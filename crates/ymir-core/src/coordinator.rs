//! The inference role: catalog and routing over model nodes (so far, the local one).

use std::collections::{BTreeMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use ymir_backends::ModelStatus;
use ymir_infer::{
    BoxFuture, CatalogModel, ChatRequest, CompletionRequest, EmbedRequest, Embeddings, Event, EventStream, InferError, ModelState, Pool, Reply,
};
use ymir_semcache::AnswerCache;

use crate::ModelNode;

/// Cold-start policy.
#[derive(Debug, Clone)]
pub struct CoordinatorOptions {
    /// A small model that answers, flagged, while a big one loads.
    pub standin: Option<String>,
    /// Use the stand-in only when the estimated load takes longer than this.
    pub standin_after: Duration,
}

impl Default for CoordinatorOptions {
    fn default() -> Self {
        Self { standin: None, standin_after: Duration::from_secs(10) }
    }
}

/// Catalog, routing, cold-start loading and stand-ins. Placement, the scheduler lease and resume
/// come in later milestones; the shape stays: apps call [`Pool`], this calls [`ModelNode`]s.
pub struct Coordinator {
    nodes: Vec<Arc<dyn ModelNode>>,
    main: Option<String>,
    cache: Arc<dyn AnswerCache>,
    opts: CoordinatorOptions,
    /// Background loads started for stand-in answers, so a burst of requests starts one load.
    loading: Arc<Mutex<HashSet<String>>>,
}

/// Where a request goes.
struct Target {
    node: Arc<dyn ModelNode>,
    model: String,
    standin: Option<String>,
}

/// A chat request's route, decided without waiting: a cached answer, or a model node and the
/// model that answers there (the stand-in while the requested model loads). [`Coordinator::run_chat`]
/// then loads the model if it is still cold and streams the answer. The split lets the infer/2
/// host name the answering model in `Accepted` before anything streams.
pub struct Prepared {
    /// The model that answers (the stand-in when [`Prepared::standin`] is set).
    pub model: String,
    pub standin: Option<String>,
    route: Route,
}

enum Route {
    Cached(Vec<Event>),
    Node { node: Arc<dyn ModelNode>, cold: bool },
}

impl Prepared {
    pub fn cache_hit(&self) -> bool {
        matches!(self.route, Route::Cached(_))
    }
}

impl Coordinator {
    pub fn new(nodes: Vec<Arc<dyn ModelNode>>, main: Option<String>, cache: Arc<dyn AnswerCache>) -> Self {
        Self::with_options(nodes, main, cache, CoordinatorOptions::default())
    }

    pub fn with_options(nodes: Vec<Arc<dyn ModelNode>>, main: Option<String>, cache: Arc<dyn AnswerCache>, opts: CoordinatorOptions) -> Self {
        Self { nodes, main, cache, opts, loading: Arc::new(Mutex::new(HashSet::new())) }
    }

    pub fn nodes(&self) -> &[Arc<dyn ModelNode>] {
        &self.nodes
    }

    pub fn options(&self) -> &CoordinatorOptions {
        &self.opts
    }

    fn node(&self, name: &str) -> Option<&Arc<dyn ModelNode>> {
        self.nodes.iter().find(|n| n.name() == name)
    }

    /// The pool catalog: every model on every node, with its state there.
    pub fn merged_catalog(&self) -> Vec<CatalogModel> {
        let mut by_id: BTreeMap<String, CatalogModel> = BTreeMap::new();
        for node in &self.nodes {
            for s in node.status() {
                let row = by_id.entry(s.id.clone()).or_insert_with(|| CatalogModel {
                    id: s.id.clone(),
                    nodes: Vec::new(),
                    purpose: if s.embedding { "embedding".into() } else { "chat".into() },
                });
                row.nodes.push((node.name().to_string(), s.state.clone()));
            }
        }
        by_id.into_values().collect()
    }

    /// Picks a node for `model`: a ready one with the fewest requests in flight, else the first
    /// node that has it (it gets loaded). `embedding` says which kind of model the caller wants.
    fn pick(&self, model: &str, embedding: bool) -> Result<(Arc<dyn ModelNode>, ModelStatus), InferError> {
        let mut best: Option<(Arc<dyn ModelNode>, ModelStatus)> = None;
        let mut fallback = None;
        for node in &self.nodes {
            if let Some(s) = node.status().into_iter().find(|s| s.id == model) {
                if s.embedding != embedding {
                    return Err(InferError::BadRequest(if s.embedding {
                        format!("{model} is an embedding model; use /v1/embeddings")
                    } else {
                        format!("{model} is not an embedding model")
                    }));
                }
                if s.state == ModelState::Ready && best.as_ref().is_none_or(|(_, b)| s.in_flight < b.in_flight) {
                    best = Some((node.clone(), s.clone()));
                }
                fallback.get_or_insert_with(|| (node.clone(), s));
            }
        }
        best.or(fallback).ok_or_else(|| InferError::UnknownModel(model.into()))
    }

    /// How long loading this model will take: the last measured load, else a guess from its file
    /// size (about 1 GB/s from page cache or NVMe, plus engine start-up). Unknown size (a vLLM
    /// model by name): assume it is slow.
    pub fn estimate_load(s: &ModelStatus) -> Duration {
        if s.wrapped {
            return Duration::ZERO;
        }
        match (s.last_load_secs, s.size_bytes) {
            (Some(secs), _) => Duration::from_secs_f32(secs.max(0.0)),
            (None, Some(bytes)) => Duration::from_secs_f64(bytes as f64 / 1e9 + 3.0),
            (None, None) => Duration::from_secs(120),
        }
    }

    /// Loads `model` on a node that has it (or `node`). `force` clears a crash loop.
    pub async fn load(&self, model: &str, node: Option<&str>, force: bool) -> Result<String, InferError> {
        let n = match node {
            Some(name) => self.node(name).cloned().ok_or_else(|| InferError::Unavailable(format!("no model node `{name}`")))?,
            None => {
                let embedding = self.is_embedding(model)?;
                self.pick(model, embedding)?.0
            }
        };
        n.load(model, force).await?;
        Ok(n.name().to_string())
    }

    fn is_embedding(&self, model: &str) -> Result<bool, InferError> {
        self.nodes.iter().flat_map(|n| n.status()).find(|s| s.id == model).map(|s| s.embedding).ok_or_else(|| InferError::UnknownModel(model.into()))
    }

    /// Unloads `model` everywhere it is resident.
    pub async fn unload(&self, model: &str) -> Result<Vec<String>, InferError> {
        let mut done = Vec::new();
        let mut known = false;
        for node in &self.nodes {
            if let Some(s) = node.status().into_iter().find(|s| s.id == model) {
                known = true;
                if s.state != ModelState::Unloaded {
                    node.unload(model).await?;
                    done.push(node.name().to_string());
                }
            }
        }
        if !known {
            return Err(InferError::UnknownModel(model.into()));
        }
        Ok(done)
    }

    fn resolve(&self, model: Option<&str>) -> Result<String, InferError> {
        match model {
            None | Some("default") | Some("ymir") => {
                self.main.clone().ok_or_else(|| InferError::BadRequest("no model given and this pool has no main model".into()))
            }
            Some(m) => Ok(m.to_string()),
        }
    }

    /// Picks the node for a chat or completion without waiting: a ready one, or the stand-in
    /// (whose model then loads in the background), or a cold node (`true`: load it first).
    fn choose(&self, model: &str, wait: bool) -> Result<(Target, bool), InferError> {
        let (node, status) = self.pick(model, false)?;
        if status.state == ModelState::Ready {
            return Ok((Target { node, model: model.into(), standin: None }, false));
        }
        if !wait && let Some(t) = self.standin_for(model, &node, &status) {
            self.load_in_background(node, model.to_string());
            return Ok((t, false));
        }
        Ok((Target { node, model: model.into(), standin: None }, true))
    }

    /// [`Coordinator::choose`], then loads a cold model.
    async fn target(&self, model: &str, wait: bool) -> Result<Target, InferError> {
        let (t, cold) = self.choose(model, wait)?;
        if cold {
            tracing::info!(model = %t.model, node = t.node.name(), "cold start: loading");
            t.node.load(&t.model, false).await?;
        }
        Ok(t)
    }

    /// Decides where a chat goes (see [`Prepared`]). `raw` (infer/2 raw mode) skips the cache
    /// and the stand-in.
    pub fn prepare_chat(&self, req: &ChatRequest, raw: bool) -> Result<Prepared, InferError> {
        let model = self.resolve(req.model.as_deref())?;
        if !raw && let Some(events) = self.cache.get(&model, req) {
            return Ok(Prepared { model, standin: None, route: Route::Cached(events) });
        }
        let (t, cold) = self.choose(&model, req.wait_for_model || raw)?;
        Ok(Prepared { model: t.model, standin: t.standin, route: Route::Node { node: t.node, cold } })
    }

    /// The stand-in, when policy allows it for this cold `model` on `node`.
    fn standin_for(&self, model: &str, node: &Arc<dyn ModelNode>, status: &ModelStatus) -> Option<Target> {
        let standin = self.opts.standin.as_deref().filter(|s| *s != model)?;
        let estimate = Self::estimate_load(status);
        if estimate <= self.opts.standin_after {
            return None;
        }
        for n in &self.nodes {
            let Some(s) = n.status().into_iter().find(|s| s.id == standin) else { continue };
            // Loading the model would unload a stand-in in its swap group on the same node.
            let evicted = Arc::ptr_eq(n, node) && s.swap_group.is_some() && s.swap_group == status.swap_group;
            if s.state == ModelState::Ready && !s.embedding && !evicted {
                tracing::info!(model = %model, standin, estimate_secs = estimate.as_secs_f32(), "cold start: answering with the stand-in");
                return Some(Target { node: n.clone(), model: standin.into(), standin: Some(standin.into()) });
            }
        }
        None
    }

    fn load_in_background(&self, node: Arc<dyn ModelNode>, model: String) {
        if !self.loading.lock().unwrap_or_else(|e| e.into_inner()).insert(model.clone()) {
            return;
        }
        let loading = self.loading.clone();
        tokio::spawn(async move {
            if let Err(e) = node.load(&model, false).await {
                tracing::warn!(model = %model, "background load failed: {e}");
            }
            loading.lock().unwrap_or_else(|e| e.into_inner()).remove(&model);
        });
    }

    async fn route(&self, req: ChatRequest) -> Result<Reply, InferError> {
        let p = self.prepare_chat(&req, false)?;
        self.run_chat(p, req).await
    }

    /// Streams the answer for a [`Prepared`] route, loading a cold model first.
    pub async fn run_chat(&self, p: Prepared, req: ChatRequest) -> Result<Reply, InferError> {
        let (node, cold) = match p.route {
            Route::Cached(events) => {
                return Ok(Reply {
                    model: p.model,
                    node: "cache".into(),
                    cache_hit: true,
                    standin: None,
                    events: Box::pin(futures::stream::iter(events.into_iter().map(Ok))),
                });
            }
            Route::Node { node, cold } => (node, cold),
        };
        if cold {
            tracing::info!(model = %p.model, node = node.name(), "cold start: loading");
            node.load(&p.model, false).await?;
        }
        let t = Target { node, model: p.model, standin: p.standin };
        let events = match t.node.chat(&t.model, &req).await {
            Ok(s) => s,
            // The engine died between the pick and the request. Nothing has streamed yet, so a
            // reload and one retry is safe.
            Err(InferError::Unavailable(why)) => {
                tracing::warn!(model = %t.model, "{why}; reloading once");
                t.node.load(&t.model, false).await?;
                t.node.chat(&t.model, &req).await?
            }
            Err(e) => return Err(e),
        };
        // Tee the stream into the cache (under the model that actually answered).
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let cache = self.cache.clone();
        let cache_model = t.model.clone();
        tokio::spawn(async move {
            let mut events = events;
            let mut seen: Vec<Event> = Vec::new();
            while let Some(ev) = events.next().await {
                if let Ok(e) = &ev {
                    seen.push(e.clone());
                }
                let failed = ev.is_err();
                if tx.send(ev).await.is_err() || failed {
                    return;
                }
            }
            cache.put(&cache_model, &req, seen);
        });
        Ok(Reply { model: t.model, node: t.node.name().to_string(), cache_hit: false, standin: t.standin, events: Box::pin(ReceiverStream::new(rx)) })
    }

    async fn route_completion(&self, req: CompletionRequest) -> Result<Reply, InferError> {
        let model = self.resolve(req.model.as_deref())?;
        let t = self.target(&model, req.wait_for_model).await?;
        let events: EventStream = match t.node.complete(&t.model, &req).await {
            Ok(s) => s,
            Err(InferError::Unavailable(why)) => {
                tracing::warn!(model = %t.model, "{why}; reloading once");
                t.node.load(&t.model, false).await?;
                t.node.complete(&t.model, &req).await?
            }
            Err(e) => return Err(e),
        };
        Ok(Reply { model: t.model, node: t.node.name().to_string(), cache_hit: false, standin: t.standin, events })
    }

    /// The embedding model a request means (`None`/`default`: the first one in the catalog).
    pub fn resolve_embedding(&self, model: Option<&str>) -> Result<String, InferError> {
        match model {
            None | Some("default") | Some("ymir") => self
                .merged_catalog()
                .into_iter()
                .find(|m| m.purpose == "embedding")
                .map(|m| m.id)
                .ok_or_else(|| InferError::BadRequest("no model given and this pool has no embedding model".into())),
            Some(m) => Ok(m.to_string()),
        }
    }

    /// Embeds with the model a request means.
    pub async fn embed_with(&self, req: EmbedRequest) -> Result<Embeddings, InferError> {
        self.route_embed(req).await
    }

    async fn route_embed(&self, req: EmbedRequest) -> Result<Embeddings, InferError> {
        let model = self.resolve_embedding(req.model.as_deref())?;
        let (node, status) = self.pick(&model, true)?;
        if status.state != ModelState::Ready {
            tracing::info!(model = %model, node = node.name(), "cold start: loading");
            node.load(&model, false).await?;
        }
        let (vectors, prompt_tokens) = match node.embed(&model, &req.input).await {
            Ok(v) => v,
            Err(InferError::Unavailable(why)) => {
                tracing::warn!(model = %model, "{why}; reloading once");
                node.load(&model, false).await?;
                node.embed(&model, &req.input).await?
            }
            Err(e) => return Err(e),
        };
        Ok(Embeddings { model, node: node.name().to_string(), vectors, prompt_tokens })
    }
}

impl Pool for Coordinator {
    fn catalog(&self) -> BoxFuture<'_, Result<Vec<CatalogModel>, InferError>> {
        Box::pin(async move { Ok(self.merged_catalog()) })
    }

    fn chat(&self, req: ChatRequest) -> BoxFuture<'_, Result<Reply, InferError>> {
        Box::pin(self.route(req))
    }

    fn complete(&self, req: CompletionRequest) -> BoxFuture<'_, Result<Reply, InferError>> {
        Box::pin(self.route_completion(req))
    }

    fn embed(&self, req: EmbedRequest) -> BoxFuture<'_, Result<Embeddings, InferError>> {
        Box::pin(self.route_embed(req))
    }
}
