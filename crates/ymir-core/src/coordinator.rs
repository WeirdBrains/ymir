//! The inference role: catalog and routing over model nodes, the answer cache and the
//! cold-start stand-in. It speaks `weirdnet-agent` infer/2 requests end to end; [`crate::host`]
//! serves it as an `InferHost`.

use std::collections::{BTreeMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use futures::StreamExt;
use weirdnet_agent::remote::RemoteError;
use weirdnet_agent::remote::catalog::{CatalogModel as AgentModel, ModelState as AgentState, Purpose};
use weirdnet_agent::remote::infer::v2::{Finish, Request, Usage};
use weirdnet_agent::{CancelToken, ToolCall};
use weirdnet_backends::error::{bad_request, engine_error, unavailable, unknown_model};
use weirdnet_backends::{Chunk, ModelState, ModelStatus, ToolCalls};
use weirdnet_semcache::{AnswerCache, CachedAnswer};

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

/// One model in the merged catalog, with its state on every node that has it.
#[derive(Debug, Clone, PartialEq)]
pub struct CatalogRow {
    pub id: String,
    pub nodes: Vec<(String, ModelState)>,
    /// The nodes (among `nodes`) that are draining: they finish what they have and take no more.
    pub draining: Vec<String>,
    pub embedding: bool,
}

impl CatalogRow {
    /// Whether a node that takes new requests has it ready.
    pub fn ready(&self) -> bool {
        self.nodes.iter().any(|(n, s)| *s == ModelState::Ready && !self.draining.contains(n))
    }
}

/// A model node as `ymirctl nodes` shows it.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeInfo {
    pub name: String,
    pub draining: bool,
    pub models: usize,
    pub ready: usize,
    pub in_flight: usize,
}

/// Failures that another node may not share: the node is gone, busy or broken. Asker mistakes
/// (`BadRequest`, `Refused`, `Unsupported`) and cancels are not retried elsewhere.
fn retryable(e: &RemoteError) -> bool {
    matches!(
        e,
        RemoteError::Unreachable { .. }
            | RemoteError::Busy
            | RemoteError::Unavailable { .. }
            | RemoteError::Failed { .. }
            | RemoteError::Closed
            | RemoteError::Timeout
            | RemoteError::Stalled { .. }
    )
}

/// Where an answer streams: infer/2's `Output`, or an infer/1 turn being collected.
pub trait Sink: Send + Sync {
    fn text(&self, t: String);
    fn reasoning(&self, t: String);
    fn tool_call(&self, c: ToolCall);
    fn usage(&self, u: Usage);
}

impl Sink for weirdnet_agent::remote::infer::v2::Output {
    fn text(&self, t: String) {
        weirdnet_agent::remote::infer::v2::Output::text(self, t)
    }
    fn reasoning(&self, t: String) {
        weirdnet_agent::remote::infer::v2::Output::reasoning(self, t)
    }
    fn tool_call(&self, c: ToolCall) {
        weirdnet_agent::remote::infer::v2::Output::tool_call(self, c)
    }
    fn usage(&self, u: Usage) {
        weirdnet_agent::remote::infer::v2::Output::usage(self, u)
    }
}

/// Catalog, routing, cold-start loading, stand-ins and the answer cache. Model nodes come and
/// go (remote ones as they appear among the user's devices).
pub struct Coordinator {
    nodes: RwLock<Vec<Arc<dyn ModelNode>>>,
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
/// model that answers there (the stand-in while the requested model loads). [`Coordinator::run`]
/// then loads the model if it is still cold and streams the answer. The split lets infer/2 name
/// the answering model in `Accepted` before anything streams.
pub struct Prepared {
    /// The model that answers (the stand-in when [`Prepared::standin`] is set).
    pub model: String,
    pub standin: Option<String>,
    raw: bool,
    route: Route,
}

enum Route {
    Cached(CachedAnswer),
    Node { node: Arc<dyn ModelNode>, cold: bool },
}

impl Prepared {
    pub fn cache_hit(&self) -> bool {
        matches!(self.route, Route::Cached(_))
    }
}

fn read<T>(l: &RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    l.read().unwrap_or_else(|e| e.into_inner())
}

impl Coordinator {
    pub fn new(nodes: Vec<Arc<dyn ModelNode>>, main: Option<String>, cache: Arc<dyn AnswerCache>) -> Self {
        Self::with_options(nodes, main, cache, CoordinatorOptions::default())
    }

    pub fn with_options(nodes: Vec<Arc<dyn ModelNode>>, main: Option<String>, cache: Arc<dyn AnswerCache>, opts: CoordinatorOptions) -> Self {
        Self { nodes: RwLock::new(nodes), main, cache, opts, loading: Arc::new(Mutex::new(HashSet::new())) }
    }

    pub fn nodes(&self) -> Vec<Arc<dyn ModelNode>> {
        read(&self.nodes).clone()
    }

    /// Adds a model node (replacing one of the same name).
    pub fn add_node(&self, node: Arc<dyn ModelNode>) {
        let mut nodes = self.nodes.write().unwrap_or_else(|e| e.into_inner());
        nodes.retain(|n| n.name() != node.name());
        nodes.push(node);
    }

    pub fn remove_node(&self, name: &str) -> bool {
        let mut nodes = self.nodes.write().unwrap_or_else(|e| e.into_inner());
        let before = nodes.len();
        nodes.retain(|n| n.name() != name);
        nodes.len() != before
    }

    pub fn options(&self) -> &CoordinatorOptions {
        &self.opts
    }

    pub fn main_model(&self) -> Option<&str> {
        self.main.as_deref()
    }

    fn node(&self, name: &str) -> Option<Arc<dyn ModelNode>> {
        read(&self.nodes).iter().find(|n| n.name() == name).cloned()
    }

    /// The pool catalog: every model on every node, with its state there.
    pub fn merged_catalog(&self) -> Vec<CatalogRow> {
        let mut by_id: BTreeMap<String, CatalogRow> = BTreeMap::new();
        for node in self.nodes() {
            for s in node.status() {
                let row = by_id.entry(s.id.clone()).or_insert_with(|| CatalogRow {
                    id: s.id.clone(),
                    nodes: Vec::new(),
                    draining: Vec::new(),
                    embedding: s.embedding,
                });
                row.nodes.push((node.name().to_string(), s.state.clone()));
                if node.draining() {
                    row.draining.push(node.name().to_string());
                }
            }
        }
        by_id.into_values().collect()
    }

    /// The merged catalog as catalog/1 entries; chat models name the stand-in.
    pub fn agent_catalog(&self) -> Vec<AgentModel> {
        let standin = self.opts.standin.as_deref();
        self.merged_catalog()
            .iter()
            .map(|m| {
                // A draining node takes no new requests, so it doesn't count as warm or loadable.
                let states: Vec<&ModelState> = m.nodes.iter().filter(|(n, _)| !m.draining.contains(n)).map(|(_, s)| s).collect();
                let state = if states.contains(&&ModelState::Ready) {
                    AgentState::Warm
                } else if states.contains(&&ModelState::Loading) {
                    AgentState::Loading
                } else {
                    AgentState::Loadable
                };
                let purpose = if m.embedding { Purpose::Embedding } else { Purpose::Chat };
                let ready = states.iter().filter(|s| ***s == ModelState::Ready).count() as u32;
                let mut c = AgentModel::new(&m.id, purpose, state).with_replicas(ready);
                if !m.embedding {
                    c.standin = standin.filter(|s| *s != m.id).map(str::to_string);
                }
                c
            })
            .collect()
    }

    /// Picks a node for `model`: a ready one with the fewest requests in flight, else the first
    /// node that has it (it gets loaded). `embedding` says which kind of model the caller wants.
    /// Draining nodes and the ones in `exclude` (already tried for this request) are skipped.
    fn pick(&self, model: &str, embedding: bool, exclude: &HashSet<String>) -> Result<(Arc<dyn ModelNode>, ModelStatus), RemoteError> {
        let mut best: Option<(Arc<dyn ModelNode>, ModelStatus)> = None;
        let mut fallback = None;
        let mut known = false;
        let mut draining = false;
        for node in self.nodes() {
            if let Some(s) = node.status().into_iter().find(|s| s.id == model) {
                known = true;
                if s.embedding != embedding {
                    return Err(bad_request(if s.embedding {
                        format!("{model} is an embedding model; use /v1/embeddings")
                    } else {
                        format!("{model} is not an embedding model")
                    }));
                }
                if node.draining() {
                    draining = true;
                    continue;
                }
                if exclude.contains(node.name()) {
                    continue;
                }
                if s.state == ModelState::Ready && best.as_ref().is_none_or(|(_, b)| s.in_flight < b.in_flight) {
                    best = Some((node.clone(), s.clone()));
                }
                fallback.get_or_insert_with(|| (node.clone(), s));
            }
        }
        match best.or(fallback) {
            Some(found) => Ok(found),
            None if known && draining && exclude.is_empty() => Err(unavailable(format!("{model} is only on nodes that are draining"))),
            None if known => Err(unavailable(format!("no other node can answer with {model}"))),
            None => Err(unknown_model(model)),
        }
    }

    /// The model nodes, for `ymirctl nodes`.
    pub fn node_infos(&self) -> Vec<NodeInfo> {
        self.nodes()
            .iter()
            .map(|n| {
                let st = n.status();
                NodeInfo {
                    name: n.name().to_string(),
                    draining: n.draining(),
                    models: st.len(),
                    ready: st.iter().filter(|s| s.state == ModelState::Ready).count(),
                    in_flight: st.iter().map(|s| s.in_flight).sum(),
                }
            })
            .collect()
    }

    /// Drains `node` (it finishes what it has and takes no new requests) or resumes it.
    pub async fn drain(&self, node: &str, on: bool) -> Result<(), RemoteError> {
        let n = self.node(node).ok_or_else(|| unavailable(format!("no model node `{node}`")))?;
        n.drain(on).await
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
    pub async fn load(&self, model: &str, node: Option<&str>, force: bool) -> Result<String, RemoteError> {
        let n = match node {
            Some(name) => self.node(name).ok_or_else(|| unavailable(format!("no model node `{name}`")))?,
            None => {
                let embedding = self.is_embedding(model)?;
                self.pick(model, embedding, &HashSet::new())?.0
            }
        };
        n.load(model, force).await?;
        Ok(n.name().to_string())
    }

    fn is_embedding(&self, model: &str) -> Result<bool, RemoteError> {
        self.nodes().iter().flat_map(|n| n.status()).find(|s| s.id == model).map(|s| s.embedding).ok_or_else(|| unknown_model(model))
    }

    /// Unloads `model` everywhere it is resident.
    pub async fn unload(&self, model: &str) -> Result<Vec<String>, RemoteError> {
        let mut done = Vec::new();
        let mut known = false;
        for node in self.nodes() {
            if let Some(s) = node.status().into_iter().find(|s| s.id == model) {
                known = true;
                if s.state != ModelState::Unloaded {
                    node.unload(model).await?;
                    done.push(node.name().to_string());
                }
            }
        }
        if !known {
            return Err(unknown_model(model));
        }
        Ok(done)
    }

    fn resolve(&self, model: Option<&str>) -> Result<String, RemoteError> {
        match model {
            None | Some("default") | Some("ymir") => self.main.clone().ok_or_else(|| bad_request("no model given and this pool has no main model")),
            Some(m) => Ok(m.to_string()),
        }
    }

    /// Picks the node for a chat without waiting: a ready one, or the stand-in (whose model then
    /// loads in the background), or a cold node (`true`: load it first).
    fn choose(&self, model: &str, wait: bool) -> Result<(Target, bool), RemoteError> {
        let (node, status) = self.pick(model, false, &HashSet::new())?;
        if status.state == ModelState::Ready {
            return Ok((Target { node, model: model.into(), standin: None }, false));
        }
        if !wait && let Some(t) = self.standin_for(model, &node, &status) {
            self.load_in_background(node, model.to_string());
            return Ok((t, false));
        }
        Ok((Target { node, model: model.into(), standin: None }, true))
    }

    /// Decides where a chat goes (see [`Prepared`]). `raw` requests skip the cache and the
    /// stand-in.
    pub fn prepare(&self, req: &Request) -> Result<Prepared, RemoteError> {
        let model = self.resolve(req.model.as_deref())?;
        if !req.raw
            && let Some(hit) = self.cache.get(&model, req)
        {
            return Ok(Prepared { model, standin: None, raw: false, route: Route::Cached(hit) });
        }
        let (t, cold) = self.choose(&model, req.wait_for_model || req.raw)?;
        Ok(Prepared { model: t.model, standin: t.standin, raw: req.raw, route: Route::Node { node: t.node, cold } })
    }

    /// The stand-in, when policy allows it for this cold `model` on `node`.
    fn standin_for(&self, model: &str, node: &Arc<dyn ModelNode>, status: &ModelStatus) -> Option<Target> {
        let standin = self.opts.standin.as_deref().filter(|s| *s != model)?;
        let estimate = Self::estimate_load(status);
        if estimate <= self.opts.standin_after {
            return None;
        }
        for n in self.nodes() {
            let Some(s) = n.status().into_iter().find(|s| s.id == standin) else { continue };
            // Loading the model would unload a stand-in in its swap group on the same node.
            let evicted = Arc::ptr_eq(&n, node) && s.swap_group.is_some() && s.swap_group == status.swap_group;
            if s.state == ModelState::Ready && !s.embedding && !evicted && !n.draining() {
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

    /// Streams the answer for a [`Prepared`] route into `out`, loading a cold model first.
    /// Finished answers are cached under the model that produced them.
    ///
    /// If the node fails before anything has streamed (gone, busy, engine died, load failed), the
    /// request moves to another node that has the model. After the first token the error goes to
    /// the asker: the app's pool resumes from what it received.
    pub async fn run(&self, p: Prepared, req: &Request, out: &dyn Sink, cancel: &CancelToken) -> Result<Finish, RemoteError> {
        let (mut node, mut cold) = match p.route {
            Route::Cached(hit) => return Ok(replay(hit, out)),
            Route::Node { node, cold } => (node, cold),
        };
        let mut tried: HashSet<String> = HashSet::new();
        let mut reloaded = false;
        loop {
            let (err, sent) = match self.attempt(&node, cold, &p.model, req, out, cancel, p.standin.as_deref(), p.raw).await {
                Ok(f) => return Ok(f),
                Err(e) => e,
            };
            if sent || !retryable(&err) || cancel.is_cancelled() {
                return Err(err);
            }
            tried.insert(node.name().to_string());
            match self.pick(&p.model, false, &tried) {
                Ok((next, status)) => {
                    tracing::warn!(model = %p.model, from = node.name(), to = next.name(), "{err}; moving the request to another node");
                    cold = status.state != ModelState::Ready;
                    node = next;
                }
                // No other node: the engine may just have died, so reload here and try once more.
                Err(_) if !reloaded && matches!(err, RemoteError::Unavailable { .. }) => {
                    tracing::warn!(model = %p.model, node = node.name(), "{err}; reloading once");
                    reloaded = true;
                    tried.remove(node.name());
                    cold = true;
                }
                Err(_) => return Err(err),
            }
        }
    }

    /// One try on `node`. The error says whether any output had already been sent.
    #[allow(clippy::too_many_arguments)]
    async fn attempt(
        &self,
        node: &Arc<dyn ModelNode>,
        cold: bool,
        model: &str,
        req: &Request,
        out: &dyn Sink,
        cancel: &CancelToken,
        standin: Option<&str>,
        raw: bool,
    ) -> Result<Finish, (RemoteError, bool)> {
        if cold {
            tracing::info!(model = %model, node = node.name(), "cold start: loading");
            tokio::select! {
                _ = cancel.cancelled() => return Err((RemoteError::Cancelled, false)),
                r = node.load(model, false) => r.map_err(|e| (e, false))?,
            }
        }
        let mut routed = req.clone();
        routed.model = Some(model.to_string());
        let mut stream = node.chat(model, &routed).await.map_err(|e| (e, false))?;
        let mut sent = false;
        let mut calls = ToolCalls::default();
        let mut answer = CachedAnswer::new(model.to_string(), "", "stop");
        loop {
            let next = tokio::select! {
                _ = cancel.cancelled() => return Err((RemoteError::Cancelled, sent)),
                n = stream.next() => n,
            };
            let chunk = next.ok_or_else(|| (engine_error("engine stream ended early"), sent))?.map_err(|e| (e, sent))?;
            match chunk {
                Chunk::Text(t) => {
                    sent = true;
                    answer.text.push_str(&t);
                    out.text(t);
                }
                Chunk::Reasoning(t) => {
                    sent = true;
                    answer.reasoning.push_str(&t);
                    out.reasoning(t);
                }
                Chunk::ToolCall { index, id, name, arguments } => calls.push(index, id, name, &arguments),
                Chunk::Done { finish_reason, usage } => {
                    let had_calls = !calls.is_empty();
                    for c in calls.finish() {
                        answer.tool_calls.push(c.clone());
                        out.tool_call(c);
                    }
                    let reason = finish_reason.unwrap_or_else(|| if had_calls { "tool_calls".into() } else { "stop".into() });
                    let mut f = Finish::new(reason.clone());
                    if let Some(u) = usage {
                        out.usage(u);
                        f = f.with_usage(u);
                    }
                    f.meta.served_by = Some(node.name().to_string());
                    f.meta.standin = standin.map(str::to_string);
                    answer.finish_reason = Some(reason);
                    answer.usage = usage;
                    if !raw && answer.complete() {
                        self.cache.put(model, req, &answer);
                    }
                    return Ok(f);
                }
                other => tracing::debug!(?other, "engine chunk this version does not use"),
            }
        }
    }

    /// The embedding model a request means (`None`/`default`: the first one in the catalog).
    pub fn resolve_embedding(&self, model: Option<&str>) -> Result<String, RemoteError> {
        match model {
            None | Some("default") | Some("ymir") => self
                .merged_catalog()
                .into_iter()
                .find(|m| m.embedding)
                .map(|m| m.id)
                .ok_or_else(|| bad_request("no model given and this pool has no embedding model")),
            Some(m) => Ok(m.to_string()),
        }
    }

    /// Embeds with `model`, loading it first if it is cold. Returns the vectors, the prompt
    /// tokens and the node that answered. A node that fails is skipped for another that has the
    /// model.
    pub async fn embed(&self, model: &str, input: &[String]) -> Result<(Vec<Vec<f32>>, u64, String), RemoteError> {
        let mut tried: HashSet<String> = HashSet::new();
        let mut reloaded = false;
        loop {
            let (node, status) = self.pick(model, true, &tried)?;
            let attempt = async {
                if status.state != ModelState::Ready {
                    tracing::info!(model = %model, node = node.name(), "cold start: loading");
                    node.load(model, false).await?;
                }
                node.embed(model, input).await
            }
            .await;
            let err = match attempt {
                Ok((vectors, tokens)) => return Ok((vectors, tokens, node.name().to_string())),
                Err(e) => e,
            };
            if !retryable(&err) {
                return Err(err);
            }
            tried.insert(node.name().to_string());
            if self.pick(model, true, &tried).is_ok() {
                tracing::warn!(model = %model, node = node.name(), "{err}; trying another node");
            } else if !reloaded && matches!(err, RemoteError::Unavailable { .. }) {
                tracing::warn!(model = %model, node = node.name(), "{err}; reloading once");
                reloaded = true;
                tried.remove(node.name());
                node.load(model, false).await.ok();
            } else {
                return Err(err);
            }
        }
    }
}

/// A cached answer, replayed.
fn replay(hit: CachedAnswer, out: &dyn Sink) -> Finish {
    if !hit.reasoning.is_empty() {
        out.reasoning(hit.reasoning.clone());
    }
    if !hit.text.is_empty() {
        out.text(hit.text.clone());
    }
    for c in &hit.tool_calls {
        out.tool_call(c.clone());
    }
    let mut f = Finish::new(hit.finish_reason.clone().unwrap_or_else(|| "stop".into()));
    if let Some(u) = hit.usage {
        out.usage(u);
        f = f.with_usage(u);
    }
    f.meta.cache_hit = true;
    f.meta.served_by = Some("cache".into());
    f
}
