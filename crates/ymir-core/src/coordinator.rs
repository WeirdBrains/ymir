//! The inference role: catalog and routing over model nodes (M1: the local one).

use std::collections::BTreeMap;
use std::sync::Arc;

use futures::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use ymir_infer::{BoxFuture, CatalogModel, ChatRequest, Event, InferError, ModelState, Pool, Reply};
use ymir_semcache::AnswerCache;

use crate::ModelNode;

/// Catalog, routing and cold-start loading. Placement, the scheduler lease, stand-ins and resume
/// come in later milestones; the shape stays: apps call [`Pool`], this calls [`ModelNode`]s.
pub struct Coordinator {
    nodes: Vec<Arc<dyn ModelNode>>,
    main: Option<String>,
    cache: Arc<dyn AnswerCache>,
}

impl Coordinator {
    pub fn new(nodes: Vec<Arc<dyn ModelNode>>, main: Option<String>, cache: Arc<dyn AnswerCache>) -> Self {
        Self { nodes, main, cache }
    }

    pub fn nodes(&self) -> &[Arc<dyn ModelNode>] {
        &self.nodes
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
    /// node that has it (it gets loaded).
    fn pick(&self, model: &str) -> Result<(Arc<dyn ModelNode>, bool), InferError> {
        let mut best: Option<(usize, Arc<dyn ModelNode>)> = None;
        let mut fallback = None;
        let mut embedding = false;
        for node in &self.nodes {
            if let Some(s) = node.status().into_iter().find(|s| s.id == model) {
                embedding |= s.embedding;
                if s.state == ModelState::Ready && best.as_ref().is_none_or(|(n, _)| s.in_flight < *n) {
                    best = Some((s.in_flight, node.clone()));
                }
                fallback.get_or_insert_with(|| node.clone());
            }
        }
        if embedding {
            return Err(InferError::BadRequest(format!("{model} is an embedding model")));
        }
        match (best, fallback) {
            (Some((_, n)), _) => Ok((n, true)),
            (None, Some(n)) => Ok((n, false)),
            (None, None) => Err(InferError::UnknownModel(model.into())),
        }
    }

    /// Loads `model` on a node that has it (or `node`).
    pub async fn load(&self, model: &str, node: Option<&str>) -> Result<String, InferError> {
        let n = match node {
            Some(name) => self.node(name).cloned().ok_or_else(|| InferError::Unavailable(format!("no model node `{name}`")))?,
            None => self.pick(model)?.0,
        };
        n.load(model).await?;
        Ok(n.name().to_string())
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

    async fn route(&self, req: ChatRequest) -> Result<Reply, InferError> {
        let model = match req.model.as_deref() {
            None | Some("default") | Some("ymir") => {
                self.main.clone().ok_or_else(|| InferError::BadRequest("no model given and this pool has no main model".into()))?
            }
            Some(m) => m.to_string(),
        };
        if let Some(events) = self.cache.get(&model, &req) {
            return Ok(Reply {
                model,
                node: "cache".into(),
                cache_hit: true,
                standin: None,
                events: Box::pin(futures::stream::iter(events.into_iter().map(Ok))),
            });
        }
        let (node, ready) = self.pick(&model)?;
        if !ready {
            tracing::info!(model = %model, node = node.name(), "cold start: loading");
            node.load(&model).await?;
        }
        let events = match node.chat(&model, &req).await {
            Ok(s) => s,
            // The engine died between the pick and the request. Nothing has streamed yet, so a
            // reload and one retry is safe.
            Err(InferError::Unavailable(why)) => {
                tracing::warn!(model = %model, "{why}; reloading once");
                node.load(&model).await?;
                node.chat(&model, &req).await?
            }
            Err(e) => return Err(e),
        };
        // Tee the stream into the cache.
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let cache = self.cache.clone();
        let cache_model = model.clone();
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
        Ok(Reply { model, node: node.name().to_string(), cache_hit: false, standin: None, events: Box::pin(ReceiverStream::new(rx)) })
    }
}

impl Pool for Coordinator {
    fn catalog(&self) -> BoxFuture<'_, Result<Vec<CatalogModel>, InferError>> {
        Box::pin(async move { Ok(self.merged_catalog()) })
    }

    fn chat(&self, req: ChatRequest) -> BoxFuture<'_, Result<Reply, InferError>> {
        Box::pin(self.route(req))
    }
}
