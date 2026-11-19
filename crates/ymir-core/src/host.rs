//! The inference role as a `weirdnet-agent` `InferHost`: infer/2 (and infer/1 agent turns) and
//! embeddings from the [`Coordinator`], and catalog/1 from its merged catalog. The same host
//! serves the user's devices over Weirdnet ([`crate::remote::serve`]) and this box's own
//! bridge in process ([`local_pool`]).

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use weirdnet_agent::pool::Pool;
use weirdnet_agent::remote::catalog::{self, CatalogFeed};
use weirdnet_agent::remote::infer::v2::{self, EmbedBackend, Finish, InferBackend, Output, Usage};
use weirdnet_agent::remote::infer::{self, Admission, InferHost};
use weirdnet_agent::remote::{Authorize, Caller, MemoryDialer, RemoteError};
use weirdnet_agent::{CancelToken, DeltaSink, Message, ModelBackend, ModelError, ModelTurn, ToolCall, ToolSpec};
use weirdnet_backends::error::engine_error;

use crate::coordinator::{Coordinator, Prepared, Sink};

/// Everyone: for this box's own bridge, in process.
pub struct AllowAll;

impl Authorize for AllowAll {
    fn authorize(&self, _: &Caller) -> Result<(), RemoteError> {
        Ok(())
    }
}

/// infer/1, infer/2 and embeddings from the coordinator, to the callers `auth` admits.
pub struct InferenceHost {
    coord: Arc<Coordinator>,
    auth: Arc<dyn Authorize>,
}

impl InferenceHost {
    pub fn new(coord: Arc<Coordinator>, auth: Arc<dyn Authorize>) -> Self {
        Self { coord, auth }
    }
}

impl InferHost for InferenceHost {
    fn authorize(&self, caller: &Caller) -> Result<(), RemoteError> {
        self.auth.authorize(caller)
    }

    fn open(&self, _caller: &Caller, model: Option<&str>) -> Result<Arc<dyn ModelBackend>, RemoteError> {
        Ok(Arc::new(TurnBackend { coord: self.coord.clone(), model: model.map(str::to_string) }))
    }

    fn open_request(&self, _caller: &Caller, request: &v2::Request) -> Result<Arc<dyn InferBackend>, RemoteError> {
        let prepared = self.coord.prepare(request)?;
        Ok(Arc::new(RoutedBackend { coord: self.coord.clone(), name: prepared.model.clone(), prepared: Mutex::new(Some(prepared)) }))
    }

    fn open_embed(&self, _caller: &Caller, model: Option<&str>) -> Result<Arc<dyn EmbedBackend>, RemoteError> {
        let model = self.coord.resolve_embedding(model)?;
        Ok(Arc::new(Embedder { coord: self.coord.clone(), model }))
    }
}

/// One infer/2 request, routed when it was opened (so `Accepted` names the model that answers,
/// the stand-in included).
struct RoutedBackend {
    coord: Arc<Coordinator>,
    name: String,
    prepared: Mutex<Option<Prepared>>,
}

#[async_trait]
impl InferBackend for RoutedBackend {
    fn name(&self) -> String {
        self.name.clone()
    }

    async fn infer(&self, request: &v2::Request, out: &Output, cancel: &CancelToken) -> Result<Finish, RemoteError> {
        let prepared = self.prepared.lock().unwrap_or_else(|e| e.into_inner()).take();
        let prepared = match prepared {
            Some(p) => p,
            None => self.coord.prepare(request)?,
        };
        self.coord.run(prepared, request, out, cancel).await
    }
}

/// An infer/1 turn being collected.
struct Turn<'a> {
    deltas: &'a DeltaSink,
    text: Mutex<String>,
    calls: Mutex<Vec<ToolCall>>,
}

impl Sink for Turn<'_> {
    fn text(&self, t: String) {
        self.text.lock().unwrap_or_else(|e| e.into_inner()).push_str(&t);
        self.deltas.text(t);
    }
    fn reasoning(&self, t: String) {
        self.deltas.reasoning(t);
    }
    fn tool_call(&self, c: ToolCall) {
        self.calls.lock().unwrap_or_else(|e| e.into_inner()).push(c);
    }
    fn usage(&self, _: Usage) {}
}

/// infer/1 (agent turns): the same route, collected into a turn.
struct TurnBackend {
    coord: Arc<Coordinator>,
    model: Option<String>,
}

#[async_trait]
impl ModelBackend for TurnBackend {
    fn name(&self) -> String {
        self.model.clone().unwrap_or_else(|| "ymir".into())
    }

    async fn complete(&self, messages: &[Message], tools: &[ToolSpec], deltas: &DeltaSink, cancel: &CancelToken) -> Result<ModelTurn, ModelError> {
        let mut r = v2::Request::messages(messages.to_vec()).with_tools(tools.to_vec());
        if let Some(m) = &self.model {
            r = r.with_model(m.clone());
        }
        let model_err = |e: RemoteError| match e {
            RemoteError::Cancelled => ModelError::Cancelled,
            RemoteError::Failed { message } => ModelError::Failed(message),
            other => ModelError::Unavailable(other.to_string()),
        };
        let p = self.coord.prepare(&r).map_err(model_err)?;
        let turn = Turn { deltas, text: Mutex::new(String::new()), calls: Mutex::new(Vec::new()) };
        self.coord.run(p, &r, &turn, cancel).await.map_err(model_err)?;
        Ok(ModelTurn {
            text: turn.text.into_inner().unwrap_or_else(|e| e.into_inner()),
            tool_calls: turn.calls.into_inner().unwrap_or_else(|e| e.into_inner()),
        })
    }
}

struct Embedder {
    coord: Arc<Coordinator>,
    model: String,
}

#[async_trait]
impl EmbedBackend for Embedder {
    fn name(&self) -> String {
        self.model.clone()
    }

    async fn embed(&self, input: &str, cancel: &CancelToken) -> Result<Vec<f32>, RemoteError> {
        let input = [input.to_string()];
        let (vectors, _, _) = tokio::select! {
            _ = cancel.cancelled() => return Err(RemoteError::Cancelled),
            e = self.coord.embed(&self.model, &input) => e?,
        };
        vectors.into_iter().next().ok_or_else(|| engine_error("the engine returned no vector"))
    }
}

/// catalog/1 from the coordinator: current when a caller asks, and republished when it changes
/// (checked every `every`) for callers that follow it.
pub struct LiveCatalog {
    coord: Arc<Coordinator>,
    feed: CatalogFeed,
    last: Mutex<Vec<catalog::CatalogModel>>,
}

impl LiveCatalog {
    fn refresh(&self) {
        let models = self.coord.agent_catalog();
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        if *last != models || self.feed.current().rev == 0 {
            self.feed.publish(models.clone());
            *last = models;
        }
    }
}

impl catalog::CatalogHost for LiveCatalog {
    fn authorize(&self, caller: &Caller) -> Result<(), RemoteError> {
        catalog::CatalogHost::authorize(&self.feed, caller)
    }

    fn watch(&self, caller: &Caller) -> Result<tokio::sync::watch::Receiver<catalog::Catalog>, RemoteError> {
        self.refresh();
        catalog::CatalogHost::watch(&self.feed, caller)
    }
}

/// A [`LiveCatalog`]; its refresh task stops with its last holder.
pub fn catalog_feed(coord: Arc<Coordinator>, auth: Arc<dyn Authorize>, every: Duration) -> Arc<LiveCatalog> {
    let live = Arc::new(LiveCatalog { coord, feed: CatalogFeed::new(auth), last: Mutex::new(Vec::new()) });
    live.refresh();
    let weak = Arc::downgrade(&live);
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(every).await;
            let Some(live) = weak.upgrade() else { return };
            live.refresh();
        }
    });
    live
}

/// Admission for an inference node: `slots` requests at once, `queue` more waiting.
pub fn admission(slots: usize, queue: usize) -> Admission {
    Admission::with_queue(slots.max(1), queue)
}

/// This box's own pool, in process: the bridge on an all-roles box reaches the coordinator
/// without a network hop (the same infer/2 and catalog/1 protocol, over memory channels).
/// Answers name `node` as the node that served them.
pub fn local_pool(coord: Arc<Coordinator>, node: &str, admission: Admission) -> Pool {
    let auth: Arc<dyn Authorize> = Arc::new(AllowAll);
    let host: Arc<dyn InferHost> = Arc::new(InferenceHost::new(coord.clone(), auth.clone()));
    let feed = catalog_feed(coord, auth, Duration::from_secs(1));
    let dialer = MemoryDialer::new();
    let me = node.to_string();
    dialer.route(node, move |ch, protocol| {
        let (host, feed, admission, caller) = (host.clone(), feed.clone(), admission.clone(), Caller::new(me.clone()));
        tokio::spawn(async move {
            let r = match protocol.as_str() {
                v2::PROTOCOL => v2::serve(ch, caller, host.as_ref(), &admission).await,
                infer::PROTOCOL => infer::serve(ch, &caller, host.as_ref(), &admission).await,
                catalog::PROTOCOL => catalog::serve(ch, caller, feed.as_ref()).await,
                other => Err(RemoteError::Unsupported { message: format!("{other} is not served here") }),
            };
            if let Err(e) = r {
                tracing::debug!("local {protocol}: {e}");
            }
        });
    });
    Pool::new(Arc::new(dialer), vec![node.to_string()])
}
