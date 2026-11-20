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

use crate::context::ContextStore;
use crate::coordinator::{Coordinator, Prepared, Sink};
use crate::membership::{Membership, SyncNode};

/// Everyone: for this box's own bridge, in process.
pub struct AllowAll;

impl Authorize for AllowAll {
    fn authorize(&self, _: &Caller) -> Result<(), RemoteError> {
        Ok(())
    }
}

/// The conversations an inference node answers `ContextRef` requests from, and how it fetches
/// what a caller has written that it hasn't synced yet.
#[derive(Clone)]
pub struct ContextAccess {
    pub store: ContextStore,
    /// Pull-on-demand from the caller; `None` for a caller in this process (its store is ours).
    pub sync: Option<ContextSync>,
}

#[derive(Clone)]
pub struct ContextSync {
    pub membership: Arc<Membership>,
    /// Weak, so a request still running after the daemon stops can't keep the Weirdnet node alive.
    pub node: std::sync::Weak<SyncNode>,
    pub hints: crate::remote::Hints,
}

impl ContextAccess {
    /// The prompt for `r`. A caller on another device is pulled from first, so the newest turn it
    /// wrote is here even if the periodic pass hasn't run.
    async fn prompt(&self, caller: &Caller, r: &v2::ContextRef) -> Result<Vec<Message>, RemoteError> {
        if let Some(sync) = &self.sync
            && !caller.peer.is_empty()
        {
            let (sync, peer) = (sync.clone(), caller.peer.clone());
            let addrs: Vec<_> = sync.hints.iter().filter(|(p, _)| p.to_string() == peer).flat_map(|(_, a)| a.clone()).collect();
            let pulled =
                tokio::task::spawn_blocking(move || sync.node.upgrade().is_some_and(|node| sync.membership.pull_context(&node, &peer, &addrs)))
                    .await
                    .unwrap_or(false);
            tracing::debug!(conversation = %r.conversation, pulled, "context: pulled from the caller");
        }
        let conv = self
            .store
            .load(&r.conversation)
            .ok_or_else(|| RemoteError::unavailable(format!("conversation {} is not on this node (yet)", r.conversation)))?;
        conv.prompt(r.head.as_deref()).ok_or_else(|| {
            RemoteError::unavailable(format!("conversation {} has no message {} here (yet)", r.conversation, r.head.as_deref().unwrap_or("?")))
        })
    }
}

/// infer/1, infer/2 and embeddings from the coordinator, to the callers `auth` admits.
pub struct InferenceHost {
    coord: Arc<Coordinator>,
    auth: Arc<dyn Authorize>,
    context: Option<ContextAccess>,
}

impl InferenceHost {
    pub fn new(coord: Arc<Coordinator>, auth: Arc<dyn Authorize>) -> Self {
        Self { coord, auth, context: None }
    }

    /// Answers `ContextRef` requests from `context` and writes each answer back to it.
    pub fn with_context(mut self, context: Option<ContextAccess>) -> Self {
        self.context = context;
        self
    }
}

impl InferHost for InferenceHost {
    fn authorize(&self, caller: &Caller) -> Result<(), RemoteError> {
        self.auth.authorize(caller)
    }

    fn open(&self, _caller: &Caller, model: Option<&str>) -> Result<Arc<dyn ModelBackend>, RemoteError> {
        Ok(Arc::new(TurnBackend { coord: self.coord.clone(), model: model.map(str::to_string) }))
    }

    fn open_request(&self, caller: &Caller, request: &v2::Request) -> Result<Arc<dyn InferBackend>, RemoteError> {
        if matches!(request.input, v2::Input::Context(_)) && self.context.is_none() {
            return Err(RemoteError::BadRequest { message: "this node holds no conversations".into() });
        }
        let prepared = self.coord.prepare(request)?;
        Ok(Arc::new(RoutedBackend {
            coord: self.coord.clone(),
            name: prepared.model.clone(),
            prepared: Mutex::new(Some(prepared)),
            context: self.context.clone(),
            caller: caller.clone(),
        }))
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
    context: Option<ContextAccess>,
    caller: Caller,
}

/// Passes text on and keeps it, so the finished answer can be written to the conversation.
struct Tee<'a> {
    out: &'a dyn Sink,
    text: Mutex<String>,
}

impl Sink for Tee<'_> {
    fn text(&self, t: String) {
        self.text.lock().unwrap_or_else(|e| e.into_inner()).push_str(&t);
        self.out.text(t);
    }
    fn reasoning(&self, t: String) {
        self.out.reasoning(t);
    }
    fn tool_call(&self, c: ToolCall) {
        self.out.tool_call(c);
    }
    fn usage(&self, u: Usage) {
        self.out.usage(u);
    }
}

#[async_trait]
impl InferBackend for RoutedBackend {
    fn name(&self) -> String {
        self.name.clone()
    }

    /// Continues an `assistant_prefix`: the model nodes' engines do (llama.cpp prefill, vLLM
    /// `continue_final_message`), so an asker whose answer was cut off can resume here.
    fn resumes(&self) -> bool {
        self.coord.can_resume(&self.name)
    }

    async fn infer(&self, request: &v2::Request, out: &Output, cancel: &CancelToken) -> Result<Finish, RemoteError> {
        let prepared = self.prepared.lock().unwrap_or_else(|e| e.into_inner()).take();
        let prepared = match prepared {
            Some(p) => p,
            None => self.coord.prepare(request)?,
        };
        // A conversation reference becomes the assembled messages (raw mode to a model node
        // never sees one); the answer is written back under the job's id when it completes.
        let mut resolved = request.clone();
        let mut conversation = None;
        if let v2::Input::Context(r) = &request.input {
            let access = self.context.as_ref().ok_or_else(|| RemoteError::BadRequest { message: "this node holds no conversations".into() })?;
            resolved.input = v2::Input::Messages(access.prompt(&self.caller, r).await?);
            conversation = Some((access.store.clone(), r.conversation.clone()));
        }
        let tee = Tee { out, text: Mutex::new(String::new()) };
        let finish = self.coord.run(prepared, &resolved, &tee, cancel).await?;
        if let Some((store, conv)) = conversation {
            // What the asker already had (a resume) plus what this attempt streamed: the whole
            // answer, on whichever node finishes it.
            let before = request.assistant_prefix.as_ref().map_or("", |p| p.text.as_str());
            let text = format!("{before}{}", tee.text.into_inner().unwrap_or_else(|e| e.into_inner()));
            let model = finish.meta.standin.clone().unwrap_or_else(|| self.name.clone());
            if let Err(e) = store.append(&conv, "assistant", &text, Some(&model), Some(&request.job_id)) {
                tracing::warn!(conversation = %conv, "writing the answer to the conversation: {e}");
            }
        }
        Ok(finish)
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
        Ok(self.embed_counted(input, cancel).await?.0)
    }

    /// The engine's prompt-token count rides along, so the bridge reports real embedding usage.
    async fn embed_counted(&self, input: &str, cancel: &CancelToken) -> Result<(Vec<f32>, Option<u64>), RemoteError> {
        let input = [input.to_string()];
        let (vectors, tokens, _) = tokio::select! {
            _ = cancel.cancelled() => return Err(RemoteError::Cancelled),
            e = self.coord.embed(&self.model, &input) => e?,
        };
        let vector = vectors.into_iter().next().ok_or_else(|| engine_error("the engine returned no vector"))?;
        Ok((vector, Some(tokens)))
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
pub fn local_pool(coord: Arc<Coordinator>, node: &str, admission: Admission, context: Option<ContextStore>) -> Pool {
    let auth: Arc<dyn Authorize> = Arc::new(AllowAll);
    let access = context.map(|store| ContextAccess { store, sync: None });
    let host: Arc<dyn InferHost> = Arc::new(InferenceHost::new(coord.clone(), auth.clone()).with_context(access));
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
