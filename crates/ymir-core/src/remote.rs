//! The inference role over Weirdnet, and the app's way to it.
//!
//! - [`InferenceHost`] serves `/weirdnet-agent/infer/2` (and infer/1) from the [`Coordinator`]:
//!   the cache, the stand-in policy and the model nodes, exactly as the local bridge gets them.
//! - [`catalog_feed`] serves `/weirdnet-agent/catalog/1` from the coordinator's merged catalog.
//! - [`RemotePool`] is the app side: `weirdnet-agent`'s pool client (failover across the user's
//!   inference nodes, resume, stall detection) behind the `ymir_infer::Pool` the frozen OpenAI
//!   bridge takes.
//!
//! Every call is checked against the user's roster ([`weirdnet_agent::serve::RosterAuth`]): only
//! the user's linked, unrevoked devices get an answer.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::Value;
use tokio::sync::mpsc;
use weirdnet_agent::pool::{CatalogWatch, EventSink, LinkedNodes, NetDialer, Pool as AgentPool, PoolEvent, PoolOptions, encode_cert};
use weirdnet_agent::remote::catalog::CatalogFeed;
use weirdnet_agent::remote::infer::v2::{self, EmbedBackend, Finish, InferBackend, Meta, Output};
use weirdnet_agent::remote::infer::{Admission, InferHost};
use weirdnet_agent::serve::{RosterAuth, Server, Serving};
use weirdnet_agent::{Authorize, Caller, CancelToken, DeltaSink, Message, ModelBackend, ModelError, ModelTurn, RemoteError, ToolCall, ToolSpec};
use weirdnet_net::{Multiaddr, PeerId, WEIRDNET};
use ymir_infer::{BoxFuture, CatalogModel, ChatRequest, CompletionRequest, EmbedRequest, Embeddings, Event, InferError, Pool, Reply, Usage};

use crate::coordinator::{Coordinator, Prepared};
use crate::membership::{Membership, SyncNode};
use crate::wire;

/// Only the user's linked devices.
pub fn roster_auth(m: &Membership) -> Arc<dyn Authorize> {
    Arc::new(RosterAuth::new(&WEIRDNET, m.user_key(), m.roster_view()))
}

/// infer/1 and infer/2 from the coordinator.
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
        let chat = wire::from_agent(request).map_err(wire::error_to_remote)?;
        let prepared = self.coord.prepare_chat(&chat, request.raw).map_err(wire::error_to_remote)?;
        Ok(Arc::new(RoutedBackend { coord: self.coord.clone(), name: prepared.model.clone(), prepared: Mutex::new(Some(prepared)) }))
    }

    fn open_embed(&self, _caller: &Caller, model: Option<&str>) -> Result<Arc<dyn EmbedBackend>, RemoteError> {
        let model = self.coord.resolve_embedding(model).map_err(wire::error_to_remote)?;
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

/// Tool-call fragments as the engines stream them, joined into whole calls (infer/2 carries
/// whole calls only).
#[derive(Default)]
struct Calls(BTreeMap<u32, (Option<String>, Option<String>, String)>);

impl Calls {
    fn push(&mut self, index: u32, id: Option<String>, name: Option<String>, arguments: &str) {
        let c = self.0.entry(index).or_default();
        if id.is_some() {
            c.0 = id;
        }
        if name.is_some() {
            c.1 = name;
        }
        c.2.push_str(arguments);
    }

    fn finish(self) -> Vec<ToolCall> {
        self.0
            .into_iter()
            .map(|(i, (id, name, args))| ToolCall {
                id: id.unwrap_or_else(|| format!("call_{i}")),
                name: name.unwrap_or_default(),
                arguments: if args.trim().is_empty() {
                    Value::Object(Default::default())
                } else {
                    serde_json::from_str(&args).unwrap_or(Value::String(args))
                },
            })
            .collect()
    }
}

/// What a finished stream gave, for the sinks the two protocols use.
struct Streamed {
    calls: Vec<ToolCall>,
    finish_reason: Option<String>,
    usage: Option<Usage>,
}

async fn drain(
    reply: &mut Reply,
    cancel: &CancelToken,
    text: &(dyn Fn(String) + Sync),
    reasoning: &(dyn Fn(String) + Sync),
) -> Result<Streamed, RemoteError> {
    let mut calls = Calls::default();
    let (mut finish_reason, mut usage) = (None, None);
    loop {
        let ev = tokio::select! {
            _ = cancel.cancelled() => return Err(RemoteError::Cancelled),
            ev = reply.events.next() => ev,
        };
        match ev {
            None => break,
            Some(Err(e)) => return Err(wire::error_to_remote(e)),
            Some(Ok(Event::Delta { text: t, .. })) => text(t),
            Some(Ok(Event::Reasoning { text: t, .. })) => reasoning(t),
            Some(Ok(Event::ToolCall { index, id, name, arguments })) => calls.push(index, id, name, &arguments),
            Some(Ok(Event::Queued { .. })) => {}
            Some(Ok(Event::Done { finish_reason: f, usage: u })) => {
                finish_reason = f;
                usage = u;
            }
        }
    }
    Ok(Streamed { calls: calls.finish(), finish_reason, usage })
}

#[async_trait]
impl InferBackend for RoutedBackend {
    fn name(&self) -> String {
        self.name.clone()
    }

    async fn infer(&self, request: &v2::Request, out: &Output, cancel: &CancelToken) -> Result<Finish, RemoteError> {
        let chat = wire::from_agent(request).map_err(wire::error_to_remote)?;
        let prepared = self.prepared.lock().unwrap_or_else(|e| e.into_inner()).take();
        let prepared = match prepared {
            Some(p) => p,
            None => self.coord.prepare_chat(&chat, request.raw).map_err(wire::error_to_remote)?,
        };
        let mut reply = tokio::select! {
            _ = cancel.cancelled() => return Err(RemoteError::Cancelled),
            r = self.coord.run_chat(prepared, chat) => r.map_err(wire::error_to_remote)?,
        };
        let s = drain(&mut reply, cancel, &|t| out.text(t), &|t| out.reasoning(t)).await?;
        let calls = !s.calls.is_empty();
        for c in s.calls {
            out.tool_call(c);
        }
        let reason = s.finish_reason.unwrap_or_else(|| if calls { "tool_calls".into() } else { "stop".into() });
        let mut f = Finish::new(reason);
        if let Some(u) = s.usage {
            f = f.with_usage(v2::Usage::new(u.prompt_tokens, u.completion_tokens));
        }
        let mut meta = Meta::default();
        meta.served_by = Some(reply.node.clone());
        meta.cache_hit = reply.cache_hit;
        meta.standin = reply.standin.clone();
        f.meta = meta;
        Ok(f)
    }
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
        let model_err = |e: InferError| match e {
            InferError::Engine(m) => ModelError::Failed(m),
            other => ModelError::Unavailable(other.to_string()),
        };
        let chat = wire::from_agent(&r).map_err(model_err)?;
        let mut reply = tokio::select! {
            _ = cancel.cancelled() => return Err(ModelError::Cancelled),
            r = self.coord.chat(chat) => r.map_err(model_err)?,
        };
        let text = Mutex::new(String::new());
        let s = drain(
            &mut reply,
            cancel,
            &|t| {
                deltas.text(t.clone());
                text.lock().unwrap_or_else(|e| e.into_inner()).push_str(&t);
            },
            &|t| deltas.reasoning(t),
        )
        .await
        .map_err(|e| if e == RemoteError::Cancelled { ModelError::Cancelled } else { ModelError::Failed(e.to_string()) })?;
        Ok(ModelTurn { text: text.into_inner().unwrap_or_else(|e| e.into_inner()), tool_calls: s.calls })
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
        let req = EmbedRequest { model: Some(self.model.clone()), input: vec![input.to_string()] };
        let e = tokio::select! {
            _ = cancel.cancelled() => return Err(RemoteError::Cancelled),
            e = self.coord.embed_with(req) => e.map_err(wire::error_to_remote)?,
        };
        e.vectors.into_iter().next().ok_or_else(|| RemoteError::Failed { message: "the engine returned no vector".into() })
    }
}

/// catalog/1 from the coordinator, republished when it changes (checked every `every`). The
/// task stops with the feed's last holder.
pub fn catalog_feed(coord: Arc<Coordinator>, auth: Arc<dyn Authorize>, every: Duration) -> Arc<CatalogFeed> {
    let feed = Arc::new(CatalogFeed::new(auth));
    let weak = Arc::downgrade(&feed);
    tokio::spawn(async move {
        let mut last = None;
        loop {
            let Some(feed) = weak.upgrade() else { return };
            let models = wire::catalog_to_agent(&coord.merged_catalog(), coord.options().standin.as_deref());
            if last.as_ref() != Some(&models) {
                feed.publish(models.clone());
                last = Some(models);
            }
            drop(feed);
            tokio::time::sleep(every).await;
        }
    });
    feed
}

/// Admission for this inference node: `slots` requests at once, `queue` more waiting.
pub fn admission(slots: usize, queue: usize) -> Admission {
    Admission::with_queue(slots.max(1), queue)
}

/// Serves the inference role on `node`: infer/1, infer/2 and catalog/1 to the user's linked
/// devices, announced as an inference node. Dropping the server stops serving.
pub fn serve(node: &Arc<SyncNode>, coord: Arc<Coordinator>, m: &Membership, admission: Admission) -> Result<(Server, Arc<CatalogFeed>), RemoteError> {
    let auth = roster_auth(m);
    let feed = catalog_feed(coord.clone(), auth.clone(), Duration::from_secs(1));
    let host: Arc<dyn InferHost> = Arc::new(InferenceHost::new(coord, auth));
    let server = Server::start(node, Serving { infer: Some((host, admission)), catalog: Some(feed.clone()), announce: true })?;
    Ok((server, feed))
}

/// The app's pool over Weirdnet: the inference nodes among the user's linked devices.
pub struct RemotePool {
    pool: AgentPool,
    watch: CatalogWatch,
}

#[derive(Debug, Clone)]
pub struct RemoteOptions {
    /// How long to wait to reach and identify each linked device when looking for inference nodes.
    pub find_wait: Duration,
    /// How long an inference node has to accept a request.
    pub accept_timeout: Duration,
}

impl Default for RemoteOptions {
    fn default() -> Self {
        Self { find_wait: Duration::from_secs(3), accept_timeout: Duration::from_secs(10) }
    }
}

/// Cancels the remote request when the bridge drops the stream (the client went away).
struct CancelOnDrop(CancelToken);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

fn requested(model: &Option<String>) -> Option<&str> {
    model.as_deref().filter(|m| !matches!(*m, "default" | "ymir"))
}

fn usage_of(u: v2::Usage) -> Usage {
    Usage { prompt_tokens: u.prompt_tokens, completion_tokens: u.completion_tokens }
}

type PoolTask = tokio::task::JoinHandle<Result<weirdnet_agent::pool::Answer, weirdnet_agent::pool::Interrupted>>;

struct Streaming {
    rx: mpsc::UnboundedReceiver<PoolEvent>,
    task: Option<PoolTask>,
    offset: u64,
    reasoning: u64,
    calls: u32,
    usage: Option<Usage>,
    _guard: CancelOnDrop,
}

async fn next_event(mut st: Streaming) -> Option<(Result<Event, InferError>, Streaming)> {
    loop {
        if let Some(ev) = st.rx.recv().await {
            let out = match ev {
                PoolEvent::Text(t) => {
                    let o = st.offset;
                    st.offset += t.len() as u64;
                    Event::Delta { offset: o, text: t }
                }
                PoolEvent::Reasoning(t) => {
                    let o = st.reasoning;
                    st.reasoning += t.len() as u64;
                    Event::Reasoning { offset: o, text: t }
                }
                PoolEvent::ToolCall(c) => {
                    let index = st.calls;
                    st.calls += 1;
                    Event::ToolCall { index, id: Some(c.id), name: Some(c.name), arguments: wire::arguments_text(&c.arguments) }
                }
                PoolEvent::Usage(u) => {
                    st.usage = Some(usage_of(u));
                    continue;
                }
                _ => continue,
            };
            return Some((Ok(out), st));
        }
        let task = st.task.take()?;
        let ev = match task.await {
            Ok(Ok(a)) => Ok(Event::Done { finish_reason: a.finish_reason, usage: a.usage.map(usage_of).or(st.usage) }),
            Ok(Err(i)) => Err(wire::error_from_remote(i.error)),
            Err(e) => Err(InferError::Unavailable(format!("the pool task failed: {e}"))),
        };
        return Some((ev, st));
    }
}

impl RemotePool {
    /// A pool over `node` for `m`'s devices. `hints` are addresses to reach devices at (the
    /// config's `peers`; LAN discovery and the relays find the rest).
    pub fn connect(node: Arc<SyncNode>, m: &Membership, hints: &[(PeerId, Vec<Multiaddr>)], opts: RemoteOptions) -> Self {
        let dialer = NetDialer::new(node.clone());
        let mut nodes = LinkedNodes::new(node, m.roster_view()).with_wait(opts.find_wait);
        for (peer, addrs) in hints {
            dialer.hint(*peer, addrs.clone());
            nodes = nodes.with_hint(*peer, addrs.clone());
        }
        let po = PoolOptions::default().with_cert(Some(encode_cert(&m.our_cert()))).with_accept_timeout(opts.accept_timeout);
        let pool = AgentPool::new(Arc::new(dialer), nodes).with_options(po);
        let watch = pool.subscribe();
        Self { pool, watch }
    }

    pub fn agent_pool(&self) -> &AgentPool {
        &self.pool
    }

    async fn chat_remote(&self, req: ChatRequest) -> Result<Reply, InferError> {
        let (r, dropped) = wire::to_agent(&req)?;
        if !dropped.is_empty() {
            tracing::debug!(?dropped, "OpenAI parameters infer/2 has no field for; dropped");
        }
        let (tx, mut rx) = mpsc::unbounded_channel::<PoolEvent>();
        let cancel = CancelToken::new();
        let pool = self.pool.clone();
        let c = cancel.clone();
        let mut task: PoolTask = tokio::spawn(async move {
            let sink: &dyn EventSink = &tx;
            pool.infer(r, sink, &c).await
        });
        let guard = CancelOnDrop(cancel);
        let wanted = requested(&req.model).map(str::to_string);
        // Wait for a node to take it, so the bridge can name the answering model.
        let (node, model) = loop {
            match rx.recv().await {
                Some(PoolEvent::Accepted { node, model, .. }) => break (node, model),
                Some(_) => continue,
                None => {
                    // Over before anything was accepted: a failure, or an answer with no frames.
                    return match (&mut task).await {
                        Ok(Ok(a)) => Ok(answer_reply(a, wanted)),
                        Ok(Err(i)) => Err(wire::error_from_remote(i.error)),
                        Err(e) => Err(InferError::Unavailable(format!("the pool task failed: {e}"))),
                    };
                }
            }
        };
        let standin = wanted.filter(|w| *w != model).map(|_| model.clone());
        let st = Streaming { rx, task: Some(task), offset: 0, reasoning: 0, calls: 0, usage: None, _guard: guard };
        Ok(Reply { model, node, cache_hit: false, standin, events: Box::pin(futures::stream::unfold(st, next_event)) })
    }
}

/// A whole answer as a reply (when it finished before any frame reached us).
fn answer_reply(a: weirdnet_agent::pool::Answer, wanted: Option<String>) -> Reply {
    let mut events = Vec::new();
    if !a.reasoning.is_empty() {
        events.push(Ok(Event::Reasoning { offset: 0, text: a.reasoning.clone() }));
    }
    if !a.text.is_empty() {
        events.push(Ok(Event::Delta { offset: 0, text: a.text.clone() }));
    }
    for (i, c) in a.tool_calls.iter().enumerate() {
        events.push(Ok(Event::ToolCall {
            index: i as u32,
            id: Some(c.id.clone()),
            name: Some(c.name.clone()),
            arguments: wire::arguments_text(&c.arguments),
        }));
    }
    events.push(Ok(Event::Done { finish_reason: a.finish_reason.clone(), usage: a.usage.map(usage_of) }));
    let standin = a.meta.standin.clone().or_else(|| wanted.filter(|w| *w != a.model).map(|_| a.model.clone()));
    Reply { model: a.model, node: a.node, cache_hit: a.meta.cache_hit, standin, events: Box::pin(futures::stream::iter(events)) }
}

impl Pool for RemotePool {
    fn catalog(&self) -> BoxFuture<'_, Result<Vec<CatalogModel>, InferError>> {
        Box::pin(async move {
            if let Some((node, c)) = self.watch.current() {
                return Ok(wire::catalog_from_agent(&node, &c.models));
            }
            match self.pool.catalog().await {
                Ok(c) => Ok(wire::catalog_from_agent("weirdnet", &c.models)),
                Err(e) => Err(wire::error_from_remote(e)),
            }
        })
    }

    fn chat(&self, req: ChatRequest) -> BoxFuture<'_, Result<Reply, InferError>> {
        Box::pin(self.chat_remote(req))
    }

    fn complete(&self, _req: CompletionRequest) -> BoxFuture<'_, Result<Reply, InferError>> {
        Box::pin(async { Err(InferError::Unavailable("raw completions are local only: infer/2 carries chat; use /v1/chat/completions".into())) })
    }

    fn embed(&self, req: EmbedRequest) -> BoxFuture<'_, Result<Embeddings, InferError>> {
        Box::pin(async move {
            let mut r = v2::EmbedRequest::new(req.input);
            if let Some(m) = req.model.filter(|m| !matches!(m.as_str(), "default" | "ymir")) {
                r = r.with_model(m);
            }
            let e = self.pool.embed(r, &CancelToken::new()).await.map_err(wire::error_from_remote)?;
            // infer/2 reports no token count for embeddings.
            Ok(Embeddings { model: e.model, node: e.node, vectors: e.vectors, prompt_tokens: 0 })
        })
    }
}
