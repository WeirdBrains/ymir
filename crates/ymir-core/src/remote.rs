//! The roles over Weirdnet.
//!
//! - [`serve`]: the inference role answers the user's devices on `/weirdnet-agent/infer/2`,
//!   infer/1 and `/weirdnet-agent/catalog/1` ([`crate::host::InferenceHost`]), announced as an
//!   inference node.
//! - [`connect_pool`]: the app role's way in: `weirdnet-agent`'s pool client over the inference
//!   nodes among the user's linked devices (failover, resume, stall detection), which the
//!   `weirdnet-openai` bridge serves on loopback.
//! - [`serve_model`] and [`RemoteModelNode`]: the model role on another device, over
//!   `/weirdnet-backends/node/1` (signed capability records, `Load`, `Unload`) and raw infer/2;
//!   [`discover_model_nodes`] adds the ones among the user's devices to the coordinator.
//!
//! Every call is checked against the user's roster ([`RosterAuth`]): only the user's linked,
//! unrevoked devices get an answer, and capability records must carry the user's certificate.

use std::collections::HashMap;
use std::str::FromStr;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use tokio::sync::mpsc;
use weirdnet_agent::CancelToken;
use weirdnet_agent::pool::{EventSink, LinkedNodes, NetDialer, Pool, PoolEvent, PoolOptions, encode_cert};
use weirdnet_agent::remote::infer::v2::{EmbedRequest, Request};
use weirdnet_agent::remote::infer::{Admission, InferHost};
use weirdnet_agent::remote::{Authorize, Dialer, RemoteError};
use weirdnet_agent::serve::{RosterAuth, Server, Serving};
use weirdnet_backends::node::{Capability, MODEL_NODE_APP, NodeClient, NodeServer, NodeServerOptions, NodeService, Update, Verifier};
use weirdnet_backends::{Chunk, ChunkStream, ModelHost, ModelStatus, RawInferDialer, Supervisor};
use weirdnet_net::{Multiaddr, PeerId, WEIRDNET};

use crate::context::ContextStore;
use crate::coordinator::Coordinator;
use crate::host::{ContextAccess, ContextSync, InferenceHost, LiveCatalog, catalog_feed};
use crate::membership::{Membership, SyncNode};
use crate::model_node::{BoxFuture, Embedded, ModelNode};

/// Addresses to reach devices at (the config's `peers`; LAN discovery and relays find the rest).
pub type Hints = Vec<(PeerId, Vec<Multiaddr>)>;

/// Only the user's linked devices.
pub fn roster_auth(m: &Membership) -> Arc<dyn Authorize> {
    Arc::new(RosterAuth::new(&WEIRDNET, m.user_key(), m.roster_view()))
}

/// Serves the inference role on `node`: infer/1, infer/2 and catalog/1 to the user's linked
/// devices, announced as an inference node. Dropping the server stops serving.
pub fn serve(
    node: &Arc<SyncNode>,
    coord: Arc<Coordinator>,
    m: &Arc<Membership>,
    hints: Hints,
    store: Option<ContextStore>,
    admission: Admission,
) -> Result<(Server, Arc<LiveCatalog>), RemoteError> {
    let auth = roster_auth(m);
    let feed = catalog_feed(coord.clone(), auth.clone(), Duration::from_secs(1));
    let access = store.map(|store| ContextAccess { store, sync: Some(ContextSync { membership: m.clone(), node: Arc::downgrade(node), hints }) });
    let host: Arc<dyn InferHost> = Arc::new(InferenceHost::new(coord, auth).with_context(access));
    let server = Server::start(node, Serving { infer: Some((host, admission)), catalog: Some(feed.clone()), announce: true })?;
    Ok((server, feed))
}

#[derive(Debug, Clone)]
pub struct RemoteOptions {
    /// How long to wait to reach and identify each linked device when looking for nodes.
    pub find_wait: Duration,
    /// How long a node has to accept a request.
    pub accept_timeout: Duration,
}

impl Default for RemoteOptions {
    fn default() -> Self {
        Self { find_wait: Duration::from_secs(3), accept_timeout: Duration::from_secs(10) }
    }
}

fn dialer(node: &Arc<SyncNode>, hints: &[(PeerId, Vec<Multiaddr>)]) -> NetDialer<Vec<u8>, Vec<u8>> {
    let d = NetDialer::new(node.clone());
    for (peer, addrs) in hints {
        d.hint(*peer, addrs.clone());
    }
    d
}

/// The app's pool over Weirdnet: the inference nodes among `m`'s linked devices, presenting this
/// device's certificate.
pub fn connect_pool(node: Arc<SyncNode>, m: &Membership, hints: &[(PeerId, Vec<Multiaddr>)], opts: &RemoteOptions) -> Pool {
    let d = dialer(&node, hints);
    let mut nodes = LinkedNodes::new(node, m.roster_view()).with_wait(opts.find_wait);
    for (peer, addrs) in hints {
        nodes = nodes.with_hint(*peer, addrs.clone());
    }
    let po = PoolOptions::default().with_cert(Some(encode_cert(&m.our_cert()))).with_accept_timeout(opts.accept_timeout);
    Pool::new(Arc::new(d), nodes).with_options(po)
}

/// Serves the model role on `node` over node/1 and raw inference (`/weirdnet-backends/infer/1`),
/// to the user's devices. On a box that also has the inference role (`combined`), the model
/// role leaves `/weirdnet-agent/infer/2` to the inference role, so one box serves both. The
/// inference role uses its own models in process.
/// `hardware` goes into the capability record (for placement).
pub fn serve_model(
    node: &Arc<SyncNode>,
    supervisor: Supervisor,
    m: &Membership,
    slots: usize,
    queue: usize,
    hardware: Option<(weirdnet_model::hw::Hardware, weirdnet_model::hw::Budget)>,
    combined: bool,
    workers: Option<Arc<dyn weirdnet_backends::node::GroupWorkers>>,
) -> Result<(NodeServer, Arc<NodeService>), RemoteError> {
    let auth = roster_auth(m);
    let host = Arc::new(ModelHost::new(supervisor, auth.clone()));
    let mut service = NodeService::new(host, m.node_key().clone(), Some(m.our_cert()), auth).with_admission(slots.max(1), queue);
    if let Some((hw, budget)) = hardware {
        service = service.with_hardware(hw, budget);
    }
    if let Some(w) = workers {
        service = service.with_group_workers(w);
    }
    let service = Arc::new(service);
    let server = NodeServer::start_with(node, service.clone(), NodeServerOptions::default().with_legacy_infer2(!combined))?;
    Ok((server, service))
}

/// A model node on another of the user's devices.
pub struct RemoteModelNode {
    peer: String,
    client: Arc<NodeClient>,
    pool: Pool,
    record: Arc<RwLock<Option<Capability>>>,
    follow: tokio::task::JoinHandle<()>,
}

impl Drop for RemoteModelNode {
    fn drop(&mut self) {
        self.follow.abort();
    }
}

impl RemoteModelNode {
    /// Reads `peer`'s capability record, then follows it.
    pub async fn connect(dialer: Arc<dyn Dialer>, peer: &str, m: &Membership, opts: &RemoteOptions) -> Result<Self, RemoteError> {
        let cert = Some(encode_cert(&m.our_cert()));
        let client = Arc::new(NodeClient::new(dialer.clone(), Verifier::new(&WEIRDNET, m.user_key())).with_cert(cert.clone()));
        let first = tokio::time::timeout(opts.accept_timeout, client.record(peer)).await.unwrap_or(Err(RemoteError::Timeout))?;
        let record = Arc::new(RwLock::new(Some(first)));
        let (c, p, r) = (client.clone(), peer.to_string(), record.clone());
        let follow = tokio::spawn(async move {
            loop {
                match c.subscribe(&p).await {
                    Ok(mut sub) => {
                        while let Some(u) = sub.next().await {
                            match u {
                                Ok(Update::Record(cap)) => *r.write().unwrap_or_else(|e| e.into_inner()) = Some(*cap),
                                Ok(Update::Withdrawn) => *r.write().unwrap_or_else(|e| e.into_inner()) = None,
                                Ok(_) => {}
                                Err(e) => {
                                    tracing::debug!(peer = %p, "model node records: {e}");
                                    break;
                                }
                            }
                        }
                    }
                    // Can't reach the node again: drop its record now (rather than when it lapses,
                    // up to 90 s later) so routing and discovery stop counting on it. It comes
                    // back with its next record if it returns.
                    Err(e) => {
                        tracing::debug!(peer = %p, "model node unreachable: {e}");
                        *r.write().unwrap_or_else(|e| e.into_inner()) = None;
                    }
                }
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
        });
        // A model node that stops talking is given up on sooner than the asker gives up on this inference
        // node (its pool stalls at 15 s or more), so the answer can move to another model node here
        // first instead of the asker's pool timing out on a silent stream.
        let stall = weirdnet_agent::pool::Stall::new(Duration::from_secs(5), 3, Duration::from_secs(12));
        let po = PoolOptions::default().with_cert(cert).with_accept_timeout(opts.accept_timeout).with_infer_v1_fallback(false).with_stall(stall);
        // Raw inference goes to the model node's own protocol id, falling back to infer/2 for a node
        // on backends 0.1.0.
        let pool = Pool::new(Arc::new(RawInferDialer::new(dialer)), vec![peer.to_string()]).with_options(po);
        Ok(Self { peer: peer.to_string(), client, pool, record, follow })
    }

    /// The node's record, while it is current.
    pub fn record(&self) -> Option<Capability> {
        let now = weirdnet_net::now_ms();
        self.record.read().unwrap_or_else(|e| e.into_inner()).clone().filter(|c| c.expires_ms > now)
    }

    pub fn alive(&self) -> bool {
        self.record().is_some()
    }
}

impl ModelNode for RemoteModelNode {
    fn name(&self) -> &str {
        &self.peer
    }

    fn status(&self) -> Vec<ModelStatus> {
        self.record().map(|c| c.models.into_iter().map(|m| m.status).collect()).unwrap_or_default()
    }

    fn load<'a>(&'a self, model: &'a str, force: bool) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(self.client.load(&self.peer, model, force))
    }

    fn unload<'a>(&'a self, model: &'a str) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(self.client.unload(&self.peer, model))
    }

    fn draining(&self) -> bool {
        self.record().is_some_and(|c| c.draining)
    }

    fn group_jobs(&self) -> bool {
        self.record().is_some_and(|c| c.group_jobs)
    }

    fn group_up<'a>(&'a self, plan: &'a weirdnet_backends::node::GroupPlan) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(self.client.group_up(&self.peer, plan.clone()))
    }

    fn group_down<'a>(&'a self, group: &'a str) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(self.client.group_down(&self.peer, group))
    }

    /// A model node continues an assistant prefix (llama.cpp prefill, vLLM `continue_final_message`),
    /// and `weirdnet-agent` 0.4.2's pool carries one on a first attempt.
    fn resumes(&self) -> bool {
        true
    }

    fn drain<'a>(&'a self, on: bool) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(async move {
            self.client.drain(&self.peer, on).await?;
            // The record catches up with the next update; take the flag now so routing agrees.
            if let Some(c) = self.record.write().unwrap_or_else(|e| e.into_inner()).as_mut() {
                c.draining = on;
            }
            Ok(())
        })
    }

    fn chat<'a>(&'a self, model: &'a str, req: &'a Request) -> BoxFuture<'a, Result<ChunkStream, RemoteError>> {
        Box::pin(async move {
            let mut r = req.clone();
            r.model = Some(model.to_string());
            let (tx, rx) = mpsc::unbounded_channel::<PoolEvent>();
            let cancel = CancelToken::new();
            let (pool, c) = (self.pool.clone(), cancel.clone());
            let task = tokio::spawn(async move {
                let sink: &dyn EventSink = &tx;
                pool.infer(r, sink, &c).await
            });
            Ok(chunks(rx, task, cancel))
        })
    }

    fn embed<'a>(&'a self, model: &'a str, input: &'a [String]) -> BoxFuture<'a, Result<Embedded, RemoteError>> {
        Box::pin(async move {
            let e = self.pool.embed(EmbedRequest::new(input.to_vec()).with_model(model), &CancelToken::new()).await?;
            Ok((e.vectors, e.usage.map_or(0, |u| u.prompt_tokens)))
        })
    }
}

type PoolTask = tokio::task::JoinHandle<Result<weirdnet_agent::pool::Answer, weirdnet_agent::pool::Interrupted>>;

/// Cancels the remote request when the stream is dropped.
struct CancelOnDrop(CancelToken);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

/// A pool answer as engine chunks (what the coordinator streams from any model node).
fn chunks(rx: mpsc::UnboundedReceiver<PoolEvent>, task: PoolTask, cancel: CancelToken) -> ChunkStream {
    struct St {
        rx: mpsc::UnboundedReceiver<PoolEvent>,
        task: Option<PoolTask>,
        calls: u32,
        _guard: CancelOnDrop,
    }
    let st = St { rx, task: Some(task), calls: 0, _guard: CancelOnDrop(cancel) };
    Box::pin(futures::stream::unfold(st, |mut st| async move {
        loop {
            if let Some(ev) = st.rx.recv().await {
                let c = match ev {
                    PoolEvent::Text(t) => Chunk::Text(t),
                    PoolEvent::Reasoning(t) => Chunk::Reasoning(t),
                    PoolEvent::ToolCall(c) => {
                        st.calls += 1;
                        Chunk::ToolCall { index: st.calls - 1, id: Some(c.id), name: Some(c.name), arguments: c.arguments.to_string() }
                    }
                    _ => continue,
                };
                return Some((Ok(c), st));
            }
            let task = st.task.take()?;
            let out = match task.await {
                Ok(Ok(a)) => Ok(Chunk::Done { finish_reason: a.finish_reason, usage: a.usage }),
                Ok(Err(i)) => Err(i.error),
                Err(e) => Err(RemoteError::Failed { message: format!("the pool task failed: {e}") }),
            };
            return Some((out, st));
        }
    }))
}

/// Follows the model nodes among `m`'s devices: every `every`, devices that announce
/// [`MODEL_NODE_APP`] join the coordinator; ones whose records lapse leave it. Stops when the
/// handle is aborted.
pub fn discover_model_nodes(
    coord: Arc<Coordinator>,
    node: Arc<SyncNode>,
    m: Arc<Membership>,
    hints: Hints,
    opts: RemoteOptions,
    every: Duration,
) -> tokio::task::JoinHandle<()> {
    let d: Arc<dyn Dialer> = Arc::new(dialer(&node, &hints));
    let hints: HashMap<PeerId, Vec<Multiaddr>> = hints.into_iter().collect();
    tokio::spawn(async move {
        let mut known: HashMap<String, Arc<RemoteModelNode>> = HashMap::new();
        loop {
            let me = node.peer_id().to_string();
            let peers: Vec<String> = m.roster().active().filter(|c| c.peer_id != me).map(|c| c.peer_id.clone()).collect();
            // Revoked or lapsed nodes leave.
            known.retain(|p, n| {
                let keep = peers.contains(p) && n.alive();
                if !keep {
                    coord.remove_node(p);
                    tracing::info!(peer = %p, "model node gone");
                }
                keep
            });
            for p in peers {
                if known.contains_key(&p) {
                    continue;
                }
                let Ok(peer) = PeerId::from_str(&p) else { continue };
                let (n, addrs, wait) = (node.clone(), hints.get(&peer).cloned().unwrap_or_default(), opts.find_wait);
                let announced = tokio::task::spawn_blocking(move || {
                    let mut apps = n.peer_apps(peer);
                    if apps.is_none() {
                        let _ = n.connect(peer, &addrs);
                        let until = std::time::Instant::now() + wait;
                        while apps.is_none() && std::time::Instant::now() < until {
                            std::thread::sleep(Duration::from_millis(25));
                            apps = n.peer_apps(peer);
                        }
                    }
                    apps.is_some_and(|a| a.iter().any(|a| a == MODEL_NODE_APP))
                })
                .await
                .unwrap_or(false);
                if !announced {
                    continue;
                }
                match RemoteModelNode::connect(d.clone(), &p, &m, &opts).await {
                    Ok(rn) => {
                        let rn = Arc::new(rn);
                        tracing::info!(peer = %p, models = rn.status().len(), "model node joined");
                        coord.add_node(rn.clone());
                        known.insert(p, rn);
                    }
                    Err(e) => tracing::debug!(peer = %p, "model node: {e}"),
                }
            }
            tokio::time::sleep(every).await;
        }
    })
}
