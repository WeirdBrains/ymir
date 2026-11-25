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
use weirdnet_agent::remote::{Authorize, Caller, Dialer, RemoteError};
use weirdnet_agent::serve::{RosterAuth, Server, Serving};
use weirdnet_backends::models::{ModelsClient, ModelsServer, ModelsService};
use weirdnet_backends::node::{Capability, MODEL_NODE_APP, NodeClient, NodeServer, NodeServerOptions, NodeService, Update, Verifier};
use weirdnet_backends::{Chunk, ChunkStream, EngineKind, ModelHost, ModelStatus, RawInferDialer, Supervisor};
use weirdnet_net::{Multiaddr, PeerId, WEIRDNET};

use crate::context::ContextStore;
use crate::coordinator::Coordinator;
use crate::host::{ContextAccess, ContextSync, InferenceHost, LiveCatalog, catalog_feed};
use crate::membership::{Membership, SyncNode};
use crate::model_node::{BoxFuture, Embedded, ModelNode};
use crate::models::{Directory, ModelsHub, Target};

/// Addresses to reach devices at (the config's `peers`; LAN discovery and relays find the rest).
pub type Hints = Vec<(PeerId, Vec<Multiaddr>)>;

/// Only the user's linked devices.
pub fn roster_auth(m: &Membership) -> Arc<dyn Authorize> {
    Arc::new(RosterAuth::new(&WEIRDNET, m.user_key(), m.roster_view()))
}

/// Linked devices, narrowed to `allow` (peer ids only) when it is not empty: a model node serves the
/// inference nodes, not every device of the user's (an app must go through an inference node and its
/// cache, batch caps and fairness). Revoked and unlinked devices stay refused either way. Only the peer
/// id is matched: it is covered by the certificate signature, while a roster label is not (so a label
/// can be changed without breaking the cert). An entry that is not a peer id matches nothing.
pub fn role_auth(m: &Membership, allow: &[String]) -> Arc<dyn Authorize> {
    let base = roster_auth(m);
    if allow.is_empty() {
        return base;
    }
    let allow = allow.to_vec();
    Arc::new(move |caller: &Caller| {
        base.authorize(caller)?;
        if allow.contains(&caller.peer) { Ok(()) } else { Err(RemoteError::refused("not-an-allowed-caller")) }
    })
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
#[allow(clippy::too_many_arguments)]
pub fn serve_model(
    node: &Arc<SyncNode>,
    supervisor: Supervisor,
    m: &Membership,
    slots: usize,
    queue: usize,
    hardware: Option<(weirdnet_model::hw::Hardware, weirdnet_model::hw::Budget)>,
    combined: bool,
    workers: Option<Arc<dyn weirdnet_backends::node::GroupWorkers>>,
    callers: &[String],
) -> Result<(NodeServer, Arc<NodeService>), RemoteError> {
    let auth = role_auth(m, callers);
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

/// Whether `peer` announces [`MODEL_NODE_APP`]: connects to it first if needed and waits up to `wait`
/// to learn what it announces.
pub async fn announces_model_node(node: &Arc<SyncNode>, peer: PeerId, addrs: Vec<Multiaddr>, wait: Duration) -> bool {
    let n = node.clone();
    tokio::task::spawn_blocking(move || {
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
    .unwrap_or(false)
}

/// Serves `/weirdnet-backends/models/1` on `node`: the owner (the user's linked, unrevoked devices;
/// [`roster_auth`]) searches the Hugging Face Hub, sees what fits this node, pulls models into
/// `store` (and serves them through `supervisor` without a restart when the request says so),
/// lists and removes what is on disk, and takes uploads under `policy` (any linked device; the
/// free space to keep). Nothing here edits `node.toml` (an upload's entry is saved by
/// `/ymir/uploads/1`, [`serve_uploads`]). Dropping the server stops it.
pub fn serve_models(
    node: &Arc<SyncNode>,
    supervisor: Supervisor,
    m: &Membership,
    store: weirdnet_model::ModelStore,
    budget: weirdnet_model::hw::Budget,
    engine: EngineKind,
    policy: weirdnet_backends::models::WritePolicy,
) -> Result<(ModelsServer, Arc<ModelsService>), RemoteError> {
    let service = Arc::new(ModelsService::new(store, supervisor, roster_auth(m)).with_budget(budget).with_engine(engine).with_write_policy(policy));
    let server = ModelsServer::start(node, service.clone())?;
    Ok((server, service))
}

/// The devices on `m`'s roster, and which of them announce a model node.
pub struct NetDirectory {
    node: Arc<SyncNode>,
    membership: Arc<Membership>,
    hints: HashMap<PeerId, Vec<Multiaddr>>,
    wait: Duration,
    /// This device's name for itself (the roster label may differ).
    me: String,
}

impl NetDirectory {
    pub fn new(node: Arc<SyncNode>, membership: Arc<Membership>, hints: Hints, wait: Duration, me: impl Into<String>) -> Self {
        Self { node, membership, hints: hints.into_iter().collect(), wait, me: me.into() }
    }
}

impl Directory for NetDirectory {
    fn devices(&self) -> Vec<Target> {
        let me = self.node.peer_id().to_string();
        self.membership
            .roster()
            .active()
            .map(|c| Target {
                peer: c.peer_id.clone(),
                label: if c.peer_id == me { self.me.clone() } else { c.label.clone() },
                local: c.peer_id == me,
            })
            .collect()
    }

    fn model_nodes(&self) -> futures::future::BoxFuture<'_, Vec<Target>> {
        Box::pin(async move {
            let peers: Vec<Target> = self.devices().into_iter().filter(|t| !t.local).collect();
            let checks = peers.iter().map(|t| async {
                let Ok(peer) = PeerId::from_str(&t.peer) else { return false };
                announces_model_node(&self.node, peer, self.hints.get(&peer).cloned().unwrap_or_default(), self.wait).await
            });
            let found = futures::future::join_all(checks).await;
            peers.into_iter().zip(found).filter_map(|(t, ok)| ok.then_some(t)).collect()
        })
    }
}

/// The `ymird` side of `ymirctl models --node`: this node's own model role (when it has one) in
/// process, the others over models/1 presenting this device's certificate.
pub fn models_hub(
    node: &Arc<SyncNode>,
    m: &Arc<Membership>,
    hints: &Hints,
    opts: &RemoteOptions,
    name: &str,
    local: Option<Arc<ModelsService>>,
) -> ModelsHub {
    let client = ModelsClient::new(Arc::new(dialer(node, hints))).with_cert(Some(encode_cert(&m.our_cert())));
    let dir = Arc::new(NetDirectory::new(node.clone(), m.clone(), hints.clone(), opts.find_wait, name));
    let me = Target { peer: node.peer_id().to_string(), label: name.to_string(), local: true };
    let uploads = crate::upload_net::UploadsClient::new(Arc::new(dialer(node, hints)), Some(encode_cert(&m.our_cert())));
    let rollouts = crate::rollouts_net::RolloutsClient::new(Arc::new(dialer(node, hints)), Some(encode_cert(&m.our_cert())));
    ModelsHub::new(local.map(|s| (me, s)), dir, client).with_uploads(uploads).with_rollouts(rollouts)
}

/// The `ymird` side of `ymirctl train run --node`: runs on the user's other model nodes over
/// `/ymir/train/1`, presenting this device's certificate.
pub fn train_hub(node: &Arc<SyncNode>, m: &Arc<Membership>, hints: &Hints, opts: &RemoteOptions, name: &str) -> crate::train_net::TrainHub {
    let dir = Arc::new(NetDirectory::new(node.clone(), m.clone(), hints.clone(), opts.find_wait, name));
    crate::train_net::TrainHub::new(dir, Arc::new(dialer(node, hints)), Some(encode_cert(&m.our_cert())))
}

/// Serves `/ymir/train/1` on `node` (`[train] remote = true`): runs for the user's linked devices,
/// narrowed to `[train] callers`.
pub fn serve_train(
    node: &Arc<SyncNode>,
    m: &Membership,
    paths: crate::Paths,
    cfg: crate::NodeConfig,
) -> Result<crate::train_net::TrainServer, RemoteError> {
    let auth = role_auth(m, &cfg.train.callers);
    let service = Arc::new(crate::train_net::TrainService::new(paths, cfg, auth));
    crate::train_net::TrainServer::start(node, service)
}

/// Serves `/ymir/uploads/1` on `node` next to models/1: after an upload, the uploader's `ymird`
/// has this node record the entry's revision and (unless `--no-persist`) save it in node.toml.
/// Any linked device, like models/1.
pub fn serve_uploads(
    node: &Arc<SyncNode>,
    m: &Membership,
    service: crate::upload_net::UploadsService,
) -> Result<crate::upload_net::UploadsServer, RemoteError> {
    crate::upload_net::UploadsServer::start(node, Arc::new(service.with_auth(roster_auth(m))))
}

/// Serves `/ymir/rollouts/1` on an inference node: the owner's linked devices (narrowed to
/// `allow`, `[inference] rollout_callers`) read and change its rollouts.
pub fn serve_rollouts(
    node: &Arc<SyncNode>,
    m: &Membership,
    allow: &[String],
    service: crate::rollouts_net::RolloutsService,
) -> Result<crate::rollouts_net::RolloutsServer, RemoteError> {
    crate::rollouts_net::RolloutsServer::start(node, Arc::new(service.with_auth(role_auth(m, allow))))
}

/// When a model node's stream counts as stalled: no frame for 5 s, or 3x the recent gap between
/// frames, up to 12 s; before the first frame, `first` (the prompt's processing time, from
/// [`crate::coordinator::prefill_allowance`]) when longer.
pub fn model_node_stall(first: Duration) -> weirdnet_agent::pool::Stall {
    weirdnet_agent::pool::Stall::new(Duration::from_secs(5), 3, Duration::from_secs(12).max(first))
}

/// A model node on another of the user's devices.
pub struct RemoteModelNode {
    peer: String,
    client: Arc<NodeClient>,
    pool: Pool,
    record: Arc<RwLock<Option<Capability>>>,
    follow: tokio::task::JoinHandle<()>,
    roster: Arc<dyn Fn() -> weirdnet_sync::Roster + Send + Sync>,
}

impl Drop for RemoteModelNode {
    fn drop(&mut self) {
        self.follow.abort();
    }
}

/// Whether `peer` holds an active (unrevoked) cert in the user's roster. `weirdnet-backends` 0.1.1's
/// `Verifier` checks the cert's signature but not the roster, so a revoked model node's old cert
/// still verifies; this closes that until the `Verifier` does (0.1.2).
fn on_roster(roster: &dyn Fn() -> weirdnet_sync::Roster, peer: &str) -> Result<(), RemoteError> {
    if roster().get(peer).is_some_and(|c| c.active()) { Ok(()) } else { Err(RemoteError::refused("model-node-not-in-roster")) }
}

impl RemoteModelNode {
    /// Refused unless the node is on the active roster now (checked on connect, on every request and
    /// on every record update).
    fn permitted(&self) -> Result<(), RemoteError> {
        on_roster(&*self.roster, &self.peer)
    }

    /// Reads `peer`'s capability record, then follows it.
    pub async fn connect(dialer: Arc<dyn Dialer>, peer: &str, m: &Membership, opts: &RemoteOptions) -> Result<Self, RemoteError> {
        let roster = m.roster_view();
        on_roster(&*roster, peer)?;
        let cert = Some(encode_cert(&m.our_cert()));
        let client = Arc::new(NodeClient::new(dialer.clone(), Verifier::new(&WEIRDNET, m.user_key())).with_cert(cert.clone()));
        let first = tokio::time::timeout(opts.accept_timeout, client.record(peer)).await.unwrap_or(Err(RemoteError::Timeout))?;
        let record = Arc::new(RwLock::new(Some(first)));
        let (c, p, r, rv) = (client.clone(), peer.to_string(), record.clone(), roster.clone());
        let follow = tokio::spawn(async move {
            loop {
                match c.subscribe(&p).await {
                    Ok(mut sub) => {
                        while let Some(u) = sub.next().await {
                            // A revoked node's records are dropped, not applied.
                            if on_roster(&*rv, &p).is_err() {
                                *r.write().unwrap_or_else(|e| e.into_inner()) = None;
                                break;
                            }
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
        let po = PoolOptions::default()
            .with_cert(cert)
            .with_accept_timeout(opts.accept_timeout)
            .with_infer_v1_fallback(false)
            .with_stall(model_node_stall(Duration::ZERO));
        // Raw inference goes to the model node's own protocol id, falling back to infer/2 for a node
        // on backends 0.1.0.
        let pool = Pool::new(Arc::new(RawInferDialer::new(dialer)), vec![peer.to_string()]).with_options(po);
        Ok(Self { peer: peer.to_string(), client, pool, record, follow, roster })
    }

    /// The node's record, while it is current.
    pub fn record(&self) -> Option<Capability> {
        self.permitted().ok()?;
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

    fn revision(&self, model: &str) -> Option<String> {
        self.record()?.models.into_iter().find(|m| m.status.id == model)?.revision
    }

    fn load<'a>(&'a self, model: &'a str, force: bool) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(async move {
            self.permitted()?;
            self.client.load(&self.peer, model, force).await
        })
    }

    fn unload<'a>(&'a self, model: &'a str) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(async move {
            self.permitted()?;
            self.client.unload(&self.peer, model).await
        })
    }

    fn draining(&self) -> bool {
        self.record().is_some_and(|c| c.draining || c.reserved.is_some())
    }

    fn reserved(&self) -> Option<String> {
        self.record().and_then(|c| c.reserved.map(|r| r.holder))
    }

    /// A node before `weirdnet-backends` 0.4.3 can't decode the command, so the call fails.
    fn reserve<'a>(&'a self, holder: &'a str) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(async move {
            self.permitted()?;
            self.client.reserve(&self.peer, holder).await?;
            if let Some(c) = self.record.write().unwrap_or_else(|e| e.into_inner()).as_mut() {
                c.reserved = Some(weirdnet_backends::Reservation::new(holder));
            }
            Ok(())
        })
    }

    fn release<'a>(&'a self, holder: &'a str) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(async move {
            self.permitted()?;
            self.client.release(&self.peer, holder).await?;
            if let Some(c) = self.record.write().unwrap_or_else(|e| e.into_inner()).as_mut() {
                c.reserved = None;
            }
            Ok(())
        })
    }

    fn group_jobs(&self) -> bool {
        self.record().is_some_and(|c| c.group_jobs)
    }

    fn group_up<'a>(&'a self, plan: &'a weirdnet_backends::node::GroupPlan) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(async move {
            self.permitted()?;
            self.client.group_up(&self.peer, plan.clone()).await
        })
    }

    fn group_down<'a>(&'a self, group: &'a str) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(async move {
            self.permitted()?;
            self.client.group_down(&self.peer, group).await
        })
    }

    /// A model node continues an assistant prefix (llama.cpp prefill, vLLM `continue_final_message`),
    /// and `weirdnet-agent` 0.4.2's pool carries one on a first attempt.
    fn resumes(&self) -> bool {
        true
    }

    fn drain<'a>(&'a self, on: bool) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(async move {
            self.permitted()?;
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
            self.permitted()?;
            let mut r = req.clone();
            r.model = Some(model.to_string());
            let (tx, rx) = mpsc::unbounded_channel::<PoolEvent>();
            let cancel = CancelToken::new();
            // The model node sends nothing while its engine processes the prompt: wait for the
            // first frame as long as the prompt may take (this node keeps its own asker alive
            // meanwhile, `Coordinator`), and after it as before.
            let opts = self.pool.options().clone().with_stall(model_node_stall(crate::coordinator::prefill_allowance(&r)));
            let (pool, c) = (self.pool.clone().with_options(opts), cancel.clone());
            let task = tokio::spawn(async move {
                let sink: &dyn EventSink = &tx;
                pool.infer(r, sink, &c).await
            });
            Ok(chunks(rx, task, cancel))
        })
    }

    fn embed<'a>(&'a self, model: &'a str, input: &'a [String]) -> BoxFuture<'a, Result<Embedded, RemoteError>> {
        Box::pin(async move {
            self.permitted()?;
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
                let announced = announces_model_node(&node, peer, hints.get(&peer).cloned().unwrap_or_default(), opts.find_wait).await;
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
