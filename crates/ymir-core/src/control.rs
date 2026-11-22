//! `/ymir/control/1`, local transport: JSON lines over a Unix socket in the node's private state
//! directory (mode 0600), or on Windows a local-only named pipe derived from that path (see
//! [`pipe_name`]; the pipe's default DACL lets only its owner, SYSTEM and Administrators write).
//! Over Weirdnet (M3) the same messages go to admin devices on the roster.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use weirdnet_backends::EngineKind;
pub use weirdnet_backends::ModelState;

use crate::{Coordinator, LocalModelNode, ModelNode};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum ControlRequest {
    Status,
    /// An operator's load: also clears a crash loop.
    Load {
        model: String,
        #[serde(default)]
        node: Option<String>,
    },
    Unload {
        model: String,
    },
    /// Stop sending a model node new requests (it finishes what it has), or resume it.
    Drain {
        node: String,
        #[serde(default = "yes")]
        on: bool,
    },
    /// A device this user just linked from here (`ymirctl pool add`) joins the roster.
    PoolApprove {
        cert: weirdnet_net::user::NodeCert,
    },
    /// Revoke a device (`ymirctl pool revoke`).
    PoolRevoke {
        peer: String,
    },
    /// Add a message to a conversation in this node's context store (it syncs to the user's
    /// other inference and app nodes).
    ContextAdd {
        conversation: String,
        #[serde(default = "user")]
        role: String,
        text: String,
    },
    /// The conversations this node holds, or one of them in full.
    ContextShow {
        #[serde(default)]
        conversation: Option<String>,
    },
    /// Tombstone a conversation everywhere.
    ContextDelete {
        conversation: String,
    },
    /// Add `text` to `conversation` as the user and ask the pool to answer it from the
    /// conversation (a `context_ref`): the inference node that answers writes the answer back.
    Chat {
        conversation: String,
        text: String,
        #[serde(default)]
        model: Option<String>,
    },
    /// Split models (M7): bring the group for `model` up or down, or list the groups.
    Group {
        /// `up`, `down` or `status`.
        action: String,
        #[serde(default)]
        model: Option<String>,
    },
    /// Fan `prompts` out over the pool at batch priority and wait for all the answers.
    Batch {
        prompts: Vec<String>,
        #[serde(default)]
        model: Option<String>,
        #[serde(default)]
        concurrency: Option<usize>,
        #[serde(default)]
        retries: Option<u32>,
    },
    /// Fan `tasks` (each with its own messages, sampling and seed) out over the pool at batch
    /// priority and wait for all the answers (`ymirctl batch --jsonl`, in chunks).
    BatchTasks {
        tasks: Vec<crate::batch::BatchTask>,
        #[serde(default)]
        model: Option<String>,
        #[serde(default)]
        concurrency: Option<usize>,
        #[serde(default)]
        retries: Option<u32>,
        /// Job id prefix, stable across chunks and resumed runs (default: a fresh one).
        #[serde(default)]
        job: Option<String>,
        /// Wait for the model to load rather than take a stand-in.
        #[serde(default)]
        wait_for_model: bool,
        #[serde(default)]
        cache: crate::batch::CacheMode,
    },
    /// Manage model files on the user's model nodes over Weirdnet (`ymirctl models --node`). The
    /// answer is a stream of [`crate::models::ModelsLine`]s (a pull's progress, then the result), not
    /// one [`ControlResponse`].
    Models {
        /// A roster label, a peer id, `auto` (best fit, pulls only) or `all`.
        node: String,
        request: crate::models::ModelsOp,
    },
    /// Stop `ymird` cleanly (engines too), as SIGTERM does. Local transport only: this is how
    /// `ymirctl service stop` stops a Windows scheduled task, which has no signal to send.
    Shutdown,
}

fn user() -> String {
    "user".into()
}

fn yes() -> bool {
    true
}

/// A model node, for `ymirctl nodes`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeSummary {
    pub name: String,
    /// This box's own model role.
    #[serde(default)]
    pub local: bool,
    pub draining: bool,
    pub models: usize,
    pub ready: usize,
    pub in_flight: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeModel {
    pub node: String,
    pub id: String,
    pub engine: EngineKind,
    pub state: ModelState,
    pub in_flight: usize,
    pub swap_group: Option<String>,
    pub purpose: String,
    /// The engine process, while one runs.
    pub pid: Option<u32>,
    #[serde(default)]
    pub last_load_secs: Option<f32>,
    /// A stock server Ymir forwards to but doesn't own.
    #[serde(default)]
    pub wrapped: bool,
    /// Engine crashes in the current window.
    #[serde(default)]
    pub crashes: usize,
    /// What the model takes (chat, embeddings, images, audio), when the node reports it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<weirdnet_backends::Capabilities>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatusReport {
    pub node: String,
    pub version: String,
    pub pid: u32,
    pub roles: Vec<String>,
    /// The loopback bridge port, when the app role is on.
    pub bridge_port: Option<u16>,
    pub main_model: Option<String>,
    pub models: Vec<NodeModel>,
    /// The model nodes the inference role routes to (this box's own and the user's other devices').
    #[serde(default)]
    pub nodes: Vec<NodeSummary>,
    /// Alert keys currently failing.
    #[serde(default)]
    pub alerts: Vec<String>,
    /// This device on Weirdnet, once it has an identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub network: Option<NetworkStatus>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NetworkStatus {
    pub peer: String,
    pub user: String,
    pub listening: Vec<String>,
    pub connected: Vec<String>,
    pub devices: Vec<Device>,
}

/// A device in the user's roster.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Device {
    pub peer: String,
    pub label: String,
    pub active: bool,
    /// This device.
    pub me: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ControlResponse {
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<StatusReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat: Option<ChatResult>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conversations: Vec<ConversationView>,
    /// The outcomes of a [`ControlRequest::Batch`], in prompt order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub batch: Vec<crate::batch::BatchItem>,
    /// The split-model groups that are up (`ymirctl group status`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub groups: Vec<crate::group::GroupInfo>,
    /// Groups that went down by themselves, newest last.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub group_losses: Vec<crate::group::GroupLoss>,
}

/// An answer to [`ControlRequest::Chat`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatResult {
    pub text: String,
    /// The inference node that finished it.
    pub node: String,
    pub model: String,
    /// How many times the answer moved to another inference node mid-answer.
    pub resumes: usize,
    /// What happened on the way, in order: `accepted <node>` and `resumed <node> at <offset>`.
    #[serde(default)]
    pub path: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConversationView {
    pub id: String,
    pub title: String,
    pub messages: Vec<(String, String)>,
}

impl ControlResponse {
    fn ok(message: impl Into<String>) -> Self {
        Self {
            ok: true,
            message: Some(message.into()),
            status: None,
            chat: None,
            conversations: Vec::new(),
            batch: Vec::new(),
            groups: Vec::new(),
            group_losses: Vec::new(),
        }
    }
    fn err(message: impl Into<String>) -> Self {
        Self {
            ok: false,
            message: Some(message.into()),
            status: None,
            chat: None,
            conversations: Vec::new(),
            batch: Vec::new(),
            groups: Vec::new(),
            group_losses: Vec::new(),
        }
    }
}

/// What the control socket can reach.
pub struct ControlHandler {
    pub node: String,
    pub roles: Vec<String>,
    pub bridge_port: Option<u16>,
    pub main_model: Option<String>,
    pub coordinator: Option<Arc<Coordinator>>,
    pub model_node: Option<Arc<LocalModelNode>>,
    pub alerter: Option<Arc<weirdnet_alerts::Alerter>>,
    pub network: Option<(Arc<crate::membership::Membership>, Arc<crate::membership::SyncNode>)>,
    /// Conversations (inference and app roles on Weirdnet).
    pub context: Option<crate::context::ContextStore>,
    /// The pool the app role asks through.
    pub pool: Option<Arc<weirdnet_agent::pool::Pool>>,
    /// Model management over Weirdnet (needs a device identity).
    pub models: Option<Arc<crate::models::ModelsHub>>,
    /// Woken by [`ControlRequest::Shutdown`]; `ymird` waits on it next to its signals.
    pub exit: Arc<tokio::sync::Notify>,
}

impl ControlHandler {
    fn network_status(&self) -> Option<NetworkStatus> {
        let (m, node) = self.network.as_ref()?;
        let st = node.status();
        let me = m.peer_id().to_string();
        let devices = m
            .roster()
            .certs
            .iter()
            .map(|c| Device { peer: c.peer_id.clone(), label: c.label.clone(), active: c.active(), me: c.peer_id == me })
            .collect();
        Some(NetworkStatus { peer: me, user: m.user_id(), listening: st.listening, connected: st.connected, devices })
    }

    fn nodes(&self) -> Vec<Arc<dyn ModelNode>> {
        match (&self.coordinator, &self.model_node) {
            (Some(c), _) => c.nodes(),
            (None, Some(n)) => vec![n.clone() as Arc<dyn ModelNode>],
            (None, None) => Vec::new(),
        }
    }

    /// Runs a models request and writes its lines to `w` as they happen.
    async fn stream_models<W: AsyncWrite + Unpin>(&self, node: String, op: crate::models::ModelsOp, w: &mut W) -> Result<()> {
        use crate::models::ModelsLine;
        let Some(hub) = self.models.clone() else {
            let line = ModelsLine::Failed {
                message: "model management over Weirdnet needs a device identity on this node (see `ymirctl pool link`)".into(),
            };
            return write_line(w, &line).await;
        };
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let task = tokio::spawn(async move {
            let done = hub.run(&node, op, &tx).await;
            let _ = tx.send(match done {
                Ok(d) => ModelsLine::Done(Box::new(d)),
                Err(message) => ModelsLine::Failed { message },
            });
        });
        while let Some(line) = rx.recv().await {
            write_line(w, &line).await?;
        }
        let _ = task.await;
        Ok(())
    }

    pub async fn handle(&self, req: ControlRequest) -> ControlResponse {
        match req {
            // Streamed by `stream_models`, which `connection` calls instead.
            ControlRequest::Models { .. } => ControlResponse::err("models requests are streamed"),
            ControlRequest::Shutdown => {
                tracing::info!("shutdown requested over the control socket");
                self.exit.notify_one();
                ControlResponse::ok("ymird is stopping")
            }
            ControlRequest::Status => {
                let models = self
                    .nodes()
                    .iter()
                    .flat_map(|n| {
                        n.status().into_iter().map(|s| NodeModel {
                            purpose: if crate::model_node::embeds(&s) { "embedding".into() } else { "chat".into() },
                            node: n.name().to_string(),
                            id: s.id,
                            engine: s.engine,
                            state: s.state,
                            in_flight: s.in_flight,
                            swap_group: s.swap_group,
                            pid: s.pid,
                            last_load_secs: s.last_load_secs,
                            wrapped: s.wrapped,
                            crashes: s.crashes,
                            capabilities: s.capabilities,
                        })
                    })
                    .collect();
                let nodes = self
                    .nodes()
                    .iter()
                    .map(|n| {
                        let st = n.status();
                        NodeSummary {
                            name: n.name().to_string(),
                            local: self.model_node.as_ref().is_some_and(|l| l.name() == n.name()),
                            draining: n.draining(),
                            models: st.len(),
                            ready: st.iter().filter(|s| s.state == ModelState::Ready).count(),
                            in_flight: st.iter().map(|s| s.in_flight).sum(),
                        }
                    })
                    .collect();
                ControlResponse {
                    ok: true,
                    message: None,
                    chat: None,
                    conversations: Vec::new(),
                    batch: Vec::new(),
                    groups: Vec::new(),
                    group_losses: Vec::new(),
                    status: Some(StatusReport {
                        node: self.node.clone(),
                        version: env!("CARGO_PKG_VERSION").into(),
                        pid: std::process::id(),
                        roles: self.roles.clone(),
                        bridge_port: self.bridge_port,
                        main_model: self.main_model.clone(),
                        models,
                        nodes,
                        alerts: self.alerter.as_ref().map(|a| a.failing()).unwrap_or_default(),
                        network: self.network_status(),
                    }),
                }
            }
            ControlRequest::Load { model, node } => {
                let result = match (&self.coordinator, &self.model_node) {
                    (Some(c), _) => c.load(&model, node.as_deref(), true).await,
                    (None, Some(n)) => n.load(&model, true).await.map(|_| n.name().to_string()),
                    (None, None) => return ControlResponse::err("this node runs neither the inference nor the model role"),
                };
                match result {
                    Ok(on) => ControlResponse::ok(format!("{model} is ready on {on}")),
                    Err(e) => ControlResponse::err(e.to_string()),
                }
            }
            ControlRequest::Unload { model } => {
                let result = match (&self.coordinator, &self.model_node) {
                    (Some(c), _) => c.unload(&model).await,
                    (None, Some(n)) => n.unload(&model).await.map(|_| vec![n.name().to_string()]),
                    (None, None) => return ControlResponse::err("this node runs neither the inference nor the model role"),
                };
                match result {
                    Ok(on) if on.is_empty() => ControlResponse::ok(format!("{model} was not loaded")),
                    Ok(on) => ControlResponse::ok(format!("{model} unloaded on {}", on.join(", "))),
                    Err(e) => ControlResponse::err(e.to_string()),
                }
            }
            ControlRequest::Drain { node, on } => {
                let n = self.nodes().into_iter().find(|n| n.name() == node);
                let Some(n) = n else { return ControlResponse::err(format!("no model node `{node}` (see `ymirctl nodes`)")) };
                match n.drain(on).await {
                    Ok(()) if on => ControlResponse::ok(format!("{node} is drained: no new requests, nothing in flight")),
                    Ok(()) => ControlResponse::ok(format!("{node} takes requests again")),
                    Err(e) => ControlResponse::err(format!("{node}: {e}")),
                }
            }
            ControlRequest::PoolApprove { cert } => {
                let Some((m, _)) = &self.network else { return ControlResponse::err("this node is not on Weirdnet") };
                let label = cert.label.clone();
                match m.approve(cert) {
                    Ok(()) => ControlResponse::ok(format!("{label} is linked")),
                    Err(e) => ControlResponse::err(format!("{e:#}")),
                }
            }
            ControlRequest::ContextAdd { conversation, role, text } => {
                let Some(store) = &self.context else { return no_context() };
                match store.append(&conversation, &role, &text, None, None) {
                    Ok(id) => ControlResponse::ok(id),
                    Err(e) => ControlResponse::err(format!("{e:#}")),
                }
            }
            ControlRequest::ContextShow { conversation } => {
                let Some(store) = &self.context else { return no_context() };
                let all = match conversation {
                    Some(id) => match store.load(&id) {
                        Some(c) => vec![c],
                        None => return ControlResponse::err(format!("no conversation `{id}` here")),
                    },
                    None => store.list(),
                };
                let mut r = ControlResponse::ok(format!("{} conversation(s)", all.len()));
                r.conversations = all
                    .into_iter()
                    .map(|c| ConversationView {
                        title: c.meta.title.clone(),
                        messages: c.messages.into_iter().map(|m| (m.role, m.text)).collect(),
                        id: c.id,
                    })
                    .collect();
                r
            }
            ControlRequest::ContextDelete { conversation } => {
                let Some(store) = &self.context else { return no_context() };
                match store.delete(&conversation) {
                    Ok(()) => ControlResponse::ok(format!("{conversation} is deleted")),
                    Err(e) => ControlResponse::err(format!("{e:#}")),
                }
            }
            ControlRequest::Chat { conversation, text, model } => {
                let (Some(store), Some(pool)) = (&self.context, &self.pool) else {
                    return ControlResponse::err("chat needs the app role and a device identity (see `ymirctl pool link`)");
                };
                if let Err(e) = store.append(&conversation, "user", &text, None, None) {
                    return ControlResponse::err(format!("{e:#}"));
                }
                let mut request =
                    weirdnet_agent::remote::infer::v2::Request::context(weirdnet_agent::remote::infer::v2::ContextRef::new(&conversation, None));
                if let Some(m) = model {
                    request = request.with_model(m);
                }
                // Count the times the answer moved to another inference node.
                let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
                let result = pool.infer(request, &tx, &weirdnet_agent::CancelToken::new()).await;
                drop(tx);
                let (mut resumes, mut path) = (0, Vec::new());
                while let Some(e) = rx.recv().await {
                    match e {
                        weirdnet_agent::pool::PoolEvent::Accepted { node, .. } => path.push(format!("accepted {node}")),
                        weirdnet_agent::pool::PoolEvent::Resumed { node, offset } => {
                            resumes += 1;
                            path.push(format!("resumed {node} at {offset}"));
                        }
                        _ => {}
                    }
                }
                match result {
                    Ok(a) => {
                        let mut r = ControlResponse::ok("answered");
                        r.chat = Some(ChatResult { text: a.text, node: a.node, model: a.model, resumes, path });
                        r
                    }
                    Err(i) => ControlResponse::err(format!("{} (after {} characters)", i.error, i.partial.text.chars().count())),
                }
            }
            ControlRequest::Group { action, model } => {
                let Some(c) = &self.coordinator else {
                    return ControlResponse::err("split models are decided by an inference node; this one has no inference role");
                };
                match (action.as_str(), model) {
                    ("up", Some(m)) => match c.group_up(&m).await {
                        Ok(i) => {
                            let mut r = ControlResponse::ok(format!(
                                "{m} is up: {} leads, with {}",
                                i.leader,
                                i.members.iter().skip(1).cloned().collect::<Vec<_>>().join(", ")
                            ));
                            r.groups = vec![i];
                            r
                        }
                        Err(e) => ControlResponse::err(e.to_string()),
                    },
                    ("down", Some(m)) => match c.group_down(&m).await {
                        Ok(_) => ControlResponse::ok(format!("{m}: group down; normal placement resumes")),
                        Err(e) => ControlResponse::err(e.to_string()),
                    },
                    ("status", _) => {
                        let mut r = ControlResponse::ok("groups");
                        r.groups = c.groups();
                        r.group_losses = c.group_losses();
                        r
                    }
                    _ => ControlResponse::err("usage: group up MODEL | group down MODEL | group status"),
                }
            }
            ControlRequest::Batch { prompts, model, concurrency, retries } => {
                let Some(pool) = &self.pool else { return ControlResponse::err("batch needs the app role") };
                let mut o = crate::batch::BatchOptions::new(format!(
                    "batch-{:x}",
                    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos())
                ));
                o.model = model;
                o.concurrency = concurrency.unwrap_or(o.concurrency);
                o.retries = retries.unwrap_or(o.retries);
                let items = crate::batch::run(pool.clone(), prompts, o, weirdnet_agent::CancelToken::new()).await;
                let failed = items.iter().filter(|i| !i.ok).count();
                let mut r = ControlResponse::ok(format!("{} task(s), {failed} failed", items.len()));
                r.batch = items;
                r
            }
            ControlRequest::BatchTasks { tasks, model, concurrency, retries, job, wait_for_model, cache } => {
                let Some(pool) = &self.pool else { return ControlResponse::err("batch needs the app role") };
                let job = job.unwrap_or_else(|| {
                    format!("batch-{:x}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos()))
                });
                let mut o = crate::batch::BatchOptions::new(job);
                o.model = model;
                o.concurrency = concurrency.unwrap_or(o.concurrency);
                o.retries = retries.unwrap_or(o.retries);
                o.wait_for_model = wait_for_model;
                o.cache = cache;
                let items = crate::batch::run_tasks(pool.clone(), tasks, o, weirdnet_agent::CancelToken::new()).await;
                let failed = items.iter().filter(|i| !i.ok).count();
                let mut r = ControlResponse::ok(format!("{} task(s), {failed} failed", items.len()));
                r.batch = items;
                r
            }
            ControlRequest::PoolRevoke { peer } => {
                let Some((m, _)) = &self.network else { return ControlResponse::err("this node is not on Weirdnet") };
                match m.revoke(&peer) {
                    Ok(c) => ControlResponse::ok(format!("{} ({}) is revoked", c.label, c.peer_id)),
                    Err(e) => ControlResponse::err(format!("{e:#}")),
                }
            }
        }
    }
}

/// The control endpoint `ymird` listens on.
#[cfg(unix)]
pub type Listener = tokio::net::UnixListener;

/// The control endpoint `ymird` listens on: the next free instance of its named pipe.
#[cfg(windows)]
pub struct Listener {
    name: String,
    next: tokio::net::windows::named_pipe::NamedPipeServer,
}

/// The local named pipe that stands in for the control socket at `socket` on Windows:
/// `\\.\pipe\ymird-<hash of the socket path>`, so every `YMIR_HOME` gets its own pipe and
/// `ymirctl` finds it from the same path. Defined on every OS so it can be tested anywhere.
pub fn pipe_name(socket: &Path) -> String {
    let abs = std::path::absolute(socket).unwrap_or_else(|_| socket.to_path_buf());
    let key = abs.to_string_lossy().replace('/', "\\").to_lowercase();
    // `canonicalize` on Windows gives `\\?\C:\...`: the same file as `C:\...`.
    let key = key.strip_prefix(r"\\?\").map(str::to_string).unwrap_or(key);
    // FNV-1a, 64 bit: stable across builds and platforms (std's hasher is not).
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in key.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!(r"\\.\pipe\ymird-{h:016x}")
}

/// Binds the socket, replacing a stale one. Fails if another `ymird` answers on it.
#[cfg(unix)]
pub async fn bind(path: &Path) -> Result<Listener> {
    use tokio::net::{UnixListener, UnixStream};
    if path.exists() {
        if UnixStream::connect(path).await.is_ok() {
            bail!("another ymird is already running (control socket {})", path.display());
        }
        std::fs::remove_file(path).with_context(|| format!("removing stale {}", path.display()))?;
    }
    let listener = UnixListener::bind(path).with_context(|| format!("binding {}", path.display()))?;
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(listener)
}

/// Creates the first instance of the control pipe. Fails if another `ymird` holds it.
#[cfg(windows)]
pub async fn bind(path: &Path) -> Result<Listener> {
    use tokio::net::windows::named_pipe::ServerOptions;
    let name = pipe_name(path);
    let next = ServerOptions::new()
        .first_pipe_instance(true)
        .reject_remote_clients(true)
        .create(&name)
        .map_err(|e| anyhow::anyhow!("another ymird is already running (control pipe {name}): {e}"))?;
    Ok(Listener { name, next })
}

#[cfg(unix)]
async fn accept(listener: &mut Listener) -> std::io::Result<tokio::net::UnixStream> {
    listener.accept().await.map(|(s, _)| s)
}

#[cfg(windows)]
async fn accept(listener: &mut Listener) -> std::io::Result<tokio::net::windows::named_pipe::NamedPipeServer> {
    use tokio::net::windows::named_pipe::ServerOptions;
    listener.next.connect().await?;
    let fresh = ServerOptions::new().reject_remote_clients(true).create(&listener.name)?;
    Ok(std::mem::replace(&mut listener.next, fresh))
}

pub async fn serve(mut listener: Listener, handler: Arc<ControlHandler>, mut shutdown: tokio::sync::watch::Receiver<bool>) {
    loop {
        let accepted = tokio::select! {
            accepted = accept(&mut listener) => accepted,
            _ = shutdown.wait_for(|stop| *stop) => return,
        };
        match accepted {
            Ok(stream) => {
                let h = handler.clone();
                tokio::spawn(async move {
                    if let Err(e) = connection(stream, h).await {
                        tracing::debug!("control connection: {e:#}");
                    }
                });
            }
            Err(e) => {
                tracing::debug!("control accept: {e}");
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        }
    }
}

async fn connection<S: AsyncRead + AsyncWrite + Send + 'static>(stream: S, handler: Arc<ControlHandler>) -> Result<()> {
    let (r, mut w) = tokio::io::split(stream);
    let mut lines = BufReader::new(r.take(64 * 1024)).lines();
    while let Some(line) = lines.next_line().await? {
        let resp = match serde_json::from_str::<ControlRequest>(&line) {
            Ok(ControlRequest::Models { node, request }) => {
                handler.stream_models(node, request, &mut w).await?;
                continue;
            }
            Ok(req) => handler.handle(req).await,
            Err(e) => ControlResponse::err(format!("bad request: {e}")),
        };
        let mut out = serde_json::to_vec(&resp)?;
        out.push(b'\n');
        w.write_all(&out).await?;
    }
    Ok(())
}

#[cfg(unix)]
async fn connect(socket: &Path) -> Result<tokio::net::UnixStream> {
    tokio::net::UnixStream::connect(socket).await.with_context(|| format!("ymird is not running (no control socket at {})", socket.display()))
}

#[cfg(windows)]
async fn connect(socket: &Path) -> Result<tokio::net::windows::named_pipe::NamedPipeClient> {
    use tokio::net::windows::named_pipe::ClientOptions;
    const ERROR_PIPE_BUSY: i32 = 231;
    let name = pipe_name(socket);
    let mut tries = 0;
    loop {
        match ClientOptions::new().open(&name) {
            Ok(c) => return Ok(c),
            Err(e) if e.raw_os_error() == Some(ERROR_PIPE_BUSY) && tries < 40 => {
                tries += 1;
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                bail!("ymird is not running (no control pipe {name} for {})", socket.display())
            }
            Err(e) => return Err(e).with_context(|| format!("opening control pipe {name} (an elevated prompt is needed for a system service)")),
        }
    }
}

async fn write_line<W: AsyncWrite + Unpin>(w: &mut W, line: &impl Serialize) -> Result<()> {
    let mut out = serde_json::to_vec(line)?;
    out.push(b'\n');
    w.write_all(&out).await?;
    Ok(())
}

/// Sends a models request to the `ymird` at `socket` and calls `on_line` for each progress line.
/// Returns the final answer.
pub async fn request_models(
    socket: &Path,
    node: &str,
    op: crate::models::ModelsOp,
    mut on_line: impl FnMut(&crate::models::ModelsLine),
) -> Result<crate::models::ModelsDone> {
    use crate::models::ModelsLine;
    let stream = connect(socket).await?;
    let (r, mut w) = tokio::io::split(stream);
    write_line(&mut w, &ControlRequest::Models { node: node.to_string(), request: op }).await?;
    let mut lines = BufReader::new(r).lines();
    while let Some(line) = lines.next_line().await? {
        match serde_json::from_str::<ModelsLine>(&line) {
            Ok(ModelsLine::Done(d)) => return Ok(*d),
            Ok(ModelsLine::Failed { message }) => bail!("{message}"),
            Ok(p) => on_line(&p),
            // An older ymird answers an unknown request with one `ControlResponse` error.
            Err(_) => match serde_json::from_str::<ControlResponse>(&line) {
                Ok(r) if !r.ok => bail!("this ymird does not manage models over Weirdnet (upgrade it): {}", r.message.unwrap_or_default()),
                _ => bail!("ymird sent something this ymirctl does not understand (are they the same version?)"),
            },
        }
    }
    bail!("ymird closed the control connection")
}

/// Sends one request to the `ymird` at `socket`.
pub async fn request(socket: &Path, req: &ControlRequest) -> Result<ControlResponse> {
    let stream = connect(socket).await?;
    let (r, mut w) = tokio::io::split(stream);
    let mut line = serde_json::to_vec(req)?;
    line.push(b'\n');
    w.write_all(&line).await?;
    let mut lines = BufReader::new(r).lines();
    let resp = lines.next_line().await?.context("ymird closed the control connection")?;
    Ok(serde_json::from_str(&resp)?)
}

/// Whether a `ymird` answers on `socket`.
pub async fn reachable(socket: &Path) -> bool {
    request(socket, &ControlRequest::Status).await.is_ok()
}

fn no_context() -> ControlResponse {
    ControlResponse::err("this node keeps no conversations: it needs the inference or app role and a device identity (see `ymirctl pool link`)")
}
