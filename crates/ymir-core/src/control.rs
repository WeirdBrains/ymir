//! `/ymir/control/1`, local transport: JSON lines over a Unix socket in the node's private state
//! directory (mode 0600). Over Weirdnet (M3) the same messages go to admin devices on the roster.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
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

    pub async fn handle(&self, req: ControlRequest) -> ControlResponse {
        match req {
            ControlRequest::Status => {
                let models = self
                    .nodes()
                    .iter()
                    .flat_map(|n| {
                        n.status().into_iter().map(|s| NodeModel {
                            node: n.name().to_string(),
                            id: s.id,
                            engine: s.engine,
                            state: s.state,
                            in_flight: s.in_flight,
                            swap_group: s.swap_group,
                            purpose: if s.embedding { "embedding".into() } else { "chat".into() },
                            pid: s.pid,
                            last_load_secs: s.last_load_secs,
                            wrapped: s.wrapped,
                            crashes: s.crashes,
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

/// Binds the socket, replacing a stale one. Fails if another `ymird` answers on it.
pub async fn bind(path: &Path) -> Result<UnixListener> {
    if path.exists() {
        if UnixStream::connect(path).await.is_ok() {
            bail!("another ymird is already running (control socket {})", path.display());
        }
        std::fs::remove_file(path).with_context(|| format!("removing stale {}", path.display()))?;
    }
    let listener = UnixListener::bind(path).with_context(|| format!("binding {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(listener)
}

pub async fn serve(listener: UnixListener, handler: Arc<ControlHandler>, mut shutdown: tokio::sync::watch::Receiver<bool>) {
    loop {
        tokio::select! {
            accepted = listener.accept() => {
                let Ok((stream, _)) = accepted else { continue };
                let h = handler.clone();
                tokio::spawn(async move {
                    if let Err(e) = connection(stream, h).await {
                        tracing::debug!("control connection: {e:#}");
                    }
                });
            }
            _ = shutdown.wait_for(|stop| *stop) => return,
        }
    }
}

async fn connection(stream: UnixStream, handler: Arc<ControlHandler>) -> Result<()> {
    let (r, mut w) = stream.into_split();
    let mut lines = BufReader::new(r.take(64 * 1024)).lines();
    while let Some(line) = lines.next_line().await? {
        let resp = match serde_json::from_str::<ControlRequest>(&line) {
            Ok(req) => handler.handle(req).await,
            Err(e) => ControlResponse::err(format!("bad request: {e}")),
        };
        let mut out = serde_json::to_vec(&resp)?;
        out.push(b'\n');
        w.write_all(&out).await?;
    }
    Ok(())
}

/// Sends one request to the `ymird` at `socket`.
pub async fn request(socket: &Path, req: &ControlRequest) -> Result<ControlResponse> {
    let stream = UnixStream::connect(socket).await.with_context(|| format!("ymird is not running (no control socket at {})", socket.display()))?;
    let (r, mut w) = stream.into_split();
    let mut line = serde_json::to_vec(req)?;
    line.push(b'\n');
    w.write_all(&line).await?;
    let mut lines = BufReader::new(r).lines();
    let resp = lines.next_line().await?.context("ymird closed the control connection")?;
    Ok(serde_json::from_str(&resp)?)
}

fn no_context() -> ControlResponse {
    ControlResponse::err("this node keeps no conversations: it needs the inference or app role and a device identity (see `ymirctl pool link`)")
}
