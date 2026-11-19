//! `/ymir/control/1`, local transport: JSON lines over a Unix socket in the node's private state
//! directory (mode 0600). Over Weirdnet (M3) the same messages go to admin devices on the roster.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use ymir_backends::EngineKind;
pub use ymir_infer::ModelState;

use crate::{Coordinator, LocalModelNode, ModelNode};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum ControlRequest {
    Status,
    Load {
        model: String,
        #[serde(default)]
        node: Option<String>,
    },
    Unload {
        model: String,
    },
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
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ControlResponse {
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<StatusReport>,
}

impl ControlResponse {
    fn ok(message: impl Into<String>) -> Self {
        Self { ok: true, message: Some(message.into()), status: None }
    }
    fn err(message: impl Into<String>) -> Self {
        Self { ok: false, message: Some(message.into()), status: None }
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
}

impl ControlHandler {
    fn nodes(&self) -> Vec<Arc<dyn ModelNode>> {
        match (&self.coordinator, &self.model_node) {
            (Some(c), _) => c.nodes().to_vec(),
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
                        })
                    })
                    .collect();
                ControlResponse {
                    ok: true,
                    message: None,
                    status: Some(StatusReport {
                        node: self.node.clone(),
                        version: env!("CARGO_PKG_VERSION").into(),
                        pid: std::process::id(),
                        roles: self.roles.clone(),
                        bridge_port: self.bridge_port,
                        main_model: self.main_model.clone(),
                        models,
                    }),
                }
            }
            ControlRequest::Load { model, node } => {
                let result = match (&self.coordinator, &self.model_node) {
                    (Some(c), _) => c.load(&model, node.as_deref()).await,
                    (None, Some(n)) => n.load(&model).await.map(|_| n.name().to_string()),
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
