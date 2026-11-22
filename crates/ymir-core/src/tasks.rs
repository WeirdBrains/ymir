//! Agent tasks (`/weirdnet-agent/task/1`): the inference role runs an agent loop for the user's
//! apps (Bragi's agent mode). The model is the pool's own (through the coordinator); the one tool
//! is `files.read` on a directory the operator names, and it asks for approval on every call.
//!
//! Only the user's linked devices get a task (the same roster check as every other protocol), a
//! task never gets more than the tools below (plus the tools of the outside MCP servers the operator
//! lists in `[[inference.agent.mcp]]`, each call approved on the asking device), and a path that leaves the workspace (`..`, an
//! absolute path, or a link pointing out of it) is refused.

use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use tokio::task::JoinHandle;
use weirdnet_agent::pool::stream_limits;
use weirdnet_agent::remote::task::{self, TaskHost, TaskRequest};
use weirdnet_agent::remote::{Authorize, Caller, RemoteError};
use weirdnet_agent::serve::channel;
use weirdnet_agent::{
    AgentConfig, AgentIdentity, AgentKind, AgentSession, Approver, CallTarget, LocalTools, ModelBackend, Policy, Tier, Tool, ToolContext, ToolError,
    ToolOutput, ToolRegistry, ToolSpec,
};

use crate::config::AgentSection;
use crate::coordinator::Coordinator;
use crate::host::turn_backend;
use crate::membership::SyncNode;

/// The companion ids this node runs: Bragi's default and Ymir's own name.
pub const COMPANIONS: [&str; 2] = ["assistant", "ymir"];
/// The tool, when a workspace is set.
pub const FILES_READ: &str = "files.read";

/// Serves task/1 on `node` to the callers `auth` admits. Dropping the handle stops accepting
/// (tasks under way carry on).
pub async fn serve(
    node: &Arc<SyncNode>,
    coord: Arc<Coordinator>,
    auth: Arc<dyn Authorize>,
    label: &str,
    cfg: &AgentSection,
) -> Result<JoinHandle<()>, RemoteError> {
    let mut registry = ToolRegistry::new();
    if let Some(dir) = cfg.workspace.as_deref().filter(|d| !d.is_empty()) {
        let root = std::fs::canonicalize(dir).map_err(|e| RemoteError::Failed { message: format!("agent workspace {dir}: {e}") })?;
        registry.add_source(Arc::new(LocalTools::new("files").with(FilesRead { root, max_bytes: cfg.max_file_kib.max(1) * 1024 })));
    }
    for server in cfg.mcp.iter().filter(|s| s.enabled) {
        // A server that won't start or list its tools is left out; the node and the other tools carry on.
        match server.connect().await {
            Ok(source) => match source.client().list_tools().await {
                Ok(tools) => {
                    tracing::info!(server = %server.id, tools = tools.len(), "mcp server ready");
                    registry.add_source(Arc::new(source));
                }
                Err(e) => tracing::warn!(server = %server.id, "mcp server left out: tools/list failed: {e:?}"),
            },
            Err(e) => tracing::warn!("mcp server left out: {e}"),
        }
    }
    registry.refresh().await.map_err(|e| RemoteError::Failed { message: e.to_string() })?;
    let host: Arc<dyn TaskHost> = Arc::new(Tasks { coord, tools: Arc::new(registry), label: label.to_string() });
    let mut incoming = node.accept_streams(task::PROTOCOL, stream_limits()).map_err(|e| RemoteError::Failed { message: e.to_string() })?;
    Ok(tokio::spawn(async move {
        while let Some(stream) = incoming.next().await {
            let caller = Caller::new(stream.peer().to_string());
            if auth.authorize(&caller).is_err() {
                tracing::warn!(peer = %caller.peer, "task refused: not a linked device");
                continue;
            }
            let host = host.clone();
            tokio::spawn(async move {
                let _ = task::serve(channel(stream), host.as_ref()).await;
            });
        }
    }))
}

struct Tasks {
    coord: Arc<Coordinator>,
    tools: Arc<ToolRegistry>,
    label: String,
}

impl TaskHost for Tasks {
    fn session(&self, request: &TaskRequest, approver: Arc<dyn Approver>) -> Result<AgentSession, RemoteError> {
        let id = request.companion.to_string();
        if !COMPANIONS.contains(&id.as_str()) {
            return Err(RemoteError::Refused { message: format!("{id} does not run here") });
        }
        let identity = AgentIdentity::new(id.clone(), "Assistant", AgentKind::Node).with_node(self.label.clone());
        let model: Arc<dyn ModelBackend> = turn_backend(self.coord.clone(), None);
        Ok(AgentSession::new(AgentConfig::new(identity), model, self.tools.clone(), Policy::new().shared(), approver))
    }
}

/// Reads one UTF-8 file under the workspace.
struct FilesRead {
    root: PathBuf,
    max_bytes: u64,
}

/// The file `rel` names inside `root`, or why not. Resolves links, so one that points out of the
/// workspace is refused too.
pub fn resolve(root: &Path, rel: &str) -> Result<PathBuf, String> {
    let rel = Path::new(rel);
    if rel.as_os_str().is_empty() || rel.is_absolute() || rel.components().any(|c| !matches!(c, Component::Normal(_) | Component::CurDir)) {
        return Err("the path must be relative and stay inside the workspace".into());
    }
    let full = std::fs::canonicalize(root.join(rel)).map_err(|_| "no such file".to_string())?;
    if !full.starts_with(root) {
        return Err("the path must be relative and stay inside the workspace".into());
    }
    Ok(full)
}

#[async_trait]
impl Tool for FilesRead {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(FILES_READ, "Read a text file from the node's workspace folder")
            .schema(serde_json::json!({
                "type": "object",
                "properties": { "path": { "type": "string", "description": "Path relative to the workspace" } },
                "required": ["path"]
            }))
            .tier(Tier::Navigate)
    }

    fn target(&self, args: &serde_json::Value) -> CallTarget {
        CallTarget::url(format!("https://workspace.ymir/{}", args["path"].as_str().unwrap_or_default().trim_start_matches('/')))
    }

    async fn call(&self, _ctx: &ToolContext, args: serde_json::Value) -> Result<ToolOutput, ToolError> {
        let rel = args["path"].as_str().unwrap_or_default();
        let refuse = |m: String| Ok(ToolOutput { content: m, is_error: true, steps: Vec::new() });
        let path = match resolve(&self.root, rel) {
            Ok(p) => p,
            Err(m) => return refuse(m),
        };
        let max = self.max_bytes;
        let read = tokio::task::spawn_blocking(move || -> Result<String, String> {
            let meta = std::fs::metadata(&path).map_err(|e| e.to_string())?;
            if !meta.is_file() {
                return Err("not a file".into());
            }
            if meta.len() > max {
                return Err(format!("the file is {} bytes; the limit is {max}", meta.len()));
            }
            let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
            String::from_utf8(bytes).map_err(|_| "not a text file".to_string())
        })
        .await
        .map_err(|e| ToolError::Failed(e.to_string()))?;
        match read {
            Ok(text) => Ok(ToolOutput::text(text)),
            Err(m) => refuse(m),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_stay_inside_the_workspace() {
        let tmp = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(tmp.path()).unwrap();
        std::fs::create_dir(root.join("notes")).unwrap();
        std::fs::write(root.join("notes/a.txt"), "hi").unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret"), "no").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(outside.path().join("secret"), root.join("link")).unwrap();

        assert!(resolve(&root, "notes/a.txt").is_ok());
        assert!(resolve(&root, "./notes/a.txt").is_ok());
        for bad in ["", "/etc/passwd", "../x", "notes/../../x", "missing"] {
            assert!(resolve(&root, bad).is_err(), "{bad}");
        }
        #[cfg(unix)]
        assert!(resolve(&root, "link").is_err(), "a link out of the workspace");
    }
}
