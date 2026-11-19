//! How each engine is launched and probed.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EngineKind {
    /// llama.cpp `llama-server`.
    Llamacpp,
    /// `vllm serve` (Linux, NVIDIA sm_80+ or supported ROCm).
    Vllm,
}

impl EngineKind {
    pub fn name(self) -> &'static str {
        match self {
            EngineKind::Llamacpp => "llamacpp",
            EngineKind::Vllm => "vllm",
        }
    }
}

/// One engine binary and how long it may take to come up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineConfig {
    pub kind: EngineKind,
    pub program: PathBuf,
    /// Arguments added to every launch of this engine.
    pub extra_args: Vec<String>,
    pub ready_timeout: Duration,
}

impl EngineConfig {
    pub fn llamacpp(program: impl Into<PathBuf>) -> Self {
        Self { kind: EngineKind::Llamacpp, program: program.into(), extra_args: Vec::new(), ready_timeout: Duration::from_secs(300) }
    }

    pub fn vllm(program: impl Into<PathBuf>) -> Self {
        Self { kind: EngineKind::Vllm, program: program.into(), extra_args: Vec::new(), ready_timeout: Duration::from_secs(1200) }
    }

    /// The command for `model` on `127.0.0.1:port`. The API key goes in the environment, not the
    /// command line, so other local users can't read it from the process list.
    pub fn command(&self, model: &ModelLaunch, port: u16, api_key: &str) -> Command {
        let mut cmd = Command::new(&self.program);
        match self.kind {
            EngineKind::Llamacpp => {
                cmd.args(["--model", &model.source, "--host", "127.0.0.1", "--port", &port.to_string(), "--alias", &model.id, "--no-webui"]);
                if let Some(ctx) = model.context {
                    cmd.args(["--ctx-size", &ctx.to_string()]);
                }
                if let Some(slots) = model.slots {
                    cmd.args(["--parallel", &slots.to_string()]);
                }
                if let Some(ngl) = model.gpu_layers {
                    cmd.args(["--n-gpu-layers", &ngl.to_string()]);
                }
                if model.embedding {
                    cmd.arg("--embedding");
                }
                cmd.env("LLAMA_API_KEY", api_key);
            }
            EngineKind::Vllm => {
                cmd.args(["serve", &model.source, "--host", "127.0.0.1", "--port", &port.to_string(), "--served-model-name", &model.id]);
                if let Some(ctx) = model.context {
                    cmd.args(["--max-model-len", &ctx.to_string()]);
                }
                if let Some(slots) = model.slots {
                    cmd.args(["--max-num-seqs", &slots.to_string()]);
                }
                cmd.env("VLLM_API_KEY", api_key);
            }
        }
        cmd.args(&self.extra_args).args(&model.extra_args);
        cmd
    }
}

/// A model this node may be told to load.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelLaunch {
    /// Pool id; also the engine's served model name.
    pub id: String,
    pub engine: EngineKind,
    /// A GGUF path for llama.cpp; a path or Hub id for vLLM.
    pub source: String,
    pub context: Option<u32>,
    /// Concurrent requests (llama.cpp slots, vLLM max sequences).
    pub slots: Option<u32>,
    /// llama.cpp `--n-gpu-layers`; `None` keeps the engine's default (all that fit).
    pub gpu_layers: Option<i32>,
    /// Models in the same group are never resident together.
    pub swap_group: Option<String>,
    pub embedding: bool,
    pub extra_args: Vec<String>,
}

impl ModelLaunch {
    pub fn new(id: impl Into<String>, engine: EngineKind, source: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            engine,
            source: source.into(),
            context: None,
            slots: None,
            gpu_layers: None,
            swap_group: None,
            embedding: false,
            extra_args: Vec::new(),
        }
    }
}

/// The first executable called `name` on `PATH`.
pub fn find_in_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|d| d.join(name)).find(|p| is_executable(p))
}

fn is_executable(p: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(p).is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
    }
    #[cfg(not(unix))]
    {
        p.is_file()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(cmd: &Command) -> Vec<String> {
        cmd.as_std().get_args().map(|a| a.to_string_lossy().into_owned()).collect()
    }

    #[test]
    fn llama_args() {
        let mut m = ModelLaunch::new("qwen3-8b", EngineKind::Llamacpp, "/m/q.gguf");
        m.context = Some(8192);
        m.slots = Some(2);
        let cmd = EngineConfig::llamacpp("/bin/llama-server").command(&m, 4321, "k");
        let a = args(&cmd);
        assert!(a.windows(2).any(|w| w == ["--host", "127.0.0.1"]));
        assert!(a.windows(2).any(|w| w == ["--port", "4321"]));
        assert!(a.windows(2).any(|w| w == ["--alias", "qwen3-8b"]));
        assert!(a.windows(2).any(|w| w == ["--parallel", "2"]));
        assert!(!a.iter().any(|x| x == "k"), "the key stays out of argv");
    }

    #[test]
    fn vllm_args() {
        let m = ModelLaunch::new("big", EngineKind::Vllm, "Qwen/Qwen3-8B");
        let a = args(&EngineConfig::vllm("vllm").command(&m, 9000, "k"));
        assert_eq!(&a[..2], ["serve", "Qwen/Qwen3-8B"]);
        assert!(a.windows(2).any(|w| w == ["--served-model-name", "big"]));
    }
}
