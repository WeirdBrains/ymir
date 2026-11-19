//! `node.toml`: one file per node. Each role reads only its own section.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use ymir_backends::{EngineConfig, EngineKind, ModelLaunch, find_in_path};

use crate::Paths;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct NodeConfig {
    #[serde(default)]
    pub node: NodeSection,
    #[serde(default)]
    pub roles: Roles,
    /// Weirdnet settings; read from M3 on (roles over Weirdnet).
    #[serde(default, skip_serializing_if = "toml::Table::is_empty")]
    pub network: toml::Table,
    #[serde(default)]
    pub model: ModelSection,
    #[serde(default)]
    pub inference: InferenceSection,
    #[serde(default)]
    pub app: AppSection,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct NodeSection {
    #[serde(default)]
    pub name: String,
}

/// The three roles, independent toggles.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Roles {
    #[serde(default)]
    pub model: bool,
    #[serde(default)]
    pub inference: bool,
    #[serde(default)]
    pub app: bool,
}

impl Roles {
    pub const ALL: Roles = Roles { model: true, inference: true, app: true };

    /// `model,inference,app` (any subset).
    pub fn parse(s: &str) -> Result<Self> {
        let mut r = Roles::default();
        for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            match part {
                "model" => r.model = true,
                "inference" => r.inference = true,
                "app" => r.app = true,
                "all" => r = Roles::ALL,
                other => bail!("unknown role `{other}` (roles: model, inference, app)"),
            }
        }
        Ok(r)
    }

    pub fn names(&self) -> Vec<&'static str> {
        [("model", self.model), ("inference", self.inference), ("app", self.app)].into_iter().filter(|(_, on)| *on).map(|(n, _)| n).collect()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelSection {
    /// `auto` or a size such as `100GiB`.
    #[serde(default = "auto")]
    pub memory_budget: String,
    /// Unified memory kept for the OS, such as `12GiB`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reserve: Option<String>,
    /// Optional private-link address for group jobs (M7).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub private_link: Option<String>,
    /// `managed` | `system:/path/to/llama-server` | `off`.
    #[serde(default = "managed")]
    pub llamacpp: String,
    /// `auto` (only where supported) | `system:/path/to/vllm` | `off`.
    #[serde(default = "auto")]
    pub vllm: String,
    /// Where GGUF files live; default `$WEIRDNET_MODELS_DIR` or the shared Weirdnet models dir.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub models_dir: Option<PathBuf>,
    #[serde(default)]
    pub local: Vec<LocalModel>,
}

fn auto() -> String {
    "auto".into()
}

fn managed() -> String {
    "managed".into()
}

impl Default for ModelSection {
    fn default() -> Self {
        Self { memory_budget: auto(), reserve: None, private_link: None, llamacpp: managed(), vllm: auto(), models_dir: None, local: Vec::new() }
    }
}

/// A model this node keeps on disk and may be told to load.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LocalModel {
    pub id: String,
    /// `llamacpp` (default) or `vllm`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub swap_group: Option<String>,
    /// `main`, `standin` or `embedding`; default `chat`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// GGUF path (absolute, or relative to `models_dir`), or a vLLM model path or Hub id. Default:
    /// the catalog file for `id` in `models_dir`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slots: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_layers: Option<i32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
}

impl LocalModel {
    pub fn is_embedding(&self) -> bool {
        self.role.as_deref() == Some("embedding")
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceSection {
    #[serde(default = "yes")]
    pub scheduler_eligible: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_disk_budget: Option<String>,
    #[serde(default)]
    pub cache: CacheSection,
    /// Alert sinks; read from M2 on.
    #[serde(default, skip_serializing_if = "toml::Table::is_empty")]
    pub alerts: toml::Table,
}

fn yes() -> bool {
    true
}

impl Default for InferenceSection {
    fn default() -> Self {
        Self { scheduler_eligible: true, context_disk_budget: None, cache: CacheSection::default(), alerts: toml::Table::new() }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheSection {
    /// `off` | `local` | `pool` (pool mode is M6).
    #[serde(default = "local")]
    pub mode: String,
    /// `embedded` | `redis` (redis is not in M1).
    #[serde(default = "embedded")]
    pub backend: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub redis_url: Option<String>,
    #[serde(default = "threshold")]
    pub threshold: f32,
}

fn local() -> String {
    "local".into()
}

fn embedded() -> String {
    "embedded".into()
}

fn threshold() -> f32 {
    0.12
}

impl Default for CacheSection {
    fn default() -> Self {
        Self { mode: local(), backend: embedded(), redis_url: None, threshold: threshold() }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AppSection {
    #[serde(default)]
    pub bridge: BridgeSection,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BridgeSection {
    /// Always on 127.0.0.1; there is no address setting.
    #[serde(default = "bridge_port")]
    pub port: u16,
}

fn bridge_port() -> u16 {
    8484
}

impl Default for BridgeSection {
    fn default() -> Self {
        Self { port: bridge_port() }
    }
}

impl NodeConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let cfg: Self = toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn to_toml(&self) -> Result<String> {
        Ok(toml::to_string_pretty(self)?)
    }

    /// Writes `path` atomically.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = path.with_extension("toml.tmp");
        std::fs::write(&tmp, format!("# Ymir node config. No secrets belong here.\n{}", self.to_toml()?))?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        let mut seen = std::collections::HashSet::new();
        for m in &self.model.local {
            if m.id.is_empty() || m.id.contains(char::is_whitespace) {
                bail!("model id `{}` must be non-empty and have no spaces", m.id);
            }
            if !seen.insert(&m.id) {
                bail!("model `{}` is listed twice", m.id);
            }
            if let Some(b) = &m.backend
                && b != "llamacpp"
                && b != "vllm"
            {
                bail!("model `{}`: unknown backend `{b}` (llamacpp or vllm)", m.id);
            }
        }
        Ok(())
    }

    pub fn name(&self) -> String {
        if self.node.name.is_empty() { crate::setup::hostname() } else { self.node.name.clone() }
    }

    /// The main chat model: the first with `role = "main"`, else the first chat model.
    pub fn main_model(&self) -> Option<&LocalModel> {
        self.model.local.iter().find(|m| m.role.as_deref() == Some("main")).or_else(|| self.model.local.iter().find(|m| !m.is_embedding()))
    }

    pub fn models_dir(&self) -> PathBuf {
        self.model.models_dir.clone().unwrap_or_else(|| weirdnet_model::ModelStore::from_env().dir().to_path_buf())
    }

    /// The engines this node can run, from `llamacpp` and `vllm`.
    pub fn engines(&self, paths: &Paths) -> Vec<EngineConfig> {
        let mut out = Vec::new();
        if let Some(p) = resolve_engine(&self.model.llamacpp, "llama-server", &paths.engines().join("llama.cpp").join("llama-server")) {
            out.push(EngineConfig::llamacpp(p));
        }
        if let Some(p) = resolve_engine(&self.model.vllm, "vllm", &paths.engines().join("vllm").join("bin").join("vllm")) {
            out.push(EngineConfig::vllm(p));
        }
        out
    }

    /// Launch specs for `[[model.local]]`.
    pub fn launches(&self) -> Vec<ModelLaunch> {
        let dir = self.models_dir();
        self.model
            .local
            .iter()
            .map(|m| {
                let engine = if m.backend.as_deref() == Some("vllm") { EngineKind::Vllm } else { EngineKind::Llamacpp };
                let source = match (&m.file, engine) {
                    (Some(f), EngineKind::Llamacpp) => {
                        let p = Path::new(f);
                        if p.is_absolute() { p.to_path_buf() } else { dir.join(p) }.to_string_lossy().into_owned()
                    }
                    (Some(f), EngineKind::Vllm) => f.clone(),
                    (None, _) => match ymir_hw::find(&m.id) {
                        Some(e) => weirdnet_model::ModelStore::new(&dir).path(&e.spec).to_string_lossy().into_owned(),
                        None => dir.join(format!("{}.gguf", m.id)).to_string_lossy().into_owned(),
                    },
                };
                let mut l = ModelLaunch::new(&m.id, engine, source);
                l.context = m.context;
                l.slots = m.slots;
                l.gpu_layers = m.gpu_layers;
                l.swap_group.clone_from(&m.swap_group);
                l.embedding = m.is_embedding();
                l.extra_args.clone_from(&m.args);
                l
            })
            .collect()
    }
}

/// `managed` looks in Ymir's engines dir, then `PATH`; `system:/path` uses that; `auto` (vLLM)
/// looks on `PATH`; `off` disables.
fn resolve_engine(setting: &str, binary: &str, managed_path: &Path) -> Option<PathBuf> {
    match setting.trim() {
        "off" => None,
        s if s.starts_with("system:") => Some(PathBuf::from(&s["system:".len()..])),
        "managed" => Some(managed_path.to_path_buf()).filter(|p| p.exists()).or_else(|| find_in_path(binary)),
        _ => find_in_path(binary),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_design_example() {
        let cfg: NodeConfig = toml::from_str(
            r#"
            [node]
            name = "server-a"
            [roles]
            model = true
            inference = true
            app = false
            [network]
            bootstrap = "default"
            [model]
            memory_budget = "auto"
            reserve = "12GiB"
            llamacpp = "managed"
            vllm = "auto"
            [[model.local]]
            id = "qwen3.6-35b-a3b"
            backend = "llamacpp"
            quant = "Q4_K_M"
            swap_group = "main"
            [[model.local]]
            id = "gemma4-e4b"
            [inference]
            scheduler_eligible = true
            context_disk_budget = "2GiB"
            [inference.cache]
            mode = "local"
            threshold = 0.12
            [app.bridge]
            port = 8484
            "#,
        )
        .unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.roles, Roles { model: true, inference: true, app: false });
        assert_eq!(cfg.main_model().unwrap().id, "qwen3.6-35b-a3b");
        assert_eq!(cfg.app.bridge.port, 8484);
        let back: NodeConfig = toml::from_str(&cfg.to_toml().unwrap()).unwrap();
        assert_eq!(back, cfg);
    }

    #[test]
    fn roles_parse() {
        assert_eq!(Roles::parse("model,app").unwrap(), Roles { model: true, inference: false, app: true });
        assert_eq!(Roles::parse("all").unwrap(), Roles::ALL);
        assert!(Roles::parse("gpu").is_err());
    }

    #[test]
    fn rejects_duplicates() {
        let mut cfg = NodeConfig::default();
        cfg.model.local = vec![LocalModel { id: "a".into(), ..Default::default() }, LocalModel { id: "a".into(), ..Default::default() }];
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn launch_paths() {
        let mut cfg = NodeConfig::default();
        cfg.model.models_dir = Some("/models".into());
        cfg.model.local = vec![
            LocalModel { id: "qwen3-8b".into(), ..Default::default() },
            LocalModel { id: "mine".into(), file: Some("x/mine.gguf".into()), role: Some("embedding".into()), ..Default::default() },
        ];
        let l = cfg.launches();
        assert_eq!(l[0].source, "/models/Qwen3-8B-Q4_K_M.gguf");
        assert_eq!(l[1].source, "/models/x/mine.gguf");
        assert!(l[1].embedding);
    }
}
