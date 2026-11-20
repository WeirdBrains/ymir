//! `node.toml`: one file per node. Each role reads only its own section.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use weirdnet_backends::{EngineConfig, EngineKind, ModelLaunch, find_in_path};

use crate::Paths;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct NodeConfig {
    #[serde(default)]
    pub node: NodeSection,
    #[serde(default)]
    pub roles: Roles,
    /// Weirdnet: how the roles reach the user's other devices (M3).
    #[serde(default, skip_serializing_if = "NetworkSection::is_default")]
    pub network: NetworkSection,
    #[serde(default)]
    pub model: ModelSection,
    #[serde(default)]
    pub inference: InferenceSection,
    #[serde(default)]
    pub app: AppSection,
}

/// `[network]`. Weirdnet runs once this node has an identity (`ymirctl pool init`, or a link
/// from another device); these settings say how it is reached.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NetworkSection {
    /// Turn Weirdnet off even with an identity.
    #[serde(default = "yes")]
    pub enabled: bool,
    /// TCP and QUIC port on all addresses; 0 picks a free one.
    #[serde(default = "network_port")]
    pub port: u16,
    /// Listen on exactly these addresses instead (such as `/ip4/127.0.0.1/tcp/0`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub listen: Option<Vec<String>>,
    /// Find the user's devices on the local network.
    #[serde(default = "yes")]
    pub mdns: bool,
    /// Peer discovery (mDNS and the DHT). Off: only `peers` and the relays.
    #[serde(default = "yes")]
    pub discovery: bool,
    /// Relays to be reachable through behind NAT. Default: the Weird Brains relay.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relays: Option<Vec<String>>,
    /// Devices to dial directly: `/ip4/…/tcp/…/p2p/PEER` (a peer id is required). Adding one
    /// grants nothing; the roster decides who is answered.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub peers: Vec<String>,
    /// Requests the inference role runs at once for other devices, and how many may wait.
    #[serde(default = "slots")]
    pub slots: usize,
    #[serde(default = "queue")]
    pub queue: usize,
    /// Batch or background requests one device may have here at once (default: half of
    /// `slots + queue`), so it can't leave the user's other devices with `Busy`.
    #[serde(default)]
    pub batch_per_device: Option<usize>,
    /// Seconds between sync passes against the user's other devices (roster, conversations).
    #[serde(default = "sync_secs")]
    pub sync_secs: u64,
}

fn sync_secs() -> u64 {
    30
}

fn network_port() -> u16 {
    4848
}

fn slots() -> usize {
    4
}

fn queue() -> usize {
    32
}

impl Default for NetworkSection {
    fn default() -> Self {
        Self {
            enabled: true,
            port: network_port(),
            listen: None,
            mdns: true,
            discovery: true,
            relays: None,
            peers: Vec::new(),
            slots: slots(),
            queue: queue(),
            batch_per_device: None,
            sync_secs: sync_secs(),
        }
    }
}

impl NetworkSection {
    fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// `peers` as peer ids and addresses.
    pub fn hints(&self) -> Result<Vec<(weirdnet_net::PeerId, Vec<weirdnet_net::Multiaddr>)>> {
        let mut out: Vec<(weirdnet_net::PeerId, Vec<weirdnet_net::Multiaddr>)> = Vec::new();
        for p in &self.peers {
            let hint = weirdnet_net::PeerHint::from_addr(p).map_err(|e| anyhow::anyhow!("network.peers `{p}`: {e}"))?;
            let id: weirdnet_net::PeerId = hint.peer_id.parse().map_err(|e| anyhow::anyhow!("network.peers `{p}`: {e}"))?;
            let addrs: Vec<weirdnet_net::Multiaddr> = hint.addrs.iter().filter_map(|a| a.parse().ok()).collect();
            match out.iter_mut().find(|(i, _)| *i == id) {
                Some((_, a)) => a.extend(addrs),
                None => out.push((id, addrs)),
            }
        }
        Ok(out)
    }

    /// The Weirdnet node settings.
    pub fn node_config(&self, label: &str) -> Result<weirdnet_net::NodeConfig> {
        let mut c = weirdnet_sync::node_config(false);
        c.listen = match &self.listen {
            Some(l) => l.iter().map(|a| a.parse().map_err(|e| anyhow::anyhow!("network.listen `{a}`: {e}"))).collect::<Result<_>>()?,
            None => weirdnet_net::NodeConfig::default_listen(self.port),
        };
        c.mdns = self.mdns;
        c.discovery = if self.discovery { weirdnet_net::DiscoveryMode::Enabled } else { weirdnet_net::DiscoveryMode::Disabled };
        let relays: Vec<String> = match &self.relays {
            Some(r) => r.clone(),
            None => weirdnet_device_link::KNOWN_RELAYS.iter().map(|s| s.to_string()).collect(),
        };
        c.relays = relays.iter().map(|r| r.parse().map_err(|e| anyhow::anyhow!("network.relays `{r}`: {e}"))).collect::<Result<_>>()?;
        c.peers = self
            .peers
            .iter()
            .map(|p| weirdnet_net::PeerHint::from_addr(p).map_err(|e| anyhow::anyhow!("network.peers `{p}`: {e}")))
            .collect::<Result<_>>()?;
        c.label = label.to_string();
        c.version = format!("ymir/{}", env!("CARGO_PKG_VERSION"));
        Ok(c)
    }
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
    /// Which linked devices may use this model node, by peer id (the certificate signature covers the
    /// peer id; a roster label is not signed, so labels are rejected). Empty (the default) means any
    /// linked device, an app included; set it to the inference nodes' peer ids so apps must go through them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub callers: Vec<String>,
    /// `managed` | `system:/path/to/llama-server` | `off`.
    #[serde(default = "managed")]
    pub llamacpp: String,
    /// llama.cpp's `rpc-server`, for taking part in a split model: `managed` (next to the
    /// managed llama-server, else on `PATH`) | `system:/path/to/rpc-server` | `off`.
    #[serde(default = "managed")]
    pub rpc_server: String,
    /// More `rpc-server` arguments (such as `-d CPU` or `-t 8`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rpc_args: Vec<String>,
    /// `auto` (only where supported) | `system:/path/to/vllm` | `off`.
    #[serde(default = "auto")]
    pub vllm: String,
    /// MLX-LM's `mlx_lm.server`, for `backend = "mlx"` (Apple Silicon): `managed` (the venv
    /// `ymirctl setup` installs, else on `PATH`) | `system:/path/to/mlx_lm.server` | `off`.
    #[serde(default = "managed", skip_serializing_if = "is_managed")]
    pub mlx: String,
    /// Where GGUF files live; default `$WEIRDNET_MODELS_DIR` or the shared Weirdnet models dir.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub models_dir: Option<PathBuf>,
    #[serde(default)]
    pub local: Vec<LocalModel>,
    /// Split models this node can lead (M7); each needs a group to be up before it answers.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub split: Vec<crate::group::SplitModel>,
}

fn auto() -> String {
    "auto".into()
}

fn managed() -> String {
    "managed".into()
}

fn is_managed(s: &str) -> bool {
    s == "managed"
}

impl Default for ModelSection {
    fn default() -> Self {
        Self {
            memory_budget: auto(),
            reserve: None,
            private_link: None,
            callers: Vec::new(),
            llamacpp: managed(),
            rpc_server: managed(),
            rpc_args: Vec::new(),
            vllm: auto(),
            mlx: managed(),
            models_dir: None,
            local: Vec::new(),
            split: Vec::new(),
        }
    }
}

/// A model this node keeps on disk and may be told to load.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LocalModel {
    pub id: String,
    /// `llamacpp` (default), `mlx` (Apple Silicon) or `vllm`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub swap_group: Option<String>,
    /// `main`, `standin` or `embedding`; default `chat`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// GGUF path (absolute, or relative to `models_dir`), an MLX weights directory (likewise) or
    /// Hub id, or a vLLM model path or Hub id. Default: the catalog file for `id` in `models_dir`
    /// (for `mlx`, the catalog model's MLX weights in `models_dir/mlx/`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    /// Vision projector for llama.cpp (`--mmproj`): a path, absolute or relative to `models_dir`,
    /// or `off`. Default: the catalog model's projector, when downloaded (see [`crate::vision`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mmproj: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slots: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_layers: Option<i32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
    /// Unload after this long without requests: `600`, `30s`, `10m`, `2h`. Default: stay loaded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idle_ttl: Option<String>,
    /// Serve through a stock OpenAI-compatible server already running on loopback (such as
    /// `vllm serve` started by hand) instead of starting an engine: `http://127.0.0.1:8000`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wrap: Option<String>,
    /// Where the wrapped server's API key comes from: `env:NAME` or `file:/path`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wrap_api_key: Option<String>,
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
    /// Agent tasks for the user's apps (task/1), e.g. Bragi's agent mode.
    #[serde(default)]
    pub agent: AgentSection,
    /// The cold-start stand-in; default the model with `role = "standin"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub standin: Option<String>,
    /// Answer with the stand-in only when the load is expected to take longer than this.
    #[serde(default = "standin_after")]
    pub standin_after: String,
    /// Load the stand-in when the daemon starts, so it is warm for the first cold start.
    #[serde(default = "yes")]
    pub standin_warm: bool,
    #[serde(default)]
    pub alerts: weirdnet_alerts::AlertsConfig,
    /// Models that may be split across model nodes (M7). Off unless listed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub split: Vec<crate::group::SplitPolicy>,
}

fn standin_after() -> String {
    "10s".into()
}

fn yes() -> bool {
    true
}

impl Default for InferenceSection {
    fn default() -> Self {
        Self {
            scheduler_eligible: true,
            context_disk_budget: None,
            cache: CacheSection::default(),
            agent: AgentSection::default(),
            standin: None,
            standin_after: standin_after(),
            standin_warm: true,
            alerts: weirdnet_alerts::AlertsConfig::default(),
            split: Vec::new(),
        }
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

/// `[inference.agent]`. Tasks answer with the pool's models and use no tools unless a
/// `workspace` is set; then the one tool is `files.read`, which asks the user's approval on
/// every call and reads only inside that directory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentSection {
    /// Serve tasks (default on). Off: the node refuses them.
    #[serde(default = "yes")]
    pub enabled: bool,
    /// The directory `files.read` may read from; unset means no tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace: Option<String>,
    /// Largest file `files.read` returns, in KiB.
    #[serde(default = "max_file_kib")]
    pub max_file_kib: u64,
}

fn max_file_kib() -> u64 {
    64
}

impl Default for AgentSection {
    fn default() -> Self {
        Self { enabled: true, workspace: None, max_file_kib: max_file_kib() }
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
        for c in &self.model.callers {
            if c.parse::<weirdnet_net::PeerId>().is_err() {
                bail!(
                    "[model] callers: `{c}` is not a peer id. Roster labels are not covered by the certificate signature, so they \
                     are no longer accepted; list the device's peer id instead (replace the label with the peer id of that linked device, as the pool's device list shows it)"
                );
            }
        }
        let mut seen = std::collections::HashSet::new();
        for m in &self.model.local {
            if m.id.is_empty() || m.id.contains(char::is_whitespace) {
                bail!("model id `{}` must be non-empty and have no spaces", m.id);
            }
            if !seen.insert(&m.id) {
                bail!("model `{}` is listed twice", m.id);
            }
            if let Some(b) = &m.backend
                && !BACKENDS.contains(&b.as_str())
            {
                bail!("model `{}`: unknown backend `{b}` (llamacpp, mlx or vllm)", m.id);
            }
            if m.backend.as_deref() == Some("mlx") && m.is_embedding() {
                bail!("model `{}`: MLX-LM's server has no embeddings endpoint; use backend = \"llamacpp\" for embedding models", m.id);
            }
            if let Some(t) = &m.idle_ttl {
                parse_duration(t).with_context(|| format!("model `{}`: idle_ttl", m.id))?;
            }
            if let Some(url) = &m.wrap {
                weirdnet_backends::Wrap { url: url.clone(), api_key: None }.check().map_err(|e| anyhow::anyhow!("model `{}`: {e}", m.id))?;
            }
            if let Some(src) = &m.wrap_api_key
                && !(src.starts_with("env:") || src.starts_with("file:"))
            {
                bail!("model `{}`: wrap_api_key must be env:NAME or file:/path, not the key itself", m.id);
            }
        }
        self.validate_splits()?;
        parse_duration(&self.inference.standin_after).context("inference.standin_after")?;
        self.network.hints()?;
        if let Some(s) = &self.inference.standin
            && !self.model.local.iter().any(|m| &m.id == s)
            && self.roles.model
        {
            bail!("inference.standin `{s}` is not one of this node's [[model.local]] models");
        }
        Ok(())
    }

    fn validate_splits(&self) -> Result<()> {
        if let Some(l) = &self.model.private_link {
            crate::group::parse_link(l).map_err(|e| anyhow::anyhow!("model.private_link: {e}"))?;
        }
        let mut ids = std::collections::HashSet::new();
        for s in &self.model.split {
            if s.id.is_empty() || s.id.contains(char::is_whitespace) {
                bail!("model.split id `{}` must be non-empty and have no spaces", s.id);
            }
            if !ids.insert(&s.id) || self.model.local.iter().any(|m| m.id == s.id) {
                bail!("model `{}` is listed twice (split models are not also [[model.local]])", s.id);
            }
        }
        let mut seen = std::collections::HashSet::new();
        for p in &self.inference.split {
            if p.backend != crate::group::BACKEND {
                bail!("inference.split `{}`: backend `{}` is not built (only `{}`)", p.id, p.backend, crate::group::BACKEND);
            }
            if !seen.insert(&p.id) {
                bail!("inference.split `{}` is listed twice", p.id);
            }
            if weirdnet_model::hw::parse_size(&p.size).is_none() {
                bail!("inference.split `{}`: size `{}` is not a size like 7GiB", p.id, p.size);
            }
            if p.nodes.is_empty() {
                bail!("inference.split `{}` lists no node", p.id);
            }
            for n in &p.nodes {
                crate::group::parse_link(&n.link).map_err(|e| anyhow::anyhow!("inference.split `{}` node `{}`: {e}", p.id, n.node))?;
                if weirdnet_model::hw::parse_size(&n.memory).is_none() {
                    bail!("inference.split `{}` node `{}`: memory `{}` is not a size like 10GiB", p.id, n.node, n.memory);
                }
            }
            if let Some(l) = &p.leader
                && !p.nodes.iter().any(|n| &n.node == l)
            {
                bail!("inference.split `{}`: leader `{l}` is not one of its nodes", p.id);
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

    /// The cold-start stand-in: `inference.standin`, else the model with `role = "standin"`.
    pub fn standin(&self) -> Option<String> {
        self.inference.standin.clone().or_else(|| self.model.local.iter().find(|m| m.role.as_deref() == Some("standin")).map(|m| m.id.clone()))
    }

    pub fn coordinator_options(&self) -> crate::coordinator::CoordinatorOptions {
        crate::coordinator::CoordinatorOptions {
            standin: self.standin(),
            standin_after: parse_duration(&self.inference.standin_after).unwrap_or(std::time::Duration::from_secs(10)),
            splits: self.inference.split.clone(),
        }
    }

    pub fn models_dir(&self) -> PathBuf {
        self.model.models_dir.clone().unwrap_or_else(|| weirdnet_model::ModelStore::from_env().dir().to_path_buf())
    }

    /// The engines this node can run, from `llamacpp`, `mlx` and `vllm`.
    ///
    /// llama.cpp and MLX-LM both run as the supervisor's llama.cpp kind: when a model uses MLX-LM,
    /// the program is `ymir-engine` (see [`crate::shim`]), which runs `llama-server` for GGUF
    /// sources and MLX-LM for `mlx:` sources.
    pub fn engines(&self, paths: &Paths) -> Vec<EngineConfig> {
        let mut out = Vec::new();
        let llama = resolve_engine(&self.model.llamacpp, "llama-server", &paths.engines().join("llama.cpp").join("llama-server"));
        let uses = |b: &str| self.model.local.iter().any(|m| m.backend.as_deref() == Some(b));
        let mlx = uses("mlx").then(|| crate::mlx::resolve(&self.model.mlx, &paths.engines())).flatten();
        match crate::shim::locate().filter(|_| uses("mlx")) {
            Some(shim) => {
                let mut e = EngineConfig::llamacpp(shim);
                for (flag, p) in [(crate::shim::LLAMA_FLAG, &llama), (crate::shim::MLX_FLAG, &mlx)] {
                    if let Some(p) = p {
                        e.extra_args.extend([flag.to_string(), p.to_string_lossy().into_owned()]);
                    }
                }
                out.push(e);
            }
            None => {
                if uses("mlx") {
                    tracing::warn!("models use mlx but {} is not installed next to ymird; they will fail to load", crate::shim::PROGRAM);
                }
                if let Some(p) = llama {
                    out.push(EngineConfig::llamacpp(p));
                }
            }
        }
        if let Some(p) = resolve_engine(&self.model.vllm, "vllm", &paths.engines().join("vllm").join("bin").join("vllm")) {
            out.push(EngineConfig::vllm(p));
        }
        out
    }

    /// `rpc-server`, when this node can be a worker.
    pub fn rpc_server(&self, paths: &Paths) -> Option<PathBuf> {
        resolve_engine(&self.model.rpc_server, "rpc-server", &paths.engines().join("llama.cpp").join("rpc-server"))
    }

    /// What this node's group workers need (`key`: the leader's `llama-server` key).
    pub fn worker_config(&self, paths: &Paths, key: String) -> crate::group::WorkerConfig {
        crate::group::WorkerConfig {
            rpc_server: self.rpc_server(paths),
            rpc_args: self.model.rpc_args.clone(),
            llama_server: resolve_engine(&self.model.llamacpp, "llama-server", &paths.engines().join("llama.cpp").join("llama-server")),
            private_link: self.model.private_link.clone(),
            splits: self.model.split.clone(),
            models_dir: self.models_dir(),
            key,
            log_dir: Some(paths.logs()),
        }
    }

    /// Launch specs for `[[model.split]]`: the leader's `llama-server`, which the group starts,
    /// wrapped like any stock server on loopback (it answers once the group is up).
    pub fn split_launches(&self, key: &str) -> Vec<ModelLaunch> {
        self.model
            .split
            .iter()
            .map(|s| {
                let mut l = ModelLaunch::new(&s.id, EngineKind::Llamacpp, s.file.clone().unwrap_or_default());
                l.context = s.context;
                l.slots = s.slots;
                l.wrap = Some(weirdnet_backends::Wrap { url: format!("http://127.0.0.1:{}", s.serve_port), api_key: Some(key.to_string()) });
                l
            })
            .collect()
    }

    /// Launch specs for `[[model.local]]`.
    pub fn launches(&self) -> Vec<ModelLaunch> {
        let dir = self.models_dir();
        self.model
            .local
            .iter()
            .map(|m| {
                let backend = m.backend.as_deref().unwrap_or("llamacpp");
                let engine = if backend == "vllm" { EngineKind::Vllm } else { EngineKind::Llamacpp };
                let source = match (&m.file, engine) {
                    _ if backend == "mlx" => format!("{}{}", crate::shim::MLX, mlx_source(m, &dir)),
                    (Some(f), EngineKind::Llamacpp) => {
                        let p = Path::new(f);
                        if p.is_absolute() { p.to_path_buf() } else { dir.join(p) }.to_string_lossy().into_owned()
                    }
                    (Some(f), _) => f.clone(),
                    (None, _) => match weirdnet_model::hw::find(&m.id) {
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
                let has_own = m.args.iter().any(|a| a == "--mmproj" || a == "-mm" || a == "--no-mmproj");
                if engine == EngineKind::Llamacpp
                    && backend == "llamacpp"
                    && m.wrap.is_none()
                    && !m.is_embedding()
                    && !has_own
                    && let Some(p) = crate::vision::resolve(&m.id, m.mmproj.as_deref(), &dir)
                {
                    l.extra_args.extend(["--mmproj".to_string(), p.to_string_lossy().into_owned()]);
                }
                l.idle_ttl = m.idle_ttl.as_deref().and_then(|t| parse_duration(t).ok());
                l.wrap = m.wrap.as_ref().map(|url| weirdnet_backends::Wrap {
                    url: url.clone(),
                    api_key: m.wrap_api_key.as_deref().and_then(|src| match weirdnet_alerts::config::resolve_secret(src) {
                        Ok(k) => Some(k),
                        Err(e) => {
                            tracing::warn!(model = %m.id, "wrap_api_key: {e}");
                            None
                        }
                    }),
                });
                l
            })
            .collect()
    }
}

/// The `[[model.local]]` backends.
pub const BACKENDS: &[&str] = &["llamacpp", "mlx", "vllm"];

/// Where an MLX model's weights are: `file` (a directory, absolute or under `models_dir`, or a Hub
/// id), else the catalog model's MLX weights under `models_dir/mlx/`.
fn mlx_source(m: &LocalModel, dir: &Path) -> String {
    match &m.file {
        Some(f) if Path::new(f).is_absolute() => f.clone(),
        Some(f) if dir.join(f).exists() || !f.contains('/') => dir.join(f).to_string_lossy().into_owned(),
        Some(f) => f.clone(),
        None => match crate::mlx::weights(&m.id) {
            Some(w) => w.dir(dir).to_string_lossy().into_owned(),
            None => dir.join("mlx").join(&m.id).to_string_lossy().into_owned(),
        },
    }
}

/// `600` (seconds), `30s`, `10m`, `2h` or `1d`.
pub fn parse_duration(s: &str) -> Result<std::time::Duration> {
    let s = s.trim();
    let (num, unit) = s.find(|c: char| !c.is_ascii_digit() && c != '.').map_or((s, ""), |i| s.split_at(i));
    let n: f64 = num.parse().with_context(|| format!("`{s}` is not a duration (like 30s, 10m, 2h)"))?;
    let mult = match unit.trim() {
        "" | "s" => 1.0,
        "m" | "min" => 60.0,
        "h" => 3600.0,
        "d" => 86400.0,
        "ms" => 0.001,
        other => bail!("`{s}`: unknown unit `{other}` (s, m, h, d)"),
    };
    if !(n.is_finite() && n >= 0.0) {
        bail!("`{s}` is not a duration");
    }
    Ok(std::time::Duration::from_secs_f64(n * mult))
}

/// `managed` looks in Ymir's engines dir, then at the engine shipped with this install (see
/// [`crate::engine`]), then on `PATH`; `system:/path` uses that; `auto` (vLLM) looks on `PATH`;
/// `off` disables.
fn resolve_engine(setting: &str, binary: &str, managed_path: &Path) -> Option<PathBuf> {
    resolve_engine_with(setting, binary, managed_path, crate::engine::bundled().as_ref())
}

fn resolve_engine_with(setting: &str, binary: &str, managed_path: &Path, bundled: Option<&crate::engine::Bundled>) -> Option<PathBuf> {
    match setting.trim() {
        "off" => None,
        s if s.starts_with("system:") => Some(PathBuf::from(&s["system:".len()..])),
        "managed" => Some(managed_path.to_path_buf())
            .filter(|p| p.exists())
            .or_else(|| bundled.map(|b| crate::engine::program(&b.dir, binary)).filter(|p| p.is_file()))
            .or_else(|| find_in_path(binary))
            // llama.cpp renamed `rpc-server` to `ggml-rpc-server` in its release builds.
            .or_else(|| (binary == "rpc-server").then(|| find_in_path("ggml-rpc-server")).flatten()),
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
    fn m2_settings() {
        let cfg: NodeConfig = toml::from_str(
            r#"
            [roles]
            model = true
            inference = true
            [[model.local]]
            id = "big"
            swap_group = "main"
            idle_ttl = "10m"
            [[model.local]]
            id = "small"
            role = "standin"
            [[model.local]]
            id = "vl"
            backend = "vllm"
            wrap = "http://127.0.0.1:8000/v1"
            wrap_api_key = "env:YMIR_TEST_UNSET_KEY"
            [inference]
            standin_after = "5s"
            [inference.alerts]
            dry_run = true
            [inference.alerts.smtp]
            server = "smtp.example.com"
            from = "a@example.com"
            to = ["b@example.com"]
            "#,
        )
        .unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.standin().as_deref(), Some("small"));
        assert_eq!(cfg.coordinator_options().standin_after, std::time::Duration::from_secs(5));
        let l = cfg.launches();
        assert_eq!(l[0].idle_ttl, Some(std::time::Duration::from_secs(600)));
        assert_eq!(l[2].wrap.as_ref().unwrap().base(), "http://127.0.0.1:8000");
        assert!(cfg.inference.alerts.dry_run);
        let back: NodeConfig = toml::from_str(&cfg.to_toml().unwrap()).unwrap();
        assert_eq!(back, cfg);

        let mut bad = cfg.clone();
        bad.model.local[2].wrap = Some("http://10.1.2.3:8000".into());
        assert!(bad.validate().is_err(), "wrap must be loopback");
        let mut bad = cfg.clone();
        bad.model.local[2].wrap_api_key = Some("sk-123".into());
        assert!(bad.validate().is_err(), "no inline keys");
        assert_eq!(parse_duration("90").unwrap().as_secs(), 90);
        assert_eq!(parse_duration("2h").unwrap().as_secs(), 7200);
        assert!(parse_duration("soon").is_err());
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
        // Compared as paths: on Windows the join uses `\`.
        assert_eq!(Path::new(&l[0].source), Path::new("/models/Qwen3-8B-Q4_K_M.gguf"));
        assert_eq!(Path::new(&l[1].source), Path::new("/models/x/mine.gguf"));
        assert!(l[1].embedding);
    }
    #[test]
    fn managed_prefers_the_engines_dir_then_the_bundled_engine() {
        let t = tempfile::tempdir().unwrap();
        let bundle = t.path().join("lib/ymir/llama.cpp");
        std::fs::create_dir_all(&bundle).unwrap();
        // `llama-server.exe` on Windows.
        let program = crate::engine::program(&bundle, "llama-server");
        std::fs::write(&program, b"x").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let b = crate::engine::Bundled::at(&bundle).unwrap();
        let managed = t.path().join("home/engines/llama.cpp/llama-server");
        let got = resolve_engine_with("managed", "llama-server", &managed, Some(&b));
        assert_eq!(got, Some(program));
        std::fs::create_dir_all(managed.parent().unwrap()).unwrap();
        std::fs::write(&managed, b"x").unwrap();
        assert_eq!(resolve_engine_with("managed", "llama-server", &managed, Some(&b)), Some(managed.clone()));
        assert_eq!(resolve_engine_with("system:/opt/ls", "llama-server", &managed, Some(&b)), Some(PathBuf::from("/opt/ls")));
        assert_eq!(resolve_engine_with("off", "llama-server", &managed, Some(&b)), None);
    }
}
