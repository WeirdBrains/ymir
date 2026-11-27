//! `node.toml`: one file per node. Each role reads only its own section.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use weirdnet_backends::{EngineConfig, EngineKind, LoraAdapter, ModelLaunch, find_in_path};

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
    /// Fine-tuning on this node (training plan P3a): the trainer and the build it is pinned to.
    #[serde(default, skip_serializing_if = "crate::train::TrainSection::is_default")]
    pub train: crate::train::TrainSection,
    /// Hands-off self-update (`ymir_core::autoupdate`): on by default where an update root is pinned.
    #[serde(default, skip_serializing_if = "crate::autoupdate::UpdateSection::is_default")]
    pub update: crate::autoupdate::UpdateSection,
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
    /// Let the owner's linked devices search, pull, upload, list and remove models on this node over
    /// Weirdnet (`ymirctl models --node`, `/weirdnet-backends/models/1`). Pulled models are served in
    /// memory only; an upload is also saved here unless the uploader passes `--no-persist`
    /// (`/ymir/uploads/1`). `false` turns both protocols off.
    #[serde(default = "yes", skip_serializing_if = "is_yes")]
    pub remote_manage: bool,
    /// Take model uploads from the owner's linked devices (`ymirctl models upload --node`,
    /// `/ymir/uploads/1` and models/1 uploads). `false` refuses every upload (`not-allowed`, "uploads
    /// are off on this node") while pulls, listing and removals keep working; `remote_manage = false`
    /// turns all of it off instead. Uploads placed on this machine itself are not affected.
    #[serde(default = "yes", skip_serializing_if = "is_yes")]
    pub upload: bool,
    /// Free space an upload must leave on the models directory's filesystem, such as `50GiB`.
    /// Unset: the larger of 10% of the filesystem and 20 GiB. Set: exactly this much.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upload_keep_free: Option<String>,
    /// The engine an uploaded full model must suit: `llamacpp` (GGUF), `vllm` (safetensors) or
    /// `mlx` (MLX weights). Unset: llama.cpp, unless `llamacpp` is `off`; then vLLM, unless that is
    /// `off` too; then MLX. Adapters and projectors follow their base model's engine instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upload_engine: Option<String>,
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
    /// `auto` (the venv `ymirctl engines install vllm` makes, else `PATH`) |
    /// `system:/path/to/vllm` | `image:NAME` | `off`.
    #[serde(default = "auto")]
    pub vllm: String,
    /// Ray, for `vllm-ray` split models: `auto` (the managed vLLM venv, else `PATH`) |
    /// `system:/path/to/ray` | `off`.
    #[serde(default = "auto", skip_serializing_if = "is_auto")]
    pub ray: String,
    /// More `ray start` arguments (such as `--num-gpus=2`); never the bind flags Ymir sets.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ray_args: Vec<String>,
    /// Allow Ray listeners on all interfaces. Ray has no authentication and its GCS ignores
    /// `--node-ip-address`, so a `vllm-ray` group is refused unless this says a host firewall
    /// drops Ray's ports on every interface but the private link.
    #[serde(default, skip_serializing_if = "is_false")]
    pub ray_trust_firewall: bool,
    /// The ports the host firewall drops off the private link (`"6390-6399"`, `"10001"`), where
    /// `ray_trust_firewall` allows all-interface listeners; default 6390-6399 and 10000-10999. An
    /// all-interface listener on any other port is refused.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ray_trusted_ports: Vec<String>,
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

fn is_yes(b: &bool) -> bool {
    *b
}

fn is_false(b: &bool) -> bool {
    !*b
}

fn is_auto(s: &str) -> bool {
    s == "auto"
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
            remote_manage: true,
            upload: true,
            upload_keep_free: None,
            upload_engine: None,
            llamacpp: managed(),
            rpc_server: managed(),
            rpc_args: Vec::new(),
            vllm: auto(),
            ray: auto(),
            ray_args: Vec::new(),
            ray_trust_firewall: false,
            ray_trusted_ports: Vec::new(),
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
    /// The weights' revision: a Hub commit, a file hash or a fine-tune id. Optional; when set it
    /// goes in the capability record and keys the answer cache, so answers from other weights
    /// under the same `id` are never replayed. Change it when the weights change.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub swap_group: Option<String>,
    /// `main`, `standin`, `embedding` or `transcribe` (speech to text: Voxtral through llama.cpp
    /// with its audio projector, Whisper through vLLM; served over infer/2 `Transcribe` and
    /// published with catalog/1 purpose `transcribe`); default `chat`.
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
    /// A container image that provides `vllm` (for example `vllm/vllm-openai:inkling-2-mini`).
    /// When `[model] vllm` is `auto` and no `wrap` is set, ymird runs this image through docker
    /// or podman (see [`crate::vllm::write_image_wrapper`]). Every unwrapped vLLM model on the
    /// node must name the same image, because the supervisor has one vLLM program.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
    /// LoRA adapters served on this model (`[[model.local.lora]]`), each under its own pool id
    /// and sharing this model's engine: llama.cpp and vLLM only, not wrapped, not embedding.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lora: Vec<LocalAdapter>,
    /// vLLM only: start the engine with LoRA on (`--enable-lora`, `--max-loras` at least this many
    /// adapters at once) even with no `lora` listed, so adapters can be served on it later without
    /// a restart (`ymirctl models adapter serve`, or a pull with `--adapter-for --serve`). A vLLM
    /// base with adapters listed takes more at run time without it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora_slots: Option<u32>,
    /// Media and request limits for this model on this hardware, for limits the node can't
    /// measure (`limits = { max_images = 8, image_tokens = 256, image_edge = 896 }`; also
    /// `image_patch`, `audio_max_seconds`, `audio_tokens_per_minute`, `max_request_bytes`).
    /// Published in the capability record and catalog/1 (`weirdnet-agent` `MediaLimits`). Unset
    /// fields stay unknown: there are no defaults in code (video design, milestone 1).
    #[serde(default, skip_serializing_if = "weirdnet_agent::MediaLimits::is_unknown")]
    pub limits: weirdnet_agent::MediaLimits,
}

/// A LoRA adapter on a `[[model.local]]` base, served as its own model id (for example
/// `qwen3-8b+support-v2`). Requests for that id run on the base's engine with the adapter
/// applied; the base keeps answering under its own id.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LocalAdapter {
    /// The pool id it serves as. Unique among this node's models and adapters.
    pub id: String,
    /// For llama.cpp a LoRA GGUF (absolute, or relative to `models_dir`) made for this exact base
    /// (its architecture is checked); for vLLM a PEFT adapter directory (likewise) or a Hub id.
    pub file: String,
    /// How strongly it applies, per request on llama.cpp (default 1.0). vLLM ignores it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<f32>,
    /// The adapter's revision (a training run id or file hash): keys the answer cache like a
    /// model's `revision`, so answers from an older adapter under the same id are never replayed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    /// The base model's `revision` this adapter was trained on. When both it and the base's
    /// `revision` are set and differ, the adapter is not served (a LoRA only fits the weights it
    /// was trained on). `ymirctl models adapter add` records the base's revision here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_revision: Option<String>,
    /// The base's `quant` this adapter was trained on (for example `Q4_K_M`); checked like
    /// `base_revision` against the base's `quant`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_quant: Option<String>,
}

impl ModelSection {
    /// What an upload must leave free on the models directory's filesystem ([`Self::upload_keep_free`]).
    pub fn keep_free(&self) -> weirdnet_backends::models::KeepFree {
        match self.upload_keep_free.as_deref().and_then(weirdnet_model::hw::parse_size) {
            Some(bytes) => weirdnet_backends::models::KeepFree::new(0, bytes),
            None => weirdnet_backends::models::KeepFree::default(),
        }
    }

    /// The engine an uploaded full model must suit ([`Self::upload_engine`]).
    pub fn upload_engine(&self) -> EngineKind {
        match self.upload_engine.as_deref() {
            Some("vllm") => EngineKind::Vllm,
            Some("mlx") => EngineKind::Mlx,
            Some(_) => EngineKind::Llamacpp,
            None if self.llamacpp != "off" => EngineKind::Llamacpp,
            None if self.vllm != "off" => EngineKind::Vllm,
            None if self.mlx != "off" => EngineKind::Mlx,
            None => EngineKind::Llamacpp,
        }
    }
}

impl LocalModel {
    pub fn is_embedding(&self) -> bool {
        self.role.as_deref() == Some("embedding")
    }

    /// Whether it has the transcription role (`role = "transcribe"`, video design milestone 2).
    pub fn is_transcriber(&self) -> bool {
        self.role.as_deref() == Some("transcribe")
    }
}

impl LocalAdapter {
    /// Why this adapter doesn't fit `base`'s weights: its `base_revision` or `base_quant` differs
    /// from the base's `revision` or `quant` (both set). `None` when they match or either is unset.
    pub fn base_mismatch(&self, base: &LocalModel) -> Option<String> {
        let differs = |want: &Option<String>, have: &Option<String>| match (want, have) {
            (Some(w), Some(h)) if !w.trim().eq_ignore_ascii_case(h.trim()) => Some((w.clone(), h.clone())),
            _ => None,
        };
        if let Some((w, h)) = differs(&self.base_revision, &base.revision) {
            return Some(format!(
                "it was trained on revision {w} of `{}`, which is now at revision {h}; retrain it, or set its base_revision if it still fits",
                base.id
            ));
        }
        if let Some((w, h)) = differs(&self.base_quant, &base.quant) {
            return Some(format!(
                "it was trained on the {w} quantisation of `{}`, which is now {h}; retrain it, or set its base_quant if it still fits",
                base.id
            ));
        }
        None
    }

    /// The adapter path the engine takes: a path under `models_dir` when relative and present
    /// there (or not a Hub id), else as written.
    pub fn path(&self, models_dir: &Path) -> String {
        let p = Path::new(&self.file);
        if p.is_absolute() {
            return self.file.clone();
        }
        let under = models_dir.join(p);
        if under.exists() || self.file.matches('/').count() != 1 || self.file.ends_with(".gguf") {
            under.to_string_lossy().into_owned()
        } else {
            self.file.clone()
        }
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
    /// Placement and replication (M6): models to keep loaded on a number of model nodes.
    #[serde(default, skip_serializing_if = "crate::placement::PlacementPolicy::is_empty")]
    pub placement: crate::placement::PlacementPolicy,
    /// Versioned rollouts (training plan P4b): a name, its versions and their traffic weights.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rollout: Vec<crate::rollout::Rollout>,
    /// Let the owner's linked devices read and change these rollouts over Weirdnet
    /// (`/ymir/rollouts/1`: Bragi, `ymirctl models rollout --inference`), with the same rules as
    /// `ymirctl models rollout` here; changes are saved in this file unless the caller asks not to.
    /// `false` turns the protocol off.
    #[serde(default = "yes", skip_serializing_if = "is_yes")]
    pub remote_rollouts: bool,
    /// Which linked devices may do that, by peer id (as `[train] callers`). Empty (the default):
    /// any linked device.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rollout_callers: Vec<String>,
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
            placement: Default::default(),
            rollout: Vec::new(),
            remote_rollouts: true,
            rollout_callers: Vec::new(),
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
    /// Cosine distance (0 = the same meaning) under which a stored answer stands in for a prompt
    /// in the semantic cache.
    #[serde(default = "threshold")]
    pub threshold: f32,
    /// The embedding model (an id in the pool's catalog) the semantic cache embeds prompts with.
    /// Unset: exact matches only. It is used only while loaded somewhere.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_model: Option<String>,
    /// How long a semantic-cache entry may stand in (`24h`).
    #[serde(default = "ttl")]
    pub ttl: String,
    /// At most this many semantic-cache entries; the least recently used go first.
    #[serde(default = "entries")]
    pub entries: usize,
}

fn ttl() -> String {
    "24h".into()
}

fn entries() -> usize {
    1024
}

impl CacheSection {
    /// The semantic cache's settings, `None` when it is off (no embedding model, or `mode = "off"`).
    pub fn semantic(&self) -> Option<crate::semantic::SemanticOptions> {
        let model = self.embedding_model.clone().filter(|m| !m.trim().is_empty() && self.mode != "off")?;
        Some(crate::semantic::SemanticOptions {
            embedding_model: model,
            similarity: 1.0 - self.threshold,
            ttl: parse_duration(&self.ttl).ok(),
            max_entries: self.entries,
            pool: self.mode == "pool",
        })
    }

    pub fn validate(&self) -> Result<()> {
        if !matches!(self.mode.as_str(), "off" | "local" | "pool") {
            bail!("[inference.cache] mode must be off, local or pool, not {:?}", self.mode);
        }
        if !(0.0..=1.0).contains(&self.threshold) {
            bail!("[inference.cache] threshold is a cosine distance from 0 to 1, not {}", self.threshold);
        }
        if parse_duration(&self.ttl).is_err() {
            bail!("[inference.cache] ttl {:?} is not a duration (like 24h)", self.ttl);
        }
        if !(1..=1_000_000).contains(&self.entries) {
            bail!("[inference.cache] entries must be 1 to 1000000, not {}", self.entries);
        }
        Ok(())
    }
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
        Self { mode: local(), backend: embedded(), redis_url: None, threshold: threshold(), embedding_model: None, ttl: ttl(), entries: entries() }
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
    /// Outside MCP servers whose tools tasks may use (`[[inference.agent.mcp]]`). Every call
    /// asks the user's approval on the asking device.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mcp: Vec<crate::mcp_servers::McpServerConfig>,
}

fn max_file_kib() -> u64 {
    64
}

impl Default for AgentSection {
    fn default() -> Self {
        Self { enabled: true, workspace: None, max_file_kib: max_file_kib(), mcp: Vec::new() }
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
    /// Largest request body the bridge accepts, in bytes (`weirdnet-openai`'s 32 MiB when unset).
    /// A larger one gets a 413 that names this setting; `/v1/models` reports it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_request_bytes: Option<u64>,
}

fn bridge_port() -> u16 {
    8484
}

impl Default for BridgeSection {
    fn default() -> Self {
        Self { port: bridge_port(), max_request_bytes: None }
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
        self.inference.placement.validate()?;
        self.inference.cache.validate()?;
        for c in &self.inference.rollout_callers {
            if c.parse::<weirdnet_net::PeerId>().is_err() {
                bail!("[inference] rollout_callers: `{c}` is not a peer id (list the device's peer id, as the pool's device list shows it)");
            }
        }
        for c in &self.model.callers {
            if c.parse::<weirdnet_net::PeerId>().is_err() {
                bail!(
                    "[model] callers: `{c}` is not a peer id. Roster labels are not covered by the certificate signature, so they \
                     are no longer accepted; list the device's peer id instead (replace the label with the peer id of that linked device, as the pool's device list shows it)"
                );
            }
        }
        if let Some(k) = &self.model.upload_keep_free
            && weirdnet_model::hw::parse_size(k).is_none()
        {
            bail!("[model] upload_keep_free: `{k}` is not a size (such as `50GiB`)");
        }
        if let Some(e) = &self.model.upload_engine
            && !matches!(e.as_str(), "llamacpp" | "vllm" | "mlx")
        {
            bail!("[model] upload_engine: `{e}` is not `llamacpp`, `vllm` or `mlx`");
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
            if m.backend.as_deref() == Some("mlx") && m.is_embedding() && !EngineKind::Mlx.capabilities().embeddings {
                bail!("model `{}`: MLX-LM's server has no embeddings endpoint; use backend = \"llamacpp\" for embedding models", m.id);
            }
            if m.backend.as_deref() == Some("mlx") && m.is_transcriber() {
                bail!("model `{}`: role = \"transcribe\" needs backend llamacpp (with an audio projector) or vllm, not mlx", m.id);
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
            if let Some(img) = &m.image {
                if m.backend.as_deref() != Some("vllm") {
                    bail!("model `{}`: image is only valid with backend = \"vllm\"", m.id);
                }
                if !crate::vllm::valid_image(img) {
                    bail!("model `{}`: image `{img}` is not a container image name (registry/name:tag)", m.id);
                }
            }
            if m.backend.as_deref() == Some("mlx") && crate::vllm::is_inkling(m.file.as_deref().unwrap_or(&m.id)) {
                bail!("model `{}`: {}", m.id, crate::vllm::mlx_not_supported_message());
            }
            if !m.lora.is_empty() {
                if m.backend.as_deref() == Some("mlx") {
                    bail!("model `{}`: LoRA adapters need backend llamacpp or vllm, not mlx", m.id);
                }
                if m.wrap.is_some() {
                    bail!("model `{}`: a wrapped server can't take LoRA adapters from ymird", m.id);
                }
                if m.is_embedding() {
                    bail!("model `{}`: embedding models take no LoRA adapters", m.id);
                }
            }
            if let Some(n) = m.lora_slots {
                if m.backend.as_deref() != Some("vllm") {
                    bail!(
                        "model `{}`: lora_slots is for backend = \"vllm\" (llama-server loads adapter files only at start: list them under [[model.local.lora]])",
                        m.id
                    );
                }
                if m.wrap.is_some() {
                    bail!("model `{}`: a wrapped server can't take LoRA adapters from ymird; start it with --enable-lora yourself", m.id);
                }
                if m.is_embedding() {
                    bail!("model `{}`: embedding models take no LoRA adapters", m.id);
                }
                if n == 0 {
                    bail!("model `{}`: lora_slots must be at least 1 (leave it out to keep LoRA off)", m.id);
                }
            }
        }
        for m in &self.model.local {
            for a in &m.lora {
                if a.id.is_empty() || a.id.contains(char::is_whitespace) {
                    bail!("model `{}`: adapter id `{}` must be non-empty and have no spaces", m.id, a.id);
                }
                if !seen.insert(&a.id) {
                    bail!("adapter `{}` on `{}`: that id is already a model or adapter on this node", a.id, m.id);
                }
                if a.file.trim().is_empty() {
                    bail!("adapter `{}` on `{}`: file is required", a.id, m.id);
                }
                if a.scale.is_some_and(|s| !s.is_finite()) {
                    bail!("adapter `{}` on `{}`: scale must be a finite number", a.id, m.id);
                }
            }
        }
        let vllm_images: Vec<&str> =
            self.model.local.iter().filter(|m| m.backend.as_deref() == Some("vllm") && m.wrap.is_none()).filter_map(|m| m.image.as_deref()).collect();
        if vllm_images.windows(2).any(|w| w[0] != w[1]) {
            bail!(
                "[[model.local]] image: every unwrapped vLLM model on this node must use the same image (the supervisor has one vLLM program); wrap a server you started if you need more than one"
            );
        }
        crate::mcp_servers::validate(&self.inference.agent.mcp)?;
        self.validate_splits()?;
        crate::rollout::validate(&self.inference.rollout).context("[[inference.rollout]]")?;
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
        crate::ray::check_args(&self.model.ray_args).map_err(|e| anyhow::anyhow!("model.{e}"))?;
        for s in &self.model.split {
            if s.id.is_empty() || s.id.contains(char::is_whitespace) {
                bail!("model.split id `{}` must be non-empty and have no spaces", s.id);
            }
            if let Some(e) = &s.engine
                && e != "llamacpp"
                && e != "vllm"
            {
                bail!("model.split `{}`: engine `{e}` is not `llamacpp` or `vllm`", s.id);
            }
            if !ids.insert(&s.id) || self.model.local.iter().any(|m| m.id == s.id) {
                bail!("model `{}` is listed twice (split models are not also [[model.local]])", s.id);
            }
        }
        let mut seen = std::collections::HashSet::new();
        for p in &self.inference.split {
            if p.backend != crate::group::BACKEND && p.backend != crate::group::RAY_BACKEND {
                bail!("inference.split `{}`: backend `{}` is not `{}` or `{}`", p.id, p.backend, crate::group::BACKEND, crate::group::RAY_BACKEND);
            }
            if p.tensor_parallel == Some(0) || p.pipeline_parallel == Some(0) || p.nodes.iter().any(|n| n.gpus == Some(0)) {
                bail!("inference.split `{}`: tensor_parallel, pipeline_parallel and gpus are at least 1", p.id);
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
        self.model
            .local
            .iter()
            .find(|m| m.role.as_deref() == Some("main"))
            .or_else(|| self.model.local.iter().find(|m| !m.is_embedding() && !m.is_transcriber()))
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
            rollouts: self.inference.rollout.clone(),
            semantic: self.inference.cache.semantic(),
        }
    }

    pub fn models_dir(&self) -> PathBuf {
        self.model.models_dir.clone().unwrap_or_else(|| weirdnet_model::ModelStore::from_env().dir().to_path_buf())
    }

    /// The engines this node can run, from `llamacpp`, `mlx` and `vllm`. MLX-LM is configured
    /// only when a model uses it (it runs on Apple Silicon only).
    pub fn engines(&self, paths: &Paths) -> Vec<EngineConfig> {
        let mut out = Vec::new();
        if let Some(p) = resolve_engine(&self.model.llamacpp, "llama-server", &paths.engines().join("llama.cpp").join("llama-server")) {
            out.push(EngineConfig::llamacpp(p));
        }
        if self.model.local.iter().any(|m| m.backend.as_deref() == Some("mlx")) {
            match crate::mlx::resolve(&self.model.mlx, &paths.engines()) {
                Some(p) => out.push(EngineConfig::mlx(p)),
                None => tracing::warn!("models use mlx but MLX-LM is not installed (run `ymirctl setup --download`); they will fail to load"),
            }
        }
        if let Some(p) = resolve_vllm(&self.model, paths) {
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
            ray: resolve_ray(&self.model, paths),
            ray_args: self.model.ray_args.clone(),
            vllm: resolve_vllm(&self.model, paths),
            ray_trust_firewall: self.model.ray_trust_firewall,
            ray_trusted_ports: self.model.ray_trusted_ports.clone(),
        }
    }

    /// Launch specs for `[[model.split]]`: the leader's `llama-server`, which the group starts,
    /// wrapped like any stock server on loopback (it answers once the group is up).
    pub fn split_launches(&self, key: &str) -> Vec<ModelLaunch> {
        self.model
            .split
            .iter()
            .map(|s| {
                let engine = if s.engine.as_deref() == Some("vllm") { EngineKind::Vllm } else { EngineKind::Llamacpp };
                let mut l = ModelLaunch::new(&s.id, engine, s.file.clone().unwrap_or_default());
                l.context = s.context;
                l.slots = s.slots;
                l.wrap = Some(weirdnet_backends::Wrap { url: format!("http://127.0.0.1:{}", s.serve_port), api_key: Some(key.to_string()) });
                l
            })
            .collect()
    }

    /// `(id, revision, quant)` for each `[[model.local]]` that names a revision, and for each of
    /// their `lora` adapters that does (with the base's quant: an adapter runs on it).
    pub fn revisions(&self) -> Vec<(String, String, Option<String>)> {
        let models = self.model.local.iter().filter_map(|m| Some((m.id.clone(), m.revision.clone()?, m.quant.clone())));
        let adapters = self.model.local.iter().flat_map(|m| m.lora.iter().filter_map(|a| Some((a.id.clone(), a.revision.clone()?, m.quant.clone()))));
        models.chain(adapters).collect()
    }

    /// `(id, origin, license)` for each `[[model.local]]` model and LoRA adapter whose origin is
    /// known, for the capability record (and so the pool catalog). A `provenance.json` beside the
    /// weights (an upload, or a trained adapter) says both; else weights under `models_dir`
    /// (`ymirctl models pull`, a catalog model) or a Hub id are `hub`, with no license. A file
    /// elsewhere without a record, and a `wrap`ped server, say nothing.
    pub fn origins(&self) -> Vec<(String, Option<String>, Option<String>)> {
        let dir = self.models_dir();
        let mut out = Vec::new();
        for (m, l) in self.model.local.iter().zip(self.launches()) {
            if m.wrap.is_none() {
                let (o, lic) = origin_of(&l.source, &dir);
                if o.is_some() || lic.is_some() {
                    out.push((m.id.clone(), o, lic));
                }
            }
            for a in &l.adapters {
                let (o, lic) = origin_of(&a.path, &dir);
                if o.is_some() || lic.is_some() {
                    out.push((a.id.clone(), o, lic));
                }
            }
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
                let backend = m.backend.as_deref().unwrap_or("llamacpp");
                let engine = match backend {
                    "vllm" => EngineKind::Vllm,
                    "mlx" => EngineKind::Mlx,
                    _ => EngineKind::Llamacpp,
                };
                let source = match (&m.file, engine) {
                    (_, EngineKind::Mlx) => mlx_source(m, &dir),
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
                l.limits = m.limits;
                l.swap_group.clone_from(&m.swap_group);
                l.embedding = m.is_embedding();
                l = l.with_transcribe(m.is_transcriber());
                l.extra_args.clone_from(&m.args);
                // A context flag in `args` is what the engine would run with (it comes last), so it
                // is the context: one flag on the command line, and the same number advertised.
                let flags = crate::kvfit::context_flags(engine);
                if let Some(c) = crate::kvfit::arg_value(&m.args, flags).and_then(|v| v.parse::<u32>().ok()) {
                    l.context = Some(c);
                    l.extra_args = crate::kvfit::strip_arg(&m.args, flags);
                }
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
                l.lora_slots = m.lora_slots;
                l.with_adapters(
                    m.lora
                        .iter()
                        .map(|a| {
                            let la = LoraAdapter::new(&a.id, a.path(&dir));
                            match a.scale {
                                Some(s) => la.with_scale(s),
                                None => la,
                            }
                        })
                        .collect(),
                )
            })
            .collect()
    }
}

/// Where the weights at `source` (a path, or a Hub id) came from, and their license; see
/// [`NodeConfig::origins`].
fn origin_of(source: &str, models_dir: &Path) -> (Option<String>, Option<String>) {
    let p = Path::new(source);
    let record_dir = if p.is_dir() { Some(p) } else { p.parent() };
    if let Some(rec) = record_dir.filter(|d| !d.as_os_str().is_empty()).and_then(weirdnet_model::artifact::Provenance::read) {
        return (Some(rec.origin).filter(|o| !o.is_empty()), rec.license);
    }
    let hub_id = !p.is_absolute() && !p.exists() && source.matches('/').count() == 1;
    if p.starts_with(models_dir) || hub_id {
        return (Some(weirdnet_agent::remote::catalog::ORIGIN_HUB.to_string()), None);
    }
    (None, None)
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

/// `auto` or `managed` uses a container image when `[model] vllm = "image:..."` or a local
/// vLLM model sets `image` and is not wrapped.
pub fn resolve_vllm(model: &ModelSection, paths: &Paths) -> Option<PathBuf> {
    let setting = model.vllm.trim();
    if setting == "off" {
        return None;
    }
    if let Some(img) = setting.strip_prefix("image:") {
        return match crate::vllm::write_image_wrapper(&paths.engines(), img.trim()) {
            Ok(p) => Some(p),
            Err(e) => {
                tracing::warn!("vLLM image: {e}");
                None
            }
        };
    }
    if matches!(setting, "auto" | "managed")
        && let Some(img) = vllm_image_from_models(model)
    {
        return match crate::vllm::write_image_wrapper(&paths.engines(), &img) {
            Ok(p) => Some(p),
            Err(e) => {
                tracing::warn!("vLLM image: {e}");
                None
            }
        };
    }
    let managed = crate::vllm::bin_in(&crate::vllm::venv(&paths.engines()), "vllm");
    if setting == "auto" && managed.is_file() {
        return Some(managed);
    }
    resolve_engine(setting, "vllm", &managed)
}

/// `ray`: `auto` (or `managed`) is the managed vLLM venv's, else `PATH`'s.
pub fn resolve_ray(model: &ModelSection, paths: &Paths) -> Option<PathBuf> {
    let managed = crate::vllm::bin_in(&crate::vllm::venv(&paths.engines()), "ray");
    match model.ray.trim() {
        "off" => None,
        s if s.starts_with("system:") => Some(PathBuf::from(&s["system:".len()..])),
        _ => Some(managed).filter(|p| p.is_file()).or_else(|| find_in_path("ray")),
    }
}

/// The image unwrapped vLLM models agree on, if any.
pub fn vllm_image_from_models(model: &ModelSection) -> Option<String> {
    let mut imgs: Vec<&str> =
        model.local.iter().filter(|m| m.backend.as_deref() == Some("vllm") && m.wrap.is_none()).filter_map(|m| m.image.as_deref()).collect();
    imgs.sort_unstable();
    imgs.dedup();
    match imgs.as_slice() {
        [one] => Some((*one).to_string()),
        _ => None,
    }
}

/// Whether this node's vLLM program is (or will be) a container-image wrapper.
pub fn vllm_via_image(model: &ModelSection) -> bool {
    model.vllm.trim().starts_with("image:") || vllm_image_from_models(model).is_some()
}

/// `managed` looks in Ymir's engines dir, then at the engine shipped with this install (see
/// [`crate::engine`]), then on `PATH`; `system:/path` uses that; `auto` (vLLM) looks on `PATH`;
/// `off` disables. `image:name` is handled by [`resolve_vllm`].
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
    fn origins_come_from_provenance_records_else_the_models_dir_or_a_hub_id() {
        let models = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        let rec = |dir: &Path, kind: &str, id: &str, origin: &str, license: Option<&str>| {
            std::fs::create_dir_all(dir).unwrap();
            let mut r = serde_json::json!({"format": "weirdnet-provenance/1", "kind": kind, "id": id, "origin": origin, "revision": "sha256:00", "files": []});
            if let Some(l) = license {
                r["license"] = l.into();
            }
            std::fs::write(dir.join(weirdnet_model::artifact::PROVENANCE_FILE), serde_json::to_vec(&r).unwrap()).unwrap();
        };
        rec(&models.path().join("uploads/up"), "model", "up", "upload", Some("MIT"));
        rec(&models.path().join("adapters/tune"), "adapter", "tune", "train", Some("Apache-2.0"));
        std::fs::write(elsewhere.path().join("own.gguf"), b"x").unwrap();
        let text = format!(
            r#"
            [model]
            models_dir = '{m}'
            [[model.local]]
            id = "up"
            file = "uploads/up/up.gguf"
            [[model.local.lora]]
            id = "up+tune"
            file = "adapters/tune/tune.gguf"
            [[model.local]]
            id = "pulled"
            file = "pulled-Q4_K_M.gguf"
            [[model.local]]
            id = "own"
            file = '{e}/own.gguf'
            [[model.local]]
            id = "big"
            backend = "vllm"
            file = "org/big"
            [[model.local]]
            id = "wrapped"
            wrap = "http://127.0.0.1:8000"
            "#,
            m = models.path().display(),
            e = elsewhere.path().display()
        );
        let cfg: NodeConfig = toml::from_str(&text).unwrap();
        let got = cfg.origins();
        let of = |id: &str| got.iter().find(|(i, ..)| i == id).map(|(_, o, l)| (o.as_deref(), l.as_deref()));
        assert_eq!(of("up"), Some((Some("upload"), Some("MIT"))));
        assert_eq!(of("up+tune"), Some((Some("train"), Some("Apache-2.0"))));
        assert_eq!(of("pulled"), Some((Some("hub"), None)));
        assert_eq!(of("big"), Some((Some("hub"), None)));
        assert_eq!(of("own"), None, "a file elsewhere without a record says nothing");
        assert_eq!(of("wrapped"), None);
    }

    #[test]
    fn lora_adapters_on_a_local_model() {
        let cfg: NodeConfig = toml::from_str(
            r#"
            [model]
            models_dir = "/models"
            [[model.local]]
            id = "qwen3-8b"
            file = "qwen.gguf"
            revision = "base-r1"
            quant = "Q4_K_M"
            [[model.local.lora]]
            id = "qwen3-8b+support-v2"
            file = "adapters/support-v2.gguf"
            scale = 0.8
            revision = "run-42"
            [[model.local.lora]]
            id = "qwen3-8b+abs"
            file = "/srv/abs.gguf"
            [[model.local]]
            id = "big"
            backend = "vllm"
            file = "org/big"
            [[model.local.lora]]
            id = "big+tune"
            file = "org/big-tune"
            "#,
        )
        .unwrap();
        cfg.validate().unwrap();
        let l = cfg.launches();
        let ids: Vec<&str> = l[0].adapters.iter().map(|a| a.id.as_str()).collect();
        assert_eq!(ids, ["qwen3-8b+support-v2", "qwen3-8b+abs"]);
        assert_eq!(Path::new(&l[0].adapters[0].path), Path::new("/models/adapters/support-v2.gguf"));
        assert_eq!(l[0].adapters[0].scale, Some(0.8));
        assert_eq!(l[0].adapters[1].path, "/srv/abs.gguf");
        assert_eq!(l[0].served_ids(), ["qwen3-8b", "qwen3-8b+support-v2", "qwen3-8b+abs"]);
        // A Hub id stays a Hub id for vLLM.
        assert_eq!(l[1].adapters[0].path, "org/big-tune");
        // Adapters key the answer cache with their own revision, on the base's quant.
        assert!(cfg.revisions().contains(&("qwen3-8b+support-v2".into(), "run-42".into(), Some("Q4_K_M".into()))));
        assert!(!cfg.revisions().iter().any(|(id, ..)| id == "qwen3-8b+abs"));
        let back: NodeConfig = toml::from_str(&cfg.to_toml().unwrap()).unwrap();
        assert_eq!(back, cfg);

        let bad = |f: &dyn Fn(&mut NodeConfig)| {
            let mut c = cfg.clone();
            f(&mut c);
            c.validate().unwrap_err().to_string()
        };
        assert!(bad(&|c| c.model.local[0].lora[1].id = "big".into()).contains("already a model or adapter"));
        assert!(bad(&|c| c.model.local[1].lora[0].id = "qwen3-8b+abs".into()).contains("already a model or adapter"));
        assert!(bad(&|c| c.model.local[0].lora[0].scale = Some(f32::INFINITY)).contains("finite"));
        assert!(bad(&|c| c.model.local[0].backend = Some("mlx".into())).contains("llamacpp or vllm"));
        assert!(bad(&|c| c.model.local[0].role = Some("embedding".into())).contains("embedding"));
        assert!(
            bad(&|c| {
                c.model.local[0].lora.clear();
                c.model.local[0].backend = Some("mlx".into());
                c.model.local[0].role = Some("transcribe".into());
            })
            .contains("transcribe")
        );
        assert!(bad(&|c| c.model.local[0].wrap = Some("http://127.0.0.1:9000".into())).contains("wrapped"));
        assert!(bad(&|c| c.model.local[0].lora[0].file = String::new()).contains("file is required"));
    }

    #[test]
    fn lora_slots_and_the_base_revision_check() {
        let mut cfg: NodeConfig = toml::from_str(
            r#"
            [[model.local]]
            id = "qwen3-8b"
            file = "/m/qwen.gguf"
            revision = "base-r2"
            quant = "Q4_K_M"
            [[model.local.lora]]
            id = "qwen3-8b+old"
            file = "/m/old.gguf"
            base_revision = "base-r1"
            [[model.local.lora]]
            id = "qwen3-8b+q8"
            file = "/m/q8.gguf"
            base_quant = "Q8_0"
            [[model.local.lora]]
            id = "qwen3-8b+ok"
            file = "/m/ok.gguf"
            base_revision = "BASE-R2"
            base_quant = "q4_k_m"
            [[model.local.lora]]
            id = "qwen3-8b+unknown"
            file = "/m/unknown.gguf"
            [[model.local]]
            id = "big"
            backend = "vllm"
            file = "org/big"
            lora_slots = 4
            "#,
        )
        .unwrap();
        cfg.validate().unwrap();
        let mut l = cfg.launches();
        assert_eq!((l[0].lora_slots, l[1].lora_slots), (None, Some(4)));
        assert!(l[1].runtime_lora() && !l[0].runtime_lora());

        let base = &cfg.model.local[0];
        assert!(base.lora[0].base_mismatch(base).unwrap().contains("revision base-r1 of `qwen3-8b`, which is now at revision base-r2"));
        assert!(base.lora[1].base_mismatch(base).unwrap().contains("Q8_0 quantisation"));
        assert_eq!(base.lora[2].base_mismatch(base), None, "case and spaces don't count");
        assert_eq!(base.lora[3].base_mismatch(base), None, "nothing recorded, nothing checked");
        let dropped = crate::adapters::drop_wrong_base(&cfg, &mut l);
        assert_eq!(dropped.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(), ["qwen3-8b+old", "qwen3-8b+q8"]);
        assert_eq!(l[0].served_ids(), ["qwen3-8b", "qwen3-8b+ok", "qwen3-8b+unknown"]);
        let back: NodeConfig = toml::from_str(&cfg.to_toml().unwrap()).unwrap();
        assert_eq!(back, cfg);

        let bad = |f: &dyn Fn(&mut NodeConfig)| {
            let mut c = cfg.clone();
            f(&mut c);
            c.validate().unwrap_err().to_string()
        };
        assert!(bad(&|c| c.model.local[0].lora_slots = Some(2)).contains("lora_slots is for backend = \"vllm\""));
        assert!(bad(&|c| c.model.local[1].lora_slots = Some(0)).contains("at least 1"));
        assert!(bad(&|c| c.model.local[1].wrap = Some("http://127.0.0.1:9000".into())).contains("wrapped"));
        cfg.model.local[0].revision = None;
        assert_eq!(cfg.model.local[0].lora[0].base_mismatch(&cfg.model.local[0]), None, "a base without a revision can't be checked");
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

    #[test]
    fn vllm_image_and_inkling_mlx_are_validated() {
        let ok: NodeConfig = toml::from_str(
            r#"
            [node]
            name = "spark"
            [[model.local]]
            id = "inkling-2-mini"
            backend = "vllm"
            file = "thinkingmachines/Inkling-2-Mini"
            image = "vllm/vllm-openai:inkling-2-mini"
            args = ["--reasoning-parser", "inkling"]
            "#,
        )
        .unwrap();
        ok.validate().unwrap();
        assert_eq!(ok.model.local[0].image.as_deref(), Some(crate::vllm::INKLING_IMAGE));
        assert!(vllm_via_image(&ok.model));
        assert_eq!(vllm_image_from_models(&ok.model).as_deref(), Some(crate::vllm::INKLING_IMAGE));

        let mlx: NodeConfig = toml::from_str(
            r#"
            [node]
            name = "mac"
            [[model.local]]
            id = "inkling"
            backend = "mlx"
            file = "mlx-community/Inkling-2-Mini-4bit"
            "#,
        )
        .unwrap();
        let err = mlx.validate().unwrap_err().to_string();
        assert!(err.contains("mlx-lm") && err.contains("2086"), "{err}");

        let mixed: NodeConfig = toml::from_str(
            r#"
            [node]
            name = "n"
            [[model.local]]
            id = "a"
            backend = "vllm"
            file = "a"
            image = "vllm/one:tag"
            [[model.local]]
            id = "b"
            backend = "vllm"
            file = "b"
            image = "vllm/two:tag"
            "#,
        )
        .unwrap();
        let err = mixed.validate().unwrap_err().to_string();
        assert!(err.contains("same image"), "{err}");
    }
}
