//! `ymirctl models upload FILE --node X` (docs/design/model-upload.md, slice U2): a model or LoRA
//! adapter on this device sent to another of the owner's model nodes.
//!
//! - `ymirctl` checks and hashes the file and builds its provenance (`crate::upload`), then asks
//!   its local `ymird` ([`crate::models::ModelsOp::Upload`]).
//! - That `ymird` sends the bytes over `weirdnet-backends`' models/1
//!   ([`weirdnet_backends::models::ModelsClient::upload`]): 48 KiB chunks, checked against their
//!   SHA-256s on the node, staged outside the store and placed only when complete. An interrupted
//!   upload (Ctrl-C, a dropped link) leaves its parts staged on the node for 7 days, and sending
//!   the same files again resumes from them. The node refuses what doesn't fit its rules (`exists`,
//!   `no-space` below the free space it keeps, `wrong-arch` for an adapter on another base, ...).
//! - Then it asks the node, over Ymir's own [`UPLOADS_PROTOCOL`], to record the entry's revision
//!   for the pool ([`NodeService::set_model_info`]) and, unless `--no-persist`, to save it in the
//!   node's node.toml so it survives a restart (m3talsmith, 2026-11-24). The node checks the
//!   entry it holds has that revision before it records or saves anything.
//!
//! Any of the owner's linked devices may upload (the same callers as models/1).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use weirdnet_agent::remote::{Authorize, Caller, Channel, Dialer, RemoteError};
use weirdnet_agent::serve::{channel, stream_limits};
use weirdnet_backends::models::{ModelsService, Pulled, StoredEntry, Upload};
use weirdnet_backends::node::NodeService;
use weirdnet_model::ModelLayout;
use weirdnet_net::node::{Message, Node};

use crate::NodeConfig;
use crate::config::LocalAdapter;
use crate::config_edit;

/// Ymir's protocol for what follows an upload on Weirdnet.
pub const UPLOADS_PROTOCOL: &str = "/ymir/uploads/1";

/// What `ymirctl models upload --node` asks its `ymird` for. Paths are on the asking device.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UploadOp {
    pub upload: Upload,
    /// The files as inspected on this device (`upload.files` are their names).
    pub local: Vec<PathBuf>,
    /// Save it in the node's node.toml (the default; `--no-persist` turns it off).
    pub persist: bool,
    /// A model's role in that entry (default: `main` when the node has no main model, else `chat`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// With several targets (U6): the node that takes the upload, the others copying from it
    /// (default: the target with the most room).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage: Option<String>,
}

/// The request on [`UPLOADS_PROTOCOL`] (JSON), after an upload of `id` placed `revision`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Register {
    /// The caller's certificate, when the transport doesn't carry it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cert: Option<Vec<u8>>,
    pub id: String,
    pub revision: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_for: Option<String>,
    /// For a vision projector: the model it goes with (U7).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projector_for: Option<String>,
    pub persist: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
}

/// What the node did with a [`Register`].
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Registered {
    pub id: String,
    pub revision: String,
    /// The entry added to node.toml, as TOML (`None`: not saved; see `note`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub saved: Option<String>,
    /// Why it was not saved, or what to know about the entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "reply", rename_all = "snake_case")]
pub enum RegisterReply {
    Registered(Registered),
    Failed { message: String },
}

/// One upload's result, as `ymird` reports it to `ymirctl`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UploadReport {
    pub id: String,
    pub revision: String,
    /// What the node placed (`None`: it already held these exact files under this id).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pulled: Option<Pulled>,
    /// Bytes the node already held from an earlier, interrupted upload of the same files.
    #[serde(default)]
    pub resumed: u64,
    pub total: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub registered: Option<Registered>,
    /// Why the revision was not recorded (or node.toml not changed) after the upload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub register_error: Option<String>,
    /// A pool upload (U6): the node this one copied the files from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub copied_from: Option<String>,
    /// A pool upload (U6): why this node got the files from this device instead of a copy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sent_instead: Option<String>,
}

// ---------------------------------------------------------------------------------------------
// The node's side

/// Records an uploaded entry's revision and saves it in node.toml.
pub struct UploadsService {
    config: PathBuf,
    models_dir: PathBuf,
    models: Arc<ModelsService>,
    node: Arc<NodeService>,
    accelerated: bool,
    auth: Option<Arc<dyn Authorize>>,
    edit: tokio::sync::Mutex<()>,
}

impl UploadsService {
    /// `config`: this node's node.toml; `models_dir`: the store models/1 writes into;
    /// `accelerated`: whether a model entry offloads to a GPU.
    pub fn new(config: PathBuf, models_dir: PathBuf, models: Arc<ModelsService>, node: Arc<NodeService>, accelerated: bool) -> Self {
        Self { config, models_dir, models, node, accelerated, auth: None, edit: tokio::sync::Mutex::new(()) }
    }

    /// Who may ask (the default: anyone who reaches the protocol; `ymird` sets the roster).
    pub fn with_auth(mut self, auth: Arc<dyn Authorize>) -> Self {
        self.auth = Some(auth);
        self
    }

    pub async fn serve(&self, mut ch: Channel, caller: Caller) -> Result<(), RemoteError> {
        let Some(req) = ch.recv::<Register>().await? else { return Ok(()) };
        let caller = match (caller.cert_bytes.is_some(), req.cert.clone()) {
            (false, Some(cert)) => caller.with_cert(cert),
            _ => caller,
        };
        if let Some(auth) = &self.auth
            && let Err(e) = auth.authorize(&caller)
        {
            let message = format!("refused: only the owner's linked devices may upload here ({e})");
            return ch.send(&RegisterReply::Failed { message }).await;
        }
        let reply = match self.register(&req).await {
            Ok(r) => RegisterReply::Registered(r),
            Err(e) => RegisterReply::Failed { message: format!("{e:#}") },
        };
        ch.send(&reply).await
    }

    /// Checks the stored entry is `req.revision`, records it for the pool, and saves it.
    pub async fn register(&self, req: &Register) -> Result<Registered> {
        let entries = self.models.list();
        let entry = entries.iter().find(|e| e.id() == req.id).with_context(|| format!("there is no `{}` on this node", req.id))?;
        let prov = entry.provenance().with_context(|| format!("`{}` here has no provenance (it was not uploaded)", req.id))?;
        if prov.revision != req.revision {
            bail!("`{}` here is revision {}, not {}", req.id, prov.revision, req.revision);
        }
        let quant = prov.base.as_ref().and_then(|b| b.quant.clone()).filter(|_| entry.adapter.is_some());
        self.node.set_model_info(&req.id, Some(req.revision.clone()), quant);
        let mut out = Registered { id: req.id.clone(), revision: req.revision.clone(), ..Default::default() };
        if !req.persist {
            return Ok(out);
        }
        let _held = self.edit.lock().await;
        match self.save(req, entry) {
            Ok(Saved::Entry(toml)) => out.saved = Some(toml),
            Ok(Saved::Already) => out.note = Some(format!("node.toml already has `{}` at this revision", req.id)),
            Ok(Saved::Not(why)) => out.note = Some(format!("node.toml was not changed: {why}")),
            Err(e) => out.note = Some(format!("node.toml was not changed: {e:#}")),
        }
        Ok(out)
    }

    fn save(&self, req: &Register, entry: &StoredEntry) -> Result<Saved> {
        let text = std::fs::read_to_string(&self.config).with_context(|| format!("reading {}", self.config.display()))?;
        let cfg: NodeConfig = toml::from_str(&text).with_context(|| format!("parsing {}", self.config.display()))?;
        let id = req.id.as_str();
        let same = |r: &Option<String>| r.as_deref() == Some(req.revision.as_str());
        if let Some(m) = cfg.model.local.iter().find(|m| m.id == id) {
            return Ok(if same(&m.revision) { Saved::Already } else { Saved::Not(format!("it already has a model `{id}`")) });
        }
        if let Some(a) = cfg.model.local.iter().flat_map(|m| m.lora.iter()).find(|a| a.id == id) {
            return Ok(if same(&a.revision) { Saved::Already } else { Saved::Not(format!("it already has an adapter `{id}`")) });
        }
        if let Some(m) = &entry.model {
            let store = weirdnet_model::ModelStore::new(&self.models_dir);
            let (backend, file) = match m.layout {
                // vLLM takes a path as given (a relative one reads as a Hub id), so it is absolute.
                ModelLayout::Safetensors => ("vllm", store.model_dir(m.layout, &m.id)?.to_string_lossy().into_owned()),
                ModelLayout::Mlx => ("mlx", self.relative(&store.model_dir(m.layout, &m.id)?)),
                _ => ("llamacpp", m.files.first().with_context(|| format!("`{id}` has no files"))?.clone()),
            };
            let role = req.role.clone().unwrap_or_else(|| if cfg.main_model().is_some() { "chat".into() } else { "main".into() });
            let mut local = config_edit::local_model(id, backend, &role, &file, None, self.accelerated);
            local.revision = Some(req.revision.clone());
            let stanza = config_edit::stanza(&local);
            config_edit::append_model(&self.config, id, &stanza, "# Uploaded over Weirdnet (`ymirctl models upload --node`).")?;
            return Ok(Saved::Entry(stanza));
        }
        if let Some(p) = &entry.projector {
            let model = req.projector_for.clone().or(p.for_model.clone()).or(entry.base.clone()).context("the projector's model is not known")?;
            let Some(b) = cfg.model.local.iter().find(|m| m.id == model) else {
                return Ok(Saved::Not(format!("it has no [[model.local]] `{model}` to give the projector to")));
            };
            let file = self.relative(&p.path);
            if b.mmproj.as_deref() == Some(file.as_str()) {
                return Ok(Saved::Already);
            }
            let new = config_edit::set_mmproj(&text, &model, &file)?;
            let check: NodeConfig = toml::from_str(&new).context("the new entry does not parse")?;
            check.validate()?;
            config_edit::save(&self.config, &text, &new)?;
            let was = b.mmproj.as_deref().map(|m| format!(" (it was `{m}`)")).unwrap_or_default();
            return Ok(Saved::Entry(format!("mmproj = \"{file}\" on `{model}`{was}")));
        }
        let Some(a) = &entry.adapter else {
            return Ok(Saved::Not("only models, LoRA adapters and projectors are saved there".into()));
        };
        let base = req.adapter_for.clone().or(entry.base.clone()).context("the adapter's base is not known")?;
        let Some(b) = cfg.model.local.iter().find(|m| m.id == base) else {
            return Ok(Saved::Not(format!("it has no [[model.local]] `{base}` to add the adapter to")));
        };
        let trained = entry.provenance().and_then(|p| p.base.as_ref()).and_then(|pb| pb.revision.clone());
        if let (Some(t), Some(here)) = (&trained, &b.revision)
            && t != here
        {
            return Ok(Saved::Not(format!("the adapter was trained on revision {t} of `{base}`, which is revision {here} here")));
        }
        let la = LocalAdapter {
            id: id.to_string(),
            file: self.relative(&a.path),
            scale: None,
            revision: Some(req.revision.clone()),
            base_revision: b.revision.clone().or(trained),
            base_quant: b.quant.clone(),
        };
        let new = config_edit::add_adapter(&text, &base, &la)?;
        let check: NodeConfig = toml::from_str(&new).context("the new entry does not parse")?;
        check.validate()?;
        config_edit::save(&self.config, &text, &new)?;
        let row = toml::to_string(&la).unwrap_or_default();
        Ok(Saved::Entry(format!("[[model.local.lora]] on `{base}`\n{row}")))
    }

    /// `path` relative to the models directory when it is inside it.
    fn relative(&self, path: &Path) -> String {
        let p = if path.is_absolute() { path.to_path_buf() } else { self.models_dir.join(path) };
        p.strip_prefix(&self.models_dir).unwrap_or(&p).to_string_lossy().into_owned()
    }
}

enum Saved {
    Entry(String),
    Already,
    Not(String),
}

pub struct UploadsServer {
    task: tokio::task::JoinHandle<()>,
}

impl UploadsServer {
    pub fn start<Q: Message, R: Message>(node: &Arc<Node<Q, R>>, service: Arc<UploadsService>) -> Result<Self, RemoteError> {
        let mut limits = stream_limits();
        limits.max_streams = 8;
        let mut incoming = node.accept_streams(UPLOADS_PROTOCOL, limits).map_err(|e| RemoteError::Failed { message: e.to_string() })?;
        let task = tokio::spawn(async move {
            while let Some(stream) = futures::StreamExt::next(&mut incoming).await {
                let caller = Caller::new(stream.peer().to_string());
                let svc = service.clone();
                tokio::spawn(async move {
                    let _ = svc.serve(channel(stream), caller).await;
                });
            }
        });
        Ok(Self { task })
    }
}

impl Drop for UploadsServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

// ---------------------------------------------------------------------------------------------
// The uploading side

/// Asks a node to record (and save) an upload.
pub struct UploadsClient {
    dialer: Arc<dyn Dialer>,
    cert: Option<Vec<u8>>,
}

impl UploadsClient {
    pub fn new(dialer: Arc<dyn Dialer>, cert: Option<Vec<u8>>) -> Self {
        Self { dialer, cert }
    }

    pub async fn register(&self, peer: &str, mut req: Register) -> Result<Registered, RegisterError> {
        req.cert = self.cert.clone();
        let mut ch = self.dialer.open(peer, UPLOADS_PROTOCOL).await.map_err(RegisterError::Remote)?;
        ch.send(&req).await.map_err(RegisterError::Remote)?;
        match ch.recv::<RegisterReply>().await {
            Ok(Some(RegisterReply::Registered(r))) => Ok(r),
            Ok(Some(RegisterReply::Failed { message })) => Err(RegisterError::Failed(message)),
            Ok(None) => Err(RegisterError::Remote(RemoteError::Closed)),
            Err(e) => Err(RegisterError::Remote(e)),
        }
    }
}

#[derive(Debug)]
pub enum RegisterError {
    Remote(RemoteError),
    Failed(String),
}

impl RegisterError {
    /// In words, for node `label`.
    pub fn explain(&self, label: &str) -> String {
        match self {
            RegisterError::Remote(RemoteError::Unsupported { .. } | RemoteError::Protocol { .. }) => format!(
                "{label} runs a ymird without {UPLOADS_PROTOCOL}: the upload is stored there, but its revision is not recorded and node.toml was not changed (upgrade ymird there, or add the entry by hand)"
            ),
            RegisterError::Remote(e) => format!("{label}: recording the upload failed ({e}); the files are stored there"),
            RegisterError::Failed(m) => format!("{label}: {m}"),
        }
    }
}

/// What a refused upload means, in words a person can act on.
pub fn explain_refusal(label: &str, e: &RemoteError) -> Option<String> {
    use weirdnet_backends::models::{refusal, refusal_reason};
    let RemoteError::Refused { message } = e else { return None };
    let reason = refusal_reason(e)?;
    let detail = message.split_once(": ").map_or(message.as_str(), |(_, d)| d);
    let hint = match reason {
        refusal::EXISTS => "pick another id with --name, or remove it there first (`ymirctl models remove ID --node`)",
        refusal::NO_SPACE => "free space there first; uploads keep [model] upload_keep_free free (the larger of 10% and 20 GiB by default)",
        refusal::TOO_BIG | refusal::QUOTA => "the node's upload limits refuse it",
        refusal::WRONG_ARCH | refusal::NO_BASE | refusal::NOT_ADAPTER => "the adapter has to be made for that base model on that node",
        refusal::WRONG_ENGINE => "that node's engine can't serve this format",
        refusal::CHECKSUM => "the bytes arrived changed; the parts were dropped, so send it again",
        refusal::NOT_ALLOWED => "only the owner's linked devices may upload (`ymirctl pool link`)",
        _ => "",
    };
    let sep = if hint.is_empty() { "" } else { "; " };
    Some(format!("{label} refused the upload ({reason}): {detail}{sep}{hint}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refusals_read_as_what_to_do() {
        let e = RemoteError::refused("no-space: 3 GiB free, 20 GiB kept free");
        let m = explain_refusal("gpu", &e).unwrap();
        assert!(m.starts_with("gpu refused the upload (no-space): 3 GiB free") && m.contains("upload_keep_free"), "{m}");
        assert!(explain_refusal("gpu", &RemoteError::refused("not-on-roster")).is_none());
        assert!(explain_refusal("gpu", &RemoteError::Busy).is_none());
    }

    #[test]
    fn the_free_space_kept_is_the_default_or_the_configured_size() {
        use weirdnet_backends::models::KeepFree;
        let mut cfg = NodeConfig::default();
        assert_eq!(cfg.model.keep_free(), KeepFree::default());
        assert_eq!(cfg.model.keep_free().of(1 << 40), (1u64 << 40) / 100 * 10);
        cfg.model.upload_keep_free = Some("50GiB".into());
        assert_eq!(cfg.model.keep_free().of(1 << 40), 50 << 30);
        cfg.model.upload_keep_free = Some("0".into());
        assert_eq!(cfg.model.keep_free().of(1 << 40), 0);
        cfg.model.upload_keep_free = Some("lots".into());
        assert!(cfg.validate().unwrap_err().to_string().contains("upload_keep_free"));
    }

    #[test]
    fn requests_round_trip_as_json() {
        let r =
            Register { cert: None, id: "m".into(), revision: "sha256:ab".into(), adapter_for: None, projector_for: None, persist: true, role: None };
        let s = serde_json::to_string(&r).unwrap();
        assert_eq!(serde_json::from_str::<Register>(&s).unwrap(), r);
        let reply = RegisterReply::Registered(Registered { id: "m".into(), revision: "sha256:ab".into(), saved: Some("x".into()), note: None });
        let s = serde_json::to_string(&reply).unwrap();
        assert_eq!(serde_json::from_str::<RegisterReply>(&s).unwrap(), reply);
    }
}
