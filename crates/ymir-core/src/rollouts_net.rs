//! `/ymir/rollouts/1`: the versioned rollouts (training plan P4b) of an inference node, read and
//! changed from the owner's other devices (Bragi's upload dialog, `ymirctl models rollout
//! --inference NODE`). The follow-up to model-upload slice U4; decisions of 2026-11-25 in
//! docs/design/model-upload.md.
//!
//! - One JSON [`Request`] and one [`Reply`] per stream. No change: the rollouts in force.
//! - A change follows the same rules as `ymirctl models rollout` on the node
//!   ([`crate::rollout_edit`]): above the canary share it needs the scored eval rows or `force`.
//! - It is made to the rollouts in force (the coordinator's), applied at once (new conversations
//!   follow), and by default also written to node.toml (`[[inference.rollout]]`, other content
//!   kept); `persist: false` changes only the live rollouts, until `ymird` restarts.
//! - Callers: the owner's linked devices, narrowed by `[inference] rollout_callers`.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use weirdnet_agent::remote::{Authorize, Caller, Channel, Dialer, RemoteError};
use weirdnet_agent::serve::{channel, stream_limits};
use weirdnet_net::node::{Message, Node};

use crate::NodeConfig;
use crate::coordinator::Coordinator;
use crate::rollout::Rollout;
use crate::rollout_edit::{self, Change, Gate};

/// Ymir's protocol for an inference node's rollouts on Weirdnet.
pub const ROLLOUTS_PROTOCOL: &str = "/ymir/rollouts/1";

fn yes() -> bool {
    true
}

/// A request on [`ROLLOUTS_PROTOCOL`] (JSON).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Request {
    /// The caller's certificate, when the transport doesn't carry it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cert: Option<Vec<u8>>,
    /// The change (none: only read the rollouts in force).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub change: Option<Change>,
    #[serde(default)]
    pub gate: Gate,
    /// Also write it to node.toml (the default).
    #[serde(default = "yes")]
    pub persist: bool,
}

impl Request {
    pub fn list() -> Self {
        Self { cert: None, change: None, gate: Gate::default(), persist: true }
    }

    pub fn change(change: Change) -> Self {
        Self { change: Some(change), ..Self::list() }
    }
}

/// What the node answers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "reply", rename_all = "snake_case")]
pub enum Reply {
    Rollouts(Rollouts),
    Failed { message: String },
}

/// The rollouts in force after the request.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Rollouts {
    pub rollouts: Vec<Rollout>,
    /// What changed, as `ymirctl models rollout` says it (none for a read).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub said: Option<String>,
    /// Whether node.toml has it too.
    #[serde(default)]
    pub saved: bool,
    /// Why node.toml was not changed, when it was asked to be.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

// ---------------------------------------------------------------------------------------------
// The inference node's side

pub struct RolloutsService {
    config: PathBuf,
    coordinator: Arc<Coordinator>,
    auth: Option<Arc<dyn Authorize>>,
    edit: tokio::sync::Mutex<()>,
}

impl RolloutsService {
    /// `config`: this node's node.toml.
    pub fn new(config: PathBuf, coordinator: Arc<Coordinator>) -> Self {
        Self { config, coordinator, auth: None, edit: tokio::sync::Mutex::new(()) }
    }

    /// Who may ask (the default: anyone who reaches the protocol; `ymird` sets the roster).
    pub fn with_auth(mut self, auth: Arc<dyn Authorize>) -> Self {
        self.auth = Some(auth);
        self
    }

    pub async fn serve(&self, mut ch: Channel, caller: Caller) -> Result<(), RemoteError> {
        let Some(req) = ch.recv::<Request>().await? else { return Ok(()) };
        let caller = match (caller.cert_bytes.is_some(), req.cert.clone()) {
            (false, Some(cert)) => caller.with_cert(cert),
            _ => caller,
        };
        if let Some(auth) = &self.auth
            && let Err(e) = auth.authorize(&caller)
        {
            let message = format!("refused: only the owner's linked devices allowed by [inference] rollout_callers may change rollouts here ({e})");
            return ch.send(&Reply::Failed { message }).await;
        }
        let reply = match self.handle(&req).await {
            Ok(r) => Reply::Rollouts(r),
            Err(e) => Reply::Failed { message: format!("{e:#}") },
        };
        ch.send(&reply).await
    }

    /// Applies `req` to the rollouts in force (and node.toml).
    pub async fn handle(&self, req: &Request) -> Result<Rollouts> {
        let Some(change) = &req.change else {
            return Ok(Rollouts { rollouts: self.coordinator.rollouts(), saved: false, ..Default::default() });
        };
        let _held = self.edit.lock().await;
        let mut rollouts = self.coordinator.rollouts();
        let said = rollout_edit::apply(&mut rollouts, change, &req.gate)?;
        crate::rollout::validate(&rollouts)?;
        let (mut saved, mut note) = (false, None);
        if req.persist {
            match self.save(&rollouts) {
                Ok(()) => saved = true,
                Err(e) => note = Some(format!("node.toml was not changed: {e:#}")),
            }
        }
        self.coordinator.set_rollouts(rollouts.clone()).map_err(anyhow::Error::msg)?;
        Ok(Rollouts { rollouts, said: Some(said), saved, note })
    }

    fn save(&self, rollouts: &[Rollout]) -> Result<()> {
        let text = std::fs::read_to_string(&self.config).with_context(|| format!("reading {}", self.config.display()))?;
        let new = rollout_edit::write_rollouts(&text, rollouts)?;
        toml::from_str::<NodeConfig>(&new).context("the edited node.toml does not parse")?.validate()?;
        crate::config_edit::save(&self.config, &text, &new)
    }
}

pub struct RolloutsServer {
    task: tokio::task::JoinHandle<()>,
}

impl RolloutsServer {
    pub fn start<Q: Message, R: Message>(node: &Arc<Node<Q, R>>, service: Arc<RolloutsService>) -> Result<Self, RemoteError> {
        let mut limits = stream_limits();
        limits.max_streams = 8;
        let mut incoming = node.accept_streams(ROLLOUTS_PROTOCOL, limits).map_err(|e| RemoteError::Failed { message: e.to_string() })?;
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

impl Drop for RolloutsServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

// ---------------------------------------------------------------------------------------------
// The asking side

/// Reads or changes another inference node's rollouts.
pub struct RolloutsClient {
    dialer: Arc<dyn Dialer>,
    cert: Option<Vec<u8>>,
}

impl RolloutsClient {
    pub fn new(dialer: Arc<dyn Dialer>, cert: Option<Vec<u8>>) -> Self {
        Self { dialer, cert }
    }

    pub async fn request(&self, peer: &str, mut req: Request) -> Result<Rollouts, RolloutsError> {
        req.cert = self.cert.clone();
        let mut ch = self.dialer.open(peer, ROLLOUTS_PROTOCOL).await.map_err(RolloutsError::Remote)?;
        ch.send(&req).await.map_err(RolloutsError::Remote)?;
        match ch.recv::<Reply>().await {
            Ok(Some(Reply::Rollouts(r))) => Ok(r),
            Ok(Some(Reply::Failed { message })) => Err(RolloutsError::Failed(message)),
            Ok(None) => Err(RolloutsError::Remote(RemoteError::Closed)),
            Err(e) => Err(RolloutsError::Remote(e)),
        }
    }
}

#[derive(Debug)]
pub enum RolloutsError {
    Remote(RemoteError),
    Failed(String),
}

impl RolloutsError {
    /// In words, for node `label`.
    pub fn explain(&self, label: &str) -> String {
        match self {
            RolloutsError::Remote(RemoteError::Unsupported { .. } | RemoteError::Protocol { .. }) => format!(
                "{label} runs a ymird without {ROLLOUTS_PROTOCOL} (older than 0.13.0, or not an inference node): change its rollouts there with `ymirctl models rollout`"
            ),
            RolloutsError::Remote(e) => format!("{label}: {e}"),
            RolloutsError::Failed(m) => format!("{label}: {m}"),
        }
    }
}

impl std::fmt::Display for RolloutsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RolloutsError::Remote(e) => write!(f, "{e}"),
            RolloutsError::Failed(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for RolloutsError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_and_replies_round_trip_as_json() {
        let r = Request::change(Change::Add { name: "support".into(), version: 3, model: "m-r3".into(), weight: 0 });
        let s = serde_json::to_string(&r).unwrap();
        assert!(s.contains(r#""op":"add""#), "{s}");
        assert_eq!(serde_json::from_str::<Request>(&s).unwrap(), r);
        // A bare request reads the rollouts and would persist.
        let bare: Request = serde_json::from_str("{}").unwrap();
        assert_eq!(bare, Request::list());
        let reply = Reply::Rollouts(Rollouts { rollouts: vec![], said: Some("x".into()), saved: true, note: None });
        let s = serde_json::to_string(&reply).unwrap();
        assert_eq!(serde_json::from_str::<Reply>(&s).unwrap(), reply);
    }
}
