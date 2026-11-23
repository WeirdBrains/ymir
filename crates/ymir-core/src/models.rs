//! Managing model files over Weirdnet (LF-30 M2/M3): the part of `ymirctl models --node` that runs
//! inside `ymird`.
//!
//! The model role serves `/weirdnet-backends/models/1` ([`crate::remote::serve_models`]) to the user's
//! linked devices; this module is the other side. `ymirctl` asks its own `ymird` over the control
//! socket ([`crate::control::ControlRequest::Models`]); `ymird` resolves `--node` (a roster label, a
//! peer id, `auto` or `all`), runs the request on each chosen node with a
//! [`weirdnet_backends::models::ModelsClient`] (or in process for this node's own model role) and
//! streams a pull's progress back as it arrives.
//!
//! The Hugging Face token never leaves the node that downloads. Nothing in these messages carries one.

use std::sync::Arc;

use futures::future::{BoxFuture, join_all};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::UnboundedSender;
use weirdnet_agent::remote::RemoteError;
use weirdnet_backends::models::{ModelsClient, ModelsReply, ModelsService, Pull, RepoFits};
use weirdnet_model::hf::RepoModel;
use weirdnet_model::{HfToken, ModelStore};

/// A device that may manage models: `label` is the roster label (`ymirctl pool list`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Target {
    pub peer: String,
    pub label: String,
    /// This `ymird`'s own model role, reached in process.
    pub local: bool,
}

/// What to ask a node. `Files` is how `search` shows a verdict and how `auto` compares nodes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum ModelsOp {
    Search {
        query: String,
        limit: u32,
    },
    Files {
        repo: String,
        context: Option<u32>,
        slots: Option<u32>,
    },
    Pull(Pull),
    List,
    Remove {
        id: String,
    },
    /// Serve a stored adapter on a base the node serves, live where the engine allows.
    ServeAdapter {
        id: String,
        base: String,
        scale: Option<f32>,
    },
    /// Stop serving an adapter, keeping its files.
    UnserveAdapter {
        id: String,
    },
    /// Change a served adapter's scale (llama.cpp), from the next request.
    AdapterScale {
        id: String,
        scale: Option<f32>,
    },
}

/// One line of the answer on the control socket: progress while a pull runs, then `Done` or `Failed`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "line", rename_all = "snake_case")]
pub enum ModelsLine {
    Progress { node: String, done: u64, total: u64 },
    Done(Box<ModelsDone>),
    Failed { message: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelsDone {
    pub nodes: Vec<NodeResult>,
    /// With `auto`: the node picked, and why.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chosen: Option<Chosen>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Chosen {
    pub node: Target,
    /// The model it would take and its size, from the node's own fit.
    pub model: String,
    pub size: u64,
    pub usable_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeResult {
    pub node: Target,
    pub outcome: Outcome,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "result", rename_all = "snake_case")]
pub enum Outcome {
    Ok { reply: Box<ModelsReply> },
    Failed { message: String },
}

/// Who is on the roster and which of them serve models/1.
pub trait Directory: Send + Sync {
    /// Every active device, this one included.
    fn devices(&self) -> Vec<Target>;
    /// The devices that announce a model node (and so serve models/1, when they run a current `ymird`).
    fn model_nodes(&self) -> BoxFuture<'_, Vec<Target>>;
}

/// The `ymird` side of `ymirctl models --node`.
pub struct ModelsHub {
    local: Option<(Target, Arc<ModelsService>)>,
    directory: Arc<dyn Directory>,
    client: Arc<ModelsClient>,
}

impl ModelsHub {
    pub fn new(local: Option<(Target, Arc<ModelsService>)>, directory: Arc<dyn Directory>, client: ModelsClient) -> Self {
        Self { local, directory, client: Arc::new(client) }
    }

    /// Runs `op` on the nodes `spec` names, sending a pull's progress to `tx` as it goes.
    pub async fn run(&self, spec: &str, op: ModelsOp, tx: &UnboundedSender<ModelsLine>) -> Result<ModelsDone, String> {
        let mut chosen = None;
        let targets = match spec {
            "all" => {
                let t = self.model_nodes().await;
                if t.is_empty() {
                    return Err("no model nodes found among your devices (is ymird running on them, with the model role on?)".into());
                }
                t
            }
            "auto" => {
                let ModelsOp::Pull(pull) = &op else {
                    return Err("`--node auto` picks where a pull goes; for search, list and remove name a node or use `--node all`".into());
                };
                let (node, c) = self.auto(pull).await?;
                chosen = Some(c);
                vec![node]
            }
            name => vec![self.resolve(name)?],
        };
        let runs = targets.iter().map(|t| async {
            let outcome = match self.call(t, op.clone(), tx).await {
                Ok(reply) => Outcome::Ok { reply: Box::new(reply) },
                Err(e) => Outcome::Failed { message: explain(t, &e) },
            };
            NodeResult { node: t.clone(), outcome }
        });
        Ok(ModelsDone { nodes: join_all(runs).await, chosen })
    }

    async fn model_nodes(&self) -> Vec<Target> {
        let mut out: Vec<Target> = self.local.iter().map(|(t, _)| t.clone()).collect();
        for t in self.directory.model_nodes().await {
            if !out.iter().any(|o| o.peer == t.peer) {
                out.push(t);
            }
        }
        out
    }

    /// A roster label (any case), a peer id, or a peer id's start (at least 8 characters).
    fn resolve(&self, name: &str) -> Result<Target, String> {
        let devices = self.directory.devices();
        let by = |f: &dyn Fn(&Target) -> bool| -> Vec<&Target> { devices.iter().filter(|t| f(t)).collect() };
        let mut hits = by(&|t| t.peer == name);
        if hits.is_empty() {
            hits = by(&|t| t.label.eq_ignore_ascii_case(name));
        }
        if hits.is_empty() && name.len() >= 8 {
            hits = by(&|t| t.peer.starts_with(name));
        }
        match hits.as_slice() {
            [one] => Ok((*one).clone()),
            [] => Err(format!(
                "no device named `{name}` on your roster (devices: {})",
                devices.iter().map(|t| t.label.as_str()).collect::<Vec<_>>().join(", ")
            )),
            many => Err(format!("`{name}` matches {} devices; use the peer id", many.len())),
        }
    }

    /// Asks every model node how the repo fits it, and takes the node that can hold the biggest model
    /// (the node's own `fit::rank` against its memory budget), the bigger budget on a tie.
    async fn auto(&self, pull: &Pull) -> Result<(Target, Chosen), String> {
        let nodes = self.model_nodes().await;
        if nodes.is_empty() {
            return Err("no model nodes found among your devices (is ymird running on them, with the model role on?)".into());
        }
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let asks = nodes.iter().map(|t| {
            let op = ModelsOp::Files { repo: pull.repo.clone(), context: pull.context, slots: pull.slots };
            let tx = tx.clone();
            async move { self.call(t, op, &tx).await }
        });
        let answers = join_all(asks).await;
        let mut fits = Vec::new();
        let mut why = Vec::new();
        for (i, (t, a)) in nodes.iter().zip(answers).enumerate() {
            match a {
                Ok(ModelsReply::Fits(f)) => fits.push((i, *f)),
                Ok(_) => why.push(format!("{}: unexpected answer", t.label)),
                Err(e) => why.push(format!("{}: {}", t.label, explain(t, &e))),
            }
        }
        match choose_node(pull.model.as_deref(), fits.iter().map(|(i, f)| (*i, f))) {
            Some(i) => {
                let f = &fits.iter().find(|(j, _)| *j == i).expect("chosen from fits").1;
                let fit = f.fits.iter().find(|x| x.fits && wants(pull.model.as_deref(), &x.model)).expect("chosen node has a fit");
                let c = Chosen {
                    node: nodes[i].clone(),
                    model: fit.model.id.clone(),
                    size: fit.model.size,
                    usable_bytes: f.budget.map_or(0, |b| b.usable_bytes),
                };
                Ok((nodes[i].clone(), c))
            }
            None if fits.is_empty() => Err(format!("no node answered: {}", why.join("; "))),
            None => Err(format!(
                "nothing in {} fits any node{}{}",
                pull.repo,
                pull.model.as_deref().map(|m| format!(" ({m})")).unwrap_or_default(),
                if why.is_empty() { String::new() } else { format!(" (not asked: {})", why.join("; ")) }
            )),
        }
    }

    async fn call(&self, t: &Target, op: ModelsOp, tx: &UnboundedSender<ModelsLine>) -> Result<ModelsReply, RemoteError> {
        let progress = |tx: &UnboundedSender<ModelsLine>, node: &str, done: u64, total: u64| {
            let _ = tx.send(ModelsLine::Progress { node: node.to_string(), done, total });
        };
        if t.local {
            let (_, svc) = self.local.as_ref().ok_or_else(|| RemoteError::Unsupported { message: "this node does not run the model role".into() })?;
            return Ok(match op {
                ModelsOp::Search { query, limit } => ModelsReply::Hits(svc.search(&query, limit).await?),
                ModelsOp::Files { repo, context, slots } => ModelsReply::Fits(Box::new(svc.files(&repo, context, slots).await?)),
                ModelsOp::Pull(p) => {
                    let (tx, node) = (tx.clone(), t.label.clone());
                    ModelsReply::Pulled(
                        svc.pull(&p, move |pr| {
                            progress(&tx, &node, pr.done, pr.total);
                            true
                        })
                        .await?,
                    )
                }
                ModelsOp::List => ModelsReply::List(svc.list()),
                ModelsOp::Remove { id } => {
                    let freed = svc.remove(&id).await?;
                    ModelsReply::Removed { id, freed }
                }
                ModelsOp::ServeAdapter { id, base, scale } => {
                    let change = svc.serve_adapter(&id, &base, scale).await?;
                    ModelsReply::AdapterServed { id, base, change }
                }
                ModelsOp::UnserveAdapter { id } => {
                    let base = svc.unserve_adapter(&id).await?;
                    ModelsReply::AdapterUnserved { id, base }
                }
                ModelsOp::AdapterScale { id, scale } => {
                    svc.set_adapter_scale(&id, scale)?;
                    ModelsReply::AdapterScaled { id, scale }
                }
            });
        }
        let peer = t.peer.as_str();
        Ok(match op {
            ModelsOp::Search { query, limit } => ModelsReply::Hits(self.client.search(peer, &query, limit).await?),
            ModelsOp::Files { repo, context, slots } => ModelsReply::Fits(Box::new(self.client.files(peer, &repo, context, slots).await?)),
            ModelsOp::Pull(p) => {
                let (tx, node) = (tx.clone(), t.label.clone());
                ModelsReply::Pulled(self.client.pull(peer, p, move |pr| progress(&tx, &node, pr.done, pr.total)).await?)
            }
            ModelsOp::List => ModelsReply::List(self.client.list(peer).await?),
            ModelsOp::Remove { id } => {
                let freed = self.client.remove(peer, &id).await?;
                ModelsReply::Removed { id, freed }
            }
            ModelsOp::ServeAdapter { id, base, scale } => {
                let change = self.client.serve_adapter(peer, &id, &base, scale).await?;
                ModelsReply::AdapterServed { id, base, change }
            }
            ModelsOp::UnserveAdapter { id } => {
                let base = self.client.unserve_adapter(peer, &id).await?;
                ModelsReply::AdapterUnserved { id, base }
            }
            ModelsOp::AdapterScale { id, scale } => {
                self.client.set_adapter_scale(peer, &id, scale).await?;
                ModelsReply::AdapterScaled { id, scale }
            }
        })
    }
}

fn wants(want: Option<&str>, m: &RepoModel) -> bool {
    want.is_none_or(|w| m.id == w || m.parts.iter().any(|p| p.name == w || p.name.rsplit('/').next() == Some(w)))
}

/// The node that can hold the biggest model of the repo (the one `want` names, when given): each
/// node's `RepoFits` ranks the repo's models against that node's own memory budget. A larger
/// budget wins a tie, then the earlier node. `None` when no node has a model that fits.
pub fn choose_node<'a>(want: Option<&str>, fits: impl IntoIterator<Item = (usize, &'a RepoFits)>) -> Option<usize> {
    fits.into_iter()
        .filter_map(|(i, f)| {
            let fit = f.fits.iter().find(|x| x.fits && wants(want, &x.model))?;
            Some((i, fit.model.size, f.budget.map_or(0, |b| b.usable_bytes)))
        })
        .max_by(|a, b| (a.1, a.2).cmp(&(b.1, b.2)).then(b.0.cmp(&a.0)))
        .map(|(i, ..)| i)
}

/// What went wrong on `t`, in words a person can act on.
pub fn explain(t: &Target, e: &RemoteError) -> String {
    match e {
        RemoteError::Refused { message } => {
            format!("{} refused: only devices linked to its owner may manage models ({message}). Link this device with `ymirctl pool link`", t.label)
        }
        RemoteError::Unsupported { .. } | RemoteError::Protocol { .. } => format!(
            "{} does not take model management (it runs an older ymird, or has the model role off, or `[model] remote_manage = false`): upgrade it, or run `ymirctl models` on that machine",
            t.label
        ),
        RemoteError::Unreachable { message } => format!("{} is unreachable: {message}", t.label),
        RemoteError::Busy => format!("{} is already pulling that model", t.label),
        RemoteError::Timeout | RemoteError::Closed => format!("{} did not answer (is ymird running there?)", t.label),
        RemoteError::BadRequest { message } | RemoteError::Failed { message } | RemoteError::Unavailable { message } => message.clone(),
        other => other.to_string(),
    }
}

/// This node's Hugging Face token, if it has one: `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, a file named
/// by `YMIR_HF_TOKEN_FILE`, `<state>/hf-token`, then Hugging Face's own token file. Read on the node
/// that downloads and nowhere else. The text is returned for callers that talk to the Hub
/// themselves; prefer [`store`], which keeps it inside `weirdnet-model`.
pub fn find_hf_token(state_dir: &std::path::Path) -> Option<(String, TokenSource)> {
    let ok = |s: &str| {
        let s = s.trim();
        (!s.is_empty() && !s.contains(char::is_whitespace)).then(|| s.to_string())
    };
    for name in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Some(t) = std::env::var(name).ok().and_then(|v| ok(&v)) {
            return Some((t, TokenSource::Env(name)));
        }
    }
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    if let Some(p) = std::env::var_os("YMIR_HF_TOKEN_FILE").filter(|p| !p.is_empty()) {
        files.push(p.into());
    }
    files.push(state_dir.join("hf-token"));
    let hf_home = std::env::var_os("HF_HOME")
        .filter(|p| !p.is_empty())
        .map(std::path::PathBuf::from)
        .or_else(|| dirs::cache_dir().map(|c| c.join("huggingface")));
    files.extend(hf_home.map(|h| h.join("token")));
    files.into_iter().find_map(|p| {
        let t = ok(&std::fs::read_to_string(&p).ok()?)?;
        Some((t, TokenSource::File(p)))
    })
}

/// Where a token came from, for messages (never the token itself).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenSource {
    Env(&'static str),
    File(std::path::PathBuf),
}

/// The model store for `dir`: the Hub from `WEIRDNET_HF_URL` (a mirror) or Hugging Face, and this
/// node's token, sent only to the public Hub.
pub fn store(dir: impl Into<std::path::PathBuf>, state_dir: &std::path::Path) -> ModelStore {
    let env = ModelStore::from_env();
    let store = ModelStore::new(dir).with_hub(env.hub());
    match find_hf_token(state_dir).and_then(|(t, _)| HfToken::new(t)) {
        Some(t) if env.hub() == weirdnet_model::store::HUB => store.with_token(Some(t)),
        _ => store,
    }
}

#[cfg(test)]
mod tests;
