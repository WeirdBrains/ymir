//! `ymirctl train run --node X` (training plan P3a, option C): a run on another of the owner's
//! model nodes, started from any linked device, over Ymir's own Weirdnet protocol
//! [`TRAIN_PROTOCOL`].
//!
//! - `ymirctl` asks its local `ymird` ([`crate::control::ControlRequest::Train`]); that `ymird`
//!   finds the node on the owner's roster ([`crate::models::resolve`]), reads the dataset and its
//!   manifest, and opens `/ymir/train/1` to it presenting this device's certificate
//!   ([`TrainHub`]).
//! - The node's `ymird` serves the protocol only with `[train] remote = true` (off by default),
//!   to the owner's linked, unrevoked devices (narrowed by `[train] callers`), and runs
//!   [`crate::train::run`] in process with the dataset it was sent ([`TrainService`]). The run
//!   reserves the node through that `ymird`, exactly as a local run does, and its updates stream
//!   back as [`TrainLine`]s.
//! - Closing the stream (Ctrl-C in `ymirctl`) cancels the run. node.toml is never edited from the
//!   network: the adapter lands in the node's models directory with its provenance, and
//!   `ymirctl models adapter serve --node` (models/1) serves it.
//!
//! One run per node at a time (the run's lock). A synced job record that survives the caller
//! going away, and resuming, are P3b.

use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};
use weirdnet_agent::remote::{Authorize, Caller, Channel, Dialer, RemoteError};
use weirdnet_agent::serve::{channel, stream_limits};
use weirdnet_net::node::{Message, Node};

use crate::models::{Directory, Target};
use crate::train::{self, Event, Hyper, RunRecord, RunSpec, Stage, Update};
use crate::{NodeConfig, Paths};

/// Ymir's training protocol on Weirdnet.
pub const TRAIN_PROTOCOL: &str = "/ymir/train/1";
/// The largest dataset (with its manifest) sent with a run. Weirdnet messages stop at 32 MiB;
/// a larger dataset is trained on the node itself.
pub const MAX_DATASET_BYTES: usize = 24 << 20;

/// What `ymirctl train run --node` asks its `ymird` for. Paths are on the asking device.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainOp {
    pub base: String,
    /// The dataset on this device, with its manifest next to it.
    pub dataset: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// A GGUF in the node's models directory to train from (a relative path there).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
    pub hyper: Hyper,
    #[serde(default)]
    pub allow_unknown_license: bool,
    #[serde(default)]
    pub keep_full: bool,
    #[serde(default)]
    pub force: bool,
}

/// The request on [`TRAIN_PROTOCOL`] (JSON): the run, with the dataset and its manifest inline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteRun {
    /// The caller's certificate, when the transport doesn't carry it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cert: Option<Vec<u8>>,
    pub base: String,
    /// The dataset's file name (no directories).
    pub dataset_name: String,
    pub dataset: String,
    pub manifest: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
    pub hyper: Hyper,
    #[serde(default)]
    pub allow_unknown_license: bool,
    #[serde(default)]
    pub keep_full: bool,
    #[serde(default)]
    pub force: bool,
}

/// One line of a remote run, from the node (over [`TRAIN_PROTOCOL`]) and from `ymird` to
/// `ymirctl` (over the control socket). Ends with `Done` or `Failed`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "line", rename_all = "snake_case")]
pub enum TrainLine {
    Started { node: String, run: String, adapter: String },
    Reserved { in_ymird: bool },
    Step { stage: Stage, event: Event },
    Done { record: Box<RunRecord> },
    Failed { message: String },
}

impl TrainLine {
    fn of(node: &str, u: Update) -> Self {
        match u {
            Update::Started { run, adapter, .. } => TrainLine::Started { node: node.to_string(), run, adapter },
            Update::Reserved { in_ymird } => TrainLine::Reserved { in_ymird },
            Update::Event(stage, event) => TrainLine::Step { stage, event },
            Update::Checkpoint { info, .. } => TrainLine::Step {
                stage: Stage::Train,
                event: Event::Log { message: format!("checkpoint {} kept (epoch {}, step {}, root {})", info.dir, info.epoch, info.step, info.root) },
            },
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The node's side

/// Runs training requests on this node (the model role, `[train] remote = true`).
pub struct TrainService {
    paths: Paths,
    cfg: NodeConfig,
    auth: Arc<dyn Authorize>,
}

impl TrainService {
    pub fn new(paths: Paths, cfg: NodeConfig, auth: Arc<dyn Authorize>) -> Self {
        Self { paths, cfg, auth }
    }

    /// Answers one stream: authorizes the caller, runs, and streams the run's lines. The caller
    /// closing its side cancels the run.
    pub async fn serve(&self, mut ch: Channel, caller: Caller) -> Result<(), RemoteError> {
        let Some(req) = ch.recv::<RemoteRun>().await? else { return Ok(()) };
        let caller = match (caller.cert_bytes.is_some(), req.cert.clone()) {
            (false, Some(cert)) => caller.with_cert(cert),
            _ => caller,
        };
        if let Err(e) = self.auth.authorize(&caller) {
            let message = format!("refused: only the owner's linked devices allowed by [train] callers may start runs here ({e})");
            return ch.send(&TrainLine::Failed { message }).await;
        }
        let Channel { mut tx, mut rx } = ch;
        let (lines, mut out) = unbounded_channel::<TrainLine>();
        let forward = tokio::spawn(async move {
            while let Some(line) = out.recv().await {
                let Ok(bytes) = serde_json::to_vec(&line) else { continue };
                if tx.send(bytes).await.is_err() {
                    break;
                }
            }
        });
        // The caller sends nothing more; its side closing (or the stream failing) cancels the run.
        let mut cancel = Box::pin(async move { while let Ok(Some(_)) = rx.recv().await {} });
        let node = self.cfg.name();
        let sink = lines.clone();
        let mut on = move |u: Update| {
            let _ = sink.send(TrainLine::of(&node, u));
        };
        let last = match self.run(req, &mut cancel, &mut on).await {
            Ok(record) => TrainLine::Done { record: Box::new(record) },
            Err(e) => TrainLine::Failed { message: format!("{e:#}") },
        };
        let _ = lines.send(last);
        drop(lines);
        drop(on);
        let _ = forward.await;
        Ok(())
    }

    async fn run(
        &self,
        req: RemoteRun,
        cancel: &mut (dyn std::future::Future<Output = ()> + Send + Unpin),
        on: &mut (dyn FnMut(Update) + Send),
    ) -> Result<RunRecord> {
        check_name(&req.dataset_name)?;
        if req.dataset.len() + req.manifest.len() > MAX_DATASET_BYTES {
            bail!("the dataset is over {} MiB; train on the node itself", MAX_DATASET_BYTES >> 20);
        }
        let from = match &req.from {
            Some(f) => Some(in_models_dir(&self.cfg.models_dir(), f)?),
            None => None,
        };
        // Held where runs keep their work, and removed after the run (the run's record keeps the
        // dataset's name, hash and manifest).
        let mut nonce = [0u8; 8];
        let _ = getrandom::fill(&mut nonce);
        let incoming = self.cfg.train.work_dir(&self.paths).join("incoming").join(hex::encode(nonce));
        std::fs::create_dir_all(&incoming).with_context(|| format!("creating {}", incoming.display()))?;
        let dataset = incoming.join(&req.dataset_name);
        let written = std::fs::write(&dataset, req.dataset.as_bytes())
            .and_then(|_| std::fs::write(crate::dataset::manifest_path(&dataset), req.manifest.as_bytes()));
        let result = match written {
            Ok(()) => {
                let spec = RunSpec {
                    base: req.base,
                    dataset: dataset.clone(),
                    id: req.id,
                    from,
                    hyper: req.hyper,
                    allow_unknown_license: req.allow_unknown_license,
                    keep_full: req.keep_full,
                    force: req.force,
                    ..Default::default()
                };
                train::run(&self.paths, &self.cfg, &spec, cancel, on).await
            }
            Err(e) => Err(anyhow::Error::from(e).context("writing the dataset")),
        };
        let _ = std::fs::remove_dir_all(&incoming);
        result
    }
}

/// A dataset's file name: one plain component.
fn check_name(name: &str) -> Result<()> {
    let mut parts = Path::new(name).components();
    match (parts.next(), parts.next()) {
        (Some(Component::Normal(_)), None) if !name.starts_with('.') && !name.contains('\\') => Ok(()),
        _ => bail!("`{name}` is not a plain file name"),
    }
}

/// `rel` inside `models_dir`: relative, without `..`.
fn in_models_dir(models_dir: &Path, rel: &str) -> Result<PathBuf> {
    let p = Path::new(rel);
    if rel.is_empty() || rel.contains('\\') || !p.components().all(|c| matches!(c, Component::Normal(_))) {
        bail!("--from with --node names a GGUF in the node's models directory (a relative path there, no `..`), not `{rel}`");
    }
    Ok(models_dir.join(p))
}

/// Accepts [`TRAIN_PROTOCOL`] streams for `service`. Dropping it stops serving.
pub struct TrainServer {
    task: tokio::task::JoinHandle<()>,
}

impl TrainServer {
    pub fn start<Q: Message, R: Message>(node: &Arc<Node<Q, R>>, service: Arc<TrainService>) -> Result<Self, RemoteError> {
        let mut limits = stream_limits();
        limits.max_streams = 4;
        let mut incoming = node.accept_streams(TRAIN_PROTOCOL, limits).map_err(|e| RemoteError::Failed { message: e.to_string() })?;
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

impl Drop for TrainServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

// ---------------------------------------------------------------------------------------------
// The asking side

/// The `ymird` side of `ymirctl train run --node`: finds the node on the roster and runs there.
pub struct TrainHub {
    directory: Arc<dyn Directory>,
    dialer: Arc<dyn Dialer>,
    cert: Option<Vec<u8>>,
}

impl TrainHub {
    pub fn new(directory: Arc<dyn Directory>, dialer: Arc<dyn Dialer>, cert: Option<Vec<u8>>) -> Self {
        Self { directory, dialer, cert }
    }

    /// Runs `op` on `node` (a roster label or peer id), sending each line but the last to `tx`.
    pub async fn run(&self, node: &str, op: TrainOp, tx: &UnboundedSender<TrainLine>) -> Result<RunRecord, String> {
        let target = crate::models::resolve(&self.directory.devices(), node)?;
        if target.local {
            return Err(format!("`{node}` is this device; run `ymirctl train run` without --node"));
        }
        let req = self.request(op).map_err(|e| format!("{e:#}"))?;
        let mut ch = self.dialer.open(&target.peer, TRAIN_PROTOCOL).await.map_err(|e| explain(&target, &e))?;
        ch.send(&req).await.map_err(|e| explain(&target, &e))?;
        loop {
            match ch.recv::<TrainLine>().await {
                Ok(Some(TrainLine::Done { record })) => return Ok(*record),
                Ok(Some(TrainLine::Failed { message })) => return Err(format!("{}: {message}", target.label)),
                Ok(Some(line)) => {
                    let _ = tx.send(line);
                }
                Ok(None) => return Err(format!("{} closed the stream before the run ended (is ymird still running there?)", target.label)),
                Err(e) => return Err(explain(&target, &e)),
            }
        }
    }

    /// `node`'s training status over [`TRAIN_STATUS_PROTOCOL`], with its roster label.
    pub async fn status(&self, node: &str) -> Result<(String, TrainStatus), String> {
        let target = crate::models::resolve(&self.directory.devices(), node)?;
        let mut ch = self.dialer.open(&target.peer, TRAIN_STATUS_PROTOCOL).await.map_err(|e| explain_status(&target, &e))?;
        ch.send(&StatusAsk { cert: self.cert.clone() }).await.map_err(|e| explain_status(&target, &e))?;
        match ch.recv::<StatusReply>().await {
            Ok(Some(StatusReply::Status { status })) => Ok((target.label, *status)),
            Ok(Some(StatusReply::Refused { message })) => Err(format!("{}: {message}", target.label)),
            Ok(None) => Err(format!("{} closed the stream without answering", target.label)),
            Err(e) => Err(explain_status(&target, &e)),
        }
    }

    fn request(&self, op: TrainOp) -> Result<RemoteRun> {
        let name = op.dataset.file_name().and_then(|n| n.to_str()).context("the dataset has no file name")?.to_string();
        let mpath = crate::dataset::manifest_path(&op.dataset);
        let size = |p: &Path| std::fs::metadata(p).map(|m| m.len() as usize).with_context(|| format!("reading {}", p.display()));
        if size(&op.dataset)? + size(&mpath).context("a dataset needs its manifest next to it")? > MAX_DATASET_BYTES {
            bail!(
                "{} is over {} MiB with its manifest; copy it to the node and run `ymirctl train run` there",
                op.dataset.display(),
                MAX_DATASET_BYTES >> 20
            );
        }
        let read = |p: &Path| std::fs::read_to_string(p).with_context(|| format!("reading {} (a dataset is UTF-8 JSONL)", p.display()));
        Ok(RemoteRun {
            cert: self.cert.clone(),
            base: op.base,
            dataset_name: name,
            dataset: read(&op.dataset)?,
            manifest: read(&mpath)?,
            id: op.id,
            from: op.from,
            hyper: op.hyper,
            allow_unknown_license: op.allow_unknown_license,
            keep_full: op.keep_full,
            force: op.force,
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Training status (`/ymir/train-status/1`): what a model node has trained and is training

/// Ymir's read-only training-status protocol on Weirdnet: a model node's runs and resumable jobs,
/// for the owner's linked devices (Bragi's training view, `ymirctl train status --node`).
pub const TRAIN_STATUS_PROTOCOL: &str = "/ymir/train-status/1";
/// Runs in one answer, newest first.
pub const STATUS_RUNS: usize = 50;

/// The request on [`TRAIN_STATUS_PROTOCOL`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StatusAsk {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cert: Option<Vec<u8>>,
}

/// One run, as the status protocol shows it (no local paths).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunSummary {
    pub id: String,
    pub status: train::RunStatus,
    pub base: String,
    /// The adapter's id and revision, when the run made one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_revision: Option<String>,
    pub dataset: String,
    pub dataset_license: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teacher: Option<String>,
    pub rows_used: usize,
    pub epochs: u32,
    pub steps: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_loss: Option<f64>,
    pub started_ms: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ended_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Checkpoints written; whether it carried on from one.
    #[serde(default)]
    pub checkpoints: usize,
    #[serde(default)]
    pub resumed: bool,
}

impl From<&RunRecord> for RunSummary {
    fn from(r: &RunRecord) -> Self {
        Self {
            id: r.id.clone(),
            status: r.status,
            base: r.base.clone(),
            adapter: r.adapter.as_ref().map(|a| a.id.clone()),
            adapter_revision: r.adapter.as_ref().and_then(|a| a.revision.clone()),
            dataset: r.dataset.name.clone(),
            dataset_license: r.dataset.license.clone(),
            teacher: r.dataset.teacher.clone(),
            rows_used: r.dataset.rows_used,
            epochs: r.hyper.epochs,
            steps: r.steps,
            final_loss: r.final_loss,
            started_ms: r.started_ms,
            ended_ms: r.ended_ms,
            error: r.error.clone(),
            checkpoints: r.checkpoints.len(),
            resumed: r.resumed_from.is_some(),
        }
    }
}

/// One resumable job (P3b), as the status protocol shows it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobSummary {
    pub id: String,
    pub state: crate::train_job::JobState,
    pub base: String,
    pub adapter: String,
    /// The device running it, while it holds the lease.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub holder: Option<String>,
    pub attempts: u32,
    /// The last checkpoint's epoch, when there is one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_epoch: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// A model node's training: whether it takes runs from other devices, its runs (newest first,
/// at most [`STATUS_RUNS`]) and its jobs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainStatus {
    pub node: String,
    /// `[train] remote`: it takes runs from the owner's devices.
    pub remote: bool,
    pub runs: Vec<RunSummary>,
    #[serde(default)]
    pub jobs: Vec<JobSummary>,
}

impl TrainStatus {
    /// This node's training status, from its work directory.
    pub fn read(paths: &Paths, cfg: &NodeConfig) -> Self {
        let runs = train::runs(paths, &cfg.train).iter().take(STATUS_RUNS).map(RunSummary::from).collect();
        let jobs = crate::train_job::FileJobStore::open_existing(paths, cfg)
            .and_then(|s| crate::train_job::JobStore::list(&s).ok())
            .unwrap_or_default()
            .into_iter()
            .map(|j| JobSummary {
                adapter: j.payload.adapter_id.clone(),
                base: j.payload.base.clone(),
                holder: j.lease.as_ref().map(|l| l.holder.clone()),
                attempts: j.attempts,
                checkpoint_epoch: j.checkpoint.as_ref().map(|c| c.seq),
                error: j.failure.clone(),
                state: j.state,
                id: j.id,
            })
            .collect();
        Self { node: cfg.name(), remote: cfg.train.remote, runs, jobs }
    }
}

/// The answer on [`TRAIN_STATUS_PROTOCOL`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "reply", rename_all = "snake_case")]
pub enum StatusReply {
    Status { status: Box<TrainStatus> },
    Refused { message: String },
}

/// Answers [`TRAIN_STATUS_PROTOCOL`] on a model node.
pub struct TrainStatusService {
    paths: Paths,
    cfg: NodeConfig,
    auth: Arc<dyn Authorize>,
}

impl TrainStatusService {
    pub fn new(paths: Paths, cfg: NodeConfig, auth: Arc<dyn Authorize>) -> Self {
        Self { paths, cfg, auth }
    }

    pub async fn serve(&self, mut ch: Channel, caller: Caller) -> Result<(), RemoteError> {
        let Some(req) = ch.recv::<StatusAsk>().await? else { return Ok(()) };
        let caller = match (caller.cert_bytes.is_some(), req.cert) {
            (false, Some(cert)) => caller.with_cert(cert),
            _ => caller,
        };
        if let Err(e) = self.auth.authorize(&caller) {
            return ch.send(&StatusReply::Refused { message: format!("refused: only the owner's linked devices see training here ({e})") }).await;
        }
        let (paths, cfg) = (self.paths.clone(), self.cfg.clone());
        let status =
            tokio::task::spawn_blocking(move || TrainStatus::read(&paths, &cfg)).await.map_err(|e| RemoteError::Failed { message: e.to_string() })?;
        ch.send(&StatusReply::Status { status: Box::new(status) }).await
    }
}

/// Accepts [`TRAIN_STATUS_PROTOCOL`] streams. Dropping it stops serving.
pub struct TrainStatusServer {
    task: tokio::task::JoinHandle<()>,
}

impl TrainStatusServer {
    pub fn start<Q: Message, R: Message>(node: &Arc<Node<Q, R>>, service: Arc<TrainStatusService>) -> Result<Self, RemoteError> {
        let mut limits = stream_limits();
        limits.max_streams = 8;
        let mut incoming = node.accept_streams(TRAIN_STATUS_PROTOCOL, limits).map_err(|e| RemoteError::Failed { message: e.to_string() })?;
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

impl Drop for TrainStatusServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// [`explain`], for the status protocol.
fn explain_status(t: &Target, e: &RemoteError) -> String {
    match e {
        RemoteError::Unsupported { .. } | RemoteError::Protocol { .. } => {
            format!("{} does not report training over Weirdnet (it needs the model role and a ymird that knows {TRAIN_STATUS_PROTOCOL})", t.label)
        }
        other => explain(t, other),
    }
}

/// What went wrong reaching `t`, in words a person can act on.
pub fn explain(t: &Target, e: &RemoteError) -> String {
    match e {
        RemoteError::Unsupported { .. } | RemoteError::Protocol { .. } => format!(
            "{} does not take training runs over Weirdnet (it needs the model role, `[train] remote = true` and a ymird that knows {TRAIN_PROTOCOL})",
            t.label
        ),
        RemoteError::Unreachable { message } => format!("{} is unreachable: {message}", t.label),
        RemoteError::Timeout | RemoteError::Closed => format!("{} did not answer (is ymird running there?)", t.label),
        other => format!("{}: {other}", t.label),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_from_paths_stay_inside() {
        assert!(check_name("data.jsonl").is_ok());
        for bad in ["", "../x", "a/b", ".hidden", "/etc/passwd", "a\\b"] {
            assert!(check_name(bad).is_err(), "{bad}");
        }
        let m = Path::new("/m");
        assert_eq!(in_models_dir(m, "f32/base.gguf").unwrap(), PathBuf::from("/m/f32/base.gguf"));
        for bad in ["", "../base.gguf", "/abs.gguf", "a/../../b", "a\\b"] {
            assert!(in_models_dir(m, bad).is_err(), "{bad}");
        }
    }

    fn request(name: &str) -> RemoteRun {
        RemoteRun {
            cert: None,
            base: "alpha".into(),
            dataset_name: name.into(),
            dataset: String::new(),
            manifest: String::new(),
            id: None,
            from: None,
            hyper: Hyper::default(),
            allow_unknown_license: false,
            keep_full: false,
            force: false,
        }
    }

    async fn answer(auth: Arc<dyn Authorize>, req: RemoteRun) -> TrainLine {
        let tmp = tempfile::tempdir().unwrap();
        let svc = TrainService::new(Paths::new(tmp.path()), NodeConfig::default(), auth);
        let (mut ours, theirs) = weirdnet_agent::remote::memory_pair();
        let serving = tokio::spawn(async move { svc.serve(theirs, Caller::new("12D3KooWcaller")).await });
        ours.send(&req).await.unwrap();
        let line = ours.recv::<TrainLine>().await.unwrap().unwrap();
        drop(ours);
        serving.await.unwrap().unwrap();
        line
    }

    #[tokio::test]
    async fn a_refused_caller_gets_no_run_and_a_bad_name_is_refused() {
        let refuse: Arc<dyn Authorize> = Arc::new(|_: &Caller| Err(RemoteError::refused("not-on-roster")));
        let TrainLine::Failed { message } = answer(refuse, request("data.jsonl")).await else { panic!("not refused") };
        assert!(message.contains("refused") && message.contains("not-on-roster"), "{message}");
        let allow: Arc<dyn Authorize> = Arc::new(|_: &Caller| Ok(()));
        let TrainLine::Failed { message } = answer(allow, request("../data.jsonl")).await else { panic!("not refused") };
        assert!(message.contains("not a plain file name"), "{message}");
    }

    #[test]
    fn lines_round_trip_as_json() {
        let l = TrainLine::Step { stage: Stage::Train, event: Event::Progress { step: 1, steps: 2, loss: Some(0.5) } };
        let s = serde_json::to_string(&l).unwrap();
        assert_eq!(serde_json::from_str::<TrainLine>(&s).unwrap(), l);
    }
}
