//! Fine-tuning on this node (training plan P3a): one run, one node, no hand-off.
//!
//! Ymir doesn't train. It runs a separate trainer, pinned to one build, through the interface
//! below, and keeps everything around it: the dataset and its license, the node's GPU, the run
//! record and the adapter it produces. The trainer Michael chose (2026-11-24) is upstream
//! llama.cpp: a full fine-tune of a GGUF base with its pinned `llama-finetune`, then a
//! conversion step that extracts a LoRA adapter from the difference, which P4a serves on the
//! base. A wrapper program puts both behind the interface (`scripts/trainer.py`); Ymir checks
//! the build it reports against `[train] build` and refuses any other.
//!
//! The trainer interface (version [`INTERFACE`]):
//!
//! - `TRAINER version` prints one JSON line: `{"interface": 1, "engine": "llama.cpp", "build": "b1234"}`.
//! - `TRAINER train --base BASE.gguf --data DATA.jsonl --config CONFIG.json --out FULL.gguf`
//!   fine-tunes. `DATA.jsonl` holds one `{"messages": [{"role", "content"}, ...]}` per line, each
//!   with an assistant turn; `CONFIG.json` is a [`Hyper`].
//! - `TRAINER extract-lora --base BASE.gguf --tuned FULL.gguf --rank N --out ADAPTER.gguf`
//!   writes the LoRA adapter.
//! - Both steps print JSON lines on stdout: `{"event": "progress", "step": 3, "steps": 100,
//!   "loss": 1.25}`, `{"event": "log", "message": "..."}`, `{"event": "done"}` and
//!   `{"event": "error", "message": "..."}`. Other lines go to the run's log. Exit 0 and the
//!   output file mean success. SIGTERM cancels; the trainer gets [`CANCEL_GRACE`] to stop.
//!
//! The node's GPU is held for the run: a lock file (`state/train/lock.json`) that `ymird`
//! honours at start, and, while `ymird` runs, a `weirdnet-backends` reservation through its
//! control socket, so its models unload and nothing loads until the run ends.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::control::{self, ControlRequest};
use crate::{NodeConfig, Paths};

/// The trainer interface version Ymir speaks.
pub const INTERFACE: u32 = 1;
/// The run record's format.
pub const RUN_FORMAT: &str = "ymir-train-run/1";
/// How long a cancelled trainer gets after SIGTERM before it is killed.
pub const CANCEL_GRACE: Duration = Duration::from_secs(10);
/// The holder name for a reservation taken by hand (`ymirctl train reserve`).
pub const OPERATOR: &str = "operator";

/// `[train]` in node.toml.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TrainSection {
    /// The trainer: a program that speaks the trainer interface (see this module). For
    /// llama.cpp, a wrapper around the pinned `llama-finetune` and the LoRA extraction step.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trainer: Option<PathBuf>,
    /// The trainer build this node runs (`build` in `TRAINER version`); a run refuses any other.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build: Option<String>,
    /// Where runs keep their work, the full fine-tuned model too (as large as the base).
    /// Default `state/train`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub work_dir: Option<PathBuf>,
    /// Extra environment for the trainer (for example `CUDA_VISIBLE_DEVICES`).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
}

impl TrainSection {
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// Where runs keep their work.
    pub fn work_dir(&self, paths: &Paths) -> PathBuf {
        self.work_dir.clone().unwrap_or_else(|| paths.state().join("train"))
    }
}

/// Hyperparameters, passed to the trainer as `CONFIG.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hyper {
    pub epochs: u32,
    pub learning_rate: f64,
    /// Tokens per training sequence.
    pub context: u32,
    pub batch: u32,
    pub seed: u64,
    /// The LoRA rank the adapter is extracted at.
    pub rank: u32,
}

impl Default for Hyper {
    fn default() -> Self {
        Self { epochs: 1, learning_rate: 1e-5, context: 2048, batch: 4, seed: 0, rank: 16 }
    }
}

// ---------------------------------------------------------------------------------------------
// The lock

/// `state/train/lock.json`: who holds this node's GPU. `ymird` reserves its supervisor for the
/// holder at start while the lock is live.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Lock {
    pub holder: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run: Option<String>,
    /// The process that holds it (a run's `ymirctl`); the lock is stale once it is gone. `None`:
    /// held until released by hand.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pid: Option<u32>,
    pub since_ms: i64,
}

pub fn lock_path(paths: &Paths) -> PathBuf {
    paths.state().join("train").join("lock.json")
}

/// The live lock, if any (a lock whose process is gone is stale and ignored).
pub fn held(paths: &Paths) -> Option<Lock> {
    let text = std::fs::read_to_string(lock_path(paths)).ok()?;
    let lock: Lock = serde_json::from_str(&text).ok()?;
    match lock.pid {
        Some(pid) if !alive(pid) => None,
        _ => Some(lock),
    }
}

#[cfg(unix)]
fn alive(pid: u32) -> bool {
    // SAFETY: signal 0 only checks that the process exists.
    let r = unsafe { libc::kill(pid as libc::pid_t, 0) };
    r == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

/// Not checked off unix: a lock there holds until `ymirctl train release`
/// (or the run that took it) frees it.
#[cfg(not(unix))]
fn alive(_pid: u32) -> bool {
    true
}

/// Takes the lock for `holder`, unless someone else holds a live one.
pub fn take_lock(paths: &Paths, holder: &str, run: Option<&str>, pid: Option<u32>) -> Result<Lock> {
    if let Some(l) = held(paths)
        && l.holder != holder
    {
        bail!(
            "this node's GPU is held by {}{} (`ymirctl train release` frees a hold taken by hand)",
            l.holder,
            l.run.map(|r| format!(", run {r}")).unwrap_or_default()
        );
    }
    let lock = Lock { holder: holder.into(), run: run.map(str::to_string), pid, since_ms: now_ms() };
    let p = lock_path(paths);
    std::fs::create_dir_all(p.parent().expect("has a parent"))?;
    write_private(&p, &serde_json::to_vec_pretty(&lock)?)?;
    Ok(lock)
}

/// Removes the lock if `holder` holds it (or it is stale). Returns whether one was removed.
pub fn drop_lock(paths: &Paths, holder: &str) -> Result<bool> {
    let p = lock_path(paths);
    let Ok(text) = std::fs::read_to_string(&p) else { return Ok(false) };
    let lock: Option<Lock> = serde_json::from_str(&text).ok();
    let stale = lock.as_ref().is_none_or(|l| l.pid.is_some_and(|pid| !alive(pid)));
    if !stale && lock.as_ref().is_some_and(|l| l.holder != holder) {
        bail!("the hold on this node's GPU is {}'s, not {holder}'s", lock.map(|l| l.holder).unwrap_or_default());
    }
    std::fs::remove_file(&p)?;
    Ok(true)
}

/// Reserves this node's model role through its running `ymird`. `Ok(false)`: no `ymird` runs
/// here (the lock alone keeps one from loading models when it starts).
pub async fn reserve_running(paths: &Paths, holder: &str, force: bool) -> Result<bool> {
    if control::request(&paths.socket(), &ControlRequest::Status).await.is_err() {
        return Ok(false);
    }
    let r = control::request(&paths.socket(), &ControlRequest::Reserve { node: None, holder: holder.into(), force }).await?;
    if !r.ok {
        bail!("{}", r.message.unwrap_or_default());
    }
    Ok(true)
}

/// Releases `holder`'s reservation in the running `ymird`, if one runs.
pub async fn release_running(paths: &Paths, holder: &str) -> Result<bool> {
    if control::request(&paths.socket(), &ControlRequest::Status).await.is_err() {
        return Ok(false);
    }
    let r = control::request(&paths.socket(), &ControlRequest::Release { node: None, holder: holder.into() }).await?;
    if !r.ok {
        bail!("{}", r.message.unwrap_or_default());
    }
    Ok(true)
}

// ---------------------------------------------------------------------------------------------
// The dataset

/// What a run trained on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetRecord {
    pub name: String,
    pub sha256: String,
    /// The manifest's format: `ymir-dataset/1` (generated, P1) or `ymir-conversations/1`
    /// (consented conversations, P2).
    pub format: String,
    /// The license of the rows: the teacher's for a generated dataset.
    pub license: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teacher: Option<String>,
    pub rows_in: usize,
    /// Rows given to the trainer (usable and with an answer).
    pub rows_used: usize,
}

#[derive(Serialize)]
struct Turn {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct TrainRow {
    messages: Vec<Turn>,
}

/// Reads `dataset` and its manifest (`DATASET.manifest.json`) and writes the trainer's
/// `DATA.jsonl` to `out`. Only datasets with a manifest are taken, so the run can say what it
/// trained on and under which license.
pub fn prepare_dataset(dataset: &Path, allow_unknown_license: bool, out: &Path) -> Result<DatasetRecord> {
    let mpath = crate::dataset::manifest_path(dataset);
    let mtext = std::fs::read_to_string(&mpath).with_context(|| {
        format!(
            "reading {} (a dataset needs the manifest `ymirctl batch --jsonl` or `ymirctl context export --training` writes next to it, so the run can record its source and license)",
            mpath.display()
        )
    })?;
    let bytes = std::fs::read(dataset).with_context(|| format!("reading {}", dataset.display()))?;
    let sha256 = hex::encode(Sha256::digest(&bytes));
    let text = String::from_utf8(bytes).context("the dataset is not UTF-8")?;
    let format = serde_json::from_str::<serde_json::Value>(&mtext)
        .with_context(|| format!("parsing {}", mpath.display()))?
        .get("format")
        .and_then(|f| f.as_str())
        .unwrap_or_default()
        .to_string();
    let name = dataset.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let mut rows = Vec::new();
    let (license, teacher, rows_in) = if format == crate::dataset::MANIFEST_FORMAT {
        let m: crate::dataset::Manifest = serde_json::from_str(&mtext).with_context(|| format!("parsing {}", mpath.display()))?;
        crate::dataset::check_license(&m.teacher.license, allow_unknown_license)?;
        let all = crate::dataset::read_output(&text);
        for r in all.iter().filter(|r| r.usable && !r.completion.trim().is_empty()) {
            let mut messages: Vec<Turn> =
                r.messages.iter().map(|m| Turn { role: role_name(&m.role), content: m.content.text().into_owned() }).collect();
            messages.push(Turn { role: "assistant".into(), content: r.completion.clone() });
            rows.push(TrainRow { messages });
        }
        (m.teacher.license.clone(), Some(m.teacher.model.clone()), all.len())
    } else if format == crate::convexport::FORMAT {
        let m: crate::convexport::Manifest = serde_json::from_str(&mtext).with_context(|| format!("parsing {}", mpath.display()))?;
        if m.output_sha256 != sha256 {
            bail!("{} changed since its manifest was written; export it again", dataset.display());
        }
        let mut n = 0;
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            n += 1;
            let r: crate::convexport::Row = serde_json::from_str(line).with_context(|| format!("row {n} of {}", dataset.display()))?;
            if r.messages.iter().any(|t| t.role == "assistant") {
                rows.push(TrainRow { messages: r.messages.into_iter().map(|t| Turn { role: t.role, content: t.content }).collect() });
            }
        }
        ("the user's own conversations, marked for training".to_string(), None, n)
    } else {
        bail!("{}: format `{format}` is not a Ymir dataset manifest", mpath.display());
    };
    if rows.is_empty() {
        bail!("{} has no usable row with an answer to train on", dataset.display());
    }
    let mut body = Vec::new();
    for r in &rows {
        serde_json::to_writer(&mut body, r)?;
        body.push(b'\n');
    }
    write_private(out, &body)?;
    Ok(DatasetRecord { name, sha256, format, license, teacher, rows_in, rows_used: rows.len() })
}

fn role_name(role: &weirdnet_agent::Role) -> String {
    serde_json::to_value(role).ok().and_then(|v| v.as_str().map(str::to_string)).unwrap_or_else(|| "user".into())
}

// ---------------------------------------------------------------------------------------------
// The trainer

/// What `TRAINER version` reports.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainerVersion {
    pub interface: u32,
    pub engine: String,
    pub build: String,
}

/// Runs `TRAINER version`.
pub async fn trainer_version(program: &Path, env: &BTreeMap<String, String>) -> Result<TrainerVersion> {
    let out = tokio::process::Command::new(program)
        .arg("version")
        .envs(env)
        .stdin(Stdio::null())
        .output()
        .await
        .with_context(|| format!("starting the trainer {}", program.display()))?;
    if !out.status.success() {
        bail!("`{} version` failed ({}): {}", program.display(), out.status, String::from_utf8_lossy(&out.stderr).trim());
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.lines().rev().find(|l| l.trim_start().starts_with('{')).unwrap_or_default();
    serde_json::from_str(line)
        .with_context(|| format!("`{} version` printed no version line (it should speak Ymir's trainer interface {INTERFACE})", program.display()))
}

/// Checks the trainer against `[train]`: the interface Ymir speaks and the pinned build.
pub async fn check_trainer(train: &TrainSection) -> Result<(PathBuf, TrainerVersion)> {
    let Some(program) = train.trainer.clone() else {
        bail!("no trainer is set: add `trainer = \"/path/to/trainer\"` and the build it runs (`build = \"...\"`) under [train] in node.toml");
    };
    let Some(pin) = train.build.as_deref() else {
        bail!("[train] has no `build`: pin the trainer build this node runs (one llama.cpp build), and Ymir refuses any other");
    };
    let v = trainer_version(&program, &train.env).await?;
    if v.interface != INTERFACE {
        bail!("the trainer speaks interface {}, and this Ymir speaks {INTERFACE}", v.interface);
    }
    if v.build.trim() != pin.trim() {
        bail!(
            "the trainer reports build {} ({}), and [train] build pins {pin}; install the pinned build, or change the pin after checking the new one",
            v.build,
            v.engine
        );
    }
    Ok((program, v))
}

/// One event from a trainer step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    Progress {
        step: u64,
        steps: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        loss: Option<f64>,
    },
    Log {
        message: String,
    },
    Done,
    Error {
        message: String,
    },
}

/// The two steps of a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Stage {
    Train,
    ExtractLora,
}

impl std::fmt::Display for Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Stage::Train => "train",
            Stage::ExtractLora => "extract-lora",
        })
    }
}

/// How a step ended, when it didn't succeed.
#[derive(Debug)]
enum StepEnd {
    Cancelled,
    Failed(String),
}

/// The last progress a step reported.
#[derive(Debug, Clone, Copy, Default)]
struct Tally {
    step: u64,
    steps: u64,
    loss: Option<f64>,
}

async fn step(
    program: &Path,
    stage: Stage,
    args: &[String],
    env: &BTreeMap<String, String>,
    log: &Path,
    cancel: &mut (dyn std::future::Future<Output = ()> + Send + Unpin),
    on: &mut (dyn FnMut(Stage, &Event) + Send),
) -> std::result::Result<Tally, StepEnd> {
    let logf =
        std::fs::OpenOptions::new().create(true).append(true).open(log).map_err(|e| StepEnd::Failed(format!("opening {}: {e}", log.display())))?;
    let errf = logf.try_clone().map_err(|e| StepEnd::Failed(e.to_string()))?;
    let mut cmd = tokio::process::Command::new(program);
    cmd.arg(stage.to_string()).args(args).envs(env).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::from(errf)).kill_on_drop(true);
    let mut child = cmd.spawn().map_err(|e| StepEnd::Failed(format!("starting {}: {e}", program.display())))?;
    let stdout = child.stdout.take().expect("piped");
    let mut lines = BufReader::new(stdout).lines();
    let mut logw = std::io::LineWriter::new(logf);
    let mut tally = Tally::default();
    let mut error = None;
    let mut cancelled = false;
    loop {
        tokio::select! {
            line = lines.next_line() => match line {
                Ok(Some(line)) => {
                    let _ = writeln!(logw, "{line}");
                    let Ok(ev) = serde_json::from_str::<Event>(&line) else { continue };
                    match &ev {
                        Event::Progress { step, steps, loss } => tally = Tally { step: *step, steps: *steps, loss: loss.or(tally.loss) },
                        Event::Error { message } => error = Some(message.clone()),
                        _ => {}
                    }
                    on(stage, &ev);
                }
                Ok(None) | Err(_) => break,
            },
            _ = &mut *cancel, if !cancelled => {
                cancelled = true;
                terminate(&mut child).await;
                break;
            }
        }
    }
    let status = child.wait().await.map_err(|e| StepEnd::Failed(e.to_string()))?;
    if cancelled {
        return Err(StepEnd::Cancelled);
    }
    if !status.success() {
        return Err(StepEnd::Failed(error.unwrap_or_else(|| format!("the trainer's {stage} step exited with {status}; see the run's trainer.log"))));
    }
    if let Some(e) = error {
        return Err(StepEnd::Failed(e));
    }
    Ok(tally)
}

/// SIGTERM, then a kill after [`CANCEL_GRACE`].
async fn terminate(child: &mut tokio::process::Child) {
    #[cfg(unix)]
    if let Some(pid) = child.id() {
        // SAFETY: sends SIGTERM to the trainer we started.
        unsafe {
            libc::kill(pid as libc::pid_t, libc::SIGTERM);
        }
        if tokio::time::timeout(CANCEL_GRACE, child.wait()).await.is_ok() {
            return;
        }
    }
    let _ = child.kill().await;
}

// ---------------------------------------------------------------------------------------------
// A run

/// What to train.
#[derive(Debug, Clone, Default)]
pub struct RunSpec {
    /// The base: a llama.cpp `[[model.local]]` on this node.
    pub base: String,
    /// A JSONL dataset with its manifest next to it.
    pub dataset: PathBuf,
    /// The adapter's pool id (default `BASE+ft-RUN`).
    pub id: Option<String>,
    /// Train from this GGUF instead of the served file (an F32 copy of the same model when the
    /// served file is quantised; llama.cpp's trainer trains F32 models only). The adapter then
    /// records no `base_quant`.
    pub from: Option<PathBuf>,
    pub hyper: Hyper,
    pub allow_unknown_license: bool,
    /// Keep the full fine-tuned model (as large as the base) in the run's directory.
    pub keep_full: bool,
    /// Reserve the node even when it is the only one serving a loaded model.
    pub force: bool,
}

/// The adapter a run produced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdapterRecord {
    pub id: String,
    /// Its file, relative to `models_dir`.
    pub file: String,
    pub sha256: String,
    pub bytes: u64,
    /// The base's `revision` it was trained on, when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_revision: Option<String>,
    /// The base's `quant`, when it was trained on the served file itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_quant: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Running,
    Done,
    Failed,
    Cancelled,
}

/// `run.json` in the run's directory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunRecord {
    pub format: String,
    pub id: String,
    pub node: String,
    pub status: RunStatus,
    pub base: String,
    /// The file trained from (its name, not its path).
    pub trained_from: String,
    pub dataset: DatasetRecord,
    pub hyper: Hyper,
    pub trainer: TrainerVersion,
    pub steps: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_loss: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter: Option<AdapterRecord>,
    /// Whether `ymird` was running and reserved for the run (else the lock alone held the GPU).
    pub reserved_in_ymird: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub started_ms: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ended_ms: Option<i64>,
}

/// One update while a run goes.
#[derive(Debug, Clone, PartialEq)]
pub enum Update {
    /// The run's directory and the adapter id, before anything starts.
    Started {
        run: String,
        dir: PathBuf,
        adapter: String,
    },
    /// The node is held (`in_ymird`: through the running `ymird`).
    Reserved {
        in_ymird: bool,
    },
    Event(Stage, Event),
}

/// Runs the trainer on this node and leaves the adapter in `models_dir/adapters/`. `cancel`
/// resolving stops the run. The GPU is held for the whole run and released however it ends.
pub async fn run(
    paths: &Paths,
    cfg: &NodeConfig,
    spec: &RunSpec,
    cancel: &mut (dyn std::future::Future<Output = ()> + Send + Unpin),
    on: &mut (dyn FnMut(Update) + Send),
) -> Result<RunRecord> {
    let (program, version) = check_trainer(&cfg.train).await?;
    let base =
        cfg.model.local.iter().find(|m| m.id == spec.base).with_context(|| format!("no [[model.local]] with id `{}` on this node", spec.base))?;
    if base.backend.as_deref().unwrap_or("llamacpp") != "llamacpp" || base.wrap.is_some() {
        bail!("`{}` is not a llama.cpp model this node runs; the llama.cpp trainer takes a GGUF base", base.id);
    }
    if base.is_embedding() {
        bail!("`{}` is an embedding model", base.id);
    }
    let launch = cfg.launches().into_iter().find(|l| l.id == base.id).expect("a launch per [[model.local]]");
    let served = PathBuf::from(&launch.source);
    let from = spec.from.clone().unwrap_or_else(|| served.clone());
    if !from.is_file() {
        bail!("{} is not a file", from.display());
    }
    if spec.hyper.epochs == 0 || spec.hyper.rank == 0 || spec.hyper.batch == 0 || spec.hyper.context == 0 {
        bail!("epochs, rank, batch and context must be above 0");
    }

    let run_id = new_run_id();
    let adapter_id = spec.id.clone().unwrap_or_else(|| format!("{}+ft-{run_id}", base.id));
    let taken = cfg.model.local.iter().any(|m| m.id == adapter_id || m.lora.iter().any(|a| a.id == adapter_id));
    if taken {
        bail!("`{adapter_id}` is already a model or adapter on this node; pick another --id");
    }
    let models_dir = cfg.models_dir();
    let rel = format!("adapters/{}.gguf", file_safe(&adapter_id));
    let dest = models_dir.join(&rel);
    if dest.exists() {
        bail!("{} exists already; pick another --id", dest.display());
    }
    let dir = cfg.train.work_dir(paths).join(&run_id);
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    on(Update::Started { run: run_id.clone(), dir: dir.clone(), adapter: adapter_id.clone() });
    let data = dir.join("data.jsonl");
    let dataset = prepare_dataset(&spec.dataset, spec.allow_unknown_license, &data)?;
    let config = dir.join("config.json");
    write_private(&config, &serde_json::to_vec_pretty(&spec.hyper)?)?;

    let holder = format!("ymir-train {run_id}");
    take_lock(paths, &holder, Some(&run_id), Some(std::process::id()))?;
    let in_ymird = match reserve_running(paths, &holder, spec.force).await {
        Ok(b) => b,
        Err(e) => {
            let _ = drop_lock(paths, &holder);
            return Err(e.context("reserving this node for the run"));
        }
    };
    on(Update::Reserved { in_ymird });

    let mut record = RunRecord {
        format: RUN_FORMAT.into(),
        id: run_id.clone(),
        node: cfg.name(),
        status: RunStatus::Running,
        base: base.id.clone(),
        trained_from: from.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
        dataset,
        hyper: spec.hyper.clone(),
        trainer: version,
        steps: 0,
        final_loss: None,
        adapter: None,
        reserved_in_ymird: in_ymird,
        error: None,
        started_ms: now_ms(),
        ended_ms: None,
    };
    let record_path = dir.join("run.json");
    write_private(&record_path, &serde_json::to_vec_pretty(&record)?)?;

    let full = dir.join("full.gguf");
    let adapter = dir.join("adapter.gguf");
    let log = dir.join("trainer.log");
    let env = &cfg.train.env;
    let mut forward = |stage: Stage, ev: &Event| on(Update::Event(stage, ev.clone()));
    let s = |p: &Path| p.to_string_lossy().into_owned();
    let outcome = async {
        let train_args = vec!["--base".into(), s(&from), "--data".into(), s(&data), "--config".into(), s(&config), "--out".into(), s(&full)];
        let t = step(&program, Stage::Train, &train_args, env, &log, cancel, &mut forward).await?;
        if !full.is_file() {
            return Err(StepEnd::Failed("the train step wrote no model".into()));
        }
        let extract_args =
            vec!["--base".into(), s(&from), "--tuned".into(), s(&full), "--rank".into(), spec.hyper.rank.to_string(), "--out".into(), s(&adapter)];
        step(&program, Stage::ExtractLora, &extract_args, env, &log, cancel, &mut forward).await?;
        if !adapter.is_file() {
            return Err(StepEnd::Failed("the extract-lora step wrote no adapter".into()));
        }
        Ok(t)
    }
    .await;

    let result = match outcome {
        Ok(t) => {
            record.steps = t.step.max(t.steps);
            record.final_loss = t.loss;
            install(&adapter, &dest).map(|(sha256, bytes)| {
                record.adapter = Some(AdapterRecord {
                    id: adapter_id.clone(),
                    file: rel.clone(),
                    sha256,
                    bytes,
                    base_revision: base.revision.clone(),
                    base_quant: if from == served { base.quant.clone() } else { None },
                });
                record.status = RunStatus::Done;
                if !spec.keep_full {
                    let _ = std::fs::remove_file(&full);
                }
            })
        }
        Err(StepEnd::Cancelled) => {
            record.status = RunStatus::Cancelled;
            Err(anyhow::anyhow!("the run was cancelled"))
        }
        Err(StepEnd::Failed(e)) => Err(anyhow::anyhow!(e)),
    };
    if let Err(e) = &result {
        if record.status == RunStatus::Running {
            record.status = RunStatus::Failed;
        }
        record.error = Some(format!("{e:#}"));
        let _ = std::fs::remove_file(&full);
    }
    record.ended_ms = Some(now_ms());
    let _ = write_private(&record_path, &serde_json::to_vec_pretty(&record).unwrap_or_default());

    let released = if in_ymird { release_running(paths, &holder).await.map(|_| ()) } else { Ok(()) };
    let unlocked = drop_lock(paths, &holder).map(|_| ());
    result.with_context(|| format!("run {run_id} (see {})", log.display()))?;
    released.context("releasing this node after the run")?;
    unlocked?;
    Ok(record)
}

/// Moves the adapter to `dest`; returns its SHA-256 and size.
fn install(adapter: &Path, dest: &Path) -> Result<(String, u64)> {
    let bytes = std::fs::read(adapter)?;
    let sha = hex::encode(Sha256::digest(&bytes));
    std::fs::create_dir_all(dest.parent().expect("has a parent"))?;
    if std::fs::rename(adapter, dest).is_err() {
        std::fs::copy(adapter, dest).with_context(|| format!("copying the adapter to {}", dest.display()))?;
        std::fs::remove_file(adapter)?;
    }
    Ok((sha, bytes.len() as u64))
}

/// The runs under the work directory, newest first.
pub fn runs(paths: &Paths, train: &TrainSection) -> Vec<RunRecord> {
    let mut out: Vec<RunRecord> = std::fs::read_dir(train.work_dir(paths))
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| std::fs::read_to_string(e.path().join("run.json")).ok())
        .filter_map(|t| serde_json::from_str(&t).ok())
        .collect();
    out.sort_by_key(|r: &RunRecord| std::cmp::Reverse(r.started_ms));
    out
}

fn new_run_id() -> String {
    let mut b = [0u8; 2];
    let _ = getrandom::fill(&mut b);
    let secs = now_ms() / 1000;
    let t = chrono_like(secs);
    format!("{t}-{}", hex::encode(b))
}

/// `YYYYMMDD-HHMMSS` in UTC.
fn chrono_like(secs: i64) -> String {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}{m:02}{d:02}-{:02}{:02}{:02}", rem / 3600, rem % 3600 / 60, rem % 60)
}

fn file_safe(id: &str) -> String {
    id.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '+') { c } else { '_' }).collect()
}

fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64)
}

fn write_private(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    {
        let mut o = std::fs::OpenOptions::new();
        o.write(true).create(true).truncate(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut o, 0o600);
        let mut f = o.open(&tmp).with_context(|| format!("writing {}", tmp.display()))?;
        f.write_all(bytes)?;
    }
    std::fs::rename(&tmp, path).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_ids_are_utc_timestamps() {
        assert_eq!(chrono_like(0), "19700101-000000");
        assert_eq!(chrono_like(1_795_000_000), "20261118-110640");
        assert_eq!(file_safe("qwen3-8b+ft/x y"), "qwen3-8b+ft_x_y");
    }

    #[test]
    fn events_parse_and_others_are_logs() {
        let e: Event = serde_json::from_str(r#"{"event":"progress","step":2,"steps":4,"loss":0.5}"#).unwrap();
        assert_eq!(e, Event::Progress { step: 2, steps: 4, loss: Some(0.5) });
        assert!(serde_json::from_str::<Event>(r#"{"event":"checkpoint","path":"x"}"#).is_err(), "unknown events are only logged");
        assert_eq!(serde_json::from_str::<Event>(r#"{"event":"done"}"#).unwrap(), Event::Done);
    }

    #[test]
    fn a_lock_held_by_a_dead_process_is_stale() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());
        assert!(held(&paths).is_none());
        take_lock(&paths, "a", Some("r1"), Some(std::process::id())).unwrap();
        assert_eq!(held(&paths).unwrap().holder, "a");
        assert!(take_lock(&paths, "b", None, None).is_err(), "held by a live process");
        assert!(drop_lock(&paths, "b").is_err(), "not b's");
        assert!(drop_lock(&paths, "a").unwrap());
        // A pid no process has (only checked on unix).
        #[cfg(unix)]
        {
            take_lock(&paths, "a", None, Some(u32::MAX / 2)).unwrap();
            assert!(held(&paths).is_none(), "stale");
        }
        take_lock(&paths, "b", None, None).unwrap();
        assert_eq!(held(&paths).unwrap().pid, None, "held until released by hand");
    }
}
