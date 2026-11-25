//! `ymirctl train`: fine-tuning on this node (training plan P3a, see `ymir_core::train`).
//! `run` trains a LoRA adapter for a llama.cpp base with the pinned trainer, holding the GPU for
//! the run; `reserve` and `release` hold and free a node's GPU by hand; `check` tests the
//! trainer against the pin; `list` shows past runs.

use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Args, Subcommand};
use ymir_core::config::LocalAdapter;
use ymir_core::control::{self, ControlRequest};
use ymir_core::train::{self, Event, Hyper, RunSpec, Stage, Update};
use ymir_core::{NodeConfig, Paths};

#[derive(Subcommand, Debug)]
pub enum TrainCmd {
    /// Train a LoRA adapter for a llama.cpp base on this node: a full fine-tune with the pinned
    /// trainer, then the adapter extracted from it, written to models_dir/adapters/ID/ with its
    /// provenance.json (as an upload of it would be). The node's GPU is held for the run (its
    /// models unload and none load until it ends). Ctrl-C cancels.
    Run(RunArgs),
    /// Hold a model node's GPU by hand: its models unload and none load until `train release`.
    /// Without --node, this box (also kept across a ymird restart).
    Reserve {
        #[arg(long)]
        node: Option<String>,
        /// Even when it is the only node serving a loaded model.
        #[arg(long)]
        force: bool,
    },
    /// Free a GPU held by `train reserve`.
    Release {
        #[arg(long)]
        node: Option<String>,
    },
    /// Check the trainer against [train] in node.toml: the interface and the pinned build.
    Check,
    /// The runs on this node, newest first.
    List,
}

#[derive(Args, Debug)]
pub struct RunArgs {
    /// The base: a llama.cpp [[model.local]] id on this node.
    #[arg(long)]
    pub base: String,
    /// The dataset: JSONL from `ymirctl batch --jsonl` or `ymirctl context export --training`,
    /// with its manifest next to it.
    #[arg(long)]
    pub dataset: PathBuf,
    /// The adapter's id in the pool (default BASE+ft-RUN).
    #[arg(long)]
    pub id: Option<String>,
    /// Train from this GGUF instead of the served file (an F32 copy of the same model when the
    /// served file is quantised: llama.cpp's trainer trains F32 models only).
    #[arg(long)]
    pub from: Option<PathBuf>,
    #[arg(long, default_value_t = Hyper::default().epochs)]
    pub epochs: u32,
    #[arg(long = "lr", default_value_t = Hyper::default().learning_rate)]
    pub learning_rate: f64,
    /// Tokens per training sequence.
    #[arg(long, default_value_t = Hyper::default().context)]
    pub context: u32,
    #[arg(long, default_value_t = Hyper::default().batch)]
    pub batch: u32,
    #[arg(long, default_value_t = Hyper::default().seed)]
    pub seed: u64,
    /// The adapter's LoRA rank.
    #[arg(long, default_value_t = Hyper::default().rank)]
    pub rank: u32,
    /// Train on a generated dataset whose teacher license is recorded as unknown.
    #[arg(long)]
    pub allow_unknown_license: bool,
    /// Keep the full fine-tuned model (as large as the base) in the run's directory.
    #[arg(long)]
    pub keep_full: bool,
    /// Reserve this node even when it is the only one serving a loaded model.
    #[arg(long)]
    pub force: bool,
    /// Add the adapter to node.toml (`[[model.local.lora]]`) when the run succeeds.
    #[arg(long)]
    pub add: bool,
    /// Run on this model node instead (a roster label or peer id), over Weirdnet through this
    /// device's ymird. The node needs `[train] remote = true`; the dataset is sent with the run
    /// (up to 24 MiB), --from names a GGUF in the node's models directory, and node.toml there is
    /// never edited (serve the adapter with `ymirctl models adapter serve --node`).
    #[arg(long)]
    pub node: Option<String>,
}

pub async fn run(paths: &Paths, out: &mut dyn Write, cmd: TrainCmd) -> Result<()> {
    match cmd {
        TrainCmd::Run(a) => run_cmd(paths, out, a).await,
        TrainCmd::Check => {
            let cfg = crate::load_cfg(paths)?;
            let (program, v) = train::check_trainer(&cfg.train).await?;
            writeln!(out, "{}: {} build {} (interface {}), as pinned.", program.display(), v.engine, v.build, v.interface)?;
            Ok(())
        }
        TrainCmd::List => {
            let cfg = crate::load_cfg(paths)?;
            let runs = train::runs(paths, &cfg.train);
            if runs.is_empty() {
                writeln!(out, "No training runs on this node.")?;
            }
            for r in runs {
                let status = serde_json::to_value(r.status)?.as_str().unwrap_or_default().to_string();
                let adapter = r.adapter.as_ref().map(|a| format!("  -> {} ({})", a.id, a.file)).unwrap_or_default();
                let loss = r.final_loss.map(|l| format!("  loss {l:.4}")).unwrap_or_default();
                writeln!(out, "{}  {status:<9} {} on {} ({} rows){loss}{adapter}", r.id, r.base, r.dataset.name, r.dataset.rows_used)?;
                if let Some(e) = &r.error {
                    writeln!(out, "    {e}")?;
                }
            }
            if let Some(l) = train::held(paths) {
                writeln!(out, "This node's GPU is held by {}.", l.holder)?;
            }
            Ok(())
        }
        TrainCmd::Reserve { node: Some(node), force } => {
            let r = control::request(&paths.socket(), &ControlRequest::Reserve { node: Some(node), holder: train::OPERATOR.into(), force }).await?;
            done(out, r)
        }
        TrainCmd::Release { node: Some(node) } => {
            let r = control::request(&paths.socket(), &ControlRequest::Release { node: Some(node), holder: train::OPERATOR.into() }).await?;
            done(out, r)
        }
        TrainCmd::Reserve { node: None, force } => {
            train::take_lock(paths, train::OPERATOR, None, None)?;
            match train::reserve_running(paths, train::OPERATOR, force).await {
                Ok(true) => writeln!(out, "This node is reserved: its models are unloaded and none load until `ymirctl train release`.")?,
                Ok(false) => {
                    writeln!(out, "This node is reserved: ymird isn't running, and won't load a model when it starts until `ymirctl train release`.")?
                }
                Err(e) => {
                    let _ = train::drop_lock(paths, train::OPERATOR);
                    return Err(e);
                }
            }
            Ok(())
        }
        TrainCmd::Release { node: None } => {
            if let Some(l) = train::held(paths)
                && l.holder != train::OPERATOR
            {
                bail!("this node's GPU is held by {}, a training run; it is released when the run ends (Ctrl-C cancels it)", l.holder);
            }
            let in_ymird = train::release_running(paths, train::OPERATOR).await?;
            let had = train::drop_lock(paths, train::OPERATOR)?;
            if in_ymird || had {
                writeln!(out, "This node is released: models load again.")?;
            } else {
                writeln!(out, "This node was not reserved.")?;
            }
            Ok(())
        }
    }
}

fn done(out: &mut dyn Write, r: control::ControlResponse) -> Result<()> {
    if !r.ok {
        bail!("{}", r.message.unwrap_or_default());
    }
    writeln!(out, "{}", r.message.unwrap_or_default())?;
    Ok(())
}

/// A progress line for `u`, at most one per percent of a step (or per step on short runs).
fn show(shown: &mut Option<(Stage, u64)>, stage: Stage, event: &Event) -> Option<String> {
    match event {
        Event::Progress { step, steps, loss } => {
            let pct = (step * 100).checked_div(*steps).unwrap_or(0);
            (*shown != Some((stage, pct))).then(|| {
                *shown = Some((stage, pct));
                let loss = loss.map(|l| format!("  loss {l:.4}")).unwrap_or_default();
                format!("{stage}: {step}/{steps}{loss}")
            })
        }
        Event::Log { message } => Some(format!("{stage}: {message}")),
        Event::Error { message } => Some(format!("{stage}: error: {message}")),
        Event::Done => None,
    }
}

async fn run_on_node(paths: &Paths, out: &mut dyn Write, node: &str, a: RunArgs) -> Result<()> {
    use ymir_core::train_net::{TrainLine, TrainOp};
    if a.add {
        bail!(
            "--add edits node.toml, which is never done over the network; serve the adapter with `ymirctl models adapter serve ID --base BASE --node {node}`"
        );
    }
    let from = match a.from {
        Some(p) => Some(p.to_str().context("--from is not UTF-8")?.to_string()),
        None => None,
    };
    let op = TrainOp {
        base: a.base.clone(),
        dataset: std::path::absolute(&a.dataset).unwrap_or(a.dataset),
        id: a.id,
        from,
        hyper: Hyper { epochs: a.epochs, learning_rate: a.learning_rate, context: a.context, batch: a.batch, seed: a.seed, rank: a.rank },
        allow_unknown_license: a.allow_unknown_license,
        keep_full: a.keep_full,
        force: a.force,
    };
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let mut shown: Option<(Stage, u64)> = None;
    let on = |l: &TrainLine| {
        let line = match l {
            TrainLine::Started { node, run, adapter } => Some(format!("Run {run} for `{adapter}` on {node}")),
            TrainLine::Reserved { in_ymird: true } => Some("Reserved the node in its ymird: its models are unloaded until the run ends.".into()),
            TrainLine::Reserved { in_ymird: false } => Some("The run holds the node's GPU.".into()),
            TrainLine::Step { stage, event } => show(&mut shown, *stage, event),
            TrainLine::Done { .. } | TrainLine::Failed { .. } => None,
        };
        if let Some(l) = line {
            let _ = tx.send(l);
        }
    };
    let result = {
        let socket = paths.socket();
        let run = control::request_train(&socket, node, op, on);
        tokio::pin!(run);
        let cancel = tokio::signal::ctrl_c();
        tokio::pin!(cancel);
        loop {
            tokio::select! {
                Some(l) = rx.recv() => writeln!(out, "{l}")?,
                r = &mut run => break r,
                _ = &mut cancel => break Err(anyhow::anyhow!("cancelled: the run on {node} stops")),
            }
        }
    };
    while let Ok(l) = rx.try_recv() {
        writeln!(out, "{l}")?;
    }
    let record = result?;
    let adapter = record.adapter.clone().context("the run produced no adapter")?;
    let loss = record.final_loss.map(|l| format!(", final loss {l:.4}")).unwrap_or_default();
    writeln!(out, "Done on {}: {} steps{loss}. Adapter `{}` is {} in its models directory.", record.node, record.steps, adapter.id, adapter.file)?;
    writeln!(out, "To serve it: ymirctl models adapter serve {} --base {} --node {node}", adapter.id, record.base)?;
    Ok(())
}

async fn run_cmd(paths: &Paths, out: &mut dyn Write, mut a: RunArgs) -> Result<()> {
    if let Some(node) = a.node.take() {
        return run_on_node(paths, out, &node, a).await;
    }
    let config = paths.config();
    let text = std::fs::read_to_string(&config).with_context(|| format!("reading {} (run `ymirctl setup` first)", config.display()))?;
    let cfg: NodeConfig = toml::from_str(&text).with_context(|| format!("parsing {}", config.display()))?;
    let spec = RunSpec {
        base: a.base.clone(),
        dataset: a.dataset,
        id: a.id,
        from: a.from,
        hyper: Hyper { epochs: a.epochs, learning_rate: a.learning_rate, context: a.context, batch: a.batch, seed: a.seed, rank: a.rank },
        allow_unknown_license: a.allow_unknown_license,
        keep_full: a.keep_full,
        force: a.force,
    };
    let mut cancel = Box::pin(async {
        let _ = tokio::signal::ctrl_c().await;
    });
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let mut shown: Option<(Stage, u64)> = None;
    let mut on = |u: Update| {
        let line = match u {
            Update::Started { run, dir, adapter } => Some(format!("Run {run} for `{adapter}` (work in {})", dir.display())),
            Update::Reserved { in_ymird: true } => Some("Reserved this node in ymird: its models are unloaded until the run ends.".into()),
            Update::Reserved { in_ymird: false } => Some("ymird isn't running; the run holds the GPU (ymird loads nothing if it starts).".into()),
            Update::Event(stage, event) => show(&mut shown, stage, &event),
        };
        if let Some(l) = line {
            let _ = tx.send(l);
        }
    };
    let result = {
        let run = train::run(paths, &cfg, &spec, &mut cancel, &mut on);
        tokio::pin!(run);
        loop {
            tokio::select! {
                Some(l) = rx.recv() => writeln!(out, "{l}")?,
                r = &mut run => break r,
            }
        }
    };
    while let Ok(l) = rx.try_recv() {
        writeln!(out, "{l}")?;
    }
    let record = result?;
    let adapter = record.adapter.clone().context("the run produced no adapter")?;
    let loss = record.final_loss.map(|l| format!(", final loss {l:.4}")).unwrap_or_default();
    writeln!(out, "Done: {} steps{loss}. Adapter `{}` is {} in the models directory.", record.steps, adapter.id, adapter.file)?;
    // The content revision, as provenance.json and an upload of the same file give it, so the same
    // adapter has the same revision on every node it reaches.
    let revision = adapter.revision.clone().unwrap_or_else(|| record.id.clone());
    if let Some((dir, _)) = adapter.file.rsplit_once('/') {
        writeln!(out, "Provenance: {dir}/{} (origin train, run {})", ymir_core::upload::PROVENANCE_FILE, record.id)?;
    }
    let la = LocalAdapter {
        id: adapter.id.clone(),
        file: adapter.file.clone(),
        scale: None,
        revision: Some(revision.clone()),
        base_revision: adapter.base_revision.clone(),
        base_quant: adapter.base_quant.clone(),
    };
    if a.add {
        let new = crate::models::adapters_add(&text, &a.base, &la)?;
        crate::models::adapters_save(&config, &text, &new)?;
        writeln!(out, "Added it to node.toml on `{}` (revision {revision}). Restart ymird to serve it.", a.base)?;
    } else {
        writeln!(out, "To serve it: ymirctl models adapter add {} {} --id {} --revision {revision}", a.base, adapter.file, adapter.id)?;
    }
    Ok(())
}
