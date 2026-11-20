//! `ymirctl`: an app node for `chat`, and the admin front end. The logic lives here so tests can
//! drive it in process.

use std::io::{BufRead, Write};
use std::path::PathBuf;

pub mod pool;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use serde_json::{Value, json};
use weirdnet_model::hw::human_gib;
use ymir_core::client::BridgeClient;
use ymir_core::config::Roles;
use ymir_core::control::{self, ControlRequest, ModelState, StatusReport};
use ymir_core::setup::{self, SetupOptions};
use ymir_core::{NodeConfig, Paths};

#[derive(Parser, Debug)]
#[command(name = "ymirctl", version, about = "Ymir: a private cloud of LLM models on your own machines")]
pub struct Cli {
    /// Config and state directory (default ~/.config/ymir).
    #[arg(long, global = true, env = "YMIR_HOME")]
    pub home: Option<PathBuf>,
    #[command(subcommand)]
    pub cmd: Cmd,
}

#[derive(Subcommand, Debug)]
pub enum UpdateCmd {
    /// Pin the org root the updates must chain to (hex fingerprint and the first signed root file).
    Trust {
        #[arg(long)]
        pin: String,
        #[arg(long)]
        root: PathBuf,
    },
    /// Verify a bundle directory and stage it for the next start.
    Stage { bundle: PathBuf },
    /// What is pinned, the good release, a trial under way, and releases marked bad.
    Status,
    /// Choose what to start (the good release, a candidate on trial, or a rollback) and print its path;
    /// with `--exec`, run it with the arguments after `--`. Use this as the service's ExecStart.
    Boot {
        #[arg(long)]
        exec: bool,
        /// The daemon to run when no release is installed (default `ymird` on PATH).
        #[arg(long, default_value = "ymird")]
        installed: PathBuf,
        #[arg(last = true)]
        args: Vec<String>,
    },
}

#[derive(Subcommand, Debug)]
pub enum Cmd {
    /// Show what hardware this machine has and the model memory budget.
    Hw {
        #[arg(long)]
        json: bool,
    },
    /// Detect the hardware, pick a backend and models that fit, and write node.toml.
    Setup(SetupArgs),
    /// Self-update: pin the org root, stage a signed bundle, start with rollback.
    Update {
        #[command(subcommand)]
        cmd: UpdateCmd,
    },
    /// Show the running node: roles, models and their state.
    Status {
        #[arg(long)]
        json: bool,
    },
    /// List the pool's models (the merged catalog).
    Models,
    /// Load a model on a model node.
    Load {
        model: String,
        #[arg(long)]
        node: Option<String>,
    },
    /// Unload a model everywhere it is loaded.
    Unload { model: String },
    /// The model nodes the pool routes to.
    Nodes,
    /// Stop sending a model node new requests; it finishes what it has. `--resume` undoes it.
    Drain {
        /// The node's name (see `ymirctl nodes`).
        node: String,
        #[arg(long)]
        resume: bool,
    },
    /// Conversations kept by the inference and app nodes and synced among them.
    Context {
        #[command(subcommand)]
        cmd: ContextCmd,
    },
    /// Ask the pool something and stream the answer.
    Chat {
        /// Model id (default: the main model).
        #[arg(long, short)]
        model: Option<String>,
        #[arg(long)]
        system: Option<String>,
        #[arg(long)]
        temperature: Option<f64>,
        #[arg(long)]
        max_tokens: Option<u64>,
        /// Wait for the model to load instead of taking a stand-in's answer.
        #[arg(long)]
        wait: bool,
        /// The prompt; `-` reads it from stdin.
        #[arg(required = true, num_args = 1..)]
        prompt: Vec<String>,
    },
    /// Split models: a model too big for one node, served by a group of nodes over a private link.
    Group {
        #[command(subcommand)]
        cmd: GroupCmd,
    },
    /// Spread a list of prompts (one per line of FILE, or stdin with `-`) over the pool at batch
    /// priority; prints each answer in order.
    Batch {
        file: String,
        #[arg(long, short)]
        model: Option<String>,
        /// Tasks in flight at once.
        #[arg(long)]
        concurrency: Option<usize>,
        /// Extra tries for a task that fails in a way another try could fix.
        #[arg(long)]
        retries: Option<u32>,
    },
    /// This node's place in your Weirdnet: link it, link other devices, revoke.
    Pool {
        #[command(subcommand)]
        cmd: pool::PoolCmd,
    },
    /// The loopback bridge.
    Bridge {
        #[command(subcommand)]
        cmd: BridgeCmd,
    },
    /// Run ymird as a user-level service (systemd --user on Linux, a launchd agent on macOS).
    Service {
        #[command(subcommand)]
        cmd: ServiceCmd,
    },
}

#[derive(clap::Args, Debug, Default, Clone)]
pub struct ServiceArgs {
    /// Write the unit here instead of the user-service directory (never starts it).
    #[arg(long)]
    pub unit_dir: Option<PathBuf>,
    /// Print what would happen; change nothing.
    #[arg(long)]
    pub dry_run: bool,
}

#[derive(Subcommand, Debug)]
pub enum ServiceCmd {
    /// Write the user service, then enable and start it.
    Install {
        #[command(flatten)]
        args: ServiceArgs,
        /// Write it but don't enable or start it.
        #[arg(long)]
        no_start: bool,
        /// The ymird binary (default: next to ymirctl, else on PATH).
        #[arg(long)]
        ymird: Option<PathBuf>,
    },
    /// Stop, disable and remove the user service.
    Uninstall {
        #[command(flatten)]
        args: ServiceArgs,
    },
    /// Is it installed and running?
    Status {
        #[command(flatten)]
        args: ServiceArgs,
    },
}

#[derive(Subcommand, Debug)]
pub enum BridgeCmd {
    /// Print the bridge token (the API key for Jan and other OpenAI clients).
    Token {
        /// Make a new token (restart ymird to use it).
        #[arg(long)]
        rotate: bool,
    },
}

#[derive(clap::Args, Debug, Default)]
pub struct SetupArgs {
    #[arg(long)]
    pub name: Option<String>,
    /// Use this llama-server binary (default: look on PATH).
    #[arg(long)]
    pub llama_server: Option<PathBuf>,
    /// Serve this GGUF (repeatable; the first is the main model) instead of catalog models.
    #[arg(long = "model")]
    pub models: Vec<PathBuf>,
    /// Also propose catalog models when --model is given.
    #[arg(long)]
    pub catalog: bool,
    /// Memory models may use, such as `10GiB` (default: auto).
    #[arg(long)]
    pub memory_budget: Option<String>,
    /// Unified memory kept for the OS, such as `12GiB`.
    #[arg(long)]
    pub reserve: Option<String>,
    /// Where GGUF files live (default $WEIRDNET_MODELS_DIR or the shared Weirdnet models dir).
    #[arg(long)]
    pub models_dir: Option<PathBuf>,
    /// Bridge port on 127.0.0.1 (default 8484).
    #[arg(long)]
    pub port: Option<u16>,
    /// Roles to turn on (default: model,inference,app).
    #[arg(long)]
    pub roles: Option<String>,
    /// Download the proposed catalog models (asks first unless --yes).
    #[arg(long)]
    pub download: bool,
    /// Answer yes to download prompts.
    #[arg(long)]
    pub yes: bool,
    /// Replace an existing node.toml.
    #[arg(long)]
    pub force: bool,
    /// Print the proposal and write nothing.
    #[arg(long)]
    pub dry_run: bool,
    /// Also install and start ymird as a user service.
    #[arg(long)]
    pub service: bool,
    /// With --service: write the unit here instead (never starts it).
    #[arg(long, requires = "service")]
    pub unit_dir: Option<PathBuf>,
    /// With --service: write the unit but don't start it.
    #[arg(long, requires = "service")]
    pub no_start: bool,
    /// With --service: the ymird binary (default: next to ymirctl, else on PATH).
    #[arg(long, requires = "service")]
    pub ymird: Option<PathBuf>,
}

#[derive(clap::Subcommand, Debug)]
pub enum GroupCmd {
    /// Plan and start the group for MODEL (marked `split` in the pool policy).
    Up { model: String },
    /// Stop the group; its members return to normal placement.
    Down { model: String },
    /// The groups that are up, and why the last ones went down.
    Status,
}

#[derive(clap::Subcommand, Debug)]
pub enum ContextCmd {
    /// List the conversations here, or show one.
    Show { conversation: Option<String> },
    /// Add a message to a conversation.
    Add {
        conversation: String,
        text: String,
        #[arg(long, default_value = "user")]
        role: String,
    },
    /// Add a user message, then have the pool answer the conversation; the answer is written back
    /// and syncs to the other nodes.
    Ask {
        conversation: String,
        text: String,
        #[arg(long, short)]
        model: Option<String>,
    },
    /// Delete a conversation on every node (a tombstone that syncs).
    Delete { conversation: String },
}

pub async fn run(cli: Cli, out: &mut dyn Write, input: &mut dyn BufRead) -> Result<()> {
    let paths = cli.home.map(Paths::new).unwrap_or_else(Paths::from_env);
    match cli.cmd {
        Cmd::Hw { json } => hw(out, json),
        Cmd::Setup(args) => setup_cmd(&paths, args, out, input).await,
        Cmd::Status { json } => status(&paths, out, json).await,
        Cmd::Models => models(&paths, out).await,
        Cmd::Load { model, node } => admin(&paths, out, ControlRequest::Load { model, node }).await,
        Cmd::Service { cmd } => service_cmd(&paths, out, cmd),
        Cmd::Update { cmd } => update_cmd(&paths, out, cmd),
        Cmd::Unload { model } => admin(&paths, out, ControlRequest::Unload { model }).await,
        Cmd::Nodes => nodes(&paths, out).await,
        Cmd::Drain { node, resume } => admin(&paths, out, ControlRequest::Drain { node, on: !resume }).await,
        Cmd::Context { cmd } => context_cmd(&paths, out, cmd).await,
        Cmd::Group { cmd } => match cmd {
            GroupCmd::Up { model } => admin(&paths, out, ControlRequest::Group { action: "up".into(), model: Some(model) }).await,
            GroupCmd::Down { model } => admin(&paths, out, ControlRequest::Group { action: "down".into(), model: Some(model) }).await,
            GroupCmd::Status => {
                let r = control::request(&paths.socket(), &ControlRequest::Group { action: "status".into(), model: None }).await?;
                if !r.ok {
                    bail!("{}", r.message.unwrap_or_default());
                }
                if r.groups.is_empty() {
                    writeln!(out, "no group is up")?;
                }
                for g in &r.groups {
                    writeln!(out, "{}  leader {}  members {}", g.model, g.leader, g.members.join(", "))?;
                }
                for l in &r.group_losses {
                    writeln!(out, "lost  {}  {}", l.model, l.reason)?;
                }
                Ok(())
            }
        },
        Cmd::Chat { model, system, temperature, max_tokens, wait, prompt } => {
            chat(&paths, out, input, model, system, temperature, max_tokens, wait, prompt).await
        }
        Cmd::Batch { file, model, concurrency, retries } => {
            let text = if file == "-" {
                let mut s = String::new();
                input.read_to_string(&mut s)?;
                s
            } else {
                std::fs::read_to_string(&file).with_context(|| format!("reading {file}"))?
            };
            let prompts: Vec<String> = text.lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_string).collect();
            let r = control::request(&paths.socket(), &ControlRequest::Batch { prompts, model, concurrency, retries }).await?;
            if !r.ok {
                bail!("{}", r.message.unwrap_or_default());
            }
            for i in &r.batch {
                if i.ok {
                    writeln!(out, "{}\t{}\t{}", i.index, i.node, i.text.replace('\n', " "))?;
                } else {
                    writeln!(out, "{}\tFAILED after {} tries: {}", i.index, i.attempts, i.text)?;
                }
            }
            writeln!(out, "# {}", r.message.unwrap_or_default())?;
            Ok(())
        }
        Cmd::Pool { cmd } => {
            let cfg = load_cfg(&paths)?;
            pool::run(&paths, &cfg, cmd, out, input).await
        }
        Cmd::Bridge { cmd: BridgeCmd::Token { rotate } } => {
            let t = if rotate { ymir_core::token::rotate(&paths)? } else { ymir_core::token::load_or_create(&paths)? };
            writeln!(out, "{t}")?;
            if rotate {
                writeln!(out, "# restart ymird to use the new token")?;
            }
            Ok(())
        }
    }
}

fn hw(out: &mut dyn Write, json: bool) -> Result<()> {
    let hw = weirdnet_model::hw::detect();
    let b = weirdnet_model::hw::budget(&hw, None, None);
    if json {
        writeln!(out, "{}", serde_json::to_string_pretty(&json!({"hardware": hw, "budget": b}))?)?;
        return Ok(());
    }
    print_hw(out, &hw, &b)
}

fn print_hw(out: &mut dyn Write, hw: &weirdnet_model::hw::Hardware, b: &weirdnet_model::hw::Budget) -> Result<()> {
    writeln!(out, "Machine: {} {}, {} CPU threads, {} RAM", hw.os, hw.arch, hw.cpu_threads, human_gib(hw.ram_bytes))?;
    if hw.gpus.is_empty() {
        writeln!(out, "Accelerator: none found (CPU only)")?;
    }
    for g in &hw.gpus {
        let cc = g.compute_capability.map(|(a, b)| format!(", sm_{a}{b}")).unwrap_or_default();
        let kind = if g.unified { "unified" } else { "dedicated" };
        writeln!(out, "Accelerator: {} ({:?}{cc}), {} {kind}", g.name, g.accel, human_gib(g.memory_bytes))?;
    }
    writeln!(out, "Model memory: {} usable of {} (reserve {})", human_gib(b.usable_bytes), human_gib(b.pool_bytes), human_gib(b.reserve_bytes))?;
    writeln!(
        out,
        "vLLM: {}",
        if hw.vllm_supported() { "supported on this hardware (optional)" } else { "not supported on this hardware (llama.cpp only)" }
    )?;
    if hw.needs_cuda13() {
        writeln!(out, "Note: sm_121 needs CUDA 13 builds of llama.cpp and vLLM")?;
    }
    Ok(())
}

async fn setup_cmd(paths: &Paths, args: SetupArgs, out: &mut dyn Write, input: &mut dyn BufRead) -> Result<()> {
    let cfg_path = paths.config();
    if cfg_path.exists() && !args.force && !args.dry_run {
        bail!("{} already exists; use --force to replace it", cfg_path.display());
    }
    for m in &args.models {
        if !m.is_file() {
            bail!("{} is not a file", m.display());
        }
    }
    let opts = SetupOptions {
        name: args.name,
        llama_server: args.llama_server,
        models: args.models,
        catalog: args.catalog,
        memory_budget: args.memory_budget,
        reserve: args.reserve,
        models_dir: args.models_dir,
        bridge_port: args.port,
    };
    let mut p = setup::propose(weirdnet_model::hw::detect(), &opts);
    if let Some(r) = &args.roles {
        p.config.roles = Roles::parse(r)?;
    }
    print_hw(out, &p.hardware, &p.plan.budget)?;
    for n in &p.plan.notes {
        writeln!(out, "Note: {n}")?;
    }
    match &p.llama_server {
        Some(l) => writeln!(out, "llama.cpp: {}", l.display())?,
        None => writeln!(
            out,
            "llama.cpp: not found. Install llama.cpp (llama-server) or pass --llama-server PATH; managed builds come with weirdnet-backends"
        )?,
    }
    match &p.vllm {
        Some(v) => writeln!(out, "vLLM: {}", v.display())?,
        None if p.hardware.vllm_supported() => writeln!(out, "vLLM: not installed (optional; GGUF models use llama.cpp)")?,
        None => {}
    }
    writeln!(out, "Models:")?;
    for m in &p.config.model.local {
        let size = weirdnet_model::hw::find(&m.id).map(|e| format!(" ({})", weirdnet_model::human(e.spec.size))).unwrap_or_default();
        let missing = if p.to_download.iter().any(|e| e.spec.id == m.id) { ", not downloaded" } else { "" };
        writeln!(out, "  {} [{}]{size}{missing}", m.id, m.role.as_deref().unwrap_or("chat"))?;
    }
    if args.dry_run {
        writeln!(out, "\n{}", p.config.to_toml()?)?;
        return Ok(());
    }
    if args.download && !p.to_download.is_empty() {
        let store = weirdnet_model::ModelStore::new(p.config.models_dir());
        for e in p.to_download.clone() {
            for line in e.spec.consent_text(store.hub()) {
                writeln!(out, "{line}")?;
            }
            if !args.yes && !confirm(out, input, "Download it? [y/N] ")? {
                writeln!(out, "Skipped {}", e.spec.id)?;
                continue;
            }
            let spec = e.spec.clone();
            let st = store.clone();
            let path = tokio::task::spawn_blocking(move || {
                let consent = weirdnet_model::Consent::given(&spec);
                let mut last = 0u32;
                weirdnet_model::download(&st, &spec, &consent, &mut |pr| {
                    let pct = (pr.fraction() * 100.0) as u32;
                    if pct >= last + 10 {
                        last = pct;
                        eprintln!("  {pct}%");
                    }
                    true
                })
            })
            .await??;
            writeln!(out, "Downloaded {}", path.display())?;
            p.to_download.retain(|d| d.spec.id != e.spec.id);
        }
    }
    p.config.save(&cfg_path)?;
    ymir_core::token::load_or_create(paths)?;
    writeln!(out, "\nWrote {}", cfg_path.display())?;
    writeln!(out, "Roles: {}", p.config.roles.names().join(", "))?;
    if !p.to_download.is_empty() {
        writeln!(
            out,
            "Not downloaded yet: {} (run `ymirctl setup --download --force`)",
            p.to_download.iter().map(|e| e.spec.id.as_str()).collect::<Vec<_>>().join(", ")
        )?;
    }
    if args.service {
        let opts = ymir_core::service::InstallOptions { dir: args.unit_dir.clone(), start: !args.no_start, dry_run: false };
        for line in ymir_core::service::install(&service_spec(paths, args.ymird.clone())?, &opts)? {
            writeln!(out, "{line}")?;
        }
        writeln!(out, "Next: `ymirctl status`, then `ymirctl chat \"Hello\"`.")?;
    } else {
        writeln!(out, "Next: run `ymird` (or `ymirctl service install`), then `ymirctl chat \"Hello\"`.")?;
    }
    if p.config.roles.app {
        writeln!(out, "OpenAI clients (Jan): base URL http://127.0.0.1:{}/v1, API key from `ymirctl bridge token`.", p.config.app.bridge.port)?;
    }
    Ok(())
}

fn service_spec(paths: &Paths, ymird: Option<PathBuf>) -> Result<ymir_core::service::ServiceSpec> {
    let ymird = ymird.or_else(ymir_core::service::find_ymird).context("ymird not found next to ymirctl or on PATH; pass --ymird")?;
    let ymird = std::fs::canonicalize(&ymird).unwrap_or(ymird);
    let home = std::fs::canonicalize(&paths.home).unwrap_or_else(|_| paths.home.clone());
    Ok(ymir_core::service::ServiceSpec { ymird, home })
}

fn service_cmd(paths: &Paths, out: &mut dyn Write, cmd: ServiceCmd) -> Result<()> {
    use ymir_core::service;
    let lines = match cmd {
        ServiceCmd::Install { args, no_start, ymird } => {
            if !paths.config().exists() && !args.dry_run {
                bail!("{} does not exist; run `ymirctl setup` first", paths.config().display());
            }
            service::install(&service_spec(paths, ymird)?, &service::InstallOptions { dir: args.unit_dir, start: !no_start, dry_run: args.dry_run })?
        }
        ServiceCmd::Uninstall { args } => service::uninstall(args.unit_dir.as_deref(), args.dry_run)?,
        ServiceCmd::Status { args } => service::status(args.unit_dir.as_deref())?,
    };
    for l in lines {
        writeln!(out, "{l}")?;
    }
    Ok(())
}

fn confirm(out: &mut dyn Write, input: &mut dyn BufRead, prompt: &str) -> Result<bool> {
    write!(out, "{prompt}")?;
    out.flush()?;
    let mut line = String::new();
    input.read_line(&mut line)?;
    Ok(matches!(line.trim(), "y" | "Y" | "yes"))
}

fn load_cfg(paths: &Paths) -> Result<NodeConfig> {
    NodeConfig::load(&paths.config()).with_context(|| "run `ymirctl setup` first")
}

async fn status(paths: &Paths, out: &mut dyn Write, json: bool) -> Result<()> {
    let r = control::request(&paths.socket(), &ControlRequest::Status).await?;
    let s = r.status.context(r.message.unwrap_or_else(|| "no status".into()))?;
    if json {
        writeln!(out, "{}", serde_json::to_string_pretty(&s)?)?;
        return Ok(());
    }
    writeln!(out, "Node {} (ymird {}, pid {}), roles: {}", s.node, s.version, s.pid, s.roles.join(", "))?;
    if let Some(p) = s.bridge_port {
        writeln!(out, "Bridge: http://127.0.0.1:{p}/v1")?;
    }
    if let Some(m) = &s.main_model {
        writeln!(out, "Main model: {m}")?;
    }
    for m in &s.models {
        let state = match &m.state {
            ModelState::Failed(why) => format!("failed: {why}"),
            other => format!("{other:?}").to_lowercase(),
        };
        let mut extra = String::new();
        if m.wrapped {
            extra.push_str(", wrapped");
        }
        if let Some(s) = m.last_load_secs {
            extra.push_str(&format!(", loads in {s:.1}s"));
        }
        if m.crashes > 0 {
            extra.push_str(&format!(", {} crashes", m.crashes));
        }
        writeln!(out, "  {:<24} {:<10} {:<9} on {} ({}, {} in flight{extra})", m.id, m.purpose, state, m.node, m.engine.name(), m.in_flight)?;
    }
    if s.nodes.len() > 1 || s.nodes.iter().any(|n| !n.local) {
        writeln!(out, "Model nodes:")?;
        write_nodes(out, &s.nodes)?;
    }
    if !s.alerts.is_empty() {
        writeln!(out, "Alerts: {}", s.alerts.join(", "))?;
    }
    match &s.network {
        Some(n) => {
            let linked = n.devices.iter().filter(|d| d.active && !d.me).count();
            writeln!(out, "Weirdnet: {} as {}, {} linked devices, {} connected", n.user, n.peer, linked, n.connected.len())?;
            for a in &n.listening {
                writeln!(out, "  listening {a}")?;
            }
        }
        None => writeln!(out, "Weirdnet: off (link with `ymirctl pool link`)")?,
    }
    Ok(())
}

fn write_nodes(out: &mut dyn Write, nodes: &[control::NodeSummary]) -> Result<()> {
    for n in nodes {
        let state = if n.draining { "draining" } else { "serving" };
        let here = if n.local { " (this box)" } else { "" };
        writeln!(out, "  {:<24} {:<9} {} of {} models ready, {} in flight{here}", n.name, state, n.ready, n.models, n.in_flight)?;
    }
    Ok(())
}

async fn nodes(paths: &Paths, out: &mut dyn Write) -> Result<()> {
    let r = control::request(&paths.socket(), &ControlRequest::Status).await?;
    let s = r.status.context(r.message.unwrap_or_else(|| "no status".into()))?;
    if s.nodes.is_empty() {
        writeln!(out, "No model nodes (this node has neither the inference nor the model role).")?;
    }
    write_nodes(out, &s.nodes)
}

async fn bridge(paths: &Paths) -> Result<BridgeClient> {
    let cfg = load_cfg(paths)?;
    let mut client = BridgeClient::for_node(paths, &cfg)?;
    // The running daemon knows the real port (it may differ from node.toml when that says 0).
    if let Ok(r) = control::request(&paths.socket(), &ControlRequest::Status).await
        && let Some(StatusReport { bridge_port: Some(port), .. }) = r.status
    {
        client = BridgeClient::new(port, ymir_core::token::load(paths)?.unwrap_or_default());
    }
    Ok(client)
}

async fn models(paths: &Paths, out: &mut dyn Write) -> Result<()> {
    let v = bridge(paths).await?.models().await?;
    for m in v["data"].as_array().into_iter().flatten() {
        let ready = match m["weirdnet"]["state"].as_str() {
            Some("warm") => "ready",
            Some("busy") => "ready (busy)",
            Some("loading") => "loading",
            _ => "on disk",
        };
        writeln!(out, "{:<24} {:<10} {}", m["id"].as_str().unwrap_or("?"), m["weirdnet"]["purpose"].as_str().unwrap_or(""), ready)?;
    }
    Ok(())
}

async fn context_cmd(paths: &Paths, out: &mut dyn Write, cmd: ContextCmd) -> Result<()> {
    let req = match cmd {
        ContextCmd::Show { conversation } => ControlRequest::ContextShow { conversation },
        ContextCmd::Add { conversation, text, role } => ControlRequest::ContextAdd { conversation, role, text },
        ContextCmd::Ask { conversation, text, model } => ControlRequest::Chat { conversation, text, model },
        ContextCmd::Delete { conversation } => ControlRequest::ContextDelete { conversation },
    };
    let r = control::request(&paths.socket(), &req).await?;
    let msg = r.message.clone().unwrap_or_default();
    if !r.ok {
        bail!("{msg}");
    }
    if let Some(c) = r.chat {
        writeln!(out, "{}", c.text)?;
        let moved = if c.resumes > 0 { format!(", moved {} time(s) mid-answer", c.resumes) } else { String::new() };
        writeln!(out, "# {} via {}{moved}", c.model, c.node)?;
        for step in &c.path {
            writeln!(out, "#   {step}")?;
        }
    } else if !r.conversations.is_empty() {
        for c in r.conversations {
            writeln!(out, "{}  {}  ({} messages)", c.id, if c.title.is_empty() { "-" } else { &c.title }, c.messages.len())?;
            for (role, text) in c.messages {
                writeln!(out, "  {role}: {text}")?;
            }
        }
    } else {
        writeln!(out, "{msg}")?;
    }
    Ok(())
}

async fn admin(paths: &Paths, out: &mut dyn Write, req: ControlRequest) -> Result<()> {
    let r = control::request(&paths.socket(), &req).await?;
    let msg = r.message.unwrap_or_default();
    if !r.ok {
        bail!("{msg}");
    }
    writeln!(out, "{msg}")?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn chat(
    paths: &Paths,
    out: &mut dyn Write,
    input: &mut dyn BufRead,
    model: Option<String>,
    system: Option<String>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    wait: bool,
    prompt: Vec<String>,
) -> Result<()> {
    let text = if prompt == ["-"] {
        let mut s = String::new();
        input.read_to_string(&mut s)?;
        s
    } else {
        prompt.join(" ")
    };
    let mut messages = Vec::new();
    if let Some(s) = system {
        messages.push(json!({"role": "system", "content": s}));
    }
    messages.push(json!({"role": "user", "content": text}));
    let mut extra = serde_json::Map::new();
    if let Some(t) = temperature {
        extra.insert("temperature".into(), Value::from(t));
    }
    if let Some(n) = max_tokens {
        extra.insert("max_tokens".into(), Value::from(n));
    }
    if wait {
        extra.insert("wait_for_model".into(), Value::Bool(true));
    }
    let client = bridge(paths).await?;
    let mut io_err = None;
    let outcome = client
        .chat_stream(model.as_deref(), messages, extra, |t| {
            if let Err(e) = out.write_all(t.as_bytes()).and_then(|_| out.flush()) {
                io_err.get_or_insert(e);
            }
        })
        .await?;
    if let Some(e) = io_err {
        return Err(e.into());
    }
    writeln!(out)?;
    if outcome.cache_hit {
        writeln!(out, "(cached)")?;
    }
    if let Some(s) = &outcome.standin {
        writeln!(out, "(answered by the stand-in {s} while the model loads)")?;
    }
    Ok(())
}

fn update_cmd(paths: &Paths, out: &mut dyn Write, cmd: UpdateCmd) -> Result<()> {
    use ymir_core::update::{Boot, Updater};
    let now = weirdnet_update::Timestamp::now();
    let mut u = Updater::open(paths, now)?;
    match cmd {
        UpdateCmd::Trust { pin, root } => {
            u.trust(&pin, &std::fs::read(&root).with_context(|| format!("reading {}", root.display()))?, now)?;
            writeln!(out, "pinned: updates must chain to {pin}")?;
        }
        UpdateCmd::Stage { bundle } => {
            let running = semver::Version::parse(env!("CARGO_PKG_VERSION"))?;
            let v = u.stage(&bundle, &running, now)?;
            writeln!(out, "staged {} {}; it runs on trial at the next start", v.manifest.update_id, v.manifest.version)?;
        }
        UpdateCmd::Status => {
            let s = u.status();
            writeln!(out, "root: {}", if s.trusted { "pinned" } else { "not pinned" })?;
            writeln!(out, "good release: {}", s.good.as_deref().unwrap_or("none (the installed binary)"))?;
            writeln!(out, "health: {:?}", s.health)?;
            if !s.bad.is_empty() {
                writeln!(out, "marked bad: {}", s.bad.join(", "))?;
            }
        }
        UpdateCmd::Boot { exec, installed, args } => {
            let boot = u.boot()?;
            let (path, note) = match &boot {
                Boot::Installed => (installed, "installed binary".to_string()),
                Boot::Run { id, path, trial } => (path.clone(), format!("release {id}{}", if *trial { " (on trial)" } else { "" })),
                Boot::RolledBack { failed, run } => (run.as_ref().map(|r| r.1.clone()).unwrap_or(installed), format!("rolled back from {failed}")),
            };
            if !exec {
                writeln!(out, "{}", path.display())?;
                return Ok(());
            }
            eprintln!("ymirctl update boot: starting {note}");
            #[cfg(unix)]
            {
                use std::os::unix::process::CommandExt;
                let e = std::process::Command::new(&path).args(&args).exec();
                bail!("starting {}: {e}", path.display());
            }
            #[cfg(not(unix))]
            {
                let st = std::process::Command::new(&path).args(&args).status()?;
                std::process::exit(st.code().unwrap_or(1));
            }
        }
    }
    Ok(())
}
