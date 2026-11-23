//! `ymirctl models search | pull | list | remove`: find models on the Hugging Face Hub, download
//! one that fits this node, and see or delete what is on disk. All of it is local to the node it
//! runs on; it never edits `node.toml` unless `pull --register` says so.
//!
//! Without `--node` everything runs on this machine and needs no daemon. With `--node` (a device
//! name, a peer id, `auto` or `all`) the request goes through the local `ymird` to the model nodes
//! among the user's devices over Weirdnet (`remote.rs`).
//!
//! Hub search, repo files, GGUF headers, downloads, the model store and the memory fit are
//! `weirdnet-model`'s; this directory keeps MLX (`hub.rs`, which that crate does not read), the
//! quality-first quant choice (`plan.rs`), `node.toml` handling (`local.rs`) and the output.

mod ctxcmd;
mod hub;
mod local;
mod plan;
mod remote;

use std::io::{BufRead, IsTerminal, Write};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Args, Subcommand, ValueEnum};
use weirdnet_model::fit::Fit;
use weirdnet_model::hf::{RepoFile, RepoModel, Search, SearchHit};
use weirdnet_model::hw::{Hardware, human_gib};
use weirdnet_model::{Consent, ModelSpec, ModelStore, download_parts};
use ymir_core::{NodeConfig, Paths};

use hub::{Hit, MlxHub};
use plan::{Capacity, KvSource, Verdict};

#[derive(Subcommand, Debug)]
pub enum ModelsCmd {
    /// Search the Hugging Face Hub for GGUF, MLX and (where vLLM can run) safetensors models, with a fit verdict for this node.
    Search(SearchArgs),
    /// Download a model into this node's model directory and print its `[[model.local]]` entry.
    Pull(PullArgs),
    /// List the model files on this node and whether `node.toml` uses each.
    List(ListArgs),
    /// Delete a downloaded model's files (never edits `node.toml`).
    Remove(RemoveArgs),
    /// Show the context each model runs with on this node and why, or put models back on the
    /// automatic context (their full trained length, cut only where the KV cache would not fit).
    Context(ContextArgs),
}

#[derive(Args, Debug)]
pub struct ContextArgs {
    /// Model ids to put on the automatic context (with --auto).
    pub ids: Vec<String>,
    /// Remove `context` and any `--ctx-size` / `--max-model-len` in `args` from these models (a
    /// copy of node.toml is kept as node.toml.bak). Restart ymird to apply.
    #[arg(long)]
    pub auto: bool,
    /// Like --auto, for every model whose context is still one an older `ymirctl models pull`
    /// wrote by default: 32768, or 262144 for Inkling 2 Mini on vLLM.
    #[arg(long)]
    pub old_defaults: bool,
    #[command(flatten)]
    pub dir: DirArgs,
}

/// Run on the user's model nodes over Weirdnet instead of this machine.
#[derive(Args, Debug, Default, Clone)]
pub struct NodeArgs {
    /// A device name (`ymirctl pool list`), a peer id, `auto` (the node that fits best; pulls only) or
    /// `all`. Needs a running ymird on this machine, linked to your devices.
    #[arg(long)]
    pub node: Option<String>,
}

#[derive(Args, Debug)]
pub struct ListArgs {
    #[command(flatten)]
    pub dir: DirArgs,
    #[command(flatten)]
    pub node: NodeArgs,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Format {
    Gguf,
    Mlx,
    /// Safetensors / transformers repos, served with vLLM.
    Vllm,
    All,
}

#[derive(Args, Debug)]
pub struct SearchArgs {
    /// Words to search for, such as `qwen3 8b`.
    #[arg(required = true)]
    pub query: Vec<String>,
    /// Which kind of repo to show (default: GGUF, plus MLX on Apple Silicon and vLLM/safetensors on Linux NVIDIA).
    #[arg(long, value_enum)]
    pub format: Option<Format>,
    /// How many repos to show per format.
    #[arg(long, default_value_t = 5)]
    pub limit: usize,
    /// The context length to size the KV cache for when picking a quant (default 32768, capped at
    /// what the model was trained for).
    #[arg(long)]
    pub context: Option<u64>,
    #[command(flatten)]
    pub node: NodeArgs,
}

#[derive(Args, Debug)]
pub struct PullArgs {
    /// The Hub repo, `owner/name`.
    pub repo: String,
    /// The quant to take, such as `Q4_K_M` (default: the best that fits this node).
    #[arg(long, short)]
    pub quant: Option<String>,
    /// One exact file in the repo (a shard of a split GGUF takes the whole set).
    #[arg(long, short)]
    pub file: Option<String>,
    /// A fixed context for the entry, also used to pick the quant. Default: none in the entry, so
    /// ymird runs the model at its full trained length, cut only where the KV cache would not fit
    /// (the quant is picked at 32768).
    #[arg(long)]
    pub context: Option<u64>,
    /// The id for the entry (default: the repo name in lower case).
    #[arg(long)]
    pub id: Option<String>,
    /// The entry's role: `main`, `standin`, `embedding` or `chat` (default: `main` when node.toml has no main model, else `chat`).
    #[arg(long)]
    pub role: Option<String>,
    /// Append the entry to node.toml (a copy is kept as node.toml.bak). Without this the entry is only printed.
    #[arg(long)]
    pub register: bool,
    /// Show what would be downloaded and stop.
    #[arg(long)]
    pub dry_run: bool,
    /// Download the smallest quant even though none fits this node.
    #[arg(long)]
    pub force: bool,
    /// With --node: also serve it from the node's ymird right away, no restart (kept in memory only;
    /// `node.toml` is not edited on another node).
    #[arg(long)]
    pub serve: bool,
    /// With --node: serve it and load it now.
    #[arg(long)]
    pub load: bool,
    #[command(flatten)]
    pub dir: DirArgs,
    #[command(flatten)]
    pub node: NodeArgs,
}

#[derive(Args, Debug)]
pub struct DirArgs {
    /// The model directory (default: `models_dir` in node.toml, else the Weirdnet models directory).
    #[arg(long, env = "WEIRDNET_MODELS_DIR")]
    pub dir: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct RemoveArgs {
    /// A model id (from `ymirctl models list`) or a file name.
    pub id: String,
    /// Don't ask first.
    #[arg(long, short)]
    pub yes: bool,
    #[command(flatten)]
    pub dir: DirArgs,
    #[command(flatten)]
    pub node: NodeArgs,
}

/// Everything the commands read from the machine, so tests can supply their own.
pub struct Ctx {
    /// The model directory, the Hub to use and this node's token (see `ymir_core::models::store`).
    pub store: ModelStore,
    pub mlx: MlxHub,
    pub hw: Hardware,
    pub state: PathBuf,
    pub config: PathBuf,
    pub cfg: Option<NodeConfig>,
    pub tty: bool,
}

impl Ctx {
    fn dir(&self) -> &std::path::Path {
        self.store.dir()
    }

    fn hub(&self) -> weirdnet_model::Hub {
        self.store.hub_client()
    }
}

/// Runs a `models` command. With `--node` it goes to the model nodes through the local `ymird`.
pub async fn run(paths: &Paths, out: &mut dyn Write, input: &mut dyn BufRead, cmd: ModelsCmd) -> Result<()> {
    let node = match &cmd {
        ModelsCmd::Search(a) => a.node.node.clone(),
        ModelsCmd::Pull(a) => a.node.node.clone(),
        ModelsCmd::List(a) => a.node.node.clone(),
        ModelsCmd::Remove(a) => a.node.node.clone(),
        ModelsCmd::Context(_) => None,
    };
    if let Some(node) = node {
        return remote::run(paths, out, input, &node, cmd).await;
    }
    if let ModelsCmd::Pull(a) = &cmd
        && (a.serve || a.load)
    {
        bail!(
            "--serve and --load need --node (name this machine, or another one): serving a pulled model without a restart is done by ymird. Without --node, add the printed entry to node.toml, or use --register, and restart ymird"
        );
    }
    run_local(paths, out, input, cmd)
}

fn run_local(paths: &Paths, out: &mut dyn Write, input: &mut dyn BufRead, cmd: ModelsCmd) -> Result<()> {
    let cfg = NodeConfig::load(&paths.config()).ok();
    let dir_arg = match &cmd {
        ModelsCmd::Pull(a) => a.dir.dir.clone(),
        ModelsCmd::List(a) => a.dir.dir.clone(),
        ModelsCmd::Remove(a) => a.dir.dir.clone(),
        ModelsCmd::Context(a) => a.dir.dir.clone(),
        ModelsCmd::Search(_) => None,
    };
    let env_store = ModelStore::from_env();
    let dir = dir_arg.or_else(|| cfg.as_ref().map(NodeConfig::models_dir)).unwrap_or_else(|| env_store.dir().to_path_buf());
    let store = ymir_core::models::store(dir, &paths.state());
    let token = ymir_core::models::find_hf_token(&paths.state()).map(|(t, _)| t);
    let ctx = Ctx {
        mlx: MlxHub::new(store.hub(), token),
        store,
        hw: weirdnet_model::hw::detect(),
        state: paths.state(),
        config: paths.config(),
        cfg,
        tty: std::io::stdout().is_terminal(),
    };
    match cmd {
        ModelsCmd::Search(a) => search(&ctx, out, &a),
        ModelsCmd::Pull(a) => pull(&ctx, out, &a),
        ModelsCmd::List(_) => list(&ctx, out),
        ModelsCmd::Remove(a) => remove(&ctx, out, input, &a),
        ModelsCmd::Context(a) => ctxcmd::run(&ctx, out, &a),
    }
}

/// `812 MiB` under a GiB, else `12.4 GiB`.
fn gib(b: u64) -> String {
    if b < plan::GIB { format!("{} MiB", b.div_ceil(1 << 20)) } else { human_gib(b) }
}

fn count(n: u64) -> String {
    match n {
        0..1000 => n.to_string(),
        1000..1_000_000 => format!("{:.1}k", n as f64 / 1e3),
        _ => format!("{:.1}M", n as f64 / 1e6),
    }
}

fn is_mac_arm(hw: &Hardware) -> bool {
    hw.os == "macos" && matches!(hw.arch.as_str(), "aarch64" | "arm64")
}

/// Which Hub libraries to search: GGUF always by default, MLX on Apple Silicon, safetensors on
/// vLLM-capable nodes. `--format` overrides that.
fn search_libs(format: Option<Format>, hw: &Hardware) -> Vec<(&'static str, &'static str)> {
    match format {
        Some(Format::Gguf) => vec![("gguf", "GGUF")],
        Some(Format::Mlx) => vec![("mlx", "MLX")],
        Some(Format::Vllm) => vec![("safetensors", "vLLM")],
        Some(Format::All) => vec![("gguf", "GGUF"), ("mlx", "MLX"), ("safetensors", "vLLM")],
        None => {
            let mut v = vec![("gguf", "GGUF")];
            if is_mac_arm(hw) {
                v.push(("mlx", "MLX"));
            }
            if hw.vllm_supported() {
                v.push(("safetensors", "vLLM"));
            }
            v
        }
    }
}

/// What this node can say about one repo: the quant it would take and whether it fits.
struct Assessment {
    /// Complete models of the repo, smallest first.
    models: Vec<RepoModel>,
    chosen: usize,
    fit: Fit,
    context: u64,
    verdict: Verdict,
    /// The header's KV figure, for sizing other quants of the same repo.
    kv: Option<u64>,
}

/// Sizes every quant of a repo for this node and picks one. `None` when it has no GGUF files.
fn assess(ctx: &Ctx, repo: &str, requested_context: Option<u64>) -> Result<Option<Assessment>> {
    let models: Vec<RepoModel> = ctx.hub().models(repo)?.into_iter().filter(|m| m.complete).collect();
    let Some(small) = plan::smallest(&models) else { return Ok(None) };
    let info = ctx.hub().gguf_info(repo, &small.first().name).ok();
    let kv = info.as_ref().map(weirdnet_model::gguf::GgufInfo::kv_bytes_per_token).filter(|b| *b > 0);
    let context = plan::context_for(requested_context, info.as_ref().and_then(|i| i.context_length));
    let cap = plan::capacity(&ctx.hw);
    let pick = plan::pick(&models, kv, context, &cap).unwrap_or(small);
    let chosen = models.iter().position(|m| m == pick).unwrap_or(0);
    let fit = plan::measure(pick, kv, context);
    let verdict = plan::verdict(fit.footprint, &cap);
    Ok(Some(Assessment { models, chosen, fit, context, verdict, kv }))
}

fn search(ctx: &Ctx, out: &mut dyn Write, a: &SearchArgs) -> Result<()> {
    let query = a.query.join(" ");
    let libs = search_libs(a.format, &ctx.hw);
    let cap = plan::capacity(&ctx.hw);
    writeln!(
        out,
        "This node: {} usable for models ({}); sizes include the KV cache and overhead.",
        gib(cap.fast),
        if cap.accelerated { "accelerated" } else { "CPU" }
    )?;
    let mut shown = 0;
    for (lib, label) in libs {
        let limit = a.limit.clamp(1, 25);
        let hits: Vec<Hit> = if lib == "gguf" {
            ctx.hub().search(&Search::new(&query).with_limit(limit))?.into_iter().map(Hit::from).collect()
        } else {
            ctx.mlx.search(&query, lib, limit)?
        };
        for h in &hits {
            shown += 1;
            writeln!(out, "\n{}", describe(h, label))?;
            if lib == "gguf" {
                if let Err(e) = print_gguf(ctx, out, &h.id, a.context) {
                    writeln!(out, "    (could not read the files: {e})")?;
                }
            } else if lib == "mlx" {
                match ctx.mlx.repo(&h.id) {
                    Err(e) => writeln!(out, "    (could not read the files: {e})")?,
                    Ok(info) => print_mlx(ctx, out, &info.files, &cap)?,
                }
            } else if let Err(e) = print_vllm(ctx, out, &h.id, &cap) {
                writeln!(out, "    (could not read the files: {e})")?;
            }
        }
    }
    if shown == 0 {
        writeln!(out, "No repos matched `{query}`.")?;
    } else {
        writeln!(out, "\nDownload one with `ymirctl models pull <repo>`.")?;
    }
    Ok(())
}

fn describe(h: &Hit, label: &str) -> String {
    let params = h.params.map(|p| format!(", {:.1}B params", p as f64 / 1e9)).unwrap_or_default();
    let gated = if h.gated { ", gated (needs a token)" } else { "" };
    format!("{}  ({label}, {} downloads, {} likes{params}{gated})", h.id, count(h.downloads), count(h.likes))
}

impl From<SearchHit> for Hit {
    fn from(h: SearchHit) -> Self {
        Hit { id: h.id, downloads: h.downloads, likes: h.likes, gated: h.gated, params: None }
    }
}

fn print_gguf(ctx: &Ctx, out: &mut dyn Write, repo: &str, context: Option<u64>) -> Result<()> {
    let Some(a) = assess(ctx, repo, context)? else {
        return Ok(writeln!(out, "    no usable GGUF files in this repo")?);
    };
    let s = &a.models[a.chosen];
    let src = if a.fit.kv_source == KvSource::Header { "" } else { ", KV cache estimated" };
    let split = if s.parts.len() > 1 { format!(", {} files", s.parts.len()) } else { String::new() };
    writeln!(
        out,
        "    best: {}  {} on disk{split}, needs {} at {} context{src}: {}",
        plan::label(s),
        gib(s.size),
        gib(a.fit.footprint),
        a.context,
        a.verdict.label()
    )?;
    let others: Vec<String> = a.models.iter().filter(|o| *o != s).map(|o| format!("{} {}", plan::label(o), gib(o.size))).collect();
    if !others.is_empty() {
        let more = others.len().saturating_sub(8);
        let tail = if more > 0 { format!(" (+{more} more)") } else { String::new() };
        writeln!(out, "    other quants: {}{tail}", others[..others.len() - more].join(", "))?;
    }
    Ok(())
}

fn print_mlx(ctx: &Ctx, out: &mut dyn Write, files: &[RepoFile], cap: &Capacity) -> Result<()> {
    let size: u64 = files.iter().map(|f| f.size).sum();
    if !is_mac_arm(&ctx.hw) {
        return Ok(writeln!(out, "    {} on disk; MLX runs on Apple Silicon only, not on this node", gib(size))?);
    }
    let need = plan::footprint_of(size, None, plan::DEFAULT_CONTEXT);
    writeln!(out, "    {} on disk, needs about {} (KV cache estimated): {}", gib(size), gib(need), plan::verdict(need, cap).label())?;
    Ok(())
}

fn print_vllm(ctx: &Ctx, out: &mut dyn Write, repo: &str, cap: &Capacity) -> Result<()> {
    let info = ctx.mlx.repo(repo)?;
    let size: u64 = info.files.iter().filter(|f| f.name.ends_with(".safetensors") || f.name.ends_with(".json")).map(|f| f.size).sum();
    let ctxlen = plan::DEFAULT_CONTEXT;
    let need = plan::footprint_of(size, None, ctxlen);
    if !ctx.hw.vllm_supported() {
        writeln!(out, "    {} of weights; vLLM is not detected on this node. Pull still records the Hub id.", gib(size))?;
    } else {
        writeln!(
            out,
            "    {} of weights, needs about {} at {} context (vLLM fetches on first load): {}",
            gib(size),
            gib(need),
            ctxlen,
            plan::verdict(need, cap).label()
        )?;
    }
    if ymir_core::vllm::is_inkling(repo) {
        writeln!(
            out,
            "    Inkling 2 Mini: needs vLLM {}.{}.{}+ or the {} image (not a released vLLM yet).",
            ymir_core::vllm::INKLING_MIN_VERSION.0,
            ymir_core::vllm::INKLING_MIN_VERSION.1,
            ymir_core::vllm::INKLING_MIN_VERSION.2,
            ymir_core::vllm::INKLING_IMAGE
        )?;
    }
    Ok(())
}

/// Picks the files to pull for a GGUF repo from `--file`, `--quant` or the best fit.
fn choose<'a>(ctx: &Ctx, a: &PullArgs, repo: &str, asm: &'a Assessment) -> Result<&'a RepoModel> {
    let models = &asm.models;
    if let Some(f) = &a.file {
        return models
            .iter()
            .find(|m| m.parts.iter().any(|x| x.name == *f || x.name.rsplit('/').next() == Some(f.as_str())))
            .with_context(|| format!("`{f}` is not a complete GGUF in {repo} (shards of a split file must all be present)"));
    }
    if let Some(q) = &a.quant {
        let hits: Vec<&RepoModel> = models
            .iter()
            .filter(|m| {
                let l = plan::label(m);
                l.eq_ignore_ascii_case(q) || l.to_ascii_uppercase().strip_prefix("UD-").is_some_and(|l| l.eq_ignore_ascii_case(q))
            })
            .collect();
        return match hits.as_slice() {
            [one] => Ok(one),
            [] => bail!("{repo} has no `{q}` quant; it has: {}", models.iter().map(plan::label).collect::<Vec<_>>().join(", ")),
            _ => bail!("{repo} has several `{q}` files; pick one with --file"),
        };
    }
    let s = &models[asm.chosen];
    if asm.verdict != Verdict::Fits && !a.force {
        let cap = plan::capacity(&ctx.hw);
        bail!(
            "no quant of {repo} fits this node ({} usable): the smallest, {}, needs {} at {} context ({}). Try `--context` lower, `--quant` or `--file` to pick one yourself, or `--force` to take the smallest",
            gib(cap.fast),
            plan::label(s),
            gib(asm.fit.footprint),
            asm.context,
            asm.verdict.label()
        );
    }
    Ok(s)
}

struct Meter<'a> {
    out: &'a mut dyn Write,
    total: u64,
    tty: bool,
    last: u64,
}

impl Meter<'_> {
    fn tick(&mut self, name: &str, done: u64) {
        let pct = (done * 100).checked_div(self.total).unwrap_or(0);
        let step = if self.tty { 1 } else { 10 };
        if pct / step == self.last / step && done != 0 {
            return;
        }
        self.last = pct;
        let line = format!("{name}: {pct}% ({} of {})", gib(done), gib(self.total));
        let _ = if self.tty { write!(self.out, "\r{line}   ") } else { writeln!(self.out, "{line}") };
        let _ = self.out.flush();
    }
}

/// Downloads every file of `specs` (a split GGUF's shards, or an MLX repo's files) into `store`:
/// resumes, checks size and SHA-256, and says what happened.
fn download(ctx: &Ctx, out: &mut dyn Write, store: &ModelStore, name: &str, specs: &[ModelSpec]) -> Result<()> {
    let total: u64 = specs.iter().map(|s| s.size).sum();
    let dir = store.dir();
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let have: u64 = specs.iter().map(|s| store.partial_bytes(s)).sum();
    if let Some(free) = hub::free_bytes(dir)
        && free < total.saturating_sub(have) + (256 << 20)
    {
        bail!("not enough free space in {}: {} free, {} still to download", dir.display(), gib(free), gib(total.saturating_sub(have)));
    }
    let here = specs.iter().all(|s| store.installed(s));
    let mut meter = Meter { out: &mut *out, total, tty: ctx.tty, last: 0 };
    download_parts(store, specs, &Consent::given(&specs[0]), &mut |p| {
        meter.tick(name, p.done);
        true
    })?;
    if ctx.tty {
        writeln!(out)?;
    }
    let how = if here {
        "already here, verified"
    } else if specs.iter().any(|s| s.sha256.is_some()) {
        "downloaded, checksum verified"
    } else {
        "downloaded, size verified"
    };
    writeln!(out, "{name}: {how}")?;
    Ok(())
}

struct PullPlan {
    specs: Vec<ModelSpec>,
    store: ModelStore,
    main: PathBuf,
    backend: &'static str,
    id_default: String,
    context: u64,
    need: u64,
    label: String,
    mmproj: Option<String>,
}

fn pull(ctx: &Ctx, out: &mut dyn Write, a: &PullArgs) -> Result<()> {
    let ggufs = ctx.hub().models(&a.repo)?;
    let info = if ggufs.is_empty() { Some(ctx.mlx.repo(&a.repo)?) } else { ctx.mlx.repo(&a.repo).ok() };
    if ggufs.is_empty() {
        let info = info.as_ref().expect("repo metadata");
        if info.tags.iter().any(|t| t == "mlx") {
            if ymir_core::vllm::is_inkling(&a.repo) {
                bail!("{}", ymir_core::vllm::mlx_not_supported_message());
            }
        } else if plan::is_safetensors_repo(&info.files, &info.tags) {
            return pull_vllm(ctx, out, a, info);
        } else {
            bail!("{} has no GGUF files and is not an MLX or safetensors (vLLM) repo; `ymirctl models search` shows repos that work", a.repo);
        }
    }
    let cap = plan::capacity(&ctx.hw);
    let plan = if ggufs.is_empty() {
        let info = info.as_ref().expect("MLX repo");
        if !is_mac_arm(&ctx.hw) {
            writeln!(out, "Note: MLX runs on Apple Silicon only; this node can store the files but not serve them.")?;
        }
        let name = a.repo.rsplit('/').next().unwrap_or(&a.repo);
        let files: Vec<&RepoFile> = info.files.iter().filter(|f| !f.name.starts_with('.')).collect();
        let size: u64 = files.iter().map(|f| f.size).sum();
        let ctxlen = plan::context_for(a.context, None);
        let need = plan::footprint_of(size, None, ctxlen);
        let dest = ctx.dir().join("mlx").join(name);
        let store = ymir_core::models::store(&dest, &ctx.state);
        // Follow this node's Hub (the one the repo was read from), not only the environment's.
        let store = if store.hub() == ctx.store.hub() { store } else { ModelStore::new(&dest).with_hub(ctx.store.hub()) };
        let specs = files
            .iter()
            .map(|f| ModelSpec {
                id: name.to_string(),
                label: name.to_string(),
                repo: a.repo.clone(),
                file: f.name.clone(),
                size: f.size,
                sha256: f.sha256.clone(),
            })
            .collect();
        PullPlan {
            specs,
            store,
            main: dest.clone(),
            backend: "mlx",
            id_default: name.to_ascii_lowercase(),
            context: ctxlen,
            need,
            label: "MLX weights".to_string(),
            mmproj: None,
        }
    } else {
        let asm = assess(ctx, &a.repo, a.context)?.context("no usable GGUF files in this repo")?;
        let model = choose(ctx, a, &a.repo, &asm)?;
        let need = plan::measure(model, asm.kv, asm.context).footprint;
        let mut specs = model.specs(&a.repo)?;
        let mmproj = info.as_ref().and_then(|i| plan::pick_mmproj(&i.files)).cloned();
        if let Some(p) = &mmproj {
            // Same id as the weights so download_parts accepts one Consent (it rejects mixed ids).
            let id = specs.first().map(|s| s.id.clone()).unwrap_or_else(|| plan::id_for_repo(&a.repo));
            specs.push(ModelSpec { id, label: p.name.clone(), repo: a.repo.clone(), file: p.name.clone(), size: p.size, sha256: p.sha256.clone() });
        }
        PullPlan {
            specs,
            store: ctx.store.clone(),
            main: ctx.dir().join(&model.first().name),
            backend: "llamacpp",
            id_default: plan::id_for_repo(&a.repo),
            context: asm.context,
            need,
            label: plan::label(model),
            mmproj: mmproj.map(|p| p.name.rsplit('/').next().unwrap_or(&p.name).to_string()),
        }
    };
    let total: u64 = plan.specs.iter().map(|s| s.size).sum();
    writeln!(
        out,
        "{}: {}, {} file(s), {} to download, needs {} at {} context: {}",
        a.repo,
        plan.label,
        plan.specs.len(),
        gib(total),
        gib(plan.need),
        plan.context,
        plan::verdict(plan.need, &cap).label()
    )?;
    if plan::verdict(plan.need, &cap) != Verdict::Fits {
        writeln!(out, "Warning: this will not fit entirely in this node's {} of model memory.", gib(cap.fast))?;
    }
    if a.dry_run {
        for s in &plan.specs {
            writeln!(out, "  {}  {}", gib(s.size), s.file)?;
        }
        return Ok(writeln!(out, "Dry run: nothing was downloaded.")?);
    }
    let name = plan.specs[0].id.clone();
    download(ctx, out, &plan.store, &name, &plan.specs)?;
    let role = a.role.clone().unwrap_or_else(|| {
        if a.repo.to_ascii_lowercase().contains("embed") {
            "embedding".into()
        } else if ctx.cfg.as_ref().is_some_and(|c| c.main_model().is_some()) {
            "chat".into()
        } else {
            "main".into()
        }
    });
    let id = a.id.clone().unwrap_or(plan.id_default);
    // MLX-LM takes the context as its answer-length limit, so MLX keeps one; llama.cpp entries get
    // a context only when asked for, and ymird sizes the rest.
    let fixed = if plan.backend == "mlx" { Some(plan.context) } else { a.context };
    let mut entry = plan::local_model(&id, plan.backend, &role, &plan.main.to_string_lossy(), fixed, cap.accelerated);
    if let Some(proj) = plan.mmproj {
        entry.mmproj = Some(proj);
    }
    let stanza = plan::stanza(&entry);
    writeln!(out, "\nnode.toml entry:\n{stanza}")?;
    if let Some(p) = ctxcmd::planned(ctx, &entry) {
        writeln!(out, "Context on this node: {}", p.note)?;
    }
    if a.register {
        local::register(&ctx.config, &id, &stanza)?;
        writeln!(out, "Added `{id}` to {} (the old file is {}.bak). Restart ymird to load it.", ctx.config.display(), ctx.config.display())?;
    } else {
        writeln!(out, "node.toml was not changed. Add the entry above and restart ymird, or pull again with --register.")?;
    }
    Ok(())
}

fn pull_vllm(ctx: &Ctx, out: &mut dyn Write, a: &PullArgs, info: &hub::RepoInfo) -> Result<()> {
    if a.quant.is_some() || a.file.is_some() {
        bail!(
            "{} is a safetensors repo for vLLM; --quant and --file apply to GGUF repos. Pull it without those flags to register the Hub id",
            a.repo
        );
    }
    let cap = plan::capacity(&ctx.hw);
    let size: u64 = info.files.iter().filter(|f| f.name.ends_with(".safetensors")).map(|f| f.size).sum();
    let inkling = ymir_core::vllm::is_inkling(&a.repo);
    // The model's config.json sizes its KV cache; ymird reads the copy saved here (vLLM fetches
    // the weights itself on first load).
    let config = ctx.mlx.file(&a.repo, "config.json").ok();
    let shape = config.as_deref().and_then(|c| serde_json::from_slice(c).ok()).and_then(|v| ymir_core::kvfit::config_shape(&v).ok());
    let context = plan::context_for(a.context, shape.as_ref().and_then(|s| s.trained));
    let need = match &shape {
        Some(s) => ymir_core::kvfit::footprint(size, s.kv_bytes(context, 1, ymir_core::kvfit::CacheType::F16)),
        None => plan::footprint_of(size, None, context),
    };
    writeln!(
        out,
        "{}: vLLM Hub id, {} of weights (fetched into the Hugging Face cache on first load), needs about {} at {} context: {}",
        a.repo,
        gib(size),
        gib(need),
        context,
        plan::verdict(need, &cap).label()
    )?;
    if !ctx.hw.vllm_supported() {
        writeln!(
            out,
            "Note: vLLM is not detected on this hardware; the entry is still written. Serve it on a Linux NVIDIA node, or wrap a container."
        )?;
    }
    if plan::verdict(need, &cap) != Verdict::Fits {
        writeln!(out, "Warning: this will not fit entirely in this node's {} of model memory.", gib(cap.fast))?;
    }
    if inkling {
        writeln!(
            out,
            "Inkling 2 Mini: needs vLLM {}.{}.{}+ or image `{}`. A wrap path is in the docs if you start the image yourself.",
            ymir_core::vllm::INKLING_MIN_VERSION.0,
            ymir_core::vllm::INKLING_MIN_VERSION.1,
            ymir_core::vllm::INKLING_MIN_VERSION.2,
            ymir_core::vllm::INKLING_IMAGE
        )?;
    }
    let role =
        a.role.clone().unwrap_or_else(|| if ctx.cfg.as_ref().is_some_and(|c| c.main_model().is_some()) { "chat".into() } else { "main".into() });
    let id = a.id.clone().unwrap_or_else(|| plan::id_for_repo(&a.repo));
    let mut entry = plan::local_model(&id, "vllm", &role, &a.repo, a.context, cap.accelerated);
    if inkling {
        entry.args = ymir_core::vllm::inkling_args();
        entry.image = Some(ymir_core::vllm::INKLING_IMAGE.into());
    }
    let stanza = plan::stanza(&entry);
    if a.dry_run {
        writeln!(out, "\nnode.toml entry:\n{stanza}")?;
        return Ok(writeln!(out, "Dry run: nothing was downloaded.")?);
    }
    match &config {
        Some(c) => ymir_core::kvfit::save_vllm_meta(ctx.dir(), &a.repo, c, size)?,
        None => writeln!(
            out,
            "Note: could not read {}'s config.json; ymird will run it at {} context until it can",
            a.repo,
            ymir_core::kvfit::FALLBACK_CONTEXT
        )?,
    }
    writeln!(out, "\nnode.toml entry:\n{stanza}")?;
    if let Some(p) = ctxcmd::planned(ctx, &entry) {
        writeln!(out, "Context on this node: {}", p.note)?;
    }
    if a.register {
        local::register(&ctx.config, &id, &stanza)?;
        writeln!(out, "Added `{id}` to {} (the old file is {}.bak). Restart ymird to load it.", ctx.config.display(), ctx.config.display())?;
    } else {
        writeln!(out, "node.toml was not changed. Add the entry above and restart ymird, or pull again with --register.")?;
    }
    Ok(())
}

fn list(ctx: &Ctx, out: &mut dyn Write) -> Result<()> {
    let entries = local::scan(&ctx.store, ctx.cfg.as_ref());
    writeln!(out, "Models in {}:", ctx.dir().display())?;
    if entries.is_empty() {
        return Ok(writeln!(out, "  none downloaded (try `ymirctl models search`)")?);
    }
    for e in &entries {
        let state = match (&e.registered, e.partial) {
            (_, Some(n)) => format!("unfinished download ({} so far; run pull again to resume)", gib(n)),
            (Some(r), _) => format!("registered as `{r}`"),
            (None, _) if ctx.cfg.is_none() => "no node.toml".to_string(),
            (None, _) => "not registered".to_string(),
        };
        let kind = if e.kind == local::Kind::Mlx { "mlx" } else { "gguf" };
        let shards = if e.paths.len() > 1 { format!(" ({} files)", e.paths.len()) } else { String::new() };
        writeln!(out, "  {:<40} {kind:<5} {:>10}{shards}  {state}", e.id, gib(e.size))?;
    }
    Ok(())
}

fn remove(ctx: &Ctx, out: &mut dyn Write, input: &mut dyn BufRead, a: &RemoveArgs) -> Result<()> {
    let entries = local::scan(&ctx.store, ctx.cfg.as_ref());
    let e = local::find(&entries, &a.id)?;
    writeln!(out, "Delete {} ({}, {} file(s)) from {}?", e.id, gib(e.size), e.paths.len(), ctx.dir().display())?;
    for p in &e.paths {
        writeln!(out, "  {}", p.display())?;
    }
    if let Some(r) = &e.registered {
        writeln!(out, "Note: node.toml still lists `{r}`; this only deletes files, so remove that entry yourself or ymird will fail to load it.")?;
    }
    if !a.yes {
        write!(out, "Type `yes` to delete: ")?;
        out.flush()?;
        let mut line = String::new();
        input.read_line(&mut line)?;
        if !matches!(line.trim(), "y" | "yes") {
            return Ok(writeln!(out, "Nothing deleted.")?);
        }
    }
    local::remove(&ctx.store, e)?;
    writeln!(out, "Deleted {}.", e.id)?;
    Ok(())
}

#[cfg(test)]
mod tests;
