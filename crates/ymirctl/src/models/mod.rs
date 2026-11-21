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
    /// Search the Hugging Face Hub for GGUF and MLX models, with a fit verdict for this node.
    Search(SearchArgs),
    /// Download a model into this node's model directory and print its `[[model.local]]` entry.
    Pull(PullArgs),
    /// List the model files on this node and whether `node.toml` uses each.
    List(ListArgs),
    /// Delete a downloaded model's files (never edits `node.toml`).
    Remove(RemoveArgs),
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
    All,
}

#[derive(Args, Debug)]
pub struct SearchArgs {
    /// Words to search for, such as `qwen3 8b`.
    #[arg(required = true)]
    pub query: Vec<String>,
    /// Which kind of repo to show (default: GGUF, plus MLX on Apple Silicon).
    #[arg(long, value_enum)]
    pub format: Option<Format>,
    /// How many repos to show per format.
    #[arg(long, default_value_t = 5)]
    pub limit: usize,
    /// The context length to size the KV cache for (default 32768, capped at what the model was trained for).
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
    /// The context length to size for and to put in the entry (default 32768, capped at the trained length).
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
    let format = a.format.unwrap_or(if is_mac_arm(&ctx.hw) { Format::All } else { Format::Gguf });
    let cap = plan::capacity(&ctx.hw);
    writeln!(
        out,
        "This node: {} usable for models ({}); sizes include the KV cache and overhead.",
        gib(cap.fast),
        if cap.accelerated { "accelerated" } else { "CPU" }
    )?;
    let mut shown = 0;
    for (lib, label) in [("gguf", "GGUF"), ("mlx", "MLX")] {
        if !(format == Format::All || (format == Format::Gguf) == (lib == "gguf")) {
            continue;
        }
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
            } else {
                match ctx.mlx.repo(&h.id) {
                    Err(e) => writeln!(out, "    (could not read the files: {e})")?,
                    Ok(info) => print_mlx(ctx, out, &info.files, &cap)?,
                }
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

fn pull(ctx: &Ctx, out: &mut dyn Write, a: &PullArgs) -> Result<()> {
    let ggufs = ctx.hub().models(&a.repo)?;
    let mlx = if ggufs.is_empty() {
        let info = ctx.mlx.repo(&a.repo)?;
        if !info.tags.iter().any(|t| t == "mlx") {
            bail!("{} has no GGUF files and is not an MLX repo; `ymirctl models search` shows repos that work", a.repo);
        }
        Some(info)
    } else {
        None
    };
    let cap = plan::capacity(&ctx.hw);
    // (files to download, store to download into, backend, default id, context, memory needed, what to call it)
    let (specs, store, main, backend, id_default, context, need, label): (Vec<ModelSpec>, ModelStore, PathBuf, &str, String, u64, u64, String) =
        if let Some(info) = &mlx {
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
            (specs, store, dest.clone(), "mlx", name.to_ascii_lowercase(), ctxlen, need, "MLX weights".to_string())
        } else {
            let asm = assess(ctx, &a.repo, a.context)?.context("no usable GGUF files in this repo")?;
            let model = choose(ctx, a, &a.repo, &asm)?;
            let need = plan::measure(model, asm.kv, asm.context).footprint;
            let specs = model.specs(&a.repo)?;
            (
                specs,
                ctx.store.clone(),
                ctx.dir().join(&model.first().name),
                "llamacpp",
                plan::id_for_repo(&a.repo),
                asm.context,
                need,
                plan::label(model),
            )
        };
    let total: u64 = specs.iter().map(|s| s.size).sum();
    writeln!(
        out,
        "{}: {label}, {} file(s), {} to download, needs {} at {} context: {}",
        a.repo,
        specs.len(),
        gib(total),
        gib(need),
        context,
        plan::verdict(need, &cap).label()
    )?;
    if plan::verdict(need, &cap) != Verdict::Fits {
        writeln!(out, "Warning: this will not fit entirely in this node's {} of model memory.", gib(cap.fast))?;
    }
    if a.dry_run {
        for s in &specs {
            writeln!(out, "  {}  {}", gib(s.size), s.file)?;
        }
        return Ok(writeln!(out, "Dry run: nothing was downloaded.")?);
    }
    let name = specs[0].id.clone();
    download(ctx, out, &store, &name, &specs)?;
    let role = a.role.clone().unwrap_or_else(|| {
        if a.repo.to_ascii_lowercase().contains("embed") {
            "embedding".into()
        } else if ctx.cfg.as_ref().is_some_and(|c| c.main_model().is_some()) {
            "chat".into()
        } else {
            "main".into()
        }
    });
    let id = a.id.clone().unwrap_or(id_default);
    let entry = plan::local_model(&id, backend, &role, &main.to_string_lossy(), context, cap.accelerated);
    let stanza = plan::stanza(&entry);
    writeln!(out, "\nnode.toml entry:\n{stanza}")?;
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
