//! `ymirctl models search | pull | list | remove`: find models on the Hugging Face Hub, download
//! one that fits this node, and see or delete what is on disk. All of it is local to the node it
//! runs on; it never edits `node.toml` unless `pull --register` says so.
//!
//! The Hub client, GGUF header reader and quant chooser live in this directory and are meant to be
//! replaced by `weirdnet-model` once that crate has them (see `hub.rs`).

mod gguf;
mod hub;
mod local;
mod plan;

use std::io::{BufRead, IsTerminal, Write};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Args, Subcommand, ValueEnum};
use weirdnet_model::hw::{Hardware, human_gib};
use ymir_core::{NodeConfig, Paths};

use hub::{Hit, Hub, RepoInfo};
use plan::{Capacity, KvSource, QuantSet, RepoFile, Verdict};

#[derive(Subcommand, Debug)]
pub enum ModelsCmd {
    /// Search the Hugging Face Hub for GGUF and MLX models, with a fit verdict for this node.
    Search(SearchArgs),
    /// Download a model into this node's model directory and print its `[[model.local]]` entry.
    Pull(PullArgs),
    /// List the model files on this node and whether `node.toml` uses each.
    List(DirArgs),
    /// Delete a downloaded model's files (never edits `node.toml`).
    Remove(RemoveArgs),
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
    #[command(flatten)]
    pub dir: DirArgs,
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
}

/// Everything the commands read from the machine, so tests can supply their own.
pub struct Ctx {
    pub hub: Hub,
    pub hw: Hardware,
    pub dir: PathBuf,
    pub config: PathBuf,
    pub cfg: Option<NodeConfig>,
    pub tty: bool,
}

pub fn run(paths: &Paths, out: &mut dyn Write, input: &mut dyn BufRead, cmd: ModelsCmd) -> Result<()> {
    let cfg = NodeConfig::load(&paths.config()).ok();
    let dir_arg = match &cmd {
        ModelsCmd::Pull(a) => a.dir.dir.clone(),
        ModelsCmd::List(a) => a.dir.clone(),
        ModelsCmd::Remove(a) => a.dir.dir.clone(),
        ModelsCmd::Search(_) => None,
    };
    let store = weirdnet_model::ModelStore::from_env();
    let token = hub::find_token(&paths.state());
    let ctx = Ctx {
        hub: Hub::new(store.hub(), token.map(|(t, _)| t)),
        hw: weirdnet_model::hw::detect(),
        dir: dir_arg.or_else(|| cfg.as_ref().map(NodeConfig::models_dir)).unwrap_or_else(|| store.dir().to_path_buf()),
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
    sets: Vec<QuantSet>,
    chosen: Option<usize>,
    need: u64,
    context: u64,
    kv_source: KvSource,
    verdict: Verdict,
}

/// Sizes every quant of a repo for this node and picks one. `None` when it has no GGUF files.
fn assess(ctx: &Ctx, info: &RepoInfo, requested_context: Option<u64>) -> Option<Assessment> {
    let sets = plan::quant_sets(&info.files);
    let small = plan::smallest(&sets)?;
    let meta = ctx.hub.gguf_meta(&info.id, &small.files[0].name);
    let (kv, kv_source) = plan::kv_per_token(meta.as_ref().and_then(gguf::GgufMeta::kv_bytes_per_token), small.size);
    let context = plan::context_for(requested_context, meta.as_ref().map(|m| m.context_train));
    let cap: Capacity = plan::capacity(&ctx.hw);
    let pick = plan::pick(&sets, kv, context, &cap).or(Some(small))?;
    let idx = sets.iter().position(|s| s == pick)?;
    let need = plan::need(pick.size, kv, context);
    let verdict = plan::verdict(need, &cap);
    Some(Assessment { sets, chosen: Some(idx), need, context, kv_source, verdict })
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
        let hits = ctx.hub.search(&query, lib, a.limit.clamp(1, 25))?;
        for h in &hits {
            shown += 1;
            writeln!(out, "\n{}", describe(h, label))?;
            match ctx.hub.repo(&h.id) {
                Err(e) => writeln!(out, "    (could not read the files: {e})")?,
                Ok(info) if lib == "gguf" => print_gguf(ctx, out, &info, a.context)?,
                Ok(info) => print_mlx(ctx, out, &info, &cap)?,
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

fn print_gguf(ctx: &Ctx, out: &mut dyn Write, info: &RepoInfo, context: Option<u64>) -> Result<()> {
    let Some(a) = assess(ctx, info, context) else {
        return Ok(writeln!(out, "    no usable GGUF files in this repo")?);
    };
    let s = &a.sets[a.chosen.unwrap_or(0)];
    let src = if a.kv_source == KvSource::Header { "" } else { ", KV cache estimated" };
    let split = if s.files.len() > 1 { format!(", {} files", s.files.len()) } else { String::new() };
    writeln!(
        out,
        "    best: {}  {} on disk{split}, needs {} at {} context{src}: {}",
        s.label,
        gib(s.size),
        gib(a.need),
        a.context,
        a.verdict.label()
    )?;
    let others: Vec<String> = a.sets.iter().filter(|o| *o != s).map(|o| format!("{} {}", o.label, gib(o.size))).collect();
    if !others.is_empty() {
        let more = others.len().saturating_sub(8);
        let tail = if more > 0 { format!(" (+{more} more)") } else { String::new() };
        writeln!(out, "    other quants: {}{tail}", others[..others.len() - more].join(", "))?;
    }
    Ok(())
}

fn print_mlx(ctx: &Ctx, out: &mut dyn Write, info: &RepoInfo, cap: &Capacity) -> Result<()> {
    let size: u64 = info.files.iter().map(|f| f.size).sum();
    if !is_mac_arm(&ctx.hw) {
        return Ok(writeln!(out, "    {} on disk; MLX runs on Apple Silicon only, not on this node", gib(size))?);
    }
    let (kv, _) = plan::kv_per_token(None, size);
    let need = plan::need(size, kv, plan::DEFAULT_CONTEXT);
    writeln!(out, "    {} on disk, needs about {} (KV cache estimated): {}", gib(size), gib(need), plan::verdict(need, cap).label())?;
    Ok(())
}

/// Picks the files to pull for a GGUF repo from `--file`, `--quant` or the best fit.
fn choose<'a>(ctx: &Ctx, a: &PullArgs, info: &RepoInfo, asm: &'a Assessment) -> Result<&'a QuantSet> {
    let sets = &asm.sets;
    if let Some(f) = &a.file {
        return sets
            .iter()
            .find(|s| s.files.iter().any(|x| x.name == *f || x.name.rsplit('/').next() == Some(f.as_str())))
            .with_context(|| format!("`{f}` is not a complete GGUF in {} (shards of a split file must all be present)", info.id));
    }
    if let Some(q) = &a.quant {
        let hits: Vec<&QuantSet> = sets
            .iter()
            .filter(|s| {
                s.label.eq_ignore_ascii_case(q) || s.label.to_ascii_uppercase().strip_prefix("UD-").is_some_and(|l| l.eq_ignore_ascii_case(q))
            })
            .collect();
        return match hits.as_slice() {
            [one] => Ok(one),
            [] => bail!("{} has no `{q}` quant; it has: {}", info.id, sets.iter().map(|s| s.label.as_str()).collect::<Vec<_>>().join(", ")),
            _ => bail!("{} has several `{q}` files; pick one with --file", info.id),
        };
    }
    let s = &sets[asm.chosen.unwrap_or(0)];
    if asm.verdict != Verdict::Fits && !a.force {
        let cap = plan::capacity(&ctx.hw);
        bail!(
            "no quant of {} fits this node ({} usable): the smallest, {}, needs {} at {} context ({}). Try `--context` lower, `--quant` or `--file` to pick one yourself, or `--force` to take the smallest",
            info.id,
            gib(cap.fast),
            s.label,
            gib(asm.need),
            asm.context,
            asm.verdict.label()
        );
    }
    Ok(s)
}

struct Meter<'a> {
    out: &'a mut dyn Write,
    total: u64,
    base: u64,
    tty: bool,
    last: u64,
}

impl Meter<'_> {
    fn tick(&mut self, name: &str, done: u64) {
        let all = self.base + done;
        let pct = (all * 100).checked_div(self.total).unwrap_or(0);
        let step = if self.tty { 1 } else { 10 };
        if pct / step == self.last / step && done != 0 {
            return;
        }
        self.last = pct;
        let line = format!("{name}: {pct}% ({} of {})", gib(all), gib(self.total));
        let _ = if self.tty { write!(self.out, "\r{line}   ") } else { writeln!(self.out, "{line}") };
        let _ = self.out.flush();
    }
}

fn pull(ctx: &Ctx, out: &mut dyn Write, a: &PullArgs) -> Result<()> {
    let info = ctx.hub.repo(&a.repo)?;
    let has_gguf = info.files.iter().any(|f| f.name.to_ascii_lowercase().ends_with(".gguf"));
    let mlx = !has_gguf && info.tags.iter().any(|t| t == "mlx");
    if !has_gguf && !mlx {
        bail!("{} has no GGUF files and is not an MLX repo; `ymirctl models search` shows repos that work", a.repo);
    }
    let cap = plan::capacity(&ctx.hw);
    let (files, dest_dir, backend, id_default, context, need): (Vec<RepoFile>, PathBuf, &str, String, u64, u64) = if mlx {
        if !is_mac_arm(&ctx.hw) {
            writeln!(out, "Note: MLX runs on Apple Silicon only; this node can store the files but not serve them.")?;
        }
        let name = a.repo.rsplit('/').next().unwrap_or(&a.repo);
        let files: Vec<RepoFile> = info.files.iter().filter(|f| !f.name.starts_with('.')).cloned().collect();
        let size: u64 = files.iter().map(|f| f.size).sum();
        let ctxlen = plan::context_for(a.context, None);
        let need = plan::need(size, plan::kv_per_token(None, size).0, ctxlen);
        (files, ctx.dir.join("mlx").join(name), "mlx", name.to_ascii_lowercase(), ctxlen, need)
    } else {
        let asm = assess(ctx, &info, a.context).context("no usable GGUF files in this repo")?;
        let set = choose(ctx, a, &info, &asm)?;
        let weights = set.size;
        let (kv, _) = plan::kv_per_token(ctx.hub.gguf_meta(&info.id, &set.files[0].name).and_then(|m| m.kv_bytes_per_token()), weights);
        let need = plan::need(weights, kv, asm.context);
        (set.files.clone(), ctx.dir.clone(), "llamacpp", plan::id_for_repo(&a.repo), asm.context, need)
    };
    let total: u64 = files.iter().map(|f| f.size).sum();
    let label = if mlx { "MLX weights".to_string() } else { plan::quant_of(&files[0].name).unwrap_or_else(|| files[0].name.clone()) };
    writeln!(
        out,
        "{}: {label}, {} file(s), {} to download, needs {} at {} context: {}",
        a.repo,
        files.len(),
        gib(total),
        gib(need),
        context,
        plan::verdict(need, &cap).label()
    )?;
    if plan::verdict(need, &cap) != Verdict::Fits {
        writeln!(out, "Warning: this will not fit entirely in this node's {} of model memory.", gib(cap.fast))?;
    }
    if a.dry_run {
        for f in &files {
            writeln!(out, "  {}  {}", gib(f.size), f.name)?;
        }
        return Ok(writeln!(out, "Dry run: nothing was downloaded.")?);
    }
    std::fs::create_dir_all(&dest_dir).with_context(|| format!("creating {}", dest_dir.display()))?;
    let target = |f: &RepoFile| if mlx { dest_dir.join(&f.name) } else { dest_dir.join(f.name.rsplit('/').next().unwrap_or(&f.name)) };
    let have: u64 = files.iter().map(|f| std::fs::metadata(format!("{}.part", target(f).display())).map_or(0, |m| m.len())).sum();
    if let Some(free) = hub::free_bytes(&dest_dir)
        && free < total.saturating_sub(have) + (256 << 20)
    {
        bail!("not enough free space in {}: {} free, {} still to download", dest_dir.display(), gib(free), gib(total.saturating_sub(have)));
    }
    let mut base = 0;
    for f in &files {
        let dest = target(f);
        let name = dest.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let mut meter = Meter { out: &mut *out, total, base, tty: ctx.tty, last: 0 };
        let fresh = ctx.hub.download(&ctx.hub.file_url(&a.repo, &f.name), &dest, f.size, f.sha256.as_deref(), &mut |d, _| meter.tick(&name, d))?;
        if ctx.tty {
            writeln!(out)?;
        }
        writeln!(
            out,
            "{name}: {}",
            if fresh {
                if f.sha256.is_some() { "downloaded, checksum verified" } else { "downloaded, size verified" }
            } else {
                "already here, verified"
            }
        )?;
        base += f.size;
    }
    let main = if mlx { dest_dir.clone() } else { target(&files[0]) };
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
    let entries = local::scan(&ctx.dir, ctx.cfg.as_ref());
    writeln!(out, "Models in {}:", ctx.dir.display())?;
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
    let entries = local::scan(&ctx.dir, ctx.cfg.as_ref());
    let e = local::find(&entries, &a.id)?;
    writeln!(out, "Delete {} ({}, {} file(s)) from {}?", e.id, gib(e.size), e.paths.len(), ctx.dir.display())?;
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
    local::remove(&ctx.dir, e)?;
    writeln!(out, "Deleted {}.", e.id)?;
    Ok(())
}

#[cfg(test)]
mod tests;
