//! `ymirctl models search | pull | list | remove --node <name|peer-id|auto|all>`: the same commands on
//! the user's model nodes over Weirdnet.
//!
//! `ymirctl` has no network identity of its own: it asks the `ymird` on this machine (control
//! socket), which dials the nodes with `weirdnet-backends`' `ModelsClient` presenting this device's
//! certificate. A node answers only devices linked to its owner. A pull's progress streams back as it
//! downloads. The Hub token is read on the node that downloads and is never sent.

use std::io::{BufRead, IsTerminal, Write};

use anyhow::{Context, Result, bail};
use weirdnet_backends::models::{ModelsReply, Pull, RepoFits};
use ymir_core::Paths;
use ymir_core::control;
use ymir_core::models::{Chosen, ModelsDone, ModelsLine, ModelsOp, NodeResult, Outcome};

use super::adapters::AdapterCmd;
use super::{Format, ModelsCmd, PullArgs, RemoveArgs, SearchArgs, gib, plan};

pub async fn run(paths: &Paths, out: &mut dyn Write, input: &mut dyn BufRead, node: &str, cmd: ModelsCmd) -> Result<()> {
    match cmd {
        ModelsCmd::Search(a) => search(paths, out, node, &a).await,
        ModelsCmd::Pull(a) => pull(paths, out, node, &a).await,
        ModelsCmd::List(_) => list(paths, out, node).await,
        ModelsCmd::Remove(a) => remove(paths, out, input, node, &a).await,
        ModelsCmd::Context(_) => {
            bail!("`ymirctl models context` edits this machine's node.toml; run it on the node itself (without --node)")
        }
        ModelsCmd::Adapter { cmd } => adapter(paths, out, node, cmd).await,
        ModelsCmd::Rollout { .. } => bail!("rollouts are this inference node's pool policy; run `ymirctl models rollout` on it (without --node)"),
        ModelsCmd::Upload(_) => bail!("uploading over Weirdnet is not built yet; run `ymirctl models upload` on the node itself"),
    }
}

/// Prints a pull's progress lines: every 10% when piped, in place on a terminal.
struct Progress {
    tty: bool,
    last: std::collections::HashMap<String, u64>,
    open: bool,
}

impl Progress {
    fn new() -> Self {
        Self { tty: std::io::stdout().is_terminal(), last: Default::default(), open: false }
    }

    fn line(&mut self, out: &mut dyn Write, node: &str, done: u64, total: u64) {
        let pct = (done * 100).checked_div(total).unwrap_or(0);
        let step = if self.tty { 1 } else { 10 };
        let prev = self.last.insert(node.to_string(), pct);
        if prev.is_some_and(|p| p / step == pct / step) && done != total {
            return;
        }
        let line = format!("{node}: {pct}% ({} of {})", gib(done), gib(total));
        let _ = if self.tty { write!(out, "\r{line}   ") } else { writeln!(out, "{line}") };
        self.open = self.tty;
        let _ = out.flush();
    }

    fn finish(&mut self, out: &mut dyn Write) {
        if std::mem::take(&mut self.open) {
            let _ = writeln!(out);
        }
    }
}

async fn ask(paths: &Paths, out: &mut dyn Write, node: &str, op: ModelsOp) -> Result<ModelsDone> {
    let mut progress = Progress::new();
    let done = control::request_models(&paths.socket(), node, op, |l| {
        if let ModelsLine::Progress { node, done, total } = l {
            progress.line(out, node, *done, *total);
        }
    })
    .await;
    progress.finish(out);
    done.context("asking the pool (this needs a running ymird on this machine, linked to your devices: `ymirctl status`)")
}

/// The replies of the nodes that answered; the ones that did not are printed and counted.
fn answers<'a>(out: &mut dyn Write, done: &'a ModelsDone) -> Result<(Vec<(&'a str, &'a ModelsReply)>, usize)> {
    let mut ok = Vec::new();
    let mut failed = 0;
    for NodeResult { node, outcome } in &done.nodes {
        match outcome {
            Outcome::Ok { reply } => ok.push((node.label.as_str(), &**reply)),
            Outcome::Failed { message } => {
                failed += 1;
                writeln!(out, "{}: {message}", node.label)?;
            }
        }
    }
    Ok((ok, failed))
}

fn finish(failed: usize, total: usize) -> Result<()> {
    match failed {
        0 => Ok(()),
        _ if total == 1 => bail!("the node could not do it (see above)"),
        n => bail!("{n} of {total} nodes could not do it (see above)"),
    }
}

fn count(n: u64) -> String {
    match n {
        0..1000 => n.to_string(),
        1000..1_000_000 => format!("{:.1}k", n as f64 / 1e3),
        _ => format!("{:.1}M", n as f64 / 1e6),
    }
}

/// What a node says about one repo, in the words `search` uses locally.
fn verdict_line(f: &RepoFits) -> String {
    let usable = f.budget.map_or(0, |b| b.usable_bytes);
    let src = |k: weirdnet_model::fit::KvSource| if k == weirdnet_model::fit::KvSource::Header { "" } else { ", KV cache estimated" };
    if let Some(fit) = f.best.as_deref().and_then(|b| f.fits.iter().find(|x| x.model.id == b)) {
        let split = if fit.model.parts.len() > 1 { format!(", {} files", fit.model.parts.len()) } else { String::new() };
        return format!(
            "best: {}  {} on disk{split}, needs {} at {} context{}: fits",
            plan::label(&fit.model),
            gib(fit.model.size),
            gib(fit.footprint),
            f.context,
            src(fit.kv_source)
        );
    }
    match f.fits.iter().filter(|x| x.model.complete).min_by_key(|x| x.model.size) {
        Some(s) if usable == 0 => format!("no memory budget reported; smallest is {} {}", plan::label(&s.model), gib(s.model.size)),
        Some(s) => format!(
            "nothing fits ({} usable): the smallest, {}, needs {} at {} context{}",
            gib(usable),
            plan::label(&s.model),
            gib(s.footprint),
            f.context,
            src(s.kv_source)
        ),
        None => "no usable GGUF files in this repo".to_string(),
    }
}

async fn search(paths: &Paths, out: &mut dyn Write, node: &str, a: &SearchArgs) -> Result<()> {
    if a.format == Some(Format::Mlx) {
        bail!("searching over Weirdnet covers GGUF repos; MLX repos are searched on the machine itself (leave out --node)");
    }
    let query = a.query.join(" ");
    let limit = a.limit.clamp(1, 25) as u32;
    let done = ask(paths, out, node, ModelsOp::Search { query: query.clone(), limit }).await?;
    let (ok, failed) = answers(out, &done)?;
    let Some((_, ModelsReply::Hits(hits))) = ok.iter().find(|(_, r)| matches!(r, ModelsReply::Hits(_))) else {
        return finish(failed.max(1), done.nodes.len());
    };
    if hits.is_empty() {
        writeln!(out, "No repos matched `{query}`.")?;
        return finish(failed, done.nodes.len());
    }
    writeln!(out, "Sizes include the KV cache and overhead, measured against each node's own memory budget.")?;
    for h in hits {
        let gated = if h.gated { ", gated (needs a token)" } else { "" };
        writeln!(out, "\n{}  (GGUF, {} downloads, {} likes{gated})", h.id, count(h.downloads), count(h.likes))?;
        let files =
            ask(paths, out, node, ModelsOp::Files { repo: h.id.clone(), context: a.context.map(|c| c.min(u64::from(u32::MAX)) as u32), slots: None })
                .await?;
        for r in &files.nodes {
            match &r.outcome {
                Outcome::Ok { reply } => match &**reply {
                    ModelsReply::Fits(f) => writeln!(out, "    {}: {}", r.node.label, verdict_line(f))?,
                    _ => writeln!(out, "    {}: unexpected answer", r.node.label)?,
                },
                Outcome::Failed { message } => writeln!(out, "    {}", message)?,
            }
        }
    }
    writeln!(out, "\nDownload one with `ymirctl models pull <repo> --node <name|auto>`.")?;
    finish(failed, done.nodes.len())
}

async fn list(paths: &Paths, out: &mut dyn Write, node: &str) -> Result<()> {
    let done = ask(paths, out, node, ModelsOp::List).await?;
    let mut failed = 0;
    for r in &done.nodes {
        match &r.outcome {
            Outcome::Ok { reply } => {
                let ModelsReply::List(entries) = &**reply else { continue };
                writeln!(out, "Models on {}:", r.node.label)?;
                if entries.is_empty() {
                    writeln!(out, "  none downloaded")?;
                }
                for e in entries {
                    let complete = e.model.as_ref().is_none_or(|m| m.complete);
                    let state = match (&e.served_as, e.state.clone(), &e.base) {
                        (Some(id), _, Some(base)) => format!("adapter `{id}` on `{base}`"),
                        (Some(id), Some(st), None) => format!("serving as `{id}` ({st:?})"),
                        (Some(id), None, None) => format!("serving as `{id}`"),
                        _ if !complete => "incomplete (parts missing)".to_string(),
                        _ if e.adapter.is_some() => "adapter, not served".to_string(),
                        _ => "not served".to_string(),
                    };
                    let files = e.model.as_ref().map_or(1, |m| m.files.len());
                    let shards = if files > 1 { format!(" ({files} files)") } else { String::new() };
                    writeln!(out, "  {:<40} {:>10}{shards}  {state}", e.id(), gib(e.size()))?;
                }
            }
            Outcome::Failed { message } => {
                failed += 1;
                writeln!(out, "{}: {message}", r.node.label)?;
            }
        }
    }
    finish(failed, done.nodes.len())
}

async fn remove(paths: &Paths, out: &mut dyn Write, input: &mut dyn BufRead, node: &str, a: &RemoveArgs) -> Result<()> {
    if node == "auto" {
        bail!("`--node auto` is for pulls; name the node (or `all`) to remove from");
    }
    if !a.yes {
        writeln!(
            out,
            "Delete `{}` on {}? That stops serving it there and deletes its files (node.toml is not edited).",
            a.id,
            if node == "all" { "every model node".to_string() } else { node.to_string() }
        )?;
        write!(out, "Type `yes` to delete: ")?;
        out.flush()?;
        let mut line = String::new();
        input.read_line(&mut line)?;
        if !matches!(line.trim(), "y" | "yes") {
            return Ok(writeln!(out, "Nothing deleted.")?);
        }
    }
    let done = ask(paths, out, node, ModelsOp::Remove { id: a.id.clone() }).await?;
    let mut failed = 0;
    for r in &done.nodes {
        match &r.outcome {
            Outcome::Ok { reply } => {
                if let ModelsReply::Removed { id, freed } = &**reply {
                    writeln!(out, "{}: deleted {id} ({} freed)", r.node.label, gib(*freed))?;
                }
            }
            Outcome::Failed { message } => {
                failed += 1;
                writeln!(out, "{}: {message}", r.node.label)?;
            }
        }
    }
    finish(failed, done.nodes.len())
}

fn explain_pick(out: &mut dyn Write, c: &Chosen) -> Result<()> {
    Ok(writeln!(out, "Picked {}: it can hold {} ({}) in its {} of model memory.", c.node.label, c.model, gib(c.size), gib(c.usable_bytes))?)
}

/// `models adapter serve | unserve | scale --node`: live adapter changes on one node.
async fn adapter(paths: &Paths, out: &mut dyn Write, node: &str, cmd: AdapterCmd) -> Result<()> {
    use weirdnet_backends::AdapterChange;
    if node == "auto" {
        bail!("`--node auto` is for pulls; name the node that serves the adapter (or `all`)");
    }
    let op = match &cmd {
        AdapterCmd::Serve(a) => ModelsOp::ServeAdapter { id: a.id.clone(), base: a.base.clone(), scale: a.scale },
        AdapterCmd::Unserve(a) => ModelsOp::UnserveAdapter { id: a.id.clone() },
        AdapterCmd::Scale(a) => ModelsOp::AdapterScale { id: a.id.clone(), scale: if a.reset { None } else { a.scale } },
        AdapterCmd::Add(_) | AdapterCmd::Remove(_) | AdapterCmd::List | AdapterCmd::Slots(_) => {
            bail!("`ymirctl models adapter add | remove | list | slots` edit this machine's node.toml; run them on the node itself (without --node)")
        }
    };
    let done = ask(paths, out, node, op).await?;
    let (ok, failed) = answers(out, &done)?;
    for (label, reply) in &ok {
        match reply {
            ModelsReply::AdapterServed { id, base, change: AdapterChange::NextLoad } => writeln!(
                out,
                "{label}: `{id}` is added to `{base}`, whose engine can't load a new adapter while it runs (llama-server, or vLLM started without LoRA: see `ymirctl models adapter slots`): the first request for `{id}` restarts `{base}` with it"
            )?,
            ModelsReply::AdapterServed { id, base, .. } => writeln!(out, "{label}: serving `{id}` now, on `{base}` (no restart)")?,
            ModelsReply::AdapterUnserved { id, base } => writeln!(out, "{label}: stopped serving `{id}` on `{base}`; its files are kept")?,
            ModelsReply::AdapterScaled { id, scale } => {
                writeln!(out, "{label}: `{id}` now applies at scale {} from its next request", scale.unwrap_or(1.0))?
            }
            _ => {}
        }
    }
    if !ok.is_empty() {
        writeln!(out, "This is in memory only; node.toml was not changed (`ymirctl models adapter add` on the node keeps an adapter).")?;
    }
    finish(failed, done.nodes.len())
}

/// What to tell the operator about an adapter pulled for `base`.
fn adapter_pulled(out: &mut dyn Write, label: &str, base: &str, p: &weirdnet_backends::models::Pulled) -> Result<()> {
    use weirdnet_backends::AdapterChange;
    match (p.served, p.change) {
        (true, Some(AdapterChange::NextLoad)) => writeln!(
            out,
            "{label}: `{}` is added to `{base}`, which is running without it: the first request for `{}` restarts `{base}` with it",
            p.id, p.id
        )?,
        (true, _) => writeln!(out, "{label}: serving `{}` now, on `{base}`", p.id)?,
        _ => writeln!(out, "{label}: not serving yet; pull again with --serve (no restart needed), or add it to node.toml there")?,
    }
    writeln!(
        out,
        "\nTo keep it after a restart, run on {label}:\n  ymirctl models adapter add {base} {} --id {}",
        p.files.first().map(String::as_str).unwrap_or_default(),
        p.id
    )?;
    if p.served {
        writeln!(out, "{label}: serving without a restart is in memory only; node.toml was not changed.")?;
    }
    Ok(())
}

async fn pull(paths: &Paths, out: &mut dyn Write, node: &str, a: &PullArgs) -> Result<()> {
    if a.register {
        bail!(
            "--register edits node.toml on the machine you run it on; another node's node.toml is edited there. Add --serve to serve the model from its ymird now (in memory), and add the printed entry to its node.toml to keep it"
        );
    }
    if a.id.is_some() || a.role.is_some() {
        bail!(
            "--id and --role shape a node.toml entry on this machine; with --node the model keeps its file name as its id (the printed entry shows where to change it)"
        );
    }
    if a.adapter_for.is_some() {
        if node == "auto" || node == "all" {
            bail!("--adapter-for needs the node that serves the base: name it with --node");
        }
        if a.quant.is_some() || a.dry_run || a.context.is_some() {
            bail!("--quant, --context and --dry-run are for models; pick an adapter file with --file");
        }
    }
    let lookup = if node == "auto" { "all" } else { node };
    let context = a.context.map(|c| c.min(u64::from(u32::MAX)) as u32);
    let mut model = a.file.clone();
    if let Some(q) = &a.quant {
        let done = ask(paths, out, lookup, ModelsOp::Files { repo: a.repo.clone(), context, slots: None }).await?;
        let (ok, _) = answers(out, &done)?;
        let fits = ok.iter().find_map(|(_, r)| if let ModelsReply::Fits(f) = r { Some(f) } else { None }).context("no node could read that repo")?;
        let hits: Vec<_> = fits
            .fits
            .iter()
            .filter(|x| {
                let l = plan::label(&x.model);
                x.model.complete
                    && (l.eq_ignore_ascii_case(q) || l.to_ascii_uppercase().strip_prefix("UD-").is_some_and(|l| l.eq_ignore_ascii_case(q)))
            })
            .collect();
        model = match hits.as_slice() {
            [one] => Some(one.model.id.clone()),
            [] => bail!("{} has no `{q}` quant; it has: {}", a.repo, fits.fits.iter().map(|x| plan::label(&x.model)).collect::<Vec<_>>().join(", ")),
            _ => bail!("{} has several `{q}` files; pick one with --file", a.repo),
        };
    }
    if a.dry_run {
        let done = ask(paths, out, lookup, ModelsOp::Files { repo: a.repo.clone(), context, slots: None }).await?;
        for r in &done.nodes {
            match &r.outcome {
                Outcome::Ok { reply } => {
                    if let ModelsReply::Fits(f) = &**reply {
                        writeln!(out, "{}: {}", r.node.label, verdict_line(f))?;
                    }
                }
                Outcome::Failed { message } => writeln!(out, "{message}")?,
            }
        }
        return Ok(writeln!(out, "Dry run: nothing was downloaded.")?);
    }
    let mut pull = Pull::new(&a.repo);
    pull.model = model;
    pull.context = context;
    pull.serve = a.serve || a.load;
    pull.load = a.load;
    pull.adapter_for.clone_from(&a.adapter_for);
    let done = ask(paths, out, node, ModelsOp::Pull(pull)).await?;
    if let Some(c) = &done.chosen {
        explain_pick(out, c)?;
    }
    let mut failed = 0;
    for r in &done.nodes {
        match &r.outcome {
            Outcome::Ok { reply } => {
                let ModelsReply::Pulled(p) = &**reply else { continue };
                writeln!(out, "{}: pulled {} ({}, {} file(s), checksum verified)", r.node.label, p.id, gib(p.size), p.files.len())?;
                if let Some(base) = &p.adapter_for {
                    adapter_pulled(out, &r.node.label, base, p)?;
                    continue;
                }
                match (p.served, p.loaded) {
                    (_, true) => writeln!(out, "{}: serving `{}` now, loaded", r.node.label, p.id)?,
                    (true, false) => writeln!(out, "{}: serving `{}` now (loads on first use)", r.node.label, p.id)?,
                    _ => writeln!(out, "{}: not serving yet; pull again with --serve (no restart needed), or add the entry below", r.node.label)?,
                }
                let entry = plan::local_model(&p.id, "llamacpp", "chat", &p.files[0], context.map(u64::from), false);
                writeln!(
                    out,
                    "\nnode.toml entry for {} (to keep it after a restart; its file path is under that node's models directory):\n{}",
                    r.node.label,
                    plan::stanza(&entry)
                )?;
                if p.served {
                    writeln!(out, "{}: serving without a restart is in memory only; node.toml was not changed.", r.node.label)?;
                }
            }
            Outcome::Failed { message } => {
                failed += 1;
                writeln!(out, "{}: {message}", r.node.label)?;
            }
        }
    }
    finish(failed, done.nodes.len())
}
