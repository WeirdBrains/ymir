//! `ymirctl models catalog` and catalog pulls (`ymirctl models pull ID`): the models
//! `weirdnet-model` ships pinned entries for, the text-to-speech ones included (0.5.5), with their
//! purpose, engine, size, license and speech profile. A restricted license (Fish Audio Drama 3's
//! `fish-audio-community`) needs `--accept-license` before any request leaves this machine.

use std::io::Write;

use anyhow::{Result, bail};
use clap::Args;
use serde_json::{Value, json};
use weirdnet_model::hw::{self, CatalogEntry, Purpose};
use weirdnet_model::{ModelSpec, SpeechProfile};

use super::{Ctx, PullArgs, gib};

#[derive(Args, Debug)]
pub struct CatalogArgs {
    /// Only this purpose: chat, embedding, transcribe or speak.
    #[arg(long)]
    pub purpose: Option<String>,
    /// JSON instead of a table.
    #[arg(long)]
    pub json: bool,
}

fn purpose_of(s: &str) -> Result<Purpose> {
    match Purpose::ALL.into_iter().find(|p| p.as_str().eq_ignore_ascii_case(s.trim())) {
        Some(p) => Ok(p),
        None => bail!("unknown purpose `{s}` (chat, embedding, transcribe or speak)"),
    }
}

/// The engine an entry runs on: GGUF on llama.cpp, the rest of the speakers on vLLM-Omni.
fn engine(e: &CatalogEntry) -> &'static str {
    match (e.is_gguf(), e.speaks()) {
        (true, _) => "llamacpp",
        (false, true) => "vllm-omni",
        (false, false) => "vllm",
    }
}

/// Weights, projector and companions (the footprint's file part, without KV and headroom).
fn download_bytes(e: &CatalogEntry) -> u64 {
    e.spec.size + e.projector().map_or(0, |p| p.size)
}

/// A short line of what a speaker does.
pub(crate) fn speech_summary(p: &SpeechProfile) -> String {
    let mut parts = Vec::new();
    match &p.presets {
        Some(v) if !v.is_empty() => parts.push(format!("{} preset voice{}", v.len(), if v.len() == 1 { "" } else { "s" })),
        _ => {}
    }
    if p.design {
        parts.push("designs voices".into());
    }
    if let Some(c) = &p.clone {
        let how = serde_json::to_value(c).ok().and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default();
        parts.push(format!("clones ({})", how.replace('_', " ")));
    }
    if p.instructions {
        parts.push("instructions".into());
    }
    if let Some(n) = p.script_speakers.filter(|n| *n > 1) {
        parts.push(format!("scripts up to {n} speakers"));
    }
    if p.raw {
        parts.push("raw input".into());
    }
    if let Some(r) = p.sample_rate {
        parts.push(format!("{} kHz", f64::from(r) / 1000.0));
    }
    if !p.formats.is_empty() {
        parts.push(p.formats.iter().map(|f| f.as_str()).collect::<Vec<_>>().join("/"));
    }
    if let Some(l) = &p.languages {
        parts.push(format!("{} language{}", l.len(), if l.len() == 1 { "" } else { "s" }));
    }
    parts.join(", ")
}

/// One entry as `catalog --json` prints it.
pub(crate) fn entry_json(e: &CatalogEntry) -> Value {
    json!({
        "id": e.spec.id,
        "label": e.spec.label,
        "purpose": e.purpose.as_str(),
        "engine": engine(e),
        "repo": e.spec.repo,
        "file": e.spec.file,
        "projector": e.projector().map(|p| json!({"repo": p.repo, "file": p.file, "bytes": p.size})),
        "download_bytes": download_bytes(e),
        "footprint_bytes": e.footprint(1),
        "params_b": e.params_b,
        "license": e.license(),
        "restricted": e.restricted(),
        "speech": e.speech(),
    })
}

/// `ymirctl models catalog`.
pub fn run(out: &mut dyn Write, a: &CatalogArgs) -> Result<()> {
    let want = a.purpose.as_deref().map(purpose_of).transpose()?;
    let entries: Vec<CatalogEntry> = match want {
        Some(p) => hw::by_purpose(p),
        None => hw::catalog(),
    };
    if a.json {
        let v: Vec<Value> = entries.iter().map(entry_json).collect();
        writeln!(out, "{}", serde_json::to_string_pretty(&v)?)?;
        return Ok(());
    }
    for e in &entries {
        let license = match (e.license(), e.restricted()) {
            (Some(l), true) => format!(", license {l} (restricted: pull needs --accept-license {l})"),
            (Some(l), false) => format!(", license {l}"),
            (None, _) => String::new(),
        };
        writeln!(out, "{} [{}] {}, {}{license}", e.spec.id, e.purpose, engine(e), gib(download_bytes(e)))?;
        if let Some(p) = e.speech() {
            writeln!(out, "    speaks: {}", speech_summary(&p))?;
        }
    }
    if want.is_none_or(|p| p == Purpose::Speak) {
        writeln!(
            out,
            "\nPull a speaker with `ymirctl models pull ID --register` (role = \"speak\"); `ymirctl speak` and the bridge's /v1/audio/speech use it."
        )?;
    }
    Ok(())
}

/// The license a pull of `target` (a catalog id or a Hub repo) needs accepted, checked before any
/// request: a restricted catalog entry by id, or any file from its repos
/// ([`hw::restricted_license`]). `Ok(Some(license))` when it is accepted; an error naming the flag
/// when not.
pub fn license_gate(target: &str, accepted: Option<&str>) -> Result<Option<&'static str>> {
    let probe = ModelSpec { id: target.to_string(), label: String::new(), repo: target.to_string(), file: String::new(), size: 0, sha256: None };
    let Some(license) = hw::find(target).filter(CatalogEntry::restricted).and_then(|e| e.license()).or_else(|| hw::restricted_license(&probe)) else {
        return Ok(None);
    };
    match accepted {
        Some(a) if a.trim().eq_ignore_ascii_case(license) => Ok(Some(license)),
        Some(a) => bail!("{target} is under the `{license}` license; --accept-license {a} accepts another one. Nothing was downloaded"),
        None => bail!(
            "{target} is under the `{license}` license, which restricts how the model may be used{}. Read it first; \
             if you accept it, pull again with --accept-license {license}. Nothing was downloaded",
            if license == weirdnet_model::speech::FISH_AUDIO_COMMUNITY {
                " (free for research and non-commercial use; commercial use is capped by annual revenue, and no hosted text-to-speech service)"
            } else {
                ""
            }
        ),
    }
}

/// `ymirctl models pull ID` for a catalog entry: its pinned files (model and projector, each
/// checked against its SHA-256), or for a vLLM-Omni checkpoint nothing (the engine fetches the
/// repo). Prints the `[[model.local]]` entry (a speaker's with `role = "speak"`), and with
/// `--register` adds it to node.toml.
pub fn pull(ctx: &Ctx, out: &mut dyn Write, a: &PullArgs, e: &CatalogEntry) -> Result<()> {
    let license = license_gate(&e.spec.id, a.accept_license.as_deref())?;
    if a.quant.is_some() || a.file.is_some() {
        bail!("{} is a pinned catalog entry; --quant and --file apply to Hub repos (pull {} for its other files)", e.spec.id, e.spec.repo);
    }
    let id = a.id.clone().unwrap_or_else(|| e.spec.id.clone());
    let role = match e.purpose {
        Purpose::Speak => "speak".to_string(),
        Purpose::Embedding => "embedding".to_string(),
        Purpose::Transcribe => "transcribe".to_string(),
        _ => a.role.clone().unwrap_or_else(|| if ctx.cfg.as_ref().is_some_and(|c| c.main_model().is_some()) { "chat".into() } else { "main".into() }),
    };
    let fetch: Vec<ModelSpec> = if e.is_gguf() { std::iter::once(e.spec.clone()).chain(e.projector()).collect() } else { Vec::new() };
    writeln!(
        out,
        "{}: {} [{}] on {}, {} to download, needs about {}{}",
        e.spec.id,
        e.spec.label,
        e.purpose,
        engine(e),
        if fetch.is_empty() { "nothing".to_string() } else { gib(fetch.iter().map(|s| s.size).sum()) },
        gib(e.footprint(1)),
        license.map(|l| format!(", license {l} (accepted)")).unwrap_or_default(),
    )?;
    if let Some(p) = e.speech() {
        writeln!(out, "Speaks: {}", speech_summary(&p))?;
    }
    if e.restricted() {
        writeln!(
            out,
            "Guests of a shared pool don't get this speaker (its license restricts serving it); to open it to them, add `[model.local.speak]` with `guests = true` under the entry."
        )?;
    }
    let mut stanza = format!("[[model.local]]\nid = \"{id}\"\nbackend = \"{}\"\nrole = \"{role}\"\n", engine(e));
    if e.is_gguf() {
        stanza.push_str(&format!("file = \"{}\"\n", ctx.store.path(&e.spec).display()));
        if let Some(p) = e.projector() {
            stanza.push_str(&format!("mmproj = \"{}\"\n", ctx.store.path(&p).display()));
        }
    } else {
        stanza.push_str(&format!("file = \"{}\"\n", e.spec.repo));
    }
    if a.dry_run {
        writeln!(out, "\nnode.toml entry:\n{stanza}")?;
        return Ok(writeln!(out, "Dry run: nothing was downloaded.")?);
    }
    for spec in &fetch {
        super::download(ctx, out, &ctx.store, &spec.file, std::slice::from_ref(spec), license)?;
    }
    if fetch.is_empty() {
        writeln!(out, "{} fetches {} into the Hugging Face cache on first load.", if e.speaks() { "vLLM-Omni" } else { "vLLM" }, e.spec.repo)?;
    }
    writeln!(out, "\nnode.toml entry:\n{stanza}")?;
    if a.register {
        super::local::register(&ctx.config, &id, &stanza)?;
        writeln!(out, "Added `{id}` to {} (the old file is {}.bak). Restart ymird to load it.", ctx.config.display(), ctx.config.display())?;
    } else {
        writeln!(out, "node.toml was not changed. Add the entry above and restart ymird, or pull again with --register.")?;
    }
    Ok(())
}
