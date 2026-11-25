//! `ymirctl models upload FILE`: put a model or LoRA adapter trained elsewhere into this node's
//! models directory (docs/design/model-upload.md, slice U1, local only). It checks the GGUF
//! header, hashes the file while copying it in atomically, writes `provenance.json` beside it
//! (`ymir_core::upload`), and adds it to node.toml so it survives a restart (`--no-register` only
//! prints the entry). `--from-run RUN_DIR` takes a `ymirctl train run` output with its run record
//! as the provenance (origin `train`, slice U5). With `--node X` it goes to another of the owner's
//! model nodes over Weirdnet instead (slice U2, [`remote_op`]; `ymir_core::upload_net`), which
//! also takes a directory of safetensors (for vLLM) or MLX weights, a PEFT adapter directory (for
//! a vLLM base) and a GGUF vision projector (`--projector-for`, slice U7).

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Args;
use weirdnet_backends::EngineKind;
use ymir_core::NodeConfig;
use ymir_core::config::LocalAdapter;
use ymir_core::train::{self, AdapterRecord, RunRecord, RunStatus};
use ymir_core::upload::{self, ArtifactKind, ImportSpec, ProvenanceBase};

use super::plan::{self, id_for_file};
use super::{Ctx, DirArgs, Meter, adapters, gib, hub, local};

#[derive(Args, Debug)]
pub struct UploadArgs {
    /// The GGUF file: a model, or a LoRA adapter (with --adapter-for). With --node also a
    /// directory of safetensors or MLX weights, a PEFT adapter directory (with --adapter-for), or a
    /// GGUF vision projector (with --projector-for). With --from-run it may be left out when the
    /// run's adapter is in this node's models directory.
    #[arg(required_unless_present = "from_run")]
    pub file: Option<PathBuf>,
    /// A `ymirctl train run` directory (with its run.json): the adapter it made, with the run's
    /// base, dataset record and trainer as its provenance (origin `train`). The base defaults to
    /// the run's, the id to the run's adapter id.
    #[arg(long, value_name = "RUN_DIR")]
    pub from_run: Option<PathBuf>,
    /// The id it serves as (default: the file name in lower case; for an adapter `BASE+NAME`).
    #[arg(long)]
    pub name: Option<String>,
    /// It is a LoRA adapter for this model (a `[[model.local]]` in node.toml: llama.cpp for a GGUF
    /// LoRA, vLLM for a PEFT adapter).
    #[arg(long, value_name = "BASE")]
    pub adapter_for: Option<String>,
    /// With --node: it is a vision projector (mmproj GGUF) for this model, which the node serves
    /// on llama.cpp; it is saved as that model's `mmproj`.
    #[arg(long, value_name = "MODEL", requires = "node", conflicts_with = "adapter_for")]
    pub projector_for: Option<String>,
    /// The license the weights are under (an SPDX id such as apache-2.0, or the license's name),
    /// recorded in provenance.json. Optional.
    #[arg(long)]
    pub license: Option<String>,
    /// The entry's role for a model: `main`, `standin`, `embedding` or `chat` (default: `main`
    /// when node.toml has no main model, else `chat`).
    #[arg(long)]
    pub role: Option<String>,
    /// Don't add it to node.toml (with --node: the node's); only print the entry. By default it
    /// is added (a copy is kept as node.toml.bak) so it survives a restart.
    #[arg(long, visible_alias = "no-persist")]
    pub no_register: bool,
    #[command(flatten)]
    pub dir: DirArgs,
    /// Send it to this model node over Weirdnet (a device name or peer id) instead of this
    /// machine. Needs a running ymird here, linked to your devices; an interrupted upload resumes.
    /// Several nodes: `--node A,B` or `--node all` (every model node but this one) sends the files
    /// once and the others copy them from that node.
    #[arg(long)]
    pub node: Option<String>,
    /// With several nodes: the one the files are sent to, the others copying from it (default:
    /// the one with the most free space).
    #[arg(long, value_name = "NODE", requires = "node")]
    pub stage: Option<String>,
    #[command(flatten)]
    pub rollout: super::upload_rollout::RolloutArgs,
}

fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

/// The nearest existing directory at or above `dir`, for the free-space check.
fn existing(dir: &Path) -> Option<&Path> {
    dir.ancestors().find(|p| p.is_dir())
}

/// The adapter's base in node.toml, checked: served by llama.cpp, its file here, same architecture.
fn check_base(ctx: &Ctx, base: &str, item: &upload::Inspected) -> Result<ProvenanceBase> {
    let cfg: &NodeConfig =
        ctx.cfg.as_ref().with_context(|| format!("{} not found or not readable (run `ymirctl setup` first)", ctx.config.display()))?;
    let Some(m) = cfg.model.local.iter().find(|m| m.id == base) else {
        bail!("node.toml has no [[model.local]] with id `{base}`; the adapter's base must be a model on this node");
    };
    let launch = cfg.launches().into_iter().find(|l| l.id == base).with_context(|| format!("`{base}` has no launch on this node"))?;
    if launch.engine != EngineKind::Llamacpp {
        bail!("`{base}` runs on {}: a GGUF LoRA serves on llama.cpp only (send a PEFT adapter directory for it with --node)", launch.engine.name());
    }
    if !Path::new(&launch.source).is_file() {
        bail!("{base}'s model file {} is not on this machine, so the adapter can't be checked against it", launch.source);
    }
    let header = upload::check_adapter_base(item, Path::new(&launch.source))?;
    let mut b = ProvenanceBase::default();
    b.id = Some(base.to_string());
    b.hub = item.header.base_model.clone().or(header.repo).or(header.base_model);
    b.revision = m.revision.clone();
    b.quant = m.quant.clone();
    Ok(b)
}

/// What `ymirctl models upload FILE --node X` asks this machine's ymird to do: the file checked and
/// hashed here, with its provenance. The node checks it again (and an adapter against its base)
/// as the bytes arrive.
pub fn remote_op(out: &mut dyn Write, a: &UploadArgs) -> Result<ymir_core::upload_net::UploadOp> {
    let license = upload::check_license(a.license.as_deref())?;
    let run = a.from_run.as_deref().map(read_run).transpose()?;
    let file = match (&a.file, &run, &a.from_run) {
        (Some(f), _, _) => f.clone(),
        (None, Some((r, ad)), Some(dir)) => {
            let name = Path::new(&ad.file).file_name().map(PathBuf::from).unwrap_or_default();
            let f = dir.join(&name);
            if !f.is_file() {
                bail!("run {}'s adapter is not in {}; give its file: ymirctl models upload FILE --from-run RUN_DIR --node NODE", r.id, dir.display());
            }
            f
        }
        _ => bail!("give the GGUF file, or --from-run RUN_DIR"),
    };
    let file = std::fs::canonicalize(&file).with_context(|| format!("reading {}", file.display()))?;
    let item = upload::inspect_outgoing(&file, &mut |_, _| {})?;
    let adapter_for = a.adapter_for.clone().or_else(|| run.as_ref().map(|(r, _)| r.base.clone()));
    if let Some((r, ad)) = &run {
        if !item.kind().is_adapter() {
            bail!("{} is a full model; run {} made a LoRA adapter", item.name, r.id);
        }
        let sha = item.artifact.files.first().map(|f| f.sha256.as_str()).unwrap_or_default();
        if sha.trim_start_matches("sha256:") != ad.sha256.trim_start_matches("sha256:") {
            bail!("{} is not the adapter run {} made: its SHA-256 differs from the run record's", file.display(), r.id);
        }
    }
    match (item.kind(), &a.projector_for) {
        (ArtifactKind::Projector, None) => {
            bail!("{} is a vision projector: say which model on the node it goes with with --projector-for MODEL", item.name)
        }
        (ArtifactKind::Projector, Some(_)) => {}
        (k, Some(_)) => bail!("{} is a {}, not a vision projector; leave out --projector-for", item.name, upload::kind_name(k)),
        _ => {}
    }
    let base = match (item.kind(), &adapter_for) {
        (ArtifactKind::Adapter { .. }, None) => {
            bail!("{} is a LoRA adapter: say which model on the node it goes on with --adapter-for BASE", item.name)
        }
        (ArtifactKind::Model | ArtifactKind::Projector, Some(_)) => {
            bail!("{} is a {}, not an adapter; leave out --adapter-for", item.name, upload::kind_name(item.kind()))
        }
        (ArtifactKind::Adapter { .. }, Some(b)) => {
            // The run's base (the revision and quant it trained on); the node checks the rest.
            let mut base = run.as_ref().and_then(|(r, ad)| train::import_spec(r, ad, 0).base).unwrap_or_default();
            base.id = Some(b.clone());
            base.hub = item.artifact.base_model.clone().or(base.hub);
            Some(base)
        }
        _ => None,
    };
    let id = match (&a.name, &adapter_for) {
        (Some(n), _) => n.clone(),
        (None, Some(_)) if run.is_some() => run.as_ref().map(|(_, ad)| ad.id.clone()).unwrap_or_default(),
        (None, Some(b)) => adapters::default_id(b, &item.name),
        (None, None) => id_for_file(&item.name),
    };
    upload::check_id(&id)?;
    let spec = match &run {
        Some((r, ad)) => ImportSpec { id: id.clone(), license, base, ..train::import_spec(r, ad, now_ms()) },
        None => ImportSpec {
            id: id.clone(),
            license,
            base,
            origin: upload::ORIGIN_UPLOAD.into(),
            uploaded_ms: now_ms().max(0) as u64,
            ..Default::default()
        },
    };
    let goes = adapter_for.as_deref().or(a.projector_for.as_deref()).map(|b| format!(", for `{b}`")).unwrap_or_default();
    writeln!(
        out,
        "{}: {} {} ({}), {}{goes}, as `{id}`",
        item.name,
        item.format_name(),
        upload::kind_name(item.kind()),
        item.architecture(),
        gib(item.size()),
    )?;
    let mut up = weirdnet_backends::models::Upload::new(id, &item.artifact, upload::provenance_for(&item.artifact, &spec)).serving();
    if let Some(b) = adapter_for {
        up = up.with_adapter_for(b);
    }
    if let Some(m) = &a.projector_for {
        up = up.with_projector_for(m.clone());
    }
    Ok(ymir_core::upload_net::UploadOp { upload: up, local: vec![file], persist: !a.no_register, role: a.role.clone(), stage: a.stage.clone() })
}

/// A finished run's record and its adapter, from `RUN_DIR/run.json`.
fn read_run(dir: &Path) -> Result<(RunRecord, AdapterRecord)> {
    let path = dir.join("run.json");
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {} (--from-run takes a `ymirctl train run` directory)", path.display()))?;
    let record: RunRecord = serde_json::from_str(&text).with_context(|| format!("{} is not a run record", path.display()))?;
    if record.format != train::RUN_FORMAT {
        bail!("{} is a `{}` record; this version reads `{}`", path.display(), record.format, train::RUN_FORMAT);
    }
    if record.status != RunStatus::Done {
        bail!("run {} did not finish (it is {:?}), so it has no adapter to upload", record.id, record.status);
    }
    let adapter = record.adapter.clone().with_context(|| format!("run {} recorded no adapter", record.id))?;
    Ok((record, adapter))
}

pub fn run(ctx: &Ctx, out: &mut dyn Write, a: &UploadArgs) -> Result<()> {
    let disk = existing(ctx.dir()).and_then(hub::disk_space);
    run_with(ctx, out, a, disk)
}

/// [`run`] with the models disk's free and total bytes given (`None`: unknown, not checked).
pub(super) fn run_with(ctx: &Ctx, out: &mut dyn Write, a: &UploadArgs, disk: Option<(u64, u64)>) -> Result<()> {
    if a.rollout.rollout.is_some() {
        bail!("--rollout goes with --node: the upload joins the rollout on this machine's inference role after the node serves it");
    }
    let license = upload::check_license(a.license.as_deref())?;
    // Registered by default when there is a node.toml to add it to (m3talsmith, 2026-11-24).
    let register = !a.no_register && ctx.cfg.is_some();
    let run = a.from_run.as_deref().map(read_run).transpose()?;
    let file = match (&a.file, &run) {
        (Some(f), _) => f.clone(),
        (None, Some((r, ad))) => {
            let f = ctx.dir().join(&ad.file);
            if !f.is_file() {
                bail!(
                    "run {}'s adapter {} is not in this node's models directory; give its file: ymirctl models upload FILE --from-run RUN_DIR",
                    r.id,
                    ad.file
                );
            }
            f
        }
        (None, None) => bail!("give the GGUF file, or --from-run RUN_DIR"),
    };
    let item = upload::inspect(&file)?;
    let adapter_for = a.adapter_for.clone().or_else(|| run.as_ref().map(|(r, _)| r.base.clone()));
    if let Some((r, ad)) = &run {
        if !item.kind().is_adapter() {
            bail!("{} is a full model; run {} made a LoRA adapter", item.file_name, r.id);
        }
        let (sha, _) = upload::sha256_file(&file)?;
        if sha != ad.sha256 {
            bail!("{} is not the adapter run {} made: its SHA-256 differs from the run record's", file.display(), r.id);
        }
    }
    let base = match (item.kind(), &adapter_for) {
        (ArtifactKind::Adapter { .. }, None) => {
            bail!("{} is a LoRA adapter: say which model it goes on with --adapter-for BASE", item.file_name)
        }
        (ArtifactKind::Model, Some(_)) => bail!("{} is a full model, not an adapter; leave out --adapter-for", item.file_name),
        (ArtifactKind::Adapter { .. }, Some(b)) => Some(check_base(ctx, b, &item)?),
        _ => None,
    };
    let base = match (&run, base) {
        // The base the run trained on (its revision and quant), which must be the one here.
        (Some((r, ad)), Some(here)) => {
            if let (Some(trained), Some(now)) = (&ad.base_revision, &here.revision)
                && trained != now
            {
                bail!(
                    "run {} trained on revision {trained} of `{}`, but `{}` here is revision {now}; the adapter would not match its base",
                    r.id,
                    r.base,
                    here.id.as_deref().unwrap_or_default()
                );
            }
            let mut b = train::import_spec(r, ad, 0).base.unwrap_or_default();
            b.id = here.id;
            b.hub = here.hub;
            Some(b)
        }
        (_, b) => b,
    };
    let id = match (&a.name, &adapter_for) {
        (Some(n), _) => n.clone(),
        (None, Some(_)) if run.is_some() => run.as_ref().map(|(_, ad)| ad.id.clone()).unwrap_or_default(),
        (None, Some(b)) => adapters::default_id(b, &item.file_name),
        (None, None) => id_for_file(&item.file_name),
    };
    upload::check_id(&id)?;
    if register {
        let cfg = ctx.cfg.as_ref().context("node.toml")?;
        let taken = cfg.model.local.iter().any(|m| m.id == id || m.lora.iter().any(|l| l.id == id));
        if taken {
            bail!("node.toml already has a model or adapter `{id}`; choose another id with --name, or pass --no-register");
        }
    }
    writeln!(
        out,
        "{}: {} ({}), {}{}",
        item.file_name,
        upload::kind_name(item.kind()),
        item.header.architecture,
        gib(item.size()),
        base.as_ref().and_then(|b| b.id.as_deref()).map(|b| format!(", for `{b}`")).unwrap_or_default()
    )?;
    let models_dir = ctx.dir().to_path_buf();
    let spec = match &run {
        Some((r, ad)) => ImportSpec { id: id.clone(), license: license.clone(), base: base.clone(), disk, ..train::import_spec(r, ad, now_ms()) },
        None => ImportSpec {
            id: id.clone(),
            license: license.clone(),
            base: base.clone(),
            origin: upload::ORIGIN_UPLOAD.into(),
            uploaded_ms: now_ms().max(0) as u64,
            disk,
            ..Default::default()
        },
    };
    let mut meter = Meter { out: &mut *out, total: item.size(), tty: ctx.tty, last: 0 };
    let imported = upload::import(&models_dir, &item, &spec, &mut |done, _| meter.tick(&id, done))?;
    if ctx.tty {
        writeln!(out)?;
    }
    let p = &imported.provenance;
    if imported.already {
        writeln!(out, "Already in the models directory with the same contents; nothing was copied.")?;
    }
    let lic = p.license.as_deref().map(|l| format!(", license {l}")).unwrap_or_default();
    if let Some((r, _)) = &run {
        writeln!(out, "From run {} on {} (origin train).", r.id, r.node)?;
    }
    writeln!(out, "Imported `{id}` as {} (revision {}{lic}).", imported.file.display(), p.revision)?;
    writeln!(out, "Provenance: {}", imported.dir.join(upload::PROVENANCE_FILE).display())?;
    if ctx.cfg.is_none() && !a.no_register {
        writeln!(out, "No node.toml at {}, so nothing was registered (run `ymirctl setup` first).", ctx.config.display())?;
    }
    let file = imported.file.to_string_lossy().into_owned();
    match &adapter_for {
        None => register_model(ctx, out, a, register, &id, &file, &p.revision),
        Some(b) => register_adapter(ctx, out, register, b, &id, &file, base.as_ref(), &p.revision),
    }
}

fn register_model(ctx: &Ctx, out: &mut dyn Write, a: &UploadArgs, register: bool, id: &str, file: &str, revision: &str) -> Result<()> {
    let role =
        a.role.clone().unwrap_or_else(|| if ctx.cfg.as_ref().is_some_and(|c| c.main_model().is_some()) { "chat".into() } else { "main".into() });
    let cap = plan::capacity(&ctx.hw);
    let mut entry = plan::local_model(id, "llamacpp", &role, file, None, cap.accelerated);
    entry.revision = Some(revision.to_string());
    let stanza = plan::stanza(&entry);
    writeln!(out, "\nnode.toml entry:\n{stanza}")?;
    if register {
        local::register(&ctx.config, id, &stanza)?;
        writeln!(out, "Added `{id}` to {} (the old file is {}.bak). Restart ymird to load it.", ctx.config.display(), ctx.config.display())?;
    } else {
        writeln!(out, "node.toml was not changed. Add the entry above and restart ymird.")?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn register_adapter(
    ctx: &Ctx,
    out: &mut dyn Write,
    register: bool,
    base: &str,
    id: &str,
    file: &str,
    rec: Option<&ProvenanceBase>,
    revision: &str,
) -> Result<()> {
    let la = LocalAdapter {
        id: id.to_string(),
        file: file.to_string(),
        scale: None,
        revision: Some(revision.to_string()),
        base_revision: rec.and_then(|r| r.revision.clone()),
        base_quant: rec.and_then(|r| r.quant.clone()),
    };
    // llama.cpp loads adapter files only when the base starts, so a new one serves after a restart.
    let when = format!("Restart ymird (or wait for `{base}`'s next load) to serve it: llama.cpp loads a new adapter file only when its base starts.");
    if register {
        let text = std::fs::read_to_string(&ctx.config).with_context(|| format!("reading {}", ctx.config.display()))?;
        let new = adapters::add(&text, base, &la)?;
        adapters::validated(&new, base, id, ctx)?;
        adapters::save(&ctx.config, &text, &new)?;
        writeln!(out, "Added LoRA adapter `{id}` on `{base}` to node.toml (revision {revision}). {when}")?;
    } else {
        writeln!(out, "node.toml was not changed. To add it:\n  ymirctl models adapter add {base} {file} --id {id} --revision {revision}")?;
        writeln!(out, "{when}")?;
    }
    Ok(())
}
