//! Uploading a model trained elsewhere (docs/design/model-upload.md): check a local GGUF model or
//! GGUF LoRA adapter, hash it, place it in the models directory atomically, with a
//! `provenance.json` beside it saying where it came from.
//!
//! On `weirdnet-model` 0.5.3 since 2026-11-24: [`weirdnet_model::artifact::inspect`] reads and
//! hashes the file, [`Provenance`] is the record, and [`ModelStore::import`] places it. This module
//! keeps Ymir's own rules on top: the file kinds this slice takes (one GGUF file, a model or a LoRA
//! adapter), the ids and licenses it accepts, the free-space floor for local imports, the
//! "same bytes again" answer, and how a training run's record maps onto [`Provenance`].
//!
//! Layout ([`ModelStore::import`]): a model goes to `models_dir/<id>/<file>.gguf`, an adapter to
//! `models_dir/adapters/<id>/<file>.gguf` with `adapter.json`, each with `provenance.json`. Models
//! uploaded before the move (in `models_dir/uploads/<id>/`, U1) stay where they are. The file is
//! first copied into a staging directory next to the models directory (outside it), as an upload
//! over Weirdnet stages its parts, so the user's file is never moved. Nothing is ever executed.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use sha2::{Digest, Sha256};
use weirdnet_backends::models::KeepFree;
use weirdnet_model::artifact::{self, Artifact, ArtifactFormat};
use weirdnet_model::gguf::{GgufHeader, GgufKind, read_header};
use weirdnet_model::{ModelLayout, ModelStore};

pub use weirdnet_model::AdapterFormat;
pub use weirdnet_model::artifact::{
    ArtifactFile, ArtifactKind, ORIGIN_TRAIN, ORIGIN_UPLOAD, PROVENANCE_FILE, PROVENANCE_FORMAT, Provenance, ProvenanceBase, ProvenanceDataset,
    ProvenanceKind, ProvenanceTrainer,
};

/// Where models uploaded before `weirdnet-model` 0.5.3 went (U1), still looked at for "already
/// there".
pub const LEGACY_UPLOADS_DIR: &str = "uploads";
/// Where adapters go under the models directory.
pub const ADAPTERS_DIR: &str = weirdnet_model::store::ADAPTERS_DIR;

/// A name for what an upload is, for messages.
pub fn kind_name(kind: ArtifactKind) -> &'static str {
    match kind {
        ArtifactKind::Model => "model",
        ArtifactKind::Adapter { .. } => "LoRA adapter",
        ArtifactKind::Projector => "vision projector",
        _ => "artifact",
    }
}

/// A local GGUF checked (and hashed) for upload, before anything is copied.
#[derive(Debug, Clone)]
pub struct Inspected {
    pub path: PathBuf,
    /// The file name it keeps in the store (a training run sets its adapter's own).
    pub file_name: String,
    /// What `weirdnet-model` read: kind, format, architecture, the file's SHA-256, the revision.
    pub artifact: Artifact,
    pub header: GgufHeader,
}

impl Inspected {
    pub fn kind(&self) -> ArtifactKind {
        self.artifact.kind
    }

    pub fn size(&self) -> u64 {
        self.artifact.size()
    }

    pub fn sha256(&self) -> &str {
        &self.artifact.files[0].sha256
    }

    /// The artifact as it is stored: under [`Inspected::file_name`], with its revision for that.
    pub fn stored(&self) -> Artifact {
        let mut a = self.artifact.clone();
        if a.files[0].path != self.file_name {
            a.files[0].path = self.file_name.clone();
            a.revision = artifact::revision(&a.files);
        }
        a
    }
}

/// Checks a license when one is given: an id made of letters, digits and `.+-_ ()`. None (or
/// blank) is fine and records nothing; a license is not required (m3talsmith, 2026-11-24).
pub fn check_license(license: Option<&str>) -> Result<Option<String>> {
    let l = license.map(str::trim).unwrap_or("");
    if l.is_empty() {
        return Ok(None);
    }
    let ok = l.len() <= 64 && l.chars().all(|c| c.is_ascii_alphanumeric() || ".+-_ ()".contains(c));
    if !ok {
        bail!("{l:?} does not look like a license id (letters, digits and . + - _ only, at most 64 characters)");
    }
    Ok(Some(l.to_string()))
}

/// Checks an id for a model or adapter: a single path segment of letters, digits and `._+-`.
pub fn check_id(id: &str) -> Result<()> {
    let ok = !id.is_empty() && id.len() <= 128 && !id.starts_with('.') && id.chars().all(|c| c.is_ascii_alphanumeric() || "._+-".contains(c));
    if ok { Ok(()) } else { bail!("{id:?} is not a usable id: use letters, digits and . _ + - only, not starting with a dot") }
}

/// Whether a file name is a shard of a split GGUF (`-00001-of-00003.gguf`).
fn is_split(name: &str) -> bool {
    let stem = name.to_ascii_lowercase();
    let stem = stem.trim_end_matches(".gguf");
    let mut parts = stem.rsplitn(4, '-');
    let (Some(of), Some(word), Some(part)) = (parts.next(), parts.next(), parts.next()) else { return false };
    word == "of" && !of.is_empty() && of.chars().all(|c| c.is_ascii_digit()) && !part.is_empty() && part.chars().all(|c| c.is_ascii_digit())
}

/// Reads, checks and hashes a local file for upload: a regular file (not a symlink) named
/// `*.gguf`, not a split shard, whose header is a model or a LoRA adapter. Projectors are refused
/// for now (U7).
pub fn inspect(path: &Path) -> Result<Inspected> {
    inspect_with(path, &mut |_, _| {})
}

/// [`inspect`], calling `progress(done, total)` while it hashes.
pub fn inspect_with(path: &Path, progress: &mut dyn FnMut(u64, u64)) -> Result<Inspected> {
    let meta = std::fs::symlink_metadata(path).with_context(|| format!("reading {}", path.display()))?;
    if meta.file_type().is_symlink() {
        bail!("{} is a symlink; give the real file", path.display());
    }
    if !meta.is_file() {
        bail!("{} is not a file: a directory (safetensors, MLX weights or a PEFT adapter) is sent to a node with --node", path.display());
    }
    let file_name = path.file_name().map(|n| n.to_string_lossy().into_owned()).context("the path has no file name")?;
    if !file_name.to_ascii_lowercase().ends_with(".gguf") {
        bail!(
            "{file_name} is not a .gguf file: this machine takes GGUF models and GGUF LoRA adapters (send safetensors, MLX weights and PEFT adapters, as a directory, to a node with --node)"
        );
    }
    if file_name.starts_with('.') || file_name.contains(['/', '\\', ':']) {
        bail!("{file_name:?} is not a usable file name");
    }
    if is_split(&file_name) {
        bail!(
            "{file_name} is one shard of a split GGUF; split models are not taken yet (merge it with llama-gguf-split --merge, or wait for a later slice)"
        );
    }
    let header = read_header(std::io::BufReader::new(File::open(path)?)).with_context(|| format!("{file_name} is not a readable GGUF file"))?;
    match &header.kind {
        GgufKind::Projector => bail!("{file_name} is a vision projector (mmproj): send it to a node with --node and --projector-for MODEL"),
        GgufKind::Model if header.architecture.eq_ignore_ascii_case("clip") || header.architecture.eq_ignore_ascii_case("mmproj") => {
            bail!("{file_name} is a vision projector (mmproj): send it to a node with --node and --projector-for MODEL")
        }
        GgufKind::Model => {}
        k if k.is_lora() => {}
        GgufKind::Adapter { adapter_type, .. } => bail!("{file_name} is a `{adapter_type}` adapter; only LoRA adapters are served"),
        _ => bail!("{file_name} is a kind of GGUF this version does not take"),
    }
    let found = artifact::inspect_with(&[path.to_path_buf()], &mut |p| {
        progress(p.done, p.total);
        true
    })
    .map_err(|e| anyhow!("{file_name}: {e}"))?;
    match found.kind {
        ArtifactKind::Model | ArtifactKind::Adapter { format: AdapterFormat::GgufLora } => {}
        other => bail!("{file_name} reads as a {}, which this slice does not take", kind_name(other)),
    }
    Ok(Inspected { path: path.to_path_buf(), file_name, artifact: found, header })
}

/// What `ymirctl models upload --node` sends (U7): a GGUF model, GGUF LoRA or GGUF projector
/// file, or a directory of safetensors, MLX weights or a PEFT adapter.
#[derive(Debug, Clone)]
pub struct Outgoing {
    /// The file or directory, as given to [`artifact::inspect`].
    pub path: PathBuf,
    /// Its file or directory name.
    pub name: String,
    pub artifact: Artifact,
}

impl Outgoing {
    pub fn kind(&self) -> ArtifactKind {
        self.artifact.kind
    }

    pub fn size(&self) -> u64 {
        self.artifact.size()
    }

    /// The architecture it says, or `?` (a PEFT adapter does not say).
    pub fn architecture(&self) -> &str {
        self.artifact.architecture.as_deref().unwrap_or("?")
    }

    /// The format in words, for messages.
    pub fn format_name(&self) -> &'static str {
        match (&self.artifact.format, self.artifact.kind) {
            (_, ArtifactKind::Adapter { format: AdapterFormat::Peft }) => "PEFT",
            (ArtifactFormat::Safetensors, _) => "safetensors",
            (ArtifactFormat::Mlx, _) => "MLX",
            _ => "GGUF",
        }
    }
}

/// Whether `path` is a GGUF vision projector (mmproj), by its header.
pub fn is_projector(path: &Path) -> bool {
    let Ok(f) = File::open(path) else { return false };
    read_header(std::io::BufReader::new(f)).is_ok_and(|h| {
        matches!(h.kind, GgufKind::Projector) || h.architecture.eq_ignore_ascii_case("clip") || h.architecture.eq_ignore_ascii_case("mmproj")
    })
}

/// Whether `path` goes only through [`inspect_outgoing`]: a directory or a vision projector
/// (U9: placed with [`local::place`] on a machine that serves models, else sent to a node).
pub fn is_outgoing_only(path: &Path) -> bool {
    std::fs::symlink_metadata(path).is_ok_and(|m| m.is_dir()) || is_projector(path)
}

/// Reads, checks and hashes what an upload to a node sends: a GGUF file as [`inspect`] takes it, a
/// GGUF vision projector, or a directory (not a symlink) that `weirdnet-model` reads as safetensors
/// or MLX weights, or as a PEFT LoRA adapter. It refuses pickles, code and symlinks inside.
pub fn inspect_outgoing(path: &Path, progress: &mut dyn FnMut(u64, u64)) -> Result<Outgoing> {
    let meta = std::fs::symlink_metadata(path).with_context(|| format!("reading {}", path.display()))?;
    if meta.file_type().is_symlink() {
        bail!("{} is a symlink; give the real file or directory", path.display());
    }
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).context("the path has no name")?;
    if meta.is_dir() {
        let found = artifact::inspect_with(&[path.to_path_buf()], &mut |p| {
            progress(p.done, p.total);
            true
        })
        .map_err(|e| anyhow!("{name}: {e}"))?;
        return match (found.kind, &found.format) {
            (ArtifactKind::Model, ArtifactFormat::Safetensors | ArtifactFormat::Mlx) | (ArtifactKind::Adapter { format: AdapterFormat::Peft }, _) => {
                Ok(Outgoing { path: path.to_path_buf(), name, artifact: found })
            }
            (k, _) => bail!("{name} reads as a {}, which is not sent as a directory", kind_name(k)),
        };
    }
    File::open(path).with_context(|| format!("opening {}", path.display()))?;
    if !is_projector(path) {
        let item = inspect_with(path, progress)?;
        return Ok(Outgoing { path: path.to_path_buf(), name: item.file_name, artifact: item.artifact });
    }
    if is_split(&name) {
        bail!("{name} is one shard of a split GGUF; give one projector file");
    }
    let found = artifact::inspect_with(&[path.to_path_buf()], &mut |p| {
        progress(p.done, p.total);
        true
    })
    .map_err(|e| anyhow!("{name}: {e}"))?;
    if found.kind != ArtifactKind::Projector {
        bail!("{name} reads as a {}, not a vision projector", kind_name(found.kind));
    }
    Ok(Outgoing { path: path.to_path_buf(), name, artifact: found })
}

/// The directory an upload of `kind` with `id` goes to under `models_dir`.
pub fn target_dir(models_dir: &Path, kind: ArtifactKind, id: &str) -> Result<PathBuf> {
    check_id(id)?;
    let store = ModelStore::new(models_dir);
    match kind {
        ArtifactKind::Model => Ok(store.model_dir(ModelLayout::Gguf, id)?),
        ArtifactKind::Adapter { .. } => Ok(store.adapter_dir(id)?),
        _ => bail!("projectors are not taken yet"),
    }
}

/// Reads an existing record, if there is one.
pub fn read_provenance(dir: &Path) -> Option<Provenance> {
    Provenance::read(dir)
}

/// What [`import`] did.
#[derive(Debug, Clone, PartialEq)]
pub struct Imported {
    pub dir: PathBuf,
    /// The weights' path in the store.
    pub file: PathBuf,
    pub provenance: Provenance,
    /// The same bytes were already there under this id; nothing was copied.
    pub already: bool,
}

/// The least room always kept free on the models disk: 20 GiB.
pub const KEEP_FREE_MIN: u64 = 20 << 30;

/// What must stay free on a disk of `total` bytes after an upload: 10 percent of it or 20 GiB,
/// whichever is more (m3talsmith, 2026-11-24: no size caps or quotas, but always this much free;
/// `weirdnet-backends`' default `KeepFree`, which a node applies to uploads over Weirdnet too).
pub fn keep_free(total: u64) -> u64 {
    KeepFree::default().of(total)
}

/// The options for one import or upload.
#[derive(Debug, Clone, Default)]
pub struct ImportSpec {
    pub id: String,
    pub license: Option<String>,
    pub base: Option<ProvenanceBase>,
    pub origin: String,
    pub uploaded_ms: u64,
    /// Free and total bytes on the models disk, when known: a local import is refused unless
    /// [`keep_free`] of the disk stays free after it.
    pub disk: Option<(u64, u64)>,
    /// The dataset's summary, when known (a training run's dataset record).
    pub dataset: Option<ProvenanceDataset>,
    /// The trainer, when known (name, build, run).
    pub trainer: Option<ProvenanceTrainer>,
    /// Ymir's own fields in the record (`architecture`, a run's settings, ...).
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// The record for `item` stored as `spec.id`: files, kind and revision from the bytes, the rest
/// from `spec`. The same record goes with an upload to another node.
pub fn provenance(item: &Inspected, spec: &ImportSpec) -> Provenance {
    let mut p = provenance_for(&item.stored(), spec);
    p.extra.insert("architecture".into(), serde_json::Value::String(item.header.architecture.clone()));
    p
}

/// The provenance of `artifact` (as stored) under `spec`.
pub fn provenance_for(artifact: &Artifact, spec: &ImportSpec) -> Provenance {
    let mut p = Provenance::for_artifact(artifact, spec.id.clone(), spec.origin.clone());
    p.license = spec.license.clone();
    if let Some(b) = &spec.base {
        let mut base = b.clone();
        if base.hub.is_none() {
            base.hub = p.base.as_ref().and_then(|h| h.hub.clone());
        }
        p.base = Some(base);
    }
    p.dataset = spec.dataset.clone();
    p.trainer = spec.trainer.clone();
    p.uploaded_ms = Some(spec.uploaded_ms);
    p.extra = spec.extra.clone();
    if let Some(a) = &artifact.architecture {
        p.extra.insert("architecture".into(), serde_json::Value::String(a.clone()));
    }
    p
}

/// Where a local import stages its copy: a directory next to the models directory (on the same
/// disk, outside the store), else the system's temporary directory.
fn staging_dir(models_dir: &Path) -> Result<PathBuf> {
    let name = models_dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "models".into());
    let unique = format!(
        "local-{}-{}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)
    );
    let beside = models_dir.with_file_name(format!("{name}.importing")).join(&unique);
    if std::fs::create_dir_all(&beside).is_ok() {
        return Ok(beside);
    }
    let tmp = std::env::temp_dir().join(format!("ymir-import-{unique}"));
    std::fs::create_dir_all(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
    Ok(tmp)
}

/// Copies `src` to `dest` (a new file), hashing on the way and syncing. Returns the SHA-256.
fn copy_hashed(src: &Path, dest: &Path, progress: &mut dyn FnMut(u64, u64)) -> Result<String> {
    let total = std::fs::metadata(src)?.len();
    let mut input = File::open(src).with_context(|| format!("opening {}", src.display()))?;
    let mut output = std::fs::OpenOptions::new().write(true).create_new(true).open(dest).with_context(|| format!("creating {}", dest.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut done = 0u64;
    loop {
        let n = input.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        output.write_all(&buf[..n])?;
        done += n as u64;
        progress(done, total);
    }
    output.sync_all()?;
    if done != total {
        bail!("{} changed size while it was copied ({total} then {done} bytes); try again when nothing is writing it", src.display());
    }
    Ok(hex::encode(hasher.finalize()))
}

/// The record already under `id` (in its directory, or where U1 put models), if it is for the
/// same bytes; an error if something else is there.
fn already(models_dir: &Path, item: &Inspected, stored: &Artifact, id: &str, dir: &Path) -> Result<Option<Imported>> {
    let mut places = vec![dir.to_path_buf()];
    if item.kind() == ArtifactKind::Model {
        places.push(models_dir.join(LEGACY_UPLOADS_DIR).join(id));
    }
    for d in places {
        if !d.exists() {
            continue;
        }
        if let Some(old) = Provenance::read(&d)
            && old.files.len() == 1
            && old.files[0].path == item.file_name
            && old.files[0].sha256.eq_ignore_ascii_case(&stored.files[0].sha256)
            && std::fs::metadata(d.join(&item.file_name)).is_ok_and(|m| m.len() == stored.files[0].size)
        {
            return Ok(Some(Imported { file: d.join(&item.file_name), dir: d, provenance: old, already: true }));
        }
        bail!("{} already exists with other contents; choose another id with --name, or remove it first", d.display());
    }
    Ok(None)
}

/// Imports a checked file into `models_dir` with [`ModelStore::import`]: copied (and hashed) into
/// a staging directory outside the store, checked again there and moved into place whole, with
/// its record. The same bytes already there under the same id are reported as `already` and not
/// copied again; anything else under that id is refused.
pub fn import(models_dir: &Path, item: &Inspected, spec: &ImportSpec, progress: &mut dyn FnMut(u64, u64)) -> Result<Imported> {
    check_id(&spec.id)?;
    let stored = item.stored();
    let dir = target_dir(models_dir, item.kind(), &spec.id)?;
    if let Some(done) = already(models_dir, item, &stored, &spec.id, &dir)? {
        return Ok(done);
    }
    if let Some((free, total)) = spec.disk {
        let keep = keep_free(total);
        if free < item.size().saturating_add(keep) {
            bail!(
                "not enough free space for {}: {} bytes free, {} needed, and {} must stay free (10 percent of the disk or 20 GiB, whichever is more)",
                item.file_name,
                free,
                item.size(),
                keep
            );
        }
    }
    let staging = staging_dir(models_dir)?;
    let result = (|| -> Result<Imported> {
        let sha = copy_hashed(&item.path, &staging.join(&item.file_name), progress)?;
        if !sha.eq_ignore_ascii_case(item.sha256()) {
            bail!("{} changed while it was being read; try again when nothing is writing it", item.path.display());
        }
        let record = provenance(item, spec);
        let placed = ModelStore::new(models_dir).import(&staging, &stored, &record, &spec.id).map_err(|e| anyhow!("placing {}: {e}", spec.id))?;
        let provenance = placed.provenance().cloned().unwrap_or(record);
        Ok(Imported { file: dir.join(&item.file_name), dir: dir.clone(), provenance, already: false })
    })();
    let _ = std::fs::remove_dir_all(&staging);
    if let Some(parent) = staging.parent() {
        let _ = std::fs::remove_dir(parent);
    }
    result
}

/// The SHA-256 and size of a file.
pub fn sha256_file(path: &Path) -> Result<(String, u64)> {
    let mut f = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut n_total = 0u64;
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        n_total += n as u64;
    }
    Ok((hex::encode(hasher.finalize()), n_total))
}

/// Checks that an adapter's architecture is its base's, reading the base GGUF's header.
pub fn check_adapter_base(adapter: &Inspected, base_file: &Path) -> Result<GgufHeader> {
    let base =
        read_header(std::io::BufReader::new(File::open(base_file).with_context(|| format!("opening the base model {}", base_file.display()))?))
            .with_context(|| format!("reading the base model's header ({})", base_file.display()))?;
    if !matches!(base.kind, GgufKind::Model) {
        bail!("{} is not a model, so nothing can be served on it", base_file.display());
    }
    if !base.architecture.eq_ignore_ascii_case(&adapter.header.architecture) {
        bail!(
            "{} is a LoRA for `{}`, but the base model is `{}`: it was trained on a different model",
            adapter.file_name,
            adapter.header.architecture,
            base.architecture
        );
    }
    Ok(base)
}

pub mod local;

#[cfg(test)]
mod tests;
