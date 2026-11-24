//! Uploading a model trained elsewhere (docs/design/model-upload.md, slice U1): check a local GGUF
//! model or GGUF LoRA adapter, hash it, import it into the models directory atomically, and write a
//! `provenance.json` beside it saying where it came from.
//!
//! Interim: the artifact inspection and the [`Provenance`] record are generic and are to move to
//! `weirdnet-model` (request A to the Weirdnet lead, 2026-11-24: `artifact::inspect`,
//! `Provenance`, `ModelStore::import`). The record is shaped like the one requested so Ymir can
//! switch to it without changing what is on disk. Keep this module free of other Ymir types.
//!
//! Layout: a model goes to `models_dir/uploads/<id>/<file>.gguf` (which `ModelStore::list` finds);
//! an adapter to `models_dir/adapters/<id>/<file>.gguf` (which `ModelStore::adapters` finds). The
//! record is `provenance.json` in the same directory. Nothing is ever executed, and only GGUF files
//! are taken in this slice: split GGUFs, projectors, safetensors, MLX and PEFT come later (U7).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use weirdnet_model::gguf::{GgufHeader, GgufKind, read_header};
use weirdnet_model::hf::AdapterFormat;

/// The record's format name.
pub const PROVENANCE_FORMAT: &str = "ymir-provenance/1";
/// The record's file name, beside the weights.
pub const PROVENANCE_FILE: &str = "provenance.json";
/// Where uploaded models go under the models directory.
pub const UPLOADS_DIR: &str = "uploads";
/// Where adapters go under the models directory (as `weirdnet-model`'s store).
pub const ADAPTERS_DIR: &str = "adapters";

/// What an uploaded file is, shaped like the `weirdnet-model` `ArtifactKind` the Weirdnet lead
/// accepted (request A, model 0.5.3): projectors are their own kind (`GgufKind::Projector` there;
/// here, a model header whose architecture is `clip`). U1 takes models and GGUF LoRA adapters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactKind {
    Model,
    Adapter { format: AdapterFormat },
    Projector,
}

impl ArtifactKind {
    pub fn name(self) -> &'static str {
        match self {
            ArtifactKind::Model => "model",
            ArtifactKind::Adapter { .. } => "LoRA adapter",
            ArtifactKind::Projector => "vision projector",
        }
    }

    /// The kind as the record writes it.
    pub fn record(self) -> RecordKind {
        match self {
            ArtifactKind::Model => RecordKind::Model,
            ArtifactKind::Adapter { .. } => RecordKind::Adapter,
            ArtifactKind::Projector => RecordKind::Projector,
        }
    }

    pub fn is_adapter(self) -> bool {
        matches!(self, ArtifactKind::Adapter { .. })
    }
}

/// `kind` in `provenance.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RecordKind {
    Model,
    Adapter,
    Projector,
}

/// One file of an artifact, relative to its directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactFile {
    pub path: String,
    pub size: u64,
    pub sha256: String,
}

/// The base model an adapter (or a fine-tune) was made from.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BaseRecord {
    /// The base's model id in this pool, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The base as a Hub id (`owner/name`), from the GGUF header when it has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hub: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

/// Where an upload came from: what the uploader said, checked against the bytes where it can be.
/// Metadata for people and rollouts, not a trust anchor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    pub format: String,
    pub kind: RecordKind,
    /// For an adapter: `gguf_lora` or `peft`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_format: Option<AdapterFormat>,
    pub id: String,
    pub files: Vec<ArtifactFile>,
    /// `sha256:` and 16 hex digits over the files ([`revision_of`]); the entry's `revision`.
    pub revision: String,
    /// `upload`, `train` or `hub`.
    pub origin: String,
    /// An SPDX id or a license name, when the uploader gave one (not required: m3talsmith,
    /// 2026-11-24).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    /// `general.architecture` from the header.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base: Option<BaseRecord>,
    /// A dataset manifest's summary (format, hash, counts, licenses), when given (later slices).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset: Option<serde_json::Value>,
    /// The trainer (name, build, run), when known (later slices).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trainer: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uploaded_by: Option<String>,
    pub uploaded_ms: i64,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// A local GGUF checked for upload, before anything is copied.
#[derive(Debug, Clone)]
pub struct Inspected {
    pub path: PathBuf,
    /// The file name it keeps in the store.
    pub file_name: String,
    pub kind: ArtifactKind,
    pub header: GgufHeader,
    pub size: u64,
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
    let Some(stem) = stem.strip_suffix(".gguf") else { return false };
    let mut parts = stem.rsplitn(4, '-');
    let (Some(of), Some(word), Some(part)) = (parts.next(), parts.next(), parts.next()) else { return false };
    word == "of" && !of.is_empty() && of.chars().all(|c| c.is_ascii_digit()) && !part.is_empty() && part.chars().all(|c| c.is_ascii_digit())
}

/// Reads and checks a local file for upload: a regular file (not a symlink) named `*.gguf`, not a
/// split shard, whose header is a model or a LoRA adapter. Projectors (`clip`) are refused for now.
pub fn inspect(path: &Path) -> Result<Inspected> {
    let meta = std::fs::symlink_metadata(path).with_context(|| format!("reading {}", path.display()))?;
    if meta.file_type().is_symlink() {
        bail!("{} is a symlink; give the real file", path.display());
    }
    if !meta.is_file() {
        bail!("{} is not a file (this slice takes one GGUF file)", path.display());
    }
    let file_name = path.file_name().map(|n| n.to_string_lossy().into_owned()).context("the path has no file name")?;
    if !file_name.to_ascii_lowercase().ends_with(".gguf") {
        bail!("{file_name} is not a .gguf file: this slice takes GGUF models and GGUF LoRA adapters (safetensors, MLX and PEFT come later)");
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
    let kind = match &header.kind {
        GgufKind::Model if header.architecture.eq_ignore_ascii_case("clip") || header.architecture.eq_ignore_ascii_case("mmproj") => {
            bail!("{file_name} is a vision projector (mmproj); projectors are not taken yet")
        }
        GgufKind::Model => ArtifactKind::Model,
        k if k.is_lora() => ArtifactKind::Adapter { format: AdapterFormat::GgufLora },
        GgufKind::Adapter { adapter_type, .. } => bail!("{file_name} is a `{adapter_type}` adapter; only LoRA adapters are served"),
        _ => bail!("{file_name} is a kind of GGUF this version does not take"),
    };
    Ok(Inspected { path: path.to_path_buf(), file_name, kind, header, size: meta.len() })
}

/// The revision of a set of files: `sha256:` and the first 16 hex digits of the SHA-256 of their
/// sorted `path:sha256` lines.
pub fn revision_of(files: &[ArtifactFile]) -> String {
    let mut lines: Vec<String> = files.iter().map(|f| format!("{}:{}\n", f.path, f.sha256)).collect();
    lines.sort();
    let digest = Sha256::digest(lines.concat().as_bytes());
    format!("sha256:{}", &hex::encode(digest)[..16])
}

/// The directory an upload of `kind` with `id` goes to under `models_dir`.
pub fn target_dir(models_dir: &Path, kind: ArtifactKind, id: &str) -> Result<PathBuf> {
    check_id(id)?;
    let sub = match kind {
        ArtifactKind::Model => UPLOADS_DIR,
        ArtifactKind::Adapter { .. } => ADAPTERS_DIR,
        ArtifactKind::Projector => bail!("projectors are not taken yet"),
    };
    Ok(models_dir.join(sub).join(id))
}

/// Reads an existing record, if there is one.
pub fn read_provenance(dir: &Path) -> Option<Provenance> {
    serde_json::from_slice(&std::fs::read(dir.join(PROVENANCE_FILE)).ok()?).ok()
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
/// whichever is more (m3talsmith, 2026-11-24: no size caps or quotas, but always this much free).
pub fn keep_free(total: u64) -> u64 {
    (total / 10).max(KEEP_FREE_MIN)
}

/// Copies `src` into `dir` (created) as `name`, hashing on the way: into a hidden temporary file in
/// the same directory, synced, then moved into place without replacing anything. Returns the
/// SHA-256 and size. The temporary file is removed on any failure.
fn copy_hashed(src: &Path, dir: &Path, name: &str, progress: &mut dyn FnMut(u64, u64)) -> Result<(String, u64)> {
    let total = std::fs::metadata(src)?.len();
    let tmp = dir.join(format!(".{name}.upload-{}", std::process::id()));
    let result = (|| -> Result<(String, u64)> {
        let mut input = File::open(src).with_context(|| format!("opening {}", src.display()))?;
        let mut output =
            std::fs::OpenOptions::new().write(true).create_new(true).open(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
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
        let dest = dir.join(name);
        // A hard link never replaces an existing file, so two imports can't clobber each other.
        if let Err(e) = std::fs::hard_link(&tmp, &dest) {
            if dest.exists() {
                bail!("{} already exists", dest.display());
            }
            // Filesystems without hard links: rename (checked above that nothing is there).
            std::fs::rename(&tmp, &dest).with_context(|| format!("moving into place ({e})"))?;
        }
        Ok((hex::encode(hasher.finalize()), done))
    })();
    let _ = std::fs::remove_file(&tmp);
    result
}

/// The options for one import.
#[derive(Debug, Clone, Default)]
pub struct ImportSpec {
    pub id: String,
    pub license: Option<String>,
    pub base: Option<BaseRecord>,
    pub origin: String,
    pub uploaded_ms: i64,
    /// Free and total bytes on the models disk, when known: the import is refused unless
    /// [`keep_free`] of the disk stays free after it.
    pub disk: Option<(u64, u64)>,
    /// The dataset's summary for the record, when known (a training run's dataset record).
    pub dataset: Option<serde_json::Value>,
    /// The trainer for the record, when known (name, build, run, ...).
    pub trainer: Option<serde_json::Value>,
}

/// Imports a checked file into `models_dir`: into its target directory, atomically, then writes the
/// record (through a temporary file). The same bytes already there under the same id are reported
/// as `already` and not copied again; anything else under that id is refused.
pub fn import(models_dir: &Path, item: &Inspected, spec: &ImportSpec, progress: &mut dyn FnMut(u64, u64)) -> Result<Imported> {
    let dir = target_dir(models_dir, item.kind, &spec.id)?;
    let dest = dir.join(&item.file_name);
    if dir.exists() {
        if let Some(old) = read_provenance(&dir)
            && old.files.len() == 1
            && old.files[0].path == item.file_name
            && dest.is_file()
            && std::fs::metadata(&dest).is_ok_and(|m| m.len() == item.size)
        {
            let (sha, _) = sha256_file(&item.path)?;
            if old.files[0].sha256 == sha {
                return Ok(Imported { dir, file: dest, provenance: old, already: true });
            }
        }
        bail!("{} already exists with other contents; choose another id with --name, or remove it first", dir.display());
    }
    if let Some((free, total)) = spec.disk {
        let keep = keep_free(total);
        if free < item.size.saturating_add(keep) {
            bail!(
                "not enough free space for {}: {} bytes free, {} needed, and {} must stay free (10 percent of the disk or 20 GiB, whichever is more)",
                item.file_name,
                free,
                item.size,
                keep
            );
        }
    }
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let copied = copy_hashed(&item.path, &dir, &item.file_name, progress);
    let (sha256, size) = match copied {
        Ok(v) => v,
        Err(e) => {
            let _ = std::fs::remove_dir(&dir);
            return Err(e);
        }
    };
    let files = vec![ArtifactFile { path: item.file_name.clone(), size, sha256 }];
    let provenance = Provenance {
        format: PROVENANCE_FORMAT.to_string(),
        kind: item.kind.record(),
        adapter_format: match item.kind {
            ArtifactKind::Adapter { format } => Some(format),
            _ => None,
        },
        id: spec.id.clone(),
        revision: revision_of(&files),
        files,
        origin: spec.origin.clone(),
        license: spec.license.clone(),
        architecture: Some(item.header.architecture.clone()),
        base: spec.base.clone(),
        dataset: spec.dataset.clone(),
        trainer: spec.trainer.clone(),
        uploaded_by: None,
        uploaded_ms: spec.uploaded_ms,
        extra: BTreeMap::new(),
    };
    let tmp = dir.join(format!(".{PROVENANCE_FILE}.tmp"));
    let written = serde_json::to_vec_pretty(&provenance)
        .map_err(anyhow::Error::from)
        .and_then(|b| std::fs::write(&tmp, b).map_err(anyhow::Error::from))
        .and_then(|()| std::fs::rename(&tmp, dir.join(PROVENANCE_FILE)).map_err(anyhow::Error::from));
    if let Err(e) = written {
        let _ = std::fs::remove_file(&tmp);
        let _ = std::fs::remove_file(&dest);
        let _ = std::fs::remove_dir(&dir);
        return Err(e.context("writing provenance.json"));
    }
    if let ArtifactKind::Adapter { format } = item.kind
        && let Err(e) = write_adapter_info(&dir, &item.file_name, format, &item.header, spec.base.as_ref())
    {
        let _ = std::fs::remove_file(dir.join(PROVENANCE_FILE));
        let _ = std::fs::remove_file(&dest);
        let _ = std::fs::remove_dir(&dir);
        return Err(e.context("writing adapter.json"));
    }
    Ok(Imported { dir, file: dest, provenance, already: false })
}

/// `adapter.json`, in the shape `weirdnet-model`'s store writes for downloaded adapters and reads
/// back in `ModelStore::adapters` (id, format, file name, base model, architecture, repo, rank,
/// alpha), so an uploaded adapter lists like a pulled one. `ModelStore::import` writes it itself
/// once request A ships.
fn write_adapter_info(dir: &Path, file: &str, format: AdapterFormat, header: &GgufHeader, base: Option<&BaseRecord>) -> Result<()> {
    let alpha = match &header.kind {
        GgufKind::Adapter { alpha, .. } => *alpha,
        _ => None,
    };
    let id = dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let info = serde_json::json!({
        "id": id,
        "format": format,
        "file": file,
        "base_model": header.base_model.clone().or_else(|| base.and_then(|b| b.hub.clone())),
        "architecture": header.architecture,
        "repo": header.repo,
        "rank": serde_json::Value::Null,
        "alpha": alpha,
    });
    let tmp = dir.join(".adapter.json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(&info)?)?;
    std::fs::rename(&tmp, dir.join(weirdnet_model::store::ADAPTER_INFO))?;
    Ok(())
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

#[cfg(test)]
mod tests;
