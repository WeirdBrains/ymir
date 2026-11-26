//! U9: an upload placed in this machine's models directory, the way a node places one sent over
//! Weirdnet: a directory of safetensors or MLX weights, a PEFT adapter or a GGUF vision projector
//! (what [`super::inspect_outgoing`] takes), for `ymirctl models upload` without `--node` on a
//! machine that serves models. The files are copied to a staging directory beside the store,
//! checked against their hashes and moved into place with their provenance.

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use weirdnet_backends::models::{StoredEntry, Upload};
use weirdnet_model::ModelStore;

/// What [`place`] did.
#[derive(Debug, Clone)]
pub struct Placed {
    /// The entry as the store lists it now.
    pub entry: StoredEntry,
    pub revision: String,
    /// The store already held these exact files under this id; nothing was copied.
    pub already: bool,
}

/// The models, adapters and projectors in `store`, as a node's models/1 lists them (not served).
pub fn entries(store: &ModelStore) -> Vec<StoredEntry> {
    // `StoredEntry` is non-exhaustive and has no public constructor, so it is built from its
    // serialized form (the fields left out are the served state, `None` here).
    let entry = |v: serde_json::Value| serde_json::from_value::<StoredEntry>(v).ok();
    let models = store.list().into_iter().chain(store.dir_models()).filter_map(|m| entry(serde_json::json!({ "model": m })));
    let adapters = store.adapters().into_iter().filter_map(|a| entry(serde_json::json!({ "adapter": a })));
    let projectors = store.projectors().into_iter().filter_map(|p| entry(serde_json::json!({ "projector": p })));
    models.chain(adapters).chain(projectors).collect()
}

/// Places `upload` (made from the file or directory `local`) in `models_dir`. `room` is that
/// disk's free bytes and the bytes that must stay free after the copy (`None`: unknown, not
/// checked). `progress` gets bytes copied and the total.
pub fn place(models_dir: &Path, upload: &Upload, local: &Path, room: Option<(u64, u64)>, progress: &mut dyn FnMut(u64, u64)) -> Result<Placed> {
    let store = ModelStore::new(models_dir);
    let id = upload.id.as_str();
    let revision = weirdnet_model::artifact::revision(&upload.files);
    if let Some(e) = entries(&store).into_iter().find(|e| e.id() == id) {
        if e.provenance().is_some_and(|p| p.revision == revision) {
            return Ok(Placed { entry: e, revision, already: true });
        }
        bail!("the models directory already has `{id}`; choose another id with --name");
    }
    let total: u64 = upload.files.iter().map(|f| f.size).sum();
    if let Some((free, keep)) = room
        && free < total.saturating_add(keep)
    {
        bail!(
            "not enough free space for `{id}`: {free} bytes free, {total} needed, and {keep} must stay free ([model] upload_keep_free; the larger of 10 percent of the disk and 20 GiB by default)"
        );
    }
    let staging = super::staging_dir(models_dir)?;
    let result = (|| -> Result<()> {
        let mut done = 0u64;
        for f in &upload.files {
            let src = if local.is_dir() { local.join(&f.path) } else { local.to_path_buf() };
            let dest = staging.join(&f.path);
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
            }
            let base = done;
            let sha = super::copy_hashed(&src, &dest, &mut |d, _| progress(base + d, total))?;
            if !sha.eq_ignore_ascii_case(&f.sha256) {
                bail!("{} changed while it was being read; try again when nothing is writing it", src.display());
            }
            done += f.size;
        }
        store.import_files(&staging, &upload.files, &upload.provenance, id, false, &mut |_| true).map_err(|e| anyhow!("placing {id}: {e}"))?;
        Ok(())
    })();
    let _ = std::fs::remove_dir_all(&staging);
    if let Some(parent) = staging.parent() {
        let _ = std::fs::remove_dir(parent);
    }
    result?;
    let entry = entries(&store).into_iter().find(|e| e.id() == id).with_context(|| format!("`{id}` was placed but the store does not list it"))?;
    Ok(Placed { entry, revision, already: false })
}
