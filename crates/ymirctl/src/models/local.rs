use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use weirdnet_model::{ModelStore, StoredModel};
use ymir_core::NodeConfig;
use ymir_core::config::LocalModel;

use super::plan::id_for_file;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Gguf,
    Mlx,
}

/// One downloaded model: a GGUF (all its shards) or an MLX directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub id: String,
    pub kind: Kind,
    /// The files to delete (for MLX, the one directory).
    pub paths: Vec<PathBuf>,
    /// The file a `file =` line points at (first shard, or the MLX directory).
    pub main: PathBuf,
    pub size: u64,
    /// Bytes of an unfinished `.part` download, if that is all there is.
    pub partial: Option<u64>,
    /// The id of the `[[model.local]]` that uses it.
    pub registered: Option<String>,
    /// The store's record of a GGUF, which deletes it with `ModelStore::remove_local`.
    pub stored: Option<StoredModel>,
}

fn dir_size(p: &Path) -> u64 {
    std::fs::read_dir(p)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| match e.metadata() {
            Ok(m) if m.is_dir() => dir_size(&e.path()),
            Ok(m) => m.len(),
            Err(_) => 0,
        })
        .sum()
}

/// The path a `[[model.local]]` points at, or the catalog default for its id.
fn resolved(m: &LocalModel, dir: &Path) -> Option<PathBuf> {
    match &m.file {
        Some(f) => {
            let p = PathBuf::from(f);
            Some(if p.is_absolute() { p } else { dir.join(p) })
        }
        None => weirdnet_model::hw::find(&m.id).map(|c| dir.join(&c.spec.file)),
    }
}

/// `.part` files (unfinished downloads) in `dir` and one folder down.
fn part_files(dir: &Path, depth: usize, out: &mut Vec<(PathBuf, u64)>) {
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let (path, name) = (e.path(), e.file_name().to_string_lossy().to_ascii_lowercase());
        let Ok(meta) = e.metadata() else { continue };
        if meta.is_file() && name.ends_with(".gguf.part") {
            out.push((path, meta.len()));
        } else if meta.is_dir() && depth < 1 && name != "mlx" && !e.file_type().is_ok_and(|t| t.is_symlink()) {
            part_files(&path, depth + 1, out);
        }
    }
}

pub fn scan(store: &ModelStore, cfg: Option<&NodeConfig>) -> Vec<Entry> {
    let dir = store.dir();
    let mut out = Vec::new();
    for m in store.list() {
        let paths: Vec<PathBuf> = m.files.iter().map(|f| dir.join(f)).collect();
        out.push(Entry {
            id: m.id.to_ascii_lowercase(),
            kind: Kind::Gguf,
            size: m.size,
            main: paths[0].clone(),
            paths,
            partial: None,
            registered: None,
            stored: Some(m),
        });
    }
    let mut parts = Vec::new();
    part_files(dir, 0, &mut parts);
    for (path, len) in parts {
        let whole = PathBuf::from(path.to_string_lossy().trim_end_matches(".part").to_string());
        if out.iter().any(|e| e.paths.contains(&whole)) {
            continue;
        }
        out.push(Entry {
            id: id_for_file(&whole.to_string_lossy()),
            kind: Kind::Gguf,
            paths: vec![path.clone()],
            main: path,
            size: len,
            partial: Some(len),
            registered: None,
            stored: None,
        });
    }
    for e in std::fs::read_dir(dir.join("mlx")).into_iter().flatten().flatten() {
        if e.metadata().is_ok_and(|m| m.is_dir()) && !e.file_type().is_ok_and(|t| t.is_symlink()) {
            let path = e.path();
            out.push(Entry {
                id: e.file_name().to_string_lossy().to_ascii_lowercase(),
                kind: Kind::Mlx,
                size: dir_size(&path),
                paths: vec![path.clone()],
                main: path,
                partial: None,
                registered: None,
                stored: None,
            });
        }
    }
    if let Some(cfg) = cfg {
        for e in &mut out {
            e.registered = cfg.model.local.iter().find(|m| resolved(m, dir).is_some_and(|p| p == e.main)).map(|m| m.id.clone());
        }
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

/// The entry named by `key`: a registered id, a derived id or a file name.
pub fn find<'a>(entries: &'a [Entry], key: &str) -> Result<&'a Entry> {
    let k = key.to_ascii_lowercase();
    let hits: Vec<&Entry> = entries
        .iter()
        .filter(|e| {
            e.id == k
                || e.registered.as_deref().is_some_and(|r| r.eq_ignore_ascii_case(key))
                || e.paths.iter().any(|p| p.file_name().is_some_and(|n| n.to_string_lossy().eq_ignore_ascii_case(key)))
        })
        .collect();
    match hits.as_slice() {
        [one] => Ok(one),
        [] => bail!("no downloaded model named `{key}` (see `ymirctl models list`)"),
        many => {
            bail!("`{key}` matches {} models ({}); use a file name", many.len(), many.iter().map(|e| e.id.as_str()).collect::<Vec<_>>().join(", "))
        }
    }
}

/// Deletes the entry's files (and the `.part` of each). Files only: `node.toml` is not touched.
pub fn remove(store: &ModelStore, e: &Entry) -> Result<()> {
    if let Some(m) = &e.stored {
        store.remove_local(m).with_context(|| format!("deleting {}", e.id))?;
        return Ok(());
    }
    let dir = store.dir();
    let root = std::fs::canonicalize(dir)?;
    for p in &e.paths {
        let meta = std::fs::symlink_metadata(p).with_context(|| format!("reading {}", p.display()))?;
        let parent = std::fs::canonicalize(p.parent().unwrap_or(dir))?;
        if !parent.starts_with(&root) {
            bail!("{} is outside the model directory; not deleting it", p.display());
        }
        if meta.file_type().is_symlink() {
            bail!("{} is a symlink; not deleting through it", p.display());
        }
        if meta.is_dir() {
            std::fs::remove_dir_all(p)?;
        } else {
            std::fs::remove_file(p)?;
            let _ = std::fs::remove_file(format!("{}.part", p.display()));
        }
    }
    Ok(())
}

/// Appends `stanza` to `node.toml` (after saving `node.toml.bak`), keeping every other byte of
/// the file. Refuses a duplicate id, and writes nothing if the result would not load.
pub fn register(config: &Path, id: &str, stanza: &str) -> Result<()> {
    ymir_core::config_edit::append_model(config, id, stanza, "# Added by `ymirctl models pull --register`.")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(p: &Path, n: usize) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, vec![1u8; n]).unwrap();
    }

    fn cfg_with(dir: &Path, local: &[(&str, &str)]) -> NodeConfig {
        let mut c = NodeConfig::default();
        for (id, file) in local {
            c.model.local.push(LocalModel { id: (*id).into(), file: Some((*file).into()), ..Default::default() });
        }
        c.model.models_dir = Some(dir.into());
        c
    }

    #[test]
    fn lists_files_shards_partials_and_mlx_and_marks_registered() {
        let d = tempfile::tempdir().unwrap();
        touch(&d.path().join("a-Q8_0.gguf"), 10);
        touch(&d.path().join("big-Q4_K_M-00001-of-00002.gguf"), 5);
        touch(&d.path().join("big-Q4_K_M-00002-of-00002.gguf"), 6);
        touch(&d.path().join("half-Q4_K_M.gguf.part"), 3);
        touch(&d.path().join("mlx/Some-Model-4bit/model.safetensors"), 7);
        let cfg = cfg_with(d.path(), &[("alpha", "a-Q8_0.gguf"), ("big", "big-Q4_K_M-00001-of-00002.gguf")]);
        let e = scan(&ModelStore::new(d.path()), Some(&cfg));
        let row = |id: &str| e.iter().find(|x| x.id == id).unwrap();
        assert_eq!(row("a-q8_0").registered.as_deref(), Some("alpha"));
        assert_eq!((row("big-q4_k_m").size, row("big-q4_k_m").paths.len(), row("big-q4_k_m").registered.as_deref()), (11, 2, Some("big")));
        assert_eq!(row("half-q4_k_m").partial, Some(3));
        assert_eq!((row("some-model-4bit").kind, row("some-model-4bit").size), (Kind::Mlx, 7));
        assert!(scan(&ModelStore::new(d.path()), None).iter().all(|x| x.registered.is_none()));
    }

    #[test]
    fn find_matches_ids_registered_names_and_files() {
        let d = tempfile::tempdir().unwrap();
        touch(&d.path().join("a-Q8_0.gguf"), 1);
        touch(&d.path().join("b-Q8_0.gguf"), 1);
        let cfg = cfg_with(d.path(), &[("alpha", "a-Q8_0.gguf")]);
        let e = scan(&ModelStore::new(d.path()), Some(&cfg));
        assert_eq!(find(&e, "alpha").unwrap().id, "a-q8_0");
        assert_eq!(find(&e, "B-Q8_0").unwrap().id, "b-q8_0");
        assert_eq!(find(&e, "b-Q8_0.gguf").unwrap().id, "b-q8_0");
        assert!(find(&e, "nope").unwrap_err().to_string().contains("no downloaded model"));
    }

    #[test]
    fn remove_deletes_the_files_and_shards_but_not_other_files() {
        let d = tempfile::tempdir().unwrap();
        touch(&d.path().join("big-Q4_K_M-00001-of-00002.gguf"), 5);
        touch(&d.path().join("big-Q4_K_M-00002-of-00002.gguf"), 6);
        touch(&d.path().join("keep.gguf"), 1);
        touch(&d.path().join("mlx/m/w.safetensors"), 1);
        let e = scan(&ModelStore::new(d.path()), None);
        remove(&ModelStore::new(d.path()), find(&e, "big-q4_k_m").unwrap()).unwrap();
        remove(&ModelStore::new(d.path()), find(&e, "m").unwrap()).unwrap();
        let left: Vec<_> = std::fs::read_dir(d.path()).unwrap().flatten().map(|x| x.file_name().to_string_lossy().into_owned()).collect();
        assert!(left.contains(&"keep.gguf".to_string()) && !left.iter().any(|n| n.starts_with("big")), "{left:?}");
        assert!(!d.path().join("mlx/m").exists());
    }

    #[cfg(unix)]
    #[test]
    fn remove_does_not_follow_a_symlink() {
        let d = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        touch(&outside.path().join("real.gguf"), 1);
        std::os::unix::fs::symlink(outside.path().join("real.gguf"), d.path().join("link.gguf")).unwrap();
        // A symlink is not listed, so `remove` can never be pointed at it or through it.
        let e = scan(&ModelStore::new(d.path()), None);
        assert!(e.is_empty());
        let fake = Entry {
            id: "link".into(),
            kind: Kind::Gguf,
            paths: vec![d.path().join("link.gguf")],
            main: d.path().join("link.gguf"),
            size: 1,
            partial: None,
            registered: None,
            stored: None,
        };
        assert!(remove(&ModelStore::new(d.path()), &fake).unwrap_err().to_string().contains("symlink"));
        assert!(outside.path().join("real.gguf").exists());
    }

    #[test]
    fn register_appends_keeps_the_rest_and_refuses_duplicates() {
        let d = tempfile::tempdir().unwrap();
        let path = d.path().join("node.toml");
        let original = "# my notes\n[node]\nname = \"n\"\n\n[[model.local]]\nid = \"old\"\n";
        std::fs::write(&path, original).unwrap();
        let m = super::super::plan::local_model("new", "llamacpp", "chat", "/x/new.gguf", Some(4096), false);
        register(&path, "new", &super::super::plan::stanza(&m)).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.starts_with(original) && text.contains("id = \"new\""));
        assert_eq!(std::fs::read_to_string(d.path().join("node.toml.bak")).unwrap(), original);
        assert_eq!(toml::from_str::<NodeConfig>(&text).unwrap().model.local.len(), 2);
        assert!(register(&path, "new", "x").unwrap_err().to_string().contains("already has"));
        assert!(register(&path, "old", "x").is_err());
    }

    #[test]
    fn register_writes_nothing_when_the_result_would_not_parse() {
        let d = tempfile::tempdir().unwrap();
        let path = d.path().join("node.toml");
        std::fs::write(&path, "[node]\n").unwrap();
        assert!(register(&path, "x", "[[model.local]\nbroken").is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "[node]\n");
        assert!(!d.path().join("node.toml.bak").exists());
    }
}
