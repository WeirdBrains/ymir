use super::*;

/// GGUF header bytes (version 3, no tensors) with these string keys, then `pad` bytes of weights.
fn gguf(kv: &[(&str, &str)], pad: usize) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend(0u64.to_le_bytes());
    out.extend((kv.len() as u64).to_le_bytes());
    let s = |out: &mut Vec<u8>, s: &str| {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    };
    for (k, v) in kv {
        s(&mut out, k);
        out.extend(8u32.to_le_bytes());
        s(&mut out, v);
    }
    out.extend(std::iter::repeat_n(7u8, pad));
    out
}

fn model(arch: &str) -> Vec<u8> {
    gguf(&[("general.architecture", arch), ("general.type", "model")], 4096)
}

fn lora(arch: &str) -> Vec<u8> {
    gguf(&[("general.architecture", arch), ("general.type", "adapter"), ("adapter.type", "lora")], 512)
}

fn write(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, bytes).unwrap();
    p
}

fn spec(id: &str) -> ImportSpec {
    ImportSpec { id: id.into(), license: Some("apache-2.0".into()), base: None, origin: "upload".into(), uploaded_ms: 1, disk: None }
}

#[test]
fn inspect_tells_models_from_lora_adapters() {
    let d = tempfile::tempdir().unwrap();
    let m = inspect(&write(d.path(), "tuned-Q4_K_M.gguf", &model("qwen3"))).unwrap();
    assert_eq!((m.kind, m.header.architecture.as_str()), (ArtifactKind::Model, "qwen3"));
    let a = inspect(&write(d.path(), "support.gguf", &lora("qwen3"))).unwrap();
    assert_eq!(a.kind, ArtifactKind::Adapter { format: AdapterFormat::GgufLora });
}

#[test]
fn inspect_refuses_what_this_slice_does_not_take() {
    let d = tempfile::tempdir().unwrap();
    let err = |p: PathBuf| inspect(&p).unwrap_err().to_string();
    assert!(err(write(d.path(), "w.safetensors", b"{}")).contains("not a .gguf"));
    assert!(err(write(d.path(), "junk.gguf", b"not a gguf at all")).contains("not a readable GGUF"));
    assert!(err(write(d.path(), "big-00001-of-00002.gguf", &model("llama"))).contains("split GGUF"));
    assert!(err(write(d.path(), "mmproj-x.gguf", &model("clip"))).contains("projector"));
    let other = gguf(&[("general.architecture", "llama"), ("general.type", "adapter"), ("adapter.type", "control_vector")], 8);
    assert!(err(write(d.path(), "cv.gguf", &other)).contains("only LoRA"));
    assert!(err(d.path().to_path_buf()).contains("not a file"));
    #[cfg(unix)]
    {
        let real = write(d.path(), "real.gguf", &model("llama"));
        std::os::unix::fs::symlink(&real, d.path().join("link.gguf")).unwrap();
        assert!(err(d.path().join("link.gguf")).contains("symlink"));
    }
}

#[test]
fn a_license_is_recorded_when_given_and_not_required() {
    assert_eq!(check_license(Some("apache-2.0")).unwrap().as_deref(), Some("apache-2.0"));
    assert_eq!(check_license(Some(" MIT ")).unwrap().as_deref(), Some("MIT"));
    assert_eq!(check_license(None).unwrap(), None);
    assert_eq!(check_license(Some("  ")).unwrap(), None);
    assert!(check_license(Some("../../etc")).is_err());
}

#[test]
fn ten_percent_or_twenty_gib_stays_free() {
    assert_eq!(keep_free(100 << 30), KEEP_FREE_MIN);
    assert_eq!(keep_free(1000 << 30), 100 << 30);
}

#[test]
fn ids_are_one_safe_path_segment() {
    for ok in ["qwen3-8b-support-r3", "qwen3-8b+support", "a.b_c"] {
        check_id(ok).unwrap();
    }
    for bad in ["", "..", ".hidden", "a/b", "a\\b", "a b", "c:d"] {
        assert!(check_id(bad).is_err(), "{bad:?}");
    }
}

#[test]
fn import_copies_hashes_and_writes_the_record() {
    let src = tempfile::tempdir().unwrap();
    let models = tempfile::tempdir().unwrap();
    let bytes = model("qwen3");
    let item = inspect(&write(src.path(), "tuned-Q4_K_M.gguf", &bytes)).unwrap();
    let mut ticks = Vec::new();
    let got = import(models.path(), &item, &spec("tuned"), &mut |d, t| ticks.push((d, t))).unwrap();
    assert!(!got.already);
    assert_eq!(got.file, models.path().join("uploads/tuned/tuned-Q4_K_M.gguf"));
    assert_eq!(std::fs::read(&got.file).unwrap(), bytes);
    assert_eq!(ticks.last(), Some(&(bytes.len() as u64, bytes.len() as u64)));
    let sha = hex::encode(Sha256::digest(&bytes));
    let p = read_provenance(&got.dir).unwrap();
    assert_eq!(p, got.provenance);
    assert_eq!(
        (p.format.as_str(), p.kind, p.origin.as_str(), p.license.as_deref()),
        (PROVENANCE_FORMAT, RecordKind::Model, "upload", Some("apache-2.0"))
    );
    assert_eq!(p.files, vec![ArtifactFile { path: "tuned-Q4_K_M.gguf".into(), size: bytes.len() as u64, sha256: sha.clone() }]);
    assert_eq!(p.revision, revision_of(&p.files));
    assert!(p.revision.starts_with("sha256:") && p.revision.len() == 7 + 16);
    assert_eq!(p.architecture.as_deref(), Some("qwen3"));
    // No temporary files are left, and weirdnet-model's store lists the upload as a model.
    let names: Vec<String> = std::fs::read_dir(&got.dir).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
    assert_eq!(names.len(), 2, "{names:?}");
    let listed = weirdnet_model::ModelStore::new(models.path()).list();
    assert!(listed.iter().any(|m| m.id == "tuned-Q4_K_M" && m.files == ["uploads/tuned/tuned-Q4_K_M.gguf"]), "{listed:?}");
}

#[test]
fn an_adapter_lands_where_the_store_finds_adapters() {
    let src = tempfile::tempdir().unwrap();
    let models = tempfile::tempdir().unwrap();
    let item = inspect(&write(src.path(), "support.gguf", &lora("qwen3"))).unwrap();
    let mut s = spec("qwen3-8b+support");
    s.base = Some(BaseRecord { id: Some("qwen3-8b".into()), revision: Some("r1".into()), ..Default::default() });
    let got = import(models.path(), &item, &s, &mut |_, _| {}).unwrap();
    assert_eq!(got.file, models.path().join("adapters/qwen3-8b+support/support.gguf"));
    assert_eq!(got.provenance.base.as_ref().and_then(|b| b.revision.as_deref()), Some("r1"));
    let adapters = weirdnet_model::ModelStore::new(models.path()).adapters();
    assert_eq!(adapters.len(), 1);
    assert_eq!((adapters[0].id.as_str(), adapters[0].architecture.as_deref()), ("qwen3-8b+support", Some("qwen3")));
    assert_eq!(adapters[0].format, AdapterFormat::GgufLora);
    assert_eq!(adapters[0].path, got.file);
    assert!(got.dir.join("adapter.json").is_file());
    assert_eq!(got.provenance.adapter_format, Some(AdapterFormat::GgufLora));
    assert!(weirdnet_model::ModelStore::new(models.path()).list().is_empty(), "an adapter is not a model");
}

#[test]
fn the_same_bytes_again_are_already_there_and_other_bytes_are_refused() {
    let src = tempfile::tempdir().unwrap();
    let models = tempfile::tempdir().unwrap();
    let path = write(src.path(), "m.gguf", &model("llama"));
    let first = import(models.path(), &inspect(&path).unwrap(), &spec("m"), &mut |_, _| {}).unwrap();
    let again = import(models.path(), &inspect(&path).unwrap(), &spec("m"), &mut |_, _| {}).unwrap();
    assert!(again.already);
    assert_eq!(again.provenance, first.provenance);
    let mut changed = model("llama");
    changed.push(1);
    std::fs::write(&path, changed).unwrap();
    let err = import(models.path(), &inspect(&path).unwrap(), &spec("m"), &mut |_, _| {}).unwrap_err().to_string();
    assert!(err.contains("already exists with other contents"), "{err}");
}

#[test]
fn not_enough_space_copies_nothing() {
    let src = tempfile::tempdir().unwrap();
    let models = tempfile::tempdir().unwrap();
    let item = inspect(&write(src.path(), "m.gguf", &model("llama"))).unwrap();
    let mut s = spec("m");
    // A 1000 GiB disk keeps 100 GiB free: 100 GiB free plus a little is not enough for this file.
    s.disk = Some(((100 << 30) + 10, 1000 << 30));
    assert!(import(models.path(), &item, &s, &mut |_, _| {}).unwrap_err().to_string().contains("must stay free"));
    assert!(!models.path().join("uploads/m").exists());
    s.disk = Some(((100 << 30) + (1 << 20), 1000 << 30));
    import(models.path(), &item, &s, &mut |_, _| {}).unwrap();
}

#[test]
fn adapters_must_match_their_base() {
    let d = tempfile::tempdir().unwrap();
    let base = write(d.path(), "base.gguf", &model("qwen3"));
    let other = write(d.path(), "other.gguf", &model("llama"));
    let a = inspect(&write(d.path(), "a.gguf", &lora("qwen3"))).unwrap();
    check_adapter_base(&a, &base).unwrap();
    assert!(check_adapter_base(&a, &other).unwrap_err().to_string().contains("different model"));
    let not_model = write(d.path(), "l2.gguf", &lora("qwen3"));
    assert!(check_adapter_base(&a, &not_model).unwrap_err().to_string().contains("not a model"));
}

#[test]
fn the_revision_does_not_depend_on_file_order() {
    let f = |p: &str, s: &str| ArtifactFile { path: p.into(), size: 1, sha256: s.into() };
    assert_eq!(revision_of(&[f("a", "1"), f("b", "2")]), revision_of(&[f("b", "2"), f("a", "1")]));
    assert_ne!(revision_of(&[f("a", "1")]), revision_of(&[f("a", "2")]));
}
