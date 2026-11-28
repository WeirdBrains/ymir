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
    ImportSpec {
        id: id.into(),
        license: Some("apache-2.0".into()),
        base: None,
        origin: "upload".into(),
        uploaded_ms: 1,
        disk: None,
        ..Default::default()
    }
}

#[test]
fn inspect_tells_models_from_lora_adapters() {
    let d = tempfile::tempdir().unwrap();
    let m = inspect(&write(d.path(), "tuned-Q4_K_M.gguf", &model("qwen3"))).unwrap();
    assert_eq!((m.kind(), m.header.architecture.as_str()), (ArtifactKind::Model, "qwen3"));
    assert_eq!(m.sha256(), hex::encode(Sha256::digest(model("qwen3"))));
    let a = inspect(&write(d.path(), "support.gguf", &lora("qwen3"))).unwrap();
    assert_eq!(a.kind(), ArtifactKind::Adapter { format: AdapterFormat::GgufLora });
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
    assert_eq!(got.file, models.path().join("tuned/tuned-Q4_K_M.gguf"));
    assert_eq!(std::fs::read(&got.file).unwrap(), bytes);
    assert_eq!(ticks.last(), Some(&(bytes.len() as u64, bytes.len() as u64)));
    let sha = hex::encode(Sha256::digest(&bytes));
    let p = read_provenance(&got.dir).unwrap();
    assert_eq!(p, got.provenance);
    assert_eq!(
        (p.format.as_str(), p.kind, p.origin.as_str(), p.license.as_deref()),
        (PROVENANCE_FORMAT, ProvenanceKind::Model, "upload", Some("apache-2.0"))
    );
    assert_eq!(p.files, vec![ArtifactFile { path: "tuned-Q4_K_M.gguf".into(), size: bytes.len() as u64, sha256: sha.clone() }]);
    assert_eq!(p.revision, weirdnet_model::artifact::revision(&p.files));
    assert!(p.revision.starts_with("sha256:") && p.revision.len() == 7 + 16);
    assert_eq!(p.extra.get("architecture").and_then(|v| v.as_str()), Some("qwen3"));
    assert_eq!(p.uploaded_ms, Some(1));
    // No temporary files are left (in the entry, the store or the staging area), and
    // weirdnet-model's store lists the upload as a model with its record.
    let names: Vec<String> = std::fs::read_dir(&got.dir).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
    assert_eq!(names.len(), 2, "{names:?}");
    let top: Vec<String> = std::fs::read_dir(models.path()).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
    assert_eq!(top, ["tuned"], "{top:?}");
    assert!(!models.path().with_file_name(format!("{}.importing", models.path().file_name().unwrap().to_string_lossy())).exists());
    let listed = weirdnet_model::ModelStore::new(models.path()).list();
    assert!(listed.iter().any(|m| m.files == ["tuned/tuned-Q4_K_M.gguf"] && m.provenance.is_some()), "{listed:?}");
    // The source is left alone.
    assert_eq!(std::fs::read(&item.path).unwrap(), bytes);
}

#[test]
fn an_adapter_lands_where_the_store_finds_adapters() {
    let src = tempfile::tempdir().unwrap();
    let models = tempfile::tempdir().unwrap();
    let item = inspect(&write(src.path(), "support.gguf", &lora("qwen3"))).unwrap();
    let mut s = spec("qwen3-8b+support");
    let mut base = ProvenanceBase::default();
    base.id = Some("qwen3-8b".into());
    base.revision = Some("r1".into());
    s.base = Some(base);
    let got = import(models.path(), &item, &s, &mut |_, _| {}).unwrap();
    assert_eq!(got.file, models.path().join("adapters/qwen3-8b+support/support.gguf"));
    assert_eq!(got.provenance.base.as_ref().and_then(|b| b.revision.as_deref()), Some("r1"));
    let adapters = weirdnet_model::ModelStore::new(models.path()).adapters();
    assert_eq!(adapters.len(), 1);
    assert_eq!((adapters[0].id.as_str(), adapters[0].architecture.as_deref()), ("qwen3-8b+support", Some("qwen3")));
    assert_eq!(adapters[0].format, AdapterFormat::GgufLora);
    assert_eq!(adapters[0].path, got.file);
    assert!(got.dir.join("adapter.json").is_file());
    assert_eq!(got.provenance.kind, ProvenanceKind::Adapter);
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
    assert!(!models.path().join("m").exists());
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
fn a_model_uploaded_before_the_move_is_still_already_there() {
    let src = tempfile::tempdir().unwrap();
    let models = tempfile::tempdir().unwrap();
    let path = write(src.path(), "m.gguf", &model("llama"));
    let item = inspect(&path).unwrap();
    // As U1 left it: models_dir/uploads/m/ with a ymir-provenance/1 record.
    let old = models.path().join("uploads/m");
    std::fs::create_dir_all(&old).unwrap();
    std::fs::copy(&path, old.join("m.gguf")).unwrap();
    let rec = serde_json::json!({"format": "ymir-provenance/1", "kind": "model", "id": "m", "origin": "upload", "uploaded_ms": 5,
        "revision": item.artifact.revision, "files": item.artifact.files, "architecture": "llama"});
    std::fs::write(old.join(PROVENANCE_FILE), serde_json::to_vec(&rec).unwrap()).unwrap();
    let again = import(models.path(), &item, &spec("m"), &mut |_, _| {}).unwrap();
    assert!(again.already && again.dir == old);
    assert_eq!((again.provenance.format.as_str(), again.provenance.uploaded_ms), ("ymir-provenance/1", Some(5)));
}

/// A safetensors file with `n` F32 weights and these header metadata.
fn safetensors(n: u64, metadata: Option<serde_json::Value>) -> Vec<u8> {
    let mut header = serde_json::json!({"w": {"dtype": "F32", "shape": [n], "data_offsets": [0, n * 4]}});
    if let Some(m) = metadata {
        header["__metadata__"] = m;
    }
    let h = serde_json::to_vec(&header).unwrap();
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(vec![7u8; (n * 4) as usize]);
    out
}

#[test]
fn inspect_outgoing_takes_directories_and_projectors() {
    let d = tempfile::tempdir().unwrap();
    let out = |p: &Path| inspect_outgoing(p, &mut |_, _| {});
    let st = d.path().join("Tune-8B");
    std::fs::create_dir(&st).unwrap();
    write(&st, "config.json", br#"{"architectures":["Qwen3ForCausalLM"],"_name_or_path":"Qwen/Qwen3-8B"}"#);
    write(&st, "model.safetensors", &safetensors(4, None));
    let o = out(&st).unwrap();
    assert_eq!((o.kind(), o.format_name(), o.architecture(), o.name.as_str()), (ArtifactKind::Model, "safetensors", "Qwen3ForCausalLM", "Tune-8B"));

    let mlx = d.path().join("tune-mlx");
    std::fs::create_dir(&mlx).unwrap();
    write(&mlx, "config.json", br#"{"architectures":["LlamaForCausalLM"],"quantization":{"bits":4,"group_size":64}}"#);
    write(&mlx, "model.safetensors", &safetensors(2, None));
    assert_eq!(out(&mlx).unwrap().format_name(), "MLX");

    let peft = d.path().join("lora");
    std::fs::create_dir(&peft).unwrap();
    write(&peft, "adapter_config.json", br#"{"peft_type":"LORA","r":8,"lora_alpha":16,"base_model_name_or_path":"Qwen/Qwen3-8B"}"#);
    write(&peft, "adapter_model.safetensors", &safetensors(3, None));
    let o = out(&peft).unwrap();
    assert_eq!((o.kind(), o.format_name(), o.architecture()), (ArtifactKind::Adapter { format: AdapterFormat::Peft }, "PEFT", "?"));
    assert_eq!(o.artifact.base_model.as_deref(), Some("Qwen/Qwen3-8B"));

    let mm = write(d.path(), "mmproj-F16.gguf", &gguf(&[("general.architecture", "clip"), ("general.type", "mmproj")], 256));
    assert_eq!(out(&mm).unwrap().kind(), ArtifactKind::Projector);
    assert!(inspect(&mm).unwrap_err().to_string().contains("--projector-for"));
    let m = write(d.path(), "m.gguf", &model("qwen3"));
    assert_eq!(out(&m).unwrap().kind(), ArtifactKind::Model);

    // Pickles and code are refused, and so is a directory on this machine's own path.
    write(&st, "pytorch_model.bin", b"pickle");
    assert!(out(&st).is_err());
    assert!(inspect(&mlx).unwrap_err().to_string().contains("--node"));
    let empty = d.path().join("empty");
    std::fs::create_dir(&empty).unwrap();
    assert!(out(&empty).is_err());
}
