//! Training plan P3a with the real trainer program, `scripts/trainer.py` (llama.cpp's
//! `llama-finetune` plus LoRA extraction), in front of a fake `llama-finetune`
//! (`scripts/testdata/fake_llama_finetune.py`): Ymir checks the trainer's pinned build, the run
//! relays its progress, and the adapter it extracts lands in the models directory. No model is
//! trained.
//!
//! Needs a python with the trainer's requirements (`scripts/trainer-requirements.txt`) in
//! `YMIR_TEST_TRAINER_PYTHON`; without it the test says so and passes. CI sets it on Linux.
#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Result, ensure};
use clap::Parser as _;
use ymir_core::train::{self, RunStatus};
use ymir_core::{NodeConfig, Paths};

const MOCK: &str = env!("CARGO_BIN_EXE_ymir-mock-engine");

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

async fn ctl(home: &Path, args: &[&str]) -> Result<String> {
    let mut argv = vec!["ymirctl".to_string(), "--home".into(), home.display().to_string()];
    argv.extend(args.iter().map(|a| a.to_string()));
    let cli = ymirctl::Cli::try_parse_from(argv)?;
    let mut out = Vec::new();
    let mut input = std::io::Cursor::new(Vec::new());
    ymirctl::run(cli, &mut out, &mut input).await?;
    Ok(String::from_utf8(out)?)
}

fn python(py: &str, code: &str) -> Result<String> {
    let out = Command::new(py).arg("-c").arg(code).env("PYTHONPATH", repo().join("scripts")).output()?;
    ensure!(out.status.success(), "{py}: {}", String::from_utf8_lossy(&out.stderr));
    Ok(String::from_utf8(out.stdout)?)
}

#[tokio::test(flavor = "multi_thread")]
async fn the_llamacpp_trainer_trains_and_extracts_an_adapter_ymir_serves() -> Result<()> {
    let Ok(py) = std::env::var("YMIR_TEST_TRAINER_PYTHON") else {
        eprintln!("skipped: set YMIR_TEST_TRAINER_PYTHON to a python with scripts/trainer-requirements.txt");
        return Ok(());
    };
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();
    let home = root.join("ymir");
    let models = root.join("models");
    std::fs::create_dir_all(&home)?;
    std::fs::create_dir_all(&models)?;
    let base = models.join("alpha.gguf");
    python(&py, &format!("import test_trainer; test_trainer.write_model({:?})", base.display().to_string()))?;
    let finetune = root.join("llama-finetune");
    let fake = repo().join("scripts/testdata/fake_llama_finetune.py");
    std::fs::write(&finetune, format!("#!/bin/sh\nexec \"{py}\" \"{}\" \"$@\"\n", fake.display()))?;
    std::fs::set_permissions(&finetune, std::fs::Permissions::from_mode(0o755))?;
    let toml = format!(
        "[node]\nname = \"gpu1\"\n[roles]\nmodel = true\ninference = true\napp = false\n[model]\nllamacpp = \"system:{MOCK}\"\nvllm = \"off\"\nmodels_dir = \"{}\"\n[[model.local]]\nid = \"alpha\"\nfile = \"alpha.gguf\"\nrole = \"main\"\nrevision = \"r1\"\nquant = \"F32\"\n[inference.alerts]\nprobe_secs = 3600\n[train]\ntrainer = \"{}\"\nbuild = \"b12628\"\n[train.env]\nYMIR_TRAINER_PYTHON = \"{py}\"\nYMIR_LLAMA_FINETUNE = \"{}\"\n",
        models.display(),
        repo().join("scripts/trainer.py").display(),
        finetune.display()
    );
    std::fs::write(home.join("node.toml"), toml)?;
    let paths = Paths::new(&home);

    let out = ctl(&home, &["train", "check"]).await?;
    assert!(out.contains("llama.cpp build b12628 (interface 1), as pinned"), "{out}");

    let data = root.join("chats.jsonl");
    let body = "{\"id\": \"c1\", \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}, {\"role\": \"assistant\", \"content\": \"hello\"}]}\n";
    std::fs::write(&data, body)?;
    let sha: String = <sha2::Sha256 as sha2::Digest>::digest(body.as_bytes()).iter().map(|b| format!("{b:02x}")).collect();
    let manifest = serde_json::json!({
        "format": ymir_core::convexport::FORMAT, "output_name": "chats.jsonl", "output_sha256": sha,
        "rules": {"keep_system": false, "redact_patterns": 0}, "conversations": [], "conversations_flagged": 1,
        "conversations_exported": 1, "turns_exported": 2,
        "dropped": {"system": 0, "tool": 0, "attachments": 0, "reasoning": 0, "incomplete": 0, "standin": 0, "after_cut": 0},
        "redactions": 0, "created_ms": 0
    });
    std::fs::write(ymir_core::dataset::manifest_path(&data), serde_json::to_vec(&manifest)?)?;

    let d = data.display().to_string();
    let out =
        ctl(&home, &["train", "run", "--base", "alpha", "--dataset", &d, "--epochs", "2", "--rank", "4", "--id", "alpha+chats", "--add"]).await?;
    assert!(out.contains("train: 8/8") && out.contains("extract-lora: 4/4"), "{out}");
    assert!(out.contains("Done: 8 steps, final loss 0.2500. Adapter `alpha+chats` is adapters/alpha+chats.gguf"), "{out}");
    assert!(train::held(&paths).is_none(), "the GPU is free again");

    let run = out.split_whitespace().nth(1).expect("the run id").to_string();
    let r: train::RunRecord = serde_json::from_str(&std::fs::read_to_string(paths.state().join("train").join(&run).join("run.json"))?)?;
    assert_eq!((r.status, r.trainer.engine.as_str(), r.trainer.build.as_str()), (RunStatus::Done, "llama.cpp", "b12628"));
    let log = std::fs::read_to_string(paths.state().join("train").join(&run).join("trainer.log")).unwrap_or_default();
    let adapter = models.join("adapters/alpha+chats.gguf");
    let names = python(
        &py,
        &format!(
            "import trainer, gguf; r = gguf.GGUFReader({:?}); print(trainer.field_str(r, 'adapter.type'), *sorted(t.name for t in r.tensors))",
            adapter.display().to_string()
        ),
    )?;
    assert_eq!(
        names.trim(),
        "lora blk.0.attn_q.weight.lora_a blk.0.attn_q.weight.lora_b blk.0.ffn_down.weight.lora_a blk.0.ffn_down.weight.lora_b blk.0.ffn_up.weight.lora_a blk.0.ffn_up.weight.lora_b",
        "{log}"
    );
    let cfg = NodeConfig::load(&paths.config())?;
    assert_eq!(cfg.model.local[0].lora[0].file, "adapters/alpha+chats.gguf");
    Ok(())
}
