//! A stand-in for a trainer that speaks Ymir's trainer interface (`ymir_core::train`): no model
//! is read or trained.
//!
//! `version` prints `{"interface": 1, "engine": "mock", "build": B}` with B from
//! `YMIR_MOCK_TRAINER_BUILD` (default `b0000`). `train` checks every row of `--data` (a
//! `messages` list with an assistant turn) and `--config`, reports one progress event per row
//! and epoch with a falling loss, sleeping `YMIR_MOCK_TRAINER_STEP_MS` (default 0) per step, and
//! writes `--out`. `extract-lora` checks that `--tuned` is its own output and writes a small GGUF
//! LoRA adapter to `--out` (a header only: architecture `llama`, `general.name` `mock lora rank N`). `YMIR_MOCK_TRAINER_FAIL=train` (or `extract-lora`) makes that step report
//! an error and exit 1. `YMIR_MOCK_TRAINER_RECORD=FILE` appends each command line to FILE.
//!
//! Interface 2 (the default; `YMIR_MOCK_TRAINER_INTERFACE=1` reports 1): with `--checkpoint-dir
//! DIR`, `train` writes `DIR/epoch-NNNN/` (`model.gguf` and `checkpoint.json`) after each epoch
//! and prints a `checkpoint` event; `--resume CKPT` checks the checkpoint's data and config hashes
//! and carries on after its epoch. `YMIR_MOCK_TRAINER_DIE_AFTER_EPOCH=N` reports an error and
//! exits 1 right after epoch N's checkpoint (a node lost mid-run).

use std::io::Write;
use std::time::Duration;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn sha(path: &str) -> (String, u64) {
    let b = std::fs::read(path).unwrap_or_default();
    (hex::encode(Sha256::digest(&b)), b.len() as u64)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let env = |name: &str| std::env::var(name).ok().filter(|v| !v.is_empty());
    if let Some(rec) = env("YMIR_MOCK_TRAINER_RECORD")
        && let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(rec)
    {
        let _ = writeln!(f, "{}", args.join(" "));
    }
    let cmd = args.first().map(String::as_str).unwrap_or_default();
    let fail = env("YMIR_MOCK_TRAINER_FAIL");
    let say = |v: Value| {
        println!("{v}");
        let _ = std::io::stdout().flush();
    };
    let die = |message: String| -> ! {
        say(json!({"event": "error", "message": message}));
        std::process::exit(1)
    };
    match cmd {
        "version" => say(
            json!({"interface": env("YMIR_MOCK_TRAINER_INTERFACE").and_then(|v| v.parse::<u32>().ok()).unwrap_or(2), "engine": "mock", "build": env("YMIR_MOCK_TRAINER_BUILD").unwrap_or_else(|| "b0000".into())}),
        ),
        "train" => {
            let (Some(base), Some(data), Some(config), Some(out)) = (flag("--base"), flag("--data"), flag("--config"), flag("--out")) else {
                die("train needs --base, --data, --config and --out".into())
            };
            let base_len = std::fs::metadata(&base).map(|m| m.len()).unwrap_or_else(|e| die(format!("{base}: {e}")));
            let config_path = config;
            let config: Value =
                serde_json::from_str(&std::fs::read_to_string(&config_path).unwrap_or_default()).unwrap_or_else(|e| die(format!("config: {e}")));
            let epochs = config["epochs"].as_u64().unwrap_or_else(|| die("config has no epochs".into()));
            let text = std::fs::read_to_string(&data).unwrap_or_else(|e| die(format!("{data}: {e}")));
            let mut rows = 0u64;
            for (i, line) in text.lines().enumerate() {
                let row: Value = serde_json::from_str(line).unwrap_or_else(|e| die(format!("row {}: {e}", i + 1)));
                let turns = row["messages"].as_array().cloned().unwrap_or_default();
                if !turns.iter().any(|t| t["role"] == "assistant" && t["content"].as_str().is_some_and(|c| !c.is_empty())) {
                    die(format!("row {} has no assistant turn", i + 1));
                }
                rows += 1;
            }
            if fail.as_deref() == Some("train") {
                die("mock trainer: out of memory".into());
            }
            let (data_sha, _) = sha(&data);
            let (config_sha, _) = sha(&config_path);
            let mut first = 1u64;
            if let Some(ck) = flag("--resume") {
                let c: Value = serde_json::from_str(&std::fs::read_to_string(format!("{ck}/checkpoint.json")).unwrap_or_default())
                    .unwrap_or_else(|e| die(format!("{ck}: {e}")));
                if c["data_sha256"] != data_sha.as_str() || c["config_sha256"] != config_sha.as_str() {
                    die(format!("{ck} is a checkpoint of other data or settings"));
                }
                if !std::fs::read_to_string(format!("{ck}/model.gguf")).unwrap_or_default().starts_with("MOCK-FULL") {
                    die(format!("{ck}/model.gguf is not a model this trainer made"));
                }
                first = c["epoch"].as_u64().unwrap_or(0) + 1;
                say(json!({"event": "log", "message": format!("resuming after epoch {}", first - 1)}));
            }
            say(json!({"event": "log", "message": format!("{rows} rows, {epochs} epochs")}));
            let steps = rows * epochs;
            let pause = Duration::from_millis(env("YMIR_MOCK_TRAINER_STEP_MS").and_then(|v| v.parse().ok()).unwrap_or(0));
            let die_after = env("YMIR_MOCK_TRAINER_DIE_AFTER_EPOCH").and_then(|v| v.parse::<u64>().ok());
            for epoch in first..=epochs {
                for r in 1..=rows {
                    std::thread::sleep(pause);
                    let step = (epoch - 1) * rows + r;
                    say(json!({"event": "progress", "step": step, "steps": steps, "loss": 2.0 / (step as f64 + 1.0)}));
                }
                if let Some(dir) = flag("--checkpoint-dir") {
                    let name = format!("epoch-{epoch:04}");
                    let cdir = format!("{dir}/{name}");
                    std::fs::create_dir_all(&cdir).unwrap_or_else(|e| die(format!("{cdir}: {e}")));
                    let model = format!("{cdir}/model.gguf");
                    std::fs::write(&model, format!("MOCK-FULL {base_len} epoch {epoch}")).unwrap_or_else(|e| die(format!("{model}: {e}")));
                    let (msha, mbytes) = sha(&model);
                    let step = epoch * rows;
                    let c = json!({
                        "format": "ymir-train-checkpoint/1", "epoch": epoch, "epochs": epochs, "step": step, "steps": steps,
                        "loss": 2.0 / (step as f64 + 1.0), "seed": config["seed"].as_u64().unwrap_or(0),
                        "data_sha256": data_sha, "config_sha256": config_sha, "model": "model.gguf", "optimizer": "reset",
                        "files": [{"path": "model.gguf", "sha256": msha, "bytes": mbytes}],
                    });
                    std::fs::write(format!("{cdir}/checkpoint.json"), serde_json::to_vec_pretty(&c).unwrap_or_default())
                        .unwrap_or_else(|e| die(format!("{cdir}: {e}")));
                    say(json!({"event": "checkpoint", "dir": name, "epoch": epoch, "step": step}));
                }
                if die_after == Some(epoch) && epoch < epochs {
                    die(format!("mock trainer: the GPU went away after epoch {epoch}"));
                }
            }
            std::fs::write(&out, format!("MOCK-FULL {base_len}")).unwrap_or_else(|e| die(format!("{out}: {e}")));
            say(json!({"event": "done"}));
        }
        "extract-lora" => {
            let (Some(_base), Some(tuned), Some(rank), Some(out)) = (flag("--base"), flag("--tuned"), flag("--rank"), flag("--out")) else {
                die("extract-lora needs --base, --tuned, --rank and --out".into())
            };
            if !std::fs::read_to_string(&tuned).unwrap_or_default().starts_with("MOCK-FULL") {
                die(format!("{tuned} is not a model this trainer made"));
            }
            if fail.as_deref() == Some("extract-lora") {
                die("mock trainer: the tensors differ in shape".into());
            }
            say(json!({"event": "progress", "step": 1, "steps": 1}));
            std::fs::write(&out, lora(&rank)).unwrap_or_else(|e| die(format!("{out}: {e}")));
            say(json!({"event": "done"}));
        }
        other => {
            eprintln!("ymir-mock-trainer: unknown command `{other}`");
            std::process::exit(2);
        }
    }
}

/// A GGUF LoRA adapter header with no tensors, padded: enough for Ymir to read its kind.
fn lora(rank: &str) -> Vec<u8> {
    let name = format!("mock lora rank {rank}");
    let kv = [("general.architecture", "llama"), ("general.type", "adapter"), ("adapter.type", "lora"), ("general.name", name.as_str())];
    let mut o = b"GGUF".to_vec();
    o.extend(3u32.to_le_bytes());
    o.extend(0u64.to_le_bytes());
    o.extend((kv.len() as u64).to_le_bytes());
    for (k, v) in kv {
        o.extend((k.len() as u64).to_le_bytes());
        o.extend(k.as_bytes());
        o.extend(8u32.to_le_bytes());
        o.extend((v.len() as u64).to_le_bytes());
        o.extend(v.as_bytes());
    }
    o.resize(o.len() + 256, 0);
    o
}
