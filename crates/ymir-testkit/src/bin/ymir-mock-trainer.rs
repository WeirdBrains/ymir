//! A stand-in for a trainer that speaks Ymir's trainer interface (`ymir_core::train`): no model
//! is read or trained.
//!
//! `version` prints `{"interface": 1, "engine": "mock", "build": B}` with B from
//! `YMIR_MOCK_TRAINER_BUILD` (default `b0000`). `train` checks every row of `--data` (a
//! `messages` list with an assistant turn) and `--config`, reports one progress event per row
//! and epoch with a falling loss, sleeping `YMIR_MOCK_TRAINER_STEP_MS` (default 0) per step, and
//! writes `--out`. `extract-lora` checks that `--tuned` is its own output and writes a small
//! adapter to `--out`. `YMIR_MOCK_TRAINER_FAIL=train` (or `extract-lora`) makes that step report
//! an error and exit 1. `YMIR_MOCK_TRAINER_RECORD=FILE` appends each command line to FILE.

use std::io::Write;
use std::time::Duration;

use serde_json::{Value, json};

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
        "version" => say(json!({"interface": 1, "engine": "mock", "build": env("YMIR_MOCK_TRAINER_BUILD").unwrap_or_else(|| "b0000".into())})),
        "train" => {
            let (Some(base), Some(data), Some(config), Some(out)) = (flag("--base"), flag("--data"), flag("--config"), flag("--out")) else {
                die("train needs --base, --data, --config and --out".into())
            };
            let base_len = std::fs::metadata(&base).map(|m| m.len()).unwrap_or_else(|e| die(format!("{base}: {e}")));
            let config: Value =
                serde_json::from_str(&std::fs::read_to_string(&config).unwrap_or_default()).unwrap_or_else(|e| die(format!("config: {e}")));
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
            say(json!({"event": "log", "message": format!("{rows} rows, {epochs} epochs")}));
            let steps = rows * epochs;
            let pause = Duration::from_millis(env("YMIR_MOCK_TRAINER_STEP_MS").and_then(|v| v.parse().ok()).unwrap_or(0));
            for step in 1..=steps {
                std::thread::sleep(pause);
                say(json!({"event": "progress", "step": step, "steps": steps, "loss": 2.0 / (step as f64 + 1.0)}));
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
            std::fs::write(&out, format!("GGUF mock lora rank {rank}")).unwrap_or_else(|e| die(format!("{out}: {e}")));
            say(json!({"event": "done"}));
        }
        other => {
            eprintln!("ymir-mock-trainer: unknown command `{other}`");
            std::process::exit(2);
        }
    }
}
