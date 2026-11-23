//! LoRA adapters on a loaded base (training plan P4a), end to end on one box: `ymirctl models
//! adapter add` writes `[[model.local.lora]]`, ymird serves the adapter as its own model id on the
//! base's engine (the mock llama-server, started with `--lora`), requests for that id carry the
//! adapter and its scale, the base still answers as itself, and the status shows the adapter with
//! its base.

use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::Parser as _;
use ymir_core::control::{self, ControlRequest, NodeModel};
use ymir_core::{Daemon, NodeConfig, Paths};

const MOCK: &str = env!("CARGO_BIN_EXE_ymir-mock-engine");

async fn ctl(home: &Path, args: &[&str]) -> Result<String> {
    let mut argv = vec!["ymirctl".to_string(), "--home".into(), home.display().to_string()];
    argv.extend(args.iter().map(|a| a.to_string()));
    let cli = ymirctl::Cli::try_parse_from(argv)?;
    let mut out = Vec::new();
    ymirctl::run(cli, &mut out, &mut std::io::empty()).await?;
    Ok(String::from_utf8(out)?)
}

/// GGUF header bytes (version 3, no tensors) with these string and u32 keys.
fn gguf(strs: &[(&str, &str)], ints: &[(&str, u32)]) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend(0u64.to_le_bytes());
    out.extend(((strs.len() + ints.len()) as u64).to_le_bytes());
    let s = |out: &mut Vec<u8>, s: &str| {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    };
    for (k, v) in strs {
        s(&mut out, k);
        out.extend(8u32.to_le_bytes());
        s(&mut out, v);
    }
    for (k, v) in ints {
        s(&mut out, k);
        out.extend(4u32.to_le_bytes());
        out.extend(v.to_le_bytes());
    }
    out
}

fn base_gguf() -> Vec<u8> {
    gguf(
        &[("general.architecture", "llama")],
        &[
            ("llama.block_count", 1),
            ("llama.context_length", 4096),
            ("llama.embedding_length", 64),
            ("llama.attention.head_count", 1),
            ("llama.attention.head_count_kv", 1),
        ],
    )
}

fn lora_gguf(arch: &str) -> Vec<u8> {
    gguf(&[("general.architecture", arch), ("general.type", "adapter"), ("adapter.type", "lora")], &[])
}

async fn models(paths: &Paths) -> Vec<NodeModel> {
    control::request(&paths.socket(), &ControlRequest::Status).await.unwrap().status.unwrap().models
}

struct Node {
    _dir: tempfile::TempDir,
    home: PathBuf,
    models: PathBuf,
}

async fn setup() -> Result<Node> {
    let dir = tempfile::tempdir()?;
    let home = dir.path().join("home");
    let models = dir.path().join("models");
    std::fs::create_dir_all(models.join("adapters"))?;
    std::fs::write(models.join("base.gguf"), base_gguf())?;
    std::fs::write(models.join("adapters/support-v2.gguf"), lora_gguf("llama"))?;
    std::fs::write(models.join("adapters/qwen-tune.gguf"), lora_gguf("qwen3"))?;
    let b = models.join("base.gguf");
    let out = ctl(
        &home,
        &["setup", "--name", "lora", "--llama-server", MOCK, "--model", b.to_str().unwrap(), "--models-dir", models.to_str().unwrap(), "--port", "0"],
    )
    .await?;
    assert!(out.contains("Wrote"), "{out}");
    Ok(Node { _dir: dir, home, models })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_adapter_serves_as_its_own_model_on_the_base_engine() -> Result<()> {
    let n = setup().await?;
    let home = &n.home;

    // node.toml: add one, refuse one made for another architecture, list.
    let out = ctl(home, &["models", "adapter", "add", "base", "adapters/support-v2.gguf", "--scale", "0.5", "--revision", "run-7"]).await?;
    assert!(out.contains("Added LoRA adapter `base+support-v2` on `base`"), "{out}");
    let err = ctl(home, &["models", "adapter", "add", "base", "adapters/qwen-tune.gguf"]).await.unwrap_err().to_string();
    assert!(err.contains("doesn't match"), "{err}");
    let err = ctl(home, &["models", "adapter", "add", "nope", "adapters/support-v2.gguf"]).await.unwrap_err().to_string();
    assert!(err.contains("no [[model.local]] with id `nope`"), "{err}");
    let listed = ctl(home, &["models", "adapter", "list"]).await?;
    assert!(listed.contains("base+support-v2") && listed.contains("scale 0.5") && listed.contains("revision run-7"), "{listed}");
    let paths = Paths::new(home);
    let cfg = NodeConfig::load(&paths.config())?;
    assert_eq!(cfg.model.local[0].lora[0].id, "base+support-v2");
    assert!(paths.config().with_extension("toml.bak").is_file());

    let daemon = Daemon::start(cfg, paths.clone()).await?;

    // The adapter is a model of its own in the status and the catalog, with its base.
    let st = models(&paths).await;
    let ad = st.iter().find(|m| m.id == "base+support-v2").expect("adapter row");
    assert_eq!(ad.base.as_deref(), Some("base"));
    assert!(st.iter().find(|m| m.id == "base").unwrap().base.is_none());
    let status = ctl(home, &["status"]).await?;
    assert!(status.contains("base+support-v2") && status.contains("LoRA adapter on base"), "{status}");
    let listed = ctl(home, &["models"]).await?;
    assert!(listed.contains("base+support-v2"), "{listed}");
    // catalog/1 names the base, so apps (Bragi) can label the adapter; the bridge shows it too.
    let port = daemon.bridge_addr.expect("the bridge runs").port();
    let token = ymir_core::token::load(&paths)?.unwrap();
    let http = reqwest::Client::builder().no_proxy().build()?;
    let v: serde_json::Value = http.get(format!("http://127.0.0.1:{port}/v1/models")).bearer_auth(&token).send().await?.json().await?;
    let base_of = |id: &str| v["data"].as_array().unwrap().iter().find(|m| m["id"] == id).map(|m| m["weirdnet"]["base"].clone());
    assert_eq!(base_of("base+support-v2"), Some(serde_json::json!("base")), "{v}");
    assert_eq!(base_of("base"), Some(serde_json::Value::Null), "{v}");

    // Asking for the adapter loads the base (with --lora) and applies it at its scale.
    let reply = ctl(home, &["chat", "--model", "base+support-v2", "hi"]).await?;
    assert_eq!(reply.trim(), "Hello from base with support-v2 at 0.5! You said: hi");
    // The base on the same engine answers without it.
    let reply = ctl(home, &["chat", "--model", "base", "hi"]).await?;
    assert_eq!(reply.trim(), "Hello from base! You said: hi");
    let st = models(&paths).await;
    let pid = |id: &str| st.iter().find(|m| m.id == id).and_then(|m| m.pid);
    assert!(pid("base").is_some());
    assert_eq!(pid("base"), pid("base+support-v2"), "one engine for both");

    // An adapter is unloaded through its base.
    let err = ctl(home, &["unload", "base+support-v2"]).await.unwrap_err().to_string();
    assert!(err.contains("adapter"), "{err}");

    daemon.shutdown().await;

    let out = ctl(home, &["models", "adapter", "remove", "base+support-v2"]).await?;
    assert!(out.contains("Removed adapter `base+support-v2` from `base`"), "{out}");
    assert!(NodeConfig::load(&paths.config())?.model.local[0].lora.is_empty());
    assert!(n.models.join("adapters/support-v2.gguf").is_file(), "the file is kept");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_adapter_that_cannot_be_served_is_dropped_and_the_base_still_serves() -> Result<()> {
    let n = setup().await?;
    let paths = Paths::new(&n.home);
    // Written by hand, past `ymirctl models adapter add`'s check.
    let mut cfg = NodeConfig::load(&paths.config())?;
    cfg.model.local[0].lora = vec![
        ymir_core::config::LocalAdapter { id: "base+qwen".into(), file: "adapters/qwen-tune.gguf".into(), ..Default::default() },
        ymir_core::config::LocalAdapter { id: "base+support".into(), file: "adapters/support-v2.gguf".into(), ..Default::default() },
    ];
    cfg.validate()?;
    let daemon = Daemon::start(cfg, paths.clone()).await?;
    let ids: Vec<String> = models(&paths).await.into_iter().map(|m| m.id).collect();
    assert_eq!(ids, ["base", "base+support"]);
    let reply = ctl(&n.home, &["chat", "--model", "base+support", "yo"]).await?;
    assert_eq!(reply.trim(), "Hello from base with support-v2 at 1! You said: yo");
    let err = ctl(&n.home, &["chat", "--model", "base+qwen", "yo"]).await.unwrap_err().to_string();
    assert!(err.contains("unknown model"), "{err}");
    daemon.shutdown().await;
    Ok(())
}
