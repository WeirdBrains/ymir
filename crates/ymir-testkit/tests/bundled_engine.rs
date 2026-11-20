//! The engine shipped with Ymir: `ymirctl setup` with no `--llama-server` picks the bundled
//! `llama-server` (`llamacpp = "managed"`), and `ymird` runs it. The mock engine stands in for the
//! real build, laid out as a release file installs it (`lib/ymir/llama.cpp/`).
//!
//! One test in its own binary: it sets `YMIR_ENGINE_DIR` before any thread starts.

use std::path::Path;

use anyhow::Result;
use clap::Parser as _;
use ymir_core::client::BridgeClient;
use ymir_core::engine::{self, ENGINE_DIR_ENV, INFO_FILE};
use ymir_core::{Daemon, NodeConfig, Paths};

const MOCK: &str = env!("CARGO_BIN_EXE_ymir-mock-engine");

async fn ctl(home: &Path, args: &[&str]) -> Result<String> {
    let mut argv = vec!["ymirctl".to_string(), "--home".into(), home.display().to_string()];
    argv.extend(args.iter().map(|a| a.to_string()));
    let mut out = Vec::new();
    ymirctl::run(ymirctl::Cli::try_parse_from(argv)?, &mut out, &mut std::io::empty()).await?;
    Ok(String::from_utf8(out)?)
}

#[test]
fn setup_and_ymird_use_the_bundled_engine() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let prefix = dir.path().join("prefix");
    let eng = prefix.join("lib/ymir/llama.cpp");
    std::fs::create_dir_all(&eng)?;
    std::fs::copy(MOCK, engine::program(&eng, "llama-server"))?;
    std::fs::write(eng.join(INFO_FILE), "engine = \"llama.cpp\"\nbuild = \"b0\"\nvariant = \"default\"\nbackends = [\"cpu\", \"vulkan\"]\n")?;
    assert_eq!(engine::bundled_from(&prefix.join("bin/ymird")).map(|b| b.dir), Some(eng.clone()));
    // SAFETY: no other thread exists yet in this test binary (one test, runtime not started).
    unsafe { std::env::set_var(ENGINE_DIR_ENV, &eng) };

    tokio::runtime::Builder::new_multi_thread().worker_threads(4).enable_all().build()?.block_on(async {
        let home = dir.path().join("home");
        let models = dir.path().join("models");
        std::fs::create_dir_all(&models)?;
        let gguf = models.join("tiny.gguf");
        std::fs::write(&gguf, b"GGUF mock")?;
        let out = ctl(&home, &["setup", "--name", "bundled", "--model", gguf.to_str().unwrap(), "--port", "0"]).await?;
        assert!(out.contains("llama.cpp: llama.cpp b0 (default: cpu, vulkan), shipped with Ymir"), "{out}");
        let paths = Paths::new(&home);
        let cfg = NodeConfig::load(&paths.config())?;
        assert_eq!(cfg.model.llamacpp, "managed");
        assert_eq!(cfg.engines(&paths)[0].program, engine::program(&eng, "llama-server"));

        let daemon = Daemon::start(cfg, paths.clone()).await?;
        let client = BridgeClient::new(daemon.bridge_addr.unwrap().port(), ymir_core::token::load(&paths)?.unwrap());
        let reply = client.chat_stream(None, vec![serde_json::json!({"role": "user", "content": "hi"})], Default::default(), |_| {}).await;
        daemon.shutdown().await;
        assert_eq!(reply?.text.trim(), "Hello from tiny! You said: hi");
        anyhow::Ok(())
    })
}
