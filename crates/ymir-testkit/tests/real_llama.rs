//! The M1 path with a real llama-server and a real GGUF. Ignored by default:
//!
//! YMIR_TEST_LLAMA_SERVER=/path/llama-server YMIR_TEST_GGUF=/path/model.gguf \
//!   cargo test -p ymir-testkit --test real_llama -- --ignored

use std::path::PathBuf;

use anyhow::{Context, Result};
use serde_json::json;
use ymir_core::client::BridgeClient;
use ymir_core::setup::{self, SetupOptions};
use ymir_core::{Daemon, Paths};

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs YMIR_TEST_LLAMA_SERVER and YMIR_TEST_GGUF"]
async fn real_llama_server_streams_an_answer() -> Result<()> {
    let server = PathBuf::from(std::env::var("YMIR_TEST_LLAMA_SERVER").context("YMIR_TEST_LLAMA_SERVER")?);
    let gguf = PathBuf::from(std::env::var("YMIR_TEST_GGUF").context("YMIR_TEST_GGUF")?);
    let dir = tempfile::tempdir()?;
    let paths = Paths::new(dir.path());
    let mut p = setup::propose(
        ymir_hw::detect(),
        &SetupOptions { llama_server: Some(server), models: vec![gguf], bridge_port: Some(0), ..Default::default() },
    );
    p.config.model.local[0].context = Some(2048);
    p.config.save(&paths.config())?;
    let daemon = Daemon::start(p.config.clone(), paths.clone()).await?;
    let port = daemon.bridge_addr.unwrap().port();
    let client = BridgeClient::new(port, ymir_core::token::load_or_create(&paths)?);
    let mut extra = serde_json::Map::new();
    extra.insert("max_tokens".into(), 24.into());
    let mut pieces = 0;
    let out = client.chat_stream(None, vec![json!({"role": "user", "content": "Say hello in five words."})], extra, |_| pieces += 1).await;
    let log = std::fs::read_dir(paths.logs())
        .ok()
        .and_then(|mut d| d.next())
        .and_then(|e| e.ok())
        .and_then(|e| std::fs::read_to_string(e.path()).ok())
        .unwrap_or_default();
    daemon.shutdown().await;
    let out = out.with_context(|| format!("engine log:\n{log}"))?;
    eprintln!("real llama-server answered ({pieces} pieces): {:?}", out.text);
    assert!(!out.text.trim().is_empty());
    assert!(pieces > 1, "the answer streamed in pieces");
    Ok(())
}
