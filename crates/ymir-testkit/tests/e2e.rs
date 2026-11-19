//! The whole M1 path on one box: `ymirctl setup` writes node.toml, `ymird` runs the model,
//! inference and app roles in one process, the model role supervises an engine (the mock
//! llama-server), the inference role routes to it, and `ymirctl chat` streams the answer through
//! the loopback bridge.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use clap::Parser as _;
use serde_json::{Value, json};
use ymir_core::client::BridgeClient;
use ymir_core::control::{self, ControlRequest, ModelState, NodeModel};
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

async fn models(paths: &Paths) -> Vec<NodeModel> {
    control::request(&paths.socket(), &ControlRequest::Status).await.unwrap().status.unwrap().models
}

async fn state(paths: &Paths, id: &str) -> NodeModel {
    models(paths).await.into_iter().find(|m| m.id == id).unwrap()
}

async fn wait_for(paths: &Paths, id: &str, want: impl Fn(&ModelState) -> bool) -> NodeModel {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let m = state(paths, id).await;
        if want(&m.state) {
            return m;
        }
        assert!(Instant::now() < deadline, "{id} stuck in {:?}", m.state);
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

fn alive(pid: u32) -> bool {
    // SAFETY: signal 0 only checks that the process exists.
    unsafe { libc::kill(pid as i32, 0) == 0 }
}

struct Node {
    _dir: tempfile::TempDir,
    home: PathBuf,
    paths: Paths,
    daemon: Option<Daemon>,
}

/// `ymirctl setup` with two mock GGUFs in one swap group, then `ymird` with all three roles.
async fn node() -> Result<Node> {
    let dir = tempfile::tempdir()?;
    let home = dir.path().join("home");
    let models = dir.path().join("models");
    std::fs::create_dir_all(&models)?;
    for f in ["alpha.gguf", "beta.gguf"] {
        std::fs::write(models.join(f), b"GGUF mock")?;
    }
    let a = models.join("alpha.gguf");
    let b = models.join("beta.gguf");
    let out = ctl(
        &home,
        &[
            "setup",
            "--name",
            "e2e",
            "--llama-server",
            MOCK,
            "--model",
            a.to_str().unwrap(),
            "--model",
            b.to_str().unwrap(),
            "--models-dir",
            models.to_str().unwrap(),
            "--port",
            "0",
        ],
    )
    .await?;
    assert!(out.contains("Wrote"), "{out}");
    let paths = Paths::new(&home);
    let cfg = NodeConfig::load(&paths.config())?;
    assert_eq!(cfg.roles.names(), ["model", "inference", "app"]);
    assert_eq!(cfg.main_model().unwrap().id, "alpha");
    let daemon = Daemon::start(cfg, paths.clone()).await?;
    Ok(Node { _dir: dir, home, paths, daemon: Some(daemon) })
}

impl Node {
    fn bridge(&self) -> BridgeClient {
        let port = self.daemon.as_ref().unwrap().bridge_addr.unwrap().port();
        BridgeClient::new(port, ymir_core::token::load(&self.paths).unwrap().unwrap())
    }

    async fn stop(mut self) {
        self.daemon.take().unwrap().shutdown().await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn setup_daemon_chat_end_to_end() -> Result<()> {
    let n = node().await?;
    let home = n.home.clone();

    // Nothing is loaded until it is needed.
    let st = ctl(&home, &["status"]).await?;
    assert!(st.contains("roles: model, inference, app"), "{st}");
    assert!(st.contains("alpha") && st.contains("unloaded"), "{st}");

    // ymirctl chat → bridge → inference role → model role → engine, streamed back.
    let reply = ctl(&home, &["chat", "ping", "pong"]).await?;
    assert_eq!(reply.trim(), "Hello from alpha! You said: ping pong");
    let alpha = wait_for(&n.paths, "alpha", |s| *s == ModelState::Ready).await;
    let pid = alpha.pid.unwrap();
    assert!(alive(pid));

    // The catalog through the bridge.
    let listed = ctl(&home, &["models"]).await?;
    assert!(listed.contains("alpha") && listed.contains("ready") && listed.contains("beta"), "{listed}");

    // Swap group: asking for beta unloads alpha first.
    let reply = ctl(&home, &["chat", "--model", "beta", "hi"]).await?;
    assert_eq!(reply.trim(), "Hello from beta! You said: hi");
    assert_eq!(state(&n.paths, "alpha").await.state, ModelState::Unloaded);
    assert!(!alive(pid), "alpha's engine was stopped");

    // Unload and load through the control socket.
    assert!(ctl(&home, &["unload", "beta"]).await?.contains("unloaded"));
    assert_eq!(state(&n.paths, "beta").await.state, ModelState::Unloaded);
    assert!(ctl(&home, &["load", "alpha"]).await?.contains("ready"));
    let alpha = state(&n.paths, "alpha").await;
    assert_eq!(alpha.state, ModelState::Ready);

    // A crashed engine is noticed, and the next prompt restarts it.
    let pid = alpha.pid.unwrap();
    // SAFETY: the mock engine this test started.
    unsafe { libc::kill(pid as i32, libc::SIGKILL) };
    wait_for(&n.paths, "alpha", |s| matches!(s, ModelState::Failed(_))).await;
    let reply = ctl(&home, &["chat", "again"]).await?;
    assert_eq!(reply.trim(), "Hello from alpha! You said: again");

    // Unknown models and bad loads are clear errors.
    assert!(ctl(&home, &["chat", "--model", "nope", "x"]).await.unwrap_err().to_string().contains("unknown model"));
    assert!(ctl(&home, &["load", "nope"]).await.is_err());

    let pid = state(&n.paths, "alpha").await.pid.unwrap();
    n.stop().await;
    assert!(!alive(pid), "shutdown stops engines");
    assert!(!home.join("state/ymird.sock").exists());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn openai_clients_and_the_cache() -> Result<()> {
    let n = node().await?;
    let port = n.daemon.as_ref().unwrap().bridge_addr.unwrap().port();
    let token = ymir_core::token::load(&n.paths)?.unwrap();
    let url = format!("http://127.0.0.1:{port}/v1");
    let http = reqwest::Client::builder().no_proxy().build()?;

    // The token is required everywhere, including /v1/models and /metrics.
    for path in ["models", "chat/completions"] {
        let r = http.post(format!("{url}/{path}")).json(&json!({})).send().await?;
        assert_eq!(r.status(), 401, "{path}");
    }
    assert_eq!(http.get(format!("http://127.0.0.1:{port}/metrics")).send().await?.status(), 401);
    assert_eq!(http.get(format!("{url}/models")).bearer_auth("x".repeat(64)).send().await?.status(), 401);

    // A plain (non-streaming) OpenAI request, as Jan or a script would send.
    let v: Value = http
        .post(format!("{url}/chat/completions"))
        .bearer_auth(&token)
        .json(&json!({"model": "alpha", "messages": [{"role": "user", "content": "plain"}]}))
        .send()
        .await?
        .json()
        .await?;
    assert_eq!(v["choices"][0]["message"]["content"], "Hello from alpha! You said: plain");
    assert_eq!(v["usage"]["prompt_tokens"], 5);

    // Deterministic requests are answered from the embedded cache the second time.
    let bridge = n.bridge();
    let mut extra = serde_json::Map::new();
    extra.insert("temperature".into(), 0.into());
    let msgs = vec![json!({"role": "user", "content": "cache me"})];
    let first = bridge.chat_stream(None, msgs.clone(), extra.clone(), |_| {}).await?;
    let second = bridge.chat_stream(None, msgs, extra, |_| {}).await?;
    assert!(!first.cache_hit);
    assert!(second.cache_hit);
    assert_eq!(first.text, second.text);
    let metrics = http.get(format!("http://127.0.0.1:{port}/metrics")).bearer_auth(&token).send().await?.text().await?;
    assert!(metrics.contains("weirdnet_bridge_cache_hits_total 1"), "{metrics}");

    // No embedding model on this node, and chat models refuse to embed. agent 0.4's
    // Pool::embed folds every node error into `unavailable`, so these are 503 with the reason in
    // the message (M3 status: known gaps).
    let r = http.post(format!("{url}/embeddings")).bearer_auth(&token).json(&json!({"input": "x"})).send().await?;
    assert_eq!(r.status(), 503);
    assert!(r.text().await?.contains("no embedding model"));
    let r = http.post(format!("{url}/embeddings")).bearer_auth(&token).json(&json!({"model": "alpha", "input": "x"})).send().await?;
    assert_eq!(r.status(), 503);
    n.stop().await;
    Ok(())
}

#[tokio::test]
async fn setup_refuses_to_overwrite_and_dry_run_writes_nothing() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let home = dir.path().join("home");
    let gguf = dir.path().join("m.gguf");
    std::fs::write(&gguf, b"x")?;
    let out = ctl(&home, &["setup", "--dry-run", "--llama-server", MOCK, "--model", gguf.to_str().unwrap()]).await?;
    assert!(out.contains("[roles]") && out.contains("model = true"), "{out}");
    assert!(!home.join("node.toml").exists());
    ctl(&home, &["setup", "--llama-server", MOCK, "--model", gguf.to_str().unwrap()]).await?;
    assert!(home.join("node.toml").exists());
    let err = ctl(&home, &["setup", "--model", gguf.to_str().unwrap()]).await.unwrap_err();
    assert!(err.to_string().contains("--force"));
    // No daemon: status says so.
    if ctl(&home, &["status"]).await.is_ok() {
        bail!("status should fail without ymird");
    }
    Ok(())
}

#[tokio::test]
async fn hw_reports_this_machine() -> Result<()> {
    let out = ctl(Path::new("/nonexistent-ymir-home"), &["hw", "--json"]).await?;
    let v: Value = serde_json::from_str(&out)?;
    assert!(v["hardware"]["cpu_threads"].as_u64().unwrap() >= 1);
    assert!(v["budget"]["usable_bytes"].as_u64().is_some());
    Ok(())
}
