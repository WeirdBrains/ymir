//! M2: model-node parity with existing llama-swap servers. Idle unload, crash backoff and crash loops, the stand-in
//! cold start, /v1/completions and /v1/embeddings, wrapping a stock server, alerts, and the user
//! service, all against the mock engine in temp dirs.

use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Parser as _;
use serde_json::{Value, json};
use weirdnet_backends::{EngineConfig, EngineKind, ModelLaunch, Supervisor, SupervisorEvent, SupervisorOptions, Wrap};
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

async fn state(paths: &Paths, id: &str) -> NodeModel {
    let s = control::request(&paths.socket(), &ControlRequest::Status).await.unwrap().status.unwrap();
    s.models.into_iter().find(|m| m.id == id).unwrap()
}

async fn wait_for(paths: &Paths, id: &str, secs: u64, want: impl Fn(&ModelState) -> bool) -> NodeModel {
    let deadline = Instant::now() + Duration::from_secs(secs);
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
    paths: Paths,
    daemon: Option<Daemon>,
    http: reqwest::Client,
    url: String,
    token: String,
}

/// A node from a hand-written node.toml (`models` is the `[[model.local]]` part).
async fn node(models: &str, inference: &str) -> Result<Node> {
    let dir = tempfile::tempdir()?;
    let home = dir.path().join("home");
    let mdir = dir.path().join("models");
    std::fs::create_dir_all(&mdir)?;
    std::fs::create_dir_all(&home)?;
    for f in ["big.gguf", "small.gguf", "emb.gguf"] {
        std::fs::write(mdir.join(f), b"GGUF mock")?;
    }
    let toml = format!(
        "[node]\nname = \"m2e2e\"\n[roles]\nmodel = true\ninference = true\napp = true\n[model]\nllamacpp = \"system:{MOCK}\"\nvllm = \"off\"\nmodels_dir = \"{}\"\n{models}\n[inference]\n{inference}\n[inference.alerts]\nprobe_secs = 3600\n[app.bridge]\nport = 0\n",
        mdir.display()
    );
    std::fs::write(home.join("node.toml"), toml)?;
    let paths = Paths::new(&home);
    let cfg = NodeConfig::load(&paths.config())?;
    let daemon = Daemon::start(cfg, paths.clone()).await?;
    let port = daemon.bridge_addr.unwrap().port();
    let token = ymir_core::token::load(&paths)?.unwrap();
    Ok(Node {
        _dir: dir,
        paths,
        daemon: Some(daemon),
        http: reqwest::Client::builder().no_proxy().build()?,
        url: format!("http://127.0.0.1:{port}/v1"),
        token,
    })
}

impl Node {
    async fn post(&self, path: &str, body: Value, wait: bool) -> Result<reqwest::Response> {
        let mut r = self.http.post(format!("{}/{path}", self.url)).bearer_auth(&self.token).json(&body);
        if wait {
            r = r.header("x-wait", "1");
        }
        Ok(r.send().await?)
    }

    async fn stop(mut self) {
        self.daemon.take().unwrap().shutdown().await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn standin_answers_while_the_big_model_loads() -> Result<()> {
    let n = node(
        "[[model.local]]\nid = \"big\"\nfile = \"big.gguf\"\nrole = \"main\"\nswap_group = \"main\"\n[[model.local]]\nid = \"small\"\nfile = \"small.gguf\"\nrole = \"standin\"\n",
        "standin_after = \"1s\"\n",
    )
    .await?;
    // The stand-in is kept warm from the start.
    wait_for(&n.paths, "small", 15, |s| *s == ModelState::Ready).await;
    assert_eq!(state(&n.paths, "big").await.state, ModelState::Unloaded);

    // Cold big model (estimated ~3 s from its size, over standin_after): the stand-in answers,
    // flagged, and big loads in the background.
    let r = n.post("chat/completions", json!({"messages": [{"role": "user", "content": "hi"}]}), false).await?;
    assert_eq!(r.headers()["x-weirdnet-standin"], "small");
    let v: Value = r.json().await?;
    assert_eq!(v["choices"][0]["message"]["content"], "Hello from small! You said: hi");
    assert_eq!(v["weirdnet"]["standin"], "small");
    let big = wait_for(&n.paths, "big", 15, |s| *s == ModelState::Ready).await;
    assert!(big.last_load_secs.is_some());

    // Warm now: big answers itself.
    let r = n.post("chat/completions", json!({"messages": [{"role": "user", "content": "again"}]}), false).await?;
    assert!(r.headers().get("x-weirdnet-standin").is_none());
    let v: Value = r.json().await?;
    assert_eq!(v["choices"][0]["message"]["content"], "Hello from big! You said: again");

    // x-wait opts out: after an unload the request waits for big instead.
    ctl(&n.paths.home, &["unload", "big"]).await?;
    let r = n.post("chat/completions", json!({"messages": [{"role": "user", "content": "wait"}]}), true).await?;
    assert!(r.headers().get("x-weirdnet-standin").is_none());
    let v: Value = r.json().await?;
    assert_eq!(v["choices"][0]["message"]["content"], "Hello from big! You said: wait");

    // The measured load (~0.2 s) is now under standin_after, so the next cold start waits.
    ctl(&n.paths.home, &["unload", "big"]).await?;
    let r = n.post("chat/completions", json!({"messages": [{"role": "user", "content": "fast"}]}), false).await?;
    assert!(r.headers().get("x-weirdnet-standin").is_none(), "a fast load is waited for");

    // ymirctl chat says when a stand-in answered.
    ctl(&n.paths.home, &["unload", "big"]).await?;
    let out = ctl(&n.paths.home, &["chat", "--model", "small", "direct"]).await?;
    assert!(!out.contains("stand-in"), "asking for the stand-in itself is not a stand-in answer: {out}");
    n.stop().await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn completions_embeddings_and_idle_unload() -> Result<()> {
    let n = node(
        "[[model.local]]\nid = \"big\"\nfile = \"big.gguf\"\nidle_ttl = \"1s\"\n[[model.local]]\nid = \"emb\"\nfile = \"emb.gguf\"\nrole = \"embedding\"\n",
        "standin_warm = false\n",
    )
    .await?;
    let v: Value = n.post("completions", json!({"model": "big", "prompt": "Tell me", "max_tokens": 4}), false).await?.json().await?;
    assert_eq!(v["object"], "text_completion");
    assert_eq!(v["choices"][0]["text"], " once upon a time");
    assert_eq!(v["usage"]["total_tokens"], 6);
    let text = n.post("completions", json!({"model": "big", "prompt": "Tell me", "stream": true}), false).await?.text().await?;
    assert!(text.contains("\"text\":\" upon\""), "{text}");
    assert!(text.trim_end().ends_with("data: [DONE]"));
    ctl(&n.paths.home, &["load", "big"]).await?;

    let v: Value = n.post("embeddings", json!({"input": ["two words", "x"]}), false).await?.json().await?;
    assert_eq!(v["model"], "emb");
    assert_eq!(v["data"][0]["embedding"], json!([9.0, 2.0, 1.0, 0.0]));
    assert_eq!(v["data"][1]["index"], 1);
    assert!(v["usage"]["prompt_tokens"].as_u64().unwrap_or(0) > 0, "embedding usage is real: {v}");
    let r = n.post("chat/completions", json!({"model": "emb", "messages": [{"role": "user", "content": "x"}]}), false).await?;
    assert_eq!(r.status(), 400, "embedding models don't chat");

    // big has idle_ttl = 1s: the reaper unloads it and stops its engine; emb has none and stays.
    let pid = state(&n.paths, "big").await.pid.unwrap();
    wait_for(&n.paths, "big", 10, |s| *s == ModelState::Unloaded).await;
    let deadline = Instant::now() + Duration::from_secs(10);
    while alive(pid) {
        assert!(Instant::now() < deadline, "idle engine still running");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(state(&n.paths, "emb").await.state, ModelState::Ready);
    n.stop().await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn wraps_a_stock_server_without_owning_it() -> Result<()> {
    // A "stock" server started by hand (the mock, no API key).
    let port = std::net::TcpListener::bind("127.0.0.1:0")?.local_addr()?.port();
    let mut stock = tokio::process::Command::new(MOCK)
        .args(["--host", "127.0.0.1", "--port", &port.to_string(), "--alias", "stock"])
        .env("YMIR_MOCK_LOAD_MS", "0")
        .env_remove("LLAMA_API_KEY")
        .kill_on_drop(true)
        .spawn()?;
    let deadline = Instant::now() + Duration::from_secs(10);
    while reqwest::get(format!("http://127.0.0.1:{port}/health")).await.is_err() {
        assert!(Instant::now() < deadline);
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let n = node(&format!("[[model.local]]\nid = \"stock\"\nbackend = \"vllm\"\nwrap = \"http://127.0.0.1:{port}/v1\"\n"), "").await?;
    let v: Value =
        n.post("chat/completions", json!({"model": "stock", "messages": [{"role": "user", "content": "yo"}]}), false).await?.json().await?;
    assert_eq!(v["choices"][0]["message"]["content"], "Hello from stock! You said: yo");
    let m = state(&n.paths, "stock").await;
    assert!(m.wrapped && m.pid.is_none());
    assert!(ctl(&n.paths.home, &["status"]).await?.contains("wrapped"));
    // Unload detaches; the stock server keeps running.
    ctl(&n.paths.home, &["unload", "stock"]).await?;
    assert_eq!(state(&n.paths, "stock").await.state, ModelState::Unloaded);
    assert!(stock.try_wait()?.is_none(), "Ymir doesn't stop a server it doesn't own");
    n.stop().await;
    assert!(stock.try_wait()?.is_none());
    stock.kill().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn crash_backoff_then_crash_loop() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let gguf = dir.path().join("m.gguf");
    std::fs::write(&gguf, b"GGUF mock")?;
    let opts = SupervisorOptions {
        log_dir: Some(dir.path().join("logs")),
        backoff_base: Duration::from_millis(100),
        backoff_max: Duration::from_millis(400),
        max_crashes: 3,
        ..Default::default()
    };
    let sup = Supervisor::new(vec![EngineConfig::llamacpp(MOCK)], vec![ModelLaunch::new("m", EngineKind::Llamacpp, gguf.to_string_lossy())], opts);
    let mut events = sup.subscribe();
    sup.load("m").await?;
    let pid_of = |s: &Supervisor| s.status()[0].pid.unwrap();
    let wait_ready = |s: Supervisor, old: u32| async move {
        let deadline = Instant::now() + Duration::from_secs(10);
        while s.state("m") != Some(ModelState::Ready) || s.status()[0].pid == Some(old) {
            assert!(Instant::now() < deadline, "no restart: {:?}", s.state("m"));
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    };
    // Two crashes while serving: each restarts on its own after the backoff.
    for crash in 1..=2 {
        let pid = pid_of(&sup);
        // SAFETY: the mock engine this test started.
        unsafe { libc::kill(pid as i32, libc::SIGKILL) };
        wait_ready(sup.clone(), pid).await;
        assert_ne!(pid_of(&sup), pid);
        assert_eq!(sup.status()[0].crashes, crash);
    }
    // The third within the window is a crash loop: no restart, and loads refuse...
    // SAFETY: as above.
    unsafe { libc::kill(pid_of(&sup) as i32, libc::SIGKILL) };
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match events.recv().await? {
            SupervisorEvent::CrashLoop { crashes, .. } => {
                assert_eq!(crashes, 3);
                break;
            }
            _ => assert!(Instant::now() < deadline),
        }
    }
    tokio::time::sleep(Duration::from_millis(600)).await;
    assert!(matches!(sup.state("m"), Some(ModelState::Failed(why)) if why.contains("crash loop")));
    let err = sup.load("m").await.unwrap_err().to_string();
    assert!(err.contains("crash loop"), "{err}");
    // ...until an operator forces it.
    sup.load_with("m", true).await?;
    assert_eq!(sup.state("m"), Some(ModelState::Ready));
    assert_eq!(sup.status()[0].crashes, 0);
    sup.shutdown().await;

    // An engine that dies while loading is reported to the loader, and counted.
    let sup = Supervisor::new(
        vec![EngineConfig::llamacpp("/bin/false")],
        vec![ModelLaunch::new("m", EngineKind::Llamacpp, gguf.to_string_lossy())],
        SupervisorOptions { max_crashes: 2, ..Default::default() },
    );
    assert!(sup.load("m").await.unwrap_err().to_string().contains("while loading"));
    assert!(sup.load("m").await.is_err());
    assert!(sup.load("m").await.unwrap_err().to_string().contains("crash loop"), "repeated load failures stop retrying");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn wrap_refuses_non_loopback_and_reports_a_dead_server() -> Result<()> {
    let mut l = ModelLaunch::new("w", EngineKind::Vllm, "unused");
    l.wrap = Some(Wrap { url: "http://127.0.0.1:9/v1".into(), api_key: None });
    let sup = Supervisor::new(Vec::new(), vec![l.clone()], SupervisorOptions::default());
    let err = sup.load("w").await.unwrap_err().to_string();
    assert!(err.contains("not answering"), "{err}");
    l.wrap = Some(Wrap { url: "http://192.168.1.5:8000".into(), api_key: None });
    let sup = Supervisor::new(Vec::new(), vec![l], SupervisorOptions::default());
    assert!(sup.load("w").await.unwrap_err().to_string().contains("loopback"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn engine_failures_raise_deduped_alerts() -> Result<()> {
    use std::sync::Arc;
    use weirdnet_alerts::{Alerter, MemorySink, rules};
    let mem = Arc::new(MemorySink::default());
    let alerter = Arc::new(Alerter::new("m2e2e", rules::default_rules(&rules::Thresholds::default()), vec![mem.clone()]).with_product("ymir"));
    let dir = tempfile::tempdir()?;
    let gguf = dir.path().join("m.gguf");
    std::fs::write(&gguf, b"GGUF mock")?;
    let sup = Supervisor::new(
        vec![EngineConfig::llamacpp("/bin/false")],
        vec![ModelLaunch::new("m", EngineKind::Llamacpp, gguf.to_string_lossy())],
        SupervisorOptions::default(),
    );
    let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);
    let mon = tokio::spawn(weirdnet_alerts::engine::monitor(alerter.clone(), Some(sup.clone()), Vec::new(), Duration::from_secs(3600), stop_rx));
    tokio::time::sleep(Duration::from_millis(100)).await;
    let _ = sup.load("m").await;
    let _ = sup.load("m").await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let sent = mem.sent();
    // (Probes run too; on a Wi-Fi machine net.route fires as well, which is right.)
    let failures: Vec<_> = sent.iter().filter(|a| a.key == "engine.m").collect();
    assert_eq!(failures.len(), 1, "two failed loads, one alert within the hour: {sent:?}");
    assert!(failures[0].title.contains("failed to load"));
    assert!(alerter.failing().contains(&"engine.m".to_string()));
    let _ = stop_tx.send(true);
    mon.await?;
    Ok(())
}

#[tokio::test]
async fn user_service_unit_in_a_temp_dir() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let home = dir.path().join("home");
    let units = dir.path().join("units");
    let gguf = dir.path().join("m.gguf");
    std::fs::write(&gguf, b"x")?;
    // setup --service with --unit-dir writes the unit there and never touches systemd.
    let out = ctl(
        &home,
        &[
            "setup",
            "--llama-server",
            MOCK,
            "--model",
            gguf.to_str().unwrap(),
            "--service",
            "--unit-dir",
            units.to_str().unwrap(),
            "--ymird",
            "/opt/ymir/bin/ymird",
        ],
    )
    .await?;
    assert!(out.contains("wrote"), "{out}");
    let unit = std::fs::read_to_string(units.join("ymird.service"))?;
    assert!(
        unit.contains("ExecStart=/opt/ymir/bin/ymird --home") && unit.contains(&format!("--home {}", std::fs::canonicalize(&home)?.display())),
        "{unit}"
    );
    assert!(ctl(&home, &["service", "status", "--unit-dir", units.to_str().unwrap()]).await?.starts_with("installed"));
    let dry = ctl(&home, &["service", "install", "--dry-run", "--ymird", "/opt/ymir/bin/ymird"]).await?;
    assert!(dry.contains("would write") && dry.contains("ExecStart=/opt/ymir/bin/ymird"), "{dry}");
    ctl(&home, &["service", "uninstall", "--unit-dir", units.to_str().unwrap()]).await?;
    assert!(!units.join("ymird.service").exists());
    Ok(())
}
