//! First-token latency over Weirdnet on loopback: an app, an inference node and a model node
//! (separate daemons on 127.0.0.1, mock engine). The first token of an answer has to reach the
//! app's bridge in well under a second. Here the app can't reach the model node (apps never talk
//! to model nodes), which is one linked device the app pool's `LinkedNodes` can't identify: with
//! weirdnet-agent 0.4.2 every request waits its whole 3 s `with_wait` for it before the first
//! dial (12 to 15 s in the M5 and M6 setups, which have four or five such devices).
//! `YMIR_TEST_LOG=debug` turns on tracing.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use clap::Parser as _;
use serde_json::{Value, json};
use weirdnet_net::{Keypair, WEIRDNET};
use weirdnet_sync::Roster;
use ymir_core::control::{self, ControlRequest};
use ymir_core::membership::Membership;
use ymir_core::{Daemon, NodeConfig, Paths};

const MOCK: &str = env!("CARGO_BIN_EXE_ymir-mock-engine");

async fn ctl_with(home: &Path, args: &[&str], out: &mut dyn Write, input: &str) -> Result<()> {
    let mut argv = vec!["ymirctl".to_string(), "--home".into(), home.display().to_string()];
    argv.extend(args.iter().map(|a| a.to_string()));
    let cli = ymirctl::Cli::try_parse_from(argv)?;
    let mut input = std::io::Cursor::new(input.as_bytes().to_vec());
    ymirctl::run(cli, out, &mut input).await
}

async fn ctl(home: &Path, args: &[&str]) -> Result<String> {
    let mut out = Vec::new();
    ctl_with(home, args, &mut out, "").await?;
    Ok(String::from_utf8(out)?)
}

/// A daemon's home in `root/name`.
struct Home {
    home: PathBuf,
    paths: Paths,
}

/// node.toml for `roles`, on Weirdnet over 127.0.0.1 only, dialing `peers`.
fn home(root: &Path, name: &str, roles: &str, peers: &[String], models: &[&str]) -> Result<Home> {
    let home = root.join(name);
    let models_dir = root.join("models");
    std::fs::create_dir_all(&home)?;
    std::fs::create_dir_all(&models_dir)?;
    let mut local = String::new();
    for (i, id) in models.iter().enumerate() {
        std::fs::write(models_dir.join(format!("{id}.gguf")), b"GGUF mock")?;
        let role = if i == 0 { "role = \"main\"\n" } else { "" };
        local.push_str(&format!("[[model.local]]\nid = \"{id}\"\nfile = \"{id}.gguf\"\n{role}"));
    }
    let peers = peers.iter().map(|p| format!("\"{p}\"")).collect::<Vec<_>>().join(", ");
    let (m, i, a) = (roles.contains("model"), roles.contains("inference"), roles.contains("app"));
    let toml = format!(
        "[node]\nname = \"{name}\"\n[roles]\nmodel = {m}\ninference = {i}\napp = {a}\n[network]\nlisten = [\"/ip4/127.0.0.1/tcp/0\"]\nmdns = false\ndiscovery = false\nrelays = []\nsync_secs = 1\npeers = [{peers}]\n[model]\nllamacpp = \"system:{MOCK}\"\nvllm = \"off\"\nmodels_dir = \"{}\"\n{local}[inference.alerts]\nprobe_secs = 3600\n[app.bridge]\nport = 0\n",
        models_dir.display()
    );
    std::fs::write(home.join("node.toml"), toml)?;
    Ok(Home { paths: Paths::new(&home), home })
}

/// What `pool link` and `pool add` do, without the network: `from` certifies a new node for
/// `to` and the identity lands there.
fn prelink(from: &Membership, to: &Home, label: &str) -> Membership {
    let key = Keypair::generate_ed25519();
    let mut seed = from.seed().unwrap().0;
    let user = Keypair::ed25519_from_bytes(&mut seed).unwrap();
    let cert = Roster::self_certify(&WEIRDNET, &user, &key.public().to_peer_id().to_string(), label, 1).unwrap();
    from.approve(cert.clone()).unwrap();
    Membership::adopt(&to.paths.pool(), key, &from.seed().unwrap().0, cert, &from.link_settings()).unwrap()
}

async fn start(h: &Home) -> Result<Daemon> {
    Daemon::start(NodeConfig::load(&h.paths.config())?, h.paths.clone()).await
}

/// `/ip4/127.0.0.1/tcp/PORT/p2p/PEER` for a running daemon, from its control status.
async fn addr(h: &Home) -> Result<String> {
    let s = control::request(&h.paths.socket(), &ControlRequest::Status).await?.status.unwrap();
    let n = s.network.expect("on Weirdnet");
    let a = n.listening.iter().find(|a| a.starts_with("/ip4/127.0.0.1/tcp/")).expect("a loopback listener");
    let a = a.split("/p2p/").next().unwrap();
    Ok(format!("{a}/p2p/{}", n.peer))
}

fn bridge(d: &Daemon, h: &Home) -> (reqwest::Client, String, String) {
    let port = d.bridge_addr.unwrap().port();
    let token = ymir_core::token::load(&h.paths).unwrap().unwrap();
    (reqwest::Client::builder().no_proxy().build().unwrap(), format!("http://127.0.0.1:{port}/v1"), token)
}

/// Retries `f` until it is true or `secs` pass.
async fn eventually(secs: u64, what: &str, mut f: impl AsyncFnMut() -> Result<bool>) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(secs);
    loop {
        if f().await? {
            return Ok(());
        }
        if Instant::now() > deadline {
            bail!("gave up waiting for: {what}");
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
}

/// Streams a chat through the app's bridge, calling `on_text` with each piece of the answer as
/// it arrives (so a test can break something mid-answer); returns the whole text.
async fn stream_chat(d: &Daemon, h: &Home, model: &str, q: &str, mut on_text: impl AsyncFnMut(usize)) -> Result<String> {
    let (http, url, token) = bridge(d, h);
    let mut r = http
        .post(format!("{url}/chat/completions"))
        .bearer_auth(&token)
        .json(&json!({"model": model, "stream": true, "temperature": 1, "messages": [{"role": "user", "content": q}]}))
        .send()
        .await?;
    if r.status() != 200 {
        bail!("{}: {}", r.status(), r.text().await?);
    }
    let (mut text, mut buf, mut pieces) = (String::new(), String::new(), 0);
    while let Some(chunk) = r.chunk().await? {
        buf.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(i) = buf.find("\n\n") {
            let event: String = buf.drain(..i + 2).collect();
            let Some(data) = event.trim().strip_prefix("data:") else { continue };
            let Ok(v) = serde_json::from_str::<Value>(data.trim()) else { continue };
            if v.get("error").is_some() {
                bail!("the stream failed after {text:?}: {v}");
            }
            if let Some(t) = v["choices"][0]["delta"]["content"].as_str().filter(|t| !t.is_empty()) {
                text.push_str(t);
                pieces += 1;
                on_text(pieces).await;
            }
        }
    }
    Ok(text)
}

/// Time to the first piece of text and to the whole answer, for one streamed chat.
async fn timed(d: &Daemon, h: &Home, model: &str, q: &str) -> Result<(Duration, Duration)> {
    let t = Instant::now();
    let mut first = None;
    stream_chat(d, h, model, q, async |_| {
        first.get_or_insert(t.elapsed());
    })
    .await?;
    Ok((first.unwrap_or_default(), t.elapsed()))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn first_token_is_fast_over_loopback() -> Result<()> {
    let _ = tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::from_env("YMIR_TEST_LOG")).with_test_writer().try_init();
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();
    let inf_tmp = home(root, "inf", "inference", &[], &[])?;
    let first = Membership::create(&inf_tmp.paths.pool(), "inf")?;
    let (m_tmp, app_tmp) = (home(root, "ma", "model", &[], &["alpha"])?, home(root, "dev", "app", &[], &[])?);
    for (h, label) in [(&m_tmp, "ma"), (&app_tmp, "dev")] {
        prelink(&first, h, label);
    }
    drop(first);

    let model = start(&m_tmp).await?;
    let inf_h = home(root, "inf", "inference", &[addr(&m_tmp).await?], &[])?;
    let inf = start(&inf_h).await?;
    let app_h = home(root, "dev", "app", &[addr(&inf_h).await?], &[])?;
    let app = start(&app_h).await?;
    eventually(60, "the model node joins", async || Ok(ctl(&inf_h.home, &["nodes"]).await?.contains("serving"))).await?;

    // Warm up (admission, and the model loads once).
    let deadline = Instant::now() + Duration::from_secs(100);
    loop {
        match stream_chat(&app, &app_h, "alpha", "warm up", async |_| {}).await {
            Ok(_) => break,
            Err(e) if Instant::now() > deadline => bail!("never answered: {e:#}"),
            Err(_) => tokio::time::sleep(Duration::from_millis(500)).await,
        }
    }

    let mut firsts = Vec::new();
    for i in 0..3 {
        let (f, all) = timed(&app, &app_h, "alpha", &format!("question {i}")).await?;
        eprintln!("question {i}: first token after {f:?}, whole answer after {all:?}");
        firsts.push(f);
    }
    let worst = *firsts.iter().max().unwrap();
    assert!(worst < Duration::from_secs(1), "first token took up to {worst:?} over loopback: {firsts:?}");

    app.shutdown().await;
    inf.shutdown().await;
    model.shutdown().await;
    Ok(())
}
