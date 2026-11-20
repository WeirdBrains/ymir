//! M6: workload spreading. Separate daemons over Weirdnet on 127.0.0.1 with a mock engine:
//! two inference nodes, three model nodes (two replicas of `alpha`, one `beta`) and two app
//! devices. One device fans a batch out; the other chats interactively at the same time. The
//! interactive answers stay fast (p95 under a target), the batch device is not starved (every
//! task answers), and the per-device cap on batch work (`batch_per_device`) is what keeps one
//! device from filling the inference nodes' queues.

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

fn p95(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[((v.len() as f64 * 0.95).ceil() as usize).clamp(1, v.len()) - 1]
}

/// Adds a line to the `[network]` section of a home's node.toml.
fn network_line(h: &Home, line: &str) -> Result<()> {
    let p = h.home.join("node.toml");
    let t = std::fs::read_to_string(&p)?;
    std::fs::write(&p, t.replacen("sync_secs = 1\n", &format!("sync_secs = 1\n{line}\n"), 1))?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn interactive_stays_fast_while_another_device_runs_a_batch() -> Result<()> {
    // SAFETY: set before any daemon starts in this test.
    unsafe { std::env::set_var("YMIR_MOCK_WORD_MS", "40") };
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();
    let i1_tmp = home(root, "inf1", "inference", &[], &[])?;
    let first = Membership::create(&i1_tmp.paths.pool(), "inf1")?;
    let i2_tmp = home(root, "inf2", "inference", &[], &[])?;
    let (ma1_tmp, ma2_tmp, mb_tmp) =
        (home(root, "ma1", "model", &[], &["alpha"])?, home(root, "ma2", "model", &[], &["alpha"])?, home(root, "mb", "model", &[], &["beta"])?);
    let (ap_tmp, bp_tmp) = (home(root, "dev-batch", "app", &[], &[])?, home(root, "dev-chat", "app", &[], &[])?);
    for (h, label) in [(&i2_tmp, "inf2"), (&ma1_tmp, "ma1"), (&ma2_tmp, "ma2"), (&mb_tmp, "mb"), (&ap_tmp, "dev-batch"), (&bp_tmp, "dev-chat")] {
        prelink(&first, h, label);
    }
    drop(first);

    let (ma1, ma2, mb) = (start(&ma1_tmp).await?, start(&ma2_tmp).await?, start(&mb_tmp).await?);
    let models = vec![addr(&ma1_tmp).await?, addr(&ma2_tmp).await?, addr(&mb_tmp).await?];
    let i1_h = home(root, "inf1", "inference", &models, &[])?;
    network_line(&i1_h, "batch_per_device = 2")?;
    let inf1 = start(&i1_h).await?;
    let i1_addr = addr(&i1_h).await?;
    let mut peers = models.clone();
    peers.push(i1_addr.clone());
    let i2_h = home(root, "inf2", "inference", &peers, &[])?;
    network_line(&i2_h, "batch_per_device = 2")?;
    let inf2 = start(&i2_h).await?;
    let i2_addr = addr(&i2_h).await?;
    let ap_h = home(root, "dev-batch", "app", &[i1_addr.clone(), i2_addr.clone()], &[])?;
    let batch_dev = start(&ap_h).await?;
    let bp_h = home(root, "dev-chat", "app", &[i1_addr, i2_addr], &[])?;
    let chat_dev = start(&bp_h).await?;
    for h in [&i1_h, &i2_h] {
        eventually(60, "all three model nodes join", async || Ok(ctl(&h.home, &["nodes"]).await?.matches("serving").count() == 3)).await?;
    }

    // Warm up both devices (admission, and the models load once).
    for (d, h) in [(&batch_dev, &ap_h), (&chat_dev, &bp_h)] {
        let deadline = Instant::now() + Duration::from_secs(100);
        loop {
            match stream_chat(d, h, "alpha", "warm up", async |_| {}).await {
                Ok(_) => break,
                Err(e) if Instant::now() > deadline => bail!("a device was never admitted: {e:#}"),
                Err(_) => tokio::time::sleep(Duration::from_millis(500)).await,
            }
        }
    }
    stream_chat(&chat_dev, &bp_h, "beta", "warm up beta", async |_| {}).await?;

    // Baseline: five questions with nothing else running.
    let mut quiet = Vec::new();
    for i in 0..5 {
        let model = if i % 3 == 2 { "beta" } else { "alpha" };
        let t = Instant::now();
        stream_chat(&chat_dev, &bp_h, model, &format!("quiet {i}"), async |_| {}).await?;
        quiet.push(t.elapsed());
    }

    // The batch device: 24 prompts, 8 at a time, on a blocking thread (`ymirctl` isn't Send).
    let prompts: String = (0..24).map(|i| format!("batch question {i}\n")).collect();
    let file = root.join("prompts.txt");
    std::fs::write(&file, prompts)?;
    let batch = {
        let (home, file) = (ap_h.home.clone(), file.display().to_string());
        let rt = tokio::runtime::Handle::current();
        tokio::task::spawn_blocking(move || rt.block_on(ctl(&home, &["batch", &file, "--model", "alpha", "--concurrency", "8", "--retries", "40"])))
    };
    // Meanwhile the other device chats interactively, one question after another.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let mut waits = Vec::new();
    for i in 0..10 {
        let model = if i % 3 == 2 { "beta" } else { "alpha" };
        let q = format!("interactive {i}");
        let t = Instant::now();
        let text = stream_chat(&chat_dev, &bp_h, model, &q, async |_| {}).await?;
        waits.push(t.elapsed());
        assert!(text.contains(&q), "{text}");
    }
    let out = batch.await??;

    // The batch device is not starved: all 24 answered, in order, each to its own question.
    let lines: Vec<&str> = out.lines().filter(|l| !l.starts_with('#')).collect();
    assert_eq!(lines.len(), 24, "{out}");
    for (i, l) in lines.iter().enumerate() {
        assert!(l.starts_with(&format!("{i}\t")) && l.ends_with(&format!("batch question {i}")), "{l}");
    }
    assert!(out.contains("24 task(s), 0 failed"), "{out}");
    // The interactive device stayed fast while that ran: a whole answer is about half a second.
    let (worst, p) = (waits.iter().max().unwrap(), p95(waits.clone()));
    let base = p95(quiet.clone());
    eprintln!("interactive: quiet p95 {base:?}; with a batch running p95 {p:?}, worst {worst:?} over {} answers; all {waits:?}", waits.len());
    // The target: under a batch, interactive p95 stays within 2 s plus twice the quiet p95 (the
    // box this runs on is shared and noisy, so the bound is relative).
    assert!(p < Duration::from_secs(2) + base * 2, "interactive p95 {p:?} with a batch running, {base:?} without");

    batch_dev.shutdown().await;
    chat_dev.shutdown().await;
    inf1.shutdown().await;
    inf2.shutdown().await;
    for m in [ma1, ma2, mb] {
        m.shutdown().await;
    }
    Ok(())
}
