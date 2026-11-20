//! M5: answers that survive a node dying mid-answer, and conversations kept by the inference
//! nodes. Separate daemons over Weirdnet on 127.0.0.1 (nothing leaves the box), with a mock
//! engine that streams slowly so a node can be killed in the middle of an answer.
//!
//! The two failover tests need `weirdnet-agent` with the prefix fix (unreleased after 0.4.1), so
//! they run with `--features agent-prefix` and a local `[patch]`; the conversation test needs
//! nothing unreleased.
//!
//! - A model node dies mid-answer: the inference node continues the answer on another model node
//!   from the text already streamed (an assistant prefix), and the asker sees one whole answer.
//! - An inference node dies mid-answer: the app's pool resumes on the other inference node, which
//!   gets the conversation by sync (pull-on-demand from the app), continues from the text already
//!   shown, writes the finished answer once, and the answer reaches every node's copy.

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

/// What the mock engine says for the question `q` on a model called alpha.
fn said(q: &str) -> String {
    format!("Hello from alpha! You said: {q}")
}

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

async fn peer_of(h: &Home) -> Result<String> {
    Ok(control::request(&h.paths.socket(), &ControlRequest::Status).await?.status.unwrap().network.expect("on Weirdnet").peer)
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
            if let Some(t) = v["choices"][0]["delta"]["content"].as_str().filter(|t| !t.is_empty()) {
                text.push_str(t);
                pieces += 1;
                on_text(pieces).await;
            }
        }
    }
    Ok(text)
}

#[cfg(feature = "agent-prefix")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_model_node_dying_mid_answer_is_resumed_on_another() -> Result<()> {
    // SAFETY: set before any daemon starts in this test; both tests set the same value.
    unsafe { std::env::set_var("YMIR_MOCK_WORD_MS", "300") };
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();
    let inf_tmp = home(root, "inf", "inference", &[], &[])?;
    let first = Membership::create(&inf_tmp.paths.pool(), "inf")?;
    let (m1_tmp, m2_tmp, app_tmp) =
        (home(root, "server-a", "model", &[], &["alpha"])?, home(root, "server-b", "model", &[], &["alpha"])?, home(root, "app", "app", &[], &[])?);
    for (h, label) in [(&m1_tmp, "server-a"), (&m2_tmp, "server-b"), (&app_tmp, "app")] {
        prelink(&first, h, label);
    }
    drop(first);

    let mut model1 = Some(start(&m1_tmp).await?);
    let model2 = start(&m2_tmp).await?;
    let peers = vec![addr(&m1_tmp).await?, addr(&m2_tmp).await?];
    let inf_h = home(root, "inf", "inference", &peers, &[])?;
    let inf = start(&inf_h).await?;
    let app_h = home(root, "app", "app", &[addr(&inf_h).await?], &[])?;
    let app = start(&app_h).await?;
    let p2 = peer_of(&m2_tmp).await?;
    eventually(40, "both model nodes join", async || Ok(ctl(&inf_h.home, &["nodes"]).await?.matches("serving").count() == 2)).await?;

    // A first answer proves the app is admitted (roster propagation can take a moment) and
    // loads the model on whichever node got it. Then only server-a takes the next request.
    let deadline = Instant::now() + Duration::from_secs(100);
    loop {
        match stream_chat(&app, &app_h, "alpha", "warm up", async |_| {}).await {
            Ok(_) => break,
            Err(e) if Instant::now() > deadline => bail!("the app was never admitted: {e:#}"),
            Err(_) => tokio::time::sleep(Duration::from_millis(500)).await,
        }
    }
    ctl(&inf_h.home, &["drain", &p2]).await?;

    let q = "tell me a long story please about a very long road home";
    let mut killed = false;
    let text = stream_chat(&app, &app_h, "alpha", q, async |pieces| {
        // Three words in, server-b takes requests again and server-a dies: the answer must go on from there.
        if pieces == 3 && !killed {
            killed = true;
            ctl(&inf_h.home, &["drain", &p2, "--resume"]).await.unwrap();
            model1.take().expect("killed once").shutdown().await;
        }
    })
    .await?;
    assert!(killed, "the answer was too short to cut");
    assert_eq!(text, said(q), "whole, in order, nothing repeated");
    let nodes = ctl(&inf_h.home, &["nodes"]).await?;
    assert!(nodes.contains(&p2), "{nodes}");

    app.shutdown().await;
    inf.shutdown().await;
    model2.shutdown().await;
    Ok(())
}

#[cfg(feature = "agent-prefix")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_inference_node_dying_mid_answer_is_resumed_with_the_full_context() -> Result<()> {
    // SAFETY: as above.
    unsafe { std::env::set_var("YMIR_MOCK_WORD_MS", "300") };
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();
    let i1_tmp = home(root, "inf1", "inference", &[], &[])?;
    let first = Membership::create(&i1_tmp.paths.pool(), "inf1")?;
    let (i2_tmp, m_tmp, app_tmp) =
        (home(root, "inf2", "inference", &[], &[])?, home(root, "m", "model", &[], &["alpha"])?, home(root, "app", "app", &[], &[])?);
    for (h, label) in [(&i2_tmp, "inf2"), (&m_tmp, "m"), (&app_tmp, "app")] {
        prelink(&first, h, label);
    }
    drop(first);

    let model = start(&m_tmp).await?;
    let m_addr = addr(&m_tmp).await?;
    let i1_h = home(root, "inf1", "inference", std::slice::from_ref(&m_addr), &[])?;
    let inf1 = start(&i1_h).await?;
    let i1_addr = addr(&i1_h).await?;
    let i2_h = home(root, "inf2", "inference", &[m_addr, i1_addr.clone()], &[])?;
    let inf2 = start(&i2_h).await?;
    let app_h = home(root, "app", "app", &[i1_addr, addr(&i2_h).await?], &[])?;
    let app = start(&app_h).await?;
    for h in [&i1_h, &i2_h] {
        eventually(40, "the model node joins", async || Ok(ctl(&h.home, &["nodes"]).await?.matches("serving").count() == 1)).await?;
    }

    // A first conversation, to be sure the app is admitted and the key chain is in place (a
    // sealed write waits for the first rotation). Its answer lands on every node by sync.
    let deadline = Instant::now() + Duration::from_secs(100);
    let warm = loop {
        match ctl(&app_h.home, &["context", "ask", "warm", "warm up", "--model", "alpha"]).await {
            Ok(a) => break a,
            Err(e) if Instant::now() > deadline => bail!("warm-up never worked: {e:#}"),
            Err(_) => tokio::time::sleep(Duration::from_millis(500)).await,
        }
    };
    assert!(warm.contains(&said("warm up")), "{warm}");
    for h in [&i1_h, &i2_h, &app_h] {
        eventually(30, "the warm-up conversation reaches every node", async || {
            Ok(ctl(&h.home, &["context", "show", "warm"]).await.is_ok_and(|s| s.contains("assistant: Hello from alpha! You said: warm up")))
        })
        .await?;
    }

    // The real one. Whichever inference node answers is killed three words in.
    let q = "tell me a long story please about a very long road home";
    let ask = {
        let home = app_h.home.clone();
        let rt = tokio::runtime::Handle::current();
        // `ymirctl`'s futures aren't `Send`: run it on a blocking thread.
        tokio::task::spawn_blocking(move || rt.block_on(ctl(&home, &["context", "ask", "story", q, "--model", "alpha"])))
    };
    let coords = [inf1.coordinator.clone().unwrap(), inf2.coordinator.clone().unwrap()];
    eventually(30, "an inference node starts the answer", async || Ok(coords.iter().any(|c| c.in_flight() > 0))).await?;
    tokio::time::sleep(Duration::from_millis(500)).await;
    let (dying, survivor, surv_h) = if coords[0].in_flight() > 0 { (inf1, inf2, &i2_h) } else { (inf2, inf1, &i1_h) };
    dying.shutdown().await;

    let out = ask.await??;
    assert!(out.contains(&said(q)), "{out}");
    assert!(out.contains("moved 1 time"), "the answer moved to the surviving node: {out}");
    // The text came out whole: the answer appears once in the output.
    assert_eq!(out.matches(&said(q)).count(), 1, "{out}");

    // The surviving node wrote the finished answer once, with the whole text, and the app has it.
    for h in [surv_h, &app_h] {
        eventually(30, "the answer is in the conversation", async || {
            let s = ctl(&h.home, &["context", "show", "story"]).await.unwrap_or_default();
            Ok(s.matches("assistant:").count() == 1 && s.contains(&format!("assistant: {}", said(q))))
        })
        .await?;
    }
    let shown = ctl(&surv_h.home, &["context", "show", "story"]).await?;
    assert!(shown.contains(&format!("user: {q}")), "{shown}");

    app.shutdown().await;
    survivor.shutdown().await;
    model.shutdown().await;
    Ok(())
}

/// Conversations sync among the inference nodes: the app asks over a conversation, one inference
/// node answers and writes the answer, and every node ends with the same conversation; a later
/// question is answered by whichever node, from the whole context.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn conversations_sync_and_any_inference_node_answers_from_them() -> Result<()> {
    // SAFETY: as above; a quick mock here.
    unsafe { std::env::set_var("YMIR_MOCK_WORD_MS", "5") };
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();
    let i1_tmp = home(root, "inf1", "inference", &[], &[])?;
    let first = Membership::create(&i1_tmp.paths.pool(), "inf1")?;
    let (i2_tmp, m_tmp, app_tmp) =
        (home(root, "inf2", "inference", &[], &[])?, home(root, "m", "model", &[], &["alpha"])?, home(root, "app", "app", &[], &[])?);
    for (h, label) in [(&i2_tmp, "inf2"), (&m_tmp, "m"), (&app_tmp, "app")] {
        prelink(&first, h, label);
    }
    drop(first);

    let model = start(&m_tmp).await?;
    let m_addr = addr(&m_tmp).await?;
    let i1_h = home(root, "inf1", "inference", std::slice::from_ref(&m_addr), &[])?;
    let inf1 = start(&i1_h).await?;
    let i1_addr = addr(&i1_h).await?;
    let i2_h = home(root, "inf2", "inference", &[m_addr, i1_addr.clone()], &[])?;
    let inf2 = start(&i2_h).await?;
    let app_h = home(root, "app", "app", &[i1_addr, addr(&i2_h).await?], &[])?;
    let app = start(&app_h).await?;
    for h in [&i1_h, &i2_h] {
        eventually(40, "the model node joins", async || Ok(ctl(&h.home, &["nodes"]).await?.matches("serving").count() == 1)).await?;
    }

    // A conversation of two turns; the second sees the first (the mock echoes the last user turn,
    // and the conversation must hold the earlier ones for the node that answers).
    let deadline = Instant::now() + Duration::from_secs(100);
    let a1 = loop {
        match ctl(&app_h.home, &["context", "ask", "c1", "first question", "--model", "alpha"]).await {
            Ok(a) => break a,
            Err(e) if Instant::now() > deadline => bail!("never answered: {e:#}"),
            Err(_) => tokio::time::sleep(Duration::from_millis(500)).await,
        }
    };
    assert!(a1.contains(&said("first question")), "{a1}");
    let a2 = ctl(&app_h.home, &["context", "ask", "c1", "second question", "--model", "alpha"]).await?;
    assert!(a2.contains(&said("second question")), "{a2}");

    // Every node ends with the whole conversation: four messages, in order, each answer once.
    let want = [
        "user: first question",
        "assistant: Hello from alpha! You said: first question",
        "user: second question",
        "assistant: Hello from alpha! You said: second question",
    ];
    for h in [&i1_h, &i2_h, &app_h] {
        eventually(40, "the conversation reaches every node", async || {
            let s = ctl(&h.home, &["context", "show", "c1"]).await.unwrap_or_default();
            let lines: Vec<&str> = s.lines().skip(1).map(str::trim).collect();
            Ok(lines == want)
        })
        .await?;
    }

    // A tombstone written on one node removes it everywhere.
    ctl(&i2_h.home, &["context", "delete", "c1"]).await?;
    for h in [&i1_h, &app_h] {
        eventually(40, "the deletion reaches every node", async || Ok(ctl(&h.home, &["context", "show", "c1"]).await.is_err())).await?;
    }
    // A conversation that was never synced is not answered from thin air.
    assert!(ctl(&app_h.home, &["context", "show", "nope"]).await.is_err());

    app.shutdown().await;
    inf1.shutdown().await;
    inf2.shutdown().await;
    model.shutdown().await;
    Ok(())
}
