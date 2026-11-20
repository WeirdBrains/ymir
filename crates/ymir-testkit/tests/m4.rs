//! M4: several model nodes on separate daemons over Weirdnet on 127.0.0.1 (nothing leaves the
//! box). Two model nodes with different model sets sit behind one inference node; an app node
//! asks through it. Covers the merged catalog, routing by model, drain and resume, a model node
//! dying (the request moves to the other before the first token, and the node leaves the list),
//! and a model node joining later.

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
        "[node]\nname = \"{name}\"\n[roles]\nmodel = {m}\ninference = {i}\napp = {a}\n[network]\nlisten = [\"/ip4/127.0.0.1/tcp/0\"]\nmdns = false\ndiscovery = false\nrelays = []\npeers = [{peers}]\n[model]\nllamacpp = \"system:{MOCK}\"\nvllm = \"off\"\nmodels_dir = \"{}\"\n{local}[inference.alerts]\nprobe_secs = 3600\n[app.bridge]\nport = 0\n",
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

async fn chat_model(d: &Daemon, h: &Home, model: &str, text: &str) -> Result<(u16, Value)> {
    let (http, url, token) = bridge(d, h);
    let r = http
        .post(format!("{url}/chat/completions"))
        .bearer_auth(&token)
        .json(&json!({"model": model, "temperature": 1, "messages": [{"role": "user", "content": text}]}))
        .send()
        .await?;
    Ok((r.status().as_u16(), r.json().await.unwrap_or(Value::Null)))
}

/// Retries `f` until it answers 200 or `secs` pass (discovery and dialing take a moment).
async fn until_ok(secs: u64, mut f: impl AsyncFnMut() -> Result<(u16, Value)>) -> Result<Value> {
    let deadline = Instant::now() + Duration::from_secs(secs);
    loop {
        let (status, v) = f().await?;
        if status == 200 {
            return Ok(v);
        }
        if Instant::now() > deadline {
            bail!("still {status}: {v}");
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

fn futures_nodes(h: &Home) -> String {
    let h = h.home.clone();
    std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(ctl(&h, &["nodes"])).unwrap_or_default()
    })
    .join()
    .unwrap_or_default()
}

async fn peer_of(h: &Home) -> Result<String> {
    Ok(control::request(&h.paths.socket(), &ControlRequest::Status).await?.status.unwrap().network.expect("on Weirdnet").peer)
}

/// Asks `model` through the app and returns which model node answered. A `needs-admission`
/// refusal is the upstream roster-propagation flake (see DESIGN.md, "Flaky upstream roster
/// propagation"): the device just linked isn't admitted everywhere yet, so this retries those
/// for up to 100 s (it clears on its own after a roster pass) and fails on anything else.
async fn served_by(d: &Daemon, h: &Home, model: &str, text: &str) -> Result<String> {
    let deadline = Instant::now() + Duration::from_secs(100);
    loop {
        let (status, v) = chat_model(d, h, model, text).await?;
        if status == 200 {
            return Ok(v["weirdnet"]["served_by"].as_str().unwrap_or("?").to_string());
        }
        if !v.to_string().contains("needs-admission") || Instant::now() > deadline {
            bail!("{model}: {status} {v}");
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn several_model_nodes_on_separate_daemons() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();

    // One identity: the inference node starts it and links the two model nodes, the app node and
    // a third model node that joins later.
    let inf_tmp = home(root, "inf", "inference", &[], &[])?;
    let first = Membership::create(&inf_tmp.paths.pool(), "inf")?;
    let (m1_tmp, m2_tmp, app_tmp, m3_tmp) = (
        home(root, "server-a", "model", &[], &["alpha"])?,
        home(root, "server-b", "model", &[], &["alpha", "beta"])?,
        home(root, "app", "app", &[], &[])?,
        home(root, "m3", "model", &[], &["gamma"])?,
    );
    for (h, label) in [(&m1_tmp, "server-a"), (&m2_tmp, "server-b"), (&app_tmp, "app"), (&m3_tmp, "m3")] {
        prelink(&first, h, label);
    }
    drop(first);

    let model1 = start(&m1_tmp).await?;
    let model2 = start(&m2_tmp).await?;
    let peers = vec![addr(&m1_tmp).await?, addr(&m2_tmp).await?];
    let inf_h = home(root, "inf", "inference", &peers, &[])?;
    let inf = start(&inf_h).await?;
    let inf_addr = addr(&inf_h).await?;
    let app_h = home(root, "app", "app", std::slice::from_ref(&inf_addr), &[])?;
    let app = start(&app_h).await?;
    let (p1, p2) = (peer_of(&m1_tmp).await?, peer_of(&m2_tmp).await?);

    // The inference node finds both model nodes and merges their models: alpha is on both.
    eventually(40, "both model nodes join", async || Ok(ctl(&inf_h.home, &["nodes"]).await?.matches("serving").count() == 2)).await?;
    let nodes = ctl(&inf_h.home, &["nodes"]).await?;
    assert!(nodes.contains(&p1) && nodes.contains(&p2), "{nodes}");
    let models = ctl(&app_h.home, &["models"]).await?;
    assert!(models.contains("alpha") && models.contains("beta"), "{models}");

    // Each model is answered by a node that has it: beta only by server-b.
    for i in 0..2 {
        until_ok(30, async || chat_model(&app, &app_h, "beta", &format!("beta {i}")).await).await?;
    }
    assert_eq!(served_by(&app, &app_h, "beta", "again").await?, p2);
    let first_alpha =
        served_by(&app, &app_h, "alpha", "which node?").await.map_err(|e| anyhow::anyhow!("first alpha: {e}; nodes: {}", futures_nodes(&inf_h)))?;
    assert!(first_alpha == p1 || first_alpha == p2);

    // Drain server-a from the inference node: it stops getting requests, and `nodes` says so.
    let out = ctl(&inf_h.home, &["drain", &p1]).await?;
    assert!(out.contains("drained"), "{out}");
    for i in 0..4 {
        assert_eq!(
            served_by(&app, &app_h, "alpha", &format!("drained {i}")).await.map_err(|e| anyhow::anyhow!("drained alpha: {e}"))?,
            p2,
            "server-a is draining"
        );
    }
    let nodes = ctl(&inf_h.home, &["nodes"]).await?;
    assert!(nodes.lines().any(|l| l.contains(&p1) && l.contains("draining")), "{nodes}");
    assert!(ctl(&inf_h.home, &["drain", "no-such-node"]).await.is_err());
    // The drain reached the node itself (its capability record says so), and resuming undoes it.
    let out = ctl(&inf_h.home, &["drain", &p1, "--resume"]).await?;
    assert!(out.contains("again"), "{out}");
    eventually(20, "server-a serves again", async || Ok(!ctl(&inf_h.home, &["nodes"]).await?.contains("draining"))).await?;

    // server-a dies. Requests for alpha still succeed, from server-b: the router may try the dead node
    // first, and moves on before the first token.
    model1.shutdown().await;
    for i in 0..4 {
        assert_eq!(served_by(&app, &app_h, "alpha", &format!("after the crash {i}")).await?, p2);
    }
    // The dead node leaves the list (its record stream ended), and its models with it.
    eventually(60, "server-a leaves the list", async || Ok(!ctl(&inf_h.home, &["nodes"]).await?.contains(&p1))).await?;

    // A third model node joins later, with a model of its own.
    let m3_h = home(root, "m3", "model", std::slice::from_ref(&inf_addr), &["gamma"])?;
    let model3 = start(&m3_h).await?;
    let p3 = peer_of(&m3_h).await?;
    eventually(60, "m3 joins", async || Ok(ctl(&inf_h.home, &["nodes"]).await?.contains(&p3))).await?;
    let v = until_ok(30, async || chat_model(&app, &app_h, "gamma", "new here").await).await?;
    assert_eq!(v["weirdnet"]["served_by"], p3);
    let models = ctl(&app_h.home, &["models"]).await?;
    assert!(models.contains("gamma") && models.contains("beta"), "{models}");

    app.shutdown().await;
    inf.shutdown().await;
    model3.shutdown().await;
    model2.shutdown().await;
    Ok(())
}
