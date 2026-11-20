//! M3: the roles on separate daemons, over Weirdnet on 127.0.0.1 (no mDNS, discovery or relays;
//! nothing leaves the box). A model-only node runs the mock engine, an inference-only node finds
//! it through the roster, and an app-only node's bridge reaches the inference node. A device
//! outside the roster is refused, and `ymirctl pool link` / `pool add` move an identity between
//! two homes with the 6-digit check.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use clap::Parser as _;
use serde_json::{Value, json};
use weirdnet_net::{Keypair, WEIRDNET};
use weirdnet_sync::Roster;
use ymir_core::control::{self, ControlRequest, ModelState};
use ymir_core::membership::Membership;
use ymir_core::{Daemon, NodeConfig, Paths};

const MOCK: &str = env!("CARGO_BIN_EXE_ymir-mock-engine");

/// Writes into a buffer another future can read while the command runs.
#[derive(Clone, Default)]
struct Shared(Arc<Mutex<Vec<u8>>>);

impl Write for Shared {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Shared {
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.0.lock().unwrap()).into_owned()
    }
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
fn home(root: &Path, name: &str, roles: &str, peers: &[String]) -> Result<Home> {
    let home = root.join(name);
    let models = root.join("models");
    std::fs::create_dir_all(&home)?;
    std::fs::create_dir_all(&models)?;
    std::fs::write(models.join("alpha.gguf"), b"GGUF mock")?;
    let local = if roles.contains("model") { "[[model.local]]\nid = \"alpha\"\nfile = \"alpha.gguf\"\nrole = \"main\"\n" } else { "" };
    let peers = peers.iter().map(|p| format!("\"{p}\"")).collect::<Vec<_>>().join(", ");
    let (m, i, a) = (roles.contains("model"), roles.contains("inference"), roles.contains("app"));
    let toml = format!(
        "[node]\nname = \"{name}\"\n[roles]\nmodel = {m}\ninference = {i}\napp = {a}\n[network]\nlisten = [\"/ip4/127.0.0.1/tcp/0\"]\nmdns = false\ndiscovery = false\nrelays = []\npeers = [{peers}]\n[model]\nllamacpp = \"system:{MOCK}\"\nvllm = \"off\"\nmodels_dir = \"{}\"\n{local}[inference.alerts]\nprobe_secs = 3600\n[app.bridge]\nport = 0\n",
        models.display()
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

async fn chat(d: &Daemon, h: &Home, text: &str) -> Result<(u16, Value)> {
    let (http, url, token) = bridge(d, h);
    let r = http
        .post(format!("{url}/chat/completions"))
        .bearer_auth(&token)
        .json(&json!({"model": "alpha", "temperature": 0, "messages": [{"role": "user", "content": text}]}))
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn roles_on_three_daemons_over_weirdnet() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();

    // Identities: the model node starts the user's Weirdnet, the inference node is linked from
    // it, and the app node from the inference node (so the inference node knows both).
    let model_h = home(root, "model", "model", &[])?;
    let first = Membership::create(&model_h.paths.pool(), "model")?;
    let inf_tmp = home(root, "inf", "inference", &[])?;
    let inf_m = prelink(&first, &inf_tmp, "inf");
    drop(first);

    let model = start(&model_h).await?;
    let model_addr = addr(&model_h).await?;
    let inf_h = home(root, "inf", "inference", std::slice::from_ref(&model_addr))?;
    let app_tmp = home(root, "app", "app", &[])?;
    let app_m = prelink(&inf_m, &app_tmp, "app");
    drop((inf_m, app_m));
    let inf = start(&inf_h).await?;
    let inf_addr = addr(&inf_h).await?;
    let app_h = home(root, "app", "app", std::slice::from_ref(&inf_addr))?;
    let app = start(&app_h).await?;

    // ymirctl on the app node: on Weirdnet, no models of its own.
    let st = ctl(&app_h.home, &["status"]).await?;
    assert!(st.contains("roles: app") && st.contains("Weirdnet:") && st.contains("2 linked devices"), "{st}");
    let devices = ctl(&app_h.home, &["pool", "devices"]).await?;
    assert!(devices.contains("this node") && devices.contains("model") && devices.contains("inf"), "{devices}");

    // The inference node finds the model node; `ymirctl load` there loads it remotely.
    let inf_home = inf_h.home.clone();
    until_ok(20, async || {
        let r = control::request(&inf_h.paths.socket(), &ControlRequest::Load { model: "alpha".into(), node: None }).await?;
        Ok((if r.ok { 200 } else { 503 }, json!(r.message)))
    })
    .await?;
    let s = control::request(&model_h.paths.socket(), &ControlRequest::Status).await?.status.unwrap();
    assert_eq!(s.models.iter().find(|m| m.id == "alpha").unwrap().state, ModelState::Ready);
    // The inference node hears the change through the model node's capability updates.
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let listed = ctl(&inf_home, &["status"]).await?;
        if listed.contains("alpha") && listed.contains("ready") {
            break;
        }
        assert!(Instant::now() < deadline, "{listed}");
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    // app bridge → Weirdnet → inference → Weirdnet → model node → engine.
    let v = until_ok(20, async || chat(&app, &app_h, "ping").await).await?;
    assert_eq!(v["choices"][0]["message"]["content"], "Hello from alpha! You said: ping");
    let models = ctl(&app_h.home, &["models"]).await?;
    assert!(models.contains("alpha") && models.contains("ready"), "{models}");
    // The same question again (temperature 0) comes from the inference node's cache.
    let (_, v) = chat(&app, &app_h, "ping").await?;
    assert_eq!(v["weirdnet"]["cache_hit"], true, "{v}");
    let reply = ctl(&app_h.home, &["chat", "--model", "alpha", "streamed"]).await?;
    assert_eq!(reply.trim(), "Hello from alpha! You said: streamed");

    // A device with an identity of its own is not in this user's roster: refused.
    let stranger_h = home(root, "stranger", "app", std::slice::from_ref(&inf_addr))?;
    Membership::create(&stranger_h.paths.pool(), "stranger")?;
    let stranger = start(&stranger_h).await?;
    let (status, v) = chat(&stranger, &stranger_h, "let me in").await?;
    assert_ne!(status, 200, "{v}");

    // Revoking the app node on the inference node cuts it off.
    let app_peer = control::request(&app_h.paths.socket(), &ControlRequest::Status).await?.status.unwrap().network.unwrap().peer;
    let out = ctl(&inf_home, &["pool", "revoke", &app_peer]).await?;
    assert!(out.contains("revoked"), "{out}");
    let (status, _) = chat(&app, &app_h, "still there?").await?;
    assert_ne!(status, 200);

    stranger.shutdown().await;
    app.shutdown().await;
    inf.shutdown().await;
    model.shutdown().await;
    Ok(())
}

/// One box with both the inference and the model role serves infer/2 to apps (inference role) and
/// the model-node protocol (model role, raw inference on its own protocol id) at the same time.
/// A second, inference-only node uses the combined box's model role over that protocol.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_combined_node_serves_both_roles() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();

    let comb_h = home(root, "comb", "model inference", &[])?;
    let first = Membership::create(&comb_h.paths.pool(), "comb")?;
    let inf_tmp = home(root, "inf", "inference", &[])?;
    let inf_m = prelink(&first, &inf_tmp, "inf");
    let app_tmp = home(root, "app", "app", &[])?;
    let app_m = prelink(&inf_m, &app_tmp, "app");
    let direct_tmp = home(root, "direct", "app", &[])?;
    let direct_m = prelink(&first, &direct_tmp, "direct");
    drop((first, inf_m, app_m, direct_m));

    let comb = start(&comb_h).await?;
    let comb_addr = addr(&comb_h).await?;
    let inf_h = home(root, "inf", "inference", std::slice::from_ref(&comb_addr))?;
    let inf = start(&inf_h).await?;
    let inf_addr = addr(&inf_h).await?;
    let app_h = home(root, "app", "app", std::slice::from_ref(&inf_addr))?;
    let app = start(&app_h).await?;
    let direct_h = home(root, "direct", "app", std::slice::from_ref(&comb_addr))?;
    let direct = start(&direct_h).await?;

    // An app straight on the combined box: its inference role (infer/2) answers from its own model.
    let v = until_ok(30, async || chat(&direct, &direct_h, "to the combined box").await).await?;
    assert_eq!(v["choices"][0]["message"]["content"], "Hello from alpha! You said: to the combined box");

    // An app on the inference-only node: that node dials the combined box's model role.
    let v = until_ok(30, async || chat(&app, &app_h, "through the second node").await).await?;
    assert_eq!(v["choices"][0]["message"]["content"], "Hello from alpha! You said: through the second node");
    let loaded = until_ok(20, async || {
        let r = control::request(&inf_h.paths.socket(), &ControlRequest::Status).await?;
        let on = r.status.is_some_and(|s| s.models.iter().any(|m| m.id == "alpha"));
        Ok((if on { 200 } else { 503 }, json!(null)))
    })
    .await;
    assert!(loaded.is_ok(), "the second node sees the combined box's model");

    direct.shutdown().await;
    app.shutdown().await;
    inf.shutdown().await;
    comb.shutdown().await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pool_link_and_add_move_the_identity() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let old = home(tmp.path(), "old", "inference", &[])?;
    let new = home(tmp.path(), "new", "app", &[])?;
    assert!(ctl(&old.home, &["pool", "init"]).await.is_err(), "init without --standalone");
    let out = ctl(&old.home, &["pool", "init", "--standalone"]).await?;
    assert!(out.contains("New identity"), "{out}");

    // `pool link` on the new home shows a code; `pool add` on the old one takes it.
    let shown = Shared::default();
    let mut link_out = shown.clone();
    let link = ctl_with(&new.home, &["pool", "link"], &mut link_out, "y\n");
    let add = async {
        let deadline = Instant::now() + Duration::from_secs(20);
        let code = loop {
            if let Some(c) = shown.text().split_whitespace().find(|w| w.starts_with("weirdnet-ymir://link/")) {
                break c.to_string();
            }
            assert!(Instant::now() < deadline, "no code shown: {}", shown.text());
            tokio::time::sleep(Duration::from_millis(50)).await;
        };
        // Morph's scheme carries the same payload, and `pool add` takes it.
        let morph = code.replacen("weirdnet-ymir://", "weirdnet-morph://", 1);
        ctl(&old.home, &["pool", "add", &morph, "--yes"]).await
    };
    let (linked, added) = tokio::join!(link, add);
    let added = added?;
    linked?;
    assert!(added.contains("the code is") && added.contains("Linked Ymir on new"), "{added}");
    assert!(shown.text().contains("Linked as Ymir on new"), "{}", shown.text());

    let a = Membership::open(&old.paths.pool())?.unwrap();
    let b = Membership::open(&new.paths.pool())?.unwrap();
    assert_eq!(a.user_id(), b.user_id());
    // The signed roster label is the hint: "Ymir on <name>".
    assert!(a.roster().active().any(|c| c.peer_id == b.peer_id().to_string() && c.label == "Ymir on new"));
    assert!(b.roster().active().any(|c| c.peer_id == a.peer_id().to_string()));
    // Linking twice is refused.
    assert!(ctl(&new.home, &["pool", "link"]).await.is_err());
    Ok(())
}
