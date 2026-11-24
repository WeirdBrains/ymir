//! M10: cert-only server links. Servers join with a certificate and a user-signed service
//! grant, never the seed. Over Weirdnet on 127.0.0.1 (no mDNS, discovery or relays; nothing
//! leaves the box): `pool link --cert-only` and `pool add --service` with the 6-digit check, the
//! seed-holder-only commands refused on a cert-only node, `pool cert-only` moving a node that
//! holds the seed (the seed file is overwritten and deleted), and daemons where a cert-only
//! model node serves, a cert-only inference node syncs sealed conversations, and revoking it
//! from the seed holder cuts it off with the identity unchanged.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use clap::Parser as _;
use serde_json::{Value, json};
use weirdnet_net::{Keypair, WEIRDNET};
use weirdnet_sync::Roster;
use ymir_core::control::{self, ControlRequest};
use ymir_core::membership::{Membership, ServiceRole};
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

/// What `pool link --cert-only` and `pool add --service` do, without the network.
fn prelink_service(from: &Membership, to: &Home, label: &str, role: ServiceRole) -> Membership {
    let key = Keypair::generate_ed25519();
    let peer = key.public().to_peer_id().to_string();
    let user = from.user_keypair().unwrap();
    let cert = Roster::self_certify(&WEIRDNET, &user, &peer, label, 1).unwrap();
    let bundle = from.service_bundle(&peer, role).unwrap();
    from.approve(cert.clone()).unwrap();
    Membership::adopt_service(&to.paths.pool(), key, &user.public().encode_protobuf(), cert, &bundle).unwrap()
}

/// Runs `pool link`-like `args` on `to` and `pool add CODE --yes ADD...` on `from` together.
async fn link_pair(to: &Home, args: &[&str], from: &Home, add: &[&str]) -> (Result<()>, Result<String>, String) {
    let shown = Shared::default();
    let mut link_out = shown.clone();
    let link = ctl_with(&to.home, args, &mut link_out, "y\n");
    let added = async {
        let deadline = Instant::now() + Duration::from_secs(20);
        let code = loop {
            if let Some(c) = shown.text().split_whitespace().find(|w| w.starts_with("weirdnet-ymir://link/")) {
                break c.to_string();
            }
            assert!(Instant::now() < deadline, "no code shown: {}", shown.text());
            tokio::time::sleep(Duration::from_millis(50)).await;
        };
        let mut argv = vec!["pool", "add", &code, "--yes"];
        argv.extend_from_slice(add);
        ctl(&from.home, &argv).await
    };
    let (linked, added) = tokio::join!(link, added);
    (linked, added, shown.text())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cert_only_links_refusals_and_the_move_off_the_seed() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let holder = home(tmp.path(), "holder", "app", &[])?;
    ctl(&holder.home, &["pool", "init", "--standalone"]).await?;
    let user_before = Membership::open(&holder.paths.pool())?.unwrap().user_id();

    // A new server links cert-only.
    let srv = home(tmp.path(), "srv", "inference", &[])?;
    let (linked, added, shown) = link_pair(&srv, &["pool", "link", "--cert-only"], &holder, &["--service", "inference"]).await;
    let added = added?;
    linked?;
    assert!(added.contains("cert-only"), "{added}");
    assert!(shown.contains("Linked cert-only as Ymir on srv"), "{shown}");
    let dir = srv.paths.pool();
    assert!(!dir.join("seed").exists(), "no seed on a cert-only node");
    assert!(dir.join("credential.json").is_file());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(dir.join("credential.json"))?.permissions().mode() & 0o777, 0o600);
    }
    let m = Membership::open(&dir)?.unwrap();
    assert!(m.is_service());
    assert_eq!(m.user_id(), user_before);
    let srv_peer = m.peer_id().to_string();
    drop(m);
    let devices = ctl(&holder.home, &["pool", "devices"]).await?;
    assert!(devices.lines().any(|l| l.contains(&srv_peer) && l.contains("cert-only ymir.inference")), "{devices}");
    assert!(ctl(&srv.home, &["pool", "devices"]).await?.contains("(this node is cert-only)"));

    // What only a seed holder may do is refused there.
    let e = format!("{:#}", ctl(&srv.home, &["pool", "add", "weirdnet-ymir://link/x", "--yes"]).await.unwrap_err());
    assert!(e.contains("cert-only"), "{e}");
    let e = format!("{:#}", ctl(&srv.home, &["pool", "revoke", "12D3KooWQH96LGJd33JquZXNyrmrs7HAeaeviiFqdwnzwfdmj1VY"]).await.unwrap_err());
    assert!(e.contains("cert-only"), "{e}");
    assert!(ctl(&srv.home, &["pool", "cert-only"]).await.unwrap_err().to_string().contains("already cert-only"));

    // A cert-only link code refuses the seed: nothing is stored.
    let strict = home(tmp.path(), "strict", "inference", &[])?;
    let (linked, _added, _) = link_pair(&strict, &["pool", "link", "--cert-only"], &holder, &[]).await;
    let e = format!("{:#}", linked.unwrap_err());
    assert!(e.contains("cert-only"), "{e}");
    assert!(!Membership::exists(&strict.paths.pool()));

    // A node linked the old way moves to cert-only; its node key and peer id stay.
    let legacy = home(tmp.path(), "legacy", "model", &[])?;
    let (linked, added, _) = link_pair(&legacy, &["pool", "link"], &holder, &[]).await;
    linked?;
    added?;
    let ldir = legacy.paths.pool();
    assert!(ldir.join("seed").is_file());
    let before = Membership::open(&ldir)?.unwrap().peer_id();
    let (moved, added, shown) = link_pair(&legacy, &["pool", "cert-only"], &holder, &["--service", "model"]).await;
    added?;
    moved?;
    assert!(shown.contains("is cert-only now"), "{shown}");
    assert!(!ldir.join("seed").exists() && !ldir.join("seed.tmp").exists());
    let after = Membership::open(&ldir)?.unwrap();
    assert!(after.is_service());
    assert_eq!(after.peer_id(), before);
    assert_eq!(after.user_id(), user_before);
    let devices = ctl(&holder.home, &["pool", "devices"]).await?;
    assert!(devices.lines().any(|l| l.contains(&before.to_string()) && l.contains("cert-only ymir.model")), "{devices}");

    // Revoking a cert-only node from the seed holder: revoked, and the identity is the same.
    let out = ctl(&holder.home, &["pool", "revoke", &srv_peer]).await?;
    assert!(out.contains("revoked"), "{out}");
    let h = Membership::open(&holder.paths.pool())?.unwrap();
    assert_eq!(h.user_id(), user_before);
    assert!(!h.roster().active().any(|c| c.peer_id == srv_peer));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cert_only_servers_serve_and_sync_until_revoked() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();

    // The app node holds the seed; the model and inference nodes are linked cert-only.
    let app_tmp = home(root, "app", "app", &[])?;
    let holder = Membership::create(&app_tmp.paths.pool(), "app")?;
    let user_before = holder.user_id();
    let model_h = home(root, "model", "model", &[])?;
    let model_m = prelink_service(&holder, &model_h, "model", ServiceRole::Model);
    let inf_tmp = home(root, "inf", "inference", &[])?;
    let inf_m = prelink_service(&holder, &inf_tmp, "inf", ServiceRole::Inference);
    let inf_peer = inf_m.peer_id().to_string();
    assert!(model_m.is_service() && inf_m.is_service());
    drop((holder, model_m, inf_m));

    let model = start(&model_h).await?;
    let model_addr = addr(&model_h).await?;
    let inf_h = home(root, "inf", "inference", std::slice::from_ref(&model_addr))?;
    let inf = start(&inf_h).await?;
    let inf_addr = addr(&inf_h).await?;
    let app_h = home(root, "app", "app", std::slice::from_ref(&inf_addr))?;
    let app = start(&app_h).await?;

    // The cert-only inference node finds the cert-only model node and serves the app.
    until_ok(30, async || {
        let r = control::request(&inf_h.paths.socket(), &ControlRequest::Load { model: "alpha".into(), node: None }).await?;
        Ok((if r.ok { 200 } else { 503 }, json!(r.message)))
    })
    .await?;
    let v = until_ok(30, async || chat(&app, &app_h, "no seed here").await).await?;
    assert_eq!(v["choices"][0]["message"]["content"], "Hello from alpha! You said: no seed here");

    // Sealed conversations reach the cert-only inference node, which can read them.
    ctl(&app_h.home, &["context", "add", "before", "hello"]).await?;
    eventually(40, "the conversation reaches the cert-only inference node", async || {
        Ok(ctl(&inf_h.home, &["context", "show"]).await.unwrap_or_default().lines().any(|l| l.starts_with("before ")))
    })
    .await?;

    // The running cert-only node refuses what needs the seed.
    let r = control::request(&inf_h.paths.socket(), &ControlRequest::PoolRevoke { peer: model_addr.rsplit('/').next().unwrap().into() }).await?;
    assert!(!r.ok && r.message.unwrap_or_default().contains("cert-only"));
    let r = control::request(&inf_h.paths.socket(), &ControlRequest::PoolServiceBundle { peer: inf_peer.clone(), role: "model".into() }).await?;
    assert!(!r.ok && r.message.unwrap_or_default().contains("cert-only"));

    // Revoked from the seed holder: later conversations are sealed under a key it doesn't get.
    let out = ctl(&app_h.home, &["pool", "revoke", &inf_peer]).await?;
    assert!(out.contains("revoked"), "{out}");
    ctl(&app_h.home, &["context", "add", "after", "not for you"]).await?;
    tokio::time::sleep(Duration::from_secs(6)).await;
    let seen = ctl(&inf_h.home, &["context", "show"]).await.unwrap_or_default();
    assert!(!seen.lines().any(|l| l.starts_with("after ")), "{seen}");
    let devices = ctl(&app_h.home, &["pool", "devices"]).await?;
    assert!(devices.starts_with(&format!("Identity {user_before}")), "the root identity is unchanged: {devices}");

    app.shutdown().await;
    inf.shutdown().await;
    model.shutdown().await;
    Ok(())
}

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
