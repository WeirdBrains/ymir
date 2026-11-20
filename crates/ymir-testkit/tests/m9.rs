//! M9: hardening checks that run for real. A combined node (model, inference and app roles, one
//! mock engine) is started and inspected: every socket it and its engine open is on loopback, the
//! secrets and the control socket are private, and nothing but the bridge token opens the bridge.
//! Then the update flow through `ymirctl`: pin a root, stage a signed bundle, start it on trial.

use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use clap::Parser as _;
use serde_json::{Value, json};
use weirdnet_agent::pool::NetDialer;
use weirdnet_net::{Keypair, Multiaddr, WEIRDNET};
use weirdnet_sync::Roster;
use weirdnet_update::envelope::Domain;
use weirdnet_update::manifest::{FullPackage, Labels, PackageFormat};
use weirdnet_update::{Delegation, FileEntry, Root, Signed, SigningKey, Timestamp, UpdateKind, UpdateManifest};
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
    Ok(Home { paths: Paths::new(&home) })
}

async fn start(h: &Home) -> Result<Daemon> {
    Daemon::start(NodeConfig::load(&h.paths.config())?, h.paths.clone()).await
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

/// Sockets in LISTEN state owned by this process or its children, as (address, port).
fn listeners() -> Vec<(String, u16)> {
    let me = std::process::id();
    let mut family = std::collections::HashMap::<u32, u32>::new();
    for e in std::fs::read_dir("/proc").unwrap().flatten() {
        let Ok(pid) = e.file_name().to_string_lossy().parse::<u32>() else { continue };
        let Ok(stat) = std::fs::read_to_string(e.path().join("stat")) else { continue };
        // pid (comm) state ppid ...: comm may hold spaces, so split after the last ')'.
        let after = stat.rsplit_once(')').map(|x| x.1).unwrap_or("");
        let ppid: u32 = after.split_whitespace().nth(1).and_then(|p| p.parse().ok()).unwrap_or(0);
        family.insert(pid, ppid);
    }
    let mut mine = std::collections::HashSet::from([me]);
    loop {
        let before = mine.len();
        for (pid, ppid) in &family {
            if mine.contains(ppid) {
                mine.insert(*pid);
            }
        }
        if mine.len() == before {
            break;
        }
    }
    let mut inodes = std::collections::HashSet::new();
    for pid in &mine {
        let Ok(rd) = std::fs::read_dir(format!("/proc/{pid}/fd")) else { continue };
        for fd in rd.flatten() {
            if let Ok(t) = std::fs::read_link(fd.path())
                && let Some(i) = t.to_string_lossy().strip_prefix("socket:[").and_then(|x| x.strip_suffix(']'))
            {
                inodes.insert(i.to_string());
            }
        }
    }
    let mut out = Vec::new();
    for table in ["/proc/net/tcp", "/proc/net/tcp6"] {
        let text = std::fs::read_to_string(table).unwrap_or_default();
        for line in text.lines().skip(1) {
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() > 9 && f[3] == "0A" && inodes.contains(f[9]) {
                let (addr, port) = f[1].split_once(':').unwrap();
                out.push((addr.to_string(), u16::from_str_radix(port, 16).unwrap()));
            }
        }
    }
    out
}

fn is_loopback(hex: &str) -> bool {
    // 127.0.0.1 in /proc/net/tcp is 0100007F; ::1 in tcp6 is 00000000000000000000000001000000;
    // an IPv4-mapped loopback is 0000000000000000FFFF00000100007F.
    matches!(hex, "0100007F" | "00000000000000000000000001000000" | "0000000000000000FFFF00000100007F")
}

#[cfg(unix)]
fn mode(p: &Path) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(p).unwrap().permissions().mode() & 0o777
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_running_node_listens_only_on_loopback_and_keeps_its_secrets_private() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let h = home(tmp.path(), "box", "model,inference,app", &[], &["alpha"])?;
    Membership::create(&h.paths.pool(), "box")?;
    let d = start(&h).await?;
    // Load the model so an engine process is running with its own port, and ask once through the bridge.
    let r = control::request(&h.paths.socket(), &ControlRequest::Load { model: "alpha".into(), node: None }).await?;
    assert!(r.ok, "{:?}", r.message);
    let v = until_ok(30, async || chat_model(&d, &h, "alpha", "hello").await).await?;
    assert!(v["choices"][0]["message"]["content"].as_str().unwrap_or("").contains("hello"));

    let ls = listeners();
    assert!(ls.len() >= 3, "expected the bridge, the Weirdnet listener and an engine: {ls:?}");
    for (addr, port) in &ls {
        assert!(is_loopback(addr), "port {port} listens on {addr}, not loopback: {ls:?}");
    }
    let bridge_port = d.bridge_addr.unwrap().port();
    assert!(ls.iter().any(|(_, p)| *p == bridge_port), "the bridge is listening");

    // Private files: state dir, control socket, bridge token, the identity and its seed.
    assert_eq!(mode(&h.paths.state()), 0o700);
    assert_eq!(mode(&h.paths.socket()), 0o600);
    assert_eq!(mode(&h.paths.token()), 0o600);
    assert_eq!(mode(&h.paths.pool()), 0o700);
    assert_eq!(mode(&h.paths.pool().join("seed")), 0o600);

    // The bridge answers only with the token, and rejects a foreign Host header.
    let http = reqwest::Client::builder().no_proxy().build()?;
    let url = format!("http://127.0.0.1:{bridge_port}/v1/models");
    assert_eq!(http.get(&url).send().await?.status().as_u16(), 401);
    assert_eq!(http.get(&url).bearer_auth("wrong").send().await?.status().as_u16(), 401);
    let token = ymir_core::token::load(&h.paths)?.unwrap();
    assert_eq!(http.get(&url).bearer_auth(&token).send().await?.status().as_u16(), 200);
    let rebind = http.get(&url).bearer_auth(&token).header("Host", "evil.example").send().await?;
    assert!(rebind.status().is_client_error(), "a DNS-rebinding Host is refused, got {}", rebind.status());
    d.shutdown().await;
    Ok(())
}

fn key(n: u8) -> SigningKey {
    SigningKey::from_seed(&[n; 32])
}

#[tokio::test]
async fn ymirctl_updates_stage_a_signed_bundle_and_start_it_on_trial() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let home = tmp.path().join("home");
    let (r1, r2, rel) = (key(1), key(2), key(10));
    let root = Root {
        version: 1,
        org: "weirdbrains".into(),
        keys: vec![r1.public(), r2.public()],
        threshold: 2,
        release: vec![Delegation::any(rel.public())],
        release_threshold: 1,
        timestamp: vec![],
        timestamp_threshold: 0,
        expires: None,
    };
    let root_file = tmp.path().join("root.env");
    std::fs::write(&root_file, Signed::sign(Domain::Root, &root, &[&r1, &r2]).unwrap().to_bytes().unwrap())?;
    let (platform, arch) = ymir_core::update::this_target().expect("a supported platform");
    let now = Timestamp::now();
    let bundle = tmp.path().join("bundle");
    std::fs::create_dir_all(&bundle)?;
    std::fs::write(bundle.join("ymird"), b"#!/bin/sh\nexit 0\n")?;
    let m = UpdateManifest {
        app: ymir_core::update::APP.into(),
        channel: "stable".into(),
        update_id: "m9-test".into(),
        seq: 1,
        kind: UpdateKind::Full(FullPackage { platform, arch, format: PackageFormat::TarGz, signing_identity: "unsigned".into() }),
        version: semver::Version::new(99, 0, 0),
        base_releases: vec![],
        files: vec![FileEntry::describe("ymird", b"#!/bin/sh\nexit 0\n")],
        min_client: None,
        max_client: None,
        capabilities: vec![],
        labels: Labels::default(),
        targeting: None,
        rollout: None,
        rollback_of: None,
        notes: "test".into(),
        created: now.minus(60),
        expires: now.plus(3600),
    };
    std::fs::write(bundle.join("manifest.env"), Signed::sign(Domain::Manifest, &m, &[&rel]).unwrap().to_bytes().unwrap())?;

    // Not pinned yet: nothing is taken.
    let e = ctl(&home, &["update", "stage", bundle.to_str().unwrap()]).await.unwrap_err().to_string();
    assert!(e.contains("no update root"), "{e}");
    let pin = root.pin().to_string();
    ctl(&home, &["update", "trust", "--pin", &pin, "--root", root_file.to_str().unwrap()]).await?;
    // A wrong pin is refused before anything is stored.
    assert!(ctl(&tmp.path().join("other"), &["update", "trust", "--pin", &"00".repeat(32), "--root", root_file.to_str().unwrap()]).await.is_err());

    let out = ctl(&home, &["update", "stage", bundle.to_str().unwrap()]).await?;
    assert!(out.contains("staged m9-test 99.0.0"), "{out}");
    // The next start picks the release on trial; reaching ready (what ymird does once serving) commits it.
    let path = ctl(&home, &["update", "boot"]).await?;
    assert!(path.trim().ends_with("releases/m9-test/ymird"), "{path}");
    assert_eq!(std::fs::read(path.trim())?, b"#!/bin/sh\nexit 0\n");
    ymir_core::update::commit_if_on_trial(&Paths::new(&home));
    let st = ctl(&home, &["update", "status"]).await?;
    assert!(st.contains("good release: m9-test") && st.contains("pinned"), "{st}");
    Ok(())
}

/// What `pool link` does, without the network: `from` certifies a new node for `to`.
fn prelink(from: &Membership, to: &Home, label: &str) {
    let key = Keypair::generate_ed25519();
    let mut seed = from.seed().unwrap().0;
    let user = Keypair::ed25519_from_bytes(&mut seed).unwrap();
    let cert = Roster::self_certify(&WEIRDNET, &user, &key.public().to_peer_id().to_string(), label, 1).unwrap();
    from.approve(cert.clone()).unwrap();
    Membership::adopt(&to.paths.pool(), key, &from.seed().unwrap().0, cert, &from.link_settings()).unwrap();
}

/// Whether a device linked as `label` can open the model node at `model`.
async fn reaches_model_node(h: &Home, model: &str, model_peer: &str) -> Result<(), String> {
    let m = Membership::open(&h.paths.pool()).unwrap().unwrap();
    let node = std::sync::Arc::new(
        weirdnet_sync::start_node(m.node_key().clone(), NodeConfig::load(&h.paths.config()).unwrap().network.node_config("test").unwrap(), m.state())
            .unwrap(),
    );
    let dialer = NetDialer::new(node.clone());
    let addr: Multiaddr = model.split("/p2p/").next().unwrap().parse().unwrap();
    dialer.hint(model_peer.parse().unwrap(), vec![addr]);
    let opts = ymir_core::remote::RemoteOptions { find_wait: Duration::from_secs(2), accept_timeout: Duration::from_secs(5) };
    let r = ymir_core::remote::RemoteModelNode::connect(std::sync::Arc::new(dialer), model_peer, &m, &opts).await;
    r.map(|_| ()).map_err(|e| e.to_string())
}

/// One model node, one device linked as "inf" and one as "app", every device the same user's.
async fn role_scenario(callers: bool) -> Result<(Result<(), String>, Result<(), String>)> {
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();
    let model_h = home(root, "model", "model", &[], &["alpha"])?;
    let first = Membership::create(&model_h.paths.pool(), "model")?;
    let (inf_h, app_h) = (home(root, "inf", "inference", &[], &[])?, home(root, "app", "app", &[], &[])?);
    prelink(&first, &inf_h, "inf");
    prelink(&first, &app_h, "app");
    drop(first);
    if callers {
        // By peer id: the label is not signed, the peer id is.
        let inf_peer = Membership::open(&inf_h.paths.pool())?.unwrap().peer_id().to_string();
        let toml = std::fs::read_to_string(model_h.paths.config())?;
        std::fs::write(model_h.paths.config(), toml.replace("[model]\n", &format!("[model]\ncallers = [\"{inf_peer}\"]\n")))?;
    }
    let model = start(&model_h).await?;
    let s = control::request(&model_h.paths.socket(), &ControlRequest::Status).await?.status.unwrap().network.expect("on Weirdnet");
    let a = s.listening.iter().find(|a| a.starts_with("/ip4/127.0.0.1/tcp/")).expect("a loopback listener").clone();
    let inf = reaches_model_node(&inf_h, &a, &s.peer).await;
    let app = reaches_model_node(&app_h, &a, &s.peer).await;
    model.shutdown().await;
    Ok((inf, app))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_model_node_with_callers_refuses_an_app_device() -> Result<()> {
    // Without `callers`, any linked device gets in, an app included (the default, and the gap).
    let (inf, app) = role_scenario(false).await?;
    assert!(inf.is_ok() && app.is_ok(), "default: inf {inf:?}, app {app:?}");
    // With `callers = [<the inference node's peer id>]` only the inference node does.
    let (inf, app) = role_scenario(true).await?;
    assert!(inf.is_ok(), "the inference node is admitted: {inf:?}");
    let e = app.expect_err("an app device must be refused by the model node");
    assert!(e.contains("not-an-allowed-caller"), "{e}");
    Ok(())
}

#[test]
fn callers_must_be_peer_ids_not_labels() {
    let mut cfg = NodeConfig::default();
    cfg.model.callers = vec!["inf".into()];
    let e = cfg.validate().expect_err("a label is not accepted").to_string();
    assert!(e.contains("not a peer id") && e.contains("not covered by the certificate signature"), "{e}");
    cfg.model.callers = vec![Keypair::generate_ed25519().public().to_peer_id().to_string()];
    cfg.validate().expect("a peer id is");
}

/// `weirdnet-backends` 0.1.1's Verifier ignores the roster, so Ymir checks it: a model node whose cert
/// is revoked is refused on connect and, once connected, on every later request and record read.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_revoked_model_node_is_refused_on_connect_and_on_each_request() -> Result<()> {
    use weirdnet_agent::remote::infer::v2::Request;
    use ymir_core::model_node::ModelNode;
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();
    let model_h = home(root, "model", "model", &[], &["alpha"])?;
    let first = Membership::create(&model_h.paths.pool(), "model")?;
    let inf_h = home(root, "inf", "inference", &[], &[])?;
    prelink(&first, &inf_h, "inf");
    drop(first);
    let model = start(&model_h).await?;
    let s = control::request(&model_h.paths.socket(), &ControlRequest::Status).await?.status.unwrap().network.expect("on Weirdnet");
    let a: Multiaddr =
        s.listening.iter().find(|a| a.starts_with("/ip4/127.0.0.1/tcp/")).expect("a loopback listener").split("/p2p/").next().unwrap().parse()?;

    let m = Membership::open(&inf_h.paths.pool())?.unwrap();
    let node = std::sync::Arc::new(weirdnet_sync::start_node(
        m.node_key().clone(),
        NodeConfig::load(&inf_h.paths.config())?.network.node_config("test")?,
        m.state(),
    )?);
    let dialer = NetDialer::new(node.clone());
    dialer.hint(s.peer.parse()?, vec![a]);
    let dialer = std::sync::Arc::new(dialer);
    let opts = ymir_core::remote::RemoteOptions { find_wait: Duration::from_secs(2), accept_timeout: Duration::from_secs(5) };

    // Connected and answering while the node is on the roster.
    let rn = ymir_core::remote::RemoteModelNode::connect(dialer.clone(), &s.peer, &m, &opts).await.map_err(|e| anyhow::anyhow!("{e}"))?;
    assert!(rn.alive());
    assert!(rn.load("alpha", false).await.is_ok() || rn.record().is_some());

    // Revoked: the old cert would still verify, but the node is dropped everywhere.
    m.revoke(&s.peer)?;
    assert!(rn.record().is_none() && !rn.alive(), "the record is withdrawn");
    for e in [
        rn.load("alpha", false).await.err(),
        rn.unload("alpha").await.err(),
        rn.drain(true).await.err(),
        rn.chat("alpha", &Request::messages(vec![weirdnet_agent::Message::user("hi")])).await.err(),
        rn.embed("alpha", &["x".to_string()]).await.err(),
    ] {
        let e = e.expect("a request to a revoked model node is refused").to_string();
        assert!(e.contains("model-node-not-in-roster"), "{e}");
    }
    let again = ymir_core::remote::RemoteModelNode::connect(dialer, &s.peer, &m, &opts).await.err().expect("connect refused").to_string();
    assert!(again.contains("model-node-not-in-roster"), "{again}");
    model.shutdown().await;
    Ok(())
}
