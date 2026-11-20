//! Repro for weirdnet-agent 0.4.2: `LinkedNodes::nodes()` (which `Pool::infer` and
//! `Pool::embed` call on every request) checks each linked device in turn and, for each one it
//! can't identify, waits the whole `with_wait` (3 s by default) even when `connect` failed at
//! once. So every request pays `wait` times the number of offline or unreachable devices in the
//! roster before its first dial. Ignored by default: it only measures (run with `--ignored`).

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use weirdnet_agent::pool::{LinkedNodes, NetDialer, Pool};
use weirdnet_net::{Keypair, WEIRDNET};
use weirdnet_sync::Roster;
use ymir_core::control::{self, ControlRequest};
use ymir_core::daemon::Network;
use ymir_core::membership::Membership;
use ymir_core::{Daemon, NodeConfig, Paths};

const MOCK: &str = env!("CARGO_BIN_EXE_ymir-mock-engine");

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
        "[node]\nname = \"{name}\"\n[roles]\nmodel = {m}\ninference = {i}\napp = {a}\n[network]\nlisten = [\"/ip4/127.0.0.1/tcp/0\"]\nmdns = false\ndiscovery = false\nrelays = []\nsync_secs = 1\npeers = [{peers}]\n[model]\nllamacpp = \"system:{MOCK}\"\nvllm = \"off\"\nmodels_dir = \"{}\"\n{local}[inference.alerts]\nprobe_secs = 3600\n[app.bridge]\nport = 0\n",
        models_dir.display()
    );
    std::fs::write(home.join("node.toml"), toml)?;
    Ok(Home { paths: Paths::new(&home) })
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "measures weirdnet-agent LinkedNodes; run with --ignored"]
async fn linked_nodes_waits_for_every_unreachable_device() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();
    let inf_tmp = home(root, "inf", "inference", &[], &[])?;
    let first = Membership::create(&inf_tmp.paths.pool(), "inf")?;
    // Three linked devices that are never started (a phone that is off, a laptop asleep).
    let mut seed = first.seed().unwrap().0;
    let user = Keypair::ed25519_from_bytes(&mut seed).unwrap();
    for i in 0..3 {
        let k = Keypair::generate_ed25519();
        first.approve(Roster::self_certify(&WEIRDNET, &user, &k.public().to_peer_id().to_string(), &format!("offline{i}"), 1).unwrap()).unwrap();
    }
    let app_tmp = home(root, "dev", "app", &[], &[])?;
    prelink(&first, &app_tmp, "dev");
    drop(first);
    let inf = start(&inf_tmp).await?;
    let inf_addr = addr(&inf_tmp).await?;
    let app_h = home(root, "dev", "app", std::slice::from_ref(&inf_addr), &[])?;
    let cfg = NodeConfig::load(&app_h.paths.config())?;
    let net = Network::start(&cfg, &app_h.paths).await?.expect("on Weirdnet");
    let (inf_peer, inf_ma) = {
        let (a, p) = inf_addr.split_once("/p2p/").unwrap();
        (p.parse()?, vec![a.parse()?])
    };
    eprintln!("roster: {} active devices (this one, the inference node, 3 offline)", net.membership.roster().active().count());
    for wait in [Duration::from_secs(3), Duration::from_millis(500)] {
        let nodes = LinkedNodes::new(net.node.clone(), net.membership.roster_view()).with_hint(inf_peer, inf_ma.clone()).with_wait(wait);
        let pool = Pool::new(Arc::new(NetDialer::new(net.node.clone())), nodes);
        for round in 0..2 {
            let t = Instant::now();
            let found = pool.nodes().await;
            eprintln!("wait {wait:?}, call {round}: nodes() took {:?} and found {found:?}", t.elapsed());
        }
    }
    inf.shutdown().await;
    Ok(())
}
