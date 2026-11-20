//! M7: split models. Separate daemons over Weirdnet on 127.0.0.1: an inference node, a model node
//! that leads the group and one that is only a worker. The "engines" are mocks (a stand-in
//! `rpc-server` and the mock chat engine), so this checks the orchestration: planning from the
//! pool policy, refusing what the policy forbids, starting workers before the leader, serving
//! the model, tearing the group down on `group down` and when a member leaves the pool. The real
//! llama.cpp split is `real_split.rs`.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use clap::Parser as _;
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


const MOCK_RPC: &str = env!("CARGO_BIN_EXE_ymir-mock-rpc");

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

fn append(h: &Home, text: &str) -> Result<()> {
    use std::io::Write as _;
    std::fs::OpenOptions::new().append(true).open(h.home.join("node.toml"))?.write_all(text.as_bytes())?;
    Ok(())
}

/// Makes a model node a group member: its private link, the mock `rpc-server`, and (for the
/// leader) the `[[model.split]]` entry.
fn member(h: &Home, leader_port: Option<u16>) -> Result<()> {
    let p = h.home.join("node.toml");
    let t = std::fs::read_to_string(&p)?;
    std::fs::write(&p, t.replacen("vllm = \"off\"\n", &format!("vllm = \"off\"\nprivate_link = \"127.0.0.1\"\nrpc_server = \"system:{MOCK_RPC}\"\n"), 1))?;
    if let Some(port) = leader_port {
        std::fs::write(h.paths.config().parent().unwrap().join("..").join("models").join("big.gguf"), b"GGUF mock")?;
        append(h, &format!("[[model.split]]\nid = \"big\"\nfile = \"big.gguf\"\nserve_port = {port}\n"))?;
    }
    Ok(())
}

async fn peer(h: &Home) -> Result<String> {
    Ok(control::request(&h.paths.socket(), &ControlRequest::Status).await?.status.unwrap().network.unwrap().peer)
}

async fn listening(port: u16) -> bool {
    tokio::net::TcpStream::connect(("127.0.0.1", port)).await.is_ok()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_split_model_is_planned_served_and_torn_down() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();
    let inf_tmp = home(root, "inf", "inference app", &[], &[])?;
    let first = Membership::create(&inf_tmp.paths.pool(), "inf")?;
    let (lead_tmp, work_tmp) = (home(root, "lead", "model", &[], &[])?, home(root, "work", "model", &[], &[])?);
    for (h, label) in [(&lead_tmp, "lead"), (&work_tmp, "work")] {
        prelink(&first, h, label);
    }
    drop(first);
    let (serve, rpc_port) = (free_port(), free_port());
    member(&lead_tmp, Some(serve))?;
    member(&work_tmp, None)?;
    let (lead, work) = (start(&lead_tmp).await?, start(&work_tmp).await?);
    let (lead_peer, work_peer) = (peer(&lead_tmp).await?, peer(&work_tmp).await?);
    let models = vec![addr(&lead_tmp).await?, addr(&work_tmp).await?];
    let inf_h = home(root, "inf", "inference app", &models, &[])?;
    append(
        &inf_h,
        &format!(
            "[[inference.split]]\nid = \"big\"\nsize = \"8GiB\"\nleader = \"{lead_peer}\"\n\
             [[inference.split.node]]\nnode = \"{lead_peer}\"\nlink = \"127.0.0.1\"\nport = {}\nmemory = \"6GiB\"\n\
             [[inference.split.node]]\nnode = \"{work_peer}\"\nlink = \"127.0.0.1\"\nport = {rpc_port}\nmemory = \"6GiB\"\n\
             [[inference.split]]\nid = \"slow\"\nsize = \"8GiB\"\nleader = \"{lead_peer}\"\n\
             [[inference.split.node]]\nnode = \"{lead_peer}\"\nlink = \"10.9.9.1\"\nmemory = \"6GiB\"\n\
             [[inference.split.node]]\nnode = \"{work_peer}\"\nlink = \"10.9.9.2\"\nmemory = \"6GiB\"\ngbps = 1.0\n",
            free_port()
        ),
    )?;
    let inf = start(&inf_h).await?;
    eventually(60, "both model nodes join", async || Ok(ctl(&inf_h.home, &["nodes"]).await?.matches("serving").count() == 2)).await?;

    // Policy refusals come before anything starts.
    let e = ctl(&inf_h.home, &["group", "up", "slow"]).await.unwrap_err().to_string();
    assert!(e.contains("Gb/s") || e.contains("link"), "a slow link is refused: {e}");
    let e = ctl(&inf_h.home, &["group", "up", "nothing"]).await.unwrap_err().to_string();
    assert!(e.contains("not marked `split`"), "{e}");
    assert!(!listening(rpc_port).await);

    // Up: the worker listens on its private link, the leader serves, the pool answers.
    let up = ctl(&inf_h.home, &["group", "up", "big"]).await?;
    assert!(up.contains("big is up"), "{up}");
    assert!(listening(rpc_port).await, "the worker's rpc-server listens");
    assert!(ctl(&inf_h.home, &["group", "status"]).await?.contains("big  leader"));
    let a = ctl(&inf_h.home, &["chat", "--model", "big", "--wait", "hello"]).await?;
    assert!(a.contains("Hello from big"), "the group answers: {a}");

    // Down: workers stop, the model stops being served, status is empty.
    ctl(&inf_h.home, &["group", "down", "big"]).await?;
    eventually(10, "the worker stops", async || Ok(!listening(rpc_port).await)).await?;
    assert!(ctl(&inf_h.home, &["group", "status"]).await?.contains("no group is up"));
    assert!(ctl(&inf_h.home, &["group", "down", "big"]).await.is_err(), "nothing to bring down twice");

    // Up again, then a member leaves the pool: the group is torn down and the loss is kept.
    ctl(&inf_h.home, &["group", "up", "big"]).await?;
    assert!(listening(rpc_port).await);
    work.shutdown().await;
    eventually(90, "the lost member takes the group down", async || Ok(ctl(&inf_h.home, &["group", "status"]).await?.contains("lost  big")))
        .await?;
    assert!(ctl(&inf_h.home, &["group", "status"]).await?.contains("no group is up"));

    inf.shutdown().await;
    lead.shutdown().await;
    Ok(())
}
