//! M7: split models over Ray. Separate daemons over Weirdnet on 127.0.0.1: an inference node, a
//! model node that leads a `vllm-ray` group and one that is only a Ray member. `ray` and `vllm`
//! are mocks (`ymir-mock-ray`, and the mock engine as `vllm serve`), so this checks the
//! orchestration: the head on the private link, the member joining it, `vllm serve` sized from
//! the plan on that cluster, the bind check, and that stopping the group (by `group down`, a
//! lost member, or a refused bind) stops every process of it.

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

const MOCK_RAY: &str = env!("CARGO_BIN_EXE_ymir-mock-ray");

fn free_port() -> u16 {
    ymir_testkit::config_port()
}

fn append(h: &Home, text: &str) -> Result<()> {
    std::fs::OpenOptions::new().append(true).open(h.home.join("node.toml"))?.write_all(text.as_bytes())?;
    Ok(())
}

/// Makes a model node a `vllm-ray` member: its private link, the mock `ray` (and `ray_args`),
/// the mock engine as `vllm`, and for the leader a `[[model.split]]` with `engine = "vllm"`.
fn member(h: &Home, leader_port: Option<u16>, ray_args: &[&str]) -> Result<()> {
    let p = h.home.join("node.toml");
    let t = std::fs::read_to_string(&p)?;
    let args = ray_args.iter().map(|a| format!("\"{a}\"")).collect::<Vec<_>>().join(", ");
    std::fs::write(
        &p,
        t.replacen(
            "vllm = \"off\"\n",
            &format!("vllm = \"system:{MOCK}\"\nprivate_link = \"127.0.0.1\"\nray = \"system:{MOCK_RAY}\"\nray_args = [{args}]\n"),
            1,
        ),
    )?;
    if let Some(port) = leader_port {
        append(h, &format!("[[model.split]]\nid = \"big\"\nengine = \"vllm\"\nfile = \"org/Big-Instruct\"\nserve_port = {port}\n"))?;
    }
    Ok(())
}

async fn peer(h: &Home) -> Result<String> {
    Ok(control::request(&h.paths.socket(), &ControlRequest::Status).await?.status.unwrap().network.unwrap().peer)
}

async fn listening(port: u16) -> bool {
    tokio::net::TcpStream::connect(("127.0.0.1", port)).await.is_ok()
}

fn log(h: &Home, role: &str) -> String {
    std::fs::read_to_string(h.paths.logs().join(format!("split-big-{role}.log"))).unwrap_or_default()
}

/// The pid of the mock head's child (Ray's `gcs_server` and raylet stand-in).
fn head_child(h: &Home) -> Option<u32> {
    log(h, "ray-head").lines().find_map(|l| l.strip_prefix("mock ray: child ")?.trim().parse().ok())
}

fn running(pid: u32) -> bool {
    // A zombie is gone for our purposes (its parent was killed with it).
    std::fs::read_to_string(format!("/proc/{pid}/stat")).is_ok_and(|s| !s.rsplit_once(')').is_some_and(|(_, r)| r.trim_start().starts_with('Z')))
}

struct Pool {
    _tmp: tempfile::TempDir,
    inf_h: Home,
    lead_h: Home,
    work_h: Home,
    inf: Daemon,
    lead: Daemon,
    work: Daemon,
    ray_port: u16,
}

async fn pool(lead_ray_args: &[&str]) -> Result<Pool> {
    let tmp = tempfile::tempdir()?;
    let root = tmp.path().to_path_buf();
    let inf_tmp = home(&root, "inf", "inference app", &[], &[])?;
    let first = Membership::create(&inf_tmp.paths.pool(), "inf")?;
    let (lead_h, work_h) = (home(&root, "lead", "model", &[], &[])?, home(&root, "work", "model", &[], &[])?);
    for (h, label) in [(&lead_h, "lead"), (&work_h, "work")] {
        prelink(&first, h, label);
    }
    drop(first);
    let (serve, ray_port) = (free_port(), free_port());
    member(&lead_h, Some(serve), lead_ray_args)?;
    member(&work_h, None, &[])?;
    let (lead, work) = (start(&lead_h).await?, start(&work_h).await?);
    let (lead_peer, work_peer) = (peer(&lead_h).await?, peer(&work_h).await?);
    let models = vec![addr(&lead_h).await?, addr(&work_h).await?];
    let inf_h = home(&root, "inf", "inference app", &models, &[])?;
    append(
        &inf_h,
        &format!(
            "[[inference.split]]\nid = \"big\"\nbackend = \"vllm-ray\"\nsize = \"8GiB\"\nleader = \"{lead_peer}\"\nray_port = {ray_port}\n\
             [[inference.split.node]]\nnode = \"{lead_peer}\"\nlink = \"127.0.0.1\"\nmemory = \"6GiB\"\n\
             [[inference.split.node]]\nnode = \"{work_peer}\"\nlink = \"127.0.0.1\"\nmemory = \"6GiB\"\n\
             [[inference.split]]\nid = \"wide\"\nbackend = \"vllm-ray\"\nsize = \"8GiB\"\nleader = \"{lead_peer}\"\ntensor_parallel = 4\n\
             [[inference.split.node]]\nnode = \"{lead_peer}\"\nlink = \"127.0.0.1\"\nmemory = \"6GiB\"\ngpus = 2\n\
             [[inference.split.node]]\nnode = \"{work_peer}\"\nlink = \"127.0.0.1\"\nmemory = \"6GiB\"\ngpus = 2\n"
        ),
    )?;
    let inf = start(&inf_h).await?;
    eventually(60, "both model nodes join", async || Ok(ctl(&inf_h.home, &["nodes"]).await?.matches("serving").count() == 2)).await?;
    Ok(Pool { _tmp: tmp, inf_h, lead_h, work_h, inf, lead, work, ray_port })
}

#[cfg(target_os = "linux")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_vllm_ray_group_comes_up_on_the_private_link_serves_and_leaves_nothing_behind() -> Result<()> {
    let p = pool(&[]).await?;
    let ctl_inf = |args: &'static [&'static str]| ctl(&p.inf_h.home, args);

    // Sizing is checked before anything starts: 4 x 2 needs 8 GPUs, the nodes have 4.
    let e = ctl_inf(&["group", "up", "wide"]).await.unwrap_err().to_string();
    assert!(e.contains("vLLM sizing"), "{e}");
    assert!(!listening(p.ray_port).await);

    // Up: the head listens on the link, the worker joined, vLLM got the cluster and the sizing.
    let up = ctl_inf(&["group", "up", "big"]).await?;
    assert!(up.contains("big is up"), "{up}");
    assert!(listening(p.ray_port).await, "the Ray head listens on the private link");
    let vllm = log(&p.lead_h, "leader");
    assert!(vllm.contains("nodes=2 tp=1 pp=2"), "vllm serve ran on a 2-node Ray cluster with pp=2: {vllm}");
    assert!(log(&p.work_h, "ray-worker").contains("joined"), "the member joined the head");
    let child = head_child(&p.lead_h).expect("the head's child pid");
    assert!(running(child));
    let a = ctl(&p.inf_h.home, &["chat", "--model", "big", "--wait", "hello"]).await?;
    assert!(a.contains("Hello from big"), "the group answers: {a}");

    // Down: vLLM, the worker, the head and the head's child all stop.
    ctl_inf(&["group", "down", "big"]).await?;
    eventually(15, "the head stops", async || Ok(!listening(p.ray_port).await)).await?;
    eventually(15, "the head's child stops with its process group", async || Ok(!running(child))).await?;
    assert!(ctl_inf(&["group", "status"]).await?.contains("no group is up"));

    // Up again, then a member leaves the pool: the whole group goes, the head's child too.
    ctl_inf(&["group", "up", "big"]).await?;
    let child = head_child(&p.lead_h).expect("the new head's child");
    assert!(running(child));
    p.work.shutdown().await;
    eventually(90, "the lost member takes the group down", async || Ok(ctl(&p.inf_h.home, &["group", "status"]).await?.contains("lost  big")))
        .await?;
    eventually(15, "the head stops", async || Ok(!listening(p.ray_port).await)).await?;
    eventually(15, "and its child", async || Ok(!running(child))).await?;

    p.inf.shutdown().await;
    p.lead.shutdown().await;
    Ok(())
}

#[cfg(target_os = "linux")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ray_listening_on_all_interfaces_is_refused_and_torn_down() -> Result<()> {
    // The mock head's child listens on 0.0.0.0 like Ray's real GCS.
    let p = pool(&["--mock-bind-any"]).await?;
    let e = ctl(&p.inf_h.home, &["group", "up", "big"]).await.unwrap_err().to_string();
    assert!(e.contains("0.0.0.0") && e.contains("ray_trust_firewall"), "{e}");
    let child = head_child(&p.lead_h).expect("the head's child pid");
    eventually(15, "the refused head stops", async || Ok(!listening(p.ray_port).await)).await?;
    eventually(15, "with its all-interface child", async || Ok(!running(child))).await?;
    assert!(!listening(p.ray_port + 1).await, "nothing listens on all interfaces any more");
    assert!(ctl(&p.inf_h.home, &["group", "status"]).await?.contains("no group is up"));
    // The worker's `ray start` also stopped: it left the cluster when the head went, and the
    // coordinator took its part down.
    eventually(15, "the worker leaves", async || {
        Ok(!log(&p.work_h, "ray-worker").contains("joined") || log(&p.work_h, "ray-worker").contains("went away"))
    })
    .await?;
    p.inf.shutdown().await;
    p.lead.shutdown().await;
    p.work.shutdown().await;
    Ok(())
}
