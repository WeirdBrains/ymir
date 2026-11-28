//! `ymirctl models upload FILE --node all | A,B [--stage NODE]` (docs/design/model-upload.md, slice
//! U6): the files are sent once, to one model node, and the others copy them from it node to node,
//! over Weirdnet on 127.0.0.1 (nothing leaves the box). Real daemons with throwaway homes and
//! identities and a mock engine.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use clap::Parser as _;
use weirdnet_net::{Keypair, WEIRDNET};
use weirdnet_sync::Roster;

use ymir_core::control::{self, ControlRequest};
use ymir_core::membership::Membership;
use ymir_core::upload;
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

/// node.toml for `roles`, on Weirdnet over 127.0.0.1:`port` only (0: any), dialing `peers`.
fn home(root: &Path, name: &str, roles: &str, port: u16, peers: &[String], models: &[&str], extra: &str) -> Result<Home> {
    let home = root.join(name);
    let models_dir = root.join(format!("{name}-models"));
    std::fs::create_dir_all(&home)?;
    std::fs::create_dir_all(&models_dir)?;
    let mut local = String::new();
    for (i, id) in models.iter().enumerate() {
        std::fs::write(models_dir.join(format!("{id}.gguf")), gguf(4000, "llama"))?;
        let role = if i == 0 { "role = \"main\"\n" } else { "" };
        local.push_str(&format!("[[model.local]]\nid = \"{id}\"\nfile = \"{id}.gguf\"\nrevision = \"r1\"\n{role}"));
    }
    let peers = peers.iter().map(|p| format!("\"{p}\"")).collect::<Vec<_>>().join(", ");
    let (m, i, a) = (roles.contains("model"), roles.contains("inference"), roles.contains("app"));
    let toml = format!(
        "[node]\nname = \"{name}\"\n[roles]\nmodel = {m}\ninference = {i}\napp = {a}\n[network]\nlisten = [\"/ip4/127.0.0.1/tcp/{port}\"]\nmdns = false\ndiscovery = false\nrelays = []\npeers = [{peers}]\n[model]\nllamacpp = \"system:{MOCK}\"\nvllm = \"off\"\nmodels_dir = \"{}\"\n{extra}{local}[inference.alerts]\nprobe_secs = 3600\n[app.bridge]\nport = 0\n",
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

/// A GGUF header (llama, one block, one KV head of size 1) padded to `len` bytes.
fn gguf(len: usize, arch: &str) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend(0u64.to_le_bytes());
    let kv: [(&str, Option<u32>); 5] = [
        ("general.architecture", None),
        ("block_count", Some(1)),
        ("attention.head_count", Some(1)),
        ("attention.head_count_kv", Some(1)),
        ("attention.key_length", Some(1)),
    ];
    out.extend((kv.len() as u64).to_le_bytes());
    for (k, v) in kv {
        let k = if v.is_some() { format!("{arch}.{k}") } else { k.to_string() };
        out.extend((k.len() as u64).to_le_bytes());
        out.extend(k.as_bytes());
        match v {
            None => {
                out.extend(8u32.to_le_bytes());
                out.extend((arch.len() as u64).to_le_bytes());
                out.extend(arch.as_bytes());
            }
            Some(n) => {
                out.extend(4u32.to_le_bytes());
                out.extend(n.to_le_bytes());
            }
        }
    }
    out.resize(len, 0);
    out
}

/// `/ip4/127.0.0.1/tcp/PORT/p2p/PEER` for a running daemon, from its control status.
async fn addr(h: &Home) -> Result<String> {
    let s = control::request(&h.paths.socket(), &ControlRequest::Status).await?.status.unwrap();
    let n = s.network.expect("on Weirdnet");
    let a = n.listening.iter().find(|a| a.starts_with("/ip4/127.0.0.1/tcp/")).expect("a loopback listener");
    let a = a.split("/p2p/").next().unwrap();
    Ok(format!("{a}/p2p/{}", n.peer))
}

/// A loopback TCP port free right now.
fn free_port() -> Result<u16> {
    Ok(std::net::TcpListener::bind("127.0.0.1:0")?.local_addr()?.port())
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

/// A laptop and four model nodes: `a` and `b` keep nothing free, `full` keeps more than it has, `c`
/// can't reach the others. `--node all` sends once (to the node with the most room) and the other
/// copies from it, both saved in node.toml; `full` is left out before anything is sent; `c` gets the
/// files from the laptop instead. `--stage` picks where it is sent;
/// the same files again are "already there"; `auto` and a stage outside the targets are refused.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_pool_upload_is_sent_once_and_copied_node_to_node() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();
    let lap_h = home(root, "lap", "inference", 0, &[], &[], "")?;
    let first = Membership::create(&lap_h.paths.pool(), "lap")?;
    // Fixed ports, so a and b can dial each other from the start (either may be the stage).
    let (pa, pb, pf) = (free_port()?, free_port()?, free_port()?);
    let keep0 = "upload_keep_free = \"0\"\n";
    let a_h = home(root, "a", "model", pa, &[], &[], keep0)?;
    let b_h = home(root, "b", "model", pb, &[], &[], keep0)?;
    let full_h = home(root, "full", "model", pf, &[], &[], "upload_keep_free = \"1000000TiB\"\n")?;
    // c can't reach a or b (no addresses, no discovery), as behind a NAT; it keeps 1 GiB free, so it
    // has the least room and is never the stage.
    let c_h = home(root, "c", "model", 0, &[], &[], "upload_keep_free = \"1GiB\"\n")?;
    let at = |port: u16, m: &Membership| format!("/ip4/127.0.0.1/tcp/{port}/p2p/{}", m.peer_id());
    let (ma, mb, mf) = (prelink(&first, &a_h, "a"), prelink(&first, &b_h, "b"), prelink(&first, &full_h, "full"));
    prelink(&first, &c_h, "c");
    let (aa, ab, af) = (at(pa, &ma), at(pb, &mb), at(pf, &mf));
    drop((first, ma, mb, mf));
    let a_h = home(root, "a", "model", pa, std::slice::from_ref(&ab), &[], keep0)?;
    let b_h = home(root, "b", "model", pb, std::slice::from_ref(&aa), &[], keep0)?;
    let a = start(&a_h).await?;
    let b = start(&b_h).await?;
    let full = start(&full_h).await?;
    let c = start(&c_h).await?;
    let ac = addr(&c_h).await?;
    let lap_h = home(root, "lap", "inference", 0, &[aa, ab, af, ac], &[], "")?;
    let lap = start(&lap_h).await?;
    for n in ["a", "b", "full", "c"] {
        eventually(40, &format!("{n} answers models/1"), async || Ok(ctl(&lap_h.home, &["models", "list", "--node", n]).await.is_ok())).await?;
    }

    let files = root.join("files");
    std::fs::create_dir_all(&files)?;
    let model = files.join("Pool-Q4_0.gguf");
    std::fs::write(&model, gguf(2 << 20, "llama"))?;
    let model_s = model.display().to_string();

    // `--node auto` is one node (U9): --stage goes with several.
    let err = format!("{:#}", ctl(&lap_h.home, &["models", "upload", &model_s, "--node", "auto", "--stage", "a"]).await.unwrap_err());
    assert!(err.contains("--stage goes with several targets"), "{err}");
    let err = format!("{:#}", ctl(&lap_h.home, &["models", "upload", &model_s, "--node", "a,b", "--stage", "full"]).await.unwrap_err());
    assert!(err.contains("--stage full is not one of the targets"), "{err}");

    // --node all: sent once, copied once; full is left out before anything is sent.
    let mut out = Vec::new();
    let err = ctl_with(&lap_h.home, &["models", "upload", &model_s, "--node", "all"], &mut out, "").await;
    let out = String::from_utf8(out)?;
    assert!(err.is_err(), "full could not take it: {out}");
    assert!(out.contains("full is left out (no-space)") && out.contains("upload_keep_free"), "{out}");
    let stored = |n: &str| root.join(format!("{n}-models/pool-q4_0/Pool-Q4_0.gguf"));
    let (sent, copied) = if out.contains("a: stored `pool-q4_0`") { ("a", "b") } else { ("b", "a") };
    assert!(out.contains(&format!("{sent}: stored `pool-q4_0`")), "{out}");
    assert!(out.contains(&format!("{copied}: copied `pool-q4_0` from {sent}, node to node")), "{out}");
    assert!(!out.contains(&format!("{copied} could not copy")), "{out}");
    // c could not reach the stage, so it got the files from this device instead.
    assert!(out.contains(&format!("c could not copy it from {sent}")) && out.contains("so it was sent from this device"), "{out}");
    assert!(out.contains("c: stored `pool-q4_0`"), "{out}");
    assert_eq!(std::fs::read(stored("c"))?, std::fs::read(&model)?);
    for n in ["a", "b"] {
        assert_eq!(std::fs::read(stored(n))?, std::fs::read(&model)?, "{n} has the same bytes");
        let h = if n == "a" { &a_h } else { &b_h };
        let cfg = NodeConfig::load(&h.paths.config())?;
        assert!(cfg.model.local.iter().any(|m| m.id == "pool-q4_0"), "saved in {n}'s node.toml");
        let prov = upload::read_provenance(stored(n).parent().unwrap()).expect("provenance.json");
        assert_eq!(prov.origin.as_str(), upload::ORIGIN_UPLOAD, "{n}");
    }
    assert!(!root.join("full-models/pool-q4_0").exists());

    // The same files to a and b again: already there on both.
    let out = ctl(&lap_h.home, &["models", "upload", &model_s, "--node", "a,b"]).await?;
    assert!(out.contains("a: already has `pool-q4_0`") && out.contains("b: already has `pool-q4_0`"), "{out}");

    // --stage b: sent to b, copied by a.
    let other = files.join("Second-Q4_0.gguf");
    std::fs::write(&other, gguf(1 << 20, "llama"))?;
    let out = ctl(&lap_h.home, &["models", "upload", &other.display().to_string(), "--node", "a,b", "--stage", "b", "--no-persist"]).await?;
    assert!(out.contains("b: stored `second-q4_0`") && out.contains("a: copied `second-q4_0` from b, node to node"), "{out}");
    assert_eq!(std::fs::read(root.join("a-models/second-q4_0/Second-Q4_0.gguf"))?, std::fs::read(&other)?);
    assert!(!NodeConfig::load(&a_h.paths.config())?.model.local.iter().any(|m| m.id == "second-q4_0"), "--no-persist on the copy too");

    lap.shutdown().await;
    c.shutdown().await;
    full.shutdown().await;
    b.shutdown().await;
    a.shutdown().await;
    Ok(())
}
