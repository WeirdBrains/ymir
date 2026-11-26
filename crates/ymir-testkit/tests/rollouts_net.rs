//! `/ymir/rollouts/1` (docs/design/model-upload.md, the U4 follow-up): `ymirctl models rollout
//! ... --inference NODE` reads and changes another inference node's rollouts over Weirdnet on
//! 127.0.0.1 (nothing leaves the box), by that node's rules. Real daemons with throwaway homes and
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
fn home(root: &Path, name: &str, roles: &str, peers: &[String], models: &[&str], extra: &str) -> Result<Home> {
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
        "[node]\nname = \"{name}\"\n[roles]\nmodel = {m}\ninference = {i}\napp = {a}\n[network]\nlisten = [\"/ip4/127.0.0.1/tcp/0\"]\nmdns = false\ndiscovery = false\nrelays = []\npeers = [{peers}]\n[model]\nllamacpp = \"system:{MOCK}\"\nvllm = \"off\"\nmodels_dir = \"{}\"\n{extra}{local}[inference.alerts]\nprobe_secs = 3600\n[app.bridge]\nport = 0\n",
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

/// A LoRA adapter GGUF header for an `arch` base, padded to `len` bytes.
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

async fn live(h: &Home) -> Result<Vec<ymir_core::rollout::Rollout>> {
    Ok(control::request(&h.paths.socket(), &ControlRequest::Rollouts { set: None }).await?.rollouts.unwrap_or_default())
}

fn weight(r: &[ymir_core::rollout::Rollout], name: &str, n: u32) -> Option<u32> {
    r.iter().find(|r| r.name == name)?.version(n).map(|v| v.weight)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn another_inference_nodes_rollouts_change_by_its_rules() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();
    let lap_h = home(root, "lap", "inference", &[], &[], "")?;
    let first = Membership::create(&lap_h.paths.pool(), "lap")?;
    let lap_peer = first.peer_id().to_string();
    // inf takes rollout changes from lap only.
    let inf_h = home(
        root,
        "inf",
        "inference,model",
        &[],
        &["base", "cand"],
        &format!("upload_keep_free = \"0\"\n[inference]\nrollout_callers = [\"{lap_peer}\"]\n"),
    )?;
    let other_h = home(root, "other", "inference", &[], &[], "")?;
    prelink(&first, &inf_h, "inf");
    prelink(&first, &other_h, "other");
    drop(first);
    let inf = start(&inf_h).await?;
    let inf_addr = addr(&inf_h).await?;
    let lap_h = home(root, "lap", "inference", std::slice::from_ref(&inf_addr), &[], "")?;
    let other_h = home(root, "other", "inference", std::slice::from_ref(&inf_addr), &[], "")?;
    let lap = start(&lap_h).await?;
    let other = start(&other_h).await?;
    let ro = |args: &[&str]| -> Vec<String> {
        let mut v: Vec<String> = ["models", "rollout"].iter().map(|s| s.to_string()).collect();
        v.extend(args.iter().map(|s| s.to_string()));
        v.extend(["--inference".to_string(), "inf".to_string()]);
        v
    };
    let run = async |h: &Home, args: &[&str]| -> Result<String> {
        let argv = ro(args);
        let refs: Vec<&str> = argv.iter().map(String::as_str).collect();
        ctl(&h.home, &refs).await
    };
    eventually(40, "inf answers /ymir/rollouts/1", async || Ok(run(&lap_h, &["list"]).await.is_ok())).await?;

    // Not this device.
    let err = format!("{:#}", ctl(&lap_h.home, &["models", "rollout", "list", "--inference", "lap"]).await.unwrap_err());
    assert!(err.contains("is this device"), "{err}");

    // A new rollout takes all the traffic (nothing to protect), saved in inf's node.toml and live.
    let out = run(&lap_h, &["add", "support@1", "--model", "base", "--weight", "100"]).await?;
    assert!(out.contains("Added support@1 (`base`) with weight 100") && out.contains("saved in its node.toml"), "{out}");
    assert_eq!(weight(&live(&inf_h).await?, "support", 1), Some(100));
    assert_eq!(weight(&NodeConfig::load(&inf_h.paths.config())?.inference.rollout, "support", 1), Some(100));

    // A second version: weight 0, then the canary share is free, more needs an eval or force.
    run(&lap_h, &["add", "support@2", "--model", "cand"]).await?;
    let err = format!("{:#}", run(&lap_h, &["weight", "support@2", "50"]).await.unwrap_err());
    assert!(err.contains("canary share"), "{err}");
    assert_eq!(weight(&live(&inf_h).await?, "support", 2), Some(0), "a refused change changes nothing");
    run(&lap_h, &["weight", "support@2", "5"]).await?;
    assert_eq!(weight(&live(&inf_h).await?, "support", 2), Some(5));
    // A percentage named outright is set as asked, above the canary share and without an eval.
    let out = run(&lap_h, &["weight", "support@2", "--canary", "20"]).await?;
    assert!(out.contains("support@2 now gets 20.0%") && out.contains("support@1 80.0%"), "{out}");
    assert_eq!((weight(&live(&inf_h).await?, "support", 1), weight(&live(&inf_h).await?, "support", 2)), (Some(80), Some(20)));
    let err = format!("{:#}", run(&lap_h, &["weight", "support@2", "--canary", "120"]).await.unwrap_err());
    assert!(err.contains("120"), "{err}");

    // --no-persist: live only.
    let argv = [ro(&["weight", "support@2", "8"]), vec!["--no-persist".into()]].concat();
    let refs: Vec<&str> = argv.iter().map(String::as_str).collect();
    let out = ctl(&lap_h.home, &refs).await?;
    assert!(out.contains("live only"), "{out}");
    assert_eq!(weight(&live(&inf_h).await?, "support", 2), Some(8));
    assert_eq!(weight(&NodeConfig::load(&inf_h.paths.config())?.inference.rollout, "support", 2), Some(20));

    // The list shows inf's live rollouts.
    let out = run(&lap_h, &["list"]).await?;
    assert!(out.contains("support@2") && out.contains("on inf, live"), "{out}");

    // Rollback and retire work from here too.
    let out = run(&lap_h, &["rollback", "support"]).await?;
    assert!(out.contains("Rolled back `support`"), "{out}");
    assert_eq!(weight(&live(&inf_h).await?, "support", 2), Some(0));
    run(&lap_h, &["retire", "support@2"]).await?;
    assert_eq!(weight(&live(&inf_h).await?, "support", 2), None);
    assert_eq!(weight(&NodeConfig::load(&inf_h.paths.config())?.inference.rollout, "support", 2), None);

    // An upload joins inf's rollout with --inference: the next version there (2, as 2 was
    // retired), with the share asked for, saved in inf's node.toml.
    let file = root.join("tune.gguf");
    std::fs::write(&file, gguf(4000, "llama"))?;
    let f = file.display().to_string();
    let out = ctl(&lap_h.home, &["models", "upload", &f, "--node", "inf", "--rollout", "support", "--inference", "inf", "--canary", "20"]).await?;
    assert!(out.contains("As `tune-r2`, version support@2.") && out.contains("support@2 now gets 20.0%"), "{out}");
    assert_eq!(weight(&live(&inf_h).await?, "support", 2), Some(20));
    assert_eq!(weight(&NodeConfig::load(&inf_h.paths.config())?.inference.rollout, "support", 2), Some(20));
    assert!(lap_h.paths.config().exists() && NodeConfig::load(&lap_h.paths.config())?.inference.rollout.is_empty(), "not on this machine");

    // A linked device not in rollout_callers is refused.
    let err = format!("{:#}", run(&other_h, &["list"]).await.unwrap_err());
    assert!(err.contains("rollout_callers"), "{err}");

    drop((inf, lap, other));
    Ok(())
}
