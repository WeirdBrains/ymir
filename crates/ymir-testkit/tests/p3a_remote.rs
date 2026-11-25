//! `ymirctl train run --node` (P3a, option C): a run on another of the user's model nodes, started
//! from an app-only device over Weirdnet on 127.0.0.1 (nothing leaves the box). Real daemons with
//! throwaway homes and identities, the mock engine and the mock trainer, and the synthetic dataset
//! fixture. The node takes runs only with `[train] remote = true`, reserves itself through its own
//! ymird for the run, and its node.toml is never edited.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Parser as _;
use weirdnet_net::{Keypair, WEIRDNET};
use weirdnet_sync::Roster;
use ymir_core::control::{self, ControlRequest};
use ymir_core::membership::Membership;
use ymir_core::train::{self, RunStatus};
use ymir_core::{Daemon, NodeConfig, Paths};

const MOCK: &str = env!("CARGO_BIN_EXE_ymir-mock-engine");
const TRAINER: &str = env!("CARGO_BIN_EXE_ymir-mock-trainer");
const DATASET: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/decentralized-networks.dataset.jsonl");

async fn ctl(home: &Path, args: &[&str]) -> Result<String> {
    let mut argv = vec!["ymirctl".to_string(), "--home".into(), home.display().to_string()];
    argv.extend(args.iter().map(|a| a.to_string()));
    let cli = ymirctl::Cli::try_parse_from(argv)?;
    let mut out = Vec::new();
    let mut input = std::io::Cursor::new(Vec::new());
    ymirctl::run(cli, &mut out as &mut dyn Write, &mut input).await?;
    Ok(String::from_utf8(out)?)
}

struct Home {
    root: PathBuf,
    home: PathBuf,
    paths: Paths,
}

impl Home {
    fn models(&self) -> PathBuf {
        self.root.join(format!("{}-models", self.home.file_name().unwrap().to_string_lossy()))
    }
}

/// node.toml on Weirdnet over 127.0.0.1 only, dialing `peers`. A model node gets base `alpha` and
/// the mock trainer, taking runs from other devices when `remote`.
fn home(root: &Path, name: &str, model: bool, remote: bool, peers: &[String]) -> Result<Home> {
    home_with(root, name, model, remote, peers, "")
}

/// [`home`], with `env` (TOML lines) added to the trainer's environment.
fn home_with(root: &Path, name: &str, model: bool, remote: bool, peers: &[String], env: &str) -> Result<Home> {
    let home = root.join(name);
    let h = Home { root: root.into(), paths: Paths::new(&home), home };
    std::fs::create_dir_all(&h.home)?;
    std::fs::create_dir_all(h.models())?;
    let peers = peers.iter().map(|p| format!("\"{p}\"")).collect::<Vec<_>>().join(", ");
    let mut toml = format!(
        "[node]\nname = \"{name}\"\n[roles]\nmodel = {model}\ninference = false\napp = {}\n[network]\nlisten = [\"/ip4/127.0.0.1/tcp/0\"]\nmdns = false\ndiscovery = false\nrelays = []\npeers = [{peers}]\n[model]\nllamacpp = \"system:{MOCK}\"\nvllm = \"off\"\nmodels_dir = \"{}\"\n[inference.alerts]\nprobe_secs = 3600\n[app.bridge]\nport = 0\n",
        !model,
        h.models().display()
    );
    if model {
        std::fs::write(h.models().join("alpha.gguf"), b"GGUF mock base")?;
        toml.push_str("[[model.local]]\nid = \"alpha\"\nfile = \"alpha.gguf\"\nrole = \"main\"\nrevision = \"r1\"\nquant = \"Q4_K_M\"\n");
        toml.push_str(&format!("[train]\ntrainer = \"{TRAINER}\"\nbuild = \"b0000\"\nremote = {remote}\n"));
        toml.push_str(&format!("[train.env]\nYMIR_MOCK_TRAINER_RECORD = \"{}\"\n{env}", root.join(format!("{name}.calls")).display()));
    }
    std::fs::write(h.home.join("node.toml"), toml)?;
    Ok(h)
}

/// What `pool link` does, without the network.
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

async fn addr(h: &Home) -> Result<String> {
    let s = control::request(&h.paths.socket(), &ControlRequest::Status).await?.status.unwrap();
    let n = s.network.expect("on Weirdnet");
    let a = n.listening.iter().find(|a| a.starts_with("/ip4/127.0.0.1/tcp/")).expect("a loopback listener");
    let a = a.split("/p2p/").next().unwrap();
    Ok(format!("{a}/p2p/{}", n.peer))
}

/// Retries `f` while it fails with "unreachable" or "did not answer" (the devices are still
/// finding each other), up to `secs`.
async fn reaching(secs: u64, mut f: impl AsyncFnMut() -> Result<String>) -> Result<String> {
    let deadline = Instant::now() + Duration::from_secs(secs);
    loop {
        match f().await {
            Err(e)
                if Instant::now() < deadline && {
                    let m = format!("{e:#}");
                    m.contains("unreachable") || m.contains("did not answer")
                } =>
            {
                tokio::time::sleep(Duration::from_millis(300)).await
            }
            r => return r,
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_app_only_device_trains_on_a_model_node_over_weirdnet() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();
    let ctl_h = home(root, "laptop", false, false, &[])?;
    let first = Membership::create(&ctl_h.paths.pool(), "laptop")?;
    let gpu_h = home(root, "gpu", true, true, &[])?;
    let off_h = home(root, "off", true, false, &[])?;
    prelink(&first, &gpu_h, "gpu");
    prelink(&first, &off_h, "off");
    drop(first);
    let gpu_toml = std::fs::read_to_string(gpu_h.paths.config())?;

    let gpu = start(&gpu_h).await?;
    let off = start(&off_h).await?;
    let peers = vec![addr(&gpu_h).await?, addr(&off_h).await?];
    let ctl_h = home(root, "laptop", false, false, &peers)?;
    let laptop = start(&ctl_h).await?;

    let args = |node: &'static str| -> Vec<String> {
        ["train", "run", "--base", "alpha", "--dataset", DATASET, "--epochs", "1", "--context", "256", "--id", "alpha+dn", "--node", node]
            .iter()
            .map(|s| s.to_string())
            .collect()
    };
    let run = async |node: &'static str| {
        let a = args(node);
        ctl(&ctl_h.home, &a.iter().map(String::as_str).collect::<Vec<_>>()).await
    };

    // A node that hasn't opted in refuses; nothing runs there.
    let err = format!("{:#}", reaching(40, async || run("off").await).await.unwrap_err());
    assert!(err.contains("does not take training runs over Weirdnet") && err.contains("[train] remote = true"), "{err}");
    assert!(!root.join("off.calls").exists(), "no trainer ran on the node that did not opt in");

    // The opted-in node runs it, reserving itself in its own ymird, and the adapter lands there.
    let out = reaching(40, async || run("gpu").await).await?;
    assert!(out.contains("Run ") && out.contains("for `alpha+dn` on gpu"), "{out}");
    assert!(out.contains("Reserved the node in its ymird"), "{out}");
    assert!(out.contains("train: ") && out.contains("extract-lora: 1/1"), "{out}");
    assert!(out.contains("Done on gpu: ") && out.contains("Adapter `alpha+dn` is adapters/alpha+dn/alpha+dn.gguf"), "{out}");
    assert!(out.contains("To serve it: ymirctl models adapter serve alpha+dn --base alpha --node gpu"), "{out}");
    let stored = gpu_h.models().join("adapters/alpha+dn/alpha+dn.gguf");
    assert!(std::fs::read(&stored)?.windows(16).any(|w| w == b"mock lora rank 1"), "the trainer's adapter is on the node");
    let runs = train::runs(&gpu_h.paths, &NodeConfig::load(&gpu_h.paths.config())?.train);
    assert_eq!(runs.len(), 1);
    let r = &runs[0];
    assert_eq!((r.status, r.node.as_str(), r.reserved_in_ymird), (RunStatus::Done, "gpu", true));
    assert_eq!(
        (r.dataset.name.as_str(), r.dataset.rows_in, r.dataset.rows_used, r.dataset.license.as_str()),
        ("decentralized-networks.dataset.jsonl", 5, 4, "CC0-1.0")
    );
    assert!(train::held(&gpu_h.paths).is_none(), "the node's GPU is free again");
    let incoming = gpu_h.paths.state().join("train/incoming");
    assert!(std::fs::read_dir(&incoming).map(|d| d.count()).unwrap_or(0) == 0, "the sent dataset is removed after the run");
    assert_eq!(std::fs::read_to_string(gpu_h.paths.config())?, gpu_toml, "node.toml on the node is never edited");
    assert!(train::runs(&ctl_h.paths, &NodeConfig::load(&ctl_h.paths.config())?.train).is_empty(), "nothing ran on the laptop");

    // The same id again is refused by the node, and --add or a --from outside the models directory never leave the laptop.
    let err = format!("{:#}", run("gpu").await.unwrap_err());
    assert!(err.contains("gpu: ") && err.contains("already"), "{err}");
    let err =
        format!("{:#}", ctl(&ctl_h.home, &["train", "run", "--base", "alpha", "--dataset", DATASET, "--node", "gpu", "--add"]).await.unwrap_err());
    assert!(err.contains("never done over the network"), "{err}");
    let err = format!(
        "{:#}",
        ctl(&ctl_h.home, &["train", "run", "--base", "alpha", "--dataset", DATASET, "--node", "gpu", "--from", "../x.gguf"]).await.unwrap_err()
    );
    assert!(err.contains("models directory"), "{err}");
    let err = format!("{:#}", ctl(&ctl_h.home, &["train", "run", "--base", "alpha", "--dataset", DATASET, "--node", "nobody"]).await.unwrap_err());
    assert!(err.contains("no device named `nobody`"), "{err}");
    let err = format!("{:#}", ctl(&ctl_h.home, &["train", "run", "--base", "alpha", "--dataset", DATASET, "--node", "laptop"]).await.unwrap_err());
    assert!(err.contains("is this device"), "{err}");

    drop((laptop, gpu, off));
    Ok(())
}

/// Stopping `ymirctl` (Ctrl-C drops its connection) cancels the run on the node, which frees its GPU.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dropping_the_caller_cancels_the_run_on_the_node() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();
    let ctl_h = home(root, "laptop", false, false, &[])?;
    let first = Membership::create(&ctl_h.paths.pool(), "laptop")?;
    let gpu_h = home_with(root, "gpu", true, true, &[], "YMIR_MOCK_TRAINER_STEP_MS = \"300\"\n")?;
    prelink(&first, &gpu_h, "gpu");
    drop(first);
    let gpu = start(&gpu_h).await?;
    let ctl_h = home(root, "laptop", false, false, &[addr(&gpu_h).await?])?;
    let laptop = start(&ctl_h).await?;

    let calls = root.join("gpu.calls");
    let started = async {
        while !std::fs::read_to_string(&calls).unwrap_or_default().contains("train --base") {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        tokio::time::sleep(Duration::from_millis(700)).await;
    };
    // `ymirctl` is dropped once the node trains, as Ctrl-C does.
    tokio::select! {
        r = reaching(40, async || ctl(&ctl_h.home, &["train", "run", "--base", "alpha", "--dataset", DATASET, "--epochs", "50", "--node", "gpu"]).await) => {
            panic!("the run ended before it was dropped: {r:?}")
        }
        _ = tokio::time::timeout(Duration::from_secs(60), started) => {}
    }
    assert!(std::fs::read_to_string(&calls).unwrap_or_default().contains("train --base"), "the run never started on the node");
    let cfg = NodeConfig::load(&gpu_h.paths.config())?;
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let runs = train::runs(&gpu_h.paths, &cfg.train);
        if runs.first().is_some_and(|r| r.status == RunStatus::Cancelled) && train::held(&gpu_h.paths).is_none() {
            break;
        }
        assert!(Instant::now() < deadline, "the run was not cancelled: {:?}", runs.first().map(|r| r.status));
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    drop((laptop, gpu));
    Ok(())
}
