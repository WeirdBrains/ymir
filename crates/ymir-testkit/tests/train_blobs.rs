//! P3b across machines: a job's dataset and checkpoint travel between the owner's devices over
//! Weirdnet (`/ymir/blobs/1`), so another node resumes it without any shared folder or copying.
//! Three daemons on 127.0.0.1, each with its own home, models and blob store: the app node holds
//! the seed; gpu1 and gpu2 are cert-only model nodes (the board's voters, and its only claimers:
//! the job role is `model`). gpu1 posts a 3-epoch job through its `ymird` and loses its GPU after
//! epoch 1's checkpoint; gpu2 resumes the job through its own `ymird`, which fetches the dataset
//! and the checkpoint from gpu1, checks them against their manifests and adds gpu2 to the
//! checkpoint's holders; gpu2 finishes the job, and the app node sees it done.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use serde_json::json;
use weirdnet_net::{Keypair, WEIRDNET};
use weirdnet_sync::Roster;
use ymir_core::control::{self, ControlRequest};
use ymir_core::membership::{Membership, ServiceRole};
use ymir_core::train::{Hyper, RunSpec};
use ymir_core::train_job::{self, JobPolicy, JobState, JobUpdate};
use ymir_core::train_jobs_ctl::cli_ctx;
use ymir_core::{Daemon, NodeConfig, Paths};

const MOCK: &str = env!("CARGO_BIN_EXE_ymir-mock-engine");
const TRAINER: &str = env!("CARGO_BIN_EXE_ymir-mock-trainer");

struct Home {
    root: PathBuf,
    paths: Paths,
}

/// A model node's home of its own: base `alpha` at r1, the mock trainer, `[train] resume`.
fn home(base: &Path, name: &str, model: bool, peers: &[String], env: &[(&str, &str)]) -> Result<Home> {
    let root = base.join(name);
    std::fs::create_dir_all(root.join("ymir"))?;
    std::fs::create_dir_all(root.join("models"))?;
    std::fs::write(root.join("models/alpha.gguf"), b"GGUF mock base")?;
    let peers = peers.iter().map(|p| format!("\"{p}\"")).collect::<Vec<_>>().join(", ");
    let env: String = env.iter().map(|(k, v)| format!("{k} = \"{v}\"\n")).collect();
    let toml = format!(
        "[node]\nname = \"{name}\"\n[roles]\nmodel = {model}\ninference = false\napp = {app}\n[network]\nlisten = [\"/ip4/127.0.0.1/tcp/0\"]\nmdns = false\ndiscovery = false\nrelays = []\npeers = [{peers}]\nsync_secs = 1\n[model]\nllamacpp = \"system:{MOCK}\"\nvllm = \"off\"\nmodels_dir = \"{}\"\n[[model.local]]\nid = \"alpha\"\nfile = \"alpha.gguf\"\nrole = \"main\"\nrevision = \"r1\"\n[train]\ntrainer = \"{TRAINER}\"\nbuild = \"b0000\"\nresume = true\n[train.env]\nYMIR_MOCK_TRAINER_RECORD = \"{}\"\n{env}[app.bridge]\nport = 0\n",
        root.join("models").display(),
        root.join("trainer.calls").display(),
        app = !model,
    );
    let paths = Paths::new(root.join("ymir"));
    std::fs::write(paths.config(), toml)?;
    Ok(Home { root, paths })
}

fn cfg(h: &Home) -> Result<NodeConfig> {
    NodeConfig::load(&h.paths.config())
}

async fn addr(h: &Home) -> Result<String> {
    let s = control::request(&h.paths.socket(), &ControlRequest::Status).await?.status.unwrap();
    let n = s.network.expect("on Weirdnet");
    let a = n.listening.iter().find(|a| a.starts_with("/ip4/127.0.0.1/tcp/")).expect("a loopback listener");
    Ok(format!("{}/p2p/{}", a.split("/p2p/").next().unwrap(), n.peer))
}

/// A generated dataset (P1) of two usable rows with its manifest, outside both homes.
fn dataset(dir: &Path) -> Result<PathBuf> {
    let path = dir.join("kad.jsonl");
    let row = |id: &str, q: &str, a: &str| json!({"id": id, "ok": true, "usable": true, "messages": [{"role": "user", "content": q}], "completion": a, "attempts": 1});
    let body =
        format!("{}\n{}\n", row("1", "What is a DHT?", "A distributed hash table."), row("2", "What is Kademlia?", "A DHT with an XOR metric."));
    std::fs::write(&path, body)?;
    let manifest = ymir_core::dataset::Manifest {
        format: ymir_core::dataset::MANIFEST_FORMAT.into(),
        job: "dataset-1".into(),
        teacher: ymir_core::dataset::Teacher { model: "teacher-8b".into(), license: "apache-2.0".into(), license_source: "operator".into() },
        input_name: "prompts.jsonl".into(),
        input_sha256: "00".into(),
        output_name: "kad.jsonl".into(),
        defaults: Default::default(),
        sources: None,
        cache: ymir_core::batch::CacheMode::Bypass,
        rows_in_input: 2,
        rows_answered: 2,
        rows_usable: 2,
        rows_failed: 0,
        cache_hits: 0,
        standins: 0,
        answered_by: Default::default(),
        created_ms: 0,
        updated_ms: 0,
    };
    std::fs::write(ymir_core::dataset::manifest_path(&path), serde_json::to_vec(&manifest)?)?;
    Ok(path)
}

fn never() -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> {
    Box::pin(std::future::pending())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_job_resumes_on_another_machine_with_its_files_fetched_over_weirdnet() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let base = tmp.path();
    let data = dataset(base)?;

    // The app node holds the seed; gpu1 and gpu2 are linked cert-only as model nodes.
    let app_h = home(base, "app", false, &[], &[])?;
    let holder = Membership::create(&app_h.paths.pool(), "app")?;
    let user = holder.user_keypair()?;
    let service = |name: &str| -> Result<(Keypair, String, _, _)> {
        let key = Keypair::generate_ed25519();
        let peer = key.public().to_peer_id().to_string();
        let cert = Roster::self_certify(&WEIRDNET, &user, &peer, name, 1)?;
        let bundle = holder.service_bundle(&peer, ServiceRole::Model)?;
        holder.approve(cert.clone())?;
        Ok((key, peer, cert, bundle))
    };
    let (k1, gpu1_peer, cert1, b1) = service("gpu1")?;
    let (k2, gpu2_peer, cert2, b2) = service("gpu2")?;
    let user_pub = user.public().encode_protobuf();
    drop(holder);

    // gpu1 first; gpu2 dials gpu1; the app node dials both.
    let die = [("YMIR_MOCK_TRAINER_DIE_AFTER_EPOCH", "1")];
    let gpu1_h = home(base, "gpu1", true, &[], &die)?;
    Membership::adopt_service(&gpu1_h.paths.pool(), k1, &user_pub, cert1, &b1)?;
    let gpu1 = Daemon::start(cfg(&gpu1_h)?, gpu1_h.paths.clone()).await?;
    let started = Instant::now();
    let gpu1_addr = addr(&gpu1_h).await?;
    let gpu2_h = home(base, "gpu2", true, std::slice::from_ref(&gpu1_addr), &[])?;
    Membership::adopt_service(&gpu2_h.paths.pool(), k2, &user_pub, cert2, &b2)?;
    let gpu2 = Daemon::start(cfg(&gpu2_h)?, gpu2_h.paths.clone()).await?;
    let gpu2_addr = addr(&gpu2_h).await?;
    let app_h = home(base, "app", false, &[gpu1_addr, gpu2_addr], &[])?;
    let app = Daemon::start(cfg(&app_h)?, app_h.paths.clone()).await?;

    let (cfg1, cfg2) = (cfg(&gpu1_h)?, cfg(&gpu2_h)?);
    let c1 = cli_ctx(&gpu1_h.paths, &cfg1).await?;
    let c2 = cli_ctx(&gpu2_h.paths, &cfg2).await?;
    assert!(c1.fetch.is_some() && c2.fetch.is_some(), "both go through their ymird");
    assert_eq!((c1.me.as_str(), c2.me.as_str()), (gpu1_peer.as_str(), gpu2_peer.as_str()));
    let c_app = cli_ctx(&app_h.paths, &cfg(&app_h)?).await?;

    // The voters grant nothing inside their restart grace (the default claim time).
    let grace = Duration::from_millis(JobPolicy::default().claim_ms) + Duration::from_secs(2);
    tokio::time::sleep(grace.saturating_sub(started.elapsed())).await;

    // gpu1 posts a 3-epoch job and loses its GPU after epoch 1's checkpoint.
    let policy = JobPolicy { claim_ms: 4_000, renew_ms: 500, takeover_after_ms: 0, max_attempts: 3, retry_backoff_ms: 0, ..Default::default() };
    let spec = RunSpec { base: "alpha".into(), dataset: data.clone(), hyper: Hyper { epochs: 3, ..Default::default() }, ..Default::default() };
    let mut seen = Vec::new();
    let err = train_job::start(&gpu1_h.paths, &cfg1, &c1, &spec, policy, &mut never(), &mut |u| seen.push(u)).await.unwrap_err();
    assert!(format!("{err:#}").contains("GPU went away"), "{err:#}");
    let id = seen.iter().find_map(|u| if let JobUpdate::Posted { job } = u { Some(job.clone()) } else { None }).expect("posted");
    let job = c1.store.get(&id)?.expect("the job");
    let ckpt = job.checkpoint.clone().expect("epoch 1's checkpoint");
    assert_eq!(ckpt.holders, vec![c1.me.clone()]);
    let dataset_root = job.payload.dataset_root.clone();
    assert!(c1.blobs.has(&ckpt.root) && c1.blobs.has(&dataset_root));
    // Nothing is shared: gpu2's store has neither.
    assert!(!c2.blobs.has(&ckpt.root) && !c2.blobs.has(&dataset_root));

    // The failed attempt (and its checkpoint) reaches gpu2's board through the sync pass.
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let j = c2.store.get(&id)?;
        if j.as_ref().is_some_and(|j| j.state == JobState::Queued && j.checkpoint.as_ref().is_some_and(|c| c.seq == 1)) {
            break;
        }
        if Instant::now() > deadline {
            bail!("gpu2 never saw the job queued with its checkpoint: {j:?}");
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }

    // gpu2 resumes it: its ymird fetches both bundles from gpu1 and registers gpu2 as a holder.
    let holders_at_resume = Arc::new(Mutex::new(None));
    let (h, store, jid) = (holders_at_resume.clone(), c2.store.clone(), id.clone());
    let mut seen2 = Vec::new();
    let record = train_job::resume(&gpu2_h.paths, &cfg2, &c2, &id, false, false, &mut never(), &mut |u| {
        if matches!(u, JobUpdate::Resuming { .. }) {
            *h.lock().unwrap() = store.get(&jid).ok().flatten().and_then(|j| j.checkpoint).map(|c| (c.seq, c.holders));
        }
        seen2.push(u)
    })
    .await?;
    assert!(seen2.iter().any(|u| matches!(u, JobUpdate::Resuming { epoch: 1, .. })), "{seen2:?}");
    assert_eq!(record.resumed_from.as_deref(), Some(ckpt.root.as_str()));
    assert_eq!(record.steps, 6);
    assert!(c2.blobs.has(&ckpt.root) && c2.blobs.has(&dataset_root), "fetched into gpu2's own store");
    assert_eq!(*holders_at_resume.lock().unwrap(), Some((1, vec![c1.me.clone(), c2.me.clone()])), "gpu2 holds epoch 1 too");
    let calls = std::fs::read_to_string(gpu2_h.root.join("trainer.calls")).unwrap_or_default();
    assert!(calls.lines().any(|l| l.starts_with("train ") && l.contains("--resume")), "{calls}");
    // No staging folders are left behind.
    let store_dir = cfg2.train.work_dir(&gpu2_h.paths).join("blobs");
    assert!(!std::fs::read_dir(&store_dir)?.flatten().any(|e| e.file_name().to_string_lossy().starts_with(".fetch-")));

    // The app node sees it done, finished by gpu2.
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let j = c_app.store.get(&id)?;
        if let Some(j) = j.as_ref().filter(|j| j.state == JobState::Done) {
            assert_eq!(j.result.as_ref().and_then(|r| r["node"].as_str()), Some("gpu2"));
            break;
        }
        if Instant::now() > deadline {
            bail!("the app node never saw the job done: {j:?}");
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }

    app.shutdown().await;
    gpu1.shutdown().await;
    gpu2.shutdown().await;
    Ok(())
}
