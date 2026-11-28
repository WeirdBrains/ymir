//! P3b through `ymird`: on Weirdnet each daemon holds the synced training job board, and
//! `ymirctl train run --job` drives it over the control socket (`ControlJobStore`). Two daemons
//! on 127.0.0.1 (no mDNS, discovery or relays): the seed holder (app role) posts a job; a
//! cert-only model node, the only voter, sees it on its board after a sync pass, claims it as
//! its own peer id (once its voter's restart grace, the claim time, is over), checkpoints and
//! finishes it; the holder sees it done, and `train status` on the model node lists it.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use weirdnet_net::{Keypair, WEIRDNET};
use weirdnet_sync::Roster;
use ymir_core::control::{self, ControlRequest};
use ymir_core::membership::{Membership, ServiceRole};
use ymir_core::train::Hyper;
use ymir_core::train_job::{CheckpointRef, JOB_KIND, JobPolicy, JobRecord, JobState, JobStore, Needs, TrainJob};
use ymir_core::train_jobs_ctl::ControlJobStore;
use ymir_core::{Daemon, NodeConfig, Paths};

const MOCK: &str = env!("CARGO_BIN_EXE_ymir-mock-engine");

struct Home {
    paths: Paths,
}

fn home(root: &Path, name: &str, roles: &str, peers: &[String]) -> Result<Home> {
    let home: PathBuf = root.join(name);
    let models = root.join("models");
    std::fs::create_dir_all(&home)?;
    std::fs::create_dir_all(&models)?;
    std::fs::write(models.join("alpha.gguf"), b"GGUF mock")?;
    let local = if roles.contains("model") { "[[model.local]]\nid = \"alpha\"\nfile = \"alpha.gguf\"\nrole = \"main\"\n" } else { "" };
    let peers = peers.iter().map(|p| format!("\"{p}\"")).collect::<Vec<_>>().join(", ");
    let (m, i, a) = (roles.contains("model"), roles.contains("inference"), roles.contains("app"));
    let toml = format!(
        "[node]\nname = \"{name}\"\n[roles]\nmodel = {m}\ninference = {i}\napp = {a}\n[network]\nlisten = [\"/ip4/127.0.0.1/tcp/0\"]\nmdns = false\ndiscovery = false\nrelays = []\npeers = [{peers}]\nsync_secs = 1\n[model]\nllamacpp = \"system:{MOCK}\"\nvllm = \"off\"\nmodels_dir = \"{}\"\n{local}[app.bridge]\nport = 0\n",
        models.display()
    );
    std::fs::write(home.join("node.toml"), toml)?;
    Ok(Home { paths: Paths::new(&home) })
}

async fn start(h: &Home) -> Result<Daemon> {
    Daemon::start(NodeConfig::load(&h.paths.config())?, h.paths.clone()).await
}

async fn addr(h: &Home) -> Result<String> {
    let s = control::request(&h.paths.socket(), &ControlRequest::Status).await?.status.unwrap();
    let n = s.network.expect("on Weirdnet");
    let a = n.listening.iter().find(|a| a.starts_with("/ip4/127.0.0.1/tcp/")).expect("a loopback listener");
    Ok(format!("{}/p2p/{}", a.split("/p2p/").next().unwrap(), n.peer))
}

fn job(id: &str, owner: &str) -> JobRecord<TrainJob> {
    let digest = "ab".repeat(32);
    JobRecord {
        id: id.into(),
        kind: JOB_KIND.into(),
        owner: owner.into(),
        base_revision: None,
        dataset: digest.clone(),
        payload: TrainJob {
            base: "alpha".into(),
            base_revision: None,
            from: None,
            dataset_name: "d.jsonl".into(),
            dataset_sha256: "cd".repeat(32),
            manifest_sha256: digest,
            dataset_root: "ef".repeat(32),
            adapter_id: "alpha+ft".into(),
            hyper: Hyper::default(),
            trainer_build: "b1".into(),
            trainer_interface: 2,
            allow_unknown_license: false,
            needs: Needs::default(),
        },
        state: JobState::Queued,
        lease: None,
        checkpoint: None,
        attempts: 0,
        policy: JobPolicy::default(),
        failure: None,
        result: None,
        created_ms: 1,
        updated_ms: 1,
        not_before_ms: 0,
    }
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64
}

/// Runs a sync store call off the runtime's workers.
async fn blocking<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    tokio::task::spawn_blocking(f).await.unwrap()
}

async fn until_job(
    store: &std::sync::Arc<ControlJobStore>,
    id: &str,
    secs: u64,
    what: &str,
    f: fn(&JobRecord<TrainJob>) -> bool,
) -> Result<JobRecord<TrainJob>> {
    let deadline = Instant::now() + Duration::from_secs(secs);
    loop {
        let (s, i) = (store.clone(), id.to_string());
        if let Some(j) = blocking(move || s.get(&i)).await?
            && f(&j)
        {
            return Ok(j);
        }
        if Instant::now() > deadline {
            bail!("gave up waiting for: {what}");
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn ymird_holds_the_job_board_and_ymirctl_drives_it_over_the_control_socket() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();

    // The app node holds the seed; the model node is linked cert-only (`pool add --service`).
    let app_tmp = home(root, "app", "app", &[])?;
    let holder = Membership::create(&app_tmp.paths.pool(), "app")?;
    let model_h = home(root, "model", "model", &[])?;
    let key = Keypair::generate_ed25519();
    let peer = key.public().to_peer_id().to_string();
    let user = holder.user_keypair()?;
    let cert = Roster::self_certify(&WEIRDNET, &user, &peer, "model", 1)?;
    let bundle = holder.service_bundle(&peer, ServiceRole::Model)?;
    holder.approve(cert.clone())?;
    Membership::adopt_service(&model_h.paths.pool(), key, &user.public().encode_protobuf(), cert, &bundle)?;
    drop(holder);

    let model = start(&model_h).await?;
    let started = Instant::now();
    let model_addr = addr(&model_h).await?;
    let app_h = home(root, "app", "app", std::slice::from_ref(&model_addr))?;
    let app = start(&app_h).await?;

    // Both daemons hold a board; each answers as its own peer id.
    let (app_store, app_me) = ControlJobStore::connect(app_h.paths.socket()).await.expect("the app daemon has a board");
    let (model_store, model_me) = ControlJobStore::connect(model_h.paths.socket()).await.expect("the model daemon has a board");
    assert_eq!(model_me, peer);
    assert_ne!(app_me, model_me);
    let (app_store, model_store) = (std::sync::Arc::new(app_store), std::sync::Arc::new(model_store));

    // The holder posts (a job is posted by its owner only).
    let id = "daemon-1";
    let (s, j) = (app_store.clone(), job(id, &model_me));
    assert!(blocking(move || s.post(j)).await.is_err(), "a job for another owner is refused");
    let (s, j) = (app_store.clone(), job(id, &app_me));
    blocking(move || s.post(j)).await?;

    // It reaches the model node's board through the sync pass.
    until_job(&model_store, id, 30, "the job reaches the model node", |j| j.state == JobState::Queued).await?;

    // A claim for another device is refused by the daemon; its own goes through once the
    // voter's restart grace (the default claim time) is over.
    assert!(model_store.claim(id, &app_me, now()).await.is_err(), "ymird claims only as itself");
    let grace = Duration::from_millis(JobPolicy::default().claim_ms);
    let deadline = started + grace + Duration::from_secs(40);
    let claim = loop {
        match model_store.claim(id, &model_me, now()).await {
            Ok(c) => break c,
            Err(_) if Instant::now() < deadline => tokio::time::sleep(Duration::from_millis(500)).await,
            Err(e) => bail!("the model node claims the job: {e}"),
        }
    };
    assert!(started.elapsed() + Duration::from_secs(1) >= grace, "no grant inside the restart grace");

    // Renew, checkpoint and finish, each over the control socket.
    let (s, c) = (model_store.clone(), claim.clone());
    assert!(blocking(move || s.renew(&c, now())).await?);
    let at = CheckpointRef { seq: 1, step: 10, root: "01".repeat(32), bytes: 1_001, holders: vec![model_me.clone()] };
    let (s, c, a) = (model_store.clone(), claim.clone(), at.clone());
    blocking(move || s.checkpoint(&c, a, now())).await?;
    let seen =
        until_job(&app_store, id, 30, "the holder sees the model node's checkpoint", |j| j.checkpoint.as_ref().is_some_and(|c| c.seq == 1)).await?;
    assert_eq!(seen.lease.map(|l| l.holder), Some(model_me.clone()));

    // `train status` on the model node lists the board's job.
    let (p, c, b) = (model_h.paths.clone(), NodeConfig::load(&model_h.paths.config())?, model_store.clone());
    let status = blocking(move || ymir_core::train_net::TrainStatus::read_with(&p, &c, Some(b.as_ref() as &dyn JobStore))).await;
    assert!(status.jobs.iter().any(|j| j.id == id && j.checkpoint_epoch == Some(1)), "{:?}", status.jobs);

    model_store.finish(&claim, serde_json::json!({ "adapter": "alpha+ft" }), now()).await?;
    let done = until_job(&app_store, id, 30, "the holder sees the job done", |j| j.state == JobState::Done).await?;
    assert_eq!(done.checkpoint.map(|c| c.root), Some(at.root));

    app.shutdown().await;
    model.shutdown().await;
    Ok(())
}
