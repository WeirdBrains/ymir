//! Training plan P3b: a resumable training job with the mock trainer (interface 2). Two node
//! homes share one job store and blob store, standing in for the synced job record
//! (`weirdnet-agent` 0.6.0's `jobs`, not published yet): the first node loses its GPU after an
//! epoch, and the second carries the job on from that epoch's checkpoint. Also: a node refuses a
//! job it can't carry on, Ctrl-C pauses a job, and a run that loses its lease stops.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use serde_json::json;
use ymir_core::train::{self, Hyper, RunSpec};
use ymir_core::train_job::{self, BlobStore, Ctx, FileJobStore, JobPolicy, JobState, JobStore, JobUpdate, LocalBlobs};
use ymir_core::{NodeConfig, Paths};

#[path = "common/board.rs"]
mod board;

const MOCK: &str = env!("CARGO_BIN_EXE_ymir-mock-engine");
const TRAINER: &str = env!("CARGO_BIN_EXE_ymir-mock-trainer");

struct Node {
    root: PathBuf,
    paths: Paths,
}

impl Node {
    fn new(base: &Path, name: &str) -> Result<Self> {
        let root = base.join(name);
        std::fs::create_dir_all(root.join("ymir"))?;
        std::fs::create_dir_all(root.join("models"))?;
        std::fs::write(root.join("models/alpha.gguf"), b"GGUF mock base")?;
        Ok(Self { paths: Paths::new(root.join("ymir")), root })
    }

    /// node.toml for `name`: base `alpha` at `revision`, the mock trainer, `[train] resume`, and
    /// trainer env.
    fn config(&self, name: &str, revision: &str, resume: bool, env: &[(&str, &str)]) -> Result<NodeConfig> {
        let env: String = env.iter().map(|(k, v)| format!("{k} = \"{v}\"\n")).collect();
        let toml = format!(
            "[node]\nname = \"{name}\"\n[roles]\nmodel = true\ninference = true\napp = false\n[model]\nllamacpp = \"system:{MOCK}\"\nvllm = \"off\"\nmodels_dir = \"{}\"\n[[model.local]]\nid = \"alpha\"\nfile = \"alpha.gguf\"\nrole = \"main\"\nrevision = \"{revision}\"\n[train]\ntrainer = \"{TRAINER}\"\nbuild = \"b0000\"\nresume = {resume}\n[train.env]\nYMIR_MOCK_TRAINER_RECORD = \"{}\"\n{env}",
            self.root.join("models").display(),
            self.root.join("trainer.calls").display()
        );
        std::fs::write(self.paths.config(), &toml)?;
        Ok(toml::from_str(&toml)?)
    }

    fn calls(&self) -> String {
        std::fs::read_to_string(self.root.join("trainer.calls")).unwrap_or_default()
    }
}

/// A generated dataset (P1) of two usable rows with its manifest.
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

/// The shared stores (the synced record's stand-in) as seen by device `me`.
fn ctx(shared: &Path, me: &str) -> Result<Ctx> {
    Ok(Ctx { store: Arc::new(FileJobStore::open(shared.join("jobs"))?), blobs: Arc::new(LocalBlobs::open(shared.join("blobs"))?), me: me.into(), fetch: None })
}

fn quick() -> JobPolicy {
    JobPolicy { claim_ms: 2_000, renew_ms: 100, takeover_after_ms: 0, max_attempts: 3, retry_backoff_ms: 0, ..Default::default() }
}

fn spec(data: &Path, epochs: u32) -> RunSpec {
    RunSpec { base: "alpha".into(), dataset: data.to_path_buf(), hyper: Hyper { epochs, ..Default::default() }, ..Default::default() }
}

fn never() -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> {
    Box::pin(std::future::pending())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_job_lost_after_an_epoch_is_carried_on_by_another_node_from_its_checkpoint() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let shared = tmp.path().join("shared");
    let data = dataset(tmp.path())?;
    let n1 = Node::new(tmp.path(), "gpu1")?;
    let cfg1 = n1.config("gpu1", "r1", false, &[("YMIR_MOCK_TRAINER_DIE_AFTER_EPOCH", "1")])?;
    let c1 = ctx(&shared, "gpu1")?;

    // gpu1 starts a 3-epoch job and loses its GPU after epoch 1's checkpoint.
    let mut seen = Vec::new();
    let mut cancel = never();
    let err = train_job::start(&n1.paths, &cfg1, &c1, &spec(&data, 3), quick(), &mut cancel, &mut |u| seen.push(u)).await.unwrap_err();
    assert!(format!("{err:#}").contains("GPU went away"), "{err:#}");
    let id = seen.iter().find_map(|u| if let JobUpdate::Posted { job } = u { Some(job.clone()) } else { None }).expect("posted");
    assert!(seen.iter().any(|u| matches!(u, JobUpdate::Published { seq: 1, .. })), "{seen:?}");
    let job = c1.store.get(&id)?.expect("the job");
    assert_eq!(job.state, JobState::Queued, "a failure after the GPU was held is retried");
    assert_eq!(job.attempts, 1);
    let ckpt = job.checkpoint.clone().expect("epoch 1's checkpoint");
    assert_eq!((ckpt.seq, ckpt.step, ckpt.holders.clone()), (1, 2, vec!["gpu1".to_string()]));
    assert!(c1.blobs.has(&ckpt.root) && c1.blobs.has(&job.payload.dataset_root));
    // The failed run keeps its checkpoint on disk; its full model is gone.
    let run1 = train::runs(&n1.paths, &cfg1.train).into_iter().next().expect("gpu1's run");
    assert_eq!(run1.status, train::RunStatus::Failed);
    assert_eq!(run1.checkpoints.len(), 1);

    // gpu2 has the base at another revision: it refuses the job.
    let n2 = Node::new(tmp.path(), "gpu2")?;
    let c2 = ctx(&shared, "gpu2")?;
    let other = n2.config("gpu2", "r2", true, &[])?;
    let e = train_job::resume(&n2.paths, &other, &c2, &id, false, false, &mut never(), &mut |_| {}).await.unwrap_err();
    assert!(format!("{e:#}").contains("revision r1"), "{e:#}");
    // Same revision, but it doesn't take other nodes' jobs unless `[train] resume = true`.
    let off = n2.config("gpu2", "r1", false, &[])?;
    let e = train_job::resume(&n2.paths, &off, &c2, &id, false, false, &mut never(), &mut |_| {}).await.unwrap_err();
    assert!(format!("{e:#}").contains("resume = true"), "{e:#}");
    assert_eq!(c2.store.get(&id)?.unwrap().attempts, 1, "a refusal spends no attempt");

    // With it, gpu2 carries the job on from epoch 1 and finishes it.
    let cfg2 = n2.config("gpu2", "r1", true, &[])?;
    let mut seen2 = Vec::new();
    let record = train_job::resume(&n2.paths, &cfg2, &c2, &id, false, false, &mut never(), &mut |u| seen2.push(u)).await?;
    assert!(seen2.iter().any(|u| matches!(u, JobUpdate::Resuming { epoch: 1, .. })), "{seen2:?}");
    assert_eq!(record.resumed_from.as_deref(), Some(ckpt.root.as_str()));
    assert_eq!(record.steps, 6);
    assert_eq!(record.node, "gpu2");
    assert!(record.checkpoints.iter().map(|c| c.epoch).eq([2, 3]), "{:?}", record.checkpoints);
    let calls = n2.calls();
    assert!(calls.lines().any(|l| l.starts_with("train ") && l.contains("--resume") && l.contains("--checkpoint-dir")), "{calls}");
    let adapter = record.adapter.clone().expect("the adapter");
    assert!(n2.root.join("models").join(&adapter.file).is_file());
    let job = c2.store.get(&id)?.unwrap();
    assert_eq!((job.state, job.attempts), (JobState::Done, 2));
    assert_eq!(job.result.as_ref().and_then(|r| r["node"].as_str()), Some("gpu2"));
    assert_eq!(job.checkpoint.unwrap().seq, 3);
    // A done run drops its checkpoints and the job's work folder.
    assert!(!n2.paths.state().join("train").join(&record.id).join("checkpoints").exists());
    assert!(!n2.paths.state().join("train/job-work").join(&id).exists());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn ctrl_c_pauses_a_job_and_resume_on_the_same_node_finishes_it() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let shared = tmp.path().join("shared");
    let data = dataset(tmp.path())?;
    let n = Node::new(tmp.path(), "gpu1")?;
    let slow = n.config("gpu1", "r1", false, &[("YMIR_MOCK_TRAINER_STEP_MS", "200")])?;
    let c = ctx(&shared, "gpu1")?;
    let mut id = None;
    let mut cancel: std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> = Box::pin(tokio::time::sleep(Duration::from_millis(700)));
    let e = train_job::start(&n.paths, &slow, &c, &spec(&data, 4), quick(), &mut cancel, &mut |u| {
        if let JobUpdate::Posted { job } = &u {
            id = Some(job.clone());
        }
    })
    .await
    .unwrap_err();
    assert!(format!("{e:#}").contains("cancelled"), "{e:#}");
    let id = id.expect("posted");
    let job = c.store.get(&id)?.unwrap();
    assert_eq!((job.state, job.attempts), (JobState::Paused, 0), "Ctrl-C pauses without spending an attempt");
    // A paused job is not taken until its owner resumes it.
    let cfg = n.config("gpu1", "r1", false, &[])?;
    assert!(train_job::resume(&n.paths, &cfg, &c, &id, false, false, &mut never(), &mut |_| {}).await.is_err());
    c.store.unpause(&id, "gpu1", 0)?;
    let record = train_job::resume(&n.paths, &cfg, &c, &id, false, false, &mut never(), &mut |_| {}).await?;
    assert_eq!(record.steps, 8);
    assert_eq!(c.store.get(&id)?.unwrap().state, JobState::Done);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_run_that_loses_its_lease_stops_and_writes_nothing_more() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let shared = tmp.path().join("shared");
    let data = dataset(tmp.path())?;
    let n = Node::new(tmp.path(), "gpu1")?;
    let cfg = n.config("gpu1", "r1", false, &[("YMIR_MOCK_TRAINER_STEP_MS", "100")])?;
    let c = ctx(&shared, "gpu1")?;
    let thief = ctx(&shared, "gpu2")?;
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    // Another device takes the job over (its clock says the lease lapsed long ago).
    let taker = std::thread::spawn(move || {
        let id = rx.recv().unwrap();
        std::thread::sleep(Duration::from_millis(400));
        futures::executor::block_on(thief.store.claim(&id, "gpu2", u64::MAX / 2)).map(|c| c.term)
    });
    let mut lost = false;
    let e = train_job::start(&n.paths, &cfg, &c, &spec(&data, 30), quick(), &mut never(), &mut |u| match u {
        JobUpdate::Posted { job } => tx.send(job).unwrap(),
        JobUpdate::Lost => lost = true,
        _ => {}
    })
    .await
    .unwrap_err();
    assert!(lost, "the run saw the lease go");
    assert!(format!("{e:#}").contains("lease was lost"), "{e:#}");
    assert_eq!(taker.join().unwrap().expect("the takeover"), 2);
    let job = c.store.list()?.into_iter().next().unwrap();
    assert_eq!((job.state, job.lease.unwrap().holder), (JobState::Running, "gpu2".to_string()), "the old holder wrote no failure over the new term");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_trainer_without_checkpoints_cannot_run_a_job() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let n = Node::new(tmp.path(), "gpu1")?;
    let data = dataset(tmp.path())?;
    let cfg = n.config("gpu1", "r1", false, &[("YMIR_MOCK_TRAINER_INTERFACE", "1")])?;
    let c = ctx(&tmp.path().join("shared"), "gpu1")?;
    let e = train_job::start(&n.paths, &cfg, &c, &spec(&data, 1), quick(), &mut never(), &mut |_| {}).await.unwrap_err();
    assert!(format!("{e:#}").contains("interface 1"), "{e:#}");
    assert!(c.store.list()?.is_empty(), "nothing is posted");
    // A plain run still works with an interface 1 trainer.
    let mut cancel = never();
    let r = train::run(&n.paths, &cfg, &spec(&data, 1), &mut cancel, &mut |_| {}).await?;
    assert_eq!(r.status, train::RunStatus::Done);
    Ok(())
}

async fn ctl(home: &Path, args: &[&str]) -> Result<String> {
    use clap::Parser as _;
    let mut argv = vec!["ymirctl".to_string(), "--home".into(), home.display().to_string()];
    argv.extend(args.iter().map(|a| a.to_string()));
    let cli = ymirctl::Cli::try_parse_from(argv)?;
    let mut out = Vec::new();
    let mut input = std::io::Cursor::new(Vec::new());
    ymirctl::run(cli, &mut out, &mut input).await?;
    Ok(String::from_utf8(out)?)
}

#[tokio::test(flavor = "multi_thread")]
async fn ymirctl_runs_lists_and_resumes_a_job() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let data = dataset(tmp.path())?;
    let n = Node::new(tmp.path(), "gpu1")?;
    let home = n.root.join("ymir");
    n.config("gpu1", "r1", false, &[("YMIR_MOCK_TRAINER_DIE_AFTER_EPOCH", "1")])?;
    let d = data.display().to_string();
    let e = ctl(&home, &["train", "run", "--base", "alpha", "--dataset", &d, "--epochs", "2", "--job"]).await.unwrap_err();
    assert!(format!("{e:#}").contains("GPU went away"), "{e:#}");
    let jobs = ctl(&home, &["train", "jobs"]).await?;
    assert!(jobs.contains("queued") && jobs.contains("checkpoint epoch 1 of 2") && jobs.contains("attempt 1 of 3"), "{jobs}");
    let id = jobs.split_whitespace().next().unwrap().to_string();
    let st = ctl(&home, &["train", "status"]).await?;
    assert!(st.contains(&format!("job {id}  queued")) && st.contains("(attempt 1, checkpoint epoch 1)") && st.contains("failed"), "{st}");
    // The retry waits out its back-off; the owner pauses and unpauses it to go now.
    ctl(&home, &["train", "job", "pause", &id]).await?;
    assert!(ctl(&home, &["train", "job", "unpause", &id]).await?.contains("queued"));
    n.config("gpu1", "r1", false, &[])?;
    let out = ctl(&home, &["train", "resume", &id]).await?;
    assert!(out.contains("Carrying on after epoch 1") && out.contains("Done: 4 steps"), "{out}");
    assert!(ctl(&home, &["train", "jobs"]).await?.contains("done"));
    Ok(())
}

/// The first scenario on the synced board (`BoardJobStore`): three linked devices with voters;
/// gpu1 (device a) loses its GPU after epoch 1, and gpu2 (device b) carries the job on from that
/// checkpoint. The blobs stay on a shared folder (moving the bytes is the blob store's job).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn on_the_synced_board_a_lost_job_is_carried_on_by_another_device_from_its_checkpoint() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let shared = tmp.path().join("shared");
    let data = dataset(tmp.path())?;
    let devs = Arc::new(board::devices(&["gpu1", "gpu2", "gpu3"], 2_000, 0));
    let _sync = board::keep_syncing(devs.clone());
    let blobs: Arc<LocalBlobs> = Arc::new(LocalBlobs::open(shared.join("blobs"))?);
    let on = |i: usize| Ctx { store: devs[i].store.clone(), blobs: blobs.clone(), me: devs[i].peer.clone(), fetch: None };
    let policy = JobPolicy { role: train_job::JOB_ROLE.into(), ..quick() };

    let n1 = Node::new(tmp.path(), "gpu1")?;
    let cfg1 = n1.config("gpu1", "r1", false, &[("YMIR_MOCK_TRAINER_DIE_AFTER_EPOCH", "1")])?;
    let c1 = on(0);
    let mut seen = Vec::new();
    let err = train_job::start(&n1.paths, &cfg1, &c1, &spec(&data, 3), policy, &mut never(), &mut |u| seen.push(u)).await.unwrap_err();
    assert!(format!("{err:#}").contains("GPU went away"), "{err:#}");
    let id = seen.iter().find_map(|u| if let JobUpdate::Posted { job } = u { Some(job.clone()) } else { None }).expect("posted");
    assert!(seen.iter().any(|u| matches!(u, JobUpdate::Claimed { attempt: 1, .. })), "{seen:?}");
    assert!(seen.iter().any(|u| matches!(u, JobUpdate::Published { seq: 1, .. })), "{seen:?}");

    // gpu2 sees the failed attempt and epoch 1's checkpoint through sync.
    let c2 = on(1);
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    let job = loop {
        board::sync_all(&devs);
        if let Some(j) = c2.store.get(&id)?
            && j.state == JobState::Queued
            && j.checkpoint.is_some()
        {
            break j;
        }
        assert!(std::time::Instant::now() < deadline, "the job on gpu2: {:?}", c2.store.get(&id)?);
        tokio::time::sleep(Duration::from_millis(100)).await;
    };
    assert_eq!(job.owner, devs[0].peer);
    assert_eq!(job.attempts, 1);
    let ckpt = job.checkpoint.clone().unwrap();
    assert_eq!((ckpt.seq, ckpt.step, ckpt.holders.clone()), (1, 2, vec![devs[0].peer.clone()]));
    assert!(blobs.has(&ckpt.root) && blobs.has(&job.payload.dataset_root));

    let n2 = Node::new(tmp.path(), "gpu2")?;
    let cfg2 = n2.config("gpu2", "r1", true, &[])?;
    let mut seen2 = Vec::new();
    let record = train_job::resume(&n2.paths, &cfg2, &c2, &id, false, false, &mut never(), &mut |u| seen2.push(u)).await?;
    assert!(seen2.iter().any(|u| matches!(u, JobUpdate::Resuming { epoch: 1, .. })), "{seen2:?}");
    assert_eq!(record.resumed_from.as_deref(), Some(ckpt.root.as_str()));
    assert_eq!(record.steps, 6);
    assert!(record.checkpoints.iter().map(|c| c.epoch).eq([2, 3]), "{:?}", record.checkpoints);

    // Done everywhere, with gpu2's result and the last checkpoint.
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    loop {
        board::sync_all(&devs);
        if let Some(j) = c1.store.get(&id)?
            && j.state == JobState::Done
        {
            assert_eq!(j.attempts, 2);
            assert_eq!(j.result.as_ref().and_then(|r| r["node"].as_str()), Some("gpu2"));
            assert_eq!(j.checkpoint.map(|c| c.seq), Some(3));
            break;
        }
        assert!(std::time::Instant::now() < deadline, "done on gpu1: {:?}", c1.store.get(&id)?);
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    Ok(())
}

/// The board's policy: `quick()` with the model role (the voters are the devices with it).
fn board_policy() -> JobPolicy {
    JobPolicy { role: train_job::JOB_ROLE.into(), ..quick() }
}

/// Waits until device `on`'s view of job `id` satisfies `f`.
async fn board_until(
    devs: &[board::Dev],
    on: usize,
    id: &str,
    what: &str,
    f: impl Fn(&train_job::JobRecord<train_job::TrainJob>) -> bool,
) -> Result<train_job::JobRecord<train_job::TrainJob>> {
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    loop {
        if let Some(j) = devs[on].store.get(id)?
            && f(&j)
        {
            return Ok(j);
        }
        assert!(std::time::Instant::now() < deadline, "{what}: {:?}", devs[on].store.get(id)?);
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn on_the_synced_board_ctrl_c_pauses_a_job_and_its_owner_resumes_it() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let data = dataset(tmp.path())?;
    let devs = Arc::new(board::devices(&["gpu1", "gpu2", "gpu3"], 2_000, 0));
    let _sync = board::keep_syncing(devs.clone());
    let c = Ctx { store: devs[0].store.clone(), blobs: Arc::new(LocalBlobs::open(tmp.path().join("shared/blobs"))?), me: devs[0].peer.clone(), fetch: None };
    let n = Node::new(tmp.path(), "gpu1")?;
    let slow = n.config("gpu1", "r1", false, &[("YMIR_MOCK_TRAINER_STEP_MS", "400")])?;
    // Ctrl-C once the run holds the GPU.
    let (go, started) = tokio::sync::oneshot::channel::<()>();
    let mut go = Some(go);
    let mut cancel: std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> = Box::pin(async move {
        let _ = started.await;
        tokio::time::sleep(Duration::from_millis(500)).await;
    });
    let mut id = None;
    let e = train_job::start(&n.paths, &slow, &c, &spec(&data, 4), board_policy(), &mut cancel, &mut |u| match &u {
        JobUpdate::Posted { job } => id = Some(job.clone()),
        JobUpdate::Run(train::Update::Reserved { .. }) => {
            if let Some(g) = go.take() {
                let _ = g.send(());
            }
        }
        _ => {}
    })
    .await
    .unwrap_err();
    assert!(format!("{e:#}").contains("cancelled"), "{e:#}");
    let id = id.expect("posted");
    let job = board_until(&devs, 1, &id, "the pause on gpu2", |j| j.state == JobState::Paused).await?;
    assert_eq!(job.attempts, 0, "Ctrl-C pauses without spending an attempt");
    let cfg = n.config("gpu1", "r1", false, &[])?;
    assert!(train_job::resume(&n.paths, &cfg, &c, &id, false, false, &mut never(), &mut |_| {}).await.is_err(), "a paused job waits for its owner");
    c.store.unpause(&id, &c.me, 0)?;
    let record = train_job::resume(&n.paths, &cfg, &c, &id, false, false, &mut never(), &mut |_| {}).await?;
    assert_eq!(record.steps, 8);
    board_until(&devs, 2, &id, "done on gpu3", |j| j.state == JobState::Done).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn on_the_synced_board_a_run_that_loses_its_lease_stops_and_writes_nothing_more() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let data = dataset(tmp.path())?;
    let devs = Arc::new(board::devices(&["gpu1", "gpu2", "gpu3"], 2_000, 0));
    let _sync = board::keep_syncing(devs.clone());
    let c = Ctx { store: devs[0].store.clone(), blobs: Arc::new(LocalBlobs::open(tmp.path().join("shared/blobs"))?), me: devs[0].peer.clone(), fetch: None };
    let n = Node::new(tmp.path(), "gpu1")?;
    let cfg = n.config("gpu1", "r1", false, &[("YMIR_MOCK_TRAINER_STEP_MS", "100")])?;
    // Once gpu2 has seen the run's first checkpoint, gpu1 drops off the network and gpu2 takes
    // the job over when gpu1's lease lapses.
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    let taker = {
        let devs = devs.clone();
        tokio::spawn(async move {
            let id = tokio::task::spawn_blocking(move || rx.recv().unwrap()).await.unwrap();
            board_until(&devs, 1, &id, "the checkpoint on gpu2", |j| j.checkpoint.is_some()).await.unwrap();
            board::isolate(&devs, 0);
            let deadline = std::time::Instant::now() + Duration::from_secs(20);
            loop {
                match devs[1].store.claim(&id, &devs[1].peer, board::now()).await {
                    Ok(c) => return (id, c.term),
                    Err(e) => assert!(std::time::Instant::now() < deadline, "the takeover: {e}"),
                }
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
        })
    };
    let mut lost = false;
    let e = train_job::start(&n.paths, &cfg, &c, &spec(&data, 30), board_policy(), &mut never(), &mut |u| match u {
        JobUpdate::Posted { job } => tx.send(job).unwrap(),
        JobUpdate::Lost => lost = true,
        _ => {}
    })
    .await
    .unwrap_err();
    assert!(lost, "the run saw the lease go");
    assert!(format!("{e:#}").contains("lease was lost"), "{e:#}");
    let (id, term) = taker.await?;
    let job = devs[1].store.get(&id)?.unwrap();
    assert_eq!(
        (job.state, job.lease.as_ref().map(|l| (l.holder.clone(), l.term))),
        (JobState::Running, Some((devs[1].peer.clone(), term))),
        "the old holder wrote no failure over the new term"
    );
    assert!(job.failure.is_none());
    Ok(())
}
