//! Resumable training jobs (training plan P3b): a run whose record outlives the node that runs
//! it, so it carries on from its last checkpoint after a crash, a restart or, once the record is
//! synced, on another node.
//!
//! The job record and its lease are generic and belong to Weirdnet: `weirdnet-agent` 0.6.0's
//! `jobs` feature (`JobRecord<P>` over `weirdnet-sync` 0.8's job log and majority-vote leases),
//! asked for on 2026-11-25. Until it ships, [`JobStore`] is the shape Ymir needs from it and
//! [`FileJobStore`] stands in for it on one node: same states, lease, fencing term, attempts and
//! back-off, kept in files under the work directory. Ymir doesn't sync it; the swap is one type.
//! Checkpoints are content-addressed folders ([`BlobStore`], [`LocalBlobs`]); the record carries
//! only their root and who holds them.
//!
//! What Ymir owns here: the job's payload ([`TrainJob`]), the checks before a node takes a job
//! over ([`check_resumable`]), and driving a run under a claim ([`start`], [`resume`]): renewing
//! the lease, publishing each checkpoint, and finishing, failing (with or without a retry) or
//! pausing the job when the run ends.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::train::{self, CheckpointFile, Hyper, RunRecord, RunSpec, TrainerVersion, Update};
use crate::{NodeConfig, Paths};

/// The job kind (the record type in the job log).
pub const JOB_KIND: &str = "ymir.train";
/// The grant role a device needs to claim (and vote on) training jobs.
pub const JOB_ROLE: &str = "ymir.model";
/// How long a claim waits for the voters to answer (a job just posted syncs to them first).
pub const CLAIM_QUORUM_WAIT: Duration = Duration::from_secs(30);
/// A dataset bundle's index format.
pub const DATASET_FORMAT: &str = "ymir-train-dataset/1";

// ---------------------------------------------------------------------------------------------
// The record (the shape asked of weirdnet-agent 0.6.0)

/// What a training job trains: everything another node needs to check before it carries on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainJob {
    /// The base's id (a llama.cpp `[[model.local]]`).
    pub base: String,
    /// The base's `revision`: a node carries on only with the same one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_revision: Option<String>,
    /// The GGUF trained from, by file name inside `models_dir`, when not the served file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
    /// The dataset's file name and the sha256 of it and of its manifest.
    pub dataset_name: String,
    pub dataset_sha256: String,
    pub manifest_sha256: String,
    /// The dataset bundle's root in the [`BlobStore`] (the dataset and its manifest).
    pub dataset_root: String,
    pub adapter_id: String,
    pub hyper: Hyper,
    /// The trainer that must carry it on.
    pub trainer_build: String,
    pub trainer_interface: u32,
    #[serde(default)]
    pub allow_unknown_license: bool,
    #[serde(default)]
    pub needs: Needs,
}

/// What a node needs free to take a job (advisory until the pool reports it).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Needs {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_mem_mb: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disk_mb: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Queued,
    Running,
    Paused,
    Failed,
    Done,
    Cancelled,
}

impl std::fmt::Display for JobState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            JobState::Queued => "queued",
            JobState::Running => "running",
            JobState::Paused => "paused",
            JobState::Failed => "failed",
            JobState::Done => "done",
            JobState::Cancelled => "cancelled",
        })
    }
}

/// Who holds a job, under which term (the fencing token), and until when.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeaseInfo {
    pub holder: String,
    pub term: u64,
    pub expires_ms: u64,
}

/// Where the last checkpoint is: its content root and who holds the files.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointRef {
    /// Rises with each checkpoint of a job (Ymir: the epoch).
    pub seq: u64,
    pub step: u64,
    pub root: String,
    pub bytes: u64,
    /// Devices that hold it, the writer first.
    pub holders: Vec<String>,
}

/// How a job is claimed and retried.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobPolicy {
    pub claim_ms: u64,
    pub renew_ms: u64,
    /// After a lease lapses, how long before another device may claim the job.
    pub takeover_after_ms: u64,
    pub max_attempts: u32,
    /// The least time between attempts after a failure.
    pub retry_backoff_ms: u64,
    /// Empty: any device with `role`; else only these.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub candidates: Vec<String>,
    pub role: String,
}

impl Default for JobPolicy {
    fn default() -> Self {
        Self {
            claim_ms: 60_000,
            renew_ms: 20_000,
            takeover_after_ms: 300_000,
            max_attempts: 3,
            retry_backoff_ms: 60_000,
            candidates: Vec::new(),
            role: JOB_ROLE.into(),
        }
    }
}

/// A job as every device sees it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobRecord<P> {
    pub id: String,
    pub kind: String,
    /// The device that posted it; only it pauses, resumes or cancels it.
    pub owner: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_revision: Option<String>,
    /// sha256 of the dataset's manifest.
    pub dataset: String,
    pub payload: P,
    pub state: JobState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lease: Option<LeaseInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint: Option<CheckpointRef>,
    pub attempts: u32,
    pub policy: JobPolicy,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    pub created_ms: u64,
    pub updated_ms: u64,
    /// No claim before this (back-off after a failure).
    #[serde(default)]
    pub not_before_ms: u64,
}

impl<P> JobRecord<P> {
    /// Whether a `Running` job's lease has lapsed (its holder is gone or cut off).
    pub fn stalled(&self, now_ms: u64) -> bool {
        self.state == JobState::Running && self.lease.as_ref().is_none_or(|l| l.expires_ms <= now_ms)
    }
}

/// A held claim on a job: its term fences every write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Claim {
    pub id: String,
    pub holder: String,
    pub term: u64,
}

/// Why a claim was refused.
#[derive(Debug)]
pub enum ClaimError {
    NoJob,
    NotClaimable(JobState),
    Held {
        holder: String,
        until_ms: u64,
    },
    TooSoon {
        until_ms: u64,
    },
    NotCandidate,
    OutOfAttempts {
        attempts: u32,
    },
    /// The synced board's voters didn't answer (or hadn't synced the job yet): worth retrying.
    NoQuorum {
        reached: usize,
        needed: usize,
    },
    Store(anyhow::Error),
}

impl std::fmt::Display for ClaimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClaimError::NoJob => write!(f, "no such job"),
            ClaimError::NotClaimable(s) => write!(f, "the job is {s}"),
            ClaimError::Held { holder, until_ms } => write!(f, "{holder} holds the job (its lease runs to {until_ms} ms)"),
            ClaimError::TooSoon { until_ms } => write!(f, "the job may be claimed again from {until_ms} ms"),
            ClaimError::NotCandidate => write!(f, "this device is not one of the job's candidates"),
            ClaimError::OutOfAttempts { attempts } => write!(f, "the job has used its {attempts} attempts"),
            ClaimError::NoQuorum { reached, needed } => write!(f, "no quorum among the voters ({reached} of {needed} grants)"),
            ClaimError::Store(e) => write!(f, "{e:#}"),
        }
    }
}

impl std::error::Error for ClaimError {}

/// The job record and its lease: [`crate::train_board::BoardJobStore`] (`weirdnet-agent`'s
/// synced `JobBoard<TrainJob>`) between nodes, [`FileJobStore`] on one node. Every write under a
/// claim is fenced by its term: it fails once a later term exists. Claiming and ending a claim
/// are async (the board asks the voters); checkpoints and renewals are not (the board's claim
/// renews itself).
#[async_trait::async_trait]
pub trait JobStore: Send + Sync {
    fn post(&self, job: JobRecord<TrainJob>) -> Result<()>;
    fn get(&self, id: &str) -> Result<Option<JobRecord<TrainJob>>>;
    fn list(&self) -> Result<Vec<JobRecord<TrainJob>>>;
    /// Claims a queued or stalled job for `me`. The caller checks eligibility first
    /// ([`check_resumable`]).
    async fn claim(&self, id: &str, me: &str, now_ms: u64) -> std::result::Result<Claim, ClaimError>;
    /// Renews the lease; `false` once the claim is lost (lapsed and taken, or superseded).
    fn renew(&self, claim: &Claim, now_ms: u64) -> Result<bool>;
    fn checkpoint(&self, claim: &Claim, at: CheckpointRef, now_ms: u64) -> Result<()>;
    /// Replaces the holders of the job's checkpoint `seq` (a node that fetched it to resume
    /// adds itself), without a new checkpoint.
    fn set_holders(&self, claim: &Claim, seq: u64, holders: Vec<String>, now_ms: u64) -> Result<()>;
    async fn finish(&self, claim: &Claim, result: serde_json::Value, now_ms: u64) -> Result<()>;
    /// A failure: `retry` puts the job back in the queue (after the back-off) while attempts
    /// are left.
    async fn fail(&self, claim: &Claim, reason: &str, retry: bool, now_ms: u64) -> Result<()>;
    /// Lets go of a claim without a failure or a spent attempt (the job is `Paused`).
    async fn release(&self, claim: &Claim, now_ms: u64) -> Result<()>;
    /// Owner operations.
    fn pause(&self, id: &str, by: &str, now_ms: u64) -> Result<()>;
    fn unpause(&self, id: &str, by: &str, now_ms: u64) -> Result<()>;
    fn cancel(&self, id: &str, by: &str, now_ms: u64) -> Result<()>;
}

/// Content-addressed folders: datasets and checkpoints. A bundle's root is the sha256 of its
/// manifest file, which lists every file's hash. The synced store fetches from the holders in a
/// [`CheckpointRef`] over Weirdnet; [`LocalBlobs`] keeps them on this node.
pub trait BlobStore: Send + Sync {
    /// Stores the files of `dir` that `manifest` (a file in `dir`) lists; returns the root.
    fn put(&self, dir: &Path, manifest: &str, files: &[CheckpointFile]) -> Result<String>;
    /// Puts the bundle `root` into `dest` (created), each file checked against its hash.
    fn fetch(&self, root: &str, dest: &Path) -> Result<()>;
    fn has(&self, root: &str) -> bool;
}

// ---------------------------------------------------------------------------------------------
// The stand-ins

/// The job store on one node: one JSON file per job under `dir`, writes serialised by a lock
/// file. Same semantics as the synced record, without the sync.
pub struct FileJobStore {
    dir: PathBuf,
}

impl FileJobStore {
    pub fn open(dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        Ok(Self { dir })
    }

    /// The store a node keeps under its work directory.
    pub fn for_node(paths: &Paths, cfg: &NodeConfig) -> Result<Self> {
        Self::open(cfg.train.work_dir(paths).join("jobs"))
    }

    /// The node's store when it has one (reading creates nothing).
    pub fn open_existing(paths: &Paths, cfg: &NodeConfig) -> Option<Self> {
        let dir = cfg.train.work_dir(paths).join("jobs");
        dir.is_dir().then_some(Self { dir })
    }

    fn path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{}.json", train::file_safe(id)))
    }

    fn locked<T>(&self, f: impl FnOnce() -> Result<T>) -> Result<T> {
        let lock = self.dir.join(".lock");
        let start = std::time::Instant::now();
        loop {
            match std::fs::OpenOptions::new().write(true).create_new(true).open(&lock) {
                Ok(_) => break,
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    // A lock left by a process that died mid-write is stale after 30 s.
                    let old = std::fs::metadata(&lock).and_then(|m| m.modified()).ok().and_then(|t| t.elapsed().ok());
                    if old.is_some_and(|a| a > Duration::from_secs(30)) {
                        let _ = std::fs::remove_file(&lock);
                    } else if start.elapsed() > Duration::from_secs(10) {
                        bail!("{} is held by another process", lock.display());
                    } else {
                        std::thread::sleep(Duration::from_millis(20));
                    }
                }
                Err(e) => return Err(e).with_context(|| format!("locking {}", lock.display())),
            }
        }
        let r = f();
        let _ = std::fs::remove_file(&lock);
        r
    }

    fn read(&self, id: &str) -> Result<Option<JobRecord<TrainJob>>> {
        match std::fs::read(self.path(id)) {
            Ok(b) => Ok(Some(serde_json::from_slice(&b).with_context(|| format!("parsing job {id}"))?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e).with_context(|| format!("reading job {id}")),
        }
    }

    fn write(&self, job: &JobRecord<TrainJob>) -> Result<()> {
        train::write_private(&self.path(&job.id), &serde_json::to_vec_pretty(job)?)
    }

    /// Reads, changes and writes one job under the lock.
    fn update<T>(&self, id: &str, f: impl FnOnce(&mut JobRecord<TrainJob>) -> Result<T>) -> Result<T> {
        self.locked(|| {
            let mut job = self.read(id)?.with_context(|| format!("no job {id}"))?;
            let r = f(&mut job)?;
            self.write(&job)?;
            Ok(r)
        })
    }

    fn fenced(job: &JobRecord<TrainJob>, claim: &Claim) -> Result<()> {
        match &job.lease {
            Some(l) if l.term == claim.term && l.holder == claim.holder && job.state == JobState::Running => Ok(()),
            Some(l) if l.term > claim.term => bail!("job {}: term {} superseded by {}'s term {}", job.id, claim.term, l.holder, l.term),
            _ => bail!("job {}: the claim of term {} is no longer held", job.id, claim.term),
        }
    }
}

fn owner_only(job: &JobRecord<TrainJob>, by: &str) -> Result<()> {
    if job.owner != by {
        bail!("job {} belongs to {}; only its owner pauses, resumes or cancels it", job.id, job.owner);
    }
    Ok(())
}

#[async_trait::async_trait]
impl JobStore for FileJobStore {
    fn post(&self, job: JobRecord<TrainJob>) -> Result<()> {
        if job.id.is_empty() || job.id.len() > 128 || !job.id.chars().all(|c| c.is_ascii_alphanumeric() || ".-_:+#".contains(c)) {
            bail!("bad job id {:?}", job.id);
        }
        self.locked(|| {
            if self.read(&job.id)?.is_some() {
                bail!("job {} exists already", job.id);
            }
            self.write(&job)
        })
    }

    fn get(&self, id: &str) -> Result<Option<JobRecord<TrainJob>>> {
        self.read(id)
    }

    fn list(&self) -> Result<Vec<JobRecord<TrainJob>>> {
        let mut v: Vec<JobRecord<TrainJob>> = std::fs::read_dir(&self.dir)
            .with_context(|| format!("reading {}", self.dir.display()))?
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
            .filter_map(|e| std::fs::read(e.path()).ok())
            .filter_map(|b| serde_json::from_slice(&b).ok())
            .collect();
        v.sort_by_key(|j| std::cmp::Reverse(j.created_ms));
        Ok(v)
    }

    async fn claim(&self, id: &str, me: &str, now_ms: u64) -> std::result::Result<Claim, ClaimError> {
        let r = self.locked(|| {
            let Some(mut job) = self.read(id)? else { return Ok(Err(ClaimError::NoJob)) };
            if !job.policy.candidates.is_empty() && !job.policy.candidates.iter().any(|c| c == me) {
                return Ok(Err(ClaimError::NotCandidate));
            }
            let term = job.lease.as_ref().map_or(0, |l| l.term) + 1;
            match job.state {
                JobState::Queued => {
                    if now_ms < job.not_before_ms {
                        return Ok(Err(ClaimError::TooSoon { until_ms: job.not_before_ms }));
                    }
                }
                JobState::Running => {
                    let l = job.lease.clone().expect("a running job has a lease");
                    // The holder itself may claim again at once (a restart on the same node).
                    if l.holder != me {
                        if l.expires_ms > now_ms {
                            return Ok(Err(ClaimError::Held { holder: l.holder, until_ms: l.expires_ms }));
                        }
                        let from = l.expires_ms + job.policy.takeover_after_ms;
                        if now_ms < from {
                            return Ok(Err(ClaimError::TooSoon { until_ms: from }));
                        }
                    }
                }
                s => return Ok(Err(ClaimError::NotClaimable(s))),
            }
            if job.attempts >= job.policy.max_attempts {
                job.state = JobState::Failed;
                job.failure.get_or_insert_with(|| "out of attempts".into());
                job.lease = None;
                job.updated_ms = now_ms;
                self.write(&job)?;
                return Ok(Err(ClaimError::OutOfAttempts { attempts: job.attempts }));
            }
            job.attempts += 1;
            job.state = JobState::Running;
            job.lease = Some(LeaseInfo { holder: me.to_string(), term, expires_ms: now_ms + job.policy.claim_ms });
            job.updated_ms = now_ms;
            self.write(&job)?;
            Ok(Ok(Claim { id: id.to_string(), holder: me.to_string(), term }))
        });
        match r {
            Ok(r) => r,
            Err(e) => Err(ClaimError::Store(e)),
        }
    }

    fn renew(&self, claim: &Claim, now_ms: u64) -> Result<bool> {
        self.locked(|| {
            let Some(mut job) = self.read(&claim.id)? else { return Ok(false) };
            if Self::fenced(&job, claim).is_err() {
                return Ok(false);
            }
            let l = job.lease.as_mut().expect("fenced");
            // Lapsed but not yet taken: the holder may keep it.
            l.expires_ms = now_ms + job.policy.claim_ms;
            job.updated_ms = now_ms;
            self.write(&job)?;
            Ok(true)
        })
    }

    fn checkpoint(&self, claim: &Claim, at: CheckpointRef, now_ms: u64) -> Result<()> {
        self.update(&claim.id, |job| {
            Self::fenced(job, claim)?;
            if job.checkpoint.as_ref().is_some_and(|c| c.seq > at.seq) {
                bail!("job {}: checkpoint {} is older than the job's {}", job.id, at.seq, job.checkpoint.as_ref().map_or(0, |c| c.seq));
            }
            job.checkpoint = Some(at);
            job.updated_ms = now_ms;
            Ok(())
        })
    }

    fn set_holders(&self, claim: &Claim, seq: u64, holders: Vec<String>, now_ms: u64) -> Result<()> {
        self.update(&claim.id, |job| {
            Self::fenced(job, claim)?;
            match job.checkpoint.as_mut() {
                Some(c) if c.seq == seq => c.holders = holders,
                _ => bail!("job {}: checkpoint {seq} is not the job's latest", job.id),
            }
            job.updated_ms = now_ms;
            Ok(())
        })
    }

    async fn finish(&self, claim: &Claim, result: serde_json::Value, now_ms: u64) -> Result<()> {
        self.update(&claim.id, |job| {
            Self::fenced(job, claim)?;
            job.state = JobState::Done;
            job.result = Some(result);
            job.lease = None;
            job.failure = None;
            job.updated_ms = now_ms;
            Ok(())
        })
    }

    async fn fail(&self, claim: &Claim, reason: &str, retry: bool, now_ms: u64) -> Result<()> {
        self.update(&claim.id, |job| {
            Self::fenced(job, claim)?;
            job.failure = Some(reason.to_string());
            job.lease = job.lease.take().map(|mut l| {
                l.expires_ms = now_ms;
                l
            });
            if retry && job.attempts < job.policy.max_attempts {
                job.state = JobState::Queued;
                job.not_before_ms = now_ms + job.policy.retry_backoff_ms;
            } else {
                job.state = JobState::Failed;
            }
            job.updated_ms = now_ms;
            Ok(())
        })
    }

    async fn release(&self, claim: &Claim, now_ms: u64) -> Result<()> {
        self.update(&claim.id, |job| {
            Self::fenced(job, claim)?;
            job.state = JobState::Paused;
            // A release gives the attempt back.
            job.attempts = job.attempts.saturating_sub(1);
            if let Some(l) = job.lease.as_mut() {
                l.expires_ms = now_ms;
            }
            job.updated_ms = now_ms;
            Ok(())
        })
    }

    fn pause(&self, id: &str, by: &str, now_ms: u64) -> Result<()> {
        self.update(id, |job| {
            owner_only(job, by)?;
            if !matches!(job.state, JobState::Queued | JobState::Running) {
                bail!("job {id} is {}", job.state);
            }
            job.state = JobState::Paused;
            if let Some(l) = job.lease.as_mut() {
                l.expires_ms = now_ms;
            }
            job.updated_ms = now_ms;
            Ok(())
        })
    }

    fn unpause(&self, id: &str, by: &str, now_ms: u64) -> Result<()> {
        self.update(id, |job| {
            owner_only(job, by)?;
            if job.state != JobState::Paused {
                bail!("job {id} is {}, not paused", job.state);
            }
            job.state = JobState::Queued;
            job.not_before_ms = 0;
            job.updated_ms = now_ms;
            Ok(())
        })
    }

    fn cancel(&self, id: &str, by: &str, now_ms: u64) -> Result<()> {
        self.update(id, |job| {
            owner_only(job, by)?;
            if matches!(job.state, JobState::Done | JobState::Cancelled) {
                bail!("job {id} is {}", job.state);
            }
            job.state = JobState::Cancelled;
            if let Some(l) = job.lease.as_mut() {
                l.expires_ms = now_ms;
            }
            job.updated_ms = now_ms;
            Ok(())
        })
    }
}

/// Bundles on this node: `dir/ROOT/` with the files and the manifest, hard links where the
/// filesystem allows (a checkpoint's model is as large as the base).
pub struct LocalBlobs {
    dir: PathBuf,
}

impl LocalBlobs {
    pub fn open(dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        Ok(Self { dir })
    }

    pub fn for_node(paths: &Paths, cfg: &NodeConfig) -> Result<Self> {
        Self::open(cfg.train.work_dir(paths).join("blobs"))
    }

    /// The folder of a stored bundle and its manifest's name (what `/ymir/blobs/1` serves).
    pub fn bundle(&self, root: &str) -> Option<(PathBuf, String)> {
        if root.len() != 64 || !root.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        let dir = self.dir.join(root);
        let idx: Index = serde_json::from_slice(&std::fs::read(dir.join(INDEX)).ok()?).ok()?;
        Some((dir, idx.manifest))
    }

    /// A fresh staging folder in the store for fetching `root`.
    pub fn staging(&self, root: &str) -> Result<PathBuf> {
        let d = crate::blobs_net::staging_dir(&self.dir, root);
        std::fs::create_dir_all(&d).with_context(|| format!("creating {}", d.display()))?;
        Ok(d)
    }

    /// Stores a fetched bundle whose files were checked against its manifest, and checks that
    /// it lands under `root`.
    pub fn put_checked(&self, dir: &Path, manifest: &str, files: &[CheckpointFile], root: &str) -> Result<()> {
        let got = self.put(dir, manifest, files)?;
        if got != root {
            bail!("the fetched bundle stored as {got}, not {root}");
        }
        Ok(())
    }
}

const INDEX: &str = "index.json";

#[derive(Serialize, Deserialize)]
struct Index {
    manifest: String,
    files: Vec<CheckpointFile>,
}

fn link_or_copy(from: &Path, to: &Path) -> Result<()> {
    if let Some(p) = to.parent() {
        std::fs::create_dir_all(p).with_context(|| format!("creating {}", p.display()))?;
    }
    let _ = std::fs::remove_file(to);
    if std::fs::hard_link(from, to).is_err() {
        std::fs::copy(from, to).with_context(|| format!("copying {} to {}", from.display(), to.display()))?;
    }
    Ok(())
}

impl BlobStore for LocalBlobs {
    fn put(&self, dir: &Path, manifest: &str, files: &[CheckpointFile]) -> Result<String> {
        let mbytes = std::fs::read(dir.join(manifest)).with_context(|| format!("reading {}", dir.join(manifest).display()))?;
        let root = train::checkpoint_root(&mbytes);
        let out = self.dir.join(&root);
        if out.join(INDEX).is_file() {
            return Ok(root);
        }
        let tmp = self.dir.join(format!(".{root}.part"));
        let _ = std::fs::remove_dir_all(&tmp);
        for f in files {
            if !train::safe_rel(&f.path) {
                bail!("`{}` is not a path inside the bundle", f.path);
            }
            link_or_copy(&dir.join(&f.path), &tmp.join(&f.path))?;
        }
        link_or_copy(&dir.join(manifest), &tmp.join(manifest))?;
        train::write_private(&tmp.join(INDEX), &serde_json::to_vec(&Index { manifest: manifest.into(), files: files.to_vec() })?)?;
        std::fs::rename(&tmp, &out).with_context(|| format!("storing {}", out.display()))?;
        Ok(root)
    }

    fn fetch(&self, root: &str, dest: &Path) -> Result<()> {
        let src = self.dir.join(root);
        let idx: Index = serde_json::from_slice(&std::fs::read(src.join(INDEX)).with_context(|| format!("no bundle {root} on this node"))?)?;
        let mbytes = std::fs::read(src.join(&idx.manifest))?;
        if train::checkpoint_root(&mbytes) != root {
            bail!("bundle {root}: its manifest doesn't hash to its root");
        }
        std::fs::create_dir_all(dest).with_context(|| format!("creating {}", dest.display()))?;
        for f in &idx.files {
            if !train::safe_rel(&f.path) {
                bail!("bundle {root}: `{}` is not a path inside it", f.path);
            }
            let to = dest.join(&f.path);
            link_or_copy(&src.join(&f.path), &to)?;
            let (sha, n) = train::hash_file(&to)?;
            if sha != f.sha256 || n != f.bytes {
                bail!("bundle {root}: {} doesn't match its hash", f.path);
            }
        }
        link_or_copy(&src.join(&idx.manifest), &dest.join(&idx.manifest))?;
        Ok(())
    }

    fn has(&self, root: &str) -> bool {
        self.dir.join(root).join(INDEX).is_file()
    }
}

// ---------------------------------------------------------------------------------------------
// The dataset bundle

#[derive(Serialize, Deserialize)]
struct DatasetIndex {
    format: String,
    dataset: String,
    files: Vec<CheckpointFile>,
}

const DATASET_INDEX: &str = "dataset.json";

/// Puts a dataset and its manifest in the blob store; returns (root, dataset sha256, manifest
/// sha256).
pub fn put_dataset(blobs: &dyn BlobStore, dataset: &Path, scratch: &Path) -> Result<(String, String, String)> {
    let name = dataset.file_name().map(|n| n.to_string_lossy().into_owned()).context("the dataset has no file name")?;
    let mpath = crate::dataset::manifest_path(dataset);
    let mname = mpath.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let _ = std::fs::remove_dir_all(scratch);
    std::fs::create_dir_all(scratch)?;
    let mut files = Vec::new();
    for (src, n) in [(dataset.to_path_buf(), name.clone()), (mpath.clone(), mname)] {
        link_or_copy(&src, &scratch.join(&n))?;
        let (sha256, bytes) = train::hash_file(&scratch.join(&n))?;
        files.push(CheckpointFile { path: n, sha256, bytes });
    }
    let idx = DatasetIndex { format: DATASET_FORMAT.into(), dataset: name, files: files.clone() };
    train::write_private(&scratch.join(DATASET_INDEX), &serde_json::to_vec_pretty(&idx)?)?;
    let root = blobs.put(scratch, DATASET_INDEX, &files)?;
    let _ = std::fs::remove_dir_all(scratch);
    Ok((root, files[0].sha256.clone(), files[1].sha256.clone()))
}

// ---------------------------------------------------------------------------------------------
// Taking a job

/// Why this node may not carry `job` on, or `Ok` with the trainer it would use. `me` is this
/// device; a node carries its own jobs on without `[train] resume`.
pub fn check_resumable(job: &JobRecord<TrainJob>, cfg: &NodeConfig, me: &str, trainer: &TrainerVersion) -> Result<()> {
    let p = &job.payload;
    if job.kind != JOB_KIND {
        bail!("job {} is a `{}` job, not training", job.id, job.kind);
    }
    if job.owner != me && !job.policy.candidates.iter().any(|c| c == me) && !cfg.train.resume {
        bail!("this node doesn't take over other nodes' training jobs; set `resume = true` under [train] to allow it");
    }
    if !job.policy.candidates.is_empty() && !job.policy.candidates.iter().any(|c| c == me) {
        bail!("this node is not one of job {}'s candidates", job.id);
    }
    if !trainer.checkpoints() {
        bail!("this node's trainer speaks interface {}, which can't resume from a checkpoint", trainer.interface);
    }
    if trainer.build.trim() != p.trainer_build.trim() {
        bail!("job {} was trained with trainer build {}, and this node runs {}", job.id, p.trainer_build, trainer.build);
    }
    let base = cfg.model.local.iter().find(|m| m.id == p.base).with_context(|| format!("this node has no base `{}`", p.base))?;
    if base.revision != p.base_revision {
        bail!(
            "job {} trains `{}` at revision {}, and this node has revision {}",
            job.id,
            p.base,
            p.base_revision.as_deref().unwrap_or("(none)"),
            base.revision.as_deref().unwrap_or("(none)")
        );
    }
    if let Some(from) = &p.from
        && !cfg.models_dir().join(from).is_file()
    {
        bail!("job {} trains from `{from}`, which is not in this node's models directory", job.id);
    }
    let taken = cfg.model.local.iter().any(|m| m.id == p.adapter_id || m.lora.iter().any(|a| a.id == p.adapter_id));
    if taken {
        bail!("`{}` is already a model or adapter on this node", p.adapter_id);
    }
    Ok(())
}

/// Where a job's run fetches its dataset and checkpoint.
fn job_dir(paths: &Paths, cfg: &NodeConfig, id: &str) -> PathBuf {
    cfg.train.work_dir(paths).join("job-work").join(train::file_safe(id))
}

/// What a job driver shares.
#[derive(Clone)]
pub struct Ctx {
    pub store: Arc<dyn JobStore>,
    pub blobs: Arc<dyn BlobStore>,
    /// This device (its peer id when it has one, else the node's name).
    pub me: String,
    /// Brings a bundle into `blobs` from the devices that hold it (on Weirdnet: `ymird` over
    /// `/ymir/blobs/1`). `None`: `blobs` is all there is (one box, or a shared folder).
    pub fetch: Option<Arc<dyn BlobFetch>>,
}

impl Ctx {
    /// The stand-in stores of this node.
    pub fn local(paths: &Paths, cfg: &NodeConfig) -> Result<Self> {
        Ok(Self {
            store: Arc::new(FileJobStore::for_node(paths, cfg)?),
            blobs: Arc::new(LocalBlobs::for_node(paths, cfg)?),
            me: cfg.name(),
            fetch: None,
        })
    }
}

/// Makes a bundle local from its holders.
#[async_trait::async_trait]
pub trait BlobFetch: Send + Sync {
    /// Returns the device it came from, `None` when it was here already.
    async fn ensure(&self, root: &str, holders: &[String]) -> Result<Option<String>>;
}

/// What a job's run reports, besides the run's own updates.
#[derive(Debug, Clone, PartialEq)]
pub enum JobUpdate {
    Posted {
        job: String,
    },
    Claimed {
        job: String,
        term: u64,
        attempt: u32,
    },
    Resuming {
        job: String,
        epoch: u64,
        root: String,
    },
    Run(Update),
    /// A checkpoint is in the blob store and the job's record.
    Published {
        seq: u64,
        root: String,
    },
    /// The lease was lost: another node holds the job now, and this run stops.
    Lost,
}

/// Posts `spec` as a resumable job owned by this device and runs it here.
pub async fn start(
    paths: &Paths,
    cfg: &NodeConfig,
    ctx: &Ctx,
    spec: &RunSpec,
    policy: JobPolicy,
    cancel: &mut (dyn std::future::Future<Output = ()> + Send + Unpin),
    on: &mut (dyn FnMut(JobUpdate) + Send),
) -> Result<(String, RunRecord)> {
    let (_, version) = train::check_trainer(&cfg.train).await?;
    if !version.checkpoints() {
        bail!("the trainer speaks interface {}; a resumable job needs interface {}", version.interface, train::CHECKPOINT_INTERFACE);
    }
    let base =
        cfg.model.local.iter().find(|m| m.id == spec.base).with_context(|| format!("no [[model.local]] with id `{}` on this node", spec.base))?;
    let id = format!("train-{}", train::new_run_id());
    let adapter_id = spec.id.clone().unwrap_or_else(|| format!("{}+ft-{}", base.id, id.trim_start_matches("train-")));
    let from = match &spec.from {
        Some(f) => {
            let md = cfg.models_dir();
            let rel = f.strip_prefix(&md).ok().map(|r| r.to_string_lossy().into_owned());
            Some(
                rel.with_context(|| format!("a job trains from a file inside the models directory ({}), so another node can find it", md.display()))?,
            )
        }
        None => None,
    };
    let (dataset_root, dataset_sha256, manifest_sha256) = put_dataset(ctx.blobs.as_ref(), &spec.dataset, &job_dir(paths, cfg, &id).join("put"))?;
    let now = train::now_ms() as u64;
    let job = JobRecord {
        id: id.clone(),
        kind: JOB_KIND.into(),
        owner: ctx.me.clone(),
        base_revision: base.revision.clone(),
        dataset: manifest_sha256.clone(),
        payload: TrainJob {
            base: base.id.clone(),
            base_revision: base.revision.clone(),
            from,
            dataset_name: spec.dataset.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
            dataset_sha256,
            manifest_sha256,
            dataset_root,
            adapter_id,
            hyper: spec.hyper.clone(),
            trainer_build: version.build.clone(),
            trainer_interface: version.interface,
            allow_unknown_license: spec.allow_unknown_license,
            needs: Needs::default(),
        },
        state: JobState::Queued,
        lease: None,
        checkpoint: None,
        attempts: 0,
        policy,
        failure: None,
        result: None,
        created_ms: now,
        updated_ms: now,
        not_before_ms: 0,
    };
    ctx.store.post(job)?;
    on(JobUpdate::Posted { job: id.clone() });
    let record = resume(paths, cfg, ctx, &id, spec.force, spec.keep_full, cancel, on).await?;
    Ok((id, record))
}

/// Claims job `id` and runs it on this node from its last checkpoint (or from the start).
#[allow(clippy::too_many_arguments)]
pub async fn resume(
    paths: &Paths,
    cfg: &NodeConfig,
    ctx: &Ctx,
    id: &str,
    force: bool,
    keep_full: bool,
    cancel: &mut (dyn std::future::Future<Output = ()> + Send + Unpin),
    on: &mut (dyn FnMut(JobUpdate) + Send),
) -> Result<RunRecord> {
    let job = ctx.store.get(id)?.with_context(|| format!("no job {id}"))?;
    let (_, version) = train::check_trainer(&cfg.train).await?;
    check_resumable(&job, cfg, &ctx.me, &version)?;
    // On the synced board a job just posted may not have reached the voters yet.
    let deadline = std::time::Instant::now() + CLAIM_QUORUM_WAIT;
    let claim = loop {
        match ctx.store.claim(id, &ctx.me, train::train_now()).await {
            Err(ClaimError::NoQuorum { .. }) if std::time::Instant::now() < deadline => tokio::time::sleep(Duration::from_millis(250)).await,
            r => break r.map_err(|e| anyhow::anyhow!("claiming job {id}: {e}"))?,
        }
    };
    let job = ctx.store.get(id)?.context("the job went away")?;
    on(JobUpdate::Claimed { job: id.to_string(), term: claim.term, attempt: job.attempts });
    let p = &job.payload;
    let work = job_dir(paths, cfg, id);
    // Off-box bundles first: the dataset (from its poster, or whoever ran the job before) and
    // the last checkpoint (from its holders). A node that fetched the checkpoint holds it too.
    if let Some(fetch) = &ctx.fetch {
        let fetched = async {
            let mut from = vec![job.owner.clone()];
            from.extend(job.checkpoint.iter().flat_map(|c| c.holders.iter().cloned()));
            fetch.ensure(&p.dataset_root, &from).await.context("fetching the job's dataset")?;
            if let Some(c) = &job.checkpoint {
                fetch.ensure(&c.root, &c.holders).await.context("fetching the job's last checkpoint")?;
                if !c.holders.contains(&ctx.me) {
                    let mut holders = c.holders.clone();
                    holders.push(ctx.me.clone());
                    if let Err(e) = ctx.store.set_holders(&claim, c.seq, holders, train::train_now()) {
                        tracing::warn!(job = id, "registering as a holder of checkpoint {}: {e:#}", c.seq);
                    }
                }
            }
            anyhow::Ok(())
        }
        .await;
        if let Err(e) = fetched {
            let _ = ctx.store.fail(&claim, &format!("{e:#}"), true, train::train_now()).await;
            return Err(e);
        }
    }
    // The dataset, from its bundle (checked file by file).
    let data_dir = work.join("dataset");
    let prepared = (|| -> Result<(PathBuf, Option<PathBuf>)> {
        let _ = std::fs::remove_dir_all(&data_dir);
        ctx.blobs.fetch(&p.dataset_root, &data_dir).context("fetching the job's dataset")?;
        let dataset = data_dir.join(&p.dataset_name);
        let resume_dir = match &job.checkpoint {
            Some(c) => {
                let d = work.join(format!("resume-{}", &c.root[..c.root.len().min(16)]));
                if !d.join(train::CHECKPOINT_FILE).is_file() {
                    let _ = std::fs::remove_dir_all(&d);
                    ctx.blobs.fetch(&c.root, &d).context("fetching the job's last checkpoint")?;
                }
                Some(d)
            }
            None => None,
        };
        Ok((dataset, resume_dir))
    })();
    let (dataset, resume_dir) = match prepared {
        Ok(v) => v,
        Err(e) => {
            // Fetching may work on another node (or later): retryable.
            let _ = ctx.store.fail(&claim, &format!("{e:#}"), true, train::train_now()).await;
            return Err(e);
        }
    };
    if let (Some(c), Some(_)) = (&job.checkpoint, &resume_dir) {
        on(JobUpdate::Resuming { job: id.to_string(), epoch: c.seq, root: c.root.clone() });
    }
    let spec = RunSpec {
        base: p.base.clone(),
        dataset,
        id: Some(p.adapter_id.clone()),
        from: p.from.as_ref().map(|f| cfg.models_dir().join(f)),
        hyper: p.hyper.clone(),
        allow_unknown_license: p.allow_unknown_license,
        keep_full,
        force,
        checkpoints: true,
        resume: resume_dir.clone(),
    };
    let result = drive(paths, cfg, ctx, &claim, &job.policy, &spec, cancel, on).await;
    if result.is_ok() {
        let _ = std::fs::remove_dir_all(&work);
    }
    result
}

/// Why a run under a claim stopped early.
const STOP_NONE: u8 = 0;
const STOP_USER: u8 = 1;
const STOP_LOST: u8 = 2;

#[allow(clippy::too_many_arguments)]
async fn drive(
    paths: &Paths,
    cfg: &NodeConfig,
    ctx: &Ctx,
    claim: &Claim,
    policy: &JobPolicy,
    spec: &RunSpec,
    cancel: &mut (dyn std::future::Future<Output = ()> + Send + Unpin),
    on: &mut (dyn FnMut(JobUpdate) + Send),
) -> Result<RunRecord> {
    let stop = Arc::new(AtomicU8::new(STOP_NONE));
    let lost = Arc::new(tokio::sync::Notify::new());
    // Renew the lease while the run goes; stop the run once it is lost.
    let renewer = {
        let (store, claim, stop, lost) = (ctx.store.clone(), claim.clone(), stop.clone(), lost.clone());
        let every = Duration::from_millis(policy.renew_ms.max(50));
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(every).await;
                let held = tokio::task::spawn_blocking({
                    let (store, claim) = (store.clone(), claim.clone());
                    move || store.renew(&claim, train::train_now())
                })
                .await;
                if !matches!(held, Ok(Ok(true))) {
                    let _ = stop.compare_exchange(STOP_NONE, STOP_LOST, Ordering::SeqCst, Ordering::SeqCst);
                    lost.notify_one();
                    return;
                }
            }
        })
    };
    let mut both = Box::pin({
        let stop = stop.clone();
        async move {
            tokio::select! {
                _ = cancel => { let _ = stop.compare_exchange(STOP_NONE, STOP_USER, Ordering::SeqCst, Ordering::SeqCst); }
                _ = lost.notified() => {}
            }
        }
    });
    let mut started = false;
    let mut publish_err: Option<String> = None;
    let mut forward = |u: Update| {
        if matches!(u, Update::Reserved { .. }) {
            started = true;
        }
        if let Update::Checkpoint { dir, info } = &u {
            let published = train::read_checkpoint(dir).and_then(|(c, root)| {
                let r = ctx.blobs.put(dir, train::CHECKPOINT_FILE, &c.files)?;
                debug_assert_eq!(r, root);
                let at =
                    CheckpointRef { seq: u64::from(info.epoch), step: info.step, root: r.clone(), bytes: info.bytes, holders: vec![ctx.me.clone()] };
                ctx.store.checkpoint(claim, at, train::train_now())?;
                Ok(r)
            });
            match published {
                Ok(root) => {
                    on(JobUpdate::Run(u.clone()));
                    on(JobUpdate::Published { seq: u64::from(info.epoch), root });
                    return;
                }
                Err(e) => publish_err = Some(format!("{e:#}")),
            }
        }
        on(JobUpdate::Run(u))
    };
    let result = train::run(paths, cfg, spec, &mut both, &mut forward).await;
    drop(forward);
    drop(both);
    renewer.abort();
    let now = train::train_now();
    match (&result, stop.load(Ordering::SeqCst)) {
        (_, STOP_LOST) => {
            on(JobUpdate::Lost);
            let lost = format!("job {}: the lease was lost, so another node carries it on", claim.id);
            return Err(match result {
                Ok(_) => anyhow::anyhow!(lost),
                Err(e) => e.context(lost),
            });
        }
        (Ok(record), _) => {
            let adapter = record.adapter.clone();
            ctx.store.finish(claim, serde_json::json!({ "node": record.node, "run": record.id, "adapter": adapter, "steps": record.steps, "final_loss": record.final_loss }), now).await?;
        }
        (Err(_), STOP_USER) => {
            // Ctrl-C: the job waits, paused, for `train resume`.
            ctx.store.release(claim, now).await?;
        }
        (Err(e), _) => {
            // Before the run held the GPU, the job itself is wrong for this node (or anywhere):
            // no retry. After, a crash or an out-of-memory may go better elsewhere.
            let mut reason = format!("{e:#}");
            if let Some(p) = publish_err {
                reason.push_str(&format!(" (and a checkpoint could not be published: {p})"));
            }
            ctx.store.fail(claim, &reason, started, now).await?;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(id: &str, owner: &str) -> JobRecord<TrainJob> {
        JobRecord {
            id: id.into(),
            kind: JOB_KIND.into(),
            owner: owner.into(),
            base_revision: Some("r1".into()),
            dataset: "m".into(),
            payload: TrainJob {
                base: "alpha".into(),
                base_revision: Some("r1".into()),
                from: None,
                dataset_name: "d.jsonl".into(),
                dataset_sha256: "d".into(),
                manifest_sha256: "m".into(),
                dataset_root: "root".into(),
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
            policy: JobPolicy {
                claim_ms: 1_000,
                renew_ms: 300,
                takeover_after_ms: 5_000,
                max_attempts: 2,
                retry_backoff_ms: 2_000,
                ..Default::default()
            },
            failure: None,
            result: None,
            created_ms: 0,
            updated_ms: 0,
            not_before_ms: 0,
        }
    }

    fn store() -> (tempfile::TempDir, FileJobStore) {
        let t = tempfile::tempdir().unwrap();
        let s = FileJobStore::open(t.path().join("jobs")).unwrap();
        (t, s)
    }

    fn cp(seq: u64) -> CheckpointRef {
        CheckpointRef { seq, step: seq * 10, root: format!("r{seq}"), bytes: 1, holders: vec!["a".into()] }
    }

    #[tokio::test]
    async fn a_claim_fences_writes_and_a_lapsed_lease_is_taken_over_after_the_wait() {
        let (_t, s) = store();
        s.post(job("j1", "a")).unwrap();
        assert!(s.post(job("j1", "a")).is_err(), "ids are unique");
        let a = s.claim("j1", "a", 100).await.unwrap();
        assert_eq!(a.term, 1);
        assert!(matches!(s.claim("j1", "b", 500).await, Err(ClaimError::Held { .. })));
        s.checkpoint(&a, cp(1), 600).unwrap();
        assert!(s.renew(&a, 900).unwrap());
        // a's lease runs to 1900; b waits takeover_after past that.
        assert!(matches!(s.claim("j1", "b", 2_000).await, Err(ClaimError::TooSoon { until_ms: 6_900 })));
        let b = s.claim("j1", "b", 7_000).await.unwrap();
        assert_eq!(b.term, 2);
        // a comes back: every write of its term is refused.
        assert!(!s.renew(&a, 7_100).unwrap());
        assert!(s.checkpoint(&a, cp(2), 7_100).is_err());
        assert!(s.finish(&a, serde_json::json!({}), 7_100).await.is_err());
        let j = s.get("j1").unwrap().unwrap();
        assert_eq!((j.state, j.attempts, j.checkpoint.unwrap().seq), (JobState::Running, 2, 1));
        s.finish(&b, serde_json::json!({"ok": true}), 7_200).await.unwrap();
        assert_eq!(s.get("j1").unwrap().unwrap().state, JobState::Done);
        assert!(matches!(s.claim("j1", "a", 8_000).await, Err(ClaimError::NotClaimable(JobState::Done))));
    }

    #[tokio::test]
    async fn the_holder_reclaims_at_once_after_a_restart() {
        let (_t, s) = store();
        s.post(job("j", "a")).unwrap();
        let a1 = s.claim("j", "a", 0).await.unwrap();
        let a2 = s.claim("j", "a", 10).await.unwrap();
        assert_eq!(a2.term, a1.term + 1);
        assert!(!s.renew(&a1, 20).unwrap());
    }

    #[tokio::test]
    async fn retryable_failures_back_off_and_stop_at_max_attempts() {
        let (_t, s) = store();
        s.post(job("j", "a")).unwrap();
        let c = s.claim("j", "a", 0).await.unwrap();
        s.fail(&c, "oom", true, 100).await.unwrap();
        let j = s.get("j").unwrap().unwrap();
        assert_eq!((j.state, j.not_before_ms), (JobState::Queued, 2_100));
        assert!(matches!(s.claim("j", "b", 1_000).await, Err(ClaimError::TooSoon { until_ms: 2_100 })));
        let c = s.claim("j", "b", 2_200).await.unwrap();
        s.fail(&c, "oom again", true, 2_300).await.unwrap();
        let j = s.get("j").unwrap().unwrap();
        assert_eq!((j.state, j.attempts, j.failure.as_deref()), (JobState::Failed, 2, Some("oom again")));
    }

    #[tokio::test]
    async fn a_failure_without_retry_is_final_and_lapsed_claims_count_as_attempts() {
        let (_t, s) = store();
        s.post(job("j", "a")).unwrap();
        let c = s.claim("j", "a", 0).await.unwrap();
        s.fail(&c, "base mismatch", false, 1).await.unwrap();
        assert_eq!(s.get("j").unwrap().unwrap().state, JobState::Failed);

        s.post(job("k", "a")).unwrap();
        s.claim("k", "a", 0).await.unwrap(); // lapses at 1000
        s.claim("k", "b", 6_000).await.unwrap(); // second attempt, lapses at 7000
        assert!(matches!(s.claim("k", "c", 20_000).await, Err(ClaimError::OutOfAttempts { attempts: 2 })));
        assert_eq!(s.get("k").unwrap().unwrap().state, JobState::Failed);
    }

    #[tokio::test]
    async fn owners_pause_resume_and_cancel_and_a_release_gives_the_attempt_back() {
        let (_t, s) = store();
        s.post(job("j", "a")).unwrap();
        let c = s.claim("j", "a", 0).await.unwrap();
        s.release(&c, 10).await.unwrap();
        let j = s.get("j").unwrap().unwrap();
        assert_eq!((j.state, j.attempts), (JobState::Paused, 0));
        assert!(matches!(s.claim("j", "a", 20).await, Err(ClaimError::NotClaimable(JobState::Paused))));
        assert!(s.unpause("j", "b", 30).is_err(), "only the owner");
        s.unpause("j", "a", 30).unwrap();
        s.claim("j", "a", 40).await.unwrap();
        s.pause("j", "a", 50).unwrap();
        s.cancel("j", "a", 60).unwrap();
        assert_eq!(s.get("j").unwrap().unwrap().state, JobState::Cancelled);
    }

    #[tokio::test]
    async fn candidates_narrow_who_may_claim() {
        let (_t, s) = store();
        let mut j = job("j", "a");
        j.policy.candidates = vec!["b".into()];
        s.post(j).unwrap();
        assert!(matches!(s.claim("j", "c", 0).await, Err(ClaimError::NotCandidate)));
        s.claim("j", "b", 0).await.unwrap();
    }

    #[tokio::test]
    async fn blobs_round_trip_and_refuse_a_changed_file() {
        let t = tempfile::tempdir().unwrap();
        let blobs = LocalBlobs::open(t.path().join("blobs")).unwrap();
        let src = t.path().join("src");
        std::fs::create_dir_all(src.join("sub")).unwrap();
        std::fs::write(src.join("sub/model.gguf"), b"weights").unwrap();
        let (sha256, bytes) = train::hash_file(&src.join("sub/model.gguf")).unwrap();
        let files = vec![CheckpointFile { path: "sub/model.gguf".into(), sha256, bytes }];
        std::fs::write(src.join("m.json"), serde_json::to_vec(&files).unwrap()).unwrap();
        let root = blobs.put(&src, "m.json", &files).unwrap();
        assert_eq!(root, train::checkpoint_root(&std::fs::read(src.join("m.json")).unwrap()));
        assert!(blobs.has(&root));
        let out = t.path().join("out");
        blobs.fetch(&root, &out).unwrap();
        assert_eq!(std::fs::read(out.join("sub/model.gguf")).unwrap(), b"weights");
        // The stored file changes (a hard link shares the source): the fetch refuses it.
        std::fs::remove_file(t.path().join("blobs").join(&root).join("sub/model.gguf")).unwrap();
        std::fs::write(t.path().join("blobs").join(&root).join("sub/model.gguf"), b"other").unwrap();
        assert!(blobs.fetch(&root, &t.path().join("out2")).is_err());
        assert!(
            blobs.put(&src, "m.json", &[CheckpointFile { path: "../x".into(), sha256: String::new(), bytes: 0 }]).is_ok(),
            "an existing root is not stored again"
        );
    }
}
