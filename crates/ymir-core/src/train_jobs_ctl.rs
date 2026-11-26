//! The synced training job board, reached through `ymird`'s control socket. On Weirdnet the
//! daemon owns the device's sync node and membership state, so it holds the
//! [`crate::train_board::BoardJobStore`] (and the claims, which renew in its background); a
//! `ymirctl train run --job` on the same box drives a job through [`ControlJobStore`], which
//! sends each [`JobStore`] call as one [`JobCall`]. Off Weirdnet (or with no `ymird`) the CLI
//! keeps the node's [`crate::train_job::FileJobStore`].

use std::path::PathBuf;

use anyhow::{Result, anyhow, bail};
use serde::{Deserialize, Serialize};

use crate::control::{ControlRequest, ControlResponse};
use crate::train_job::{CheckpointRef, Claim, ClaimError, JobRecord, JobState, JobStore, TrainJob};

/// One [`JobStore`] call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "call", rename_all = "snake_case")]
pub enum JobCall {
    /// Whether the board is up here, and this device's id on it.
    Me,
    Post {
        job: Box<JobRecord<TrainJob>>,
    },
    Get {
        id: String,
    },
    List,
    Claim {
        id: String,
        me: String,
        now_ms: u64,
    },
    Renew {
        claim: WireClaim,
        now_ms: u64,
    },
    Checkpoint {
        claim: WireClaim,
        at: CheckpointRef,
        now_ms: u64,
    },
    Finish {
        claim: WireClaim,
        result: serde_json::Value,
        now_ms: u64,
    },
    Fail {
        claim: WireClaim,
        reason: String,
        retry: bool,
        now_ms: u64,
    },
    Release {
        claim: WireClaim,
        now_ms: u64,
    },
    Pause {
        id: String,
        by: String,
        now_ms: u64,
    },
    Unpause {
        id: String,
        by: String,
        now_ms: u64,
    },
    Cancel {
        id: String,
        by: String,
        now_ms: u64,
    },
}

/// A [`Claim`] on the wire.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireClaim {
    pub id: String,
    pub holder: String,
    pub term: u64,
}

impl From<&Claim> for WireClaim {
    fn from(c: &Claim) -> Self {
        Self { id: c.id.clone(), holder: c.holder.clone(), term: c.term }
    }
}

impl From<WireClaim> for Claim {
    fn from(c: WireClaim) -> Self {
        Self { id: c.id, holder: c.holder, term: c.term }
    }
}

/// A [`ClaimError`] on the wire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "why", rename_all = "snake_case")]
pub enum WireClaimError {
    NoJob,
    NotClaimable { state: JobState },
    Held { holder: String, until_ms: u64 },
    TooSoon { until_ms: u64 },
    NotCandidate,
    OutOfAttempts { attempts: u32 },
    NoQuorum { reached: usize, needed: usize },
    Store { message: String },
}

impl From<ClaimError> for WireClaimError {
    fn from(e: ClaimError) -> Self {
        match e {
            ClaimError::NoJob => Self::NoJob,
            ClaimError::NotClaimable(state) => Self::NotClaimable { state },
            ClaimError::Held { holder, until_ms } => Self::Held { holder, until_ms },
            ClaimError::TooSoon { until_ms } => Self::TooSoon { until_ms },
            ClaimError::NotCandidate => Self::NotCandidate,
            ClaimError::OutOfAttempts { attempts } => Self::OutOfAttempts { attempts },
            ClaimError::NoQuorum { reached, needed } => Self::NoQuorum { reached, needed },
            ClaimError::Store(e) => Self::Store { message: format!("{e:#}") },
        }
    }
}

impl From<WireClaimError> for ClaimError {
    fn from(e: WireClaimError) -> Self {
        match e {
            WireClaimError::NoJob => Self::NoJob,
            WireClaimError::NotClaimable { state } => Self::NotClaimable(state),
            WireClaimError::Held { holder, until_ms } => Self::Held { holder, until_ms },
            WireClaimError::TooSoon { until_ms } => Self::TooSoon { until_ms },
            WireClaimError::NotCandidate => Self::NotCandidate,
            WireClaimError::OutOfAttempts { attempts } => Self::OutOfAttempts { attempts },
            WireClaimError::NoQuorum { reached, needed } => Self::NoQuorum { reached, needed },
            WireClaimError::Store { message } => Self::Store(anyhow!(message)),
        }
    }
}

/// The answer to a [`JobCall`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "reply", rename_all = "snake_case")]
pub enum JobReply {
    Me { peer: String },
    Done,
    Job { job: Option<Box<JobRecord<TrainJob>>> },
    Jobs { jobs: Vec<JobRecord<TrainJob>> },
    Claimed { claim: WireClaim },
    Refused { error: WireClaimError },
    Renewed { held: bool },
    Error { message: String },
}

/// Runs `call` on the daemon's board.
pub async fn serve(store: &dyn JobStore, me: &str, call: JobCall) -> JobReply {
    fn done(r: Result<()>) -> JobReply {
        match r {
            Ok(()) => JobReply::Done,
            Err(e) => JobReply::Error { message: format!("{e:#}") },
        }
    }
    let err = |e: anyhow::Error| JobReply::Error { message: format!("{e:#}") };
    match call {
        JobCall::Me => JobReply::Me { peer: me.to_string() },
        JobCall::Post { job } => done(store.post(*job)),
        JobCall::Get { id } => store.get(&id).map(|j| JobReply::Job { job: j.map(Box::new) }).unwrap_or_else(err),
        JobCall::List => store.list().map(|jobs| JobReply::Jobs { jobs }).unwrap_or_else(err),
        // Claims are for this device only: the board's claim renews in this daemon.
        JobCall::Claim { id, me: who, now_ms } if who == me => match store.claim(&id, &who, now_ms).await {
            Ok(c) => JobReply::Claimed { claim: (&c).into() },
            Err(e) => JobReply::Refused { error: e.into() },
        },
        JobCall::Claim { me: who, .. } => JobReply::Error { message: format!("this ymird claims as {me}, not {who}") },
        JobCall::Renew { claim, now_ms } => store.renew(&claim.into(), now_ms).map(|held| JobReply::Renewed { held }).unwrap_or_else(err),
        JobCall::Checkpoint { claim, at, now_ms } => done(store.checkpoint(&claim.into(), at, now_ms)),
        JobCall::Finish { claim, result, now_ms } => done(store.finish(&claim.into(), result, now_ms).await),
        JobCall::Fail { claim, reason, retry, now_ms } => done(store.fail(&claim.into(), &reason, retry, now_ms).await),
        JobCall::Release { claim, now_ms } => done(store.release(&claim.into(), now_ms).await),
        JobCall::Pause { id, by, now_ms } => done(store.pause(&id, &by, now_ms)),
        JobCall::Unpause { id, by, now_ms } => done(store.unpause(&id, &by, now_ms)),
        JobCall::Cancel { id, by, now_ms } => done(store.cancel(&id, &by, now_ms)),
    }
}

/// [`JobStore`] over the control socket of the `ymird` on this box.
pub struct ControlJobStore {
    socket: PathBuf,
}

impl ControlJobStore {
    /// The board through the `ymird` at `socket`, with this device's id on it, when that
    /// daemon is up and on Weirdnet.
    pub async fn connect(socket: PathBuf) -> Option<(Self, String)> {
        match call_async(&socket, JobCall::Me).await {
            Ok(JobReply::Me { peer }) => Some((Self { socket }, peer)),
            _ => None,
        }
    }

    /// A call from sync code, which may run on a runtime worker: on its own thread and runtime.
    fn call(&self, call: JobCall) -> Result<JobReply> {
        let socket = self.socket.clone();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
            rt.block_on(call_async(&socket, call))
        })
        .join()
        .map_err(|_| anyhow!("the control call panicked"))?
    }

    fn done(r: Result<JobReply>) -> Result<()> {
        match r? {
            JobReply::Done => Ok(()),
            other => unexpected(other),
        }
    }
}

async fn call_async(socket: &std::path::Path, call: JobCall) -> Result<JobReply> {
    let r: ControlResponse = crate::control::request(socket, &ControlRequest::Jobs { call }).await?;
    match r.job {
        Some(JobReply::Error { message }) => bail!("{message}"),
        Some(reply) => Ok(reply),
        None => bail!("{}", r.message.unwrap_or_else(|| "this ymird has no training job board (upgrade it)".into())),
    }
}

fn unexpected<T>(r: JobReply) -> Result<T> {
    bail!("ymird answered a job call with {r:?}")
}

#[async_trait::async_trait]
impl JobStore for ControlJobStore {
    fn post(&self, job: JobRecord<TrainJob>) -> Result<()> {
        Self::done(self.call(JobCall::Post { job: Box::new(job) }))
    }
    fn get(&self, id: &str) -> Result<Option<JobRecord<TrainJob>>> {
        match self.call(JobCall::Get { id: id.into() })? {
            JobReply::Job { job } => Ok(job.map(|j| *j)),
            other => unexpected(other),
        }
    }
    fn list(&self) -> Result<Vec<JobRecord<TrainJob>>> {
        match self.call(JobCall::List)? {
            JobReply::Jobs { jobs } => Ok(jobs),
            other => unexpected(other),
        }
    }
    async fn claim(&self, id: &str, me: &str, now_ms: u64) -> std::result::Result<Claim, ClaimError> {
        match call_async(&self.socket, JobCall::Claim { id: id.into(), me: me.into(), now_ms }).await {
            Ok(JobReply::Claimed { claim }) => Ok(claim.into()),
            Ok(JobReply::Refused { error }) => Err(error.into()),
            Ok(other) => Err(ClaimError::Store(anyhow!("ymird answered a claim with {other:?}"))),
            Err(e) => Err(ClaimError::Store(e)),
        }
    }
    fn renew(&self, claim: &Claim, now_ms: u64) -> Result<bool> {
        match self.call(JobCall::Renew { claim: claim.into(), now_ms })? {
            JobReply::Renewed { held } => Ok(held),
            other => unexpected(other),
        }
    }
    fn checkpoint(&self, claim: &Claim, at: CheckpointRef, now_ms: u64) -> Result<()> {
        Self::done(self.call(JobCall::Checkpoint { claim: claim.into(), at, now_ms }))
    }
    async fn finish(&self, claim: &Claim, result: serde_json::Value, now_ms: u64) -> Result<()> {
        Self::done(call_async(&self.socket, JobCall::Finish { claim: claim.into(), result, now_ms }).await)
    }
    async fn fail(&self, claim: &Claim, reason: &str, retry: bool, now_ms: u64) -> Result<()> {
        Self::done(call_async(&self.socket, JobCall::Fail { claim: claim.into(), reason: reason.into(), retry, now_ms }).await)
    }
    async fn release(&self, claim: &Claim, now_ms: u64) -> Result<()> {
        Self::done(call_async(&self.socket, JobCall::Release { claim: claim.into(), now_ms }).await)
    }
    fn pause(&self, id: &str, by: &str, now_ms: u64) -> Result<()> {
        Self::done(self.call(JobCall::Pause { id: id.into(), by: by.into(), now_ms }))
    }
    fn unpause(&self, id: &str, by: &str, now_ms: u64) -> Result<()> {
        Self::done(self.call(JobCall::Unpause { id: id.into(), by: by.into(), now_ms }))
    }
    fn cancel(&self, id: &str, by: &str, now_ms: u64) -> Result<()> {
        Self::done(self.call(JobCall::Cancel { id: id.into(), by: by.into(), now_ms }))
    }
}

/// The job stores `ymirctl train` uses: the synced board through this box's `ymird` when it is
/// up and on Weirdnet (claims as this device's peer id), else the node's file store.
pub async fn cli_ctx(paths: &crate::Paths, cfg: &crate::NodeConfig) -> Result<crate::train_job::Ctx> {
    if cfg.network.enabled
        && let Some((store, me)) = ControlJobStore::connect(paths.socket()).await
    {
        return Ok(crate::train_job::Ctx {
            store: std::sync::Arc::new(store),
            blobs: std::sync::Arc::new(crate::train_job::LocalBlobs::for_node(paths, cfg)?),
            me,
        });
    }
    crate::train_job::Ctx::local(paths, cfg)
}
