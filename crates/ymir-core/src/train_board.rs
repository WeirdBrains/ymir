//! The synced job store: [`JobStore`] over `weirdnet-agent`'s [`JobBoard<TrainJob>`], so a
//! training job one model node started can be carried on by another from its last checkpoint.
//!
//! The board keeps every job as signed events in the user's op log (record type
//! [`JOB_KIND`], synced like any other record) and its claims are majority-vote leases among
//! the devices with [`JOB_ROLE`](crate::train_job::JOB_ROLE). This adapter maps the board's records and errors onto
//! Ymir's ([`train_job`](crate::train_job)), keeps each live [`Claim`](BoardClaim) by job id
//! (the board's claim renews itself in the background), and leaves the times to the board:
//! the `now_ms` arguments are not used.

use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Mutex;
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use weirdnet_agent::jobs::{
    CheckpointRef as BoardCheckpoint, Claim as BoardClaim, ContentHash, JobBoard, JobError, JobFilter, JobPolicy as BoardPolicy,
    JobRecord as BoardRecord, JobState as BoardState, NewJob, SyncHandle,
};

use crate::train_job::{CheckpointRef, Claim, ClaimError, JOB_KIND, JobPolicy, JobRecord, JobState, JobStore, LeaseInfo, TrainJob};

/// [`JobStore`] on the synced board.
pub struct BoardJobStore {
    board: JobBoard<TrainJob>,
    /// This device's peer id (the board's owner and holder names).
    me: String,
    claims: Mutex<HashMap<String, BoardClaim<TrainJob>>>,
}

impl BoardJobStore {
    /// The board of [`JOB_KIND`] jobs on `sync`; `me` is this device's peer id.
    pub fn open(sync: &SyncHandle, me: impl Into<String>) -> Result<Self> {
        let board = JobBoard::open(sync, JOB_KIND).map_err(|e| anyhow!("the training job board: {e}"))?;
        Ok(Self { board, me: me.into(), claims: Mutex::default() })
    }

    fn take(&self, claim: &Claim) -> Result<BoardClaim<TrainJob>> {
        let mut claims = self.claims.lock().map_err(|_| anyhow!("claims lock poisoned"))?;
        match claims.remove(&claim.id) {
            Some(c) if c.term() == claim.term => Ok(c),
            Some(c) => {
                let term = c.term();
                claims.insert(claim.id.clone(), c);
                bail!("job {}: the claim of term {} is no longer held (this device holds term {term})", claim.id, claim.term)
            }
            None => bail!("job {}: the claim of term {} is no longer held", claim.id, claim.term),
        }
    }
}

fn hash(s: &str) -> Result<ContentHash> {
    ContentHash::from_str(s).map_err(|e| anyhow!("{e}"))
}

/// Ymir writes digests as plain hex.
fn hex(h: &ContentHash) -> String {
    h.0.iter().map(|b| format!("{b:02x}")).collect()
}

fn state(s: BoardState) -> JobState {
    match s {
        BoardState::Queued => JobState::Queued,
        BoardState::Running => JobState::Running,
        BoardState::Paused => JobState::Paused,
        BoardState::Done => JobState::Done,
        BoardState::Cancelled => JobState::Cancelled,
        _ => JobState::Failed,
    }
}

fn to_board_policy(p: &JobPolicy) -> BoardPolicy {
    BoardPolicy::for_role(p.role.clone())
        .with_claim(Duration::from_millis(p.claim_ms))
        .with_renew(Duration::from_millis(p.renew_ms))
        .with_takeover_after(Duration::from_millis(p.takeover_after_ms))
        .with_max_attempts(p.max_attempts)
        .with_retry_backoff(Duration::from_millis(p.retry_backoff_ms))
        .with_candidates(p.candidates.clone())
}

fn from_board_policy(p: &BoardPolicy) -> JobPolicy {
    JobPolicy {
        claim_ms: p.claim_ms,
        renew_ms: p.renew_ms,
        takeover_after_ms: p.takeover_after_ms,
        max_attempts: p.max_attempts,
        retry_backoff_ms: p.retry_backoff_ms,
        candidates: p.candidates.clone(),
        role: p.role.clone(),
    }
}

fn checkpoint(c: &BoardCheckpoint) -> CheckpointRef {
    CheckpointRef { seq: c.seq, step: c.step, root: hex(&c.root), bytes: c.bytes, holders: c.holders.clone() }
}

fn record(r: BoardRecord<TrainJob>) -> JobRecord<TrainJob> {
    JobRecord {
        id: r.id,
        kind: r.kind,
        owner: r.owner,
        base_revision: Some(r.base_revision).filter(|b| !b.is_empty()),
        dataset: hex(&r.dataset),
        payload: r.payload,
        state: state(r.state),
        lease: r.lease.map(|l| LeaseInfo { holder: l.holder, term: l.term, expires_ms: l.expires_ms }),
        checkpoint: r.checkpoint.as_ref().map(checkpoint),
        attempts: r.attempts,
        policy: from_board_policy(&r.policy),
        failure: r.failure,
        result: r.result.and_then(|b| serde_json::from_slice(&b).ok()),
        created_ms: r.created_ms,
        updated_ms: r.updated.physical_ms,
        not_before_ms: r.not_before_ms.unwrap_or(0),
    }
}

fn ms(d: Duration) -> u64 {
    d.as_millis().min(u128::from(u64::MAX)) as u64
}

fn claim_error(e: JobError, id: &str, now_ms: u64) -> ClaimError {
    match e {
        JobError::NotFound(_) => ClaimError::NoJob,
        JobError::NotClaimable(s) => ClaimError::NotClaimable(state(s)),
        JobError::Ineligible => ClaimError::NotCandidate,
        JobError::NotYet { retry_in } => ClaimError::TooSoon { until_ms: now_ms + ms(retry_in) },
        JobError::Held { holder, retry_in } => ClaimError::Held { holder, until_ms: now_ms + ms(retry_in) },
        JobError::NoQuorum { reached, needed } => ClaimError::NoQuorum { reached, needed },
        other => ClaimError::Store(anyhow!("claiming job {id}: {other}")),
    }
}

#[async_trait::async_trait]
impl JobStore for BoardJobStore {
    fn post(&self, job: JobRecord<TrainJob>) -> Result<()> {
        if job.owner != self.me {
            bail!("job {} is posted by {}, not this device ({})", job.id, job.owner, self.me);
        }
        let new = NewJob::new(job.id.clone(), job.base_revision.clone().unwrap_or_default(), hash(&job.dataset)?, job.payload)
            .with_policy(to_board_policy(&job.policy));
        let posted = self.board.post(new).map_err(|e| anyhow!("posting job {}: {e}", job.id))?;
        if posted.owner != self.me {
            bail!("job {} exists already (posted by {})", job.id, posted.owner);
        }
        Ok(())
    }

    fn get(&self, id: &str) -> Result<Option<JobRecord<TrainJob>>> {
        Ok(self.board.get(id).map_err(|e| anyhow!("reading job {id}: {e}"))?.map(record))
    }

    fn list(&self) -> Result<Vec<JobRecord<TrainJob>>> {
        let mut v: Vec<_> = self.board.list(&JobFilter::all()).map_err(|e| anyhow!("listing jobs: {e}"))?.into_iter().map(record).collect();
        v.sort_by_key(|j| std::cmp::Reverse(j.created_ms));
        Ok(v)
    }

    async fn claim(&self, id: &str, me: &str, now_ms: u64) -> std::result::Result<Claim, ClaimError> {
        if me != self.me {
            return Err(ClaimError::Store(anyhow!("this store claims for {}, not {me}", self.me)));
        }
        let c = self.board.claim(id, |_| true).await.map_err(|e| claim_error(e, id, now_ms))?;
        let term = c.term();
        self.claims.lock().map_err(|_| ClaimError::Store(anyhow!("claims lock poisoned")))?.insert(id.to_string(), c);
        Ok(Claim { id: id.to_string(), holder: self.me.clone(), term })
    }

    fn renew(&self, claim: &Claim, _now_ms: u64) -> Result<bool> {
        let claims = self.claims.lock().map_err(|_| anyhow!("claims lock poisoned"))?;
        Ok(claims.get(&claim.id).is_some_and(|c| c.term() == claim.term && c.is_valid()))
    }

    fn checkpoint(&self, claim: &Claim, at: CheckpointRef, _now_ms: u64) -> Result<()> {
        let claims = self.claims.lock().map_err(|_| anyhow!("claims lock poisoned"))?;
        let c = claims
            .get(&claim.id)
            .filter(|c| c.term() == claim.term)
            .ok_or_else(|| anyhow!("job {}: the claim of term {} is no longer held", claim.id, claim.term))?;
        c.checkpoint(BoardCheckpoint::new(at.seq, at.step, hash(&at.root)?, at.bytes, at.holders))
            .map_err(|e| anyhow!("job {}: checkpoint {}: {e}", claim.id, at.seq))
    }

    async fn finish(&self, claim: &Claim, result: serde_json::Value, _now_ms: u64) -> Result<()> {
        let c = self.take(claim)?;
        c.finish(serde_json::to_vec(&result)?).await.map_err(|e| anyhow!("job {}: finishing: {e}", claim.id))
    }

    async fn fail(&self, claim: &Claim, reason: &str, retry: bool, _now_ms: u64) -> Result<()> {
        let c = self.take(claim)?;
        c.fail(reason, retry).await.map_err(|e| anyhow!("job {}: recording the failure: {e}", claim.id))
    }

    async fn release(&self, claim: &Claim, _now_ms: u64) -> Result<()> {
        let c = self.take(claim)?;
        c.pause().await.map_err(|e| anyhow!("job {}: pausing: {e}", claim.id))
    }

    fn pause(&self, id: &str, _by: &str, _now_ms: u64) -> Result<()> {
        self.board.pause(id).map_err(|e| anyhow!("pausing job {id}: {e}"))
    }

    fn unpause(&self, id: &str, _by: &str, _now_ms: u64) -> Result<()> {
        self.board.resume(id).map_err(|e| anyhow!("resuming job {id}: {e}"))
    }

    fn cancel(&self, id: &str, _by: &str, _now_ms: u64) -> Result<()> {
        self.board.cancel(id).map_err(|e| anyhow!("cancelling job {id}: {e}"))
    }
}
