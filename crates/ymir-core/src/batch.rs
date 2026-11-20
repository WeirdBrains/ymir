//! Batch fan-out (M6): a list of independent prompts spread over the pool at `batch` priority.
//!
//! The submitting node keeps at most `concurrency` tasks going, each a separate request the pool
//! routes (so they spread over the inference nodes, and the inference nodes spread them over the
//! replicas). A task that fails in a way another try could fix (`Busy`, unreachable, timed out,
//! failed) is tried again up to `retries` times under the same job id, with a growing pause;
//! `Busy` includes a node's per-device cap, so the fan-out slows to what the pool will take.
//! Results come back in prompt order.
//! Not built yet: the job record is not synced (the submitting node must stay up), and tasks
//! with tools or approvals.

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use serde::{Deserialize, Serialize};
use weirdnet_agent::pool::{NoEvents, Pool};
use weirdnet_agent::remote::RemoteError;
use weirdnet_agent::remote::infer::v2::{Priority, Request};
use weirdnet_agent::{CancelToken, Message};

#[derive(Debug, Clone)]
pub struct BatchOptions {
    pub model: Option<String>,
    /// Tasks in flight at once.
    pub concurrency: usize,
    /// Extra tries per task.
    pub retries: u32,
    /// The job id prefix: task `i` is `{job}-{i}`.
    pub job: String,
}

impl BatchOptions {
    pub fn new(job: impl Into<String>) -> Self {
        Self { model: None, concurrency: 4, retries: 4, job: job.into() }
    }
}

/// One task's outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchItem {
    pub index: usize,
    pub ok: bool,
    /// The answer, or the error.
    pub text: String,
    pub node: String,
    pub model: String,
    /// Tries it took (1 = the first worked).
    pub attempts: u32,
}

fn transient(e: &RemoteError) -> bool {
    matches!(
        e,
        RemoteError::Busy
            | RemoteError::Unreachable { .. }
            | RemoteError::Unavailable { .. }
            | RemoteError::Failed { .. }
            | RemoteError::Closed
            | RemoteError::Timeout
            | RemoteError::Stalled { .. }
    )
}

async fn task(pool: &Pool, index: usize, prompt: &str, o: &BatchOptions, cancel: &CancelToken) -> BatchItem {
    let mut attempts = 0;
    loop {
        attempts += 1;
        let mut req = Request::messages(vec![Message::user(prompt)]).with_priority(Priority::Batch).with_job_id(format!("{}-{index}", o.job));
        req.attempt = attempts - 1;
        if let Some(m) = &o.model {
            req = req.with_model(m.clone());
        }
        match pool.infer(req, &NoEvents, cancel).await {
            Ok(a) => return BatchItem { index, ok: true, text: a.text, node: a.node, model: a.model, attempts },
            Err(i) if attempts <= o.retries && transient(&i.error) && !cancel.is_cancelled() => {
                tokio::select! {
                    _ = cancel.cancelled() => {}
                    _ = tokio::time::sleep(Duration::from_millis(250 * u64::from(attempts))) => {}
                }
            }
            Err(i) => return BatchItem { index, ok: false, text: i.error.to_string(), node: String::new(), model: String::new(), attempts },
        }
    }
}

/// Runs `prompts` and returns the outcomes in prompt order.
pub async fn run(pool: Arc<Pool>, prompts: Vec<String>, o: BatchOptions, cancel: CancelToken) -> Vec<BatchItem> {
    let n = prompts.len();
    let conc = o.concurrency.max(1);
    let o = Arc::new(o);
    let mut done = futures::stream::iter(prompts.into_iter().enumerate())
        .map(move |(i, p)| {
            let (pool, o, cancel) = (pool.clone(), o.clone(), cancel.clone());
            async move { task(&pool, i, &p, &o, &cancel).await }
        })
        .buffer_unordered(conc);
    let mut out: Vec<Option<BatchItem>> = vec![None; n];
    while let Some(item) = done.next().await {
        let i = item.index;
        out[i] = Some(item);
    }
    out.into_iter().flatten().collect()
}
