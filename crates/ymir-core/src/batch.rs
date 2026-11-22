//! Batch fan-out (M6): a list of independent tasks spread over the pool at `batch` priority.
//!
//! The submitting node keeps at most `concurrency` tasks going, each a separate request the pool
//! routes (so they spread over the inference nodes, and the inference nodes spread them over the
//! replicas). A task that fails in a way another try could fix (`Busy`, unreachable, timed out,
//! failed) is tried again up to `retries` times under the same job id, with a growing pause;
//! `Busy` includes a node's per-device cap, so the fan-out slows to what the pool will take.
//! Results come back in task order.
//!
//! A task is either a plain prompt ([`run`], `ymirctl batch FILE`) or a [`BatchTask`] with its own
//! messages, sampling and seed ([`run_tasks`], `ymirctl batch --jsonl`, see [`crate::dataset`]).
//! Each outcome says whether the answer came from the answer cache or from a stand-in, read from
//! the answer's metadata, so data generation can leave those rows out.
//!
//! Not built yet: the job record is not synced (the submitting node must stay up), and tasks
//! with tools or approvals.

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use serde::{Deserialize, Serialize};
use weirdnet_agent::pool::{NoEvents, Pool};
use weirdnet_agent::remote::RemoteError;
use weirdnet_agent::remote::infer::v2::{Priority, Request, Sampling, Usage};
use weirdnet_agent::{CancelToken, Message};

#[derive(Debug, Clone)]
pub struct BatchOptions {
    pub model: Option<String>,
    /// Tasks in flight at once.
    pub concurrency: usize,
    /// Extra tries per task.
    pub retries: u32,
    /// The job id prefix: task `id` is `{job}-{id}` (for plain prompts the id is the index).
    pub job: String,
    /// Wait for the requested model to load rather than take a stand-in's answer.
    pub wait_for_model: bool,
    /// Whether answers may come from the inference nodes' answer cache.
    pub cache: CacheMode,
}

impl BatchOptions {
    pub fn new(job: impl Into<String>) -> Self {
        Self { model: None, concurrency: 4, retries: 4, job: job.into(), wait_for_model: false, cache: CacheMode::Use }
    }
}

/// Answer-cache mode for a request (`weirdnet-agent` 0.5.1 infer/2): `use` lets the nodes'
/// answer cache serve and keep the answer, `bypass` neither reads nor stores, `refresh` skips the
/// read and stores the fresh answer. Hits still show in [`BatchItem::cache_hit`].
pub use weirdnet_agent::remote::infer::v2::CacheMode;

/// One task of a batch: its own messages, sampling and seed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchTask {
    /// Stable id, part of the job id (so a retry, or a resumed run, is the same job).
    pub id: String,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub sampling: Sampling,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

impl BatchTask {
    /// A one-message task.
    pub fn prompt(id: impl Into<String>, prompt: &str) -> Self {
        Self { id: id.into(), messages: vec![Message::user(prompt)], sampling: Sampling::default(), seed: None }
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
    /// The task's id (the index, for plain prompts).
    #[serde(default)]
    pub id: String,
    /// The answer came from an inference node's answer cache.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub cache_hit: bool,
    /// A stand-in model answered for the requested one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub standin: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

impl BatchItem {
    fn failed(index: usize, id: &str, error: String, attempts: u32) -> Self {
        Self {
            index,
            ok: false,
            text: error,
            node: String::new(),
            model: String::new(),
            attempts,
            id: id.to_string(),
            cache_hit: false,
            standin: None,
            finish_reason: None,
            usage: None,
        }
    }
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

/// The request for `task`'s try number `attempt` (0 for the first).
pub fn request_for(task: &BatchTask, o: &BatchOptions, attempt: u32) -> Request {
    let mut req = Request::messages(task.messages.clone())
        .with_priority(Priority::Batch)
        .with_job_id(format!("{}-{}", o.job, task.id))
        .with_sampling(task.sampling.clone());
    req.attempt = attempt;
    let mut req = req.with_wait_for_model(o.wait_for_model).with_cache(o.cache);
    if let Some(seed) = task.seed {
        req = req.with_seed(seed);
    }
    if let Some(m) = &o.model {
        req = req.with_model(m.clone());
    }
    req
}

async fn task(pool: &Pool, index: usize, t: &BatchTask, o: &BatchOptions, cancel: &CancelToken) -> BatchItem {
    let mut attempts = 0;
    loop {
        attempts += 1;
        let req = request_for(t, o, attempts - 1);
        match pool.infer(req, &NoEvents, cancel).await {
            Ok(a) => {
                return BatchItem {
                    index,
                    ok: true,
                    text: a.text,
                    node: a.node,
                    model: a.model,
                    attempts,
                    id: t.id.clone(),
                    cache_hit: a.meta.cache_hit,
                    standin: a.meta.standin,
                    finish_reason: a.finish_reason,
                    usage: a.usage,
                };
            }
            Err(i) if attempts <= o.retries && transient(&i.error) && !cancel.is_cancelled() => {
                tokio::select! {
                    _ = cancel.cancelled() => {}
                    _ = tokio::time::sleep(Duration::from_millis(250 * u64::from(attempts))) => {}
                }
            }
            Err(i) => return BatchItem::failed(index, &t.id, i.error.to_string(), attempts),
        }
    }
}

/// Runs `prompts` and returns the outcomes in prompt order.
pub async fn run(pool: Arc<Pool>, prompts: Vec<String>, o: BatchOptions, cancel: CancelToken) -> Vec<BatchItem> {
    let tasks = prompts.iter().enumerate().map(|(i, p)| BatchTask::prompt(i.to_string(), p)).collect();
    run_tasks(pool, tasks, o, cancel).await
}

/// Runs `tasks` and returns the outcomes in task order.
pub async fn run_tasks(pool: Arc<Pool>, tasks: Vec<BatchTask>, o: BatchOptions, cancel: CancelToken) -> Vec<BatchItem> {
    let n = tasks.len();
    let conc = o.concurrency.max(1);
    let o = Arc::new(o);
    let mut done = futures::stream::iter(tasks.into_iter().enumerate())
        .map(move |(i, t)| {
            let (pool, o, cancel) = (pool.clone(), o.clone(), cancel.clone());
            async move { task(&pool, i, &t, &o, &cancel).await }
        })
        .buffer_unordered(conc);
    let mut out: Vec<Option<BatchItem>> = vec![None; n];
    while let Some(item) = done.next().await {
        let i = item.index;
        out[i] = Some(item);
    }
    out.into_iter().flatten().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_carries_the_task_and_the_options() {
        let mut t = BatchTask::prompt("row-7", "hi");
        t.sampling.temperature = Some(0.5);
        t.sampling.max_tokens = Some(64);
        t.seed = Some(42);
        let mut o = BatchOptions::new("job");
        o.model = Some("teacher".into());
        o.wait_for_model = true;
        let r = request_for(&t, &o, 2);
        assert_eq!(r.job_id, "job-row-7");
        assert_eq!(r.attempt, 2);
        assert_eq!(r.priority, Priority::Batch);
        assert_eq!(r.model.as_deref(), Some("teacher"));
        assert!(r.wait_for_model);
        assert_eq!(r.seed, Some(42));
        assert_eq!((r.sampling.temperature, r.sampling.max_tokens), (Some(0.5), Some(64)));
        // Plain prompts keep today's job ids and take stand-ins.
        let r = request_for(&BatchTask::prompt("3", "x"), &BatchOptions::new("b"), 0);
        assert_eq!(r.job_id, "b-3");
        assert!(!r.wait_for_model && r.seed.is_none());
    }

    #[test]
    fn the_cache_mode_and_wait_go_on_the_wire() {
        let t = BatchTask::prompt("1", "x");
        let mut o = BatchOptions::new("b");
        let plain = request_for(&t, &o, 0);
        assert_eq!(plain.cache, CacheMode::Use);
        // `use` is left off the wire, so older nodes read the request as before.
        assert!(serde_json::to_value(&plain).unwrap().get("cache").is_none());
        o.cache = CacheMode::Bypass;
        o.wait_for_model = true;
        let bypass = request_for(&t, &o, 0);
        assert!(bypass.wait_for_model && bypass.cache == CacheMode::Bypass);
        assert_eq!(serde_json::to_value(&bypass).unwrap()["cache"], "bypass");
        assert_eq!(serde_json::to_string(&CacheMode::Refresh).unwrap(), "\"refresh\"");
    }

    #[test]
    fn items_read_back_without_the_new_fields() {
        let i: BatchItem = serde_json::from_str(r#"{"index":0,"ok":true,"text":"a","node":"n","model":"m","attempts":1}"#).unwrap();
        assert!(!i.cache_hit && i.standin.is_none() && i.id.is_empty());
    }
}
