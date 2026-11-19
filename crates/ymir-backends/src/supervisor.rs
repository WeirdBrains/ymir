//! Starts, probes, drains and stops engine processes for one model node, the way llama-swap did
//! locally: swap groups, idle unload, crash detection with restart backoff, and readiness probes.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::{broadcast, oneshot, watch};
use tokio_stream::wrappers::ReceiverStream;
use ymir_infer::{ChatRequest, CompletionRequest, Event, EventStream, InferError, ModelState};

use crate::engine::{EngineConfig, EngineKind, ModelLaunch};
use crate::sse::{ChunkDecoder, SseParser};

#[derive(Debug, Clone)]
pub struct SupervisorOptions {
    /// Per-model engine logs (`<id>.log`); `None` discards engine output.
    pub log_dir: Option<PathBuf>,
    /// How long `unload` waits for in-flight requests.
    pub drain_timeout: Duration,
    /// How long a stopped engine gets after SIGTERM before SIGKILL.
    pub stop_grace: Duration,
    /// Restart an engine that crashed while serving.
    pub restart: bool,
    /// First restart delay; doubles per crash in the window, up to `backoff_max`.
    pub backoff_base: Duration,
    pub backoff_max: Duration,
    /// This many crashes within `crash_window` is a crash loop: no more restarts until an explicit
    /// (forced) load.
    pub max_crashes: usize,
    pub crash_window: Duration,
}

impl Default for SupervisorOptions {
    fn default() -> Self {
        Self {
            log_dir: None,
            drain_timeout: Duration::from_secs(30),
            stop_grace: Duration::from_secs(10),
            restart: true,
            backoff_base: Duration::from_secs(1),
            backoff_max: Duration::from_secs(60),
            max_crashes: 5,
            crash_window: Duration::from_secs(600),
        }
    }
}

/// Something that happened to an engine; the node's alerting and metrics listen to these.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SupervisorEvent {
    Starting { model: String },
    Ready { model: String, load_secs: f32 },
    LoadFailed { model: String, why: String },
    Crashed { model: String, why: String, crashes: usize, restart_in_secs: Option<f32> },
    CrashLoop { model: String, crashes: usize },
    Unloaded { model: String, idle: bool },
}

/// One model's row in the node's capability record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelStatus {
    pub id: String,
    pub engine: EngineKind,
    pub state: ModelState,
    pub in_flight: usize,
    pub swap_group: Option<String>,
    pub embedding: bool,
    pub pid: Option<u32>,
    /// The last measured load time.
    pub last_load_secs: Option<f32>,
    /// Model file size, for load-time estimates before one was measured.
    pub size_bytes: Option<u64>,
    /// A stock server Ymir only forwards to.
    pub wrapped: bool,
    /// Crashes in the current window.
    pub crashes: usize,
}

struct Running {
    base: String,
    api_key: Option<String>,
    pid: Option<u32>,
    stop: Option<oneshot::Sender<()>>,
    exited: watch::Receiver<Option<String>>,
    /// Keeps a wrapped server's never-firing exit channel open.
    _exit_tx: Option<watch::Sender<Option<String>>>,
}

struct Slot {
    launch: ModelLaunch,
    state: ModelState,
    running: Option<Running>,
    in_flight: Arc<AtomicUsize>,
    /// Milliseconds since the supervisor's epoch when the last request finished.
    last_used: Arc<AtomicU64>,
    /// Bumped on every start, so a late exit notice from an old process is ignored.
    generation: u64,
    load_secs: Option<f32>,
    crashes: Vec<Instant>,
    crash_loop: bool,
    restart_at: Option<Instant>,
}

struct Inner {
    engines: HashMap<EngineKind, EngineConfig>,
    opts: SupervisorOptions,
    http: reqwest::Client,
    slots: Mutex<HashMap<String, Slot>>,
    /// Serializes load and unload, so swaps never interleave.
    ops: tokio::sync::Mutex<()>,
    events: broadcast::Sender<SupervisorEvent>,
    epoch: Instant,
}

/// The per-node supervisor. Cheap to clone.
#[derive(Clone)]
pub struct Supervisor {
    inner: Arc<Inner>,
}

impl Supervisor {
    pub fn new(engines: Vec<EngineConfig>, models: Vec<ModelLaunch>, opts: SupervisorOptions) -> Self {
        let slots = models
            .into_iter()
            .map(|launch| {
                let slot = Slot {
                    launch: launch.clone(),
                    state: ModelState::Unloaded,
                    running: None,
                    in_flight: Arc::new(AtomicUsize::new(0)),
                    last_used: Arc::new(AtomicU64::new(0)),
                    generation: 0,
                    load_secs: None,
                    crashes: Vec::new(),
                    crash_loop: false,
                    restart_at: None,
                };
                (launch.id, slot)
            })
            .collect();
        let http =
            reqwest::Client::builder().connect_timeout(Duration::from_secs(5)).no_proxy().build().expect("an HTTP client without TLS always builds");
        let (events, _) = broadcast::channel(256);
        Self {
            inner: Arc::new(Inner {
                engines: engines.into_iter().map(|e| (e.kind, e)).collect(),
                opts,
                http,
                slots: Mutex::new(slots),
                ops: tokio::sync::Mutex::new(()),
                events,
                epoch: Instant::now(),
            }),
        }
    }

    fn slots(&self) -> std::sync::MutexGuard<'_, HashMap<String, Slot>> {
        self.inner.slots.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn emit(&self, ev: SupervisorEvent) {
        let _ = self.inner.events.send(ev);
    }

    /// Engine events from now on.
    pub fn subscribe(&self) -> broadcast::Receiver<SupervisorEvent> {
        self.inner.events.subscribe()
    }

    fn now_ms(&self) -> u64 {
        self.inner.epoch.elapsed().as_millis() as u64
    }

    pub fn status(&self) -> Vec<ModelStatus> {
        let mut out: Vec<ModelStatus> = self
            .slots()
            .values()
            .map(|s| ModelStatus {
                id: s.launch.id.clone(),
                engine: s.launch.engine,
                state: s.state.clone(),
                in_flight: s.in_flight.load(Ordering::SeqCst),
                swap_group: s.launch.swap_group.clone(),
                embedding: s.launch.embedding,
                pid: s.running.as_ref().and_then(|r| r.pid),
                last_load_secs: s.load_secs,
                size_bytes: if s.launch.wrap.is_some() { None } else { std::fs::metadata(&s.launch.source).ok().map(|m| m.len()) },
                wrapped: s.launch.wrap.is_some(),
                crashes: s.crashes.len(),
            })
            .collect();
        out.sort_by(|a, b| a.id.cmp(&b.id));
        out
    }

    pub fn state(&self, id: &str) -> Option<ModelState> {
        self.slots().get(id).map(|s| s.state.clone())
    }

    /// Starts `id` and waits until it is ready (see [`Supervisor::load_with`]).
    pub async fn load(&self, id: &str) -> Result<(), InferError> {
        self.load_with(id, false).await
    }

    /// Starts `id` and waits until its engine is ready. Loading a model in a swap group unloads the
    /// group's other models first. Already ready: returns at once. A model in a crash loop or
    /// waiting for its restart refuses unless `force` (an operator's explicit load), which also
    /// clears the crash history.
    pub async fn load_with(&self, id: &str, force: bool) -> Result<(), InferError> {
        // A request during a crash backoff waits it out (at most `backoff_max`) rather than
        // failing; the scheduled restart and this load serialize on `ops`.
        if !force {
            let wait = {
                let slots = self.slots();
                let slot = slots.get(id).ok_or_else(|| InferError::UnknownModel(id.into()))?;
                slot.restart_at.and_then(|at| at.checked_duration_since(Instant::now()))
            };
            if let Some(d) = wait {
                tokio::time::sleep(d.min(self.inner.opts.backoff_max)).await;
            }
        }
        let _op = self.inner.ops.lock().await;
        let (launch, group_mates) = {
            let mut slots = self.slots();
            let slot = slots.get_mut(id).ok_or_else(|| InferError::UnknownModel(id.into()))?;
            if slot.state == ModelState::Ready {
                return Ok(());
            }
            if force {
                slot.crashes.clear();
                slot.crash_loop = false;
                slot.restart_at = None;
            } else if slot.crash_loop {
                return Err(InferError::Unavailable(format!(
                    "{id} is in a crash loop ({} crashes); fix it, then `ymirctl load {id}`",
                    slot.crashes.len()
                )));
            }
            let launch = slot.launch.clone();
            let mates: Vec<String> = match &launch.swap_group {
                Some(g) => slots
                    .values()
                    .filter(|s| s.launch.id != id && s.launch.swap_group.as_ref() == Some(g) && s.running.is_some())
                    .map(|s| s.launch.id.clone())
                    .collect(),
                None => Vec::new(),
            };
            (launch, mates)
        };
        for mate in group_mates {
            tracing::info!(model = %mate, for_model = %id, "swap group: unloading");
            self.stop(&mate, false).await;
        }
        // A failed engine may still have a process record; clear it.
        self.stop(id, false).await;
        let started = Instant::now();
        self.emit(SupervisorEvent::Starting { model: id.into() });
        let result = match &launch.wrap {
            Some(_) => self.attach(&launch).await,
            None => self.start(&launch).await,
        };
        match &result {
            Ok(()) => {
                let secs = started.elapsed().as_secs_f32();
                if let Some(s) = self.slots().get_mut(id) {
                    s.load_secs = Some(secs);
                    s.restart_at = None;
                    s.last_used.store(self.now_ms(), Ordering::SeqCst);
                }
                self.emit(SupervisorEvent::Ready { model: id.into(), load_secs: secs });
            }
            Err(e) => self.emit(SupervisorEvent::LoadFailed { model: id.into(), why: e.to_string() }),
        }
        result
    }

    /// Drains and stops `id`. A wrapped server is only detached (Ymir doesn't own it).
    pub async fn unload(&self, id: &str) -> Result<(), InferError> {
        let _op = self.inner.ops.lock().await;
        if !self.slots().contains_key(id) {
            return Err(InferError::UnknownModel(id.into()));
        }
        self.stop(id, false).await;
        Ok(())
    }

    /// Stops every engine (daemon shutdown).
    pub async fn shutdown(&self) {
        let _op = self.inner.ops.lock().await;
        let ids: Vec<String> = self.slots().keys().cloned().collect();
        for id in ids {
            self.stop(&id, false).await;
        }
    }

    /// Unloads ready models idle past their `idle_ttl`. Returns what it unloaded. Run it every few
    /// seconds ([`Supervisor::spawn_reaper`]).
    pub async fn reap_idle(&self) -> Vec<String> {
        let now = self.now_ms();
        let idle: Vec<String> = self
            .slots()
            .values()
            .filter(|s| s.state == ModelState::Ready && s.in_flight.load(Ordering::SeqCst) == 0)
            .filter(|s| s.launch.idle_ttl.is_some_and(|ttl| now.saturating_sub(s.last_used.load(Ordering::SeqCst)) >= ttl.as_millis() as u64))
            .map(|s| s.launch.id.clone())
            .collect();
        let mut done = Vec::new();
        for id in idle {
            // Never wait behind a load or swap; try again next round.
            let Ok(_op) = self.inner.ops.try_lock() else { break };
            let still_idle = self.slots().get(&id).is_some_and(|s| s.state == ModelState::Ready && s.in_flight.load(Ordering::SeqCst) == 0);
            if still_idle {
                tracing::info!(model = %id, "idle: unloading");
                self.stop(&id, true).await;
                done.push(id);
            }
        }
        done
    }

    /// Runs [`Supervisor::reap_idle`] every `every` until the returned handle is aborted.
    pub fn spawn_reaper(&self, every: Duration) -> tokio::task::JoinHandle<()> {
        let me = self.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(every);
            loop {
                tick.tick().await;
                me.reap_idle().await;
            }
        })
    }

    /// A wrapped stock server: probe it and mark it ready.
    async fn attach(&self, launch: &ModelLaunch) -> Result<(), InferError> {
        let wrap = launch.wrap.as_ref().expect("attach is for wrapped models");
        wrap.check().map_err(InferError::Unavailable)?;
        let base = wrap.base();
        let ok = self.inner.http.get(format!("{base}/health")).timeout(Duration::from_secs(5)).send().await.is_ok_and(|r| r.status().is_success());
        let (tx, rx) = watch::channel(None);
        let mut slots = self.slots();
        let slot = slots.get_mut(&launch.id).expect("checked by load");
        slot.generation += 1;
        if !ok {
            slot.state = ModelState::Failed(format!("wrapped server {base} is not answering /health"));
            return Err(InferError::Unavailable(format!("{}: wrapped server {base} is not answering /health", launch.id)));
        }
        slot.state = ModelState::Ready;
        slot.running = Some(Running { base, api_key: wrap.api_key.clone(), pid: None, stop: None, exited: rx, _exit_tx: Some(tx) });
        Ok(())
    }

    async fn start(&self, launch: &ModelLaunch) -> Result<(), InferError> {
        let engine = self
            .inner
            .engines
            .get(&launch.engine)
            .ok_or_else(|| InferError::Unavailable(format!("engine {} is not configured on this node", launch.engine.name())))?
            .clone();
        let port = free_port().map_err(|e| InferError::Unavailable(format!("no free loopback port: {e}")))?;
        let api_key = random_hex(24);
        let mut cmd = engine.command(launch, port, &api_key);
        cmd.stdin(Stdio::null());
        match self.log_file(&launch.id) {
            Some((out, err)) => {
                cmd.stdout(out).stderr(err);
            }
            None => {
                cmd.stdout(Stdio::null()).stderr(Stdio::null());
            }
        }
        #[cfg(unix)]
        {
            cmd.process_group(0);
            #[cfg(target_os = "linux")]
            // SAFETY: prctl is async-signal-safe and touches no memory of the parent.
            unsafe {
                cmd.pre_exec(|| {
                    libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
                    Ok(())
                });
            }
        }
        let mut child = cmd.spawn().map_err(|e| InferError::Unavailable(format!("could not start {}: {e}", engine.program.display())))?;
        let pid = child.id();
        let (exit_tx, exit_rx) = watch::channel(None);
        let (stop_tx, stop_rx) = oneshot::channel::<()>();
        let generation = {
            let mut slots = self.slots();
            let slot = slots.get_mut(&launch.id).expect("checked by load");
            slot.generation += 1;
            slot.state = ModelState::Loading;
            slot.running = Some(Running {
                base: format!("http://127.0.0.1:{port}"),
                api_key: Some(api_key),
                pid,
                stop: Some(stop_tx),
                exited: exit_rx.clone(),
                _exit_tx: None,
            });
            slot.generation
        };
        tracing::info!(model = %launch.id, pid, port, engine = launch.engine.name(), "engine starting");

        // The monitor owns the child: it reaps it and does the stopping.
        let grace = self.inner.opts.stop_grace;
        let me = self.clone();
        let id = launch.id.clone();
        tokio::spawn(async move {
            let status = tokio::select! {
                s = child.wait() => s,
                _ = stop_rx => terminate(&mut child, pid, grace).await,
            };
            let desc = match status {
                Ok(s) => format!("engine exited ({s})"),
                Err(e) => format!("engine wait failed: {e}"),
            };
            // Record the exit (and count a crash) before anyone waiting on it wakes up.
            me.exited(&id, generation, desc.clone());
            let _ = exit_tx.send(Some(desc));
        });

        let started = Instant::now();
        let health = format!("http://127.0.0.1:{port}/health");
        let mut exited = exit_rx;
        loop {
            let gone = exited.borrow().clone();
            if let Some(desc) = gone {
                return Err(self.failed(&launch.id, generation, format!("{desc} while loading{}", self.log_tail(&launch.id))));
            }
            if let Ok(resp) = self.inner.http.get(&health).timeout(Duration::from_secs(2)).send().await
                && resp.status().is_success()
            {
                break;
            }
            if started.elapsed() > engine.ready_timeout {
                self.stop(&launch.id, false).await;
                return Err(self.failed(&launch.id, generation, format!("not ready after {:?}{}", engine.ready_timeout, self.log_tail(&launch.id))));
            }
            let _ = tokio::time::timeout(Duration::from_millis(200), exited.changed()).await;
        }
        let mut slots = self.slots();
        let slot = slots.get_mut(&launch.id).expect("slot exists");
        if slot.generation == generation && slot.state == ModelState::Loading {
            slot.state = ModelState::Ready;
            tracing::info!(model = %launch.id, secs = started.elapsed().as_secs_f32(), "engine ready");
            Ok(())
        } else {
            Err(InferError::Unavailable(format!("{} stopped while loading", launch.id)))
        }
    }

    /// The monitor saw the engine exit. A stop sets `Unloaded` before signalling, so anything else
    /// is a crash: count it, and schedule a restart with backoff unless it is a crash loop.
    fn exited(&self, id: &str, generation: u64, desc: String) {
        let opts = &self.inner.opts;
        let (event, restart_in) = {
            let mut slots = self.slots();
            let Some(slot) = slots.get_mut(id).filter(|s| s.generation == generation) else { return };
            slot.running = None;
            if slot.state == ModelState::Unloaded {
                return;
            }
            let was_serving = slot.state == ModelState::Ready;
            let now = Instant::now();
            slot.crashes.retain(|t| now.duration_since(*t) < opts.crash_window);
            slot.crashes.push(now);
            let n = slot.crashes.len();
            tracing::warn!(model = %id, crashes = n, "{desc}");
            if n >= opts.max_crashes {
                slot.crash_loop = true;
                slot.state = ModelState::Failed(format!("crash loop: {n} crashes in {}s ({desc})", opts.crash_window.as_secs()));
                (SupervisorEvent::CrashLoop { model: id.into(), crashes: n }, None)
            } else {
                slot.state = ModelState::Failed(desc.clone());
                // A crash while loading is reported to whoever is loading; only a crash while
                // serving restarts on its own.
                let delay = (opts.restart && was_serving).then(|| backoff(opts.backoff_base, opts.backoff_max, n));
                slot.restart_at = delay.map(|d| now + d);
                (SupervisorEvent::Crashed { model: id.into(), why: desc, crashes: n, restart_in_secs: delay.map(|d| d.as_secs_f32()) }, delay)
            }
        };
        self.emit(event);
        if let Some(delay) = restart_in {
            let me = self.clone();
            let id = id.to_string();
            tokio::spawn(async move {
                tokio::time::sleep(delay).await;
                let due = me.slots().get(&id).is_some_and(|s| {
                    matches!(s.state, ModelState::Failed(_)) && !s.crash_loop && s.restart_at.is_some_and(|at| at <= Instant::now())
                });
                if due {
                    tracing::info!(model = %id, "restarting after crash");
                    if let Some(s) = me.slots().get_mut(&id) {
                        s.restart_at = None;
                    }
                    if let Err(e) = me.load(&id).await {
                        tracing::warn!(model = %id, "restart failed: {e}");
                    }
                }
            });
        }
    }

    fn failed(&self, id: &str, generation: u64, why: String) -> InferError {
        if let Some(slot) = self.slots().get_mut(id).filter(|s| s.generation == generation)
            && !slot.crash_loop
        {
            slot.state = ModelState::Failed(why.clone());
        }
        InferError::Unavailable(format!("{id}: {why}"))
    }

    /// Drains, marks unloaded and stops the engine. Caller holds `ops`.
    async fn stop(&self, id: &str, idle: bool) {
        let in_flight = {
            let mut slots = self.slots();
            let Some(slot) = slots.get_mut(id) else { return };
            if slot.running.is_none() {
                if !matches!(slot.state, ModelState::Failed(_)) {
                    slot.state = ModelState::Unloaded;
                }
                return;
            }
            slot.state = ModelState::Draining;
            slot.in_flight.clone()
        };
        let deadline = Instant::now() + self.inner.opts.drain_timeout;
        while in_flight.load(Ordering::SeqCst) > 0 && Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let (stop, mut exited) = {
            let mut slots = self.slots();
            let Some(slot) = slots.get_mut(id) else { return };
            slot.state = ModelState::Unloaded;
            match slot.running.as_mut() {
                Some(r) if r.stop.is_some() => (r.stop.take(), r.exited.clone()),
                Some(_) => {
                    // Wrapped: detach only.
                    slot.running = None;
                    drop(slots);
                    self.emit(SupervisorEvent::Unloaded { model: id.into(), idle });
                    return;
                }
                None => return,
            }
        };
        if let Some(stop) = stop {
            let _ = stop.send(());
        }
        let wait = self.inner.opts.stop_grace + Duration::from_secs(5);
        let _ = tokio::time::timeout(wait, exited.wait_for(Option::is_some)).await;
        tracing::info!(model = %id, "engine stopped");
        self.emit(SupervisorEvent::Unloaded { model: id.into(), idle });
    }

    /// Where requests for `id` go, with the in-flight guard taken.
    fn target(&self, id: &str) -> Result<(String, Option<String>, InFlight), InferError> {
        let slots = self.slots();
        let slot = slots.get(id).ok_or_else(|| InferError::UnknownModel(id.into()))?;
        match (&slot.state, &slot.running) {
            (ModelState::Ready, Some(r)) => {
                Ok((r.base.clone(), r.api_key.clone(), InFlight::new(slot.in_flight.clone(), slot.last_used.clone(), self.inner.epoch)))
            }
            (state, _) => Err(InferError::Unavailable(format!("{id} is not ready ({state:?})"))),
        }
    }

    async fn post(&self, url: String, key: Option<&str>, body: &Value) -> Result<reqwest::Response, InferError> {
        let mut req = self.inner.http.post(url).json(body);
        if let Some(k) = key {
            req = req.bearer_auth(k);
        }
        let resp = req.send().await.map_err(|e| InferError::Engine(format!("request failed: {e}")))?;
        let status = resp.status();
        if status.as_u16() == 503 || status.as_u16() == 429 {
            return Err(InferError::Busy { retry_after_secs: 1 });
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(match status.as_u16() {
                400 | 422 => InferError::BadRequest(trim(&body)),
                _ => InferError::Engine(format!("{status}: {}", trim(&body))),
            });
        }
        Ok(resp)
    }

    /// Streams a chat answer from `id`, which must be ready. Dropping the stream aborts the engine
    /// request.
    pub async fn chat(&self, id: &str, req: &ChatRequest) -> Result<EventStream, InferError> {
        let (base, key, guard) = self.target(id)?;
        let resp = self.post(format!("{base}/v1/chat/completions"), key.as_deref(), &req.to_openai_stream(id)).await?;
        Ok(stream_events(resp, guard))
    }

    /// Streams a raw completion from `id`.
    pub async fn complete(&self, id: &str, req: &CompletionRequest) -> Result<EventStream, InferError> {
        let (base, key, guard) = self.target(id)?;
        let resp = self.post(format!("{base}/v1/completions"), key.as_deref(), &req.to_openai_stream(id)).await?;
        Ok(stream_events(resp, guard))
    }

    /// Embeds `input` with `id` (an embedding model). Returns the vectors in input order and the
    /// prompt tokens used.
    pub async fn embed(&self, id: &str, input: &[String]) -> Result<(Vec<Vec<f32>>, u64), InferError> {
        let (base, key, _guard) = self.target(id)?;
        let resp =
            self.post(format!("{base}/v1/embeddings"), key.as_deref(), &json!({"model": id, "input": input, "encoding_format": "float"})).await?;
        let v: Value = resp.json().await.map_err(|e| InferError::Engine(format!("bad embeddings response: {e}")))?;
        let mut rows: Vec<(u64, Vec<f32>)> = v["data"]
            .as_array()
            .ok_or_else(|| InferError::Engine("embeddings response has no data".into()))?
            .iter()
            .enumerate()
            .map(|(i, d)| {
                (
                    d["index"].as_u64().unwrap_or(i as u64),
                    d["embedding"].as_array().map(|e| e.iter().filter_map(Value::as_f64).map(|x| x as f32).collect()).unwrap_or_default(),
                )
            })
            .collect();
        rows.sort_by_key(|(i, _)| *i);
        if rows.len() != input.len() {
            return Err(InferError::Engine(format!("asked for {} embeddings, got {}", input.len(), rows.len())));
        }
        Ok((rows.into_iter().map(|(_, e)| e).collect(), v["usage"]["prompt_tokens"].as_u64().unwrap_or(0)))
    }

    fn log_file(&self, id: &str) -> Option<(std::fs::File, std::fs::File)> {
        let dir = self.inner.opts.log_dir.as_ref()?;
        std::fs::create_dir_all(dir).ok()?;
        let f = std::fs::OpenOptions::new().create(true).append(true).open(dir.join(format!("{id}.log"))).ok()?;
        Some((f.try_clone().ok()?, f))
    }

    fn log_tail(&self, id: &str) -> String {
        let Some(dir) = &self.inner.opts.log_dir else { return String::new() };
        let Ok(text) = std::fs::read_to_string(dir.join(format!("{id}.log"))) else { return String::new() };
        let lines: Vec<&str> = text.lines().rev().take(8).collect();
        let tail: Vec<&str> = lines.into_iter().rev().collect();
        format!("; log {}:\n{}", dir.join(format!("{id}.log")).display(), tail.join("\n"))
    }
}

/// `base * 2^(n-1)`, at most `max`.
fn backoff(base: Duration, max: Duration, n: usize) -> Duration {
    let factor = 1u32.checked_shl(n.saturating_sub(1).min(16) as u32).unwrap_or(u32::MAX);
    base.saturating_mul(factor).min(max)
}

fn stream_events(resp: reqwest::Response, guard: InFlight) -> EventStream {
    let (tx, rx) = tokio::sync::mpsc::channel(64);
    tokio::spawn(async move {
        let _guard = guard;
        let mut bytes = resp.bytes_stream();
        let mut parser = SseParser::default();
        let mut decoder = ChunkDecoder::default();
        loop {
            let payloads = match bytes.next().await {
                Some(Ok(chunk)) => parser.feed(&chunk),
                Some(Err(e)) => {
                    let _ = tx.send(Err(InferError::Engine(format!("stream broke: {e}")))).await;
                    return;
                }
                None => {
                    let rest: Vec<String> = parser.finish().into_iter().collect();
                    for p in rest {
                        if !send_all(&tx, decoder.decode(&p)).await {
                            return;
                        }
                    }
                    let last =
                        if decoder.complete() { decoder.finish().map(Ok) } else { Some(Err(InferError::Engine("engine stream ended early".into()))) };
                    if let Some(last) = last {
                        let _ = tx.send(last).await;
                    }
                    return;
                }
            };
            for p in payloads {
                let done = p.trim() == "[DONE]";
                if !send_all(&tx, decoder.decode(&p)).await || done {
                    return;
                }
            }
        }
    });
    Box::pin(ReceiverStream::new(rx))
}

/// Sends decoded frames; false when the stream should end (receiver gone, or an error was sent).
async fn send_all(tx: &tokio::sync::mpsc::Sender<Result<Event, InferError>>, evs: Result<Vec<Event>, InferError>) -> bool {
    match evs {
        Ok(evs) => {
            for ev in evs {
                if tx.send(Ok(ev)).await.is_err() {
                    return false;
                }
            }
            true
        }
        Err(e) => {
            let _ = tx.send(Err(e)).await;
            false
        }
    }
}

/// Counts a request in flight; on drop, records when the model was last used.
struct InFlight {
    n: Arc<AtomicUsize>,
    last_used: Arc<AtomicU64>,
    epoch: Instant,
}

impl InFlight {
    fn new(n: Arc<AtomicUsize>, last_used: Arc<AtomicU64>, epoch: Instant) -> Self {
        n.fetch_add(1, Ordering::SeqCst);
        Self { n, last_used, epoch }
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        self.last_used.store(self.epoch.elapsed().as_millis() as u64, Ordering::SeqCst);
        self.n.fetch_sub(1, Ordering::SeqCst);
    }
}

async fn terminate(child: &mut tokio::process::Child, pid: Option<u32>, grace: Duration) -> std::io::Result<std::process::ExitStatus> {
    #[cfg(unix)]
    if let Some(pid) = pid.and_then(|p| i32::try_from(p).ok()) {
        // SAFETY: plain syscall; the negative pid targets the engine's own process group.
        unsafe {
            libc::kill(-pid, libc::SIGTERM);
        }
        if let Ok(status) = tokio::time::timeout(grace, child.wait()).await {
            return status;
        }
        // SAFETY: as above.
        unsafe {
            libc::kill(-pid, libc::SIGKILL);
        }
        return child.wait().await;
    }
    let _ = (pid, grace);
    child.start_kill()?;
    child.wait().await
}

fn free_port() -> std::io::Result<u16> {
    Ok(std::net::TcpListener::bind(("127.0.0.1", 0))?.local_addr()?.port())
}

/// Random hex from the OS RNG.
pub(crate) fn random_hex(bytes: usize) -> String {
    let mut buf = vec![0u8; bytes];
    getrandom::fill(&mut buf).expect("the OS RNG is available");
    buf.iter().map(|b| format!("{b:02x}")).collect()
}

fn trim(s: &str) -> String {
    let s = s.trim();
    if s.len() > 500 { format!("{}...", &s[..s.floor_char_boundary(500)]) } else { s.to_string() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_doubles_and_caps() {
        let b = |n| backoff(Duration::from_secs(1), Duration::from_secs(60), n);
        assert_eq!(b(1), Duration::from_secs(1));
        assert_eq!(b(2), Duration::from_secs(2));
        assert_eq!(b(4), Duration::from_secs(8));
        assert_eq!(b(10), Duration::from_secs(60));
        assert_eq!(b(100), Duration::from_secs(60));
    }
}
