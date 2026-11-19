//! Starts, probes, drains and stops engine processes for one model node.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::{oneshot, watch};
use tokio_stream::wrappers::ReceiverStream;
use ymir_infer::{ChatRequest, EventStream, InferError, ModelState};

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
}

impl Default for SupervisorOptions {
    fn default() -> Self {
        Self { log_dir: None, drain_timeout: Duration::from_secs(30), stop_grace: Duration::from_secs(10) }
    }
}

/// One model's row in the node's capability record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelStatus {
    pub id: String,
    pub engine: EngineKind,
    pub state: ModelState,
    pub in_flight: usize,
    pub swap_group: Option<String>,
    pub embedding: bool,
    pub pid: Option<u32>,
}

struct Running {
    port: u16,
    api_key: String,
    pid: Option<u32>,
    stop: Option<oneshot::Sender<()>>,
    exited: watch::Receiver<Option<String>>,
}

struct Slot {
    launch: ModelLaunch,
    state: ModelState,
    running: Option<Running>,
    in_flight: Arc<AtomicUsize>,
    /// Bumped on every start, so a late exit notice from an old process is ignored.
    generation: u64,
}

struct Inner {
    engines: HashMap<EngineKind, EngineConfig>,
    opts: SupervisorOptions,
    http: reqwest::Client,
    slots: Mutex<HashMap<String, Slot>>,
    /// Serializes load and unload, so swaps never interleave.
    ops: tokio::sync::Mutex<()>,
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
                (
                    launch.id.clone(),
                    Slot { launch, state: ModelState::Unloaded, running: None, in_flight: Arc::new(AtomicUsize::new(0)), generation: 0 },
                )
            })
            .collect();
        let http =
            reqwest::Client::builder().connect_timeout(Duration::from_secs(5)).no_proxy().build().expect("an HTTP client without TLS always builds");
        Self {
            inner: Arc::new(Inner {
                engines: engines.into_iter().map(|e| (e.kind, e)).collect(),
                opts,
                http,
                slots: Mutex::new(slots),
                ops: tokio::sync::Mutex::new(()),
            }),
        }
    }

    fn slots(&self) -> std::sync::MutexGuard<'_, HashMap<String, Slot>> {
        self.inner.slots.lock().unwrap_or_else(|e| e.into_inner())
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
            })
            .collect();
        out.sort_by(|a, b| a.id.cmp(&b.id));
        out
    }

    pub fn state(&self, id: &str) -> Option<ModelState> {
        self.slots().get(id).map(|s| s.state.clone())
    }

    /// Starts `id` and waits until its engine is ready. Loading a model in a swap group unloads the
    /// group's other models first. Already ready: returns at once.
    pub async fn load(&self, id: &str) -> Result<(), InferError> {
        let _op = self.inner.ops.lock().await;
        let (launch, group_mates) = {
            let slots = self.slots();
            let slot = slots.get(id).ok_or_else(|| InferError::UnknownModel(id.into()))?;
            if slot.state == ModelState::Ready {
                return Ok(());
            }
            let mates: Vec<String> = match &slot.launch.swap_group {
                Some(g) => slots
                    .values()
                    .filter(|s| s.launch.id != id && s.launch.swap_group.as_ref() == Some(g) && s.running.is_some())
                    .map(|s| s.launch.id.clone())
                    .collect(),
                None => Vec::new(),
            };
            (slot.launch.clone(), mates)
        };
        for mate in group_mates {
            tracing::info!(model = %mate, for_model = %id, "swap group: unloading");
            self.stop(&mate).await;
        }
        // A failed engine may still have a process record; clear it.
        self.stop(id).await;
        self.start(&launch).await
    }

    /// Drains and stops `id`.
    pub async fn unload(&self, id: &str) -> Result<(), InferError> {
        let _op = self.inner.ops.lock().await;
        if !self.slots().contains_key(id) {
            return Err(InferError::UnknownModel(id.into()));
        }
        self.stop(id).await;
        Ok(())
    }

    /// Stops every engine (daemon shutdown).
    pub async fn shutdown(&self) {
        let _op = self.inner.ops.lock().await;
        let ids: Vec<String> = self.slots().keys().cloned().collect();
        for id in ids {
            self.stop(&id).await;
        }
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
            slot.running = Some(Running { port, api_key: api_key.clone(), pid, stop: Some(stop_tx), exited: exit_rx.clone() });
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
                _ = stop_rx => {
                    terminate(&mut child, pid, grace).await
                }
            };
            let desc = match status {
                Ok(s) => format!("engine exited ({s})"),
                Err(e) => format!("engine wait failed: {e}"),
            };
            let _ = exit_tx.send(Some(desc.clone()));
            let mut slots = me.slots();
            if let Some(slot) = slots.get_mut(&id).filter(|s| s.generation == generation) {
                // A stop sets Unloaded before signalling; anything else is a crash.
                if slot.state != ModelState::Unloaded {
                    tracing::warn!(model = %id, "{desc}");
                    slot.state = ModelState::Failed(desc);
                }
                slot.running = None;
            }
        });

        let started = Instant::now();
        let health = format!("http://127.0.0.1:{port}/health");
        let mut exited = exit_rx;
        loop {
            if let Some(desc) = exited.borrow().clone() {
                return Err(self.failed(&launch.id, generation, format!("{desc} while loading{}", self.log_tail(&launch.id))));
            }
            if let Ok(resp) = self.inner.http.get(&health).timeout(Duration::from_secs(2)).send().await
                && resp.status().is_success()
            {
                break;
            }
            if started.elapsed() > engine.ready_timeout {
                self.stop(&launch.id).await;
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

    fn failed(&self, id: &str, generation: u64, why: String) -> InferError {
        if let Some(slot) = self.slots().get_mut(id).filter(|s| s.generation == generation) {
            slot.state = ModelState::Failed(why.clone());
        }
        InferError::Unavailable(format!("{id}: {why}"))
    }

    /// Drains, marks unloaded and stops the engine. Caller holds `ops`.
    async fn stop(&self, id: &str) {
        let in_flight = {
            let mut slots = self.slots();
            let Some(slot) = slots.get_mut(id) else { return };
            if slot.running.is_none() {
                if matches!(slot.state, ModelState::Failed(_)) {
                    return;
                }
                slot.state = ModelState::Unloaded;
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
                Some(r) => (r.stop.take(), r.exited.clone()),
                None => return,
            }
        };
        if let Some(stop) = stop {
            let _ = stop.send(());
        }
        let wait = self.inner.opts.stop_grace + Duration::from_secs(5);
        let _ = tokio::time::timeout(wait, exited.wait_for(Option::is_some)).await;
        tracing::info!(model = %id, "engine stopped");
    }

    /// Streams a chat answer from `id`, which must be ready. Dropping the stream aborts the engine
    /// request.
    pub async fn chat(&self, id: &str, req: &ChatRequest) -> Result<EventStream, InferError> {
        let (port, key, guard) = {
            let slots = self.slots();
            let slot = slots.get(id).ok_or_else(|| InferError::UnknownModel(id.into()))?;
            match (&slot.state, &slot.running) {
                (ModelState::Ready, Some(r)) => (r.port, r.api_key.clone(), InFlight::new(slot.in_flight.clone())),
                (state, _) => return Err(InferError::Unavailable(format!("{id} is not ready ({state:?})"))),
            }
        };
        let resp = self
            .inner
            .http
            .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .bearer_auth(key)
            .json(&req.to_openai_stream(id))
            .send()
            .await
            .map_err(|e| InferError::Engine(format!("request failed: {e}")))?;
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
                        let ok = decoder.complete();
                        for p in rest {
                            if !send_all(&tx, decoder.decode(&p)).await {
                                return;
                            }
                        }
                        let last = if ok || decoder.complete() {
                            decoder.finish().map(Ok)
                        } else {
                            Some(Err(InferError::Engine("engine stream ended early".into())))
                        };
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
        Ok(Box::pin(ReceiverStream::new(rx)))
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

/// Sends decoded frames; false when the stream should end (receiver gone, or an error was sent).
async fn send_all(tx: &tokio::sync::mpsc::Sender<Result<ymir_infer::Event, InferError>>, evs: Result<Vec<ymir_infer::Event>, InferError>) -> bool {
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

struct InFlight(Arc<AtomicUsize>);

impl InFlight {
    fn new(n: Arc<AtomicUsize>) -> Self {
        n.fetch_add(1, Ordering::SeqCst);
        Self(n)
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
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
