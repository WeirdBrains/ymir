//! The model role, and the seam the inference role calls it through.

use std::collections::HashMap;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use weirdnet_agent::remote::RemoteError;
use weirdnet_agent::remote::infer::v2::{Input, Request};
use weirdnet_backends::error::unavailable;
use weirdnet_backends::node::{GroupPlan, GroupWorkers};
use weirdnet_backends::{ChunkStream, EngineKind, ModelHost, ModelStatus, Supervisor};

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Embedding vectors in input order, and the prompt tokens used.
pub type Embedded = (Vec<Vec<f32>>, u64);

/// A transcript and the token counts the engine reported, if any.
pub type Transcribed = (weirdnet_agent::remote::infer::v2::Transcript, Option<weirdnet_agent::remote::infer::v2::Usage>);

/// Whether a model has the transcription role here and its engine can serve it
/// (`weirdnet-backends` 0.4.9 `Capabilities::transcribe`); records from older nodes say no.
pub fn transcribes(s: &ModelStatus) -> bool {
    s.capabilities.is_some_and(|c| c.transcribe)
}

/// Whether a model has the speech role here and its engine can serve it (`weirdnet-backends`
/// 0.4.10 `Capabilities::speak`); records from older nodes say no.
pub fn speaks(s: &ModelStatus) -> bool {
    s.capabilities.is_some_and(|c| c.speak)
}

/// The context one request gets: what the engine reported at load (`weirdnet-backends` 0.4.9
/// `ModelStatus::context_tokens`, llama-server `/props` or vLLM `max_model_len`), else the
/// configured context split over the slots ([`request_context`]); `None` when neither is known.
pub fn status_context(s: &ModelStatus) -> Option<u32> {
    s.context_tokens.filter(|c| *c > 0).or_else(|| request_context(s.engine, s.context, s.slots))
}

/// What an inference node asks of a model node. [`LocalModelNode`] answers in process;
/// [`crate::remote::RemoteModelNode`] speaks `/weirdnet-backends/node/1` (capability records,
/// `Load`, `Unload`) and raw `/weirdnet-agent/infer/2` to a model node on another device.
/// Whether a model only embeds: from its capabilities when the node reports them
/// (`weirdnet-backends` 0.1.3 and later), else from the older `embedding` flag.
pub fn embeds(s: &ModelStatus) -> bool {
    s.capabilities.map_or(s.embedding, |c| c.embeddings && !c.chat)
}

/// The inputs beyond text a model takes, from its capabilities when the node reports them
/// (`weirdnet-backends` 0.1.3 and later); a node that doesn't is taken to read text only.
pub fn inputs_of(s: &ModelStatus) -> weirdnet_agent::Modalities {
    s.capabilities.map_or(weirdnet_agent::Modalities::TEXT, |c| c.inputs())
}

/// The context window one request gets, in tokens: the configured context, split over the
/// parallel slots for llama.cpp (it divides `--ctx-size` between them unless the KV cache is
/// unified, so the split is the most a request can count on). `None` when no context is set:
/// Ymir reports a limit it knows and never fills in one it doesn't (video design, milestone 1).
pub fn request_context(engine: EngineKind, context: Option<u32>, slots: Option<u32>) -> Option<u32> {
    let ctx = context.filter(|c| *c > 0)?;
    let slots = if matches!(engine, EngineKind::Llamacpp) { slots.unwrap_or(1).max(1) } else { 1 };
    Some(ctx / slots)
}

/// The known media and request limits in a few words (`8 images, 256 tokens per image, edge
/// 896 px`), or `None` when none are known; `ymirctl status` shows it.
pub fn limits_line(l: &weirdnet_agent::MediaLimits) -> Option<String> {
    let mut parts = Vec::new();
    if let Some(n) = l.max_images {
        parts.push(format!("{n} image{}", if n == 1 { "" } else { "s" }));
    }
    if let Some(t) = l.image_tokens {
        parts.push(format!("{t} tokens per image{}", if l.image_patch.is_some() { " at most" } else { "" }));
    }
    if let Some(p) = l.image_patch {
        parts.push(format!("{p} px patches"));
    }
    if let Some(e) = l.image_edge {
        parts.push(format!("edge {e} px"));
    }
    if let Some(s) = l.audio_max_seconds {
        parts.push(format!("audio up to {s} s"));
    }
    if let Some(t) = l.audio_tokens_per_minute {
        parts.push(format!("{t} audio tokens per minute"));
    }
    if let Some(s) = l.video_max_seconds {
        parts.push(format!("video up to {s} s"));
    }
    if let Some(b) = l.video_max_bytes {
        parts.push(format!("video up to {b} bytes"));
    }
    if let Some(f) = l.video_frames_per_minute {
        parts.push(format!("{f} video frames per minute"));
    }
    if let Some(b) = l.max_request_bytes {
        parts.push(format!("requests up to {b} bytes"));
    }
    (!parts.is_empty()).then(|| parts.join(", "))
}

/// A model's origin (`hub`, `upload`, `train`) and license, either unknown.
pub type Origin = (Option<String>, Option<String>);

pub trait ModelNode: Send + Sync {
    fn name(&self) -> &str;
    /// The node's models and their states (its capability record).
    fn status(&self) -> Vec<ModelStatus>;
    /// Loads `model`. `force` is an operator's explicit load: it also clears a crash loop.
    fn load<'a>(&'a self, model: &'a str, force: bool) -> BoxFuture<'a, Result<(), RemoteError>>;
    fn unload<'a>(&'a self, model: &'a str) -> BoxFuture<'a, Result<(), RemoteError>>;
    /// Streams an answer from loaded `model`: fully assembled messages in, chunks out.
    fn chat<'a>(&'a self, model: &'a str, req: &'a Request) -> BoxFuture<'a, Result<ChunkStream, RemoteError>>;
    /// Embeds `input` with loaded embedding `model`.
    fn embed<'a>(&'a self, model: &'a str, input: &'a [String]) -> BoxFuture<'a, Result<Embedded, RemoteError>>;
    /// Transcribes `request.audio` with loaded speech-to-text `model` (video design, milestone 2).
    /// The audio and the transcript are content: never logged, never cached.
    fn transcribe<'a>(
        &'a self,
        model: &'a str,
        request: &'a weirdnet_agent::remote::infer::v2::TranscribeRequest,
    ) -> BoxFuture<'a, Result<Transcribed, RemoteError>> {
        let _ = (model, request);
        Box::pin(async { Err(RemoteError::Unsupported { message: "this node serves no transcription".into() }) })
    }
    /// Speaks `request.input` with loaded text-to-speech `model` (audio design milestone 7). The
    /// text, any reference clip and the audio are content: never logged, never cached.
    fn speak<'a>(
        &'a self,
        model: &'a str,
        request: &'a weirdnet_agent::remote::infer::v2::SpeakRequest,
    ) -> BoxFuture<'a, Result<weirdnet_agent::remote::infer::v2::AudioStream, RemoteError>> {
        let _ = (model, request);
        Box::pin(async { Err(RemoteError::Unsupported { message: "this node serves no speech".into() }) })
    }
    /// The weights revision of `model` here (a Hub commit, a file hash, a fine-tune id), when
    /// known. The answer cache keys on it, so new weights under the same id don't replay old answers.
    fn revision(&self, _model: &str) -> Option<String> {
        None
    }
    /// Where `model` here came from (`hub`, `upload`, `train`) and its license, when known; the
    /// pool catalog shows them (U8).
    fn origin(&self, _model: &str) -> Origin {
        (None, None)
    }
    /// Whether the node is draining: it finishes what it has and takes no new requests. A
    /// reserved node ([`ModelNode::reserved`]) counts as draining, so nothing is routed or
    /// placed there.
    fn draining(&self) -> bool {
        false
    }
    /// Who holds the node's GPU, when it is reserved (training plan P3a): it runs no model until
    /// released.
    fn reserved(&self) -> Option<String> {
        None
    }
    /// Reserves the node's GPU for `holder` (`weirdnet-backends` 0.4.3 `Reserve`): its engines
    /// stop and none starts until [`ModelNode::release`].
    fn reserve<'a>(&'a self, _holder: &'a str) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(async { Err(RemoteError::Unsupported { message: "this node can't be reserved".into() }) })
    }
    fn release<'a>(&'a self, _holder: &'a str) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(async { Err(RemoteError::Unsupported { message: "this node can't be reserved".into() }) })
    }
    /// Whether a request carrying an `assistant_prefix` (an answer to continue) works here.
    fn resumes(&self) -> bool {
        true
    }
    /// Whether the node runs on this device (it wins ties when balancing).
    fn local(&self) -> bool {
        false
    }
    /// Whether the node takes group jobs (`GroupUp`): it can run an `rpc-server` or lead one.
    fn group_jobs(&self) -> bool {
        false
    }
    /// Joins the group `plan` describes: unloads `plan.drop`, then starts this node's worker (or,
    /// for the leader, the serving process).
    fn group_up<'a>(&'a self, _plan: &'a GroupPlan) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(async { Err(RemoteError::Unsupported { message: "this node takes no group jobs".into() }) })
    }
    fn group_down<'a>(&'a self, _group: &'a str) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(async { Err(RemoteError::Unsupported { message: "this node takes no group jobs".into() }) })
    }
    /// Whether this node's processes of `group` still run. A remote node can't say (node/1 has
    /// no such question), so it is taken to be fine while it stays in the pool.
    fn group_alive<'a>(&'a self, _group: &'a str) -> BoxFuture<'a, bool> {
        Box::pin(async { true })
    }
    /// Starts draining (waiting for requests under way) or resumes.
    fn drain<'a>(&'a self, _on: bool) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(async { Err(RemoteError::Unsupported { message: "this node can't drain".into() }) })
    }
}

/// The model role on this machine: a thin wrapper over the supervisor.
pub struct LocalModelNode {
    name: String,
    supervisor: Supervisor,
    sources: Vec<(String, EngineKind, String, bool)>,
    draining: AtomicBool,
    /// The model-node service on Weirdnet, when this box serves one: draining here drains it too.
    service_host: Mutex<Option<Arc<ModelHost>>>,
    /// Starts this node's part of a group, when it takes group jobs.
    workers: Mutex<Option<Arc<crate::group::RpcWorkers>>>,
    /// Weights revisions from node.toml (`revision` on `[[model.local]]`).
    revisions: Mutex<HashMap<String, String>>,
    /// Origins and licenses (from node.toml and provenance records, and models placed since).
    origins: Mutex<HashMap<String, Origin>>,
    /// Sizes a model's context beside what runs before it loads.
    planner: Mutex<Option<Arc<crate::ctxplan::Planner>>>,
}

/// How long a drain waits for requests under way.
const DRAIN_WAIT: Duration = Duration::from_secs(60);

impl LocalModelNode {
    pub fn new(name: impl Into<String>, supervisor: Supervisor, launches: &[weirdnet_backends::ModelLaunch]) -> Self {
        Self {
            name: name.into(),
            supervisor,
            sources: launches.iter().map(|l| (l.id.clone(), l.engine, l.source.clone(), l.wrap.is_some())).collect(),
            draining: AtomicBool::new(false),
            service_host: Mutex::new(None),
            workers: Mutex::new(None),
            revisions: Mutex::new(HashMap::new()),
            origins: Mutex::new(HashMap::new()),
            planner: Mutex::new(None),
        }
    }

    /// Has loads made room for the model and size its context first ([`crate::ctxplan`]).
    pub fn set_planner(&self, planner: Arc<crate::ctxplan::Planner>) {
        *self.planner.lock().unwrap_or_else(|e| e.into_inner()) = Some(planner);
    }

    fn planner(&self) -> Option<Arc<crate::ctxplan::Planner>> {
        self.planner.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Sets the weights revision of each model that has one (see [`ModelNode::revision`]).
    pub fn set_revisions(&self, revisions: impl IntoIterator<Item = (String, String)>) {
        *self.revisions.lock().unwrap_or_else(|e| e.into_inner()) = revisions.into_iter().collect();
    }

    /// Sets a model's origin and license (see [`ModelNode::origin`]); both `None` forgets them.
    pub fn set_origin(&self, model: &str, origin: Option<String>, license: Option<String>) {
        let mut o = self.origins.lock().unwrap_or_else(|e| e.into_inner());
        if origin.is_none() && license.is_none() {
            o.remove(model);
        } else {
            o.insert(model.to_string(), (origin, license));
        }
    }

    /// Lets this node join split-model groups (M7).
    pub fn set_group_workers(&self, workers: Arc<crate::group::RpcWorkers>) {
        *self.workers.lock().unwrap_or_else(|e| e.into_inner()) = Some(workers);
    }

    /// Ties this node's drain state to the host that serves the model-node protocol (the
    /// service's `Drain` and this node's `drain` then agree).
    pub fn attach_host(&self, host: Arc<ModelHost>) {
        host.set_draining(self.draining.load(Ordering::SeqCst));
        *self.service_host.lock().unwrap_or_else(|e| e.into_inner()) = Some(host);
    }

    pub fn supervisor(&self) -> &Supervisor {
        &self.supervisor
    }

    /// A clear message for the common setup mistake: the GGUF isn't there.
    fn check_file(&self, model: &str) -> Result<(), RemoteError> {
        match self.sources.iter().find(|(id, ..)| id == model) {
            Some((_, EngineKind::Mlx, dir, false)) if Path::new(dir).is_absolute() && !Path::new(dir).is_dir() => Err(unavailable(format!(
                "{model}: MLX weights {dir} are missing (download them with `ymirctl setup --download`, or set `file` in node.toml)"
            ))),
            Some((_, EngineKind::Llamacpp, f, false)) if !Path::new(f).is_file() => Err(unavailable(format!(
                "{model}: model file {f} is missing (download it with `ymirctl setup --download`, or set `file` in node.toml)"
            ))),
            _ => Ok(()),
        }
    }
}

impl ModelNode for LocalModelNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn status(&self) -> Vec<ModelStatus> {
        self.supervisor.status()
    }

    fn revision(&self, model: &str) -> Option<String> {
        self.revisions.lock().unwrap_or_else(|e| e.into_inner()).get(model).cloned()
    }

    fn origin(&self, model: &str) -> Origin {
        self.origins.lock().unwrap_or_else(|e| e.into_inner()).get(model).cloned().unwrap_or_default()
    }

    fn load<'a>(&'a self, model: &'a str, force: bool) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(async move {
            self.check_file(model)?;
            match self.planner() {
                Some(p) => p.load(model, force).await,
                None => self.supervisor.load_with(model, force).await,
            }
        })
    }

    fn unload<'a>(&'a self, model: &'a str) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(self.supervisor.unload(model))
    }

    fn chat<'a>(&'a self, model: &'a str, req: &'a Request) -> BoxFuture<'a, Result<ChunkStream, RemoteError>> {
        Box::pin(async move {
            if let Some(p) = self.planner() {
                p.touch(model);
            }
            // A raw prompt (`/v1/completions`) goes to the engine's completions route, with the
            // assistant prefix (a resume) appended to the prompt.
            if let Input::Prompt(prompt) = &req.input {
                let mut prompt = prompt.clone();
                if let Some(p) = &req.assistant_prefix {
                    prompt.push_str(&p.text);
                }
                return self.supervisor.complete(model, &prompt, &req.sampling, req.seed).await;
            }
            self.supervisor.chat(model, req).await
        })
    }

    fn embed<'a>(&'a self, model: &'a str, input: &'a [String]) -> BoxFuture<'a, Result<Embedded, RemoteError>> {
        if let Some(p) = self.planner() {
            p.touch(model);
        }
        Box::pin(self.supervisor.embed(model, input))
    }

    fn transcribe<'a>(
        &'a self,
        model: &'a str,
        request: &'a weirdnet_agent::remote::infer::v2::TranscribeRequest,
    ) -> BoxFuture<'a, Result<Transcribed, RemoteError>> {
        if let Some(p) = self.planner() {
            p.touch(model);
        }
        Box::pin(self.supervisor.transcribe(model, request))
    }

    fn speak<'a>(
        &'a self,
        model: &'a str,
        request: &'a weirdnet_agent::remote::infer::v2::SpeakRequest,
    ) -> BoxFuture<'a, Result<weirdnet_agent::remote::infer::v2::AudioStream, RemoteError>> {
        if let Some(p) = self.planner() {
            p.touch(model);
        }
        Box::pin(self.supervisor.speak(model, request))
    }

    fn draining(&self) -> bool {
        self.draining.load(Ordering::SeqCst) || self.supervisor.reservation().is_some()
    }

    fn reserved(&self) -> Option<String> {
        self.supervisor.reservation().map(|r| r.holder)
    }

    fn reserve<'a>(&'a self, holder: &'a str) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(async move { self.supervisor.reserve(holder).await.map(|_| ()) })
    }

    fn release<'a>(&'a self, holder: &'a str) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(async move { self.supervisor.release(holder).map(|_| ()) })
    }

    fn local(&self) -> bool {
        true
    }

    fn group_jobs(&self) -> bool {
        self.workers.lock().unwrap_or_else(|e| e.into_inner()).is_some()
    }

    fn group_up<'a>(&'a self, plan: &'a GroupPlan) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(async move {
            let workers = self.workers.lock().unwrap_or_else(|e| e.into_inner()).clone();
            let workers = workers.ok_or_else(|| RemoteError::Unsupported { message: "this node takes no group jobs".into() })?;
            for model in &plan.drop {
                self.supervisor.unload(model).await?;
            }
            workers.up(&self.name, plan).await
        })
    }

    fn group_alive<'a>(&'a self, group: &'a str) -> BoxFuture<'a, bool> {
        Box::pin(async move {
            let workers = self.workers.lock().unwrap_or_else(|e| e.into_inner()).clone();
            match workers {
                Some(w) => w.alive(group).await,
                None => true,
            }
        })
    }

    fn group_down<'a>(&'a self, group: &'a str) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(async move {
            let workers = self.workers.lock().unwrap_or_else(|e| e.into_inner()).clone();
            let workers = workers.ok_or_else(|| RemoteError::Unsupported { message: "this node takes no group jobs".into() })?;
            workers.down(group).await
        })
    }

    fn drain<'a>(&'a self, on: bool) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(async move {
            self.draining.store(on, Ordering::SeqCst);
            if let Some(h) = self.service_host.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
                h.set_draining(on);
            }
            if on {
                let deadline = Instant::now() + DRAIN_WAIT;
                while self.supervisor.in_flight() > 0 {
                    if Instant::now() >= deadline {
                        return Err(RemoteError::Timeout);
                    }
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
            }
            Ok(())
        })
    }
}
