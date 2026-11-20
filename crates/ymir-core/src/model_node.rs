//! The model role, and the seam the inference role calls it through.

use std::future::Future;
use std::path::Path;
use std::pin::Pin;

use weirdnet_agent::remote::RemoteError;
use weirdnet_agent::remote::infer::v2::{Input, Request};
use weirdnet_backends::error::unavailable;
use weirdnet_backends::{ChunkStream, EngineKind, ModelStatus, Supervisor};

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Embedding vectors in input order, and the prompt tokens used.
pub type Embedded = (Vec<Vec<f32>>, u64);

/// What an inference node asks of a model node. [`LocalModelNode`] answers in process;
/// [`crate::remote::RemoteModelNode`] speaks `/weirdnet-backends/node/1` (capability records,
/// `Load`, `Unload`) and raw `/weirdnet-agent/infer/2` to a model node on another device.
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
}

/// The model role on this machine: a thin wrapper over the supervisor.
pub struct LocalModelNode {
    name: String,
    supervisor: Supervisor,
    sources: Vec<(String, EngineKind, String, bool)>,
}

impl LocalModelNode {
    pub fn new(name: impl Into<String>, supervisor: Supervisor, launches: &[weirdnet_backends::ModelLaunch]) -> Self {
        Self {
            name: name.into(),
            supervisor,
            sources: launches.iter().map(|l| (l.id.clone(), l.engine, l.source.clone(), l.wrap.is_some())).collect(),
        }
    }

    pub fn supervisor(&self) -> &Supervisor {
        &self.supervisor
    }

    /// A clear message for the common setup mistake: the GGUF isn't there.
    fn check_file(&self, model: &str) -> Result<(), RemoteError> {
        match self.sources.iter().find(|(id, ..)| id == model) {
            Some((_, EngineKind::Llamacpp, src, false)) if !Path::new(src).is_file() => Err(unavailable(format!(
                "{model}: model file {src} is missing (download it with `ymirctl setup --download`, or set `file` in node.toml)"
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

    fn load<'a>(&'a self, model: &'a str, force: bool) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(async move {
            self.check_file(model)?;
            self.supervisor.load_with(model, force).await
        })
    }

    fn unload<'a>(&'a self, model: &'a str) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(self.supervisor.unload(model))
    }

    fn chat<'a>(&'a self, model: &'a str, req: &'a Request) -> BoxFuture<'a, Result<ChunkStream, RemoteError>> {
        Box::pin(async move {
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
        Box::pin(self.supervisor.embed(model, input))
    }
}
