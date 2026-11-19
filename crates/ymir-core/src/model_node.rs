//! The model role, and the seam the inference role calls it through.

use std::path::Path;

use ymir_backends::{EngineKind, ModelStatus, Supervisor};
use ymir_infer::{BoxFuture, ChatRequest, EventStream, InferError};

/// What an inference node can ask of a model node. Trait seam: in M1 [`LocalModelNode`] answers
/// in process; in M3 a remote implementation speaks `/weirdnet-backends/node/1` (capabilities,
/// `Load`, `Unload`) and raw `/weirdnet-agent/infer/2` for `chat`.
pub trait ModelNode: Send + Sync {
    fn name(&self) -> &str;
    /// The node's models and their states (its capability record).
    fn status(&self) -> Vec<ModelStatus>;
    fn load<'a>(&'a self, model: &'a str) -> BoxFuture<'a, Result<(), InferError>>;
    fn unload<'a>(&'a self, model: &'a str) -> BoxFuture<'a, Result<(), InferError>>;
    /// Streams an answer from a loaded model: fully assembled messages in, tokens out.
    fn chat<'a>(&'a self, model: &'a str, req: &'a ChatRequest) -> BoxFuture<'a, Result<EventStream, InferError>>;
}

/// The model role on this machine: a thin wrapper over the supervisor.
pub struct LocalModelNode {
    name: String,
    supervisor: Supervisor,
    sources: Vec<(String, EngineKind, String)>,
}

impl LocalModelNode {
    pub fn new(name: impl Into<String>, supervisor: Supervisor, launches: &[ymir_backends::ModelLaunch]) -> Self {
        Self { name: name.into(), supervisor, sources: launches.iter().map(|l| (l.id.clone(), l.engine, l.source.clone())).collect() }
    }

    pub fn supervisor(&self) -> &Supervisor {
        &self.supervisor
    }

    /// A clear message for the common setup mistake: the GGUF isn't there.
    fn check_file(&self, model: &str) -> Result<(), InferError> {
        match self.sources.iter().find(|(id, _, _)| id == model) {
            Some((_, EngineKind::Llamacpp, src)) if !Path::new(src).is_file() => Err(InferError::Unavailable(format!(
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

    fn load<'a>(&'a self, model: &'a str) -> BoxFuture<'a, Result<(), InferError>> {
        Box::pin(async move {
            self.check_file(model)?;
            self.supervisor.load(model).await
        })
    }

    fn unload<'a>(&'a self, model: &'a str) -> BoxFuture<'a, Result<(), InferError>> {
        Box::pin(self.supervisor.unload(model))
    }

    fn chat<'a>(&'a self, model: &'a str, req: &'a ChatRequest) -> BoxFuture<'a, Result<EventStream, InferError>> {
        Box::pin(self.supervisor.chat(model, req))
    }
}
