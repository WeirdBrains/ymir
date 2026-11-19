//! Inference frames and the pool seam shared by Ymir's roles.
//!
//! To be extracted into `weirdnet-agent`: [`Event`] mirrors the `/weirdnet-agent/infer/2` frames,
//! [`Pool`] is the app-side `pool` call surface (and the inference side's serve host trait).

use std::future::Future;
use std::pin::Pin;

use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
pub type EventStream = Pin<Box<dyn Stream<Item = Result<Event, InferError>> + Send>>;

/// Scheduling class (infer/2 `priority`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Priority {
    #[default]
    Interactive,
    Batch,
}

/// One chat request: fully assembled messages plus generation parameters.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    /// A pool model id; `None` means the node's main model.
    pub model: Option<String>,
    /// OpenAI chat messages, passed through unchanged.
    pub messages: Vec<Value>,
    /// Other OpenAI parameters (`temperature`, `max_tokens`, `tools`, ...), passed through.
    pub params: Map<String, Value>,
    #[serde(default)]
    pub priority: Priority,
    #[serde(default)]
    pub job_id: Option<String>,
    /// Wait for the requested model even if it is cold, instead of a stand-in answering
    /// (`wait_for_model: true` in the body, or the bridge's `x-wait: 1` header).
    #[serde(default)]
    pub wait_for_model: bool,
}

/// Keys the bridge and engines handle themselves rather than passing through.
const OWN_KEYS: &[&str] = &["model", "messages", "prompt", "input", "stream", "stream_options", "n", "wait_for_model"];

fn take_common(obj: &mut Map<String, Value>) -> Result<(Option<String>, bool, bool), InferError> {
    if obj.get("n").and_then(Value::as_u64).is_some_and(|n| n != 1) {
        return Err(InferError::BadRequest("only n = 1 is supported".into()));
    }
    let model = obj.get("model").and_then(|m| m.as_str().map(str::to_string)).filter(|m| !m.is_empty());
    let stream = obj.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let wait = obj.get("wait_for_model").and_then(Value::as_bool).unwrap_or(false);
    for k in OWN_KEYS {
        obj.remove(*k);
    }
    Ok((model, stream, wait))
}

impl ChatRequest {
    /// From an OpenAI `/v1/chat/completions` body. Returns the request and whether it asked to
    /// stream.
    pub fn from_openai(body: Value) -> Result<(Self, bool), InferError> {
        let Value::Object(mut obj) = body else { return Err(InferError::BadRequest("body must be a JSON object".into())) };
        let messages = match obj.remove("messages") {
            Some(Value::Array(m)) if !m.is_empty() => m,
            _ => return Err(InferError::BadRequest("`messages` must be a non-empty array".into())),
        };
        let (model, stream, wait_for_model) = take_common(&mut obj)?;
        Ok((Self { model, messages, params: obj, priority: Priority::Interactive, job_id: None, wait_for_model }, stream))
    }

    /// A one-message request.
    pub fn user(model: Option<&str>, text: &str) -> Self {
        Self { model: model.map(str::to_string), messages: vec![serde_json::json!({"role": "user", "content": text})], ..Default::default() }
    }

    /// The streaming OpenAI body an engine gets, with `model` set to the engine's alias.
    pub fn to_openai_stream(&self, model: &str) -> Value {
        let mut obj = self.params.clone();
        obj.insert("model".into(), model.into());
        obj.insert("messages".into(), Value::Array(self.messages.clone()));
        obj.insert("stream".into(), true.into());
        obj.insert("stream_options".into(), serde_json::json!({"include_usage": true}));
        Value::Object(obj)
    }

    /// Deterministic requests (temperature 0, no tools) are the only ones an exact cache may answer.
    pub fn deterministic(&self) -> bool {
        self.params.get("temperature").and_then(Value::as_f64) == Some(0.0) && !self.params.contains_key("tools")
    }
}

/// A raw text completion (`/v1/completions`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: String,
    pub params: Map<String, Value>,
    #[serde(default)]
    pub wait_for_model: bool,
}

impl CompletionRequest {
    /// From an OpenAI `/v1/completions` body: `prompt` is a string or a one-string array.
    pub fn from_openai(body: Value) -> Result<(Self, bool), InferError> {
        let Value::Object(mut obj) = body else { return Err(InferError::BadRequest("body must be a JSON object".into())) };
        let prompt = match obj.remove("prompt") {
            Some(Value::String(p)) => p,
            Some(Value::Array(a)) if a.len() == 1 && a[0].is_string() => a[0].as_str().unwrap_or_default().to_string(),
            _ => return Err(InferError::BadRequest("`prompt` must be a string (or an array of one string)".into())),
        };
        let (model, stream, wait_for_model) = take_common(&mut obj)?;
        Ok((Self { model, prompt, params: obj, wait_for_model }, stream))
    }

    pub fn to_openai_stream(&self, model: &str) -> Value {
        let mut obj = self.params.clone();
        obj.insert("model".into(), model.into());
        obj.insert("prompt".into(), self.prompt.clone().into());
        obj.insert("stream".into(), true.into());
        obj.insert("stream_options".into(), serde_json::json!({"include_usage": true}));
        Value::Object(obj)
    }
}

/// An embeddings request (`/v1/embeddings`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EmbedRequest {
    pub model: Option<String>,
    pub input: Vec<String>,
}

impl EmbedRequest {
    pub fn from_openai(body: Value) -> Result<Self, InferError> {
        let Value::Object(mut obj) = body else { return Err(InferError::BadRequest("body must be a JSON object".into())) };
        let input = match obj.remove("input") {
            Some(Value::String(s)) => vec![s],
            Some(Value::Array(a)) if !a.is_empty() && a.iter().all(Value::is_string) => {
                a.into_iter().filter_map(|v| v.as_str().map(str::to_string)).collect()
            }
            _ => return Err(InferError::BadRequest("`input` must be a string or a non-empty array of strings".into())),
        };
        if obj.get("encoding_format").and_then(Value::as_str).is_some_and(|f| f != "float") {
            return Err(InferError::BadRequest("only encoding_format = \"float\" is supported".into()));
        }
        let (model, _, _) = take_common(&mut obj)?;
        Ok(Self { model, input })
    }
}

/// Embedding vectors, one per input, in order. (On infer/2 they travel as little-endian f32 byte
/// strings.)
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Embeddings {
    pub model: String,
    pub node: String,
    pub vectors: Vec<Vec<f32>>,
    pub prompt_tokens: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

/// A stream frame (infer/2). Offsets are byte offsets into the answer so far, so a resumed stream
/// can skip what the caller already has.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    Queued { position: u32 },
    Delta { offset: u64, text: String },
    Reasoning { offset: u64, text: String },
    ToolCall { index: u32, id: Option<String>, name: Option<String>, arguments: String },
    Done { finish_reason: Option<String>, usage: Option<Usage> },
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, Serialize, Deserialize)]
#[serde(tag = "kind", content = "detail", rename_all = "snake_case")]
pub enum InferError {
    #[error("unknown model `{0}`")]
    UnknownModel(String),
    #[error("busy; retry in {retry_after_secs}s")]
    Busy { retry_after_secs: u32 },
    #[error("unavailable: {0}")]
    Unavailable(String),
    #[error("bad request: {0}")]
    BadRequest(String),
    #[error("engine error: {0}")]
    Engine(String),
}

/// A model's state on one node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", content = "detail", rename_all = "snake_case")]
pub enum ModelState {
    Unloaded,
    Loading,
    Ready,
    Draining,
    Failed(String),
}

/// One pool catalog row (`/weirdnet-agent/catalog/1`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatalogModel {
    pub id: String,
    /// Model nodes that have it, with its state on each.
    pub nodes: Vec<(String, ModelState)>,
    /// `chat` or `embedding`.
    pub purpose: String,
}

impl CatalogModel {
    pub fn ready(&self) -> bool {
        self.nodes.iter().any(|(_, s)| *s == ModelState::Ready)
    }
}

/// What the pool answers with: who serves the request, then the token stream.
pub struct Reply {
    pub model: String,
    pub node: String,
    pub cache_hit: bool,
    /// Set when a stand-in answers for the requested model.
    pub standin: Option<String>,
    pub events: EventStream,
}

/// The app side of the cloud: list models and send prompts. The inference role implements it in
/// process; `weirdnet-agent`'s `pool` client will implement it over Weirdnet.
pub trait Pool: Send + Sync {
    fn catalog(&self) -> BoxFuture<'_, Result<Vec<CatalogModel>, InferError>>;
    fn chat(&self, req: ChatRequest) -> BoxFuture<'_, Result<Reply, InferError>>;
    /// A raw completion; same frames as chat.
    fn complete(&self, req: CompletionRequest) -> BoxFuture<'_, Result<Reply, InferError>> {
        let _ = req;
        Box::pin(async { Err(InferError::Unavailable("completions are not supported by this pool".into())) })
    }
    fn embed(&self, req: EmbedRequest) -> BoxFuture<'_, Result<Embeddings, InferError>> {
        let _ = req;
        Box::pin(async { Err(InferError::Unavailable("embeddings are not supported by this pool".into())) })
    }
}

/// A finished answer, collected from a stream.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Collected {
    pub text: String,
    pub reasoning: String,
    pub tool_calls: Vec<(u32, Option<String>, Option<String>, String)>,
    pub finish_reason: Option<String>,
    pub usage: Option<Usage>,
}

impl Collected {
    pub fn push(&mut self, ev: &Event) {
        match ev {
            Event::Delta { text, .. } => self.text.push_str(text),
            Event::Reasoning { text, .. } => self.reasoning.push_str(text),
            Event::ToolCall { index, id, name, arguments } => match self.tool_calls.iter_mut().find(|c| c.0 == *index) {
                Some(c) => {
                    if c.1.is_none() {
                        c.1.clone_from(id);
                    }
                    if c.2.is_none() {
                        c.2.clone_from(name);
                    }
                    c.3.push_str(arguments);
                }
                None => self.tool_calls.push((*index, id.clone(), name.clone(), arguments.clone())),
            },
            Event::Done { finish_reason, usage } => {
                self.finish_reason.clone_from(finish_reason);
                self.usage = *usage;
            }
            Event::Queued { .. } => {}
        }
    }
}

/// Drains a stream into a [`Collected`].
pub async fn collect(mut events: EventStream) -> Result<Collected, InferError> {
    let mut out = Collected::default();
    while let Some(ev) = events.next().await {
        out.push(&ev?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn openai_round_trip() {
        let (req, stream) =
            ChatRequest::from_openai(json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": true, "temperature": 0}))
                .unwrap();
        assert!(stream);
        assert_eq!(req.model.as_deref(), Some("m"));
        assert!(req.deterministic());
        let body = req.to_openai_stream("alias");
        assert_eq!(body["model"], "alias");
        assert_eq!(body["stream"], true);
        assert_eq!(body["temperature"], 0);
    }

    #[test]
    fn completions_and_embeddings() {
        let (c, stream) = CompletionRequest::from_openai(json!({"model": "m", "prompt": ["Once"], "max_tokens": 5, "wait_for_model": true})).unwrap();
        assert!(!stream && c.wait_for_model);
        assert_eq!(c.prompt, "Once");
        assert_eq!(c.to_openai_stream("m")["max_tokens"], 5);
        assert!(CompletionRequest::from_openai(json!({"prompt": 3})).is_err());
        let e = EmbedRequest::from_openai(json!({"model": "e", "input": ["a", "b"]})).unwrap();
        assert_eq!(e.input, ["a", "b"]);
        assert_eq!(EmbedRequest::from_openai(json!({"input": "x"})).unwrap().input, ["x"]);
        assert!(EmbedRequest::from_openai(json!({"input": [1]})).is_err());
        assert!(EmbedRequest::from_openai(json!({"input": "x", "encoding_format": "base64"})).is_err());
    }

    #[test]
    fn rejects_bad_bodies() {
        assert!(ChatRequest::from_openai(json!([])).is_err());
        assert!(ChatRequest::from_openai(json!({"messages": []})).is_err());
        assert!(ChatRequest::from_openai(json!({"messages": [{"role": "user", "content": "x"}], "n": 2})).is_err());
    }

    #[tokio::test]
    async fn collects() {
        let evs = vec![
            Ok(Event::Delta { offset: 0, text: "He".into() }),
            Ok(Event::Delta { offset: 2, text: "llo".into() }),
            Ok(Event::Done { finish_reason: Some("stop".into()), usage: Some(Usage { prompt_tokens: 1, completion_tokens: 2 }) }),
        ];
        let c = collect(Box::pin(futures::stream::iter(evs))).await.unwrap();
        assert_eq!(c.text, "Hello");
        assert_eq!(c.finish_reason.as_deref(), Some("stop"));
    }
}
