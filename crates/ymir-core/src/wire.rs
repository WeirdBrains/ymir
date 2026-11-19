//! Conversions between `weirdnet-agent`'s request types (what crosses Weirdnet: infer/2,
//! catalog/1) and the `ymir-infer` types the frozen interim crates still speak (the OpenAI bridge,
//! the answer cache and the backends supervisor). They go away at the swap-over to
//! `weirdnet-openai`, `weirdnet-semcache` and `weirdnet-backends`.
//!
//! What infer/2 can't carry is refused, never silently changed: message content other than text
//! (images, audio) is a bad request. OpenAI parameters with no infer/2 field are dropped, and
//! [`to_agent`] names them so the caller can log them.

use serde_json::{Map, Value, json};
use weirdnet_agent::remote::RemoteError;
use weirdnet_agent::remote::catalog::{self, ModelState as AgentState, Purpose};
use weirdnet_agent::remote::infer::v2::{self, Priority as AgentPriority, ResponseFormat, Sampling};
use weirdnet_agent::{Message, Role, ToolCall, ToolSpec};
use ymir_infer::{CatalogModel, ChatRequest, InferError, ModelState, Priority};

/// OpenAI parameters with an infer/2 field.
const MAPPED: &[&str] = &[
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "max_tokens",
    "max_completion_tokens",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "repeat_penalty",
    "seed",
    "response_format",
    "tools",
];

fn f32_of(params: &Map<String, Value>, k: &str) -> Option<f32> {
    params.get(k).and_then(Value::as_f64).map(|v| v as f32)
}

fn u32_of(params: &Map<String, Value>, k: &str) -> Option<u32> {
    params.get(k).and_then(Value::as_u64).and_then(|v| u32::try_from(v).ok())
}

fn text_of(content: &Value) -> Result<String, InferError> {
    match content {
        Value::Null => Ok(String::new()),
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                match (p.get("type").and_then(Value::as_str), p.get("text").and_then(Value::as_str)) {
                    (Some("text"), Some(t)) => out.push_str(t),
                    (kind, _) => {
                        return Err(InferError::BadRequest(format!(
                            "message content of type {} can't be sent to another node yet (text only over infer/2)",
                            kind.unwrap_or("unknown")
                        )));
                    }
                }
            }
            Ok(out)
        }
        _ => Err(InferError::BadRequest("message content must be a string or an array of parts".into())),
    }
}

/// An OpenAI chat message as an agent [`Message`].
pub fn message_to_agent(m: &Value) -> Result<Message, InferError> {
    let role = match m.get("role").and_then(Value::as_str) {
        Some("system" | "developer") => Role::System,
        Some("user") => Role::User,
        Some("assistant") => Role::Assistant,
        Some("tool") => Role::Tool,
        other => return Err(InferError::BadRequest(format!("unknown message role {other:?}"))),
    };
    let content = text_of(m.get("content").unwrap_or(&Value::Null))?;
    let mut msg = match role {
        Role::System => Message::system(content),
        Role::User => Message::user(content),
        Role::Tool => Message::tool(m.get("tool_call_id").and_then(Value::as_str).unwrap_or_default(), content),
        _ => Message::assistant(content, Vec::new()),
    };
    if let Some(calls) = m.get("tool_calls").and_then(Value::as_array) {
        for c in calls {
            let f = c.get("function").unwrap_or(&Value::Null);
            let args = match f.get("arguments") {
                Some(Value::String(s)) => serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone())),
                Some(v) => v.clone(),
                None => json!({}),
            };
            msg.tool_calls.push(ToolCall {
                id: c.get("id").and_then(Value::as_str).unwrap_or_default().to_string(),
                name: f.get("name").and_then(Value::as_str).unwrap_or_default().to_string(),
                arguments: args,
            });
        }
    }
    Ok(msg)
}

/// An agent [`Message`] as an OpenAI chat message.
pub fn message_to_openai(m: &Message) -> Value {
    let role = match m.role {
        Role::System => "system",
        Role::User => "user",
        Role::Tool => "tool",
        _ => "assistant",
    };
    let mut v = json!({"role": role, "content": m.content});
    if !m.tool_calls.is_empty() {
        v["tool_calls"] = m
            .tool_calls
            .iter()
            .map(|c| json!({"id": c.id, "type": "function", "function": {"name": c.name, "arguments": arguments_text(&c.arguments)}}))
            .collect();
    }
    if let Some(id) = &m.tool_call_id {
        v["tool_call_id"] = id.clone().into();
    }
    v
}

/// Tool-call arguments as the JSON text OpenAI clients expect.
pub fn arguments_text(args: &Value) -> String {
    match args {
        Value::String(s) => s.clone(),
        v => v.to_string(),
    }
}

fn tool_to_agent(t: &Value) -> Option<ToolSpec> {
    let f = t.get("function")?;
    let name = f.get("name")?.as_str()?;
    let desc = f.get("description").and_then(Value::as_str).unwrap_or_default();
    let mut spec = ToolSpec::new(name, desc);
    if let Some(p) = f.get("parameters") {
        spec = spec.schema(p.clone());
    }
    Some(spec)
}

fn tool_to_openai(t: &ToolSpec) -> Value {
    json!({"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}})
}

fn format_to_agent(v: &Value) -> Option<ResponseFormat> {
    match v.get("type").and_then(Value::as_str)? {
        "text" => Some(ResponseFormat::Text),
        "json_object" => Some(ResponseFormat::Json),
        "json_schema" => {
            let s = v.get("json_schema")?;
            Some(ResponseFormat::JsonSchema {
                name: s.get("name").and_then(Value::as_str).unwrap_or("schema").to_string(),
                schema: s.get("schema").cloned().unwrap_or(json!({})),
                strict: s.get("strict").and_then(Value::as_bool).unwrap_or(false),
            })
        }
        _ => None,
    }
}

fn format_to_openai(f: &ResponseFormat) -> Value {
    match f {
        ResponseFormat::Json => json!({"type": "json_object"}),
        ResponseFormat::JsonSchema { name, schema, strict } => {
            json!({"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": strict}})
        }
        _ => json!({"type": "text"}),
    }
}

/// A bridge request as an infer/2 request, and the OpenAI parameters infer/2 has no field for.
pub fn to_agent(req: &ChatRequest) -> Result<(v2::Request, Vec<String>), InferError> {
    let messages = req.messages.iter().map(message_to_agent).collect::<Result<Vec<_>, _>>()?;
    let p = &req.params;
    let mut sampling = Sampling::default();
    sampling.temperature = f32_of(p, "temperature");
    sampling.top_p = f32_of(p, "top_p");
    sampling.top_k = u32_of(p, "top_k");
    sampling.min_p = f32_of(p, "min_p");
    sampling.max_tokens = u32_of(p, "max_tokens").or_else(|| u32_of(p, "max_completion_tokens"));
    sampling.stop = match p.get("stop") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().filter_map(|s| s.as_str().map(str::to_string)).collect(),
        _ => Vec::new(),
    };
    sampling.presence_penalty = f32_of(p, "presence_penalty");
    sampling.frequency_penalty = f32_of(p, "frequency_penalty");
    sampling.repeat_penalty = f32_of(p, "repeat_penalty");
    let mut r = v2::Request::messages(messages).with_sampling(sampling).with_priority(match req.priority {
        Priority::Batch => AgentPriority::Batch,
        Priority::Interactive => AgentPriority::Interactive,
    });
    if let Some(m) = &req.model {
        r = r.with_model(m.clone());
    }
    if let Some(seed) = p.get("seed").and_then(Value::as_u64) {
        r = r.with_seed(seed);
    }
    if let Some(f) = p.get("response_format").and_then(format_to_agent) {
        r = r.with_response_format(f);
    }
    if let Some(tools) = p.get("tools").and_then(Value::as_array) {
        r = r.with_tools(tools.iter().filter_map(tool_to_agent).collect());
    }
    if let Some(j) = &req.job_id {
        r = r.with_job_id(j.clone());
    }
    r.wait_for_model = req.wait_for_model;
    let dropped = p.keys().filter(|k| !MAPPED.contains(&k.as_str())).cloned().collect();
    Ok((r, dropped))
}

/// An infer/2 request as a bridge request (for the answer cache and the backends supervisor).
pub fn from_agent(r: &v2::Request) -> Result<ChatRequest, InferError> {
    let Some(messages) = r.inline_messages() else {
        return Err(InferError::Unavailable("this inference node keeps no conversations yet (M5); send the messages".into()));
    };
    let s = &r.sampling;
    let mut params = Map::new();
    let mut put = |k: &str, v: Option<Value>| {
        if let Some(v) = v {
            params.insert(k.into(), v);
        }
    };
    put("temperature", s.temperature.map(|v| json!(v)));
    put("top_p", s.top_p.map(|v| json!(v)));
    put("top_k", s.top_k.map(|v| json!(v)));
    put("min_p", s.min_p.map(|v| json!(v)));
    put("max_tokens", s.max_tokens.map(|v| json!(v)));
    put("stop", (!s.stop.is_empty()).then(|| json!(s.stop)));
    put("presence_penalty", s.presence_penalty.map(|v| json!(v)));
    put("frequency_penalty", s.frequency_penalty.map(|v| json!(v)));
    put("repeat_penalty", s.repeat_penalty.map(|v| json!(v)));
    put("seed", r.seed.map(|v| json!(v)));
    put("response_format", r.response_format.as_ref().map(format_to_openai));
    put("tools", (!r.tools.is_empty()).then(|| r.tools.iter().map(tool_to_openai).collect()));
    Ok(ChatRequest {
        model: r.model.clone(),
        messages: messages.iter().map(message_to_openai).collect(),
        params,
        priority: if r.priority == AgentPriority::Interactive { Priority::Interactive } else { Priority::Batch },
        job_id: Some(r.job_id.clone()),
        wait_for_model: r.wait_for_model,
    })
}

/// A coordinator error as it crosses Weirdnet.
pub fn error_to_remote(e: InferError) -> RemoteError {
    match e {
        InferError::UnknownModel(m) => RemoteError::unavailable(format!("unknown model `{m}`")),
        InferError::Busy { .. } => RemoteError::Busy,
        InferError::Unavailable(m) => RemoteError::unavailable(m),
        InferError::BadRequest(m) => RemoteError::BadRequest { message: m },
        InferError::Engine(m) => RemoteError::Failed { message: m },
    }
}

/// A Weirdnet error as the bridge reports it.
pub fn error_from_remote(e: RemoteError) -> InferError {
    match e {
        RemoteError::Busy => InferError::Busy { retry_after_secs: 1 },
        RemoteError::BadRequest { message } => InferError::BadRequest(message),
        RemoteError::Unavailable { message } => match message.strip_prefix("unknown model `").and_then(|m| m.strip_suffix('`')) {
            Some(m) => InferError::UnknownModel(m.to_string()),
            None => InferError::Unavailable(message),
        },
        RemoteError::Failed { message } => InferError::Engine(message),
        RemoteError::Refused { message } => InferError::Unavailable(format!("refused by the inference node: {message}")),
        other => InferError::Unavailable(other.to_string()),
    }
}

/// The coordinator's merged catalog as catalog/1 entries. `standin` is the policy's stand-in,
/// named on every other chat model.
pub fn catalog_to_agent(models: &[CatalogModel], standin: Option<&str>) -> Vec<catalog::CatalogModel> {
    models
        .iter()
        .map(|m| {
            let states: Vec<&ModelState> = m.nodes.iter().map(|(_, s)| s).collect();
            let state = if states.contains(&&ModelState::Ready) {
                AgentState::Warm
            } else if states.contains(&&ModelState::Loading) {
                AgentState::Loading
            } else {
                AgentState::Loadable
            };
            let purpose = if m.purpose == "embedding" { Purpose::Embedding } else { Purpose::Chat };
            let ready = states.iter().filter(|s| ***s == ModelState::Ready).count() as u32;
            let mut c = catalog::CatalogModel::new(&m.id, purpose, state).with_replicas(ready);
            if purpose == Purpose::Chat {
                c.standin = standin.filter(|s| *s != m.id).map(str::to_string);
            }
            c
        })
        .collect()
}

/// catalog/1 entries from inference node `node` as bridge catalog rows.
pub fn catalog_from_agent(node: &str, models: &[catalog::CatalogModel]) -> Vec<CatalogModel> {
    models
        .iter()
        .map(|m| CatalogModel {
            id: m.id.clone(),
            nodes: vec![(
                node.to_string(),
                match m.state {
                    AgentState::Warm | AgentState::Busy => ModelState::Ready,
                    AgentState::Loading => ModelState::Loading,
                    _ => ModelState::Unloaded,
                },
            )],
            purpose: if m.purpose == Purpose::Embedding { "embedding".into() } else { "chat".into() },
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_a_bridge_request() {
        let (req, _) = ChatRequest::from_openai(json!({
            "model": "m",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": [{"type": "text", "text": "hi "}, {"type": "text", "text": "there"}]},
                {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"a\":1}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "ok"}
            ],
            "temperature": 0.5, "max_tokens": 7, "stop": "END", "seed": 3, "logprobs": true,
            "response_format": {"type": "json_object"},
            "tools": [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}}]
        }))
        .unwrap();
        let (a, dropped) = to_agent(&req).unwrap();
        assert_eq!(dropped, ["logprobs"]);
        assert_eq!(a.model.as_deref(), Some("m"));
        let msgs = a.inline_messages().unwrap();
        assert_eq!(msgs[1].content, "hi there");
        assert_eq!(msgs[2].tool_calls[0].arguments, json!({"a": 1}));
        assert_eq!(msgs[3].tool_call_id.as_deref(), Some("c1"));
        assert_eq!(a.sampling.max_tokens, Some(7));
        assert_eq!(a.sampling.stop, ["END"]);
        assert_eq!(a.tools[0].name, "f");

        let back = from_agent(&a).unwrap();
        assert_eq!(back.params["max_tokens"], 7);
        assert_eq!(back.params["seed"], 3);
        assert_eq!(back.params["response_format"], json!({"type": "json_object"}));
        assert_eq!(back.params["tools"][0]["function"]["name"], "f");
        assert_eq!(back.messages[2]["tool_calls"][0]["function"]["arguments"], "{\"a\":1}");
        assert!(!back.params.contains_key("logprobs"));
    }

    #[test]
    fn refuses_what_infer2_cannot_carry() {
        let (req, _) =
            ChatRequest::from_openai(json!({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]})).unwrap();
        assert!(matches!(to_agent(&req), Err(InferError::BadRequest(_))));
        let ctx = v2::Request::context(v2::ContextRef::new("c", None));
        assert!(matches!(from_agent(&ctx), Err(InferError::Unavailable(_))));
    }

    #[test]
    fn errors_and_catalogs() {
        assert_eq!(error_from_remote(error_to_remote(InferError::UnknownModel("x".into()))), InferError::UnknownModel("x".into()));
        assert_eq!(error_from_remote(error_to_remote(InferError::BadRequest("b".into()))), InferError::BadRequest("b".into()));
        let rows = vec![
            CatalogModel { id: "big".into(), nodes: vec![("n".into(), ModelState::Unloaded)], purpose: "chat".into() },
            CatalogModel { id: "small".into(), nodes: vec![("n".into(), ModelState::Ready)], purpose: "chat".into() },
            CatalogModel { id: "emb".into(), nodes: vec![("n".into(), ModelState::Loading)], purpose: "embedding".into() },
        ];
        let c = catalog_to_agent(&rows, Some("small"));
        assert_eq!(c[0].state, AgentState::Loadable);
        assert_eq!(c[0].standin.as_deref(), Some("small"));
        assert_eq!(c[1].standin, None);
        assert_eq!((c[1].state, c[1].replicas), (AgentState::Warm, 1));
        assert_eq!(c[2].purpose, Purpose::Embedding);
        let back = catalog_from_agent("inf", &c);
        assert_eq!(back[1].nodes, [("inf".to_string(), ModelState::Ready)]);
        assert_eq!(back[2].purpose, "embedding");
    }
}
