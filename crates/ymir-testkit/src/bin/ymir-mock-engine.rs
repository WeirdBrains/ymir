//! A stand-in for `llama-server`: same flags and endpoints, canned answers.
//!
//! Answers "Hello from `<alias>`! You said: `<last user message>`" word by word. `/health` says 503
//! for `YMIR_MOCK_LOAD_MS` (default 200) after start, like a model loading. `/v1/completions`
//! continues the prompt with " once upon a time"; `/v1/embeddings` returns
//! `[chars, words, 1, 0]` per input.
//!
//! Images: with `--mmproj FILE` (a file that exists), image parts in a chat message are counted and
//! the answer ends with ` [N image(s)]`; without one, a request with an image fails like
//! `llama-server`'s ("image input is not supported").
//!
//! vLLM: as `vllm serve MODEL --served-model-name ID` (key in `VLLM_API_KEY`). With
//! `--distributed-executor-backend ray` it asks the Ray cluster at `RAY_ADDRESS` (a
//! `ymir-mock-ray` head) how many nodes it has and exits unless there are
//! `--pipeline-parallel-size` of them, like vLLM failing to place its workers.
//!
//! LoRA: `--lora FILE` (repeatable) as llama-server takes it; a chat body's `"lora": [{"id": i,
//! "scale": s}]` applies file `i`, and the answer reads "Hello from `<alias>` with `<file stem>` at
//! `<s>`!". An id past the list is refused like llama-server's "invalid adapter id".

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::Router;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde_json::{Value, json};

struct Mock {
    alias: String,
    key: Option<String>,
    vision: bool,
    ready_at: Instant,
    /// `--lora` files, in order (llama-server's adapter ids are their positions).
    loras: Vec<String>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let host = flag("--host").unwrap_or_else(|| "127.0.0.1".into());
    let port: u16 = flag("--port").and_then(|p| p.parse().ok()).unwrap_or(8080);
    let alias = flag("--alias").or_else(|| flag("--served-model-name")).or_else(|| flag("--model")).unwrap_or_else(|| "mock".into());
    if flag("--distributed-executor-backend").as_deref() == Some("ray") {
        let addr = std::env::var("RAY_ADDRESS").expect("RAY_ADDRESS for a Ray executor");
        let pp: usize = flag("--pipeline-parallel-size").and_then(|p| p.parse().ok()).unwrap_or(1);
        let tp: usize = flag("--tensor-parallel-size").and_then(|p| p.parse().ok()).unwrap_or(1);
        let nodes = ray_nodes(&addr);
        eprintln!("mock vllm: ray {addr} nodes={nodes} tp={tp} pp={pp}");
        if nodes < pp {
            eprintln!("mock vllm: the Ray cluster has {nodes} nodes; pipeline parallel {pp} needs {pp}");
            std::process::exit(1);
        }
    }
    let load_ms = std::env::var("YMIR_MOCK_LOAD_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
    assert_eq!(host, "127.0.0.1", "engines must bind loopback only");
    let vision = flag("--mmproj").is_some_and(|p| std::path::Path::new(&p).is_file());
    let loras: Vec<String> = args.windows(2).filter(|w| w[0] == "--lora").map(|w| w[1].clone()).collect();
    let st = Arc::new(Mock {
        alias,
        key: std::env::var("LLAMA_API_KEY").or_else(|_| std::env::var("VLLM_API_KEY")).ok(),
        vision,
        ready_at: Instant::now() + Duration::from_millis(load_ms),
        loras,
    });
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .with_state(st);
    let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await.expect("bind");
    // With `--port 0` the OS picks the port; tests read it from this line.
    eprintln!("mock engine listening on {}", listener.local_addr().expect("local addr"));
    axum::serve(listener, app).await.expect("serve");
}

async fn health(State(st): State<Arc<Mock>>) -> Response {
    if Instant::now() < st.ready_at {
        return (StatusCode::SERVICE_UNAVAILABLE, axum::Json(json!({"error": {"message": "Loading model", "code": 503}}))).into_response();
    }
    axum::Json(json!({"status": "ok"})).into_response()
}

fn authorized(st: &Mock, headers: &HeaderMap) -> bool {
    match &st.key {
        None => true,
        Some(k) => headers.get("authorization").and_then(|v| v.to_str().ok()) == Some(&format!("Bearer {k}")),
    }
}

/// Behind the key, like llama-server (b12628) and vLLM: the supervisor checks it is talking to its
/// own engine here before calling it ready.
async fn models(State(st): State<Arc<Mock>>, headers: HeaderMap) -> Response {
    if !authorized(&st, &headers) {
        return (StatusCode::UNAUTHORIZED, axum::Json(json!({"error": {"message": "Invalid API Key"}}))).into_response();
    }
    axum::Json(json!({"object": "list", "data": [{"id": st.alias, "object": "model"}]})).into_response()
}

fn sse(events: Vec<Value>) -> Response {
    let stream = futures::stream::unfold((events.into_iter(), false), |(mut it, done)| async move {
        if done {
            return None;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
        match it.next() {
            Some(v) => Some((Ok::<_, std::convert::Infallible>(Event::default().data(v.to_string())), (it, false))),
            None => Some((Ok(Event::default().data("[DONE]")), (it, true))),
        }
    });
    Sse::new(stream).into_response()
}

fn check(st: &Mock, headers: &HeaderMap, body: &Value) -> Option<Response> {
    if !authorized(st, headers) {
        return Some((StatusCode::UNAUTHORIZED, axum::Json(json!({"error": {"message": "Invalid API Key"}}))).into_response());
    }
    if !matches!(body["model"].as_str(), Some(m) if m == st.alias || m == "default_model") {
        return Some(
            (StatusCode::BAD_REQUEST, axum::Json(json!({"error": {"message": format!("unknown model {}", body["model"])}}))).into_response(),
        );
    }
    None
}

async fn completions(State(st): State<Arc<Mock>>, headers: HeaderMap, axum::Json(body): axum::Json<Value>) -> Response {
    if let Some(r) = check(&st, &headers, &body) {
        return r;
    }
    let words = [" once", " upon", " a", " time"];
    let alias = st.alias.clone();
    let mut events: Vec<Value> = words
        .iter()
        .map(|w| json!({"object": "text_completion", "model": alias, "choices": [{"index": 0, "text": w, "finish_reason": null}]}))
        .collect();
    events.push(json!({"object": "text_completion", "model": alias, "choices": [{"index": 0, "text": "", "finish_reason": "length"}], "usage": {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6}}));
    sse(events)
}

async fn embeddings(State(st): State<Arc<Mock>>, headers: HeaderMap, axum::Json(body): axum::Json<Value>) -> Response {
    if let Some(r) = check(&st, &headers, &body) {
        return r;
    }
    let inputs: Vec<String> = match &body["input"] {
        Value::String(s) => vec![s.clone()],
        Value::Array(a) => a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect(),
        _ => Vec::new(),
    };
    let data: Vec<Value> = inputs
        .iter()
        .enumerate()
        .map(|(i, s)| json!({"object": "embedding", "index": i, "embedding": [s.chars().count() as f32, s.split_whitespace().count() as f32, 1.0, 0.0]}))
        .collect();
    axum::Json(json!({"object": "list", "model": st.alias, "data": data, "usage": {"prompt_tokens": inputs.len(), "total_tokens": inputs.len()}}))
        .into_response()
}

async fn chat(State(st): State<Arc<Mock>>, headers: HeaderMap, axum::Json(body): axum::Json<Value>) -> Response {
    if !authorized(&st, &headers) {
        return (StatusCode::UNAUTHORIZED, axum::Json(json!({"error": {"message": "Invalid API Key"}}))).into_response();
    }
    if !matches!(body["model"].as_str(), Some(m) if m == st.alias || m == "default_model") {
        return (StatusCode::BAD_REQUEST, axum::Json(json!({"error": {"message": format!("unknown model {}", body["model"])}}))).into_response();
    }
    let last_user = body["messages"].as_array().and_then(|m| m.iter().rev().find(|m| m["role"] == "user"));
    let content = last_user.map(|m| &m["content"]);
    let parts: &[Value] = content.and_then(Value::as_array).map_or(&[], Vec::as_slice);
    let images = parts.iter().filter(|p| p["type"] == "image_url").count();
    if images > 0 && !st.vision {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(json!({"error": {"code": 500, "message": "image input is not supported - hint: if this is unexpected, you may need to provide the mmproj", "type": "server_error"}})),
        )
            .into_response();
    }
    let said = match content.and_then(Value::as_str) {
        Some(t) => t.to_string(),
        None => parts.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join(" "),
    };
    let seen = if images > 0 { format!(" [{images} image(s)]") } else { String::new() };
    // Per-request adapters as llama-server takes them: `"lora": [{"id": index, "scale": s}]`.
    let mut applied = Vec::new();
    for l in body["lora"].as_array().into_iter().flatten() {
        let Some(file) = l["id"].as_u64().and_then(|i| st.loras.get(i as usize)) else {
            return (StatusCode::BAD_REQUEST, axum::Json(json!({"error": {"message": format!("invalid adapter id {}", l["id"])}}))).into_response();
        };
        let scale = l["scale"].as_f64().unwrap_or(1.0);
        if scale != 0.0 {
            let name = std::path::Path::new(file).file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            applied.push(format!("{name} at {scale}"));
        }
    }
    let with = if applied.is_empty() { String::new() } else { format!(" with {}", applied.join(", ")) };
    let answer = format!("Hello from {}{with}! You said: {said}{seen}", st.alias);
    // A trailing assistant message is a prefix to continue (llama.cpp prefill), as a resume
    // sends: only what comes after it streams.
    let prefix = body["messages"].as_array().and_then(|m| m.last()).filter(|m| m["role"] == "assistant").and_then(|m| m["content"].as_str());
    let answer = match prefix {
        Some(p) => answer.strip_prefix(p).unwrap_or(&answer).to_string(),
        None => answer,
    };
    let words: Vec<String> = answer.split_inclusive(' ').map(str::to_string).collect();
    let word_ms: u64 = std::env::var("YMIR_MOCK_WORD_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
    let n = words.len() as u64;
    let alias = st.alias.clone();
    let chunk = move |delta: Value, finish: Option<&str>| json!({"object": "chat.completion.chunk", "model": alias, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
    let mut events: Vec<Value> = vec![chunk(json!({"role": "assistant", "content": null}), None)];
    events.extend(words.into_iter().map(|w| chunk(json!({"content": w}), None)));
    events.push(chunk(json!({}), Some("stop")));
    events.push(
        json!({"object": "chat.completion.chunk", "choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": n, "total_tokens": 5 + n}}),
    );
    let stream = futures::stream::unfold((events.into_iter(), false), move |(mut it, done)| async move {
        if done {
            return None;
        }
        tokio::time::sleep(Duration::from_millis(word_ms)).await;
        match it.next() {
            Some(v) => Some((Ok::<_, std::convert::Infallible>(Event::default().data(v.to_string())), (it, false))),
            None => Some((Ok(Event::default().data("[DONE]")), (it, true))),
        }
    });
    Sse::new(stream).into_response()
}

/// The node count a `ymir-mock-ray` head reports.
fn ray_nodes(addr: &str) -> usize {
    use std::io::{Read, Write};
    let Ok(mut s) = std::net::TcpStream::connect(addr) else { return 0 };
    let _ = s.write_all(b"status\n");
    let mut out = String::new();
    let _ = s.read_to_string(&mut out);
    out.trim().parse().unwrap_or(0)
}
