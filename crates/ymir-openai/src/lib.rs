//! Loopback-only OpenAI-compatible bridge over a [`Pool`].
//!
//! To be extracted into `weirdnet-openai`.

use std::convert::Infallible;
use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::Router;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use futures::StreamExt;
use serde_json::{Value, json};
use ymir_infer::{ChatRequest, Collected, Event, InferError, Pool};

/// Where the bridge listens and the token it requires.
#[derive(Clone)]
pub struct BridgeOptions {
    /// Port on 127.0.0.1; 0 picks a free one (tests).
    pub port: u16,
    pub token: String,
}

impl std::fmt::Debug for BridgeOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BridgeOptions").field("port", &self.port).field("token", &"<redacted>").finish()
    }
}

/// A bound, not yet serving, bridge.
pub struct Bridge {
    listener: tokio::net::TcpListener,
    token: String,
}

#[derive(Default)]
struct Metrics {
    requests: AtomicU64,
    errors: AtomicU64,
    cache_hits: AtomicU64,
}

struct AppState {
    pool: Arc<dyn Pool>,
    token: String,
    port: u16,
    metrics: Metrics,
}

impl Bridge {
    /// Binds `127.0.0.1:port`. Refuses an empty token.
    pub async fn bind(opts: BridgeOptions) -> std::io::Result<Self> {
        if opts.token.len() < 16 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "the bridge token must be at least 16 characters"));
        }
        let listener = tokio::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], opts.port))).await?;
        Ok(Self { listener, token: opts.token })
    }

    pub fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.listener.local_addr()
    }

    /// Serves until `shutdown` resolves.
    pub async fn serve(self, pool: Arc<dyn Pool>, shutdown: impl Future<Output = ()> + Send + 'static) -> std::io::Result<()> {
        let port = self.listener.local_addr()?.port();
        let state = Arc::new(AppState { pool, token: self.token, port, metrics: Metrics::default() });
        axum::serve(self.listener, router(state)).with_graceful_shutdown(shutdown).await
    }
}

fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(not_yet))
        .route("/v1/embeddings", post(not_yet))
        .route("/metrics", get(metrics))
        .fallback(not_found)
        .layer(DefaultBodyLimit::max(16 << 20))
        .layer(middleware::from_fn_with_state(state.clone(), guard))
        .with_state(state)
}

/// Host check, then the token, on every request.
async fn guard(State(st): State<Arc<AppState>>, req: Request, next: Next) -> Response {
    let host = req.headers().get(header::HOST).and_then(|h| h.to_str().ok()).unwrap_or("");
    if host != format!("127.0.0.1:{}", st.port) && host != format!("localhost:{}", st.port) {
        return error(StatusCode::FORBIDDEN, "forbidden", "Host must be 127.0.0.1 or localhost with this port");
    }
    let presented = req.headers().get(header::AUTHORIZATION).and_then(|h| h.to_str().ok()).and_then(|h| h.strip_prefix("Bearer ")).unwrap_or("");
    if !same(presented.as_bytes(), st.token.as_bytes()) {
        return error(StatusCode::UNAUTHORIZED, "invalid_api_key", "missing or wrong bridge token (see `ymirctl bridge token`)");
    }
    st.metrics.requests.fetch_add(1, Ordering::Relaxed);
    next.run(req).await
}

fn same(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn error(status: StatusCode, code: &str, message: &str) -> Response {
    let kind = if status.is_server_error() { "server_error" } else { "invalid_request_error" };
    (status, axum::Json(json!({"error": {"message": message, "type": kind, "code": code}}))).into_response()
}

fn infer_error(st: &AppState, e: &InferError) -> Response {
    st.metrics.errors.fetch_add(1, Ordering::Relaxed);
    let msg = e.to_string();
    match e {
        InferError::UnknownModel(_) => error(StatusCode::NOT_FOUND, "model_not_found", &msg),
        InferError::BadRequest(_) => error(StatusCode::BAD_REQUEST, "bad_request", &msg),
        InferError::Busy { retry_after_secs } => {
            let mut r = error(StatusCode::SERVICE_UNAVAILABLE, "busy", &msg);
            r.headers_mut().insert(header::RETRY_AFTER, HeaderValue::from(*retry_after_secs));
            r
        }
        InferError::Unavailable(_) => error(StatusCode::SERVICE_UNAVAILABLE, "unavailable", &msg),
        InferError::Engine(_) => error(StatusCode::BAD_GATEWAY, "engine_error", &msg),
    }
}

async fn not_found() -> Response {
    error(StatusCode::NOT_FOUND, "not_found", "no such endpoint")
}

async fn not_yet() -> Response {
    error(StatusCode::NOT_IMPLEMENTED, "not_implemented", "not available yet (Ymir M1 serves /v1/models and /v1/chat/completions)")
}

async fn metrics(State(st): State<Arc<AppState>>) -> Response {
    let m = &st.metrics;
    let body = format!(
        "# TYPE weirdnet_bridge_requests_total counter\nweirdnet_bridge_requests_total {}\n# TYPE weirdnet_bridge_errors_total counter\nweirdnet_bridge_errors_total {}\n# TYPE weirdnet_bridge_cache_hits_total counter\nweirdnet_bridge_cache_hits_total {}\n",
        m.requests.load(Ordering::Relaxed),
        m.errors.load(Ordering::Relaxed),
        m.cache_hits.load(Ordering::Relaxed)
    );
    ([(header::CONTENT_TYPE, "text/plain; version=0.0.4")], body).into_response()
}

async fn models(State(st): State<Arc<AppState>>) -> Response {
    match st.pool.catalog().await {
        Ok(models) => {
            let data: Vec<Value> = models
                .iter()
                .map(|m| {
                    json!({
                        "id": m.id,
                        "object": "model",
                        "created": 0,
                        "owned_by": "weirdnet",
                        "weirdnet": {"ready": m.ready(), "purpose": m.purpose, "nodes": m.nodes.iter().map(|(n, _)| n).collect::<Vec<_>>()},
                    })
                })
                .collect();
            axum::Json(json!({"object": "list", "data": data})).into_response()
        }
        Err(e) => infer_error(&st, &e),
    }
}

async fn chat(State(st): State<Arc<AppState>>, body: Bytes) -> Response {
    let body: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return error(StatusCode::BAD_REQUEST, "bad_request", &format!("invalid JSON: {e}")),
    };
    let (req, stream) = match ChatRequest::from_openai(body) {
        Ok(r) => r,
        Err(e) => return infer_error(&st, &e),
    };
    let reply = match st.pool.chat(req).await {
        Ok(r) => r,
        Err(e) => return infer_error(&st, &e),
    };
    let mut headers = HeaderMap::new();
    if let Ok(v) = HeaderValue::from_str(&reply.model) {
        headers.insert(HeaderName::from_static("x-weirdnet-model"), v);
    }
    if reply.cache_hit {
        st.metrics.cache_hits.fetch_add(1, Ordering::Relaxed);
        headers.insert(HeaderName::from_static("x-weirdnet-cache"), HeaderValue::from_static("hit"));
    }
    if let Some(v) = reply.standin.as_deref().and_then(|s| HeaderValue::from_str(s).ok()) {
        headers.insert(HeaderName::from_static("x-weirdnet-standin"), v);
    }
    let id = format!("chatcmpl-{}", random_hex(12));
    let created = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let model = reply.model.clone();
    if !stream {
        return match ymir_infer::collect(reply.events).await {
            Ok(c) => (headers, axum::Json(completion(&id, created, &model, &c))).into_response(),
            Err(e) => infer_error(&st, &e),
        };
    }
    let chunk = move |delta: Value, finish: Option<&str>, usage: Option<Value>| -> Result<SseEvent, Infallible> {
        let mut v = json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
        if let Some(u) = usage {
            v["usage"] = u;
        }
        Ok(SseEvent::default().data(v.to_string()))
    };
    let first = futures::stream::once(std::future::ready(chunk(json!({"role": "assistant", "content": ""}), None, None)));
    let body = reply
        .events
        .flat_map(move |ev| {
            let out: Vec<Result<SseEvent, Infallible>> = match ev {
                Ok(Event::Delta { text, .. }) => vec![chunk(json!({"content": text}), None, None)],
                Ok(Event::Reasoning { text, .. }) => vec![chunk(json!({"reasoning_content": text}), None, None)],
                Ok(Event::ToolCall { index, id, name, arguments }) => {
                    let mut call = json!({"index": index, "function": {"arguments": arguments}});
                    if let Some(id) = id {
                        call["id"] = id.into();
                        call["type"] = "function".into();
                    }
                    if let Some(name) = name {
                        call["function"]["name"] = name.into();
                    }
                    vec![chunk(json!({"tool_calls": [call]}), None, None)]
                }
                Ok(Event::Done { finish_reason, usage }) => {
                    let usage = usage.map(|u| json!({"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens, "total_tokens": u.prompt_tokens + u.completion_tokens}));
                    vec![chunk(json!({}), Some(finish_reason.as_deref().unwrap_or("stop")), usage), Ok(SseEvent::default().data("[DONE]"))]
                }
                Ok(Event::Queued { .. }) => Vec::new(),
                Err(e) => vec![Ok(SseEvent::default().data(json!({"error": {"message": e.to_string(), "type": "server_error"}}).to_string()))],
            };
            futures::stream::iter(out)
        });
    (headers, Sse::new(first.chain(body)).keep_alive(KeepAlive::default())).into_response()
}

fn completion(id: &str, created: u64, model: &str, c: &Collected) -> Value {
    let mut message = json!({"role": "assistant", "content": c.text});
    if !c.reasoning.is_empty() {
        message["reasoning_content"] = c.reasoning.clone().into();
    }
    if !c.tool_calls.is_empty() {
        message["tool_calls"] = c
            .tool_calls
            .iter()
            .map(|(_, id, name, args)| json!({"id": id, "type": "function", "function": {"name": name, "arguments": args}}))
            .collect::<Vec<_>>()
            .into();
    }
    let usage = c.usage.unwrap_or_default();
    json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": c.finish_reason.as_deref().unwrap_or("stop")}],
        "usage": {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens, "total_tokens": usage.prompt_tokens + usage.completion_tokens},
    })
}

/// A new random bridge token (32 bytes, hex).
pub fn new_token() -> String {
    random_hex(32)
}

fn random_hex(bytes: usize) -> String {
    let mut buf = vec![0u8; bytes];
    getrandom::fill(&mut buf).expect("the OS RNG is available");
    buf.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use ymir_infer::{BoxFuture, CatalogModel, ModelState, Reply, Usage};

    struct Fake;

    impl Pool for Fake {
        fn catalog(&self) -> BoxFuture<'_, Result<Vec<CatalogModel>, InferError>> {
            Box::pin(async { Ok(vec![CatalogModel { id: "m".into(), nodes: vec![("n".into(), ModelState::Ready)], purpose: "chat".into() }]) })
        }
        fn chat(&self, req: ChatRequest) -> BoxFuture<'_, Result<Reply, InferError>> {
            Box::pin(async move {
                if req.model.as_deref() == Some("nope") {
                    return Err(InferError::UnknownModel("nope".into()));
                }
                let evs = vec![
                    Ok(Event::Delta { offset: 0, text: "Hel".into() }),
                    Ok(Event::Delta { offset: 3, text: "lo".into() }),
                    Ok(Event::Done { finish_reason: Some("stop".into()), usage: Some(Usage { prompt_tokens: 2, completion_tokens: 2 }) }),
                ];
                Ok(Reply { model: "m".into(), node: "n".into(), cache_hit: false, standin: None, events: Box::pin(futures::stream::iter(evs)) })
            })
        }
    }

    const TOKEN: &str = "0123456789abcdef0123456789abcdef";

    async fn start() -> SocketAddr {
        let b = Bridge::bind(BridgeOptions { port: 0, token: TOKEN.into() }).await.unwrap();
        let addr = b.local_addr().unwrap();
        tokio::spawn(b.serve(Arc::new(Fake), std::future::pending()));
        addr
    }

    #[tokio::test]
    async fn binds_loopback_only() {
        let addr = start().await;
        assert!(addr.ip().is_loopback());
        assert!(Bridge::bind(BridgeOptions { port: 0, token: "short".into() }).await.is_err());
    }

    #[tokio::test]
    async fn token_on_every_endpoint() {
        let addr = start().await;
        let c = reqwest::Client::new();
        for path in ["/v1/models", "/metrics", "/nothing"] {
            let r = c.get(format!("http://{addr}{path}")).send().await.unwrap();
            assert_eq!(r.status(), 401, "{path}");
        }
        let r = c.get(format!("http://{addr}/v1/models")).bearer_auth(TOKEN).send().await.unwrap();
        assert_eq!(r.status(), 200);
        assert!(r.headers().get("access-control-allow-origin").is_none());
        let v: Value = r.json().await.unwrap();
        assert_eq!(v["data"][0]["id"], "m");
    }

    #[tokio::test]
    async fn host_header_is_checked() {
        let addr = start().await;
        let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
        s.write_all(
            format!("GET /v1/models HTTP/1.1\r\nHost: evil.example:{}\r\nAuthorization: Bearer {TOKEN}\r\nConnection: close\r\n\r\n", addr.port())
                .as_bytes(),
        )
        .await
        .unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).await.unwrap();
        assert!(out.starts_with("HTTP/1.1 403"), "{out}");
    }

    #[tokio::test]
    async fn chat_plain_and_streaming() {
        let addr = start().await;
        let c = reqwest::Client::new();
        let url = format!("http://{addr}/v1/chat/completions");
        let msg = json!([{"role": "user", "content": "hi"}]);
        let v: Value = c.post(&url).bearer_auth(TOKEN).json(&json!({"messages": msg})).send().await.unwrap().json().await.unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "Hello");
        assert_eq!(v["usage"]["total_tokens"], 4);
        let text = c.post(&url).bearer_auth(TOKEN).json(&json!({"messages": msg, "stream": true})).send().await.unwrap().text().await.unwrap();
        assert!(text.contains("\"content\":\"Hel\""), "{text}");
        assert!(text.trim_end().ends_with("data: [DONE]"), "{text}");
        let r = c.post(&url).bearer_auth(TOKEN).json(&json!({"model": "nope", "messages": msg})).send().await.unwrap();
        assert_eq!(r.status(), 404);
        let r = c.post(&url).bearer_auth(TOKEN).body("{").send().await.unwrap();
        assert_eq!(r.status(), 400);
    }
}
