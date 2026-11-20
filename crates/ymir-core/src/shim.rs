//! `ymir-engine`: the program the supervisor starts for llama.cpp-kind models when a node also
//! serves models on MLX (macOS) or mistral.rs. It takes `llama-server`'s arguments and picks the
//! engine from the model source:
//!
//! | Source | Engine |
//! |---|---|
//! | `mlx:<dir or Hub id>` | `mlx_lm.server`, behind a small proxy |
//! | `mistralrs:<GGUF or dir>` | `mistralrs serve`, behind the same proxy |
//! | anything else (a GGUF path) | `llama-server`, exec'd with the arguments unchanged |
//!
//! The proxy makes the other engines look like `llama-server` to the supervisor and to `ymird`: it
//! answers `/health` (200 once the model is loaded and has produced a token), `/v1/models` (the
//! alias), checks `LLAMA_API_KEY` as a bearer token, rewrites the request's `model` to what the
//! engine expects, and streams everything else through. Each engine still runs as its own process
//! (the proxy and the engine die together), so a crash stays out of `ymird`.
//!
//! Interim: `weirdnet-backends` keys engines by kind with one program per kind, and has no MLX or
//! mistral.rs adapter. Once it has (proposal in docs/DESIGN.md), this goes away.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use axum::Router;
use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use serde_json::{Value, json};

/// Source prefix for models served by MLX-LM.
pub const MLX: &str = "mlx:";
/// Source prefix for models served by mistral.rs.
pub const MISTRALRS: &str = "mistralrs:";
/// The program name of the shim.
pub const PROGRAM: &str = "ymir-engine";
/// Overrides where `ymird` looks for the shim (default: next to the running executable).
pub const SHIM_ENV: &str = "YMIR_ENGINE_SHIM";

/// Shim-only flags (stripped before an engine sees the arguments).
pub const LLAMA_FLAG: &str = "--ymir-llama-server";
pub const MLX_FLAG: &str = "--ymir-mlx-server";
pub const MISTRALRS_FLAG: &str = "--ymir-mistralrs";

/// Where the shim is: `$YMIR_ENGINE_SHIM`, else next to the running executable.
pub fn locate() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os(SHIM_ENV) {
        return Some(PathBuf::from(p)).filter(|p| p.is_file());
    }
    let exe = std::env::current_exe().ok()?;
    let exe = std::fs::canonicalize(&exe).unwrap_or(exe);
    Some(crate::engine::program(exe.parent()?, PROGRAM)).filter(|p| p.is_file())
}

/// Which engine a source string asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Llamacpp,
    Mlx,
    Mistralrs,
}

impl Backend {
    pub fn of(source: &str) -> (Self, &str) {
        if let Some(s) = source.strip_prefix(MLX) {
            (Self::Mlx, s)
        } else if let Some(s) = source.strip_prefix(MISTRALRS) {
            (Self::Mistralrs, s)
        } else {
            (Self::Llamacpp, source)
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Llamacpp => "llama.cpp",
            Self::Mlx => "MLX-LM",
            Self::Mistralrs => "mistral.rs",
        }
    }
}

/// The arguments the supervisor gave, split into what the shim needs.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Args {
    pub model: String,
    pub host: String,
    pub port: u16,
    pub alias: String,
    pub ctx: Option<u32>,
    pub slots: Option<u32>,
    pub embedding: bool,
    pub mmproj: Option<String>,
    pub llama_server: Option<PathBuf>,
    pub mlx_server: Option<PathBuf>,
    pub mistralrs: Option<PathBuf>,
    /// Everything but the shim's own flags, for `llama-server`.
    pub passthrough: Vec<String>,
}

pub fn parse(args: &[String]) -> Args {
    let mut a = Args { host: "127.0.0.1".into(), port: 8080, ..Default::default() };
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        let val = args.get(i + 1).cloned();
        let shim_only = matches!(flag, LLAMA_FLAG | MLX_FLAG | MISTRALRS_FLAG);
        match (flag, &val) {
            (LLAMA_FLAG, Some(v)) => a.llama_server = Some(v.into()),
            (MLX_FLAG, Some(v)) => a.mlx_server = Some(v.into()),
            (MISTRALRS_FLAG, Some(v)) => a.mistralrs = Some(v.into()),
            ("--model" | "-m", Some(v)) => a.model.clone_from(v),
            ("--host", Some(v)) => a.host.clone_from(v),
            ("--port", Some(v)) => a.port = v.parse().unwrap_or(8080),
            ("--alias", Some(v)) => a.alias.clone_from(v),
            ("--ctx-size" | "-c", Some(v)) => a.ctx = v.parse().ok(),
            ("--parallel" | "-np", Some(v)) => a.slots = v.parse().ok(),
            ("--mmproj", Some(v)) => a.mmproj = Some(v.clone()),
            ("--embedding" | "--embeddings", _) => a.embedding = true,
            _ => {}
        }
        if shim_only {
            i += 2;
            continue;
        }
        a.passthrough.push(args[i].clone());
        i += 1;
    }
    if a.alias.is_empty() {
        a.alias.clone_from(&a.model);
    }
    a
}

/// The command for the engine behind the proxy, on loopback `port`, and the `model` value
/// requests must carry (empty: whatever the engine lists first in `/v1/models`).
pub fn upstream(a: &Args, backend: Backend, source: &str, port: u16) -> Result<(PathBuf, Vec<String>, String)> {
    let port = port.to_string();
    match backend {
        Backend::Mlx => {
            if a.embedding {
                bail!("MLX-LM's server has no embeddings endpoint; serve embedding models with backend = \"llamacpp\"");
            }
            let prog = a.mlx_server.clone().context("mlx-lm is not installed (run `ymirctl setup --download` on this Mac)")?;
            // mlx_lm.server stops at 512 tokens unless told otherwise; llama-server runs to the context.
            let max = a.ctx.unwrap_or(32768).to_string();
            let args = vec!["--model".into(), source.into(), "--host".into(), "127.0.0.1".into(), "--port".into(), port, "--max-tokens".into(), max];
            Ok((prog, args, "default_model".into()))
        }
        Backend::Mistralrs => {
            let prog = a.mistralrs.clone().context("this install has no mistral.rs (set backend = \"llamacpp\" for this model)")?;
            let mut args: Vec<String> =
                vec!["serve".into(), "--host".into(), "127.0.0.1".into(), "-p".into(), port, "--no-ui".into(), "--disable-metrics".into()];
            args.extend(["--token-source".into(), "none".into()]);
            if let Some(s) = a.slots {
                args.extend(["--max-seqs".into(), s.to_string()]);
            }
            if let Some(c) = a.ctx {
                args.extend(["--max-model-len".into(), c.to_string()]);
                // Size the paged KV cache to the context, not to a fraction of (unified) memory.
                args.extend(["--kv-cache".into(), format!("{}tokens", u64::from(c) * u64::from(a.slots.unwrap_or(1)))]);
            }
            if a.embedding {
                args.push("embedding".into());
            }
            let p = std::path::Path::new(source);
            if p.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf")) {
                let dir = p.parent().map_or_else(|| ".".into(), |d| d.to_string_lossy().into_owned());
                let file = p.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
                args.extend(["-m".into(), dir, "-f".into(), file]);
                if let Some(m) = &a.mmproj {
                    args.extend([
                        "--mmproj".into(),
                        std::path::Path::new(m).file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default(),
                    ]);
                }
            } else {
                args.extend(["-m".into(), source.into()]);
            }
            Ok((prog, args, String::new()))
        }
        Backend::Llamacpp => bail!("llama.cpp runs directly"),
    }
}

/// Entry point for the `ymir-engine` program.
pub fn main(args: Vec<String>) -> ExitCode {
    let a = parse(&args);
    let (backend, source) = Backend::of(&a.model);
    let source = source.to_string();
    if backend == Backend::Llamacpp {
        return exec_llama(&a);
    }
    let rt = match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("ymir-engine: {e}");
            return ExitCode::FAILURE;
        }
    };
    match rt.block_on(proxy(a, backend, source)) {
        Ok(code) => ExitCode::from(code),
        Err(e) => {
            eprintln!("ymir-engine: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn exec_llama(a: &Args) -> ExitCode {
    let Some(prog) = a.llama_server.clone() else {
        eprintln!("ymir-engine: no llama-server (this install ships none and none is configured)");
        return ExitCode::FAILURE;
    };
    let mut cmd = std::process::Command::new(&prog);
    cmd.args(&a.passthrough);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let e = cmd.exec();
        eprintln!("ymir-engine: {}: {e}", prog.display());
        ExitCode::FAILURE
    }
    #[cfg(not(unix))]
    match cmd.status() {
        Ok(s) => ExitCode::from(u8::try_from(s.code().unwrap_or(1)).unwrap_or(1)),
        Err(e) => {
            eprintln!("ymir-engine: {}: {e}", prog.display());
            ExitCode::FAILURE
        }
    }
}

struct Proxy {
    upstream: String,
    model: std::sync::OnceLock<String>,
    alias: String,
    key: Option<String>,
    http: reqwest::Client,
    ready: AtomicBool,
}

async fn proxy(a: Args, backend: Backend, source: String) -> Result<u8> {
    let port = std::net::TcpListener::bind("127.0.0.1:0")?.local_addr()?.port();
    let (prog, args, model) = upstream(&a, backend, &source, port)?;
    eprintln!("ymir-engine: {} {} {}", backend.name(), prog.display(), args.join(" "));
    let mut child = tokio::process::Command::new(&prog)
        .args(&args)
        .env_remove("LLAMA_API_KEY")
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("starting {}", prog.display()))?;
    let st = Arc::new(Proxy {
        upstream: format!("http://127.0.0.1:{port}"),
        model: if model.is_empty() { std::sync::OnceLock::new() } else { std::sync::OnceLock::from(model) },
        alias: a.alias.clone(),
        key: std::env::var("LLAMA_API_KEY").ok().filter(|k| !k.is_empty()),
        http: reqwest::Client::builder().connect_timeout(Duration::from_secs(5)).build()?,
        ready: AtomicBool::new(false),
    });
    tokio::spawn(warm(st.clone()));
    let app = Router::new().route("/health", get(health)).route("/v1/models", get(models)).fallback(forward).with_state(st);
    let listener = tokio::net::TcpListener::bind((a.host.as_str(), a.port)).await.with_context(|| format!("binding {}:{}", a.host, a.port))?;
    let code = tokio::select! {
        s = child.wait() => {
            let s = s?;
            eprintln!("ymir-engine: {} exited ({s})", backend.name());
            u8::try_from(s.code().unwrap_or(1)).unwrap_or(1).max(1)
        }
        r = axum::serve(listener, app) => { r?; 0 }
        () = terminated() => 0,
    };
    let _ = child.start_kill();
    let _ = tokio::time::timeout(Duration::from_secs(10), child.wait()).await;
    Ok(code)
}

async fn terminated() {
    #[cfg(unix)]
    {
        if let Ok(mut s) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            tokio::select! { _ = s.recv() => {}, _ = tokio::signal::ctrl_c() => {} }
            return;
        }
    }
    let _ = tokio::signal::ctrl_c().await;
}

/// Ready means the engine has loaded the model and produced a token, not just opened its port
/// (MLX-LM answers `/health` before the weights are in).
async fn warm(st: Arc<Proxy>) {
    while st.model.get().is_none() {
        let listed = async {
            let v: Value = st.http.get(format!("{}/v1/models", st.upstream)).timeout(Duration::from_secs(5)).send().await.ok()?.json().await.ok()?;
            v["data"][0]["id"].as_str().map(str::to_string)
        };
        match listed.await {
            Some(id) => {
                let _ = st.model.set(id);
            }
            None => tokio::time::sleep(Duration::from_millis(500)).await,
        }
    }
    let body = json!({"model": st.model.get(), "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "stream": false});
    loop {
        let r = st.http.post(format!("{}/v1/chat/completions", st.upstream)).json(&body).timeout(Duration::from_secs(600)).send().await;
        match r {
            Ok(r) if r.status().is_success() => {
                st.ready.store(true, Ordering::SeqCst);
                eprintln!("ymir-engine: model ready");
                return;
            }
            Ok(r) if r.status().is_client_error() => {
                // An embedding model (or one without a chat template) answers 4xx: it is up.
                st.ready.store(true, Ordering::SeqCst);
                return;
            }
            _ => tokio::time::sleep(Duration::from_millis(500)).await,
        }
    }
}

async fn health(State(st): State<Arc<Proxy>>) -> Response {
    if st.ready.load(Ordering::SeqCst) {
        axum::Json(json!({"status": "ok"})).into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, axum::Json(json!({"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}})))
            .into_response()
    }
}

async fn models(State(st): State<Arc<Proxy>>, headers: HeaderMap) -> Response {
    if let Some(r) = deny(&st, &headers) {
        return r;
    }
    axum::Json(json!({"object": "list", "data": [{"id": st.alias, "object": "model", "owned_by": "ymir"}]})).into_response()
}

fn deny(st: &Proxy, headers: &HeaderMap) -> Option<Response> {
    let key = st.key.as_deref()?;
    let ok = headers.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()).and_then(|v| v.strip_prefix("Bearer ")) == Some(key);
    (!ok).then(|| {
        (StatusCode::UNAUTHORIZED, axum::Json(json!({"error": {"code": 401, "message": "Invalid API Key", "type": "authentication_error"}})))
            .into_response()
    })
}

/// Request bodies can carry images; allow them.
const MAX_BODY: usize = 256 << 20;

async fn forward(State(st): State<Arc<Proxy>>, req: Request) -> Response {
    if let Some(r) = deny(&st, req.headers()) {
        return r;
    }
    let method = req.method().clone();
    let path = req.uri().path_and_query().map_or("/", |p| p.as_str()).to_string();
    let ctype = req.headers().get(header::CONTENT_TYPE).cloned();
    let bytes = match axum::body::to_bytes(req.into_body(), MAX_BODY).await {
        Ok(b) => b,
        Err(e) => return (StatusCode::PAYLOAD_TOO_LARGE, e.to_string()).into_response(),
    };
    let body = match serde_json::from_slice::<Value>(&bytes) {
        Ok(mut v) if v.get("model").is_some() => {
            if let Some(m) = st.model.get() {
                v["model"] = Value::String(m.clone());
            }
            serde_json::to_vec(&v).unwrap_or_else(|_| bytes.to_vec())
        }
        _ => bytes.to_vec(),
    };
    let mut up = st.http.request(method, format!("{}{path}", st.upstream)).body(body);
    if let Some(c) = ctype {
        up = up.header(header::CONTENT_TYPE, c);
    }
    match up.send().await {
        Ok(r) => {
            let status = StatusCode::from_u16(r.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            let ctype = r.headers().get(header::CONTENT_TYPE).and_then(|v| HeaderValue::from_bytes(v.as_bytes()).ok());
            let mut resp = Response::new(Body::from_stream(r.bytes_stream()));
            *resp.status_mut() = status;
            if let Some(c) = ctype {
                resp.headers_mut().insert(header::CONTENT_TYPE, c);
            }
            resp
        }
        Err(e) => (StatusCode::BAD_GATEWAY, axum::Json(json!({"error": {"code": 502, "message": format!("engine: {e}")}}))).into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| (*x).to_string()).collect()
    }

    #[test]
    fn shim_flags_are_stripped_and_llama_args_kept() {
        let a = parse(&v(&[
            "--model",
            "/m/x.gguf",
            "--host",
            "127.0.0.1",
            "--port",
            "4000",
            "--alias",
            "x",
            "--no-webui",
            "--ctx-size",
            "8192",
            "--parallel",
            "2",
            "--ymir-llama-server",
            "/e/llama-server",
            "--ymir-mlx-server",
            "/v/bin/mlx_lm.server",
            "-fa",
            "on",
        ]));
        assert_eq!((a.port, a.alias.as_str(), a.ctx, a.slots), (4000, "x", Some(8192), Some(2)));
        assert_eq!(a.llama_server.as_deref(), Some(std::path::Path::new("/e/llama-server")));
        assert!(!a.passthrough.iter().any(|s| s.starts_with("--ymir") || s.contains("mlx_lm")));
        assert!(a.passthrough.windows(2).any(|w| w == ["-fa", "on"]));
        assert_eq!(Backend::of(&a.model).0, Backend::Llamacpp);
    }

    #[test]
    fn mlx_and_mistralrs_commands() {
        let mut a = parse(&v(&["--model", "mlx:/m/mlx/Qwen3-8B-4bit", "--port", "1", "--ctx-size", "4096", "--ymir-mlx-server", "/v/mlx_lm.server"]));
        let (b, src) = Backend::of(&a.model);
        let (prog, args, model) = upstream(&a, b, src, 9).unwrap();
        assert_eq!(prog, PathBuf::from("/v/mlx_lm.server"));
        assert_eq!(args, v(&["--model", "/m/mlx/Qwen3-8B-4bit", "--host", "127.0.0.1", "--port", "9", "--max-tokens", "4096"]));
        assert_eq!(model, "default_model");
        a.embedding = true;
        assert!(upstream(&a, b, src, 9).is_err());

        let a = parse(&v(&[
            "--model",
            "mistralrs:/m/g.gguf",
            "--mmproj",
            "/m/p.gguf",
            "--parallel",
            "2",
            "--ctx-size",
            "4096",
            "--ymir-mistralrs",
            "/b/mistralrs",
        ]));
        let (b, src) = Backend::of(&a.model);
        let (_, args, _) = upstream(&a, b, src, 9).unwrap();
        for w in [["-m", "/m"], ["-f", "g.gguf"], ["--mmproj", "p.gguf"], ["--max-seqs", "2"], ["--kv-cache", "8192tokens"]] {
            assert!(args.windows(2).any(|x| x == w), "{w:?} in {args:?}");
        }
    }
}
