//! The MCP protocol front end: everything between a client and the tool table.
//!
//! This module is the one place that knows how messages arrive (a stdio loop, a loopback HTTP
//! route) and which crate speaks JSON-RPC (`weirdnet-mcp`). It is kept small and self-contained
//! so it can be swapped: the tools in `mcp.rs` only see `weirdnet_mcp::Tools`.
//!
//! What `weirdnet-mcp` 0.5.3 does not do, and this module fills in (listed in HANDOFF.md as the
//! upstream need): async tools (each call runs on a blocking thread), richer input schemas than
//! flat scalars (patched into `tools/list`), a stdio server loop and a streamable HTTP route.

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
use weirdnet_mcp::{Server, Tools, rpc_err};

/// Where the streamable HTTP endpoint lives.
pub const HTTP_PATH: &str = "/mcp";

/// A server plus the schema additions its tool table cannot express.
pub struct Rpc<T> {
    server: Arc<Server<T>>,
    patch: fn(&str, &mut Value),
}

impl<T> Clone for Rpc<T> {
    fn clone(&self) -> Self {
        Self { server: self.server.clone(), patch: self.patch }
    }
}

impl<T: Tools + Send + Sync + 'static> Rpc<T> {
    /// `patch(tool, input_schema)` may add what `weirdnet_mcp::Param` cannot say (arrays of
    /// objects, enums, `anyOf`).
    pub fn new(server: Server<T>, patch: fn(&str, &mut Value)) -> Self {
        Self { server: Arc::new(server), patch }
    }

    /// Handles one decoded message or batch. Tool calls block (they drive async work with
    /// `Handle::block_on`), so the whole call runs on a blocking thread.
    pub async fn handle(&self, token: Option<String>, msg: Value) -> Option<Value> {
        let (server, patch) = (self.server.clone(), self.patch);
        let mut reply = tokio::task::spawn_blocking(move || server.handle(token.as_deref(), msg)).await.ok()??;
        for r in std::iter::once(&mut reply).flat_map(|v| match v {
            Value::Array(a) => a.iter_mut().collect::<Vec<_>>(),
            other => vec![other],
        }) {
            if let Some(tools) = r.get_mut("result").and_then(|v| v.get_mut("tools")).and_then(Value::as_array_mut) {
                for t in tools {
                    let name = t["name"].as_str().unwrap_or_default().to_string();
                    patch(&name, &mut t["inputSchema"]);
                }
            }
        }
        Some(reply)
    }

    /// Parses a raw line or body and handles it; bad JSON gets a JSON-RPC parse error.
    pub async fn handle_raw(&self, token: Option<String>, raw: &str) -> Option<Value> {
        match serde_json::from_str::<Value>(raw) {
            Ok(v) => self.handle(token, v).await,
            Err(e) => Some(rpc_err(Value::Null, -32700, &format!("parse error: {e}"))),
        }
    }
}

/// Serves newline-delimited JSON-RPC on `input` and `output` until `input` ends and every request
/// has been answered. Requests run concurrently, so a slow chat does not block a ping.
pub async fn serve_stdio<T, R, W>(rpc: Rpc<T>, input: R, mut output: W) -> Result<()>
where
    T: Tools + Send + Sync + 'static,
    R: tokio::io::AsyncRead + Unpin,
    W: tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let writer = tokio::spawn(async move {
        while let Some(line) = rx.recv().await {
            if output.write_all(line.as_bytes()).await.is_err() || output.write_all(b"\n").await.is_err() || output.flush().await.is_err() {
                break;
            }
        }
    });
    let mut tasks = tokio::task::JoinSet::new();
    let mut lines = tokio::io::BufReader::new(input).lines();
    while let Some(line) = lines.next_line().await.context("reading the MCP client's messages")? {
        if line.trim().is_empty() {
            continue;
        }
        let (rpc, tx) = (rpc.clone(), tx.clone());
        tasks.spawn(async move {
            if let Some(r) = rpc.handle_raw(None, &line).await {
                let _ = tx.send(r.to_string());
            }
        });
    }
    while tasks.join_next().await.is_some() {}
    drop(tx);
    writer.await.ok();
    Ok(())
}

/// Parses `--http ADDR` and refuses anything but a loopback address.
pub fn loopback_addr(s: &str) -> Result<SocketAddr> {
    let s = s.trim();
    let addr: SocketAddr = match s.strip_prefix("localhost:") {
        Some(port) => SocketAddr::from(([127, 0, 0, 1], port.parse().with_context(|| format!("bad port in {s}"))?)),
        None => s.parse().with_context(|| format!("`{s}` is not an address like 127.0.0.1:8485"))?,
    };
    if !addr.ip().is_loopback() {
        bail!("the MCP server only listens on loopback addresses (127.0.0.1 or ::1), not {}", addr.ip());
    }
    Ok(addr)
}

/// Compares two tokens in time that depends only on their lengths.
fn same(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// A `Host` or `Origin` value that names this machine (DNS rebinding and browser pages are refused).
fn local_host(h: &str) -> bool {
    let h = h.trim().trim_start_matches("http://").trim_start_matches("https://");
    let host = if let Some(rest) = h.strip_prefix('[') { rest.split(']').next().unwrap_or("") } else { h.split(['/', ':']).next().unwrap_or("") };
    matches!(host, "localhost" | "127.0.0.1" | "::1")
}

/// Serves stateless streamable HTTP (JSON replies) at `/mcp` on `listener`, which must be bound to
/// a loopback address, until `shutdown`. Every request needs `Authorization: Bearer <token>`.
pub async fn serve_http<T>(
    listener: tokio::net::TcpListener,
    rpc: Rpc<T>,
    token: String,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> Result<()>
where
    T: Tools + Send + Sync + 'static,
{
    use axum::http::{HeaderMap, StatusCode, header};
    use axum::response::IntoResponse;
    if !listener.local_addr()?.ip().is_loopback() {
        bail!("refusing to serve MCP on {}: loopback only", listener.local_addr()?);
    }
    let token = Arc::new(token);
    let post = move |headers: HeaderMap, body: String| {
        let (rpc, token) = (rpc.clone(), token.clone());
        async move {
            let host_ok = headers.get(header::HOST).and_then(|v| v.to_str().ok()).is_some_and(local_host);
            let origin_ok = headers.get(header::ORIGIN).and_then(|v| v.to_str().ok()).is_none_or(local_host);
            if !host_ok || !origin_ok {
                return (StatusCode::FORBIDDEN, "this MCP server only answers requests addressed to this machine\n").into_response();
            }
            let presented = headers.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()).and_then(|v| v.strip_prefix("Bearer ")).map(str::trim);
            if !presented.is_some_and(|t| same(t.as_bytes(), token.as_bytes())) {
                return (
                    StatusCode::UNAUTHORIZED,
                    [(header::WWW_AUTHENTICATE, "Bearer")],
                    "missing or wrong bearer token (the bridge token: `ymirctl bridge token`)\n",
                )
                    .into_response();
            }
            match rpc.handle_raw(presented.map(str::to_string), &body).await {
                Some(v) => ([(header::CONTENT_TYPE, "application/json")], v.to_string()).into_response(),
                None => StatusCode::ACCEPTED.into_response(),
            }
        }
    };
    // Stateless: no sessions and no server-initiated stream, so GET and DELETE are not allowed.
    let app = axum::Router::new().route(HTTP_PATH, axum::routing::post(post));
    axum::serve(listener, app).with_graceful_shutdown(shutdown).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_loopback_addresses() {
        assert_eq!(loopback_addr("127.0.0.1:9").unwrap().port(), 9);
        assert!(loopback_addr("[::1]:9").is_ok());
        assert!(loopback_addr("localhost:9").is_ok());
        assert!(loopback_addr("0.0.0.0:9").is_err());
        assert!(loopback_addr("192.0.2.10:9").is_err());
        assert!(loopback_addr("nonsense").is_err());
    }

    #[test]
    fn token_compare() {
        assert!(same(b"abc", b"abc"));
        assert!(!same(b"abc", b"abd"));
        assert!(!same(b"abc", b"abcd"));
    }

    #[test]
    fn hosts_and_origins_on_this_machine() {
        for ok in ["localhost:8485", "127.0.0.1:1", "[::1]:2", "http://localhost:8485", "http://127.0.0.1:9/x"] {
            assert!(local_host(ok), "{ok}");
        }
        for bad in ["evil.example", "evil.example:8485", "http://evil.example", "localhost.evil.example", "192.0.2.1:80"] {
            assert!(!local_host(bad), "{bad}");
        }
    }
}
