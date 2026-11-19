//! A client for the loopback bridge: what `ymirctl chat` uses, like any OpenAI client would.

use anyhow::{Context, Result, bail};
use futures::StreamExt;
use serde_json::{Value, json};
use weirdnet_backends::SseParser;

use crate::{NodeConfig, Paths};

pub struct BridgeClient {
    base: String,
    token: String,
    http: reqwest::Client,
}

/// A finished streamed answer.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChatOutcome {
    pub model: Option<String>,
    pub text: String,
    pub finish_reason: Option<String>,
    pub cache_hit: bool,
    /// The stand-in that answered while the requested model loads.
    pub standin: Option<String>,
}

impl BridgeClient {
    pub fn new(port: u16, token: impl Into<String>) -> Self {
        let http = reqwest::Client::builder().no_proxy().build().expect("an HTTP client without TLS always builds");
        Self { base: format!("http://127.0.0.1:{port}/v1"), token: token.into(), http }
    }

    /// The bridge of the node at `paths`.
    pub fn for_node(paths: &Paths, cfg: &NodeConfig) -> Result<Self> {
        if !cfg.roles.app {
            bail!("the app role (the loopback bridge) is off in {}", paths.config().display());
        }
        let token = crate::token::load(paths)?.context("no bridge token yet; run `ymirctl setup` or start ymird once")?;
        Ok(Self::new(cfg.app.bridge.port, token))
    }

    pub async fn models(&self) -> Result<Value> {
        let r = self
            .http
            .get(format!("{}/models", self.base))
            .bearer_auth(&self.token)
            .send()
            .await
            .context("the bridge is not reachable; is ymird running?")?;
        check(r).await?.json().await.context("bad /v1/models response")
    }

    /// Streams a chat answer, calling `on_text` with each piece.
    pub async fn chat_stream(
        &self,
        model: Option<&str>,
        messages: Vec<Value>,
        extra: serde_json::Map<String, Value>,
        mut on_text: impl FnMut(&str),
    ) -> Result<ChatOutcome> {
        let mut body = json!({"messages": messages, "stream": true});
        if let Some(m) = model {
            body["model"] = m.into();
        }
        for (k, v) in extra {
            body[k] = v;
        }
        let r = self
            .http
            .post(format!("{}/chat/completions", self.base))
            .bearer_auth(&self.token)
            .json(&body)
            .send()
            .await
            .context("the bridge is not reachable; is ymird running?")?;
        let r = check(r).await?;
        let mut out = ChatOutcome {
            model: r.headers().get("x-weirdnet-model").and_then(|v| v.to_str().ok()).map(str::to_string),
            cache_hit: r.headers().get("x-weirdnet-cache").is_some_and(|v| v == "hit"),
            standin: r.headers().get("x-weirdnet-standin").and_then(|v| v.to_str().ok()).map(str::to_string),
            ..Default::default()
        };
        let mut parser = SseParser::default();
        let mut bytes = r.bytes_stream();
        while let Some(chunk) = bytes.next().await {
            for payload in parser.feed(&chunk.context("the stream broke")?) {
                if payload.trim() == "[DONE]" {
                    return Ok(out);
                }
                let v: Value = serde_json::from_str(&payload).context("bad stream chunk")?;
                if let Some(err) = v.get("error") {
                    bail!("{}", err.get("message").and_then(Value::as_str).unwrap_or("stream error"));
                }
                let choice = &v["choices"][0];
                if let Some(t) = choice["delta"]["content"].as_str().filter(|t| !t.is_empty()) {
                    out.text.push_str(t);
                    on_text(t);
                }
                if let Some(f) = choice["finish_reason"].as_str() {
                    out.finish_reason = Some(f.to_string());
                }
                // On a stream the bridge says on the last chunk whether the cache or a stand-in
                // answered.
                let w = &v["weirdnet"];
                if w["cache_hit"].as_bool() == Some(true) {
                    out.cache_hit = true;
                }
                if let Some(s) = w["standin"].as_str() {
                    out.standin = Some(s.to_string());
                }
            }
        }
        if out.finish_reason.is_none() {
            bail!("the stream ended early");
        }
        Ok(out)
    }
}

async fn check(r: reqwest::Response) -> Result<reqwest::Response> {
    if r.status().is_success() {
        return Ok(r);
    }
    let status = r.status();
    let v: Value = r.json().await.unwrap_or(Value::Null);
    let msg = v["error"]["message"].as_str().unwrap_or("no details");
    bail!("bridge answered {status}: {msg}")
}
