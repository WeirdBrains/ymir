//! A client for the loopback bridge: what `ymirctl chat` uses, like any OpenAI client would.

use std::time::{Duration, Instant};

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

/// A streamed chat answer with its timings, for benchmarks.
#[derive(Debug, Clone, Default)]
pub struct TimedChat {
    pub text: String,
    /// From sending the request to the first streamed token (answer or reasoning text).
    pub ttft: Option<Duration>,
    /// From sending the request to the end of the stream.
    pub total: Duration,
    pub prompt_tokens: Option<u64>,
    /// Prompt tokens the node served from its prefix cache.
    pub cached_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    /// The node that answered (`cache` for an answer-cache hit).
    pub served_by: Option<String>,
    pub standin: Option<String>,
    pub finish_reason: Option<String>,
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

    /// Streams a chat answer like [`Self::chat_stream`] and times it: the first token, the end, the
    /// token counts the node reported and which node answered.
    pub async fn chat_timed(&self, model: &str, messages: Vec<Value>, extra: serde_json::Map<String, Value>) -> Result<TimedChat> {
        let mut body = json!({"model": model, "messages": messages, "stream": true, "stream_options": {"include_usage": true}});
        for (k, v) in extra {
            body[k] = v;
        }
        let start = Instant::now();
        let r = self
            .http
            .post(format!("{}/chat/completions", self.base))
            .bearer_auth(&self.token)
            .json(&body)
            .send()
            .await
            .context("the bridge is not reachable; is ymird running?")?;
        let r = check(r).await?;
        let mut out = TimedChat {
            standin: r.headers().get("x-weirdnet-standin").and_then(|v| v.to_str().ok()).map(str::to_string),
            ..Default::default()
        };
        let mut parser = SseParser::default();
        let mut bytes = r.bytes_stream();
        let mut done = false;
        while let Some(chunk) = bytes.next().await {
            for payload in parser.feed(&chunk.context("the stream broke")?) {
                if payload.trim() == "[DONE]" {
                    done = true;
                    break;
                }
                let v: Value = serde_json::from_str(&payload).context("bad stream chunk")?;
                if let Some(err) = v.get("error") {
                    bail!("{}", err.get("message").and_then(Value::as_str).unwrap_or("stream error"));
                }
                let choice = &v["choices"][0];
                let delta = &choice["delta"];
                let answer = delta["content"].as_str().filter(|t| !t.is_empty());
                let thought = ["reasoning_content", "reasoning"].iter().find_map(|k| delta[*k].as_str().filter(|t| !t.is_empty()));
                if (answer.is_some() || thought.is_some()) && out.ttft.is_none() {
                    out.ttft = Some(start.elapsed());
                }
                if let Some(t) = answer {
                    out.text.push_str(t);
                }
                if let Some(f) = choice["finish_reason"].as_str() {
                    out.finish_reason = Some(f.to_string());
                }
                let u = &v["usage"];
                if u.is_object() {
                    out.prompt_tokens = u["prompt_tokens"].as_u64();
                    out.completion_tokens = u["completion_tokens"].as_u64();
                    out.cached_tokens = u["prompt_tokens_details"]["cached_tokens"].as_u64();
                }
                let w = &v["weirdnet"];
                if let Some(s) = w["served_by"].as_str() {
                    out.served_by = Some(s.to_string());
                }
                if let Some(s) = w["standin"].as_str() {
                    out.standin = Some(s.to_string());
                }
            }
            if done {
                break;
            }
        }
        out.total = start.elapsed();
        if !done && out.finish_reason.is_none() {
            bail!("the stream ended early");
        }
        Ok(out)
    }

    /// Embeds `inputs`; returns the vectors' length and how long it took.
    pub async fn embed_timed(&self, model: &str, inputs: &[String]) -> Result<(usize, Duration)> {
        let start = Instant::now();
        let r = self
            .http
            .post(format!("{}/embeddings", self.base))
            .bearer_auth(&self.token)
            .json(&json!({"model": model, "input": inputs}))
            .send()
            .await
            .context("the bridge is not reachable; is ymird running?")?;
        let v: Value = check(r).await?.json().await.context("bad /v1/embeddings response")?;
        let took = start.elapsed();
        let dims = v["data"][0]["embedding"].as_array().map_or(0, Vec::len);
        if dims == 0 {
            bail!("the embedding answer had no vector");
        }
        Ok((dims, took))
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
