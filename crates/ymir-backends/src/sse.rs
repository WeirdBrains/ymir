//! OpenAI-style server-sent events from an engine, decoded into infer frames.

use serde_json::Value;
use ymir_infer::{Event, InferError, Usage};

/// Splits a byte stream into SSE `data:` payloads.
#[derive(Debug, Default)]
pub struct SseParser {
    buf: Vec<u8>,
    data: String,
}

impl SseParser {
    /// Feeds bytes; returns the payloads of every event completed by them.
    pub fn feed(&mut self, bytes: &[u8]) -> Vec<String> {
        self.buf.extend_from_slice(bytes);
        let mut out = Vec::new();
        while let Some(nl) = self.buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=nl).collect();
            let line = String::from_utf8_lossy(&line);
            let line = line.trim_end_matches(['\n', '\r']);
            if line.is_empty() {
                if !self.data.is_empty() {
                    out.push(std::mem::take(&mut self.data));
                }
            } else if let Some(rest) = line.strip_prefix("data:") {
                if !self.data.is_empty() {
                    self.data.push('\n');
                }
                self.data.push_str(rest.strip_prefix(' ').unwrap_or(rest));
            }
        }
        out
    }

    /// Whatever is left when the stream ends without a blank line.
    pub fn finish(&mut self) -> Option<String> {
        let rest = self.feed(b"\n\n");
        rest.into_iter().next()
    }
}

/// Turns `chat.completion.chunk` JSON into frames, tracking byte offsets.
#[derive(Debug, Default)]
pub struct ChunkDecoder {
    offset: u64,
    reasoning_offset: u64,
    finish_reason: Option<String>,
    usage: Option<Usage>,
    done: bool,
}

impl ChunkDecoder {
    /// Decodes one payload. `[DONE]` yields the final `Done` frame.
    pub fn decode(&mut self, payload: &str) -> Result<Vec<Event>, InferError> {
        if payload.trim() == "[DONE]" {
            return Ok(self.finish().into_iter().collect());
        }
        let v: Value = serde_json::from_str(payload).map_err(|e| InferError::Engine(format!("bad chunk: {e}")))?;
        if let Some(err) = v.get("error") {
            let msg = err.get("message").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| err.to_string());
            return Err(InferError::Engine(msg));
        }
        if let Some(u) = v.get("usage").filter(|u| u.is_object()) {
            self.usage = Some(Usage {
                prompt_tokens: u.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0),
                completion_tokens: u.get("completion_tokens").and_then(Value::as_u64).unwrap_or(0),
            });
        }
        let mut out = Vec::new();
        let Some(choice) = v.get("choices").and_then(Value::as_array).and_then(|c| c.first()) else { return Ok(out) };
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            self.finish_reason = Some(reason.to_string());
        }
        // `/v1/completions` chunks carry `text` instead of a delta.
        if let Some(text) = choice.get("text").and_then(Value::as_str).filter(|t| !t.is_empty()) {
            out.push(Event::Delta { offset: self.offset, text: text.to_string() });
            self.offset += text.len() as u64;
        }
        let Some(delta) = choice.get("delta") else { return Ok(out) };
        if let Some(text) = delta.get("reasoning_content").and_then(Value::as_str).filter(|t| !t.is_empty()) {
            out.push(Event::Reasoning { offset: self.reasoning_offset, text: text.to_string() });
            self.reasoning_offset += text.len() as u64;
        }
        if let Some(text) = delta.get("content").and_then(Value::as_str).filter(|t| !t.is_empty()) {
            out.push(Event::Delta { offset: self.offset, text: text.to_string() });
            self.offset += text.len() as u64;
        }
        for (i, call) in delta.get("tool_calls").and_then(Value::as_array).into_iter().flatten().enumerate() {
            let f = call.get("function");
            out.push(Event::ToolCall {
                index: call.get("index").and_then(Value::as_u64).unwrap_or(i as u64) as u32,
                id: call.get("id").and_then(Value::as_str).map(str::to_string),
                name: f.and_then(|f| f.get("name")).and_then(Value::as_str).map(str::to_string),
                arguments: f.and_then(|f| f.get("arguments")).and_then(Value::as_str).unwrap_or("").to_string(),
            });
        }
        Ok(out)
    }

    /// The `Done` frame, once.
    pub fn finish(&mut self) -> Option<Event> {
        if self.done {
            return None;
        }
        self.done = true;
        Some(Event::Done { finish_reason: self.finish_reason.take(), usage: self.usage })
    }

    /// The engine sent a finish reason or `[DONE]`.
    pub fn complete(&self) -> bool {
        self.done || self.finish_reason.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_split_events() {
        let mut p = SseParser::default();
        assert!(p.feed(b"data: {\"a\":").is_empty());
        assert_eq!(p.feed(b"1}\r\n\r\ndata: [DONE]\n\n"), vec!["{\"a\":1}".to_string(), "[DONE]".to_string()]);
    }

    #[test]
    fn decodes_llama_chunks() {
        let mut d = ChunkDecoder::default();
        assert!(d.decode(r#"{"choices":[{"index":0,"delta":{"role":"assistant","content":null}}]}"#).unwrap().is_empty());
        assert_eq!(
            d.decode(r#"{"choices":[{"index":0,"delta":{"content":"Hello"}}]}"#).unwrap(),
            vec![Event::Delta { offset: 0, text: "Hello".into() }]
        );
        assert_eq!(d.decode(r#"{"choices":[{"index":0,"delta":{"content":"!"}}]}"#).unwrap(), vec![Event::Delta { offset: 5, text: "!".into() }]);
        d.decode(r#"{"choices":[{"index":0,"finish_reason":"stop","delta":{}}]}"#).unwrap();
        d.decode(r#"{"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}"#).unwrap();
        assert_eq!(
            d.decode("[DONE]").unwrap(),
            vec![Event::Done { finish_reason: Some("stop".into()), usage: Some(Usage { prompt_tokens: 3, completion_tokens: 2 }) }]
        );
        assert!(d.finish().is_none());
    }

    #[test]
    fn completion_chunks() {
        let mut d = ChunkDecoder::default();
        assert_eq!(d.decode(r#"{"choices":[{"index":0,"text":"Once"}]}"#).unwrap(), vec![Event::Delta { offset: 0, text: "Once".into() }]);
    }

    #[test]
    fn tool_calls_and_errors() {
        let mut d = ChunkDecoder::default();
        let evs = d.decode(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"f","arguments":"{\"x\""}}]}}]}"#).unwrap();
        assert!(matches!(&evs[0], Event::ToolCall { index: 0, name: Some(n), .. } if n == "f"));
        assert!(d.decode(r#"{"error":{"message":"boom"}}"#).is_err());
    }
}
