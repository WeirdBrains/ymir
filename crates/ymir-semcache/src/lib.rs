//! Answer cache seam for an inference node.
//!
//! To be extracted into `weirdnet-semcache`. [`ExactCache`] is a placeholder that only answers
//! identical deterministic requests; semantic matching comes with the real crate.

use std::collections::HashMap;
use std::sync::Mutex;

use ymir_infer::{ChatRequest, Event};

/// What an inference node asks before and after routing a prompt.
pub trait AnswerCache: Send + Sync {
    /// A stored answer for this request on `model`, as frames to replay.
    fn get(&self, model: &str, req: &ChatRequest) -> Option<Vec<Event>>;
    /// Offers a complete answer for storage; the cache decides whether to keep it.
    fn put(&self, model: &str, req: &ChatRequest, answer: Vec<Event>);
}

/// Caches nothing.
pub struct NoCache;

impl AnswerCache for NoCache {
    fn get(&self, _: &str, _: &ChatRequest) -> Option<Vec<Event>> {
        None
    }
    fn put(&self, _: &str, _: &ChatRequest, _: Vec<Event>) {}
}

/// A use counter plus entries keyed by model and request, each with its last use.
type Entries = (u64, HashMap<String, (u64, Vec<Event>)>);

/// Exact-match, in-memory, least-recently-used cache for deterministic requests.
pub struct ExactCache {
    max_entries: usize,
    inner: Mutex<Entries>,
}

impl ExactCache {
    pub fn new(max_entries: usize) -> Self {
        Self { max_entries: max_entries.max(1), inner: Mutex::new((0, HashMap::new())) }
    }

    fn key(model: &str, req: &ChatRequest) -> Option<String> {
        if !req.deterministic() {
            return None;
        }
        Some(format!("{model}\u{0}{}\u{0}{}", serde_json::Value::Array(req.messages.clone()), serde_json::Value::Object(req.params.clone())))
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).1.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl AnswerCache for ExactCache {
    fn get(&self, model: &str, req: &ChatRequest) -> Option<Vec<Event>> {
        let key = Self::key(model, req)?;
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.0 += 1;
        let tick = g.0;
        let hit = g.1.get_mut(&key)?;
        hit.0 = tick;
        Some(hit.1.clone())
    }

    fn put(&self, model: &str, req: &ChatRequest, answer: Vec<Event>) {
        let Some(key) = Self::key(model, req) else { return };
        let complete = answer.iter().any(|e| matches!(e, Event::Done { finish_reason: Some(r), .. } if r == "stop" || r == "length"));
        if !complete {
            return;
        }
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.0 += 1;
        let tick = g.0;
        if g.1.len() >= self.max_entries
            && !g.1.contains_key(&key)
            && let Some(oldest) = g.1.iter().min_by_key(|(_, v)| v.0).map(|(k, _)| k.clone())
        {
            g.1.remove(&oldest);
        }
        g.1.insert(key, (tick, answer));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(temp: f64, text: &str) -> ChatRequest {
        let mut r = ChatRequest::user(None, text);
        r.params.insert("temperature".into(), temp.into());
        r
    }

    fn answer() -> Vec<Event> {
        vec![Event::Delta { offset: 0, text: "hi".into() }, Event::Done { finish_reason: Some("stop".into()), usage: None }]
    }

    #[test]
    fn only_deterministic_hits() {
        let c = ExactCache::new(2);
        c.put("m", &req(0.7, "a"), answer());
        assert!(c.is_empty());
        c.put("m", &req(0.0, "a"), answer());
        assert!(c.get("m", &req(0.0, "a")).is_some());
        assert!(c.get("other", &req(0.0, "a")).is_none());
        assert!(c.get("m", &req(0.0, "b")).is_none());
    }

    #[test]
    fn evicts_least_recent() {
        let c = ExactCache::new(2);
        c.put("m", &req(0.0, "a"), answer());
        c.put("m", &req(0.0, "b"), answer());
        c.get("m", &req(0.0, "a"));
        c.put("m", &req(0.0, "c"), answer());
        assert!(c.get("m", &req(0.0, "a")).is_some());
        assert!(c.get("m", &req(0.0, "b")).is_none());
    }

    #[test]
    fn ignores_incomplete() {
        let c = ExactCache::new(2);
        c.put("m", &req(0.0, "a"), vec![Event::Delta { offset: 0, text: "h".into() }]);
        assert!(c.is_empty());
    }
}
