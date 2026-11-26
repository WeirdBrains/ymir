//! The semantic answer cache (M6): a finished answer stands in for a later prompt that means the
//! same, not only for an identical one. It sits behind the exact cache in
//! [`crate::coordinator::Coordinator::run`] and runs on `weirdnet-semcache`'s [`SemanticCache`].
//!
//! The rules are `rvl-app.py`'s `cacheable()` (see docs/DESIGN.md, "Shared semantic cache"): one
//! user turn after any system messages, text only, no tools, no prefix to continue. The user
//! turn is embedded with the pool's embedding model; everything else that shapes the answer (the
//! system prompt, sampling, seed, response format, adapter scale) goes into the scope with the
//! model, its weights' revision and the embedding model, so answers never cross them.

use std::time::Duration;

use sha2::{Digest, Sha256};
use weirdnet_agent::model::{Content, ContentPart, Message, Role};
use weirdnet_agent::remote::infer::v2::{Input, Request};
use weirdnet_semcache::{CachedAnswer, Lookup, SemanticCache, SemanticConfig, scope_key};

/// `[inference.cache]` settings for the semantic layer.
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticOptions {
    /// The embedding model prompts are embedded with (it must be loaded somewhere: the cache never
    /// cold-starts it).
    pub embedding_model: String,
    /// Cosine similarity a lookup needs (1 − the configured distance).
    pub similarity: f32,
    pub ttl: Option<Duration>,
    pub max_entries: usize,
}

/// A prompt the semantic cache may answer: the text to embed and the context it is scoped by.
#[derive(Debug, Clone, PartialEq)]
pub struct Cacheable {
    pub prompt: String,
    /// Hex SHA-256 of everything else that shapes the answer.
    pub context: String,
}

/// What a miss remembers so the finished answer can be stored.
#[derive(Debug, Clone)]
pub struct Pending {
    pub(crate) model: String,
    pub(crate) scope: String,
    pub(crate) embedding: Vec<f32>,
}

pub struct Semantic {
    pub opts: SemanticOptions,
    cache: SemanticCache<CachedAnswer>,
}

impl Semantic {
    pub fn new(opts: SemanticOptions) -> Self {
        let mut config = SemanticConfig::new(opts.max_entries.max(1)).with_threshold(opts.similarity);
        if let Some(ttl) = opts.ttl {
            config = config.with_ttl(ttl);
        }
        Self { cache: SemanticCache::with_config(config), opts }
    }

    /// The scope for `model` at `revision`, embedded with `embed_revision` of the embedding model.
    pub fn scope(&self, model: &str, revision: Option<&str>, embed_revision: Option<&str>, context: &str) -> String {
        scope_key(&[model, revision.unwrap_or(""), &self.opts.embedding_model, embed_revision.unwrap_or(""), context])
    }

    pub fn lookup(&self, scope: &str, embedding: &[f32]) -> Option<(CachedAnswer, f32)> {
        match self.cache.lookup(scope, embedding) {
            Ok(Lookup::Hit { value, score, .. }) => Some((value, score)),
            Ok(_) => None,
            Err(e) => {
                tracing::debug!("semantic cache lookup: {e}");
                None
            }
        }
    }

    pub fn store(&self, pending: &Pending, model: &str, answer: &CachedAnswer) {
        if pending.model != model {
            return;
        }
        if let Err(e) = self.cache.insert(&pending.scope, &pending.embedding, answer.clone()) {
            tracing::debug!("semantic cache insert: {e}");
        }
    }

    /// Drops every entry (a model's weights changed under its id, or the operator asked).
    pub fn clear(&self) {
        self.cache.clear();
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

fn text_of(content: &Content) -> Option<String> {
    match content {
        Content::Text(t) => Some(t.clone()),
        Content::Parts(parts) => {
            let mut out = String::new();
            for p in parts {
                match p {
                    ContentPart::Text { text, .. } => out.push_str(text),
                    _ => return None,
                }
            }
            Some(out)
        }
    }
}

/// The prompt and context when `req` is one the semantic cache may answer.
pub fn cacheable(req: &Request) -> Option<Cacheable> {
    if req.raw || req.assistant_prefix.is_some() || !req.tools.is_empty() {
        return None;
    }
    let Input::Messages(messages) = &req.input else { return None };
    let (last, before) = messages.split_last()?;
    if last.role != Role::User || !last.tool_calls.is_empty() || before.iter().any(|m: &Message| m.role != Role::System) {
        return None;
    }
    let prompt = text_of(&last.content)?;
    if prompt.trim().is_empty() {
        return None;
    }
    let mut system = Vec::new();
    for m in before {
        system.push(text_of(&m.content)?);
    }
    let shape = serde_json::json!({
        "system": system,
        "sampling": req.sampling,
        "seed": req.seed,
        "response_format": req.response_format,
        "adapter_scale": req.adapter_scale,
    });
    let context = Sha256::digest(shape.to_string().as_bytes()).iter().map(|b| format!("{b:02x}")).collect();
    Some(Cacheable { prompt, context })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> SemanticOptions {
        SemanticOptions { embedding_model: "embed".into(), similarity: 0.88, ttl: None, max_entries: 8 }
    }

    #[test]
    fn single_user_turn_text_only() {
        let r = Request::messages(vec![Message::system("be brief"), Message::user("what is rust?")]);
        let c = cacheable(&r).unwrap();
        assert_eq!(c.prompt, "what is rust?");
        let two_turns = Request::messages(vec![Message::user("a"), Message::assistant("b", vec![]), Message::user("c")]);
        assert!(cacheable(&two_turns).is_none());
        let mut raw = r.clone();
        raw.raw = true;
        assert!(cacheable(&raw).is_none());
        assert!(cacheable(&Request::messages(vec![Message::user("  ")])).is_none());
    }

    #[test]
    fn context_covers_system_and_sampling() {
        let a = cacheable(&Request::messages(vec![Message::system("x"), Message::user("q")])).unwrap();
        let b = cacheable(&Request::messages(vec![Message::system("y"), Message::user("q")])).unwrap();
        assert_ne!(a.context, b.context);
        let mut hot = Request::messages(vec![Message::system("x"), Message::user("other words")]);
        assert_eq!(cacheable(&hot).unwrap().context, a.context);
        hot.sampling.temperature = Some(1.2);
        assert_ne!(cacheable(&hot).unwrap().context, a.context);
    }

    #[test]
    fn hits_near_misses_far_and_scopes_apart() {
        let s = Semantic::new(opts());
        let scope = s.scope("m", Some("r1"), None, "ctx");
        let p = Pending { model: "m".into(), scope: scope.clone(), embedding: vec![1.0, 0.0, 0.0] };
        s.store(&p, "m", &CachedAnswer::new("m", "hello", "stop"));
        assert_eq!(s.lookup(&scope, &[0.99, 0.05, 0.0]).unwrap().0.text, "hello");
        assert!(s.lookup(&scope, &[0.0, 1.0, 0.0]).is_none());
        assert!(s.lookup(&s.scope("m", Some("r2"), None, "ctx"), &[1.0, 0.0, 0.0]).is_none());
        // An answer from another model (a stand-in) is not stored under the asked one.
        s.store(&Pending { embedding: vec![0.0, 0.0, 1.0], ..p }, "standin", &CachedAnswer::new("standin", "x", "stop"));
        assert_eq!(s.len(), 1);
    }
}
