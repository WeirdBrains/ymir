//! The semantic answer cache (M6): a finished answer stands in for a later prompt that means the
//! same, not only for an identical one. It sits behind the exact cache in
//! [`crate::coordinator::Coordinator::run`] and runs on `weirdnet-semcache`'s [`SemanticCache`].
//!
//! The rules are `rvl-app.py`'s `cacheable()` (see docs/DESIGN.md, "Shared semantic cache"): one
//! user turn after any system messages, text only, no tools, no prefix to continue. The user
//! turn is embedded with the pool's embedding model; everything else that shapes the answer (the
//! system prompt, sampling, seed, response format, adapter scale) goes into the scope with the
//! model, its weights' revision and the embedding model, so answers never cross them.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use weirdnet_agent::model::{Content, ContentPart, Message, Role};
use weirdnet_agent::remote::infer::v2::{Input, Request};
use weirdnet_semcache::pool::{PoolLookup, PoolStats, PooledSemanticCache};
use weirdnet_semcache::{CachedAnswer, Lookup, SemanticCache, SemanticConfig, scope_key};

/// The pool-mode cache over this node's shards (M6, `weirdnet-semcache` 0.2.3).
pub type Pooled = PooledSemanticCache<CachedAnswer>;

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
    /// `[inference.cache] mode = "pool"`: share the cache with the pool's other inference nodes
    /// (attached by `ymird` once Weirdnet is up; without it the cache stays local).
    pub pool: bool,
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
    /// The entries this node holds: all of them in local mode, its shards in pool mode.
    shard: Arc<SemanticCache<CachedAnswer>>,
    /// Pool mode, once attached ([`Self::attach_pool`]).
    pool: OnceLock<Arc<Pooled>>,
    /// Pool-mode stores running in the background ([`Self::settle`]).
    writes: Arc<tokio::sync::watch::Sender<usize>>,
}

/// The semantic cache in `ymirctl status`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheStatus {
    /// `local`, or `pool` once shared.
    pub mode: String,
    pub embedding_model: String,
    /// Entries this node holds (its shards in pool mode).
    pub entries: usize,
    /// Pool mode: the inference nodes sharing it, this one included.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub members: Vec<String>,
    /// Pool mode: members skipped for now after a failure or timeout.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub down: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stats: Option<PoolCounters>,
}

/// `weirdnet-semcache`'s [`PoolStats`], as Ymir reports them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolCounters {
    pub l1_hits: u64,
    pub shard_hits: u64,
    pub peer_hits: u64,
    pub misses: u64,
    pub degraded: u64,
    pub peer_errors: u64,
    pub timeouts: u64,
    pub skipped: u64,
}

impl From<PoolStats> for PoolCounters {
    fn from(s: PoolStats) -> Self {
        Self {
            l1_hits: s.l1_hits,
            shard_hits: s.shard_hits,
            peer_hits: s.peer_hits,
            misses: s.misses,
            degraded: s.degraded,
            peer_errors: s.peer_errors,
            timeouts: s.timeouts,
            skipped: s.skipped,
        }
    }
}

impl Semantic {
    pub fn new(opts: SemanticOptions) -> Self {
        let mut config = SemanticConfig::new(opts.max_entries.max(1)).with_threshold(opts.similarity);
        if let Some(ttl) = opts.ttl {
            config = config.with_ttl(ttl);
        }
        Self {
            shard: Arc::new(SemanticCache::with_config(config)),
            pool: OnceLock::new(),
            writes: Arc::new(tokio::sync::watch::Sender::new(0)),
            opts,
        }
    }

    /// The entries this node holds, for [`Pooled::new`] (pool mode keeps its shards there).
    pub fn shard(&self) -> &Arc<SemanticCache<CachedAnswer>> {
        &self.shard
    }

    /// Shares the cache through `pool` from now on (built over [`Self::shard`]). Once only;
    /// `false` when a pool was already attached.
    pub fn attach_pool(&self, pool: Arc<Pooled>) -> bool {
        self.pool.set(pool).is_ok()
    }

    pub fn pool(&self) -> Option<&Arc<Pooled>> {
        self.pool.get()
    }

    /// The scope for `model` at `revision`, embedded with `embed_revision` of the embedding model.
    pub fn scope(&self, model: &str, revision: Option<&str>, embed_revision: Option<&str>, context: &str) -> String {
        scope_key(&[model, revision.unwrap_or(""), &self.opts.embedding_model, embed_revision.unwrap_or(""), context])
    }

    /// The most similar stored answer. In pool mode: L1, then the scope's home, then its
    /// replica if the home is unreachable, each bounded (150 ms); an outage is a miss.
    pub async fn lookup(&self, scope: &str, embedding: &[f32]) -> Option<(CachedAnswer, f32)> {
        let Some(pool) = self.pool.get() else {
            return match self.shard.lookup(scope, embedding) {
                Ok(Lookup::Hit { value, score, .. }) => Some((value, score)),
                Ok(_) => None,
                Err(e) => {
                    tracing::debug!("semantic cache lookup: {e}");
                    None
                }
            };
        };
        match pool.lookup(scope, embedding).await {
            Ok(PoolLookup::Hit { value, score, from }) => {
                tracing::debug!(?from, score, "pool semantic cache hit");
                Some((value, score))
            }
            Ok(PoolLookup::Miss { degraded: true, .. }) => {
                tracing::debug!("pool semantic cache: an owner was unreachable; answering from a model");
                None
            }
            Ok(_) => None,
            Err(e) => {
                tracing::debug!("semantic cache lookup: {e}");
                None
            }
        }
    }

    /// Stores a finished answer. In pool mode the write to the scope's owners runs in the
    /// background, so the answer is never held up by it.
    pub fn store(&self, pending: &Pending, model: &str, answer: &CachedAnswer) {
        if pending.model != model {
            return;
        }
        let Some(pool) = self.pool.get() else {
            if let Err(e) = self.shard.insert(&pending.scope, &pending.embedding, answer.clone()) {
                tracing::debug!("semantic cache insert: {e}");
            }
            return;
        };
        let (pool, scope, embedding, answer) = (pool.clone(), pending.scope.clone(), pending.embedding.clone(), answer.clone());
        let writes = self.writes.clone();
        writes.send_modify(|n| *n += 1);
        tokio::spawn(async move {
            match pool.insert(&scope, &embedding, answer).await {
                Ok(w) if !w.failed.is_empty() => tracing::debug!(failed = ?w.failed, "pool semantic cache insert"),
                Ok(_) => {}
                Err(e) => tracing::debug!("semantic cache insert: {e}"),
            }
            writes.send_modify(|n| *n -= 1);
        });
    }

    /// Waits until the background stores (and, in pool mode, their replica writes) are done;
    /// each is bounded by the pool's write timeout.
    pub async fn settle(&self) {
        let mut rx = self.writes.subscribe();
        let _ = rx.wait_for(|n| *n == 0).await;
        if let Some(p) = self.pool.get() {
            p.settle().await;
        }
    }

    /// Drops every entry this node holds (a model's weights changed under its id, or the
    /// operator asked). Other members keep theirs; use [`Self::remove_scope`] for one scope
    /// everywhere.
    pub fn clear(&self) {
        self.shard.clear();
    }

    /// Drops `scope` here and, in pool mode, on every member (bounded). Returns the members that
    /// could not be reached.
    pub async fn remove_scope(&self, scope: &str) -> Vec<String> {
        match self.pool.get() {
            Some(p) => p.remove_scope(scope).await.failed,
            None => {
                self.shard.remove_scope(scope);
                Vec::new()
            }
        }
    }

    pub fn len(&self) -> usize {
        self.shard.len()
    }

    pub fn is_empty(&self) -> bool {
        self.shard.is_empty()
    }

    pub fn status(&self) -> CacheStatus {
        let mut st =
            CacheStatus { mode: "local".into(), embedding_model: self.opts.embedding_model.clone(), entries: self.len(), ..Default::default() };
        if let Some(p) = self.pool.get() {
            st.mode = "pool".into();
            st.members = p.ring().nodes().map(str::to_string).collect();
            st.down = st.members.iter().filter(|m| p.is_down(m)).cloned().collect();
            st.stats = Some(p.stats().into());
        }
        st
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
        SemanticOptions { embedding_model: "embed".into(), similarity: 0.88, ttl: None, max_entries: 8, pool: false }
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

    #[tokio::test]
    async fn hits_near_misses_far_and_scopes_apart() {
        let s = Semantic::new(opts());
        let scope = s.scope("m", Some("r1"), None, "ctx");
        let p = Pending { model: "m".into(), scope: scope.clone(), embedding: vec![1.0, 0.0, 0.0] };
        s.store(&p, "m", &CachedAnswer::new("m", "hello", "stop"));
        assert_eq!(s.lookup(&scope, &[0.99, 0.05, 0.0]).await.unwrap().0.text, "hello");
        assert!(s.lookup(&scope, &[0.0, 1.0, 0.0]).await.is_none());
        assert!(s.lookup(&s.scope("m", Some("r2"), None, "ctx"), &[1.0, 0.0, 0.0]).await.is_none());
        assert_eq!(s.status().mode, "local");
        assert!(s.status().stats.is_none());
        // An answer from another model (a stand-in) is not stored under the asked one.
        s.store(&Pending { embedding: vec![0.0, 0.0, 1.0], ..p }, "standin", &CachedAnswer::new("standin", "x", "stop"));
        assert_eq!(s.len(), 1);
    }
}
