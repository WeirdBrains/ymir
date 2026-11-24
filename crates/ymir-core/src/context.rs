//! Conversations kept by the inference nodes and synced among the user's devices (M5).
//!
//! A conversation is a set of sealed `weirdnet-sync` records of the type
//! `CONTEXT_TYPE` (in [`membership`](crate::membership)):
//!
//! - `{conversation}/meta`: title, system prompt, model, and the deleted flag (a tombstone), as one
//!   last-writer-wins register;
//! - `{conversation}/m/{time}-{id}`: one message each, append-only. An assistant message is written
//!   when its answer completes, under the job's id, so a retry or a resume on another node
//!   writes the same record and the conversation never holds an answer twice.
//! - `{conversation}/train`: the user's training consent ([`Train`], training plan P2), a
//!   last-writer-wins register of its own, so an older client that rewrites `meta` never clears
//!   it. Absent means "not for training"; it is never inferred.
//!
//! An inference node assembles the prompt for a `ContextRef` request from this store. This module
//! is the generic part that belongs in `weirdnet-agent`'s `context` module (Weirdnet lead); it
//! lives here until that exists, behind this one interface.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use weirdnet_agent::Message;

use crate::membership::{CONTEXT_TYPE, Membership};

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Meta {
    #[serde(default)]
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// A tombstone: the conversation is gone and its messages are not shown or used.
    #[serde(default)]
    pub deleted: bool,
    #[serde(default)]
    pub at_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stored {
    /// The record id's message part: sorts by time.
    #[serde(skip)]
    pub id: String,
    pub role: String,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub at_ms: i64,
}

/// The training consent record (`{conversation}/train`). Only a conversation whose latest
/// record says `on` is exported for training (`ymirctl context export --training`); turning it
/// off withdraws consent for later exports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Train {
    #[serde(default)]
    pub on: bool,
    /// When the user set it.
    #[serde(default)]
    pub at_ms: i64,
}

/// A conversation with its raw message values (other apps' extra fields kept), for the
/// training export.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawConversation {
    pub id: String,
    pub meta: Meta,
    pub train: Train,
    /// `(message id, value)` in the order the conversation reads.
    pub messages: Vec<(String, serde_json::Value)>,
}

/// A conversation as read back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conversation {
    pub id: String,
    pub meta: Meta,
    pub messages: Vec<Stored>,
}

impl Conversation {
    /// Newest activity.
    pub fn updated_ms(&self) -> i64 {
        self.messages.last().map_or(self.meta.at_ms, |m| m.at_ms.max(self.meta.at_ms))
    }

    /// The prompt: the system prompt, then the messages through `head` (a message id from
    /// [`Stored::id`]; `None`: all).
    pub fn prompt(&self, head: Option<&str>) -> Option<Vec<Message>> {
        let upto = match head {
            None => self.messages.len(),
            Some(h) => self.messages.iter().position(|m| m.id == h)? + 1,
        };
        let mut out = Vec::new();
        if let Some(s) = self.meta.system.as_deref().filter(|s| !s.is_empty()) {
            out.push(Message::system(s));
        }
        for m in &self.messages[..upto] {
            out.push(match m.role.as_str() {
                "assistant" => Message::assistant(m.text.clone(), Vec::new()),
                "system" => Message::system(m.text.clone()),
                _ => Message::user(m.text.clone()),
            });
        }
        Some(out)
    }
}

/// The conversations this node holds.
#[derive(Clone)]
pub struct ContextStore {
    m: Arc<Membership>,
}

fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

/// Record ids are `/`-separated, so a conversation id may not hold one.
fn check_id(conversation: &str) -> Result<()> {
    if conversation.is_empty() || conversation.contains('/') {
        anyhow::bail!("a conversation id is not empty and has no `/`");
    }
    Ok(())
}

impl ContextStore {
    pub fn new(m: Arc<Membership>) -> Self {
        Self { m }
    }

    /// Appends a message and returns its id (the head after it). `key` makes the write
    /// idempotent: the same key writes the same record (an answer's job id).
    pub fn append(&self, conversation: &str, role: &str, text: &str, model: Option<&str>, key: Option<&str>) -> Result<String> {
        check_id(conversation)?;
        let at = now_ms();
        // A keyed write must land on one id from every node, so the id is the key alone;
        // otherwise time (so messages sort by when they were written) and a random tail.
        let id = match key {
            Some(k) => format!("k-{k}"),
            None => format!("{at:013}-{:08x}", rand_u32()),
        };
        if let Some(k) = key
            && let Some(existing) = self.find_keyed(conversation, k)
        {
            return Ok(existing);
        }
        let msg = Stored { id: String::new(), role: role.into(), text: text.into(), model: model.map(str::to_string), at_ms: at };
        self.m.write_record(CONTEXT_TYPE, &format!("{conversation}/m/{id}"), &serde_json::to_vec(&msg)?)?;
        Ok(id)
    }

    fn find_keyed(&self, conversation: &str, key: &str) -> Option<String> {
        let want = format!("{conversation}/m/k-{key}");
        self.m.records(CONTEXT_TYPE).into_iter().find(|(id, _)| *id == want).map(|_| format!("k-{key}"))
    }

    /// Sets title, system prompt and model (the latest write wins).
    pub fn set_meta(&self, conversation: &str, meta: Meta) -> Result<()> {
        check_id(conversation)?;
        let meta = Meta { at_ms: now_ms(), ..meta };
        self.m.write_record(CONTEXT_TYPE, &format!("{conversation}/meta"), &serde_json::to_vec(&meta)?)
    }

    /// Deletes a conversation: a tombstone, which syncs. Its messages stop being used and read.
    pub fn delete(&self, conversation: &str) -> Result<()> {
        let mut meta = self.load(conversation).map(|c| c.meta).unwrap_or_default();
        meta.deleted = true;
        self.set_meta(conversation, meta)
    }

    /// Sets or withdraws a conversation's training consent. The conversation has to exist here.
    pub fn set_train(&self, conversation: &str, on: bool) -> Result<Train> {
        check_id(conversation)?;
        if self.load(conversation).is_none() {
            anyhow::bail!("no conversation `{conversation}` here");
        }
        let t = Train { on, at_ms: now_ms() };
        self.m.write_record(CONTEXT_TYPE, &format!("{conversation}/train"), &serde_json::to_vec(&t)?)?;
        Ok(t)
    }

    /// A conversation's training consent (off when never set).
    pub fn train(&self, conversation: &str) -> Train {
        let want = format!("{conversation}/train");
        self.m.records(CONTEXT_TYPE).into_iter().find(|(id, _)| *id == want).and_then(|(_, v)| serde_json::from_slice(&v).ok()).unwrap_or_default()
    }

    /// Every live conversation whose consent is on, oldest id first, with raw message values.
    pub fn for_training(&self) -> Vec<RawConversation> {
        let mut all: BTreeMap<String, RawConversation> = BTreeMap::new();
        let mut at: BTreeMap<(String, String), i64> = BTreeMap::new();
        for (id, value) in self.m.records(CONTEXT_TYPE) {
            let mut parts = id.splitn(3, '/');
            let (Some(conv), Some(kind)) = (parts.next(), parts.next()) else { continue };
            let c = all.entry(conv.to_string()).or_insert_with(|| RawConversation {
                id: conv.to_string(),
                meta: Meta::default(),
                train: Train::default(),
                messages: Vec::new(),
            });
            match (kind, parts.next()) {
                ("meta", None) => {
                    if let Ok(m) = serde_json::from_slice(&value) {
                        c.meta = m;
                    }
                }
                ("train", None) => {
                    if let Ok(t) = serde_json::from_slice(&value) {
                        c.train = t;
                    }
                }
                ("m", Some(mid)) => {
                    if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&value) {
                        at.insert((conv.to_string(), mid.to_string()), v.get("at_ms").and_then(|a| a.as_i64()).unwrap_or(0));
                        c.messages.push((mid.to_string(), v));
                    }
                }
                _ => {}
            }
        }
        all.into_values()
            .filter(|c| c.train.on && !c.meta.deleted)
            .map(|mut c| {
                let key = |mid: &String| (at.get(&(c.id.clone(), mid.clone())).copied().unwrap_or(0), mid.clone());
                c.messages.sort_by_key(|(mid, _)| key(mid));
                c
            })
            .collect()
    }

    /// Every live conversation, newest first.
    pub fn list(&self) -> Vec<Conversation> {
        let mut all: Vec<Conversation> = self.read_all().into_values().filter(|c| !c.meta.deleted).collect();
        all.sort_by_key(|c| std::cmp::Reverse(c.updated_ms()));
        all
    }

    /// A live conversation, or `None` (never seen here, or deleted).
    pub fn load(&self, conversation: &str) -> Option<Conversation> {
        self.read_all().remove(conversation).filter(|c| !c.meta.deleted)
    }

    /// Tombstones every conversation whose newest activity is older than `max_age_ms`
    /// (retention). Returns how many.
    pub fn expire(&self, max_age_ms: i64) -> usize {
        let cutoff = now_ms() - max_age_ms;
        let old: Vec<String> = self.list().into_iter().filter(|c| c.updated_ms() < cutoff).map(|c| c.id).collect();
        old.iter().filter(|c| self.delete(c).is_ok()).count()
    }

    fn read_all(&self) -> BTreeMap<String, Conversation> {
        let mut all: BTreeMap<String, Conversation> = BTreeMap::new();
        for (id, value) in self.m.records(CONTEXT_TYPE) {
            let mut parts = id.splitn(3, '/');
            let (Some(conv), Some(kind)) = (parts.next(), parts.next()) else { continue };
            let c = all.entry(conv.to_string()).or_insert_with(|| Conversation { id: conv.to_string(), meta: Meta::default(), messages: Vec::new() });
            match (kind, parts.next()) {
                ("meta", None) => {
                    if let Ok(m) = serde_json::from_slice(&value) {
                        c.meta = m;
                    }
                }
                ("m", Some(mid)) => {
                    if let Ok(mut m) = serde_json::from_slice::<Stored>(&value) {
                        m.id = mid.to_string();
                        c.messages.push(m);
                    }
                }
                _ => {}
            }
        }
        for c in all.values_mut() {
            // By time written, then id.
            c.messages.sort_by(|a, b| (a.at_ms, &a.id).cmp(&(b.at_ms, &b.id)));
        }
        all
    }
}

fn rand_u32() -> u32 {
    let mut b = [0u8; 4];
    let _ = getrandom::fill(&mut b);
    u32::from_le_bytes(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (tempfile::TempDir, ContextStore) {
        let dir = tempfile::tempdir().unwrap();
        let m = Arc::new(Membership::create(&dir.path().join("pool"), "t").unwrap());
        (dir, ContextStore::new(m))
    }

    #[test]
    fn a_conversation_reads_back_in_order_with_its_system_prompt() {
        let (_d, s) = store();
        s.set_meta("c1", Meta { title: "Hello".into(), system: Some("Be brief.".into()), ..Default::default() }).unwrap();
        let first = s.append("c1", "user", "hi", None, None).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(3));
        s.append("c1", "assistant", "hello", Some("alpha"), Some("job-1")).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(3));
        s.append("c1", "user", "again", None, None).unwrap();
        let c = s.load("c1").unwrap();
        assert_eq!(c.meta.title, "Hello");
        assert_eq!(c.messages.iter().map(|m| m.text.as_str()).collect::<Vec<_>>(), ["hi", "hello", "again"]);
        let p = c.prompt(None).unwrap();
        assert_eq!(p.len(), 4);
        // Through a head: the system prompt and the first message only.
        assert_eq!(c.prompt(Some(&first)).unwrap().len(), 2);
        assert!(c.prompt(Some("nope")).is_none());
        assert_eq!(s.list().len(), 1);
    }

    #[test]
    fn a_keyed_write_lands_once() {
        let (_d, s) = store();
        s.append("c", "user", "q", None, None).unwrap();
        let a = s.append("c", "assistant", "answer", None, Some("j")).unwrap();
        let b = s.append("c", "assistant", "answer", None, Some("j")).unwrap();
        assert_eq!(a, b);
        assert_eq!(s.load("c").unwrap().messages.len(), 2);
    }

    #[test]
    fn a_tombstone_hides_a_conversation_and_retention_writes_them() {
        let (_d, s) = store();
        s.append("gone", "user", "x", None, None).unwrap();
        s.append("kept", "user", "y", None, None).unwrap();
        s.delete("gone").unwrap();
        assert!(s.load("gone").is_none());
        assert_eq!(s.list().len(), 1);
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert_eq!(s.expire(5), 1);
        assert!(s.list().is_empty());
        assert!(s.append("a/b", "user", "z", None, None).is_err());
    }

    #[test]
    fn training_consent_is_its_own_record_and_off_by_default() {
        let (_d, s) = store();
        s.set_meta("c", Meta { title: "T".into(), ..Default::default() }).unwrap();
        s.append("c", "user", "q", None, None).unwrap();
        s.append("other", "user", "q2", None, None).unwrap();
        assert!(!s.train("c").on);
        assert!(s.for_training().is_empty());
        assert!(s.set_train("missing", true).is_err());
        s.set_train("c", true).unwrap();
        // A later meta write (an older client renaming the chat) does not clear it.
        s.set_meta("c", Meta { title: "Renamed".into(), ..Default::default() }).unwrap();
        let t = s.for_training();
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].id, "c");
        assert_eq!(t[0].messages.len(), 1);
        // Withdrawn, then deleted: neither is exported.
        s.set_train("c", false).unwrap();
        assert!(s.for_training().is_empty());
        s.set_train("c", true).unwrap();
        s.delete("c").unwrap();
        assert!(s.for_training().is_empty());
    }

    #[test]
    fn conversations_survive_a_restart_and_sequence_numbers_carry_on() {
        let dir = tempfile::tempdir().unwrap();
        let pool = dir.path().join("pool");
        {
            let s = ContextStore::new(Arc::new(Membership::create(&pool, "t").unwrap()));
            s.append("c", "user", "before", None, None).unwrap();
        }
        let m = Arc::new(Membership::open(&pool).unwrap().unwrap());
        let s = ContextStore::new(m);
        assert_eq!(s.load("c").unwrap().messages.len(), 1);
        std::thread::sleep(std::time::Duration::from_millis(3));
        s.append("c", "user", "after", None, None).unwrap();
        assert_eq!(s.load("c").unwrap().messages.len(), 2);
    }
}
