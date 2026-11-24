//! Versioned rollouts (training plan P4b): one name apps ask for, several versions behind it,
//! and a traffic weight per version, so a new fine-tune gets a small share (a canary) and
//! rollback is moving the weight back. Weights live in Ymir's pool policy (decision 5).
//!
//! ```toml
//! [[inference.rollout]]
//! name = "support"
//! [[inference.rollout.version]]
//! version = 2
//! model = "llama-8b+support-r2"   # any model id in the pool: an adapter, a merged GGUF, a base
//! weight = 95
//! [[inference.rollout.version]]
//! version = 3
//! model = "llama-8b+support-r3"
//! weight = 5
//! ```
//!
//! A request for `support` goes to one version, chosen by weight from a key of the conversation
//! (the synced conversation id, else its opening messages), so one conversation stays on one
//! version while the weights stay the same. `support@3` asks for version 3 whatever its weight
//! (evals, and checking a version before it gets traffic). A version at weight 0 stays in the
//! catalog and on its nodes until an operator retires it, so rollback needs no load. Answers are
//! cached under the version's own model id, so versions never share answer-cache entries.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use weirdnet_agent::remote::infer::v2::{Input, Request};

/// One rollout: a name and its versions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rollout {
    pub name: String,
    #[serde(default, rename = "version", skip_serializing_if = "Vec::is_empty")]
    pub versions: Vec<Version>,
}

/// One version of a rollout and its share of the traffic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Version {
    pub version: u32,
    /// The model id in the pool that serves this version.
    pub model: String,
    /// Relative share of new conversations (0: none, kept for rollback).
    #[serde(default)]
    pub weight: u32,
}

impl Rollout {
    pub fn version(&self, v: u32) -> Option<&Version> {
        self.versions.iter().find(|x| x.version == v)
    }

    pub fn total_weight(&self) -> u64 {
        self.versions.iter().map(|v| u64::from(v.weight)).sum()
    }

    /// Each version's share of traffic in percent (0 when no version has weight).
    pub fn shares(&self) -> Vec<(u32, f64)> {
        let t = self.total_weight();
        self.versions.iter().map(|v| (v.version, if t == 0 { 0.0 } else { 100.0 * f64::from(v.weight) / t as f64 })).collect()
    }

    /// The version for a conversation with stickiness key `key`.
    pub fn pick(&self, key: &[u8]) -> Option<&Version> {
        let total = self.total_weight();
        if total == 0 {
            return None;
        }
        let h = Sha256::new().chain_update(self.name.as_bytes()).chain_update([0]).chain_update(key).finalize();
        let mut point = u64::from_be_bytes(h[..8].try_into().expect("8 bytes")) % total;
        // Versions in ascending order, so adding a newer version moves the fewest conversations.
        let mut sorted: Vec<&Version> = self.versions.iter().collect();
        sorted.sort_by_key(|v| v.version);
        for v in sorted {
            let w = u64::from(v.weight);
            if point < w {
                return Some(v);
            }
            point -= w;
        }
        None
    }
}

/// A name with `@` splits into the rollout and a version number (`support@3`).
pub fn split_versioned(id: &str) -> Option<(&str, u32)> {
    let (name, v) = id.rsplit_once('@')?;
    Some((name, v.parse().ok()?))
}

/// The key that keeps a conversation on one version: the synced conversation, else a hash of
/// its system prompt and first user message, else (a raw prompt) the job id.
pub fn sticky_key(req: &Request) -> Vec<u8> {
    match &req.input {
        Input::Context(c) => [b"c:".as_slice(), c.conversation.as_bytes()].concat(),
        Input::Messages(ms) => {
            let mut h = Sha256::new();
            h.update(b"m:");
            let system = ms.iter().find(|m| matches!(m.role, weirdnet_agent::Role::System));
            let first = ms.iter().find(|m| matches!(m.role, weirdnet_agent::Role::User));
            for m in [system, first].into_iter().flatten() {
                h.update(m.text().as_bytes());
                h.update([0]);
            }
            h.finalize().to_vec()
        }
        _ => [b"j:".as_slice(), req.job_id.as_bytes()].concat(),
    }
}

/// Checks a set of rollouts: names are plain (no `@`) and unique, versions unique per rollout,
/// and a version's model is not another rollout's name.
pub fn validate(rollouts: &[Rollout]) -> Result<()> {
    let mut names = std::collections::HashSet::new();
    for r in rollouts {
        if r.name.is_empty() || r.name.contains('@') || r.name.chars().any(char::is_whitespace) {
            bail!("rollout name `{}`: use a plain name without `@` or spaces", r.name);
        }
        if !names.insert(r.name.as_str()) {
            bail!("rollout `{}` is listed twice", r.name);
        }
        if r.versions.is_empty() {
            bail!("rollout `{}` has no versions", r.name);
        }
        let mut vs = std::collections::HashSet::new();
        for v in &r.versions {
            if !vs.insert(v.version) {
                bail!("rollout `{}` lists version {} twice", r.name, v.version);
            }
            if v.model.is_empty() {
                bail!("{}@{}: no model", r.name, v.version);
            }
        }
    }
    for r in rollouts {
        for v in &r.versions {
            if names.contains(v.model.as_str()) || split_versioned(&v.model).is_some_and(|(n, _)| names.contains(n)) {
                bail!("{}@{}: its model `{}` is a rollout name; name a model in the pool", r.name, v.version, v.model);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use weirdnet_agent::Message;

    fn r(weights: &[(u32, u32)]) -> Rollout {
        Rollout {
            name: "support".into(),
            versions: weights.iter().map(|&(v, w)| Version { version: v, model: format!("m{v}"), weight: w }).collect(),
        }
    }

    #[test]
    fn weights_split_conversations_and_each_stays_put() {
        let ro = r(&[(2, 95), (3, 5)]);
        let mut three = 0;
        for i in 0..2000 {
            let k = format!("conv-{i}");
            let v = ro.pick(k.as_bytes()).unwrap().version;
            assert_eq!(ro.pick(k.as_bytes()).unwrap().version, v, "sticky");
            three += usize::from(v == 3);
        }
        assert!((50..=160).contains(&three), "about 5% on the canary, got {three}/2000");
        assert!(r(&[(2, 0), (3, 0)]).pick(b"x").is_none());
        assert_eq!(r(&[(2, 0), (3, 1)]).pick(b"x").unwrap().version, 3);
    }

    #[test]
    fn rollback_moves_only_the_canary_conversations() {
        let canary = r(&[(2, 90), (3, 10)]);
        let back = r(&[(2, 100), (3, 0)]);
        for i in 0..500 {
            let k = format!("c{i}");
            if canary.pick(k.as_bytes()).unwrap().version == 2 {
                assert_eq!(back.pick(k.as_bytes()).unwrap().version, 2);
            }
        }
    }

    #[test]
    fn a_conversation_keys_on_its_opening_not_its_latest_turn() {
        let mut a = Request::messages(vec![Message::system("be brief"), Message::user("hi")]);
        let k = sticky_key(&a);
        if let Input::Messages(ms) = &mut a.input {
            ms.push(Message::assistant("hello", Vec::new()));
            ms.push(Message::user("more"));
        }
        assert_eq!(sticky_key(&a), k);
        assert_ne!(sticky_key(&Request::messages(vec![Message::user("other")])), k);
    }

    #[test]
    fn names_and_versions_are_checked() {
        assert_eq!(split_versioned("support@3"), Some(("support", 3)));
        assert_eq!(split_versioned("support"), None);
        assert_eq!(split_versioned("a@b"), None);
        assert!(validate(&[r(&[(2, 1)])]).is_ok());
        assert!(validate(&[r(&[(2, 1), (2, 3)])]).is_err());
        assert!(validate(&[r(&[(2, 1)]), r(&[(1, 1)])]).is_err());
        assert!(validate(&[Rollout { name: "a@1".into(), ..r(&[(1, 1)]) }]).is_err());
        let mut own = r(&[(1, 1)]);
        own.versions[0].model = "support".into();
        assert!(validate(&[own]).is_err());
    }
}
