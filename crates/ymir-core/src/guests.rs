//! Sharing a pool with another Weirdnet identity (weirdnet-agent 0.5.9 guests).
//!
//! The pool owner records a [`GuestGrant`] for an external user id in the synced `ymir.pool`
//! type (record id `guest/<user_id>`), so every inference node reads the same list. A guest
//! device presents its own user's certificate plus that user's public key (a guest credential,
//! see `weirdnet_agent::serve::GuestAuth`); the node verifies it cryptographically, then asks
//! [`policy`], which reads the grants live on every check: revoking a grant (`unshare`) refuses
//! the next request and ends live catalog subscriptions within `catalog::RECHECK_EVERY`.
//!
//! A guest only ever gets catalog/1 and infer/2 (infer/1): never `ymir.context` conversations
//! (`Input::Context` is refused), wake cards, tasks, admin, models, uploads or training, which all
//! stay on the roster check. See `docs/design/pool-sharing.md`.

use std::sync::Arc;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use weirdnet_agent::pool::PoolShare;
use weirdnet_agent::remote::RemoteError;
use weirdnet_agent::serve::{Guest, GuestPolicy, GuestScope};

use crate::membership::{Membership, SYNC_TYPE};

/// Record id prefix of guest grants in [`SYNC_TYPE`].
pub const GRANT_PREFIX: &str = "guest/";

/// One shared identity: what its devices may use. Unknown fields are ignored, so a newer grant
/// stays readable here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuestGrant {
    /// The guest's Weirdnet user id (32 hex characters).
    pub user_id: String,
    /// The owner's name for them (shown in `ymirctl pool guests`; not part of any check).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub label: String,
    /// Only these devices (peer ids) of the guest; empty: any device the guest user certified.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub peers: Vec<String>,
    /// Only these models (as requested or as routed); empty: every model in the catalog.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<String>,
    /// `catalog` and/or `infer`; empty reads as both.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scopes: Vec<String>,
    /// Unix ms after which the grant no longer admits anyone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<i64>,
    /// When it was granted (Unix ms).
    #[serde(default)]
    pub granted_at: i64,
    /// Set by `unshare` (records are overwritten, never deleted, so the revocation syncs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revoked_at: Option<i64>,
}

/// Whether `s` looks like a Weirdnet user id (32 lowercase hex characters).
pub fn is_user_id(s: &str) -> bool {
    s.len() == 32 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

impl GuestGrant {
    pub fn new(user_id: impl Into<String>, now_ms: i64) -> Self {
        Self {
            user_id: user_id.into(),
            label: String::new(),
            peers: Vec::new(),
            models: Vec::new(),
            scopes: Vec::new(),
            expires_at: None,
            granted_at: now_ms,
            revoked_at: None,
        }
    }

    /// Not revoked and not expired at `now_ms`.
    pub fn is_active(&self, now_ms: i64) -> bool {
        self.revoked_at.is_none() && self.expires_at.is_none_or(|t| now_ms < t)
    }

    fn allows_scope(&self, scope: GuestScope) -> bool {
        self.scopes.is_empty() || self.scopes.iter().any(|s| s == scope.as_str())
    }

    /// Whether `model` (a requested name or the routed one) is in the allowlist.
    pub fn allows_model(&self, model: &str) -> bool {
        self.models.is_empty() || self.models.iter().any(|m| m == model)
    }

    /// The verdict for a verified guest `user_id` on device `peer` asking for `scope` at `now_ms`.
    pub fn admits(&self, user_id: &str, peer: &str, scope: GuestScope, now_ms: i64) -> Result<(), RemoteError> {
        if self.user_id != user_id {
            return Err(RemoteError::refused("not-shared"));
        }
        if self.revoked_at.is_some() {
            return Err(RemoteError::refused("unshared"));
        }
        if self.expires_at.is_some_and(|t| now_ms >= t) {
            return Err(RemoteError::refused("share-expired"));
        }
        if !self.peers.is_empty() && !self.peers.iter().any(|p| p == peer) {
            return Err(RemoteError::refused("guest-peer-not-pinned"));
        }
        if !self.allows_scope(scope) {
            return Err(RemoteError::refused("guest-scope"));
        }
        Ok(())
    }

    /// Checks the fields an operator typed.
    pub fn validate(&self) -> Result<()> {
        if !is_user_id(&self.user_id) {
            bail!("{:?} is not a Weirdnet user id (32 hex characters)", self.user_id);
        }
        for p in &self.peers {
            if p.parse::<weirdnet_net::PeerId>().is_err() {
                bail!("{p:?} is not a peer id");
            }
        }
        for s in &self.scopes {
            if s != "catalog" && s != "infer" {
                bail!("unknown scope {s:?} (catalog or infer)");
            }
        }
        Ok(())
    }
}

/// Every grant recorded in `records` (the `ymir.pool` records), revoked and expired ones included.
pub fn parse_grants(records: &[(String, Vec<u8>)]) -> Vec<GuestGrant> {
    let mut out: Vec<GuestGrant> = records
        .iter()
        .filter(|(id, _)| id.starts_with(GRANT_PREFIX))
        .filter_map(|(id, v)| serde_json::from_slice::<GuestGrant>(v).ok().filter(|g| id[GRANT_PREFIX.len()..] == g.user_id))
        .collect();
    out.sort_by(|a, b| a.user_id.cmp(&b.user_id));
    out
}

/// Every grant this node holds (revoked and expired ones included).
pub fn all_grants(m: &Membership) -> Vec<GuestGrant> {
    parse_grants(&m.records(SYNC_TYPE))
}

/// The grant for `user_id`, if any.
pub fn grant(m: &Membership, user_id: &str) -> Option<GuestGrant> {
    all_grants(m).into_iter().find(|g| g.user_id == user_id)
}

/// Records (or replaces) a grant; it syncs to the pool's nodes with the next pass.
pub fn share(m: &Membership, g: &GuestGrant) -> Result<()> {
    g.validate()?;
    if g.user_id == m.user_id() {
        bail!("{} is this pool's own identity: link the device instead (`ymirctl pool add`)", g.user_id);
    }
    m.write_record(SYNC_TYPE, &format!("{GRANT_PREFIX}{}", g.user_id), &serde_json::to_vec(g)?)
}

/// Revokes `user_id`'s grant (kept as a revoked record so the revocation syncs).
pub fn unshare(m: &Membership, user_id: &str, now_ms: i64) -> Result<GuestGrant> {
    let Some(mut g) = grant(m, user_id).filter(|g| g.revoked_at.is_none()) else { bail!("{user_id} has no active share") };
    g.revoked_at = Some(now_ms);
    m.write_record(SYNC_TYPE, &format!("{GRANT_PREFIX}{user_id}"), &serde_json::to_vec(&g)?)?;
    Ok(g)
}

/// Where grants are read from on every check.
pub type GrantView = Arc<dyn Fn(&str) -> Option<GuestGrant> + Send + Sync>;

/// The live grants of `m`.
pub fn grant_view(m: &Arc<Membership>) -> GrantView {
    let m = m.clone();
    Arc::new(move |user_id: &str| grant(&m, user_id))
}

/// The `GuestPolicy` over `grants`: reads them on every check.
pub fn policy(grants: GrantView) -> GuestPolicy {
    Arc::new(move |guest: &Guest, scope: GuestScope| {
        let g = grants(&guest.user_id).ok_or_else(|| RemoteError::refused("not-shared"))?;
        g.admits(&guest.user_id, &guest.peer, scope, weirdnet_net::now_ms())
    })
}

/// The share link for a guest: the pool owner, an optional name, the inference nodes with their
/// addresses, and the grant's expiry. Dial info only; the grant is the authority.
pub fn pool_share(owner: &str, name: Option<&str>, nodes: &[(String, Vec<String>)], expires_at: Option<i64>) -> Result<PoolShare> {
    if nodes.is_empty() {
        bail!("no inference node to share (this node's role or a linked inference node is needed)");
    }
    let mut s = PoolShare::new(owner);
    if let Some(n) = name.filter(|n| !n.is_empty()) {
        s = s.with_name(n);
    }
    for (peer, addrs) in nodes {
        s = s.with_node(peer.clone(), addrs.clone());
    }
    if let Some(t) = expires_at {
        s = s.with_expires_at(t);
    }
    s.validate().map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(s)
}

/// Parses an expiry: Unix ms, or a duration from `now_ms` like `30m`, `12h`, `7d`.
pub fn parse_expiry(s: &str, now_ms: i64) -> Result<i64> {
    let s = s.trim();
    let (num, mult) = match s.chars().last() {
        Some('m') => (&s[..s.len() - 1], 60_000),
        Some('h') => (&s[..s.len() - 1], 3_600_000),
        Some('d') => (&s[..s.len() - 1], 86_400_000),
        Some('w') => (&s[..s.len() - 1], 7 * 86_400_000),
        _ => (s, 0),
    };
    let n: i64 = num.parse().map_err(|_| anyhow::anyhow!("{s:?}: an expiry is 30m, 12h, 7d, 2w or Unix ms"))?;
    if n <= 0 {
        bail!("{s:?}: the expiry has to be in the future");
    }
    Ok(if mult == 0 { n } else { now_ms + n * mult })
}

#[cfg(test)]
mod tests {
    use super::*;

    const U: &str = "0123456789abcdef0123456789abcdef";
    const P: &str = "12D3KooWDpJ7As7BWAwRMfu1VU2WCqNjvq387JEYKDBj4kx6nXTN";

    #[test]
    fn grants_round_trip_and_ignore_unknown_fields() {
        let mut g = GuestGrant::new(U, 5);
        g.label = "morph".into();
        g.models = vec!["small".into()];
        let v = serde_json::to_vec(&g).unwrap();
        assert_eq!(serde_json::from_slice::<GuestGrant>(&v).unwrap(), g);
        let newer = format!(r#"{{"user_id":"{U}","granted_at":1,"future":true}}"#);
        assert_eq!(serde_json::from_str::<GuestGrant>(&newer).unwrap().user_id, U);
        // Minimal grants stay minimal on the wire.
        assert!(!String::from_utf8(serde_json::to_vec(&GuestGrant::new(U, 1)).unwrap()).unwrap().contains("peers"));
    }

    #[test]
    fn only_records_under_their_own_id_count() {
        let g = GuestGrant::new(U, 1);
        let other = "ffffffffffffffffffffffffffffffff";
        let recs = vec![
            (format!("guest/{U}"), serde_json::to_vec(&g).unwrap()),
            (format!("guest/{other}"), serde_json::to_vec(&g).unwrap()),
            ("something/else".into(), b"{}".to_vec()),
            (format!("guest/{other}x"), b"not json".to_vec()),
        ];
        assert_eq!(parse_grants(&recs), vec![g]);
    }

    #[test]
    fn matching_scopes_expiry_revocation_and_pinning() {
        let mut g = GuestGrant::new(U, 1);
        assert!(g.admits(U, P, GuestScope::Infer, 10).is_ok());
        assert!(g.admits(U, P, GuestScope::Catalog, 10).is_ok());
        assert!(g.admits("ffffffffffffffffffffffffffffffff", P, GuestScope::Infer, 10).is_err());

        g.scopes = vec!["catalog".into()];
        assert!(g.admits(U, P, GuestScope::Catalog, 10).is_ok());
        assert!(g.admits(U, P, GuestScope::Infer, 10).is_err());
        g.scopes.clear();

        g.peers = vec![P.into()];
        assert!(g.admits(U, P, GuestScope::Infer, 10).is_ok());
        assert!(g.admits(U, "12D3KooWother", GuestScope::Infer, 10).is_err());

        g.expires_at = Some(100);
        assert!(g.is_active(99) && g.admits(U, P, GuestScope::Infer, 99).is_ok());
        assert!(!g.is_active(100) && g.admits(U, P, GuestScope::Infer, 100).is_err());

        g.expires_at = None;
        g.revoked_at = Some(50);
        assert!(!g.is_active(10));
        assert!(g.admits(U, P, GuestScope::Infer, 10).is_err());
    }

    #[test]
    fn model_allowlist() {
        let mut g = GuestGrant::new(U, 1);
        assert!(g.allows_model("anything"));
        g.models = vec!["small".into()];
        assert!(g.allows_model("small"));
        assert!(!g.allows_model("big"));
    }

    #[test]
    fn validation() {
        assert!(GuestGrant::new(U, 1).validate().is_ok());
        assert!(GuestGrant::new("nope", 1).validate().is_err());
        let mut g = GuestGrant::new(U, 1);
        g.scopes = vec!["admin".into()];
        assert!(g.validate().is_err());
        let mut g = GuestGrant::new(U, 1);
        g.peers = vec!["not-a-peer".into()];
        assert!(g.validate().is_err());
    }

    #[test]
    fn share_links_carry_dial_info_and_expiry() {
        let s = pool_share(U, Some("home"), &[(P.into(), vec![format!("/ip4/10.0.0.2/udp/4001/quic-v1/p2p/{P}")])], Some(i64::MAX - 1)).unwrap();
        let token = s.to_token().unwrap();
        assert!(token.starts_with("wnet-pool1:"));
        assert!(s.to_link().unwrap().starts_with("weirdnet-pool://share/"));
        let back = PoolShare::parse(&token).unwrap();
        assert_eq!(back.node_ids(), vec![P.to_string()]);
        assert_eq!(back.owner, U);
        assert_eq!(back.name.as_deref(), Some("home"));
        assert!(pool_share(U, None, &[], None).is_err());
    }

    #[test]
    fn expiries() {
        assert_eq!(parse_expiry("30m", 1000).unwrap(), 1000 + 30 * 60_000);
        assert_eq!(parse_expiry("2d", 0).unwrap(), 2 * 86_400_000);
        assert_eq!(parse_expiry("1900000000000", 0).unwrap(), 1_900_000_000_000);
        assert!(parse_expiry("soon", 0).is_err());
        assert!(parse_expiry("0h", 0).is_err());
    }
}
