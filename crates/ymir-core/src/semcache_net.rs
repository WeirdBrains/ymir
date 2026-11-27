//! Pool mode for the semantic answer cache (M6, `[inference.cache] mode = "pool"`): the pool's
//! inference nodes share one cache through `weirdnet-semcache` 0.2.3's [`PooledSemanticCache`]
//! (rendezvous sharding on the scope to a home node and one replica, so a lookup is one call).
//! This module only wires it to Weirdnet; the sharding, the wire format and the fallbacks are
//! the crate's.
//!
//! - **Serving:** each sharing node serves its shards ([`PooledSemanticCache::host`]) on
//!   [`PROTOCOL`] (`/weirdnet-semcache/shared/1`) with [`weirdnet_semcache::pool::serve`],
//!   behind the roster ([`crate::remote::roster_auth`]) narrowed to the current members, so an
//!   app device or a model node can neither read nor fill the cache.
//! - **Finding each other:** a sharing node announces [`app_name`] (it names a hash of the
//!   embedding model, so only nodes embedding with the same model share). Every [`FOLLOW_EVERY`]
//!   the active devices in the roster that announce it, plus this node, are the ring's members
//!   ([`PooledSemanticCache::set_nodes`]); a revoked or silent one drops out.
//! - **One node:** with no other member, every lookup and insert stays in this process (the ring
//!   holds only this node), with no network call.
//! - **Outages never block an answer:** every remote call is bounded (150 ms lookups, 2 s
//!   writes, in the background for the answer), a failed node is skipped for 10 s and its
//!   replica asked instead, and when no owner answers the lookup is a miss
//!   (`PoolLookup::Miss { degraded: true }`): the request goes to a model as usual.

use std::collections::HashSet;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use sha2::{Digest, Sha256};
use weirdnet_agent::remote::{Authorize, Caller, RemoteError};
use weirdnet_agent::serve::{channel, stream_limits};
use weirdnet_net::node::{Message, Node};
use weirdnet_net::{AppBody, AppResponse, PeerId};
use weirdnet_semcache::PROTOCOL;
use weirdnet_semcache::pool::{DialerTransport, PoolConfig};

use crate::membership::{Membership, SyncNode};
use crate::semantic::{Pooled, Semantic};

/// How often the members are looked up again.
pub const FOLLOW_EVERY: Duration = Duration::from_secs(10);

/// What a node sharing its semantic cache announces: `ymir-semcache-pool-` and the first 16 hex
/// digits of the embedding model id's SHA-256.
pub fn app_name(embedding_model: &str) -> String {
    let h: String = Sha256::digest(embedding_model.as_bytes()).iter().take(8).map(|b| format!("{b:02x}")).collect();
    format!("ymir-semcache-pool-{h}")
}

/// The current members, read by the server's authorization.
pub type Members = Arc<RwLock<HashSet<String>>>;

/// `base` (the roster), and only the current members.
pub fn members_auth(base: Arc<dyn Authorize>, members: Members) -> Arc<dyn Authorize> {
    Arc::new(move |caller: &Caller| {
        base.authorize(caller)?;
        if members.read().unwrap_or_else(|e| e.into_inner()).contains(&caller.peer) {
            Ok(())
        } else {
            Err(RemoteError::refused("not-a-cache-member"))
        }
    })
}

/// Replaces the ring's members with `me` and `others`, and the set the server admits. Returns
/// whether anything changed.
pub fn set_members(cache: &Pooled, members: &Members, me: &str, others: impl IntoIterator<Item = String>) -> bool {
    let mut all: HashSet<String> = others.into_iter().collect();
    all.insert(me.to_string());
    let mut cur = members.write().unwrap_or_else(|e| e.into_inner());
    if *cur == all {
        return false;
    }
    let mut sorted: Vec<&String> = all.iter().collect();
    sorted.sort();
    cache.set_nodes(sorted.iter().map(|s| s.as_str()));
    *cur = all;
    true
}

/// Serves this node's shards on [`PROTOCOL`] and announces [`app_name`]. Dropping it stops both.
pub struct SharedCacheServer<Q: Message, R: Message> {
    node: Arc<Node<Q, R>>,
    app: String,
    task: tokio::task::JoinHandle<()>,
}

impl<Q: Message, R: Message> SharedCacheServer<Q, R> {
    pub fn start(node: &Arc<Node<Q, R>>, cache: &Pooled, auth: Arc<dyn Authorize>, app: String) -> Result<Self, RemoteError> {
        let mut limits = stream_limits();
        limits.max_streams = 64;
        let mut incoming = node.accept_streams(PROTOCOL, limits).map_err(|e| RemoteError::Failed { message: e.to_string() })?;
        let host = Arc::new(cache.host());
        let task = tokio::spawn(async move {
            while let Some(stream) = futures::StreamExt::next(&mut incoming).await {
                let caller = Caller::new(stream.peer().to_string());
                let (host, auth) = (host.clone(), auth.clone());
                tokio::spawn(async move {
                    if let Err(e) = weirdnet_semcache::pool::serve(channel(stream), &caller, auth.as_ref(), host.as_ref()).await {
                        tracing::debug!(peer = %caller.peer, "shared semantic cache: {e}");
                    }
                });
            }
        });
        node.serve_app(
            &app,
            Arc::new(|_peer, _req| async {
                Ok(AppResponse::new(200).header("content-type", "text/plain").body(AppBody::bytes(PROTOCOL.as_bytes().to_vec())))
            }),
        )
        .map_err(|e| RemoteError::Failed { message: e.to_string() })?;
        Ok(Self { node: node.clone(), app, task })
    }
}

impl<Q: Message, R: Message> Drop for SharedCacheServer<Q, R> {
    fn drop(&mut self) {
        self.task.abort();
        self.node.stop_serving_app(&self.app);
    }
}

/// What [`start`] leaves running: the server and the member follower.
pub struct PoolCache {
    _server: SharedCacheServer<Vec<u8>, Vec<u8>>,
    follow: tokio::task::JoinHandle<()>,
}

impl Drop for PoolCache {
    fn drop(&mut self) {
        self.follow.abort();
    }
}

/// Turns on pool mode for `semantic` on this inference node: builds the [`Pooled`] cache over
/// its shard, attaches it, serves it and follows the members.
pub fn start(
    semantic: &Semantic,
    node: &Arc<SyncNode>,
    m: &Arc<Membership>,
    hints: &crate::remote::Hints,
    find_wait: Duration,
) -> Result<PoolCache, RemoteError> {
    let me = node.peer_id().to_string();
    let transport = Arc::new(DialerTransport::new(crate::remote::net_dialer(node, hints)));
    let cache = Arc::new(Pooled::new(me.clone(), semantic.shard().clone(), transport, PoolConfig::new()));
    let members: Members = Arc::new(RwLock::new(HashSet::from([me.clone()])));
    let app = app_name(&semantic.opts.embedding_model);
    let server = SharedCacheServer::start(node, &cache, members_auth(crate::remote::roster_auth(m), members.clone()), app.clone())?;
    if !semantic.attach_pool(cache.clone()) {
        return Err(RemoteError::Failed { message: "the semantic cache is already shared".into() });
    }
    let (node, m, hints) = (node.clone(), m.clone(), hints.clone());
    let follow = tokio::spawn(async move {
        loop {
            let peers: Vec<String> = m.roster().active().filter(|c| c.peer_id != me).map(|c| c.peer_id.clone()).collect();
            let mut others = Vec::new();
            for p in peers {
                let Ok(peer) = p.parse::<PeerId>() else { continue };
                let addrs = hints.iter().find(|(h, _)| *h == peer).map(|(_, a)| a.clone()).unwrap_or_default();
                if crate::remote::announces(&node, peer, addrs, find_wait, &app).await {
                    others.push(p);
                }
            }
            if set_members(&cache, &members, &me, others) {
                let ring: Vec<String> = cache.ring().nodes().map(str::to_string).collect();
                tracing::info!(members = ?ring, "shared semantic cache: members changed");
            }
            tokio::time::sleep(FOLLOW_EVERY).await;
        }
    });
    Ok(PoolCache { _server: server, follow })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_app_name_follows_the_embedding_model() {
        assert_eq!(app_name("bge-m3"), app_name("bge-m3"));
        assert_ne!(app_name("bge-m3"), app_name("nomic-embed"));
        assert!(app_name("bge-m3").starts_with("ymir-semcache-pool-") && app_name("bge-m3").len() == "ymir-semcache-pool-".len() + 16);
    }
}
