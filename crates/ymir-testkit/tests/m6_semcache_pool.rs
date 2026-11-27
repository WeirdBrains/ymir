//! M6 pool-mode semantic cache over Weirdnet (`[inference.cache] mode = "pool"`,
//! `weirdnet-semcache` 0.2.3): three inference nodes linked cert-only serve their shards on
//! `/weirdnet-semcache/shared/1` (`ymir_core::semcache_net`), find each other by the announced
//! app and share entries. An answer stored on one is found from the others; a linked device
//! that is not a member (the seed holder, here) is refused; when a member stops, lookups that
//! needed it are quick degraded misses and the others keep answering.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use weirdnet_agent::remote::RemoteError;
use weirdnet_net::{Keypair, Multiaddr, Node, PeerId, WEIRDNET};
use weirdnet_semcache::pool::{CacheTransport, DialerTransport};
use weirdnet_semcache::{CachedAnswer, SharedCacheReq, SharedCacheResp};
use weirdnet_sync::{Roster, SyncState};
use ymir_core::membership::{Membership, SYNC_TYPE, ServiceRole};
use ymir_core::semantic::{Semantic, SemanticOptions};

type SyncNode = Node<Vec<u8>, Vec<u8>>;

fn addrs(n: &SyncNode) -> Vec<Multiaddr> {
    n.reachable_addrs().into_iter().map(|a| a.parse().unwrap()).collect()
}

fn pull(st: &Arc<Mutex<SyncState>>, node: &SyncNode, from: &SyncNode, typ: &str) {
    let _ = weirdnet_sync::pull_from_shared(st, node, from.peer_id(), &addrs(from), typ);
}

fn cert_only(holder: &Membership, dir: &std::path::Path, label: &str) -> Membership {
    let key = Keypair::generate_ed25519();
    let peer = key.public().to_peer_id().to_string();
    let user = holder.user_keypair().unwrap();
    let cert = Roster::self_certify(&WEIRDNET, &user, &peer, label, 1).unwrap();
    let bundle = holder.service_bundle(&peer, ServiceRole::Inference).unwrap();
    holder.approve(cert.clone()).unwrap();
    Membership::adopt_service(&dir.join(label), key, &user.public().encode_protobuf(), cert, &bundle).unwrap()
}

fn semantic() -> Semantic {
    Semantic::new(SemanticOptions { embedding_model: "embed".into(), similarity: 0.88, ttl: None, max_entries: 64, pool: true })
}

#[tokio::test(flavor = "multi_thread")]
async fn inference_nodes_share_the_semantic_cache_over_weirdnet() {
    let tmp = tempfile::tempdir().unwrap();
    let holder = Membership::create(&tmp.path().join("holder"), "holder").unwrap();
    let ms: Vec<Arc<Membership>> = ["inf1", "inf2", "inf3"].iter().map(|l| Arc::new(cert_only(&holder, tmp.path(), l))).collect();
    let start = |m: &Membership| Arc::new(weirdnet_sync::start_node(m.node_key().clone(), weirdnet_sync::node_config(true), m.state()).unwrap());
    let hn = start(&holder);
    let nodes: Vec<Arc<SyncNode>> = ms.iter().map(|m| start(m)).collect();
    let all: Vec<(&Membership, &Arc<SyncNode>)> = std::iter::once((&holder, &hn)).chain(ms.iter().map(|m| m.as_ref()).zip(nodes.iter())).collect();
    for _ in 0..3 {
        for (m, n) in &all {
            for (_, other) in &all {
                if !Arc::ptr_eq(n, other) {
                    pull(&m.state(), n, other, SYNC_TYPE);
                }
            }
        }
    }
    let hints: Vec<(PeerId, Vec<Multiaddr>)> = all.iter().map(|(_, n)| (n.peer_id(), addrs(n))).collect();

    let sems: Vec<Semantic> = (0..3).map(|_| semantic()).collect();
    let mut running: Vec<_> =
        (0..3).map(|i| Some(ymir_core::semcache_net::start(&sems[i], &nodes[i], &ms[i], &hints, Duration::from_secs(3)).unwrap())).collect();

    // Each finds the other two (the follower looks every 10 s).
    let deadline = Instant::now() + Duration::from_secs(60);
    while !sems.iter().all(|s| s.status().members.len() == 3) {
        assert!(Instant::now() < deadline, "members: {:?}", sems.iter().map(|s| s.status().members).collect::<Vec<_>>());
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    // Stored through node 1 (to the scope's home and replica), found from every node.
    let pool1 = sems[0].pool().unwrap().clone();
    let scope = sems[0].scope("alpha", Some("r1"), None, "ctx");
    let w = pool1.insert(&scope, &[1.0, 0.0, 0.1], CachedAnswer::new("alpha", "shared answer", "stop")).await.unwrap();
    assert!(w.failed.is_empty(), "{w:?}");
    sems[0].settle().await;
    let plan = pool1.plan(&scope).unwrap();
    let owners = [plan.home.clone(), plan.replica.clone().unwrap()];
    for (i, s) in sems.iter().enumerate() {
        let me = nodes[i].peer_id().to_string();
        assert_eq!(s.len(), usize::from(owners.contains(&me)), "node {i} holds the entry only if it owns the scope");
        let hit = s.lookup(&scope, &[0.98, 0.02, 0.1]).await.unwrap_or_else(|| panic!("node {i} missed"));
        assert_eq!(hit.0.text, "shared answer");
    }

    // The seed holder is a linked device but not a member: refused.
    let outsider = DialerTransport::new(ymir_core::remote::net_dialer(&hn, &hints));
    let target = nodes[0].peer_id().to_string();
    match CacheTransport::<CachedAnswer>::call(&outsider, &target, SharedCacheReq::Ping).await {
        Ok(SharedCacheResp::Error { message, .. }) => assert!(message.contains("not-a-cache-member"), "{message}"),
        Err(RemoteError::Refused { .. }) | Err(RemoteError::Closed) => {}
        other => panic!("a non-member reached the cache: {other:?}"),
    }

    // The scope's home stops serving: from a node that isn't the home, a lookup still answers
    // (from the replica, or its own shard) and never waits long.
    let home_i = (0..3).find(|i| nodes[*i].peer_id().to_string() == plan.home).unwrap();
    running[home_i] = None;
    let asker = (0..3).find(|i| *i != home_i).unwrap();
    let t = Instant::now();
    assert_eq!(sems[asker].lookup(&scope, &[1.0, 0.0, 0.1]).await.map(|h| h.0.text), Some("shared answer".into()));
    assert!(t.elapsed() < Duration::from_secs(5), "{:?}", t.elapsed());
    // A scope homed on the stopped node and never stored: a miss, bounded.
    let other_scope = (0..200)
        .map(|n| sems[asker].scope("alpha", Some("r1"), None, &format!("ctx{n}")))
        .find(|s| sems[asker].pool().unwrap().plan(s).is_some_and(|p| p.home == plan.home && !p.owns(&nodes[asker].peer_id().to_string())))
        .unwrap();
    let t = Instant::now();
    assert!(sems[asker].lookup(&other_scope, &[0.0, 1.0, 0.1]).await.is_none());
    assert!(t.elapsed() < Duration::from_secs(5), "{:?}", t.elapsed());
    drop(running);
}
