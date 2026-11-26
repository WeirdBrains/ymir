//! A synced job board on in-memory Weirdnet nodes: linked full devices of one test user, each
//! granted the model role and the job type, with a lease voter.

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use weirdnet_agent::jobs::{JobPolicy as BoardPolicy, SyncHandle};
use weirdnet_net::{Keypair, Multiaddr, Node};
use weirdnet_sync::{DeviceClass, Leases, Roster, SyncState};
use ymir_core::train_board::BoardJobStore;
use ymir_core::train_job::{JOB_KIND, JOB_ROLE};

type SyncNode = Node<Vec<u8>, Vec<u8>>;
type Book = Arc<Mutex<HashMap<String, Vec<Multiaddr>>>>;

pub struct Dev {
    pub st: Arc<Mutex<SyncState>>,
    pub node: Arc<SyncNode>,
    pub peer: String,
    /// Where this device's leases dial the others; emptied to cut it off.
    pub book: Book,
    pub store: Arc<BoardJobStore>,
}

/// `n` linked full devices of one user, each granted the model role and the job type, with a
/// voter, on memory nodes that know each other's addresses.
pub fn devices(labels: &[&str], claim_ms: u64, takeover_ms: u64) -> Vec<Dev> {
    // The identity seed is the user key's secret (a test seed).
    let seed = [7u8; 32];
    let user = Keypair::ed25519_from_bytes(seed).unwrap();
    let made: Vec<(Arc<Mutex<SyncState>>, Keypair)> = labels
        .iter()
        .map(|l| {
            let k = Keypair::generate_ed25519();
            let peer = k.public().to_peer_id().to_string();
            let cert = Roster::self_certify(&weirdnet_net::WEIRDNET, &user, &peer, l, 1).unwrap();
            let st = SyncState::new(seed.to_vec(), user.public(), cert, Some(&k), vec![JOB_KIND.to_string()]).unwrap();
            (Arc::new(Mutex::new(st)), k)
        })
        .collect();
    let certs: Vec<_> = made.iter().map(|(s, _)| s.lock().unwrap().our_cert.clone()).collect();
    let chains: Vec<_> = made.iter().map(|(s, _)| s.lock().unwrap().roster.chain.clone()).collect();
    // Newer than the grant each state holds for itself from its creation.
    let at = now() as i64 + 60_000;
    for (s, _) in &made {
        let mut st = s.lock().unwrap();
        for c in &certs {
            st.roster.upsert(c.clone());
        }
        for ch in &chains {
            st.roster.chain.merge(ch);
        }
        for c in &certs {
            st.grant(&c.peer_id, DeviceClass::Full, &[JOB_ROLE.to_string()], &[JOB_KIND.to_string()], at).unwrap();
        }
        st.set_voter(Some(
            BoardPolicy::for_role(JOB_ROLE)
                .with_claim(Duration::from_millis(claim_ms))
                .with_takeover_after(Duration::from_millis(takeover_ms))
                .voter(JOB_KIND),
        ));
    }
    let nodes: Vec<(Arc<Mutex<SyncState>>, Arc<SyncNode>)> = made
        .into_iter()
        .map(|(s, k)| {
            let node = weirdnet_sync::start_node(k, weirdnet_sync::node_config(true), s.clone()).unwrap();
            (s, Arc::new(node))
        })
        .collect();
    let all: HashMap<String, Vec<Multiaddr>> =
        nodes.iter().map(|(_, n)| (n.peer_id().to_string(), n.reachable_addrs().into_iter().map(|a| a.parse().unwrap()).collect())).collect();
    nodes
        .into_iter()
        .map(|(st, node)| {
            let peer = node.peer_id().to_string();
            let book: Book = Arc::new(Mutex::new(all.clone()));
            let b = book.clone();
            let leases = Leases::new(node.clone(), st.clone()).with_addrs(move |p| b.lock().unwrap().get(p).cloned().unwrap_or_default());
            let store = Arc::new(BoardJobStore::open(&SyncHandle::from_leases(leases), peer.clone()).unwrap());
            Dev { st, node, peer, book, store }
        })
        .collect()
}

/// Takes device `i` off the network. Pausing its node alone is not enough (a paused node still
/// dials out to the addresses it remembers), so every node pauses and all but `i` listen again on
/// fresh memory addresses, which only the others learn.
pub fn isolate(devs: &[Dev], i: usize) {
    devs[i].book.lock().unwrap().clear();
    for d in devs {
        d.node.pause().unwrap();
    }
    let rest: Vec<&Dev> = devs.iter().enumerate().filter(|(k, _)| *k != i).map(|(_, d)| d).collect();
    for d in &rest {
        d.node.resume().unwrap();
    }
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while rest.iter().any(|d| d.node.reachable_addrs().is_empty()) {
        assert!(std::time::Instant::now() < deadline, "the nodes listen again");
        std::thread::sleep(Duration::from_millis(50));
    }
    let fresh: HashMap<String, Vec<Multiaddr>> =
        rest.iter().map(|d| (d.peer.clone(), d.node.reachable_addrs().into_iter().map(|a| a.parse().unwrap()).collect())).collect();
    for d in &rest {
        *d.book.lock().unwrap() = fresh.clone();
    }
}

/// One pull of the job records from every reachable device by every device.
pub fn sync_all(devs: &[Dev]) {
    for a in devs {
        for b in devs {
            if a.peer == b.peer || a.book.lock().unwrap().is_empty() || b.book.lock().unwrap().is_empty() {
                continue;
            }
            let addrs: Vec<Multiaddr> = b.node.reachable_addrs().into_iter().map(|s| s.parse().unwrap()).collect();
            let _ = weirdnet_sync::pull_from(&mut a.st.lock().unwrap(), &a.node, b.node.peer_id(), &addrs, JOB_KIND);
        }
    }
}

pub fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64
}

/// Syncs every device with every other every 100 ms until dropped, as the daemon's sync loop does.
pub struct Syncer(Arc<std::sync::atomic::AtomicBool>);

impl Drop for Syncer {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

pub fn keep_syncing(devs: Arc<Vec<Dev>>) -> Syncer {
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let s = stop.clone();
    std::thread::spawn(move || {
        while !s.load(std::sync::atomic::Ordering::SeqCst) {
            sync_all(&devs);
            std::thread::sleep(Duration::from_millis(100));
        }
    });
    Syncer(stop)
}
