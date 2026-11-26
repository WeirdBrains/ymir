//! P3b on the synced job board: three linked devices of one user, each on an in-memory Weirdnet
//! node with the model role and a lease voter, carry a training job through `BoardJobStore`
//! (the same `JobStore` calls `train_job` makes): a claim held elsewhere, a checkpoint the
//! next holder resumes from, a release and the owner's resume, a retryable failure with its
//! back-off, and a takeover once the holder's lease lapsed. The same scenario runs on
//! `FileJobStore` for comparison.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use weirdnet_agent::jobs::{JobPolicy as BoardPolicy, SyncHandle};
use weirdnet_net::{Keypair, Multiaddr, Node};
use weirdnet_sync::{DeviceClass, Leases, Roster, SyncState};
use ymir_core::train::Hyper;
use ymir_core::train_board::BoardJobStore;
use ymir_core::train_job::{CheckpointRef, ClaimError, FileJobStore, JOB_KIND, JOB_ROLE, JobPolicy, JobRecord, JobState, JobStore, Needs, TrainJob};

type SyncNode = Node<Vec<u8>, Vec<u8>>;
type Book = Arc<Mutex<HashMap<String, Vec<Multiaddr>>>>;

const CLAIM_MS: u64 = 1_500;
const TAKEOVER_MS: u64 = 1_000;
const BACKOFF_MS: u64 = 500;

struct Dev {
    st: Arc<Mutex<SyncState>>,
    node: Arc<SyncNode>,
    peer: String,
    /// Where this device's leases dial the others; emptied to cut it off.
    book: Book,
    store: BoardJobStore,
}

fn policy() -> JobPolicy {
    JobPolicy {
        claim_ms: CLAIM_MS,
        renew_ms: 300,
        takeover_after_ms: TAKEOVER_MS,
        max_attempts: 4,
        retry_backoff_ms: BACKOFF_MS,
        candidates: Vec::new(),
        role: JOB_ROLE.into(),
    }
}

fn voter_policy() -> BoardPolicy {
    BoardPolicy::for_role(JOB_ROLE).with_claim(Duration::from_millis(CLAIM_MS)).with_takeover_after(Duration::from_millis(TAKEOVER_MS))
}

/// `n` linked full devices of one user, each granted the model role and the job type, with a
/// voter, on memory nodes that know each other's addresses.
fn devices(labels: &[&str]) -> Vec<Dev> {
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
        st.set_voter(Some(voter_policy().voter(JOB_KIND)));
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
            let store = BoardJobStore::open(&SyncHandle::from_leases(leases), peer.clone()).unwrap();
            Dev { st, node, peer, book, store }
        })
        .collect()
}

/// Takes device `i` off the network. Pausing its node alone is not enough (a paused node still
/// dials out to the addresses it remembers), so every node pauses and all but `i` listen again on
/// fresh memory addresses, which only the others learn.
fn isolate(devs: &[Dev], i: usize) {
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
fn sync_all(devs: &[Dev]) {
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

fn job(id: &str, owner: &str) -> JobRecord<TrainJob> {
    let digest = "ab".repeat(32);
    JobRecord {
        id: id.into(),
        kind: JOB_KIND.into(),
        owner: owner.into(),
        base_revision: Some("sha256:0011223344556677".into()),
        dataset: digest.clone(),
        payload: TrainJob {
            base: "alpha".into(),
            base_revision: Some("sha256:0011223344556677".into()),
            from: None,
            dataset_name: "d.jsonl".into(),
            dataset_sha256: "cd".repeat(32),
            manifest_sha256: digest.clone(),
            dataset_root: "ef".repeat(32),
            adapter_id: "alpha+ft".into(),
            hyper: Hyper::default(),
            trainer_build: "b1".into(),
            trainer_interface: 2,
            allow_unknown_license: false,
            needs: Needs::default(),
        },
        state: JobState::Queued,
        lease: None,
        checkpoint: None,
        attempts: 0,
        policy: policy(),
        failure: None,
        result: None,
        created_ms: 1,
        updated_ms: 1,
        not_before_ms: 0,
    }
}

fn cp(seq: u64, holder: &str) -> CheckpointRef {
    CheckpointRef { seq, step: seq * 10, root: format!("{:02x}", seq).repeat(32), bytes: 1_000 + seq, holders: vec![holder.to_string()] }
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64
}

/// Waits until `f` holds on `dev`'s view, syncing in between.
async fn until(devs: &[Dev], dev: usize, what: &str, f: impl Fn(&JobRecord<TrainJob>) -> bool, id: &str) -> JobRecord<TrainJob> {
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    loop {
        sync_all(devs);
        if let Some(j) = devs[dev].store.get(id).unwrap()
            && f(&j)
        {
            return j;
        }
        assert!(std::time::Instant::now() < deadline, "waiting for {what}: {:?}", devs[dev].store.get(id).unwrap());
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Claims until it works (the voters may not have synced the job yet).
async fn claim_when_ready(devs: &[Dev], dev: usize, id: &str) -> ymir_core::train_job::Claim {
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    loop {
        sync_all(devs);
        match devs[dev].store.claim(id, &devs[dev].peer, now()).await {
            Ok(c) => return c,
            Err(e) => assert!(std::time::Instant::now() < deadline, "claiming {id} on {}: {e}", devs[dev].peer),
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_board_job_moves_between_devices_and_resumes_from_its_checkpoint() {
    let devs = devices(&["a", "b", "c"]);
    let (a, b) = (&devs[0], &devs[1]);
    a.store.post(job("j1", &a.peer)).unwrap();
    assert!(a.store.post(job("j1", &b.peer)).is_err(), "a device posts only its own jobs");

    // a holds it; b sees it held.
    let ca = claim_when_ready(&devs, 0, "j1").await;
    assert_eq!(ca.holder, a.peer);
    assert!(a.store.renew(&ca, now()).unwrap());
    a.store.checkpoint(&ca, cp(1, &a.peer), now()).unwrap();
    let seen = until(&devs, 1, "a's claim and checkpoint on b", |j| j.state == JobState::Running && j.checkpoint.is_some(), "j1").await;
    assert_eq!(seen.lease.as_ref().unwrap().holder, a.peer);
    assert_eq!(seen.checkpoint.as_ref().unwrap(), &cp(1, &a.peer), "the checkpoint pointer round-trips (plain hex roots)");
    assert!(matches!(b.store.claim("j1", &b.peer, now()).await, Err(ClaimError::Held { .. })));

    // A retryable failure: queued again after the back-off, then b carries it on from a's checkpoint.
    a.store.fail(&ca, "out of memory", true, now()).await.unwrap();
    assert!(!a.store.renew(&ca, now()).unwrap(), "a let go");
    until(&devs, 1, "the failure on b", |j| j.state == JobState::Queued && j.failure.is_some(), "j1").await;
    let cb = claim_when_ready(&devs, 1, "j1").await;
    assert!(cb.term > ca.term, "a new fencing term");
    let resumed = b.store.get("j1").unwrap().unwrap();
    assert_eq!(resumed.checkpoint.as_ref().map(|c| c.seq), Some(1), "b resumes from a's last checkpoint");
    assert_eq!(resumed.attempts, 2);

    // A release (Ctrl-C) pauses it without spending the attempt; the owner resumes it.
    b.store.release(&cb, now()).await.unwrap();
    let paused = until(&devs, 0, "the pause on a", |j| j.state == JobState::Paused, "j1").await;
    assert!(matches!(b.store.claim("j1", &b.peer, now()).await, Err(ClaimError::NotClaimable(JobState::Paused))));
    a.store.unpause("j1", &a.peer, now()).unwrap();
    until(&devs, 1, "the resume on b", |j| j.state == JobState::Queued, "j1").await;
    let cb = claim_when_ready(&devs, 1, "j1").await;
    b.store.checkpoint(&cb, cp(2, &b.peer), now()).unwrap();
    b.store.finish(&cb, serde_json::json!({ "adapter": "alpha+ft", "steps": 20 }), now()).await.unwrap();
    let done = until(&devs, 0, "done on a", |j| j.state == JobState::Done, "j1").await;
    assert_eq!(done.result, Some(serde_json::json!({ "adapter": "alpha+ft", "steps": 20 })));
    assert_eq!(done.checkpoint.map(|c| c.seq), Some(2));
    assert!(done.attempts <= paused.attempts + 1, "the release gave its attempt back");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_lapsed_board_claim_is_taken_over_after_the_wait() {
    let devs = devices(&["a", "b", "c"]);
    let (a, b) = (&devs[0], &devs[1]);
    a.store.post(job("j2", &a.peer)).unwrap();
    let ca = claim_when_ready(&devs, 0, "j2").await;
    a.store.checkpoint(&ca, cp(3, &a.peer), now()).unwrap();
    until(&devs, 1, "the checkpoint on b", |j| j.checkpoint.is_some(), "j2").await;

    // a drops off the network: it can't reach a voter majority, so its lease lapses.
    isolate(&devs, 0);
    let t0 = std::time::Instant::now();
    let cb = claim_when_ready(&devs, 1, "j2").await;
    assert!(t0.elapsed() >= Duration::from_millis(CLAIM_MS / 2), "not before a's lease could lapse ({:?})", t0.elapsed());
    assert!(cb.term > ca.term);
    assert_eq!(b.store.get("j2").unwrap().unwrap().checkpoint.map(|c| c.seq), Some(3), "b resumes from a's checkpoint");
    // a learns it lost the claim: its writes are refused.
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while a.store.renew(&ca, now()).unwrap() {
        assert!(std::time::Instant::now() < deadline, "a still thinks it holds the job");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(a.store.checkpoint(&ca, cp(4, &a.peer), now()).is_err(), "a stale holder's checkpoint is refused");
    b.store.finish(&cb, serde_json::json!({}), now()).await.unwrap();
}

/// The same moves on the one-node store, with explicit clocks.
#[tokio::test]
async fn the_file_store_makes_the_same_moves() {
    let tmp = tempfile::tempdir().unwrap();
    let s = FileJobStore::open(tmp.path().join("jobs")).unwrap();
    s.post(job("j1", "a")).unwrap();
    let ca = s.claim("j1", "a", 0).await.unwrap();
    s.checkpoint(&ca, cp(1, "a"), 10).unwrap();
    assert!(matches!(s.claim("j1", "b", 100).await, Err(ClaimError::Held { .. })));
    s.fail(&ca, "out of memory", true, 200).await.unwrap();
    assert!(matches!(s.claim("j1", "b", 300).await, Err(ClaimError::TooSoon { .. })));
    let cb = s.claim("j1", "b", 200 + BACKOFF_MS).await.unwrap();
    assert_eq!(s.get("j1").unwrap().unwrap().checkpoint.map(|c| c.seq), Some(1));
    s.release(&cb, 800).await.unwrap();
    s.unpause("j1", "a", 900).unwrap();
    let cb = s.claim("j1", "b", 1_000).await.unwrap();
    // b lapses; c takes over after the wait.
    let late = 1_000 + CLAIM_MS + TAKEOVER_MS;
    assert!(matches!(s.claim("j1", "c", late - 1).await, Err(ClaimError::TooSoon { .. })));
    let cc = s.claim("j1", "c", late).await.unwrap();
    assert!(cc.term > cb.term);
    assert!(s.checkpoint(&cb, cp(2, "b"), late + 1).is_err(), "the stale holder is fenced");
    s.finish(&cc, serde_json::json!({}), late + 2).await.unwrap();
    assert_eq!(s.get("j1").unwrap().unwrap().state, JobState::Done);
}
