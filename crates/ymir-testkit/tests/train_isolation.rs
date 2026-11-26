//! A cert-only model node granted `["ymir.pool", "ymir.train"]` (what `pool add --service` /
//! `pool link --cert-only` give `ServiceRole::Model`) takes part in the synced training job
//! board: it syncs the job records, votes on claims and claims a job the seed holder posted. It
//! is cryptographically unable to open the user's sealed conversations (`ymir.context`): it
//! holds no epoch key and no `ymir.context` key, a context pull fails (no key to seal it; a peer
//! would refuse it `type-not-granted`),
//! and a sealed context entry handed to it directly does not open.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use weirdnet_agent::jobs::SyncHandle;
use weirdnet_net::{Keypair, Multiaddr, Node, WEIRDNET};
use weirdnet_sync::{DeviceClass, Leases, RecordValue, Roster, SyncState};
use ymir_core::membership::{CONTEXT_TYPE, Membership, ROLE_MODEL, SYNC_TYPE, ServiceRole, TRAIN_TYPE};
use ymir_core::train::Hyper;
use ymir_core::train_board::BoardJobStore;
use ymir_core::train_job::{JOB_KIND, JOB_ROLE, JobPolicy, JobRecord, JobState, JobStore, Needs, TrainJob};

type SyncNode = Node<Vec<u8>, Vec<u8>>;

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64
}

fn policy() -> JobPolicy {
    JobPolicy {
        claim_ms: 2_000,
        renew_ms: 400,
        takeover_after_ms: 1_000,
        max_attempts: 3,
        retry_backoff_ms: 500,
        candidates: Vec::new(),
        role: JOB_ROLE.into(),
    }
}

fn job(id: &str, owner: &str) -> JobRecord<TrainJob> {
    let digest = "ab".repeat(32);
    JobRecord {
        id: id.into(),
        kind: JOB_KIND.into(),
        owner: owner.into(),
        base_revision: None,
        dataset: digest.clone(),
        payload: TrainJob {
            base: "alpha".into(),
            base_revision: None,
            from: None,
            dataset_name: "d.jsonl".into(),
            dataset_sha256: "cd".repeat(32),
            manifest_sha256: digest,
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

fn addrs(n: &SyncNode) -> Vec<Multiaddr> {
    n.reachable_addrs().into_iter().map(|a| a.parse().unwrap()).collect()
}

fn board(m: &Membership, node: &Arc<SyncNode>, other: &Arc<SyncNode>) -> BoardJobStore {
    let (peer, theirs) = (other.peer_id().to_string(), addrs(other));
    let leases = Leases::new(node.clone(), m.state()).with_addrs(move |p| if p == peer { theirs.clone() } else { Vec::new() });
    BoardJobStore::open(&SyncHandle::from_leases(leases), m.peer_id().to_string()).unwrap()
}

fn pull(st: &Arc<Mutex<SyncState>>, node: &SyncNode, from: &SyncNode, typ: &str) -> Option<String> {
    weirdnet_sync::pull_from_shared(st, node, from.peer_id(), &addrs(from), typ).error.map(|e| e.to_string())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_cert_only_model_node_works_the_job_board_but_cannot_open_conversations() {
    let tmp = tempfile::tempdir().unwrap();

    // The seed holder, alone, starts the key chain with a sealed conversation.
    let holder = Membership::create(&tmp.path().join("app"), "app").unwrap();
    holder.write_record(CONTEXT_TYPE, "c1", b"a private conversation").unwrap();

    // The model node, linked cert-only with the model role.
    let key = Keypair::generate_ed25519();
    let mpeer = key.public().to_peer_id().to_string();
    let user = holder.user_keypair().unwrap();
    let cert = Roster::self_certify(&WEIRDNET, &user, &mpeer, "model", 1).unwrap();
    let bundle = holder.service_bundle(&mpeer, ServiceRole::Model).unwrap();
    holder.approve(cert.clone()).unwrap();
    let model = Membership::adopt_service(&tmp.path().join("model"), key, &user.public().encode_protobuf(), cert, &bundle).unwrap();
    assert!(model.is_service());

    // The grant: the pool and the training jobs, never the conversations.
    let g = model.state().lock().unwrap().roster.grant_of(&mpeer).cloned().unwrap();
    assert_eq!(g.class, DeviceClass::Service);
    assert_eq!(g.roles, vec![ROLE_MODEL.to_string()]);
    assert_eq!(g.types, vec![SYNC_TYPE.to_string(), TRAIN_TYPE.to_string()]);

    // The holder votes too (a full device with the model role), so a claim needs both.
    {
        let st = holder.state();
        let mut st = st.lock().unwrap();
        let hp = st.our_cert.peer_id.clone();
        st.grant(
            &hp,
            DeviceClass::Full,
            &[ROLE_MODEL.to_string()],
            &[SYNC_TYPE.to_string(), CONTEXT_TYPE.to_string(), TRAIN_TYPE.to_string()],
            now() as i64 + 60_000,
        )
        .unwrap();
    }
    assert!(holder.enable_train_voter_with(&policy()).unwrap(), "the holder votes");
    assert!(model.enable_train_voter_with(&policy()).unwrap(), "the model node votes");

    let hnode = Arc::new(weirdnet_sync::start_node(holder.node_key().clone(), weirdnet_sync::node_config(true), holder.state()).unwrap());
    let mnode = Arc::new(weirdnet_sync::start_node(model.node_key().clone(), weirdnet_sync::node_config(true), model.state()).unwrap());
    let (hstore, mstore) = (board(&holder, &hnode, &mnode), board(&model, &mnode, &hnode));
    let sync = |typ: &str| {
        let a = pull(&model.state(), &mnode, &hnode, typ);
        let b = pull(&holder.state(), &hnode, &mnode, typ);
        (a, b)
    };
    // Rosters and grants first (each side learns the other's grant), then the jobs.
    for _ in 0..3 {
        sync(SYNC_TYPE);
    }

    // The holder posts a job; it reaches the model node.
    let id = "isolation-1";
    hstore.post(job(id, &holder.peer_id().to_string())).unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    loop {
        let (a, _) = sync(JOB_KIND);
        if mstore.get(id).unwrap().is_some() {
            break;
        }
        assert!(std::time::Instant::now() < deadline, "the job reaches the model node (last pull: {a:?})");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // The model node claims it: both voters (the holder's and its own) grant the lease.
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    let claim = loop {
        sync(JOB_KIND);
        match mstore.claim(id, &mpeer, now()).await {
            Ok(c) => break c,
            Err(e) => {
                assert!(std::time::Instant::now() < deadline, "the model node claims the job: {e:?}");
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
        }
    };
    assert!(mstore.renew(&claim, now()).unwrap(), "the claim holds");
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    loop {
        sync(JOB_KIND);
        let j = hstore.get(id).unwrap().unwrap();
        if j.state == JobState::Running && j.lease.as_ref().is_some_and(|l| l.holder == mpeer) {
            break;
        }
        assert!(std::time::Instant::now() < deadline, "the holder sees the model node's claim: {:?} {:?}", j.state, j.lease);
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    mstore.release(&claim, now()).await.unwrap();

    // Conversations: a pull cannot even be sealed on the model node (no key for the type), and
    // a peer refuses one for an ungranted type (`type-not-granted`), ...
    let (refused, _) = sync(CONTEXT_TYPE);
    assert!(
        refused.as_deref().is_some_and(|e| e.contains("no key for this type") || e.contains("type-not-granted")),
        "a context pull fails: {refused:?}"
    );
    assert!(model.records(CONTEXT_TYPE).is_empty());
    // ... no key for them is held, ...
    let st = model.state();
    let st = st.lock().unwrap();
    let rotations: Vec<String> = holder.state().lock().unwrap().roster.chain.rotations.iter().map(|r| r.id()).collect();
    assert!(!rotations.is_empty());
    for r in &rotations {
        assert!(!st.keys.has_epoch_key(r), "no epoch key for {r}");
        assert!(st.keys.get_scoped(r, CONTEXT_TYPE).is_none(), "no context key for {r}");
    }
    assert!(rotations.iter().any(|r| st.keys.get_scoped(r, TRAIN_TYPE).is_some()), "it holds the training jobs' key");
    // ... and the sealed entry itself, handed over, does not open.
    let hst = holder.state();
    let hst = hst.lock().unwrap();
    let entry = hst.latest_entries(CONTEXT_TYPE).into_iter().find(|e| e.record_id == "c1").cloned().unwrap();
    assert!(matches!(hst.read(&entry), Ok(RecordValue::Ready(_))), "the holder reads it");
    assert!(st.read(&entry).is_err(), "the model node cannot open it");
}
