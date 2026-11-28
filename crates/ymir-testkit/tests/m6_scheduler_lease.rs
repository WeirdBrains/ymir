//! M6, the placement scheduler's lease (`ymir.scheduler`, weirdnet-sync 0.8 majority-vote
//! leases): two inference nodes linked cert-only (`ServiceRole::Inference`, so their grants carry
//! `ymir.inference`) stand for it and vote on it; the seed holder doesn't (no inference-role
//! grant). Exactly one of the two holds it at a time, and when the holder lets go the other
//! takes over with a higher term. Each holder publishes the placement plan (`ymir.plan`); the
//! other inference node reads it after a sync pass, and the old holder's plan is refused once a
//! newer term has published.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use weirdnet_net::{Keypair, Multiaddr, Node, WEIRDNET};
use weirdnet_sync::{LeaseConfig, LeaseName, LeaseState, Leases, Roster, SyncState};
use ymir_core::membership::{Membership, PLAN_TYPE, ROLE_INFERENCE, SYNC_TYPE, ServiceRole};
use ymir_core::placement::{PLAN_NAME, SCHEDULER_LEASE};

type SyncNode = Node<Vec<u8>, Vec<u8>>;

fn addrs(n: &SyncNode) -> Vec<Multiaddr> {
    n.reachable_addrs().into_iter().map(|a| a.parse().unwrap()).collect()
}

fn pull(st: &Arc<Mutex<SyncState>>, node: &SyncNode, from: &SyncNode, typ: &str) {
    let _ = weirdnet_sync::pull_from_shared(st, node, from.peer_id(), &addrs(from), typ);
}

/// An inference node linked cert-only from `holder`.
fn cert_only(holder: &Membership, dir: &std::path::Path, label: &str) -> Membership {
    let key = Keypair::generate_ed25519();
    let peer = key.public().to_peer_id().to_string();
    let user = holder.user_keypair().unwrap();
    let cert = Roster::self_certify(&WEIRDNET, &user, &peer, label, 1).unwrap();
    let bundle = holder.service_bundle(&peer, ServiceRole::Inference).unwrap();
    holder.approve(cert.clone()).unwrap();
    Membership::adopt_service(&dir.join(label), key, &user.public().encode_protobuf(), cert, &bundle).unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn one_cert_only_inference_node_holds_the_scheduler_lease_and_the_other_takes_over() {
    let tmp = tempfile::tempdir().unwrap();
    let holder = Membership::create(&tmp.path().join("holder"), "holder").unwrap();
    let (i1, i2) = (cert_only(&holder, tmp.path(), "inf1"), cert_only(&holder, tmp.path(), "inf2"));
    let start = |m: &Membership| Arc::new(weirdnet_sync::start_node(m.node_key().clone(), weirdnet_sync::node_config(true), m.state()).unwrap());
    let (hn, n1, n2) = (start(&holder), start(&i1), start(&i2));
    // Rosters and grants reach everyone.
    for _ in 0..3 {
        for (m, n) in [(&holder, &hn), (&i1, &n1), (&i2, &n2)] {
            for other in [&hn, &n1, &n2] {
                if !Arc::ptr_eq(n, other) {
                    pull(&m.state(), n, other, SYNC_TYPE);
                }
            }
        }
    }
    assert!(!holder.scheduler_candidate(), "the seed holder has no inference-role grant");
    assert!(i1.scheduler_candidate() && i2.scheduler_candidate());
    assert_eq!((holder.scheduler_voters(), i1.scheduler_voters(), i2.scheduler_voters()), (2, 2, 2));
    assert!(!holder.enable_scheduler_voter().unwrap());
    assert!(i1.enable_scheduler_voter().unwrap() && i2.enable_scheduler_voter().unwrap());

    let leases = |m: &Membership, n: &Arc<SyncNode>, other: &Arc<SyncNode>| {
        let (peer, theirs) = (other.peer_id().to_string(), addrs(other));
        Leases::new(n.clone(), m.state()).with_addrs(move |p| if p == peer { theirs.clone() } else { Vec::new() })
    };
    let cfg = LeaseConfig::for_role(ROLE_INFERENCE).with_duration(Duration::from_secs(3));
    let name = LeaseName::new(SCHEDULER_LEASE).unwrap();
    let e1 = leases(&i1, &n1, &n2).elect(name.clone(), cfg.clone());
    let e2 = leases(&i2, &n2, &n1).elect(name, cfg);

    // A voter refuses takeovers for one lease after it starts, so allow for that.
    let deadline = std::time::Instant::now() + Duration::from_secs(40);
    let (first, second, term, first_m, second_m, first_n, second_n) = loop {
        match (e1.handle(), e2.handle()) {
            (Some(_), Some(_)) => panic!("both hold the scheduler lease"),
            (Some(h), None) => break (e1, e2, h.term(), &i1, &i2, &n1, &n2),
            (None, Some(h)) => break (e2, e1, h.term(), &i2, &i1, &n2, &n1),
            (None, None) => {
                assert!(std::time::Instant::now() < deadline, "nobody got the lease: {:?} {:?}", e1.state(), e2.state());
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    };
    // It stays with one holder.
    for _ in 0..20 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(second.handle().is_none(), "two holders");
    }
    // The holder publishes its plan; the other reads it after a sync pass.
    let old_handle = first.handle().unwrap();
    let p1 = first_m.publish_plan(PLAN_NAME, &old_handle, b"{\"models\":{}}").unwrap();
    assert_eq!((p1.term, p1.version), (term, 1));
    pull(&second_m.state(), second_n, first_n, PLAN_TYPE);
    let seen = second_m.current_plan(PLAN_NAME).expect("the plan reached the other inference node");
    assert_eq!((seen.term, seen.version, seen.body.as_slice()), (term, 1, b"{\"models\":{}}".as_slice()));
    // Only a lease holder publishes: the follower has no handle, and the seed holder neither.
    assert!(second.handle().is_none());

    // The holder lets go (a daemon stopping): the other takes over with a higher term.
    first.release();
    let state = tokio::task::spawn_blocking(move || {
        let s = second.wait_for(Duration::from_secs(30), |s| matches!(s, LeaseState::Held { .. }));
        (s, second)
    })
    .await
    .unwrap();
    let new_term = match state.0 {
        LeaseState::Held { term: t } => {
            assert!(t > term, "the new term {t} is above {term}");
            t
        }
        other => panic!("the other inference node did not take over: {other:?}"),
    };
    // The new holder publishes under its term; once that reaches the old holder, the old one's
    // plan is refused (it is fenced), and the newer plan stays current.
    let h2 = state.1.handle().unwrap();
    let p2 = second_m.publish_plan(PLAN_NAME, &h2, b"{\"models\":{\"a\":{\"want\":1,\"nodes\":[]}}}").unwrap();
    assert_eq!((p2.term, p2.version), (new_term, 1));
    pull(&first_m.state(), first_n, second_n, PLAN_TYPE);
    assert_eq!(first_m.current_plan(PLAN_NAME).unwrap().term, new_term);
    assert!(first_m.publish_plan(PLAN_NAME, &old_handle, b"{}").is_err(), "a stale holder published over a newer term");
    assert_eq!(first_m.current_plan(PLAN_NAME).unwrap().term, new_term);
}
