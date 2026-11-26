//! M9 push wake-ups, Ymir's side (Bragi B4): a Bragi device's wake card (`bragi.wake`, sealed)
//! reaches the seed holder over sync, which wakes the device through the relay the device keeps
//! a `/weirdnet/wake/1` subscription to; the device opens the wake. A cert-only inference node
//! is never granted the type: it can't pull the cards and holds none. Loopback nodes and a
//! stand-in relay in this process; nothing leaves the box.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use weirdnet_net::{Handler, Keypair, Limits, Multiaddr, Node, PeerId, WEIRDNET};
use weirdnet_push::relaywake::{self, Delivery, FromRelay, Subscription, ToRelay};
use weirdnet_push::{Payload96, PushKey, Reason, WakeCard, WakeHandle};
use weirdnet_sync::{Roster, SyncState};
use ymir_core::membership::{CONTEXT_TYPE, Membership, SYNC_TYPE, ServiceRole, SyncNode, WAKE_TYPE};
use ymir_core::wake::{self, DeviceWake};

fn addrs(n: &SyncNode) -> Vec<Multiaddr> {
    n.reachable_addrs().into_iter().map(|a| a.parse().unwrap()).collect()
}

fn pull(st: &Arc<Mutex<SyncState>>, node: &SyncNode, from: &SyncNode, typ: &str) -> Option<String> {
    weirdnet_sync::pull_from_shared(st, node, from.peer_id(), &addrs(from), typ).error.map(|e| e.to_string())
}

fn node(m: &Membership) -> Arc<SyncNode> {
    Arc::new(weirdnet_sync::start_node(m.node_key().clone(), weirdnet_sync::node_config(true), m.state()).unwrap())
}

struct NoRequests;

impl Handler<Vec<u8>, Vec<u8>> for NoRequests {
    fn request(&self, _from: &PeerId, _req: Vec<u8>) -> Vec<u8> {
        Vec::new()
    }
}

type Subs = Arc<Mutex<HashMap<WakeHandle, tokio::sync::mpsc::Sender<(WakeHandle, Payload96)>>>>;

/// A stand-in for the relay's wake service: subscriptions and deliveries, one process.
fn relay() -> Arc<Node<Vec<u8>, Vec<u8>>> {
    let node = Arc::new(Node::start(&WEIRDNET, Keypair::generate_ed25519(), weirdnet_sync::node_config(true), Arc::new(NoRequests)).unwrap());
    let mut incoming = node.accept_streams(relaywake::PROTOCOL, Limits::default()).unwrap();
    let subs: Subs = Arc::default();
    let rt = tokio::runtime::Handle::current();
    std::thread::spawn(move || {
        while let Some(mut io) = incoming.accept() {
            let subs = subs.clone();
            rt.spawn(async move {
                match relaywake::read::<_, ToRelay>(&mut io).await {
                    Ok(Some(ToRelay::Subscribe { handles })) => {
                        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
                        for h in &handles {
                            subs.lock().unwrap().insert(h.clone(), tx.clone());
                        }
                        if relaywake::write(&mut io, &FromRelay::Subscribed { handles: handles.len(), keepalive_s: 30 }).await.is_err() {
                            return;
                        }
                        while let Some((handle, payload)) = rx.recv().await {
                            if relaywake::write(&mut io, &FromRelay::Wake { handle, payload }).await.is_err() {
                                return;
                            }
                        }
                    }
                    Ok(Some(ToRelay::Deliver { handle, payload })) => {
                        let tx = subs.lock().unwrap().get(&handle).cloned();
                        let status = match tx {
                            Some(tx) if tx.send((handle, payload)).await.is_ok() => Delivery::Delivered,
                            _ => Delivery::Offline,
                        };
                        let _ = relaywake::write(&mut io, &FromRelay::Delivered { status }).await;
                        let _ = futures::AsyncWriteExt::close(&mut io).await;
                    }
                    _ => {}
                }
            });
        }
    });
    node
}

fn wait(what: &str, mut f: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while !f() {
        assert!(Instant::now() < deadline, "waiting for {what}");
        std::thread::sleep(Duration::from_millis(100));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn the_seed_holder_wakes_a_bragi_device_and_a_cert_only_node_never_sees_its_card() {
    let tmp = tempfile::tempdir().unwrap();

    // The seed holder starts the key chain.
    let holder = Membership::create(&tmp.path().join("holder"), "holder").unwrap();
    holder.write_record(CONTEXT_TYPE, "c1", b"a private conversation").unwrap();
    let user = holder.user_keypair().unwrap();

    // A Bragi phone: a full device (it holds the seed).
    let pkey = Keypair::generate_ed25519();
    let ppeer = pkey.public().to_peer_id().to_string();
    let pcert = Roster::self_certify(&WEIRDNET, &user, &ppeer, "Bragi on Android", 1).unwrap();
    holder.approve(pcert.clone()).unwrap();
    let phone = Membership::adopt(&tmp.path().join("phone"), pkey, &holder.seed().unwrap().0, pcert, &holder.link_settings()).unwrap();

    // A cert-only inference node.
    let ikey = Keypair::generate_ed25519();
    let ipeer = ikey.public().to_peer_id().to_string();
    let icert = Roster::self_certify(&WEIRDNET, &user, &ipeer, "inference", 1).unwrap();
    let bundle = holder.service_bundle(&ipeer, ServiceRole::Inference).unwrap();
    holder.approve(icert.clone()).unwrap();
    let inference = Membership::adopt_service(&tmp.path().join("inference"), ikey, &user.public().encode_protobuf(), icert, &bundle).unwrap();
    let g = inference.state().lock().unwrap().roster.grant_of(&ipeer).cloned().unwrap();
    assert!(!g.types.iter().any(|t| t == WAKE_TYPE), "the inference grant lists the wake cards: {:?}", g.types);

    let (hnode, pnode, inode) = (node(&holder), node(&phone), node(&inference));
    let relay = relay();
    let relay_addr = format!("{}/p2p/{}", relay.reachable_addrs()[0].split("/p2p/").next().unwrap(), relay.peer_id());

    // The phone gets the key chain, then writes its card (what Bragi's wake settings do).
    let key = PushKey::generate();
    let card = WakeCard {
        service: "https://push.weirdbrains.com".into(),
        handle: WakeHandle::generate(),
        key: key.public().iter().map(|b| format!("{b:02x}")).collect(),
    };
    let record = serde_json::to_vec(&DeviceWake { card: Some(card.clone()), relay: Some(relay_addr.clone()) }).unwrap();
    wait("the phone's card written", || {
        let _ = pull(&phone.state(), &pnode, &hnode, SYNC_TYPE);
        phone.write_record(WAKE_TYPE, &ppeer, &record).is_ok()
    });

    // The seed holder's sync pass brings it in.
    wait("the card on the holder", || {
        holder.tick(&hnode, &|c| if c.peer_id == ppeer { addrs(&pnode) } else { Vec::new() });
        !wake::wakeable(&holder).is_empty()
    });
    let devices = wake::wakeable(&holder);
    let d = wake::find(&devices, "Bragi on Android").expect("the phone by label");
    assert_eq!((d.peer_id.as_str(), d.relay.as_deref()), (ppeer.as_str(), Some(relay_addr.as_str())));

    // The cert-only node: its pulls of the cards are refused and it holds none.
    let _ = pull(&inference.state(), &inode, &hnode, SYNC_TYPE);
    assert!(pull(&inference.state(), &inode, &hnode, WAKE_TYPE).is_some(), "a cert-only node pulled the wake cards");
    inference.tick(&inode, &|c| if c.peer_id == holder.peer_id().to_string() { addrs(&hnode) } else { Vec::new() });
    assert!(inference.records(WAKE_TYPE).is_empty());
    assert!(wake::wakeable(&inference).is_empty());

    // Not subscribed: the relay holds no route, so it answers offline.
    assert_eq!(wake::wake(&hnode, d, Reason::MessageWaiting).await.unwrap(), Delivery::Offline);

    // The phone keeps its relay wake connection; the holder wakes it; the phone opens it once.
    let io = pnode.open_stream_async(relay.peer_id(), &addrs(&relay), relaywake::PROTOCOL, Limits::default()).await.unwrap();
    let mut sub = Subscription::start(io, std::slice::from_ref(&card.handle)).await.unwrap();
    assert_eq!(wake::wake(&hnode, d, Reason::MessageWaiting).await.unwrap(), Delivery::Delivered);
    let (h, payload) = tokio::time::timeout(Duration::from_secs(10), sub.next()).await.unwrap().unwrap().unwrap();
    assert_eq!(h, card.handle);
    assert_eq!(key.open(&h, &payload).unwrap().reason, Reason::MessageWaiting);

    // A device that turned wake-ups off withdraws its card: nothing to wake.
    phone.write_record(WAKE_TYPE, &ppeer, br#"{"card":null}"#).unwrap();
    holder.refresh_wake_cards();
    wait("the withdrawn card on the holder", || {
        holder.tick(&hnode, &|c| if c.peer_id == ppeer { addrs(&pnode) } else { Vec::new() });
        wake::wakeable(&holder).is_empty()
    });
}
