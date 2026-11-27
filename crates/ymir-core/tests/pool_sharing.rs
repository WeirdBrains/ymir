//! Cross-identity pool sharing (weirdnet-agent 0.5.9 guests; speech refusals from 0.5.11), in one process: a pool owner's
//! inference node and a device of a second, separate Weirdnet identity that joins with a share
//! link and its own certificate (`connect_shared`). A fake model node stands in for the engines;
//! nothing touches the real network.

use std::sync::Arc;
use std::time::Duration;

use serde_json::json;
use weirdnet_agent::pool::{Answer, NoEvents, Pool, PoolShare, connect_shared, encode_credential, share_dialer};
use weirdnet_agent::remote::RemoteError;
use weirdnet_agent::remote::infer::v2::{ContextRef, Request, SpeakRequest, Usage, Voice};
use weirdnet_agent::remote::{Caller, Dialer, catalog};
use weirdnet_agent::{CancelToken, Message};
use weirdnet_backends::{Chunk, ChunkStream, ModelStatus};
use weirdnet_net::{Keypair, Multiaddr, WEIRDNET};
use weirdnet_sync::Roster;
use ymir_core::context::ContextStore;
use ymir_core::coordinator::Coordinator;
use ymir_core::guests::{self, GuestGrant};
use ymir_core::host;
use ymir_core::membership::{Membership, SyncNode};
use ymir_core::model_node::{BoxFuture, Embedded, ModelNode};
use ymir_core::remote;

/// A model node with a chat model and an embedding model, always loaded. It answers with its
/// own name and the last user message, and calls `lookup` when it is offered tools.
struct FakeNode(String);

fn status(id: &str, embedding: bool) -> ModelStatus {
    serde_json::from_value(json!({
        "id": id, "engine": "llamacpp", "state": {"state": "ready"}, "in_flight": 0, "swap_group": null,
        "embedding": embedding, "pid": null, "last_load_secs": null, "size_bytes": null, "wrapped": false, "crashes": 0
    }))
    .unwrap()
}

impl ModelNode for FakeNode {
    fn name(&self) -> &str {
        &self.0
    }
    fn status(&self) -> Vec<ModelStatus> {
        vec![status("small", false), status("emb", true)]
    }
    fn load<'a>(&'a self, _: &'a str, _: bool) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(async { Ok(()) })
    }
    fn unload<'a>(&'a self, _: &'a str) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(async { Ok(()) })
    }
    fn chat<'a>(&'a self, _model: &'a str, req: &'a Request) -> BoxFuture<'a, Result<ChunkStream, RemoteError>> {
        Box::pin(async move {
            let last = req
                .inline_messages()
                .and_then(|m| m.iter().rev().find(|m| m.role == weirdnet_agent::Role::User))
                .map(|m| m.text().into_owned())
                .unwrap_or_default();
            let mut ev = vec![Chunk::Text(format!("{} says: ", self.0)), Chunk::Text(last)];
            if !req.tools.is_empty() {
                ev.push(Chunk::ToolCall { index: 0, id: Some("c1".into()), name: Some("lookup".into()), arguments: "{\"q\":".into() });
                ev.push(Chunk::ToolCall { index: 0, id: None, name: None, arguments: "\"ymir\"}".into() });
            }
            ev.push(Chunk::Done { finish_reason: Some("stop".into()), usage: Some(Usage::new(3, 4)) });
            Ok(Box::pin(futures::stream::iter(ev.into_iter().map(Ok))) as ChunkStream)
        })
    }
    fn embed<'a>(&'a self, _: &'a str, input: &'a [String]) -> BoxFuture<'a, Result<Embedded, RemoteError>> {
        Box::pin(async move { Ok((input.iter().map(|s| vec![s.len() as f32, 1.0]).collect(), 1)) })
    }
}

struct Device {
    m: Arc<Membership>,
    node: Arc<SyncNode>,
    addrs: Vec<Multiaddr>,
}

impl Device {
    fn start(m: Membership) -> Self {
        let m = Arc::new(m);
        let node = Arc::new(weirdnet_sync::start_node(m.node_key().clone(), weirdnet_sync::node_config(true), m.state()).unwrap());
        let mut addrs = Vec::new();
        for _ in 0..200 {
            addrs = node.status().listening.iter().filter_map(|a| a.parse().ok()).collect();
            if !addrs.is_empty() {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(!addrs.is_empty(), "the node never listened");
        Self { m, node, addrs }
    }

    fn peer(&self) -> String {
        self.m.peer_id().to_string()
    }
}

fn coordinator(name: &str) -> Arc<Coordinator> {
    Arc::new(Coordinator::new(
        vec![Arc::new(FakeNode(name.into()))],
        Some("small".into()),
        Arc::new(weirdnet_semcache::ExactCache::<weirdnet_semcache::MemoryStore>::new(16)),
    ))
}

fn user(text: &str) -> Request {
    Request::messages(vec![Message::user(text)]).with_model("small")
}

async fn answer(pool: &Pool, req: Request) -> Result<Answer, RemoteError> {
    pool.infer(req, &NoEvents, &CancelToken::new()).await.map_err(|i| i.error)
}

fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as i64
}

fn share_of(owner: &Device) -> PoolShare {
    let addrs: Vec<String> = owner.addrs.iter().map(|a| format!("{a}/p2p/{}", owner.peer())).collect();
    guests::pool_share(&owner.m.user_id(), Some("test pool"), &[(owner.peer(), addrs)], None).unwrap()
}

fn guest_pool(guest: &Device, share: &PoolShare) -> Pool {
    connect_shared(guest.node.clone(), share, &guest.m.user_key(), &guest.m.our_cert()).unwrap()
}

fn refused(r: Result<Answer, RemoteError>, why: &str) -> bool {
    matches!(&r, Err(RemoteError::Refused { message }) if message.contains(why)) || {
        eprintln!("expected a refusal ({why}), got {:?}", r.as_ref().map(|a| &a.text));
        false
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_second_identity_uses_a_shared_pool_until_unshared() {
    let tmp = tempfile::tempdir().unwrap();
    let owner = Device::start(Membership::create(&tmp.path().join("owner"), "inference").unwrap());
    // A separate Weirdnet identity (Morph's profile, say), not linked to the owner's roster.
    let guest = Device::start(Membership::create(&tmp.path().join("guest"), "morph").unwrap());
    assert_ne!(owner.m.user_id(), guest.m.user_id());
    let store = ContextStore::new(owner.m.clone());
    let (_server, _feed) = remote::serve(&owner.node, coordinator("owner"), &owner.m, Vec::new(), Some(store), host::admission(2, 8)).unwrap();
    let share = share_of(&owner);
    let parsed = PoolShare::parse(&share.to_link().unwrap()).unwrap();
    let pool = guest_pool(&guest, &parsed);

    // Not shared yet: the guest's valid certificate alone gets nothing.
    assert!(matches!(pool.catalog().await, Err(RemoteError::Refused { message }) if message.contains("not-shared")));
    assert!(refused(answer(&pool, user("hi")).await, "not-shared"));

    // Shared: catalog and chat work.
    let mut g = GuestGrant::new(guest.m.user_id(), now_ms());
    g.label = "morph".into();
    guests::share(&owner.m, &g).unwrap();
    assert!(pool.catalog().await.unwrap().models.iter().any(|m| m.id == "small"));
    assert_eq!(answer(&pool, user("hello")).await.unwrap().text, "owner says: hello");

    // Never the owner's conversations.
    let ctx = Request::context(ContextRef::new("c1", None)).with_model("small");
    assert!(refused(answer(&pool, ctx).await, "guest-context"));

    // Speech: a guest never clones a voice through a share link, consent or not; a preset gets
    // the plain answer that no model here speaks yet.
    let clone = SpeakRequest::text("hi").with_voice(Voice::clone_from(vec![1; 64], "wav", None, true));
    let r = pool.speak(clone, &CancelToken::new()).await.map(|s| s.model);
    assert!(matches!(&r, Err(RemoteError::Refused { message }) if message.contains("guest-clone")), "{r:?}");
    let r = pool.speak(SpeakRequest::text("hi").with_voice(Voice::preset("vivian")), &CancelToken::new()).await.map(|s| s.model);
    assert!(matches!(&r, Err(e) if e.to_string().contains("speaks yet")), "{r:?}");

    // Everything outside catalog/infer stays on the roster check (models, train, tasks, wake, admin).
    let caller = Caller { peer: guest.peer(), cert_bytes: Some(encode_credential(&guest.m.user_key(), &guest.m.our_cert())) };
    assert!(remote::roster_auth(&owner.m).authorize(&caller).is_err());
    assert!(remote::role_auth(&owner.m, &[guest.peer()]).authorize(&caller).is_err());
    assert!(remote::guests(&owner.m).auth.identify(&caller, weirdnet_agent::serve::GuestScope::Infer).unwrap().is_guest());

    // A model allowlist.
    g.models = vec!["other".into()];
    guests::share(&owner.m, &g).unwrap();
    assert!(refused(answer(&pool, user("hello")).await, "guest-model"));
    g.models.clear();

    // Peer pinning: another device of the same guest identity is refused.
    let mut seed = guest.m.seed().unwrap().0;
    let guest_user = Keypair::ed25519_from_bytes(&mut seed).unwrap();
    let second_key = Keypair::generate_ed25519();
    let second_cert = Roster::self_certify(&WEIRDNET, &guest_user, &second_key.public().to_peer_id().to_string(), "phone", 1).unwrap();
    guest.m.approve(second_cert.clone()).unwrap();
    let second = Device::start(
        Membership::adopt(&tmp.path().join("guest2"), second_key, &guest.m.seed().unwrap().0, second_cert, &guest.m.link_settings()).unwrap(),
    );
    let second_pool = guest_pool(&second, &parsed);
    g.peers = vec![guest.peer()];
    guests::share(&owner.m, &g).unwrap();
    assert_eq!(answer(&pool, user("pinned")).await.unwrap().text, "owner says: pinned");
    assert!(refused(answer(&second_pool, user("not pinned")).await, "guest-peer-not-pinned"));

    // A live catalog subscription, then unshare: it closes within RECHECK_EVERY, and the next
    // inference call is refused.
    let dialer = share_dialer(guest.node.clone(), &parsed).unwrap();
    let mut ch = dialer.open(&owner.peer(), catalog::PROTOCOL).await.unwrap();
    let cred = Some(encode_credential(&guest.m.user_key(), &guest.m.our_cert()));
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let sub = tokio::spawn(async move { catalog::subscribe(&mut ch, cred, move |c| drop(tx.send(c))).await });
    assert!(tokio::time::timeout(Duration::from_secs(10), rx.recv()).await.unwrap().is_some(), "the first catalog");
    guests::unshare(&owner.m, &guest.m.user_id(), now_ms()).unwrap();
    let started = std::time::Instant::now();
    assert!(refused(answer(&pool, user("after")).await, "unshared"));
    let ended =
        tokio::time::timeout(catalog::RECHECK_EVERY + Duration::from_secs(3), sub).await.expect("the subscription outlived the share").unwrap();
    assert!(matches!(&ended, Err(RemoteError::Refused { message }) if message.contains("unshared")), "{ended:?}");
    assert!(started.elapsed() <= catalog::RECHECK_EVERY + Duration::from_secs(3));
    assert!(guests::all_grants(&owner.m).iter().all(|g| !g.is_active(now_ms())));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_expired_share_admits_no_one() {
    let tmp = tempfile::tempdir().unwrap();
    let owner = Device::start(Membership::create(&tmp.path().join("owner"), "inference").unwrap());
    let guest = Device::start(Membership::create(&tmp.path().join("guest"), "bragi").unwrap());
    let (_server, _feed) = remote::serve(&owner.node, coordinator("owner"), &owner.m, Vec::new(), None, host::admission(2, 8)).unwrap();
    let pool = guest_pool(&guest, &share_of(&owner));
    let mut g = GuestGrant::new(guest.m.user_id(), now_ms());
    g.expires_at = Some(now_ms() - 1);
    guests::share(&owner.m, &g).unwrap();
    assert!(refused(answer(&pool, user("late")).await, "share-expired"));
    // The owner's own identity can't be a guest.
    assert!(guests::share(&owner.m, &GuestGrant::new(owner.m.user_id(), now_ms())).is_err());
}
