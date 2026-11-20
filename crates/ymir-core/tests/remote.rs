//! The inference role over Weirdnet, in one process: inference nodes and app nodes on the
//! in-memory transport, each with its own identity files in a temp dir. A fake model node stands
//! in for the engines; nothing touches the real network.

use std::sync::Arc;
use std::time::Duration;

use serde_json::json;
use weirdnet_agent::pool::{Answer, NetDialer, NoEvents, Pool, PoolOptions, encode_cert};
use weirdnet_agent::remote::RemoteError;
use weirdnet_agent::remote::infer::v2::{EmbedRequest, Request, Sampling, Usage};
use weirdnet_agent::{CancelToken, Message, ToolSpec};
use weirdnet_backends::{Chunk, ChunkStream, ModelStatus};
use weirdnet_net::{Keypair, Multiaddr, PeerId, WEIRDNET};
use weirdnet_sync::Roster;
use ymir_core::coordinator::Coordinator;
use ymir_core::host;
use ymir_core::membership::{Membership, SyncNode};
use ymir_core::model_node::{BoxFuture, Embedded, ModelNode};
use ymir_core::remote::{self, RemoteOptions, connect_pool};

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
                .map(|m| m.content.clone())
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

    fn hint(&self) -> (PeerId, Vec<Multiaddr>) {
        (self.m.peer_id(), self.addrs.clone())
    }

    fn peer(&self) -> String {
        self.m.peer_id().to_string()
    }
}

/// Links a new device to `from`'s identity the way a device link does: the seed holder
/// certifies its node key and approves it; the new device adopts the seed and the roster.
fn link(from: &Membership, dir: &std::path::Path, label: &str) -> Membership {
    let key = Keypair::generate_ed25519();
    let mut seed = from.seed().unwrap().0;
    let user = Keypair::ed25519_from_bytes(&mut seed).unwrap();
    let cert = Roster::self_certify(&WEIRDNET, &user, &key.public().to_peer_id().to_string(), label, 1).unwrap();
    from.approve(cert.clone()).unwrap();
    Membership::adopt(dir, key, &from.seed().unwrap().0, cert, &from.link_settings()).unwrap()
}

fn coordinator(name: &str) -> Arc<Coordinator> {
    Arc::new(Coordinator::new(
        vec![Arc::new(FakeNode(name.into()))],
        Some("small".into()),
        Arc::new(weirdnet_semcache::ExactCache::<weirdnet_semcache::MemoryStore>::new(16)),
    ))
}

fn opts() -> RemoteOptions {
    RemoteOptions { find_wait: Duration::from_secs(2), accept_timeout: Duration::from_secs(5) }
}

fn user(text: &str) -> Request {
    Request::messages(vec![Message::user(text)]).with_model("small")
}

async fn answer(pool: &Pool, req: Request) -> Result<Answer, RemoteError> {
    pool.infer(req, &NoEvents, &CancelToken::new()).await.map_err(|i| i.error)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn app_reaches_inference_over_weirdnet() {
    let tmp = tempfile::tempdir().unwrap();
    let inf = Device::start(Membership::create(&tmp.path().join("inf"), "inference").unwrap());
    let app = Device::start(link(&inf.m, &tmp.path().join("app"), "app"));
    let (_server, _feed) = remote::serve(&inf.node, coordinator("inf"), &inf.m, Vec::new(), None, host::admission(2, 8)).unwrap();
    let pool = connect_pool(app.node.clone(), &app.m, &[inf.hint()], &opts());

    // Chat: streamed text, joined tool call, usage, and who served it.
    let a = answer(&pool, user("hello")).await.unwrap();
    assert_eq!(a.text, "inf says: hello");
    assert_eq!(a.node, inf.peer());
    assert_eq!(a.model, "small");
    assert_eq!(a.usage.map(|u| u.completion_tokens), Some(4));
    assert_eq!(a.meta.served_by.as_deref(), Some("inf"));

    let a = answer(&pool, user("use a tool").with_tools(vec![ToolSpec::new("lookup", "look it up")])).await.unwrap();
    assert_eq!(a.tool_calls.len(), 1, "{:?}", a.tool_calls);
    assert_eq!(a.tool_calls[0].name, "lookup");
    assert_eq!(a.tool_calls[0].arguments, json!({"q": "ymir"}));

    // Unknown models come back as such.
    let err = answer(&pool, Request::messages(vec![Message::user("hi")]).with_model("nope")).await.err().unwrap();
    assert!(err.to_string().contains("unknown model `nope`"), "{err}");

    // Deterministic answers come from the inference node's cache the second time.
    let det = || {
        let mut s = Sampling::default();
        s.temperature = Some(0.0);
        user("same").with_sampling(s)
    };
    assert!(!answer(&pool, det()).await.unwrap().meta.cache_hit);
    let again = answer(&pool, det()).await.unwrap();
    assert!(again.meta.cache_hit);
    assert_eq!(again.text, "inf says: same");

    // Catalog and embeddings.
    let cat = pool.catalog().await.unwrap();
    let ids: Vec<&str> = cat.models.iter().map(|m| m.id.as_str()).collect();
    assert!(ids.contains(&"small") && ids.contains(&"emb"), "{ids:?}");
    let e = pool.embed(EmbedRequest::new(vec!["abc".into(), "hello".into()]), &CancelToken::new()).await.unwrap();
    assert_eq!(e.vectors, vec![vec![3.0, 1.0], vec![5.0, 1.0]]);
    assert_eq!(e.model, "emb");

    // A revoked device is refused from then on.
    inf.m.revoke(&app.peer()).unwrap();
    let err = answer(&pool, user("still there?")).await.err().unwrap();
    assert!(err.to_string().contains("revoked"), "{err}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn this_box_uses_its_own_pool_in_process() {
    let pool = host::local_pool(coordinator("box"), "box", host::admission(2, 8), None);
    let a = answer(&pool, user("local")).await.unwrap();
    assert_eq!((a.text.as_str(), a.node.as_str()), ("box says: local", "box"));
    assert!(pool.catalog().await.unwrap().models.iter().any(|m| m.id == "small"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn strangers_are_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let inf = Device::start(Membership::create(&tmp.path().join("inf"), "inference").unwrap());
    let stranger = Device::start(Membership::create(&tmp.path().join("x"), "stranger").unwrap());
    let (_server, _feed) = remote::serve(&inf.node, coordinator("inf"), &inf.m, Vec::new(), None, host::admission(2, 8)).unwrap();

    // Another identity's device that knows where the node is: its certificate is not ours.
    let dialer = NetDialer::new(stranger.node.clone());
    let (peer, addrs) = inf.hint();
    dialer.hint(peer, addrs);
    let pool =
        Pool::new(Arc::new(dialer), vec![inf.peer()]).with_options(PoolOptions::default().with_cert(Some(encode_cert(&stranger.m.our_cert()))));
    let r = pool.infer(user("hi"), &NoEvents, &CancelToken::new()).await;
    let e = r.err().unwrap().error;
    assert!(e.to_string().contains("wrong-user"), "{e}");
    assert!(
        matches!(pool.catalog().await, Err(RemoteError::Refused { message }) if message.contains("wrong-user")),
        "client errors pass through the pool (agent 0.4.1)"
    );

    // And with no certificate at all: an unknown device.
    let dialer = NetDialer::new(stranger.node.clone());
    let (peer, addrs) = inf.hint();
    dialer.hint(peer, addrs);
    let pool = Pool::new(Arc::new(dialer), vec![inf.peer()]);
    let r = pool.infer(user("hi"), &NoEvents, &CancelToken::new()).await;
    let e = r.err().unwrap().error;
    assert!(e.to_string().contains("unknown-device"), "{e}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fails_over_between_inference_nodes() {
    let tmp = tempfile::tempdir().unwrap();
    let one = Device::start(Membership::create(&tmp.path().join("one"), "one").unwrap());
    let two = Device::start(link(&one.m, &tmp.path().join("two"), "two"));
    let app = Device::start(link(&one.m, &tmp.path().join("app"), "app"));
    // Both inference nodes know every device (as a sync pass would give them).
    for d in [&one, &two] {
        d.m.state().lock().unwrap().roster.merge(&app.m.roster(), &app.m.user_key(), &WEIRDNET);
    }
    let mut s1 = Some(remote::serve(&one.node, coordinator("one"), &one.m, Vec::new(), None, host::admission(2, 8)).unwrap());
    let mut s2 = Some(remote::serve(&two.node, coordinator("two"), &two.m, Vec::new(), None, host::admission(2, 8)).unwrap());
    let pool = connect_pool(app.node.clone(), &app.m, &[one.hint(), two.hint()], &opts());

    let a = answer(&pool, user("a")).await.unwrap();
    assert!(a.text.ends_with("says: a"));
    let first = a.node;
    // The node that answered stops serving; the other one answers.
    let other = if first == one.peer() {
        s1.take();
        two.peer()
    } else {
        s2.take();
        one.peer()
    };
    let b = answer(&pool, user("b")).await.unwrap();
    assert_eq!(b.node, other, "{}", b.text);
    assert!(b.text.ends_with("says: b"));
}

/// One sync pass on every device against every other one.
async fn sync_round(devices: &[&Device]) {
    let hints: Vec<(String, Vec<Multiaddr>)> = devices.iter().map(|d| (d.peer(), d.addrs.clone())).collect();
    for d in devices {
        let (m, node, hints) = (d.m.clone(), d.node.clone(), hints.clone());
        tokio::task::spawn_blocking(move || {
            m.tick(&node, &|c| hints.iter().find(|(p, _)| *p == c.peer_id).map(|(_, a)| a.clone()).unwrap_or_default())
        })
        .await
        .unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn roster_changes_reach_other_devices() {
    let tmp = tempfile::tempdir().unwrap();
    let inf = Device::start(Membership::create(&tmp.path().join("inf"), "inference").unwrap());
    let app = Device::start(link(&inf.m, &tmp.path().join("app"), "app"));
    let (_server, _feed) = remote::serve(&inf.node, coordinator("inf"), &inf.m, Vec::new(), None, host::admission(2, 8)).unwrap();
    // The app device links a third device the inference node has not seen; it is refused
    // there until the roster reaches the inference node.
    let late = Device::start(link(&app.m, &tmp.path().join("late"), "late"));
    let late_peer = late.peer();
    assert!(inf.m.roster().get(&late_peer).is_none());

    let mut rounds = 0;
    while !inf.m.roster().active().any(|c| c.peer_id == late_peer) {
        rounds += 1;
        assert!(rounds <= 4, "the new device never reached the inference node");
        sync_round(&[&inf, &app, &late]).await;
    }
    eprintln!("the new device reached the inference node after {rounds} sync round(s)");
    let pool = connect_pool(late.node.clone(), &late.m, &[inf.hint()], &opts());
    let a = answer(&pool, user("new here")).await.unwrap();
    assert!(a.text.ends_with("says: new here"));

    // A revocation made on the inference node reaches the others the same way.
    inf.m.revoke(&late_peer).unwrap();
    sync_round(&[&inf, &app]).await;
    assert!(!app.m.roster().active().any(|c| c.peer_id == late_peer), "the revocation reached the app device");
    let err = answer(&pool, user("again")).await.err().unwrap();
    assert!(err.to_string().contains("revoked"), "{err}");
    // And it survives a restart.
    let reopened = Membership::open(&tmp.path().join("app")).unwrap().unwrap();
    assert!(!reopened.roster().active().any(|c| c.peer_id == late_peer));
}
