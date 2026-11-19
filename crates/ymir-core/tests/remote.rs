//! The inference role over Weirdnet, in one process: inference nodes and app nodes on the
//! in-memory transport, each with its own identity files in a temp dir. A fake model node stands
//! in for the engines; nothing touches the real network.

use std::sync::Arc;
use std::time::Duration;

use weirdnet_agent::pool::{NetDialer, Pool as AgentPool, PoolOptions, encode_cert};
use weirdnet_agent::remote::infer::v2;
use weirdnet_agent::{CancelToken, RemoteError};
use weirdnet_net::{Keypair, Multiaddr, PeerId, WEIRDNET};
use weirdnet_sync::Roster;
use ymir_backends::{EngineKind, ModelStatus};
use ymir_core::coordinator::Coordinator;
use ymir_core::membership::{Membership, SyncNode};
use ymir_core::model_node::{Embedded, ModelNode};
use ymir_core::remote::{self, RemoteOptions, RemotePool};
use ymir_infer::{BoxFuture, ChatRequest, CompletionRequest, EmbedRequest, Event, EventStream, InferError, ModelState, Pool, collect};

/// A model node with a chat model and an embedding model, always loaded. It answers with its
/// own name and the last user message, and calls `lookup` when it is offered tools.
struct FakeNode(String);

fn status(id: &str, embedding: bool) -> ModelStatus {
    ModelStatus {
        id: id.into(),
        engine: EngineKind::Llamacpp,
        state: ModelState::Ready,
        in_flight: 0,
        swap_group: None,
        embedding,
        pid: None,
        last_load_secs: None,
        size_bytes: None,
        wrapped: false,
        crashes: 0,
    }
}

impl ModelNode for FakeNode {
    fn name(&self) -> &str {
        &self.0
    }
    fn status(&self) -> Vec<ModelStatus> {
        vec![status("small", false), status("emb", true)]
    }
    fn load<'a>(&'a self, _: &'a str, _: bool) -> BoxFuture<'a, Result<(), InferError>> {
        Box::pin(async { Ok(()) })
    }
    fn unload<'a>(&'a self, _: &'a str) -> BoxFuture<'a, Result<(), InferError>> {
        Box::pin(async { Ok(()) })
    }
    fn chat<'a>(&'a self, model: &'a str, req: &'a ChatRequest) -> BoxFuture<'a, Result<EventStream, InferError>> {
        Box::pin(async move {
            let last = req.messages.iter().rev().find(|m| m["role"] == "user").and_then(|m| m["content"].as_str()).unwrap_or("").to_string();
            let mut ev = vec![Event::Delta { offset: 0, text: format!("{} says: ", self.0) }, Event::Delta { offset: 0, text: last }];
            if req.params.contains_key("tools") {
                ev.push(Event::ToolCall { index: 0, id: Some("c1".into()), name: Some("lookup".into()), arguments: "{\"q\":".into() });
                ev.push(Event::ToolCall { index: 0, id: None, name: None, arguments: "\"ymir\"}".into() });
            }
            ev.push(Event::Done { finish_reason: Some("stop".into()), usage: Some(ymir_infer::Usage { prompt_tokens: 3, completion_tokens: 4 }) });
            let _ = model;
            Ok(Box::pin(futures::stream::iter(ev.into_iter().map(Ok))) as EventStream)
        })
    }
    fn complete<'a>(&'a self, _: &'a str, _: &'a CompletionRequest) -> BoxFuture<'a, Result<EventStream, InferError>> {
        Box::pin(async { Err(InferError::Unavailable("no".into())) })
    }
    fn embed<'a>(&'a self, _: &'a str, input: &'a [String]) -> BoxFuture<'a, Result<Embedded, InferError>> {
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
    Arc::new(Coordinator::new(vec![Arc::new(FakeNode(name.into()))], Some("small".into()), Arc::new(ymir_semcache::NoCache)))
}

fn opts() -> RemoteOptions {
    RemoteOptions { find_wait: Duration::from_secs(2), accept_timeout: Duration::from_secs(5) }
}

fn user(text: &str) -> ChatRequest {
    ChatRequest::user(Some("small"), text)
}

async fn answer(pool: &dyn Pool, req: ChatRequest) -> Result<(ymir_infer::Collected, String, String), InferError> {
    let reply = pool.chat(req).await?;
    let (node, model) = (reply.node.clone(), reply.model.clone());
    Ok((collect(reply.events).await?, node, model))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn app_reaches_inference_over_weirdnet() {
    let tmp = tempfile::tempdir().unwrap();
    let inf = Device::start(Membership::create(&tmp.path().join("inf"), "inference").unwrap());
    let app = Device::start(link(&inf.m, &tmp.path().join("app"), "app"));
    let (_server, _feed) = remote::serve(&inf.node, coordinator("inf"), &inf.m, remote::admission(2, 8)).unwrap();
    let pool = RemotePool::connect(app.node.clone(), &app.m, &[inf.hint()], opts());

    // Chat: streamed text, joined tool call, usage, and who served it.
    let (c, node, model) = answer(&pool, user("hello")).await.unwrap();
    assert_eq!(c.text, "inf says: hello");
    assert_eq!(node, inf.peer());
    assert_eq!(model, "small");
    assert_eq!(c.usage.map(|u| u.completion_tokens), Some(4));

    let mut req = user("use a tool");
    req.params.insert("tools".into(), serde_json::json!([{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]));
    let (c, _, _) = answer(&pool, req).await.unwrap();
    assert_eq!(c.tool_calls.len(), 1, "{:?}", c.tool_calls);
    let (_, _, name, args) = &c.tool_calls[0];
    assert_eq!(name.as_deref(), Some("lookup"));
    assert_eq!(serde_json::from_str::<serde_json::Value>(args).unwrap(), serde_json::json!({"q": "ymir"}));

    // Unknown models come back as such.
    let err = answer(&pool, ChatRequest::user(Some("nope"), "hi")).await.err().unwrap();
    assert_eq!(err, InferError::UnknownModel("nope".into()));

    // Catalog and embeddings.
    let cat = pool.catalog().await.unwrap();
    let ids: Vec<&str> = cat.iter().map(|m| m.id.as_str()).collect();
    assert!(ids.contains(&"small") && ids.contains(&"emb"), "{ids:?}");
    assert!(cat.iter().find(|m| m.id == "emb").unwrap().purpose == "embedding");
    let e = pool.embed(EmbedRequest { model: None, input: vec!["abc".into(), "hello".into()] }).await.unwrap();
    assert_eq!(e.vectors, vec![vec![3.0, 1.0], vec![5.0, 1.0]]);
    assert_eq!(e.model, "emb");

    // Completions stay local.
    assert!(matches!(
        pool.complete(CompletionRequest { model: None, prompt: "x".into(), params: Default::default(), wait_for_model: false }).await,
        Err(InferError::Unavailable(_))
    ));

    // A revoked device is refused from then on.
    inf.m.revoke(&app.peer()).unwrap();
    let err = answer(&pool, user("still there?")).await.err().unwrap();
    assert!(err.to_string().contains("revoked"), "{err}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn strangers_are_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let inf = Device::start(Membership::create(&tmp.path().join("inf"), "inference").unwrap());
    let stranger = Device::start(Membership::create(&tmp.path().join("x"), "stranger").unwrap());
    let (_server, _feed) = remote::serve(&inf.node, coordinator("inf"), &inf.m, remote::admission(2, 8)).unwrap();

    // Another identity's device that knows where the node is: its certificate is not ours.
    let dialer = NetDialer::new(stranger.node.clone());
    let (peer, addrs) = inf.hint();
    dialer.hint(peer, addrs);
    let pool =
        AgentPool::new(Arc::new(dialer), vec![inf.peer()]).with_options(PoolOptions::default().with_cert(Some(encode_cert(&stranger.m.our_cert()))));
    let r = pool.infer(v2::Request::messages(vec![weirdnet_agent::Message::user("hi")]), &weirdnet_agent::pool::NoEvents, &CancelToken::new()).await;
    let e = r.err().unwrap().error;
    assert!(e.to_string().contains("wrong-user"), "{e}");
    assert!(matches!(pool.catalog().await, Err(RemoteError::Unavailable { message }) if message.contains("wrong-user")));

    // And with no certificate at all: an unknown device.
    let dialer = NetDialer::new(stranger.node.clone());
    let (peer, addrs) = inf.hint();
    dialer.hint(peer, addrs);
    let pool = AgentPool::new(Arc::new(dialer), vec![inf.peer()]);
    let r = pool.infer(v2::Request::messages(vec![weirdnet_agent::Message::user("hi")]), &weirdnet_agent::pool::NoEvents, &CancelToken::new()).await;
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
    let mut s1 = Some(remote::serve(&one.node, coordinator("one"), &one.m, remote::admission(2, 8)).unwrap());
    let mut s2 = Some(remote::serve(&two.node, coordinator("two"), &two.m, remote::admission(2, 8)).unwrap());
    let pool = RemotePool::connect(app.node.clone(), &app.m, &[one.hint(), two.hint()], opts());

    let (c, first, _) = answer(&pool, user("a")).await.unwrap();
    assert!(c.text.ends_with("says: a"));
    // The node that answered stops serving; the other one answers.
    let other = if first == one.peer() {
        s1.take();
        two.peer()
    } else {
        s2.take();
        one.peer()
    };
    let (c, node, _) = answer(&pool, user("b")).await.unwrap();
    assert_eq!(node, other, "{}", c.text);
    assert!(c.text.ends_with("says: b"));
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
    let (_server, _feed) = remote::serve(&inf.node, coordinator("inf"), &inf.m, remote::admission(2, 8)).unwrap();
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
    let pool = RemotePool::connect(late.node.clone(), &late.m, &[inf.hint()], opts());
    let (c, _, _) = answer(&pool, user("new here")).await.unwrap();
    assert!(c.text.ends_with("says: new here"));

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
