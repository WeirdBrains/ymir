//! M6, moving a model between nodes (`ymirctl move`): the model loads on the target, new
//! requests go there at once, the source finishes what it has (up to the grace time) and then
//! unloads. A target without its own copy, a model loaded nowhere else and a failed load on
//! the target are refused, with the source left as it was. Fake model nodes in one process.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::json;
use weirdnet_agent::remote::RemoteError;
use weirdnet_agent::remote::infer::v2::{Request, Usage};
use weirdnet_agent::{CancelToken, Message, ToolCall};
use weirdnet_backends::{Chunk, ChunkStream, ModelStatus};
use weirdnet_semcache::NoCache;
use ymir_core::coordinator::{Coordinator, Sink};
use ymir_core::model_node::{BoxFuture, Embedded, ModelNode};

/// A node holding copies of some models, each loaded or not; a chat takes `delay` and counts
/// as in flight meanwhile.
struct Node {
    name: String,
    loaded: Mutex<HashMap<String, bool>>,
    in_flight: AtomicU32,
    delay: Duration,
    load_fails: AtomicBool,
}

impl Node {
    fn new(name: &str, models: &[(&str, bool)], delay_ms: u64) -> Arc<Self> {
        Arc::new(Self {
            name: name.into(),
            loaded: Mutex::new(models.iter().map(|(m, l)| (m.to_string(), *l)).collect()),
            in_flight: AtomicU32::new(0),
            delay: Duration::from_millis(delay_ms),
            load_fails: AtomicBool::new(false),
        })
    }
    fn is_loaded(&self, m: &str) -> bool {
        self.loaded.lock().unwrap().get(m).copied().unwrap_or(false)
    }
}

impl ModelNode for Node {
    fn name(&self) -> &str {
        &self.name
    }
    fn status(&self) -> Vec<ModelStatus> {
        let flight = self.in_flight.load(Ordering::SeqCst);
        self.loaded
            .lock()
            .unwrap()
            .iter()
            .map(|(id, on)| {
                let state = if *on { json!({"state": "ready"}) } else { json!({"state": "unloaded"}) };
                serde_json::from_value(json!({
                    "id": id, "engine": "llamacpp", "state": state, "in_flight": if *on { flight } else { 0 }, "swap_group": null,
                    "embedding": false, "pid": null, "last_load_secs": null, "size_bytes": null, "wrapped": false, "crashes": 0
                }))
                .unwrap()
            })
            .collect()
    }
    fn load<'a>(&'a self, model: &'a str, _: bool) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(async move {
            if self.load_fails.load(Ordering::SeqCst) {
                return Err(RemoteError::unavailable("engine won't start"));
            }
            self.loaded.lock().unwrap().insert(model.to_string(), true);
            Ok(())
        })
    }
    fn unload<'a>(&'a self, model: &'a str) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(async move {
            self.loaded.lock().unwrap().insert(model.to_string(), false);
            Ok(())
        })
    }
    fn chat<'a>(&'a self, _model: &'a str, _req: &'a Request) -> BoxFuture<'a, Result<ChunkStream, RemoteError>> {
        Box::pin(async move {
            self.in_flight.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(self.delay).await;
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            let ev = vec![Chunk::Text(self.name.clone()), Chunk::Done { finish_reason: Some("stop".into()), usage: Some(Usage::new(1, 1)) }];
            Ok(Box::pin(futures::stream::iter(ev.into_iter().map(Ok))) as ChunkStream)
        })
    }
    fn embed<'a>(&'a self, _: &'a str, _: &'a [String]) -> BoxFuture<'a, Result<Embedded, RemoteError>> {
        Box::pin(async { Err(RemoteError::unavailable("no embeddings")) })
    }
}

#[derive(Default)]
struct Collect(Mutex<String>);

impl Sink for Collect {
    fn text(&self, t: String) {
        self.0.lock().unwrap().push_str(&t);
    }
    fn reasoning(&self, _: String) {}
    fn tool_call(&self, _: ToolCall) {}
    fn usage(&self, _: Usage) {}
}

/// Which node answered.
async fn ask(c: &Coordinator, model: &str) -> Result<String, RemoteError> {
    let req = Request::messages(vec![Message::user("hi")]).with_model(model);
    let p = c.prepare(&req)?;
    let out = Collect::default();
    c.run(p, &req, &out, &CancelToken::new()).await?;
    Ok(out.0.lock().unwrap().clone())
}

fn pool(nodes: &[&Arc<Node>]) -> Arc<Coordinator> {
    Arc::new(Coordinator::new(nodes.iter().map(|n| (*n).clone() as Arc<dyn ModelNode>).collect(), None, Arc::new(NoCache)))
}

#[tokio::test(flavor = "multi_thread")]
async fn a_model_moves_and_the_source_finishes_first() {
    let a = Node::new("a", &[("alpha", true)], 600);
    let b = Node::new("b", &[("alpha", false)], 0);
    let c = pool(&[&a, &b]);
    assert_eq!(ask(&c, "alpha").await.unwrap(), "a");

    // A request is running on a when the move starts.
    let slow = tokio::spawn({
        let c = c.clone();
        async move { ask(&c, "alpha").await }
    });
    let t0 = Instant::now();
    while a.in_flight.load(Ordering::SeqCst) == 0 {
        assert!(t0.elapsed() < Duration::from_secs(5), "the request never reached a");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let mv = tokio::spawn({
        let c = c.clone();
        async move { c.move_model("alpha", None, "b", Duration::from_secs(10)).await }
    });
    // b is ready and takes new requests while a finishes.
    let t0 = Instant::now();
    while !b.is_loaded("alpha") {
        assert!(t0.elapsed() < Duration::from_secs(5), "b never loaded alpha");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(ask(&c, "alpha").await.unwrap(), "b", "a new request went to the leaving node");
    assert!(a.is_loaded("alpha"), "a unloaded before its request finished");
    assert_eq!(slow.await.unwrap().unwrap(), "a", "the running request was cut");
    assert_eq!(mv.await.unwrap().unwrap(), ("a".to_string(), "b".to_string()));
    assert!(!a.is_loaded("alpha") && b.is_loaded("alpha"));
    assert_eq!(ask(&c, "alpha").await.unwrap(), "b");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_move_that_cannot_happen_leaves_the_source_alone() {
    let a = Node::new("a", &[("alpha", true), ("beta", false)], 0);
    let b = Node::new("b", &[("beta", false)], 0);
    let c = pool(&[&a, &b]);

    // b has no copy of alpha.
    let e = c.move_model("alpha", None, "b", Duration::from_secs(1)).await.unwrap_err();
    assert!(e.to_string().contains("no copy"), "{e}");
    // beta is loaded nowhere.
    assert!(c.move_model("beta", None, "b", Duration::from_secs(1)).await.is_err());
    // Already there, and an unknown node.
    assert!(c.move_model("alpha", Some("a"), "a", Duration::from_secs(1)).await.is_err());
    assert!(c.move_model("alpha", None, "zeta", Duration::from_secs(1)).await.is_err());
    // The target's engine won't start: alpha stays on a and keeps answering.
    b.loaded.lock().unwrap().insert("alpha".into(), false);
    b.load_fails.store(true, Ordering::SeqCst);
    assert!(c.move_model("alpha", None, "b", Duration::from_secs(1)).await.is_err());
    assert!(a.is_loaded("alpha"));
    assert_eq!(ask(&c, "alpha").await.unwrap(), "a");
}
