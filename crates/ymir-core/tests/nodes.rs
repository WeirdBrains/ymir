//! M4: several model nodes behind one inference role, in one process with scripted nodes:
//! the merged catalog, routing by model and load, failover before the first token, no failover
//! after it, drain and resume, and nodes joining and leaving.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use futures::stream;
use serde_json::json;
use weirdnet_agent::remote::RemoteError;
use weirdnet_agent::remote::catalog::ModelState as AgentState;
use weirdnet_agent::remote::infer::v2::{Request, Usage};
use weirdnet_agent::{CancelToken, Message, ToolCall};
use weirdnet_backends::{Chunk, ChunkStream, ModelState, ModelStatus};
use weirdnet_semcache::NoCache;
use ymir_core::coordinator::{Coordinator, Sink};
use ymir_core::model_node::{BoxFuture, Embedded, ModelNode};

/// How a scripted node answers a chat.
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// Streams its name and the question.
    Works,
    /// Fails to start the stream: the node is gone.
    Dead,
    /// Streams one chunk, then breaks.
    DiesMidAnswer,
    /// The asker's request is bad; no node would take it.
    Rejects,
}

struct Scripted {
    name: String,
    models: Vec<(&'static str, bool, ModelState)>,
    mode: Mutex<Mode>,
    draining: AtomicBool,
    chats: AtomicUsize,
    loads: AtomicUsize,
    load_fails: AtomicBool,
}

impl Scripted {
    fn new(name: &str, models: &[(&'static str, bool, ModelState)]) -> Arc<Self> {
        Arc::new(Self {
            name: name.into(),
            models: models.to_vec(),
            mode: Mutex::new(Mode::Works),
            draining: AtomicBool::new(false),
            chats: AtomicUsize::new(0),
            loads: AtomicUsize::new(0),
            load_fails: AtomicBool::new(false),
        })
    }
    fn set(&self, m: Mode) {
        *self.mode.lock().unwrap() = m;
    }
}

fn status(id: &str, embedding: bool, state: ModelState) -> ModelStatus {
    serde_json::from_value(json!({
        "id": id, "engine": "llamacpp", "state": state, "in_flight": 0, "swap_group": null,
        "embedding": embedding, "pid": null, "last_load_secs": null, "size_bytes": null, "wrapped": false, "crashes": 0
    }))
    .unwrap()
}

impl ModelNode for Scripted {
    fn name(&self) -> &str {
        &self.name
    }
    fn status(&self) -> Vec<ModelStatus> {
        self.models.iter().map(|(id, e, s)| status(id, *e, s.clone())).collect()
    }
    fn load<'a>(&'a self, _model: &'a str, _force: bool) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(async move {
            self.loads.fetch_add(1, Ordering::SeqCst);
            if self.load_fails.load(Ordering::SeqCst) { Err(RemoteError::unavailable("engine won't start")) } else { Ok(()) }
        })
    }
    fn unload<'a>(&'a self, _model: &'a str) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(async { Ok(()) })
    }
    fn chat<'a>(&'a self, model: &'a str, req: &'a Request) -> BoxFuture<'a, Result<ChunkStream, RemoteError>> {
        Box::pin(async move {
            self.chats.fetch_add(1, Ordering::SeqCst);
            let mode = *self.mode.lock().unwrap();
            let text = format!("{} says {}", self.name, model);
            let _ = req;
            let items: Vec<Result<Chunk, RemoteError>> = match mode {
                Mode::Works => vec![Ok(Chunk::Text(text)), Ok(Chunk::Done { finish_reason: Some("stop".into()), usage: Some(Usage::new(1, 1)) })],
                Mode::Dead => return Err(RemoteError::Unreachable { message: format!("{} is gone", self.name) }),
                Mode::DiesMidAnswer => vec![Ok(Chunk::Text(text)), Err(RemoteError::Closed)],
                Mode::Rejects => return Err(RemoteError::BadRequest { message: "no".into() }),
            };
            Ok(Box::pin(stream::iter(items)) as ChunkStream)
        })
    }
    fn embed<'a>(&'a self, _model: &'a str, input: &'a [String]) -> BoxFuture<'a, Result<Embedded, RemoteError>> {
        Box::pin(async move {
            if *self.mode.lock().unwrap() == Mode::Dead {
                return Err(RemoteError::Unreachable { message: "gone".into() });
            }
            Ok((input.iter().map(|_| vec![1.0, 2.0]).collect(), 3))
        })
    }
    fn draining(&self) -> bool {
        self.draining.load(Ordering::SeqCst)
    }
    fn drain<'a>(&'a self, on: bool) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(async move {
            self.draining.store(on, Ordering::SeqCst);
            Ok(())
        })
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

async fn ask(c: &Coordinator, model: &str) -> Result<String, RemoteError> {
    let req = Request::messages(vec![Message::user("hi")]).with_model(model);
    let prepared = c.prepare(&req)?;
    let out = Collect::default();
    let f = c.run(prepared, &req, &out, &CancelToken::new()).await?;
    let text = out.0.lock().unwrap().clone();
    assert_eq!(f.finish_reason.as_deref(), Some("stop"));
    Ok(text)
}

fn coordinator(nodes: &[&Arc<Scripted>]) -> Coordinator {
    let nodes: Vec<Arc<dyn ModelNode>> = nodes.iter().map(|n| (*n).clone() as Arc<dyn ModelNode>).collect();
    Coordinator::new(nodes, Some("alpha".into()), Arc::new(NoCache))
}

const READY: ModelState = ModelState::Ready;

#[tokio::test]
async fn the_catalog_merges_nodes_with_different_models() {
    let a = Scripted::new("a", &[("alpha", false, READY), ("emb", true, READY)]);
    let b = Scripted::new("b", &[("beta", false, READY), ("alpha", false, ModelState::Unloaded)]);
    let c = coordinator(&[&a, &b]);

    let rows = c.merged_catalog();
    let ids: Vec<&str> = rows.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(ids, ["alpha", "beta", "emb"]);
    let alpha = &rows[0];
    assert_eq!(alpha.nodes.len(), 2, "alpha is on both nodes");
    assert!(alpha.ready());
    assert!(rows[2].embedding);

    // catalog/1: alpha is warm with one ready replica, beta warm, emb an embedding model.
    let cat = c.agent_catalog();
    let alpha = cat.iter().find(|m| m.id == "alpha").unwrap();
    assert_eq!((alpha.state.clone(), alpha.replicas), (AgentState::Warm, 1));

    // Each model is answered by the node that has it.
    assert_eq!(ask(&c, "alpha").await.unwrap(), "a says alpha");
    assert_eq!(ask(&c, "beta").await.unwrap(), "b says beta");
    assert!(matches!(
        ask(&c, "gamma").await,
        Err(RemoteError::Unavailable { .. }) | Err(RemoteError::BadRequest { .. }) | Err(RemoteError::Failed { .. })
    ));
}

#[tokio::test]
async fn a_dead_node_fails_over_before_the_first_token() {
    let a = Scripted::new("a", &[("alpha", false, READY)]);
    let b = Scripted::new("b", &[("alpha", false, READY)]);
    let c = coordinator(&[&a, &b]);
    a.set(Mode::Dead);
    b.set(Mode::Dead);
    // Both dead: the error reaches the asker.
    assert!(matches!(ask(&c, "alpha").await, Err(RemoteError::Unreachable { .. })));
    // One back: whichever the router tries first, the answer comes from the live one.
    b.set(Mode::Works);
    for _ in 0..3 {
        assert_eq!(ask(&c, "alpha").await.unwrap(), "b says alpha");
    }
    assert!(a.chats.load(Ordering::SeqCst) >= 1, "the dead node was tried");
}

#[tokio::test]
async fn a_cold_load_that_fails_moves_on() {
    let a = Scripted::new("a", &[("alpha", false, ModelState::Unloaded)]);
    let b = Scripted::new("b", &[("alpha", false, ModelState::Unloaded)]);
    let c = coordinator(&[&a, &b]);
    a.load_fails.store(true, Ordering::SeqCst);
    assert_eq!(ask(&c, "alpha").await.unwrap(), "b says alpha");
    assert_eq!(a.loads.load(Ordering::SeqCst), 1);
    assert_eq!(b.loads.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn nothing_moves_after_the_first_token_or_for_a_bad_request() {
    let a = Scripted::new("a", &[("alpha", false, READY)]);
    let b = Scripted::new("b", &[("alpha", false, READY)]);
    let c = coordinator(&[&a, &b]);
    a.set(Mode::DiesMidAnswer);
    b.set(Mode::DiesMidAnswer);
    // The answer breaks after a token: the asker gets the error (its pool resumes), and the
    // other node isn't asked to start over.
    assert!(ask(&c, "alpha").await.is_err());
    assert_eq!(a.chats.load(Ordering::SeqCst) + b.chats.load(Ordering::SeqCst), 1);

    // A bad request is the asker's mistake, not a node's.
    let (x, y) = (Scripted::new("x", &[("alpha", false, READY)]), Scripted::new("y", &[("alpha", false, READY)]));
    let c = coordinator(&[&x, &y]);
    x.set(Mode::Rejects);
    y.set(Mode::Rejects);
    assert!(matches!(ask(&c, "alpha").await, Err(RemoteError::BadRequest { .. })));
    assert_eq!(x.chats.load(Ordering::SeqCst) + y.chats.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn a_draining_node_takes_no_new_requests_and_resumes() {
    let a = Scripted::new("a", &[("alpha", false, READY)]);
    let b = Scripted::new("b", &[("alpha", false, READY), ("beta", false, READY)]);
    let c = coordinator(&[&a, &b]);

    c.drain("b", true).await.unwrap();
    for _ in 0..3 {
        assert_eq!(ask(&c, "alpha").await.unwrap(), "a says alpha", "b is draining");
    }
    let infos = c.node_infos();
    assert!(infos.iter().find(|n| n.name == "b").unwrap().draining);
    // beta is only on b, which is draining: unavailable, and not warm in the catalog.
    assert!(matches!(ask(&c, "beta").await, Err(RemoteError::Unavailable { .. })));
    let beta = c.agent_catalog().into_iter().find(|m| m.id == "beta").unwrap();
    assert_ne!(beta.state, AgentState::Warm);
    assert!(!c.merged_catalog().iter().find(|r| r.id == "beta").unwrap().ready());
    assert_eq!(b.chats.load(Ordering::SeqCst), 0);

    c.drain("b", false).await.unwrap();
    assert_eq!(ask(&c, "beta").await.unwrap(), "b says beta");
    assert!(c.drain("nope", true).await.is_err());
}

#[tokio::test]
async fn nodes_join_and_leave() {
    let a = Scripted::new("a", &[("alpha", false, READY)]);
    let c = coordinator(&[&a]);
    assert!(matches!(
        ask(&c, "beta").await,
        Err(RemoteError::Unavailable { .. }) | Err(RemoteError::BadRequest { .. }) | Err(RemoteError::Failed { .. })
    ));

    let b = Scripted::new("b", &[("beta", false, READY)]);
    c.add_node(b.clone());
    assert_eq!(c.merged_catalog().len(), 2);
    assert_eq!(ask(&c, "beta").await.unwrap(), "b says beta");

    // A node of the same name replaces the old one (a device that came back).
    let b2 = Scripted::new("b", &[("beta", false, READY), ("gamma", false, READY)]);
    c.add_node(b2);
    assert_eq!(c.nodes().len(), 2);
    assert_eq!(c.merged_catalog().len(), 3);

    assert!(c.remove_node("b"));
    assert!(!c.remove_node("b"));
    assert_eq!(c.merged_catalog().len(), 1);
    assert!(ask(&c, "beta").await.is_err());
    assert_eq!(ask(&c, "alpha").await.unwrap(), "a says alpha");
}

#[tokio::test]
async fn embeddings_fail_over_too() {
    let a = Scripted::new("a", &[("emb", true, READY)]);
    let b = Scripted::new("b", &[("emb", true, READY)]);
    let c = coordinator(&[&a, &b]);
    a.set(Mode::Dead);
    for _ in 0..2 {
        let (v, tokens, node) = c.embed("emb", &["x".to_string()]).await.unwrap();
        assert_eq!((v.len(), tokens, node.as_str()), (1, 3, "b"));
    }
    c.drain("b", true).await.unwrap();
    assert!(c.embed("emb", &["x".to_string()]).await.is_err(), "a is dead and b is draining");
}
