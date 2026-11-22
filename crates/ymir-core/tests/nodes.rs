//! M4 and M5: several model nodes behind one inference role, in one process with scripted nodes:
//! the merged catalog, routing by model and load, failover before the first token, no failover
//! after it, resuming an answer on another node from the text already streamed, drain and resume,
//! and nodes joining and leaving.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use futures::stream;
use serde_json::json;
use weirdnet_agent::remote::RemoteError;
use weirdnet_agent::remote::catalog::ModelState as AgentState;
use weirdnet_agent::remote::infer::v2::{CacheMode, Prefix, Request, Usage};
use weirdnet_agent::{CancelToken, ContentPart, Message, Modalities, ToolCall};
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
    /// Streams [`SENTENCE`] a word at a time (after any prefix it is given, as an engine does).
    Words,
    /// Like `Words`, but breaks after this many words.
    WordsThenDies(usize),
}

/// What the `Words` modes say, whichever node says it.
const SENTENCE: &str = "one two three four five six seven eight ";

struct Scripted {
    name: String,
    models: Vec<(&'static str, bool, ModelState)>,
    mode: Mutex<Mode>,
    /// The `assistant_prefix` of each chat it was asked.
    prefixes: Mutex<Vec<Option<String>>>,
    draining: AtomicBool,
    chats: AtomicUsize,
    loads: AtomicUsize,
    load_fails: AtomicBool,
    /// Milliseconds a chat takes to start answering.
    delay_ms: AtomicUsize,
    /// Whether its models take images and audio.
    media: AtomicBool,
    /// The weights revision it reports for every model.
    revision: Mutex<Option<String>>,
}

impl Scripted {
    fn new(name: &str, models: &[(&'static str, bool, ModelState)]) -> Arc<Self> {
        Arc::new(Self {
            name: name.into(),
            models: models.to_vec(),
            mode: Mutex::new(Mode::Works),
            prefixes: Mutex::new(Vec::new()),
            draining: AtomicBool::new(false),
            chats: AtomicUsize::new(0),
            loads: AtomicUsize::new(0),
            load_fails: AtomicBool::new(false),
            delay_ms: AtomicUsize::new(0),
            media: AtomicBool::new(false),
            revision: Mutex::new(None),
        })
    }
    fn set(&self, m: Mode) {
        *self.mode.lock().unwrap() = m;
    }
    fn set_revision(&self, rev: Option<&str>) {
        *self.revision.lock().unwrap() = rev.map(str::to_string);
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
        self.models
            .iter()
            .map(|(id, e, s)| {
                let mut st = status(id, *e, s.clone());
                if self.media.load(Ordering::SeqCst) {
                    st.capabilities = Some(weirdnet_backends::Capabilities::text().with_vision(true).with_audio(true));
                }
                st
            })
            .collect()
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
            let delay = self.delay_ms.load(Ordering::SeqCst);
            if delay > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(delay as u64)).await;
            }
            let mode = *self.mode.lock().unwrap();
            let text = format!("{} says {}", self.name, model);
            let prefix = req.assistant_prefix.as_ref().map(|p| p.text.clone());
            self.prefixes.lock().unwrap().push(prefix.clone());
            let words = |die_after: Option<usize>| -> Vec<Result<Chunk, RemoteError>> {
                let rest = SENTENCE.strip_prefix(prefix.as_deref().unwrap_or("")).expect("a prefix of the sentence");
                let mut items: Vec<Result<Chunk, RemoteError>> = Vec::new();
                for (i, w) in rest.split_inclusive(' ').enumerate() {
                    if die_after == Some(i) {
                        items.push(Err(RemoteError::Closed));
                        return items;
                    }
                    items.push(Ok(Chunk::Text(w.to_string())));
                }
                items.push(Ok(Chunk::Done { finish_reason: Some("stop".into()), usage: Some(Usage::new(1, 1)) }));
                items
            };
            let items: Vec<Result<Chunk, RemoteError>> = match mode {
                Mode::Works => vec![Ok(Chunk::Text(text)), Ok(Chunk::Done { finish_reason: Some("stop".into()), usage: Some(Usage::new(1, 1)) })],
                Mode::Dead => return Err(RemoteError::Unreachable { message: format!("{} is gone", self.name) }),
                Mode::DiesMidAnswer => vec![Ok(Chunk::Text(text)), Err(RemoteError::Closed)],
                Mode::Rejects => return Err(RemoteError::BadRequest { message: "no".into() }),
                Mode::Words => words(None),
                Mode::WordsThenDies(n) => words(Some(n)),
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
    fn revision(&self, _model: &str) -> Option<String> {
        self.revision.lock().unwrap().clone()
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

/// With the exact-match answer cache on, as `ymird` runs it by default.
fn cached(nodes: &[&Arc<Scripted>]) -> Coordinator {
    let nodes: Vec<Arc<dyn ModelNode>> = nodes.iter().map(|n| (*n).clone() as Arc<dyn ModelNode>).collect();
    Coordinator::new(nodes, Some("alpha".into()), Arc::new(weirdnet_semcache::ExactCache::<weirdnet_semcache::MemoryStore>::new(16)))
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
    assert_eq!((alpha.state, alpha.replicas), (AgentState::Warm, 1));

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
async fn a_bad_request_is_not_moved_and_a_node_that_keeps_breaking_ends_the_answer() {
    let a = Scripted::new("a", &[("alpha", false, READY)]);
    let b = Scripted::new("b", &[("alpha", false, READY)]);
    let c = coordinator(&[&a, &b]);
    a.set(Mode::DiesMidAnswer);
    b.set(Mode::DiesMidAnswer);
    // The answer breaks after a token and again on the node it resumed on, with no node left:
    // the asker gets the error (its pool may resume elsewhere).
    assert!(ask(&c, "alpha").await.is_err());
    assert_eq!(a.chats.load(Ordering::SeqCst) + b.chats.load(Ordering::SeqCst), 2);

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

async fn say(c: &Coordinator, req: Request) -> (Result<(), RemoteError>, String) {
    let prepared = c.prepare(&req).unwrap();
    let out = Collect::default();
    let r = c.run(prepared, &req, &out, &CancelToken::new()).await.map(|_| ());
    let text = out.0.lock().unwrap().clone();
    (r, text)
}

#[tokio::test]
async fn an_answer_resumes_on_another_node_from_the_text_already_streamed() {
    let a = Scripted::new("a", &[("alpha", false, READY)]);
    let b = Scripted::new("b", &[("alpha", false, READY)]);
    let c = coordinator(&[&a, &b]);
    // Whichever node is picked first breaks after three words; drain the other so the pick is a's.
    b.drain(true).await.unwrap();
    a.set(Mode::WordsThenDies(3));
    b.set(Mode::Words);
    let req = Request::messages(vec![Message::user("count")]).with_model("alpha");
    let prepared = c.prepare(&req).unwrap();
    b.drain(false).await.unwrap();
    let out = Collect::default();
    let f = c.run(prepared, &req, &out, &CancelToken::new()).await.unwrap();
    // The text is whole and exact: b continued from "one two three ", nothing repeated.
    assert_eq!(*out.0.lock().unwrap(), SENTENCE);
    assert_eq!(*a.prefixes.lock().unwrap(), [None]);
    assert_eq!(*b.prefixes.lock().unwrap(), [Some("one two three ".to_string())]);
    assert_eq!(f.meta.resumed_from, Some(0));
    assert_eq!(f.meta.served_by.as_deref(), Some("b"));
}

/// A question the cache will keep (deterministic sampling).
fn plain() -> Request {
    let mut r = Request::messages(vec![Message::user("count")]);
    r.sampling.temperature = Some(0.0);
    r
}

#[tokio::test]
async fn a_request_with_a_prefix_continues_it_and_skips_the_cache() {
    let a = Scripted::new("a", &[("alpha", false, READY)]);
    a.set(Mode::Words);
    let c = cached(&[&a]);
    let req = plain().with_model("alpha");
    let (r, full) = say(&c, req.clone()).await;
    r.unwrap();
    assert_eq!(full, SENTENCE);
    // Asked again with the first words as the prefix (the asker's pool resuming here): only the
    // rest streams, and it comes from the engine, not from the cached whole answer.
    let prefix = "one two three ";
    let (r, rest) = say(&c, req.with_prefix(Prefix::new(prefix, ""))).await;
    r.unwrap();
    assert_eq!(format!("{prefix}{rest}"), SENTENCE);
    assert_eq!(a.chats.load(Ordering::SeqCst), 2, "the prefixed request reached the engine");
    // The plain request again is a cache hit: no third engine call.
    say(&c, plain().with_model("alpha")).await.0.unwrap();
    assert_eq!(a.chats.load(Ordering::SeqCst), 2);
    // And a resumed answer is not cached as if it were whole.
    let b = Scripted::new("b", &[("beta", false, READY)]);
    b.set(Mode::Words);
    let c = cached(&[&b]);
    let req = plain().with_model("beta").with_prefix(Prefix::new(prefix, ""));
    say(&c, req).await.0.unwrap();
    let (r, full) = say(&c, plain().with_model("beta")).await;
    r.unwrap();
    assert_eq!(full, SENTENCE, "the plain request got a whole answer");
    assert_eq!(b.chats.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn replicas_share_a_burst_and_a_slow_one_gets_less() {
    let nodes: Vec<_> = ["a", "b", "c"].iter().map(|n| Scripted::new(n, &[("alpha", false, READY)])).collect();
    for n in &nodes {
        n.delay_ms.store(40, Ordering::SeqCst);
    }
    let c = Arc::new(coordinator(&nodes.iter().collect::<Vec<_>>()));
    for _ in 0..6 {
        let waves: Vec<_> = (0..10).map(|_| ask(&c, "alpha")).collect();
        for r in futures::future::join_all(waves).await {
            r.unwrap();
        }
    }
    let got: Vec<usize> = nodes.iter().map(|n| n.chats.load(Ordering::SeqCst)).collect();
    assert_eq!(got.iter().sum::<usize>(), 60);
    assert!(got.iter().all(|&g| g >= 10), "every replica takes a fair share of a burst: {got:?}");

    // One replica turns slow: after it has been measured, it takes clearly less than the others.
    nodes[0].delay_ms.store(400, Ordering::SeqCst);
    for n in &nodes {
        n.chats.store(0, Ordering::SeqCst);
    }
    for _ in 0..8 {
        let wave: Vec<_> = (0..6).map(|_| ask(&c, "alpha")).collect();
        for r in futures::future::join_all(wave).await {
            r.unwrap();
        }
    }
    let got: Vec<usize> = nodes.iter().map(|n| n.chats.load(Ordering::SeqCst)).collect();
    assert!(got[0] < got[1] && got[0] < got[2], "the slow replica gets the least: {got:?}");
}

fn with_image() -> Request {
    Request::messages(vec![Message::user(vec![ContentPart::text("what is this"), ContentPart::image("data:image/png;base64,AAAA")])])
}

#[tokio::test]
async fn images_and_audio_go_only_to_nodes_that_take_them() {
    // The same model name on two nodes, but only b's copy has a projector.
    let a = Scripted::new("a", &[("alpha", false, READY)]);
    let b = Scripted::new("b", &[("alpha", false, READY)]);
    b.media.store(true, Ordering::SeqCst);
    let c = coordinator(&[&a, &b]);

    let alpha = c.merged_catalog().remove(0);
    assert_eq!(alpha.inputs, Modalities::new(true, true), "the entry lists what some node takes");
    assert_eq!(c.agent_catalog()[0].inputs, Modalities::new(true, true));

    for _ in 0..8 {
        let (r, text) = say(&c, with_image().with_model("alpha")).await;
        r.unwrap();
        assert_eq!(text, "b says alpha");
    }
    assert_eq!(a.chats.load(Ordering::SeqCst), 0, "a text-only node never saw an image");

    // Text is not restricted: it may go to either node.
    ask(&c, "alpha").await.unwrap();

    // With no model named, the pool picks one that takes images, even if the main model doesn't.
    let text_only = Scripted::new("t", &[("alpha", false, READY)]);
    let vision = Scripted::new("v", &[("gemma", false, READY)]);
    vision.media.store(true, Ordering::SeqCst);
    let c = coordinator(&[&text_only, &vision]);
    let (r, text) = say(&c, with_image()).await;
    r.unwrap();
    assert_eq!(text, "v says gemma");

    // A model that no node reads media with is refused, not tried.
    let c = coordinator(&[&text_only]);
    let e = c.prepare(&with_image().with_model("alpha")).err().expect("refused");
    assert!(matches!(&e, RemoteError::BadRequest { message } if message.contains("does not take images")), "{e:?}");
    let e = c.prepare(&with_image()).err().expect("no model reads it");
    assert!(matches!(e, RemoteError::BadRequest { .. }), "{e:?}");
}

#[tokio::test]
async fn the_request_cache_mode_skips_or_refreshes_the_answer_cache() {
    let a = Scripted::new("a", &[("alpha", false, READY)]);
    let c = cached(&[&a]);
    let chats = || a.chats.load(Ordering::SeqCst);
    // `bypass` neither reads nor stores: two engine calls, and still nothing cached after.
    let bypass = plain().with_model("alpha").with_cache(CacheMode::Bypass);
    say(&c, bypass.clone()).await.0.unwrap();
    say(&c, bypass).await.0.unwrap();
    assert_eq!(chats(), 2);
    assert!(!c.prepare(&plain().with_model("alpha")).unwrap().cache_hit(), "bypass stored nothing");
    // `refresh` skips the read but stores the fresh answer, which `use` then reads.
    let refresh = plain().with_model("alpha").with_cache(CacheMode::Refresh);
    assert!(!c.prepare(&refresh).unwrap().cache_hit());
    say(&c, refresh).await.0.unwrap();
    assert_eq!(chats(), 3);
    assert!(c.prepare(&plain().with_model("alpha")).unwrap().cache_hit());
    // A cached answer is there, but `bypass` and `refresh` still go to the engine.
    assert!(!c.prepare(&plain().with_model("alpha").with_cache(CacheMode::Bypass)).unwrap().cache_hit());
    assert!(!c.prepare(&plain().with_model("alpha").with_cache(CacheMode::Refresh)).unwrap().cache_hit());
}

#[tokio::test]
async fn cached_answers_are_keyed_on_the_weights_revision() {
    let a = Scripted::new("a", &[("alpha", false, READY)]);
    a.set_revision(Some("r1"));
    let c = cached(&[&a]);
    let req = || plain().with_model("alpha");
    let prepared = c.prepare(&req()).unwrap();
    let out = Collect::default();
    let f = c.run(prepared, &req(), &out, &CancelToken::new()).await.unwrap();
    assert_eq!(f.meta.revision.as_deref(), Some("r1"), "the answer says which weights gave it");
    assert!(c.prepare(&req()).unwrap().cache_hit(), "stored under r1");
    // New weights under the same id: the old answer is not replayed.
    a.set_revision(Some("r2"));
    assert!(!c.prepare(&req()).unwrap().cache_hit());
    say(&c, req()).await.0.unwrap();
    assert!(c.prepare(&req()).unwrap().cache_hit(), "stored under r2");
    // A node with no revision: the answers keyed on r1 and r2 aren't its own.
    a.set_revision(None);
    assert!(!c.prepare(&req()).unwrap().cache_hit());
    // Replicas on different weights: which one answers isn't known in advance, so no read.
    let b = Scripted::new("b", &[("alpha", false, READY)]);
    a.set_revision(Some("r2"));
    b.set_revision(Some("r3"));
    let c2 = cached(&[&a, &b]);
    say(&c2, req()).await.0.unwrap();
    assert!(!c2.prepare(&req()).unwrap().cache_hit());
    // Both on r2 again: the cache reads under r2.
    b.set_revision(Some("r2"));
    say(&c2, req()).await.0.unwrap();
    assert!(c2.prepare(&req()).unwrap().cache_hit());
}
