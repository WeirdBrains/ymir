//! M6 semantic answer cache: a finished answer stands in for a later prompt that means the same
//! (by the pool's embedding model), only within the same model, weights, system prompt and
//! sampling, never by cold-starting the embedding model, and not for `bypass` requests.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use futures::stream;
use serde_json::json;
use weirdnet_agent::remote::RemoteError;
use weirdnet_agent::remote::infer::v2::{CacheMode, Request, Usage};
use weirdnet_agent::{CancelToken, Message, ToolCall};
use weirdnet_backends::{Chunk, ChunkStream, ModelState, ModelStatus};
use ymir_core::coordinator::{Coordinator, CoordinatorOptions, Sink};
use ymir_core::model_node::{BoxFuture, Embedded, ModelNode};
use ymir_core::semantic::SemanticOptions;

/// Words the toy embedding counts: prompts with the same words (in any order or case, with any
/// punctuation) embed the same.
const VOCAB: [&str; 6] = ["rust", "python", "fast", "what", "is", "weather"];

fn embed_text(s: &str) -> Vec<f32> {
    let words: Vec<String> = s.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(str::to_string).collect();
    let mut v: Vec<f32> = VOCAB.iter().map(|w| words.iter().filter(|x| x == w).count() as f32).collect();
    v.push(0.01); // never all zero
    v
}

struct Node {
    chats: AtomicUsize,
    embeds: AtomicUsize,
    embed_ready: AtomicBool,
    revision: Mutex<Option<String>>,
}

fn status(id: &str, embedding: bool, state: ModelState) -> ModelStatus {
    serde_json::from_value(json!({
        "id": id, "engine": "llamacpp", "state": state, "in_flight": 0, "swap_group": null,
        "embedding": embedding, "pid": null, "last_load_secs": null, "size_bytes": null, "wrapped": false, "crashes": 0
    }))
    .unwrap()
}

impl ModelNode for Node {
    fn name(&self) -> &str {
        "a"
    }
    fn status(&self) -> Vec<ModelStatus> {
        let e = if self.embed_ready.load(Ordering::SeqCst) { ModelState::Ready } else { ModelState::Unloaded };
        vec![status("alpha", false, ModelState::Ready), status("embed", true, e)]
    }
    fn load<'a>(&'a self, _model: &'a str, _force: bool) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(async { Ok(()) })
    }
    fn unload<'a>(&'a self, _model: &'a str) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(async { Ok(()) })
    }
    fn chat<'a>(&'a self, _model: &'a str, _req: &'a Request) -> BoxFuture<'a, Result<ChunkStream, RemoteError>> {
        Box::pin(async move {
            let n = self.chats.fetch_add(1, Ordering::SeqCst) + 1;
            let items =
                vec![Ok(Chunk::Text(format!("answer {n}"))), Ok(Chunk::Done { finish_reason: Some("stop".into()), usage: Some(Usage::new(1, 1)) })];
            Ok(Box::pin(stream::iter(items)) as ChunkStream)
        })
    }
    fn embed<'a>(&'a self, _model: &'a str, input: &'a [String]) -> BoxFuture<'a, Result<Embedded, RemoteError>> {
        Box::pin(async move {
            self.embeds.fetch_add(1, Ordering::SeqCst);
            Ok((input.iter().map(|s| embed_text(s)).collect(), 1))
        })
    }
    fn revision(&self, _model: &str) -> Option<String> {
        self.revision.lock().unwrap().clone()
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

fn setup(threshold_similarity: f32) -> (Arc<Node>, Coordinator) {
    let node = Arc::new(Node {
        chats: AtomicUsize::new(0),
        embeds: AtomicUsize::new(0),
        embed_ready: AtomicBool::new(true),
        revision: Mutex::new(Some("r1".into())),
    });
    let opts = CoordinatorOptions {
        semantic: Some(SemanticOptions {
            embedding_model: "embed".into(),
            similarity: threshold_similarity,
            ttl: None,
            max_entries: 64,
            pool: false,
        }),
        ..Default::default()
    };
    let nodes: Vec<Arc<dyn ModelNode>> = vec![node.clone()];
    // No exact cache, so every hit here is the semantic one.
    (node, Coordinator::with_options(nodes, Some("alpha".into()), Arc::new(weirdnet_semcache::NoCache), opts))
}

fn req(system: &str, user: &str) -> Request {
    Request::messages(vec![Message::system(system), Message::user(user)]).with_model("alpha")
}

/// The answer text and who served it.
async fn ask(c: &Coordinator, r: Request) -> (String, Option<String>) {
    let p = c.prepare(&r).unwrap();
    let out = Collect::default();
    let f = c.run(p, &r, &out, &CancelToken::new()).await.unwrap();
    let text = out.0.lock().unwrap().clone();
    (text, f.meta.served_by)
}

#[tokio::test]
async fn a_rephrased_prompt_replays_the_answer() {
    let (node, c) = setup(0.88);
    assert_eq!(ask(&c, req("brief", "What is Rust?")).await, ("answer 1".into(), Some("a".into())));
    assert_eq!(c.semantic_entries(), 1);
    // Same words, other case and punctuation: the stored answer, no engine call.
    assert_eq!(ask(&c, req("brief", "what is rust")).await, ("answer 1".into(), Some("semantic-cache".into())));
    assert_eq!(node.chats.load(Ordering::SeqCst), 1);
    // A different question goes to the model.
    assert_eq!(ask(&c, req("brief", "is python fast")).await.0, "answer 2");
    // Another system prompt, or other sampling, is another scope.
    assert_eq!(ask(&c, req("verbose", "what is rust")).await.0, "answer 3");
    let mut hot = req("brief", "what is rust");
    hot.sampling.temperature = Some(1.5);
    assert_eq!(ask(&c, hot).await.0, "answer 4");
    // New weights under the same id: nothing from r1 is replayed.
    *node.revision.lock().unwrap() = Some("r2".into());
    assert_eq!(ask(&c, req("brief", "what is rust")).await.0, "answer 5");
    assert_eq!(ask(&c, req("brief", "What is Rust!")).await.0, "answer 5");
}

#[tokio::test]
async fn bypass_refresh_and_follow_ups() {
    let (node, c) = setup(0.88);
    ask(&c, req("s", "what is rust")).await;
    // `bypass` neither reads nor stores (and doesn't embed).
    let embeds = node.embeds.load(Ordering::SeqCst);
    assert_eq!(ask(&c, req("s", "what is rust").with_cache(CacheMode::Bypass)).await.0, "answer 2");
    assert_eq!(node.embeds.load(Ordering::SeqCst), embeds);
    // `refresh` skips the read and stores the fresh answer, which the next one replays.
    assert_eq!(ask(&c, req("s", "what is rust").with_cache(CacheMode::Refresh)).await.0, "answer 3");
    assert_eq!(ask(&c, req("s", "rust: what is")).await.0, "answer 3");
    // A follow-up turn (history) is not a single user turn: never from the semantic cache.
    let follow = Request::messages(vec![Message::user("what is rust"), Message::assistant("answer", vec![]), Message::user("what is rust")])
        .with_model("alpha");
    assert_eq!(ask(&c, follow).await.0, "answer 4");
}

#[tokio::test]
async fn the_embedding_model_is_never_cold_started() {
    let (node, c) = setup(0.88);
    node.embed_ready.store(false, Ordering::SeqCst);
    ask(&c, req("s", "what is rust")).await;
    ask(&c, req("s", "what is rust")).await;
    assert_eq!(node.embeds.load(Ordering::SeqCst), 0, "not loaded: skipped, not loaded for it");
    assert_eq!(node.chats.load(Ordering::SeqCst), 2);
    assert_eq!(c.semantic_entries(), 0);
}

#[tokio::test]
async fn a_strict_threshold_needs_closer_prompts() {
    let (_node, c) = setup(0.999);
    ask(&c, req("s", "what is rust")).await;
    // One extra word drops the similarity below 0.999.
    assert_eq!(ask(&c, req("s", "what is rust fast")).await.0, "answer 2");
    assert_eq!(ask(&c, req("s", "WHAT is rust?")).await.0, "answer 1");
}
