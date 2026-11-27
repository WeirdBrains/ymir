//! M6 pool-mode semantic cache (`[inference.cache] mode = "pool"`, `weirdnet-semcache` 0.2.3):
//! three inference nodes (coordinators with their own model node) share one cache through
//! `PooledSemanticCache`, over an in-process transport that calls each node's
//! `SharedCacheHost` (the same bytes as `/weirdnet-semcache/shared/1`) and can cut a node off.
//! Over Weirdnet itself: `ymir-testkit/tests/m6_semcache_pool.rs`.
//!
//! - An answer stored through one node replays on another for a rephrased prompt: from the
//!   scope's home, then from the asker's L1 copy; the replica got the background write.
//! - The home unreachable: the replica answers. Both unreachable: a degraded miss, quickly,
//!   and the model answers; nothing blocks.
//! - Removing a scope removes it on every member.
//! - One node alone: no transport call at all.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures::stream;
use serde_json::json;
use weirdnet_agent::remote::RemoteError;
use weirdnet_agent::remote::infer::v2::{Request, Usage};
use weirdnet_agent::{CancelToken, Message, ToolCall};
use weirdnet_backends::{Chunk, ChunkStream, ModelState, ModelStatus};
use weirdnet_semcache::pool::{CacheTransport, PoolConfig};
use weirdnet_semcache::{CachedAnswer, SharedCacheHost, SharedCacheReq, SharedCacheResp};
use ymir_core::coordinator::{Coordinator, CoordinatorOptions, Sink};
use ymir_core::model_node::{BoxFuture, Embedded, ModelNode};
use ymir_core::semantic::{Pooled, SemanticOptions};

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

/// The in-process network: each member's host, members cut off, and calls made.
#[derive(Default)]
struct Net {
    hosts: Mutex<HashMap<String, SharedCacheHost<CachedAnswer>>>,
    cut: Mutex<HashSet<String>>,
    calls: AtomicUsize,
}

struct Link(Arc<Net>);

#[async_trait::async_trait]
impl CacheTransport<CachedAnswer> for Link {
    async fn call(&self, node: &str, req: SharedCacheReq<CachedAnswer>) -> Result<SharedCacheResp<CachedAnswer>, RemoteError> {
        self.0.calls.fetch_add(1, Ordering::SeqCst);
        if self.0.cut.lock().unwrap().contains(node) {
            // An unreachable node: the call hangs until the pool's timeout.
            tokio::time::sleep(Duration::from_secs(30)).await;
            return Err(RemoteError::Timeout);
        }
        let host = self.0.hosts.lock().unwrap().get(node).cloned().ok_or(RemoteError::Closed)?;
        // Through the wire encoding, as over /weirdnet-semcache/shared/1.
        SharedCacheResp::decode(&host.handle_bytes(&req.encode()?)?)
    }
}

struct Member {
    name: String,
    node: Arc<Node>,
    coord: Coordinator,
    pool: Arc<Pooled>,
}

fn member(name: &str, net: &Arc<Net>) -> Member {
    let node = Arc::new(Node {
        chats: AtomicUsize::new(0),
        embeds: AtomicUsize::new(0),
        embed_ready: AtomicBool::new(true),
        revision: Mutex::new(Some("r1".into())),
    });
    let opts = CoordinatorOptions {
        semantic: Some(SemanticOptions { embedding_model: "embed".into(), similarity: 0.88, ttl: None, max_entries: 64, pool: true }),
        ..Default::default()
    };
    let coord = Coordinator::with_options(vec![node.clone() as Arc<dyn ModelNode>], Some("alpha".into()), Arc::new(weirdnet_semcache::NoCache), opts);
    let sem = coord.semantic().unwrap();
    let pool = Arc::new(Pooled::new(name, sem.shard().clone(), Arc::new(Link(net.clone())), PoolConfig::new()));
    assert!(sem.attach_pool(pool.clone()));
    assert!(!sem.attach_pool(pool.clone()), "attached twice");
    net.hosts.lock().unwrap().insert(name.into(), pool.host());
    Member { name: name.into(), node, coord, pool }
}

fn pool_of(names: &[&str]) -> (Arc<Net>, Vec<Member>) {
    let net = Arc::new(Net::default());
    let members: Vec<Member> = names.iter().map(|n| member(n, &net)).collect();
    for m in &members {
        m.pool.set_nodes(names.iter().copied());
    }
    (net, members)
}

fn req(user: &str) -> Request {
    Request::messages(vec![Message::system("brief"), Message::user(user)]).with_model("alpha")
}

async fn ask(c: &Coordinator, r: Request) -> (String, Option<String>) {
    let p = c.prepare(&r).unwrap();
    let out = Collect::default();
    let f = c.run(p, &r, &out, &CancelToken::new()).await.unwrap();
    let text = out.0.lock().unwrap().clone();
    (text, f.meta.served_by)
}

/// The scope every `req` lands in, and its home and replica.
fn owners(ms: &[Member]) -> (String, String, String) {
    let sem = ms[0].coord.semantic().unwrap();
    let ctx = ymir_core::semantic::cacheable(&req("x")).unwrap().context;
    let scope = sem.scope("alpha", Some("r1"), Some("r1"), &ctx);
    let plan = ms[0].pool.plan(&scope).unwrap();
    (scope, plan.home, plan.replica.unwrap())
}

fn by<'a>(ms: &'a [Member], name: &str) -> &'a Member {
    ms.iter().find(|m| m.name == name).unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn an_answer_stored_on_one_node_replays_on_another() {
    let (_net, ms) = pool_of(&["a", "b", "c"]);
    let (scope, home, replica) = owners(&ms);
    let other = ["a", "b", "c"].into_iter().find(|n| *n != home && *n != replica).unwrap();
    let (writer, asker) = (by(&ms, &replica), by(&ms, other));

    // Answered by the replica node's model; stored on the home (waited for) and the replica.
    assert_eq!(ask(&writer.coord, req("What is Rust?")).await, ("answer 1".into(), Some("a".into())));
    for m in &ms {
        m.coord.semantic().unwrap().settle().await;
    }
    assert_eq!(by(&ms, &home).coord.semantic_entries(), 1, "the home holds it");
    assert_eq!(writer.coord.semantic_entries(), 1, "the replica holds it");
    assert_eq!(asker.coord.semantic_entries(), 0, "a third node holds no shard of it");

    // A rephrased prompt on the third node: replayed from the home, no model call there.
    assert_eq!(ask(&asker.coord, req("what is rust")).await, ("answer 1".into(), Some("semantic-cache".into())));
    assert_eq!(asker.node.chats.load(Ordering::SeqCst), 0);
    // Again: from its L1 copy.
    assert_eq!(ask(&asker.coord, req("rust, what is?")).await.1.as_deref(), Some("semantic-cache"));
    let st = asker.coord.semantic().unwrap().status();
    assert_eq!(st.mode, "pool");
    assert_eq!(st.members, vec!["a".to_string(), "b".to_string(), "c".to_string()]);
    let c = st.stats.unwrap();
    assert_eq!((c.peer_hits, c.l1_hits, c.degraded), (1, 1, 0), "{c:?}");

    // Removing the scope removes it everywhere.
    assert!(asker.coord.semantic().unwrap().remove_scope(&scope).await.is_empty());
    assert!(ms.iter().all(|m| m.coord.semantic_entries() == 0));
    let (text, served) = ask(&by(&ms, &home).coord, req("what is rust")).await;
    assert_eq!((text.as_str(), served.as_deref()), ("answer 1", Some("a")), "a fresh answer from the home's model");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_replica_answers_when_the_home_is_unreachable_and_an_outage_is_a_quick_miss() {
    let (net, ms) = pool_of(&["a", "b", "c"]);
    let (_scope, home, replica) = owners(&ms);
    let other = ["a", "b", "c"].into_iter().find(|n| *n != home && *n != replica).unwrap();
    let asker = by(&ms, other);
    ask(&by(&ms, &home).coord, req("is python fast")).await;
    for m in &ms {
        m.coord.semantic().unwrap().settle().await;
    }

    // The home is cut off: the replica answers (after the 150 ms lookup timeout at most).
    net.cut.lock().unwrap().insert(home.clone());
    let t = Instant::now();
    assert_eq!(ask(&asker.coord, req("python is fast?")).await.1.as_deref(), Some("semantic-cache"));
    assert!(t.elapsed() < Duration::from_secs(2), "{:?}", t.elapsed());
    assert!(asker.pool.is_down(&home));
    assert_eq!(asker.coord.semantic().unwrap().status().down, vec![home.clone()]);

    // Home and replica cut off, a new asker with no L1 copy: a degraded miss, answered by its
    // own model at once.
    net.cut.lock().unwrap().insert(replica.clone());
    // A fourth node whose ring leaves the scope on the two cut-off owners.
    let (scope, _, _) = owners(&ms);
    let fresh = ["d", "e", "f", "g", "h", "i", "j", "k"]
        .into_iter()
        .map(|n| {
            let m = member(n, &net);
            m.pool.set_nodes(["a", "b", "c", n]);
            m
        })
        .find(|m| m.pool.plan(&scope).is_some_and(|p| p.home == home && p.replica.as_deref() == Some(replica.as_str())))
        .expect("some fourth node leaves the scope where it was");
    let t = Instant::now();
    assert_eq!(ask(&fresh.coord, req("fast python is")).await.0, "answer 1");
    assert!(t.elapsed() < Duration::from_secs(2), "the outage held the answer up: {:?}", t.elapsed());
    let st = fresh.coord.semantic().unwrap().status().stats.unwrap();
    assert!(st.degraded >= 1 && st.timeouts >= 1, "{st:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn one_node_alone_never_calls_out() {
    let (net, ms) = pool_of(&["solo"]);
    let solo = &ms[0];
    ask(&solo.coord, req("what is rust")).await;
    solo.coord.semantic().unwrap().settle().await;
    assert_eq!(ask(&solo.coord, req("Rust: what is it")).await.1.as_deref(), Some("semantic-cache"));
    assert_eq!(net.calls.load(Ordering::SeqCst), 0);
    let st = solo.coord.semantic().unwrap().status();
    assert_eq!((st.members.len(), st.entries, st.stats.unwrap().shard_hits), (1, 1, 1));
}
