//! M6, the published placement plan with fencing and cloud-wide swap, in one process with fake
//! model nodes and a fake plan book (the lease and `ymir.plan` are exercised over Weirdnet in
//! `ymir-testkit/tests/m6_scheduler_lease.rs`).
//!
//! - The lease holder publishes where its managed models are after a round, and publishes again
//!   only when that changes.
//! - A holder that has seen a plan of a newer term, or whose lease ran out (also mid-round), takes
//!   no step and publishes nothing; a holder never goes back below a term it has seen.
//! - Cloud-wide swap: a cold request loads where it evicts nothing and the node is emptiest;
//!   an unmanaged model idle past `idle_unload_secs` is unloaded, never the main model or a busy
//!   one, and never one asked for recently.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::json;
use weirdnet_agent::remote::RemoteError;
use weirdnet_agent::remote::infer::v2::{Request, Usage};
use weirdnet_agent::{CancelToken, Message, ToolCall};
use weirdnet_backends::{Chunk, ChunkStream, ModelStatus};
use weirdnet_semcache::NoCache;
use weirdnet_sync::Term;
use ymir_core::coordinator::{Coordinator, Sink};
use ymir_core::model_node::{BoxFuture, Embedded, ModelNode};
use ymir_core::placement::{Fence, Lead, PlacementPlan, PlacementPolicy, PlanBook, PublishedPlan, Step};

/// A node holding copies of some models, each loaded or not; a chat takes `delay` and counts
/// as in flight meanwhile.
struct Node {
    name: String,
    loaded: Mutex<HashMap<String, bool>>,
    in_flight: AtomicU32,
    delay: Duration,
    load_fails: AtomicBool,
    group: Option<&'static str>,
}

impl Node {
    fn new(name: &str, models: &[(&str, bool)], delay_ms: u64) -> Arc<Self> {
        Arc::new(Self {
            name: name.into(),
            loaded: Mutex::new(models.iter().map(|(m, l)| (m.to_string(), *l)).collect()),
            in_flight: AtomicU32::new(0),
            delay: Duration::from_millis(delay_ms),
            load_fails: AtomicBool::new(false),
            group: None,
        })
    }
    fn grouped(name: &str, models: &[(&str, bool)], group: &'static str) -> Arc<Self> {
        let mut n = Arc::try_unwrap(Self::new(name, models, 0)).ok().unwrap();
        n.group = Some(group);
        Arc::new(n)
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
                    "id": id, "engine": "llamacpp", "state": state, "in_flight": if *on { flight } else { 0 }, "swap_group": self.group,
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

async fn ask(c: &Coordinator, model: &str) -> Result<String, RemoteError> {
    let req = Request::messages(vec![Message::user("hi")]).with_model(model);
    let p = c.prepare(&req)?;
    let out = Collect::default();
    c.run(p, &req, &out, &CancelToken::new()).await?;
    Ok(out.0.lock().unwrap().clone())
}

fn pool(nodes: &[&Arc<Node>], main: Option<&str>) -> Arc<Coordinator> {
    Arc::new(Coordinator::new(nodes.iter().map(|n| (*n).clone() as Arc<dyn ModelNode>).collect(), main.map(str::to_string), Arc::new(NoCache)))
}

/// A plan book standing in for the lease and `ymir.plan`: this node holds `term` while `holds`;
/// `seen` is the newest plan that reached it (its own, or another holder's).
struct Book {
    term: Term,
    holds: AtomicBool,
    seen: Mutex<Option<PublishedPlan>>,
    published: Mutex<Vec<PublishedPlan>>,
}

impl Book {
    fn new(term: u64) -> Arc<Self> {
        Arc::new(Self { term: Term(term), holds: AtomicBool::new(true), seen: Mutex::new(None), published: Mutex::new(Vec::new()) })
    }
    fn others(&self, term: u64) {
        *self.seen.lock().unwrap() = Some(PublishedPlan { term: Term(term), version: 1, signer: "other".into(), plan: PlacementPlan::default() });
    }
}

impl PlanBook for Book {
    fn lead(&self) -> Lead {
        if self.holds.load(Ordering::SeqCst) { Lead::Held(self.term) } else { Lead::No }
    }
    fn still_holds(&self, term: Term) -> bool {
        self.holds.load(Ordering::SeqCst) && term == self.term
    }
    fn current(&self) -> Option<PublishedPlan> {
        self.seen.lock().unwrap().clone()
    }
    fn publish(&self, term: Term, plan: &PlacementPlan) -> anyhow::Result<PublishedPlan> {
        let mut seen = self.seen.lock().unwrap();
        // As `publish_plan`: refused under a newer term, versioned within one.
        let version = match &*seen {
            Some(p) if p.term > term => anyhow::bail!("a newer term has published"),
            Some(p) if p.term == term => p.version + 1,
            _ => 1,
        };
        let p = PublishedPlan { term, version, signer: "me".into(), plan: plan.clone() };
        *seen = Some(p.clone());
        self.published.lock().unwrap().push(p.clone());
        Ok(p)
    }
}

fn policy(replicas: &[(&str, u32)], idle: Option<u64>) -> PlacementPolicy {
    PlacementPolicy {
        replicas: replicas.iter().map(|(m, n)| (m.to_string(), *n)).collect::<BTreeMap<_, _>>(),
        idle_unload_secs: idle,
        ..Default::default()
    }
}

#[tokio::test]
async fn the_holder_publishes_its_plan_and_republishes_only_on_change() {
    let (a, b, c) = (Node::new("a", &[("alpha", false)], 0), Node::new("b", &[("alpha", false)], 0), Node::new("c", &[("alpha", false)], 0));
    let co = pool(&[&a, &b, &c], None);
    let book = Book::new(3);
    let fence = Fence::new();
    let p = policy(&[("alpha", 2)], None);

    let r = co.place_round(&p, &*book, &fence, None).await;
    assert!(!r.fenced);
    assert_eq!(r.done.len(), 1);
    assert_eq!(r.published, Some((Term(3), 1)));
    let r = co.place_round(&p, &*book, &fence, Some(&co.current_plan(&p))).await;
    assert_eq!(r.published, Some((Term(3), 2)));
    let plan = co.current_plan(&p);
    assert_eq!(plan.models["alpha"].want, 2);
    assert_eq!(plan.models["alpha"].nodes, vec!["a".to_string(), "b".to_string()]);
    assert_eq!(book.published.lock().unwrap().last().unwrap().plan, plan);
    // Settled: no step and nothing new to publish.
    let r = co.place_round(&p, &*book, &fence, Some(&plan)).await;
    assert!(r.done.is_empty() && r.published.is_none(), "{r:?}");
    // An unload that the round puts back in place leaves the plan as it was: nothing published.
    a.loaded.lock().unwrap().insert("alpha".into(), false);
    let r = co.place_round(&p, &*book, &fence, Some(&plan)).await;
    assert_eq!((r.done.len(), r.published), (1, None));
    // a loses its copy: alpha moves to c, published again, same term, next version.
    a.loaded.lock().unwrap().remove("alpha");
    let r = co.place_round(&p, &*book, &fence, Some(&plan)).await;
    assert_eq!(r.published, Some((Term(3), 3)));
    assert_eq!(book.published.lock().unwrap().last().unwrap().plan.models["alpha"].nodes, vec!["b".to_string(), "c".to_string()]);
}

#[tokio::test]
async fn a_stale_holder_takes_no_step_and_publishes_nothing() {
    let (a, b) = (Node::new("a", &[("alpha", false)], 0), Node::new("b", &[("alpha", false)], 0));
    let co = pool(&[&a, &b], None);
    let p = policy(&[("alpha", 2)], None);

    // A plan of term 5 has reached this holder of term 4 (it was paused while another took over).
    let book = Book::new(4);
    book.others(5);
    let fence = Fence::new();
    let r = co.place_round(&p, &*book, &fence, None).await;
    assert!(r.fenced && r.done.is_empty() && r.published.is_none(), "{r:?}");
    assert!(!a.is_loaded("alpha") && !b.is_loaded("alpha"));
    // Even if that plan is later not visible, the fence remembers term 5.
    *book.seen.lock().unwrap() = None;
    let r = co.place_round(&p, &*book, &fence, None).await;
    assert!(r.fenced && r.done.is_empty(), "{r:?}");

    // A holder whose lease ran out takes no step.
    let book = Book::new(6);
    book.holds.store(false, Ordering::SeqCst);
    let r = co.place_round(&p, &*book, &Fence::new(), None).await;
    assert!(r.done.is_empty() && r.published.is_none());

    // A lease lost mid-round: the step already planned is not taken.
    struct Flaky(Arc<Book>, AtomicU32);
    impl PlanBook for Flaky {
        fn lead(&self) -> Lead {
            Lead::Held(self.0.term)
        }
        fn still_holds(&self, _: Term) -> bool {
            // Valid at the round's start, gone by the first step.
            self.1.fetch_add(1, Ordering::SeqCst) == 0
        }
        fn current(&self) -> Option<PublishedPlan> {
            None
        }
        fn publish(&self, t: Term, plan: &PlacementPlan) -> anyhow::Result<PublishedPlan> {
            self.0.publish(t, plan)
        }
    }
    let flaky = Flaky(Book::new(7), AtomicU32::new(0));
    let r = co.place_round(&p, &flaky, &Fence::new(), None).await;
    assert!(r.fenced && r.done.is_empty() && r.published.is_none(), "{r:?}");
    assert!(!a.is_loaded("alpha") && !b.is_loaded("alpha"));
}

#[tokio::test]
async fn a_cold_request_loads_where_it_evicts_nothing() {
    // a would have to evict its ready `beta` (same swap group); b is free.
    let a = Node::grouped("a", &[("alpha", false), ("beta", true)], "gpu");
    let b = Node::grouped("b", &[("alpha", false)], "gpu");
    let co = pool(&[&a, &b], None);
    assert_eq!(ask(&co, "alpha").await.unwrap(), "b");
    assert!(b.is_loaded("alpha") && !a.is_loaded("alpha") && a.is_loaded("beta"));

    // Neither evicts: the emptier node.
    let a = Node::new("a", &[("alpha", false), ("beta", true), ("gamma", true)], 0);
    let b = Node::new("b", &[("alpha", false), ("beta", true)], 0);
    let co = pool(&[&a, &b], None);
    assert_eq!(ask(&co, "alpha").await.unwrap(), "b");
}

#[tokio::test]
async fn idle_models_are_swapped_out_but_not_the_main_model_or_a_managed_one() {
    let a = Node::new("a", &[("main", true), ("idle", true), ("hot", true), ("kept", true)], 0);
    let b = Node::new("b", &[("kept", true)], 0);
    let co = pool(&[&a, &b], Some("main"));
    let p = policy(&[("kept", 2)], Some(60));
    p.validate().unwrap();
    for m in ["main", "idle", "kept"] {
        co.demand().set_idle(m, Duration::from_secs(600));
    }
    co.demand().set_idle("hot", Duration::from_secs(5));

    // No lease (one inference node, no Weirdnet): it places and swaps, publishing nothing.
    let done = co.place_once(&p).await;
    assert_eq!(done.iter().map(|d| d.step.clone()).collect::<Vec<_>>(), vec![Step::Unload { model: "idle".into(), node: "a".into() }]);
    assert!(!a.is_loaded("idle") && a.is_loaded("main") && a.is_loaded("hot") && a.is_loaded("kept") && b.is_loaded("kept"));

    // Asked for again: it loads (cold) and its idle clock restarts.
    assert_eq!(ask(&co, "idle").await.unwrap(), "a");
    assert!(co.place_once(&p).await.is_empty());

    // Under a lease the swap is part of the published plan.
    co.demand().set_idle("idle", Duration::from_secs(600));
    let book = Book::new(1);
    let r = co.place_round(&p, &*book, &Fence::new(), None).await;
    assert_eq!(r.published, Some((Term(1), 1)));
    assert_eq!(book.published.lock().unwrap()[0].plan.swapped_out, vec!["idle".to_string()]);

    // A model with an answer in flight is never swapped out.
    let slow = Node::new("s", &[("idle", true)], 400);
    let co = pool(&[&slow], None);
    co.demand().set_idle("idle", Duration::from_secs(600));
    let c2 = co.clone();
    let asking = tokio::spawn(async move { ask(&c2, "idle").await });
    tokio::time::sleep(Duration::from_millis(100)).await;
    co.demand().set_idle("idle", Duration::from_secs(600));
    assert!(co.place_once(&p).await.is_empty());
    assert_eq!(asking.await.unwrap().unwrap(), "s");
}

#[test]
fn idle_unload_needs_a_minute() {
    assert!(policy(&[], Some(30)).validate().is_err());
    assert!(!policy(&[], Some(60)).is_empty());
}
