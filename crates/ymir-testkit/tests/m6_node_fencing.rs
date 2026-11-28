//! M6, model-node fencing between a real model daemon and the placement lease holder
//! (`weirdnet-backends` 0.4.11 node/1 `LoadFenced`/`UnloadFenced`), over Weirdnet on 127.0.0.1
//! with the mock engine:
//!
//! - The model node's record says `fencing`, and it takes the `ymir.scheduler` term on load
//!   and unload, refusing an older term with the model left as it is.
//! - The terms it has seen live in `state/model-fence.cbor`, so a restart keeps refusing them.
//! - A placement round of an older term is refused at the node, stops, and the holder stays
//!   stopped at that term; a newer holder places.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, anyhow};
use weirdnet_agent::pool::NetDialer;
use weirdnet_backends::fence::{Fence as NodeFence, FenceGuard, FenceRefusal};
use weirdnet_backends::{ModelState, ModelStatus};
use weirdnet_net::Multiaddr;
use weirdnet_semcache::NoCache;
use weirdnet_sync::Term;
use ymir_core::NodeConfig;
use ymir_core::control::{self, ControlRequest};
use ymir_core::coordinator::Coordinator;
use ymir_core::membership::{Membership, SyncNode};
use ymir_core::model_node::ModelNode;
use ymir_core::placement::{Fence, Lead, PlacementPlan, PlacementPolicy, PlanBook, PublishedPlan, SCHEDULER_LEASE};
use ymir_core::remote::{RemoteModelNode, RemoteOptions};

#[path = "common/pool.rs"]
mod pool;
use pool::*;

/// This inference node holds the lease at `term` and publishes nowhere.
struct Holder(Term);

impl PlanBook for Holder {
    fn lead(&self) -> Lead {
        Lead::Held(self.0)
    }
    fn still_holds(&self, term: Term) -> bool {
        term == self.0
    }
    fn current(&self) -> Option<PublishedPlan> {
        None
    }
    fn publish(&self, term: Term, plan: &PlacementPlan) -> anyhow::Result<PublishedPlan> {
        Ok(PublishedPlan { term, version: 1, signer: "inf".into(), plan: plan.clone() })
    }
}

fn scheduler(term: u64) -> NodeFence {
    NodeFence::new(SCHEDULER_LEASE, term)
}

fn state(s: &[ModelStatus], id: &str) -> Option<ModelState> {
    s.iter().find(|m| m.id == id).map(|m| m.state.clone())
}

/// Connects to the running model daemon at `model_h` as the device in `inf_h`.
async fn connect(model_h: &Home, inf_h: &Home) -> Result<(Arc<RemoteModelNode>, Arc<SyncNode>)> {
    let s = control::request(&model_h.paths.socket(), &ControlRequest::Status).await?.status.unwrap().network.expect("on Weirdnet");
    let a: Multiaddr =
        s.listening.iter().find(|a| a.starts_with("/ip4/127.0.0.1/tcp/")).expect("a loopback listener").split("/p2p/").next().unwrap().parse()?;
    let m = Membership::open(&inf_h.paths.pool())?.unwrap();
    let node =
        Arc::new(weirdnet_sync::start_node(m.node_key().clone(), NodeConfig::load(&inf_h.paths.config())?.network.node_config("test")?, m.state())?);
    let dialer = NetDialer::new(node.clone());
    dialer.hint(s.peer.parse()?, vec![a]);
    let opts = RemoteOptions { find_wait: Duration::from_secs(2), accept_timeout: Duration::from_secs(5) };
    let rn = RemoteModelNode::connect(Arc::new(dialer), &s.peer, &m, &opts).await.map_err(|e| anyhow!("{e}"))?;
    Ok((Arc::new(rn), node))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_model_node_refuses_an_older_scheduler_term_even_after_a_restart() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();
    let model_h = home(root, "model", "model", &[], &["alpha"])?;
    let first = Membership::create(&model_h.paths.pool(), "model")?;
    let inf_h = home(root, "inf", "inference", &[], &[])?;
    prelink(&first, &inf_h, "inf");
    drop(first);

    let model = start(&model_h).await?;
    let (rn, _net) = connect(&model_h, &inf_h).await?;
    assert!(rn.fencing(), "a 0.4.11 model node says it fences");

    // Term 2 loads; term 1 can't unload what term 2 placed.
    rn.load_fenced("alpha", false, &scheduler(2)).await.map_err(|e| anyhow!("{e}"))?;
    eventually(30, "alpha ready", async || Ok(state(&rn.status(), "alpha") == Some(ModelState::Ready))).await?;
    let e = rn.unload_fenced("alpha", &scheduler(1)).await.expect_err("an older term is refused");
    assert_eq!(FenceRefusal::from_error(&e), Some(FenceRefusal::Stale { lease: SCHEDULER_LEASE.into(), term: 1, seen: 2 }), "{e}");
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(state(&rn.status(), "alpha"), Some(ModelState::Ready), "a refused unload leaves the model loaded");
    // An operator's unfenced commands still work (no fence required).
    assert!(!rn.record().unwrap().fence_required);

    // Kept on disk, so a restart keeps refusing term 1.
    let saved = FenceGuard::load(model_h.paths.model_fence())?;
    assert_eq!(saved.seen(SCHEDULER_LEASE), Some(2));
    drop(rn);
    model.shutdown().await;
    let model = start(&model_h).await?;
    let (rn, _net2) = connect(&model_h, &inf_h).await?;
    let e = rn.load_fenced("alpha", false, &scheduler(1)).await.expect_err("still refused after a restart");
    assert!(matches!(FenceRefusal::from_error(&e), Some(FenceRefusal::Stale { seen: 2, .. })), "{e}");

    // Term 3 unloads it; then a placement round from term 2 (an ex-holder that hasn't seen term 3's
    // plan) is refused at the node, stops, and stays stopped.
    rn.unload_fenced("alpha", &scheduler(3)).await.map_err(|e| anyhow!("{e}"))?;
    eventually(30, "alpha unloaded", async || Ok(state(&rn.status(), "alpha") == Some(ModelState::Unloaded))).await?;
    let co = Coordinator::new(vec![rn.clone() as Arc<dyn ModelNode>], None, Arc::new(NoCache));
    let policy: PlacementPolicy = toml::from_str("replicas = { alpha = 1 }")?;
    let fence = Fence::new();
    let r = co.place_round(&policy, &Holder(Term(2)), &fence, None).await;
    assert!(r.fenced && r.done.len() == 1 && r.done[0].fenced && r.published.is_none(), "{r:?}");
    assert_eq!(fence.seen(), Some(Term(3)));
    let r = co.place_round(&policy, &Holder(Term(2)), &fence, None).await;
    assert!(r.fenced && r.done.is_empty(), "{r:?}");
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(state(&rn.status(), "alpha"), Some(ModelState::Unloaded));

    // The newer holder places.
    let r = co.place_round(&policy, &Holder(Term(4)), &Fence::new(), None).await;
    assert!(!r.fenced && r.done.len() == 1 && r.done[0].error.is_none(), "{r:?}");
    eventually(30, "alpha ready again", async || Ok(state(&rn.status(), "alpha") == Some(ModelState::Ready))).await?;
    assert_eq!(FenceGuard::load(model_h.paths.model_fence())?.seen(SCHEDULER_LEASE), Some(4));
    drop(co);
    drop(rn);
    model.shutdown().await;
    Ok(())
}
