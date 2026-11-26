//! Placement and replication (M6): keep a model loaded on a set number of model nodes.
//!
//! `[inference.placement] replicas = { "qwen3-8b" = 2 }` names the models to manage and how many
//! nodes should have each loaded. Every `every_secs` the scheduler (below) compares that with
//! what the nodes report and takes **one step per model**: load the model on one more node, or
//! unload it from one idle extra node. Models not listed are left alone, as before.
//!
//! The rules ([`plan`]):
//! - A replica counts while its node takes new requests (not draining, not reserved, not being
//!   moved off) and the model is loading or ready there.
//! - Too few: load it on a node that has its own copy, is not draining or reserved, has not
//!   failed to load it (a crash loop stays an operator's call) and would not have to evict
//!   another ready model of the same swap group. Fewest loaded models first, then the
//!   quietest, then by name.
//! - Too many: unload it from a ready replica with nothing in flight (an answer is never cut
//!   off; a busy extra waits for the next round), the one with the most loaded models first.
//!   Never below the number asked for.
//!
//! **Who acts.** Only one inference node should place at a time (DESIGN: "only placement needs a
//! single writer"). An inference node linked cert-only (an `ymir.inference` grant) votes on and
//! stands for the `ymir.scheduler` lease (`weirdnet-sync` 0.8 majority-vote leases among the
//! devices with that role) and places only while it holds it. An inference node that holds the
//! seed (no grant, so no vote) places only when no cert-only inference node exists to hold the
//! lease; a pool with several seed-holding inference nodes should leave placement on one of
//! them (or link the others cert-only). `scheduler_eligible = false` never places.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use weirdnet_backends::ModelState;

use crate::coordinator::Coordinator;
use crate::model_node::ModelNode;

/// The lease the placing inference node holds.
pub const SCHEDULER_LEASE: &str = "ymir.scheduler";

/// `[inference.placement]`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PlacementPolicy {
    /// Model id to the number of model nodes that should have it loaded (at least 1).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub replicas: BTreeMap<String, u32>,
    /// How often the scheduler looks (default 15 s, at least 2 s).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub every_secs: Option<u64>,
}

impl PlacementPolicy {
    pub fn is_empty(&self) -> bool {
        self.replicas.is_empty()
    }

    pub fn every(&self) -> Duration {
        Duration::from_secs(self.every_secs.unwrap_or(15).max(2))
    }

    /// Refuses a count of 0 (unload it instead) or a nonsense one.
    pub fn validate(&self) -> anyhow::Result<()> {
        for (m, n) in &self.replicas {
            if *n == 0 || *n > 64 {
                anyhow::bail!("[inference.placement] replicas: {m} = {n}; give 1 to 64 (to keep it unloaded, unload it and leave it out)");
            }
        }
        Ok(())
    }
}

/// One model node as the planner sees it, for one model.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeView {
    pub name: String,
    /// The node has its own copy of the model (it is in its status).
    pub has_copy: bool,
    pub state: Option<ModelState>,
    pub in_flight: usize,
    /// Draining, reserved, or the model is being moved off it.
    pub closed: bool,
    /// Models loading or ready on the node.
    pub loaded: usize,
    /// Loading the model there would evict another ready model in its swap group.
    pub evicts: bool,
}

impl NodeView {
    fn serving(&self) -> bool {
        !self.closed && matches!(self.state, Some(ModelState::Ready | ModelState::Loading))
    }

    fn can_take(&self) -> bool {
        self.has_copy && !self.closed && !self.evicts && matches!(self.state, Some(ModelState::Unloaded))
    }
}

/// One step for one model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Step {
    Load { model: String, node: String },
    Unload { model: String, node: String },
}

/// The next step for `model` (wanted on `want` nodes), if any.
pub fn plan(model: &str, want: u32, nodes: &[NodeView]) -> Option<Step> {
    let have = nodes.iter().filter(|n| n.serving()).count();
    let want = want as usize;
    if have < want {
        let best = nodes.iter().filter(|n| n.can_take()).min_by(|a, b| (a.loaded, a.in_flight, &a.name).cmp(&(b.loaded, b.in_flight, &b.name)))?;
        return Some(Step::Load { model: model.into(), node: best.name.clone() });
    }
    if have > want {
        let idle = nodes
            .iter()
            .filter(|n| n.serving() && n.state == Some(ModelState::Ready) && n.in_flight == 0)
            .max_by(|a, b| (a.loaded, &b.name).cmp(&(b.loaded, &a.name)))?;
        return Some(Step::Unload { model: model.into(), node: idle.name.clone() });
    }
    None
}

/// What [`Coordinator::place_once`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Done {
    pub step: Step,
    pub error: Option<String>,
}

fn view(c: &Coordinator, node: &Arc<dyn ModelNode>, model: &str) -> NodeView {
    let st = node.status();
    let mine = st.iter().find(|s| s.id == model);
    let active = |s: &ModelState| matches!(s, ModelState::Ready | ModelState::Loading);
    let group = mine.and_then(|s| s.swap_group.clone());
    NodeView {
        name: node.name().to_string(),
        has_copy: mine.is_some(),
        state: mine.map(|s| s.state.clone()),
        in_flight: mine.map_or(0, |s| s.in_flight),
        closed: node.draining() || node.reserved().is_some() || c.is_leaving(node.name(), model),
        loaded: st.iter().filter(|s| active(&s.state)).count(),
        evicts: group.is_some_and(|g| st.iter().any(|s| s.id != model && s.swap_group.as_deref() == Some(&g) && active(&s.state))),
    }
}

impl Coordinator {
    /// One placement round: at most one step per managed model, done in turn.
    pub async fn place_once(&self, policy: &PlacementPolicy) -> Vec<Done> {
        let mut done = Vec::new();
        for (model, want) in &policy.replicas {
            let nodes = self.nodes();
            let views: Vec<NodeView> = nodes.iter().map(|n| view(self, n, model)).collect();
            let Some(step) = plan(model, *want, &views) else { continue };
            let (Step::Load { node, .. } | Step::Unload { node, .. }) = &step;
            let Some(n) = nodes.iter().find(|n| n.name() == node) else { continue };
            let r = match &step {
                Step::Load { model, .. } => n.load(model, false).await,
                Step::Unload { model, .. } => n.unload(model).await,
            };
            match &r {
                Ok(()) => tracing::info!(?step, "placement"),
                Err(e) => tracing::warn!(?step, "placement: {e}"),
            }
            done.push(Done { step, error: r.err().map(|e| e.to_string()) });
        }
        done
    }

    /// Runs [`Self::place_once`] every `policy.every()` while `leads()` says this node places.
    pub fn spawn_placement(self: &Arc<Self>, policy: PlacementPolicy, leads: Box<dyn Fn() -> bool + Send + Sync>) -> tokio::task::JoinHandle<()> {
        let me = Arc::downgrade(self);
        tokio::spawn(async move {
            let mut leading = false;
            loop {
                tokio::time::sleep(policy.every()).await;
                let Some(c) = me.upgrade() else { return };
                let now = leads();
                if now != leading {
                    tracing::info!("placement: {}", if now { "this node places models now" } else { "another inference node places models" });
                    leading = now;
                }
                if now {
                    c.place_once(&policy).await;
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(name: &str, state: Option<ModelState>, loaded: usize) -> NodeView {
        NodeView { name: name.into(), has_copy: state.is_some(), state, in_flight: 0, closed: false, loaded, evicts: false }
    }
    const U: Option<ModelState> = Some(ModelState::Unloaded);
    const R: Option<ModelState> = Some(ModelState::Ready);

    #[test]
    fn loads_on_the_emptiest_node_with_a_copy() {
        let nodes = [n("a", R, 1), n("b", U, 2), n("c", U, 0), n("d", None, 0)];
        assert_eq!(plan("m", 2, &nodes), Some(Step::Load { model: "m".into(), node: "c".into() }));
        // Enough already: nothing.
        assert_eq!(plan("m", 1, &nodes), None);
        // Nowhere to put it: nothing (d has no copy).
        let nodes = [n("a", R, 1), n("d", None, 0)];
        assert_eq!(plan("m", 2, &nodes), None);
    }

    #[test]
    fn closed_failed_and_evicting_nodes_are_not_used_and_do_not_count() {
        let mut drained = n("a", R, 1);
        drained.closed = true;
        let failed = n("b", Some(ModelState::Failed("crash".into())), 0);
        let mut evicts = n("c", U, 1);
        evicts.evicts = true;
        let ok = n("d", U, 3);
        // The drained replica doesn't count, so one is missing; only d may take it.
        assert_eq!(plan("m", 1, &[drained.clone(), failed.clone(), evicts.clone(), ok]), Some(Step::Load { model: "m".into(), node: "d".into() }));
        assert_eq!(plan("m", 1, &[drained, failed, evicts]), None);
    }

    #[test]
    fn unloads_an_idle_extra_never_a_busy_one() {
        let mut busy = n("a", R, 1);
        busy.in_flight = 2;
        let idle = n("b", R, 3);
        let idle2 = n("c", R, 1);
        assert_eq!(plan("m", 2, &[busy.clone(), idle.clone(), idle2.clone()]), Some(Step::Unload { model: "m".into(), node: "b".into() }));
        // Only busy extras: wait.
        let mut busy2 = n("b", R, 3);
        busy2.in_flight = 1;
        assert_eq!(plan("m", 1, &[busy, busy2]), None);
        // A loading one counts but is never unloaded.
        assert_eq!(plan("m", 1, &[n("a", Some(ModelState::Loading), 1), idle2]), Some(Step::Unload { model: "m".into(), node: "c".into() }));
    }

    #[test]
    fn ties_break_by_name_so_coordinators_agree() {
        assert_eq!(plan("m", 1, &[n("b", U, 0), n("a", U, 0)]), Some(Step::Load { model: "m".into(), node: "a".into() }));
        assert_eq!(plan("m", 1, &[n("a", R, 0), n("b", R, 0)]), Some(Step::Unload { model: "m".into(), node: "a".into() }));
    }

    #[test]
    fn policy_checks_counts() {
        let p: PlacementPolicy = toml::from_str("replicas = { alpha = 2 }\nevery_secs = 1").unwrap();
        assert_eq!(p.replicas["alpha"], 2);
        assert_eq!(p.every(), Duration::from_secs(2));
        p.validate().unwrap();
        let p: PlacementPolicy = toml::from_str("replicas = { alpha = 0 }").unwrap();
        assert!(p.validate().is_err());
    }
}
