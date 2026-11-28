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
//!
//! **The published plan and fencing.** The lease holder publishes what it placed as a signed
//! plan (`weirdnet-sync` 0.8 `SyncState::publish_plan`, sealed type [`crate::membership::PLAN_TYPE`],
//! name [`PLAN_NAME`]), ordered by `(term, version)`, never by clock. Before every round, and
//! before every step in it, the holder checks that its lease is still valid and that no plan of a
//! newer term has reached it (a [`weirdnet_sync::FenceGuard`] over the terms seen). A paused or
//! partitioned ex-holder that wakes up sees the new holder's plan on its next sync pass and stops,
//! and `publish_plan` itself refuses a stale term. Model nodes do not check terms yet: node/1
//! `Load`/`Unload` carry none (a `weirdnet-backends` gap; see docs/DESIGN.md, M6 status).
//!
//! **Cloud-wide swap.** A cold request loads its model where it is cheapest ([`cold_rank`]): a
//! node already loading it, then one that would evict nothing, the fewest models loaded, the
//! quietest, this device, by name. With `idle_unload_secs` set, the placing node also unloads a
//! model nobody has asked for in that long ([`idle_unloads`]), so the memory goes to what is in
//! demand; the replica floor, the main model, the stand-in and the cache's embedding model stay.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use weirdnet_backends::{ModelState, ModelStatus};
use weirdnet_sync::{FenceGuard, LeaseName, Term};

use crate::coordinator::Coordinator;
use crate::model_node::ModelNode;

/// The lease the placing inference node holds.
pub const SCHEDULER_LEASE: &str = "ymir.scheduler";

/// The name of the published placement plan (under [`crate::membership::PLAN_TYPE`]).
pub const PLAN_NAME: &str = "placement";

/// `[inference.placement]`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PlacementPolicy {
    /// Model id to the number of model nodes that should have it loaded (at least 1).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub replicas: BTreeMap<String, u32>,
    /// How often the scheduler looks (default 15 s, at least 2 s).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub every_secs: Option<u64>,
    /// Cloud-wide swap: unload a model nobody has asked for in this many seconds (off when
    /// unset; at least 60). Never below a model's `replicas`, never the main model, the
    /// stand-in or the semantic cache's embedding model, never with answers in flight.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idle_unload_secs: Option<u64>,
}

impl PlacementPolicy {
    pub fn is_empty(&self) -> bool {
        self.replicas.is_empty() && self.idle_unload_secs.is_none()
    }

    pub fn idle_unload(&self) -> Option<Duration> {
        self.idle_unload_secs.map(Duration::from_secs)
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
        if let Some(s) = self.idle_unload_secs
            && s < 60
        {
            anyhow::bail!("[inference.placement] idle_unload_secs = {s}; give at least 60 (a model reloading every few seconds only thrashes)");
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

fn active(s: &ModelState) -> bool {
    matches!(s, ModelState::Ready | ModelState::Loading)
}

fn view(c: &Coordinator, node: &Arc<dyn ModelNode>, model: &str) -> NodeView {
    let st = node.status();
    let mine = st.iter().find(|s| s.id == model);
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

/// How good a node is for a cold load of `model` (lower is better; see the module docs): the
/// model's row there, the node's whole status, whether it is on this device, and its name.
pub fn cold_rank(mine: &ModelStatus, all: &[ModelStatus], local: bool, name: &str) -> (u8, bool, usize, usize, bool, String) {
    let state = match mine.state {
        ModelState::Ready | ModelState::Loading => 0,
        ModelState::Unloaded => 1,
        ModelState::Failed(_) => 3,
        _ => 2,
    };
    let evicts =
        mine.swap_group.as_ref().is_some_and(|g| all.iter().any(|s| s.id != mine.id && s.swap_group.as_ref() == Some(g) && active(&s.state)));
    let loaded = all.iter().filter(|s| s.id != mine.id && s.base.is_none() && active(&s.state)).count();
    let busy = all.iter().map(|s| s.in_flight).sum();
    (state, evicts, loaded, busy, !local, name.to_string())
}

/// When each model was last asked for (or first seen loaded) on this inference node, for
/// cloud-wide swap's idle unloads.
#[derive(Default)]
pub struct Demand {
    last: Mutex<HashMap<String, Instant>>,
}

impl Demand {
    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, Instant>> {
        self.last.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// `model` was asked for now.
    pub fn touch(&self, model: &str) {
        self.lock().insert(model.to_string(), Instant::now());
    }

    /// How long since `model` was last asked for; a model never seen starts its clock now.
    pub fn idle_for(&self, model: &str) -> Duration {
        self.lock().entry(model.to_string()).or_insert_with(Instant::now).elapsed()
    }

    /// For tests: pretend `model` was last asked for `ago`.
    pub fn set_idle(&self, model: &str, ago: Duration) {
        let at = Instant::now().checked_sub(ago).unwrap_or_else(Instant::now);
        self.lock().insert(model.to_string(), at);
    }
}

/// Cloud-wide swap's unloads this round: for each loaded model that is not `keep`, not managed
/// by `replicas`, and idle for at least `idle`, one idle replica (never one with answers in
/// flight). `views` gives the planner's view of every node per model.
pub fn idle_unloads(
    replicas: &BTreeMap<String, u32>,
    keep: &[String],
    idle: Duration,
    views: &BTreeMap<String, Vec<NodeView>>,
    idle_for: impl Fn(&str) -> Duration,
) -> Vec<Step> {
    views
        .iter()
        .filter(|(m, _)| !replicas.contains_key(*m) && !keep.contains(m))
        .filter(|(_, nodes)| nodes.iter().any(|n| n.serving()))
        .filter(|(m, _)| idle_for(m) >= idle)
        .filter_map(|(m, nodes)| plan(m, 0, nodes))
        .collect()
}

/// The plan the lease holder publishes: for each managed model, how many replicas it wants and
/// the nodes serving it after the round (sorted, so an unchanged round is an unchanged body).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlacementPlan {
    pub models: BTreeMap<String, PlannedModel>,
    /// Models cloud-wide swap unloaded for idleness in the last round that changed the plan.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub swapped_out: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannedModel {
    pub want: u32,
    pub nodes: Vec<String>,
}

/// A plan as read back from sync: whose it is and where it sits in `(term, version)` order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishedPlan {
    pub term: Term,
    pub version: u64,
    pub signer: String,
    pub plan: PlacementPlan,
}

/// Whether this inference node places now, and under what.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lead {
    /// Another inference node places (or this one may not).
    No,
    /// It places with no lease (no Weirdnet, or a seed holder while no cert-only inference node
    /// exists): nothing to fence against and no plan to publish.
    Alone,
    /// It holds the `ymir.scheduler` lease at this term.
    Held(Term),
}

/// Who places, and where the plan is published (the lease and `ymir.plan` in production,
/// [`Alone`] without Weirdnet, fakes in tests).
pub trait PlanBook: Send + Sync {
    fn lead(&self) -> Lead;
    /// Whether this node still holds the lease at `term` (checked before every step).
    fn still_holds(&self, term: Term) -> bool;
    /// The newest plan this node has seen.
    fn current(&self) -> Option<PublishedPlan>;
    /// Publishes `plan` under the lease held at `term`.
    fn publish(&self, term: Term, plan: &PlacementPlan) -> anyhow::Result<PublishedPlan>;
}

/// Placing with no lease and nothing published.
pub struct Alone(pub bool);

impl PlanBook for Alone {
    fn lead(&self) -> Lead {
        if self.0 { Lead::Alone } else { Lead::No }
    }
    fn still_holds(&self, _: Term) -> bool {
        false
    }
    fn current(&self) -> Option<PublishedPlan> {
        None
    }
    fn publish(&self, _: Term, _: &PlacementPlan) -> anyhow::Result<PublishedPlan> {
        anyhow::bail!("no lease, so no plan to publish")
    }
}

/// What a placement round did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Round {
    pub done: Vec<Done>,
    /// The round stopped (or never started) because a newer term has published or the lease
    /// ran out.
    pub fenced: bool,
    pub published: Option<(Term, u64)>,
}

/// The holder's own fence: the terms of plans seen, so a holder whose term is older stops.
pub struct Fence {
    guard: Mutex<FenceGuard>,
    name: LeaseName,
}

impl Default for Fence {
    fn default() -> Self {
        Self::new()
    }
}

impl Fence {
    pub fn new() -> Self {
        Self { guard: Mutex::new(FenceGuard::new()), name: LeaseName::new(SCHEDULER_LEASE).expect("a valid lease name") }
    }

    /// Records `seen` (a published plan's term) and checks `ours` against everything seen.
    pub fn admit(&self, ours: Term, seen: Option<Term>) -> Result<(), weirdnet_sync::Stale> {
        let mut g = self.guard.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(t) = seen {
            let _ = g.check(&self.name, t);
        }
        g.check(&self.name, ours)
    }
}

impl Coordinator {
    /// The models cloud-wide swap never unloads: the main model, the stand-in and the
    /// semantic cache's embedding model.
    fn keep_warm(&self) -> Vec<String> {
        let o = self.options();
        self.main_model()
            .map(str::to_string)
            .into_iter()
            .chain(o.standin.clone())
            .chain(o.semantic.as_ref().map(|s| s.embedding_model.clone()))
            .collect()
    }

    /// One placement round with no lease (see [`Self::place_round`]).
    pub async fn place_once(&self, policy: &PlacementPolicy) -> Vec<Done> {
        self.place_round(policy, &Alone(true), &Fence::new(), None).await.done
    }

    /// One placement round: at most one step per managed model (and, with `idle_unload_secs`,
    /// one idle unload per unmanaged model), done in turn, each only while `book` says this
    /// node still leads. Under a lease, the round is fenced on the newest plan seen and the
    /// result is published when it changed from `last`.
    pub async fn place_round(&self, policy: &PlacementPolicy, book: &dyn PlanBook, fence: &Fence, last: Option<&PlacementPlan>) -> Round {
        let mut round = Round::default();
        let term = match book.lead() {
            Lead::No => return round,
            Lead::Alone => None,
            Lead::Held(t) => Some(t),
        };
        let leads = |round: &mut Round| match term {
            None => true,
            Some(t) => {
                let ok = book.still_holds(t) && fence.admit(t, book.current().map(|p| p.term)).is_ok();
                if !ok {
                    round.fenced = true;
                }
                ok
            }
        };
        if !leads(&mut round) {
            tracing::info!("placement: a newer scheduler term has published, or the lease ran out; not placing");
            return round;
        }
        let mut swapped = Vec::new();
        for (model, want) in &policy.replicas {
            let nodes = self.nodes();
            let views: Vec<NodeView> = nodes.iter().map(|n| view(self, n, model)).collect();
            let Some(step) = plan(model, *want, &views) else { continue };
            if !leads(&mut round) {
                return round;
            }
            round.done.push(self.take(&nodes, step).await);
        }
        if let Some(idle) = policy.idle_unload() {
            let nodes = self.nodes();
            let mut views: BTreeMap<String, Vec<NodeView>> = BTreeMap::new();
            for n in &nodes {
                for s in n.status() {
                    if s.base.is_some() || s.wrapped {
                        continue;
                    }
                    if s.in_flight > 0 {
                        self.demand().touch(&s.id);
                    }
                    if !views.contains_key(&s.id) {
                        let v = nodes.iter().map(|n| view(self, n, &s.id)).collect();
                        views.insert(s.id.clone(), v);
                    }
                }
            }
            for step in idle_unloads(&policy.replicas, &self.keep_warm(), idle, &views, |m| self.demand().idle_for(m)) {
                if !leads(&mut round) {
                    return round;
                }
                let d = self.take(&nodes, step).await;
                if d.error.is_none()
                    && let Step::Unload { model, .. } = &d.step
                {
                    swapped.push(model.clone());
                }
                round.done.push(d);
            }
        }
        if let Some(t) = term {
            let mut now = self.current_plan(policy);
            now.swapped_out = swapped;
            let unchanged = last.is_some_and(|l| l.models == now.models) && now.swapped_out.is_empty();
            let ours = book.current().is_some_and(|p| p.term == t);
            if !(unchanged && ours) && leads(&mut round) {
                match book.publish(t, &now) {
                    Ok(p) => round.published = Some((p.term, p.version)),
                    Err(e) => tracing::warn!("placement: publishing the plan: {e:#}"),
                }
            }
        }
        round
    }

    /// Where the managed models are now, as the plan to publish.
    pub fn current_plan(&self, policy: &PlacementPolicy) -> PlacementPlan {
        let nodes = self.nodes();
        let models = policy
            .replicas
            .iter()
            .map(|(m, want)| {
                let mut on: Vec<String> = nodes.iter().map(|n| view(self, n, m)).filter(|v| v.serving()).map(|v| v.name).collect();
                on.sort();
                (m.clone(), PlannedModel { want: *want, nodes: on })
            })
            .collect();
        PlacementPlan { models, swapped_out: Vec::new() }
    }

    async fn take(&self, nodes: &[Arc<dyn ModelNode>], step: Step) -> Done {
        let (Step::Load { node, .. } | Step::Unload { node, .. }) = &step;
        let Some(n) = nodes.iter().find(|n| n.name() == node) else { return Done { step, error: Some("the node went away".into()) } };
        let r = match &step {
            Step::Load { model, .. } => n.load(model, false).await,
            Step::Unload { model, .. } => n.unload(model).await,
        };
        match &r {
            Ok(()) => tracing::info!(?step, "placement"),
            Err(e) => tracing::warn!(?step, "placement: {e}"),
        }
        Done { step, error: r.err().map(|e| e.to_string()) }
    }

    /// Runs [`Self::place_round`] every `policy.every()`, publishing through `book`.
    pub fn spawn_placement(self: &Arc<Self>, policy: PlacementPolicy, book: Box<dyn PlanBook>) -> tokio::task::JoinHandle<()> {
        let me = Arc::downgrade(self);
        tokio::spawn(async move {
            let fence = Fence::new();
            let mut leading = false;
            let mut last: Option<PlacementPlan> = None;
            loop {
                tokio::time::sleep(policy.every()).await;
                let Some(c) = me.upgrade() else { return };
                let now = book.lead() != Lead::No;
                if now != leading {
                    tracing::info!("placement: {}", if now { "this node places models now" } else { "another inference node places models" });
                    leading = now;
                }
                if now {
                    let r = c.place_round(&policy, book.as_ref(), &fence, last.as_ref()).await;
                    if r.published.is_some() {
                        last = Some(c.current_plan(&policy));
                    }
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

    fn row(id: &str, state: ModelState, group: Option<&str>, in_flight: usize) -> ModelStatus {
        serde_json::from_value(serde_json::json!({
            "id": id, "engine": "llamacpp", "state": state, "in_flight": in_flight, "swap_group": group,
            "embedding": false, "pid": null, "last_load_secs": null, "size_bytes": null, "wrapped": false, "crashes": 0
        }))
        .unwrap()
    }

    #[test]
    fn cold_loads_prefer_loading_then_no_eviction_then_empty_quiet_local_nodes() {
        let m = |s| row("m", s, Some("g"), 0);
        let loading = m(ModelState::Loading);
        let free = m(ModelState::Unloaded);
        let evicting = [free.clone(), row("other", ModelState::Ready, Some("g"), 0)];
        let busy = [free.clone(), row("x", ModelState::Ready, None, 3)];
        let r = |mine: &ModelStatus, all: &[ModelStatus], local, name| cold_rank(mine, all, local, name);
        assert!(r(&loading, std::slice::from_ref(&loading), false, "z") < r(&free, std::slice::from_ref(&free), true, "a"));
        assert!(r(&free, std::slice::from_ref(&free), false, "z") < r(&free, &evicting, true, "a"));
        assert!(r(&free, std::slice::from_ref(&free), false, "z") < r(&free, &busy, true, "a"));
        let quiet = [free.clone(), row("x", ModelState::Ready, None, 0)];
        assert!(r(&free, &quiet, false, "z") < r(&free, &busy, false, "a"));
        assert!(r(&free, &quiet, true, "z") < r(&free, &quiet, false, "a"));
        let failed = m(ModelState::Failed("crash".into()));
        assert!(r(&free, &evicting, false, "z") < r(&failed, std::slice::from_ref(&failed), true, "a"));
    }

    #[test]
    fn idle_unloads_skip_kept_managed_busy_and_recent_models() {
        let mut views = BTreeMap::new();
        let busy = NodeView { in_flight: 1, ..n("a", R, 2) };
        views.insert("idle".to_string(), vec![n("a", R, 3), n("b", R, 1)]);
        views.insert("main".to_string(), vec![n("a", R, 3)]);
        views.insert("managed".to_string(), vec![n("a", R, 3)]);
        views.insert("busy".to_string(), vec![busy]);
        views.insert("recent".to_string(), vec![n("a", R, 3)]);
        views.insert("cold".to_string(), vec![n("a", U, 3)]);
        let replicas = BTreeMap::from([("managed".to_string(), 1)]);
        let idle_for = |m: &str| if m == "recent" { Duration::from_secs(1) } else { Duration::from_secs(3600) };
        let steps = idle_unloads(&replicas, &["main".to_string()], Duration::from_secs(60), &views, idle_for);
        // One replica per model per round, the most loaded first.
        assert_eq!(steps, vec![Step::Unload { model: "idle".into(), node: "a".into() }]);
    }

    #[test]
    fn the_fence_refuses_a_term_older_than_any_seen() {
        let f = Fence::new();
        f.admit(Term(2), None).unwrap();
        f.admit(Term(2), Some(Term(2))).unwrap();
        assert_eq!(f.admit(Term(2), Some(Term(3))).unwrap_err().seen, Term(3));
        // Remembered even when the newer plan is no longer in view.
        assert!(f.admit(Term(2), None).is_err());
        f.admit(Term(4), None).unwrap();
    }

    #[test]
    fn plans_round_trip_and_sort() {
        let mut p = PlacementPlan::default();
        p.models.insert("m".into(), PlannedModel { want: 2, nodes: vec!["a".into(), "b".into()] });
        let back: PlacementPlan = serde_json::from_slice(&serde_json::to_vec(&p).unwrap()).unwrap();
        assert_eq!(back, p);
        assert!(!serde_json::to_string(&p).unwrap().contains("swapped_out"));
    }

    #[test]
    fn policy_checks_counts() {
        let p: PlacementPolicy = toml::from_str("replicas = { alpha = 2 }\nevery_secs = 1").unwrap();
        assert_eq!(p.replicas["alpha"], 2);
        assert_eq!(p.every(), Duration::from_secs(2));
        p.validate().unwrap();
        let p: PlacementPolicy = toml::from_str("replicas = { alpha = 0 }").unwrap();
        assert!(p.validate().is_err());
        let p: PlacementPolicy = toml::from_str("idle_unload_secs = 600").unwrap();
        assert!(!p.is_empty() && p.validate().is_ok() && p.idle_unload() == Some(Duration::from_secs(600)));
        let p: PlacementPolicy = toml::from_str("idle_unload_secs = 10").unwrap();
        assert!(p.validate().is_err());
    }
}
