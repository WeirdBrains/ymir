//! Contexts at load time, beside the models already running on this node.
//!
//! [`crate::kvfit::plan_launches`] gives each model a context at start, sharing the node among
//! the models that could be resident together. When a model is asked for, the [`Planner`] looks
//! at what actually runs now:
//!
//! - Memory free for it is the smaller of the estimate (the budget less what the running models
//!   take at their contexts) and the measurement (`MemAvailable` less the OS reserve on a
//!   unified-memory node, `nvidia-smi`'s free memory on a discrete GPU).
//! - If its full trained context does not fit beside them, running models with no request under
//!   way and none for [`GRACE`] are unloaded first, the longest idle first, until it does (a
//!   model with a request in flight, or loading, is never unloaded; the grace keeps two models
//!   from evicting each other on every turn).
//! - It then gets the largest context that fits, up to its trained length, with
//!   [`crate::kvfit::HEADROOM_PCT`] of the free memory kept free. An explicit `context` is kept.
//!
//! The context is set on the supervisor's launch, so the capability record (and every catalog
//! built from it) reports the context the engine runs with. A keeper task also re-sizes models
//! that are not loaded whenever what runs changes, so a load that does not come through
//! [`Planner::prepare`] (a `Load` from another device's inference node, a crash restart) starts
//! with a context that fits beside what runs then. A model counts as loading from its
//! `Starting` event (the supervisor reports it `Unloaded` until its engine is spawned), so it is
//! neither re-sized under a load nor left out of what the others are sized beside.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use weirdnet_backends::{EngineKind, ModelLaunch, ModelState, ModelStatus, Supervisor, SupervisorEvent};
use weirdnet_model::hw::Budget;

use crate::kvfit::{self, Resolved, Run};

/// How long a model must have been without requests before it may be unloaded for another.
pub const GRACE: Duration = Duration::from_secs(30);
/// How often the keeper looks at what is idle and at measured memory.
const TICK: Duration = Duration::from_secs(2);
/// How often the keeper re-sizes unloaded models on measured memory alone (no event).
const RESIZE_EVERY: Duration = Duration::from_secs(30);

fn gib(b: u64) -> String {
    format!("{:.1} GiB", b as f64 / (1u64 << 30) as f64)
}

// ---------------------------------------------------------------------------------------------
// Measuring

/// The memory free for another model right now, measured. `None`: this node can't say, and the
/// estimate alone is used.
pub trait MemProbe: Send + Sync {
    fn free_bytes(&self) -> Option<u64>;
}

/// `/proc/meminfo` on a unified-memory (or CPU-only) Linux node, `nvidia-smi` on a discrete
/// NVIDIA GPU.
pub struct SystemProbe {
    unified: bool,
    nvidia: bool,
    reserve: u64,
}

impl SystemProbe {
    pub fn new(hw: &weirdnet_model::hw::Hardware, budget: &Budget) -> Self {
        let unified = hw.gpus.is_empty() || hw.gpus.iter().any(|g| g.unified);
        let nvidia = hw.gpus.iter().any(|g| matches!(g.accel, weirdnet_model::hw::Accel::Nvidia));
        Self { unified, nvidia, reserve: budget.reserve_bytes }
    }
}

/// `MemAvailable` in a `/proc/meminfo` text.
pub fn mem_available(meminfo: &str) -> Option<u64> {
    let line = meminfo.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib * 1024)
}

/// The free memory summed over `nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits`
/// lines (MiB).
pub fn nvidia_free(csv: &str) -> Option<u64> {
    let v: Vec<u64> = csv.lines().filter_map(|l| l.trim().parse::<u64>().ok()).collect();
    (!v.is_empty()).then(|| v.iter().sum::<u64>() << 20)
}

impl MemProbe for SystemProbe {
    fn free_bytes(&self) -> Option<u64> {
        if self.unified {
            let text = std::fs::read_to_string("/proc/meminfo").ok()?;
            return mem_available(&text).map(|a| a.saturating_sub(self.reserve));
        }
        if self.nvidia {
            let out = std::process::Command::new("nvidia-smi").args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"]).output().ok()?;
            return out.status.success().then(|| nvidia_free(&String::from_utf8_lossy(&out.stdout))).flatten();
        }
        None
    }
}

// ---------------------------------------------------------------------------------------------
// Deciding (no I/O, so it can be tested)

/// A model running (or loading) on the node.
#[derive(Debug, Clone)]
pub struct Resident {
    pub id: String,
    pub swap_group: Option<String>,
    /// What it takes at its context (the estimate).
    pub bytes: u64,
    pub in_flight: usize,
    pub loading: bool,
    /// How long since it last had a request under way (or was loaded).
    pub idle: Duration,
}

impl Resident {
    fn evictable(&self, grace: Duration) -> bool {
        self.in_flight == 0 && !self.loading && self.idle >= grace
    }
}

/// The model being sized.
pub struct Ask<'a> {
    pub launch: &'a ModelLaunch,
    pub resolved: &'a Resolved,
    /// Its context was set by the operator (`context =` or a flag in `args`).
    pub explicit: bool,
}

/// What [`decide`] chose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decision {
    /// Models to unload first.
    pub evict: Vec<String>,
    /// The context to launch with.
    pub context: u32,
    /// The bytes free for the model, after the evictions.
    pub free: u64,
    /// Whether the full trained length (or the explicit context) did not fit.
    pub short: bool,
    pub note: String,
}

/// Whether `r` stays in memory when `ask` loads (a swap-group mate is unloaded by the supervisor).
fn stays(ask: &ModelLaunch, r: &Resident) -> bool {
    r.id != ask.id && !(ask.swap_group.is_some() && r.swap_group == ask.swap_group)
}

/// The memory free for `ask` beside `residents` that stay: the budget less their estimates, and
/// at most the measured free memory plus what the models that go (`gone`) give back.
pub fn free_beside(ask: &ModelLaunch, residents: &[Resident], gone: &HashSet<String>, budget: &Budget, measured: Option<u64>) -> u64 {
    let (mut kept, mut freed) = (0u64, 0u64);
    for r in residents {
        if stays(ask, r) && !gone.contains(&r.id) {
            kept = kept.saturating_add(r.bytes);
        } else if r.id != ask.id {
            freed = freed.saturating_add(r.bytes);
        }
    }
    let estimate = budget.usable_bytes.saturating_sub(kept);
    match measured {
        Some(m) => estimate.min(m.saturating_add(freed)),
        None => estimate,
    }
}

/// The context `ask` gets in `free` bytes: (context, cut).
fn fit(ask: &Ask, budget: &Budget, free: u64) -> (u32, bool) {
    let room = kvfit::room_for_context(ask.resolved, ask.launch, budget, free);
    kvfit::fit_context(&ask.resolved.shape, room, Run::of(ask.launch))
}

/// Chooses what to unload for `ask` and the context it gets. `measured` is the free memory now
/// ([`MemProbe`]); [`Planner::prepare`] measures again after unloading and calls this once more
/// with `evict_ok` off.
pub fn decide(ask: &Ask, residents: &[Resident], budget: &Budget, measured: Option<u64>, grace: Duration, evict_ok: bool) -> Decision {
    let run = Run::of(ask.launch);
    let want = |free: u64| -> bool {
        match (ask.explicit, ask.launch.context) {
            (true, Some(c)) => {
                kvfit::context_budget(free.saturating_sub(kvfit::footprint(ask.resolved.weights, 0)))
                    >= ask.resolved.shape.context_bytes(u64::from(c), run)
            }
            _ => !fit(ask, budget, free).1,
        }
    };
    let mut gone = HashSet::new();
    let mut free = free_beside(ask.launch, residents, &gone, budget, measured);
    if evict_ok && !want(free) {
        let mut idle: Vec<&Resident> = residents.iter().filter(|r| stays(ask.launch, r) && r.evictable(grace)).collect();
        idle.sort_by_key(|r| std::cmp::Reverse(r.idle));
        for r in idle {
            gone.insert(r.id.clone());
            free = free_beside(ask.launch, residents, &gone, budget, measured);
            if want(free) {
                break;
            }
        }
    }
    let mut evict: Vec<String> = residents.iter().filter(|r| gone.contains(&r.id)).map(|r| r.id.clone()).collect();
    evict.sort();
    let beside: Vec<&str> = residents.iter().filter(|r| stays(ask.launch, r) && !gone.contains(&r.id)).map(|r| r.id.as_str()).collect();
    let beside = if beside.is_empty() { String::from("with nothing else running") } else { format!("beside {}", beside.join(", ")) };
    let evicted = if evict.is_empty() { String::new() } else { format!("; unloaded idle {} for it", evict.join(", ")) };
    let measured_note = measured.map_or(String::new(), |m| format!(", {} measured free", gib(m)));
    if let (true, Some(c)) = (ask.explicit, ask.launch.context) {
        let short = !want(free);
        let note = format!(
            "context {c} (set in node.toml), {} free for it {beside}{measured_note}{evicted}{}",
            gib(free),
            if short { ": it may not fit" } else { "" }
        );
        return Decision { evict, context: c, free, short, note };
    }
    let (context, short) = fit(ask, budget, free);
    let trained = ask.resolved.shape.trained.unwrap_or(u64::from(kvfit::FALLBACK_CONTEXT));
    let note = format!(
        "context {context}{} ({} for KV and buffers): {} free for it {beside}{measured_note}, {}% kept free{evicted}",
        if short { format!(", cut from {trained}") } else { format!(", its full {trained}") },
        gib(ask.resolved.shape.context_bytes(u64::from(context), run)),
        gib(free),
        kvfit::HEADROOM_PCT,
    );
    Decision { evict, context, free, short, note }
}

// ---------------------------------------------------------------------------------------------
// The planner

struct Known {
    resolved: Option<Resolved>,
    explicit: bool,
}

/// Sizes contexts on this node at load time. Cheap to share (`Arc`).
pub struct Planner {
    sup: Supervisor,
    budget: Budget,
    probe: Arc<dyn MemProbe>,
    models_dir: PathBuf,
    grace: Duration,
    known: Mutex<HashMap<String, Known>>,
    /// When each model last had a request under way, or was loaded.
    busy: Mutex<HashMap<String, Instant>>,
    /// Models re-added with a new context, and how many of their `Removed`/`Added` events are
    /// still to come (those are not news).
    resizing: Mutex<HashMap<String, u32>>,
    /// Models whose load has begun (`Starting`) and not yet ended (`Ready`, `LoadFailed`,
    /// `Unloaded`, ...). The supervisor still reports such a model `Unloaded` while it waits for
    /// its turn and spawns the engine.
    starting: Mutex<HashSet<String>>,
    /// One plan at a time, so two loads don't both count the same free memory.
    ops: tokio::sync::Mutex<()>,
}

impl Planner {
    /// `explicit` are the ids whose context the operator set; every other llama.cpp or vLLM model
    /// is sized here.
    pub fn new(sup: Supervisor, budget: Budget, probe: Arc<dyn MemProbe>, models_dir: PathBuf, explicit: &HashSet<String>) -> Self {
        let known = sup
            .launches()
            .into_iter()
            .map(|l| {
                let k =
                    Known { resolved: resolvable(&l).then(|| kvfit::resolve(&l, &models_dir).ok()).flatten(), explicit: explicit.contains(&l.id) };
                (l.id, k)
            })
            .collect();
        Self {
            sup,
            budget,
            probe,
            models_dir,
            grace: GRACE,
            known: Mutex::new(known),
            busy: Mutex::new(HashMap::new()),
            resizing: Mutex::new(HashMap::new()),
            starting: Mutex::new(HashSet::new()),
            ops: tokio::sync::Mutex::new(()),
        }
    }

    pub fn with_grace(mut self, grace: Duration) -> Self {
        self.grace = grace;
        self
    }

    /// Notes a request for `id` (its idle time starts again). A request for a LoRA adapter keeps
    /// its base busy.
    pub fn touch(&self, id: &str) {
        let id = self.base(id);
        self.busy.lock().unwrap_or_else(|e| e.into_inner()).insert(id, Instant::now());
    }

    /// The model `id` runs on: its base for an adapter, else `id`.
    fn base(&self, id: &str) -> String {
        crate::adapters::base_of(&self.sup.launches(), id).unwrap_or_else(|| id.to_string())
    }

    fn residents(&self, status: &[ModelStatus]) -> Vec<Resident> {
        let launches: HashMap<String, ModelLaunch> = self.sup.launches().into_iter().map(|l| (l.id.clone(), l)).collect();
        let known = self.known.lock().unwrap_or_else(|e| e.into_inner());
        let busy = self.busy.lock().unwrap_or_else(|e| e.into_inner());
        let starting = self.starting.lock().unwrap_or_else(|e| e.into_inner());
        status
            .iter()
            // An adapter's row mirrors its base's; the base's cost counts the adapters.
            .filter(|s| s.base.is_none())
            .filter(|s| {
                matches!(s.state, ModelState::Ready | ModelState::Loading | ModelState::Draining)
                    || (s.state == ModelState::Unloaded && starting.contains(&s.id))
            })
            .map(|s| {
                let bytes = match (launches.get(&s.id), known.get(&s.id).and_then(|k| k.resolved.as_ref())) {
                    (Some(l), Some(r)) => kvfit::cost(r, l),
                    _ if s.wrapped => 0,
                    _ => kvfit::footprint(s.size_bytes.unwrap_or(0), 0),
                };
                Resident {
                    id: s.id.clone(),
                    swap_group: s.swap_group.clone(),
                    bytes,
                    in_flight: s.in_flight,
                    loading: s.state == ModelState::Loading || starting.contains(&s.id),
                    idle: busy.get(&s.id).map_or(Duration::MAX, Instant::elapsed),
                }
            })
            .collect()
    }

    fn is_starting(&self, id: &str) -> bool {
        self.starting.lock().unwrap_or_else(|e| e.into_inner()).contains(id)
    }

    /// Whether `id` may be re-added with a new context now: not loaded, and no load under way.
    fn may_resize(&self, id: &str, force: bool) -> bool {
        resizable(self.sup.state(id), force) && !self.is_starting(id)
    }

    /// Readies `id`'s launch for a load: unloads idle models it needs the room of, then sets the
    /// largest context that fits beside what still runs. Logs what it did and why. A model that
    /// failed is re-sized only on a `force` load (re-adding it clears its crash history, which an
    /// operator's forced load does anyway).
    pub async fn prepare(&self, id: &str, force: bool) {
        let _op = self.ops.lock().await;
        self.prepare_locked(&self.base(id), force).await;
    }

    /// [`Planner::prepare`], then the load, with no re-size between them.
    pub async fn load(&self, id: &str, force: bool) -> Result<(), weirdnet_agent::remote::RemoteError> {
        let _op = self.ops.lock().await;
        // An adapter is sized and loaded as its base (the supervisor loads the base for it).
        self.prepare_locked(&self.base(id), force).await;
        let r = self.sup.load_with(id, force).await;
        if r.is_ok() {
            self.touch(id);
        }
        r
    }

    async fn prepare_locked(&self, id: &str, force: bool) {
        if !self.may_resize(id, force) {
            return;
        }
        let Some(launch) = self.sup.launches().into_iter().find(|l| l.id == id) else { return };
        let Some((resolved, explicit)) = self.known_of(&launch) else { return };
        let ask = Ask { launch: &launch, resolved: &resolved, explicit };
        let d = decide(&ask, &self.residents(&self.sup.status()), &self.budget, self.probe.free_bytes(), self.grace, true);
        for other in &d.evict {
            tracing::info!(model = %other, for_model = %id, "unloading idle {other} to give {id} its context");
            if let Err(e) = self.sup.unload(other).await {
                tracing::warn!(model = %other, "unloading for {id}: {e}");
            }
        }
        // Measure again now the room is made.
        let after = if d.evict.is_empty() {
            d
        } else {
            let mut again = decide(&ask, &self.residents(&self.sup.status()), &self.budget, self.probe.free_bytes(), self.grace, false);
            again.note = format!("{}; unloaded idle {} for it", again.note, d.evict.join(", "));
            again.evict = d.evict;
            again
        };
        if after.short && explicit {
            tracing::warn!(model = %id, "{}", after.note);
        } else {
            tracing::info!(model = %id, "{}", after.note);
        }
        self.set_context(&launch, after.context, force).await;
    }

    /// `launch`'s shape and whether its context is the operator's, when it is sized here.
    fn known_of(&self, launch: &ModelLaunch) -> Option<(Resolved, bool)> {
        let mut known = self.known.lock().unwrap_or_else(|e| e.into_inner());
        let k = known.entry(launch.id.clone()).or_insert_with(|| Known {
            resolved: resolvable(launch).then(|| kvfit::resolve(launch, &self.models_dir).ok()).flatten(),
            explicit: launch.context.is_some(),
        });
        Some((k.resolved.clone()?, k.explicit))
    }

    /// Re-adds `launch` with `context`, when that is a change (the supervisor has no setter; the
    /// model is not loaded, so nothing is lost).
    ///
    /// The supervisor has no "re-size if still unloaded": `remove_model` waits for a load that
    /// holds the supervisor and then stops what it loaded. So the events are watched from before
    /// the check: a `Starting` for the model ahead of its `Removed` means a load came in between,
    /// and the model goes back with the context it loaded with and is loaded again.
    async fn set_context(&self, launch: &ModelLaunch, context: u32, force: bool) {
        let mut events = self.sup.subscribe();
        if launch.context == Some(context) || !self.may_resize(&launch.id, force) {
            return;
        }
        let id = launch.id.clone();
        *self.resizing.lock().unwrap_or_else(|e| e.into_inner()).entry(id.clone()).or_default() += 2;
        let mut l = launch.clone();
        l.context = Some(context);
        match self.sup.remove_model(&id).await {
            Ok(removed) => {
                if loaded_before_removal(&mut events, &id) {
                    tracing::warn!(model = %id, "a load of {id} came in while it was re-sized to {context}; loading it again as it was");
                    if let Err(e) = self.sup.add_model(removed) {
                        tracing::error!(model = %id, "re-adding: {e}");
                        return;
                    }
                    let sup = self.sup.clone();
                    tokio::spawn(async move {
                        if let Err(e) = sup.load(&id).await {
                            tracing::warn!(model = %id, "loading again: {e}");
                        }
                    });
                    return;
                }
                if let Err(e) = self.sup.add_model(l) {
                    tracing::error!(model = %id, "re-adding with context {context}: {e}");
                    let _ = self.sup.add_model(launch.clone());
                }
            }
            Err(e) => {
                tracing::warn!(model = %id, "setting context {context}: {e}");
                let mut r = self.resizing.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(n) = r.get_mut(&id) {
                    *n = n.saturating_sub(2);
                }
            }
        }
    }

    /// Re-sizes every automatic model that is not loaded to what fits beside what runs now,
    /// sharing the room among those that could load together (as the start-up plan does).
    pub async fn refresh(&self) {
        let _op = self.ops.lock().await;
        let status = self.sup.status();
        let residents = self.residents(&status);
        let measured = self.probe.free_bytes();
        let launches = self.sup.launches();
        let unloaded: Vec<&ModelLaunch> =
            launches.iter().filter(|l| status.iter().any(|s| s.id == l.id && s.state == ModelState::Unloaded) && resolvable(l)).collect();
        let unloaded: Vec<&ModelLaunch> = unloaded.into_iter().filter(|l| !self.is_starting(&l.id)).collect();
        let groups = |l: &ModelLaunch| l.swap_group.clone().unwrap_or_else(|| format!("\0{}", l.id));
        let running_groups: HashSet<String> = residents.iter().map(|r| r.swap_group.clone().unwrap_or_else(|| format!("\0{}", r.id))).collect();
        // Models not loaded that could load beside each other, as at start
        // ([`kvfit::plan_launches`]): each takes its weights (an explicit one its whole cost; the
        // largest per swap group), and the automatic ones share the room left for contexts.
        let mut waiting: HashSet<String> = HashSet::new();
        let mut fixed: HashMap<String, u64> = HashMap::new();
        for l in &unloaded {
            let g = groups(l);
            if running_groups.contains(&g) {
                continue;
            }
            match self.known_of(l) {
                Some((r, false)) => {
                    waiting.insert(g.clone());
                    let e = fixed.entry(g).or_default();
                    *e = (*e).max(kvfit::footprint(r.weights, 0));
                }
                Some((r, true)) => {
                    let e = fixed.entry(g).or_default();
                    *e = (*e).max(kvfit::cost(&r, l));
                }
                None => {}
            }
        }
        for l in unloaded {
            let Some((resolved, explicit)) = self.known_of(l) else { continue };
            if explicit {
                continue;
            }
            let held: u64 = fixed.iter().filter(|(g, _)| **g != groups(l)).map(|(_, c)| *c).sum();
            let free = free_beside(l, &residents, &HashSet::new(), &self.budget, measured).saturating_sub(held);
            let sharers = 1 + waiting.iter().filter(|g| **g != groups(l)).count() as u64;
            let room = kvfit::room_for_context(&resolved, l, &self.budget, free) / sharers;
            let (context, _) = kvfit::fit_context(&resolved.shape, room, Run::of(l));
            if l.context != Some(context) {
                tracing::info!(
                    model = %l.id,
                    "context {context} until it is asked for ({} free beside what runs{}): was {}",
                    gib(free),
                    if sharers > 1 { format!(", shared with {} other model(s) not loaded", sharers - 1) } else { String::new() },
                    l.context.map_or(String::from("unset"), |c| c.to_string())
                );
                self.set_context(l, context, false).await;
            }
        }
    }

    /// Follows the supervisor: notes requests under way, learns models added at run time, and
    /// re-sizes unloaded models when what runs changes (and every half minute, for measured
    /// memory).
    pub fn spawn_keeper(self: &Arc<Self>) -> tokio::task::JoinHandle<()> {
        let me = self.clone();
        let mut events = self.sup.subscribe();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(TICK);
            let mut last = Instant::now();
            let mut dirty = true;
            loop {
                tokio::select! {
                    ev = events.recv() => match ev {
                        Ok(ev) => dirty |= me.note(&ev),
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => dirty = me.lagged(),
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => return,
                    },
                    _ = tick.tick() => {
                        for s in me.sup.status() {
                            if s.in_flight > 0 {
                                me.touch(&s.id);
                            }
                        }
                        // Take in what has already happened (a `Starting` above all) first.
                        loop {
                            match events.try_recv() {
                                Ok(ev) => dirty |= me.note(&ev),
                                Err(tokio::sync::broadcast::error::TryRecvError::Lagged(_)) => dirty = me.lagged(),
                                Err(_) => break,
                            }
                        }
                        if dirty || last.elapsed() >= RESIZE_EVERY {
                            me.refresh().await;
                            last = Instant::now();
                            dirty = false;
                        }
                    }
                }
            }
        })
    }

    /// Events were missed: a load whose end went by unseen is not under way any more unless the
    /// supervisor says it is still loading. Always `true` (anything may have changed).
    fn lagged(&self) -> bool {
        self.starting.lock().unwrap_or_else(|e| e.into_inner()).retain(|id| self.sup.state(id) == Some(ModelState::Loading));
        true
    }

    /// Takes in one supervisor event; `true` when what runs changed.
    fn note(&self, ev: &SupervisorEvent) -> bool {
        match ev {
            SupervisorEvent::Starting { model } => {
                self.starting.lock().unwrap_or_else(|e| e.into_inner()).insert(model.clone());
                true
            }
            SupervisorEvent::Ready { model, .. } => {
                self.starting.lock().unwrap_or_else(|e| e.into_inner()).remove(model);
                self.touch(model);
                true
            }
            SupervisorEvent::Added { model } | SupervisorEvent::Removed { model } => {
                if matches!(ev, SupervisorEvent::Removed { .. }) {
                    self.starting.lock().unwrap_or_else(|e| e.into_inner()).remove(model);
                }
                {
                    let mut r = self.resizing.lock().unwrap_or_else(|e| e.into_inner());
                    if let Some(n) = r.get_mut(model).filter(|n| **n > 0) {
                        *n -= 1;
                        return false;
                    }
                }
                let launches = self.sup.launches();
                // An adapter added to (or removed from) a base changes the base's weights: take
                // the base in again.
                let base = launches.iter().find(|l| l.adapters.iter().any(|a| &a.id == model)).map(|l| l.id.clone());
                let adapter = base.is_some();
                let model = &base.unwrap_or_else(|| model.clone());
                let mut known = self.known.lock().unwrap_or_else(|e| e.into_inner());
                // Whether the base's context is the operator's does not change with its adapters.
                let explicit = known.get(model).map(|k| k.explicit).filter(|_| adapter);
                known.remove(model);
                if let Some(l) = launches.into_iter().find(|l| &l.id == model) {
                    let resolved = resolvable(&l).then(|| kvfit::resolve(&l, &self.models_dir).ok()).flatten();
                    let explicit = explicit.unwrap_or(l.context.is_some());
                    known.insert(model.clone(), Known { resolved, explicit });
                }
                true
            }
            SupervisorEvent::Unloaded { model, .. } | SupervisorEvent::LoadFailed { model, .. } | SupervisorEvent::Crashed { model, .. } => {
                self.starting.lock().unwrap_or_else(|e| e.into_inner()).remove(model);
                true
            }
            _ => false,
        }
    }
}

/// Whether a model in `state` may be re-added with a new context now.
fn resizable(state: Option<ModelState>, force: bool) -> bool {
    match state {
        Some(ModelState::Unloaded) => true,
        Some(ModelState::Failed(_)) => force,
        _ => false,
    }
}

/// Whether `events` (subscribed before a re-size's check) show a load of `id` beginning before
/// `id` was removed: the removal then stopped what that load brought up.
fn loaded_before_removal(events: &mut tokio::sync::broadcast::Receiver<SupervisorEvent>, id: &str) -> bool {
    loop {
        match events.try_recv() {
            Ok(SupervisorEvent::Starting { model }) if model == id => return true,
            Ok(SupervisorEvent::Removed { model }) if model == id => return false,
            Ok(_) => {}
            // Missed some: what follows still holds the `Removed`.
            Err(tokio::sync::broadcast::error::TryRecvError::Lagged(_)) => {}
            Err(_) => return false,
        }
    }
}

/// Whether `l` is sized here: a llama.cpp or vLLM model ymird starts itself.
fn resolvable(l: &ModelLaunch) -> bool {
    l.wrap.is_none() && matches!(l.engine, EngineKind::Llamacpp | EngineKind::Vllm)
}

#[cfg(test)]
mod tests;
