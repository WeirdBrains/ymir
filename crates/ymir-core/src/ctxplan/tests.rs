use super::*;
use crate::kvfit::tests::{GIB, gemma4_e4b_header, inkling_header, spark_budget, sparse};

/// Inkling 2 Mini IQ4_NL with its Q8_0 mmproj, as on m2 (ggml-org/Inkling-2-Mini-GGUF).
const INK_WEIGHTS: u64 = 17_627_940_768 + 345_910_944;
/// Gemma 4 E4B Q4_0.
const GEMMA_WEIGHTS: u64 = 4_590_807_392;
/// What llama-server took for Inkling at 1M tokens, 4 slots, on m2 (a 128 GB Spark).
const M2_MEASURED_AT_1M: u64 = 104_600 * GIB / 1000;

fn ink() -> (ModelLaunch, Resolved) {
    let l = ModelLaunch::new("inkling-2-mini", EngineKind::Llamacpp, "/m/ink.gguf");
    (l, Resolved { shape: kvfit::gguf_shape(&inkling_header()[..]).unwrap(), weights: INK_WEIGHTS })
}

fn gemma() -> (ModelLaunch, Resolved) {
    let l = ModelLaunch::new("gemma4-e4b", EngineKind::Llamacpp, "/m/gemma.gguf");
    (l, Resolved { shape: kvfit::gguf_shape(&gemma4_e4b_header()[..]).unwrap(), weights: GEMMA_WEIGHTS })
}

/// Gemma running at its full 131072 context.
fn gemma_running(in_flight: usize, idle: Duration) -> Resident {
    let (mut l, r) = gemma();
    l.context = Some(131_072);
    Resident { id: l.id.clone(), swap_group: None, bytes: kvfit::cost(&r, &l), in_flight, loading: false, idle }
}

fn ask<'a>(l: &'a ModelLaunch, r: &'a Resolved) -> Ask<'a> {
    Ask { launch: l, resolved: r, explicit: false }
}

const LONG: Duration = Duration::from_secs(600);

#[test]
fn inkling_alone_on_a_spark_is_cut_to_what_fits_with_its_score_buffers() {
    let b = spark_budget();
    let (l, r) = ink();
    let d = decide(&ask(&l, &r), &[], &b, None, GRACE, true);
    assert_eq!((d.context, d.short), (721_920, true), "{}", d.note);
    assert!(d.evict.is_empty());
    assert!(d.note.contains("cut from 1048576") && d.note.contains("20% kept free"), "{}", d.note);
    // At the chosen context the estimate, with its 10%, leaves the headroom free.
    let mut at = l.clone();
    at.context = Some(d.context);
    assert!(kvfit::cost(&r, &at) <= b.usable_bytes / 100 * 85, "{} GiB", kvfit::cost(&r, &at) / GIB);
    // At 1M the estimate is above what m2 measured, never below it (v0.2.0 said 53.9 GiB).
    at.context = Some(1_048_576);
    assert!(kvfit::cost(&r, &at) > M2_MEASURED_AT_1M);
}

#[test]
fn a_busy_neighbour_is_kept_and_the_context_shrinks_to_fit_beside_it() {
    let b = spark_budget();
    let (l, r) = ink();
    // A request under way: never unloaded, however long since it last started one.
    let d = decide(&ask(&l, &r), &[gemma_running(1, LONG)], &b, None, GRACE, true);
    assert!(d.evict.is_empty(), "{d:?}");
    assert_eq!(d.context, 655_360, "{}", d.note);
    assert!(d.note.contains("beside gemma4-e4b"), "{}", d.note);
    // Loading counts as busy too.
    let mut loading = gemma_running(0, LONG);
    loading.loading = true;
    assert!(decide(&ask(&l, &r), &[loading], &b, None, GRACE, true).evict.is_empty());
}

#[test]
fn an_idle_neighbour_is_unloaded_and_the_model_grows_to_its_alone_context() {
    let b = spark_budget();
    let (l, r) = ink();
    let d = decide(&ask(&l, &r), &[gemma_running(0, LONG)], &b, None, GRACE, true);
    assert_eq!(d.evict, vec!["gemma4-e4b".to_string()]);
    assert_eq!(d.context, 721_920, "{}", d.note);
    assert!(d.note.contains("unloaded idle gemma4-e4b"), "{}", d.note);
    // Used within the grace period: kept (no ping-pong between two models).
    let d = decide(&ask(&l, &r), &[gemma_running(0, Duration::from_secs(5))], &b, None, GRACE, true);
    assert!(d.evict.is_empty());
    assert_eq!(d.context, 655_360);
    // Eviction off (the second pass, after unloading): nothing unloaded.
    assert!(decide(&ask(&l, &r), &[gemma_running(0, LONG)], &b, None, GRACE, false).evict.is_empty());
}

#[test]
fn nothing_is_unloaded_for_a_model_that_fits_whole_anyway() {
    let b = spark_budget();
    let (gl, gr) = gemma();
    let (mut il, ir) = ink();
    il.context = Some(721_920);
    let idle_ink = Resident { id: il.id.clone(), swap_group: None, bytes: kvfit::cost(&ir, &il), in_flight: 0, loading: false, idle: LONG };
    let d = decide(&ask(&gl, &gr), &[idle_ink], &b, None, GRACE, true);
    assert!(d.evict.is_empty(), "{d:?}");
    assert_eq!((d.context, d.short), (131_072, false));
}

#[test]
fn measured_memory_wins_when_it_is_lower_than_the_estimate() {
    let b = spark_budget();
    let (l, r) = ink();
    let estimated = decide(&ask(&l, &r), &[gemma_running(1, LONG)], &b, None, GRACE, true);
    // The node says only 70 GiB are free (the neighbour uses more than estimated).
    let measured = decide(&ask(&l, &r), &[gemma_running(1, LONG)], &b, Some(70 * GIB), GRACE, true);
    assert!(measured.context < estimated.context, "{} vs {}", measured.context, estimated.context);
    assert_eq!(measured.free, 70 * GIB);
    assert!(measured.note.contains("70.0 GiB measured free"), "{}", measured.note);
    // An idle neighbour unloaded gives back what it took: measured plus its bytes, up to the estimate.
    let g = gemma_running(0, LONG);
    let d = decide(&ask(&l, &r), std::slice::from_ref(&g), &b, Some(70 * GIB), GRACE, true);
    assert_eq!(d.free, (70 * GIB + g.bytes).min(b.usable_bytes));
    // More measured than the budget: the budget.
    assert_eq!(decide(&ask(&l, &r), &[], &b, Some(500 * GIB), GRACE, true).free, b.usable_bytes);
}

#[test]
fn a_swap_group_mate_is_not_counted_and_not_evicted_here() {
    let b = spark_budget();
    let (mut l, r) = ink();
    l.swap_group = Some("main".into());
    let mut mate = gemma_running(1, LONG);
    mate.swap_group = Some("main".into());
    let d = decide(&ask(&l, &r), &[mate], &b, None, GRACE, true);
    assert!(d.evict.is_empty());
    assert_eq!(d.context, 721_920);
}

#[test]
fn an_explicit_context_is_kept_and_may_still_make_room() {
    let b = spark_budget();
    let (mut l, r) = ink();
    l.context = Some(1_048_576);
    let a = Ask { launch: &l, resolved: &r, explicit: true };
    // It doesn't fit beside an idle neighbour: the neighbour goes, the context stays, with a warning.
    let d = decide(&a, &[gemma_running(0, LONG)], &b, None, GRACE, true);
    assert_eq!(d.context, 1_048_576);
    assert_eq!(d.evict, vec!["gemma4-e4b".to_string()]);
    assert!(d.short && d.note.contains("may not fit"), "{}", d.note);
    // A small explicit context that fits: nothing unloaded.
    l.context = Some(32_768);
    let a = Ask { launch: &l, resolved: &r, explicit: true };
    let d = decide(&a, &[gemma_running(0, LONG)], &b, None, GRACE, true);
    assert!(d.evict.is_empty() && !d.short && d.context == 32_768, "{d:?}");
}

#[test]
fn meminfo_and_nvidia_smi_are_read() {
    let meminfo = "MemTotal:       131072000 kB\nMemFree:         1000 kB\nMemAvailable:   100000000 kB\n";
    assert_eq!(mem_available(meminfo), Some(100_000_000 * 1024));
    assert_eq!(mem_available("MemTotal: 1 kB\n"), None);
    assert_eq!(nvidia_free("24000\n 1000 \n"), Some(25_000 << 20));
    assert_eq!(nvidia_free("N/A\n"), None);
}

struct Fixed(Option<u64>);

impl MemProbe for Fixed {
    fn free_bytes(&self) -> Option<u64> {
        self.0
    }
}

#[tokio::test]
async fn the_planner_sets_the_context_the_catalog_reports_and_keeps_explicit_ones() {
    let dir = tempfile::tempdir().unwrap();
    let ink_file = sparse(dir.path(), "ink.gguf", &inkling_header(), INK_WEIGHTS);
    let gemma_file = sparse(dir.path(), "gemma.gguf", &gemma4_e4b_header(), GEMMA_WEIGHTS);
    let mut fixed = ModelLaunch::new("fixed", EngineKind::Llamacpp, &gemma_file);
    fixed.context = Some(8192);
    let mut launches = vec![
        ModelLaunch::new("inkling-2-mini", EngineKind::Llamacpp, &ink_file),
        ModelLaunch::new("gemma4-e4b", EngineKind::Llamacpp, &gemma_file),
        fixed,
    ];
    let budget = spark_budget();
    let plans = kvfit::plan_launches(&mut launches, &budget, dir.path());
    let explicit: HashSet<String> = plans.iter().filter(|p| p.explicit).map(|p| p.id.clone()).collect();
    assert_eq!(explicit, HashSet::from(["fixed".to_string()]));
    let sup = Supervisor::new(vec![], launches, Default::default());
    let planner = Planner::new(sup.clone(), budget, Arc::new(Fixed(None)), dir.path().into(), &explicit);
    let context = |id: &str| sup.status().into_iter().find(|s| s.id == id).unwrap().context;
    // At start the three share the node.
    let shared = context("inkling-2-mini").unwrap();
    assert!(shared < 655_360, "{shared}");

    // Asked for, with nothing running: everything free is its (the catalog says so too).
    planner.prepare("inkling-2-mini", false).await;
    let (_, r) = ink();
    let mut l = sup.launches().into_iter().find(|l| l.id == "inkling-2-mini").unwrap();
    let want = decide(&Ask { launch: &l, resolved: &r, explicit: false }, &[], &budget, None, GRACE, false);
    assert_eq!(context("inkling-2-mini"), Some(want.context));
    assert_eq!(want.context, 721_920);
    // Its other settings survive the re-add.
    l.context = None;
    assert_eq!(l.source, ink_file);

    // The explicit one is never touched.
    planner.prepare("fixed", false).await;
    planner.refresh().await;
    assert_eq!(context("fixed"), Some(8192));
    // A refresh with nothing running shares the room among those not loaded again, as the
    // start-up plan does (each one's weights count against the others).
    assert_eq!(context("inkling-2-mini"), Some(shared));
    assert_eq!(context("gemma4-e4b"), Some(131_072));
}

/// Two Inklings on a Spark, planned to share it.
fn two_inklings(dir: &std::path::Path) -> (Supervisor, Planner, u32) {
    let file = sparse(dir, "ink.gguf", &inkling_header(), INK_WEIGHTS);
    let mut launches = vec![ModelLaunch::new("a", EngineKind::Llamacpp, &file), ModelLaunch::new("b", EngineKind::Llamacpp, &file)];
    let budget = spark_budget();
    kvfit::plan_launches(&mut launches, &budget, dir);
    let shared = launches[0].context.unwrap();
    let sup = Supervisor::new(vec![], launches, Default::default());
    let planner = Planner::new(sup.clone(), budget, Arc::new(Fixed(None)), dir.into(), &HashSet::new());
    (sup, planner, shared)
}

#[tokio::test]
async fn a_model_whose_load_has_begun_is_not_resized_and_the_others_are_sized_beside_it() {
    let dir = tempfile::tempdir().unwrap();
    let (sup, planner, shared) = two_inklings(dir.path());
    let context = |id: &str| sup.status().into_iter().find(|s| s.id == id).unwrap().context;
    // A refresh with nothing running keeps the start-up share.
    planner.refresh().await;
    assert_eq!((context("a"), context("b")), (Some(shared), Some(shared)));
    // A load of a from elsewhere has begun; the supervisor still says a is unloaded.
    assert!(planner.note(&SupervisorEvent::Starting { model: "a".into() }));
    assert_eq!(sup.state("a"), Some(ModelState::Unloaded));
    planner.refresh().await;
    planner.prepare("a", false).await;
    assert_eq!(context("a"), Some(shared), "a is loading at its share: left alone");
    let b = context("b").unwrap();
    assert!(b > shared && b < 721_920, "b gets what is left beside a: {b} vs {shared}");
    // Its load failed: a is sized again (beside nothing, sharing with b).
    assert!(planner.note(&SupervisorEvent::LoadFailed { model: "a".into(), why: String::new() }));
    planner.refresh().await;
    assert_eq!((context("a"), context("b")), (Some(shared), Some(shared)));
}

#[test]
fn a_load_that_came_in_before_a_resize_removed_the_model_is_seen() {
    let (tx, _) = tokio::sync::broadcast::channel(16);
    let starting = |m: &str| SupervisorEvent::Starting { model: m.into() };
    let removed = |m: &str| SupervisorEvent::Removed { model: m.into() };
    // a was loading when the removal got the supervisor: it stopped a.
    let mut rx = tx.subscribe();
    for ev in [starting("b"), starting("a"), removed("a")] {
        tx.send(ev).unwrap();
    }
    assert!(loaded_before_removal(&mut rx, "a"));
    // A load of a after its removal (or of another model) is not one.
    let mut rx = tx.subscribe();
    for ev in [starting("b"), removed("a"), starting("a")] {
        tx.send(ev).unwrap();
    }
    assert!(!loaded_before_removal(&mut rx, "a"));
}
