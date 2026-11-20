//! Load balancing across replicas (M6): which model node answers a request for a model that
//! several have ready.
//!
//! Each candidate gets a predicted completion time and the best of two random candidates wins
//! (power of two choices), so stale capability records don't send every request to the same
//! "least loaded" node:
//!
//! ```text
//! wait = queue_depth / slots * mean_service
//! cost = wait + mean_service + (cold ? load_estimate : 0) + penalty
//! ```
//!
//! `queue_depth` is the larger of what the node reported and what this coordinator itself has
//! sent it that hasn't finished (so a burst doesn't pile onto one node between reports).
//! `mean_service` is measured here (an average of finished requests per node and model, 2 s until
//! one finishes). A `Busy` answer adds a penalty that fades. A node on this device wins ties.
//! Not built yet: prefill and decode speed per node, uncached prompt length from prefix hints,
//! round-trip time and a relay penalty (they need data the capability record doesn't carry).

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Service time assumed before a node has finished any request.
const DEFAULT_SERVICE: Duration = Duration::from_secs(2);
/// How long a `Busy` answer counts against a node, and what it adds at first.
const PENALTY_FOR: Duration = Duration::from_secs(5);
const PENALTY: Duration = Duration::from_secs(4);
/// Costs closer than this are ties (the local node wins them).
const TIE: f64 = 0.05;

/// What the scorer knows about one candidate.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub node: String,
    pub local: bool,
    /// Requests the node reports in flight for this model.
    pub reported: usize,
    /// Concurrent sequences the model runs (1 if unknown).
    pub slots: usize,
    /// Estimated load time when the model isn't ready there.
    pub cold: Option<Duration>,
}

#[derive(Default)]
struct Stats {
    /// Requests sent here and not finished.
    sent: usize,
    /// Exponential average of service time, per model, in seconds.
    service: HashMap<String, f64>,
    penalty_until: Option<Instant>,
}

#[derive(Default)]
pub struct Balancer {
    stats: Mutex<HashMap<String, Stats>>,
    seed: AtomicU64,
}

/// A request counted against a node until it ends.
pub struct Dispatch<'a> {
    balancer: &'a Balancer,
    node: String,
    model: String,
    started: Instant,
    done: bool,
}

impl Balancer {
    pub fn new() -> Self {
        Self { stats: Mutex::default(), seed: AtomicU64::new(0x9E37_79B9_7F4A_7C15 ^ std::process::id() as u64) }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, Stats>> {
        self.stats.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn rand(&self, n: usize) -> usize {
        // splitmix64 over a counter: enough spread for choosing two candidates.
        let mut z = self.seed.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed).wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) % n as u64) as usize
    }

    /// Predicted completion time of a request for `model` on `c`, in seconds.
    pub fn cost(&self, model: &str, c: &Candidate) -> f64 {
        let stats = self.lock();
        let st = stats.get(&c.node);
        let service = st.and_then(|s| s.service.get(model)).copied().unwrap_or(DEFAULT_SERVICE.as_secs_f64());
        let depth = c.reported.max(st.map_or(0, |s| s.sent)) as f64;
        let wait = depth / c.slots.max(1) as f64 * service;
        let penalty = st
            .and_then(|s| s.penalty_until)
            .and_then(|until| until.checked_duration_since(Instant::now()))
            .map_or(0.0, |left| PENALTY.as_secs_f64() * left.as_secs_f64() / PENALTY_FOR.as_secs_f64());
        wait + service + c.cold.map_or(0.0, |d| d.as_secs_f64()) + penalty
    }

    /// Index into `cands` of the node to use: score two random candidates (all, if two or
    /// fewer), the cheaper wins, and a tie goes to a local node.
    pub fn choose(&self, model: &str, cands: &[Candidate]) -> Option<usize> {
        let n = cands.len();
        if n == 0 {
            return None;
        }
        if n == 1 {
            return Some(0);
        }
        let (a, b) = if n == 2 {
            (0, 1)
        } else {
            let a = self.rand(n);
            let mut b = self.rand(n - 1);
            if b >= a {
                b += 1;
            }
            (a, b)
        };
        let (ca, cb) = (self.cost(model, &cands[a]), self.cost(model, &cands[b]));
        let close = (ca - cb).abs() <= TIE * ca.max(cb);
        Some(match (close, cands[a].local, cands[b].local) {
            (true, true, false) => a,
            (true, false, true) => b,
            _ if ca <= cb => a,
            _ => b,
        })
    }

    /// Counts a request against `node` until the returned guard is finished or dropped.
    pub fn begin(&self, node: &str, model: &str) -> Dispatch<'_> {
        self.lock().entry(node.to_string()).or_default().sent += 1;
        Dispatch { balancer: self, node: node.into(), model: model.into(), started: Instant::now(), done: false }
    }

    /// A `Busy` answer from `node`: it costs more for a few seconds.
    pub fn busy(&self, node: &str) {
        self.lock().entry(node.to_string()).or_default().penalty_until = Some(Instant::now() + PENALTY_FOR);
    }

    /// Requests this coordinator has sent `node` that haven't finished.
    pub fn sent(&self, node: &str) -> usize {
        self.lock().get(node).map_or(0, |s| s.sent)
    }

    /// Forgets a node that left.
    pub fn forget(&self, node: &str) {
        self.lock().remove(node);
    }
}

impl Dispatch<'_> {
    /// The request finished well: its duration feeds the node's service-time average.
    pub fn finish(mut self) {
        self.done = true;
        let secs = self.started.elapsed().as_secs_f64();
        let mut stats = self.balancer.lock();
        if let Some(s) = stats.get_mut(&self.node) {
            s.sent = s.sent.saturating_sub(1);
            let e = s.service.entry(self.model.clone()).or_insert(secs);
            *e = 0.7 * *e + 0.3 * secs;
        }
    }
}

impl Drop for Dispatch<'_> {
    fn drop(&mut self) {
        if !self.done
            && let Some(s) = self.balancer.lock().get_mut(&self.node)
        {
            s.sent = s.sent.saturating_sub(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(node: &str, local: bool, reported: usize) -> Candidate {
        Candidate { node: node.into(), local, reported, slots: 1, cold: None }
    }

    #[test]
    fn the_shorter_queue_wins_and_a_tie_goes_local() {
        let b = Balancer::new();
        for _ in 0..50 {
            assert_eq!(b.choose("m", &[c("a", false, 3), c("b", false, 0)]), Some(1));
        }
        for _ in 0..50 {
            assert_eq!(b.choose("m", &[c("a", false, 1), c("b", true, 1)]), Some(1));
        }
    }

    #[test]
    fn our_own_sends_count_before_the_node_reports_them() {
        let b = Balancer::new();
        let _g: Vec<_> = (0..4).map(|_| b.begin("a", "m")).collect();
        assert_eq!(b.sent("a"), 4);
        assert_eq!(b.choose("m", &[c("a", false, 0), c("b", false, 0)]), Some(1));
    }

    #[test]
    fn a_busy_node_is_avoided_for_a_while() {
        let b = Balancer::new();
        b.busy("a");
        assert_eq!(b.choose("m", &[c("a", true, 0), c("b", false, 0)]), Some(1));
    }

    #[test]
    fn a_slower_node_gets_less_and_cold_costs_the_load() {
        let b = Balancer::new();
        b.lock().entry("slow".into()).or_default().service.insert("m".into(), 10.0);
        assert_eq!(b.choose("m", &[c("slow", false, 0), c("fast", false, 0)]), Some(1));
        let cold = Candidate { cold: Some(Duration::from_secs(30)), ..c("cold", false, 0) };
        assert_eq!(b.choose("m", &[cold, c("busy", false, 4)]), Some(1));
    }

    #[test]
    fn two_choices_spread_over_many_equal_nodes() {
        let b = Balancer::new();
        let cands: Vec<_> = (0..4).map(|i| c(&format!("n{i}"), false, 0)).collect();
        let mut hits = [0usize; 4];
        for _ in 0..400 {
            hits[b.choose("m", &cands).unwrap()] += 1;
        }
        assert!(hits.iter().all(|&h| h > 40), "{hits:?}");
    }

    #[test]
    fn finishing_updates_the_average_and_releases_the_slot() {
        let b = Balancer::new();
        let d = b.begin("a", "m");
        std::thread::sleep(Duration::from_millis(20));
        d.finish();
        assert_eq!(b.sent("a"), 0);
        assert!(b.lock()["a"].service["m"] < 1.0);
        drop(b.begin("a", "m"));
        assert_eq!(b.sent("a"), 0);
    }
}
