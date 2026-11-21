//! `ymirctl bench`: compare the models of the pool, on every node that serves them.
//!
//! By default the benchmark talks to each inference node directly over Weirdnet
//! ([`ymir_core::direct::Direct`]), so a node×model matrix is real: the same model can be compared on
//! different nodes. `--via bridge` goes through the loopback bridge instead, where the pool picks the
//! node (and the node that answered is recorded). For each target it measures time to first token,
//! decode and prompt-processing speed and total latency (one warm-up, then N repeats, median and p90),
//! then scores deterministic quality checks at temperature 0 (see [`suites`]).
//!
//! Safety: one request at a time, strictly in sequence, so no node runs two benchmark jobs. A model that
//! is busy or loading is re-checked and then skipped, and a model that is not loaded is skipped unless
//! `--cold`, which warns first because loading can evict what the node is serving. Every request carries
//! its own seed and a nonce, so neither the answer cache nor a node's prefix cache flatters the numbers.

mod compare;
pub mod hints;
mod report;
mod score;
mod suites;

use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use serde_json::{Map, Value, json};
use ymir_core::client::{BridgeClient, TimedChat};
use ymir_core::direct::{AskOptions, Direct};

pub use compare::CompareArgs;
use score::{judge, median, python_available};
pub use suites::Suite;
use suites::{perf_suite, quality_suite};

#[derive(Clone, Copy, clap::ValueEnum, Debug, PartialEq, Eq)]
pub enum Via {
    /// Each node directly over Weirdnet (a real node pin).
    Weirdnet,
    /// The loopback bridge: the pool picks the node.
    Bridge,
}

#[derive(clap::Args, Debug, Clone)]
pub struct BenchArgs {
    /// Only these nodes (repeatable): a name, a peer id or its start, or `all`. Default: every node.
    #[arg(long, short = 'n')]
    pub node: Vec<String>,
    /// Only these models (repeatable; a trailing `*` matches a prefix), or `all`. Default: every model.
    #[arg(long, short = 'm')]
    pub model: Vec<String>,
    /// How to reach the nodes.
    #[arg(long, value_enum, default_value_t = Via::Weirdnet)]
    pub via: Via,
    /// Extra address to dial a node at (`/ip4/…/tcp/…/p2p/PEER`), or `@FILE` with one per line;
    /// repeatable. `[network] peers` and the `bench-hints` file in the Ymir home are used too (see
    /// `bench hints add`).
    #[arg(long)]
    pub hint: Vec<String>,
    /// Seconds the quality checks of one model may take before the rest are left unrun (default: hard
    /// 480, others no limit; 0 for no limit).
    #[arg(long)]
    pub quality_budget: Option<u64>,
    /// Which prompts: quick (smoke test), standard, hard (built to tell models apart).
    #[arg(long, value_enum, default_value_t = Suite::Standard)]
    pub suite: Suite,
    /// Size of the long prompt in tokens, about (default about 1300).
    #[arg(long)]
    pub context: Option<u32>,
    /// Tokens the decode-speed story may generate (default 256, quick 128).
    #[arg(long)]
    pub max_tokens: Option<u64>,
    /// Tokens a quality answer may use, thinking included (default by suite: 1024, 2048, 3072).
    #[arg(long)]
    pub quality_tokens: Option<u64>,
    /// Timed runs of each performance prompt.
    #[arg(long, default_value_t = 3)]
    pub repeats: usize,
    /// Untimed runs before the timed ones (the first also loads a cold model).
    #[arg(long, default_value_t = 1)]
    pub warmup: usize,
    /// Also benchmark models that are not loaded, loading them one at a time. This can evict models the
    /// node is serving, so it warns first.
    #[arg(long, alias = "include-cold")]
    pub cold: bool,
    /// Do not ask before `--cold` loads models.
    #[arg(long, short = 'y')]
    pub yes: bool,
    /// Skip the quality checks.
    #[arg(long)]
    pub skip_quality: bool,
    /// Never run model-written code (skips the code quality checks).
    #[arg(long)]
    pub no_code_exec: bool,
    /// Seconds one request may take.
    #[arg(long, default_value_t = 300)]
    pub timeout: u64,
    /// Times to re-check a busy or loading model before skipping it.
    #[arg(long, default_value_t = 6)]
    pub busy_retries: usize,
    /// Seconds between those re-checks.
    #[arg(long, default_value_t = 20)]
    pub busy_wait: u64,
    /// Where the timestamped `.json`, `.md` and `.csv` results go.
    #[arg(long, default_value = "bench-results")]
    pub out_dir: PathBuf,
    /// Name nodes as they are. By default they are labelled node-a, node-b, ... so results can be shared.
    #[arg(long)]
    pub real_names: bool,
}

impl Default for BenchArgs {
    fn default() -> Self {
        Self {
            node: Vec::new(),
            model: Vec::new(),
            via: Via::Weirdnet,
            hint: Vec::new(),
            quality_budget: None,
            suite: Suite::Standard,
            context: None,
            max_tokens: None,
            quality_tokens: None,
            repeats: 3,
            warmup: 1,
            cold: false,
            yes: false,
            skip_quality: false,
            no_code_exec: false,
            timeout: 300,
            busy_retries: 6,
            busy_wait: 20,
            out_dir: PathBuf::from("bench-results"),
            real_names: false,
        }
    }
}

#[derive(clap::Args, Debug, Clone)]
pub struct ListArgs {
    /// How to reach the nodes.
    #[arg(long, value_enum, default_value_t = Via::Weirdnet)]
    pub via: Via,
    /// Extra address to dial a node at, or `@FILE`; repeatable (and the `bench-hints` file).
    #[arg(long)]
    pub hint: Vec<String>,
    /// Name nodes as they are (otherwise node-a, node-b, ...).
    #[arg(long)]
    pub real_names: bool,
}

#[derive(clap::Subcommand, Debug, Clone)]
pub enum BenchSub {
    /// The nodes and models a benchmark can run on.
    List(ListArgs),
    /// Two saved results side by side, with the change in each number.
    Compare(CompareArgs),
    /// Addresses to dial nodes at: `hints add` reads `ymirctl pool addresses` output from stdin.
    Hints(HintsArgs),
}

#[derive(clap::Args, Debug, Clone)]
pub struct HintsArgs {
    #[command(subcommand)]
    pub action: HintsAction,
}

#[derive(clap::Subcommand, Debug, Clone)]
pub enum HintsAction {
    /// Save the dialable addresses piped in (`ssh NODE ymirctl pool addresses | ymirctl bench hints add`).
    Add,
    /// The saved addresses.
    List,
}

#[derive(clap::Args, Debug, Clone)]
#[command(args_conflicts_with_subcommands = true)]
pub struct BenchCmd {
    #[command(subcommand)]
    pub sub: Option<BenchSub>,
    #[command(flatten)]
    pub run: BenchArgs,
}

// ---- backends ----

/// One model on one node (or, over the bridge, one model wherever the pool puts it).
#[derive(Debug, Clone, PartialEq)]
pub struct Target {
    /// The node's name; `routed` over the bridge.
    pub node: String,
    pub peer: Option<String>,
    /// `model` (the hardware itself), `inference` (an inference node) or `routed` (the bridge).
    pub role: String,
    pub model: String,
    pub purpose: String,
    pub state: String,
    pub replicas: u64,
    pub context: Option<u64>,
}

/// What a backend can see.
#[derive(Debug, Default)]
pub struct Listing {
    /// (peer id, name) of every linked node, answering or not.
    pub nodes: Vec<(String, String)>,
    pub targets: Vec<Target>,
    /// Nodes that could not be asked, with why.
    pub unreachable: Vec<(String, String)>,
}

#[async_trait]
pub trait Backend: Send + Sync {
    fn via(&self) -> &'static str;
    /// Whether a request can be pinned to a node.
    fn pinned(&self) -> bool;
    async fn listing(&self) -> Result<Listing>;
    async fn chat(&self, t: &Target, prompt: &str, o: &AskOptions) -> Result<TimedChat>;
    async fn embed(&self, t: &Target, inputs: &[String]) -> Result<(usize, Duration)>;
}

/// The loopback bridge.
pub struct BridgeBackend(pub BridgeClient);

#[async_trait]
impl Backend for BridgeBackend {
    fn via(&self) -> &'static str {
        "bridge"
    }

    fn pinned(&self) -> bool {
        false
    }

    async fn listing(&self) -> Result<Listing> {
        let v = self.0.models().await?;
        let mut l = Listing::default();
        for m in v["data"].as_array().into_iter().flatten() {
            let Some(id) = m["id"].as_str() else { continue };
            let w = &m["weirdnet"];
            l.targets.push(Target {
                node: "routed".into(),
                peer: None,
                role: "routed".into(),
                model: id.into(),
                purpose: w["purpose"].as_str().unwrap_or("chat").into(),
                state: w["state"].as_str().unwrap_or("unknown").into(),
                replicas: w["replicas"].as_u64().unwrap_or(0),
                context: w["context"].as_u64(),
            });
        }
        Ok(l)
    }

    async fn chat(&self, t: &Target, prompt: &str, o: &AskOptions) -> Result<TimedChat> {
        let mut extra = Map::new();
        extra.insert("temperature".into(), Value::from(f64::from(o.temperature)));
        extra.insert("max_tokens".into(), Value::from(o.max_tokens));
        extra.insert("seed".into(), Value::from(o.seed));
        self.0.chat_timed(&t.model, vec![json!({"role": "user", "content": prompt})], extra).await
    }

    async fn embed(&self, t: &Target, inputs: &[String]) -> Result<(usize, Duration)> {
        self.0.embed_timed(&t.model, inputs).await
    }
}

/// Each node over Weirdnet.
pub struct DirectBackend {
    pub direct: Direct,
}

#[async_trait]
impl Backend for DirectBackend {
    fn via(&self) -> &'static str {
        "weirdnet"
    }

    fn pinned(&self) -> bool {
        true
    }

    async fn listing(&self) -> Result<Listing> {
        let mut l = Listing::default();
        for n in self.direct.nodes().await {
            let name = short_label(&n.label);
            l.nodes.push((n.peer.clone(), name.clone()));
            match n.models {
                Ok(models) => {
                    for m in models {
                        l.targets.push(Target {
                            node: name.clone(),
                            peer: Some(n.peer.clone()),
                            role: n.role.into(),
                            model: m.id,
                            purpose: if m.embedding { "embedding".into() } else { "chat".into() },
                            state: m.state.into(),
                            replicas: u64::from(m.replicas),
                            context: m.context.map(u64::from),
                        });
                    }
                }
                Err(why) => l.unreachable.push((name, why)),
            }
        }
        Ok(l)
    }

    async fn chat(&self, t: &Target, prompt: &str, o: &AskOptions) -> Result<TimedChat> {
        let peer = t.peer.as_deref().context("a target without a node")?;
        self.direct.chat_timed(peer, &t.model, prompt, o).await
    }

    async fn embed(&self, t: &Target, inputs: &[String]) -> Result<(usize, Duration)> {
        let peer = t.peer.as_deref().context("a target without a node")?;
        self.direct.embed_timed(peer, &t.model, inputs).await
    }
}

/// `Ymir on m2` is `m2`.
fn short_label(label: &str) -> String {
    label.strip_prefix("Ymir on ").unwrap_or(label).to_string()
}

// ---- selection ----

/// The peers of `nodes` that one `--node` query names; an error when it names none or several.
fn resolve_node(q: &str, nodes: &[(String, String)]) -> Result<Vec<String>> {
    let ql = q.to_lowercase();
    let hits: Vec<&(String, String)> = nodes
        .iter()
        .filter(|(peer, name)| {
            let nl = name.to_lowercase();
            nl == ql
                || nl.split_whitespace().any(|w| w == ql)
                || peer == q
                || (q.len() >= 6 && peer.starts_with(q))
                || (q.len() >= 6 && peer.ends_with(q))
        })
        .collect();
    match hits.as_slice() {
        [] => bail!("no node `{q}` (see `ymirctl bench list`)"),
        [(peer, _)] => Ok(vec![peer.clone()]),
        many => bail!(
            "`{q}` matches {} nodes ({}); use a longer name or the peer id",
            many.len(),
            many.iter().map(|n| n.1.as_str()).collect::<Vec<_>>().join(", ")
        ),
    }
}

fn model_matches(wanted: &str, id: &str) -> bool {
    match wanted.strip_suffix('*') {
        Some(prefix) => id.starts_with(prefix),
        None => id == wanted,
    }
}

fn is_all(v: &[String]) -> bool {
    v.is_empty() || v.iter().any(|s| s == "all")
}

// ---- samples ----

#[derive(Debug, Clone)]
pub struct Sample {
    pub prompt: &'static str,
    /// The node the request went to (or, over the bridge, the node that answered).
    pub node: String,
    /// The node or model node that produced the answer.
    pub served_by: String,
    pub ttft_ms: f64,
    pub total_ms: f64,
    pub decode_tps: Option<f64>,
    pub pp_tps: Option<f64>,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
}

fn sample_of(prompt: &'static str, r: &TimedChat, pinned: Option<&str>) -> Sample {
    let total = r.total.as_secs_f64();
    let ttft = r.ttft.map_or(total, |t| t.as_secs_f64());
    let tokens = r.completion_tokens.or_else(|| (!r.text.is_empty()).then(|| (r.text.len() as u64).div_ceil(4)));
    let decode_tps = tokens.filter(|n| *n > 1 && total > ttft).map(|n| (n - 1) as f64 / (total - ttft));
    let fresh = r.prompt_tokens.map(|p| p.saturating_sub(r.cached_tokens.unwrap_or(0)));
    let pp_tps = fresh.filter(|n| *n > 0 && ttft > 0.0).map(|n| n as f64 / ttft);
    let served_by = r.served_by.clone().unwrap_or_else(|| "unknown".into());
    Sample {
        prompt,
        node: pinned.map_or_else(|| served_by.clone(), str::to_string),
        served_by,
        ttft_ms: ttft * 1000.0,
        total_ms: total * 1000.0,
        decode_tps,
        pp_tps,
        prompt_tokens: r.prompt_tokens,
        completion_tokens: r.completion_tokens,
    }
}

/// One quality check's outcome.
#[derive(Debug, Clone)]
pub struct QualityResult {
    pub check: &'static str,
    pub category: &'static str,
    pub node: String,
    /// `None` when the check could not run (no python3).
    pub passed: Option<bool>,
    /// The start of a wrong answer, to see why it failed.
    pub got: Option<String>,
}

#[derive(Debug, Default)]
pub struct TargetRun {
    pub model: String,
    pub node: String,
    pub role: String,
    pub purpose: String,
    pub state_at_start: String,
    pub replicas: u64,
    pub context: Option<u64>,
    pub skipped: Option<String>,
    pub cold_load_ms: Option<f64>,
    pub samples: Vec<Sample>,
    pub errors: Vec<String>,
    pub requests: usize,
    pub quality: Vec<QualityResult>,
    pub embed: Option<(usize, f64)>,
}

fn nonce() -> String {
    let n = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos());
    format!("{n:x}")
}

fn seed() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64 & 0x7fff_ffff_ffff)
}

struct Bench<'a> {
    backend: &'a dyn Backend,
    args: &'a BenchArgs,
    err: &'a mut dyn Write,
}

impl Bench<'_> {
    async fn ask_once(&self, t: &Target, prompt: &str, max_tokens: u64) -> Result<TimedChat> {
        let o = AskOptions { max_tokens: max_tokens as u32, temperature: 0.0, seed: seed(), wait: true };
        let call = self.backend.chat(t, prompt, &o);
        let r = tokio::time::timeout(Duration::from_secs(self.args.timeout), call)
            .await
            .map_err(|_| anyhow::anyhow!("timed out after {}s", self.args.timeout))??;
        if let Some(s) = &r.standin {
            bail!("the stand-in {s} answered instead");
        }
        Ok(r)
    }

    async fn ask(&self, t: &Target, prompt: &str, max_tokens: u64, fresh_prefix: bool) -> Result<TimedChat> {
        let nonced = |p: &str| format!("[run {}]\n{p}", nonce());
        let first = if fresh_prefix { nonced(prompt) } else { prompt.to_string() };
        let mut r = self.ask_once(t, &first, max_tokens).await?;
        if r.served_by.as_deref() == Some("cache") {
            // An answer-cache hit measures nothing; ask again with a prompt nobody has asked.
            r = self.ask_once(t, &nonced(prompt), max_tokens).await?;
            if r.served_by.as_deref() == Some("cache") {
                bail!("the answer cache answered");
            }
        }
        Ok(r)
    }

    /// Waits while the model is busy or loading; `Err(reason)` to skip it.
    async fn settle(&mut self, t: &Target) -> std::result::Result<Target, String> {
        let mut last = String::new();
        for attempt in 0..=self.args.busy_retries {
            let l = self.backend.listing().await.map_err(|e| format!("could not list models: {e}"))?;
            let Some(now) = l.targets.iter().find(|x| x.node == t.node && x.model == t.model) else {
                return Err(match l.unreachable.iter().find(|(n, _)| *n == t.node) {
                    Some((_, why)) => format!("node unreachable: {why}"),
                    None => "no longer listed".into(),
                });
            };
            match now.state.as_str() {
                "warm" => return Ok(now.clone()),
                "loadable" if self.args.cold => return Ok(now.clone()),
                "loadable" => return Err("not loaded (use --cold to load it)".into()),
                other => {
                    last = other.to_string();
                    if attempt < self.args.busy_retries {
                        let _ = writeln!(
                            self.err,
                            "  {} on {} is {last}; waiting {}s ({}/{})",
                            t.model,
                            t.node,
                            self.args.busy_wait,
                            attempt + 1,
                            self.args.busy_retries
                        );
                        tokio::time::sleep(Duration::from_secs(self.args.busy_wait)).await;
                    }
                }
            }
        }
        Err(format!("still {last} after {} checks", self.args.busy_retries))
    }

    async fn run_target(&mut self, t: &Target) -> TargetRun {
        let mut run =
            TargetRun { model: t.model.clone(), node: t.node.clone(), purpose: t.purpose.clone(), context: t.context, ..Default::default() };
        let t = match self.settle(t).await {
            Ok(now) => now,
            Err(why) => {
                run.skipped = Some(why);
                return run;
            }
        };
        run.state_at_start = t.state.clone();
        run.replicas = t.replicas;
        run.context = t.context;
        if t.purpose == "embedding" {
            self.embedding(&t, &mut run).await;
            return run;
        }
        let pinned = self.backend.pinned().then(|| t.node.clone());
        let cold = t.state == "loadable";
        let mut consecutive_errors = 0;
        // Warm-up; the first one of a cold model times the load.
        for i in 0..self.args.warmup.max(usize::from(cold)) {
            let started = std::time::Instant::now();
            run.requests += 1;
            match self.ask(&t, "Say hello in five words.", 32, true).await {
                Ok(_) if i == 0 && cold => run.cold_load_ms = Some(started.elapsed().as_secs_f64() * 1000.0),
                Ok(_) => {}
                Err(e) => {
                    run.errors.push(format!("warm-up: {e:#}"));
                    consecutive_errors += 1;
                }
            }
        }
        'perf: for p in perf_suite(self.args.suite, self.args.context, self.args.max_tokens) {
            for _ in 0..self.args.repeats {
                run.requests += 1;
                match self.ask(&t, &p.text, p.max_tokens, true).await {
                    Ok(r) => {
                        consecutive_errors = 0;
                        run.samples.push(sample_of(p.name, &r, pinned.as_deref()));
                    }
                    Err(e) => {
                        run.errors.push(format!("{}: {e:#}", p.name));
                        consecutive_errors += 1;
                        if consecutive_errors >= 3 {
                            run.errors.push("giving up on this model after three errors in a row".into());
                            break 'perf;
                        }
                    }
                }
            }
        }
        if !self.args.skip_quality && consecutive_errors < 3 {
            self.quality(&t, pinned.as_deref(), &mut run).await;
        }
        run
    }

    async fn quality(&mut self, t: &Target, pinned: Option<&str>, run: &mut TargetRun) {
        let code_ok = !self.args.no_code_exec && python_available();
        let tokens = self.args.quality_tokens.unwrap_or_else(|| self.args.suite.quality_tokens());
        let budget = match self.args.quality_budget {
            Some(0) => None,
            Some(s) => Some(Duration::from_secs(s)),
            None => (self.args.suite == Suite::Hard).then_some(Duration::from_secs(480)),
        };
        let started = std::time::Instant::now();
        for check in quality_suite(self.args.suite, self.args.context) {
            if budget.is_some_and(|b| started.elapsed() > b) {
                run.quality.push(QualityResult {
                    check: check.name,
                    category: check.category,
                    node: pinned.unwrap_or("unknown").to_string(),
                    passed: None,
                    got: Some("not run: over the quality time budget".into()),
                });
                continue;
            }
            run.requests += 1;
            match self.ask(t, &check.prompt, tokens, false).await {
                Ok(r) => {
                    let node = pinned.map_or_else(|| r.served_by.clone().unwrap_or_else(|| "unknown".into()), str::to_string);
                    let passed = judge(&check, &r.text, code_ok);
                    let got = (passed == Some(false)).then(|| {
                        let t = score::strip_think(&r.text);
                        let t = if t.is_empty() { "(nothing after the reasoning)".to_string() } else { t };
                        t.chars().take(160).collect::<String>().replace('\n', " ")
                    });
                    run.quality.push(QualityResult { check: check.name, category: check.category, node, passed, got });
                }
                Err(e) => {
                    run.errors.push(format!("{}: {e:#}", check.name));
                    let node = pinned.unwrap_or("unknown").to_string();
                    run.quality.push(QualityResult {
                        check: check.name,
                        category: check.category,
                        node,
                        passed: Some(false),
                        got: Some(format!("error: {e:#}")),
                    });
                }
            }
        }
    }

    async fn embedding(&mut self, t: &Target, run: &mut TargetRun) {
        let inputs: Vec<String> =
            ["The quick brown fox.", "A lighthouse stands on the rock.", "Prime numbers are only divisible by one and themselves."]
                .iter()
                .map(|s| s.to_string())
                .collect();
        let mut times = Vec::new();
        for i in 0..self.args.warmup + self.args.repeats {
            run.requests += 1;
            match tokio::time::timeout(Duration::from_secs(self.args.timeout), self.backend.embed(t, &inputs)).await {
                Ok(Ok((dims, took))) => {
                    if i >= self.args.warmup {
                        times.push(took.as_secs_f64() * 1000.0);
                        run.embed = Some((dims, 0.0));
                    }
                }
                Ok(Err(e)) => run.errors.push(format!("embedding: {e:#}")),
                Err(_) => run.errors.push("embedding: timed out".into()),
            }
        }
        if let (Some((dims, _)), Some(m)) = (run.embed, median(&times)) {
            run.embed = Some((dims, m));
        }
    }
}

// ---- running ----

/// Days since 1970-01-01 to (year, month, day), proleptic Gregorian.
fn civil(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m, d)
}

/// `20261121T133000Z` for a Unix time.
fn stamp(secs: u64) -> String {
    let (y, m, d) = civil((secs / 86_400) as i64);
    let r = secs % 86_400;
    format!("{y:04}{m:02}{d:02}T{:02}{:02}{:02}Z", r / 3600, r % 3600 / 60, r % 60)
}

fn display(p: &Path) -> String {
    p.display().to_string()
}

/// (node, what happened); the node is empty for a note that is not about one.
type Notes = Vec<(String, String)>;

/// The targets `args` selects, plus notes about what was asked for and is not there.
fn select(l: &Listing, args: &BenchArgs, pinned: bool) -> Result<(Vec<Target>, Notes)> {
    let mut notes = Vec::new();
    let mut peers: Option<Vec<String>> = None;
    if !is_all(&args.node) {
        if !pinned {
            bail!("--node needs `--via weirdnet`: over the bridge the pool picks the node");
        }
        let mut v = Vec::new();
        for q in &args.node {
            v.extend(resolve_node(q, &l.nodes)?);
        }
        peers = Some(v);
    }
    let mut targets: Vec<Target> = l
        .targets
        .iter()
        .filter(|t| peers.as_ref().is_none_or(|p| t.peer.as_ref().is_some_and(|x| p.contains(x))))
        .filter(|t| is_all(&args.model) || args.model.iter().any(|m| model_matches(m, &t.model)))
        .cloned()
        .collect();
    if let Some(p) = &peers {
        for (_, name) in l.nodes.iter().filter(|(peer, _)| p.contains(peer)) {
            if let Some((_, why)) = l.unreachable.iter().find(|(n, _)| n == name) {
                notes.push((name.clone(), format!("skipped, node unreachable: {why}")));
            }
        }
    } else {
        for (name, why) in &l.unreachable {
            notes.push((name.clone(), format!("skipped, node unreachable: {why}")));
        }
    }
    if !is_all(&args.model) {
        for m in &args.model {
            if !targets.iter().any(|t| model_matches(m, &t.model)) {
                notes.push((String::new(), format!("model {m}: not served on the selected nodes")));
            }
        }
    }
    targets.sort_by(|a, b| (&a.node, &a.model).cmp(&(&b.node, &b.model)));
    Ok((targets, notes))
}

fn confirm_cold(cold: &[&Target], args: &BenchArgs, err: &mut dyn Write) -> Result<()> {
    if cold.is_empty() {
        return Ok(());
    }
    writeln!(err, "WARNING: --cold will load models that are not loaded, one at a time:")?;
    for t in cold {
        writeln!(err, "  {} on {}", t.model, t.node)?;
    }
    writeln!(err, "Loading can evict models the node is serving right now (same swap group or memory), and they reload on next use.")?;
    if args.yes {
        return Ok(());
    }
    if !std::io::stdin().is_terminal() {
        bail!("refusing to load models without a person to ask; pass --yes to go ahead");
    }
    write!(err, "Continue? [y/N] ")?;
    err.flush()?;
    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    if !matches!(line.trim().to_lowercase().as_str(), "y" | "yes") {
        bail!("cancelled");
    }
    Ok(())
}

/// Runs the benchmark, writes `bench-<time>.json`, `.md` and `.csv` into the output directory and
/// prints the report. Progress goes to `err`.
pub async fn run(backend: &dyn Backend, args: &BenchArgs, out: &mut dyn Write, err: &mut dyn Write) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let listing = backend.listing().await?;
    let (targets, mut notes) = select(&listing, args, backend.pinned())?;
    if targets.is_empty() {
        bail!(
            "nothing to benchmark: no model matches (see `ymirctl bench list`){}",
            if notes.is_empty() {
                String::new()
            } else {
                format!("; {}", notes.iter().map(|(n, w)| if n.is_empty() { w.clone() } else { format!("{n}: {w}") }).collect::<Vec<_>>().join("; "))
            }
        );
    }
    if args.cold {
        let cold: Vec<&Target> = targets.iter().filter(|t| t.state == "loadable").collect();
        confirm_cold(&cold, args, err)?;
    }
    let t0 = std::time::Instant::now();
    let started = stamp(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs());
    let mut runs = Vec::new();
    let mut bench = Bench { backend, args, err };
    for t in &targets {
        let _ = writeln!(bench.err, "benchmarking {} on {}", t.model, t.node);
        let r = bench.run_target(t).await;
        let _ = writeln!(
            bench.err,
            "  {}",
            r.skipped.as_deref().map_or_else(|| format!("{} samples, {} errors", r.samples.len(), r.errors.len()), |w| format!("skipped: {w}"))
        );
        runs.push(r);
    }
    let meta = report::Meta {
        via: backend.via(),
        started: started.clone(),
        seconds: t0.elapsed().as_secs_f64(),
        roster: listing.nodes.clone(),
        notes: std::mem::take(&mut notes),
    };
    let report = report::build(args, &runs, &meta);
    std::fs::create_dir_all(&args.out_dir).with_context(|| format!("creating {}", args.out_dir.display()))?;
    let jp = args.out_dir.join(format!("bench-{started}.json"));
    let mp = args.out_dir.join(format!("bench-{started}.md"));
    let cp = args.out_dir.join(format!("bench-{started}.csv"));
    std::fs::write(&jp, serde_json::to_string_pretty(&report.json)? + "\n")?;
    std::fs::write(&mp, &report.markdown)?;
    std::fs::write(&cp, &report.csv)?;
    write!(out, "{}", report.markdown)?;
    writeln!(out, "\nSaved {}, {} and {}", display(&jp), display(&mp), display(&cp))?;
    Ok((jp, mp, cp))
}

/// `bench compare`.
pub fn compare(args: &CompareArgs, out: &mut dyn Write) -> Result<()> {
    compare::run(args, out)
}

/// `bench list`: the nodes and models a benchmark can run on.
pub async fn list(backend: &dyn Backend, real_names: bool, out: &mut dyn Write) -> Result<()> {
    let l = backend.listing().await?;
    let mut namer = report::Namer::new(real_names, &l.nodes);
    writeln!(out, "| node | role | model | kind | state | replicas | context |")?;
    writeln!(out, "|---|---|---|---|---|---|---|")?;
    let mut ts = l.targets.clone();
    ts.sort_by(|a, b| (&a.node, &a.model).cmp(&(&b.node, &b.model)));
    for t in &ts {
        writeln!(
            out,
            "| {} | {} | {} | {} | {} | {} | {} |",
            namer.name(&t.node),
            t.role,
            t.model,
            t.purpose,
            t.state,
            t.replicas,
            t.context.map_or("-".into(), |c| c.to_string())
        )?;
    }
    for (name, why) in &l.unreachable {
        writeln!(out, "| {} | - | - | - | unreachable: {} | - | - |", namer.name(name), report::scrub(why, real_names))?;
    }
    if !backend.pinned() {
        writeln!(out, "\nOver the bridge the pool picks the node, so models are listed once; use `--via weirdnet` to see each node.")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
