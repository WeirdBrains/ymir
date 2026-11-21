//! `ymirctl bench`: compare every model on the pool.
//!
//! The benchmark is an ordinary client of the loopback bridge, so every request takes the normal
//! pool route. For each model it measures time to first token, decode and prompt-processing speed
//! and total latency over a fixed prompt suite (one warm-up, then N repeats, median and p90), then
//! scores a handful of deterministic quality checks at temperature 0. Models run one at a time.
//!
//! The bridge cannot pin a node, so the node that answered each request (`served_by`) is recorded
//! and a model served by several nodes gets one row per node. Every request carries its own seed and
//! a nonce, so neither the answer cache nor a node's prefix cache can flatter the numbers.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde_json::{Map, Value, json};
use ymir_core::client::{BridgeClient, TimedChat};

#[derive(clap::Args, Debug, Clone)]
pub struct BenchArgs {
    /// Only these models (repeatable). Default: every model on the pool.
    #[arg(long, short)]
    pub model: Vec<String>,
    /// Timed runs of each performance prompt.
    #[arg(long, default_value_t = 3)]
    pub repeats: usize,
    /// Untimed runs before the timed ones (the first also loads a cold model).
    #[arg(long, default_value_t = 1)]
    pub warmup: usize,
    /// Also benchmark models that are not loaded, loading them one at a time. Off by default so a
    /// benchmark never evicts what the pool is serving.
    #[arg(long)]
    pub include_cold: bool,
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
    /// Where the timestamped `.json` and `.md` results go.
    #[arg(long, default_value = "bench-results")]
    pub out_dir: PathBuf,
    /// Name nodes as they are. By default they are labelled node-a, node-b, ... so results can be
    /// shared.
    #[arg(long)]
    pub real_names: bool,
}

/// A performance prompt: a name, the text and the most tokens to generate.
struct PerfPrompt {
    name: &'static str,
    text: String,
    max_tokens: u64,
}

fn perf_suite() -> Vec<PerfPrompt> {
    let facts: String =
        (1..=60).map(|i| format!("Fact {i}: the item numbered {i} weighs {} grams and is kept in box {}.\n", i * 7 + 3, i % 9 + 1)).collect();
    vec![
        PerfPrompt { name: "short", text: "Explain in two sentences why the sky is blue.".into(), max_tokens: 128 },
        PerfPrompt { name: "decode", text: "Write a story of about 200 words about a lighthouse keeper and a stray cat.".into(), max_tokens: 256 },
        PerfPrompt {
            name: "long_prompt",
            text: format!("Read these facts.\n{facts}\nIn one short sentence, say how heavy item 17 is."),
            max_tokens: 48,
        },
    ]
}

/// A quality check: a prompt and how its answer is judged.
struct Check {
    name: &'static str,
    prompt: &'static str,
    kind: Kind,
}

enum Kind {
    /// The text, normalized, equals one of these.
    Exact(&'static [&'static str]),
    /// The last whole number in the text is this one.
    Number(i64),
    /// A JSON object with these fields (name "Ada", age 36, tags ["x", "y"]).
    Json,
    /// Python code that, followed by these lines, prints OK.
    Code(&'static str),
}

fn quality_suite() -> Vec<Check> {
    vec![
        Check { name: "arithmetic_1", prompt: "Compute 17 * 23 + 5. Reply with only the final number.", kind: Kind::Number(396) },
        Check { name: "arithmetic_2", prompt: "What is 12345 + 6789? Reply with only the number.", kind: Kind::Number(19134) },
        Check {
            name: "json_format",
            prompt: "Reply with only a JSON object, no prose and no code fences, with exactly these keys: \"name\" set to the string \"Ada\", \"age\" set to the number 36, and \"tags\" set to an array of the two strings \"x\" and \"y\".",
            kind: Kind::Json,
        },
        Check {
            name: "code_fib",
            prompt: "Write a Python function fib(n) that returns the n-th Fibonacci number, with fib(0) = 0 and fib(1) = 1. Reply with only the code.",
            kind: Kind::Code("assert [fib(i) for i in range(10)] == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]\nassert fib(30) == 832040\nprint('OK')\n"),
        },
        Check {
            name: "code_palindrome",
            prompt: "Write a Python function is_palindrome(s) that returns True when s reads the same backwards, ignoring case and anything that is not a letter or digit. Reply with only the code.",
            kind: Kind::Code(
                "assert is_palindrome('A man, a plan, a canal: Panama') is True\nassert is_palindrome('hello') is False\nassert is_palindrome('') is True\nprint('OK')\n",
            ),
        },
        Check { name: "fact_capital", prompt: "What is the capital of Australia? Reply with one word.", kind: Kind::Exact(&["canberra"]) },
        Check { name: "fact_symbol", prompt: "What is the chemical symbol for gold? Reply with only the symbol.", kind: Kind::Exact(&["au"]) },
        Check { name: "fact_leap", prompt: "How many days are in a leap year? Reply with only the number.", kind: Kind::Number(366) },
    ]
}

// ---- scoring ----

/// The answer without a reasoning block.
fn strip_think(s: &str) -> String {
    let mut out = s.to_string();
    while let Some(a) = out.find("<think>") {
        match out[a..].find("</think>") {
            Some(b) => out.replace_range(a..a + b + "</think>".len(), ""),
            None => out.truncate(a),
        }
    }
    if let Some(b) = out.find("</think>") {
        out = out[b + "</think>".len()..].to_string();
    }
    out.trim().to_string()
}

fn normalize(s: &str) -> String {
    strip_think(s).trim_matches(|c: char| c.is_whitespace() || ".,!?\"'`*_:;".contains(c)).to_lowercase()
}

fn last_number(s: &str) -> Option<i64> {
    let s = strip_think(s).replace(',', "");
    let mut last = None;
    let mut cur = String::new();
    for c in s.chars().chain(std::iter::once(' ')) {
        if c.is_ascii_digit() || (c == '-' && cur.is_empty()) {
            cur.push(c);
        } else {
            if let Ok(n) = cur.parse::<i64>() {
                last = Some(n);
            }
            cur.clear();
        }
    }
    last
}

/// The first fenced block, else the whole text.
fn extract_block(s: &str) -> String {
    let s = strip_think(s);
    if let Some(a) = s.find("```") {
        let rest = &s[a + 3..];
        let rest = rest.split_once('\n').map_or(rest, |(_, r)| r);
        return rest.split("```").next().unwrap_or(rest).trim().to_string();
    }
    s
}

fn json_ok(s: &str) -> bool {
    let Ok(v) = serde_json::from_str::<Value>(&extract_block(s)) else { return false };
    v["name"] == "Ada" && v["age"].as_f64() == Some(36.0) && v["tags"] == json!(["x", "y"]) && v.as_object().is_some_and(|o| o.len() == 3)
}

/// Whether network isolation is available for running model code.
fn can_unshare() -> bool {
    Command::new("unshare").args(["--net", "--map-root-user", "true"]).stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok_and(|s| s.success())
}

/// Runs `code` plus `tests` with python3 in an empty temp directory with an empty environment,
/// CPU, memory and file-size limits, a wall-clock limit and, where the machine allows it, no
/// network. This limits accidents; it is not a security boundary.
fn run_python(code: &str, tests: &str) -> Result<bool> {
    let dir = tempfile::tempdir()?;
    let file = dir.path().join("t.py");
    std::fs::write(&file, format!("{code}\n\n{tests}"))?;
    let run = "ulimit -t 5 -v 2000000 -f 1024 -u 64 2>/dev/null; exec python3 -B -I t.py";
    let mut cmd = if can_unshare() {
        let mut c = Command::new("unshare");
        c.args(["--net", "--map-root-user", "sh", "-c", run]);
        c
    } else {
        let mut c = Command::new("sh");
        c.args(["-c", run]);
        c
    };
    let mut child = cmd
        .current_dir(dir.path())
        .env_clear()
        .env("PATH", "/usr/local/bin:/usr/bin:/bin")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("starting python3")?;
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        if child.try_wait()?.is_some() {
            break;
        }
        if std::time::Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Ok(false);
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    let out = child.wait_with_output()?;
    Ok(out.status.success() && String::from_utf8_lossy(&out.stdout).trim() == "OK")
}

fn python_available() -> bool {
    Command::new("python3").arg("--version").stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok_and(|s| s.success())
}

/// `Some(passed)`, or `None` when the check could not run.
fn judge(check: &Check, answer: &str, code_ok: bool) -> Option<bool> {
    match &check.kind {
        Kind::Exact(ok) => Some(ok.contains(&normalize(answer).as_str())),
        Kind::Number(n) => Some(last_number(answer) == Some(*n)),
        Kind::Json => Some(json_ok(answer)),
        Kind::Code(tests) => code_ok.then(|| run_python(&extract_block(answer), tests).unwrap_or(false)),
    }
}

// ---- statistics ----

fn median(v: &[f64]) -> Option<f64> {
    percentile(v, 50.0)
}

/// Nearest-rank percentile.
fn percentile(v: &[f64], p: f64) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let rank = ((p / 100.0) * s.len() as f64).ceil().max(1.0) as usize;
    Some(s[rank.min(s.len()) - 1])
}

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

// ---- samples ----

#[derive(Debug, Clone)]
struct Sample {
    prompt: &'static str,
    node: String,
    ttft_ms: f64,
    total_ms: f64,
    decode_tps: Option<f64>,
    pp_tps: Option<f64>,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
}

fn sample_of(prompt: &'static str, r: &TimedChat) -> Sample {
    let total = r.total.as_secs_f64();
    let ttft = r.ttft.map_or(total, |t| t.as_secs_f64());
    let tokens = r.completion_tokens.or_else(|| (!r.text.is_empty()).then(|| (r.text.len() as u64).div_ceil(4)));
    let decode_tps = tokens.filter(|n| *n > 1 && total > ttft).map(|n| (n - 1) as f64 / (total - ttft));
    let fresh = r.prompt_tokens.map(|p| p.saturating_sub(r.cached_tokens.unwrap_or(0)));
    let pp_tps = fresh.filter(|n| *n > 0 && ttft > 0.0).map(|n| n as f64 / ttft);
    Sample {
        prompt,
        node: r.served_by.clone().unwrap_or_else(|| "unknown".into()),
        ttft_ms: ttft * 1000.0,
        total_ms: total * 1000.0,
        decode_tps,
        pp_tps,
        prompt_tokens: r.prompt_tokens,
        completion_tokens: r.completion_tokens,
    }
}

#[derive(Debug, Default)]
struct ModelRun {
    id: String,
    purpose: String,
    state_at_start: String,
    replicas: u64,
    skipped: Option<String>,
    cold_load_ms: Option<f64>,
    samples: Vec<Sample>,
    errors: Vec<String>,
    requests: usize,
    /// (check, node, passed or None when it could not run)
    quality: Vec<(&'static str, String, Option<bool>)>,
    embed: Option<(usize, f64)>,
}

fn nonce() -> String {
    let n = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos());
    format!("{n:x}")
}

fn seed() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64 & 0x7fff_ffff_ffff)
}

fn request_extra(max_tokens: u64) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("temperature".into(), Value::from(0.0));
    m.insert("max_tokens".into(), Value::from(max_tokens));
    m.insert("seed".into(), Value::from(seed()));
    m
}

struct Bench<'a> {
    client: &'a BridgeClient,
    args: &'a BenchArgs,
    err: &'a mut dyn Write,
}

impl Bench<'_> {
    async fn ask(&self, model: &str, prompt: &str, max_tokens: u64, fresh_prefix: bool) -> Result<TimedChat> {
        let text = if fresh_prefix { format!("[run {}]\n{prompt}", nonce()) } else { prompt.to_string() };
        let messages = vec![json!({"role": "user", "content": text})];
        let call = self.client.chat_timed(model, messages, request_extra(max_tokens));
        let r = tokio::time::timeout(Duration::from_secs(self.args.timeout), call)
            .await
            .map_err(|_| anyhow::anyhow!("timed out after {}s", self.args.timeout))??;
        if let Some(s) = &r.standin {
            anyhow::bail!("the stand-in {s} answered instead");
        }
        if r.served_by.as_deref() == Some("cache") {
            anyhow::bail!("the answer cache answered");
        }
        Ok(r)
    }

    /// Waits while the model is busy or loading; `Err(reason)` to skip it.
    async fn settle(&mut self, id: &str) -> std::result::Result<(String, u64, String), String> {
        let mut last = String::new();
        for attempt in 0..=self.args.busy_retries {
            let v = self.client.models().await.map_err(|e| format!("could not list models: {e}"))?;
            let Some(m) = v["data"].as_array().and_then(|d| d.iter().find(|m| m["id"] == id)) else {
                return Err("no longer listed".into());
            };
            let state = m["weirdnet"]["state"].as_str().unwrap_or("unknown").to_string();
            let replicas = m["weirdnet"]["replicas"].as_u64().unwrap_or(0);
            let purpose = m["weirdnet"]["purpose"].as_str().unwrap_or("chat").to_string();
            match state.as_str() {
                "warm" => return Ok((state, replicas, purpose)),
                "loadable" if self.args.include_cold => return Ok((state, replicas, purpose)),
                "loadable" => return Err("not loaded (use --include-cold to load it)".into()),
                _ => {
                    last = state;
                    if attempt < self.args.busy_retries {
                        let _ = writeln!(self.err, "  {id} is {last}; waiting {}s ({}/{})", self.args.busy_wait, attempt + 1, self.args.busy_retries);
                        tokio::time::sleep(Duration::from_secs(self.args.busy_wait)).await;
                    }
                }
            }
        }
        Err(format!("still {last} after {} checks", self.args.busy_retries))
    }

    async fn run_model(&mut self, id: &str, purpose: &str) -> ModelRun {
        let mut run = ModelRun { id: id.into(), purpose: purpose.into(), ..Default::default() };
        match self.settle(id).await {
            Ok((state, replicas, p)) => {
                run.state_at_start = state;
                run.replicas = replicas;
                run.purpose = p;
            }
            Err(why) => {
                run.skipped = Some(why);
                return run;
            }
        }
        if run.purpose == "embedding" {
            self.embedding(&mut run).await;
            return run;
        }
        let cold = run.state_at_start == "loadable";
        let mut consecutive_errors = 0;
        // Warm-up; the first one of a cold model times the load.
        for i in 0..self.args.warmup.max(usize::from(cold)) {
            let t = std::time::Instant::now();
            run.requests += 1;
            match self.ask(id, "Say hello in five words.", 32, true).await {
                Ok(_) if i == 0 && cold => run.cold_load_ms = Some(t.elapsed().as_secs_f64() * 1000.0),
                Ok(_) => {}
                Err(e) => {
                    run.errors.push(format!("warm-up: {e:#}"));
                    consecutive_errors += 1;
                }
            }
        }
        'perf: for p in perf_suite() {
            for _ in 0..self.args.repeats {
                run.requests += 1;
                match self.ask(id, &p.text, p.max_tokens, true).await {
                    Ok(r) => {
                        consecutive_errors = 0;
                        run.samples.push(sample_of(p.name, &r));
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
            self.quality(&mut run).await;
        }
        run
    }

    async fn quality(&mut self, run: &mut ModelRun) {
        let code_ok = !self.args.no_code_exec && python_available();
        for check in quality_suite() {
            run.requests += 1;
            match self.ask(&run.id, check.prompt, 2048, false).await {
                Ok(r) => {
                    let node = r.served_by.clone().unwrap_or_else(|| "unknown".into());
                    run.quality.push((check.name, node, judge(&check, &r.text, code_ok)));
                }
                Err(e) => {
                    run.errors.push(format!("{}: {e:#}", check.name));
                    run.quality.push((check.name, "unknown".into(), Some(false)));
                }
            }
        }
    }

    async fn embedding(&mut self, run: &mut ModelRun) {
        let inputs: Vec<String> =
            ["The quick brown fox.", "A lighthouse stands on the rock.", "Prime numbers are only divisible by one and themselves."]
                .iter()
                .map(|s| s.to_string())
                .collect();
        let mut times = Vec::new();
        for i in 0..self.args.warmup + self.args.repeats {
            run.requests += 1;
            match tokio::time::timeout(Duration::from_secs(self.args.timeout), self.client.embed_timed(&run.id, &inputs)).await {
                Ok(Ok((dims, t))) => {
                    if i >= self.args.warmup {
                        times.push(t.as_secs_f64() * 1000.0);
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

// ---- report ----

/// Labels for node names: node-a, node-b, ... in sorted order, unless real names are wanted.
fn labels(runs: &[ModelRun], real: bool) -> BTreeMap<String, String> {
    let mut names: Vec<String> =
        runs.iter().flat_map(|r| r.samples.iter().map(|s| s.node.clone()).chain(r.quality.iter().map(|q| q.1.clone()))).collect();
    names.sort();
    names.dedup();
    let mut map = BTreeMap::new();
    let mut n = 0u8;
    for name in names {
        let label = if real || name == "unknown" {
            name.clone()
        } else {
            let l = format!("node-{}", (b'a' + n % 26) as char);
            n += 1;
            l
        };
        map.insert(name, label);
    }
    map
}

fn fmt_opt(x: Option<f64>, digits: usize) -> String {
    x.map_or("-".into(), |v| format!("{v:.digits$}"))
}

pub struct Report {
    pub json: Value,
    pub markdown: String,
}

fn stats(values: &[f64]) -> Value {
    json!({"median": median(values).map(round1), "p90": percentile(values, 90.0).map(round1), "n": values.len()})
}

fn build_report(args: &BenchArgs, runs: &[ModelRun], started: &str, secs: f64) -> Report {
    let label = labels(runs, args.real_names);
    let name = |n: &str| label.get(n).cloned().unwrap_or_else(|| n.to_string());
    let mut models = Vec::new();
    let mut rows = Vec::new();
    let mut skipped = Vec::new();
    for r in runs {
        if let Some(why) = &r.skipped {
            skipped.push(format!("{}: {why}", r.id));
            models.push(json!({"model": r.id, "purpose": r.purpose, "skipped": why}));
            continue;
        }
        let q_pass = r.quality.iter().filter(|q| q.2 == Some(true)).count();
        let q_ran = r.quality.iter().filter(|q| q.2.is_some()).count();
        let quality = (q_ran > 0).then(|| json!({"passed": q_pass, "ran": q_ran, "checks": r.quality.iter().map(|(c, n, p)| json!({"check": c, "node": name(n), "passed": p})).collect::<Vec<_>>()}));
        let mut by_node: BTreeMap<String, Vec<&Sample>> = BTreeMap::new();
        for s in &r.samples {
            by_node.entry(name(&s.node)).or_default().push(s);
        }
        let mut node_json = Vec::new();
        for (node, ss) in &by_node {
            let col = |f: &dyn Fn(&Sample) -> Option<f64>| ss.iter().filter_map(|s| f(s)).collect::<Vec<f64>>();
            let ttft = col(&|s| Some(s.ttft_ms));
            let total = col(&|s| Some(s.total_ms));
            let dec = col(&|s| s.decode_tps);
            let pp = col(&|s| s.pp_tps);
            // Decode speed comes from the long-form prompt; prompt processing from the long prompt.
            let dec_long = ss.iter().filter(|s| s.prompt == "decode").filter_map(|s| s.decode_tps).collect::<Vec<_>>();
            let pp_long = ss.iter().filter(|s| s.prompt == "long_prompt").filter_map(|s| s.pp_tps).collect::<Vec<_>>();
            let ttft_short = ss.iter().filter(|s| s.prompt == "short").map(|s| s.ttft_ms).collect::<Vec<_>>();
            node_json.push(json!({
                "node": node, "samples": ss.len(),
                "ttft_ms_short": stats(&ttft_short), "ttft_ms_all": stats(&ttft), "total_ms_all": stats(&total),
                "decode_tps_all": stats(&dec), "decode_tps_long": stats(&dec_long),
                "prompt_tps_all": stats(&pp), "prompt_tps_long": stats(&pp_long),
                "by_prompt": perf_suite().iter().map(|p| {
                    let of = |f: &dyn Fn(&Sample) -> Option<f64>| ss.iter().filter(|s| s.prompt == p.name).filter_map(|s| f(s)).collect::<Vec<f64>>();
                    json!({"prompt": p.name, "ttft_ms": stats(&of(&|s| Some(s.ttft_ms))), "total_ms": stats(&of(&|s| Some(s.total_ms))),
                           "decode_tps": stats(&of(&|s| s.decode_tps)), "prompt_tps": stats(&of(&|s| s.pp_tps)),
                           "prompt_tokens": ss.iter().find(|s| s.prompt == p.name).and_then(|s| s.prompt_tokens),
                           "completion_tokens": ss.iter().find(|s| s.prompt == p.name).and_then(|s| s.completion_tokens)})
                }).collect::<Vec<_>>(),
            }));
            rows.push(format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                r.id,
                node,
                fmt_opt(median(&ttft_short), 0),
                fmt_opt(percentile(&ttft_short, 90.0), 0),
                fmt_opt(median(&dec_long), 1),
                fmt_opt(median(&pp_long), 0),
                fmt_opt(median(&col(&|s| Some(s.total_ms))), 0),
                quality.as_ref().map_or("-".into(), |_| format!("{q_pass}/{q_ran}")),
                r.errors.len(),
            ));
        }
        if by_node.is_empty() {
            let detail = match r.embed {
                Some((dims, ms)) => format!("embedding, {dims} dims, {ms:.0} ms per 3 inputs (median)"),
                None => "no successful request".into(),
            };
            rows.push(format!("| {} | - | - | - | - | - | - | - | {} ({detail}) |", r.id, r.errors.len()));
        }
        models.push(json!({
            "model": r.id, "purpose": r.purpose, "state_at_start": r.state_at_start, "replicas": r.replicas,
            "cold_load_ms": r.cold_load_ms.map(round1), "requests": r.requests, "errors": r.errors,
            "nodes": node_json, "quality": quality,
            "embedding": r.embed.map(|(d, ms)| json!({"dims": d, "ms_median": round1(ms)})),
        }));
    }
    let mut md = String::new();
    md.push_str(&format!("# Pool benchmark, {started}\n\n"));
    md.push_str(&format!("Warm-up {}, repeats {}, temperature 0, one model at a time; took {:.0} s.\n\n", args.warmup, args.repeats, secs));
    md.push_str(
        "| model | node | TTFT ms (median) | TTFT ms (p90) | decode tok/s | prompt tok/s | total ms (all prompts, median) | quality | errors |\n",
    );
    md.push_str("|---|---|---|---|---|---|---|---|---|\n");
    for r in &rows {
        md.push_str(r);
        md.push('\n');
    }
    md.push_str("\nTTFT is for the short prompt. Decode tok/s is from the 200-word story (tokens after the first, divided by the time after it). Prompt tok/s is from the long prompt (tokens not served from a prefix cache, divided by TTFT, so it includes queueing and the network). Quality is checks passed over checks run (arithmetic, JSON format, executed code, exact-match facts).\n");
    if !skipped.is_empty() {
        md.push_str("\nSkipped:\n\n");
        for s in &skipped {
            md.push_str(&format!("- {s}\n"));
        }
    }
    let json = json!({"bench": "ymirctl bench", "version": 1, "started": started, "seconds": round1(secs),
        "settings": {"warmup": args.warmup, "repeats": args.repeats, "include_cold": args.include_cold, "temperature": 0},
        "models": models, "skipped": skipped});
    Report { json, markdown: md }
}

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

/// Runs the benchmark, writes `bench-<time>.json` and `.md` into the output directory and prints
/// the table. Progress goes to `err`.
pub async fn run(client: &BridgeClient, args: &BenchArgs, out: &mut dyn Write, err: &mut dyn Write) -> Result<(PathBuf, PathBuf)> {
    let listing = client.models().await?;
    let mut targets: Vec<(String, String)> = listing["data"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|m| Some((m["id"].as_str()?.to_string(), m["weirdnet"]["purpose"].as_str().unwrap_or("chat").to_string())))
        .filter(|(id, _)| args.model.is_empty() || args.model.contains(id))
        .collect();
    targets.sort();
    for wanted in &args.model {
        if !targets.iter().any(|(id, _)| id == wanted) {
            writeln!(err, "model {wanted} is not on the pool")?;
        }
    }
    let t0 = std::time::Instant::now();
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let started = stamp(now);
    let mut runs = Vec::new();
    let mut bench = Bench { client, args, err };
    for (id, purpose) in &targets {
        let _ = writeln!(bench.err, "benchmarking {id}");
        let r = bench.run_model(id, purpose).await;
        let _ = writeln!(
            bench.err,
            "  {}",
            r.skipped.as_deref().map_or_else(|| format!("{} samples, {} errors", r.samples.len(), r.errors.len()), |w| format!("skipped: {w}"))
        );
        runs.push(r);
    }
    let report = build_report(args, &runs, &started, t0.elapsed().as_secs_f64());
    std::fs::create_dir_all(&args.out_dir).with_context(|| format!("creating {}", args.out_dir.display()))?;
    let jp = args.out_dir.join(format!("bench-{started}.json"));
    let mp = args.out_dir.join(format!("bench-{started}.md"));
    std::fs::write(&jp, serde_json::to_string_pretty(&report.json)? + "\n")?;
    std::fs::write(&mp, &report.markdown)?;
    write!(out, "{}", report.markdown)?;
    writeln!(out, "\nSaved {} and {}", display(&jp), display(&mp))?;
    Ok((jp, mp))
}

fn display(p: &Path) -> String {
    p.display().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_use_nearest_rank() {
        let v = [5.0, 1.0, 3.0, 2.0, 4.0];
        assert_eq!(median(&v), Some(3.0));
        assert_eq!(percentile(&v, 90.0), Some(5.0));
        assert_eq!(percentile(&[7.0], 90.0), Some(7.0));
        assert_eq!(median(&[]), None);
    }

    #[test]
    fn answers_are_judged_strictly_but_ignore_reasoning_and_decoration() {
        assert_eq!(normalize("<think>hmm, Sydney?</think>\n**Canberra.**"), "canberra");
        assert_eq!(normalize("Au"), "au");
        assert_ne!(normalize("The capital is Canberra"), "canberra");
        assert_eq!(last_number("17*23 = 391, plus 5 is 396."), Some(396));
        assert_eq!(last_number("19,134"), Some(19134));
        assert_eq!(last_number("no digits"), None);
        assert!(json_ok("```json\n{\"name\":\"Ada\",\"age\":36,\"tags\":[\"x\",\"y\"]}\n```"));
        assert!(!json_ok("{\"name\":\"Ada\",\"age\":\"36\",\"tags\":[\"x\",\"y\"]}"));
        assert!(!json_ok("Sure! {\"name\":\"Ada\"}"));
        assert_eq!(extract_block("Here:\n```python\ndef f():\n    return 1\n```\nDone"), "def f():\n    return 1");
    }

    #[test]
    fn model_code_is_run_with_a_time_limit() {
        if !python_available() {
            return;
        }
        assert!(run_python("def f():\n    return 2", "assert f() == 2\nprint('OK')").unwrap());
        assert!(!run_python("def f():\n    return 3", "assert f() == 2\nprint('OK')").unwrap());
        assert!(!run_python("while True:\n    pass", "print('OK')").unwrap());
    }

    #[test]
    fn stamps_are_utc() {
        assert_eq!(stamp(0), "19700101T000000Z");
        assert_eq!(stamp(1_795_267_771), "20261121T132931Z");
    }

    #[test]
    fn samples_compute_speeds_from_the_stream() {
        let r = TimedChat {
            text: "x".into(),
            ttft: Some(Duration::from_millis(500)),
            total: Duration::from_millis(2500),
            prompt_tokens: Some(1100),
            cached_tokens: Some(100),
            completion_tokens: Some(41),
            served_by: Some("m".into()),
            ..Default::default()
        };
        let s = sample_of("p", &r);
        assert_eq!(s.decode_tps, Some(20.0));
        assert_eq!(s.pp_tps, Some(2000.0));
        assert_eq!(s.node, "m");
    }
}
