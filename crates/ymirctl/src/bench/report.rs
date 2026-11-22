//! The report: JSON, Markdown (a table per target and side-by-side matrices) and CSV.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Value, json};

use super::score::{median, percentile, round1};
use super::suites::perf_suite;
use super::{BenchArgs, Sample, TargetRun};

/// Node names for output: the real ones, or node-a, node-b, ... assigned in peer-id order over the
/// linked devices (so they stay the same between runs while the roster does), then in first-seen order.
pub struct Namer {
    real: bool,
    map: BTreeMap<String, String>,
    next: usize,
}

fn letter(n: usize) -> String {
    if n < 26 { format!("node-{}", (b'a' + n as u8) as char) } else { format!("node-{n}") }
}

impl Namer {
    pub fn new(real: bool, roster: &[(String, String)]) -> Self {
        let mut n = Self { real, map: BTreeMap::new(), next: 0 };
        let mut sorted = roster.to_vec();
        sorted.sort();
        for (peer, name) in sorted {
            let l = n.name(&name);
            n.map.insert(peer, l.clone());
            n.map.insert(format!("Ymir on {name}"), l);
        }
        n
    }

    pub fn name(&mut self, real: &str) -> String {
        if self.real || real == "unknown" || real == "routed" {
            return real.to_string();
        }
        if let Some(l) = self.map.get(real) {
            return l.clone();
        }
        let l = letter(self.next);
        self.next += 1;
        self.map.insert(real.to_string(), l.clone());
        l
    }
}

/// Hides peer ids and addresses in text that may be shared (error messages carry them).
pub fn scrub(s: &str, real: bool) -> String {
    if real {
        return s.to_string();
    }
    let hidden = |w: &str| {
        let w = w.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '/' && c != '.');
        w.contains("12D3Koo")
            || w.starts_with("/ip")
            || w.starts_with("/dns")
            || (w.split('.').count() == 4
                && w.split('.').all(|p| p.split(':').next().is_some_and(|d| !d.is_empty() && d.chars().all(|c| c.is_ascii_digit()))))
    };
    s.split(' ').map(|w| if hidden(w) { "<hidden>" } else { w }).collect::<Vec<_>>().join(" ")
}

pub struct Meta {
    pub via: &'static str,
    pub started: String,
    pub seconds: f64,
    pub roster: Vec<(String, String)>,
    /// (node, what happened): things asked for that did not run.
    pub notes: Vec<(String, String)>,
}

pub struct Report {
    pub json: Value,
    pub markdown: String,
    pub csv: String,
}

fn fmt_opt(x: Option<f64>, digits: usize) -> String {
    x.map_or("-".into(), |v| format!("{v:.digits$}"))
}

fn stats(values: &[f64]) -> Value {
    json!({"median": median(values).map(round1), "p90": percentile(values, 90.0).map(round1), "n": values.len()})
}

/// One row of the report: a model on a node.
struct Row {
    model: String,
    node: String,
    served_by: Vec<String>,
    purpose: String,
    samples: usize,
    ttft: Vec<f64>,
    decode: Vec<f64>,
    prompt: Vec<f64>,
    total: Vec<f64>,
    q_pass: usize,
    q_ran: usize,
    by_category: BTreeMap<&'static str, (usize, usize)>,
    errors: usize,
    skipped: Option<String>,
    embed: Option<(usize, f64)>,
    json: Value,
}

impl Row {
    fn quality(&self) -> Option<String> {
        (self.q_ran > 0).then(|| format!("{}/{}", self.q_pass, self.q_ran))
    }
}

fn rows(args: &BenchArgs, runs: &[TargetRun], namer: &mut Namer) -> Vec<Row> {
    let mut out = Vec::new();
    for r in runs {
        let node0 = namer.name(&r.node);
        if let Some(why) = &r.skipped {
            out.push(Row {
                model: r.model.clone(),
                node: node0.clone(),
                served_by: vec![],
                purpose: r.purpose.clone(),
                samples: 0,
                ttft: vec![],
                decode: vec![],
                prompt: vec![],
                total: vec![],
                q_pass: 0,
                q_ran: 0,
                by_category: BTreeMap::new(),
                errors: 0,
                skipped: Some(scrub(why, args.real_names)),
                embed: None,
                json: json!({"model": r.model, "node": node0, "purpose": r.purpose, "skipped": scrub(why, args.real_names)}),
            });
            continue;
        }
        // Over the bridge a model may be answered by several nodes: one row per node that answered.
        let mut groups: BTreeMap<String, Vec<&Sample>> = BTreeMap::new();
        for s in &r.samples {
            groups.entry(s.node.clone()).or_default().push(s);
        }
        if groups.is_empty() {
            groups.insert(r.node.clone(), Vec::new());
        }
        for (real_node, ss) in &groups {
            let node = namer.name(real_node);
            let mut served: BTreeSet<String> = ss.iter().map(|s| namer.name(&s.served_by)).collect();
            served.remove(&node);
            let col = |f: &dyn Fn(&Sample) -> Option<f64>| ss.iter().filter_map(|s| f(s)).collect::<Vec<f64>>();
            let of =
                |name: &str, f: &dyn Fn(&Sample) -> Option<f64>| ss.iter().filter(|s| s.prompt == name).filter_map(|s| f(s)).collect::<Vec<f64>>();
            let ttft = of("short", &|s| Some(s.ttft_ms));
            let decode = of("decode", &|s| s.decode_tps);
            let prompt = of("long_prompt", &|s| s.pp_tps);
            let total = col(&|s| Some(s.total_ms));
            let qs: Vec<_> = r.quality.iter().filter(|q| &q.node == real_node || r.quality.iter().all(|x| x.node == "unknown")).collect();
            let q_pass = qs.iter().filter(|q| q.passed == Some(true)).count();
            let q_ran = qs.iter().filter(|q| q.passed.is_some()).count();
            let mut by_category: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
            for q in &qs {
                if let Some(p) = q.passed {
                    let e = by_category.entry(q.category).or_default();
                    e.1 += 1;
                    e.0 += usize::from(p);
                }
            }
            let quality = (q_ran > 0).then(|| {
                json!({"passed": q_pass, "ran": q_ran, "percent": percent(q_pass, q_ran),
                    "by_category": by_category.iter().map(|(k, (p, n))| (k.to_string(), json!({"passed": p, "ran": n, "percent": percent(*p, *n)}))).collect::<Map>(),
                    "checks": qs.iter().map(|q| {
                        let mut c = json!({"check": q.check, "category": q.category, "passed": q.passed});
                        if let Some(g) = &q.got { c["got"] = json!(g); }
                        c
                    }).collect::<Vec<_>>()})
            });
            let by_prompt: Vec<Value> = perf_suite(args.suite, args.context, args.max_tokens)
                .iter()
                .map(|p| {
                    json!({"prompt": p.name, "ttft_ms": stats(&of(p.name, &|s| Some(s.ttft_ms))), "total_ms": stats(&of(p.name, &|s| Some(s.total_ms))),
                           "decode_tps": stats(&of(p.name, &|s| s.decode_tps)), "prompt_tps": stats(&of(p.name, &|s| s.pp_tps)),
                           "prompt_tokens": ss.iter().find(|s| s.prompt == p.name).and_then(|s| s.prompt_tokens),
                           "completion_tokens": ss.iter().find(|s| s.prompt == p.name).and_then(|s| s.completion_tokens)})
                })
                .collect();
            let json = json!({
                "model": r.model, "node": node, "role": r.role, "served_by": served, "purpose": r.purpose, "state_at_start": r.state_at_start,
                "replicas": r.replicas, "context": r.context, "cold_load_ms": r.cold_load_ms.map(round1), "requests": r.requests,
                "samples": ss.len(), "ttft_ms": stats(&ttft), "decode_tps": stats(&decode), "prompt_tps": stats(&prompt),
                "total_ms": stats(&total), "by_prompt": by_prompt, "quality": quality, "errors": r.errors.iter().map(|e| scrub(e, args.real_names)).collect::<Vec<_>>(),
                "embedding": r.embed.map(|(d, ms)| json!({"dims": d, "ms_median": round1(ms)})),
            });
            out.push(Row {
                model: r.model.clone(),
                node,
                served_by: served.into_iter().collect(),
                purpose: r.purpose.clone(),
                samples: ss.len(),
                ttft,
                decode,
                prompt,
                total,
                q_pass,
                q_ran,
                by_category,
                errors: r.errors.len(),
                skipped: None,
                embed: r.embed,
                json,
            });
        }
    }
    out.sort_by(|a, b| (&a.node, &a.model).cmp(&(&b.node, &b.model)));
    out
}

type Map = serde_json::Map<String, Value>;

fn matrix(md: &mut String, title: &str, rows: &[Row], cell: &dyn Fn(&Row) -> String) {
    let nodes: BTreeSet<&str> = rows.iter().filter(|r| r.skipped.is_none()).map(|r| r.node.as_str()).collect();
    let models: BTreeSet<&str> = rows.iter().filter(|r| r.skipped.is_none()).map(|r| r.model.as_str()).collect();
    md.push_str(&format!("\n**{title}**\n\n| model |"));
    for n in &nodes {
        md.push_str(&format!(" {n} |"));
    }
    md.push_str("\n|---|");
    for _ in &nodes {
        md.push_str("---|");
    }
    md.push('\n');
    for m in &models {
        md.push_str(&format!("| {m} |"));
        for n in &nodes {
            let c = rows.iter().find(|r| r.model == *m && r.node == *n && r.skipped.is_none()).map_or("-".into(), cell);
            md.push_str(&format!(" {c} |"));
        }
        md.push('\n');
    }
}

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n']) { format!("\"{}\"", s.replace('"', "\"\"")) } else { s.to_string() }
}

pub fn build(args: &BenchArgs, runs: &[TargetRun], meta: &Meta) -> Report {
    let mut namer = Namer::new(args.real_names, &meta.roster);
    let rows = rows(args, runs, &mut namer);
    let mut md = String::new();
    md.push_str(&format!("# Pool benchmark, {}\n\n", meta.started));
    md.push_str(&format!(
        "Suite {}, via {}; warm-up {}, repeats {}, temperature 0, one request at a time; took {:.0} s.\n\n",
        args.suite.name(),
        meta.via,
        args.warmup,
        args.repeats,
        meta.seconds
    ));
    md.push_str(
        "| model | node | TTFT ms (median) | TTFT ms (p90) | decode tok/s | prompt tok/s | total ms (all prompts, median) | quality | errors |\n",
    );
    md.push_str("|---|---|---|---|---|---|---|---|---|\n");
    for r in rows.iter().filter(|r| r.skipped.is_none()) {
        if r.samples == 0 {
            let detail = match r.embed {
                Some((dims, ms)) => format!("embedding, {dims} dims, {ms:.0} ms per 3 inputs (median)"),
                None => "no successful request".into(),
            };
            md.push_str(&format!("| {} | {} | - | - | - | - | - | - | {} ({detail}) |\n", r.model, r.node, r.errors));
            continue;
        }
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            r.model,
            r.node,
            fmt_opt(median(&r.ttft), 0),
            fmt_opt(percentile(&r.ttft, 90.0), 0),
            fmt_opt(median(&r.decode), 1),
            fmt_opt(median(&r.prompt), 0),
            fmt_opt(median(&r.total), 0),
            r.quality().unwrap_or_else(|| "-".into()),
            r.errors,
        ));
    }
    let measured: Vec<&Row> = rows.iter().filter(|r| r.skipped.is_none() && r.samples > 0).collect();
    let node_count = measured.iter().map(|r| r.node.as_str()).collect::<BTreeSet<_>>().len();
    let model_count = measured.iter().map(|r| r.model.as_str()).collect::<BTreeSet<_>>().len();
    if node_count > 1 && model_count > 0 {
        md.push_str("\n## Side by side\n");
        matrix(&mut md, "Decode tok/s (median)", &rows, &|r| fmt_opt(median(&r.decode), 1));
        matrix(&mut md, "Time to first token, ms (median)", &rows, &|r| fmt_opt(median(&r.ttft), 0));
        matrix(&mut md, "Prompt tok/s (median)", &rows, &|r| fmt_opt(median(&r.prompt), 0));
        matrix(&mut md, "Quality (passed/ran)", &rows, &|r| r.quality().unwrap_or_else(|| "-".into()));
    }
    let categories: BTreeSet<&str> = rows.iter().flat_map(|r| r.by_category.keys().copied()).collect();
    if !categories.is_empty() {
        md.push_str("\n## Quality by category (percent, passed/ran)\n\n| model | node |");
        for c in &categories {
            md.push_str(&format!(" {c} |"));
        }
        md.push_str(" total |\n|---|---|");
        for _ in &categories {
            md.push_str("---|");
        }
        md.push_str("---|\n");
        for r in rows.iter().filter(|r| r.q_ran > 0) {
            md.push_str(&format!("| {} | {} |", r.model, r.node));
            for c in &categories {
                md.push_str(&match r.by_category.get(c) {
                    Some((p, n)) => format!(" {} |", pct(*p, *n)),
                    None => " - |".to_string(),
                });
            }
            md.push_str(&format!(" {} |\n", pct(r.q_pass, r.q_ran)));
        }
    }
    md.push_str("\nTTFT is for the short prompt. Decode tok/s is from the story (tokens after the first, divided by the time after it). Prompt tok/s is from the long prompt (tokens not served from a prefix cache, divided by TTFT, so it includes queueing and the network). Quality is checks passed over checks run, scored by exact match, parsed JSON or tests run in a sandbox.\n");
    let forwarded: Vec<String> = rows
        .iter()
        .filter(|r| !r.served_by.is_empty())
        .map(|r| format!("{} on {} was answered by {}", r.model, r.node, r.served_by.join(", ")))
        .collect();
    if !forwarded.is_empty() {
        md.push_str("\nForwarded: a node placed these on another node of the pool, so the speed is that node's, plus a hop:\n\n");
        for f in &forwarded {
            md.push_str(&format!("- {f}\n"));
        }
    }
    let mut skipped: Vec<String> = rows.iter().filter_map(|r| r.skipped.as_ref().map(|w| format!("{} on {}: {w}", r.model, r.node))).collect();
    skipped.extend(meta.notes.iter().map(|(node, what)| {
        let what = scrub(what, args.real_names);
        if node.is_empty() { what } else { format!("{}: {what}", namer.name(node)) }
    }));
    if !skipped.is_empty() {
        md.push_str("\nSkipped:\n\n");
        for s in &skipped {
            md.push_str(&format!("- {s}\n"));
        }
    }

    let mut csv = String::from(
        "started,suite,via,model,node,served_by,purpose,samples,ttft_ms_median,ttft_ms_p90,decode_tps_median,prompt_tps_median,total_ms_median,quality_passed,quality_ran,errors,skipped\n",
    );
    for r in &rows {
        let f = |x: Option<f64>, d: usize| x.map_or(String::new(), |v| format!("{v:.d$}"));
        let fields = [
            meta.started.clone(),
            args.suite.name().into(),
            meta.via.into(),
            r.model.clone(),
            r.node.clone(),
            r.served_by.join(" "),
            r.purpose.clone(),
            r.samples.to_string(),
            f(median(&r.ttft), 0),
            f(percentile(&r.ttft, 90.0), 0),
            f(median(&r.decode), 1),
            f(median(&r.prompt), 0),
            f(median(&r.total), 0),
            if r.q_ran > 0 { r.q_pass.to_string() } else { String::new() },
            if r.q_ran > 0 { r.q_ran.to_string() } else { String::new() },
            r.errors.to_string(),
            r.skipped.clone().unwrap_or_default(),
        ];
        csv.push_str(&fields.iter().map(|x| csv_field(x)).collect::<Vec<_>>().join(","));
        csv.push('\n');
    }

    let json = json!({
        "bench": "ymirctl bench", "version": 2, "started": meta.started, "seconds": round1(meta.seconds),
        "via": meta.via, "suite": args.suite.name(),
        "settings": {"warmup": args.warmup, "repeats": args.repeats, "cold": args.cold, "temperature": 0,
                     "context": args.context, "max_tokens": args.max_tokens,
                     "quality_tokens": args.quality_tokens.unwrap_or_else(|| args.suite.quality_tokens())},
        "rows": rows.iter().map(|r| r.json.clone()).collect::<Vec<_>>(),
        "skipped": skipped,
    });
    Report { json, markdown: md, csv }
}

fn percent(p: usize, n: usize) -> f64 {
    if n == 0 { 0.0 } else { (1000.0 * p as f64 / n as f64).round() / 10.0 }
}

/// `73% (8/11)`.
fn pct(p: usize, n: usize) -> String {
    format!("{:.0}% ({p}/{n})", percent(p, n))
}
