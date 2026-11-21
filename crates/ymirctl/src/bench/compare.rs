//! `bench compare A.json B.json`: two saved results side by side, with the change in each number.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use serde_json::Value;

#[derive(clap::Args, Debug, Clone)]
pub struct CompareArgs {
    /// The earlier (baseline) result.
    pub a: PathBuf,
    /// The later result.
    pub b: PathBuf,
    /// Also write the comparison here as Markdown.
    #[arg(long)]
    pub out: Option<PathBuf>,
}

fn load(p: &PathBuf) -> Result<Value> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?)
        .with_context(|| format!("{} is not JSON", p.display()))?;
    if v["bench"] != "ymirctl bench" {
        bail!("{} is not a `ymirctl bench` result", p.display());
    }
    if v["rows"].as_array().is_none() {
        bail!("{} is from an older bench (no per-node rows); rerun it", p.display());
    }
    Ok(v)
}

#[derive(Default, Clone)]
struct Nums {
    ttft: Option<f64>,
    decode: Option<f64>,
    prompt: Option<f64>,
    q: Option<(u64, u64)>,
}

fn index(v: &Value) -> BTreeMap<(String, String), Nums> {
    let mut m = BTreeMap::new();
    for r in v["rows"].as_array().into_iter().flatten() {
        if r["skipped"].is_string() {
            continue;
        }
        let key = (r["model"].as_str().unwrap_or("?").to_string(), r["node"].as_str().unwrap_or("?").to_string());
        let q = r["quality"].as_object().and_then(|q| Some((q.get("passed")?.as_u64()?, q.get("ran")?.as_u64()?)));
        m.insert(
            key,
            Nums { ttft: r["ttft_ms"]["median"].as_f64(), decode: r["decode_tps"]["median"].as_f64(), prompt: r["prompt_tps"]["median"].as_f64(), q },
        );
    }
    m
}

/// `a → b (+x%)`.
fn change(a: Option<f64>, b: Option<f64>, digits: usize) -> String {
    match (a, b) {
        (Some(a), Some(b)) if a != 0.0 => format!("{a:.digits$} → {b:.digits$} ({:+.0}%)", (b - a) / a * 100.0),
        (Some(a), Some(b)) => format!("{a:.digits$} → {b:.digits$}"),
        (Some(a), None) => format!("{a:.digits$} → -"),
        (None, Some(b)) => format!("- → {b:.digits$}"),
        (None, None) => "-".into(),
    }
}

fn quality(a: Option<(u64, u64)>, b: Option<(u64, u64)>) -> String {
    let f = |q: Option<(u64, u64)>| q.map_or("-".to_string(), |(p, n)| format!("{p}/{n}"));
    format!("{} → {}", f(a), f(b))
}

pub fn run(args: &CompareArgs, out: &mut dyn Write) -> Result<()> {
    let (a, b) = (load(&args.a)?, load(&args.b)?);
    let (ia, ib) = (index(&a), index(&b));
    let mut md = format!(
        "# Benchmark comparison\n\nA: {} (suite {}, via {})\nB: {} (suite {}, via {})\n\nTTFT is lower-is-better; tok/s higher-is-better.\n\n",
        a["started"].as_str().unwrap_or("?"),
        a["suite"].as_str().unwrap_or("?"),
        a["via"].as_str().unwrap_or("?"),
        b["started"].as_str().unwrap_or("?"),
        b["suite"].as_str().unwrap_or("?"),
        b["via"].as_str().unwrap_or("?"),
    );
    if a["suite"] != b["suite"] {
        md.push_str("Note: the suites differ, so quality scores are not comparable.\n\n");
    }
    md.push_str("| model | node | TTFT ms | decode tok/s | prompt tok/s | quality |\n|---|---|---|---|---|---|\n");
    let mut only_a = Vec::new();
    let mut only_b = Vec::new();
    for (key, na) in &ia {
        match ib.get(key) {
            Some(nb) => md.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} |\n",
                key.0,
                key.1,
                change(na.ttft, nb.ttft, 0),
                change(na.decode, nb.decode, 1),
                change(na.prompt, nb.prompt, 0),
                quality(na.q, nb.q)
            )),
            None => only_a.push(format!("{} on {}", key.0, key.1)),
        }
    }
    for key in ib.keys().filter(|k| !ia.contains_key(*k)) {
        only_b.push(format!("{} on {}", key.0, key.1));
    }
    if !only_a.is_empty() {
        md.push_str(&format!("\nOnly in A: {}\n", only_a.join(", ")));
    }
    if !only_b.is_empty() {
        md.push_str(&format!("\nOnly in B: {}\n", only_b.join(", ")));
    }
    if ia.keys().all(|k| !ib.contains_key(k)) {
        md.push_str("\nNo model/node pair is in both results (node labels differ between real-name and anonymised runs).\n");
    }
    write!(out, "{md}")?;
    if let Some(p) = &args.out {
        std::fs::write(p, &md).with_context(|| format!("writing {}", p.display()))?;
        writeln!(out, "\nSaved {}", p.display())?;
    }
    Ok(())
}
