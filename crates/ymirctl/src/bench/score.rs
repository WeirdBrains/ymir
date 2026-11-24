//! Scoring: judging answers (exact match, numbers, JSON, lines, code run in a sandbox) and the
//! statistics over samples.

use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{Context, Result};
use serde_json::{Value, json};

use super::suites::{Check, Kind, expected_json};

/// The answer without a reasoning block.
pub fn strip_think(s: &str) -> String {
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

pub fn normalize(s: &str) -> String {
    strip_think(s).trim_matches(|c: char| c.is_whitespace() || ".,!?\"'`*_:;".contains(c)).to_lowercase()
}

pub fn last_number(s: &str) -> Option<i64> {
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
pub fn extract_block(s: &str) -> String {
    let s = strip_think(s);
    if let Some(a) = s.find("```") {
        let rest = &s[a + 3..];
        let rest = rest.split_once('\n').map_or(rest, |(_, r)| r);
        return rest.split("```").next().unwrap_or(rest).trim().to_string();
    }
    s
}

pub fn json_ok(s: &str) -> bool {
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
pub fn run_python(code: &str, tests: &str) -> Result<bool> {
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

/// Whether code answers can be run: python3 and the unix sandbox in `run_python`
/// (`sh`, `ulimit`, `unshare`). Off unix, code questions are not scored.
#[cfg(not(unix))]
pub fn python_available() -> bool {
    false
}

/// Whether code answers can be run: python3 and the unix sandbox in `run_python`.
#[cfg(unix)]
pub fn python_available() -> bool {
    Command::new("python3").arg("--version").stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok_and(|s| s.success())
}

/// `Some(passed)`, or `None` when the check could not run.
pub fn judge(check: &Check, answer: &str, code_ok: bool) -> Option<bool> {
    match &check.kind {
        Kind::JsonEq(j) => Some(serde_json::from_str::<Value>(&extract_block(answer)).is_ok_and(|v| v == expected_json(j))),
        Kind::Lines(l) => {
            let t = strip_think(answer);
            let got: Vec<&str> = t.lines().map(str::trim).filter(|x| !x.is_empty()).collect();
            Some(got == *l)
        }
        Kind::Func(f) => Some(f(answer)),
        Kind::Has(w) => Some(strip_think(answer).to_lowercase().contains(&w.to_lowercase())),
        Kind::Exact(ok) => Some(ok.contains(&normalize(answer).as_str())),
        Kind::Number(n) => Some(last_number(answer) == Some(*n)),
        Kind::Json => Some(json_ok(answer)),
        Kind::Code(tests) => code_ok.then(|| run_python(&extract_block(answer), tests).unwrap_or(false)),
    }
}

// ---- statistics ----

pub fn median(v: &[f64]) -> Option<f64> {
    percentile(v, 50.0)
}

/// Nearest-rank percentile.
pub fn percentile(v: &[f64], p: f64) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let rank = ((p / 100.0) * s.len() as f64).ceil().max(1.0) as usize;
    Some(s[rank.min(s.len()) - 1])
}

pub fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

#[cfg(test)]
/// The letters-only words of the answer, lower case, joined by single spaces.
pub fn words(s: &str) -> String {
    strip_think(s).split(|c: char| !c.is_alphabetic()).filter(|w| !w.is_empty()).map(str::to_lowercase).collect::<Vec<_>>().join(" ")
}
