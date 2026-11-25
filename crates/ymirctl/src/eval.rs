//! `ymirctl eval`: score models of the pool on a held-out file of prompts (training plan P1).
//!
//! Each row is a `ymirctl batch --jsonl` row plus how its answer is judged: an `expect`
//! (deterministic, with the `ymirctl bench` scorers) and/or a `rubric` (graded PASS/FAIL by a
//! `--judge` model). Every model answers at temperature 0 with a fixed seed unless a row says
//! otherwise, waits for its own model rather than a stand-in, and answers that came from the
//! answer cache or a stand-in are counted and reported. Deterministic and judge scores are kept
//! apart: judges favour their own family. No eval set is built in.
//!
//! ```json
//! {"id": "e1", "prompt": "What is 6 times 7?", "expect": {"number": 42}}
//! {"id": "e2", "prompt": "Name a relay's job.", "expect": {"contains": "relay"}, "rubric": "Says it forwards traffic."}
//! ```
//!
//! `expect` is one of `{"exact": ["a", "b"]}` (normalized), `{"contains": "x"}`,
//! `{"contains_all": ["x", "y"]}`, `{"number": 42}` (the last whole number), `{"json": VALUE}`
//! (the answer, or its first fenced block, parses to VALUE) or `{"lines": ["a", "b"]}`.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use ymir_core::Paths;
use ymir_core::batch::{BatchItem, BatchTask, CacheMode};
use ymir_core::dataset::{self, CHUNK, RowDefaults};

use crate::bench::score::{extract_block, last_number, normalize, strip_think};

#[derive(clap::Args, Debug, Clone)]
pub struct EvalArgs {
    /// The eval file: JSONL rows with `prompt` or `messages`, and `expect` and/or `rubric`.
    pub file: PathBuf,
    /// A model to score; repeat to compare several on the same rows.
    #[arg(long, short, required = true)]
    pub model: Vec<String>,
    /// A model that grades `rubric` rows PASS or FAIL.
    #[arg(long)]
    pub judge: Option<String>,
    /// Write one JSON row per (model, row) with the answer and both verdicts.
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Seed for rows that don't set one (the same for every model).
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    #[arg(long)]
    pub max_tokens: Option<u32>,
    #[arg(long)]
    pub system: Option<String>,
    #[arg(long)]
    pub concurrency: Option<usize>,
    #[arg(long)]
    pub retries: Option<u32>,
}

/// How an answer is checked.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Expect {
    Exact(Vec<String>),
    Contains(String),
    ContainsAll(Vec<String>),
    Number(i64),
    Json(Value),
    Lines(Vec<String>),
}

impl Expect {
    /// Whether `answer` passes.
    pub fn check(&self, answer: &str) -> bool {
        match self {
            Expect::Exact(ok) => {
                let a = normalize(answer);
                ok.iter().any(|o| normalize(o) == a)
            }
            Expect::Contains(w) => strip_think(answer).to_lowercase().contains(&w.to_lowercase()),
            Expect::ContainsAll(ws) => {
                let a = strip_think(answer).to_lowercase();
                ws.iter().all(|w| a.contains(&w.to_lowercase()))
            }
            Expect::Number(n) => last_number(answer) == Some(*n),
            Expect::Json(v) => serde_json::from_str::<Value>(&extract_block(answer)).is_ok_and(|got| &got == v),
            Expect::Lines(l) => {
                let t = strip_think(answer);
                let got: Vec<&str> = t.lines().map(str::trim).filter(|x| !x.is_empty()).collect();
                got == l.iter().map(String::as_str).collect::<Vec<_>>()
            }
        }
    }
}

/// One eval row: the task and how it is judged.
#[derive(Debug, Clone)]
pub struct EvalRow {
    pub task: BatchTask,
    pub expect: Option<Expect>,
    pub rubric: Option<String>,
}

/// Parses an eval file: `ymirctl batch --jsonl` rows with `expect` and/or `rubric`.
pub fn parse(text: &str, d: &RowDefaults) -> Result<Vec<EvalRow>> {
    // Take `expect` and `rubric` off each row, keeping the line numbers (they are default ids),
    // and parse the rest as dataset rows.
    let mut rest = String::new();
    let mut judged = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let t = line.trim();
        if t.is_empty() || t.starts_with("//") {
            rest.push('\n');
            continue;
        }
        let mut v: Value = serde_json::from_str(t).with_context(|| format!("line {}: not a valid row", n + 1))?;
        let Some(o) = v.as_object_mut() else { bail!("line {}: not a JSON object", n + 1) };
        let expect = match o.remove("expect") {
            None => None,
            Some(e) => Some(serde_json::from_value::<Expect>(e).with_context(|| format!("line {}: bad `expect`", n + 1))?),
        };
        let rubric = match o.remove("rubric") {
            None => None,
            Some(Value::String(r)) if !r.trim().is_empty() => Some(r),
            Some(_) => bail!("line {}: `rubric` must be a non-empty string", n + 1),
        };
        if expect.is_none() && rubric.is_none() {
            bail!("line {}: needs `expect` or `rubric`", n + 1);
        }
        judged.push((expect, rubric));
        rest.push_str(&serde_json::to_string(&v)?);
        rest.push('\n');
    }
    let tasks: Vec<BatchTask> = dataset::parse_rows(&rest, d)?.into_iter().map(|r| r.task).collect();
    Ok(tasks.into_iter().zip(judged).map(|(task, (expect, rubric))| EvalRow { task, expect, rubric }).collect())
}

const JUDGE_SYSTEM: &str = "You grade an answer against a rubric. Reply with PASS or FAIL on the first line, then one short sentence saying why.";

/// The judge's task for `row`'s `answer`.
pub fn judge_task(row: &EvalRow, answer: &str, seed: u64) -> BatchTask {
    let question =
        row.task.messages.iter().filter(|m| m.role == weirdnet_agent::Role::User).map(|m| m.text().into_owned()).collect::<Vec<_>>().join("\n");
    let prompt = format!("Question:\n{question}\n\nRubric:\n{}\n\nAnswer:\n{}", row.rubric.as_deref().unwrap_or_default(), strip_think(answer));
    let mut t = BatchTask::prompt(format!("{}-judge", row.task.id), &prompt);
    t.messages.insert(0, weirdnet_agent::Message::system(JUDGE_SYSTEM));
    t.sampling.temperature = Some(0.0);
    t.sampling.max_tokens = Some(128);
    t.seed = Some(seed);
    t
}

/// The judge's verdict: `Some(true)` for PASS, `Some(false)` for FAIL, `None` when it said neither.
pub fn verdict(reply: &str) -> Option<bool> {
    let t = strip_think(reply);
    let first = t.lines().map(str::trim).find(|l| !l.is_empty())?;
    let word: String = first.trim_start_matches(|c: char| !c.is_alphanumeric()).chars().take_while(|c| c.is_alphanumeric()).collect();
    match word.to_ascii_uppercase().as_str() {
        "PASS" | "PASSED" => Some(true),
        "FAIL" | "FAILED" => Some(false),
        _ => None,
    }
}

/// One (model, row) result (shared with the rollout gate).
pub use ymir_core::rollout_edit::Scored;

/// Per-model totals.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Totals {
    pub rows: usize,
    pub errors: usize,
    pub checked: usize,
    pub check_passed: usize,
    pub judged: usize,
    pub judge_passed: usize,
    /// Rubric rows the judge gave no clear verdict on (or could not answer).
    pub judge_unclear: usize,
    pub cache_hits: usize,
    pub standins: usize,
}

pub fn totals(scored: &[Scored]) -> BTreeMap<String, Totals> {
    let mut t: BTreeMap<String, Totals> = BTreeMap::new();
    for s in scored {
        let e = t.entry(s.model.clone()).or_default();
        e.rows += 1;
        e.errors += usize::from(!s.ok);
        if let Some(c) = s.check {
            e.checked += 1;
            e.check_passed += usize::from(c);
        }
        if let Some(j) = s.judge {
            e.judged += 1;
            e.judge_passed += usize::from(j);
        } else if s.judge_reply.is_some() {
            e.judge_unclear += 1;
        }
        e.cache_hits += usize::from(s.cache_hit);
        e.standins += usize::from(s.standin.is_some());
    }
    t
}

fn scored(model: &str, row: &EvalRow, item: BatchItem) -> Scored {
    let (answer, error) = if item.ok { (item.text, None) } else { (String::new(), Some(item.text)) };
    let check = if item.ok { row.expect.as_ref().map(|e| e.check(&answer)) } else { None };
    Scored {
        model: model.to_string(),
        id: row.task.id.clone(),
        ok: item.ok,
        answer,
        check,
        judge: None,
        judge_reply: None,
        cache_hit: item.cache_hit,
        standin: item.standin,
        error,
    }
}

async fn ask_all(paths: &Paths, tasks: &[BatchTask], model: &str, job: &str, a: &EvalArgs) -> Result<Vec<BatchItem>> {
    let mut out = Vec::with_capacity(tasks.len());
    for chunk in tasks.chunks(CHUNK) {
        out.extend(dataset::ask(&paths.socket(), chunk, model, job, a.concurrency, a.retries, CacheMode::Bypass).await?);
    }
    Ok(out)
}

pub async fn run(paths: &Paths, a: EvalArgs, out: &mut dyn Write) -> Result<()> {
    let text = std::fs::read_to_string(&a.file).with_context(|| format!("reading {}", a.file.display()))?;
    let d = RowDefaults { system: a.system.clone(), temperature: Some(0.0), top_p: None, max_tokens: a.max_tokens, seed: Some(a.seed) };
    let rows = parse(&text, &d)?;
    let rubrics = rows.iter().filter(|r| r.rubric.is_some()).count();
    if rubrics > 0 && a.judge.is_none() {
        writeln!(out, "note: {rubrics} row(s) have a rubric but no --judge was given; they are scored by `expect` only")?;
    }
    if let Some(j) = &a.judge
        && a.model.contains(j)
    {
        writeln!(out, "note: the judge {j} is also a model under test; its own answers are graded by itself")?;
    }
    let job = format!("eval-{:x}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos()));
    let tasks: Vec<BatchTask> = rows.iter().map(|r| r.task.clone()).collect();
    let mut all = Vec::new();
    for model in &a.model {
        let items = ask_all(paths, &tasks, model, &format!("{job}-{model}"), &a).await?;
        let mut s: Vec<Scored> = rows.iter().zip(items).map(|(r, i)| scored(model, r, i)).collect();
        if let Some(judge) = &a.judge {
            let idx: Vec<usize> = (0..rows.len()).filter(|&i| rows[i].rubric.is_some() && s[i].ok).collect();
            let jt: Vec<BatchTask> = idx.iter().map(|&i| judge_task(&rows[i], &s[i].answer, a.seed)).collect();
            let verdicts = ask_all(paths, &jt, judge, &format!("{job}-{model}-judge"), &a).await?;
            for (&i, v) in idx.iter().zip(verdicts) {
                s[i].judge = if v.ok { verdict(&v.text) } else { None };
                s[i].judge_reply = Some(if v.ok { v.text } else { format!("judge failed: {}", v.text) });
            }
        }
        all.extend(s);
    }
    if let Some(p) = &a.out {
        let mut f = std::fs::File::create(p).with_context(|| format!("writing {}", p.display()))?;
        for s in &all {
            serde_json::to_writer(&mut f, s)?;
            f.write_all(b"\n")?;
        }
    }
    writeln!(out, "{:<28} {:>9} {:>9} {:>7} {:>7} {:>7}", "model", "checks", "judge", "errors", "cached", "stand-in")?;
    for (m, t) in totals(&all) {
        let frac = |p: usize, n: usize| if n == 0 { "-".to_string() } else { format!("{p}/{n}") };
        writeln!(
            out,
            "{:<28} {:>9} {:>9} {:>7} {:>7} {:>7}",
            m,
            frac(t.check_passed, t.checked),
            frac(t.judge_passed, t.judged),
            t.errors,
            t.cache_hits,
            t.standins
        )?;
        if t.judge_unclear > 0 {
            writeln!(out, "  {m}: the judge gave no clear PASS/FAIL on {} row(s)", t.judge_unclear)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expectations_use_the_bench_scorers() {
        assert!(Expect::Exact(vec!["Paris".into()]).check("  paris. "));
        assert!(Expect::Contains("Relay".into()).check("<think>no</think>A relay forwards traffic."));
        assert!(!Expect::Contains("relay".into()).check("<think>relay</think>nothing"));
        assert!(Expect::ContainsAll(vec!["peer".into(), "hash".into()]).check("Each peer stores a hash."));
        assert!(Expect::Number(42).check("6 x 7 = 42"));
        assert!(!Expect::Number(42).check("41"));
        assert!(Expect::Json(serde_json::json!({"a": 1})).check("```json\n{\"a\": 1}\n```"));
        assert!(Expect::Lines(vec!["a".into(), "b".into()]).check("a\n\n b \n"));
    }

    #[test]
    fn rows_keep_their_ids_and_take_eval_defaults() {
        let d = RowDefaults { temperature: Some(0.0), seed: Some(5), ..Default::default() };
        let text = "// c\n{\"prompt\": \"q\", \"expect\": {\"number\": 4}}\n{\"id\": \"r\", \"prompt\": \"p\", \"rubric\": \"good\"}\n";
        let rows = parse(text, &d).unwrap();
        assert_eq!(rows[0].task.id, "2");
        assert_eq!(rows[0].expect, Some(Expect::Number(4)));
        assert_eq!((rows[0].task.sampling.temperature, rows[0].task.seed), (Some(0.0), Some(5)));
        assert_eq!(rows[1].rubric.as_deref(), Some("good"));
        for bad in ["{\"prompt\": \"q\"}", "{\"prompt\": \"q\", \"expect\": {\"maybe\": 1}}", "{\"prompt\": \"q\", \"rubric\": 3}", "[1]"] {
            assert!(parse(bad, &d).is_err(), "{bad}");
        }
    }

    #[test]
    fn verdicts_read_the_first_line() {
        assert_eq!(verdict("PASS\nit says so"), Some(true));
        assert_eq!(verdict("<think>hmm FAIL</think>\n**Pass** - fine"), Some(true));
        assert_eq!(verdict("fail: wrong"), Some(false));
        assert_eq!(verdict("Hello there"), None);
        assert_eq!(verdict(""), None);
    }

    #[test]
    fn the_judge_sees_the_question_rubric_and_answer() {
        let rows =
            parse("{\"id\": \"x\", \"system\": \"SYSPROMPT\", \"prompt\": \"Why?\", \"rubric\": \"Because.\"}", &RowDefaults::default()).unwrap();
        let t = judge_task(&rows[0], "<think>x</think>Since.", 3);
        assert_eq!(t.id, "x-judge");
        let p = t.messages[1].text().into_owned();
        assert!(p.contains("Why?") && p.contains("Because.") && p.contains("Since.") && !p.contains("<think>") && !p.contains("SYSPROMPT"), "{p}");
        assert_eq!((t.sampling.temperature, t.seed), (Some(0.0), Some(3)));
    }

    #[test]
    fn totals_keep_checks_and_judges_apart() {
        let s = |m: &str, check: Option<bool>, judge: Option<bool>, reply: bool, cache: bool| Scored {
            model: m.into(),
            id: "i".into(),
            ok: true,
            answer: String::new(),
            check,
            judge,
            judge_reply: reply.then(String::new),
            cache_hit: cache,
            standin: None,
            error: None,
        };
        let t = totals(&[s("a", Some(true), Some(false), true, false), s("a", Some(false), None, true, true), s("b", None, Some(true), true, false)]);
        let a = &t["a"];
        assert_eq!((a.rows, a.checked, a.check_passed, a.judged, a.judge_passed, a.judge_unclear, a.cache_hits), (2, 2, 1, 1, 0, 1, 1));
        assert_eq!((t["b"].checked, t["b"].judge_passed), (0, 1));
    }
}
