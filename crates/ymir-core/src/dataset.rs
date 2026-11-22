//! Data generation through the pool (training plan P1, `ymirctl batch --jsonl`).
//!
//! A teacher model in the pool answers a file of prompts, one JSON object per line, and the
//! answers are written as JSONL rows next to a manifest that records the teacher model and the
//! license the operator gave for it. Nothing here trains a model; a separate trainer reads the
//! output. No dataset is built in: the input is always the operator's file.
//!
//! Input rows (`prompt` or `messages`, the rest optional; the command line gives defaults):
//!
//! ```json
//! {"id": "q1", "prompt": "Name a prime.", "system": "Answer briefly.", "temperature": 0.7,
//!  "top_p": 0.9, "max_tokens": 128, "seed": 7, "stop": ["\n\n"]}
//! {"id": "q2", "messages": [{"role": "user", "content": "Hi"}]}
//! ```
//!
//! Output rows carry the messages sent, the completion, the model and node that answered, and
//! whether the row is `usable` as teacher data: it answered, it was not replayed from the answer
//! cache, no stand-in answered, and the model that answered is the teacher. Generation waits for
//! the teacher to load rather than take a stand-in.
//!
//! With `--sources FILE`, a sources manifest ([`Sources`]: the topic and each source's URL and
//! license) is checked, every row must name its `source`, and the dataset's manifest records the
//! sources, their licenses and rows per source.
//!
//! Runs go to `ymird` in chunks and every chunk is appended to the output as it finishes, so
//! `--resume` picks up after an interrupted run: rows already answered are skipped (failed rows
//! are tried again and appended, and the last row for an id wins).

use std::collections::{BTreeMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use weirdnet_agent::remote::infer::v2::{Sampling, Usage};
use weirdnet_agent::{Message, Role};

use crate::batch::{BatchItem, BatchTask, CacheMode};
use crate::control::{self, ControlRequest};

/// The manifest's format name.
pub const MANIFEST_FORMAT: &str = "ymir-dataset/1";
/// Rows sent to `ymird` per control request.
pub const CHUNK: usize = 32;

/// Defaults for rows that don't set their own (from the command line).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RowDefaults {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InputRow {
    #[serde(default)]
    id: Option<serde_json::Value>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    messages: Option<Vec<Message>>,
    #[serde(default)]
    system: Option<String>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<u32>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stop: Option<Vec<String>>,
    /// The id of the source (in the sources manifest) this row comes from.
    #[serde(default)]
    source: Option<String>,
}

/// A parsed input row: the task, and the source it comes from, if it names one.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub task: BatchTask,
    pub source: Option<String>,
}

/// Parses an input file: one JSON object per line; blank lines and lines starting with `//` are
/// skipped. A row without an `id` gets its line number. Ids must be unique.
pub fn parse_rows(text: &str, d: &RowDefaults) -> Result<Vec<Row>> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for (n, line) in text.lines().enumerate() {
        let line_no = n + 1;
        let line = line.trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        let row: InputRow = serde_json::from_str(line).with_context(|| format!("line {line_no}: not a valid row"))?;
        let id = match row.id {
            None => line_no.to_string(),
            Some(serde_json::Value::String(s)) if !s.is_empty() => s,
            Some(serde_json::Value::Number(x)) => x.to_string(),
            Some(_) => bail!("line {line_no}: `id` must be a non-empty string or a number"),
        };
        if !seen.insert(id.clone()) {
            bail!("line {line_no}: id {id:?} is used twice");
        }
        let mut messages = match (row.prompt, row.messages) {
            (Some(p), None) => vec![Message::user(p)],
            (None, Some(m)) if !m.is_empty() => m,
            (None, Some(_)) => bail!("line {line_no}: `messages` is empty"),
            (Some(_), Some(_)) => bail!("line {line_no}: give `prompt` or `messages`, not both"),
            (None, None) => bail!("line {line_no}: needs `prompt` or `messages`"),
        };
        let has_system = messages.first().is_some_and(|m| m.role == Role::System);
        match (row.system, has_system) {
            (Some(_), true) => bail!("line {line_no}: `system` and a system message in `messages`"),
            (Some(s), false) => messages.insert(0, Message::system(s)),
            (None, false) => {
                if let Some(s) = &d.system {
                    messages.insert(0, Message::system(s.clone()));
                }
            }
            // The row's own system message wins over the default.
            (None, true) => {}
        }
        let mut sampling = Sampling::default();
        sampling.temperature = row.temperature.or(d.temperature);
        sampling.top_p = row.top_p.or(d.top_p);
        sampling.top_k = row.top_k;
        sampling.max_tokens = row.max_tokens.or(d.max_tokens);
        sampling.stop = row.stop.unwrap_or_default();
        if row.source.as_deref().is_some_and(|x| x.trim().is_empty()) {
            bail!("line {line_no}: `source` is empty");
        }
        out.push(Row { task: BatchTask { id, messages, sampling, seed: row.seed.or(d.seed) }, source: row.source });
    }
    if out.is_empty() {
        bail!("no rows in the input");
    }
    Ok(out)
}

/// One output row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputRow {
    pub id: String,
    pub ok: bool,
    /// Fit for training: answered by the teacher itself, fresh (not cached, not a stand-in).
    pub usable: bool,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub completion: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub model: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub node: String,
    pub attempts: u32,
    #[serde(default)]
    pub cache_hit: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub standin: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// The source the row comes from (an id in the sources manifest).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

impl OutputRow {
    /// The row for `task`'s outcome; `teacher` is the model asked for.
    pub fn new(task: &BatchTask, item: BatchItem, teacher: &str) -> Self {
        let usable = item.ok && !item.cache_hit && item.standin.is_none() && item.model == teacher;
        let (completion, error) = if item.ok { (item.text, None) } else { (String::new(), Some(item.text)) };
        Self {
            id: task.id.clone(),
            ok: item.ok,
            usable,
            messages: task.messages.clone(),
            completion,
            model: item.model,
            node: item.node,
            attempts: item.attempts,
            cache_hit: item.cache_hit,
            standin: item.standin,
            finish_reason: item.finish_reason,
            usage: item.usage,
            error,
            source: None,
        }
    }
}

/// The last row for each id in an output file, in first-seen order. Lines that don't parse
/// (a run killed mid-write) are skipped.
pub fn read_output(text: &str) -> Vec<OutputRow> {
    let mut order: Vec<String> = Vec::new();
    let mut last: BTreeMap<String, OutputRow> = BTreeMap::new();
    for line in text.lines() {
        let Ok(row) = serde_json::from_str::<OutputRow>(line) else { continue };
        if !last.contains_key(&row.id) {
            order.push(row.id.clone());
        }
        last.insert(row.id.clone(), row);
    }
    order.into_iter().filter_map(|id| last.remove(&id)).collect()
}

/// Ids a resumed run skips: rows that answered.
pub fn done_ids(rows: &[OutputRow]) -> HashSet<String> {
    rows.iter().filter(|r| r.ok).map(|r| r.id.clone()).collect()
}

/// The teacher: the model asked for and the license the operator gave for its outputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Teacher {
    pub model: String,
    /// An SPDX id or license name, as given with `--license`; `unknown` only when allowed.
    pub license: String,
    /// Where the license came from (`operator`, for now).
    pub license_source: String,
}

/// The sources a dataset is drawn from (`--sources FILE`, JSON). The operator writes it; Ymir
/// fetches nothing. Every source needs a license, so the dataset's manifest can say what each row
/// may be used for.
///
/// ```json
/// {"topic": "Decentralized networks: how they work",
///  "sources": [{"id": "kad", "title": "Kademlia", "url": "https://example.org/kademlia.pdf",
///               "license": "CC-BY-4.0", "kind": "paper"}]}
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sources {
    pub topic: String,
    pub sources: Vec<Source>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub id: String,
    pub title: String,
    pub url: String,
    /// An SPDX id or license name; `unknown` only with `--allow-unknown-license`.
    pub license: String,
    /// `paper`, `spec`, `docs`, `book`...
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

impl Sources {
    /// Parses and checks a sources manifest: a topic, unique non-empty ids, http(s) URLs and a
    /// license for every source.
    pub fn parse(text: &str, allow_unknown_license: bool) -> Result<Self> {
        let s: Sources = serde_json::from_str(text).context("the sources manifest is not valid")?;
        if s.topic.trim().is_empty() {
            bail!("the sources manifest has no topic");
        }
        if s.sources.is_empty() {
            bail!("the sources manifest lists no sources");
        }
        let mut seen = HashSet::new();
        for x in &s.sources {
            if x.id.trim().is_empty() || !seen.insert(x.id.as_str()) {
                bail!("source id {:?} is empty or used twice", x.id);
            }
            if !(x.url.starts_with("https://") || x.url.starts_with("http://")) {
                bail!("source {}: the url must be http(s)", x.id);
            }
            check_license(&x.license, allow_unknown_license).with_context(|| format!("source {}", x.id))?;
        }
        Ok(s)
    }

    pub fn ids(&self) -> HashSet<&str> {
        self.sources.iter().map(|x| x.id.as_str()).collect()
    }
}

/// Checks that every row's `source` is in `sources`, and, when there are sources, that every row
/// names one.
pub fn check_row_sources(rows: &[Row], sources: Option<&Sources>) -> Result<()> {
    match sources {
        None => {
            if let Some(r) = rows.iter().find(|r| r.source.is_some()) {
                bail!("row {} names a source but no --sources manifest was given", r.task.id);
            }
        }
        Some(s) => {
            let ids = s.ids();
            for r in rows {
                match r.source.as_deref() {
                    None => bail!("row {} names no source (every row needs one with --sources)", r.task.id),
                    Some(x) if !ids.contains(x) => bail!("row {}: source {x:?} is not in the sources manifest", r.task.id),
                    Some(_) => {}
                }
            }
        }
    }
    Ok(())
}

/// The sources part of a dataset's manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourcesRecord {
    pub name: String,
    pub sha256: String,
    pub topic: String,
    pub sources: Vec<Source>,
    /// Rows per source id.
    pub rows: BTreeMap<String, usize>,
}

/// Written next to the output as `OUT.manifest.json`, rewritten after every chunk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub job: String,
    pub teacher: Teacher,
    /// The input file's name (not its path) and SHA-256.
    pub input_name: String,
    pub input_sha256: String,
    pub output_name: String,
    pub defaults: RowDefaults,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sources: Option<SourcesRecord>,
    pub cache: CacheMode,
    pub rows_in_input: usize,
    pub rows_answered: usize,
    pub rows_usable: usize,
    pub rows_failed: usize,
    pub cache_hits: usize,
    pub standins: usize,
    /// Answered rows per model that answered.
    pub answered_by: BTreeMap<String, usize>,
    pub created_ms: i64,
    pub updated_ms: i64,
}

impl Manifest {
    fn count(&mut self, rows: &[OutputRow]) {
        self.rows_answered = rows.iter().filter(|r| r.ok).count();
        self.rows_usable = rows.iter().filter(|r| r.usable).count();
        self.rows_failed = rows.iter().filter(|r| !r.ok).count();
        self.cache_hits = rows.iter().filter(|r| r.cache_hit).count();
        self.standins = rows.iter().filter(|r| r.standin.is_some()).count();
        self.answered_by = BTreeMap::new();
        for r in rows.iter().filter(|r| r.ok) {
            *self.answered_by.entry(r.model.clone()).or_default() += 1;
        }
    }
}

/// The manifest's path for an output file: `OUT.manifest.json`.
pub fn manifest_path(out: &Path) -> PathBuf {
    let mut s = out.as_os_str().to_os_string();
    s.push(".manifest.json");
    PathBuf::from(s)
}

/// A generation run.
#[derive(Debug, Clone)]
pub struct Generate {
    pub input: PathBuf,
    pub output: PathBuf,
    pub teacher: String,
    pub license: String,
    pub allow_unknown_license: bool,
    pub defaults: RowDefaults,
    /// A sources manifest ([`Sources`]); every row then names its `source`.
    pub sources: Option<PathBuf>,
    pub resume: bool,
    pub concurrency: Option<usize>,
    pub retries: Option<u32>,
    pub cache: CacheMode,
}

/// What a run did.
#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    pub skipped: usize,
    pub sent: usize,
    pub manifest: Manifest,
    pub manifest_path: PathBuf,
}

fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64)
}

fn file_name(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// Runs `tasks` on `model` through the `ymird` on `socket` (one control request), waiting for the
/// model rather than a stand-in, and returns one outcome per task, in order.
pub async fn ask(
    socket: &Path,
    tasks: &[BatchTask],
    model: &str,
    job: &str,
    concurrency: Option<usize>,
    retries: Option<u32>,
    cache: CacheMode,
) -> Result<Vec<BatchItem>> {
    let req = ControlRequest::BatchTasks {
        tasks: tasks.to_vec(),
        model: Some(model.to_string()),
        concurrency,
        retries,
        job: Some(job.to_string()),
        wait_for_model: true,
        cache,
    };
    let r = control::request(socket, &req).await?;
    if !r.ok {
        bail!("{}", r.message.unwrap_or_default());
    }
    if r.batch.len() != tasks.len() {
        bail!("ymird answered {} of {} rows", r.batch.len(), tasks.len());
    }
    Ok(r.batch)
}

/// Checks the license given for the teacher.
pub fn check_license(license: &str, allow_unknown: bool) -> Result<()> {
    let l = license.trim();
    if l.is_empty() {
        bail!("--license is empty: give the teacher model's license (an SPDX id such as apache-2.0)");
    }
    if l.eq_ignore_ascii_case("unknown") && !allow_unknown {
        bail!(
            "the teacher's license is unknown: some licenses forbid training on a model's outputs. Check it, or pass --allow-unknown-license to record it as unknown"
        );
    }
    Ok(())
}

/// Runs a generation against the `ymird` on `socket`. `progress` gets (rows done, rows to do)
/// after every chunk.
pub async fn generate(socket: &Path, g: &Generate, progress: &mut dyn FnMut(usize, usize)) -> Result<Summary> {
    check_license(&g.license, g.allow_unknown_license)?;
    let bytes = std::fs::read(&g.input).with_context(|| format!("reading {}", g.input.display()))?;
    let text = String::from_utf8(bytes.clone()).context("the input is not UTF-8")?;
    let parsed = parse_rows(&text, &g.defaults)?;
    let sources = match &g.sources {
        None => None,
        Some(p) => {
            let b = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
            let s = Sources::parse(std::str::from_utf8(&b).context("the sources manifest is not UTF-8")?, g.allow_unknown_license)?;
            Some((s, format!("{:x}", Sha256::digest(&b)), file_name(p)))
        }
    };
    check_row_sources(&parsed, sources.as_ref().map(|(s, _, _)| s))?;
    let source_of: BTreeMap<String, String> = parsed.iter().filter_map(|r| Some((r.task.id.clone(), r.source.clone()?))).collect();
    let mut rows_per_source: BTreeMap<String, usize> = BTreeMap::new();
    for x in source_of.values() {
        *rows_per_source.entry(x.clone()).or_default() += 1;
    }
    let tasks: Vec<BatchTask> = parsed.into_iter().map(|r| r.task).collect();
    let input_sha256 = format!("{:x}", Sha256::digest(&bytes));
    let mpath = manifest_path(&g.output);

    let existing = if g.output.exists() {
        if !g.resume {
            bail!("{} exists: pass --resume to continue it, or choose another --out", g.output.display());
        }
        read_output(&std::fs::read_to_string(&g.output).with_context(|| format!("reading {}", g.output.display()))?)
    } else {
        Vec::new()
    };
    let old: Option<Manifest> = if g.resume && mpath.exists() {
        Some(serde_json::from_str(&std::fs::read_to_string(&mpath)?).with_context(|| format!("reading {}", mpath.display()))?)
    } else {
        None
    };
    if let Some(m) = &old {
        if m.teacher.model != g.teacher {
            bail!("{} was made with teacher {:?}, not {:?}", g.output.display(), m.teacher.model, g.teacher);
        }
        if m.input_sha256 != input_sha256 {
            tracing::warn!("the input changed since the last run; rows are matched by id");
        }
    }
    let job = old.as_ref().map_or_else(|| format!("dataset-{:x}", now_ms()), |m| m.job.clone());
    let mut manifest = Manifest {
        format: MANIFEST_FORMAT.into(),
        job: job.clone(),
        teacher: Teacher { model: g.teacher.clone(), license: g.license.trim().to_string(), license_source: "operator".into() },
        input_name: file_name(&g.input),
        input_sha256,
        output_name: file_name(&g.output),
        defaults: g.defaults.clone(),
        sources: sources.map(|(s, sha256, name)| SourcesRecord { name, sha256, topic: s.topic, sources: s.sources, rows: rows_per_source }),
        cache: g.cache,
        rows_in_input: tasks.len(),
        rows_answered: 0,
        rows_usable: 0,
        rows_failed: 0,
        cache_hits: 0,
        standins: 0,
        answered_by: BTreeMap::new(),
        created_ms: old.as_ref().map_or_else(now_ms, |m| m.created_ms),
        updated_ms: now_ms(),
    };

    let done = done_ids(&existing);
    let todo: Vec<BatchTask> = tasks.into_iter().filter(|t| !done.contains(&t.id)).collect();
    let (skipped, total) = (done.len(), todo.len());
    let mut out = std::fs::OpenOptions::new().create(true).append(true).open(&g.output).with_context(|| format!("opening {}", g.output.display()))?;
    let mut rows = existing;
    let mut sent = 0;
    progress(0, total);
    for chunk in todo.chunks(CHUNK) {
        let items = ask(socket, chunk, &g.teacher, &job, g.concurrency, g.retries, g.cache).await?;
        let mut buf = Vec::new();
        for (t, item) in chunk.iter().zip(items) {
            let mut row = OutputRow::new(t, item, &g.teacher);
            row.source = source_of.get(&t.id).cloned();
            serde_json::to_writer(&mut buf, &row)?;
            buf.push(b'\n');
            rows.retain(|x| x.id != row.id);
            rows.push(row);
        }
        out.write_all(&buf)?;
        out.flush()?;
        sent += chunk.len();
        manifest.count(&rows);
        manifest.updated_ms = now_ms();
        write_manifest(&mpath, &manifest)?;
        progress(sent, total);
    }
    manifest.count(&rows);
    write_manifest(&mpath, &manifest)?;
    Ok(Summary { skipped, sent, manifest, manifest_path: mpath })
}

fn write_manifest(path: &Path, m: &Manifest) -> Result<()> {
    let tmp = {
        let mut s = path.as_os_str().to_os_string();
        s.push(".tmp");
        PathBuf::from(s)
    };
    std::fs::write(&tmp, serde_json::to_vec_pretty(m)?)?;
    std::fs::rename(&tmp, path).with_context(|| format!("writing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(ok: bool, text: &str, model: &str) -> BatchItem {
        serde_json::from_value(serde_json::json!({"index": 0, "ok": ok, "text": text, "node": "n1", "model": model, "attempts": 1})).unwrap()
    }

    #[test]
    fn rows_parse_with_defaults_and_overrides() {
        let d = RowDefaults { system: Some("Be brief.".into()), temperature: Some(0.7), max_tokens: Some(64), seed: Some(1), top_p: None };
        let text = r#"
// a comment
{"id": "a", "prompt": "one"}
{"prompt": "two", "temperature": 0, "seed": 9, "stop": ["\n"], "system": "Own."}
{"id": 7, "messages": [{"role": "system", "content": "Sys."}, {"role": "user", "content": "three"}]}
"#;
        let rows: Vec<BatchTask> = parse_rows(text, &d).unwrap().into_iter().map(|r| r.task).collect();
        assert_eq!(rows.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(), ["a", "4", "7"]);
        assert_eq!(rows[0].messages.len(), 2);
        assert_eq!(rows[0].messages[0].role, Role::System);
        assert_eq!(rows[0].messages[0].text(), "Be brief.");
        assert_eq!((rows[0].sampling.temperature, rows[0].sampling.max_tokens, rows[0].seed), (Some(0.7), Some(64), Some(1)));
        assert_eq!(rows[1].messages[0].text(), "Own.");
        assert_eq!((rows[1].sampling.temperature, rows[1].seed), (Some(0.0), Some(9)));
        assert_eq!(rows[1].sampling.stop, vec!["\n".to_string()]);
        // A row's own system message wins over the default.
        assert_eq!(rows[2].messages.len(), 2);
        assert_eq!(rows[2].messages[0].text(), "Sys.");
    }

    #[test]
    fn bad_rows_name_their_line() {
        let d = RowDefaults::default();
        for (text, want) in [
            ("{\"prompt\": \"a\", \"messages\": [{\"role\":\"user\",\"content\":\"b\"}]}", "not both"),
            ("{\"id\": \"x\"}", "needs `prompt`"),
            ("{\"id\": \"x\", \"prompt\": \"a\"}\n{\"id\": \"x\", \"prompt\": \"b\"}", "used twice"),
            ("{\"prompt\": \"a\", \"tempreature\": 1}", "line 1"),
            ("not json", "line 1"),
            ("{\"id\": true, \"prompt\": \"a\"}", "`id` must be"),
            ("\n\n", "no rows"),
        ] {
            let e = format!("{:#}", parse_rows(text, &d).unwrap_err());
            assert!(e.contains(want), "{text}: {e}");
        }
    }

    #[test]
    fn only_fresh_teacher_answers_are_usable() {
        let t = BatchTask::prompt("r", "q");
        assert!(OutputRow::new(&t, item(true, "a", "teacher"), "teacher").usable);
        let mut cached = item(true, "a", "teacher");
        cached.cache_hit = true;
        let row = OutputRow::new(&t, cached, "teacher");
        assert!(row.ok && !row.usable && row.cache_hit);
        let mut standin = item(true, "a", "small");
        standin.standin = Some("small".into());
        assert!(!OutputRow::new(&t, standin, "teacher").usable);
        assert!(!OutputRow::new(&t, item(true, "a", "other"), "teacher").usable);
        let failed = OutputRow::new(&t, item(false, "busy", ""), "teacher");
        assert!(!failed.ok && !failed.usable && failed.completion.is_empty());
        assert_eq!(failed.error.as_deref(), Some("busy"));
    }

    #[test]
    fn resume_takes_the_last_row_per_id_and_skips_answered_ones() {
        let t = |id: &str| BatchTask::prompt(id, "q");
        let rows = [
            OutputRow::new(&t("a"), item(true, "x", "m"), "m"),
            OutputRow::new(&t("b"), item(false, "busy", ""), "m"),
            OutputRow::new(&t("b"), item(true, "y", "m"), "m"),
            OutputRow::new(&t("c"), item(false, "busy", ""), "m"),
        ];
        let mut text: String = rows.iter().map(|r| serde_json::to_string(r).unwrap() + "\n").collect();
        text.push_str("{\"id\": \"half-writ");
        let back = read_output(&text);
        assert_eq!(back.iter().map(|r| (r.id.as_str(), r.ok)).collect::<Vec<_>>(), [("a", true), ("b", true), ("c", false)]);
        let done = done_ids(&back);
        assert!(done.contains("a") && done.contains("b") && !done.contains("c"));
    }

    #[test]
    fn sources_are_checked_and_rows_must_name_them() {
        let good = r#"{"topic": "T", "sources": [{"id": "a", "title": "A", "url": "https://example.org/a", "license": "CC-BY-4.0", "kind": "paper"},
                                               {"id": "b", "title": "B", "url": "http://example.org/b", "license": "MIT"}]}"#;
        let s = Sources::parse(good, false).unwrap();
        assert_eq!(s.ids().len(), 2);
        for (bad, want) in [
            (r#"{"topic": "", "sources": [{"id": "a", "title": "A", "url": "https://x", "license": "MIT"}]}"#, "no topic"),
            (r#"{"topic": "T", "sources": []}"#, "no sources"),
            (r#"{"topic": "T", "sources": [{"id": "a", "title": "A", "url": "ftp://x", "license": "MIT"}]}"#, "http(s)"),
            (r#"{"topic": "T", "sources": [{"id": "a", "title": "A", "url": "https://x", "license": "unknown"}]}"#, "source a"),
            (
                r#"{"topic": "T", "sources": [{"id": "a", "title": "A", "url": "https://x", "license": "MIT"}, {"id": "a", "title": "B", "url": "https://y", "license": "MIT"}]}"#,
                "twice",
            ),
            (r#"{"topic": "T", "sources": [{"id": "a", "title": "A", "url": "https://x", "license": "MIT", "licence": "x"}]}"#, "not valid"),
        ] {
            let e = format!("{:#}", Sources::parse(bad, false).unwrap_err());
            assert!(e.contains(want), "{bad}: {e}");
        }
        assert!(Sources::parse(r#"{"topic": "T", "sources": [{"id": "a", "title": "A", "url": "https://x", "license": "unknown"}]}"#, true).is_ok());

        let d = RowDefaults::default();
        let rows =
            parse_rows("{\"id\": \"1\", \"prompt\": \"q\", \"source\": \"a\"}\n{\"id\": \"2\", \"prompt\": \"q\", \"source\": \"b\"}", &d).unwrap();
        assert_eq!(rows[0].source.as_deref(), Some("a"));
        assert!(check_row_sources(&rows, Some(&s)).is_ok());
        assert!(format!("{:#}", check_row_sources(&rows, None).unwrap_err()).contains("no --sources"));
        let stray = parse_rows("{\"id\": \"1\", \"prompt\": \"q\", \"source\": \"zzz\"}", &d).unwrap();
        assert!(format!("{:#}", check_row_sources(&stray, Some(&s)).unwrap_err()).contains("not in the sources manifest"));
        let none = parse_rows("{\"id\": \"1\", \"prompt\": \"q\"}", &d).unwrap();
        assert!(format!("{:#}", check_row_sources(&none, Some(&s)).unwrap_err()).contains("names no source"));
        assert!(check_row_sources(&none, None).is_ok());
    }

    #[test]
    fn the_license_must_be_given() {
        assert!(check_license("apache-2.0", false).is_ok());
        assert!(check_license(" ", false).is_err());
        assert!(check_license("unknown", false).is_err());
        assert!(check_license("Unknown", true).is_ok());
    }

    #[test]
    fn the_manifest_sits_next_to_the_output() {
        assert_eq!(manifest_path(Path::new("out/data.jsonl")), PathBuf::from("out/data.jsonl.manifest.json"));
    }
}
