//! Seed prompts from sources (training plan P1): the operator's fetched source texts become
//! prompt rows for `ymirctl batch --jsonl`, in two passes through the teacher.
//!
//! 1. [`question_rows`]: each source's text (`DIR/ID.txt` or `DIR/ID.md`, one file per source in
//!    the sources manifest; Ymir fetches nothing) is cut into excerpts at paragraph breaks, and
//!    each excerpt becomes a row asking the teacher for questions the excerpt answers.
//! 2. [`prompts_from_answers`]: the teacher's usable answers to those rows are read back, and each
//!    `Question:` line becomes a plain prompt row, still naming its source.
//!
//! The second file then goes through `batch --jsonl` again, so the dataset's prompts stand alone
//! (no excerpt) and every row keeps the source it came from, with its license.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

use crate::dataset::Sources;

/// How excerpts are cut and how many questions each asks for.
#[derive(Debug, Clone, PartialEq)]
pub struct Seeding {
    /// The longest excerpt, in characters (paragraphs are kept whole when they fit).
    pub chunk_chars: usize,
    /// Excerpts shorter than this are dropped (headings, stray lines).
    pub min_chars: usize,
    /// Questions asked for per excerpt.
    pub questions: u32,
    /// At most this many excerpts per source.
    pub max_per_source: Option<usize>,
}

impl Default for Seeding {
    fn default() -> Self {
        Self { chunk_chars: 2400, min_chars: 200, questions: 3, max_per_source: None }
    }
}

/// What a pass made: rows written, per source.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SeedSummary {
    pub rows: usize,
    pub per_source: Vec<(String, usize)>,
    /// Pass 2: rows skipped (not usable, or no source).
    pub skipped: usize,
    /// Pass 2: questions dropped as duplicates.
    pub duplicates: usize,
}

const SYSTEM: &str = "You write study questions about a source text. Every question must be answerable from the excerpt alone, stand on its own without the excerpt (no \"this text\" or \"the author\"), and be one a newcomer to the topic would ask.";

/// The file holding source `id`'s text under `dir`.
pub fn text_file(dir: &Path, id: &str) -> Option<PathBuf> {
    ["txt", "md"].iter().map(|e| dir.join(format!("{id}.{e}"))).find(|p| p.is_file())
}

/// Cuts `text` into excerpts of at most `max` characters at paragraph breaks; a paragraph longer
/// than `max` is cut at whitespace. Excerpts under `min` characters are dropped.
pub fn excerpts(text: &str, max: usize, min: usize) -> Vec<String> {
    let max = max.max(1);
    let text = text.replace("\r\n", "\n");
    let mut pieces = Vec::new();
    for para in text.split("\n\n").map(str::trim).filter(|p| !p.is_empty()) {
        if para.chars().count() <= max {
            pieces.push(para.to_string());
            continue;
        }
        let mut cur = String::new();
        for word in para.split_whitespace() {
            if !cur.is_empty() && cur.chars().count() + 1 + word.chars().count() > max {
                pieces.push(std::mem::take(&mut cur));
            }
            if !cur.is_empty() {
                cur.push(' ');
            }
            cur.push_str(word);
        }
        if !cur.is_empty() {
            pieces.push(cur);
        }
    }
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for p in pieces {
        if !cur.is_empty() && cur.chars().count() + 2 + p.chars().count() > max {
            out.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push_str("\n\n");
        }
        cur.push_str(&p);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out.retain(|e| e.chars().count() >= min);
    out
}

/// Pass 1: one row per excerpt of every source's text, asking for `questions` questions. Returns
/// the JSONL text and what it holds. Every source in the manifest needs a text file.
pub fn question_rows(sources: &Sources, texts: &Path, s: &Seeding) -> Result<(String, SeedSummary)> {
    if s.questions == 0 || s.questions > 20 {
        bail!("--questions must be 1 to 20");
    }
    let missing: Vec<&str> = sources.sources.iter().filter(|x| text_file(texts, &x.id).is_none()).map(|x| x.id.as_str()).collect();
    if !missing.is_empty() {
        bail!("no text for source(s) {} in {} (one ID.txt or ID.md per source)", missing.join(", "), texts.display());
    }
    let mut out = format!("// Seed questions for {:?}, made by ymirctl train seed questions.\n", sources.topic);
    let mut sum = SeedSummary::default();
    for src in &sources.sources {
        let path = text_file(texts, &src.id).expect("checked");
        let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        let mut ex = excerpts(&text, s.chunk_chars, s.min_chars);
        if let Some(m) = s.max_per_source {
            ex.truncate(m);
        }
        if ex.is_empty() {
            bail!("source {}: {} has no excerpt of at least {} characters", src.id, path.display(), s.min_chars);
        }
        for (n, e) in ex.iter().enumerate() {
            let prompt = format!(
                "Topic: {topic}\nSource: {title}\n\nExcerpt:\n---\n{e}\n---\n\nWrite {q} different questions that this excerpt answers. Put each on its own line, starting with \"Question: \". Write nothing else.",
                topic = sources.topic,
                title = src.title,
                q = s.questions,
            );
            let row = json!({"id": format!("q-{}-{:03}", src.id, n + 1), "source": src.id, "system": SYSTEM, "prompt": prompt});
            out.push_str(&serde_json::to_string(&row)?);
            out.push('\n');
        }
        sum.rows += ex.len();
        sum.per_source.push((src.id.clone(), ex.len()));
    }
    Ok((out, sum))
}

/// The questions in one answer: lines starting `Question:` (or `Q:`), after any list marker.
pub fn questions_in(answer: &str) -> Vec<String> {
    answer
        .lines()
        .filter_map(|l| {
            let l = l.trim().trim_start_matches(|c: char| c.is_ascii_digit() || matches!(c, '.' | ')' | '-' | '*' | ' '));
            let l = l.trim_start_matches("**");
            let rest = ["Question:", "question:", "Q:"].iter().find_map(|p| l.strip_prefix(p))?;
            let q = rest.trim().trim_matches('*').trim();
            (q.chars().count() >= 8).then(|| q.to_string())
        })
        .collect()
}

/// Pass 2: the questions in pass 1's answers (`batch --jsonl` output) as prompt rows, each naming
/// its row's source. Rows that aren't `usable`, or name no source, are skipped; a question asked
/// twice (ignoring case and spacing) is kept once.
pub fn prompts_from_answers(answers: &str) -> Result<(String, SeedSummary)> {
    let mut out = String::from("// Seed prompts, made by ymirctl train seed prompts.\n");
    let mut sum = SeedSummary::default();
    let mut seen = HashSet::new();
    let mut per: Vec<(String, usize)> = Vec::new();
    for (n, line) in answers.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        let r: Value = serde_json::from_str(line).with_context(|| format!("line {}: not a JSON row", n + 1))?;
        let (Some(id), Some(src), Some(text)) = (r["id"].as_str(), r["source"].as_str(), r["completion"].as_str()) else {
            sum.skipped += 1;
            continue;
        };
        if r["usable"] != Value::Bool(true) {
            sum.skipped += 1;
            continue;
        }
        for (k, q) in questions_in(text).into_iter().enumerate() {
            let key = q.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
            if !seen.insert(key) {
                sum.duplicates += 1;
                continue;
            }
            let id = format!("{}-{}", id.strip_prefix("q-").unwrap_or(id), k + 1);
            out.push_str(&serde_json::to_string(&json!({"id": id, "source": src, "prompt": q}))?);
            out.push('\n');
            sum.rows += 1;
            match per.iter_mut().find(|(s, _)| s == src) {
                Some((_, c)) => *c += 1,
                None => per.push((src.to_string(), 1)),
            }
        }
    }
    if sum.rows == 0 {
        bail!("no `Question:` lines in any usable answer");
    }
    sum.per_source = per;
    Ok((out, sum))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{RowDefaults, parse_rows};

    #[test]
    fn excerpts_keep_paragraphs_and_cut_long_ones() {
        let text = "Title\n\nFirst paragraph is here.\r\n\r\nSecond paragraph follows it.\n\n".to_string() + &"word ".repeat(30);
        let ex = excerpts(&text, 60, 10);
        assert_eq!(ex[0], "Title\n\nFirst paragraph is here.");
        assert_eq!(ex[1], "Second paragraph follows it.");
        assert!(ex[2..].iter().all(|e| e.chars().count() <= 60 && e.starts_with("word")), "{ex:?}");
        assert_eq!(ex[2..].iter().map(|e| e.split_whitespace().count()).sum::<usize>(), 30);
        assert!(excerpts("tiny", 60, 10).is_empty());
    }

    #[test]
    fn questions_are_read_from_answer_lines() {
        let a = "Question: What is a DHT?\n1. Question: How do peers find each other?\n**Question:** Why use a relay at all?\nAnswer: no\nQ: Short?\nQ: What does hole punching do?";
        assert_eq!(questions_in(a), ["What is a DHT?", "How do peers find each other?", "Why use a relay at all?", "What does hole punching do?"]);
    }

    #[test]
    fn two_passes_make_rows_batch_accepts() {
        let dir = tempfile::tempdir().unwrap();
        let sources = Sources::parse(
            r#"{"topic": "Networks", "sources": [{"id": "a", "title": "A", "url": "https://example.org/a", "license": "CC0-1.0"}, {"id": "b", "title": "B", "url": "https://example.org/b", "license": "MIT"}]}"#,
            false,
        )
        .unwrap();
        std::fs::write(dir.path().join("a.txt"), "x".repeat(250)).unwrap();
        let s = Seeding::default();
        let e = question_rows(&sources, dir.path(), &s).unwrap_err();
        assert!(format!("{e:#}").contains("no text for source(s) b"), "{e:#}");
        std::fs::write(dir.path().join("b.md"), format!("{}\n\n{}", "y ".repeat(150), "z ".repeat(150))).unwrap();
        let (text, sum) = question_rows(&sources, dir.path(), &Seeding { chunk_chars: 320, ..s }).unwrap();
        assert_eq!(sum.per_source, [("a".to_string(), 1), ("b".to_string(), 2)]);
        let rows = parse_rows(&text, &RowDefaults::default()).unwrap();
        assert_eq!(rows.iter().map(|r| r.task.id.as_str()).collect::<Vec<_>>(), ["q-a-001", "q-b-001", "q-b-002"]);
        assert_eq!(rows[1].source.as_deref(), Some("b"));

        let answers = [
            json!({"id": "q-a-001", "source": "a", "usable": true, "completion": "Question: What is x for?\nQuestion: How long is x?"}),
            json!({"id": "q-b-001", "source": "b", "usable": true, "completion": "Question: what is  X for?\nQuestion: Why is y useful?"}),
            json!({"id": "q-b-002", "source": "b", "usable": false, "completion": "Question: Not used?"}),
        ]
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join("\n");
        let (text, sum) = prompts_from_answers(&answers).unwrap();
        assert_eq!((sum.rows, sum.skipped, sum.duplicates), (3, 1, 1));
        let rows = parse_rows(&text, &RowDefaults::default()).unwrap();
        assert_eq!(rows.iter().map(|r| r.task.id.as_str()).collect::<Vec<_>>(), ["a-001-1", "a-001-2", "b-001-2"]);
        assert_eq!(rows[2].source.as_deref(), Some("b"));
        assert!(prompts_from_answers("{\"id\": \"x\", \"source\": \"a\", \"usable\": true, \"completion\": \"no\"}").is_err());
    }
}
