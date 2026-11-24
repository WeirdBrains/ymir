//! Training data from conversations, with consent (training plan P2,
//! `ymirctl context export --training`).
//!
//! Only conversations whose `{conversation}/train` record is on are read (see
//! [`crate::context::Train`]); deleted ones never are. Each becomes one chat-format JSONL row
//! after the strip rules:
//!
//! - system prompts (the conversation's and any `system` message) are dropped unless kept;
//! - tool calls and results, attachments and reasoning are dropped (only message text is kept);
//! - the conversation is cut at the first assistant answer that is not a complete answer by the
//!   model asked (stopped, failed, still streaming, or a stand-in's), and trailing user turns
//!   are dropped, so every row ends on an answer;
//! - every `--redact` pattern (a regular expression) is replaced with `[redacted]` in what is kept.
//!
//! The output is a copy outside the encrypted store: it is written with owner-only permissions,
//! never synced, and the tool reports counts, not content. The manifest next to it
//! (`OUT.manifest.json`) lists the conversation ids and their consent and update times, so a
//! later export or `ymirctl context withdrawn` can tell which were withdrawn or deleted since.

use std::collections::BTreeSet;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::context::RawConversation;

/// The manifest's format name.
pub const FORMAT: &str = "ymir-conversations/1";
const REDACTED: &str = "[redacted]";

/// What to strip.
#[derive(Debug, Clone, Default)]
pub struct Rules {
    pub keep_system: bool,
    /// Regular expressions; matches become `[redacted]`.
    pub redact: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatTurn {
    pub role: String,
    pub content: String,
}

/// One output row: a conversation as chat turns.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Row {
    pub id: String,
    pub messages: Vec<ChatTurn>,
    /// The models that wrote the kept answers.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<String>,
}

/// What the strip rules removed, by kind.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dropped {
    pub system: usize,
    pub tool: usize,
    pub attachments: usize,
    pub reasoning: usize,
    /// Answers that were stopped, failed or still streaming (the conversation is cut there).
    pub incomplete: usize,
    /// Answers by a stand-in model (the conversation is cut there).
    pub standin: usize,
    /// Messages after a cut, and trailing user turns.
    pub after_cut: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestConversation {
    pub id: String,
    /// When consent was given.
    pub consent_ms: i64,
    /// The newest message or meta write.
    pub updated_ms: i64,
    /// Turns kept (0: nothing usable, no row written).
    pub turns: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RulesRecord {
    pub keep_system: bool,
    /// How many redact patterns were applied and their SHA-256 (the patterns themselves may
    /// name what they hide, so they are not written).
    pub redact_patterns: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub redact_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub output_name: String,
    pub output_sha256: String,
    pub rules: RulesRecord,
    pub conversations: Vec<ManifestConversation>,
    pub conversations_flagged: usize,
    pub conversations_exported: usize,
    pub turns_exported: usize,
    pub dropped: Dropped,
    pub redactions: usize,
    pub created_ms: i64,
}

fn text(v: &Value, key: &str) -> String {
    v.get(key).and_then(Value::as_str).unwrap_or_default().to_string()
}

fn count(v: &Value, key: &str) -> usize {
    match v.get(key) {
        Some(Value::Array(a)) => a.len(),
        Some(Value::String(s)) if !s.is_empty() => 1,
        _ => 0,
    }
}

struct Redactor {
    res: Vec<regex::Regex>,
    hits: usize,
}

impl Redactor {
    fn apply(&mut self, s: &str) -> String {
        let mut out = s.to_string();
        for re in &self.res {
            let n = re.find_iter(&out).count();
            if n > 0 {
                self.hits += n;
                out = re.replace_all(&out, REDACTED).into_owned();
            }
        }
        out
    }
}

/// Applies the strip rules to one conversation.
fn strip(c: &RawConversation, keep_system: bool, red: &mut Redactor, d: &mut Dropped) -> Row {
    let mut turns = Vec::new();
    let mut models = BTreeSet::new();
    match c.meta.system.as_deref().filter(|s| !s.is_empty()) {
        Some(s) if keep_system => turns.push(ChatTurn { role: "system".into(), content: red.apply(s) }),
        Some(_) => d.system += 1,
        None => {}
    }
    let mut cut = None;
    for (i, (_, v)) in c.messages.iter().enumerate() {
        let role = text(v, "role");
        d.attachments += count(v, "attachments");
        match role.as_str() {
            "system" if keep_system => turns.push(ChatTurn { role: role.clone(), content: red.apply(&text(v, "text")) }),
            "system" => d.system += 1,
            "user" => {
                let t = text(v, "text");
                if !t.is_empty() {
                    turns.push(ChatTurn { role, content: red.apply(&t) });
                }
            }
            "assistant" => {
                let state = v.get("state").and_then(Value::as_str).unwrap_or("complete");
                if state != "complete" {
                    d.incomplete += 1;
                    cut = Some(i);
                    break;
                }
                if v.get("standin").is_some_and(|s| !s.is_null()) {
                    d.standin += 1;
                    cut = Some(i);
                    break;
                }
                d.tool += count(v, "tools");
                d.reasoning += count(v, "reasoning");
                if let Some(m) = v.get("model").and_then(Value::as_str) {
                    models.insert(m.to_string());
                }
                turns.push(ChatTurn { role, content: red.apply(&text(v, "text")) });
            }
            _ => d.tool += 1,
        }
    }
    if let Some(i) = cut {
        d.after_cut += c.messages.len() - i - 1;
    }
    while turns.last().is_some_and(|t| t.role == "user") {
        turns.pop();
        d.after_cut += 1;
    }
    if !turns.iter().any(|t| t.role == "assistant") {
        turns.clear();
    }
    Row { id: c.id.clone(), messages: turns, models: models.into_iter().collect() }
}

fn updated_ms(c: &RawConversation) -> i64 {
    c.messages.iter().filter_map(|(_, v)| v.get("at_ms").and_then(Value::as_i64)).max().unwrap_or(0).max(c.meta.at_ms)
}

/// Builds the rows and the manifest (without the output's name and hash).
pub fn build(conversations: &[RawConversation], rules: &Rules) -> Result<(Vec<Row>, Manifest)> {
    let res = rules
        .redact
        .iter()
        .enumerate()
        .map(|(i, p)| regex::Regex::new(p).with_context(|| format!("redact pattern #{}", i + 1)))
        .collect::<Result<Vec<_>>>()?;
    let mut red = Redactor { res, hits: 0 };
    let mut dropped = Dropped::default();
    let mut rows = Vec::new();
    let mut listed = Vec::new();
    for c in conversations.iter().filter(|c| c.train.on && !c.meta.deleted) {
        let row = strip(c, rules.keep_system, &mut red, &mut dropped);
        listed.push(ManifestConversation { id: c.id.clone(), consent_ms: c.train.at_ms, updated_ms: updated_ms(c), turns: row.messages.len() });
        if !row.messages.is_empty() {
            rows.push(row);
        }
    }
    let redact_sha256 = (!rules.redact.is_empty()).then(|| hex(&Sha256::digest(rules.redact.join("\n").as_bytes())));
    let manifest = Manifest {
        format: FORMAT.into(),
        output_name: String::new(),
        output_sha256: String::new(),
        rules: RulesRecord { keep_system: rules.keep_system, redact_patterns: rules.redact.len(), redact_sha256 },
        conversations_flagged: listed.len(),
        conversations_exported: rows.len(),
        turns_exported: rows.iter().map(|r| r.messages.len()).sum(),
        conversations: listed,
        dropped,
        redactions: red.hits,
        created_ms: now_ms(),
    };
    Ok((rows, manifest))
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

/// Writes `bytes` to `path` with owner-only permissions, through a temporary file in the same
/// directory, so a reader never sees half a file.
fn write_private(path: &Path, bytes: &[u8]) -> Result<()> {
    let dir = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let name = path.file_name().context("the output needs a file name")?.to_string_lossy().into_owned();
    let tmp = dir.join(format!(".{name}.tmp-{}", std::process::id()));
    let mut o = std::fs::OpenOptions::new();
    o.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(0o600);
    }
    let mut f = o.open(&tmp).with_context(|| format!("writing {}", tmp.display()))?;
    f.write_all(bytes)?;
    f.sync_all()?;
    drop(f);
    std::fs::rename(&tmp, path).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// The manifest's path for an output file: `OUT.manifest.json`.
pub fn manifest_path(out: &Path) -> std::path::PathBuf {
    crate::dataset::manifest_path(out)
}

/// Exports `conversations` to `out` (JSONL) and its manifest, both owner-only. Returns the
/// manifest.
pub fn export(conversations: &[RawConversation], rules: &Rules, out: &Path) -> Result<Manifest> {
    let (rows, mut manifest) = build(conversations, rules)?;
    let mut body = Vec::new();
    for r in &rows {
        serde_json::to_writer(&mut body, r)?;
        body.push(b'\n');
    }
    manifest.output_name = out.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    manifest.output_sha256 = hex(&Sha256::digest(&body));
    write_private(out, &body)?;
    write_private(&manifest_path(out), &serde_json::to_vec_pretty(&manifest)?)?;
    Ok(manifest)
}

/// The conversations an earlier export listed that are no longer for training here (consent
/// withdrawn, or deleted): a dataset built from that export should be rebuilt without them.
pub fn withdrawn(earlier: &Manifest, now: &[RawConversation]) -> Vec<String> {
    let live: BTreeSet<&str> = now.iter().filter(|c| c.train.on && !c.meta.deleted).map(|c| c.id.as_str()).collect();
    earlier.conversations.iter().filter(|c| !live.contains(c.id.as_str())).map(|c| c.id.clone()).collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::context::{Meta, Train};

    fn conv(id: &str, system: Option<&str>, msgs: Vec<Value>) -> RawConversation {
        RawConversation {
            id: id.into(),
            meta: Meta { title: "t".into(), system: system.map(str::to_string), ..Default::default() },
            train: Train { on: true, at_ms: 5 },
            messages: msgs.into_iter().enumerate().map(|(i, v)| (format!("{i}"), v)).collect(),
        }
    }

    #[test]
    fn strips_system_tools_attachments_and_cuts_at_a_bad_answer() {
        let c = conv(
            "c1",
            Some("secret system prompt"),
            vec![
                json!({"role": "user", "text": "my email is a@b.example", "at_ms": 1, "attachments": [{"name": "x.txt"}]}),
                json!({"role": "assistant", "text": "ok", "model": "m", "state": "complete", "tools": ["read"], "reasoning": "hmm", "at_ms": 2}),
                json!({"role": "user", "text": "next", "at_ms": 3}),
                json!({"role": "assistant", "text": "half", "state": "stopped", "at_ms": 4}),
                json!({"role": "user", "text": "after", "at_ms": 5}),
            ],
        );
        let rules = Rules { keep_system: false, redact: vec![r"[\w.]+@[\w.]+".into()] };
        let (rows, m) = build(&[c], &rules).unwrap();
        assert_eq!(rows.len(), 1);
        let r = &rows[0];
        assert_eq!(r.messages.len(), 2);
        assert_eq!(r.messages[0].content, "my email is [redacted]");
        assert_eq!(r.messages[1], ChatTurn { role: "assistant".into(), content: "ok".into() });
        assert_eq!(r.models, ["m"]);
        assert_eq!(m.dropped, Dropped { system: 1, tool: 1, attachments: 1, reasoning: 1, incomplete: 1, standin: 0, after_cut: 2 });
        assert_eq!(m.redactions, 1);
        assert_eq!(m.rules.redact_patterns, 1);
        assert!(!serde_json::to_string(&m).unwrap().contains("@"), "the manifest never holds the patterns or content");
        assert_eq!(m.conversations[0].updated_ms, 5);
    }

    #[test]
    fn a_standin_answer_cuts_and_a_conversation_with_no_answer_is_listed_but_not_written() {
        let a = conv(
            "a",
            Some("sys"),
            vec![json!({"role": "user", "text": "q", "at_ms": 1}), json!({"role": "assistant", "text": "fast", "standin": "tiny", "at_ms": 2})],
        );
        let b = conv("b", None, vec![json!({"role": "user", "text": "q", "at_ms": 1}), json!({"role": "assistant", "text": "a", "at_ms": 2})]);
        let (rows, m) = build(&[a, b], &Rules { keep_system: true, redact: vec![] }).unwrap();
        assert_eq!(rows.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(), ["b"]);
        assert_eq!(m.conversations_flagged, 2);
        assert_eq!(m.conversations_exported, 1);
        assert_eq!(m.conversations[0].turns, 0);
        assert_eq!(m.dropped.standin, 1);
    }

    #[test]
    fn keeps_the_system_prompt_when_asked() {
        let c = conv("c", Some("Be brief."), vec![json!({"role": "user", "text": "q"}), json!({"role": "assistant", "text": "a"})]);
        let (rows, _) = build(&[c], &Rules { keep_system: true, redact: vec![] }).unwrap();
        assert_eq!(rows[0].messages[0].role, "system");
        assert_eq!(rows[0].messages.len(), 3);
    }

    #[test]
    fn a_bad_pattern_is_an_error() {
        assert!(build(&[], &Rules { keep_system: false, redact: vec!["(".into()] }).is_err());
    }

    #[test]
    fn export_is_owner_only_and_withdrawn_lists_what_left() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("train.jsonl");
        let a = conv("a", None, vec![json!({"role": "user", "text": "q"}), json!({"role": "assistant", "text": "a"})]);
        let b = conv("b", None, vec![json!({"role": "user", "text": "q"}), json!({"role": "assistant", "text": "a"})]);
        let m = export(&[a.clone(), b], &Rules::default(), &out).unwrap();
        assert_eq!(m.output_name, "train.jsonl");
        let body = std::fs::read(&out).unwrap();
        assert_eq!(m.output_sha256, hex(&Sha256::digest(&body)));
        assert_eq!(body.iter().filter(|&&c| c == b'\n').count(), 2);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&out).unwrap().permissions().mode() & 0o777, 0o600);
            assert_eq!(std::fs::metadata(manifest_path(&out)).unwrap().permissions().mode() & 0o777, 0o600);
        }
        let back: Manifest = serde_json::from_slice(&std::fs::read(manifest_path(&out)).unwrap()).unwrap();
        assert_eq!(back, m);
        assert_eq!(withdrawn(&m, &[a]), ["b"]);
    }
}
