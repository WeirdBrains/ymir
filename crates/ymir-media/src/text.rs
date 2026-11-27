//! Planning long text for speech: pieces that fit a model's input limit, cut at the most natural
//! boundary available (paragraph, sentence, clause, word), script turns kept with their speaker.

use serde::{Deserialize, Serialize};

use crate::{MediaError, Result};

/// One turn of a multi-speaker script.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Turn {
    pub speaker: u8,
    pub text: String,
}

/// What to speak.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanInput {
    Text(String),
    Script(Vec<Turn>),
}

/// The boundary a piece starts after; callers pick pauses and crossfades from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Break {
    /// The first piece.
    Start,
    Paragraph,
    /// A new speaker turn in a script.
    Turn,
    Sentence,
    Clause,
    Word,
}

/// One request's worth of text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanPiece {
    /// The script speaker, `None` for plain text.
    pub speaker: Option<u8>,
    pub text: String,
    pub after: Break,
}

/// The pieces, in order. A pure function of the input and the limit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextPlan {
    pub pieces: Vec<PlanPiece>,
}

fn chars(s: &str) -> usize {
    s.chars().count()
}

fn is_sentence_end(c: char) -> bool {
    matches!(c, '.' | '!' | '?' | '…' | '。' | '！' | '？' | '؟' | '।')
}

fn is_clause_end(c: char) -> bool {
    matches!(c, ',' | ';' | ':' | '—' | '–' | '，' | '；' | '：' | '、' | '،')
}

/// Splits after each character `end` accepts when followed by whitespace or the end (or always,
/// for CJK full-width marks, which take no space after them).
fn split_after(text: &str, end: fn(char) -> bool) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut it = text.chars().peekable();
    while let Some(c) = it.next() {
        cur.push(c);
        if end(c) {
            // Keep runs like "?!" or "..." together.
            while let Some(&n) = it.peek() {
                if end(n) || matches!(n, '"' | '\'' | '”' | '’' | ')' | '»' | '」') {
                    cur.push(n);
                    it.next();
                } else {
                    break;
                }
            }
            let full_width = (c as u32) >= 0x3000;
            if full_width || it.peek().is_none_or(|n| n.is_whitespace()) {
                let t = cur.trim();
                if !t.is_empty() {
                    out.push(t.to_string());
                }
                cur.clear();
            }
        }
    }
    let t = cur.trim();
    if !t.is_empty() {
        out.push(t.to_string());
    }
    out
}

fn join(a: &str, b: &str) -> String {
    if a.is_empty() {
        return b.to_string();
    }
    // No space between full-width (CJK) pieces.
    let cjk = a.chars().last().is_some_and(|c| (c as u32) >= 0x3000) || b.chars().next().is_some_and(|c| (c as u32) >= 0x3000);
    if cjk { format!("{a}{b}") } else { format!("{a} {b}") }
}

/// Packs `units` greedily into strings of at most `max` characters; a unit that is too long is
/// split further by `finer` (and its pieces start after `finer_break`).
fn pack(units: Vec<String>, max: usize, unit_break: Break, finer: &dyn Fn(&str) -> Vec<(String, Break)>) -> Vec<(String, Break)> {
    let mut out: Vec<(String, Break)> = Vec::new();
    let mut cur = String::new();
    let mut cur_break = unit_break;
    for u in units {
        if chars(&u) > max {
            if !cur.is_empty() {
                out.push((std::mem::take(&mut cur), cur_break));
            }
            let mut first = true;
            for (p, b) in finer(&u) {
                out.push((p, if first { unit_break } else { b }));
                first = false;
            }
            cur_break = unit_break;
            continue;
        }
        let joined = join(&cur, &u);
        if chars(&joined) <= max {
            cur = joined;
        } else {
            out.push((std::mem::replace(&mut cur, u), cur_break));
            cur_break = unit_break;
        }
    }
    if !cur.is_empty() {
        out.push((cur, cur_break));
    }
    out
}

fn words(text: &str, max: usize) -> Vec<(String, Break)> {
    let units: Vec<String> = text.split_whitespace().map(str::to_string).collect();
    pack(units, max, Break::Word, &|w: &str| {
        // A "word" longer than the limit (a URL, CJK without spaces): cut by characters.
        let cs: Vec<char> = w.chars().collect();
        cs.chunks(max).map(|c| (c.iter().collect::<String>(), Break::Word)).collect()
    })
}

fn clauses(text: &str, max: usize) -> Vec<(String, Break)> {
    pack(split_after(text, is_clause_end), max, Break::Clause, &|c: &str| words(c, max))
}

fn sentences(text: &str, max: usize) -> Vec<(String, Break)> {
    pack(split_after(text, is_sentence_end), max, Break::Sentence, &|s: &str| clauses(s, max))
}

fn paragraphs(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur: Vec<&str> = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            if !cur.is_empty() {
                out.push(cur.join(" "));
                cur.clear();
            }
        } else {
            cur.push(line.trim());
        }
    }
    if !cur.is_empty() {
        out.push(cur.join(" "));
    }
    out.into_iter().map(|p| p.split_whitespace().collect::<Vec<_>>().join(" ")).collect()
}

impl TextPlan {
    /// Plans `input` into pieces of at most `max_chars` characters. `Invalid` for empty input
    /// or a limit under 16 characters.
    pub fn new(input: &PlanInput, max_chars: usize) -> Result<Self> {
        if max_chars < 16 {
            return Err(MediaError::Invalid(format!("an input limit of {max_chars} characters is too small to plan")));
        }
        let blocks: Vec<(Option<u8>, String, Break)> = match input {
            PlanInput::Text(t) => paragraphs(t).into_iter().map(|p| (None, p, Break::Paragraph)).collect(),
            PlanInput::Script(turns) => turns
                .iter()
                .flat_map(|t| {
                    paragraphs(&t.text)
                        .into_iter()
                        .enumerate()
                        .map(move |(i, p)| (Some(t.speaker), p, if i == 0 { Break::Turn } else { Break::Paragraph }))
                })
                .collect(),
        };
        let mut pieces = Vec::new();
        for (speaker, text, block_break) in blocks {
            let parts = if chars(&text) <= max_chars { vec![(text, block_break)] } else { sentences(&text, max_chars) };
            for (i, (t, b)) in parts.into_iter().enumerate() {
                let after = if pieces.is_empty() {
                    Break::Start
                } else if i == 0 {
                    block_break
                } else {
                    b
                };
                pieces.push(PlanPiece { speaker, text: t, after });
            }
        }
        if pieces.is_empty() {
            return Err(MediaError::Invalid("nothing to speak".into()));
        }
        Ok(Self { pieces })
    }

    pub fn total_chars(&self) -> usize {
        self.pieces.iter().map(|p| chars(&p.text)).sum()
    }
}

/// Parses a script: a JSON array of `{"speaker": 0, "text": "..."}`, or lines `S0: ...` (a line
/// without a prefix continues the turn before it; a blank line is a paragraph break).
pub fn parse_script(src: &str) -> Result<Vec<Turn>> {
    let trimmed = src.trim_start();
    let turns: Vec<Turn> = if trimmed.starts_with('[') {
        serde_json::from_str(trimmed).map_err(|e| MediaError::Invalid(format!("script JSON: {e}")))?
    } else {
        let mut turns: Vec<Turn> = Vec::new();
        for (n, line) in src.lines().enumerate() {
            let l = line.trim();
            let tagged = l.strip_prefix('S').and_then(|rest| {
                let (num, text) = rest.split_once(':')?;
                Some((num.trim().parse::<u8>().ok()?, text.trim()))
            });
            match (tagged, turns.last_mut()) {
                (Some((speaker, text)), _) => turns.push(Turn { speaker, text: text.to_string() }),
                (None, Some(t)) => {
                    t.text.push('\n');
                    t.text.push_str(l);
                }
                (None, None) if l.is_empty() => {}
                (None, None) => return Err(MediaError::Invalid(format!("script line {}: expected `S<speaker>: text`", n + 1))),
            }
        }
        turns
    };
    let turns: Vec<Turn> = turns.into_iter().filter(|t| !t.text.trim().is_empty()).collect();
    if turns.is_empty() {
        return Err(MediaError::Invalid("the script has no turns".into()));
    }
    Ok(turns)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(t: &str, max: usize) -> Vec<(String, Break)> {
        TextPlan::new(&PlanInput::Text(t.into()), max).unwrap().pieces.into_iter().map(|p| (p.text, p.after)).collect()
    }

    #[test]
    fn short_text_is_one_piece() {
        assert_eq!(plan("Hello  world.\nSecond line.", 100), vec![("Hello world. Second line.".into(), Break::Start)]);
    }

    #[test]
    fn paragraphs_then_sentences_then_clauses_then_words() {
        let t = "First paragraph here.\n\nOne sentence. Two sentence! Three, with a clause; and another clause here at the end.";
        let p = plan(t, 30);
        assert_eq!(p[0], ("First paragraph here.".into(), Break::Start));
        assert_eq!(p[1], ("One sentence. Two sentence!".into(), Break::Paragraph));
        assert_eq!(p[2].1, Break::Sentence);
        assert!(p.iter().all(|(s, _)| s.chars().count() <= 30));
        assert!(p.iter().any(|(_, b)| *b == Break::Clause));
        let words: Vec<&str> = t.split_whitespace().collect();
        let back: Vec<String> = p.iter().map(|(s, _)| s.clone()).collect();
        assert_eq!(back.join(" ").split_whitespace().collect::<Vec<_>>(), words, "no text lost or reordered");
        assert_eq!(plan(t, 30), p, "deterministic");
    }

    #[test]
    fn never_splits_inside_a_word_unless_it_must() {
        let p = plan("supercalifragilisticexpialidocious and more words follow it here", 16);
        assert!(p.iter().all(|(s, _)| s.chars().count() <= 16));
        assert_eq!(p[0].0, "supercalifragili");
        assert_eq!(p[2].0, "us");
        assert!(p[3..].iter().all(|(s, _)| s.split_whitespace().all(|w| "and more words follow it here".contains(w))));
    }

    #[test]
    fn keeps_quotes_and_ellipses_and_cjk() {
        let p = plan("He said \"stop!\" Then... nothing. Done.", 20);
        assert_eq!(p[0].0, "He said \"stop!\"");
        let p = plan("今天天气很好。我们去公园吧！好的。", 16);
        assert_eq!(p[0].0, "今天天气很好。我们去公园吧！");
        assert_eq!(p[1].0, "好的。");
    }

    #[test]
    fn scripts_keep_speakers() {
        let turns = vec![Turn { speaker: 0, text: "Hi there. How are you doing today, friend?".into() }, Turn { speaker: 3, text: "Fine.".into() }];
        let p = TextPlan::new(&PlanInput::Script(turns), 20).unwrap().pieces;
        assert!(p.iter().take(p.len() - 1).all(|x| x.speaker == Some(0)));
        assert_eq!(p.last().unwrap(), &PlanPiece { speaker: Some(3), text: "Fine.".into(), after: Break::Turn });
        assert_eq!(p[0].after, Break::Start);
    }

    #[test]
    fn refusals() {
        assert!(TextPlan::new(&PlanInput::Text("  \n ".into()), 100).is_err());
        assert!(TextPlan::new(&PlanInput::Text("x".into()), 5).is_err());
    }

    #[test]
    fn parses_scripts() {
        let s = parse_script("S0: Hello.\nstill speaker zero\n\nS1: Hi!\nS15:Bye").unwrap();
        assert_eq!(s.len(), 3);
        assert_eq!(s[0], Turn { speaker: 0, text: "Hello.\nstill speaker zero\n".into() });
        assert_eq!(s[2], Turn { speaker: 15, text: "Bye".into() });
        let j = parse_script(r#"[{"speaker":2,"text":"a"},{"speaker":1,"text":" "}]"#).unwrap();
        assert_eq!(j, vec![Turn { speaker: 2, text: "a".into() }]);
        assert!(parse_script("no prefix").is_err());
        assert!(parse_script("S300: too big").is_err());
        assert!(parse_script("[{\"speaker\":-1}]").is_err());
        assert!(parse_script("").is_err());
    }
}
