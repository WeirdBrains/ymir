//! Re-aligning the transcripts of a chunked clip onto the clip's own timeline.

use serde::{Deserialize, Serialize};

use crate::vad::Chunk;

/// A timed piece of a transcript.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Segment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    /// The backend gave no timing: the bounds are the chunk's, not the words'.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub coarse: bool,
}

impl Segment {
    pub fn new(start_ms: u64, end_ms: u64, text: impl Into<String>) -> Self {
        Self { start_ms, end_ms, text: text.into(), coarse: false }
    }
}

/// One chunk's transcript, as a backend returned it (times relative to the chunk).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Piece {
    pub text: String,
    pub segments: Vec<Segment>,
}

/// The whole clip's transcript.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Joined {
    pub text: String,
    pub segments: Vec<Segment>,
    /// Milliseconds of audio covered.
    pub duration_ms: u64,
}

/// Joins `pieces` (one per chunk, in order): the text with single spaces, segments shifted by
/// their chunk's start and clamped to it; a chunk with text but no segments gets one coarse
/// segment spanning the chunk.
pub fn realign(chunks: &[Chunk], pieces: &[Piece]) -> Joined {
    let mut out = Joined::default();
    for (chunk, piece) in chunks.iter().zip(pieces) {
        let text = piece.text.trim();
        if !text.is_empty() {
            if !out.text.is_empty() {
                out.text.push(' ');
            }
            out.text.push_str(text);
        }
        if piece.segments.is_empty() {
            if !text.is_empty() {
                out.segments.push(Segment { start_ms: chunk.start_ms, end_ms: chunk.end_ms, text: text.to_string(), coarse: true });
            }
        } else {
            for s in &piece.segments {
                let start = (chunk.start_ms + s.start_ms).min(chunk.end_ms);
                let end = (chunk.start_ms + s.end_ms).clamp(start, chunk.end_ms);
                out.segments.push(Segment { start_ms: start, end_ms: end, text: s.text.trim().to_string(), coarse: s.coarse });
            }
        }
        out.duration_ms = chunk.end_ms;
    }
    out
}

/// The tail of `text` to guide the next chunk (at most `max_chars` characters, from a word start).
pub fn prompt_tail(text: &str, max_chars: usize) -> String {
    let t = text.trim();
    let n = t.chars().count();
    if n <= max_chars {
        return t.to_string();
    }
    let tail: String = t.chars().skip(n - max_chars).collect();
    match tail.find(char::is_whitespace) {
        Some(i) => tail[i..].trim_start().to_string(),
        None => tail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(start_ms: u64, end_ms: u64) -> Chunk {
        Chunk { start: (start_ms * 16) as usize, end: (end_ms * 16) as usize, start_ms, end_ms }
    }

    #[test]
    fn shifts_and_clamps() {
        let chunks = [chunk(0, 30_000), chunk(30_000, 55_000)];
        let pieces = [
            Piece { text: " Hello there. ".into(), segments: vec![Segment::new(0, 1_200, "Hello"), Segment::new(1_200, 31_000, "there.")] },
            Piece { text: "General Kenobi.".into(), segments: vec![Segment::new(500, 2_000, " General Kenobi. ")] },
        ];
        let j = realign(&chunks, &pieces);
        assert_eq!(j.text, "Hello there. General Kenobi.");
        assert_eq!(j.duration_ms, 55_000);
        assert_eq!(
            j.segments,
            vec![Segment::new(0, 1_200, "Hello"), Segment::new(1_200, 30_000, "there."), Segment::new(30_500, 32_000, "General Kenobi.")]
        );
    }

    #[test]
    fn untimed_chunks_get_coarse_segments() {
        let j = realign(
            &[chunk(0, 30_000), chunk(30_000, 40_000), chunk(40_000, 41_000)],
            &[
                Piece { text: "one".into(), segments: vec![] },
                Piece { text: "two".into(), segments: vec![] },
                Piece { text: "  ".into(), segments: vec![] },
            ],
        );
        assert_eq!(j.segments.len(), 2);
        assert!(j.segments.iter().all(|s| s.coarse));
        assert_eq!((j.segments[1].start_ms, j.segments[1].end_ms), (30_000, 40_000));
        assert_eq!(j.duration_ms, 41_000);
    }

    #[test]
    fn prompt_tail_starts_at_a_word() {
        assert_eq!(prompt_tail("short", 200), "short");
        assert_eq!(prompt_tail("alpha beta gamma delta", 12), "gamma delta");
        assert_eq!(prompt_tail("ééééé ü", 3), "ü");
    }
}
