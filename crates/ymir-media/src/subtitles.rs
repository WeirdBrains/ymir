//! SRT and WebVTT from timed segments.

use crate::transcript::Segment;

fn stamp(ms: u64, sep: char) -> String {
    let (h, m, s, f) = (ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000);
    format!("{h:02}:{m:02}:{s:02}{sep}{f:03}")
}

/// Cue text with blank lines removed (a blank line ends a cue in both formats).
fn cue_text(t: &str) -> String {
    t.lines().map(str::trim).filter(|l| !l.is_empty()).collect::<Vec<_>>().join("\n")
}

/// SubRip: numbered cues, `HH:MM:SS,mmm`.
pub fn srt(segments: &[Segment]) -> String {
    let mut out = String::new();
    let mut n = 0;
    for s in segments {
        let text = cue_text(&s.text);
        if text.is_empty() {
            continue;
        }
        n += 1;
        out.push_str(&format!("{n}\n{} --> {}\n{text}\n\n", stamp(s.start_ms, ','), stamp(s.end_ms.max(s.start_ms), ',')));
    }
    out
}

/// WebVTT: a `WEBVTT` header, `HH:MM:SS.mmm`, and `-->` in text escaped.
pub fn vtt(segments: &[Segment]) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for s in segments {
        let text = cue_text(&s.text).replace('&', "&amp;").replace('<', "&lt;").replace("-->", "--&gt;");
        if text.is_empty() {
            continue;
        }
        out.push_str(&format!("{} --> {}\n{text}\n\n", stamp(s.start_ms, '.'), stamp(s.end_ms.max(s.start_ms), '.')));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats() {
        let s = [Segment::new(0, 1_500, "Hello"), Segment::new(3_723_004, 3_725_000, "a <b> & c --> d\n\nnext"), Segment::new(9, 9, "  ")];
        assert_eq!(srt(&s), "1\n00:00:00,000 --> 00:00:01,500\nHello\n\n2\n01:02:03,004 --> 01:02:05,000\na <b> & c --> d\nnext\n\n");
        assert_eq!(vtt(&s), "WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nHello\n\n01:02:03.004 --> 01:02:05.000\na &lt;b> &amp; c --&gt; d\nnext\n\n");
        assert_eq!(vtt(&[]), "WEBVTT\n\n");
    }
}
