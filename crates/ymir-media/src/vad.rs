//! Silence-aware chunking of long audio for speech-to-text models with a window.

use crate::{MediaError, Pcm, Result};

/// RMS frame for the energy scan.
pub const FRAME_MS: u32 = 30;

/// One piece of a long clip, in frames of the source and in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chunk {
    pub start: usize,
    pub end: usize,
    pub start_ms: u64,
    pub end_ms: u64,
}

/// Splits `pcm` into pieces of at most `max_ms` each, without overlap. Each cut goes at the
/// quietest 30 ms frame in the last fifth of the window, so words are not cut where the audio
/// allows it; a clip with no quiet place is cut at the window's end. Deterministic: the same
/// samples and limit give the same plan.
pub fn plan(pcm: &Pcm, max_ms: u64) -> Result<Vec<Chunk>> {
    if max_ms < u64::from(FRAME_MS) * 10 {
        return Err(MediaError::Invalid(format!("a chunk limit of {max_ms} ms is too short to plan (at least {} ms)", FRAME_MS * 10)));
    }
    let rate = pcm.sample_rate.max(1) as usize;
    let mono = pcm.to_mono();
    let frames = mono.frames();
    let ms = |f: usize| f as u64 * 1000 / rate as u64;
    let window = (max_ms as usize).saturating_mul(rate) / 1000;
    let hop = (FRAME_MS as usize * rate / 1000).max(1);
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < frames {
        if frames - start <= window {
            chunks.push(Chunk { start, end: frames, start_ms: ms(start), end_ms: ms(frames) });
            break;
        }
        let limit = start + window;
        let search_from = start + window * 4 / 5;
        let mut best = (f32::INFINITY, limit);
        let mut f = search_from;
        while f + hop <= limit {
            let s = &mono.samples[f..f + hop];
            let e = s.iter().map(|x| x * x).sum::<f32>() / hop as f32;
            // Ties go to the later frame: longer pieces, fewer of them.
            if e <= best.0 {
                best = (e, f + hop / 2);
            }
            f += hop;
        }
        let cut = best.1.clamp(start + 1, limit);
        chunks.push(Chunk { start, end: cut, start_ms: ms(start), end_ms: ms(cut) });
        start = cut;
    }
    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Speech-like bursts: `on` ms of tone, `off` ms of silence, repeated.
    fn bursts(rate: u32, on: u32, off: u32, total_ms: u32) -> Pcm {
        let n = (rate * total_ms / 1000) as usize;
        let period = (on + off) as usize * rate as usize / 1000;
        let on_n = on as usize * rate as usize / 1000;
        Pcm::new(rate, 1, (0..n).map(|i| if i % period < on_n { (i as f32 * 0.3).sin() * 0.6 } else { 0.0 }).collect())
    }

    #[test]
    fn short_clip_is_one_chunk() {
        let p = bursts(16_000, 500, 200, 3_000);
        let c = plan(&p, 30_000).unwrap();
        assert_eq!(c, vec![Chunk { start: 0, end: 48_000, start_ms: 0, end_ms: 3_000 }]);
    }

    #[test]
    fn cuts_land_in_silence_and_cover_everything() {
        let p = bursts(16_000, 1_700, 300, 95_000);
        let c = plan(&p, 30_000).unwrap();
        assert!(c.len() >= 4);
        assert_eq!(c[0].start, 0);
        assert_eq!(c.last().unwrap().end, p.frames());
        for w in c.windows(2) {
            assert_eq!(w[0].end, w[1].start, "no gap, no overlap");
        }
        for ch in &c {
            assert!(ch.end_ms - ch.start_ms <= 30_000);
            if ch.end < p.frames() {
                assert_eq!(p.samples[ch.end], 0.0, "cut at {} ms is in a pause", ch.end_ms);
            }
        }
        assert_eq!(plan(&p, 30_000).unwrap(), c, "deterministic");
    }

    #[test]
    fn no_silence_cuts_at_the_window() {
        let rate = 16_000;
        let p = Pcm::new(rate, 1, (0..rate as usize * 70).map(|i| (i as f32 * 0.1).sin() * 0.5 + 0.4).collect());
        let c = plan(&p, 30_000).unwrap();
        assert_eq!(c.len(), 3);
        assert!(c.iter().all(|ch| ch.end_ms - ch.start_ms <= 30_000));
    }

    #[test]
    fn refuses_a_tiny_limit_and_handles_empty() {
        assert!(plan(&bursts(16_000, 10, 10, 100), 100).is_err());
        assert!(plan(&Pcm::new(16_000, 1, vec![]), 30_000).unwrap().is_empty());
    }
}
