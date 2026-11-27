//! Samples in memory: down-mix, resample, slice, stitch.

use crate::{MediaError, Result, wav};

/// The rate speech-to-text models want (Whisper, Voxtral).
pub const ASR_RATE: u32 = 16_000;

/// Interleaved `f32` samples in [-1, 1].
#[derive(Debug, Clone, PartialEq)]
pub struct Pcm {
    pub sample_rate: u32,
    pub channels: u16,
    pub samples: Vec<f32>,
}

impl Pcm {
    pub fn new(sample_rate: u32, channels: u16, samples: Vec<f32>) -> Self {
        Self { sample_rate, channels: channels.max(1), samples }
    }

    /// Frames (one sample per channel).
    pub fn frames(&self) -> usize {
        self.samples.len() / usize::from(self.channels)
    }

    pub fn duration_ms(&self) -> u64 {
        self.frames() as u64 * 1000 / u64::from(self.sample_rate.max(1))
    }

    /// The average of the channels.
    pub fn to_mono(&self) -> Pcm {
        if self.channels == 1 {
            return self.clone();
        }
        let c = usize::from(self.channels);
        let samples = self.samples.chunks_exact(c).map(|f| f.iter().sum::<f32>() / c as f32).collect();
        Pcm::new(self.sample_rate, 1, samples)
    }

    /// Frames `start..end` (clamped).
    pub fn slice(&self, start: usize, end: usize) -> Pcm {
        let c = usize::from(self.channels);
        let end = end.min(self.frames());
        let start = start.min(end);
        Pcm::new(self.sample_rate, self.channels, self.samples[start * c..end * c].to_vec())
    }

    /// Resampled to `rate` with a windowed-sinc filter (Hann window, 16 zero crossings, low-pass
    /// at the lower Nyquist frequency), channel by channel.
    pub fn resample(&self, rate: u32) -> Pcm {
        if rate == self.sample_rate || self.samples.is_empty() {
            return Pcm::new(rate, self.channels, self.samples.clone());
        }
        let c = usize::from(self.channels);
        let frames = self.frames();
        let ratio = f64::from(rate) / f64::from(self.sample_rate);
        let out_frames = ((frames as f64) * ratio).round() as usize;
        let cutoff = ratio.min(1.0);
        const ZEROS: f64 = 16.0;
        let half = (ZEROS / cutoff).ceil() as i64;
        let mut out = vec![0f32; out_frames * c];
        for o in 0..out_frames {
            let t = o as f64 / ratio;
            let center = t.floor() as i64;
            let mut acc = vec![0f64; c];
            let mut norm = 0f64;
            for i in (center - half + 1)..=(center + half) {
                if i < 0 || i as usize >= frames {
                    continue;
                }
                let x = (t - i as f64) * cutoff;
                let sinc = if x.abs() < 1e-9 { 1.0 } else { (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x) };
                let w = 0.5 + 0.5 * (std::f64::consts::PI * x / ZEROS).cos();
                let k = if x.abs() >= ZEROS { 0.0 } else { sinc * w };
                norm += k;
                for (ch, a) in acc.iter_mut().enumerate() {
                    *a += k * f64::from(self.samples[i as usize * c + ch]);
                }
            }
            for (ch, a) in acc.iter().enumerate() {
                out[o * c + ch] = if norm.abs() > 1e-9 { (a / norm).clamp(-1.0, 1.0) as f32 } else { 0.0 };
            }
        }
        Pcm::new(rate, self.channels, out)
    }
}

/// A WAV clip as speech-to-text models want it: 16 kHz, mono, 16-bit PCM WAV. Only WAV input
/// (other containers go through [`crate::ffmpeg`] first).
pub fn for_asr(wav_bytes: &[u8]) -> Result<Vec<u8>> {
    let pcm = wav::decode(wav_bytes)?;
    Ok(wav::encode_pcm16(&pcm.to_mono().resample(ASR_RATE)))
}

/// How to join pieces of speech.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Join {
    /// Equal-power crossfade, in milliseconds, between pieces with no pause.
    pub crossfade_ms: u32,
    /// Silence inserted before this piece (no crossfade when non-zero).
    pub pause_ms: u32,
}

/// Joins `pieces` in order. `joins[i]` says how piece `i + 1` follows piece `i` (missing entries
/// mean a plain crossfade of `default_crossfade_ms`). Every piece must have the same rate and
/// channels: speech from different engines or settings is not mixed.
pub fn stitch(pieces: &[Pcm], joins: &[Join], default_crossfade_ms: u32) -> Result<Pcm> {
    let Some(first) = pieces.first() else {
        return Err(MediaError::Invalid("nothing to stitch".into()));
    };
    let (rate, channels) = (first.sample_rate, first.channels);
    if let Some(p) = pieces.iter().find(|p| p.sample_rate != rate || p.channels != channels) {
        return Err(MediaError::Unsupported(format!(
            "pieces differ: {} Hz × {} channels after {rate} Hz × {channels} channels",
            p.sample_rate, p.channels
        )));
    }
    let c = usize::from(channels);
    let mut out = first.samples.clone();
    for (i, p) in pieces.iter().enumerate().skip(1) {
        let join = joins.get(i - 1).copied().unwrap_or(Join { crossfade_ms: default_crossfade_ms, pause_ms: 0 });
        if join.pause_ms > 0 {
            out.extend(std::iter::repeat_n(0.0, (rate as usize * join.pause_ms as usize / 1000) * c));
            out.extend_from_slice(&p.samples);
            continue;
        }
        let have = out.len() / c;
        let fade = (rate as usize * join.crossfade_ms as usize / 1000).min(have).min(p.frames());
        let start = (have - fade) * c;
        for f in 0..fade {
            let t = (f as f32 + 0.5) / fade as f32;
            let (a, b) = ((t * std::f32::consts::FRAC_PI_2).cos(), (t * std::f32::consts::FRAC_PI_2).sin());
            for ch in 0..c {
                let o = &mut out[start + f * c + ch];
                *o = (*o * a + p.samples[f * c + ch] * b).clamp(-1.0, 1.0);
            }
        }
        out.extend_from_slice(&p.samples[fade * c..]);
    }
    Ok(Pcm::new(rate, channels, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(rate: u32, hz: f32, ms: u32, channels: u16) -> Pcm {
        let n = (rate * ms / 1000) as usize;
        let mut s = Vec::with_capacity(n * usize::from(channels));
        for i in 0..n {
            let v = (i as f32 * hz * std::f32::consts::TAU / rate as f32).sin() * 0.5;
            for _ in 0..channels {
                s.push(v);
            }
        }
        Pcm::new(rate, channels, s)
    }

    fn rms(s: &[f32]) -> f32 {
        (s.iter().map(|x| x * x).sum::<f32>() / s.len().max(1) as f32).sqrt()
    }

    #[test]
    fn resample_keeps_length_and_tone() {
        for (from, to) in [(48_000, 16_000), (44_100, 16_000), (8_000, 16_000), (24_000, 16_000)] {
            let p = tone(from, 440.0, 500, 1).resample(to);
            assert_eq!(p.sample_rate, to);
            assert!((p.frames() as i64 - (to as i64 / 2)).abs() <= 1, "{from}->{to}: {}", p.frames());
            assert!((rms(&p.samples[1000..7000]) - 0.3536).abs() < 0.02, "{from}->{to}");
        }
        // Above the new Nyquist (8 kHz) is filtered out.
        let high = tone(48_000, 12_000.0, 200, 1).resample(16_000);
        assert!(rms(&high.samples[200..3000]) < 0.02);
    }

    #[test]
    fn for_asr_is_16k_mono_16bit() {
        let stereo = wav::encode_pcm16(&tone(44_100, 440.0, 300, 2));
        let out = for_asr(&stereo).unwrap();
        let i = wav::probe(&out).unwrap();
        assert_eq!((i.sample_rate, i.channels, i.bits), (16_000, 1, 16));
        assert!((i.duration_ms() as i64 - 300).abs() <= 1);
        assert!(for_asr(b"not audio").is_err());
    }

    #[test]
    fn stitch_crossfades_and_pauses() {
        let a = tone(16_000, 440.0, 100, 1);
        let b = tone(16_000, 440.0, 100, 1);
        let x = stitch(&[a.clone(), b.clone()], &[], 10).unwrap();
        assert_eq!(x.frames(), 1600 + 1600 - 160);
        assert!(x.samples.iter().all(|s| s.abs() <= 1.0));
        let y = stitch(&[a.clone(), b.clone()], &[Join { crossfade_ms: 10, pause_ms: 300 }], 10).unwrap();
        assert_eq!(y.frames(), 1600 + 4800 + 1600);
        assert!(y.samples[1600..6400].iter().all(|s| *s == 0.0));
        // A piece shorter than the crossfade is not over-read.
        let z = stitch(&[a.clone(), a.slice(0, 10)], &[], 50).unwrap();
        assert_eq!(z.frames(), 1600);
        assert!(stitch(&[a.clone(), tone(24_000, 440.0, 10, 1)], &[], 10).is_err());
        assert!(stitch(&[a, tone(16_000, 440.0, 10, 2)], &[], 10).is_err());
        assert!(stitch(&[], &[], 10).is_err());
    }

    #[test]
    fn mono_and_slice() {
        let p = Pcm::new(8000, 2, vec![1.0, 0.0, 0.5, 0.5, -1.0, 1.0]);
        assert_eq!(p.to_mono().samples, vec![0.5, 0.5, 0.0]);
        assert_eq!(p.slice(1, 99).samples, vec![0.5, 0.5, -1.0, 1.0]);
        assert_eq!(p.slice(5, 2).frames(), 0);
    }
}
