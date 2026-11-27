//! RIFF/WAVE: probe a header, decode the samples, write 16-bit PCM.

use crate::{MediaError, Pcm, Result};

const FORMAT_PCM: u16 = 1;
const FORMAT_FLOAT: u16 = 3;
const FORMAT_EXTENSIBLE: u16 = 0xFFFE;

/// Sample encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    /// Signed integers (unsigned for 8-bit), little-endian.
    Int,
    Float,
}

/// What a WAV header says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WavInfo {
    pub sample_rate: u32,
    pub channels: u16,
    pub bits: u16,
    pub format: SampleFormat,
    /// Where the samples start in the file.
    pub data_offset: usize,
    /// Bytes of samples present (a streamed header's unknown size counts what is there).
    pub data_len: usize,
}

impl WavInfo {
    /// Frames (one sample per channel).
    pub fn frames(&self) -> u64 {
        let frame = u64::from(self.channels) * u64::from(self.bits / 8);
        (self.data_len as u64).checked_div(frame).unwrap_or(0)
    }

    pub fn duration_ms(&self) -> u64 {
        self.frames() * 1000 / u64::from(self.sample_rate.max(1))
    }
}

/// True when `bytes` start like a WAV file.
pub fn is_wav(bytes: &[u8]) -> bool {
    bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE"
}

fn u16_at(b: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(at..at + 2)?.try_into().ok()?))
}

fn u32_at(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?))
}

/// Reads the header. `Invalid` for anything that is not a well-formed WAV with a `fmt ` chunk
/// before its `data` chunk; `Unsupported` for encodings other than PCM and float.
pub fn probe(bytes: &[u8]) -> Result<WavInfo> {
    if !is_wav(bytes) {
        return Err(MediaError::Invalid("not a WAV file (no RIFF/WAVE header)".into()));
    }
    let bad = |m: &str| MediaError::Invalid(format!("broken WAV header: {m}"));
    let mut at = 12usize;
    let mut fmt: Option<(u16, u16, u32, u16)> = None;
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let size = u32_at(bytes, at + 4).ok_or_else(|| bad("chunk size"))? as usize;
        let body = at + 8;
        match id {
            b"fmt " => {
                if size < 16 || body + 16 > bytes.len() {
                    return Err(bad("short fmt chunk"));
                }
                let mut tag = u16_at(bytes, body).ok_or_else(|| bad("fmt"))?;
                let channels = u16_at(bytes, body + 2).ok_or_else(|| bad("fmt"))?;
                let rate = u32_at(bytes, body + 4).ok_or_else(|| bad("fmt"))?;
                let bits = u16_at(bytes, body + 14).ok_or_else(|| bad("fmt"))?;
                if tag == FORMAT_EXTENSIBLE {
                    // The sub-format GUID's first two bytes are the real tag.
                    tag = u16_at(bytes, body + 24).filter(|_| size >= 40).ok_or_else(|| bad("short extensible fmt chunk"))?;
                }
                fmt = Some((tag, channels, rate, bits));
            }
            b"data" => {
                let (tag, channels, sample_rate, bits) = fmt.ok_or_else(|| bad("data before fmt"))?;
                let format = match (tag, bits) {
                    (FORMAT_PCM, 8 | 16 | 24 | 32) => SampleFormat::Int,
                    (FORMAT_FLOAT, 32 | 64) => SampleFormat::Float,
                    _ => return Err(MediaError::Unsupported(format!("WAV encoding {tag} with {bits}-bit samples is not supported"))),
                };
                if channels == 0 || channels > 32 {
                    return Err(bad("channel count"));
                }
                if !(1000..=768_000).contains(&sample_rate) {
                    return Err(bad("sample rate"));
                }
                let present = bytes.len() - body;
                let data_len = if size == 0 || size == u32::MAX as usize { present } else { size.min(present) };
                return Ok(WavInfo { sample_rate, channels, bits, format, data_offset: body, data_len });
            }
            _ => {}
        }
        at = body.checked_add(size).and_then(|x| x.checked_add(size & 1)).ok_or_else(|| bad("chunk size"))?;
    }
    Err(bad("no data chunk"))
}

/// Decodes to interleaved `f32` samples in [-1, 1].
pub fn decode(bytes: &[u8]) -> Result<Pcm> {
    let info = probe(bytes)?;
    let data = &bytes[info.data_offset..info.data_offset + info.data_len];
    let width = usize::from(info.bits / 8);
    let frame = width * usize::from(info.channels);
    let data = &data[..data.len() - data.len() % frame];
    let samples: Vec<f32> = match (info.format, info.bits) {
        (SampleFormat::Int, 8) => data.iter().map(|b| (f32::from(*b) - 128.0) / 128.0).collect(),
        (SampleFormat::Int, 16) => data.as_chunks::<2>().0.iter().map(|c| f32::from(i16::from_le_bytes(*c)) / 32768.0).collect(),
        (SampleFormat::Int, 24) => {
            data.as_chunks::<3>().0.iter().map(|c| (i32::from_le_bytes([0, c[0], c[1], c[2]]) >> 8) as f32 / 8_388_608.0).collect()
        }
        (SampleFormat::Int, 32) => data.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c) as f32 / 2_147_483_648.0).collect(),
        (SampleFormat::Float, 32) => data.as_chunks::<4>().0.iter().map(|c| clean(f32::from_le_bytes(*c))).collect(),
        (SampleFormat::Float, 64) => data.as_chunks::<8>().0.iter().map(|c| clean(f64::from_le_bytes(*c) as f32)).collect(),
        _ => return Err(MediaError::Unsupported("unsupported WAV sample size".into())),
    };
    Ok(Pcm::new(info.sample_rate, info.channels, samples))
}

fn clean(s: f32) -> f32 {
    if s.is_finite() { s.clamp(-1.0, 1.0) } else { 0.0 }
}

/// Converts `f32` samples to signed 16-bit little-endian bytes.
pub fn to_pcm16(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for s in samples {
        let v = (clean(*s) * 32767.0).round() as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Reads signed 16-bit little-endian bytes (raw PCM, no header).
pub fn from_pcm16(bytes: &[u8]) -> Vec<f32> {
    bytes.as_chunks::<2>().0.iter().map(|c| f32::from(i16::from_le_bytes(*c)) / 32768.0).collect()
}

/// A 16-bit PCM WAV file of `pcm`.
pub fn encode_pcm16(pcm: &Pcm) -> Vec<u8> {
    let data = to_pcm16(&pcm.samples);
    let channels = pcm.channels;
    let block = channels * 2;
    let mut out = Vec::with_capacity(44 + data.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&FORMAT_PCM.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&pcm.sample_rate.to_le_bytes());
    out.extend_from_slice(&(pcm.sample_rate * u32::from(block)).to_le_bytes());
    out.extend_from_slice(&block.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&data);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(rate: u32, ms: u32) -> Pcm {
        let n = (rate * ms / 1000) as usize;
        Pcm::new(rate, 1, (0..n).map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / rate as f32).sin() * 0.5).collect())
    }

    #[test]
    fn round_trip_pcm16() {
        let p = tone(16_000, 250);
        let bytes = encode_pcm16(&p);
        let info = probe(&bytes).unwrap();
        assert_eq!((info.sample_rate, info.channels, info.bits, info.format), (16_000, 1, 16, SampleFormat::Int));
        assert_eq!(info.duration_ms(), 250);
        let back = decode(&bytes).unwrap();
        assert_eq!(back.samples.len(), p.samples.len());
        assert!(back.samples.iter().zip(&p.samples).all(|(a, b)| (a - b).abs() < 1e-3));
    }

    fn header(tag: u16, channels: u16, rate: u32, bits: u16, data: &[u8], size: u32) -> Vec<u8> {
        let mut v = b"RIFF\0\0\0\0WAVEfmt ".to_vec();
        v.extend_from_slice(&16u32.to_le_bytes());
        v.extend_from_slice(&tag.to_le_bytes());
        v.extend_from_slice(&channels.to_le_bytes());
        v.extend_from_slice(&rate.to_le_bytes());
        v.extend_from_slice(&(rate * u32::from(channels) * u32::from(bits / 8)).to_le_bytes());
        v.extend_from_slice(&(channels * bits / 8).to_le_bytes());
        v.extend_from_slice(&bits.to_le_bytes());
        v.extend_from_slice(b"LIST\x04\0\0\0abcd"); // a chunk to skip
        v.extend_from_slice(b"data");
        v.extend_from_slice(&size.to_le_bytes());
        v.extend_from_slice(data);
        v
    }

    #[test]
    fn decodes_other_widths_and_float() {
        let d8 = header(1, 1, 8000, 8, &[128, 255, 0], 3);
        let p = decode(&d8).unwrap();
        assert_eq!(p.samples.len(), 3);
        assert!(p.samples[0].abs() < 1e-6 && p.samples[1] > 0.99 && p.samples[2] <= -1.0 + 1e-6);
        let d24 = header(1, 1, 8000, 24, &[0, 0, 0x40, 0, 0, 0xC0], 6);
        let p = decode(&d24).unwrap();
        assert!((p.samples[0] - 0.5).abs() < 1e-6 && (p.samples[1] + 0.5).abs() < 1e-6);
        let mut f = Vec::new();
        for x in [0.25f32, f32::NAN, 3.0] {
            f.extend_from_slice(&x.to_le_bytes());
        }
        let p = decode(&header(3, 1, 8000, 32, &f, 12)).unwrap();
        assert_eq!(p.samples, vec![0.25, 0.0, 1.0]);
    }

    #[test]
    fn streamed_size_counts_what_is_there() {
        let d = header(1, 2, 8000, 16, &[0; 8], u32::MAX);
        let i = probe(&d).unwrap();
        assert_eq!((i.data_len, i.frames()), (8, 2));
    }

    #[test]
    fn hostile_headers_fail_closed() {
        assert!(matches!(probe(b"hello, I am a text file named .wav"), Err(MediaError::Invalid(_))));
        assert!(matches!(probe(b"RIFF\0\0\0\0WAVE"), Err(MediaError::Invalid(_))));
        let mut huge = b"RIFF\0\0\0\0WAVEjunk".to_vec();
        huge.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(probe(&huge).is_err());
        assert!(matches!(probe(&header(1, 0, 8000, 16, &[0; 4], 4)), Err(MediaError::Invalid(_))));
        assert!(matches!(probe(&header(1, 1, 10, 16, &[0; 4], 4)), Err(MediaError::Invalid(_))));
        assert!(matches!(probe(&header(2, 1, 8000, 4, &[0; 4], 4)), Err(MediaError::Unsupported(_))));
        // data before fmt
        let mut v = b"RIFF\0\0\0\0WAVEdata\x02\0\0\0\0\0".to_vec();
        v.extend_from_slice(b"fmt ");
        assert!(probe(&v).is_err());
        // A truncated file decodes what is there, whole frames only.
        let mut t = encode_pcm16(&tone(8000, 10));
        t.truncate(t.len() - 3);
        let p = decode(&t).unwrap();
        assert_eq!(p.samples.len(), 78);
    }
}
