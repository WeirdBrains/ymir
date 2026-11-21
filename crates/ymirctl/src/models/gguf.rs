//! Reads the few GGUF header fields that size the KV cache, from a stream (the first bytes of a
//! file on the Hub, or a local file), without loading any weights.

use std::io::{self, Read};

/// What a GGUF header says about the attention layout.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GgufMeta {
    pub arch: String,
    pub layers: u64,
    pub context_train: u64,
    pub embedding: u64,
    pub heads: u64,
    pub kv_heads: KvHeads,
    pub key_len: Option<u64>,
    pub value_len: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum KvHeads {
    #[default]
    Unknown,
    Uniform(u64),
    PerLayer(Vec<u64>),
}

impl GgufMeta {
    /// Bytes of f16 KV cache per token of context, or `None` when the header lacks the fields.
    pub fn kv_bytes_per_token(&self) -> Option<u64> {
        let head_dim = self.embedding.checked_div(self.heads).unwrap_or(0);
        let k = self.key_len.unwrap_or(head_dim);
        let v = self.value_len.unwrap_or(head_dim);
        let kv_total = match &self.kv_heads {
            KvHeads::Unknown => {
                if self.heads == 0 {
                    return None;
                }
                self.heads * self.layers
            }
            KvHeads::Uniform(n) => n * self.layers,
            KvHeads::PerLayer(v) => v.iter().sum(),
        };
        let bytes = kv_total * (k + v) * 2;
        (bytes > 0).then_some(bytes)
    }
}

fn read_n<const N: usize>(r: &mut impl Read) -> io::Result<[u8; N]> {
    let mut b = [0u8; N];
    r.read_exact(&mut b)?;
    Ok(b)
}

fn u32le(r: &mut impl Read) -> io::Result<u32> {
    Ok(u32::from_le_bytes(read_n(r)?))
}

fn u64le(r: &mut impl Read) -> io::Result<u64> {
    Ok(u64::from_le_bytes(read_n(r)?))
}

fn string(r: &mut impl Read) -> io::Result<String> {
    let n = u64le(r)?;
    if n > 1 << 20 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "implausible string length"));
    }
    let mut b = vec![0u8; n as usize];
    r.read_exact(&mut b)?;
    Ok(String::from_utf8_lossy(&b).into_owned())
}

fn skip(r: &mut impl Read, n: u64) -> io::Result<()> {
    let copied = io::copy(&mut r.take(n), &mut io::sink())?;
    if copied == n { Ok(()) } else { Err(io::ErrorKind::UnexpectedEof.into()) }
}

fn fixed_size(t: u32) -> Option<u64> {
    match t {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

fn int(t: u32, r: &mut impl Read) -> io::Result<Option<u64>> {
    Ok(match t {
        0 => Some(u64::from(read_n::<1>(r)?[0])),
        2 => Some(u64::from(u16::from_le_bytes(read_n(r)?))),
        4 => Some(u64::from(u32le(r)?)),
        5 => Some(i32::from_le_bytes(read_n(r)?).max(0) as u64),
        10 => Some(u64le(r)?),
        11 => Some(i64::from_le_bytes(read_n(r)?).max(0) as u64),
        _ => {
            skip_value(t, r)?;
            None
        }
    })
}

fn skip_value(t: u32, r: &mut impl Read) -> io::Result<()> {
    if let Some(n) = fixed_size(t) {
        return skip(r, n);
    }
    match t {
        8 => string(r).map(drop),
        9 => {
            let et = u32le(r)?;
            let n = u64le(r)?;
            match fixed_size(et) {
                Some(sz) => skip(r, sz.saturating_mul(n)),
                None => {
                    for _ in 0..n {
                        skip_value(et, r)?;
                    }
                    Ok(())
                }
            }
        }
        _ => Err(io::Error::new(io::ErrorKind::InvalidData, "unknown GGUF value type")),
    }
}

/// Parses the header up to the tokenizer section (which comes after the architecture keys and
/// can be megabytes long), and stops there.
pub fn read_meta(r: &mut impl Read) -> io::Result<GgufMeta> {
    if read_n::<4>(r)? != *b"GGUF" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "not a GGUF file"));
    }
    let version = u32le(r)?;
    if !(2..=3).contains(&version) {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "unsupported GGUF version"));
    }
    let _tensors = u64le(r)?;
    let count = u64le(r)?;
    let mut m = GgufMeta::default();
    for _ in 0..count {
        let key = string(r)?;
        if key.starts_with("tokenizer.") {
            break;
        }
        let t = u32le(r)?;
        if key == "general.architecture" && t == 8 {
            m.arch = string(r)?;
            continue;
        }
        let Some(field) = key.strip_prefix(&format!("{}.", m.arch)).filter(|_| !m.arch.is_empty()) else {
            skip_value(t, r)?;
            continue;
        };
        match field {
            "block_count" => m.layers = int(t, r)?.unwrap_or(0),
            "context_length" => m.context_train = int(t, r)?.unwrap_or(0),
            "embedding_length" => m.embedding = int(t, r)?.unwrap_or(0),
            "attention.head_count" => m.heads = int(t, r)?.unwrap_or(0),
            "attention.key_length" => m.key_len = int(t, r)?,
            "attention.value_length" => m.value_len = int(t, r)?,
            "attention.head_count_kv" if t == 9 => {
                let et = u32le(r)?;
                let n = u64le(r)?;
                if n > 1 << 16 {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "implausible array length"));
                }
                let mut v = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    v.push(int(et, r)?.unwrap_or(0));
                }
                m.kv_heads = KvHeads::PerLayer(v);
            }
            "attention.head_count_kv" => m.kv_heads = int(t, r)?.map_or(KvHeads::Unknown, KvHeads::Uniform),
            _ => skip_value(t, r)?,
        }
    }
    Ok(m)
}

/// A minimal GGUF header for tests: Llama-like architecture keys, then a tokenizer key.
#[cfg(test)]
pub fn sample_header(layers: u32, kv_heads: u32) -> Vec<u8> {
    fn s(out: &mut Vec<u8>, v: &str) {
        out.extend((v.len() as u64).to_le_bytes());
        out.extend(v.as_bytes());
    }
    let mut o = b"GGUF".to_vec();
    o.extend(3u32.to_le_bytes());
    o.extend(0u64.to_le_bytes());
    o.extend(8u64.to_le_bytes());
    s(&mut o, "general.architecture");
    o.extend(8u32.to_le_bytes());
    s(&mut o, "llama");
    s(&mut o, "general.name");
    o.extend(8u32.to_le_bytes());
    s(&mut o, "Tiny");
    for (k, v) in [
        ("llama.block_count", layers),
        ("llama.context_length", 131_072),
        ("llama.embedding_length", 4096),
        ("llama.attention.head_count", 32),
        ("llama.attention.head_count_kv", kv_heads),
    ] {
        s(&mut o, k);
        o.extend(4u32.to_le_bytes());
        o.extend(v.to_le_bytes());
    }
    s(&mut o, "tokenizer.ggml.tokens");
    o.extend(9u32.to_le_bytes());
    o.extend(8u32.to_le_bytes());
    o.extend(2u64.to_le_bytes());
    s(&mut o, "a");
    s(&mut o, "b");
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_attention_fields_and_sizes_the_kv_cache() {
        let m = read_meta(&mut sample_header(32, 8).as_slice()).unwrap();
        assert_eq!((m.arch.as_str(), m.layers, m.heads, m.context_train), ("llama", 32, 32, 131_072));
        // 32 layers * 8 kv heads * (128 + 128) * 2 bytes.
        assert_eq!(m.kv_bytes_per_token(), Some(32 * 8 * 256 * 2));
    }

    #[test]
    fn per_layer_kv_heads_and_explicit_key_lengths() {
        let m = GgufMeta {
            arch: "x".into(),
            layers: 4,
            heads: 8,
            embedding: 512,
            kv_heads: KvHeads::PerLayer(vec![2, 0, 2, 0]),
            key_len: Some(64),
            value_len: Some(32),
            ..Default::default()
        };
        assert_eq!(m.kv_bytes_per_token(), Some(4 * 96 * 2));
    }

    #[test]
    fn rejects_other_files_and_truncated_headers() {
        assert!(read_meta(&mut &b"nope0000000000000000000000"[..]).is_err());
        let h = sample_header(32, 8);
        assert!(read_meta(&mut &h[..40]).is_err());
    }
}
