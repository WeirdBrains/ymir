//! The context each model is launched with.
//!
//! A model gets the full context it was trained for (GGUF `<arch>.context_length`, or a
//! `config.json` `max_position_embeddings` with its RoPE scaling) unless this node cannot hold the
//! KV cache for it. Then the context is cut to what fits, in steps of [`STEP`] tokens, and the
//! plan says why. An explicit `context` (or `--ctx-size` / `--max-model-len` in `args`) is the
//! operator's choice and is kept as it is.
//!
//! The KV cache is sized from the model's shape ([`KvShape`]): attention layers (loop passes
//! unrolled), KV heads, head size, the cache type in `args` (`-ctk`/`-ctv`, `--kv-cache-dtype`),
//! and sliding-window layers, which hold one window per server slot (llama-server runs 4 slots
//! with a unified cache unless told otherwise). Attention that cannot use flash attention (a
//! per-head score bias, as Inkling's relative positions, or `-fa off`) materializes its f32 score
//! matrix, context x micro-batch x heads, in the compute buffer: two of them where a bias is added
//! to the scores (97 GiB for Inkling at 1M tokens), one under plain `-fa off`. Everything gets 10%
//! on top, as `weirdnet_model::fit` counts a footprint, and
//! [`HEADROOM_PCT`] of the memory left is kept free.
//!
//! `weirdnet_model::gguf::GgufInfo` has no sliding-window or loop fields yet, so the GGUF reader
//! here reads those keys too. It is meant to move into `weirdnet-model` (a wish for the Weirdnet
//! lead); keep it free of Ymir types.

use std::collections::{BTreeMap, HashMap};
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;
use weirdnet_backends::{EngineKind, ModelLaunch};
use weirdnet_model::hw::Budget;

/// The context used when a model's trained length is unknown (the old `ymirctl models pull`
/// default).
pub const FALLBACK_CONTEXT: u32 = 32_768;
/// The smallest context a model is cut to when memory is short.
pub const MIN_CONTEXT: u32 = 4_096;
/// A cut context is rounded down to a multiple of this.
pub const STEP: u64 = 1_024;
/// The context older `ymirctl models pull` wrote into every entry it made.
pub const OLD_PULL_DEFAULT: u32 = 32_768;
/// The `--max-model-len` older pulls wrote for Inkling 2 Mini on a DGX Spark.
pub const OLD_INKLING_SPARK_DEFAULT: u32 = 262_144;
/// llama.cpp's default micro-batch, which a sliding-window cache keeps beyond its window.
const UBATCH: u64 = 512;
/// The slots llama-server runs when `--parallel` is not given (its "auto": 4, unified KV).
pub const SERVER_AUTO_SLOTS: u32 = 4;
/// The share of the memory free for a model that is kept free anyway: estimates are estimates, and
/// a unified-memory node also runs its OS out of it.
pub const HEADROOM_PCT: u64 = 20;
/// How many context x micro-batch x heads f32 score matrices the compute buffer holds where a
/// per-head bias is added to the scores (llama.cpp b12628 `build_attn_mha` with `kq_b`): the
/// gathered bias and the scores are both live at the `ggml_add`, so ggml's allocator reserves two.
/// llama.cpp's own reservation (`llama-fit-params -fitp on` on the ggml-org Inkling-2-Mini IQ4_NL
/// header) is 68,002 MiB of compute at 701,440 tokens and 101,563 MiB at 1M: 2.07 matrices. With
/// one, v0.3.0 planned 70.9 GiB for Inkling at 701,440 beside gemma4-e4b on node-a (a 128 GB Spark)
/// and llama-server mapped 108 GiB while it set up its context, and the node ran out of memory.
/// (node-b's 84.1 GiB at 788,480 was read after load: not all of the compute buffer was resident.)
const BIASED_SCORE_COPIES: u64 = 2;
/// The same under plain `-fa off` (no bias): the softmax runs in place, so one matrix (Qwen3
/// 0.6B at 32,768 tokens: 1,110 MiB of compute against a 1,024 MiB matrix).
const SCORE_COPIES: u64 = 1;
/// Bytes per token of context x micro-batch for the attention inputs beside the scores: the f32
/// mask, and the i32 relative positions where there is a bias (llama.cpp measured ~6).
const MASK_BYTES: u64 = 4;
const REL_POS_BYTES: u64 = 4;
/// The share of GPU memory vLLM takes unless `--gpu-memory-utilization` says otherwise.
const VLLM_UTILIZATION: f64 = 0.9;
/// How many header bytes [`gguf_shape`] reads at most (tokenizer vocabularies are the bulk).
const HEADER_LIMIT: u64 = 64 << 20;

/// What a model's KV cache grows with.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KvShape {
    /// K plus V elements per token, summed over the layers that attend to the whole context
    /// (each pass of a looped layer counts).
    pub full_elems: u64,
    /// The same over sliding-window layers.
    pub swa_elems: u64,
    /// The sliding window in tokens; 0 when the model has none.
    pub window: u64,
    pub full_layers: u64,
    pub swa_layers: u64,
    /// The context the model was trained for (or can take, with its RoPE scaling).
    pub trained: Option<u64>,
    /// Query heads (the most of any layer).
    pub heads: u64,
    /// Attention takes a per-head score bias, so llama.cpp cannot use flash attention for it and
    /// keeps full score matrices (Inkling's relative-position bias, T5's).
    pub dense_scores: bool,
}

impl KvShape {
    /// KV cache bytes at `context` tokens with `slots` sequences sharing it (llama.cpp's unified
    /// cache: the context is the total; a sliding-window layer keeps a window per sequence).
    pub fn kv_bytes(&self, context: u64, slots: u64, cache: CacheType) -> u64 {
        let swa_tokens = context.min(self.window.saturating_mul(slots.max(1)).saturating_add(UBATCH));
        cache.bytes(self.full_elems.saturating_mul(context).saturating_add(self.swa_elems.saturating_mul(swa_tokens)))
    }

    /// The compute buffer that grows with the context: the f32 attention scores (two matrices with
    /// a score bias, one under `-fa off`) and their mask, when they are not flash-attended (else
    /// nothing; the fixed-size rest is in [`footprint`]'s 10%).
    pub fn score_bytes(&self, context: u64, run: Run) -> u64 {
        if !(self.dense_scores || run.flash_off) || self.full_layers == 0 {
            return 0;
        }
        let (copies, inputs) = if self.dense_scores { (BIASED_SCORE_COPIES, MASK_BYTES + REL_POS_BYTES) } else { (SCORE_COPIES, MASK_BYTES) };
        let per_row = copies.saturating_mul(self.heads.max(1)).saturating_mul(4).saturating_add(inputs);
        context.saturating_mul(run.ubatch).saturating_mul(per_row)
    }

    /// Everything that grows with the context: the KV cache and the score buffers.
    pub fn context_bytes(&self, context: u64, run: Run) -> u64 {
        self.kv_bytes(context, run.slots, run.cache).saturating_add(self.score_bytes(context, run))
    }
}

/// How a launch runs, as far as its memory goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Run {
    /// Sequences sharing the cache (llama-server `--parallel`; vLLM: 1, its pool is per token).
    pub slots: u64,
    pub cache: CacheType,
    /// llama.cpp's micro-batch (`-ub`), the score matrices' other side.
    pub ubatch: u64,
    /// `-fa off`: every layer keeps full score matrices.
    pub flash_off: bool,
}

impl Run {
    pub fn of(l: &ModelLaunch) -> Self {
        let llama = l.engine == EngineKind::Llamacpp;
        let n = |names: &[&str]| arg_value(&l.extra_args, names).and_then(|v| v.parse::<u64>().ok()).filter(|v| *v > 0);
        Self {
            slots: if llama { l.slots.map(u64::from).or_else(|| n(&["-np", "--parallel"])).unwrap_or(u64::from(SERVER_AUTO_SLOTS)) } else { 1 },
            cache: cache_type(l.engine, &l.extra_args),
            ubatch: if llama { n(&["-ub", "--ubatch-size"]).unwrap_or(UBATCH) } else { UBATCH },
            flash_off: llama && arg_value(&l.extra_args, &["-fa", "--flash-attn"]).is_some_and(|v| matches!(v, "off" | "0" | "false")),
        }
    }

    /// One slot, a 16-bit cache, the default micro-batch.
    pub fn single() -> Self {
        Self { slots: 1, cache: CacheType::F16, ubatch: UBATCH, flash_off: false }
    }
}

/// A KV cache element type and its size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheType {
    pub name: &'static str,
    num: u64,
    den: u64,
}

impl CacheType {
    pub const F16: CacheType = CacheType { name: "f16", num: 2, den: 1 };

    /// A llama.cpp `--cache-type-k`/`-v` value.
    pub fn llama(s: &str) -> Option<Self> {
        let (name, num, den) = match s.to_ascii_lowercase().as_str() {
            "f32" => ("f32", 4, 1),
            "f16" => ("f16", 2, 1),
            "bf16" => ("bf16", 2, 1),
            "q8_0" => ("q8_0", 34, 32),
            "q5_1" => ("q5_1", 24, 32),
            "q5_0" => ("q5_0", 22, 32),
            "q4_1" => ("q4_1", 20, 32),
            "q4_0" => ("q4_0", 18, 32),
            "iq4_nl" => ("iq4_nl", 18, 32),
            _ => return None,
        };
        Some(Self { name, num, den })
    }

    /// A vLLM `--kv-cache-dtype` value (`auto` is the model's 16-bit type).
    pub fn vllm(s: &str) -> Option<Self> {
        let s = s.to_ascii_lowercase();
        Some(match s.as_str() {
            "auto" | "bfloat16" | "float16" | "half" | "bf16" | "fp16" => Self::F16,
            "float32" | "fp32" => Self { name: "fp32", num: 4, den: 1 },
            "nvfp4" | "fp4" => Self { name: "fp4", num: 9, den: 16 },
            s if s.starts_with("fp8") => Self { name: "fp8", num: 1, den: 1 },
            _ => return None,
        })
    }

    pub fn bytes(self, elems: u64) -> u64 {
        u64::try_from((u128::from(elems) * u128::from(self.num)).div_ceil(u128::from(self.den))).unwrap_or(u64::MAX)
    }
}

/// The value after the last of `names` in `args` (`--flag value` or `--flag=value`).
pub fn arg_value<'a>(args: &'a [String], names: &[&str]) -> Option<&'a str> {
    let mut found = None;
    for (i, a) in args.iter().enumerate() {
        if names.contains(&a.as_str()) {
            found = args.get(i + 1).map(String::as_str);
        } else if let Some((k, v)) = a.split_once('=')
            && names.contains(&k)
        {
            found = Some(v);
        }
    }
    found
}

/// `args` without any of `names` and their values.
pub fn strip_arg(args: &[String], names: &[&str]) -> Vec<String> {
    let mut out = Vec::with_capacity(args.len());
    let mut skip = false;
    for a in args {
        if skip {
            skip = false;
            continue;
        }
        if names.contains(&a.as_str()) {
            skip = true;
        } else if !a.split_once('=').is_some_and(|(k, _)| names.contains(&k)) {
            out.push(a.clone());
        }
    }
    out
}

/// The engine flags that set the context.
pub fn context_flags(engine: EngineKind) -> &'static [&'static str] {
    match engine {
        EngineKind::Llamacpp => &["-c", "--ctx-size"],
        EngineKind::Vllm => &["--max-model-len", "--max_model_len"],
        _ => &[],
    }
}

/// The KV cache type `args` ask for (the larger of K and V for llama.cpp), else 16-bit.
pub fn cache_type(engine: EngineKind, args: &[String]) -> CacheType {
    match engine {
        EngineKind::Llamacpp => {
            let k = arg_value(args, &["-ctk", "--cache-type-k"]).and_then(CacheType::llama);
            let v = arg_value(args, &["-ctv", "--cache-type-v"]).and_then(CacheType::llama);
            let (k, v) = (k.unwrap_or(CacheType::F16), v.unwrap_or(CacheType::F16));
            if k.bytes(1 << 20) >= v.bytes(1 << 20) { k } else { v }
        }
        EngineKind::Vllm => arg_value(args, &["--kv-cache-dtype", "--kv_cache_dtype"]).and_then(CacheType::vllm).unwrap_or(CacheType::F16),
        _ => CacheType::F16,
    }
}

// ---------------------------------------------------------------------------------------------
// GGUF headers

#[derive(Debug, Clone)]
enum Val {
    Int(u64),
    Bool(bool),
    Str(String),
    Ints(Vec<u64>),
    Bools(Vec<bool>),
    Other,
}

struct Reader<R> {
    inner: R,
    read: u64,
}

impl<R: Read> Reader<R> {
    fn bytes(&mut self, n: u64) -> Result<Vec<u8>> {
        if self.read + n > HEADER_LIMIT {
            bail!("GGUF header is larger than {HEADER_LIMIT} bytes");
        }
        let mut buf = vec![0; usize::try_from(n)?];
        self.inner.read_exact(&mut buf).context("GGUF header cut short")?;
        self.read += n;
        Ok(buf)
    }

    fn skip(&mut self, n: u64) -> Result<()> {
        if self.read + n > HEADER_LIMIT {
            bail!("GGUF header is larger than {HEADER_LIMIT} bytes");
        }
        let got = std::io::copy(&mut (&mut self.inner).take(n), &mut std::io::sink())?;
        if got != n {
            bail!("GGUF header cut short");
        }
        self.read += n;
        Ok(())
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().map_err(|_| anyhow!("short read"))?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().map_err(|_| anyhow!("short read"))?))
    }

    fn string(&mut self) -> Result<String> {
        let n = self.u64()?;
        String::from_utf8(self.bytes(n)?).map_err(|_| anyhow!("GGUF string is not UTF-8"))
    }

    /// One scalar of type `t`: integers (negative ones as `None`) and booleans; floats skipped.
    fn scalar(&mut self, t: u32) -> Result<Val> {
        let le = |b: Vec<u8>| b.iter().rev().fold(0u64, |acc, x| (acc << 8) | u64::from(*x));
        Ok(match t {
            0 => Val::Int(u64::from(self.bytes(1)?[0])),
            1 => u64::try_from(self.bytes(1)?[0] as i8).map_or(Val::Other, Val::Int),
            2 => Val::Int(le(self.bytes(2)?)),
            3 => u64::try_from(le(self.bytes(2)?) as u16 as i16).map_or(Val::Other, Val::Int),
            4 => Val::Int(u64::from(self.u32()?)),
            5 => u64::try_from(self.u32()? as i32).map_or(Val::Other, Val::Int),
            7 => Val::Bool(self.bytes(1)?[0] != 0),
            10 => Val::Int(self.u64()?),
            11 => u64::try_from(self.u64()? as i64).map_or(Val::Other, Val::Int),
            6 => {
                self.skip(4)?;
                Val::Other
            }
            12 => {
                self.skip(8)?;
                Val::Other
            }
            t => bail!("GGUF value type {t} is unknown"),
        })
    }

    fn value(&mut self, t: u32) -> Result<Val> {
        match t {
            8 => {
                let n = self.u64()?;
                if n > 4096 {
                    self.skip(n)?;
                    return Ok(Val::Other);
                }
                String::from_utf8(self.bytes(n)?).map(Val::Str).map_err(|_| anyhow!("GGUF string is not UTF-8"))
            }
            9 => {
                let et = self.u32()?;
                let n = self.u64()?;
                if et == 8 {
                    for _ in 0..n {
                        let len = self.u64()?;
                        self.skip(len)?;
                    }
                    return Ok(Val::Other);
                }
                let keep = n <= 4096;
                let (mut ints, mut bools) = (Vec::new(), Vec::new());
                for _ in 0..n {
                    match self.scalar(et)? {
                        Val::Int(v) if keep => ints.push(v),
                        Val::Bool(b) if keep => bools.push(b),
                        _ => {}
                    }
                }
                Ok(if !keep {
                    Val::Other
                } else if et == 7 && bools.len() as u64 == n {
                    Val::Bools(bools)
                } else if ints.len() as u64 == n {
                    Val::Ints(ints)
                } else {
                    Val::Other
                })
            }
            t => self.scalar(t),
        }
    }
}

/// Architectures whose attention takes a score bias in llama.cpp, which turns flash attention off.
const DENSE_SCORE_ARCHS: &[&str] = &["inkling2", "inkling2_loop", "t5", "t5encoder"];

/// The KV shape in a GGUF file's header (versions 2 and 3). Reads only the metadata.
pub fn gguf_shape(r: impl Read) -> Result<KvShape> {
    let mut r = Reader { inner: r, read: 0 };
    if r.bytes(4)? != b"GGUF" {
        bail!("not a GGUF file");
    }
    let version = r.u32()?;
    if !(2..=3).contains(&version) {
        bail!("GGUF version {version} is not supported");
    }
    let _tensors = r.u64()?;
    let count = r.u64()?;
    let mut kv: HashMap<String, Val> = HashMap::new();
    for _ in 0..count {
        let key = r.string()?;
        let t = r.u32()?;
        let v = r.value(t)?;
        kv.insert(key, v);
    }
    let arch = match kv.get("general.architecture") {
        Some(Val::Str(a)) => a.clone(),
        _ => bail!("GGUF header has no general.architecture"),
    };
    let get = |k: &str| kv.get(&format!("{arch}.{k}"));
    let int = |k: &str| match get(k) {
        Some(Val::Int(v)) => Some(*v),
        _ => None,
    };
    let layers = int("block_count").ok_or_else(|| anyhow!("GGUF header has no {arch}.block_count"))?;
    let n = usize::try_from(layers)?;
    let per_layer = |k: &str| -> Option<Vec<u64>> {
        match get(k) {
            Some(Val::Int(v)) => Some(vec![*v; n]),
            Some(Val::Ints(v)) if v.len() == n => Some(v.clone()),
            Some(Val::Ints(v)) => v.iter().copied().max().map(|m| vec![m; n]),
            _ => None,
        }
    };
    let heads = per_layer("attention.head_count").unwrap_or_else(|| vec![0; n]);
    let kv_heads = per_layer("attention.head_count_kv").unwrap_or_else(|| heads.clone());
    let max_heads = heads.iter().copied().max().unwrap_or(0);
    let head_dim = int("embedding_length").filter(|_| max_heads > 0).map(|e| e / max_heads);
    let k_len = int("attention.key_length").or(head_dim).ok_or_else(|| anyhow!("GGUF header has no head size"))?;
    let v_len = int("attention.value_length").unwrap_or(k_len);
    let k_swa = int("attention.key_length_swa").unwrap_or(k_len);
    let v_swa = int("attention.value_length_swa").unwrap_or(v_len);
    let window = int("attention.sliding_window").unwrap_or(0);
    let swa: Vec<bool> = match get("attention.sliding_window_pattern") {
        _ if window == 0 => vec![false; n],
        Some(Val::Bools(b)) if b.len() == n => b.clone(),
        // llama.cpp's convention: every `p`th layer is global.
        Some(Val::Int(p)) => (0..layers).map(|il| *p == 0 || il % p < p - 1).collect(),
        // A window without a pattern: llama.cpp knows the pattern per architecture; this does
        // not, so count every layer as global (more KV than real, never less).
        _ => vec![false; n],
    };
    let loops = (int("loop.start"), int("loop.end"), int("loop.count"));
    let mut shape = KvShape {
        window,
        trained: int("context_length").filter(|c| *c > 0),
        heads: max_heads,
        dense_scores: DENSE_SCORE_ARCHS.contains(&arch.as_str()),
        ..Default::default()
    };
    // Gemma 3n/4: the last `shared_kv_layers` layers reuse earlier layers' KV and cache none.
    let own_kv = n.saturating_sub(usize::try_from(int("attention.shared_kv_layers").unwrap_or(0)).unwrap_or(0));
    for il in 0..own_kv {
        let passes = match loops {
            (Some(s), Some(e), Some(c)) if (s..e).contains(&(il as u64)) => c.max(1),
            _ => 1,
        };
        if swa[il] {
            shape.swa_elems += passes * kv_heads[il] * (k_swa + v_swa);
            shape.swa_layers += passes * u64::from(kv_heads[il] > 0);
        } else {
            shape.full_elems += passes * kv_heads[il] * (k_len + v_len);
            shape.full_layers += passes * u64::from(kv_heads[il] > 0);
        }
    }
    Ok(shape)
}

// ---------------------------------------------------------------------------------------------
// config.json (vLLM)

fn num(v: &Value, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|k| v.get(*k).and_then(|x| x.as_u64().or_else(|| x.as_f64().filter(|f| *f > 0.0).map(|f| f as u64))))
}

/// The longest context vLLM derives from a config: `max_position_embeddings` (or the model's
/// other length keys) times a RoPE scaling factor, as vLLM's `_get_and_verify_max_len` does.
fn derived_max_len(t: &Value, root: &Value) -> Option<u64> {
    let keys = ["max_position_embeddings", "model_max_length", "max_seq_len", "max_sequence_length", "seq_length", "n_positions"];
    let mut len = num(t, &keys).or_else(|| num(root, &keys))?;
    let rope = t.get("rope_scaling").or_else(|| t.get("rope_parameters")).filter(|r| r.is_object());
    if let Some(r) = rope
        && let Some(factor) = r.get("factor").and_then(Value::as_f64)
    {
        let kind = r.get("rope_type").or_else(|| r.get("type")).and_then(Value::as_str).unwrap_or("");
        if !matches!(kind, "su" | "longrope" | "llama3" | "default" | "mrope") {
            if kind == "yarn"
                && let Some(orig) = num(r, &["original_max_position_embeddings"])
            {
                len = orig;
            }
            len = (len as f64 * factor) as u64;
        }
    }
    Some(len)
}

fn layer_shape(t: &Value, loops: bool) -> Result<KvShape> {
    let layers = num(t, &["num_hidden_layers", "n_layer", "num_layers"]).ok_or_else(|| anyhow!("config.json has no num_hidden_layers"))?;
    let heads = num(t, &["num_attention_heads", "n_head"]).unwrap_or(0);
    let kv_heads = num(t, &["num_key_value_heads", "num_kv_heads", "n_head_kv"]).unwrap_or(heads);
    let head_dim = num(t, &["head_dim"])
        .or_else(|| num(t, &["hidden_size", "n_embd"]).filter(|_| heads > 0).map(|h| h / heads))
        .ok_or_else(|| anyhow!("config.json has no head size"))?;
    let swa_heads = num(t, &["swa_num_key_value_heads"]).unwrap_or(kv_heads);
    let swa_dim = num(t, &["swa_head_dim"]).unwrap_or(head_dim);
    let sliding_on = t.get("use_sliding_window").and_then(Value::as_bool) != Some(false);
    let window = if sliding_on { num(t, &["sliding_window_size", "sliding_window"]).unwrap_or(0) } else { 0 };
    let n = usize::try_from(layers)?;
    // Per layer: Some(true) sliding, Some(false) global, None no KV cache (linear/recurrent).
    let kinds: Vec<Option<bool>> = if let Some(types) = t.get("layer_types").and_then(Value::as_array).filter(|a| a.len() == n) {
        types
            .iter()
            .map(|x| match x.as_str().unwrap_or("") {
                "sliding_attention" | "local_attention" | "chunked_attention" => Some(window > 0),
                "full_attention" | "attention" | "global_attention" | "" => Some(false),
                // Linear attention, Mamba and other recurrent layers keep a fixed-size state.
                _ => None,
            })
            .collect()
    } else if window == 0 {
        vec![Some(false); n]
    } else if let Some(ids) = t.get("local_layer_ids").and_then(Value::as_array) {
        let ids: Vec<u64> = ids.iter().filter_map(Value::as_u64).collect();
        (0..layers).map(|i| Some(ids.contains(&i))).collect()
    } else if let Some(p) = num(t, &["sliding_window_pattern"]) {
        (0..layers).map(|i| Some(p == 0 || i % p < p - 1)).collect()
    } else {
        vec![Some(false); n]
    };
    let lc = t.get("loop_config").filter(|_| loops);
    let range = lc.and_then(|c| Some((num(c, &["core_start"])?, num(c, &["core_end"])?, num(c, &["num_passes"]).unwrap_or(1))));
    let mut shape = KvShape { window, heads, ..Default::default() };
    for (i, kind) in kinds.iter().enumerate() {
        let passes = match range {
            Some((s, e, p)) if (s..e).contains(&(i as u64)) => p.max(1),
            _ => 1,
        };
        match kind {
            Some(true) => {
                shape.swa_elems += passes * swa_heads * swa_dim * 2;
                shape.swa_layers += passes;
            }
            Some(false) => {
                shape.full_elems += passes * kv_heads * head_dim * 2;
                shape.full_layers += passes;
            }
            None => {}
        }
    }
    Ok(shape)
}

/// The KV shape in a Hugging Face `config.json` (the text model's, for multimodal configs), with
/// a multi-token-prediction head's layers added when it has one (vLLM caches those too).
pub fn config_shape(v: &Value) -> Result<KvShape> {
    let t = v.get("text_config").filter(|t| t.is_object()).unwrap_or(v);
    let mut shape = layer_shape(t, true)?;
    shape.trained = derived_max_len(t, v);
    if let Some(mtp) = v.get("mtp_config").filter(|m| m.is_object())
        && let Ok(m) = layer_shape(mtp, false)
    {
        shape.full_elems += m.full_elems;
        shape.full_layers += m.full_layers;
        shape.swa_elems += m.swa_elems;
        shape.swa_layers += m.swa_layers;
        shape.window = shape.window.max(m.window);
    }
    Ok(shape)
}

// ---------------------------------------------------------------------------------------------
// Where a model's shape and weights are

/// The folder Ymir keeps a vLLM Hub model's `config.json` and weight size in:
/// `models_dir/vllm/<owner>--<name>` (written by `ymirctl models pull`).
pub fn vllm_meta_dir(models_dir: &Path, repo: &str) -> PathBuf {
    models_dir.join("vllm").join(repo.replace('/', "--"))
}

/// Saves what [`resolve`] needs for a vLLM Hub model that vLLM fetches itself.
pub fn save_vllm_meta(models_dir: &Path, repo: &str, config: &[u8], weights: u64) -> Result<()> {
    let dir = vllm_meta_dir(models_dir, repo);
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    std::fs::write(dir.join("config.json"), config)?;
    std::fs::write(dir.join("weights"), weights.to_string())?;
    Ok(())
}

/// The Hugging Face cache's hub folder: `HF_HUB_CACHE`, `HF_HOME/hub` or `~/.cache/huggingface/hub`.
fn hf_hub_cache() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("HF_HUB_CACHE") {
        return Some(PathBuf::from(p));
    }
    if let Some(p) = std::env::var_os("HF_HOME") {
        return Some(PathBuf::from(p).join("hub"));
    }
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/huggingface/hub"))
}

fn safetensors_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.file_name().to_string_lossy().ends_with(".safetensors"))
                .filter_map(|e| std::fs::metadata(e.path()).ok())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}

/// A model's shape and the bytes its weights take in memory (with its projector and its LoRA
/// adapters, which load beside them).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolved {
    pub shape: KvShape,
    pub weights: u64,
}

fn gguf_weights(source: &Path) -> u64 {
    let name = source.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let size = |p: &Path| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
    match (weirdnet_model::hf::split_part(&name), source.parent()) {
        (Some((base, _, of)), Some(dir)) => (1..=of).map(|i| size(&dir.join(format!("{base}-{i:05}-of-{of:05}.gguf")))).sum(),
        _ => size(source),
    }
}

/// Reads `l`'s shape: the GGUF header for llama.cpp; for vLLM the `config.json` in a local model
/// folder, the one `ymirctl models pull` saved, or the Hugging Face cache's.
pub fn resolve(l: &ModelLaunch, models_dir: &Path) -> Result<Resolved> {
    match l.engine {
        EngineKind::Llamacpp => {
            let p = Path::new(&l.source);
            let f = std::fs::File::open(p).with_context(|| format!("opening {}", p.display()))?;
            let shape = gguf_shape(std::io::BufReader::new(f)).with_context(|| format!("reading the GGUF header of {}", p.display()))?;
            let mmproj = arg_value(&l.extra_args, &["--mmproj", "-mm"]).and_then(|m| std::fs::metadata(m).ok()).map_or(0, |m| m.len());
            Ok(Resolved { shape, weights: gguf_weights(p) + mmproj + crate::adapters::bytes(l) })
        }
        EngineKind::Vllm => {
            let p = Path::new(&l.source);
            let (config, weights) = if p.is_dir() {
                (p.join("config.json"), safetensors_bytes(p))
            } else {
                let meta = vllm_meta_dir(models_dir, &l.source);
                if meta.join("config.json").is_file() {
                    let w = std::fs::read_to_string(meta.join("weights")).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(0);
                    (meta.join("config.json"), w)
                } else {
                    let repo = hf_hub_cache()
                        .map(|c| c.join(format!("models--{}", l.source.replace('/', "--"))))
                        .ok_or_else(|| anyhow!("no Hugging Face cache"))?;
                    let rev = std::fs::read_to_string(repo.join("refs/main")).ok().map(|s| s.trim().to_string());
                    let snap = match rev {
                        Some(r) => repo.join("snapshots").join(r),
                        None => std::fs::read_dir(repo.join("snapshots"))
                            .ok()
                            .and_then(|mut rd| rd.find_map(|e| e.ok().map(|e| e.path())))
                            .ok_or_else(|| anyhow!("{} is not in {} or the Hugging Face cache", l.source, meta.display()))?,
                    };
                    (snap.join("config.json"), safetensors_bytes(&snap))
                }
            };
            let text = std::fs::read_to_string(&config).with_context(|| format!("reading {}", config.display()))?;
            let v: Value = serde_json::from_str(&text).with_context(|| format!("parsing {}", config.display()))?;
            if weights == 0 {
                bail!("the weights' size for {} is unknown", l.source);
            }
            Ok(Resolved { shape: config_shape(&v)?, weights: weights + crate::adapters::bytes(l) })
        }
        _ => bail!("{} sizes its cache as it goes; there is no context to plan", l.engine.name()),
    }
}

// ---------------------------------------------------------------------------------------------
// Fitting

/// Weights plus KV plus 10%, as `weirdnet_model::fit` counts a footprint.
pub fn footprint(weights: u64, kv: u64) -> u64 {
    let t = weights.saturating_add(kv);
    t.saturating_add(t / 10)
}

/// What of `free` bytes the context may grow into: [`HEADROOM_PCT`] stays free, and the rest
/// leaves room for [`footprint`]'s 10%.
pub fn context_budget(free: u64) -> u64 {
    free / 100 * (100 - HEADROOM_PCT) / 11 * 10
}

/// The largest context up to the trained length whose KV cache and score buffers
/// ([`KvShape::context_bytes`]) fit in `budget` bytes, and whether it had to be cut.
pub fn fit_context(shape: &KvShape, budget: u64, run: Run) -> (u32, bool) {
    let max = shape.trained.unwrap_or(u64::from(FALLBACK_CONTEXT)).min(u64::from(u32::MAX));
    if shape.context_bytes(max, run) <= budget {
        return (max as u32, false);
    }
    let (mut lo, mut hi) = (0u64, max);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if shape.context_bytes(mid, run) <= budget { lo = mid } else { hi = mid - 1 }
    }
    let floor = u64::from(MIN_CONTEXT).min(max);
    ((lo / STEP * STEP).max(floor) as u32, true)
}

/// What `l` takes in memory at the context it launches with (weights alone when it has none).
pub fn cost(r: &Resolved, l: &ModelLaunch) -> u64 {
    footprint(r.weights, l.context.map_or(0, |c| r.shape.context_bytes(u64::from(c), Run::of(l))))
}

/// The bytes `l`'s context may grow into when `free` bytes are free for it (its weights still to
/// load): its weights and their 10% come off, [`context_budget`] keeps the headroom, and vLLM is
/// also held to its own pool (`--gpu-memory-utilization`, default 0.9, of the memory pool).
pub fn room_for_context(r: &Resolved, l: &ModelLaunch, budget: &Budget, free: u64) -> u64 {
    let mut free = free.saturating_sub(footprint(r.weights, 0));
    if l.engine == EngineKind::Vllm {
        let util = arg_value(&l.extra_args, &["--gpu-memory-utilization", "--gpu_memory_utilization"])
            .and_then(|u| u.parse::<f64>().ok())
            .filter(|u| *u > 0.0 && *u <= 1.0)
            .unwrap_or(VLLM_UTILIZATION);
        let pool = (budget.pool_bytes as f64 * util) as u64;
        free = free.min(pool.saturating_sub(footprint(r.weights, 0)));
    }
    context_budget(free)
}

/// What [`plan_launches`] decided for one model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub id: String,
    /// The context the model launches with (`None`: the engine decides, as for MLX).
    pub context: Option<u32>,
    pub trained: Option<u64>,
    /// `true` when the context came from `context =` or `args`.
    pub explicit: bool,
    /// `true` when the full trained length did not fit and the context was cut.
    pub clamped: bool,
    /// For an explicit context: what the automatic context would be here, when larger.
    pub auto_would_be: Option<u32>,
    /// A warning rather than information.
    pub warn: bool,
    pub note: String,
}

fn gib(b: u64) -> String {
    format!("{:.1} GiB", b as f64 / (1u64 << 30) as f64)
}

/// Sets `context` on every launch that has none, and says what each model got and why.
///
/// `budget` is the node's model memory. Models that can be resident together share it: each
/// other swap group (or ungrouped model) counts with its largest member's weights (and KV, for an
/// explicit context), and the memory left is split evenly among the automatic models that can be
/// resident at once. vLLM is also held to its own pool (`--gpu-memory-utilization`, default 0.9,
/// of the memory pool). Wrapped servers and MLX are left alone.
pub fn plan_launches(launches: &mut [ModelLaunch], budget: &Budget, models_dir: &Path) -> Vec<Plan> {
    struct Item {
        resolved: Option<Resolved>,
        error: Option<String>,
        group: String,
        auto: bool,
        planned: bool,
    }
    let items: Vec<Item> = launches
        .iter()
        .map(|l| {
            let planned = l.wrap.is_none() && matches!(l.engine, EngineKind::Llamacpp | EngineKind::Vllm);
            let (resolved, error) = if planned {
                match resolve(l, models_dir) {
                    Ok(r) => (Some(r), None),
                    Err(e) => (None, Some(format!("{e:#}"))),
                }
            } else {
                (None, None)
            };
            Item {
                resolved,
                error,
                group: l.swap_group.clone().unwrap_or_else(|| format!("\0{}", l.id)),
                auto: planned && l.context.is_none(),
                planned,
            }
        })
        .collect();
    let cost = |i: usize| -> u64 {
        let l = &launches[i];
        let Some(r) = &items[i].resolved else {
            return if items[i].planned { std::fs::metadata(&l.source).map_or(0, |m| m.len()) } else { 0 };
        };
        if items[i].auto { footprint(r.weights, 0) } else { cost(r, l) }
    };
    let costs: Vec<u64> = (0..launches.len()).map(cost).collect();
    let mut plans = Vec::new();
    for i in 0..launches.len() {
        if !items[i].planned {
            continue;
        }
        // The other groups that can be resident with this model: their largest cost, and whether
        // they hold an automatic model.
        let mut others: BTreeMap<&str, (u64, bool)> = BTreeMap::new();
        for (j, it) in items.iter().enumerate() {
            if j == i || it.group == items[i].group {
                continue;
            }
            let e = others.entry(it.group.as_str()).or_default();
            e.0 = e.0.max(costs[j]);
            e.1 |= it.auto;
        }
        let fixed: u64 = others.values().map(|(c, _)| *c).sum();
        let sharers = 1 + others.values().filter(|(_, a)| *a).count() as u64;
        let l = &launches[i];
        let id = l.id.clone();
        let run = Run::of(l);
        let cache = run.cache;
        let Some(r) = items[i].resolved.clone() else {
            let why = items[i].error.clone().unwrap_or_default();
            if items[i].auto {
                launches[i].context = Some(FALLBACK_CONTEXT);
                plans.push(Plan {
                    id,
                    context: Some(FALLBACK_CONTEXT),
                    trained: None,
                    explicit: false,
                    clamped: false,
                    auto_would_be: None,
                    warn: true,
                    note: format!("context {FALLBACK_CONTEXT}: the model's shape could not be read ({why}); set `context` to choose one"),
                });
            } else {
                plans.push(Plan {
                    id,
                    context: l.context,
                    trained: None,
                    explicit: true,
                    clamped: false,
                    auto_would_be: None,
                    warn: false,
                    note: format!("context {} (set in node.toml)", l.context.unwrap_or(0)),
                });
            }
            continue;
        };
        let kv_budget = room_for_context(&r, l, budget, budget.usable_bytes.saturating_sub(fixed)) / sharers;
        let (fit, clamped) = fit_context(&r.shape, kv_budget, run);
        let trained = r.shape.trained;
        let full = trained.map_or(String::from("its trained length is unknown, so the fallback"), |t| format!("the trained length is {t}"));
        if let (false, Some(c)) = (items[i].auto, l.context) {
            let kv = r.shape.context_bytes(u64::from(c), run);
            let warn = kv > kv_budget;
            let total = footprint(r.weights, kv);
            let note = if warn {
                format!(
                    "context {c} (set in node.toml): its {} {} KV cache and score buffers may not fit beside the weights ({} free for them here; {} in all with {} of weights and 10% for the engine's other buffers)",
                    gib(kv),
                    cache.name,
                    gib(kv_budget),
                    gib(total),
                    gib(r.weights)
                )
            } else {
                format!(
                    "context {c} (set in node.toml) ({} {} KV cache and buffers; {} in all with {} of weights and 10% for the engine's other buffers)",
                    gib(kv),
                    cache.name,
                    gib(total),
                    gib(r.weights)
                )
            };
            plans.push(Plan { id, context: Some(c), trained, explicit: true, clamped: false, auto_would_be: (fit > c).then_some(fit), warn, note });
            continue;
        }
        launches[i].context = Some(fit);
        let kv = r.shape.context_bytes(u64::from(fit), run);
        let note = if clamped {
            let want = r.shape.context_bytes(trained.unwrap_or(u64::from(FALLBACK_CONTEXT)), run);
            format!(
                "context {fit}, cut from {}: the full-length {} KV cache and buffers need {}, and {} is free for them beside {} of weights{} (budget {}, {HEADROOM_PCT}% kept free). Set `context` to override",
                trained.unwrap_or(u64::from(FALLBACK_CONTEXT)),
                cache.name,
                gib(want),
                gib(kv_budget),
                gib(r.weights),
                if sharers > 1 { format!(", shared with {} other model(s)", sharers - 1) } else { String::new() },
                gib(budget.usable_bytes)
            )
        } else {
            format!("context {fit} ({full}; {} {} KV cache and buffers, {} with the weights)", gib(kv), cache.name, gib(footprint(r.weights, kv)))
        };
        plans.push(Plan {
            id,
            context: Some(fit),
            trained,
            explicit: false,
            clamped,
            auto_would_be: None,
            warn: clamped && fit <= MIN_CONTEXT,
            note,
        });
    }
    plans
}

/// Logs [`plan_launches`]'s decisions, with a hint where an entry still carries the context an
/// older `ymirctl models pull` wrote and the model could take more.
pub fn log_plans(plans: &[Plan]) {
    for p in plans {
        if p.warn {
            tracing::warn!(model = %p.id, "{}", p.note);
        } else {
            tracing::info!(model = %p.id, "{}", p.note);
        }
        if let (Some(c), Some(auto)) = (p.context.filter(|c| *c == OLD_PULL_DEFAULT || *c == OLD_INKLING_SPARK_DEFAULT), p.auto_would_be) {
            tracing::info!(
                model = %p.id,
                "context = {c} looks like the default an older `ymirctl models pull` wrote; this node can run it at {auto}. `ymirctl models context --auto {}` lifts it (then restart ymird)",
                p.id
            );
        }
    }
}

#[cfg(test)]
pub(crate) mod tests;
