//! Choosing what to pull: a quality ranking, which quant runs fully from fast memory (the memory
//! a quant needs comes from `weirdnet_model::fit`), and the `[[model.local]]` stanza. Pure
//! functions; no network and no disk. File grouping, quant tags and split GGUFs are
//! `weirdnet_model::hf`'s.

use weirdnet_model::fit::{self, Fit, FitOptions};
use weirdnet_model::hf::{RepoModel, split_part};
use weirdnet_model::hw::{Accel, Budget, Hardware, budget};

pub use weirdnet_model::fit::KvSource;

pub const GIB: u64 = 1 << 30;
/// The context a quant is picked at unless `--context` says otherwise. The entry itself gets no
/// fixed context: ymird runs the model at its full trained length, cut only where the KV cache
/// would not fit (`ymir_core::kvfit`).
pub const DEFAULT_CONTEXT: u64 = 32_768;

/// `Q4_K_M`, `UD-Q4_K_XL`, `BF16`, ... or the model id when the name has no quant tag.
pub fn label(m: &RepoModel) -> String {
    m.quant.clone().unwrap_or_else(|| m.id.clone())
}

pub fn full_precision(m: &RepoModel) -> bool {
    matches!(label(m).to_ascii_uppercase().as_str(), "F32" | "F16" | "BF16")
}

/// Memory for weights of `size` bytes (no file list, as for an MLX folder) at `context` tokens: the
/// same sum as [`measure`], with the crate's KV estimate unless `kv_per_token` is known.
pub fn footprint_of(size: u64, kv_per_token: Option<u64>, context: u64) -> u64 {
    let kv = kv_per_token.unwrap_or_else(|| fit::estimate_kv_bytes_per_token(size));
    let total = size.saturating_add(kv.saturating_mul(context));
    total.saturating_add(total / 10)
}

/// `model` sized at `context` tokens: the header's KV figure when known, else the crate's estimate.
/// Weights + KV cache + 10%, as the pool's other nodes measure it.
pub fn measure(model: &RepoModel, kv_per_token: Option<u64>, context: u64) -> Fit {
    let mut opts = FitOptions::default().with_context(context.min(u64::from(u32::MAX)) as u32);
    if let Some(kv) = kv_per_token {
        opts = opts.with_kv_bytes_per_token(kv);
    }
    fit::measure(model, &Budget { pool_bytes: 0, reserve_bytes: 0, usable_bytes: u64::MAX }, &opts)
}

/// Rough bits per weight; higher is better quality. Unknown tags fall back to their leading digit.
pub fn quality(label: &str) -> f32 {
    let l = label.to_ascii_uppercase();
    let l = l.strip_prefix("UD-").unwrap_or(&l);
    let bonus = if label.to_ascii_uppercase().starts_with("UD-") { 0.1 } else { 0.0 };
    let base = match l {
        "F32" => 32.0,
        "F16" | "BF16" => 16.0,
        "Q8_0" => 8.5,
        "Q6_K" | "Q6_K_L" => 6.6,
        "Q5_1" => 6.0,
        "Q5_K_L" => 5.8,
        "Q5_K_M" => 5.7,
        "Q5_K_S" | "Q5_0" => 5.5,
        "Q4_1" => 5.0,
        "Q4_K_XL" | "Q4_K_L" => 4.95,
        "Q4_K_M" => 4.85,
        "Q4_K_S" => 4.6,
        "Q4_0" | "IQ4_NL" => 4.5,
        "MXFP4" | "MXFP4_MOE" | "NVFP4" | "IQ4_XS" => 4.3,
        "Q3_K_L" => 4.1,
        "Q3_K_M" => 3.9,
        "IQ3_M" => 3.7,
        "Q3_K_S" => 3.5,
        "IQ3_S" => 3.4,
        "IQ3_XS" => 3.3,
        "IQ3_XXS" => 3.1,
        "Q2_K_L" | "Q2_K_XL" => 3.1,
        "Q2_K" => 3.0,
        "IQ2_M" => 2.7,
        "IQ2_S" => 2.5,
        "IQ2_XS" => 2.3,
        "IQ2_XXS" => 2.1,
        "IQ1_M" => 1.75,
        "IQ1_S" => 1.6,
        other => {
            other.trim_start_matches(['I', 'T']).trim_start_matches('Q').chars().next().and_then(|c| c.to_digit(10)).map_or(0.5, |d| d as f32 - 0.5)
        }
    };
    base + bonus
}

/// What this node can hold: `fast` is memory the model runs fastest from (GPU memory, or the
/// unified budget), `total` adds system RAM that a partial offload can use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capacity {
    pub fast: u64,
    pub total: u64,
    pub accelerated: bool,
}

pub fn capacity(hw: &Hardware) -> Capacity {
    let b = budget(hw, None, None);
    let dedicated: u64 = hw.gpus.iter().filter(|g| !g.unified && g.accel != Accel::Cpu).map(|g| g.memory_bytes).sum();
    if dedicated > 0 {
        let fast = dedicated / 100 * 90;
        return Capacity { fast, total: fast + hw.ram_bytes / 4 * 3, accelerated: true };
    }
    Capacity { fast: b.usable_bytes, total: b.usable_bytes, accelerated: hw.gpus.iter().any(|g| g.accel != Accel::Cpu) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Fits in the memory the model runs fastest from.
    Fits,
    /// Fits only by spilling layers to system RAM (slower).
    Partial,
    TooBig,
}

impl Verdict {
    pub fn label(self) -> &'static str {
        match self {
            Verdict::Fits => "fits",
            Verdict::Partial => "partial (spills to RAM, slower)",
            Verdict::TooBig => "too big",
        }
    }
}

pub fn verdict(need: u64, cap: &Capacity) -> Verdict {
    if need <= cap.fast {
        Verdict::Fits
    } else if need <= cap.total {
        Verdict::Partial
    } else {
        Verdict::TooBig
    }
}

/// The context to size for: the request or [`DEFAULT_CONTEXT`], capped at what the model was
/// trained for when that is known.
pub fn context_for(requested: Option<u64>, trained: Option<u64>) -> u64 {
    let c = requested.unwrap_or(DEFAULT_CONTEXT);
    match trained {
        Some(t) if t > 0 && requested.is_none() => c.min(t),
        _ => c,
    }
}

/// The best quant that runs fully from fast memory: highest quality, full precision only when
/// nothing else fits. `None` when nothing fits fast; then use [`smallest`].
pub fn pick<'a>(models: &'a [RepoModel], kv: Option<u64>, context: u64, cap: &Capacity) -> Option<&'a RepoModel> {
    let fits = |m: &&RepoModel| verdict(measure(m, kv, context).footprint, cap) == Verdict::Fits;
    let best = |full: bool| {
        models
            .iter()
            .filter(|m| full_precision(m) == full)
            .filter(fits)
            .max_by(|a, b| quality(&label(a)).total_cmp(&quality(&label(b))).then(b.size.cmp(&a.size)))
    };
    best(false).or_else(|| best(true))
}

pub fn smallest(models: &[RepoModel]) -> Option<&RepoModel> {
    models.iter().min_by_key(|m| m.size)
}

/// A model id for a repo: its name in lower case without the `-GGUF` suffix.
pub fn id_for_repo(repo: &str) -> String {
    let name = repo.rsplit('/').next().unwrap_or(repo).to_ascii_lowercase();
    name.strip_suffix("-gguf").map_or(name.clone(), str::to_string)
}

/// A model id for a local file: its lower-case stem without a shard suffix.
pub fn id_for_file(file: &str) -> String {
    // Both separators: a Windows path has backslashes.
    let name = file.rsplit(['/', '\\']).next().unwrap_or(file);
    let stem = name.trim_end_matches(".gguf").trim_end_matches(".GGUF");
    split_part(name).map_or_else(|| stem.to_ascii_lowercase(), |(b, _, _)| b.to_ascii_lowercase())
}

pub use ymir_core::config_edit::{local_model, stanza};

/// A projector GGUF (`mmproj-*.gguf`), which is not a chat quant.
pub fn is_mmproj_name(file: &str) -> bool {
    let name = file.rsplit(['/', '\\']).next().unwrap_or(file).to_ascii_lowercase();
    name.starts_with("mmproj-") && name.ends_with(".gguf")
}

/// The projector to pull with a GGUF: prefer `Q8_0` (as the catalog Gemma projectors do), then
/// any non-F16/BF16 file, then whatever is there.
pub fn pick_mmproj(files: &[weirdnet_model::hf::RepoFile]) -> Option<&weirdnet_model::hf::RepoFile> {
    let mut found: Vec<&weirdnet_model::hf::RepoFile> = files.iter().filter(|f| is_mmproj_name(&f.name)).collect();
    if found.is_empty() {
        return None;
    }
    found.sort_by_key(|f| {
        let n = f.name.to_ascii_uppercase();
        if n.contains("Q8_0") {
            0u8
        } else if n.contains("BF16") || n.contains("-F16") || n.ends_with("F16.GGUF") {
            2
        } else {
            1
        }
    });
    Some(found[0])
}

/// A Hugging Face repo vLLM can load: `config.json` plus safetensors, or the Hub's tags.
pub fn is_safetensors_repo(files: &[weirdnet_model::hf::RepoFile], tags: &[String]) -> bool {
    let has_st = files.iter().any(|f| f.name.ends_with(".safetensors"));
    let has_cfg = files.iter().any(|f| f.name.rsplit('/').next() == Some("config.json"));
    (has_st && has_cfg) || tags.iter().any(|t| t == "safetensors" || t == "transformers")
}

#[cfg(test)]
mod tests {
    use weirdnet_model::hf::{RepoFile, group_files};

    use super::*;

    fn models(files: &[(&str, u64)]) -> Vec<RepoModel> {
        let files: Vec<RepoFile> = files.iter().map(|(n, s)| RepoFile { name: (*n).into(), size: *s, sha256: None }).collect();
        group_files(&files).into_iter().filter(|m| m.complete).collect()
    }

    #[test]
    fn picks_the_best_quant_that_fits_with_the_kv_cache() {
        let ms = models(&[("m-Q8_0.gguf", 8 * GIB), ("m-Q6_K.gguf", 6 * GIB), ("m-Q4_K_M.gguf", 4 * GIB), ("m-F16.gguf", 16 * GIB)]);
        let cap = Capacity { fast: 12 * GIB, total: 12 * GIB, accelerated: true };
        assert_eq!(label(pick(&ms, Some(100_000), 8192, &cap).unwrap()), "Q8_0");
        // A 1 MB/token cache at 32k is 32 GiB of KV: nothing fits.
        assert!(pick(&ms, Some(1 << 20), 32_768, &cap).is_none());
        // 192 KiB/token at 32k is 6 GiB of KV: Q6_K would need 6 + 6 + overhead > 12, so only Q4_K_M fits.
        assert_eq!(label(pick(&ms, Some(192 * 1024), 32_768, &cap).unwrap()), "Q4_K_M");
        // 128 KiB/token is 4 GiB of KV, and Q6_K (6 + 4 + overhead) still fits.
        assert_eq!(label(pick(&ms, Some(128 * 1024), 32_768, &cap).unwrap()), "Q6_K");
        assert_eq!(label(smallest(&ms).unwrap()), "Q4_K_M");
    }

    #[test]
    fn full_precision_is_a_last_resort() {
        let ms = models(&[("m-Q4_K_M.gguf", 4 * GIB), ("m-F16.gguf", 2 * GIB)]);
        let cap = Capacity { fast: 100 * GIB, total: 100 * GIB, accelerated: false };
        assert_eq!(label(pick(&ms, Some(1000), 4096, &cap).unwrap()), "Q4_K_M");
        let tight = Capacity { fast: 3 * GIB, total: 3 * GIB, accelerated: false };
        assert_eq!(label(pick(&ms, Some(1000), 4096, &tight).unwrap()), "F16");
    }

    #[test]
    fn verdicts_distinguish_fast_partial_and_too_big() {
        let cap = Capacity { fast: 10 * GIB, total: 30 * GIB, accelerated: true };
        assert_eq!(verdict(9 * GIB, &cap), Verdict::Fits);
        assert_eq!(verdict(20 * GIB, &cap), Verdict::Partial);
        assert_eq!(verdict(31 * GIB, &cap), Verdict::TooBig);
    }

    #[test]
    fn sizes_the_context_and_estimates_kv_without_a_header() {
        assert_eq!(context_for(None, Some(8192)), 8192);
        assert_eq!(context_for(None, None), DEFAULT_CONTEXT);
        assert_eq!(context_for(Some(65_536), Some(8192)), 65_536);
        let m = &models(&[("m-Q4_K_M.gguf", 10 * GIB)])[0];
        assert_eq!(measure(m, Some(5), 1000).kv_source, KvSource::Header);
        assert_eq!(measure(m, None, 1000).kv_source, KvSource::Estimate);
    }

    #[test]
    fn stanza_mirrors_the_setup_shape_and_parses_back() {
        let m = local_model("tiny", "llamacpp", "chat", "/srv/models/gguf/tiny-Q8_0.gguf", Some(32_768), true);
        let s = stanza(&m);
        assert!(s.starts_with("[[model.local]]\nid = \"tiny\"\n"), "{s}");
        for want in [
            "backend = \"llamacpp\"",
            "role = \"chat\"",
            "file = \"/srv/models/gguf/tiny-Q8_0.gguf\"",
            "context = 32768",
            "gpu_layers = 99",
            "idle_ttl = \"1800s\"",
            "args = [\"-fa\", \"on\", \"--jinja\"]",
        ] {
            assert!(s.contains(want), "missing {want} in\n{s}");
        }
        let cfg: ymir_core::NodeConfig = toml::from_str(&s).unwrap();
        assert_eq!(cfg.model.local[0], m);
        // No accelerator: no gpu_layers.
        assert!(!stanza(&local_model("t", "llamacpp", "chat", "/x.gguf", None, false)).contains("gpu_layers"));
    }

    #[test]
    fn ids_come_from_the_repo_or_the_file() {
        assert_eq!(id_for_repo("bartowski/Qwen3-8B-GGUF"), "qwen3-8b");
        assert_eq!(id_for_file("big-Q8_0-00001-of-00004.gguf"), "big-q8_0");
        assert_eq!(id_for_file("Tiny-Q4_K_M.gguf"), "tiny-q4_k_m");
    }

    #[test]
    fn picks_the_q8_0_mmproj_and_detects_safetensors_repos() {
        use weirdnet_model::hf::RepoFile;
        let files = [
            RepoFile { name: "mmproj-Inkling-2-Mini-BF16.gguf".into(), size: 642, sha256: None },
            RepoFile { name: "mmproj-Inkling-2-Mini-Q8_0.gguf".into(), size: 346, sha256: None },
            RepoFile { name: "Inkling-2-Mini-IQ4_NL.gguf".into(), size: 17_600, sha256: None },
        ];
        assert_eq!(pick_mmproj(&files).unwrap().name, "mmproj-Inkling-2-Mini-Q8_0.gguf");
        assert!(!is_mmproj_name("Inkling-2-Mini-IQ4_NL.gguf"));
        let st =
            [RepoFile { name: "config.json".into(), size: 10, sha256: None }, RepoFile { name: "model.safetensors".into(), size: 100, sha256: None }];
        assert!(is_safetensors_repo(&st, &[]));
        assert!(is_safetensors_repo(&[], &["transformers".into()]));
        assert!(!is_safetensors_repo(&[RepoFile { name: "README.md".into(), size: 1, sha256: None }], &[]));
    }
}
