//! Choosing what to pull: grouping a repo's files into quants (split GGUFs stay together), a
//! quality ranking, the memory a quant needs at a context length, and the `[[model.local]]`
//! stanza. Pure functions; no network and no disk.

use std::collections::BTreeMap;

use serde::Serialize;
use weirdnet_model::hw::{Accel, Hardware, budget};
use ymir_core::config::LocalModel;

pub const GIB: u64 = 1 << 30;
/// Compute buffers and the runtime's own overhead on top of weights and KV cache.
pub const OVERHEAD: u64 = GIB / 2;
/// The context a pulled model is sized and registered for unless `--context` says otherwise.
pub const DEFAULT_CONTEXT: u64 = 32_768;

/// One file of a Hub repo.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoFile {
    pub name: String,
    pub size: u64,
    pub sha256: Option<String>,
}

/// All the files of one quant: a single GGUF, or every shard of a split one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantSet {
    /// `Q4_K_M`, `UD-Q4_K_XL`, `BF16`, ... or the file stem when the name has no quant tag.
    pub label: String,
    /// Shards in order; the first is the file llama.cpp is pointed at.
    pub files: Vec<RepoFile>,
    pub size: u64,
}

impl QuantSet {
    pub fn full_precision(&self) -> bool {
        matches!(self.label.to_ascii_uppercase().as_str(), "F32" | "F16" | "BF16")
    }
}

fn basename(name: &str) -> &str {
    name.rsplit('/').next().unwrap_or(name)
}

/// `(stem without the shard suffix, index, count)` for `name-00002-of-00005.gguf`.
pub fn shard_of(file: &str) -> Option<(String, u32, u32)> {
    let stem = basename(file).strip_suffix(".gguf").or_else(|| basename(file).strip_suffix(".GGUF"))?;
    let (rest, total) = stem.rsplit_once("-of-")?;
    let (base, idx) = rest.rsplit_once('-')?;
    if idx.len() < 3 || total.len() < 3 {
        return None;
    }
    Some((base.to_string(), idx.parse().ok()?, total.parse().ok()?))
}

fn is_quant_token(t: &str) -> bool {
    let t = t.to_ascii_uppercase();
    if matches!(t.as_str(), "F32" | "F16" | "BF16" | "MXFP4" | "MXFP4_MOE" | "NVFP4") {
        return true;
    }
    let t = t.strip_prefix('I').unwrap_or(&t);
    let Some(rest) = t.strip_prefix('Q').or_else(|| t.strip_prefix("TQ")) else { return false };
    let digits = rest.chars().take_while(char::is_ascii_digit).count();
    digits > 0 && rest[digits..].starts_with('_') && rest[digits..].chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// The quant tag in a GGUF file name (`Qwen3-8B-Q4_K_M.gguf` is `Q4_K_M`; an Unsloth `UD-` prefix
/// stays part of the tag).
pub fn quant_of(file: &str) -> Option<String> {
    let stem = basename(file).trim_end_matches(".gguf").trim_end_matches(".GGUF");
    let stem = shard_of(file).map_or_else(|| stem.to_string(), |(b, _, _)| b);
    let tokens: Vec<&str> = stem.split(['-', '.']).collect();
    let i = tokens.iter().rposition(|t| is_quant_token(t))?;
    let tag = tokens[i].to_ascii_uppercase();
    Some(if i > 0 && tokens[i - 1].eq_ignore_ascii_case("ud") { format!("UD-{tag}") } else { tag })
}

fn skipped(name: &str) -> bool {
    let l = basename(name).to_ascii_lowercase();
    !l.ends_with(".gguf") || l.starts_with("mmproj") || l.contains("imatrix") || l.starts_with("mtp-")
}

/// (folder, file stem without shard suffix)
type GroupKey = (String, String);

/// Groups GGUF files into quants. Projectors, imatrix and draft files are left out, and so is a
/// split whose shards are not all there. Sorted smallest first.
pub fn quant_sets(files: &[RepoFile]) -> Vec<QuantSet> {
    let mut groups: BTreeMap<GroupKey, Vec<(u32, u32, &RepoFile)>> = BTreeMap::new();
    for f in files.iter().filter(|f| !skipped(&f.name)) {
        let dir = f.name.rsplit_once('/').map(|(d, _)| d.to_string()).unwrap_or_default();
        match shard_of(&f.name) {
            Some((base, i, n)) => groups.entry((dir, base)).or_default().push((i, n, f)),
            None => {
                let stem = basename(&f.name).trim_end_matches(".gguf").trim_end_matches(".GGUF").to_string();
                groups.entry((dir, stem)).or_default().push((1, 1, f));
            }
        }
    }
    let mut out: Vec<QuantSet> = groups
        .into_values()
        .filter_map(|mut v| {
            v.sort_by_key(|(i, _, _)| *i);
            let n = v[0].1;
            let complete = v.len() as u32 == n && v.iter().enumerate().all(|(k, (i, m, _))| *i == k as u32 + 1 && *m == n);
            if !complete {
                return None;
            }
            let files: Vec<RepoFile> = v.into_iter().map(|(_, _, f)| f.clone()).collect();
            let label = quant_of(&files[0].name).unwrap_or_else(|| {
                let b = basename(&files[0].name);
                b.trim_end_matches(".gguf").to_string()
            });
            Some(QuantSet { size: files.iter().map(|f| f.size).sum(), label, files })
        })
        .collect();
    out.sort_by(|a, b| a.size.cmp(&b.size).then_with(|| a.label.cmp(&b.label)));
    out
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

/// How the KV cache was sized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvSource {
    /// From the file's own GGUF header.
    Header,
    /// A guess from the file size, because the header could not be read.
    Estimate,
}

/// Bytes of KV cache per token: the header's figure, or about 32 KiB per GiB of weights.
pub fn kv_per_token(header: Option<u64>, weights: u64) -> (u64, KvSource) {
    match header {
        Some(b) => (b, KvSource::Header),
        None => ((weights / GIB).max(1) * 32 * 1024, KvSource::Estimate),
    }
}

/// Weights + KV cache at `context` + overhead.
pub fn need(weights: u64, kv_per_token: u64, context: u64) -> u64 {
    weights + kv_per_token.saturating_mul(context) + OVERHEAD
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
pub fn pick<'a>(sets: &'a [QuantSet], kv: u64, context: u64, cap: &Capacity) -> Option<&'a QuantSet> {
    let fits = |s: &&QuantSet| verdict(need(s.size, kv, context), cap) == Verdict::Fits;
    let best = |full: bool| {
        sets.iter()
            .filter(|s| s.full_precision() == full)
            .filter(fits)
            .max_by(|a, b| quality(&a.label).total_cmp(&quality(&b.label)).then(b.size.cmp(&a.size)))
    };
    best(false).or_else(|| best(true))
}

pub fn smallest(sets: &[QuantSet]) -> Option<&QuantSet> {
    sets.iter().min_by_key(|s| s.size)
}

/// A model id for a repo: its name in lower case without the `-GGUF` suffix.
pub fn id_for_repo(repo: &str) -> String {
    let name = repo.rsplit('/').next().unwrap_or(repo).to_ascii_lowercase();
    name.strip_suffix("-gguf").map_or(name.clone(), str::to_string)
}

/// A model id for a local file: its lower-case stem without a shard suffix.
pub fn id_for_file(file: &str) -> String {
    let stem = basename(file).trim_end_matches(".gguf").trim_end_matches(".GGUF");
    shard_of(file).map_or_else(|| stem.to_ascii_lowercase(), |(b, _, _)| b.to_ascii_lowercase())
}

#[derive(Serialize)]
struct Stanza<'a> {
    model: StanzaModel<'a>,
}

#[derive(Serialize)]
struct StanzaModel<'a> {
    local: [&'a LocalModel; 1],
}

/// A `[[model.local]]` entry, in the shape Ymir's own `setup` writes.
pub fn stanza(m: &LocalModel) -> String {
    toml::to_string(&Stanza { model: StanzaModel { local: [m] } }).unwrap_or_default()
}

pub fn local_model(id: &str, backend: &str, role: &str, file: &str, context: u64, accelerated: bool) -> LocalModel {
    let llama = backend == "llamacpp";
    LocalModel {
        id: id.to_string(),
        backend: Some(backend.to_string()),
        role: Some(role.to_string()),
        file: Some(file.to_string()),
        context: Some(context.min(u64::from(u32::MAX)) as u32),
        gpu_layers: (llama && accelerated).then_some(99),
        idle_ttl: Some("1800s".into()),
        args: if llama { ["-fa", "on", "--jinja"].map(String::from).to_vec() } else { Vec::new() },
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(name: &str, size: u64) -> RepoFile {
        RepoFile { name: name.into(), size, sha256: None }
    }

    #[test]
    fn finds_the_quant_tag() {
        for (name, want) in [
            ("Qwen3-8B-Q4_K_M.gguf", Some("Q4_K_M")),
            ("model.IQ4_XS.gguf", Some("IQ4_XS")),
            ("Qwen3-8B-UD-Q4_K_XL.gguf", Some("UD-Q4_K_XL")),
            ("sub/Model-7B-BF16.gguf", Some("BF16")),
            ("gpt-oss-20b-MXFP4.gguf", Some("MXFP4")),
            ("big-Q8_0-00001-of-00004.gguf", Some("Q8_0")),
            ("Qwen3-0.6B.gguf", None),
        ] {
            assert_eq!(quant_of(name).as_deref(), want, "{name}");
        }
    }

    #[test]
    fn groups_shards_and_drops_incomplete_sets_projectors_and_imatrix() {
        let files = vec![
            f("m-Q8_0-00001-of-00002.gguf", 10),
            f("m-Q8_0-00002-of-00002.gguf", 10),
            f("m-Q4_K_M.gguf", 9),
            f("m-Q6_K-00001-of-00002.gguf", 8),
            f("mmproj-m-f16.gguf", 1),
            f("m.imatrix.gguf", 1),
            f("README.md", 1),
        ];
        let sets = quant_sets(&files);
        assert_eq!(sets.iter().map(|s| (s.label.as_str(), s.size, s.files.len())).collect::<Vec<_>>(), [("Q4_K_M", 9, 1), ("Q8_0", 20, 2)]);
        assert_eq!(sets[1].files[0].name, "m-Q8_0-00001-of-00002.gguf");
    }

    #[test]
    fn same_quant_in_two_folders_stays_two_sets() {
        let sets = quant_sets(&[f("a/m-Q4_K_M.gguf", 5), f("b/m-Q4_K_M.gguf", 6)]);
        assert_eq!(sets.len(), 2);
    }

    #[test]
    fn picks_the_best_quant_that_fits_with_the_kv_cache() {
        let sets = quant_sets(&[f("m-Q8_0.gguf", 8 * GIB), f("m-Q6_K.gguf", 6 * GIB), f("m-Q4_K_M.gguf", 4 * GIB), f("m-F16.gguf", 16 * GIB)]);
        let cap = Capacity { fast: 10 * GIB, total: 10 * GIB, accelerated: true };
        assert_eq!(pick(&sets, 100_000, 8192, &cap).unwrap().label, "Q8_0");
        // A 1 MB/token cache at 32k is 32 GiB of KV: nothing fits.
        assert!(pick(&sets, 1 << 20, 32_768, &cap).is_none());
        // 128 KiB/token at 32k is 4 GiB of KV: only Q4_K_M leaves room (4 + 4 + 0.5 <= 10; Q6_K is 10.5).
        assert_eq!(pick(&sets, 128 * 1024, 32_768, &cap).unwrap().label, "Q4_K_M");
        assert_eq!(smallest(&sets).unwrap().label, "Q4_K_M");
    }

    #[test]
    fn full_precision_is_a_last_resort() {
        let sets = quant_sets(&[f("m-Q4_K_M.gguf", 4 * GIB), f("m-F16.gguf", 2 * GIB)]);
        let cap = Capacity { fast: 100 * GIB, total: 100 * GIB, accelerated: false };
        assert_eq!(pick(&sets, 1000, 4096, &cap).unwrap().label, "Q4_K_M");
        let tight = Capacity { fast: 3 * GIB, total: 3 * GIB, accelerated: false };
        assert_eq!(pick(&sets, 1000, 4096, &tight).unwrap().label, "F16");
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
        assert_eq!(kv_per_token(Some(5), 10 * GIB), (5, KvSource::Header));
        assert_eq!(kv_per_token(None, 10 * GIB), (10 * 32 * 1024, KvSource::Estimate));
    }

    #[test]
    fn stanza_mirrors_the_setup_shape_and_parses_back() {
        let m = local_model("tiny", "llamacpp", "chat", "/srv/models/gguf/tiny-Q8_0.gguf", 32_768, true);
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
        assert!(!stanza(&local_model("t", "llamacpp", "chat", "/x.gguf", 1, false)).contains("gpu_layers"));
    }

    #[test]
    fn ids_come_from_the_repo_or_the_file() {
        assert_eq!(id_for_repo("bartowski/Qwen3-8B-GGUF"), "qwen3-8b");
        assert_eq!(id_for_file("big-Q8_0-00001-of-00004.gguf"), "big-q8_0");
        assert_eq!(id_for_file("Tiny-Q4_K_M.gguf"), "tiny-q4_k_m");
    }
}
