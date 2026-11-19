//! The usable memory budget and which models fit in it.

use serde::{Deserialize, Serialize};
use weirdnet_model::ModelSpec;

use crate::{Accel, GIB, Hardware};

/// What a model is for on a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Purpose {
    Chat,
    Embedding,
}

/// A catalog model with what it needs in memory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatalogEntry {
    pub spec: ModelSpec,
    pub purpose: Purpose,
    pub quant: String,
    /// Total and active parameters, in billions (they differ for mixture-of-experts models).
    pub params_b: f32,
    pub active_params_b: f32,
    /// KV cache bytes per token at f16 (layers x KV heads x head dim x 2 x 2).
    pub kv_bytes_per_token: u64,
    /// Context the sizing assumes per slot.
    pub context: u32,
}

impl CatalogEntry {
    pub fn id(&self) -> &str {
        &self.spec.id
    }

    /// Weights plus KV for `slots` concurrent requests at the entry's context, plus 10% for compute
    /// buffers.
    pub fn footprint(&self, slots: u32) -> u64 {
        let kv = self.kv_bytes_per_token * self.context as u64 * slots as u64;
        (self.spec.size + kv) + (self.spec.size + kv) / 10
    }
}

fn spec(id: &str, label: &str, repo: &str, file: &str, size: u64, sha256: &str) -> ModelSpec {
    ModelSpec { id: id.into(), label: label.into(), repo: repo.into(), file: file.into(), size, sha256: Some(sha256.into()) }
}

#[allow(clippy::too_many_arguments)]
fn chat(spec: ModelSpec, quant: &str, params_b: f32, active_params_b: f32, kv_bytes_per_token: u64, context: u32) -> CatalogEntry {
    CatalogEntry { spec, purpose: Purpose::Chat, quant: quant.into(), params_b, active_params_b, kv_bytes_per_token, context }
}

/// The tiered catalog, smallest first. Sizes and SHA-256 from the Hugging Face API (2026-11-19);
/// the Gemma 4 files are `weirdnet-model`'s own pinned entries.
pub fn catalog() -> Vec<CatalogEntry> {
    let gemma = |id: &str| weirdnet_model::catalog::find(id).expect("weirdnet-model ships Gemma 4");
    vec![
        chat(
            spec(
                "smollm2-135m",
                "SmolLM2 135M instruct (Q8_0)",
                "bartowski/SmolLM2-135M-Instruct-GGUF",
                "SmolLM2-135M-Instruct-Q8_0.gguf",
                144_811_360,
                "5a1395716f7913741cc51d98581b9b1228d80987a9f7d3664106742eb06bba83",
            ),
            "Q8_0",
            0.135,
            0.135,
            23_040,
            8192,
        ),
        chat(gemma(weirdnet_model::catalog::GEMMA_4_E2B), "Q4_0", 5.1, 2.0, 32_768, 8192),
        chat(gemma(weirdnet_model::catalog::GEMMA_4_E4B), "Q4_0", 8.0, 4.0, 65_536, 8192),
        chat(
            spec(
                "qwen3-8b",
                "Qwen3 8B (Q4_K_M)",
                "unsloth/Qwen3-8B-GGUF",
                "Qwen3-8B-Q4_K_M.gguf",
                5_027_784_512,
                "120307ba529eb2439d6c430d94104dabd578497bc7bfe7e322b5d9933b449bd4",
            ),
            "Q4_K_M",
            8.2,
            8.2,
            147_456,
            8192,
        ),
        chat(
            spec(
                "qwen3-14b",
                "Qwen3 14B (Q4_K_M)",
                "unsloth/Qwen3-14B-GGUF",
                "Qwen3-14B-Q4_K_M.gguf",
                9_001_753_984,
                "5eaa0870bd81ed3b58a630a271234cfa604e43ffb3a19cd68e54a80dd9d52a66",
            ),
            "Q4_K_M",
            14.8,
            14.8,
            163_840,
            8192,
        ),
        chat(
            spec(
                "gpt-oss-20b",
                "gpt-oss 20B (MXFP4)",
                "ggml-org/gpt-oss-20b-GGUF",
                "gpt-oss-20b-MXFP4.gguf",
                12_109_566_624,
                "27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901",
            ),
            "MXFP4",
            21.0,
            3.6,
            49_152,
            8192,
        ),
        chat(
            spec(
                "qwen3-30b-a3b",
                "Qwen3 30B-A3B (Q4_K_M)",
                "unsloth/Qwen3-30B-A3B-GGUF",
                "Qwen3-30B-A3B-Q4_K_M.gguf",
                18_556_686_912,
                "9f1a24700a339b09c06009b729b5c809e0b64c213b8af5b711b3dbdfd0c5ba48",
            ),
            "Q4_K_M",
            30.5,
            3.3,
            98_304,
            8192,
        ),
        chat(
            spec(
                "qwen3-32b",
                "Qwen3 32B (Q4_K_M)",
                "unsloth/Qwen3-32B-GGUF",
                "Qwen3-32B-Q4_K_M.gguf",
                19_762_150_048,
                "8df67573b2c23484e02ec7af295e39bed7ee774f3771d5fda2978265b59370e7",
            ),
            "Q4_K_M",
            32.8,
            32.8,
            262_144,
            8192,
        ),
        CatalogEntry {
            spec: spec(
                "qwen3-embedding-0.6b",
                "Qwen3 Embedding 0.6B (Q8_0)",
                "Qwen/Qwen3-Embedding-0.6B-GGUF",
                "Qwen3-Embedding-0.6B-Q8_0.gguf",
                639_150_592,
                "06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439",
            ),
            purpose: Purpose::Embedding,
            quant: "Q8_0".into(),
            params_b: 0.6,
            active_params_b: 0.6,
            kv_bytes_per_token: 114_688,
            context: 2048,
        },
    ]
}

/// The catalog entry with this id.
pub fn find(id: &str) -> Option<CatalogEntry> {
    catalog().into_iter().find(|e| e.spec.id == id)
}

/// How much memory models may use on this node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Budget {
    /// VRAM of the first GPU, or system memory on unified and CPU-only machines.
    pub pool_bytes: u64,
    /// Kept for the OS (unified and CPU-only only).
    pub reserve_bytes: u64,
    pub usable_bytes: u64,
}

/// 85% of VRAM, or 85% of unified memory after the OS reserve (default: an eighth of RAM, at least
/// 4 GiB). `memory_budget` overrides the result; `reserve` overrides the reserve.
pub fn budget(hw: &Hardware, memory_budget: Option<u64>, reserve: Option<u64>) -> Budget {
    let unified = hw.unified();
    let pool_bytes = if unified { hw.ram_bytes } else { hw.gpus.first().map(|g| g.memory_bytes).unwrap_or(hw.ram_bytes) };
    let reserve_bytes = if unified { reserve.unwrap_or_else(|| (hw.ram_bytes / 8).max(4 * GIB)) } else { 0 };
    let usable = pool_bytes.saturating_sub(reserve_bytes) / 100 * 85;
    Budget { pool_bytes, reserve_bytes, usable_bytes: memory_budget.unwrap_or(usable) }
}

/// What setup proposes for this node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Plan {
    pub budget: Budget,
    pub main: Option<CatalogEntry>,
    pub standin: Option<CatalogEntry>,
    pub embedding: Option<CatalogEntry>,
    /// Concurrent requests each chat model is sized for.
    pub slots: u32,
    pub notes: Vec<String>,
}

/// Picks the largest chat model that fits with its KV budget, a Gemma 4 stand-in and the embedding
/// model. CPU-only machines only get models with at most 4B active parameters, so answers stay
/// usable.
pub fn plan(hw: &Hardware, budget: Budget) -> Plan {
    let slots = 2;
    let cpu_only = hw.accel() == Accel::Cpu;
    let mut notes = Vec::new();
    let all = catalog();
    let fits = |e: &&CatalogEntry| e.footprint(slots) <= budget.usable_bytes;
    let main = all
        .iter()
        .filter(|e| e.purpose == Purpose::Chat && e.spec.id != "smollm2-135m")
        .filter(|e| !cpu_only || e.active_params_b <= 4.0)
        .filter(fits)
        .max_by(|a, b| a.params_b.total_cmp(&b.params_b))
        .cloned();
    if cpu_only {
        notes.push("CPU only: limited to models with at most 4B active parameters".into());
    }
    if main.is_none() {
        notes.push("no catalog chat model fits; add a smaller GGUF with `ymirctl setup --model PATH`".into());
    }
    let standin_id = if budget.usable_bytes >= 16 * GIB { weirdnet_model::catalog::GEMMA_4_E4B } else { weirdnet_model::catalog::GEMMA_4_E2B };
    let standin = all.iter().find(|e| e.spec.id == standin_id).filter(fits).filter(|e| main.as_ref().is_none_or(|m| m.spec.id != e.spec.id)).cloned();
    let embedding = all.iter().find(|e| e.purpose == Purpose::Embedding).filter(|e| e.footprint(1) <= budget.usable_bytes).cloned();
    Plan { budget, main, standin, embedding, slots, notes }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Gpu, parse_nvidia_smi};

    fn box_with(gpus: Vec<Gpu>, ram_gib: u64) -> Hardware {
        Hardware { os: "linux".into(), arch: "x86_64".into(), cpu_threads: 16, ram_bytes: ram_gib * GIB, gpus }
    }

    #[test]
    fn rtx_3060_gets_an_8b() {
        let hw = box_with(parse_nvidia_smi("NVIDIA GeForce GPU, 12288, 8.6, 615.78.08", 62 * GIB), 62);
        let b = budget(&hw, None, None);
        assert_eq!(b.pool_bytes, 12 * GIB);
        let p = plan(&hw, b);
        assert_eq!(p.main.unwrap().id(), "qwen3-8b");
        assert_eq!(p.standin.unwrap().id(), "gemma-4-e2b");
        assert!(p.embedding.is_some());
    }

    #[test]
    fn spark_gets_a_big_model() {
        let hw = box_with(parse_nvidia_smi("NVIDIA GB10, [N/A], 12.1, 580.95.05", 128 * GIB), 128);
        let b = budget(&hw, None, None);
        assert_eq!(b.reserve_bytes, 16 * GIB);
        let p = plan(&hw, b);
        assert_eq!(p.main.unwrap().id(), "qwen3-32b");
        assert_eq!(p.standin.unwrap().id(), "gemma-4-e4b");
    }

    #[test]
    fn cpu_box_prefers_few_active_params() {
        let hw = box_with(Vec::new(), 62);
        let p = plan(&hw, budget(&hw, None, None));
        assert_eq!(p.main.unwrap().id(), "qwen3-30b-a3b");
    }

    #[test]
    fn tiny_box_gets_no_main() {
        let hw = box_with(Vec::new(), 4);
        let p = plan(&hw, budget(&hw, Some(GIB), None));
        assert!(p.main.is_none());
        assert!(!p.notes.is_empty());
    }

    #[test]
    fn override_budget() {
        let hw = box_with(Vec::new(), 64);
        assert_eq!(budget(&hw, Some(10 * GIB), None).usable_bytes, 10 * GIB);
        assert_eq!(budget(&hw, None, Some(14 * GIB)).reserve_bytes, 14 * GIB);
    }
}
