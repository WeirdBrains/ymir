//! LoRA adapters on a loaded base (training plan P4a). Serving them is `weirdnet-backends`'
//! (`ModelLaunch::adapters`, `Supervisor::add_adapter`, per-request adapter and scale); this module
//! keeps the Ymir side: which configured adapters can be served, which base an adapter id runs
//! on, and the memory an adapter adds to its base for the context planner.

use std::path::Path;

use weirdnet_backends::{EngineKind, ModelLaunch};

/// Drops, with an error in the log, each adapter that can't be served on its base (a file that
/// is missing or not a LoRA for that base's architecture, ...), so the base and the other
/// adapters still serve. Checked one at a time, in order.
pub fn keep_servable(launches: &mut [ModelLaunch]) {
    for l in launches.iter_mut() {
        if l.check_adapters().is_ok() {
            continue;
        }
        let all = std::mem::take(&mut l.adapters);
        for a in all {
            let mut trial = l.clone();
            trial.adapters.push(a.clone());
            match trial.check_adapters() {
                Ok(()) => l.adapters.push(a),
                Err(e) => tracing::error!(model = %a.id, base = %l.id, "not serving LoRA adapter {}: {e}", a.id),
            }
        }
    }
}

/// The base model `id` runs on: itself for a model, its base for an adapter, `None` when unknown.
pub fn base_of(launches: &[ModelLaunch], id: &str) -> Option<String> {
    launches.iter().find_map(|l| (l.id == id || l.adapters.iter().any(|a| a.id == id)).then(|| l.id.clone()))
}

/// The bytes `l`'s adapters add to its memory: each adapter's weights, as on disk. llama.cpp
/// loads every `--lora` GGUF whole at start; vLLM keeps each listed adapter's tensors
/// (`--max-loras` is the number of adapters). A Hub id not yet downloaded counts as nothing.
pub fn bytes(l: &ModelLaunch) -> u64 {
    l.adapters
        .iter()
        .map(|a| {
            let p = Path::new(&a.path);
            match (l.engine, p.is_dir()) {
                (EngineKind::Vllm, true) => dir_weights(p),
                _ => std::fs::metadata(p).ok().filter(|m| m.is_file()).map_or(0, |m| m.len()),
            }
        })
        .sum()
}

/// A PEFT directory's weight files (`*.safetensors`, `*.bin`).
fn dir_weights(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| {
                    let n = e.file_name().to_string_lossy().into_owned();
                    n.ends_with(".safetensors") || n.ends_with(".bin")
                })
                .filter_map(|e| e.metadata().ok())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use weirdnet_backends::LoraAdapter;

    use super::*;

    /// GGUF header bytes (version 3, no tensors) with these string and u32 keys.
    pub(crate) fn gguf(kv: &[(&str, &str)], ints: &[(&str, u32)]) -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend(3u32.to_le_bytes());
        out.extend(0u64.to_le_bytes());
        out.extend(((kv.len() + ints.len()) as u64).to_le_bytes());
        let s = |out: &mut Vec<u8>, s: &str| {
            out.extend((s.len() as u64).to_le_bytes());
            out.extend(s.as_bytes());
        };
        for (k, v) in kv {
            s(&mut out, k);
            out.extend(8u32.to_le_bytes());
            s(&mut out, v);
        }
        for (k, v) in ints {
            s(&mut out, k);
            out.extend(4u32.to_le_bytes());
            out.extend(v.to_le_bytes());
        }
        out
    }

    fn lora(arch: &str) -> Vec<u8> {
        gguf(&[("general.architecture", arch), ("general.type", "adapter"), ("adapter.type", "lora")], &[])
    }

    #[test]
    fn bad_adapters_are_dropped_and_good_ones_kept() {
        let t = tempfile::tempdir().unwrap();
        let w = |name: &str, b: Vec<u8>| {
            let p = t.path().join(name);
            std::fs::write(&p, b).unwrap();
            p.to_string_lossy().into_owned()
        };
        let base = w("base.gguf", gguf(&[("general.architecture", "llama")], &[("llama.block_count", 1)]));
        let good = w("good.gguf", lora("llama"));
        let other = w("other.gguf", lora("qwen3"));
        let mut launches = vec![ModelLaunch::new("m", EngineKind::Llamacpp, &base).with_adapters(vec![
            LoraAdapter::new("m+good", &good),
            LoraAdapter::new("m+other", &other),
            LoraAdapter::new("m+gone", t.path().join("gone.gguf").to_string_lossy()),
        ])];
        keep_servable(&mut launches);
        assert_eq!(launches[0].served_ids(), ["m", "m+good"]);
        assert_eq!(base_of(&launches, "m+good").as_deref(), Some("m"));
        assert_eq!(base_of(&launches, "m").as_deref(), Some("m"));
        assert_eq!(base_of(&launches, "m+other"), None);
        assert_eq!(bytes(&launches[0]), std::fs::metadata(&good).unwrap().len());
    }

    #[test]
    fn vllm_adapter_bytes_are_its_weight_files() {
        let t = tempfile::tempdir().unwrap();
        let d = t.path().join("tune");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("adapter_config.json"), br#"{"peft_type":"LORA"}"#).unwrap();
        std::fs::write(d.join("adapter_model.safetensors"), vec![0u8; 1000]).unwrap();
        let l = ModelLaunch::new("b", EngineKind::Vllm, "org/b")
            .with_adapters(vec![LoraAdapter::new("b+t", d.to_string_lossy()), LoraAdapter::new("b+hub", "org/not-here")]);
        assert_eq!(bytes(&l), 1000);
    }
}
