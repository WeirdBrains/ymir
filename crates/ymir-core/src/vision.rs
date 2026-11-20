//! Vision projectors (`mmproj` files) for the catalog models that take images.
//!
//! llama.cpp reads images through a separate projector GGUF next to the model. Ymir pins one per
//! vision-capable catalog model (from the same upstream repo as the weights), `ymirctl setup
//! --download` fetches it with the model, and `ymird` passes it to `llama-server` as `--mmproj`
//! whenever it is on disk, so images sent through the OpenAI chat API just work. `mmproj` in
//! `[[model.local]]` points at another projector, or `off` turns images off.
//!
//! The Q8_0 projectors are used: about half the size of BF16 with no visible loss for chat use.
//! The other catalog models (Qwen3, gpt-oss, SmolLM2, the embedding model) are text-only.

use std::path::{Path, PathBuf};

use weirdnet_model::ModelSpec;

/// A pinned projector for a catalog model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Projector {
    /// The catalog model it belongs to.
    pub model: &'static str,
    pub repo: &'static str,
    pub file: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

pub const PROJECTORS: &[Projector] = &[
    Projector {
        model: "gemma-4-e2b",
        repo: "ggml-org/gemma-4-E2B-it-GGUF",
        file: "mmproj-gemma-4-E2B-it-Q8_0.gguf",
        size: 557_368_096,
        sha256: "042cc2561806cd3b5cb5efb25cab76911e3100c43cc09d854904f921924b1429",
    },
    Projector {
        model: "gemma-4-e4b",
        repo: "ggml-org/gemma-4-E4B-it-GGUF",
        file: "mmproj-gemma-4-E4B-it-Q8_0.gguf",
        size: 559_874_848,
        sha256: "9257f21a0f5358204b4a612cff83fad2e56412f4f607f0049331ce60c0cf9d34",
    },
];

/// The projector for catalog model `id`, if it takes images.
pub fn for_model(id: &str) -> Option<&'static Projector> {
    PROJECTORS.iter().find(|p| p.model == id)
}

impl Projector {
    /// As a download spec for `weirdnet-model` (checksummed, with consent).
    pub fn spec(&self) -> ModelSpec {
        ModelSpec {
            id: format!("{}-mmproj", self.model),
            label: format!("{} vision projector (Q8_0)", self.model),
            repo: self.repo.into(),
            file: self.file.into(),
            size: self.size,
            sha256: Some(self.sha256.into()),
        }
    }

    pub fn path(&self, models_dir: &Path) -> PathBuf {
        models_dir.join(self.file)
    }
}

/// The projector `ymird` passes for a llama.cpp model: `mmproj` from node.toml (a path, absolute
/// or under `models_dir`; `off` for none), else the catalog projector when it is downloaded.
pub fn resolve(id: &str, setting: Option<&str>, models_dir: &Path) -> Option<PathBuf> {
    match setting.map(str::trim) {
        Some("off" | "none" | "") => None,
        Some(p) => Some(if Path::new(p).is_absolute() { PathBuf::from(p) } else { models_dir.join(p) }),
        None => for_model(id).map(|p| p.path(models_dir)).filter(|p| std::fs::metadata(p).is_ok_and(|m| m.len() > 0)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projectors_belong_to_catalog_models() {
        for p in PROJECTORS {
            let e = weirdnet_model::hw::find(p.model).unwrap_or_else(|| panic!("{} is not in the catalog", p.model));
            assert_eq!(e.spec.repo, p.repo, "{}: projector from another repo", p.model);
            assert!(p.file.starts_with("mmproj-") && p.sha256.len() == 64);
        }
        assert!(for_model("qwen3-8b").is_none());
    }

    #[test]
    fn resolve_settings() {
        let d = tempfile::tempdir().unwrap();
        assert_eq!(resolve("gemma-4-e4b", None, d.path()), None, "not downloaded");
        std::fs::write(d.path().join("mmproj-gemma-4-E4B-it-Q8_0.gguf"), b"x").unwrap();
        assert_eq!(resolve("gemma-4-e4b", None, d.path()), Some(d.path().join("mmproj-gemma-4-E4B-it-Q8_0.gguf")));
        assert_eq!(resolve("gemma-4-e4b", Some("off"), d.path()), None);
        assert_eq!(resolve("x", Some("p.gguf"), d.path()), Some(d.path().join("p.gguf")));
        let abs = d.path().join("elsewhere").join("p.gguf");
        assert_eq!(resolve("x", Some(abs.to_str().unwrap()), Path::new("models")), Some(abs));
    }
}
