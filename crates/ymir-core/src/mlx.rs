//! MLX-LM, the default engine on Apple Silicon Macs.
//!
//! On an arm64 Mac, `ymirctl setup` serves chat models with Apple's MLX through `mlx_lm.server`
//! instead of llama.cpp: MLX-format weights from `mlx-community` run faster on the Mac's GPU and
//! unified memory than GGUF through llama.cpp's Metal backend. llama.cpp stays for what MLX-LM's
//! server cannot do (embeddings), and for catalog models without MLX weights.
//!
//! MLX-LM is Python. Setup installs the `mlx-lm` that `weirdnet-backends` pins into a venv under
//! Ymir's engines dir (`<home>/engines/mlx`) with [`install`] (`uv` when it is on `PATH`, else
//! `python3 -m venv`), and downloads the MLX weights for the proposed models, pinned to a
//! revision, into `<models_dir>/mlx/<name>`; loads then need no network. The supervisor runs the
//! model as `EngineKind::Mlx`, behind its own proxy (API key, readiness after the first token,
//! text only, no embeddings), so it looks like any other engine to `ymird`. Installing and
//! running are `weirdnet-backends`'; the weights catalog and their download stay here.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use weirdnet_backends::find_in_path;

pub use weirdnet_backends::mlx::{MLX_LM_VERSION, install, server_in, supported};

/// MLX-format weights for a catalog model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Weights {
    /// The catalog id (`weirdnet_model::hw::catalog`).
    pub id: &'static str,
    /// The Hugging Face repo.
    pub repo: &'static str,
    /// The pinned revision (commit).
    pub revision: &'static str,
    /// Total size of the repo's files.
    pub size: u64,
}

impl Weights {
    /// The directory under `models_dir` the weights go in.
    pub fn dir(&self, models_dir: &Path) -> PathBuf {
        models_dir.join("mlx").join(self.repo.rsplit('/').next().unwrap_or(self.repo))
    }

    pub fn installed(&self, models_dir: &Path) -> bool {
        let d = self.dir(models_dir);
        d.join("config.json").is_file()
            && std::fs::read_dir(&d).is_ok_and(|r| r.flatten().any(|e| e.path().extension().is_some_and(|x| x == "safetensors")))
    }
}

/// Catalog models with MLX weights (4-bit, or MXFP4 for gpt-oss, close to the GGUF quants the
/// catalog sizes). The embedding model has none: MLX-LM's server does not serve embeddings.
pub const WEIGHTS: &[Weights] = &[
    Weights {
        id: "smollm2-135m",
        repo: "mlx-community/SmolLM2-135M-Instruct-8bit",
        revision: "0f0d9b8218915bc34d401e1a340b8c049d300d5e",
        size: 147_877_089,
    },
    Weights {
        id: "gemma-4-e2b",
        repo: "mlx-community/gemma-4-e2b-it-4bit",
        revision: "238767527555cb75a05732a84dff5d6ba0dd6809",
        size: 3_583_088_661,
    },
    Weights {
        id: "gemma-4-e4b",
        repo: "mlx-community/gemma-4-e4b-it-4bit",
        revision: "475b9088d29754a3379866cf5aeb6b41acd313c2",
        size: 5_179_241_512,
    },
    Weights { id: "qwen3-8b", repo: "mlx-community/Qwen3-8B-4bit", revision: "545dc4251c05440727734bcd94334791f6ab0192", size: 4_623_784_971 },
    Weights { id: "qwen3-14b", repo: "mlx-community/Qwen3-14B-4bit", revision: "a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4", size: 8_323_870_520 },
    Weights {
        id: "gpt-oss-20b",
        repo: "mlx-community/gpt-oss-20b-MXFP4-Q4",
        revision: "f356f2747216d7e98fee755df25987459fc19089",
        size: 11_206_563_096,
    },
    Weights {
        id: "qwen3-30b-a3b",
        repo: "mlx-community/Qwen3-30B-A3B-4bit",
        revision: "d388dead1515f5e085ef7a0431dd8fadf0886c57",
        size: 17_190_786_248,
    },
    Weights { id: "qwen3-32b", repo: "mlx-community/Qwen3-32B-4bit", revision: "bcaaf7f538adf166c1080a2befdb4f6019f66639", size: 18_445_874_457 },
];

pub fn weights(id: &str) -> Option<&'static Weights> {
    WEIGHTS.iter().find(|w| w.id == id)
}

/// The managed venv.
pub fn venv(engines_dir: &Path) -> PathBuf {
    engines_dir.join("mlx")
}

/// Where `mlx = "managed"` finds `mlx_lm.server`: the managed venv, then `PATH`.
pub fn resolve(setting: &str, engines_dir: &Path) -> Option<PathBuf> {
    match setting.trim() {
        "off" => None,
        s if s.starts_with("system:") => Some(PathBuf::from(&s["system:".len()..])),
        _ => Some(server_in(&venv(engines_dir))).filter(|p| p.is_file()).or_else(|| find_in_path("mlx_lm.server")),
    }
}

/// Downloads `w` (at its pinned revision) into `models_dir`, with the venv's `huggingface_hub`.
pub fn download(venv: &Path, w: &Weights, models_dir: &Path) -> Result<PathBuf> {
    let dest = w.dir(models_dir);
    std::fs::create_dir_all(&dest)?;
    let script = "import sys\nfrom huggingface_hub import snapshot_download\nsnapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])\n";
    let mut cmd = Command::new(venv.join("bin").join("python"));
    cmd.args(["-c", script, w.repo, w.revision]).arg(&dest);
    let s = cmd.status().with_context(|| format!("running {cmd:?}"))?;
    if !s.success() {
        bail!("downloading {} failed ({s})", w.repo);
    }
    Ok(dest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_mapping_is_a_catalog_chat_model() {
        for w in WEIGHTS {
            let e = weirdnet_model::hw::find(w.id).unwrap_or_else(|| panic!("{} is not in the catalog", w.id));
            assert!(!e.spec.id.contains("embedding"));
            assert!(w.repo.starts_with("mlx-community/") && w.revision.len() == 40);
        }
        assert!(weights("qwen3-embedding-0.6b").is_none());
        assert_eq!(weights("qwen3-8b").unwrap().dir(Path::new("/m")), PathBuf::from("/m/mlx/Qwen3-8B-4bit"));
    }

    #[test]
    fn resolve_settings() {
        let d = tempfile::tempdir().unwrap();
        assert_eq!(resolve("off", d.path()), None);
        assert_eq!(resolve("system:/x/mlx_lm.server", d.path()), Some(PathBuf::from("/x/mlx_lm.server")));
        let s = server_in(&venv(d.path()));
        std::fs::create_dir_all(s.parent().unwrap()).unwrap();
        std::fs::write(&s, b"").unwrap();
        assert_eq!(resolve("managed", d.path()), Some(s));
    }
}
