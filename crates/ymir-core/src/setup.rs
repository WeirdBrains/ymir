//! `ymirctl setup`: hardware in, a proposed `node.toml` out.

use std::path::PathBuf;

use weirdnet_backends::find_in_path;
use weirdnet_model::hw::{Budget, CatalogEntry, Hardware, Plan};

use crate::config::{LocalModel, NodeConfig, Roles};

#[derive(Debug, Clone, Default)]
pub struct SetupOptions {
    pub name: Option<String>,
    /// An existing `llama-server`; default: look on `PATH`.
    pub llama_server: Option<PathBuf>,
    /// GGUF files to serve instead of catalog models (the first is the main model).
    pub models: Vec<PathBuf>,
    /// Add the catalog proposal even when `models` is given.
    pub catalog: bool,
    pub memory_budget: Option<String>,
    pub reserve: Option<String>,
    pub models_dir: Option<PathBuf>,
    pub bridge_port: Option<u16>,
}

/// What setup found and proposes.
#[derive(Debug, Clone)]
pub struct Proposal {
    pub hardware: Hardware,
    pub plan: Plan,
    pub config: NodeConfig,
    pub llama_server: Option<PathBuf>,
    pub vllm: Option<PathBuf>,
    /// Catalog models in the config that aren't downloaded yet.
    pub to_download: Vec<CatalogEntry>,
}

pub fn budget(hw: &Hardware, opts: &SetupOptions) -> Budget {
    weirdnet_model::hw::budget(
        hw,
        opts.memory_budget.as_deref().and_then(weirdnet_model::hw::parse_size),
        opts.reserve.as_deref().and_then(weirdnet_model::hw::parse_size),
    )
}

pub fn propose(hw: Hardware, opts: &SetupOptions) -> Proposal {
    let plan = weirdnet_model::hw::plan(&hw, budget(&hw, opts));
    let mut cfg = NodeConfig::default();
    cfg.node.name = opts.name.clone().unwrap_or_else(hostname);
    cfg.roles = Roles::ALL;
    cfg.model.memory_budget = opts.memory_budget.clone().unwrap_or_else(|| "auto".into());
    cfg.model.reserve.clone_from(&opts.reserve);
    cfg.model.models_dir.clone_from(&opts.models_dir);
    if let Some(p) = opts.bridge_port {
        cfg.app.bridge.port = p;
    }
    let llama_server = opts.llama_server.clone().or_else(|| find_in_path("llama-server"));
    if let Some(p) = &llama_server {
        cfg.model.llamacpp = format!("system:{}", std::fs::canonicalize(p).unwrap_or_else(|_| p.clone()).display());
    }
    let vllm = if hw.vllm_supported() { find_in_path("vllm") } else { None };
    cfg.model.vllm = if hw.vllm_supported() { "auto".into() } else { "off".into() };

    for (i, path) in opts.models.iter().enumerate() {
        let stem = path.file_stem().map(|s| s.to_string_lossy().to_ascii_lowercase()).unwrap_or_else(|| format!("model-{i}"));
        cfg.model.local.push(LocalModel {
            id: stem.replace(char::is_whitespace, "-"),
            role: Some(if i == 0 { "main" } else { "chat" }.into()),
            swap_group: Some("main".into()),
            file: Some(std::fs::canonicalize(path).unwrap_or_else(|_| path.clone()).to_string_lossy().into_owned()),
            ..Default::default()
        });
    }
    let mut to_download = Vec::new();
    if opts.models.is_empty() || opts.catalog {
        let store = weirdnet_model::ModelStore::new(cfg.models_dir());
        let both_fit = match (&plan.main, &plan.standin) {
            (Some(m), Some(s)) => m.footprint(plan.slots) + s.footprint(plan.slots) <= plan.budget.usable_bytes,
            _ => true,
        };
        let picks = [
            (plan.main.as_ref(), "main", Some("main")),
            (plan.standin.as_ref(), "standin", if both_fit { None } else { Some("main") }),
            (plan.embedding.as_ref(), "embedding", None),
        ];
        for (entry, role, group) in picks {
            let Some(e) = entry else { continue };
            if cfg.model.local.iter().any(|m| m.id == e.spec.id) {
                continue;
            }
            let role = if role == "main" && !opts.models.is_empty() { "chat" } else { role };
            cfg.model.local.push(LocalModel {
                id: e.spec.id.clone(),
                backend: Some("llamacpp".into()),
                quant: Some(e.quant.clone()),
                swap_group: group.map(str::to_string),
                role: Some(role.into()),
                context: Some(e.context),
                slots: Some(if role == "embedding" { 1 } else { plan.slots }),
                ..Default::default()
            });
            if !store.installed(&e.spec) {
                to_download.push(e.clone());
            }
        }
    }
    Proposal { hardware: hw, plan, config: cfg, llama_server, vllm, to_download }
}

/// This machine's host name, else `ymir`.
pub fn hostname() -> String {
    let mut buf = [0u8; 256];
    // SAFETY: the buffer is valid for its length; gethostname NUL-terminates on success.
    let rc = unsafe { libc::gethostname(buf.as_mut_ptr().cast(), buf.len()) };
    if rc == 0 {
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        let name = String::from_utf8_lossy(&buf[..end]).trim().to_string();
        if !name.is_empty() {
            return name;
        }
    }
    "ymir".into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use weirdnet_model::hw::{GIB, parse_nvidia_smi};

    fn nvidia_pc() -> Hardware {
        Hardware {
            os: "linux".into(),
            arch: "x86_64".into(),
            cpu_threads: 16,
            ram_bytes: 62 * GIB,
            gpus: parse_nvidia_smi("NVIDIA GeForce GPU, 12288, 8.6, 615.78.08", 62 * GIB),
        }
    }

    #[test]
    fn catalog_proposal() {
        let dir = tempfile::tempdir().unwrap();
        let p = propose(
            nvidia_pc(),
            &SetupOptions {
                name: Some("box".into()),
                models_dir: Some(dir.path().into()),
                llama_server: Some("/opt/llama-server".into()),
                ..Default::default()
            },
        );
        let c = &p.config;
        assert_eq!(c.roles, Roles::ALL);
        assert_eq!(c.main_model().unwrap().id, "qwen3-8b");
        assert_eq!(c.model.llamacpp, "system:/opt/llama-server");
        assert_eq!(p.to_download.len(), 3);
        // 12 GB of VRAM holds the 8B or the stand-in, not both: they share the swap group.
        assert_eq!(c.model.local.iter().find(|m| m.id == "gemma-4-e2b").unwrap().swap_group.as_deref(), Some("main"));
        c.validate().unwrap();
    }

    #[test]
    fn own_gguf_replaces_the_catalog() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("Tiny Model.gguf");
        std::fs::write(&f, b"x").unwrap();
        let p = propose(nvidia_pc(), &SetupOptions { models: vec![f], ..Default::default() });
        assert_eq!(p.config.model.local.len(), 1);
        assert_eq!(p.config.main_model().unwrap().id, "tiny-model");
        assert!(p.to_download.is_empty());
    }
}
