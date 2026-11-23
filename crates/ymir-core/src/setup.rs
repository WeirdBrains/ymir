//! `ymirctl setup`: hardware in, a proposed `node.toml` out.

use std::path::PathBuf;

use weirdnet_backends::find_in_path;
use weirdnet_model::hw::{Budget, CatalogEntry, Hardware, Plan};

use crate::config::{LocalModel, NodeConfig, Roles};

#[derive(Debug, Clone, Default)]
pub struct SetupOptions {
    pub name: Option<String>,
    /// An external `llama-server` to use instead of the one shipped with Ymir.
    pub llama_server: Option<PathBuf>,
    /// GGUF files to serve instead of catalog models (the first is the main model).
    pub models: Vec<PathBuf>,
    /// Add the catalog proposal even when `models` is given.
    pub catalog: bool,
    pub memory_budget: Option<String>,
    pub reserve: Option<String>,
    pub models_dir: Option<PathBuf>,
    pub bridge_port: Option<u16>,
    /// The engine for chat models: `llamacpp` or `mlx` (default: `mlx` on Apple
    /// Silicon, else `llamacpp`). Embedding models stay on llama.cpp.
    pub backend: Option<String>,
    /// Ymir's engines dir, where the managed MLX-LM venv lives.
    pub engines_dir: Option<PathBuf>,
}

/// The chat backend setup picks when none is given.
pub fn default_backend(hw: &Hardware) -> &'static str {
    if crate::mlx::supported(&hw.os, &hw.arch) { "mlx" } else { "llamacpp" }
}

/// What setup found and proposes.
#[derive(Debug, Clone)]
pub struct Proposal {
    pub hardware: Hardware,
    pub plan: Plan,
    pub config: NodeConfig,
    /// The `llama-server` the node will run, if one was found.
    pub llama_server: Option<PathBuf>,
    /// Where it comes from.
    pub engine: EngineSource,
    pub vllm: Option<PathBuf>,
    /// Catalog models in the config that aren't downloaded yet.
    pub to_download: Vec<CatalogEntry>,
    /// The engine chat models use: `llamacpp` or `mlx`.
    pub backend: String,
    /// `mlx_lm.server`, when MLX-LM is installed (backend `mlx`).
    pub mlx_server: Option<PathBuf>,
    /// Vision projectors for the proposed llama.cpp models that aren't downloaded yet.
    pub projectors: Vec<&'static crate::vision::Projector>,
    /// MLX weights the config needs that aren't downloaded yet.
    pub mlx_download: Vec<&'static crate::mlx::Weights>,
}

/// Where setup found `llama-server`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineSource {
    /// Shipped with this Ymir install (the default; `llamacpp = "managed"`).
    Bundled(crate::engine::Bundled),
    /// Given with `--llama-server` (`llamacpp = "system:..."`).
    Given,
    /// Found on `PATH` because this install has no engine of its own (`llamacpp = "system:..."`).
    Path,
    /// None found: loads fail until one is installed.
    Missing,
}

pub fn budget(hw: &Hardware, opts: &SetupOptions) -> Budget {
    weirdnet_model::hw::budget(
        hw,
        opts.memory_budget.as_deref().and_then(weirdnet_model::hw::parse_size),
        opts.reserve.as_deref().and_then(weirdnet_model::hw::parse_size),
    )
}

pub fn propose(hw: Hardware, opts: &SetupOptions) -> Proposal {
    propose_with(hw, opts, crate::engine::bundled())
}

/// [`propose`] with the bundled engine given (`None`: this install ships none).
pub fn propose_with(hw: Hardware, opts: &SetupOptions, bundled: Option<crate::engine::Bundled>) -> Proposal {
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
    // The engine shipped with Ymir is the default, resolved by ymird at each start (`managed`), so
    // a package upgrade upgrades the engine too. An explicit `--llama-server` wins; an install
    // without an engine of its own falls back to one on PATH, pinned by path because a service's
    // PATH can differ from the shell's.
    let pin = |cfg: &mut NodeConfig, p: &PathBuf| {
        cfg.model.llamacpp = format!("system:{}", std::fs::canonicalize(p).unwrap_or_else(|_| p.clone()).display());
    };
    let (llama_server, engine) = if let Some(p) = &opts.llama_server {
        pin(&mut cfg, p);
        (Some(p.clone()), EngineSource::Given)
    } else if let Some(b) = bundled {
        cfg.model.llamacpp = "managed".into();
        (Some(b.llama_server()), EngineSource::Bundled(b))
    } else if let Some(p) = find_in_path("llama-server") {
        pin(&mut cfg, &p);
        (Some(p), EngineSource::Path)
    } else {
        (None, EngineSource::Missing)
    };
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
    let backend = opts.backend.clone().unwrap_or_else(|| default_backend(&hw).into());
    let engines_dir = opts.engines_dir.clone().unwrap_or_else(|| crate::Paths::from_env().engines());
    let mlx_server = (backend == "mlx").then(|| crate::mlx::resolve(&cfg.model.mlx, &engines_dir)).flatten();
    let mut to_download = Vec::new();
    let mut mlx_download = Vec::new();
    let mut projectors = Vec::new();
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
            // MLX serves chat models that have MLX weights. Embeddings, and models that take images
            // (MLX-LM's server is text-only; llama.cpp reads images with the projector), use llama.cpp.
            let vision = crate::vision::for_model(&e.spec.id);
            let can = weirdnet_backends::EngineKind::Mlx.capabilities();
            let fits = if role == "embedding" { can.embeddings } else { vision.is_none() || can.vision };
            let mlx = (backend == "mlx" && fits).then(|| crate::mlx::weights(&e.spec.id)).flatten();
            let engine = match (mlx, backend.as_str()) {
                (Some(_), _) => "mlx",
                _ => "llamacpp",
            };
            cfg.model.local.push(LocalModel {
                id: e.spec.id.clone(),
                backend: Some(engine.into()),
                quant: Some(e.quant.clone()),
                swap_group: group.map(str::to_string),
                role: Some(role.into()),
                // llama.cpp models get no fixed context: ymird gives each its full trained length,
                // cut only where the KV cache would not fit (`kvfit`). MLX-LM takes it as its
                // answer-length limit, so it keeps the catalog's.
                context: (engine == "mlx").then_some(e.context),
                slots: Some(if role == "embedding" { 1 } else { plan.slots }),
                ..Default::default()
            });
            match mlx {
                Some(w) if !w.installed(&cfg.models_dir()) => mlx_download.push(w),
                Some(_) => {}
                None if !store.installed(&e.spec) => to_download.push(e.clone()),
                None => {}
            }
            if mlx.is_none()
                && let Some(v) = vision
                && !store.installed(&v.spec())
            {
                projectors.push(v);
            }
        }
    }
    Proposal { hardware: hw, plan, config: cfg, llama_server, engine, vllm, to_download, backend, mlx_server, mlx_download, projectors }
}

/// This machine's host name, else `ymir`.
#[cfg(unix)]
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

/// This machine's host name (`COMPUTERNAME`), else `ymir`.
#[cfg(not(unix))]
pub fn hostname() -> String {
    std::env::var("COMPUTERNAME").ok().map(|n| n.trim().to_ascii_lowercase()).filter(|n| !n.is_empty()).unwrap_or_else(|| "ymir".into())
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
        assert_eq!(p.engine, EngineSource::Given);
        assert_eq!(p.to_download.len(), 3);
        // The stand-in takes images: its projector comes along, and ymird passes it once on disk.
        assert_eq!(p.projectors.iter().map(|v| v.model).collect::<Vec<_>>(), ["gemma-4-e2b"]);
        std::fs::write(dir.path().join(p.projectors[0].file), b"x").unwrap();
        let l = c.launches().into_iter().find(|l| l.id == "gemma-4-e2b").unwrap();
        assert!(l.extra_args.windows(2).any(|w| w[0] == "--mmproj" && w[1].ends_with("mmproj-gemma-4-E2B-it-Q8_0.gguf")), "{:?}", l.extra_args);
        assert!(!c.launches().iter().any(|l| l.id == "qwen3-8b" && l.extra_args.contains(&"--mmproj".to_string())));
        // 12 GB of VRAM holds the 8B or the stand-in, not both: they share the swap group.
        assert_eq!(c.model.local.iter().find(|m| m.id == "gemma-4-e2b").unwrap().swap_group.as_deref(), Some("main"));
        c.validate().unwrap();
    }

    fn mac() -> Hardware {
        Hardware { os: "macos".into(), arch: "aarch64".into(), cpu_threads: 8, ram_bytes: 24 * GIB, gpus: Vec::new() }
    }

    /// A hosted macOS runner (7 GiB, a 6 GiB budget): the catalog picks the Gemma models, which take
    /// images, so they stay on llama.cpp (MLX-LM's server is text-only) while MLX is still the backend.
    #[test]
    fn a_small_mac_keeps_image_models_on_llamacpp_and_names_mlx_the_backend() {
        let dir = tempfile::tempdir().unwrap();
        let opts = SetupOptions {
            models_dir: Some(dir.path().into()),
            engines_dir: Some(dir.path().join("engines")),
            memory_budget: Some("6GiB".into()),
            reserve: Some("1GiB".into()),
            ..Default::default()
        };
        let hw = Hardware { ram_bytes: 7 * GIB, cpu_threads: 3, ..mac() };
        let p = propose(hw, &opts);
        assert_eq!(p.backend, "mlx");
        assert!(p.mlx_download.is_empty());
        let main = p.config.model.local.iter().find(|m| m.role.as_deref() == Some("main")).unwrap();
        assert_eq!(main.id, "gemma-4-e4b");
        assert!(p.config.model.local.iter().all(|m| m.backend.as_deref() == Some("llamacpp")));
    }

    #[test]
    fn apple_silicon_serves_chat_models_with_mlx() {
        let dir = tempfile::tempdir().unwrap();
        let opts = SetupOptions { models_dir: Some(dir.path().into()), engines_dir: Some(dir.path().join("engines")), ..Default::default() };
        let p = propose(mac(), &opts);
        assert_eq!(p.backend, "mlx");
        let c = &p.config;
        let main = c.main_model().unwrap();
        assert_eq!(main.backend.as_deref(), Some("mlx"));
        assert!(p.mlx_download.iter().any(|w| w.id == main.id));
        // The Gemma stand-in takes images, so it stays on llama.cpp with its projector.
        let standin = c.model.local.iter().find(|m| m.role.as_deref() == Some("standin")).unwrap();
        assert_eq!(standin.backend.as_deref(), Some("llamacpp"));
        assert!(p.projectors.iter().any(|v| v.model == standin.id));
        // Embeddings stay on llama.cpp, and their GGUF is still downloaded.
        let emb = c.model.local.iter().find(|m| m.is_embedding()).unwrap();
        assert_eq!(emb.backend.as_deref(), Some("llamacpp"));
        assert!(p.to_download.iter().all(|e| e.spec.id == emb.id || e.spec.id == standin.id));
        c.validate().unwrap();
        // llama.cpp entries leave the context to ymird (the full trained length where it fits);
        // MLX keeps the catalog's answer-length limit.
        assert!(standin.context.is_none() && emb.context.is_none() && main.context.is_some());
        let launch = c.launches().into_iter().find(|l| l.id == main.id).unwrap();
        assert_eq!(launch.engine, weirdnet_backends::EngineKind::Mlx);
        let src = &launch.source;
        assert!(!src.starts_with("mlx:") && src.contains(&format!("{0}mlx{0}", std::path::MAIN_SEPARATOR)), "{src}");
        // llama.cpp on request, and everywhere but Apple Silicon.
        let p = propose(mac(), &SetupOptions { backend: Some("llamacpp".into()), ..opts.clone() });
        assert!(p.config.model.local.iter().all(|m| m.backend.as_deref() == Some("llamacpp")) && p.mlx_download.is_empty());
        assert_eq!(propose(nvidia_pc(), &opts).backend, "llamacpp");
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

    #[test]
    fn the_bundled_engine_is_the_default() {
        let dir = tempfile::tempdir().unwrap();
        let eng = dir.path().join("lib/ymir/llama.cpp");
        std::fs::create_dir_all(&eng).unwrap();
        let server = crate::engine::program(&eng, "llama-server");
        std::fs::write(&server, b"x").unwrap();
        let b = crate::engine::Bundled::at(&eng).unwrap();
        let opts = SetupOptions { models_dir: Some(dir.path().into()), ..Default::default() };
        let p = propose_with(nvidia_pc(), &opts, Some(b.clone()));
        assert_eq!(p.config.model.llamacpp, "managed");
        assert_eq!(p.llama_server, Some(server));
        assert_eq!(p.engine, EngineSource::Bundled(b.clone()));
        // --llama-server still picks an external engine.
        let opts = SetupOptions { llama_server: Some("/opt/llama-server".into()), ..opts };
        let p = propose_with(nvidia_pc(), &opts, Some(b));
        assert_eq!(p.config.model.llamacpp, "system:/opt/llama-server");
        assert_eq!(p.engine, EngineSource::Given);
    }
}
