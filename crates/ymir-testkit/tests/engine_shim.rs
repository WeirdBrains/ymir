//! MLX-LM models run through `ymir-engine` next to llama.cpp ones: one node serves a GGUF on
//! llama.cpp and an MLX model, and `ymird` answers for each. The mock engine stands in for both
//! servers (it takes `mlx_lm.server`'s flags too), so this checks the plumbing, the proxy's key
//! check and model rewrite, not MLX. The llama.cpp model's projector reaches `llama-server`
//! through `ymir-engine` unchanged.
//!
//! One test in its own binary: it sets `YMIR_ENGINE_SHIM` before any thread starts.

use anyhow::Result;
use ymir_core::client::BridgeClient;
use ymir_core::config::LocalModel;
use ymir_core::config::Roles;
use ymir_core::{Daemon, NodeConfig, Paths, shim};

const MOCK: &str = env!("CARGO_BIN_EXE_ymir-mock-engine");
const SHIM: &str = env!("CARGO_BIN_EXE_ymir-test-shim");

#[test]
fn llamacpp_and_mlx_models_on_one_node() -> Result<()> {
    // SAFETY: no other thread exists yet in this test binary (one test, runtime not started).
    unsafe { std::env::set_var(shim::SHIM_ENV, SHIM) };
    let dir = tempfile::tempdir()?;
    let models = dir.path().join("models");
    let mlx_dir = models.join("mlx").join("Tiny-4bit");
    std::fs::create_dir_all(&mlx_dir)?;
    std::fs::write(models.join("tiny.gguf"), b"GGUF mock")?;
    std::fs::write(models.join("tiny-mmproj.gguf"), b"GGUF mock")?;

    let mut cfg = NodeConfig::default();
    cfg.node.name = "shim".into();
    cfg.roles = Roles::ALL;
    cfg.app.bridge.port = 0;
    cfg.model.models_dir = Some(models.clone());
    cfg.model.llamacpp = format!("system:{MOCK}");
    cfg.model.mlx = format!("system:{MOCK}");
    let local = |id: &str, backend: &str, file: &str| LocalModel {
        id: id.into(),
        backend: Some(backend.into()),
        file: Some(file.into()),
        role: Some("chat".into()),
        ..Default::default()
    };
    cfg.model.local = vec![local("tiny", "llamacpp", "tiny.gguf"), local("tiny-mlx", "mlx", "mlx/Tiny-4bit")];
    cfg.model.local[0].role = Some("main".into());
    cfg.model.local[0].mmproj = Some("tiny-mmproj.gguf".into());
    cfg.validate()?;
    let home = dir.path().join("home");
    let paths = Paths::new(&home);
    std::fs::create_dir_all(&home)?;

    let engines = cfg.engines(&paths);
    assert_eq!(engines.len(), 1);
    assert_eq!(engines[0].program, std::path::PathBuf::from(SHIM));
    let launches = cfg.launches();
    assert_eq!(launches[1].source, format!("mlx:{}", mlx_dir.display()));
    assert!(launches[0].extra_args.windows(2).any(|w| w == ["--mmproj".to_string(), models.join("tiny-mmproj.gguf").display().to_string()]));
    assert!(!launches[1].extra_args.contains(&"--mmproj".to_string()));

    tokio::runtime::Builder::new_multi_thread().worker_threads(4).enable_all().build()?.block_on(async {
        let daemon = Daemon::start(cfg, paths.clone()).await?;
        let client = BridgeClient::new(daemon.bridge_addr.unwrap().port(), ymir_core::token::load(&paths)?.unwrap());
        let mut replies = Vec::new();
        for model in ["tiny", "tiny-mlx"] {
            let r = client.chat_stream(Some(model), vec![serde_json::json!({"role": "user", "content": "hi"})], Default::default(), |_| {}).await;
            replies.push((model, r.map(|r| r.text)));
        }
        daemon.shutdown().await;
        for (model, r) in replies {
            let text = r.map_err(|e| anyhow::anyhow!("{model}: {e}"))?;
            assert!(text.trim().ends_with("You said: hi"), "{model}: {text}");
        }
        anyhow::Ok(())
    })
}
