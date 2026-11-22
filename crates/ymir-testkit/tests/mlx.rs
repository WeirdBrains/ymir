//! MLX-LM models run as `weirdnet-backends`' `EngineKind::Mlx` next to llama.cpp ones: one node
//! serves a GGUF on llama.cpp and an MLX model, and `ymird` answers for each. The mock engine
//! stands in for both servers (it takes `mlx_lm.server`'s flags and `default_model` too), so this
//! checks the plumbing, the supervisor's MLX proxy, and the capabilities each model reports, not
//! MLX itself.

use anyhow::Result;
use weirdnet_backends::EngineKind;
use ymir_core::client::BridgeClient;
use ymir_core::config::{LocalModel, Roles};
use ymir_core::control::{self, ControlRequest};
use ymir_core::{Daemon, NodeConfig, Paths};

const MOCK: &str = env!("CARGO_BIN_EXE_ymir-mock-engine");

#[test]
fn llamacpp_and_mlx_models_on_one_node() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let models = dir.path().join("models");
    let mlx_dir = models.join("mlx").join("Tiny-4bit");
    std::fs::create_dir_all(&mlx_dir)?;
    std::fs::write(models.join("tiny.gguf"), b"GGUF mock")?;
    std::fs::write(models.join("tiny-mmproj.gguf"), b"GGUF mock")?;

    let mut cfg = NodeConfig::default();
    cfg.node.name = "mlx".into();
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
    let kinds: Vec<_> = engines.iter().map(|e| (e.kind, e.program.clone())).collect();
    assert_eq!(kinds, [(EngineKind::Llamacpp, MOCK.into()), (EngineKind::Mlx, MOCK.into())]);
    let launches = cfg.launches();
    assert_eq!(launches[0].engine, EngineKind::Llamacpp);
    assert_eq!(launches[1].engine, EngineKind::Mlx);
    assert_eq!(launches[1].source, mlx_dir.display().to_string());
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
        let status = control::request(&paths.socket(), &ControlRequest::Status).await?.status.unwrap();
        daemon.shutdown().await;
        for (model, r) in replies {
            let text = r.map_err(|e| anyhow::anyhow!("{model}: {e}"))?;
            assert!(text.trim().ends_with("You said: hi"), "{model}: {text}");
        }
        let caps = |id: &str| status.models.iter().find(|m| m.id == id).and_then(|m| m.capabilities).unwrap();
        let (gguf, mlx) = (caps("tiny"), caps("tiny-mlx"));
        assert!(gguf.chat && gguf.vision && gguf.audio, "{gguf:?}");
        assert!(mlx.chat && !mlx.vision && !mlx.audio && !mlx.embeddings, "{mlx:?}");
        anyhow::Ok(())
    })
}
