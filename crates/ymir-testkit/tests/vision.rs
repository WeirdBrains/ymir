//! Images on a catalog vision model. `ymird` starts llama.cpp with the pinned projector
//! (`--mmproj`) once it is on disk, and that engine takes image parts in OpenAI chat requests, one
//! or several; with the projector off it refuses them, as llama-server does. The mock engine
//! stands in for llama-server (it counts images only when given an existing `--mmproj`).
//!
//! The pool path does not carry images yet: `weirdnet-openai` turns away non-text content parts and
//! `weirdnet-agent` messages are text, so the bridge answers 400. That is pinned here so the test
//! changes when Weirdnet carries content parts (proposed in docs/DESIGN.md).

use std::time::Duration;

use anyhow::Result;
use serde_json::{Value, json};
use ymir_core::config::{LocalModel, Roles};
use ymir_core::{Daemon, NodeConfig, Paths, vision};

const MOCK: &str = env!("CARGO_BIN_EXE_ymir-mock-engine");
const PNG: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

fn body(model: &str, images: usize, stream: bool) -> Value {
    let mut parts = vec![json!({"type": "text", "text": "describe"})];
    parts.extend((0..images).map(|_| json!({"type": "image_url", "image_url": {"url": PNG}})));
    json!({"model": model, "messages": [{"role": "user", "content": parts}], "stream": stream})
}

/// The answer text of a streamed reply (the mock streams), or the body as is.
fn text_of(body: &str) -> String {
    let deltas: Vec<String> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter_map(|d| serde_json::from_str::<Value>(d).ok())
        .filter_map(|v| v["choices"][0]["delta"]["content"].as_str().map(str::to_string))
        .collect();
    if deltas.is_empty() { body.to_string() } else { deltas.concat() }
}

fn config(models: &std::path::Path) -> NodeConfig {
    let entry = weirdnet_model::hw::find("gemma-4-e4b").unwrap();
    let mut cfg = NodeConfig::default();
    cfg.node.name = "vision".into();
    cfg.roles = Roles::ALL;
    cfg.app.bridge.port = 0;
    cfg.model.models_dir = Some(models.to_path_buf());
    cfg.model.llamacpp = format!("system:{MOCK}");
    let gemma = |id: &str, mmproj: Option<&str>| LocalModel {
        id: id.into(),
        role: Some(if id == "gemma-4-e4b" { "main" } else { "chat" }.into()),
        file: Some(entry.spec.file.clone()),
        mmproj: mmproj.map(str::to_string),
        ..Default::default()
    };
    // The same weights twice: with the catalog projector (the default), and with it off.
    cfg.model.local = vec![gemma("gemma-4-e4b", None), gemma("gemma-blind", Some("off"))];
    cfg
}

fn setup() -> Result<(tempfile::TempDir, std::path::PathBuf)> {
    let dir = tempfile::tempdir()?;
    let models = dir.path().join("models");
    std::fs::create_dir_all(&models)?;
    std::fs::write(models.join(&weirdnet_model::hw::find("gemma-4-e4b").unwrap().spec.file), b"GGUF mock")?;
    std::fs::write(vision::for_model("gemma-4-e4b").unwrap().path(&models), b"GGUF mock projector")?;
    Ok((dir, models))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_engine_ymird_starts_takes_images() -> Result<()> {
    let (dir, models) = setup()?;
    let cfg = config(&models);
    cfg.validate()?;
    let proj = vision::for_model("gemma-4-e4b").unwrap().path(&models);
    let launches = cfg.launches();
    assert!(launches[0].extra_args.windows(2).any(|w| w[0] == "--mmproj" && w[1] == proj.display().to_string()), "{:?}", launches[0].extra_args);
    assert!(!launches[1].extra_args.contains(&"--mmproj".to_string()));
    let engine = &cfg.engines(&Paths::new(dir.path().join("home")))[0];

    let http = reqwest::Client::new();
    let mut results = Vec::new();
    for (i, launch) in launches.iter().enumerate() {
        let port = 18600 + u16::try_from(std::process::id() % 200).unwrap() + u16::try_from(i).unwrap();
        let mut child = engine.command(launch, port, "k").kill_on_drop(true).spawn()?;
        let base = format!("http://127.0.0.1:{port}");
        for _ in 0..100 {
            if http.get(format!("{base}/health")).send().await.is_ok_and(|r| r.status().is_success()) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        for images in [1, 3] {
            let r = http.post(format!("{base}/v1/chat/completions")).bearer_auth("k").json(&body(&launch.id, images, false)).send().await?;
            let status = r.status();
            results.push((launch.id.clone(), images, status, text_of(&r.text().await?)));
        }
        child.kill().await?;
    }
    for (id, images, status, text) in &results {
        if id == "gemma-4-e4b" {
            assert!(status.is_success() && text.contains(&format!("describe [{images} image(s)]")), "{id}: {status} {text}");
        } else {
            assert!(!status.is_success() && text.contains("image input is not supported"), "{id}: {status} {text}");
        }
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_bridge_does_not_carry_images_yet() -> Result<()> {
    let (dir, models) = setup()?;
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home)?;
    let paths = Paths::new(&home);
    let daemon = Daemon::start(config(&models), paths.clone()).await?;
    let port = daemon.bridge_addr.unwrap().port();
    let token = ymir_core::token::load(&paths)?.unwrap();
    let r = reqwest::Client::new()
        .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
        .bearer_auth(&token)
        .json(&body("gemma-4-e4b", 1, false))
        .send()
        .await?;
    let (status, text) = (r.status(), r.text().await?);
    daemon.shutdown().await;
    assert_eq!(status, reqwest::StatusCode::BAD_REQUEST, "{text}");
    assert!(text.contains("content parts of type `image_url` are not supported"), "{text}");
    Ok(())
}
