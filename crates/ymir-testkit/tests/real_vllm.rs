//! Real vLLM, opt in (needs a GPU). Ignored by default.
//!
//! Managed launch: `YMIR_TEST_VLLM=/path/to/vllm YMIR_TEST_VLLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct`
//! (extra engine args in `YMIR_TEST_VLLM_ARGS`, such as `--gpu-memory-utilization 0.5`).
//! Wrapping a stock `vllm serve`: `YMIR_TEST_VLLM_WRAP=http://127.0.0.1:8000` and
//! `YMIR_TEST_VLLM_WRAP_MODEL=<served model name>`.

use std::time::{Duration, Instant};

use futures::StreamExt;
use serde_json::json;
use ymir_backends::{EngineConfig, EngineKind, ModelLaunch, Supervisor, SupervisorOptions, Wrap};
use ymir_infer::{ChatRequest, Event};

async fn ask(sup: &Supervisor, id: &str, max_tokens: u64) -> (String, u64, Duration) {
    let (req, _) = ChatRequest::from_openai(json!({
        "messages": [{"role": "user", "content": "Write a long story about a lighthouse keeper."}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }))
    .unwrap();
    let t = Instant::now();
    let mut s = sup.chat(id, &req).await.unwrap();
    let (mut text, mut tokens) = (String::new(), 0);
    while let Some(ev) = s.next().await {
        match ev.unwrap() {
            Event::Delta { text: t, .. } => text.push_str(&t),
            Event::Done { usage, .. } => tokens = usage.map(|u| u.completion_tokens).unwrap_or(0),
            _ => {}
        }
    }
    (text, tokens, t.elapsed())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs YMIR_TEST_VLLM and YMIR_TEST_VLLM_MODEL (and a GPU)"]
async fn real_vllm_managed() {
    let program = std::env::var("YMIR_TEST_VLLM").expect("YMIR_TEST_VLLM");
    let model = std::env::var("YMIR_TEST_VLLM_MODEL").expect("YMIR_TEST_VLLM_MODEL");
    let mut l = ModelLaunch::new("vl", EngineKind::Vllm, model);
    l.context = Some(4096);
    l.extra_args = std::env::var("YMIR_TEST_VLLM_ARGS").unwrap_or_default().split_whitespace().map(str::to_string).collect();
    let logs = tempfile::tempdir().unwrap();
    let sup =
        Supervisor::new(vec![EngineConfig::vllm(program)], vec![l], SupervisorOptions { log_dir: Some(logs.path().into()), ..Default::default() });
    let t = Instant::now();
    if let Err(e) = sup.load("vl").await {
        let log = std::fs::read_to_string(logs.path().join("vl.log")).unwrap_or_default();
        panic!("{e}\n{}", &log[log.len().saturating_sub(4000)..]);
    }
    eprintln!("vLLM ready in {:.1}s", t.elapsed().as_secs_f32());
    let _ = ask(&sup, "vl", 16).await;
    for n in [256, 512] {
        let (text, tokens, d) = ask(&sup, "vl", n).await;
        assert!(!text.is_empty());
        eprintln!(
            "managed vLLM: {tokens} tokens in {:.2}s = {:.1} tok/s (streamed through the supervisor)",
            d.as_secs_f32(),
            tokens as f32 / d.as_secs_f32()
        );
    }
    let pid = sup.status()[0].pid.unwrap();
    sup.unload("vl").await.unwrap();
    // SAFETY: signal 0 only checks that the process exists.
    assert!(unsafe { libc::kill(pid as i32, 0) } != 0, "vLLM's process group is gone");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs YMIR_TEST_VLLM_WRAP and YMIR_TEST_VLLM_WRAP_MODEL (a running vllm serve)"]
async fn real_vllm_wrapped() {
    let url = std::env::var("YMIR_TEST_VLLM_WRAP").expect("YMIR_TEST_VLLM_WRAP");
    let model = std::env::var("YMIR_TEST_VLLM_WRAP_MODEL").expect("YMIR_TEST_VLLM_WRAP_MODEL");
    let mut l = ModelLaunch::new(&model, EngineKind::Vllm, "wrapped");
    l.wrap = Some(Wrap { url, api_key: None });
    let sup = Supervisor::new(Vec::new(), vec![l], SupervisorOptions::default());
    sup.load(&model).await.unwrap();
    let _ = ask(&sup, &model, 16).await;
    let (text, tokens, d) = ask(&sup, &model, 512).await;
    assert!(!text.is_empty());
    eprintln!("wrapped vLLM: {tokens} tokens in {:.2}s = {:.1} tok/s", d.as_secs_f32(), tokens as f32 / d.as_secs_f32());
    sup.unload(&model).await.unwrap();
}
