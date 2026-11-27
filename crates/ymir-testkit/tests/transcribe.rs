//! Video design, milestone 2: the transcription role and route, and limits measured at load.
//!
//! A `role = "transcribe"` model on vLLM (the mock engine stands in for `vllm serve`) is published
//! with catalog/1 purpose `transcribe`, its context per request is what the engine reports at load
//! (`max_model_len`), not a default, and the operator's `[model.local] limits` win field by field
//! while unknown fields stay absent. infer/2 `Transcribe` goes through the inference role
//! (`InferenceHost::open_transcribe`) to the model node and its engine; a pool with no
//! transcriber, or a named model that doesn't transcribe, gets a 400. Nothing is cached and
//! neither the audio nor the transcript is logged.

use std::io::Write;
use std::sync::{Arc, Mutex};

use weirdnet_agent::CancelToken;
use weirdnet_agent::MediaLimits;
use weirdnet_agent::remote::RemoteError;
use weirdnet_agent::remote::catalog::Purpose;
use weirdnet_agent::remote::infer::v2::TranscribeRequest;
use weirdnet_backends::{EngineConfig, Supervisor, SupervisorOptions};
use weirdnet_semcache::NoCache;
use ymir_core::config::LocalModel;
use ymir_core::{Coordinator, LocalModelNode, ModelNode, NodeConfig};

const MOCK: &str = env!("CARGO_BIN_EXE_ymir-mock-engine");

/// What every log line in this test binary says, to check no content reaches the logs.
#[derive(Clone, Default)]
struct Logs(Arc<Mutex<Vec<u8>>>);

impl Write for Logs {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn config(models: &std::path::Path) -> NodeConfig {
    let mut cfg = NodeConfig::default();
    cfg.node.name = "stt".into();
    cfg.model.models_dir = Some(models.to_path_buf());
    cfg.model.local = vec![
        LocalModel {
            id: "whisper".into(),
            backend: Some("vllm".into()),
            file: Some("openai/whisper-large-v3".into()),
            role: Some("transcribe".into()),
            // The operator's figure for this hardware; nothing else is known.
            limits: MediaLimits::new().with_audio_max_seconds(30),
            ..Default::default()
        },
        LocalModel { id: "chatty".into(), backend: Some("vllm".into()), file: Some("org/chatty".into()), ..Default::default() },
    ];
    cfg
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_transcriber_is_measured_routed_and_neither_cached_nor_logged() {
    let logs = Logs::default();
    let sink = logs.clone();
    tracing_subscriber::fmt()
        .with_max_level(tracing_subscriber::filter::LevelFilter::TRACE)
        .with_writer(move || sink.clone())
        .with_ansi(false)
        .init();

    let dir = tempfile::tempdir().unwrap();
    let cfg = config(dir.path());
    cfg.validate().unwrap();
    assert_eq!(cfg.main_model().map(|m| m.id.as_str()), Some("chatty"), "a transcriber is never the main chat model");
    let mut launches = cfg.launches();
    assert!(launches[0].transcribe && !launches[1].transcribe);
    // The engine is started with a context the config doesn't know (as from the engine's own
    // default): only measuring it at load can report it.
    launches[0].extra_args.extend(["--max-model-len".to_string(), "3072".to_string()]);
    let sup = Supervisor::new(
        vec![EngineConfig::vllm(MOCK)],
        launches.clone(),
        SupervisorOptions { log_dir: Some(dir.path().join("logs")), ..Default::default() },
    );
    let node: Arc<dyn ModelNode> = Arc::new(LocalModelNode::new("stt", sup.clone(), &launches));
    let coord = Arc::new(Coordinator::new(vec![node], Some("chatty".into()), Arc::new(NoCache)));

    // Before the first load: the purpose, and the operator's limit alone; no context yet.
    let entry = |id: &str| coord.agent_catalog().into_iter().find(|m| m.id == id).unwrap();
    let w = entry("whisper");
    assert_eq!(w.purpose, Purpose::Transcribe);
    assert_eq!(w.context, None, "not measured yet, and no default");
    assert_eq!(w.limits.audio_max_seconds, Some(30));
    assert_eq!((w.limits.max_images, w.limits.image_tokens, w.limits.image_edge), (None, None, None), "unknown stays absent");
    assert_eq!(entry("chatty").purpose, Purpose::Chat);
    assert!(entry("whisper").standin.is_none());

    let pool = ymir_core::host::local_pool(coord.clone(), "stt", ymir_core::host::admission(2, 4), None);
    let audio = b"RIFF-secret-words-in-the-audio".to_vec();
    let t = pool
        .transcribe(TranscribeRequest::new(audio.clone(), "wav").with_timestamps(true), &CancelToken::new())
        .await
        .expect("routed to the transcriber");
    assert_eq!(t.model, "whisper");
    assert!(t.transcript.text.starts_with("heard "), "{:?}", t.transcript.text);
    assert_eq!(t.transcript.segments.len(), 1, "timed segments when asked for");

    // Measured at load: vLLM's max_model_len is the context per request, in the node record and catalog/1.
    let st = sup.status().into_iter().find(|s| s.id == "whisper").unwrap();
    assert_eq!(st.context, None);
    assert_eq!(st.context_tokens, Some(3072));
    assert_eq!(ymir_core::model_node::status_context(&st), Some(3072));
    let w = entry("whisper");
    assert_eq!(w.context, Some(3072));
    assert_eq!(w.limits.audio_max_seconds, Some(30), "the operator's figure still wins");
    assert!(w.limits.image_tokens.is_none());

    // Not cached: the same audio again reaches the engine again (a new transcript, a new request).
    let again = pool.transcribe(TranscribeRequest::new(audio.clone(), "wav"), &CancelToken::new()).await.unwrap();
    assert!(again.transcript.segments.is_empty());
    assert!(!again.meta.cache_hit, "never replayed from a cache");

    // A named model that doesn't transcribe, and an unknown one.
    let named = pool.transcribe(TranscribeRequest::new(audio.clone(), "wav").with_model("chatty"), &CancelToken::new()).await;
    assert!(matches!(&named, Err(RemoteError::BadRequest { message }) if message.contains("does not transcribe")), "{named:?}");
    let unknown = pool.transcribe(TranscribeRequest::new(audio.clone(), "wav").with_model("nope"), &CancelToken::new()).await;
    assert!(unknown.is_err());

    // A chat request never goes to the transcriber by default: the main model answers.
    assert_eq!(coord.agent_catalog().iter().filter(|m| m.purpose == Purpose::Transcribe).count(), 1);

    sup.unload("whisper").await.ok();
    let logged = String::from_utf8_lossy(&logs.0.lock().unwrap()).into_owned();
    assert!(!logged.contains("secret-words"), "the audio is never logged");
    assert!(!logged.contains(&t.transcript.text), "the transcript is never logged");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_pool_without_a_transcriber_says_so() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = config(dir.path());
    cfg.model.local.remove(0);
    let launches = cfg.launches();
    let sup = Supervisor::new(vec![EngineConfig::vllm(MOCK)], launches.clone(), SupervisorOptions::default());
    let node: Arc<dyn ModelNode> = Arc::new(LocalModelNode::new("chat-only", sup, &launches));
    let coord = Arc::new(Coordinator::new(vec![node], Some("chatty".into()), Arc::new(NoCache)));
    let err = coord.resolve_transcriber(None).unwrap_err();
    assert!(matches!(&err, RemoteError::BadRequest { message } if message.contains("no model in the pool transcribes")), "{err:?}");
    let pool = ymir_core::host::local_pool(coord, "chat-only", ymir_core::host::admission(1, 1), None);
    let r = pool.transcribe(TranscribeRequest::new(b"RIFF".to_vec(), "wav"), &CancelToken::new()).await;
    assert!(matches!(r, Err(RemoteError::BadRequest { .. })), "{r:?}");
}
