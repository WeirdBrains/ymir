//! Audio design milestones 1 and 3: `ymirctl transcribe` through ymird's control socket.
//!
//! A control handler with the app role's pool (`local_pool` over a coordinator whose model node
//! runs the mock engine as vLLM with a `transcribe` model) serves a real socket. `ymirctl
//! transcribe` sends a clip (normalized to 16 kHz mono WAV here), the pool routes it to the
//! transcriber, and the answer comes back as text, JSON, SRT or VTT. With `--chunk auto` a long
//! clip is cut at silences to the transcriber's reported `audio_max_seconds` and the segments are
//! re-aligned onto the clip; a transcriber that reports no length is refused with the setting to
//! fill in. Neither audio nor transcript reaches the logs.

use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};

use clap::Parser;
use weirdnet_agent::MediaLimits;
use weirdnet_backends::{EngineConfig, Supervisor, SupervisorOptions};
use weirdnet_media::pcm::{Pcm, encode_wav};
use weirdnet_semcache::NoCache;
use ymir_core::config::LocalModel;
use ymir_core::control::{self, ControlHandler, ControlRequest, TranscribeJob};
use ymir_core::{Coordinator, LocalModelNode, ModelNode, NodeConfig, Paths};
use ymirctl::transcribe::{Chunking, Options};

const MOCK: &str = env!("CARGO_BIN_EXE_ymir-mock-engine");

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

/// A node with a vLLM transcriber (`limits`: the operator's figures) and a chat model, serving
/// the control socket under `home`. Returns the supervisor (to stop the engines).
async fn node(home: &Path, limits: MediaLimits) -> Supervisor {
    let models = home.join("models");
    let mut cfg = NodeConfig::default();
    cfg.node.name = "stt".into();
    cfg.model.models_dir = Some(models.clone());
    cfg.model.local = vec![
        LocalModel {
            id: "whisper".into(),
            backend: Some("vllm".into()),
            file: Some("openai/whisper-large-v3".into()),
            role: Some("transcribe".into()),
            limits,
            ..Default::default()
        },
        LocalModel { id: "chatty".into(), backend: Some("vllm".into()), file: Some("org/chatty".into()), ..Default::default() },
    ];
    cfg.validate().unwrap();
    let launches = cfg.launches();
    let sup = Supervisor::new(
        vec![EngineConfig::vllm(MOCK)],
        launches.clone(),
        SupervisorOptions { log_dir: Some(home.join("logs")), ..Default::default() },
    );
    let node: Arc<dyn ModelNode> = Arc::new(LocalModelNode::new("stt", sup.clone(), &launches));
    let coord = Arc::new(Coordinator::new(vec![node], Some("chatty".into()), Arc::new(NoCache)));
    let pool = Arc::new(ymir_core::host::local_pool(coord.clone(), "stt", ymir_core::host::admission(2, 4), None));
    let handler = Arc::new(ControlHandler {
        node: "stt".into(),
        roles: vec!["model".into(), "inference".into(), "app".into()],
        bridge_port: None,
        main_model: Some("chatty".into()),
        coordinator: Some(coord),
        model_node: None,
        alerter: None,
        network: None,
        context: None,
        pool: Some(pool),
        models: None,
        train: None,
        jobs: None,
        blobs: None,
        exit: Arc::new(tokio::sync::Notify::new()),
    });
    let paths = Paths::new(home.to_path_buf());
    std::fs::create_dir_all(paths.state()).unwrap();
    let listener = control::bind(&paths.socket()).await.unwrap();
    let (_tx, rx) = tokio::sync::watch::channel(false);
    tokio::spawn(async move {
        let _keep = _tx;
        control::serve(listener, handler, rx).await;
    });
    sup
}

/// Speech-like bursts: 1.7 s of tone, 0.3 s of silence, over `ms`, at `rate` with `channels`.
fn speech(rate: u32, channels: u16, ms: u32) -> Vec<u8> {
    let n = (rate as u64 * u64::from(ms) / 1000) as usize;
    let period = rate as usize * 2;
    let on = rate as usize * 17 / 10;
    let mut s = Vec::with_capacity(n * usize::from(channels));
    for i in 0..n {
        let v = if i % period < on { ((i as f32 * 0.07).sin() * 16_000.0) as i16 } else { 0 };
        for _ in 0..channels {
            s.push(v);
        }
    }
    encode_wav(&Pcm::new(rate, channels, s))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn transcribe_through_the_control_socket() {
    let logs = Logs::default();
    let sink = logs.clone();
    tracing_subscriber::fmt()
        .with_max_level(tracing_subscriber::filter::LevelFilter::TRACE)
        .with_writer(move || sink.clone())
        .with_ansi(false)
        .init();

    let home = tempfile::tempdir().unwrap();
    let sup = node(home.path(), MediaLimits::new().with_audio_max_seconds(30)).await;
    let socket = Paths::new(home.path().to_path_buf()).socket();

    // The transcriber the pool would pick, with the operator's limit.
    let t = control::transcribe_target(&socket, None).await.unwrap();
    assert_eq!((t.model.as_str(), t.audio_max_seconds), ("whisper", Some(30)));
    let named = control::transcribe_target(&socket, Some("chatty")).await.unwrap_err().to_string();
    assert!(named.contains("does not transcribe"), "{named}");

    // One request: a 44.1 kHz stereo clip goes out as 16 kHz mono WAV (64 000 bytes + header).
    let clip = speech(44_100, 2, 2_000);
    let opts = Options { timestamps: true, normalize: true, ..Default::default() };
    let o = ymirctl::transcribe::transcribe(&socket, &opts, clip.clone(), None, &mut |_| {}).await.unwrap();
    assert_eq!(o.model, "whisper");
    assert_eq!(o.chunks, 1);
    assert!(o.transcript.text.starts_with("heard "), "{}", o.transcript.text);
    assert_eq!(o.transcript.segments.len(), 1);
    assert_eq!(o.transcript.segments[0].end_ms, 1_500, "the engine's timing");
    assert_eq!(o.nodes, vec!["stt".to_string()]);

    // Long audio, chunked at silences to 30 s: 75 s is three requests, segments on the clip's timeline.
    let long = speech(16_000, 1, 75_000);
    let mut lines = Vec::new();
    let opts = Options { timestamps: true, chunk: Chunking::Auto, normalize: true, ..Default::default() };
    let o = ymirctl::transcribe::transcribe(&socket, &opts, long, None, &mut |l| lines.push(l)).await.unwrap();
    assert_eq!(o.chunks, 3);
    assert_eq!(lines.len(), 3);
    assert!(lines[0].starts_with("chunk 1 of 3"));
    assert_eq!(o.transcript.segments.len(), 3);
    assert_eq!(o.transcript.segments[0].start_ms, 0);
    for w in o.transcript.segments.windows(2) {
        assert!(w[1].start_ms >= 24_000 + w[0].start_ms && w[1].start_ms - w[0].start_ms <= 30_000, "{:?}", o.transcript.segments);
    }
    assert_eq!(o.duration_ms, 75_000);
    assert_eq!(o.transcript.text.matches("heard").count(), 3);

    // The CLI: a file to SRT on stdout.
    let file = home.path().join("talk.wav");
    std::fs::write(&file, &clip).unwrap();
    let cli = ymirctl::Cli::parse_from(["ymirctl", "--home", home.path().to_str().unwrap(), "transcribe", file.to_str().unwrap(), "--format", "srt"]);
    let mut out = Vec::new();
    ymirctl::run(cli, &mut out, &mut std::io::empty()).await.unwrap();
    let srt = String::from_utf8(out).unwrap();
    assert!(srt.starts_with("1\n00:00:00,000 --> 00:00:01,500\nheard "), "{srt}");

    // Stdin, VTT, to a file.
    let vtt = home.path().join("talk.vtt");
    let cli = ymirctl::Cli::parse_from([
        "ymirctl",
        "--home",
        home.path().to_str().unwrap(),
        "transcribe",
        "-",
        "--format",
        "vtt",
        "--out",
        vtt.to_str().unwrap(),
    ]);
    ymirctl::run(cli, &mut Vec::new(), &mut std::io::Cursor::new(clip.clone())).await.unwrap();
    assert!(std::fs::read_to_string(&vtt).unwrap().starts_with("WEBVTT\n\n00:00:00.000 --> "));

    // Not audio; a named model that doesn't transcribe; a request with no audio on the socket.
    assert!(
        ymirctl::transcribe::transcribe(&socket, &Options { normalize: true, ..Default::default() }, b"just text".to_vec(), None, &mut |_| {})
            .await
            .is_err()
    );
    let opts = Options { model: Some("chatty".into()), normalize: true, ..Default::default() };
    let e = ymirctl::transcribe::transcribe(&socket, &opts, clip.clone(), None, &mut |_| {}).await.unwrap_err().to_string();
    assert!(e.contains("does not transcribe"), "{e}");
    let e = control::transcribe(&socket, TranscribeJob { format: "wav".into(), ..Default::default() }, &[]).await.unwrap_err().to_string();
    assert!(e.contains("1 to"), "{e}");
    // In-process requests can't carry audio.
    let r =
        control::request(&socket, &ControlRequest::Transcribe(TranscribeJob { bytes: 0, format: "wav".into(), ..Default::default() })).await.unwrap();
    assert!(!r.ok);

    sup.unload("whisper").await.ok();
    let logged = String::from_utf8_lossy(&logs.0.lock().unwrap()).into_owned();
    assert!(!logged.contains("heard "), "the transcript is never logged");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chunking_needs_a_reported_length() {
    let home = tempfile::tempdir().unwrap();
    let sup = node(home.path(), MediaLimits::UNKNOWN).await;
    let socket = Paths::new(home.path().to_path_buf()).socket();
    assert_eq!(control::transcribe_target(&socket, None).await.unwrap().audio_max_seconds, None, "absent, never defaulted");
    let opts = Options { chunk: Chunking::Auto, normalize: true, ..Default::default() };
    let e = ymirctl::transcribe::transcribe(&socket, &opts, speech(16_000, 1, 1_000), None, &mut |_| {}).await.unwrap_err().to_string();
    assert!(e.contains("reports no audio_max_seconds") && e.contains("limits = { audio_max_seconds = N }"), "{e}");
    // Whole clips still go.
    let o = ymirctl::transcribe::transcribe(&socket, &Options { normalize: true, ..Default::default() }, speech(16_000, 1, 1_000), None, &mut |_| {})
        .await
        .unwrap();
    assert!(o.transcript.text.starts_with("heard "));
    assert!(o.transcript.segments.iter().all(|s| s.coarse), "no timing asked for: one coarse segment");
    sup.unload("whisper").await.ok();
}
