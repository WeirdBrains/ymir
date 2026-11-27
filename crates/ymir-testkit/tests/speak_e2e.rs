//! Text to speech end to end (audio design milestones 7–12) and native video in the catalog
//! (video design Phase 2): a real ymird (model, inference and app roles) whose engines are the
//! mock engine in both speech flavors, a llama-server speaker with its projector (`/props`
//! voices, `/v1/audio/speech`) and a vLLM-Omni one (`vllm serve MODEL --omni`, `GET
//! /v1/audio/voices`), plus a vLLM chat model with `video = true`. Speech goes through
//! `ymirctl speak` (the control socket), MCP `ymir_speak` and the bridge's
//! `POST /v1/audio/speech`; each reaches the engine through `InferenceHost::open_speak`,
//! `Coordinator::speak` and the supervisor. Nothing touches the network or a model.

use std::path::Path;
use std::time::Duration;

use serde_json::{Value, json};
use weirdnet_media::pcm::decode_wav;
use ymir_core::{Daemon, NodeConfig, Paths};
use ymirctl::speak::{Chunking, Options, OutFormat, Text};

const MOCK: &str = env!("CARGO_BIN_EXE_ymir-mock-engine");
/// The vLLM-Omni flavor's sample rate here (the llama-server one keeps the mock's 24 kHz).
const OMNI_RATE: u32 = 22_050;

struct Node {
    daemon: Daemon,
    paths: Paths,
    base: String,
    token: String,
    http: reqwest::Client,
    _dir: tempfile::TempDir,
}

impl Node {
    async fn start() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join("home");
        let models = dir.path().join("models");
        std::fs::create_dir_all(&home).unwrap();
        std::fs::create_dir_all(&models).unwrap();
        std::fs::write(models.join("tts.gguf"), b"GGUF mock").unwrap();
        std::fs::write(models.join("mmproj-tts.gguf"), b"GGUF mock projector").unwrap();
        let toml = format!(
            r#"[node]
name = "speech-node"
[roles]
model = true
inference = true
app = true
[model]
llamacpp = "system:{MOCK}"
vllm = "system:{MOCK}"
vllm_omni = "system:{MOCK}"
models_dir = "{models}"
[[model.local]]
id = "tts"
role = "speak"
file = "tts.gguf"
mmproj = "mmproj-tts.gguf"
args = ["--mock-voices", "af_heart,am_adam"]
[[model.local]]
id = "omni-tts"
backend = "vllm-omni"
role = "speak"
file = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
args = ["--mock-voices", "vivian,ryan", "--mock-rate", "{OMNI_RATE}"]
[model.local.speak]
max_input_chars = 4000
guests = false
[[model.local]]
id = "vl"
backend = "vllm"
file = "Qwen/Qwen3-VL-8B-Instruct"
video = true
limits = {{ max_images = 8, video_max_seconds = 60, video_max_bytes = 50000000, video_frames_per_minute = 120 }}
[inference]
[inference.alerts]
probe_secs = 3600
[app.bridge]
port = 0
"#,
            models = models.display()
        );
        std::fs::write(home.join("node.toml"), toml).unwrap();
        let paths = Paths::new(&home);
        let cfg = NodeConfig::load(&paths.config()).unwrap();
        cfg.validate().unwrap();
        let daemon = Daemon::start(cfg, paths.clone()).await.unwrap();
        let port = daemon.bridge_addr.unwrap().port();
        let token = ymir_core::token::load(&paths).unwrap().unwrap();
        let node = Self { daemon, paths, base: format!("http://127.0.0.1:{port}"), token, http: reqwest::Client::new(), _dir: dir };
        node.wait_listed(&["tts", "omni-tts", "vl"]).await;
        node
    }

    async fn models(&self) -> Vec<Value> {
        let r = self.http.get(format!("{}/v1/models", self.base)).bearer_auth(&self.token).send().await.unwrap();
        assert_eq!(r.status(), 200);
        r.json::<Value>().await.unwrap()["data"].as_array().unwrap().clone()
    }

    async fn model(&self, id: &str) -> Value {
        self.models().await.into_iter().find(|m| m["id"] == id).unwrap_or_else(|| panic!("{id} not listed"))
    }

    async fn wait_listed(&self, ids: &[&str]) {
        for _ in 0..150 {
            let listed = self.models().await;
            if ids.iter().all(|id| listed.iter().any(|m| m["id"] == *id)) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        panic!("never listed: {ids:?}");
    }

    async fn speech(&self, body: Value) -> reqwest::Response {
        self.http.post(format!("{}/v1/audio/speech", self.base)).bearer_auth(&self.token).json(&body).send().await.unwrap()
    }

    fn socket(&self) -> std::path::PathBuf {
        self.paths.socket()
    }
}

fn opts(text: &str, model: &str, voice: &str) -> Options {
    Options {
        text: Some(Text::Plain(text.into())),
        model: Some(model.into()),
        voice: Some(weirdnet_agent::remote::infer::v2::Voice::preset(voice)),
        voices: Vec::new(),
        instructions: None,
        language: None,
        speed: None,
        seed: None,
        format: OutFormat::Wav,
        chunk: Chunking::Auto,
        pause_ms: 300,
        crossfade_ms: 10,
    }
}

/// The mock's length for `words` words: 60 ms a word, at least 200 ms.
fn mock_ms(words: u64) -> u64 {
    (words * 60).max(200)
}

fn wav_ms(b: &[u8]) -> (u64, u32) {
    let p = decode_wav(b).unwrap();
    (p.samples.len() as u64 * 1000 / (u64::from(p.sample_rate) * u64::from(p.channels)), p.sample_rate)
}

fn header<'a>(r: &'a reqwest::Response, name: &str) -> &'a str {
    r.headers().get(name).and_then(|v| v.to_str().ok()).unwrap_or("")
}

async fn error_of(r: reqwest::Response) -> (u16, String) {
    let status = r.status().as_u16();
    let v: Value = r.json().await.unwrap_or_default();
    (status, v["error"]["message"].as_str().unwrap_or_default().to_string())
}

async fn ctl(home: &Path, args: &[&str]) -> String {
    let h = home.display().to_string();
    let mut argv = vec!["ymirctl", "--home", &h];
    argv.extend_from_slice(args);
    let cli = <ymirctl::Cli as clap::Parser>::try_parse_from(argv).unwrap();
    let mut out = Vec::new();
    ymirctl::run(cli, &mut out, &mut std::io::empty()).await.unwrap();
    String::from_utf8(out).unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn speakers_in_both_flavors_through_ymirctl_mcp_and_the_bridge() {
    let node = Node::start().await;
    let socket = node.socket();

    // The catalog: both speakers have purpose `speak` before they load.
    for id in ["tts", "omni-tts"] {
        assert_eq!(node.model(id).await["weirdnet"]["purpose"], "speak", "{id}");
    }

    // `ymirctl speak` (control socket): llama-server, cold, loads on first use.
    let out = ymirctl::speak::speak(&socket, &opts("Hello from the llama speaker.", "tts", "am_adam"), None, &mut |_| {}).await.unwrap();
    assert_eq!(out.model, "tts");
    assert_eq!(out.nodes, vec!["speech-node".to_string()]);
    assert_eq!(wav_ms(&out.audio), (mock_ms(5), 24_000));
    // vLLM-Omni: the probe at load found its rate; the voices come from /v1/audio/voices.
    let out = ymirctl::speak::speak(&socket, &opts("Hello from the omni speaker.", "omni-tts", "ryan"), None, &mut |_| {}).await.unwrap();
    assert_eq!(out.model, "omni-tts");
    assert_eq!(wav_ms(&out.audio), (mock_ms(5), OMNI_RATE));
    // A voice the engine doesn't report: refused before the engine is asked.
    let e = ymirctl::speak::speak(&socket, &opts("No such voice.", "tts", "nobody"), None, &mut |_| {}).await.unwrap_err().to_string();
    assert!(e.contains("nobody") || e.contains("voice"), "{e}");

    // Loaded: the profiles the engines reported are in the catalog (and /v1/models).
    let tts = node.model("tts").await;
    assert_eq!(tts["weirdnet"]["speech"]["presets"], json!(["af_heart", "am_adam"]), "{tts}");
    let omni = node.model("omni-tts").await;
    assert_eq!(omni["weirdnet"]["speech"]["presets"], json!(["vivian", "ryan"]), "{omni}");
    assert_eq!(omni["weirdnet"]["speech"]["sample_rate"], OMNI_RATE);
    assert_eq!(omni["weirdnet"]["speech"]["max_input_chars"], 4000, "the operator's narrowing is published");

    // MCP `ymir_speak`, to a file under its root.
    {
        use ymirctl::mcp::{Policy, YmirTools};
        let root = node.paths.home.join("audio");
        std::fs::create_dir_all(&root).unwrap();
        let tools = YmirTools::new(node.paths.clone(), Some(node.paths.home.clone()), Policy { read_only: false, allow_remove: false })
            .with_roots(std::slice::from_ref(&root))
            .unwrap();
        let rpc = ymirctl::mcp::server(tools);
        let file = root.join("omni.wav");
        let r = rpc
            .handle_async(
                None,
                json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "ymir_speak", "arguments": {
                    "text": "Spoken for an agent.", "model": "omni-tts", "voice": "vivian", "out": file, "confirm": true
                }}}),
            )
            .await
            .unwrap();
        let r = &r["result"];
        assert_eq!(r["isError"], false, "{r}");
        assert_eq!(r["structuredContent"]["model"], "omni-tts");
        assert_eq!(wav_ms(&std::fs::read(&file).unwrap()), (mock_ms(4), OMNI_RATE));
        // ymir_models shows the speaker with its profile.
        let r = rpc
            .handle_async(None, json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "ymir_models", "arguments": {}}}))
            .await
            .unwrap();
        let listed = r["result"]["structuredContent"]["models"].as_array().cloned().unwrap_or_default();
        let tts = listed.iter().find(|m| m["id"] == "tts").unwrap_or_else(|| panic!("{r}"));
        assert_eq!(tts["purpose"], "speak");
        assert_eq!(tts["speech"]["presets"], json!(["af_heart", "am_adam"]));
    }

    // The bridge: OpenAI's /v1/audio/speech, both flavors.
    let r = node.speech(json!({"model": "tts", "input": "Through the bridge now.", "voice": "af_heart", "response_format": "wav"})).await;
    assert_eq!(r.status(), 200);
    assert_eq!(header(&r, "content-type"), "audio/wav");
    assert_eq!(header(&r, "x-weirdnet-model"), "tts");
    assert_eq!(wav_ms(&r.bytes().await.unwrap()), (mock_ms(4), 24_000));
    let r = node.speech(json!({"model": "omni-tts", "input": "Raw samples please.", "voice": "vivian", "response_format": "pcm"})).await;
    assert_eq!(r.status(), 200);
    assert_eq!(header(&r, "content-type"), format!("audio/pcm;rate={OMNI_RATE}"));
    assert_eq!(r.bytes().await.unwrap().len() as u64, u64::from(OMNI_RATE) * mock_ms(3) / 1000 * 2);
    // Streamed WAV from llama-server.
    let r = node.speech(json!({"model": "tts", "input": "Streaming it.", "voice": "am_adam", "stream_format": "audio"})).await;
    assert_eq!(r.status(), 200);
    assert!(!r.bytes().await.unwrap().is_empty());
    // No model named: the pool picks a speaker that has the voice.
    let r = node.speech(json!({"input": "Pick for me.", "voice": "ryan"})).await;
    assert_eq!(r.status(), 200);
    assert_eq!(header(&r, "x-weirdnet-model"), "omni-tts");
    // Errors are clean 400s: an unknown voice, a speaker asked to chat, a chat model asked to speak.
    let (status, message) = error_of(node.speech(json!({"model": "tts", "input": "x", "voice": "nobody"})).await).await;
    assert_eq!(status, 400, "{message}");
    let (status, message) = error_of(node.speech(json!({"model": "vl", "input": "x"})).await).await;
    assert_eq!(status, 400, "{message}");
    assert!(message.contains("speak"), "{message}");
    let r = node
        .http
        .post(format!("{}/v1/chat/completions", node.base))
        .bearer_auth(&node.token)
        .json(&json!({"model": "tts", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();
    let (status, message) = error_of(r).await;
    assert_eq!(status, 400, "{message}");
    assert!(message.contains("text-to-speech"), "{message}");

    // `ymirctl status`: the speakers, what they do and who may use them.
    let status = ctl(&node.paths.home, &["status"]).await;
    let line = |id: &str| status.lines().find(|l| l.trim_start().starts_with(&format!("{id} "))).unwrap_or_else(|| panic!("{status}")).to_string();
    let tts = line("tts");
    assert!(tts.contains("speak") && tts.contains("speaks (text to speech): 2 voices (af_heart, am_adam)"), "{tts}");
    let omni = line("omni-tts");
    assert!(omni.contains("vllm-omni") && omni.contains("22.05 kHz") && omni.contains("not for guests"), "{omni}");
    node.daemon.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn native_video_is_published_in_the_catalog_and_v1_models() {
    let node = Node::start().await;
    let vl = node.model("vl").await;
    let w = &vl["weirdnet"];
    assert_eq!(w["purpose"], "chat");
    assert_eq!(w["video"], true, "{vl}");
    assert_eq!(w["inputs"]["video"], true, "{vl}");
    assert_eq!(w["limits"]["video_max_seconds"], 60, "{vl}");
    assert_eq!(w["limits"]["video_max_bytes"], 50_000_000u64, "{vl}");
    assert_eq!(w["limits"]["video_frames_per_minute"], 120, "{vl}");
    // Speakers and the rest take no video.
    assert_ne!(node.model("tts").await["weirdnet"]["inputs"]["video"], true);

    // catalog/1 itself, as Bragi reads it: inputs.video and the limits on the model.
    let catalog = node.daemon.coordinator.as_ref().expect("the inference role").agent_catalog();
    let m = catalog.iter().find(|m| m.id == "vl").unwrap();
    assert!(m.inputs.video);
    assert_eq!((m.limits.video_max_seconds, m.limits.video_max_bytes, m.limits.video_frames_per_minute), (Some(60), Some(50_000_000), Some(120)));
    assert_eq!(m.limits.max_images, Some(8));

    // `ymirctl status` names it.
    let status = ctl(&node.paths.home, &["status"]).await;
    let line = status.lines().find(|l| l.trim_start().starts_with("vl ")).unwrap_or_else(|| panic!("{status}"));
    assert!(line.contains("video") && line.contains("video up to 60 s") && line.contains("120 video frames per minute"), "{line}");
    node.daemon.shutdown().await;
}
