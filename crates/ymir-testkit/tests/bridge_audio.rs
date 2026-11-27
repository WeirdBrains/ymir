//! The app role's OpenAI bridge (`weirdnet-openai` 0.2.5) and audio: `POST /v1/audio/speech`,
//! `POST /v1/audio/transcriptions` and `/v1/models` `weirdnet.speech`.
//!
//! First the bridge as ymird binds it (`daemon::bind_bridge`) over a pool of the mock speaker and
//! transcriber (`common/speech.rs`, served through weirdnet-agent's own `v2::serve`), with a small
//! `max_request_bytes`. Then a real ymird with the mock engine as a vLLM transcriber: the
//! transcription goes through `InferenceHost` to the engine, and speech with no speaker in the
//! pool is a clean 400 (real speakers: `speak_e2e.rs`). Nothing touches the network or a model.

use std::sync::Arc;

#[path = "common/speech.rs"]
mod speech;

use base64::Engine as _;
use serde_json::{Value, json};
use speech::{MAX_CHARS, MS_PER_CHAR, RATE};
use weirdnet_agent::remote::infer::v2::{SpeakInput, Voice};
use weirdnet_media::pcm::{Pcm, decode_wav, encode_wav};
use ymir_core::config::BridgeSection;
use ymir_core::{Daemon, NodeConfig, Paths};

const MOCK: &str = env!("CARGO_BIN_EXE_ymir-mock-engine");
const TOKEN: &str = "bridge-audio-test-token-0123456789";
const LIMIT: u64 = 256 << 10;

struct Bridge {
    base: String,
    token: String,
    http: reqwest::Client,
}

impl Bridge {
    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        self.http.post(format!("{}{path}", self.base)).bearer_auth(&self.token)
    }

    async fn speech(&self, body: Value) -> reqwest::Response {
        self.post("/v1/audio/speech").json(&body).send().await.unwrap()
    }

    async fn models(&self) -> Vec<Value> {
        let r = self.http.get(format!("{}/v1/models", self.base)).bearer_auth(&self.token).send().await.unwrap();
        assert_eq!(r.status(), 200);
        r.json::<Value>().await.unwrap()["data"].as_array().unwrap().clone()
    }

    /// A multipart transcription form: `file` (with a name) and text fields.
    async fn transcribe(&self, file: &[u8], name: &str, fields: &[(&str, &str)]) -> reqwest::Response {
        let boundary = "ymir-bridge-audio-boundary";
        let mut body = Vec::new();
        for (k, v) in fields {
            body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").as_bytes());
        }
        body.extend_from_slice(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\nContent-Type: audio/wav\r\n\r\n")
                .as_bytes(),
        );
        body.extend_from_slice(file);
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
        self.post("/v1/audio/transcriptions")
            .header("content-type", format!("multipart/form-data; boundary={boundary}"))
            .body(body)
            .send()
            .await
            .unwrap()
    }
}

fn header<'a>(r: &'a reqwest::Response, name: &str) -> &'a str {
    r.headers().get(name).and_then(|v| v.to_str().ok()).unwrap_or("")
}

async fn error_of(r: reqwest::Response) -> (u16, String) {
    let status = r.status().as_u16();
    let v: Value = r.json().await.unwrap_or_default();
    (status, v["error"]["message"].as_str().unwrap_or_default().to_string())
}

fn clip(ms: u32) -> Vec<u8> {
    encode_wav(&Pcm::new(16_000, 1, vec![300; (16 * ms) as usize]))
}

/// The bridge ymird binds, over the mock pool.
async fn mock_bridge() -> (Bridge, speech::Seen) {
    let (pool, seen) = speech::pool();
    let section = BridgeSection { port: 0, max_request_bytes: Some(LIMIT) };
    let bridge = ymir_core::daemon::bind_bridge(&section, TOKEN.into()).await.unwrap();
    assert_eq!(bridge.max_request_bytes() as u64, LIMIT);
    let port = bridge.local_addr().unwrap().port();
    tokio::spawn(bridge.serve(Arc::new(pool), std::future::pending()));
    (Bridge { base: format!("http://127.0.0.1:{port}"), token: TOKEN.into(), http: reqwest::Client::new() }, seen)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn models_list_the_speaker_with_its_profile() {
    let (b, _) = mock_bridge().await;
    let models = b.models().await;
    let tts = models.iter().find(|m| m["id"] == "mock-tts").expect("the speaker");
    let w = &tts["weirdnet"];
    assert_eq!(w["purpose"], "speak");
    assert_eq!(w["speech"]["presets"], json!(["vivian", "ryan"]));
    assert_eq!(w["speech"]["formats"], json!(["wav", "pcm16"]));
    assert_eq!(w["speech"]["max_input_chars"], MAX_CHARS);
    assert_eq!(w["speech"]["clone"], "audio_and_text");
    assert_eq!(w["max_request_bytes"], LIMIT, "the bridge's limit, as ymird set it");
    assert!(w["inputs"].is_object());
    let stt = models.iter().find(|m| m["id"] == "mock-stt").expect("the transcriber");
    assert_eq!(stt["weirdnet"]["purpose"], "transcribe");
    assert!(stt["weirdnet"].get("speech").is_none());
    // The token guards it as every route.
    let r = b.http.get(format!("{}/v1/models", b.base)).send().await.unwrap();
    assert_eq!(r.status(), 401);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn speech_through_the_bridge() {
    let (b, seen) = mock_bridge().await;

    // A preset voice, WAV: the model's rate and channels in headers, the request as sent.
    let r = b.speech(json!({"model": "mock-tts", "input": "Hello through the bridge.", "voice": "ryan", "response_format": "wav", "speed": 1.25, "weirdnet": {"seed": 9, "language": "en"}})).await;
    assert_eq!(r.status(), 200);
    assert_eq!(header(&r, "content-type"), "audio/wav");
    assert_eq!(header(&r, "x-audio-sample-rate"), RATE.to_string());
    assert_eq!(header(&r, "x-weirdnet-audio-channels"), "1");
    assert_eq!(header(&r, "x-weirdnet-model"), "mock-tts");
    let wav = r.bytes().await.unwrap();
    let pcm = decode_wav(&wav).unwrap();
    assert_eq!(pcm.samples.len() as u64, u64::from(RATE) * 25 * MS_PER_CHAR / 1000);
    {
        let s = seen.lock().unwrap();
        let req = s.last().unwrap();
        assert_eq!(req.voice, Some(Voice::preset("ryan")));
        assert_eq!((req.seed, req.speed, req.language.as_deref()), (Some(9), Some(1250), Some("en")));
    }

    // No model named: the bridge routes to the pool's speaker. `{"id": name}` voices, raw PCM.
    let r = b.speech(json!({"input": "Raw please.", "voice": {"id": "vivian"}, "response_format": "pcm"})).await;
    assert_eq!(r.status(), 200);
    assert_eq!(header(&r, "content-type"), format!("audio/pcm;rate={RATE}"));
    assert_eq!(header(&r, "x-weirdnet-model"), "mock-tts");
    assert_eq!(r.bytes().await.unwrap().len() as u64, u64::from(RATE) * 110 / 1000 * 2);

    // Streamed as raw bytes: the same audio.
    let r = b.speech(json!({"input": "Hello through the bridge.", "voice": "ryan", "stream_format": "audio"})).await;
    assert_eq!(r.status(), 200);
    assert_eq!(r.bytes().await.unwrap(), wav);

    // A cloned voice from a `data:` URL with consent reaches the speaker intact.
    let reference = clip(500);
    let url = format!("data:audio/wav;base64,{}", base64::engine::general_purpose::STANDARD.encode(&reference));
    let r = b.speech(json!({"input": "Mine.", "weirdnet": {"ref_audio": url, "ref_text": "hum", "consent": true}})).await;
    assert_eq!(r.status(), 200);
    match &seen.lock().unwrap().last().unwrap().voice {
        Some(Voice::Clone { audio, text, consent, .. }) => assert_eq!((audio, text.as_deref(), *consent), (&reference, Some("hum"), true)),
        other => panic!("{other:?}"),
    }

    // A two-speaker script.
    let r = b.speech(json!({"input": null, "weirdnet": {"script": [{"speaker": 0, "text": "Hi."}, {"speaker": 1, "text": "Hey."}]}})).await;
    assert_eq!(r.status(), 200);
    assert!(matches!(&seen.lock().unwrap().last().unwrap().input, SpeakInput::Script(t) if t.len() == 2));

    // Refusals, each a 400 that says why, before the speaker is asked.
    let before = seen.lock().unwrap().len();
    let long = "word ".repeat(40);
    for (body, says) in [
        (json!({"input": "x", "voice": "gandalf"}), "gandalf"),
        (json!({"input": "x", "response_format": "mp3"}), "mp3"),
        (json!({"input": long, "model": "mock-tts"}), "120"),
        (json!({"input": "x", "stream_format": "sse"}), "sse"),
        (json!({"input": "x", "weirdnet": {"ref_audio": url, "consent": false}}), "consent"),
        (json!({"input": "x", "weirdnet": {"ref_audio": "https://example.com/me.wav", "consent": true}}), "data:"),
        (json!({"input": "x", "model": "mock-stt"}), "mock-stt"),
    ] {
        let (status, message) = error_of(b.speech(body.clone()).await).await;
        assert_eq!(status, 400, "{body}: {message}");
        assert!(message.to_lowercase().contains(says), "{body}: {message}");
    }
    assert_eq!(seen.lock().unwrap().len(), before);

    // Over ymird's max_request_bytes: 413 naming the limit.
    let big = format!("data:audio/wav;base64,{}", base64::engine::general_purpose::STANDARD.encode(vec![0u8; LIMIT as usize]));
    let r = b.speech(json!({"input": "x", "weirdnet": {"ref_audio": big, "consent": true}})).await;
    assert_eq!(r.status(), 413);
    assert_eq!(header(&r, "x-weirdnet-max-request-bytes"), LIMIT.to_string());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn transcriptions_through_the_bridge() {
    let (b, _) = mock_bridge().await;
    let audio = clip(1_000);

    let r = b.transcribe(&audio, "talk.wav", &[("model", "mock-stt")]).await;
    assert_eq!(r.status(), 200);
    assert_eq!(header(&r, "x-weirdnet-model"), "mock-stt");
    assert_eq!(r.json::<Value>().await.unwrap()["text"], format!("heard {} bytes of wav", audio.len()));

    let r = b.transcribe(&audio, "talk.wav", &[("response_format", "verbose_json"), ("timestamp_granularities[]", "segment")]).await;
    assert_eq!(r.status(), 200);
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["language"], "en");
    assert_eq!(v["segments"].as_array().unwrap().len(), 1);

    let r = b.transcribe(&audio, "talk.wav", &[("response_format", "srt")]).await;
    assert_eq!(r.status(), 200);
    assert!(r.text().await.unwrap().contains("00:00:00,000 --> 00:00:01,000"));

    // A speaker named as the transcriber, or a granularity it doesn't have: 400.
    assert_eq!(b.transcribe(&audio, "talk.wav", &[("model", "mock-tts")]).await.status(), 400);
    assert_eq!(b.transcribe(&audio, "talk.wav", &[("timestamp_granularities[]", "word")]).await.status(), 400);

    // Over the limit: 413 naming it.
    let r = b.transcribe(&vec![0u8; LIMIT as usize + 1], "big.wav", &[]).await;
    assert_eq!(r.status(), 413);
    assert_eq!(header(&r, "x-weirdnet-max-request-bytes"), LIMIT.to_string());
}

/// A real ymird (model, inference and app roles) with the mock engine as a vLLM transcriber.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ymird_bridge_serves_audio_routes() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let models = dir.path().join("models");
    std::fs::create_dir_all(&home).unwrap();
    std::fs::create_dir_all(&models).unwrap();
    let toml = format!(
        "[node]\nname = \"audio-node\"\n[roles]\nmodel = true\ninference = true\napp = true\n\
         [model]\nllamacpp = \"off\"\nvllm = \"system:{MOCK}\"\nmodels_dir = \"{}\"\n\
         [[model.local]]\nid = \"whisper\"\nbackend = \"vllm\"\nfile = \"openai/whisper-large-v3\"\nrole = \"transcribe\"\n\
         [[model.local]]\nid = \"chatty\"\nbackend = \"vllm\"\nfile = \"org/chatty\"\n\
         [inference]\n[inference.alerts]\nprobe_secs = 3600\n[app.bridge]\nport = 0\nmax_request_bytes = {LIMIT}\n",
        models.display()
    );
    std::fs::write(home.join("node.toml"), toml).unwrap();
    let paths = Paths::new(&home);
    let cfg = NodeConfig::load(&paths.config()).unwrap();
    let daemon = Daemon::start(cfg, paths.clone()).await.unwrap();
    let port = daemon.bridge_addr.unwrap().port();
    let token = ymir_core::token::load(&paths).unwrap().unwrap();
    let b = Bridge { base: format!("http://127.0.0.1:{port}"), token, http: reqwest::Client::new() };

    // The transcriber is listed; this node has no speaker.
    let mut listed = Vec::new();
    for _ in 0..100 {
        listed = b.models().await;
        if listed.iter().any(|m| m["id"] == "whisper") {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    let whisper = listed.iter().find(|m| m["id"] == "whisper").expect("whisper listed");
    assert_eq!(whisper["weirdnet"]["purpose"], "transcribe");
    assert_eq!(whisper["weirdnet"]["max_request_bytes"], LIMIT);
    assert!(listed.iter().all(|m| m["weirdnet"].get("speech").is_none()));

    // A transcription through InferenceHost to the engine.
    let r = b.transcribe(&clip(2_000), "talk.wav", &[("model", "whisper")]).await;
    assert_eq!(r.status(), 200);
    assert!(r.json::<Value>().await.unwrap()["text"].as_str().unwrap().starts_with("heard "));

    // Speech with no speaker: a clean 400 naming what is missing, not a hang or a 500.
    let (status, message) = error_of(b.speech(json!({"input": "Hello."})).await).await;
    assert_eq!(status, 400, "{message}");
    assert!(message.contains("speak"), "{message}");
    let (status, _) = error_of(b.speech(json!({"input": "Hello.", "model": "whisper"})).await).await;
    assert_eq!(status, 400);

    // Over max_request_bytes: 413 from ymird's bridge.
    let r = b.transcribe(&vec![0u8; LIMIT as usize + 1], "big.wav", &[]).await;
    assert_eq!(r.status(), 413);
    daemon.shutdown().await;
}
