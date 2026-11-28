//! Audio design milestone 5: the mock engine's speech endpoints, so speech routing tests can run
//! the day `weirdnet-backends` has `Supervisor::speak`. Both flavors: llama-server
//! (`/v1/audio/speech`, `/tts`, `/props` voices) and vLLM-Omni (`--omni`: `/v1/audio/voices`,
//! `x-audio-sample-rate`, `data:` references only).

use std::time::Duration;

use serde_json::{Value, json};
use weirdnet_media::probe::wav_header;

const MOCK: &str = env!("CARGO_BIN_EXE_ymir-mock-engine");

struct Engine {
    _child: tokio::process::Child,
    base: String,
}

async fn engine(args: &[&str]) -> Engine {
    use tokio::io::AsyncBufReadExt as _;
    let mut child = tokio::process::Command::new(MOCK)
        .args(["--host", "127.0.0.1", "--port", "0"])
        .args(args)
        .env("YMIR_MOCK_LOAD_MS", "0")
        .env_remove("LLAMA_API_KEY")
        .env_remove("VLLM_API_KEY")
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let mut lines = tokio::io::BufReader::new(child.stderr.take().unwrap()).lines();
    let line = tokio::time::timeout(Duration::from_secs(30), async {
        while let Some(l) = lines.next_line().await.unwrap() {
            if l.contains("listening on") {
                return l;
            }
        }
        panic!("the mock exited without saying where it listens")
    })
    .await
    .unwrap();
    tokio::spawn(async move { while let Ok(Some(_)) = lines.next_line().await {} });
    let addr = line.rsplit(' ').next().unwrap().to_string();
    Engine { _child: child, base: format!("http://{addr}") }
}

async fn speak(e: &Engine, path: &str, body: Value) -> reqwest::Response {
    reqwest::Client::new().post(format!("{}{path}", e.base)).json(&body).send().await.unwrap()
}

async fn error(r: reqwest::Response) -> String {
    assert_eq!(r.status(), 400);
    r.json::<Value>().await.unwrap()["error"]["message"].as_str().unwrap().to_string()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn llama_server_flavor() {
    let e = engine(&["--alias", "kokoro", "--mock-voices", "af_heart,am_michael"]).await;
    let props: Value = reqwest::get(format!("{}/props", e.base)).await.unwrap().json().await.unwrap();
    assert_eq!(props["voices"], json!(["af_heart", "am_michael"]));

    // WAV at the model's rate, 60 ms a word; the text is not in the audio.
    let r = speak(&e, "/v1/audio/speech", json!({"input": "one two three four five", "voice": "am_michael"})).await;
    assert_eq!(r.status(), 200);
    assert_eq!(r.headers()["content-type"], "audio/wav");
    let b = r.bytes().await.unwrap();
    let i = wav_header(&b, None).unwrap();
    assert_eq!((i.sample_rate, i.channels, i.bits_per_sample, i.duration_ms()), (24_000, 1, 16, 300));
    assert!(!b.windows(5).any(|w| w == b"three"));

    // Raw PCM, speed, and /tts.
    let r = speak(&e, "/tts", json!({"input": "hello there", "response_format": "pcm", "speed": 2.0})).await;
    assert!(r.headers()["content-type"].to_str().unwrap().starts_with("audio/pcm;rate=24000"));
    assert_eq!(r.bytes().await.unwrap().len(), 24_000 * 2 / 10, "200 ms at speed 2 is 100 ms");

    // Streamed: a WAV header with unknown sizes, then all the samples.
    let r = speak(&e, "/v1/audio/speech", json!({"input": "a b c d e f g h i j", "stream_format": "audio"})).await;
    let b = r.bytes().await.unwrap();
    assert_eq!(&b[4..8], &u32::MAX.to_le_bytes());
    assert_eq!(wav_header(&b, None).unwrap().duration_ms(), 600);

    // Refusals name the problem, never the text.
    let m = error(speak(&e, "/v1/audio/speech", json!({"input": "secret words", "voice": "alloy"})).await).await;
    assert!(m.contains("af_heart, am_michael") && !m.contains("secret"), "{m}");
    assert!(error(speak(&e, "/v1/audio/speech", json!({"input": "x", "response_format": "mp3"})).await).await.contains("unsupported"));
    assert!(error(speak(&e, "/v1/audio/speech", json!({"input": "x", "speaker_ref_b64": "UklGRg=="})).await).await.contains("no voice encoder"));
    assert!(error(speak(&e, "/v1/audio/speech", json!({"input": " "})).await).await.contains("required"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn llama_server_cloning_and_no_voices() {
    let e = engine(&["--alias", "voxcpm2", "--mock-clone", "--mock-rate", "48000"]).await;
    let props: Value = reqwest::get(format!("{}/props", e.base)).await.unwrap().json().await.unwrap();
    assert!(props.get("voices").is_none(), "no voices reported when the model has none");
    let r =
        speak(&e, "/v1/audio/speech", json!({"input": "(A calm narrator)Hello.", "speaker_ref_b64": "UklGRg==", "speaker_ref_text": "ref"})).await;
    assert_eq!(wav_header(&r.bytes().await.unwrap(), None).unwrap().sample_rate, 48_000);
    assert!(error(speak(&e, "/v1/audio/speech", json!({"input": "x", "voice": "any"})).await).await.contains("no predefined voices"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn vllm_omni_flavor() {
    let e = engine(&["--served-model-name", "qwen3-tts", "--omni", "--mock-voices", "vivian,ryan", "--mock-clone"]).await;
    let v: Value = reqwest::get(format!("{}/v1/audio/voices", e.base)).await.unwrap().json().await.unwrap();
    assert_eq!(v, json!({"voices": ["vivian", "ryan"], "uploaded_voices": []}));

    let r = speak(&e, "/v1/audio/speech", json!({"input": "hi there you", "voice": "ryan", "task_type": "CustomVoice", "instructions": "cheerful"}))
        .await;
    assert_eq!(r.headers()["x-audio-sample-rate"], "24000");
    assert_eq!(r.headers()["x-audio-channels"], "1");
    assert_eq!(r.headers()["x-audio-duration-seconds"], "0.200");
    assert_eq!(wav_header(&r.bytes().await.unwrap(), None).unwrap().duration_ms(), 200);

    let r = speak(&e, "/v1/audio/speech", json!({"input": "x", "response_format": "pcm", "stream": true, "stream_format": "audio"})).await;
    assert!(r.headers().get("x-audio-duration-seconds").is_none(), "a stream can't know its length");
    assert_eq!(r.bytes().await.unwrap().len(), 24_000 * 2 / 5);

    // Cloning takes data: URLs only.
    let ok =
        speak(&e, "/v1/audio/speech", json!({"input": "x", "task_type": "Base", "ref_audio": "data:audio/wav;base64,UklGRg==", "ref_text": "r"}))
            .await;
    assert_eq!(ok.status(), 200);
    for url in ["https://192.0.2.1/a.wav", "file:///etc/passwd"] {
        assert!(error(speak(&e, "/v1/audio/speech", json!({"input": "x", "ref_audio": url})).await).await.contains("data: URL"));
    }
    assert!(error(speak(&e, "/v1/audio/speech", json!({"input": "x", "response_format": "opus"})).await).await.contains("no opus encoder"));
    // llama-server's /props has no voices in this flavor.
    let props: Value = reqwest::get(format!("{}/props", e.base)).await.unwrap().json().await.unwrap();
    assert!(props.get("voices").is_none());
}
