//! Speech out: `ymirctl speak` and MCP `ymir_speak` through ymird's control socket.
//!
//! A mock speaker (`common/speech.rs`: an `InferHost` whose `open_speak` hands out a
//! `SpeakBackend` making a tone per voice, 10 ms per character) is served over weirdnet-agent's
//! own `v2::serve` and `catalog::serve` on an in-memory dialer, so `Pool::speak` runs exactly as
//! it does across the mesh. A control handler with that pool serves a real socket. Long text is planned to the
//! model's `max_input_chars`, sent piece by piece with one model, voice and seed, and stitched;
//! a script goes one turn at a time with that speaker's voice; MP3 goes out as WAV and is
//! encoded here with ffmpeg (skipped without it). Nothing here touches the network or a model.

use std::io::Write;
use std::path::Path;
use std::sync::Arc;

#[path = "common/speech.rs"]
mod speech;

use base64::Engine as _;
use clap::Parser;
use serde_json::{Value, json};
use speech::{MAX_CHARS, MS_PER_CHAR, RATE, Seen};
use weirdnet_agent::remote::infer::v2::{SpeakInput, SpeakRequest, Voice};
use weirdnet_agent::speech::AudioFormat;
use weirdnet_media::pcm::{Pcm, decode_wav, encode_wav};
use ymir_core::Paths;
use ymir_core::control::{self, ControlHandler};
use ymirctl::speak::{Chunking, Options, OutFormat, Text};

/// The control socket under `home`, over the mock pool.
async fn node(home: &Path) -> Seen {
    let (pool, seen) = speech::pool();
    let pool = Arc::new(pool);
    let handler = Arc::new(ControlHandler {
        node: "tts".into(),
        roles: vec!["app".into()],
        bridge_port: None,
        main_model: None,
        coordinator: None,
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
    let (tx, rx) = tokio::sync::watch::channel(false);
    tokio::spawn(async move {
        let _keep = tx;
        control::serve(listener, handler, rx).await;
    });
    seen
}

fn opts(text: Text) -> Options {
    Options {
        text: Some(text),
        model: None,
        voice: None,
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

fn wav_ms(b: &[u8]) -> u64 {
    let p = decode_wav(b).unwrap();
    p.samples.len() as u64 * 1000 / (u64::from(p.sample_rate) * u64::from(p.channels))
}

/// Three paragraphs of `n` characters each.
fn paragraphs(n: usize) -> (String, Vec<String>) {
    let ps: Vec<String> = ["a", "b", "c"].iter().map(|w| format!("{}end.", format!("{w}{w}{w} ").repeat((n - 4) / 4))).collect();
    (ps.join("\n\n"), ps)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn speak_through_the_control_socket() {
    let home = tempfile::tempdir().unwrap();
    let seen = node(home.path()).await;
    let socket = Paths::new(home.path().to_path_buf()).socket();

    // The speaker the pool would pick, with its profile; a preset it lacks is refused with the list.
    let mut want = SpeakRequest::text("hi");
    let t = control::speak_target(&socket, &want).await.unwrap();
    assert_eq!(t.model, "mock-tts");
    assert_eq!(t.profile.as_ref().and_then(|p| p.max_input_chars), Some(MAX_CHARS));
    want = want.with_voice(Voice::preset("gandalf"));
    let e = control::speak_target(&socket, &want).await.unwrap_err().to_string();
    assert!(e.contains("gandalf") || e.contains("vivian"), "{e}");

    // One short request: WAV at the model's rate, the voice and seed passed through.
    let mut o = opts(Text::Plain("Hello there, world.".into()));
    o.voice = Some(Voice::preset("vivian"));
    o.seed = Some(7);
    let out = ymirctl::speak::speak(&socket, &o, None, &mut |_| {}).await.unwrap();
    assert_eq!((out.model.as_str(), out.pieces, out.sample_rate, out.channels), ("mock-tts", 1, Some(RATE), Some(1)));
    assert_eq!(out.nodes, vec!["tts".to_string()]);
    assert_eq!(out.duration_ms, Some(19 * MS_PER_CHAR));
    assert_eq!(wav_ms(&out.audio), 19 * MS_PER_CHAR);
    assert_eq!((out.usage.input_chars, out.usage.audio_ms), (19, 190));
    {
        let s = seen.lock().unwrap();
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].voice, Some(Voice::preset("vivian")));
        assert_eq!((s[0].seed, s[0].model.as_deref()), (Some(7), Some("mock-tts")));
    }

    // PCM16 straight from the model.
    let mut o = opts(Text::Plain("Raw samples please.".into()));
    o.format = OutFormat::Pcm16;
    let out = ymirctl::speak::speak(&socket, &o, None, &mut |_| {}).await.unwrap();
    assert_eq!(out.format, AudioFormat::Pcm16);
    assert_eq!(out.audio.len() as u64, u64::from(RATE) * 190 / 1000 * 2);

    // Long text: three paragraphs over the 120-character limit go as three requests to one model
    // with one voice and seed, stitched with 300 ms pauses (each join also crossfades 10 ms).
    seen.lock().unwrap().clear();
    let (long, parts) = paragraphs(100);
    let mut o = opts(Text::Plain(long.clone()));
    o.voice = Some(Voice::preset("ryan"));
    o.seed = Some(42);
    let mut lines = Vec::new();
    let out = ymirctl::speak::speak(&socket, &o, None, &mut |l| lines.push(l)).await.unwrap();
    assert_eq!(out.pieces, 3);
    assert_eq!(lines.len(), 3, "{lines:?}");
    let chars: u64 = parts.iter().map(|p| p.chars().count() as u64).sum();
    assert_eq!(out.usage.input_chars, chars);
    let spoken = chars * MS_PER_CHAR;
    let got = wav_ms(&out.audio);
    let want = spoken + 2 * 300;
    assert!(got.abs_diff(want) <= 2 * 10 + 2, "stitched {got} ms, about {want} ms");
    assert_eq!(out.duration_ms, Some(got));
    {
        let s = seen.lock().unwrap();
        assert_eq!(s.len(), 3);
        for (r, p) in s.iter().zip(&parts) {
            assert_eq!(r.input, SpeakInput::Text(p.clone()));
            assert_eq!((r.voice.clone(), r.seed, r.model.as_deref()), (Some(Voice::preset("ryan")), Some(42), Some("mock-tts")));
            assert_eq!(r.format, Some(AudioFormat::Wav));
        }
    }

    // --chunk off: one request, which the model refuses with its limit.
    let mut o = opts(Text::Plain(long));
    o.chunk = Chunking::Off;
    let e = ymirctl::speak::speak(&socket, &o, None, &mut |_| {}).await.unwrap_err().to_string();
    assert!(e.contains("120"), "{e}");

    // A two-speaker script: one turn per request, each with only its speaker's voice.
    seen.lock().unwrap().clear();
    let mut o = opts(Text::Script(vec![(0, "Morning, Ryan.".into()), (1, "Morning, Vivian.".into()), (0, "Coffee?".into())]));
    o.voices = vec![(0, Voice::preset("vivian")), (1, Voice::preset("ryan"))];
    let out = ymirctl::speak::speak(&socket, &o, None, &mut |_| {}).await.unwrap();
    assert_eq!(out.pieces, 3);
    {
        let s = seen.lock().unwrap();
        assert_eq!(s.len(), 3);
        let speakers: Vec<(u8, String)> = s
            .iter()
            .map(|r| match (&r.input, r.voices.as_slice()) {
                (SpeakInput::Script(t), [v]) if t.len() == 1 && v.speaker == t[0].speaker => match &v.voice {
                    Voice::Preset { name } => (t[0].speaker, name.clone()),
                    other => panic!("{other:?}"),
                },
                other => panic!("one turn with its voice, got {other:?}"),
            })
            .collect();
        assert_eq!(speakers, [(0, "vivian".into()), (1, "ryan".into()), (0, "vivian".into())]);
    }

    // A cloned voice: the clip crosses the socket as bytes and reaches the backend unchanged.
    seen.lock().unwrap().clear();
    let clip = encode_wav(&Pcm::new(16_000, 1, vec![100; 16_000]));
    let mut o = opts(Text::Plain("In my own voice.".into()));
    o.voice = Some(ymirctl::speak::clone_voice(clip.clone(), Some("a reference".into()), true).unwrap());
    ymirctl::speak::speak(&socket, &o, None, &mut |_| {}).await.unwrap();
    match &seen.lock().unwrap()[0].voice {
        Some(Voice::Clone { audio, format, text, consent }) => {
            assert_eq!((audio, format.as_str(), text.as_deref(), *consent), (&clip, "wav", Some("a reference"), true));
        }
        other => panic!("{other:?}"),
    }
    assert!(ymirctl::speak::clone_voice(clip, None, false).unwrap_err().to_string().contains("consent"));

    // MP3: asked of the model as WAV, encoded here (needs ffmpeg with ffprobe).
    if weirdnet_media::ffmpeg::Runner::locate(None).is_ok() {
        let mut o = opts(Text::Plain("Compressed, please.".into()));
        o.format = OutFormat::Mp3;
        let out = ymirctl::speak::speak(&socket, &o, None, &mut |_| {}).await.unwrap();
        assert_eq!(out.format, AudioFormat::Mp3);
        assert!(out.audio.starts_with(b"ID3") || out.audio[0] == 0xFF, "not an MP3");
        assert_eq!(seen.lock().unwrap().last().unwrap().format, Some(AudioFormat::Wav));
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ymirctl_speak_cli() {
    let home = tempfile::tempdir().unwrap();
    let seen = node(home.path()).await;
    let h = home.path().display().to_string();
    let socket = Paths::new(home.path().to_path_buf()).socket();

    // To a file, with a preset voice; the summary is on stderr, so `out` stays empty.
    let wav = home.path().join("hi.wav");
    let cli =
        ymirctl::Cli::try_parse_from(["ymirctl", "--home", &h, "speak", "Hello from the CLI.", "--voice", "vivian", "-o", wav.to_str().unwrap()])
            .unwrap();
    let mut out = Vec::new();
    ymirctl::run(cli, &mut out, &mut std::io::empty()).await.unwrap();
    assert!(out.is_empty());
    assert_eq!(wav_ms(&std::fs::read(&wav).unwrap()), 19 * MS_PER_CHAR);

    // From stdin to stdout (`--out -`), as raw PCM.
    let cli = ymirctl::Cli::try_parse_from(["ymirctl", "--home", &h, "speak", "-", "--format", "pcm16", "--out", "-"]).unwrap();
    let mut out = Vec::new();
    ymirctl::run(cli, &mut out, &mut &b"Piped text."[..]).await.unwrap();
    assert_eq!(out.len() as u64, u64::from(RATE) * 110 / 1000 * 2);

    // A script file with per-speaker voices (`--voice 0=vivian --voice 1=ryan`).
    seen.lock().unwrap().clear();
    let script = home.path().join("dialog.txt");
    std::fs::write(&script, "S0: Shall we start?\nS1: Yes, let's.\n").unwrap();
    let wav = home.path().join("dialog.wav");
    let args = [
        "ymirctl",
        "--home",
        &h,
        "speak",
        "--script",
        script.to_str().unwrap(),
        "--voice",
        "0=vivian",
        "--voice",
        "1=ryan",
        "-o",
        wav.to_str().unwrap(),
    ];
    ymirctl::run(ymirctl::Cli::try_parse_from(args).unwrap(), &mut Vec::new(), &mut std::io::empty()).await.unwrap();
    assert_eq!(seen.lock().unwrap().len(), 2);
    let got = wav_ms(&std::fs::read(&wav).unwrap());
    let want = ("Shall we start?".len() + "Yes, let's.".len()) as u64 * MS_PER_CHAR + 300;
    assert!(got.abs_diff(want) <= 12, "{got} vs {want}");

    // No --out with a terminal on stdout: refused before anything is spoken.
    seen.lock().unwrap().clear();
    let cli = ymirctl::Cli::try_parse_from(["ymirctl", "--home", &h, "speak", "Not to a terminal."]).unwrap();
    let ymirctl::Cmd::Speak(args) = cli.cmd else { panic!() };
    let e = ymirctl::speak::run(&socket, args, &mut Vec::new(), &mut std::io::empty(), true).await.unwrap_err().to_string();
    assert!(e.contains("terminal"), "{e}");
    assert!(seen.lock().unwrap().is_empty());

    // A clone without consent is refused here, before the clip is sent.
    let clip = home.path().join("me.wav");
    std::fs::write(&clip, encode_wav(&Pcm::new(16_000, 1, vec![0; 8_000]))).unwrap();
    let args = ["ymirctl", "--home", &h, "speak", "Copy me.", "--clone-audio", clip.to_str().unwrap(), "-o", wav.to_str().unwrap()];
    let e = ymirctl::run(ymirctl::Cli::try_parse_from(args).unwrap(), &mut Vec::new(), &mut std::io::empty()).await.unwrap_err().to_string();
    assert!(e.contains("consent"), "{e}");
    assert!(seen.lock().unwrap().is_empty());
}

type Rpc = weirdnet_mcp::Server<ymirctl::mcp::YmirTools>;

async fn call(rpc: &Rpc, name: &str, args: Value) -> Value {
    let r = rpc
        .handle_async(None, json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": args}}))
        .await
        .unwrap();
    r["result"].clone()
}

fn text_of(r: &Value) -> String {
    r["content"][0]["text"].as_str().unwrap_or_default().to_string()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mcp_ymir_speak() {
    use ymirctl::mcp::{Policy, YmirTools};
    let home = tempfile::tempdir().unwrap();
    let seen = node(home.path()).await;
    let paths = Paths::new(home.path().to_path_buf());
    let root = home.path().join("audio");
    std::fs::create_dir_all(&root).unwrap();
    let tools = YmirTools::new(paths.clone(), Some(home.path().to_path_buf()), Policy { read_only: false, allow_remove: false })
        .with_roots(std::slice::from_ref(&root))
        .unwrap();
    let rpc = ymirctl::mcp::server(tools);

    // Read-only: not listed, and not callable.
    let ro = ymirctl::mcp::server(YmirTools::new(paths, Some(home.path().to_path_buf()), Policy { read_only: true, allow_remove: false }));
    let list = ro.handle_async(None, json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})).await.unwrap();
    assert!(!list.to_string().contains("ymir_speak"));
    let r = ro
        .handle_async(None, json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "ymir_speak", "arguments": {"text": "hi"}}}))
        .await
        .unwrap();
    assert!(r["result"]["isError"] == true || r["error"].is_object(), "{r}");

    // To a file under the root, with confirm: the path, length and usage come back.
    let out = root.join("hello.wav");
    let r = call(&rpc, "ymir_speak", json!({"text": "Hello from MCP.", "voice": "ryan", "out": out, "confirm": true})).await;
    assert_eq!(r["isError"], false, "{r}");
    let v = &r["structuredContent"];
    assert_eq!(v["path"], out.canonicalize().unwrap().display().to_string());
    assert_eq!(v["duration_ms"], 15 * MS_PER_CHAR);
    assert_eq!(v["usage"]["input_chars"], 15);
    assert_eq!((v["model"].as_str(), v["format"].as_str()), (Some("mock-tts"), Some("wav")));
    assert_eq!(wav_ms(&std::fs::read(&out).unwrap()), 150);

    // No `out`: base64 audio, and no confirm needed (nothing is written).
    let r = call(
        &rpc,
        "ymir_speak",
        json!({"script": [{"speaker": 0, "text": "One."}, {"speaker": 1, "text": "Two."}], "voices": {"0": "vivian", "1": "ryan"}}),
    )
    .await;
    assert_eq!(r["isError"], false, "{r}");
    let audio = base64::engine::general_purpose::STANDARD.decode(r["structuredContent"]["audio_base64"].as_str().unwrap()).unwrap();
    assert!(wav_ms(&audio) > 0);
    assert_eq!(r["structuredContent"]["pieces"], 2);

    // Refusals, all before anything is spoken.
    seen.lock().unwrap().clear();
    let outside = home.path().join("out.wav");
    let clip = root.join("me.wav");
    std::fs::write(&clip, encode_wav(&Pcm::new(16_000, 1, vec![0; 8_000]))).unwrap();
    let away = home.path().join("away.wav");
    std::fs::write(&away, encode_wav(&Pcm::new(16_000, 1, vec![0; 8_000]))).unwrap();
    for (args, says) in [
        (json!({"text": "x", "out": root.join("a.wav")}), "confirm"),
        (json!({"text": "x", "out": outside, "confirm": true}), "outside the allowed roots"),
        (json!({"text": "x", "out": root.join("a.mp3"), "confirm": true}), ".wav"),
        (json!({"text": "x", "out": root.join("nope/a.wav"), "confirm": true}), "must exist"),
        (json!({"text": "x", "out": "https://example.com/a.wav", "confirm": true}), "file on this machine"),
        (json!({"text": "x", "clone_path": "https://example.com/me.wav", "consent": true}), "never fetched"),
        (json!({"text": "x", "clone_path": "data:audio/wav;base64,AAAA", "consent": true}), "never fetched"),
        (json!({"text": "x", "clone_audio": "data:audio/wav;base64,AAAA", "consent": true}), "not a URL"),
        (json!({"text": "x", "clone_path": away, "consent": true}), "outside the allowed roots"),
        (json!({"text": "x", "clone_path": clip}), "consent"),
        (json!({"text": "x", "voice": "vivian", "design": "a pirate"}), "one of"),
    ] {
        let r = call(&rpc, "ymir_speak", args.clone()).await;
        assert_eq!(r["isError"], true, "{args}: {r}");
        assert!(text_of(&r).contains(says), "{args}: {}", text_of(&r));
    }
    assert!(seen.lock().unwrap().is_empty());

    // A clone from a clip under the root, with consent.
    let r = call(&rpc, "ymir_speak", json!({"text": "Mine.", "clone_path": clip, "clone_text": "silence", "consent": true})).await;
    assert_eq!(r["isError"], false, "{r}");
    assert!(matches!(seen.lock().unwrap()[0].voice, Some(Voice::Clone { consent: true, .. })));
    let _ = std::io::stdout().flush();
}
