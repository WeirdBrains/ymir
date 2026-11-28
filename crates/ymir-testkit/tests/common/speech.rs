//! A mock speaker and transcriber behind weirdnet-agent's own `v2::serve` and `catalog::serve` on
//! an in-memory dialer, as a [`Pool`] (shared by `speak_cli.rs` and `bridge_audio.rs`).
//!
//! The speaker (`mock-tts`) makes a tone per voice, 10 ms per character, as WAV or PCM16 at
//! 24 kHz mono, and refuses what its profile lacks. The transcriber (`mock-stt`) answers
//! "heard N bytes of FORMAT" with one segment.
#![allow(dead_code)]

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use weirdnet_agent::remote::catalog::{self, CatalogFeed, CatalogModel, ModelState, Purpose};
use weirdnet_agent::remote::infer::v2::{
    self, AudioHeader, AudioStream, Segment, SpeakBackend, SpeakInput, SpeakRequest, TranscribeBackend, TranscribeRequest, Transcript, Usage, Voice,
};
use weirdnet_agent::remote::infer::{self, InferHost};
use weirdnet_agent::remote::pool::Pool;
use weirdnet_agent::remote::{Caller, MemoryDialer, RemoteError};
use weirdnet_agent::speech::{AudioFormat, CloneMode, SpeechProfile};
use weirdnet_agent::{CancelToken, ModelBackend};
use weirdnet_media::pcm::{Pcm, encode_wav};
use ymir_core::host::AllowAll;

pub const RATE: u32 = 24_000;
pub const MS_PER_CHAR: u64 = 10;
pub const MAX_CHARS: u32 = 120;

/// What the speaker was asked, in order.
pub type Seen = Arc<Mutex<Vec<SpeakRequest>>>;

struct Speaker {
    profile: SpeechProfile,
    seen: Seen,
}

fn freq(voice: Option<&Voice>) -> f32 {
    match voice {
        Some(Voice::Preset { name }) if name == "ryan" => 330.0,
        Some(Voice::Preset { .. }) => 220.0,
        Some(Voice::Design { .. }) => 440.0,
        Some(Voice::Clone { .. }) => 550.0,
        _ => 110.0,
    }
}

#[async_trait]
impl SpeakBackend for Speaker {
    fn name(&self) -> String {
        "mock-tts".into()
    }

    async fn speak(&self, request: &SpeakRequest, _: &CancelToken) -> Result<AudioStream, RemoteError> {
        self.profile.covers(&request.needs()).map_err(|m| RemoteError::BadRequest { message: m })?;
        self.seen.lock().unwrap().push(request.clone());
        let (chars, voice) = match &request.input {
            SpeakInput::Text(t) | SpeakInput::Raw(t) => (t.chars().count(), request.voice.as_ref()),
            SpeakInput::Script(turns) => {
                let first = turns.first().map(|t| t.speaker).unwrap_or(0);
                let v = request.voices.iter().find(|v| v.speaker == first).map(|v| &v.voice).or(request.voice.as_ref());
                (turns.iter().map(|t| t.text.chars().count()).sum(), v)
            }
            _ => return Err(RemoteError::BadRequest { message: "an input this mock doesn't know".into() }),
        };
        let ms = chars as u64 * MS_PER_CHAR;
        let n = (u64::from(RATE) * ms / 1000) as usize;
        let f = freq(voice);
        let samples = (0..n).map(|i| ((i as f32 * f * std::f32::consts::TAU / RATE as f32).sin() * 8_000.0) as i16).collect();
        let pcm = Pcm::new(RATE, 1, samples);
        let usage = Some(Usage::speech(chars as u64, ms));
        Ok(match request.format.unwrap_or_default() {
            AudioFormat::Pcm16 => {
                let raw = pcm.samples.iter().flat_map(|s| s.to_le_bytes()).collect();
                AudioStream::from_file(AudioHeader::new(AudioFormat::Pcm16).with_sample_rate(RATE).with_channels(1), raw, usage)
            }
            AudioFormat::Wav => {
                AudioStream::from_file(AudioHeader::new(AudioFormat::Wav).with_sample_rate(RATE).with_channels(1), encode_wav(&pcm), usage)
            }
            other => return Err(RemoteError::BadRequest { message: format!("mock-tts makes wav and pcm16, not {}", other.as_str()) }),
        })
    }
}

struct Transcriber;

#[async_trait]
impl TranscribeBackend for Transcriber {
    fn name(&self) -> String {
        "mock-stt".into()
    }
    async fn transcribe(&self, r: &TranscribeRequest, _: &CancelToken) -> Result<Transcript, RemoteError> {
        let text = format!("heard {} bytes of {}", r.audio.len(), r.format);
        Ok(Transcript::new(text.clone()).with_language("en").with_duration_ms(1_000).with_segments(vec![Segment::new(0, 1_000, text)]))
    }
}

struct Host {
    speaker: Arc<Speaker>,
}

impl InferHost for Host {
    fn authorize(&self, _: &Caller) -> Result<(), RemoteError> {
        Ok(())
    }
    fn open(&self, _: &Caller, _: Option<&str>) -> Result<Arc<dyn ModelBackend>, RemoteError> {
        Err(RemoteError::unavailable("no chat here"))
    }
    fn open_transcribe(&self, _: &Caller, model: Option<&str>) -> Result<Arc<dyn TranscribeBackend>, RemoteError> {
        match model {
            None | Some("mock-stt") => Ok(Arc::new(Transcriber)),
            Some(m) => Err(RemoteError::Unsupported { message: format!("{m} does not transcribe here") }),
        }
    }
    fn open_speak(&self, _: &Caller, request: &SpeakRequest) -> Result<Arc<dyn SpeakBackend>, RemoteError> {
        match request.model.as_deref() {
            None | Some("mock-tts") => Ok(self.speaker.clone()),
            Some(m) => Err(RemoteError::Unsupported { message: format!("{m} does not speak here") }),
        }
    }
}

pub fn profile() -> SpeechProfile {
    SpeechProfile::new()
        .with_presets(vec!["vivian".into(), "ryan".into()])
        .with_design(true)
        .with_instructions(true)
        .with_clone(CloneMode::AudioAndText)
        .with_script_speakers(2)
        .with_max_voices(2)
        .with_speed(true)
        .with_formats(vec![AudioFormat::Wav, AudioFormat::Pcm16])
        .with_sample_rate(RATE)
        .with_max_input_chars(MAX_CHARS)
}

/// A pool of one node (`tts`) with the mock speaker and transcriber, and what the speaker saw.
pub fn pool() -> (Pool, Seen) {
    let seen = Seen::default();
    let host: Arc<dyn InferHost> = Arc::new(Host { speaker: Arc::new(Speaker { profile: profile(), seen: seen.clone() }) });
    let feed = Arc::new(CatalogFeed::new(Arc::new(AllowAll)));
    feed.publish(vec![
        CatalogModel::new("mock-tts", Purpose::Speak, ModelState::Warm).with_speech(profile()),
        CatalogModel::new("mock-stt", Purpose::Transcribe, ModelState::Warm),
    ]);
    let admission = ymir_core::host::admission(2, 4);
    let dialer = MemoryDialer::new();
    dialer.route("tts", move |ch, protocol| {
        let (host, feed, admission) = (host.clone(), feed.clone(), admission.clone());
        tokio::spawn(async move {
            let caller = Caller::new("tts");
            let _ = match protocol.as_str() {
                v2::PROTOCOL => v2::serve(ch, caller, host.as_ref(), &admission).await,
                infer::PROTOCOL => infer::serve(ch, &caller, host.as_ref(), &admission).await,
                catalog::PROTOCOL => catalog::serve(ch, caller, feed.as_ref()).await,
                other => Err(RemoteError::Unsupported { message: format!("{other} is not served here") }),
            };
        });
    });
    (Pool::new(Arc::new(dialer), vec!["tts".to_string()]), seen)
}
