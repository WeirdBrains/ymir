//! `ymirctl transcribe` and the MCP `transcribe` tool (audio design milestones 1, 3 and 4):
//! speech to text through ymird's control socket, which asks the pool like the bridge does.
//!
//! The clip is read here (a file the caller named, or stdin), normalized to 16 kHz mono WAV (WAV
//! in Rust, anything else through the hardened ffmpeg runner), and, with `--chunk auto`, cut at
//! silences into pieces no longer than the transcriber's reported `audio_max_seconds`, whose
//! transcripts are re-aligned onto the clip's timeline. The media work is `weirdnet-media`'s
//! (`audio::normalize`, `transcribe::plan`, `prompt_from`, `realign`, `subtitles`). Nothing here
//! logs audio or text.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use weirdnet_media::ffmpeg::{Exec, Input, Runner};
use weirdnet_media::transcribe::{Piece, PieceTranscript, Plan, Segment, TranscribeLimits, Transcript, prompt_from, realign};
use weirdnet_media::{audio, pcm, probe, subtitles};
use ymir_core::control::{self, MAX_TRANSCRIBE_BYTES, TranscribeJob};

/// The largest file read for transcription (before normalizing).
pub const MAX_INPUT_BYTES: usize = 512 << 20;
/// The longest clip normalized (4 hours: 16 kHz mono PCM16 is 460 MB of it).
pub const MAX_AUDIO_MS: u64 = 4 * 3600 * 1000;

/// What runs ffmpeg for non-WAV input: the hardened runner, or a fake in tests.
pub type Ffmpeg = Arc<dyn Exec + Send + Sync>;

#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutFormat {
    #[default]
    Text,
    Json,
    Srt,
    Vtt,
}

impl OutFormat {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "text" => Some(Self::Text),
            "json" => Some(Self::Json),
            "srt" => Some(Self::Srt),
            "vtt" => Some(Self::Vtt),
            _ => None,
        }
    }
}

#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Chunking {
    /// One request with the whole clip.
    #[default]
    Off,
    /// Cut at silences to the transcriber's reported `audio_max_seconds` (refused when it reports none).
    Auto,
}

#[derive(clap::Args, Debug, Clone)]
pub struct TranscribeArgs {
    /// An audio file (wav, mp3, flac, ogg) or a video file (mp4, mov, mkv, webm: its first audio
    /// stream); `-` reads stdin.
    pub file: String,
    /// A speech-to-text model (default: the pool's pick, catalog purpose `transcribe`).
    #[arg(long, short)]
    pub model: Option<String>,
    /// The spoken language (ISO 639-1, such as `en`); otherwise the model detects it.
    #[arg(long, alias = "lang")]
    pub language: Option<String>,
    /// Names and terms to guide the transcript.
    #[arg(long)]
    pub prompt: Option<String>,
    #[arg(long, value_enum, default_value_t)]
    pub format: OutFormat,
    /// Ask for timed segments (always on for srt and vtt).
    #[arg(long)]
    pub timestamps: bool,
    #[arg(long, value_enum, default_value_t)]
    pub chunk: Chunking,
    /// Send the file as it is (no 16 kHz mono WAV conversion). Not with `--chunk auto`.
    #[arg(long)]
    pub no_normalize: bool,
    /// Write the result here instead of stdout.
    #[arg(long, short)]
    pub out: Option<PathBuf>,
}

/// What to do with one clip.
#[derive(Debug, Clone, Default)]
pub struct Options {
    pub model: Option<String>,
    pub language: Option<String>,
    pub prompt: Option<String>,
    pub timestamps: bool,
    pub chunk: Chunking,
    pub normalize: bool,
}

/// The transcript of a whole clip.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    pub transcript: Transcript,
    /// The clip's length (0 when unknown: a file sent as it is that isn't WAV).
    pub duration_ms: u64,
    pub language: Option<String>,
    pub model: String,
    /// The nodes that served it, in order, without repeats.
    pub nodes: Vec<String>,
    pub chunks: usize,
}

fn system_ffmpeg() -> Result<Ffmpeg> {
    Ok(Arc::new(Runner::locate(None)?))
}

/// The clip as 16 kHz mono PCM: WAV in Rust, anything else through ffmpeg (on a blocking thread).
async fn normalize(bytes: Vec<u8>, ffmpeg: Option<Ffmpeg>) -> Result<pcm::Pcm> {
    if probe::looks_like_wav(&bytes) {
        let pcm = pcm::decode_wav(&bytes)?;
        if pcm.duration_ms() > MAX_AUDIO_MS {
            bail!("the audio is {} ms, over the {MAX_AUDIO_MS} ms ymirctl reads", pcm.duration_ms());
        }
        return Ok(pcm.to_transcribe());
    }
    if sniff(&bytes).is_none() {
        bail!("not an audio or video file ymirctl reads (wav, mp3, flac, ogg, mp4/m4a/mov, mkv/webm)");
    }
    let ffmpeg = match ffmpeg {
        Some(r) => r,
        None => system_ffmpeg().context("only WAV is read without ffmpeg; install ffmpeg (with ffprobe) or pass --no-normalize")?,
    };
    let pcm = tokio::task::spawn_blocking(move || audio::normalize_pcm(&*ffmpeg, &Input::bytes(bytes), MAX_AUDIO_MS)).await??;
    Ok(pcm)
}

/// The container, from its first bytes (`video` for the ones a transcriber can't take as is).
fn sniff(b: &[u8]) -> Option<&'static str> {
    if probe::looks_like_wav(b) {
        Some("wav")
    } else if b.starts_with(b"fLaC") {
        Some("flac")
    } else if b.starts_with(b"OggS") {
        Some("ogg")
    } else if b.starts_with(b"ID3") || (b.len() > 1 && b[0] == 0xFF && b[1] & 0xE0 == 0xE0) {
        Some("mp3")
    } else if (b.len() >= 8 && &b[4..8] == b"ftyp") || b.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) {
        Some("video")
    } else {
        None
    }
}

/// The format name for a clip sent as it is.
fn format_of(bytes: &[u8]) -> Result<&'static str> {
    match sniff(bytes) {
        Some("video") => bail!("a video container can't be sent as it is; leave out --no-normalize so its audio is extracted"),
        Some(f) => Ok(f),
        None => bail!("not an audio file ymirctl reads (wav, mp3, flac, ogg)"),
    }
}

fn piece_transcript(t: weirdnet_agent::remote::infer::v2::Transcript) -> PieceTranscript {
    PieceTranscript::new(t.text, t.segments.into_iter().map(|s| Segment::new(s.start_ms, s.end_ms, s.text)).collect())
}

fn dedup_push(nodes: &mut Vec<String>, node: String) {
    if nodes.last() != Some(&node) {
        nodes.push(node);
    }
}

/// Transcribes `bytes` through the ymird on `socket`. `progress` gets one line per chunk.
pub async fn transcribe(
    socket: &Path,
    opts: &Options,
    bytes: Vec<u8>,
    ffmpeg: Option<Ffmpeg>,
    progress: &mut (dyn FnMut(String) + Send),
) -> Result<Outcome> {
    if bytes.is_empty() {
        bail!("no audio");
    }
    if bytes.len() > MAX_INPUT_BYTES {
        bail!("the file is {} bytes; ymirctl reads at most {MAX_INPUT_BYTES}", bytes.len());
    }
    let job = |format: &str, model: Option<String>, prompt: Option<String>| TranscribeJob {
        format: format.into(),
        model,
        language: opts.language.clone(),
        prompt,
        timestamps: opts.timestamps,
        ..Default::default()
    };
    if opts.chunk == Chunking::Off {
        let (audio, format) = if opts.normalize {
            (pcm::encode_wav(&normalize(bytes, ffmpeg).await?), "wav")
        } else {
            let f = format_of(&bytes)?;
            (bytes, f)
        };
        if audio.len() as u64 > MAX_TRANSCRIBE_BYTES {
            bail!("the audio is {} bytes, over the {MAX_TRANSCRIBE_BYTES} bytes one request carries; use --chunk auto", audio.len());
        }
        let wav_ms = probe::wav_header(&audio, None).ok().map(|i| i.duration_ms());
        let r = control::transcribe(socket, job(format, opts.model.clone(), opts.prompt.clone()), &audio).await?;
        let (language, end) = (r.transcript.language.clone(), r.transcript.duration_ms.or(wav_ms).unwrap_or(0));
        let whole = Plan { duration_ms: end, pieces: vec![Piece { index: 0, start_ms: 0, end_ms: end }] };
        let transcript = realign(&whole, &[piece_transcript(r.transcript)])?;
        return Ok(Outcome { transcript, duration_ms: end, language, model: r.model, nodes: vec![r.node], chunks: 1 });
    }
    if !opts.normalize {
        bail!("--chunk auto decodes the audio to cut it; leave out --no-normalize");
    }
    let target = control::transcribe_target(socket, opts.model.as_deref()).await?;
    let Some(max_s) = target.audio_max_seconds else {
        bail!(
            "transcriber `{}` reports no audio_max_seconds, so the clip can't be cut to fit; set it on the model's node \
             (`limits = {{ audio_max_seconds = N }}` on its [[model.local]]) or send it whole with --chunk off",
            target.model
        );
    };
    let pcm = normalize(bytes, ffmpeg).await?;
    // One request also has to fit the control socket's frame.
    let max_request = target.max_request_bytes.map_or(MAX_TRANSCRIBE_BYTES, |m| m.min(MAX_TRANSCRIBE_BYTES));
    let plan = weirdnet_media::transcribe::plan(&pcm, &TranscribeLimits::new(Some(max_s), Some(max_request)))?;
    let n = plan.pieces.len();
    let mut results: Vec<PieceTranscript> = Vec::with_capacity(n);
    let mut nodes = Vec::new();
    let mut language = None;
    for (i, c) in plan.pieces.iter().enumerate() {
        progress(format!("chunk {} of {n} ({} to {} ms)", i + 1, c.start_ms, c.end_ms));
        let piece_wav = plan.piece_wav(&pcm, i).context("a planned piece is outside the clip")?;
        let tail = results.last().map(|p| prompt_from(&p.text)).filter(|t| !t.is_empty());
        let prompt = match (opts.prompt.as_deref(), tail) {
            (Some(p), Some(tail)) => Some(format!("{p} {tail}")),
            (None, tail) => tail,
            (Some(p), None) => Some(p.to_string()),
        };
        let r = control::transcribe(socket, job("wav", Some(target.model.clone()), prompt), &piece_wav)
            .await
            .with_context(|| format!("chunk {} of {n}", i + 1))?;
        language = language.or(r.transcript.language.clone());
        dedup_push(&mut nodes, r.node);
        results.push(piece_transcript(r.transcript));
    }
    let transcript = realign(&plan, &results)?;
    Ok(Outcome { transcript, duration_ms: plan.duration_ms, language, model: target.model, nodes, chunks: n })
}

/// The outcome as text, JSON, SRT or VTT.
pub fn render(o: &Outcome, format: OutFormat) -> String {
    match format {
        OutFormat::Text => format!("{}\n", o.transcript.text),
        OutFormat::Json => {
            let v = json!({
                "text": o.transcript.text,
                "language": o.language,
                "duration_ms": o.duration_ms,
                "segments": o.transcript.segments,
                "model": o.model,
                "nodes": o.nodes,
                "chunks": o.chunks,
            });
            format!("{}\n", serde_json::to_string_pretty(&v).unwrap_or_default())
        }
        OutFormat::Srt => subtitles::srt(&o.transcript.segments),
        OutFormat::Vtt => subtitles::vtt(&o.transcript.segments),
    }
}

/// The JSON the MCP tool returns.
pub fn to_json(o: &Outcome, format: OutFormat) -> Value {
    let mut v = json!({
        "text": o.transcript.text,
        "language": o.language,
        "duration_ms": o.duration_ms,
        "model": o.model,
        "nodes": o.nodes,
        "chunks": o.chunks,
        "coarse": o.transcript.coarse,
    });
    match format {
        OutFormat::Srt | OutFormat::Vtt => v["subtitles"] = render(o, format).into(),
        OutFormat::Json => v["segments"] = json!(o.transcript.segments),
        OutFormat::Text => {}
    }
    v
}

/// `ymirctl transcribe`.
pub async fn run(socket: &Path, args: TranscribeArgs, out: &mut dyn Write, input: &mut dyn std::io::BufRead) -> Result<()> {
    let bytes = if args.file == "-" {
        let mut b = Vec::new();
        std::io::Read::take(input, MAX_INPUT_BYTES as u64 + 1).read_to_end(&mut b)?;
        b
    } else {
        read_capped(Path::new(&args.file))?
    };
    let subtitles = matches!(args.format, OutFormat::Srt | OutFormat::Vtt);
    let opts = Options {
        model: args.model,
        language: args.language,
        prompt: args.prompt,
        timestamps: args.timestamps || subtitles || args.format == OutFormat::Json,
        chunk: args.chunk,
        normalize: !args.no_normalize,
    };
    let o = transcribe(socket, &opts, bytes, None, &mut |line| eprintln!("{line}")).await?;
    if subtitles && o.transcript.coarse {
        eprintln!("note: {} gave no timing; cues span whole chunks", o.model);
    }
    let text = render(&o, args.format);
    match args.out {
        Some(p) => std::fs::write(&p, text).with_context(|| format!("writing {}", p.display()))?,
        None => out.write_all(text.as_bytes())?,
    }
    Ok(())
}

/// Reads at most [`MAX_INPUT_BYTES`] of a regular file.
pub fn read_capped(path: &Path) -> Result<Vec<u8>> {
    let f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let meta = f.metadata()?;
    if !meta.is_file() {
        bail!("{} is not a regular file", path.display());
    }
    if meta.len() > MAX_INPUT_BYTES as u64 {
        bail!("{} is {} bytes; ymirctl reads at most {MAX_INPUT_BYTES}", path.display(), meta.len());
    }
    let mut b = Vec::with_capacity(meta.len() as usize);
    f.take(MAX_INPUT_BYTES as u64 + 1).read_to_end(&mut b)?;
    Ok(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_formats() {
        let o = Outcome {
            transcript: Transcript {
                text: "hi there".into(),
                segments: vec![Segment::new(0, 900, "hi"), Segment::new(900, 2000, "there")],
                coarse: false,
            },
            duration_ms: 2000,
            language: Some("en".into()),
            model: "whisper".into(),
            nodes: vec!["a".into()],
            chunks: 1,
        };
        assert_eq!(render(&o, OutFormat::Text), "hi there\n");
        assert!(render(&o, OutFormat::Srt).starts_with("1\n00:00:00,000 --> 00:00:00,900\nhi\n"));
        assert!(render(&o, OutFormat::Vtt).starts_with("WEBVTT\n\n00:00:00.000 --> 00:00:00.900\nhi\n"));
        let j: Value = serde_json::from_str(&render(&o, OutFormat::Json)).unwrap();
        assert_eq!(j["segments"][1]["text"], "there");
        assert_eq!(j["language"], "en");
        assert_eq!(to_json(&o, OutFormat::Vtt)["subtitles"].as_str().unwrap().lines().next(), Some("WEBVTT"));
        assert_eq!(to_json(&o, OutFormat::Text)["coarse"], false);
    }

    #[test]
    fn format_of_refuses_what_it_cannot_send() {
        assert_eq!(format_of(b"fLaC....").unwrap(), "flac");
        assert!(format_of(b"\0\0\0\x18ftypisom").is_err());
        assert!(format_of(b"text").is_err());
    }

    #[tokio::test]
    async fn non_wav_goes_through_the_runner_and_wav_does_not() {
        use weirdnet_media::ffmpeg::{FakeRunner, Tool};
        let wav16 = pcm::encode_wav(&pcm::tone(16_000, 440.0, 500, 0.3));
        let fake = Arc::new(FakeRunner::new().on(|j| j.tool == Tool::Ffmpeg, move |_| Ok(wav16.clone())));
        let mp3 = b"ID3\x04\0\0\0\0\0\0rest of an mp3".to_vec();
        let p = normalize(mp3, Some(fake.clone())).await.unwrap();
        assert_eq!((p.sample_rate, p.channels, p.duration_ms()), (16_000, 1, 500));
        let calls = fake.calls();
        assert_eq!(calls.len(), 1);
        assert!(calls[0].has_arg("-vn") && calls[0].has_arg("0:a:0"), "the first audio stream only: {:?}", calls[0].args);
        // WAV is decoded and resampled in Rust: 44.1 kHz stereo in, 16 kHz mono out, no job.
        let stereo = pcm::encode_wav(&pcm::Pcm::new(44_100, 2, vec![0; 44_100 * 2]));
        let p = normalize(stereo, Some(fake.clone())).await.unwrap();
        assert_eq!((p.sample_rate, p.channels, p.duration_ms()), (16_000, 1, 1_000));
        assert_eq!(fake.calls().len(), 1);
        assert!(normalize(b"not audio".to_vec(), Some(fake.clone())).await.is_err());
    }

    /// With the system's ffmpeg when there is one (skipped otherwise): an MP3 made by ffmpeg from a
    /// WAV comes back as 16 kHz mono of about the same length.
    #[tokio::test]
    async fn real_ffmpeg_normalizes_an_mp3() {
        let Ok(runner) = Runner::locate(None) else { return };
        let wav = pcm::encode_wav(&pcm::tone(44_100, 440.0, 1_000, 0.3));
        let Ok(mp3) = audio::encode(&runner, &pcm::decode_wav(&wav).unwrap(), audio::AudioFormat::Mp3, 1 << 20) else { return };
        let p = normalize(mp3, Some(Arc::new(runner))).await.unwrap();
        assert_eq!((p.sample_rate, p.channels), (16_000, 1));
        assert!((900..=1_200).contains(&p.duration_ms()), "{}", p.duration_ms());
    }
}
