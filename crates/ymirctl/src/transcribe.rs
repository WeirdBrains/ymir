//! `ymirctl transcribe` and the MCP `transcribe` tool (audio design milestones 1, 3 and 4):
//! speech to text through ymird's control socket, which asks the pool like the bridge does.
//!
//! The clip is read here (a file the caller named, or stdin), normalized to 16 kHz mono WAV (WAV
//! in Rust, anything else through the hardened ffmpeg runner), and, with `--chunk auto`, cut at
//! silences into pieces no longer than the transcriber's reported `audio_max_seconds`, whose
//! transcripts are re-aligned onto the clip's timeline. Nothing here logs audio or text.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use ymir_core::control::{self, MAX_TRANSCRIBE_BYTES, TranscribeJob};
use ymir_media::ffmpeg::{Demuxer, FfmpegRunner, Job, Limits, Runner};
use ymir_media::transcript::{Joined, Piece, Segment, prompt_tail, realign};
use ymir_media::{pcm, subtitles, vad, wav};

/// The largest file read for transcription (before normalizing).
pub const MAX_INPUT_BYTES: usize = 512 << 20;

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
    pub joined: Joined,
    pub language: Option<String>,
    pub model: String,
    /// The nodes that served it, in order, without repeats.
    pub nodes: Vec<String>,
    pub chunks: usize,
}

fn system_ffmpeg() -> Result<Arc<dyn Runner>> {
    Ok(Arc::new(FfmpegRunner::from_path(Limits::default())?))
}

/// The clip as 16 kHz mono WAV: WAV in Rust, anything else through ffmpeg (on a blocking thread).
async fn normalize(bytes: Vec<u8>, runner: Option<Arc<dyn Runner>>) -> Result<Vec<u8>> {
    if wav::is_wav(&bytes) {
        return Ok(pcm::for_asr(&bytes)?);
    }
    if Demuxer::sniff(&bytes).is_none() {
        bail!("not an audio or video file ymirctl reads (wav, mp3, flac, ogg, mp4/m4a/mov, mkv/webm)");
    }
    let runner = match runner {
        Some(r) => r,
        None => system_ffmpeg().context("only WAV is read without ffmpeg; install ffmpeg or pass --no-normalize")?,
    };
    let out = tokio::task::spawn_blocking(move || runner.run(Job::DecodeForAsr, &bytes)).await??;
    Ok(out)
}

/// The format name for a clip sent as it is.
fn format_of(bytes: &[u8]) -> Result<&'static str> {
    match Demuxer::sniff(bytes) {
        Some(Demuxer::Wav) => Ok("wav"),
        Some(Demuxer::Mp3) => Ok("mp3"),
        Some(Demuxer::Flac) => Ok("flac"),
        Some(Demuxer::Ogg) => Ok("ogg"),
        Some(_) => bail!("a video container can't be sent as it is; leave out --no-normalize so its audio is extracted"),
        None => bail!("not an audio file ymirctl reads (wav, mp3, flac, ogg)"),
    }
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
    ffmpeg: Option<Arc<dyn Runner>>,
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
        let (audio, format) = if opts.normalize { (normalize(bytes, ffmpeg).await?, "wav") } else { (bytes.clone(), format_of(&bytes)?) };
        if audio.len() as u64 > MAX_TRANSCRIBE_BYTES {
            bail!("the audio is {} bytes, over the {MAX_TRANSCRIBE_BYTES} bytes one request carries; use --chunk auto", audio.len());
        }
        let duration_ms = wav::probe(&audio).ok().map(|i| i.duration_ms());
        let r = control::transcribe(socket, job(format, opts.model.clone(), opts.prompt.clone()), &audio).await?;
        let t = r.transcript;
        let end = t.duration_ms.or(duration_ms).unwrap_or(0);
        let piece = Piece { text: t.text.clone(), segments: t.segments.iter().map(|s| Segment::new(s.start_ms, s.end_ms, s.text.clone())).collect() };
        let chunk = vad::Chunk { start: 0, end: 0, start_ms: 0, end_ms: end };
        return Ok(Outcome { joined: realign(&[chunk], &[piece]), language: t.language, model: r.model, nodes: vec![r.node], chunks: 1 });
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
    let audio = normalize(bytes, ffmpeg).await?;
    let pcm = wav::decode(&audio)?;
    let plan = vad::plan(&pcm, u64::from(max_s) * 1000)?;
    let mut pieces = Vec::with_capacity(plan.len());
    let mut nodes = Vec::new();
    let mut language = None;
    for (i, c) in plan.iter().enumerate() {
        progress(format!("chunk {} of {} ({} to {} ms)", i + 1, plan.len(), c.start_ms, c.end_ms));
        let piece_wav = wav::encode_pcm16(&pcm.slice(c.start, c.end));
        let prompt = match (opts.prompt.as_deref(), pieces.last().map(|p: &Piece| prompt_tail(&p.text, 200))) {
            (Some(p), Some(tail)) if !tail.is_empty() => Some(format!("{p} {tail}")),
            (None, Some(tail)) if !tail.is_empty() => Some(tail),
            (p, _) => p.map(str::to_string),
        };
        let r = control::transcribe(socket, job("wav", Some(target.model.clone()), prompt), &piece_wav)
            .await
            .with_context(|| format!("chunk {} of {}", i + 1, plan.len()))?;
        language = language.or(r.transcript.language.clone());
        dedup_push(&mut nodes, r.node);
        pieces.push(Piece {
            text: r.transcript.text,
            segments: r.transcript.segments.into_iter().map(|s| Segment::new(s.start_ms, s.end_ms, s.text)).collect(),
        });
    }
    Ok(Outcome { joined: realign(&plan, &pieces), language, model: target.model, nodes, chunks: plan.len() })
}

/// The outcome as text, JSON, SRT or VTT.
pub fn render(o: &Outcome, format: OutFormat) -> String {
    match format {
        OutFormat::Text => format!("{}\n", o.joined.text),
        OutFormat::Json => {
            let v = json!({
                "text": o.joined.text,
                "language": o.language,
                "duration_ms": o.joined.duration_ms,
                "segments": o.joined.segments,
                "model": o.model,
                "nodes": o.nodes,
                "chunks": o.chunks,
            });
            format!("{}\n", serde_json::to_string_pretty(&v).unwrap_or_default())
        }
        OutFormat::Srt => subtitles::srt(&o.joined.segments),
        OutFormat::Vtt => subtitles::vtt(&o.joined.segments),
    }
}

/// The JSON the MCP tool returns.
pub fn to_json(o: &Outcome, format: OutFormat) -> Value {
    let mut v = json!({
        "text": o.joined.text,
        "language": o.language,
        "duration_ms": o.joined.duration_ms,
        "model": o.model,
        "nodes": o.nodes,
        "chunks": o.chunks,
        "coarse": o.joined.segments.iter().any(|s| s.coarse),
    });
    match format {
        OutFormat::Srt | OutFormat::Vtt => v["subtitles"] = render(o, format).into(),
        OutFormat::Json => v["segments"] = json!(o.joined.segments),
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
    if subtitles && o.joined.segments.iter().any(|s| s.coarse) {
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
            joined: Joined {
                text: "hi there".into(),
                segments: vec![Segment::new(0, 900, "hi"), Segment::new(900, 2000, "there")],
                duration_ms: 2000,
            },
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
}
