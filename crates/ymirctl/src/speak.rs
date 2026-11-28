//! `ymirctl speak` and the MCP `ymir_speak` tool: text to speech through ymird's control socket,
//! which asks the pool (`Pool::speak`, infer/2 `Speak`) like the bridge does.
//!
//! Text comes from the command line, a file, stdin, or a multi-speaker script. With `--chunk auto`
//! (the default) text longer than the speaker's reported `max_input_chars` is split by
//! `weirdnet-media`'s [`TextPlan`] (paragraphs, sentences, clauses; script turns kept whole and
//! never mixed), each piece is spoken by the same model with the same voice and seed, and the
//! pieces are stitched with a short crossfade inside a paragraph and a pause between paragraphs
//! and turns ([`pcm::concat`]). The result is encoded in the asked format: WAV and raw PCM in Rust,
//! MP3, Opus and FLAC through the hardened ffmpeg runner ([`audio::encode`]). A model that makes
//! the asked format itself is asked for it directly when one request is enough. Voices are never
//! faked: a preset, designed or cloned voice the model lacks is the pool's 400, which names it.
//! Nothing here logs text, reference clips or audio.

use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use weirdnet_agent::remote::infer::v2::{MAX_REFERENCE_BYTES, ScriptTurn, SpeakInput, SpeakRequest, Usage, Voice};
use weirdnet_agent::speech::{AudioFormat, SpeechProfile};
use weirdnet_media::ffmpeg::Runner;
use weirdnet_media::pcm::{self, Pcm};
use weirdnet_media::speak::{self as plan, TextPlan};
use weirdnet_media::{audio, probe};
use ymir_core::control::{self, SpeechReply};

use crate::transcribe::Ffmpeg;

/// The most text read (from a file or stdin).
pub const MAX_TEXT_BYTES: u64 = 4 << 20;

#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Chunking {
    /// Split text over the speaker's `max_input_chars` into pieces and stitch them.
    #[default]
    Auto,
    /// One request with all the text (the model's own limit applies).
    Off,
}

#[derive(clap::Args, Debug, Clone, Default)]
pub struct SpeakArgs {
    /// The text to speak; `-` or nothing reads stdin (unless --file or --script).
    pub text: Option<String>,
    /// Read the text from this file (paragraphs are separated by blank lines).
    #[arg(long, conflicts_with_all = ["text", "script"])]
    pub file: Option<PathBuf>,
    /// A multi-speaker script: JSON `[{"speaker":0,"text":"..."}]` or lines `S0: ...` (`-`: stdin).
    #[arg(long, conflicts_with = "text")]
    pub script: Option<PathBuf>,
    /// A preset voice (`vivian`), or one speaker's (`1=ryan`). Repeat for more speakers.
    #[arg(long)]
    pub voice: Vec<String>,
    /// Design a voice from a description ("a warm, low voice"), when the model designs voices.
    #[arg(long, conflicts_with = "clone_audio")]
    pub design: Option<String>,
    /// Clone the voice in this reference clip (WAV, MP3, FLAC, OGG; at most 30 MB), when the model clones.
    #[arg(long, alias = "ref-audio")]
    pub clone_audio: Option<PathBuf>,
    /// What the reference clip says (some models need it, some use it, some ignore it).
    #[arg(long, alias = "ref-text", requires = "clone_audio")]
    pub clone_text: Option<String>,
    /// Confirm the cloned voice is yours or you have permission to use it (required to clone).
    #[arg(long, alias = "ref-consent", requires = "clone_audio")]
    pub clone_consent: bool,
    /// Style or emotion for the voice, when the model takes instructions.
    #[arg(long)]
    pub instructions: Option<String>,
    #[arg(long, alias = "lang")]
    pub language: Option<String>,
    /// Speaking speed (1.0: the model's pace), when the model takes one.
    #[arg(long)]
    pub speed: Option<f32>,
    #[arg(long)]
    pub seed: Option<u64>,
    /// A speaking model (default: the pool's pick, catalog purpose `speak`).
    #[arg(long, short)]
    pub model: Option<String>,
    #[arg(long, value_enum, default_value_t)]
    pub format: OutFormat,
    /// Write the audio here (`-`: stdout, even a terminal). Default: stdout when it is not a terminal.
    #[arg(long, short)]
    pub out: Option<String>,
    #[arg(long, value_enum, default_value_t)]
    pub chunk: Chunking,
    /// The pause between paragraphs and speaker turns, in milliseconds.
    #[arg(long, default_value_t = plan::DEFAULT_PAUSE_MS)]
    pub pause_ms: u64,
    /// The crossfade between pieces of one paragraph, in milliseconds.
    #[arg(long, default_value_t = plan::DEFAULT_CROSSFADE_MS)]
    pub crossfade_ms: u64,
}

#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutFormat {
    #[default]
    Wav,
    Mp3,
    Opus,
    Flac,
    /// Raw 16-bit little-endian samples (the rate is printed on stderr).
    Pcm16,
}

impl OutFormat {
    pub fn wire(self) -> AudioFormat {
        match self {
            OutFormat::Wav => AudioFormat::Wav,
            OutFormat::Mp3 => AudioFormat::Mp3,
            OutFormat::Opus => AudioFormat::Opus,
            OutFormat::Flac => AudioFormat::Flac,
            OutFormat::Pcm16 => AudioFormat::Pcm16,
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match AudioFormat::parse(s)? {
            AudioFormat::Wav => OutFormat::Wav,
            AudioFormat::Mp3 => OutFormat::Mp3,
            AudioFormat::Opus => OutFormat::Opus,
            AudioFormat::Flac => OutFormat::Flac,
            _ => OutFormat::Pcm16,
        })
    }
}

/// What to say.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Text {
    Plain(String),
    Script(Vec<(u8, String)>),
}

/// One speech job.
#[derive(Debug, Clone, Default)]
pub struct Options {
    pub text: Option<Text>,
    pub model: Option<String>,
    /// The voice (for a script: of speakers without their own).
    pub voice: Option<Voice>,
    pub voices: Vec<(u8, Voice)>,
    pub instructions: Option<String>,
    pub language: Option<String>,
    /// Per mille of the model's pace.
    pub speed: Option<u32>,
    pub seed: Option<u64>,
    pub format: OutFormat,
    pub chunk: Chunking,
    pub pause_ms: u64,
    pub crossfade_ms: u64,
}

/// The speech for a whole job.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    pub audio: Vec<u8>,
    pub format: AudioFormat,
    pub sample_rate: Option<u32>,
    pub channels: Option<u16>,
    /// Known for WAV and PCM, and for anything stitched or re-encoded here.
    pub duration_ms: Option<u64>,
    pub model: String,
    /// The nodes that served it, in order, without repeats.
    pub nodes: Vec<String>,
    pub pieces: usize,
    /// Summed over the pieces.
    pub usage: Usage,
    pub ignored: Vec<String>,
}

fn system_ffmpeg() -> Result<Ffmpeg> {
    Ok(Arc::new(Runner::locate(None)?))
}

/// A script file: a JSON array of `{"speaker": N, "text": "..."}`, or lines `S0: text` (also
/// `0: text`); blank lines are skipped and a line without a prefix continues the previous turn.
pub fn parse_script(s: &str) -> Result<Vec<(u8, String)>> {
    let t = s.trim_start();
    if t.starts_with('[') {
        #[derive(serde::Deserialize)]
        struct Turn {
            speaker: u8,
            text: String,
        }
        let turns: Vec<Turn> = serde_json::from_str(t).context("the script is not a JSON array of {\"speaker\": N, \"text\": \"...\"}")?;
        return Ok(turns.into_iter().map(|t| (t.speaker, t.text)).collect());
    }
    let mut out: Vec<(u8, String)> = Vec::new();
    for (n, line) in s.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let head = line.split_once(':').and_then(|(h, rest)| {
            let h = h.trim();
            let num = h.strip_prefix('S').or_else(|| h.strip_prefix('s')).unwrap_or(h);
            num.parse::<u8>().ok().map(|sp| (sp, rest.trim()))
        });
        match (head, out.last_mut()) {
            (Some((sp, rest)), _) => out.push((sp, rest.to_string())),
            (None, Some(last)) => {
                last.1.push(' ');
                last.1.push_str(line);
            }
            (None, None) => bail!("script line {}: start it with a speaker, such as `S0: ...`", n + 1),
        }
    }
    if out.is_empty() {
        bail!("the script has no turns");
    }
    Ok(out)
}

/// `--voice` values: `NAME` for the voice, `N=NAME` for speaker N's.
pub fn parse_voices(values: &[String]) -> Result<Voices> {
    let mut voice = None;
    let mut voices: Vec<(u8, Voice)> = Vec::new();
    for v in values {
        match v.split_once('=') {
            Some((n, name)) => {
                let n: u8 =
                    n.trim().trim_start_matches(['S', 's']).parse().with_context(|| format!("--voice {v}: the speaker is a number (1=ryan)"))?;
                if voices.iter().any(|(s, _)| *s == n) {
                    bail!("--voice gives speaker {n} two voices");
                }
                voices.push((n, Voice::preset(name.trim())));
            }
            None if voice.is_some() => bail!("--voice is given twice without a speaker; use N=NAME for a script's speakers"),
            None => voice = Some(Voice::preset(v.trim())),
        }
    }
    Ok((voice, voices))
}

/// The format of a reference clip, from its first bytes.
pub fn clip_format(b: &[u8]) -> Result<&'static str> {
    if probe::looks_like_wav(b) {
        Ok("wav")
    } else if b.starts_with(b"fLaC") {
        Ok("flac")
    } else if b.starts_with(b"OggS") {
        Ok("ogg")
    } else if b.starts_with(b"ID3") || (b.len() > 1 && b[0] == 0xFF && b[1] & 0xE0 == 0xE0) {
        Ok("mp3")
    } else {
        bail!("the reference clip is not WAV, MP3, FLAC or OGG")
    }
}

/// A cloned voice from `clip`, refused without consent here as the pool would.
pub fn clone_voice(clip: Vec<u8>, text: Option<String>, consent: bool) -> Result<Voice> {
    if !consent {
        bail!("cloning a voice needs consent: confirm the voice is yours or you have permission to use it (--clone-consent)");
    }
    if clip.is_empty() || clip.len() > MAX_REFERENCE_BYTES {
        bail!("a reference clip takes 1 to {MAX_REFERENCE_BYTES} bytes, not {}", clip.len());
    }
    let format = clip_format(&clip)?;
    Ok(Voice::clone_from(clip, format, text.filter(|t| !t.trim().is_empty()), true))
}

impl Options {
    fn request(&self, input: SpeakInput, speakers: Option<&[u8]>) -> SpeakRequest {
        let mut r = SpeakRequest::with_input(input);
        if let Some(m) = &self.model {
            r = r.with_model(m.clone());
        }
        if let Some(v) = &self.voice {
            r = r.with_voice(v.clone());
        }
        for (s, v) in &self.voices {
            if speakers.is_none_or(|only| only.contains(s)) {
                r = r.with_speaker_voice(*s, v.clone());
            }
        }
        if let Some(i) = self.instructions.as_ref().filter(|i| !i.trim().is_empty()) {
            r = r.with_instructions(i.clone());
        }
        if let Some(l) = self.language.as_ref().filter(|l| !l.trim().is_empty()) {
            r = r.with_language(l.clone());
        }
        if let Some(s) = self.speed {
            r = r.with_speed(s);
        }
        if let Some(s) = self.seed {
            r = r.with_seed(s);
        }
        r
    }

    fn whole(&self) -> Result<SpeakRequest> {
        Ok(match self.text.as_ref().context("nothing to speak")? {
            Text::Plain(t) => self.request(SpeakInput::Text(t.clone()), None),
            Text::Script(turns) => self.request(SpeakInput::Script(turns.iter().map(|(s, t)| ScriptTurn::new(*s, t.clone())).collect()), None),
        })
    }
}

fn dedup_push(nodes: &mut Vec<String>, node: &str) {
    if nodes.last().map(String::as_str) != Some(node) {
        nodes.push(node.to_string());
    }
}

fn add_usage(total: &mut Usage, u: Option<Usage>) {
    if let Some(u) = u {
        total.input_chars += u.input_chars;
        total.audio_ms += u.audio_ms;
        total.prompt_tokens += u.prompt_tokens;
        total.completion_tokens += u.completion_tokens;
    }
}

/// The samples of one spoken piece (WAV, or raw PCM16 at the reply's rate).
fn decode_piece(reply: &SpeechReply, audio: &[u8]) -> Result<Pcm> {
    match reply.format {
        AudioFormat::Wav => Ok(pcm::decode_wav(audio)?),
        AudioFormat::Pcm16 => {
            let rate = reply.sample_rate.context("the speaker sent raw PCM without its sample rate")?;
            let samples = audio.as_chunks::<2>().0.iter().map(|b| i16::from_le_bytes(*b)).collect();
            Ok(Pcm::new(rate, reply.channels.unwrap_or(1), samples))
        }
        other => bail!("the speaker sent {other}, which can't be stitched (asked for wav)"),
    }
}

/// `pcm` in `format`: WAV and PCM here, the rest through ffmpeg (on a blocking thread).
async fn encode(pcm: Pcm, format: AudioFormat, ffmpeg: Option<Ffmpeg>) -> Result<Vec<u8>> {
    match format {
        AudioFormat::Wav => Ok(pcm::encode_wav(&pcm)),
        AudioFormat::Pcm16 => Ok(pcm.samples.iter().flat_map(|s| s.to_le_bytes()).collect()),
        f => {
            let to = match f {
                AudioFormat::Mp3 => audio::AudioFormat::Mp3,
                AudioFormat::Opus => audio::AudioFormat::Opus,
                _ => audio::AudioFormat::Flac,
            };
            let ffmpeg = match ffmpeg {
                Some(r) => r,
                None => system_ffmpeg().with_context(|| format!("encoding {f} needs ffmpeg (with ffprobe); install it or ask for wav"))?,
            };
            Ok(tokio::task::spawn_blocking(move || audio::encode(&*ffmpeg, &pcm, to, 1 << 30)).await??)
        }
    }
}

/// What to ask one request for, given the profile: the wanted format when the model makes it
/// (or doesn't say), else WAV to convert here.
fn ask_format(want: AudioFormat, profile: Option<&SpeechProfile>) -> Result<AudioFormat> {
    let formats = profile.map(|p| p.formats.clone()).unwrap_or_default();
    if formats.is_empty() || formats.contains(&want) {
        return Ok(want);
    }
    for f in [AudioFormat::Wav, AudioFormat::Pcm16] {
        if formats.contains(&f) {
            return Ok(f);
        }
    }
    bail!("the speaker makes {} only, none of which can be converted here", formats.iter().map(|f| f.as_str()).collect::<Vec<_>>().join(", "))
}

/// A default voice and per-speaker voices.
pub type Voices = (Option<Voice>, Vec<(u8, Voice)>);

/// Speaks the job through the ymird on `socket`. `progress` gets one line per piece.
pub async fn speak(socket: &Path, opts: &Options, ffmpeg: Option<Ffmpeg>, progress: &mut (dyn FnMut(String) + Send)) -> Result<Outcome> {
    let whole = opts.whole()?;
    // The speaker it would go to, its profile (formats, max_input_chars) and the same model for
    // every piece.
    let target = control::speak_target(socket, &whole).await?;
    let profile = target.profile.as_ref();
    let want = opts.format.wire();
    let max_chars = profile.and_then(|p| p.max_input_chars).map(|m| m as usize);
    let plan = match (&opts.chunk, opts.text.as_ref()) {
        (Chunking::Off, _) => None,
        (Chunking::Auto, Some(Text::Plain(t))) => Some(TextPlan::text(t, max_chars)?),
        (Chunking::Auto, Some(Text::Script(turns))) => {
            Some(TextPlan::script(&turns.iter().map(|(s, t)| plan::ScriptTurn::new(*s, t.clone())).collect::<Vec<_>>(), max_chars)?)
        }
        (_, None) => bail!("nothing to speak"),
    };
    let pieces = match plan {
        Some(p) if p.pieces.len() > 1 => p,
        _ => {
            // One request.
            let ask = ask_format(want, profile)?;
            let req = whole.with_model(target.model.clone()).with_format(ask);
            progress("speaking in one request".into());
            let (reply, bytes) = control::speak(socket, req).await?;
            let mut usage = Usage::default();
            add_usage(&mut usage, reply.usage);
            let (audio, duration_ms, rate, channels) = if reply.format == want {
                let ms = match want {
                    AudioFormat::Wav => probe::wav_header(&bytes, None).ok().map(|i| i.duration_ms()),
                    AudioFormat::Pcm16 => {
                        reply.sample_rate.map(|r| bytes.len() as u64 / 2 * 1000 / u64::from(r.max(1)) / u64::from(reply.channels.unwrap_or(1).max(1)))
                    }
                    _ => (usage.audio_ms > 0).then_some(usage.audio_ms),
                };
                (bytes, ms, reply.sample_rate, reply.channels)
            } else {
                let p = decode_piece(&reply, &bytes)?;
                let (ms, rate, ch) = (p.duration_ms(), p.sample_rate, p.channels);
                (encode(p, want, ffmpeg).await?, Some(ms), Some(rate), Some(ch))
            };
            return Ok(Outcome {
                audio,
                format: want,
                sample_rate: rate,
                channels,
                duration_ms,
                model: reply.model,
                nodes: vec![reply.node],
                pieces: 1,
                usage,
                ignored: reply.ignored,
            });
        }
    };
    // Long text: piece by piece on one model, stitched here.
    let ask = ask_format(AudioFormat::Wav, profile)?;
    let n = pieces.pieces.len();
    let mut pcms = Vec::with_capacity(n);
    let mut nodes = Vec::new();
    let mut usage = Usage::default();
    let mut ignored: Vec<String> = Vec::new();
    for (i, piece) in pieces.pieces.iter().enumerate() {
        progress(format!("piece {} of {n} ({} characters)", i + 1, piece.chars));
        let req = match piece.speaker {
            Some(s) => opts.request(SpeakInput::Script(vec![ScriptTurn::new(s, piece.text.clone())]), Some(&[s])),
            None => opts.request(SpeakInput::Text(piece.text.clone()), None),
        }
        .with_model(target.model.clone())
        .with_format(ask);
        let (reply, bytes) = control::speak(socket, req).await.with_context(|| format!("piece {} of {n}", i + 1))?;
        dedup_push(&mut nodes, &reply.node);
        add_usage(&mut usage, reply.usage);
        for f in &reply.ignored {
            if !ignored.contains(f) {
                ignored.push(f.clone());
            }
        }
        pcms.push(decode_piece(&reply, &bytes)?);
    }
    let joined = pcm::concat(&pcms, &pieces.joins(opts.crossfade_ms, opts.pause_ms))?;
    let (ms, rate, ch) = (joined.duration_ms(), joined.sample_rate, joined.channels);
    Ok(Outcome {
        audio: encode(joined, want, ffmpeg).await?,
        format: want,
        sample_rate: Some(rate),
        channels: Some(ch),
        duration_ms: Some(ms),
        model: target.model,
        nodes,
        pieces: n,
        usage,
        ignored,
    })
}

/// What the MCP tool and `--format json` callers see, without the audio.
pub fn summary(o: &Outcome) -> Value {
    json!({
        "format": o.format.as_str(),
        "bytes": o.audio.len(),
        "duration_ms": o.duration_ms,
        "sample_rate": o.sample_rate,
        "channels": o.channels,
        "model": o.model,
        "nodes": o.nodes,
        "pieces": o.pieces,
        "usage": {"input_chars": o.usage.input_chars, "audio_ms": o.usage.audio_ms},
        "ignored": o.ignored,
    })
}

fn read_text(path: &Path, input: &mut dyn std::io::BufRead) -> Result<String> {
    let mut b = Vec::new();
    if path == Path::new("-") {
        std::io::Read::take(input, MAX_TEXT_BYTES + 1).read_to_end(&mut b)?;
    } else {
        std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?.take(MAX_TEXT_BYTES + 1).read_to_end(&mut b)?;
    }
    if b.len() as u64 > MAX_TEXT_BYTES {
        bail!("{} is over {MAX_TEXT_BYTES} bytes of text", path.display());
    }
    String::from_utf8(b).with_context(|| format!("{} is not UTF-8 text", path.display()))
}

/// `ymirctl speak`. `stdout_tty`: whether stdout is a terminal (raw audio is not written there
/// unless `--out -` asks).
pub async fn run(socket: &Path, args: SpeakArgs, out: &mut dyn Write, input: &mut dyn std::io::BufRead, stdout_tty: bool) -> Result<()> {
    if args.out.is_none() && stdout_tty {
        bail!("not writing audio to a terminal: pass --out FILE, or --out - (or a pipe) for stdout");
    }
    let text = match (&args.script, &args.file, args.text.as_deref()) {
        (Some(p), _, _) => Text::Script(parse_script(&read_text(p, input)?)?),
        (None, Some(p), _) => Text::Plain(read_text(p, input)?),
        (None, None, Some(t)) if t != "-" => Text::Plain(t.to_string()),
        (None, None, _) => Text::Plain(read_text(Path::new("-"), input)?),
    };
    let (mut voice, voices) = parse_voices(&args.voice)?;
    if let Some(d) = &args.design {
        if voice.is_some() {
            bail!("--design and a --voice without a speaker both set the voice; give one");
        }
        voice = Some(Voice::design(d.clone()));
    }
    if let Some(p) = &args.clone_audio {
        if voice.is_some() {
            bail!("--clone-audio and --voice/--design both set the voice; give one");
        }
        let clip = std::fs::File::open(p).with_context(|| format!("opening {}", p.display()))?;
        let mut b = Vec::new();
        clip.take(MAX_REFERENCE_BYTES as u64 + 1).read_to_end(&mut b)?;
        voice = Some(clone_voice(b, args.clone_text.clone(), args.clone_consent)?);
    }
    let speed = match args.speed {
        Some(s) if !(0.25..=4.0).contains(&s) => bail!("--speed {s}: 0.25 to 4"),
        s => s.map(|s| (s * 1000.0).round() as u32),
    };
    let opts = Options {
        text: Some(text),
        model: args.model,
        voice,
        voices,
        instructions: args.instructions,
        language: args.language,
        speed,
        seed: args.seed,
        format: args.format,
        chunk: args.chunk,
        pause_ms: args.pause_ms,
        crossfade_ms: args.crossfade_ms,
    };
    let o = speak(socket, &opts, None, &mut |line| eprintln!("{line}")).await?;
    match args.out.as_deref() {
        None | Some("-") => {
            out.write_all(&o.audio)?;
            out.flush()?;
        }
        Some(p) => std::fs::write(p, &o.audio).with_context(|| format!("writing {p}"))?,
    }
    let secs = o.duration_ms.map(|ms| format!("{:.1} s", ms as f64 / 1000.0)).unwrap_or_else(|| "?".into());
    let rate = o.sample_rate.map(|r| format!(", {r} Hz")).unwrap_or_default();
    eprintln!(
        "{} of {}{rate} from {} on {} ({} piece{})",
        secs,
        o.format,
        o.model,
        o.nodes.join(", "),
        o.pieces,
        if o.pieces == 1 { "" } else { "s" }
    );
    if !o.ignored.is_empty() {
        eprintln!("note: {} did not use {}", o.model, o.ignored.join(", "));
    }
    Ok(())
}

/// Whether stdout is a terminal.
pub fn stdout_is_tty() -> bool {
    std::io::stdout().is_terminal()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scripts_parse_both_ways() {
        let j = parse_script(r#"[{"speaker":0,"text":"Hi."},{"speaker":1,"text":"Hello."}]"#).unwrap();
        assert_eq!(j, vec![(0, "Hi.".into()), (1, "Hello.".into())]);
        let l = parse_script("S0: Hi there.\nand more\n\n1: Hello.\ns2: Hey.").unwrap();
        assert_eq!(l, vec![(0, "Hi there. and more".into()), (1, "Hello.".into()), (2, "Hey.".into())]);
        assert!(parse_script("no speaker here").is_err());
        assert!(parse_script("  ").is_err());
        assert!(parse_script("[{\"speaker\": 300, \"text\": \"x\"}]").is_err());
    }

    #[test]
    fn voices_parse() {
        let (v, vs) = parse_voices(&["vivian".into(), "1=ryan".into(), "S2=aiden".into()]).unwrap();
        assert_eq!(v, Some(Voice::preset("vivian")));
        assert_eq!(vs, vec![(1, Voice::preset("ryan")), (2, Voice::preset("aiden"))]);
        assert!(parse_voices(&["a".into(), "b".into()]).is_err());
        assert!(parse_voices(&["1=a".into(), "1=b".into()]).is_err());
        assert!(parse_voices(&["x=a".into()]).is_err());
    }

    #[test]
    fn cloning_needs_consent_and_a_clip() {
        let wav = pcm::encode_wav(&pcm::tone(16_000, 220.0, 300, 0.3));
        assert!(clone_voice(wav.clone(), None, false).unwrap_err().to_string().contains("consent"));
        assert!(clone_voice(b"text".to_vec(), None, true).is_err());
        assert!(clone_voice(Vec::new(), None, true).is_err());
        match clone_voice(wav, Some("hello".into()), true).unwrap() {
            Voice::Clone { format, text, consent, .. } => assert_eq!((format.as_str(), text.as_deref(), consent), ("wav", Some("hello"), true)),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn formats_are_asked_or_converted() {
        let p = SpeechProfile::new().with_formats(vec![AudioFormat::Wav, AudioFormat::Pcm16]);
        assert_eq!(ask_format(AudioFormat::Mp3, Some(&p)).unwrap(), AudioFormat::Wav);
        assert_eq!(ask_format(AudioFormat::Pcm16, Some(&p)).unwrap(), AudioFormat::Pcm16);
        assert_eq!(ask_format(AudioFormat::Mp3, None).unwrap(), AudioFormat::Mp3, "unknown: ask");
        let mp3_only = SpeechProfile::new().with_formats(vec![AudioFormat::Mp3]);
        assert!(ask_format(AudioFormat::Flac, Some(&mp3_only)).is_err());
        assert_eq!(OutFormat::parse("pcm"), Some(OutFormat::Pcm16));
        assert_eq!(OutFormat::parse("aac"), None);
    }

    #[test]
    fn raw_pcm_pieces_decode_at_their_rate() {
        let reply = SpeechReply {
            bytes: 8,
            format: AudioFormat::Pcm16,
            sample_rate: Some(24_000),
            channels: None,
            model: "m".into(),
            node: "n".into(),
            usage: None,
            ignored: vec![],
        };
        let p = decode_piece(&reply, &[1, 0, 2, 0, 3, 0, 4, 0]).unwrap();
        assert_eq!((p.sample_rate, p.channels, p.samples), (24_000, 1, vec![1, 2, 3, 4]));
        let no_rate = SpeechReply { sample_rate: None, ..reply };
        assert!(decode_piece(&no_rate, &[0, 0]).is_err());
    }
}
