//! A hardened ffmpeg runner, and a fake for tests.
//!
//! The rules (`docs/design/audio.md`, "Hardened ffmpeg"): the caller hands over bytes it read
//! itself, never a path or URL; the bytes are sniffed and must be one of a short list of
//! containers, named to ffmpeg with `-f`; they go to a private temp file that ffmpeg reads with
//! `-protocol_whitelist file,pipe`, writing to `pipe:1` only; the process gets a wall-time limit, an
//! address-space limit, no file writes (`RLIMIT_FSIZE` 0), one thread, no stdin, and on Linux a
//! new network namespace when the kernel lets an unprivileged process have one; output past the
//! cap kills it. Nothing it reads or writes is logged; its stderr is kept only for the error,
//! truncated.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::{MediaError, Pcm, Result, pcm::ASR_RATE, wav};

/// The containers ffmpeg may read, by sniffed magic bytes. Nothing else reaches it: no playlists,
/// no `concat` lists, no image or subtitle formats, no raw streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Demuxer {
    Wav,
    Mp3,
    Flac,
    Ogg,
    /// MP4, M4A, MOV (the `mov` demuxer, with external data references off).
    Mp4,
    /// Matroska and WebM.
    Matroska,
}

impl Demuxer {
    /// The ffmpeg demuxer name for `-f`.
    pub fn ffmpeg_name(self) -> &'static str {
        match self {
            Demuxer::Wav => "wav",
            Demuxer::Mp3 => "mp3",
            Demuxer::Flac => "flac",
            Demuxer::Ogg => "ogg",
            Demuxer::Mp4 => "mov",
            Demuxer::Matroska => "matroska",
        }
    }

    /// What `bytes` look like, from their first bytes; `None` for anything not on the list.
    pub fn sniff(bytes: &[u8]) -> Option<Demuxer> {
        let at = |i: usize, m: &[u8]| bytes.get(i..i + m.len()) == Some(m);
        if wav::is_wav(bytes) {
            Some(Demuxer::Wav)
        } else if at(0, b"fLaC") {
            Some(Demuxer::Flac)
        } else if at(0, b"OggS") {
            Some(Demuxer::Ogg)
        } else if at(4, b"ftyp") {
            Some(Demuxer::Mp4)
        } else if at(0, &[0x1A, 0x45, 0xDF, 0xA3]) {
            Some(Demuxer::Matroska)
        } else if at(0, b"ID3") || (bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] & 0xE0 == 0xE0 && bytes[1] & 0x06 != 0) {
            Some(Demuxer::Mp3)
        } else {
            None
        }
    }
}

/// Encoded output formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    Mp3,
    Opus,
    Flac,
}

impl Encoding {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "mp3" => Some(Encoding::Mp3),
            "opus" => Some(Encoding::Opus),
            "flac" => Some(Encoding::Flac),
            _ => None,
        }
    }

    fn codec_and_muxer(self) -> (&'static str, &'static str) {
        match self {
            Encoding::Mp3 => ("libmp3lame", "mp3"),
            Encoding::Opus => ("libopus", "ogg"),
            Encoding::Flac => ("flac", "flac"),
        }
    }
}

/// What to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Job {
    /// The first audio stream (of an audio file or a video container) as 16 kHz mono 16-bit
    /// PCM WAV, ready for a speech-to-text model.
    DecodeForAsr,
    /// A WAV file encoded to another format.
    Encode(Encoding),
}

/// Bounds for one call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    pub timeout: Duration,
    pub max_input_bytes: usize,
    pub max_output_bytes: usize,
    /// Address space (Unix), in bytes.
    pub max_memory_bytes: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self { timeout: Duration::from_secs(120), max_input_bytes: 512 << 20, max_output_bytes: 512 << 20, max_memory_bytes: 1 << 30 }
    }
}

/// Runs media jobs. [`FfmpegRunner`] in production, [`FakeRunner`] in tests.
pub trait Runner: Send + Sync {
    fn run(&self, job: Job, input: &[u8]) -> Result<Vec<u8>>;
}

/// The arguments for `job` on an input file of kind `demuxer`. Pure, so tests can check them.
pub fn args(job: Job, demuxer: Demuxer, input: &Path) -> Vec<String> {
    let mut a: Vec<String> = ["-nostdin", "-hide_banner", "-nostats", "-loglevel", "error", "-threads", "1", "-protocol_whitelist", "file,pipe"]
        .map(String::from)
        .to_vec();
    if demuxer == Demuxer::Mp4 {
        a.extend(["-enable_drefs", "0", "-use_absolute_path", "0"].map(String::from));
    }
    a.extend(["-f".into(), demuxer.ffmpeg_name().into(), "-i".into(), format!("file:{}", input.display())]);
    a.extend(["-map", "0:a:0", "-vn", "-sn", "-dn", "-map_metadata", "-1"].map(String::from));
    match job {
        Job::DecodeForAsr => {
            a.extend(["-ac", "1", "-ar"].map(String::from));
            a.push(ASR_RATE.to_string());
            a.extend(["-c:a", "pcm_s16le", "-f", "s16le"].map(String::from));
        }
        Job::Encode(e) => {
            let (codec, muxer) = e.codec_and_muxer();
            a.extend(["-c:a", codec, "-f", muxer].map(String::from));
        }
    }
    a.push("pipe:1".into());
    a
}

/// ffmpeg as a child process, under [`Limits`].
#[derive(Debug, Clone)]
pub struct FfmpegRunner {
    binary: PathBuf,
    limits: Limits,
}

impl FfmpegRunner {
    pub fn new(binary: impl Into<PathBuf>, limits: Limits) -> Self {
        Self { binary: binary.into(), limits }
    }

    /// `ffmpeg` from `PATH`; `Tool` when there is none.
    pub fn from_path(limits: Limits) -> Result<Self> {
        find_on_path("ffmpeg")
            .map(|b| Self::new(b, limits))
            .ok_or_else(|| MediaError::Tool("ffmpeg is not installed (needed for audio other than WAV, and for mp3, opus and flac output)".into()))
    }

    pub fn binary(&self) -> &Path {
        &self.binary
    }
}

/// The first executable named `name` on `PATH`.
pub fn find_on_path(name: &str) -> Option<PathBuf> {
    let exe = if cfg!(windows) { format!("{name}.exe") } else { name.to_string() };
    std::env::split_paths(&std::env::var_os("PATH")?).map(|d| d.join(&exe)).find(|p| p.is_file())
}

impl Runner for FfmpegRunner {
    fn run(&self, job: Job, input: &[u8]) -> Result<Vec<u8>> {
        if input.len() > self.limits.max_input_bytes {
            return Err(MediaError::Limit(format!("the input is {} bytes, over the {} byte limit", input.len(), self.limits.max_input_bytes)));
        }
        let demuxer = Demuxer::sniff(input)
            .ok_or_else(|| MediaError::Invalid("not an audio or video file this node reads (wav, mp3, flac, ogg, mp4/m4a/mov, mkv/webm)".into()))?;
        if matches!(job, Job::Encode(_)) && demuxer != Demuxer::Wav {
            return Err(MediaError::Invalid("encoding takes WAV input".into()));
        }
        // Private (0700) and removed on drop, also on error and panic.
        let dir = tempfile::Builder::new().prefix("ymir-media-").tempdir()?;
        let path = dir.path().join("input");
        {
            let mut f = std::fs::OpenOptions::new().write(true).create_new(true).open(&path)?;
            f.write_all(input)?;
        }
        let mut cmd = Command::new(&self.binary);
        cmd.args(args(job, demuxer, &path)).current_dir(dir.path()).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
        cmd.env_clear();
        if let Some(p) = std::env::var_os("PATH") {
            cmd.env("PATH", p);
        }
        confine(&mut cmd, &self.limits);
        let mut child = cmd.spawn().map_err(|e| MediaError::Tool(format!("could not start ffmpeg: {e}")))?;
        let mut stdout = child.stdout.take().ok_or_else(|| MediaError::Tool("no ffmpeg output".into()))?;
        let mut stderr = child.stderr.take().ok_or_else(|| MediaError::Tool("no ffmpeg error output".into()))?;
        let cap = self.limits.max_output_bytes;
        let out = std::thread::spawn(move || -> (Vec<u8>, bool) {
            let mut buf = Vec::new();
            let mut chunk = [0u8; 64 * 1024];
            loop {
                match stdout.read(&mut chunk) {
                    Ok(0) | Err(_) => return (buf, false),
                    Ok(n) if buf.len() + n > cap => return (buf, true),
                    Ok(n) => buf.extend_from_slice(&chunk[..n]),
                }
            }
        });
        let err = std::thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = stderr.by_ref().take(64 * 1024).read_to_end(&mut buf);
            let _ = std::io::copy(&mut stderr, &mut std::io::sink());
            buf
        });
        let deadline = Instant::now() + self.limits.timeout;
        let status = loop {
            if out.is_finished() && !child.try_wait()?.is_some() {
                // Output ended (or overflowed): give ffmpeg a moment to exit, then stop it.
                let grace = Instant::now() + Duration::from_millis(500);
                while Instant::now() < grace && child.try_wait()?.is_none() {
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
            if let Some(s) = child.try_wait()? {
                break Some(s);
            }
            if Instant::now() >= deadline || out.is_finished() {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        let (bytes, overflow) = out.join().map_err(|_| MediaError::Tool("ffmpeg output reader failed".into()))?;
        let stderr = err.join().unwrap_or_default();
        drop(dir);
        if overflow {
            return Err(MediaError::Limit(format!("ffmpeg's output went over the {cap} byte limit")));
        }
        let Some(status) = status else {
            return Err(MediaError::Limit(format!("ffmpeg took longer than {} s", self.limits.timeout.as_secs())));
        };
        if !status.success() {
            let msg = String::from_utf8_lossy(&stderr);
            let msg: String = msg.trim().chars().take(2048).collect();
            return Err(MediaError::Tool(format!("ffmpeg failed ({status}){}", if msg.is_empty() { String::new() } else { format!(": {msg}") })));
        }
        match job {
            Job::DecodeForAsr => {
                if bytes.is_empty() {
                    return Err(MediaError::Invalid("no audio in the file".into()));
                }
                Ok(wav::encode_pcm16(&Pcm::new(ASR_RATE, 1, wav::from_pcm16(&bytes))))
            }
            Job::Encode(_) => Ok(bytes),
        }
    }
}

#[cfg(unix)]
fn confine(cmd: &mut Command, limits: &Limits) {
    use std::os::unix::process::CommandExt;
    let mem = limits.max_memory_bytes;
    let cpu = limits.timeout.as_secs().saturating_add(1);
    let limit = move || -> std::io::Result<()> {
        let set = |res, v: u64| {
            let l = libc::rlimit { rlim_cur: v as libc::rlim_t, rlim_max: v as libc::rlim_t };
            // SAFETY: a valid rlimit on the stack.
            unsafe { libc::setrlimit(res, &l) }
        };
        if set(libc::RLIMIT_AS, mem) != 0 || set(libc::RLIMIT_FSIZE, 0) != 0 || set(libc::RLIMIT_CPU, cpu) != 0 {
            return Err(std::io::Error::last_os_error());
        }
        #[cfg(target_os = "linux")]
        {
            // No network: a new (empty) network namespace, when allowed. Best effort: without
            // it the protocol whitelist still keeps ffmpeg off the network.
            // SAFETY: plain system call.
            let _ = unsafe { libc::unshare(libc::CLONE_NEWUSER | libc::CLONE_NEWNET) };
        }
        Ok(())
    };
    // SAFETY: the closure runs in the child between fork and exec and only makes
    // async-signal-safe system calls (setrlimit, unshare), allocating nothing.
    unsafe {
        cmd.pre_exec(limit);
    }
}

#[cfg(not(unix))]
fn confine(_cmd: &mut Command, _limits: &Limits) {}

type Answer = Box<dyn Fn(Job, &[u8]) -> Result<Vec<u8>> + Send + Sync>;

/// A [`Runner`] for tests: answers with a closure and records the jobs (never the input).
pub struct FakeRunner {
    answer: Answer,
    calls: Mutex<Vec<(Job, usize)>>,
}

impl FakeRunner {
    pub fn new(answer: impl Fn(Job, &[u8]) -> Result<Vec<u8>> + Send + Sync + 'static) -> Self {
        Self { answer: Box::new(answer), calls: Mutex::new(Vec::new()) }
    }

    /// Decodes WAV input as [`FfmpegRunner`] would and refuses everything else the same way.
    pub fn wav_only() -> Self {
        Self::new(|job, input| match (job, Demuxer::sniff(input)) {
            (_, None) => Err(MediaError::Invalid("not an audio or video file this node reads".into())),
            (Job::DecodeForAsr, Some(Demuxer::Wav)) => crate::pcm::for_asr(input),
            (Job::DecodeForAsr, Some(d)) => Err(MediaError::Tool(format!("fake runner: no {} decoder", d.ffmpeg_name()))),
            (Job::Encode(_), _) => Err(MediaError::Tool("fake runner: no encoder".into())),
        })
    }

    /// The jobs run so far, with their input sizes.
    pub fn calls(&self) -> Vec<(Job, usize)> {
        self.calls.lock().map(|c| c.clone()).unwrap_or_default()
    }
}

impl Runner for FakeRunner {
    fn run(&self, job: Job, input: &[u8]) -> Result<Vec<u8>> {
        if let Ok(mut c) = self.calls.lock() {
            c.push((job, input.len()));
        }
        (self.answer)(job, input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone_wav(rate: u32, ms: u32, channels: u16) -> Vec<u8> {
        let n = (rate * ms / 1000) as usize;
        let mut s = Vec::new();
        for i in 0..n {
            for _ in 0..channels {
                s.push((i as f32 * 440.0 * std::f32::consts::TAU / rate as f32).sin() * 0.5);
            }
        }
        wav::encode_pcm16(&Pcm::new(rate, channels, s))
    }

    #[test]
    fn sniffs_only_the_allowlist() {
        assert_eq!(Demuxer::sniff(&tone_wav(8000, 10, 1)), Some(Demuxer::Wav));
        assert_eq!(Demuxer::sniff(b"fLaC\0\0"), Some(Demuxer::Flac));
        assert_eq!(Demuxer::sniff(b"OggS\0"), Some(Demuxer::Ogg));
        assert_eq!(Demuxer::sniff(b"\0\0\0\x20ftypisom"), Some(Demuxer::Mp4));
        assert_eq!(Demuxer::sniff(&[0x1A, 0x45, 0xDF, 0xA3, 1]), Some(Demuxer::Matroska));
        assert_eq!(Demuxer::sniff(b"ID3\x04"), Some(Demuxer::Mp3));
        assert_eq!(Demuxer::sniff(&[0xFF, 0xFB, 0x90]), Some(Demuxer::Mp3));
        for hostile in
            [&b"ffconcat version 1.0\nfile '/etc/passwd'\n"[..], b"#EXTM3U\n#EXT-X-VERSION:3\nhttp://x/a.ts", b"hello", b"", b"<?xml", b"\x89PNG"]
        {
            assert_eq!(Demuxer::sniff(hostile), None, "{:?}", String::from_utf8_lossy(hostile));
        }
    }

    #[test]
    fn arguments_are_locked_down() {
        let a = args(Job::DecodeForAsr, Demuxer::Mp4, Path::new("/tmp/x/input"));
        let s = a.join(" ");
        assert!(s.starts_with("-nostdin -hide_banner"));
        assert!(s.contains("-protocol_whitelist file,pipe"));
        assert!(s.contains("-enable_drefs 0"));
        assert!(s.contains("-f mov -i file:/tmp/x/input"));
        assert!(s.contains("-ac 1 -ar 16000 -c:a pcm_s16le -f s16le"));
        assert!(s.ends_with("pipe:1"));
        assert!(!s.contains("http") && !s.contains("concat"));
        let e = args(Job::Encode(Encoding::Opus), Demuxer::Wav, Path::new("in")).join(" ");
        assert!(e.contains("-f wav -i file:in") && e.contains("-c:a libopus -f ogg") && !e.contains("enable_drefs"));
    }

    #[test]
    fn fake_runner_records_and_refuses() {
        let f = FakeRunner::wav_only();
        let out = f.run(Job::DecodeForAsr, &tone_wav(48_000, 100, 2)).unwrap();
        assert_eq!(wav::probe(&out).unwrap().sample_rate, 16_000);
        assert!(f.run(Job::DecodeForAsr, b"text").is_err());
        assert!(f.run(Job::DecodeForAsr, b"fLaC....").is_err());
        assert_eq!(f.calls().len(), 3);
    }

    fn real() -> Option<FfmpegRunner> {
        let r = FfmpegRunner::from_path(Limits { timeout: Duration::from_secs(30), ..Limits::default() }).ok();
        if r.is_none() {
            eprintln!("ffmpeg not installed: skipping");
        }
        r
    }

    #[test]
    fn ffmpeg_decodes_and_encodes() {
        let Some(r) = real() else { return };
        let wav_in = tone_wav(44_100, 500, 2);
        let out = r.run(Job::DecodeForAsr, &wav_in).unwrap();
        let i = wav::probe(&out).unwrap();
        assert_eq!((i.sample_rate, i.channels, i.bits), (16_000, 1, 16));
        assert!((i.duration_ms() as i64 - 500).abs() <= 30, "{}", i.duration_ms());
        let flac = r.run(Job::Encode(Encoding::Flac), &wav_in).unwrap();
        assert_eq!(Demuxer::sniff(&flac), Some(Demuxer::Flac));
        let back = r.run(Job::DecodeForAsr, &flac).unwrap();
        assert!((wav::probe(&back).unwrap().duration_ms() as i64 - 500).abs() <= 30);
        assert!(r.run(Job::Encode(Encoding::Mp3), &flac).is_err(), "encoding takes WAV only");
    }

    #[test]
    fn ffmpeg_hostile_inputs_fail_closed() {
        let Some(r) = real() else { return };
        // A text file, a concat list and a playlist never reach ffmpeg.
        for bad in [&b"just some text named clip.wav"[..], b"ffconcat version 1.0\nfile 'a.wav'\n", b"#EXTM3U\nhttp://192.0.2.1/a.ts\n"] {
            assert!(matches!(r.run(Job::DecodeForAsr, bad), Err(MediaError::Invalid(_))));
        }
        // Truncated and corrupt containers fail (or decode nothing), within the limits.
        let mut t = r.run(Job::Encode(Encoding::Flac), &tone_wav(16_000, 300, 1)).unwrap();
        t.truncate(60);
        assert!(r.run(Job::DecodeForAsr, &t).is_err());
        let mut fake_mp4 = b"\0\0\0\x18ftypisom\0\0\0\0isomiso2".to_vec();
        fake_mp4.extend_from_slice(b"\0\0\0\x30moov\0\0\0\x28trak....url file:///etc/passwd http://192.0.2.1/");
        assert!(r.run(Job::DecodeForAsr, &fake_mp4).is_err());
        // Output cap.
        let tiny = FfmpegRunner::new(r.binary(), Limits { max_output_bytes: 1000, ..Limits::default() });
        assert!(matches!(tiny.run(Job::DecodeForAsr, &tone_wav(16_000, 2_000, 1)), Err(MediaError::Limit(_))));
        // Input cap.
        let small = FfmpegRunner::new(r.binary(), Limits { max_input_bytes: 10, ..Limits::default() });
        assert!(matches!(small.run(Job::DecodeForAsr, &tone_wav(16_000, 10, 1)), Err(MediaError::Limit(_))));
    }

    #[test]
    fn missing_binary_is_a_tool_error() {
        let r = FfmpegRunner::new("/nonexistent/ffmpeg", Limits::default());
        assert!(matches!(r.run(Job::DecodeForAsr, &tone_wav(8000, 10, 1)), Err(MediaError::Tool(_))));
    }
}
