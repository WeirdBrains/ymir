# Audio through Ymir: speech out, and audio processing

Status: design, 2026-11-27 (Michael, t75u: "support for transcripts to audio and for audio processing"). Milestones
1 to 10 were built on 2026-11-27, against mocks only (see "What was built" under "Milestones"), speech out included, on
`weirdnet-backends` 0.4.10. Milestone 11 (real runs on m1/m2) and 12 (later) remain.
"What exists" below is the state before them. It builds on the transcription route of video milestone 2 (`5b77934`, in v0.23.0) and needs new Weirdnet crate
versions for speech out; what Ymir can build before them is listed under "Milestones". Read
[video.md](video.md) first (limits are reported, never defaulted; the hardened ffmpeg rules) and [DESIGN.md](../DESIGN.md),
"Images and audio through the pool".

## Decisions (2026-11-27)

- **Four target speech models**, chosen by Michael: Qwen3-TTS, Kokoro-82M, VoxCPM2 and Fish Audio Drama 3. Others that
  the same engines serve (ZONOS3-Mini, CosyVoice3, Voxtral TTS) need no new design, only a catalog entry and a
  compatibility row.
- **The role is `speak`**, beside `transcribe`: `role = "speak"` in `[[model.local]]`, `Purpose::Speak` (`speak` on the
  wire), infer/2 `Speak`, `ymirctl speak`. A verb, like `transcribe`.
- **One request shape for every model, and no model fakes a feature.** A request asks for a preset voice, a designed
  voice, a cloned voice or a multi-speaker script; each model's *speech profile* says which of those it does, and a
  request for something it doesn't do is a 400 that names the model and what it lacks. Ymir never maps an unknown voice
  name to another one, never drops a voice description silently, never turns a script into one voice.
- **Profiles are reported or absent.** Preset voices, languages, formats, sample rate and limits come from the engine at
  load (llama-server `/props` `voices`, vLLM-Omni `GET /v1/audio/voices` and response headers), from a `weirdnet-model`
  catalog entry that cites its model card, or from the operator in `node.toml`. A field nobody knows is absent.
- **Speech out is content.** Neither the text nor the audio (nor a reference clip) is cached, logged or stored.
  Logs carry the model, the node, the character count and the bytes returned.
- **Long jobs are a client's job first.** Long-transcript narration and long-audio transcription are planned and stitched
  by the client (ymirctl, `ymirctl mcp`, Bragi) from the reported limits, as Bragi samples video. Server-side planning
  comes later, opt-in, from the same `weirdnet-media` code.
- **Voice cloning is opt-in for guests and needs consent** on every request (below).

## What exists today (v0.23.0)

- **Audio in.** Message content can carry `input_audio` parts (`weirdnet-agent` 0.5, `Modalities::audio`); routing sends
  them only to a model whose `Capabilities::audio` is set (llama.cpp with an audio projector). Not cached.
- **Transcription.** `role = "transcribe"` sets `ModelLaunch::transcribe` (`weirdnet-backends` 0.4.9); the model's
  catalog/1 purpose is `Purpose::Transcribe` (`weirdnet-agent` 0.5.8+, hidden from older peers by `Catalog::for_older`).
  infer/2 `ClientFrame::Transcribe(TranscribeRequest { model, audio: bytes, format, language, prompt, timestamps, .. })`
  answers `Accepted`, `Working` keepalives, `ServerFrame::Transcript { transcript: Transcript { text, language,
  duration_ms, segments } }`, `Done`. `InferenceHost::open_transcribe` → `Coordinator::resolve_transcriber`
  (`Catalog::route_transcribe`) → `Coordinator::transcribe` (ready node first, failover, guest allowlist with
  `guest-model`) → `Supervisor::transcribe` (llama.cpp: a greedy chat request with the clip; vLLM:
  `/v1/audio/transcriptions`, `verbose_json` for segments). `Pool::transcribe` is the client side.
- **Measured limits.** `MediaLimits` (`audio_max_seconds`, `audio_tokens_per_minute`, `max_request_bytes`, image
  fields) from the engine, from `[model.local] limits`, or measured at load and kept in the models dir
  (`Supervisor::set_limits_store`); catalog/1 carries the strictest per field. `weirdnet-model` 0.5.4 knows the
  Whisper-style 30 s window (`limits::WHISPER_CHUNK_SECONDS`) and Voxtral's tokens per piece.
- **Not there.** No speech out anywhere. The bridge (`weirdnet-openai` 0.2.4) has `/v1/models`, `/v1/chat/completions`,
  `/v1/completions` and `/v1/embeddings` only. No `ymirctl transcribe`, no MCP `transcribe` tool, no audio chunking;
  a clip longer than the model takes is refused by the engine.

## The four models

What their engines document today ([llama.cpp TTS models](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal/tts.md),
[llama-server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md),
[vLLM-Omni Speech API](https://github.com/vllm-project/vllm-omni/blob/main/docs/serving/speech_api.md) and its
[TTS serving examples](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/examples/online_serving/text_to_speech.md),
all read 2026-11-27). The compatibility row in [DESIGN.md](../DESIGN.md)
replaces this table once a real run fills it.

| | Qwen3-TTS | Kokoro-82M | VoxCPM2 | Fish Audio Drama 3 |
|---|---|---|---|---|
| Size, license | 0.6B and 1.7B; Apache 2.0 | 82M; Apache 2.0 | 2B; Apache 2.0 | 35B MoE, 3B active; Fish Audio Community License |
| Engines | llama-server (GGUF), vLLM-Omni | llama-server | llama-server, vLLM-Omni | llama-server (with its mmproj), vLLM-Omni |
| Checkpoints | three, one per mode: CustomVoice, VoiceDesign, Base | one | one | one |
| Preset voices | CustomVoice: named speakers (`vivian`, `ryan`, ...) | yes, shipped with the weights (default `af_heart`) | no | no |
| Voice design | VoiceDesign: a description in `instructions` (vLLM-Omni) | no | a description in parentheses at the start of the text | no |
| Style instructions | CustomVoice: `instructions` (vLLM-Omni; llama-server ignores `instructions`) | no | no | inline event tags (`[sighs]`) |
| Cloning | Base: reference clip, transcript optional (ICL mode, clip ≤ 30 s on llama.cpp) | no | reference clip only, transcript not used (llama.cpp: first 30 s, resampled to 16 kHz) | one reference clip and its transcript (llama.cpp: first 30 s) |
| Multi-speaker script | no | no | no | `<\|speaker:0\|>` to `<\|speaker:15\|>` in the text |
| Languages | 10 (llama.cpp `--tts-lang` list) | English with built-in phonemes; others need phoneme input | 30 (from the text) | from the text |
| Speed | vLLM-Omni `speed` | llama-server `speed` | not documented | not documented |
| Rate | 24 kHz | 24 kHz | 48 kHz | 44.1 kHz |
| Length cap documented | output token budget | engine splits at 510 phonemes | 6 frames per text token + 10 | llama.cpp: 4096 positions, prompt plus frames (~46 ms a frame) |
| Formats | llama-server: `wav`, `pcm` (16-bit); vLLM-Omni: `wav`, `mp3`, `flac`, `pcm`, `opus` | `wav`, `pcm` | as its engine | as its engine |

Notes that shape the design:

- **Qwen3-TTS is three pool ids**, one per checkpoint, each with its own profile (CustomVoice: presets and
  instructions; VoiceDesign: design; Base: cloning). 0.6B and 1.7B are more ids. Nothing switches mode at request time.
- **Kokoro** speaks phonemes; only English has the text-to-phoneme step in the engine. Ymir does not phonemize: its
  profile lists the languages the engine handles from text (`en-us`, `en-gb`), and other-language voices need the
  caller's own phonemes (`raw` input, below).
- **VoxCPM2's design and Qwen3-TTS VoiceDesign differ on the wire** (a text prefix on llama-server, `instructions` on
  vLLM-Omni). That is the adapter's job; the request carries `Voice::Design { description }` either way.
- **Drama 3 is the only multi-speaker model.** Its script turns become speaker tags in the adapter; a cloned voice is
  speaker 0 (one reference). Event tags stay in the text as the caller wrote them.
- **Drama 3's license** caps the licensee's annual revenue at $500k and forbids offering TTS as a hosted service. The
  catalog entry is marked restricted: `ymirctl setup --download` never picks it, a download asks for the operator's
  explicit acceptance (`--accept-license fish-audio-community`), and it is refused to guests of a shared pool
  ([pool-sharing.md](pool-sharing.md)) unless the operator sets `guests = true` on that model, because serving another
  identity can be a hosted service. `ymirctl status` shows the license name next to the model.

## Speech out through the pool

### Config

```toml
[[model.local]]
id = "qwen3-tts-1.7b-customvoice"
role = "speak"
backend = "vllm-omni"           # or "llamacpp" (GGUF with its mmproj)
# repo, file, mmproj as for any model
[model.local.speak]             # optional operator overrides, field by field, never defaults in code
voices = ["vivian", "ryan"]     # restricts the presets the engine reports; never adds one
languages = ["en", "zh"]
max_input_chars = 2000
max_reference_seconds = 30
max_output_seconds = 180
clone = false                   # turn cloning off on this node even if the model does it
guests = false                  # serve this model to guests of a shared pool (default true, false for restricted licenses)
```

`role = "speak"` sets `ModelLaunch::speak`. MLX is refused at validation (no speech engine). A speaker is never picked
as the main chat model, gets no stand-in, and is not counted in the chat model budget's KV plan (its own footprint is
the catalog entry's). `[model.local.speak]` becomes the profile overrides: the operator can narrow what the engine
reports (fewer voices, a smaller limit) or fill a limit it doesn't report; widening what the engine reports is refused
at validation.

### The speech profile

Each serving copy has a `SpeechProfile` (spec below): what it does (`presets`, `design`, `instructions`, `clone`,
`script_speakers`, `speed`, `streaming`), what it takes (`languages`, `formats`, `sample_rate`) and its limits
(`max_input_chars`, `max_reference_seconds`, `max_reference_bytes`, `max_output_seconds`). A bool is true only when the
adapter knows the model does it; a list or limit is absent when unknown. catalog/1 carries the merged profile per model:
the intersection of lists, the smallest limit, and a bool only when every node with the model has it, as
`MediaLimits::tightest` does for input limits. A node record without a profile counts as no speech.

### infer/2 `Speak`

`ClientFrame::Speak(SpeakRequest)`:

- `input`: `SpeakInput::Text(String)`, `SpeakInput::Script(Vec<ScriptTurn { speaker: u8, text }>)` (needs
  `script_speakers` > the highest speaker), or `SpeakInput::Raw(String)` (passed to the engine with no template, for
  Kokoro phonemes; needs `raw`).
- `voice`: `None` (the model's default), `Voice::Preset { name }` (one of `presets`; any other name is a 400 listing
  them), `Voice::Design { description }` (needs `design`), `Voice::Clone { audio: bytes, format, text: Option<String>,
  consent: bool }` (needs `clone`; `consent` must be true). For a script, `voices: Vec<(u8, Voice)>` per speaker, bounded
  by `max_voices` (Drama 3: one clone, speaker 0).
- `instructions`: style or emotion for the chosen voice (needs `instructions`).
- `language` (ISO 639-1 or the engine's code from `languages`), `speed` (per mille, 1000 = the model's pace; needs
  `speed`), `seed`, `format` (`AudioFormat`: `wav`, `pcm16`, `mp3`, `opus`, `flac`; must be in `formats`),
  `sample_rate` (must match, or be one the engine resamples to), `stream: bool`, `priority`, `cert`.
- `SpeakRequest::check` refuses empty input, a clone without consent and a reference over 30 MiB before any host opens
  a backend; the backend then checks the request against its profile (`Unsupported` → the next node, which may have
  a different engine; `BadRequest` when it names what the model lacks).

Server frames: `Accepted`, `Working` keepalives until the first audio, then `ServerFrame::SpeechStart { id, format,
sample_rate: Option, channels: Option }` once (the rate and channels are optional, since an encoded format may not say), `ServerFrame::Audio { id, offset, data: bytes }` (chunks of at most 256 KiB; with
`stream`, as the engine produces them; without, the finished file in chunks), then `Done` with usage (input
characters, audio milliseconds, engine tokens when known). A reference transcript the model doesn't use (VoxCPM2) is
named in `Done`'s meta (`ignored: ["reference_text"]`) rather than refused, because sending it is harmless and clients
will send it to every model.

**Failover** only before `SpeechStart`. After audio has left, a failure is `Failed`: another node would start the
speech again with another voice draw, and splicing two takes is worse than an error. No resume.

### Routing

`InferHost::open_speak(caller, &request)` on the inference node, which gets the whole `SpeakRequest` (so the host can
check the voice, a clone and the model before opening a backend), then the same chain as transcription:

1. `Catalog::route_speak(model, &needs)`: a named model must have `Purpose::Speak` and a profile that covers the
   request's needs (preset, design, clone, script speakers, format) or it is a 400 that says which; with none named, the
   best speaker whose profile covers them (warm first, then replicas), or a 400 "no model in the pool speaks" /
   "no speaker in the pool clones voices". Older peers never see `speak` entries (`Catalog::for_older`).
2. `Coordinator::speak`: a node whose copy speaks and covers the needs, ready first, else one that loads it; draining,
   reserved and leaving nodes skipped; retryable failures before audio go to the next node.
3. Admission: the same `authorize` as chat. A guest gets only speakers in its grant's allowlist (`guest-model`), only
   models with `guests = true`, and `Voice::Clone` only when the grant has `clone = true` (`guest-clone`, a new grant
   field, default off). *Built so far:* every guest clone is refused (`guest-clone`); the per-grant switch is not built.
4. Nothing is cached; the semantic cache is never asked.

### Large clips

infer/2 carries the reference clip and the answer as CBOR byte strings, so the request limit (`max_request_bytes`, 32
MiB bridge default) bounds a reference clip, and the answer is chunked. A minute of 24 kHz 16-bit mono is 2.9 MB, so a
single response is fine; hours of narration go through the long-form plan below. Content-root references
(`weirdnet-net` `content`, LF-12) replace inline bytes later for both directions.

## Audio processing

All of it lives in one place: `weirdnet-media` (0.1.0, published 2026-11-27 from the spec below; Ymir never had to keep
the interim `ymir-media` it had started). Clients use it; a node uses it only for the opt-in server-side paths.

### Long-transcript synthesis (narration, podcasts)

- **Plan** (`TextPlan::new(text_or_script, limits)`, pure and deterministic): split at paragraph, then sentence, then
  clause boundaries, never inside a word or a speaker turn, so each piece fits `max_input_chars` (and, for Drama 3, the
  position budget the profile reports). A script keeps its turns; a turn over the limit splits at sentences and keeps
  its speaker. Without `max_input_chars` the plan refuses text over 4 000 characters and says to set it (this is a
  refusal threshold of the client, not a limit claimed for the model).
- **Synthesis**: one `Speak` per piece, in order, with the same model, voice and seed, at most two in flight. Every
  piece must come back with the same format, sample rate and channels or the job fails.
- **Stitching**: PCM is joined with a 10 ms equal-power crossfade inside a paragraph and a configurable pause between
  paragraphs and speaker turns (default 300 ms), then written once as WAV, or encoded to `mp3`/`opus`/`flac` with
  ffmpeg when asked. Pieces are held as PCM in memory or in the job's private temp dir, never elsewhere.
- **Progress and cancel**: `piece i of n` with characters done; cancel stops after the piece in flight and removes the
  temp dir; nothing partial is written to the output path unless `--keep-partial`.
- **Voice drift**: a designed voice can drift between pieces. `--anchor` (off by default, only for models that clone)
  uses the first piece as the reference clip for the rest. It is the caller's choice, not something Ymir does quietly.

### Long-audio transcription and preprocessing

- **Probe**: WAV is read in Rust (the header reader `weirdnet-backends` uses for durations, generalized); any other
  container is probed with ffprobe (hardened, below).
- **Normalize**: decode to 16 kHz mono 16-bit PCM WAV (Whisper and Voxtral both want 16 kHz); WAV input is resampled in
  Rust, anything else through ffmpeg. `--no-normalize` sends the file as it is when the model takes its format.
- **Chunk**: only when the clip is longer than `audio_max_seconds` (or the transcriber's window from `weirdnet-model`).
  Silence-aware: 30 ms RMS frames, the cut goes at the quietest point in the last fifth of the window; no overlap.
  Each piece's `prompt` is the last 200 characters of the previous transcript. Without `audio_max_seconds` reported
  or set, a clip over the request limit is refused with the setting to fill in.
- **Re-align**: segments are shifted by their piece's start. A backend without segments (llama.cpp) gets one segment
  per piece with the piece's bounds, flagged `coarse` in JSON output, so SRT and VTT say what they are.
- **From video**: `-vn` extraction of the first audio stream through the same ffmpeg runner, then the steps above.

### Hardened ffmpeg

The same stance as [video.md](video.md#server-side-expansion-opt-in-later), "Server-side expansion", as code in `weirdnet-media::ffmpeg`:

- Ymir opens the input file itself (after the roots check) and feeds it on `pipe:0`; ffmpeg gets `-nostdin -hide_banner
  -loglevel error -protocol_whitelist file,pipe`, an allowlisted demuxer (`wav`, `mp3`, `flac`, `ogg`, `mov,mp4,m4a`,
  `matroska,webm`), never `concat`, playlists or a URL.
- Limits: wall time (default 120 s per call, setting), address space (`RLIMIT_AS`, 1 GiB), output size (the plan's
  bound, checked while reading `pipe:1`), CPU nice; on Linux a new network namespace when unprivileged user namespaces
  allow it.
- Output to a `tempfile` dir with mode 0700, removed on drop, also on panic and cancel.
- No input, output or transcript in logs; ffmpeg's stderr is kept only on failure and truncated to 2 KiB.
- A node runs it only with `[media] server = true` (off by default), and never on the node that holds the model when
  they differ. A missing ffmpeg is a clear error naming the setting, shown in `ymirctl status`.
- A fake runner for tests; ffmpeg-backed tests are `#[ignore]` unless it is installed.

### Voice cloning: safety and privacy

- Reference clips: at most `max_reference_seconds` (profile) and 30 MiB; longer clips are refused, not truncated by
  Ymir (the engine may still use only its first 30 s; the profile says so).
- `consent: true` on every clone request; `ymirctl speak --ref-audio` asks for `--ref-consent` (the caller confirms the
  voice is theirs or they have permission).
- A clip lives in memory for one request: llama-server gets it as `speaker_ref_b64`, vLLM-Omni as a `data:` URL in
  `ref_audio`. Ymir never uses vLLM-Omni's `POST /v1/audio/voices` or `custom_voice_dir`, never starts it with
  `--allowed-local-media-path`, and never forwards an `http(s)://` or `file://` reference. Clone requests set
  vLLM-Omni's `cache_salt` to a hash of the request id so no conditioning prefix is shared across callers.
- Guests: off unless the grant has `clone = true`. Operators can turn cloning off per model (`clone = false`).

## OpenAI bridge, CLI and MCP

### Bridge (`weirdnet-openai`)

- `POST /v1/audio/speech` → infer/2 `Speak`. `input`, `model`, `voice` (a name or `{"id": ...}`; OpenAI names such as
  `alloy` are not mapped: a 400 lists the model's presets), `instructions`, `response_format` (absent: the model's first
  format, `wav` for every target model, not OpenAI's `mp3`; one the model lacks is a 400 listing its formats),
  `speed`, `stream_format: "audio"` (raw bytes as they come; `sse` refused), and Weirdnet extensions under `weirdnet`:
  `language`, `seed`, `design`, `ref_audio` (`data:` URL only) with `ref_text` and `consent`, `script`. The answer's
  `Content-Type` says the real format and rate (`audio/pcm;rate=24000`).
- `POST /v1/audio/transcriptions` (multipart) → infer/2 `Transcribe`: `file`, `model`, `language`, `prompt`,
  `response_format` (`json`, `text`, `verbose_json`, `srt`, `vtt`), `timestamp_granularities[]=segment`.
- Both bounded by `max_request_bytes`: a 413 naming it in `x-weirdnet-max-request-bytes`. `/v1/models` lists
  `weirdnet.purpose` and `weirdnet.speech` (the profile).

### CLI (`ymirctl`)

Both go through ymird's control socket (`ControlRequest::Transcribe` and `ControlRequest::Speak`), which runs them
through the inference role like `ymirctl chat`; nothing talks to an engine directly.

- `ymirctl transcribe FILE [--model M] [--lang xx] [--prompt T] [--format text|json|srt|vtt] [--timestamps]
  [--chunk auto|off] [--out F]`: text to stdout by default. `--chunk auto` runs the long-audio plan.
- `ymirctl speak [TEXT | --file F | --script F] [--out F | -] [--model M] [--voice V | --design "..." | --ref-audio F
  [--ref-text T] --ref-consent] [--instructions "..."] [--lang xx] [--speed 1.2] [--format wav|pcm|mp3|opus|flac]
  [--seed N] [--stream] [--pause-ms N] [--anchor]`: a script file is JSON `[{"speaker":0,"text":"..."}]` or lines
  `S0: ...`. Text over one piece runs the long-form plan. `-` writes to stdout (with `--stream`, as it arrives).
- `ymirctl voices [MODEL]`: each speaker's profile, `not reported` for the gaps.
- `ymirctl status` and `pool models` show `speaks (text to speech)` and the purpose.

### MCP (`ymirctl mcp`)

- `transcribe`: `path` (under the allowed roots) or `audio` (base64) with `format`; the options of the CLI. Read-only
  (it writes nothing), so available under `--read-only`.
- `speak`: `text` or `script`, the voice options, and `out` (a path under the roots). Writing a file makes it a
  mutating tool: refused under `--read-only`, behind the `confirm` gate, and a clone also needs `consent`. Without
  `out` it returns base64 audio until `weirdnet-mcp` has audio content blocks (wish list in [video.md](video.md#weirdnet-wish-list)).
- `voices`: read-only, the profiles. `models` shows the `speak` purpose.

## Weirdnet crate spec

For the Weirdnet Lead Developer. All additive; every new enum variant and struct is `#[non_exhaustive]`, every new
field `#[serde(default, skip_serializing_if = ...)]` and absent in records from older nodes.

### `weirdnet-agent` (0.5.11, published)

Published as 0.5.11 with four departures from the spec below, which Ymir follows:

1. `InferHost::open_speak(caller, request: &SpeakRequest)` takes the whole request, not a model name.
2. `SpeechNeeds.presets` is a `Vec<String>` (every preset the request names, a script's speakers included), and it
   gained `reference_bytes: Option<u64>` beside `reference_ms`.
3. `SpeechStart`'s `sample_rate` and `channels` are optional (`AudioHeader::new(format).with_sample_rate(..)
   .with_channels(..)`).
4. There is no public `serve_speak`: speech is a private handler dispatched inside `v2::serve` on `ClientFrame::Speak`,
   so a host serving infer/2 serves speech with the same admission.

The spec as written:

- `remote::catalog::Purpose::Speak` (`"speak"` in `as_str`/`parse`). `Catalog::for_older` generalized: a peer that
  doesn't advertise `speak` doesn't get those entries (as for `transcribe` before 0.5.8). `CatalogModel::speaks()`,
  `CatalogModel::speech: Option<SpeechProfile>`, `with_speech`, `Catalog::speakers()`,
  `Catalog::route_speak(model: Option<&str>, needs: &SpeechNeeds) -> Result<String, RemoteError>`.
- `speech` module: 
  ```rust
  pub enum AudioFormat { Wav, Pcm16, Mp3, Opus, Flac }            // "wav", "pcm16", ...
  pub enum CloneMode { Audio, AudioAndText, AudioTextRequired }   // what a reference needs
  pub struct SpeechProfile {
      pub presets: Option<Vec<String>>, pub design: bool, pub instructions: bool,
      pub clone: Option<CloneMode>, pub script_speakers: Option<u8>, pub max_voices: Option<u8>,
      pub raw: bool, pub speed: bool, pub streaming: bool,
      pub languages: Option<Vec<String>>, pub formats: Vec<AudioFormat>, pub sample_rate: Option<u32>,
      pub max_input_chars: Option<u32>, pub max_reference_seconds: Option<u32>,
      pub max_reference_bytes: Option<u64>, pub max_output_seconds: Option<u32>,
      pub license: Option<String>,
  }
  impl SpeechProfile { pub fn tightest(a, b) -> Self; pub fn narrowed_by(&self, operator: &SpeechProfile) -> Result<Self, String>;
                       pub fn covers(&self, needs: &SpeechNeeds) -> Result<(), String>; }
  pub struct SpeechNeeds { presets: Vec<String> /*0.5.11; spec: preset: Option<String>*/, design, instructions, clone: Option<bool /*has text*/>,
                           speakers: u8, voices: u8, raw, speed, format: Option<AudioFormat>, input_chars: u32,
                           reference_ms: Option<u64>, reference_bytes: Option<u64> /*0.5.11*/ }   // SpeakRequest::needs()
  ```
- `remote::infer::v2`: `ClientFrame::Speak(SpeakRequest)`; `SpeakRequest { id, model, input: SpeakInput, voice:
  Option<Voice>, voices: Vec<SpeakerVoice>, instructions, language, speed: Option<u32 /*per mille*/>, seed: Option<u64>,
  format: Option<AudioFormat>, sample_rate: Option<u32>, stream: bool, priority, cert }` with `check()` and builders;
  `SpeakInput::{Text, Script(Vec<ScriptTurn { speaker: u8, text }>), Raw}`; `Voice::{Preset { name }, Design {
  description }, Clone { #[serde(with = "bytes")] audio, format, text: Option<String>, consent: bool }}`;
  `ServerFrame::SpeechStart { id, format, sample_rate: Option, channels: Option }`, `ServerFrame::Audio { id, offset, #[serde(with =
  "bytes")] data }` (≤ 256 KiB), `Working` reused; `Usage` gains `input_chars` and `audio_ms`; `Meta` gains `ignored:
  Vec<String>`. Speed is per mille so the request keeps `Eq`.
- `SpeakBackend` trait (`name`, `speak(&SpeakRequest, &CancelToken) -> Result<AudioStream, RemoteError>` where
  `AudioStream` yields `SpeechStart` then chunks then usage), `InferHost::open_speak(caller, &SpeakRequest) -> Result<Arc<dyn
  SpeakBackend>, RemoteError>` (default `unavailable("no speech model on this node")`), served inside `v2::serve` (a
  private handler, not a public `serve_speak`) with the same admission and keepalive as transcription.
- `Pool::speak(request, cancel) -> Result<Speech, RemoteError>` and `Pool::speak_stream`: failover across nodes only
  before `SpeechStart`; `Speech { audio: Vec<u8>, format, sample_rate, channels, model, node, meta, usage }`.
- Tests as `transcribe_tests.rs`: frames round-trip, an older node drops the stream (counts as `Unsupported`),
  failover before audio and none after, `for_older` hides `speak`.

### `weirdnet-backends` (0.4.10)

- `EngineKind::VllmOmni` (`vllm serve MODEL --omni`, Linux, NVIDIA; its own install and version probe).
  `EngineKind::capabilities` gains `speak` for `Llamacpp` and `VllmOmni`.
- `Capabilities::speak: bool` (stays `Copy`), `with_speak`, and `purpose` returns `Purpose::Speak` for it.
- `ModelLaunch::speak: bool` + `with_speak`, `ModelLaunch::speech: SpeechProfile` (operator overrides, narrowing only;
  `check_speech` refuses widening). `ModelLaunch::capabilities`: `speak` when the role is set and the engine is
  llama.cpp with the model's mmproj or vLLM-Omni.
- `ModelStatus::speech: Option<SpeechProfile>`: read at load (llama-server `GET /props` `voices` and `modalities`
  containing `audio`; vLLM-Omni `GET /v1/audio/voices` and a 1-word probe for `x-audio-sample-rate`), merged with the
  `weirdnet-model` entry and the operator's overrides, kept by `set_limits_store` like the other measured limits.
- `Supervisor::speak(id, &SpeakRequest) -> Result<AudioStream, RemoteError>`; `Unsupported` when the model doesn't
  speak here or its profile doesn't cover the request; `BadRequest` naming the missing feature when no engine could.
- Adapters (`speak.rs`), each mapping only what the model does:
  - llama-server `POST /v1/audio/speech`: `input`, `voice` (preset), `response_format` `wav`|`pcm` (`pcm16` maps to
    `pcm`), `stream_format: "audio"`, `speed`, `lang`, `seed`, `speaker_ref_b64` + `speaker_ref_text` (clone),
    `raw_prompt` (raw). Design for VoxCPM2 as the `(description)` text prefix; Drama 3 script turns as `<|speaker:N|>`
    prefixes. Sample rate from the `Content-Type` `rate`. Any other format is `BadRequest`.
  - vLLM-Omni `POST /v1/audio/speech`: `input`, `voice`, `task_type` from the checkpoint (`CustomVoice`, `VoiceDesign`,
    `Base`), `instructions` (design for VoiceDesign, style for CustomVoice), `language`, `speed`, `seed`,
    `response_format`, `stream: true` + `stream_format: "audio"`, `ref_audio` as a `data:` URL only, `ref_text`,
    `cache_salt`. Drama 3 turns as speaker tags. Rate from `x-audio-sample-rate`.
- Tests with the in-process mock engine (both flavors): each adapter's body for each `Voice`, refusals for each
  missing feature, no `http(s)://` or `file://` ever sent, stream chunking, `/props` voices read.

### `weirdnet-openai` (0.2.5, published)

- `POST /v1/audio/speech` and `POST /v1/audio/transcriptions` as in "Bridge" above, over `Pool::speak` /
  `Pool::speak_stream` and `Pool::transcribe`; the 413 with `x-weirdnet-max-request-bytes`; `/v1/models` with
  `weirdnet.purpose` and `weirdnet.speech`. SRT and VTT rendering from `Segment`s (shared with ymirctl: a
  `subtitles` helper in `weirdnet-agent` or here).
- Tests: OpenAI SDK request shapes, `alloy` refused with the preset list, `data:` URL only for `ref_audio`, the 413.

### `weirdnet-model` (0.5.5, published)

- `hw::Purpose::Speak` (`"speak"`), `hw::speaker()`, `by_purpose(Speak)`.
- Catalog entries, each with memory footprints and a `SpeechProfile` cited to its model card (never measured numbers
  invented): `qwen3-tts-0.6b-{customvoice,base}`, `qwen3-tts-1.7b-{customvoice,voicedesign,base}` (GGUF from
  `ggml-org` where published, HF for vLLM-Omni), `kokoro-82m`, `voxcpm2`, `fish-audio-drama-3` (with mmproj, license
  `fish-audio-community`, `restricted: true`). `CatalogEntry::license` and `restricted`; downloads of a restricted
  entry need an explicit acceptance argument.
- Detection from GGUF metadata (architecture and the mmproj's TTS heads) so a hand-placed file gets the right profile.

### `weirdnet-media` (0.1.0, published)

`probe` (WAV in Rust, others through ffprobe), `pcm` (decode WAV, resample to 16 kHz mono, crossfade, concatenate,
write WAV), `vad::silence_cuts`, `transcribe::plan` and `realign`, `speak::TextPlan` (paragraph/sentence/clause, script
aware), `ffmpeg::Runner` (the hardened invocation above) with `FakeRunner`, `subtitles` (SRT, VTT). Then the frame
sampling of [video.md](video.md) joins it.

## Testing

- `ymir-testkit` mock engine: `/v1/audio/speech` as llama-server (WAV/PCM, `/props` voices) and as vLLM-Omni
  (`/v1/audio/voices`, `x-audio-sample-rate`, `task_type`), returning a generated sine so lengths are checkable.
- Routing (`tests/speak.rs`, like `transcribe.rs`): catalog purpose and profile; a request needing a preset, design,
  clone or script goes only to a model that does it; the 400s name the gap; failover before audio, `Failed` after;
  guests: allowlist, `guests = false`, `clone` grant; not cached; no text, audio or reference bytes in the logs (a log
  capture asserts it).
- `weirdnet-media`'s own units (in that crate): WAV header fuzz, resampling length, silence cuts on generated tones, plan determinism, crossfade
  continuity, segment re-alignment, SRT/VTT.
- Hostile files through the ffmpeg runner (`#[ignore]` without ffmpeg): a text file named `.wav`, a truncated MP3, a
  `concat` playlist, an HLS `.m3u8`, an MP4 pointing at a URL, a zip bomb–sized FLAC; all fail closed within the limits.
- CLI and MCP: a scripted stdio session calling `transcribe` and `speak` against the mock; `--read-only` refuses
  `speak` with `out`; a clone without consent is refused.
- Real models (`#[ignore]`, run by hand, never in CI, never a download in tests): each of the four on its engines,
  filling the compatibility row in [DESIGN.md](../DESIGN.md).

## Milestones

**Now, with the crates Ymir already uses** (v0.23.x):

1. `ymirctl transcribe`: `ControlRequest::Transcribe` through the inference role (`InferenceHost` / `Pool::transcribe`
   on `weirdnet-agent` 0.5.9), text/JSON/SRT/VTT output from `Transcript::segments`.
2. Media processing (first planned as an interim `ymir-media`, now `weirdnet-media` 0.1.0): WAV probe, resample to 16 kHz mono, silence-aware
   chunking, segment re-alignment, PCM stitching and crossfade, the text plan, SRT/VTT, and the hardened ffmpeg runner
   with its fake. No Ymir types.
3. Long-audio transcription in ymirctl (`--chunk auto`), using catalog/1 `audio_max_seconds` and refusing when absent.
4. MCP `transcribe` tool.
5. The testkit mock `/v1/audio/speech` (both flavors) and `/props` voices, so speech routing tests can start the day the
   crates land.
6. `ymirctl engines install vllm-omni` (venv, the same CUDA wheel check as vLLM), usable with `wrap` once the crate
   knows the engine.

**After the crates** (`weirdnet-agent` 0.5.11, `weirdnet-backends` 0.4.10, `weirdnet-model` 0.5.5,
`weirdnet-openai` 0.2.5):

7. `role = "speak"`, `[model.local.speak]`, catalog/1 `speak` with profiles, `InferenceHost::open_speak`,
   `Coordinator::speak`, guest `clone` grants and `guests = false`; `ymirctl status`, `pool models`, `voices`.
8. `ymirctl speak` (single piece, then the long-form plan and scripts); MCP `speak` and `voices`.
9. Bridge `/v1/audio/speech` and `/v1/audio/transcriptions` (crate update only on Ymir's side, plus docs).
10. Catalog downloads for the four models, with Drama 3's license acceptance.
11. Real runs of the four on llama-server and vLLM-Omni, the compatibility row, measured profiles.
12. Later: opt-in server-side long-form and normalization (`[media] server = true`), content-root references.

### What was built (2026-11-27, milestones 1 to 6, mocks only)

- **1 and 3.** `ymirctl transcribe` as above, plus `--chunk off|auto` (off by default), `--no-normalize` and `--out`.
  `ControlRequest::Transcribe` carries the audio as raw bytes after its JSON line (`bytes`, at most 31 MiB), not a path,
  so ymird never opens a client's file; `TranscribeTarget` reads the transcriber and its catalog/1 limits. `--chunk auto`
  refuses without a reported `audio_max_seconds` and names `limits = { audio_max_seconds = N }`.
- **2.** `weirdnet-media` 0.1.0 from crates.io (also a `[[requires]]` in `release.toml`). An interim `ymir-media` was
  written first and removed in the same branch once 0.1.0 was published. ymirctl keeps a small magic-byte check of its
  own, only to refuse non-media early and to name the format of a file sent with `--no-normalize`; ffmpeg needs ffprobe
  next to it (`Runner::locate`).
- **4.** MCP `ymir_transcribe` (read tool). "Allowed roots" are new: `ymirctl mcp --root DIR`, applied to this tool's
  `path`; without a root any file the user can read is allowed after the extension check. Chat media is unchanged.
- **5.** The mock's `/v1/audio/speech` in both flavors. It encodes WAV and PCM only (it refuses mp3, flac and opus with each
  flavor's error), which is enough for the client tests; `weirdnet-media::audio::encode` does real encoding.
- **6.** `ymirctl engines install vllm-omni` in its own venv (`<home>/engines/vllm-omni`), vLLM and vLLM-Omni pinned to one
  release. It has the PyTorch-versus-driver check and warns when vLLM's wheel is newer than the driver. The `--vllm-cuda`
  nightly variants on the unmerged M7 pass 2 branch do not apply, because vLLM-Omni needs a release.

### What was built (2026-11-27, on `weirdnet-agent` 0.5.11, before the backends)

- **open_speak** (superseded by milestone 7 below). `InferenceHost::open_speak` checks a guest first: a request that clones a voice is refused
  `guest-clone` (all guest cloning, consent or not), and a model outside the grant's allowlist `guest-model`. Anything
  else is a clean `Unsupported` that says no model in this pool speaks yet and that a `role = "speak"` node comes with
  `weirdnet-backends` 0.4.10. Because it is `Unsupported`, the pool fails over and reports it.
- **Control socket.** `ControlRequest::Speak` carries the `SpeakRequest` with the clone clips taken out. Their lengths
  are in `clips`, and the bytes follow the JSON line (each clip 1 byte to 30 MiB, 31 MiB together, checked before
  reading). The answer line (`SpeechReply`: bytes, format, rate, channels, model, node, usage, ignored) is followed by
  the audio. `SpeakTarget` routes with `Catalog::route_speak` and returns the model and its profile. It leaves out
  input length and format, because the client splits text and converts formats itself.
- **8 (partly).** `ymirctl speak [TEXT | - | --file F | --script F|-] [--voice V | --voice N=V ...] [--design "..."]
  [--clone-audio F --clone-text T --clone-consent] [--instructions ..] [--lang xx] [--speed 1.2] [--seed N]
  [--model M] [--format wav|mp3|opus|flac|pcm16] [--out F|-] [--chunk auto|off] [--pause-ms N] [--crossfade-ms N]`.
  `--ref-*` are aliases. `--stream`, `--anchor` and `ymirctl voices` are not built.
  - Raw audio is never written to a terminal. Without `--out` it needs a pipe, or `--out -`.
  - With `--chunk auto` (the default), text is planned with `weirdnet-media` `speak::TextPlan` to the profile's
    `max_input_chars`. Every piece goes to the same model with the same voice and seed, as WAV, and the pieces are
    stitched with `pcm::concat` and `joins` (300 ms pauses, 10 ms crossfades).
  - A script goes one turn at a time, each turn carrying only its speaker's voice.
  - A format the model doesn't make is asked as WAV and encoded here with `audio::encode`.
  - The summary (model, nodes, pieces, duration, usage) goes to stderr.
- **MCP `ymir_speak`.** A mutating tool, hidden under `--read-only`.
  - With `out` it needs `confirm=true`. The path must have the format's extension and sit in an existing folder under
    `--root` (links resolved), and it must not be a link or a non-file.
  - Without `out` it returns base64 audio.
  - A clone comes from `clone_path` (under `--root`, an audio extension, 30 MB at most) or from `clone_audio`
    (base64). A URL or `data:` is refused in either, and a clone needs `consent=true`.
  - It returns the path, duration, format, model, nodes, pieces and usage.
- **Tests.**
  - `ymir-testkit/tests/speak_cli.rs`: a mock `InferHost` and `SpeakBackend` served through `v2::serve` and
    `catalog::serve` on a `MemoryDialer`, with a `Pool` and the control socket. It covers single requests, PCM,
    chunking and stitching, a two-speaker script, a clone crossing the socket, MP3 via ffmpeg when present, the CLI,
    the TTY refusal, and MCP `ymir_speak` with its refusals.
  - `ymir-core/tests/pool_sharing.rs`: a real guest identity's clone is refused, and a preset gets the "speaks yet"
    answer.

### What was built (2026-11-27, on `weirdnet-openai` 0.2.5)

- **9 (bridge, Ymir side).** Bumped to `weirdnet-openai` 0.2.5 (also a `[[requires]]` in `release.toml`). The
  crate mounts `/v1/audio/speech` and `/v1/audio/transcriptions` itself, so Ymir changes no routes.
  - ymird binds the bridge through `daemon::bind_bridge`, so tests bind exactly what ymird binds.
  - `[app.bridge] max_request_bytes` (32 MiB unset) bounds audio bodies too, and the 413 carries
    `x-weirdnet-max-request-bytes`.
  - Departures from the spec above:
    - The bridge splits no long text and converts no format: over `max_input_chars`, or a format the model doesn't
      make, is a 400.
    - Per-speaker `voices` in the speech body isn't wired yet.
    - `stream_format: "sse"` is a 400.
- **Tests.**
  - `ymir-testkit/tests/bridge_audio.rs`, against the bridge ymird binds over the mock pool (`common/speech.rs`,
    shared with `speak_cli.rs`):
    - speech: headers, PCM, streaming, a clone from `data:` and a script;
    - seven 400s and the 413;
    - transcriptions as json, verbose_json and srt, plus 400s and the 413;
    - `/v1/models` `weirdnet.speech`.
  - Against a real ymird with the mock vLLM transcriber: `/v1/models`, a transcription through `InferenceHost`, the
    speech 400 (nothing speaks yet), and the 413.

### What was built (2026-11-27, on `weirdnet-model` 0.5.5)

- **10 (catalog downloads).**
  - `ymirctl models catalog [--purpose speak] [--json]` shows the eight speakers: purpose, engine, size, license,
    restriction and profile.
  - `ymirctl models pull ID` takes catalog ids. GGUF speakers download with their projector; vLLM-Omni ones download
    nothing.
  - Drama 3 needs `--accept-license fish-audio-community` (case-insensitive) for its id and both of its repos. The
    check runs before any request to the Hub or to another node, and is passed on as
    `Consent::accepting_license`.
  - With `--node`, catalog ids and `--accept-license` are refused: the remote models protocol carries no
    acceptance, so pull on the node itself. MCP never accepts a license.
- **Departures.**
  - A speaker's `[[model.local]]` entry is printed but can't be registered yet. `node.toml` validation refuses
    `role = "speak"` and catalog speaker ids until `weirdnet-backends` 0.4.10, because they would otherwise launch
    as chat models. (Lifted with milestone 7 below.)
  - `guests = false` for restricted models and the license in `ymirctl status` wait for speakers to be served
    (milestone 7, now built).
  - Ymir uses neither `LimitsRecord::speech` nor the video frame limits yet, so nothing needed reconciling there.
- **Tests.**
  - `models::tests::the_catalog_lists_the_speakers`: the eight ids, profiles, licenses, projectors, and the table
    and JSON output.
  - `drama_3_needs_its_license_accepted_before_any_request`: a refusal with no Hub request for the id and both
    repos, a wrong license refused, and acceptance reaching the Hub. It also covers the `--register` refusal, the
    Kokoro dry run and a vLLM-Omni pull.
  - `config::tests::speakers_are_refused_until_the_engines_serve_them` (replaced in milestone 7).

### What was built (2026-11-27, on `weirdnet-backends` 0.4.10: milestones 7 to 9 served end to end)

- **7 (speakers on the node).**
  - The temporary refusal is gone, in node.toml validation and in `models pull --register`. `role = "speak"` sets
    `ModelLaunch::with_speak`; validation runs `ModelLaunch::check_speech` (llama.cpp needs the projector, only
    llama.cpp and vLLM-Omni speak, no `--allowed-local-media-path`, settings only narrow the catalog entry) and refuses a
    speaker that is also an embedding, transcription, LoRA or video model, a catalog speaker id without the role, and
    `backend = "vllm-omni"` without it. A catalog speaker's projector is found in the models dir when `mmproj` is unset.
  - `backend = "vllm-omni"` is `EngineKind::VllmOmni`, run from `EngineConfig::vllm_omni` with `[model] vllm_omni`:
    `auto` uses the venv `ymirctl engines install vllm-omni` made (`<home>/engines/vllm-omni/bin/vllm`), else `vllm` on
    `PATH`; also `system:PATH` and `off`. Configured only when a model uses it.
  - `[model.local.speak]` is `weirdnet-agent`'s `SpeechProfile` (narrowing only, `ModelLaunch::with_speech`) plus
    `guests`. Speakers are open to guests by default except under a restricting license (Drama 3's
    `fish-audio-community`, from the catalog or the settings' `license`), which needs `guests = true`; `guests = false`
    closes any speaker.
  - `ModelNode::speak`: `Supervisor::speak` locally; a remote model node over infer/2 `Speak`, relayed piece by piece.
    `Coordinator::speak` picks a node whose copy speaks (ready first, else one that loads it; failover before the audio
    starts). catalog/1: purpose `speak`, the node's profile (`ModelStatus::speech`, unknown profiles left out) and the
    license. A chat or embedding request naming a speaker is a 400.
  - `InferenceHost::open_speak`: the guest checks (`guest-clone`, `guest-model`, and new `guest-license` for a speaker
    closed to guests), then `Coordinator::resolve_speaker` (`Catalog::route_speak` among the speakers the caller may
    use) and a `SpeakBackend` over `Coordinator::speak`.
  - `ymirctl status`: `speak` purpose, `speaks (text to speech): 2 voices (af_heart, am_adam), clones, 24 kHz`, the
    license, `not for guests`. There is no `ymirctl pool models` command; MCP `ymir_models` (the pool's model list) adds
    `speech`, `license`, `limits` and `takes.video`.
- **8 and 9.** `ymirctl speak`, MCP `ymir_speak` and the bridge's `/v1/audio/speech` now reach real speakers; no change to
  them beyond the routing below.
- **Departures.**
  - The design's `voices` is the profile's `presets`, and there is no `clone = false`: the profile's bools can't say "off"
    (`SpeechProfile::narrowed_by`), so turning cloning off per node waits on a Weirdnet field. Guests never clone anyway.
  - A speaker outside the catalog has no profile until its engine loads once, and `Catalog::route_speak` would give it
    only the plainest request. A request that names such a speaker is left to its node (ymird and `SpeakTarget`), which
    loads it and checks against what the engine reports; the profile is kept from then on.
  - The guest policy is the inference node's view: its own `[model.local.speak] guests`, else the license a remote
    node publishes. A remote model node's `guests = true` doesn't reach another inference node yet.
  - No per-grant `clone` switch and no `ymirctl voices` command yet.
- **Tests.**
  - `speak_e2e.rs` `speakers_in_both_flavors_through_ymirctl_mcp_and_the_bridge`: a real ymird with the mock engine as a
    llama-server speaker (projector, `/props` voices) and a vLLM-Omni speaker (`--omni`, `/v1/audio/voices`, the probe's
    sample rate, an operator `max_input_chars`). Cold load on first use, the reported profiles in `/v1/models`, `ymirctl
    speak`, MCP `ymir_speak` and `ymir_models`, the bridge in WAV, PCM and streamed, the pool's pick with no model named,
    clean 400s, and the status lines.
  - `pool_sharing.rs`: a guest gets an open speaker by name and by the pool's pick, never Drama 3 (`guest-license`), while
    the owner's devices get it.
  - Config: `speakers_launch_with_the_speak_role`, `speakers_are_checked_against_their_engines_and_catalog`,
    `restricted_speakers_are_closed_to_guests_by_default`; `models::tests::speakers_register_with_the_speak_role`.

## Not in this design

- Live speech-to-speech and realtime sessions (vLLM-Omni's WebSocket API): after batch and streaming are solid.
- Speaker diarization: no target model does it; a transcriber that reports speakers would add a `Segment` field.
- Voice libraries stored on nodes: deliberately absent (privacy decision above).
