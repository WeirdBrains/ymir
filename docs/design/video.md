# Video through Ymir: the server and node side

Status: design, post-v1, 2026-11-22 (LF-38). Milestone 1 started 2026-11-26 (LF-43): catalog/1 reports the per-request
context (smallest across nodes, absent when any node doesn't say) and the media limits (`weirdnet-agent` 0.5.7 `MediaLimits`,
strictest across nodes, from the engine or `[model.local] limits`), `[bridge] max_request_bytes` sets the bridge's body limit,
and `ymirctl status` shows both. Milestone 2 built 2026-11-27 (mocks only): a `role = "transcribe"` model, the infer/2
`Transcribe` route through the inference role, and limits measured at load (see "Milestone 2" below). A video part still waits
on the Weirdnet wish list below. None of it blocks v1. The client side (sampling, attach,
players in all three front ends, ffmpeg packaging) is in [`bragi/docs/design/video.md`](https://github.com/WeirdBrains/bragi/blob/main/docs/design/video.md);
read it first. This document covers what Ymir does so that video works for Bragi and for every other client of the pool.

## Decisions (2026-11-22, m3talsmith)

- **No default frame count or frame size, in Ymir or in Bragi.** Ymir reports what each model supports; clients plan frames to fit
  (Bragi doc, "Fitting the clip to the model"). Ymir must therefore never invent a number for a model it knows nothing about: a missing
  field is reported as missing (absent), not filled with a default, so the client can refuse clearly.
- The Bragi desktop apps (GPUI on Linux, macOS and Windows) get an inline player with a system-player fallback; Flutter is the phones only.
  That is client-side and does not change anything here.

## What exists today

- Message content is text or parts: `text`, `image_url` (data URL only) and `input_audio` (`weirdnet-agent` 0.5, infer/2).
  `Modalities` is `VISION` and `AUDIO`; there is no video.
- Routing is by capability: each model's `Capabilities` (from its node record) feed catalog/1 `inputs` (`vision`, `audio`, the union over
  nodes that have the model). `Catalog::route` picks a node whose copy of the model takes what the request carries, and a named model
  that cannot is a 400 ("does not take images"). Answers to requests with media are not cached.
- `weirdnet-openai` passes parts through the bridge, `weirdnet-backends` forwards them to `llama-server` unchanged. See DESIGN.md,
  "Images and audio through the pool".
- llama.cpp has no video input. mistral.rs has `video_url` and `audio_url` parts, but Ymir does not send them and the matrix in
  DESIGN.md shows llama.cpp with a projector covers more than mistral.rs did.

## Phase 1: nothing to build in Ymir for Bragi

Bragi expands a clip into timestamped `image_url` parts and an `input_audio` part (or a transcript) before sending. To Ymir that is
an ordinary multimodal request, so routing, the bridge, caching rules and the backends already do the right thing. What Ymir
*does* need for phase 1 to be comfortable:

- **Limits per model, reported by Ymir.** This is the core of phase 1 on the server side, because the client has no defaults to fall
  back on. For every model on a node, `Capabilities` and catalog/1 (the union over nodes, using the most restrictive copy that the
  request could be routed to, or per node when the request names one) report, when known:
  - the context window the node actually runs the model with (`--ctx-size`, per slot), not the model's trained maximum;
  - `max_images` per request (llama.cpp takes many, mistral.rs refused two in one message);
  - the tokens one image costs: a fixed number, or a function of size (from the projector and the model's own settings), and the preferred or
    maximum image edge;
  - audio: accepted or not, and the maximum length;
  - later, native video limits (below).
  Where the node can measure a limit (a projector's token count for a test image, the real `--ctx-size`), it does so once at model load and
  records it. Where it cannot, the field is **absent**, never defaulted. `ymirctl models` shows them and says "not reported" for the gaps.
- **Request size.** Check and document the largest request the bridge, the coordinator and infer/2 accept, and make the limit a setting
  (`[bridge] max_request_bytes`) with a clear 413 that names it. It is reported in catalog/1 too, so a client sizes frames to fit the
  bytes as well as the tokens.
- **Operators can set a limit** a node cannot measure (`[[models]] image_tokens = N`, `max_images = N`) in `node.toml`; that is the
  operator's number for their hardware, not a default in the code.
- **Transcription.** A model with a `transcribe` role (Whisper-class, through llama.cpp or a small ASR backend) so the soundtrack can
  become text for models without audio. The catalog lists it and `Catalog::route` picks it for a transcription request. This is the
  one new route in phase 1, and it is optional: without it, the sound is simply left out.

## Phase 2: native video

Models that read video exist (Qwen-VL family, and others). They are fed video as sampled frames plus temporal position information,
or through a backend's own video path. When a backend takes the clip itself, sending it whole is better than the client's frames
(model-matched sampling, temporal encoding, no double work). The model's native video limits are what bound the clip, as image limits bound frames.

1. **Shared types** (`weirdnet-agent`, wish list): `ContentPart::video` and `Modalities::VIDEO`. Until it exists, nothing is sent.
2. **Capability flag.** `Capabilities.video: bool` in the node record and `inputs.video` in catalog/1 (a node that lacks the field
   counts as no video, as for audio today). Optional detail: `video_max_seconds`, `video_max_bytes`, `video_frame_rate`; they follow the same rule, reported or absent. `ymirctl models`
   and `ymirctl nodes` show it.
3. **Routing.** The coordinator treats a request with a video part like one with images: only to a node whose model reads video; with no
   model named, the main model if it does, else the best chat model in the pool that does; a named model that does not is a 400 that
   says so *and tells the client the fallback*: send frames. Bragi on that 400 falls back to phase 1, planning frames from the same reported limits.
4. **Backends.** `weirdnet-backends` forwards `video_url` to a backend that declares video input (the way it declares images today). For
   llama-server, only once upstream has it (watch the multimodal work; nothing to build until it ships). For mistral.rs or another engine that has
   a video part, a mapping behind a feature flag, checked by the same compatibility matrix in DESIGN.md (one row for video).
5. **Large clips.** Video does not fit in a request. The real answer is content roots (`weirdnet-net` `content`): the part carries a
   reference, and the node fetches the clip from the client over the existing encrypted link. This is the same gap that limits attachments
   today (LF-12). Until it exists, native video is limited to small clips (the request limit above).
6. **Caching.** As with images and audio, answers to requests with video are not cached.

## Server-side expansion (opt-in, later)

For clients that speak only the OpenAI API (Cursor, scripts, anything through the bridge) a `video_url` part can be accepted and expanded by
the bridge when the model does not read video natively: the same sampler Bragi uses, taken from `weirdnet-media` (wish list), not copied.

- Off by default; a node operator turns it on in `node.toml` (`[bridge.video] expand = true`), because it makes the node run ffmpeg on untrusted
  input and use CPU on an inference machine.
- Only `data:` URLs and uploaded bytes. Remote `https://` URLs are not fetched, as for images (no server-side request forgery, nothing
  fetched on a client's behalf).
- Which node does it: the bridge's own node, in a child process with limits (time, memory, `-protocol_whitelist file,pipe`, no network),
  never the inference node that holds the model if they differ. Output is bounded by the frame budget computed from the target model's reported limits, exactly as on the client; with no limits reported, expansion refuses with a 400 that says which are missing, and the operator can set them in `node.toml`.
- The node needs ffmpeg: not bundled with `ymird` by default. A node with expansion on and no ffmpeg refuses video with a 400 that says
  why; `ymirctl doctor`-style status shows it. Linux packages `Recommends` it; Ymir's other platform packages follow the Bragi table, but
  the server is mostly Linux, so no mobile case.
- Privacy: the raw clip reaches the node. That is why this is opt-in and why Bragi samples on the device by default.

## MCP exposure (`ymirctl mcp`)

`ymirctl mcp` already offers chat with images and audio. With video:

- `chat` takes a `video` input (path under the allowed roots, or base64). The tool samples locally with `weirdnet-media` (a client of the
  pool, so the same as Bragi's phase 1), or sends natively when the routed model has the video flag. The client never has to know which.
- Read-only helpers: `video_probe` and `video_frames` (frames as image content blocks with timestamps).
- `models` and `nodes` show the video flag.
- All keep `--read-only` and the `confirm` gates; reading a file outside the roots is refused, a URL is never fetched.

## Privacy

- Phase 1 sends no video to Ymir, only the frames and sound the user chose; the node sees exactly what the model sees.
- Frames are re-encoded, so no container or EXIF metadata travels. Nodes do not log content parts (existing rule); this also holds for
  frames and transcripts, and a transcript is content.
- Nothing is cached for requests with media, and a node never writes a received clip to disk except in a private temp file for the length of
  one expansion or backend call.
- Native video and expansion put the whole clip on a node. Both are opt-in or per-request, and Bragi says which will happen before sending.

## Testing

- Routing tests in `tests/nodes.rs` style: a request with a video part goes only to a node whose model has the flag; a 400 names the
  fallback when none does; a node record without the field counts as no video.
- Bridge tests in `tests/vision.rs` style with the in-process fake backend: a `video_url` part is refused, passed through, or expanded
  according to the flag and the setting; the request-size limit gives a 413 that names the setting.
- Expansion with a fake `FrameSource` (no ffmpeg), plus ffmpeg-backed tests marked `#[ignore]` unless it is installed; the hostile-file
  cases from the Bragi doc (a text file named `.mp4`, a truncated file, a `concat` playlist) must fail closed.
- The compatibility matrix in DESIGN.md gets a video row, filled in by a real run (llama.cpp with a projector and frames, a native
  engine if one is enabled), as with the existing matrix. Tests never download a model, start a GPU backend or touch the real network;
  real-model tests stay `#[ignore]`.
- `ymirctl mcp` tests: a scripted stdio session calling `video_probe` and `video_frames` on a generated fixture.

## Milestones

1. Per-model limits in `Capabilities` and catalog/1 (context actually in use, `max_images`, per-image token cost and edge, audio length) plus the request-size setting, with the operator overrides. Small, useful for images today too, and **the prerequisite for Bragi's video**.
2. Transcription role and route.
3. The `video` modality and capability flag once `weirdnet-agent` has them; routing and the 400 fallback hint.
4. Backend pass-through when a backend takes video.
5. Opt-in server-side expansion and the `ymirctl mcp` video inputs, after `weirdnet-media`.

## Milestone 2: what was built (2026-11-27, mocks only)

- **Config.** `role = "transcribe"` on a `[[model.local]]` sets `ModelLaunch::transcribe` (`weirdnet-backends` 0.4.9). The engine
  decides whether it can serve it (`Capabilities::transcribe`: llama.cpp with a projector, or vLLM); MLX is refused at
  validation. A transcriber is never picked as the main chat model and gets no stand-in.
- **Catalog.** The merged catalog marks a model `transcribe` when any node's copy transcribes, and catalog/1 publishes
  `Purpose::Transcribe` (hidden from peers before `weirdnet-agent` 0.5.8 by the crate). `ymirctl status` says
  `transcribes (speech to text)`, `ymirctl pool models` shows the purpose.
- **Route.** `InferenceHost::open_transcribe` resolves the model with `Catalog::route_transcribe` (a named model that does not
  transcribe, or a pool with none, is a 400) and `Coordinator::transcribe` picks a node whose copy transcribes (ready first,
  else one that loads it; draining, reserved and leaving nodes skipped; retryable failures go to the next node). Local nodes
  call `Supervisor::transcribe`, remote ones infer/2 `Transcribe` on the model node's `ModelHost`. Callers are admitted by the
  same `authorize` as chat (a guest of a shared pool, M11, only gets transcribers in its grant's model allowlist); batch priority takes the same per-device hold. Nothing is cached; logs carry the model, the node and
  the byte count, never the audio or the transcript.
- **Measured limits.** ymird hands the models dir to `Supervisor::set_limits_store`, so GGUF and projector figures measured at
  load are kept next to the model and read back after a restart. The context the engine reports at load
  (`ModelStatus::context_tokens`) is preferred over the configured context per slot for catalog/1 `context` and
  `ymirctl status` (`(measured)`). The operator's `limits` win field by field (in the supervisor), and a field nobody knows
  stays absent.
- **Not built.** The OpenAI bridge has no `/v1/audio/transcriptions` (`weirdnet-openai` 0.2.4 has none: wish list), and there is
  no `ymirctl transcribe` yet. Not tried against a real Voxtral or Whisper engine.
- **Tests.** [`transcribe.rs`](../../crates/ymir-testkit/tests/transcribe.rs): the mock engine as vLLM, catalog purpose and
  limits before and after load (measured `max_model_len`, the operator's audio limit kept, unknown fields absent), transcription
  through `local_pool` with timed segments, not cached, named-model and empty-pool 400s, and no audio or transcript in the logs.

## Weirdnet wish list

The same list as in the Bragi doc, with the Ymir-specific items first:

1. **`weirdnet-agent`**: `ContentPart::video`, `Modalities::VIDEO`, `Capabilities.video`, and the limit fields (`context_tokens`, `max_images`,
   per-image token cost by size, preferred image edge, audio maximum length, `video_max_seconds`, `video_max_bytes`), each optional and
   absent when unknown, never defaulted; catalog `inputs.video` and the same limits per model; infer/2 carrying a video part, ideally by content-root reference.
2. **`weirdnet-openai`**: accept and map `video_url`, with a hook for the expansion step; a 413 that names the configured limit;
   `/v1/audio/transcriptions` mapped to infer/2 `Transcribe` (milestone 2 leaves the bridge without it).
3. **`weirdnet-backends`**: forward video to engines that take it, a transcription call, and the compatibility row.
4. **`weirdnet-media`** (new): probe, `FrameSource` (ffmpeg CLI), frame planning (uniform, scene, hybrid), scaling and encoding, soundtrack
   extraction, hardened ffmpeg invocation, test fake. Mouser's `mouser-media` is the nearest code.
5. **`weirdnet-model`**: a speech-to-text catalog entry with a `transcribe` role; the per-model limits above (measured at load where possible, so the node can fill them without the operator).
6. **`weirdnet-mcp`**: image and audio content blocks in tool results.
7. **An LGPL ffmpeg build recipe** shared by the apps.
