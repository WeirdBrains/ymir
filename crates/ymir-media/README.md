# ymir-media

Interim crate, **to be extracted into `weirdnet-media`** (Weirdnet lead). It holds no Ymir types and depends on no
`ymir-*` or `weirdnet-*` crate, so the swap is a rename of the dependency. See
[docs/design/audio.md](../../docs/design/audio.md), "Audio processing".

- `wav`: probe and decode RIFF/WAVE (integer PCM 8/16/24/32-bit, 32/64-bit float, `WAVE_FORMAT_EXTENSIBLE`), encode 16-bit PCM.
- `pcm`: mono down-mix, windowed-sinc resampling, `for_asr` (16 kHz mono 16-bit WAV), stitching with crossfades and pauses.
- `vad`: silence-aware chunk planning for long audio.
- `transcript`: segment re-alignment across chunks (`coarse` when a backend gave no timing).
- `subtitles`: SRT and WebVTT.
- `text`: `TextPlan` for long-form speech (paragraph, sentence, clause; script turns kept whole) and script parsing.
- `ffmpeg`: the hardened `FfmpegRunner` (local files only, `-protocol_whitelist file,pipe`, an allowlisted demuxer, wall-time,
  memory and output caps, a private temp dir removed on drop, no network namespace on Linux when allowed) and `FakeRunner`.
