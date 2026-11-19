# ymir-infer

**To be extracted into `weirdnet-agent`.** A trait seam, not a protocol: these are the in-process
shapes of what `/weirdnet-agent/infer/2` and the `pool` module will carry, so Ymir's roles already
talk in those terms on one box and only the transport changes in M3.

- `ChatRequest`: messages plus OpenAI-style parameters, `priority` and `job_id`.
- `Event`: `Delta` and `Reasoning` with byte offsets, `ToolCall`, `Queued`, `Done` with usage
  (the infer/2 frames).
- `Pool`: what an app node calls (catalog and chat). In M1 the inference role implements it in
  process; in M3 `weirdnet-agent`'s `pool` client implements the same calls over Weirdnet.
- `ModelNode`-side calls live in `ymir-backends` (to be `weirdnet-backends`).

License: BSD-3-Clause.
