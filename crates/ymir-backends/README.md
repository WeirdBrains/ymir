# ymir-backends

**To be extracted into `weirdnet-backends`** (the model-node wrapper). Generic: it knows engines
and processes, not Ymir's coordinator. It decides nothing cloud-wide; it does what it is told.

What is here in M1:

- Engine adapters: llama.cpp `llama-server` (launch arguments, `/health` readiness, OpenAI SSE
  stream, abort by dropping the request) and vLLM (`vllm serve` launch arguments and the same
  probes; not exercised yet since no supported vLLM install was available).
- The local supervisor: `load` (spawns the engine on a random loopback port with a per-process API
  key passed by environment, waits for readiness), `unload` (drains in-flight requests, SIGTERM to
  the engine's process group, SIGKILL after 10 s), swap groups (loading a model unloads the others
  in its group), crash detection (an engine that exits is marked `Failed`; the next `load`
  restarts it), per-model logs.
- `chat`: streams `ymir_infer::Event`s from the engine.

Not yet (M2 and later, per the design): idle TTL, restart with backoff, slot save and restore,
pinned engine builds through `weirdnet-model`'s download and consent, wrapping a stock server by
URL, Ray and `rpc-server` adapters, and the `/weirdnet-backends/node/1` protocol.

License: BSD-3-Clause.
