# ymir-openai

**To be extracted into `weirdnet-openai`.** The loopback-only OpenAI-compatible bridge, an app
node over a `ymir_infer::Pool` (in M3, `weirdnet-agent`'s pool client). Generic: any Weirdnet app
can put it in front of its pool.

- Binds `127.0.0.1:<port>` only. There is no address option and no other listener.
- A token is required on **every** endpoint (`/v1/models` and `/metrics` included), compared in
  constant time. The caller supplies it; keeping it in the OS keyring is the app's job.
- DNS-rebinding guard: the `Host` header must be `127.0.0.1:<port>` or `localhost:<port>`.
- No CORS headers.
- `/v1/models` (the pool catalog), `/v1/chat/completions` (SSE streaming and plain JSON, tool
  calls and reasoning passed through), `/metrics`.
- `x-weirdnet-cache: hit`, `x-weirdnet-standin: <model>` and `x-weirdnet-model: <model>` response
  headers; `Busy` is a 503 with `Retry-After`.
- `/v1/completions` and `/v1/embeddings` answer 501 in M1.

License: BSD-3-Clause.
