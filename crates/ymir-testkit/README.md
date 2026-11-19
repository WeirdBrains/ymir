# ymir-testkit

Test support, never shipped:

- `ymir-mock-engine`: speaks enough of `llama-server`'s API (`/health` with a loading delay,
  `/v1/models`, streaming `/v1/chat/completions`, `LLAMA_API_KEY` checks) for the supervisor and
  the whole path to be tested without a GPU or a model file.
- `tests/e2e.rs`: `ymirctl setup` → `ymird` (all three roles in one process) → `ymirctl chat`
  through the loopback bridge, plus load, unload, swap groups, crash recovery, the cache and the
  bridge's guards. Everything runs in a temp `YMIR_HOME`.
- `tests/real_llama.rs` (ignored by default): the same path with a real `llama-server` and GGUF:
  `YMIR_TEST_LLAMA_SERVER=/path/llama-server YMIR_TEST_GGUF=/path/model.gguf cargo test -p ymir-testkit --test real_llama -- --ignored`.
