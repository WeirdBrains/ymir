<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="brand/svg/ymir-lockup.svg">
    <img src="brand/svg/ymir-lockup-light.svg" alt="Ymir" width="360">
  </picture>
</p>

# Ymir

**A cloud of LLM models: a private, self-hosted model cloud.** Start with one machine. Add
more over Weirdnet. The pool acts as one endpoint, and the models are spread across it.

> Status: M2 (2026-11-19): one box runs all three roles in one `ymird`, with llama.cpp and vLLM
> (managed or a wrapped stock server), stand-in cold start, idle unload, crash backoff, alerts and a
> user service. Roles over Weirdnet (M3), more machines and context sync come next. Read [docs/DESIGN.md](docs/DESIGN.md).

Ymir turns a computer with a decent GPU (a DGX Spark, a gaming PC, a Mac, a workstation, or a
CPU-only box) into part of your own model cloud with one command. Machines join over
[Weirdnet](https://net.weirdbrains.com) peer to peer by being linked to your Weirdnet identity:
no open HTTPS port, no reverse proxy, no API keys. Only your own linked devices can use the pool.

Three roles, placed freely (all on one box, or spread over any number of devices):

- **Model nodes** run the models (llama.cpp everywhere, vLLM where supported, or a stock server
  wrapped), advertise what they have, and load or unload on command.
- **Inference nodes** coordinate: one model catalog, load balancing, placement and cloud-wide model
  swapping, automatic mid-answer failover, and conversation context synced with `weirdnet-sync`.
  Run more than one for redundancy.
- **App nodes** ask for models and send prompts: Bragi, `ymirctl`, the loopback-only OpenAI bridge
  for Jan and scripts on `127.0.0.1`, or any `weirdnet-agent` app.

Redis, Tailscale and a second machine are all optional. Models too big for one machine can be split
across several (big mode, generalized).

The chat client, **Bragi**, has its own repo. Its desktop (GPUI) and phone (Flutter) apps work exactly
the same way: both get their models from the Ymir cloud and send prompts across it, through one
shared Rust core. Only the UI shell differs.

Generic pieces live in `weirdnet-*` crates (owned by the Weirdnet lead); this repo holds only the
Ymir-specific app.

## Try it (from source)

Needs Rust 1.99 and a `llama-server` from [llama.cpp](https://github.com/ggml-org/llama.cpp).
On an NVIDIA PC, take the release's `ubuntu-cuda-12.8-x64` build and unpack its matching `cudart-`
archive into the same directory (no CUDA toolkit needed); llama.cpp offloads all layers that fit by
default, so `node.toml` needs only `llamacpp = "system:/path/to/llama-server"`.

```sh
cargo build --release -p ymird -p ymirctl
ymirctl hw                                   # what this machine has and the model budget
ymirctl setup --llama-server /path/to/llama-server --model /path/to/model.gguf
#   or: ymirctl setup --download             # catalog models that fit, asks before downloading
ymird &                                      # model, inference and app roles in one process
ymirctl chat "Hello"                         # streams through the loopback bridge
ymirctl status                               # roles, models and their state
ymirctl load <model> / ymirctl unload <model>
ymirctl bridge token                         # API key for Jan: base URL http://127.0.0.1:8484/v1
ymirctl service install                      # or `setup --service`: a systemd user unit / launchd agent
```

The bridge serves `/v1/models`, `/v1/chat/completions`, `/v1/completions` and `/v1/embeddings`.
While a big model loads, a stand-in can answer (flagged with `x-weirdnet-standin`; send
`x-wait: true`, or `ymirctl chat --wait`, to wait for the real model). Some `node.toml` settings:

```toml
[[model.local]]
id = "qwen3-8b"
swap_group = "main"
idle_ttl = "10m"                      # unload when idle

[[model.local]]
id = "smollm2-360m"
role = "standin"                      # answers while the main model cold-starts

[[model.local]]
id = "qwen2.5-7b"
backend = "vllm"
wrap = "http://127.0.0.1:8000/v1"     # a stock server Ymir didn't start (loopback only)
wrap_api_key = "env:VLLM_API_KEY"     # secrets are env: or file: references, never inline

[inference.alerts]
dry_run = true                        # log what would be sent
[inference.alerts.smtp]
server = "smtp.example.com"
from = "Ymir <alerts@example.com>"
to = ["you@example.com"]
password = "file:/home/you/.config/ymir/smtp.pass"
```

A chat GGUF used for embeddings needs `args = ["--pooling", "mean"]` on llama-server.

Linux packages (tar.gz, zip, .deb, and the `ymir-bin` TAUR package) are built with
`scripts/package.py` from release binaries; see `release.toml` for the platform matrix.

`YMIR_HOME` moves the config and state (default `~/.config/ymir`); `WEIRDNET_MODELS_DIR` moves the
model files.

Brand assets and rules are in `brand/` and [BRAND.md](BRAND.md). The roadmap is [ROADMAP.md](ROADMAP.md), generated from `roadmap.toml` with the website in `site/` (Jaspr; `python3 site/tool/gen_roadmap.py` after editing).

Crates: `ymir-core`, `ymird`, `ymirctl` (the app), `ymir-testkit` (tests). Backends, the OpenAI bridge, the answer
cache, alerts, hardware detection and the request types come from the published `weirdnet-*` crates.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
