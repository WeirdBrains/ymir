<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="brand/svg/ymir-lockup.svg">
    <img src="brand/svg/ymir-lockup-light.svg" alt="Ymir" width="360">
  </picture>
</p>

# Ymir

**A cloud of LLM models: a private, self-hosted model cloud.** Start with one machine. Add
more over Weirdnet. The pool acts as one endpoint, and the models are spread across it.

> Status: M2 to M5 and M8 done; M6, M7 and M9 in progress; M10 next (see the [roadmap](ROADMAP.md) and the [design doc](docs/DESIGN.md)).
> One box runs all three roles in one `ymird`, with llama.cpp and vLLM (managed or a wrapped stock server), stand-in
> cold start, idle unload, crash backoff, alerts and a service. The roles talk over Weirdnet across devices, with several
> model nodes, context sync and mid-answer failover, and the servers that served models before now run on Ymir.

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

## Try it

Release files (tar.gz, zip, deb, rpm, Arch, Nix) include llama.cpp, so a server install is: unpack or install the
package, `ymirctl setup --download --yes --service`, done. There are two builds: `ymir` (CPU and Vulkan) and
`ymir-cuda` (CUDA 13 with its runtime, for NVIDIA; needs only the driver). On Apple Silicon Macs, chat models run on
MLX instead: `setup --download` installs MLX-LM and the MLX weights (`--backend llamacpp` keeps llama.cpp). The
[deployment guide](https://ymir.weirdbrains.com/docs/guides/deployment/) has the details.

[Ymir 0.3.1](https://github.com/WeirdBrains/ymir/releases/tag/v0.3.1) is out, with downloads for Linux, macOS and
Windows, and `ymir-bin` / `ymir-cuda-bin` on [TAUR](https://pkg.taur.weirdbrains.com/packages/ymir-bin). It sizes
each model's context to what fits beside the others, and 0.3.1 fixes the plan so it counts both attention score
matrices llama.cpp keeps (Inkling 2 Mini no longer runs a node out of memory). To build from source (Rust 1.99), put
the pinned llama.cpp build (`packaging/engines/llama.cpp.toml`) next to the binaries, where `ymird` looks for it.
`python3 scripts/package.py [--variant cuda]` builds the release files themselves into `dist/`.

```sh
cargo build --release -p ymird -p ymirctl
python3 scripts/engine.py fetch --target x86_64-unknown-linux-gnu --out target/release/llama.cpp   # --variant cuda for NVIDIA
ymirctl hw                                   # what this machine has and the model budget
ymirctl setup --model /path/to/model.gguf    # or --llama-server PATH for your own llama.cpp
#   or: ymirctl setup --download             # catalog models that fit, asks before downloading
ymird &                                      # model, inference and app roles in one process
ymirctl chat "Hello"                         # streams through the loopback bridge
ymirctl status                               # roles, models and their state
ymirctl load <model> / ymirctl unload <model>
ymirctl mcp                                  # MCP server for Cursor, Claude Desktop and VS Code (stdio; --http for loopback HTTP)
ymirctl bench                                # compare every model on the pool: speed and quality, saved as JSON and Markdown
ymirctl bridge token                         # API key for Jan: base URL http://127.0.0.1:8484/v1
ymirctl service install                      # or `setup --service`: systemd user unit, launchd agent or Windows logon task
ymirctl service stop / start / status        # --system: a system unit, launchd daemon, the Windows service or a FreeBSD rc.d script
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

[inference.placement]                 # optional: keep these models loaded on this many model nodes
replicas = { "qwen3-8b" = 2 }

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
