# Changelog

All notable changes to Ymir are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-11-23

### Changed

- Models run with as much context as they can take: ymird gives each model its full trained context (GGUF header, or `config.json` with RoPE scaling for vLLM) and cuts it only where the KV cache would not fit beside the weights and the node's other models, saying why in the log. Sliding-window layers, looped layers and the KV cache type are counted. An explicit `context` is kept. `ymirctl models pull` no longer writes `context = 32768` (or `--max-model-len 262144` for Inkling on a Spark), and setup leaves llama.cpp contexts to ymird. The catalog advertises the context each server is started with.

### Added

- `ymirctl models context` shows each model's context and why; `--auto ID` or `--old-defaults` lifts contexts that older pulls wrote by default.
- Second public release: models get the largest context that fits.

## [0.1.0] - 2026-11-22

### Added

- First-class Inkling 2 Mini support: `ymirctl models pull thinkingmachines/Inkling-2-Mini` registers a vLLM Hub id (with Inkling reasoning/tool parsers, DSpark, the `vllm/vllm-openai:inkling-2-mini` image, and a 262144 context cap on DGX Spark / GB10); `pull ggml-org/Inkling-2-Mini-GGUF` also fetches `mmproj-Inkling-2-Mini-Q8_0.gguf` (any `mmproj-*.gguf` in the same repo is pulled this way). `ymirctl models search` lists safetensors/vLLM repos on vLLM-capable nodes (`--format vllm` or `--format all`). ymird refuses to launch Inkling on a vLLM older than 0.35.0 unless the node uses that image or a wrapped server. MLX is not supported until mlx-lm#2086 lands in a release; pull says so.
- `[model] vllm = "image:registry/name:tag"` and `[[model.local]] image` run vLLM through docker or podman (host networking, `--entrypoint vllm` so the supervisor's `vllm serve` still works).
- Bundled llama.cpp is pinned at b12628 (Inkling 2 Mini GGUF support, including the NVFP4 loop-scale fix from ggml-org/llama.cpp#32704).
- One `ymird` for the model, inference and app roles, with `ymirctl` setup, status, load, unload and chat.
- Loopback OpenAI bridge: `/v1/models`, `/v1/chat/completions`, `/v1/completions` and `/v1/embeddings`.
- llama.cpp and vLLM engines, swap groups, idle unload, and crash restarts with backoff.
- Wrapping a stock OpenAI-compatible server, such as `vllm serve`.
- Stand-in answers, flagged, while a large model loads.
- Health alerts by email, webhook and log, including the default route on Wi-Fi.
- User-level service install (systemd user unit, launchd agent).
- `ymird` runs on Windows, as a logon task (user level) or as the Windows service `NT SERVICE\ymird` (`--system`).
- `ymirctl service start` and `stop`, and `--system` for a systemd system unit or launchd daemon.
- Windows MSI and `YmirSetup` installers that register the Windows service.
- Linux tar.gz, zip, deb and pkg.tar.zst packages, a TAUR recipe and a Nix flake.
- llama.cpp bundled in every release file, used by `ymirctl setup` by default: `ymir` (CPU and Vulkan) and `ymir-cuda` (CUDA 13 with its runtime).
- rpm packages, a release workflow for Linux, macOS and Windows, and a smoke test that runs a real model from an unpacked archive.
- `ymirctl models search`, `pull`, `list` and `remove`: find a model on Hugging Face with a fit verdict for this node, download it (resumable, checksum-verified, split GGUFs, optional token), and print or register its `[[model.local]]` entry.
- `ymirctl batch --jsonl`: data generation through the pool. JSONL rows with their own messages, system prompt, sampling and seed; the teacher model waits to load (no stand-in); answers go to `--out` with `usable`, `cache_hit` and `standin` per row and a manifest with the teacher's license (`--license` required); `--resume` continues an interrupted run; `--sources` records a sources manifest (topic, URLs and licenses) and the source of every row.
- Requests can skip or refresh the answer cache (infer/2 `cache`, `weirdnet-agent` 0.5.1): `batch --jsonl` bypasses it by default (`--cache refresh|use`) and `eval` always does; the bridge takes `"cache": false|"refresh"` or `x-weirdnet-cache: off|refresh`.
- `[[model.local]] revision`: the weights revision goes in the capability record and keys the answer cache, so new weights under the same model id never replay old answers; answers report it in their metadata.
- `ymirctl eval`: score models on a held-out JSONL file with the bench scorers (`expect`) and an optional `--judge` model for rubrics.
- `ymirctl mcp`: an MCP server for Cursor, Claude Desktop and VS Code over stdio or loopback streamable HTTP. Tools for status, nodes, models, chat with images and audio, completions, embeddings, model search and the service; load, unload and pull need `confirm: true`, and `ymir_remove_model` needs `--allow-remove` too. Built on `weirdnet-mcp`.
- `ymirctl models search | pull | list | remove --node <name|peer-id|auto|all>`: the same on the model nodes among your devices over Weirdnet (`/weirdnet-backends/models/1`, owner only). `auto` picks the node that fits best, a pull streams its progress, `pull --serve` serves the model from the node's running `ymird` without a restart (in memory; `node.toml` is not edited remotely), and `remove` stops serving and deletes. `[model] remote_manage = false` turns it off on a node.
- First public 0.1.0 release.
