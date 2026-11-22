# Changelog

All notable changes to Ymir are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

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
- `ymirctl eval`: score models on a held-out JSONL file with the bench scorers (`expect`) and an optional `--judge` model for rubrics.
- `ymirctl mcp`: an MCP server for Cursor, Claude Desktop and VS Code over stdio or loopback streamable HTTP. Tools for status, nodes, models, chat with images and audio, completions, embeddings, model search and the service; load, unload and pull need `confirm: true`, and `ymir_remove_model` needs `--allow-remove` too. Built on `weirdnet-mcp`.
- `ymirctl models search | pull | list | remove --node <name|peer-id|auto|all>`: the same on the model nodes among your devices over Weirdnet (`/weirdnet-backends/models/1`, owner only). `auto` picks the node that fits best, a pull streams its progress, `pull --serve` serves the model from the node's running `ymird` without a restart (in memory; `node.toml` is not edited remotely), and `remove` stops serving and deletes. `[model] remote_manage = false` turns it off on a node.
