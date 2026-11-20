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
- Linux tar.gz, zip, deb and pkg.tar.zst packages, a TAUR recipe and a Nix flake.
- llama.cpp bundled in every release file, used by `ymirctl setup` by default: `ymir` (CPU and Vulkan) and `ymir-cuda` (CUDA 13 with its runtime).
- rpm packages, a release workflow for Linux, macOS and Windows, and a smoke test that runs a real model from an unpacked archive.
