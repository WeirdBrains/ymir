# Ymir agents

The collective `AGENTS.md` in `WeirdBrains/weirdnet-agents` is mandatory for this project and wins on
conflict. Read it first. This file only adds Ymir-specific rules.

- Ymir is an app. App crates are `ymir-*` (`ymir-core`, `ymird`, `ymirctl`) and hold only Ymir-specific
  code. Networking, auth, routing, scheduling, backends, caching, the OpenAI bridge and alerts belong
  in `weirdnet-*` crates (see `docs/DESIGN.md`, "Crates"). Never reimplement them here.
- Until the Weirdnet lead extracts it, generic alerting code lives in the interim crate
  `ymir-alerts`, marked "to be extracted into weirdnet-alerts". Keep it free of Ymir types. The
  other interim crates are retired: `ymir-hw` became `weirdnet-model`'s `hw`, `ymir-backends`,
  `ymir-openai` and `ymir-semcache` became `weirdnet-backends`, `weirdnet-openai` and
  `weirdnet-semcache`, and `ymir-infer` gave way to `weirdnet-agent`'s request types. Never create
  or modify a `weirdnet-*` crate from this repo; use the published ones as they are.
- Ymir has three roles (model, inference, app) that are independent toggles in `node.toml` and run in
  one `ymird`. Keep the protocols identical whether a peer is local or remote; the in-process path is
  only a fast path. Apps never talk to model nodes directly.
- Supported release targets: Linux x86_64 and aarch64 (plain and CUDA), Windows x86_64 and aarch64,
  FreeBSD x86_64, and macOS on Apple Silicon only (`aarch64-apple-darwin`). Intel (`x86_64-apple-darwin`)
  macOS is dropped (v0.23.0 was its last release): never add it back to the release or macOS workflows,
  `release.toml`, the update bundles, the llama.cpp engine pin or the Homebrew formula.
- Licensed BSD-3-Clause (`LICENSE`); every crate sets `license = "BSD-3-Clause"`.
- Protocol ids are `/ymir/...` for app protocols. The URL scheme is `weirdnet-ymir://`, registered in
  `weirdnet-schemes`. Never use bare `weirdnet://`.
- The reference deployment is a two-server setup (its configs are kept off-repo).
  It is read-only from this project: never change server-a, server-b or the dev box services without m3talsmith's
  explicit go-ahead for that step.
- Tests and manual runs use `YMIR_HOME` (and `WEIRDNET_MODELS_DIR`) pointed at a temp directory. Tests never
  download a model, start a GPU backend or touch the real network; real-model tests are `#[ignore]`.
- No secrets in configs, logs, docs or commits. SMTP passwords and similar live in the OS keyring or an
  encrypted store, referenced by name only.
- On the dev box, build with `CARGO_TARGET_DIR` under a scratch disk and clean targets afterwards.
  Nothing opens on the live desktop.
- Keep `docs/DESIGN.md` current in the same change as the code.
