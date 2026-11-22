# Ymir roadmap

<!-- Generated from roadmap.toml by site/tool/gen_roadmap.py. Edit roadmap.toml, not this file. -->

Ymir is built in milestones. Each one lands its Weirdnet crate work first, published on crates.io, and then the Ymir side. Status comes from the design doc.

Updated 2026-11-22. Design: [docs/DESIGN.md](https://github.com/WeirdBrains/ymir/blob/main/docs/DESIGN.md).

## M0 Design (Done)

The repo, the names, the three roles and the decisions, written down in the design doc and reviewed by the Weirdnet lead.

Weirdnet crates: none new.

## M1 One box, three roles in one daemon (Done)

One machine runs the model, inference and app roles in a single `ymird`. It detects the hardware, picks models that fit, runs llama.cpp, and serves an OpenAI-compatible endpoint on loopback so Jan and scripts work. `ymirctl setup` writes the config.

Weirdnet crates: `weirdnet-model`, `weirdnet-backends`, `weirdnet-openai`, `weirdnet-semcache`.

## M2 Model-node parity with existing llama-swap servers (Done)

vLLM as a second engine (managed, or wrapping a server that is already running), idle unload, restart with backoff after a crash, a small stand-in model that answers while a big one loads, alerts, a user service and Linux packages.

Weirdnet crates: `weirdnet-backends`, `weirdnet-alerts`, `weirdnet-model`.

## M3 Roles over Weirdnet (Done)

The same three roles talk across devices over Weirdnet. An app node reaches inference nodes, inference nodes drive model nodes, and only devices in your own roster are let in. `ymirctl` and Jan work on another machine through that machine's own loopback endpoint.

Weirdnet crates: `weirdnet-agent`, `weirdnet-net`, `weirdnet-sync`, `weirdnet-device-link`, `weirdnet-backends`, `weirdnet-model`.

## M4 Several model nodes (Done)

More than one model node in the pool: linking devices, each node advertising what it can run, one combined model list, failover before the first token, and draining a node before maintenance.

Weirdnet crates: `weirdnet-device-link`, `weirdnet-relay`, `weirdnet-net`, `weirdnet-agent`, `weirdnet-backends`.

## M5 Context sync and failover (Done)

Conversations are kept in sync across inference nodes, so a reply can resume mid-answer after a model node or an inference node dies, with the full context and no repeated text.

Weirdnet crates: `weirdnet-agent`, `weirdnet-sync`.

## M6 Workload spreading (In progress)

Load balancing across replicas, fair sharing between devices, priority classes and batch jobs that fan out across nodes are built. Next: moving models between nodes, placement, and a shared answer cache.

Weirdnet crates: `weirdnet-agent`, `weirdnet-backends`, `weirdnet-semcache`.

## M7 Split models (In progress)

A model too big for one machine is split across several with llama.cpp RPC: an opt-in pool policy, private-link-only planning, group up and down, and teardown when a member is lost. Next: vLLM with Ray groups.

Weirdnet crates: `weirdnet-backends`.

## M8 Move existing servers onto Ymir (Done)

The servers that served models before now run on Ymir. The move was made one step at a time, with the owner approving each step.

Weirdnet crates: none new.

## M9 Breadth and hardening (In progress)

Built: self-update with rollback (pinned root, signed bundles, trial boots), agent tasks (task/1) for Bragi's agent mode, the Windows service, a security review with checks that run, and a `[model] callers` guard so a model node can refuse app devices. Next: push wake-ups, BSD packages and a bundle fetcher; MLX-LM for Inkling 2 Mini once mlx-lm#2086 is in a release (vLLM and llama.cpp already serve it).

Weirdnet crates: `weirdnet-update`, `weirdnet-push`, `weirdnet-agent`.

## M10 Cert-only server link (Planned)

Service nodes get their own certificate and sync keys without holding the root seed, so one node can be revoked without rotating your identity. The design is approved and the Weirdnet crates are published; Ymir wires it up next. A majority-vote lease follows as weirdnet-sync 0.8.0.

Weirdnet crates: `weirdnet-sync`, `weirdnet-device-link`.
