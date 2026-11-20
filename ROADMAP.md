# Ymir roadmap

<!-- Generated from roadmap.toml by site/tool/gen_roadmap.py. Edit roadmap.toml, not this file. -->

Ymir is built in milestones. Each one lands its Weirdnet crate work first, published on crates.io, and then the Ymir side. Status comes from docs/DESIGN.md.

Updated 2026-11-19. Design: [docs/DESIGN.md](https://github.com/WeirdBrains/ymir/blob/main/docs/DESIGN.md).

## M0 Design (Done)

The repo, the names, the three roles and the decisions, written down in docs/DESIGN.md and reviewed by the Weirdnet lead.

Weirdnet crates: none new.

## M1 One box, three roles in one daemon (Done)

One machine runs the model, inference and app roles in a single `ymird`. It detects the hardware, picks models that fit, runs llama.cpp, and serves an OpenAI-compatible endpoint on loopback so Jan and scripts work. `ymirctl setup` writes the config.

Weirdnet crates: `weirdnet-model`, `weirdnet-backends`, `weirdnet-openai`, `weirdnet-semcache`.

Known gaps:
- The quickstart has run on one NVIDIA PC only. A Spark, a Mac and a CPU-only box are not tried yet.

## M2 Model-node parity with server-a and server-b (Done)

vLLM as a second engine (managed, or wrapping a server that is already running), idle unload, restart with backoff after a crash, a small stand-in model that answers while a big one loads, alerts, a user service and Linux packages.

Weirdnet crates: `weirdnet-backends`, `weirdnet-alerts`, `weirdnet-model`.

Known gaps:
- vLLM is tested on an NVIDIA GPU only, not on the Spark's newer GPU.
- Nothing is released yet: no rpm, no signatures, no macOS or Windows builds.
- Alerts have only run in dry-run, so nothing real was ever sent.

## M3 Roles over Weirdnet (Done)

The same three roles talk across devices over Weirdnet. An app node reaches inference nodes, inference nodes drive model nodes, and only devices in your own roster are let in. `ymirctl` and Jan work on another machine through that machine's own loopback endpoint.

Weirdnet crates: `weirdnet-agent`, `weirdnet-net`, `weirdnet-sync`, `weirdnet-device-link`, `weirdnet-backends`, `weirdnet-model`.

Known gaps:
- Nothing is tested across real machines yet. Every multi-node test ran on one computer (in-memory and loopback).
- Linking the owner's real Weirdnet identity is still to do.
- One roster-sync test fails about half the time, a known upstream issue.

## M4 Several model nodes (Done)

More than one model node in the pool: linking devices, each node advertising what it can run, one combined model list, failover before the first token, and draining a node before maintenance.

Weirdnet crates: `weirdnet-device-link`, `weirdnet-relay`, `weirdnet-net`, `weirdnet-agent`, `weirdnet-backends`.

Known gaps:
- The cross-machine run on the laptop did not happen (the machine was busy). Every multi-node test ran on one computer.
- Routing is ready node first, then fewest requests in flight. It does not yet weigh hardware, speed or warm prompts.

## M5 Context sync and failover (Planned)

Conversations are kept in sync across inference nodes, so a reply can resume mid-answer after a model node or an inference node dies, with the full context and no repeated text.

Weirdnet crates: `weirdnet-agent`, `weirdnet-sync`.

Known gaps:
- Waits on a `weirdnet-sync` design for local eviction and working sets.

## M6 Workload spreading (Planned)

Load balancing across replicas, moving models around the cloud, fair sharing between devices, priority classes, batch jobs, a shared answer cache, and several inference nodes active at once.

Weirdnet crates: `weirdnet-agent`, `weirdnet-backends`, `weirdnet-semcache`.

## M7 Split models (Planned)

A model too big for one machine is split across several (vLLM with Ray, or llama.cpp RPC), reproducing the big mode on the two DGX Sparks.

Weirdnet crates: `weirdnet-backends`.

## M8 Move server-a and server-b onto Ymir (Planned)

The two DGX Sparks that serve models today move to Ymir, one step at a time, with the owner approving each step.

Weirdnet crates: none new.

## M9 Breadth and hardening (Planned)

Self-update with rollback, push wake-ups, a Windows service, BSD packages, a security review, and the 1.0 criteria. The open issues parked from M3 are handled here.

Weirdnet crates: `weirdnet-update`, `weirdnet-push`, `weirdnet-agent`.

Known gaps:
- `weirdnet-push` is planned and does not exist yet.

## M10 Cert-only server link (Planned)

Service nodes get their own certificate and sync keys without holding the root seed, so one node can be revoked without rotating your identity.

Weirdnet crates: `weirdnet-sync`, `weirdnet-device-link`.

Known gaps:
- Needs Weirdnet changes first.
