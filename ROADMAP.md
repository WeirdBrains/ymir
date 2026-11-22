# Ymir roadmap

<!-- Generated from roadmap.toml by site/tool/gen_roadmap.py. Edit roadmap.toml, not this file. -->

Ymir is built in milestones. Each one lands its Weirdnet crate work first, published on crates.io, and then the Ymir side. Status comes from the design document.

Updated 2026-11-19. Design: [docs/DESIGN.md](https://github.com/WeirdBrains/ymir/blob/main/docs/DESIGN.md).

## M0 Design (Done)

The repo, the names, the three roles and the decisions, written down in [docs/DESIGN.md](https://github.com/WeirdBrains/ymir/blob/main/docs/DESIGN.md) and reviewed by the Weirdnet lead.

Weirdnet crates: none new.

## M1 One box, three roles in one daemon (Done)

One machine runs the model, inference and app roles in a single `ymird`. It detects the hardware, picks models that fit, runs llama.cpp, and serves an OpenAI-compatible endpoint on loopback so Jan and scripts work. `ymirctl setup` writes the config.

Weirdnet crates: [`weirdnet-model`](https://crates.io/crates/weirdnet-model), [`weirdnet-backends`](https://crates.io/crates/weirdnet-backends), [`weirdnet-openai`](https://crates.io/crates/weirdnet-openai), [`weirdnet-semcache`](https://crates.io/crates/weirdnet-semcache).

## M2 Model-node parity with existing llama-swap servers (Done)

vLLM as a second engine (managed, or wrapping a server that is already running), idle unload, restart with backoff after a crash, a small stand-in model that answers while a big one loads, alerts, a user service and Linux packages.

Weirdnet crates: [`weirdnet-backends`](https://crates.io/crates/weirdnet-backends), [`weirdnet-alerts`](https://crates.io/crates/weirdnet-alerts), [`weirdnet-model`](https://crates.io/crates/weirdnet-model).

## M3 Roles over Weirdnet (Done)

The same three roles talk across devices over Weirdnet. An app node reaches inference nodes, inference nodes drive model nodes, and only devices in your own roster are let in. `ymirctl` and Jan work on another machine through that machine's own loopback endpoint.

Weirdnet crates: [`weirdnet-agent`](https://crates.io/crates/weirdnet-agent), [`weirdnet-net`](https://crates.io/crates/weirdnet-net), [`weirdnet-sync`](https://crates.io/crates/weirdnet-sync), [`weirdnet-device-link`](https://crates.io/crates/weirdnet-device-link), [`weirdnet-backends`](https://crates.io/crates/weirdnet-backends), [`weirdnet-model`](https://crates.io/crates/weirdnet-model).

## M4 Several model nodes (Done)

More than one model node in the pool: linking devices, each node advertising what it can run, one combined model list, failover before the first token, and draining a node before maintenance.

Weirdnet crates: [`weirdnet-device-link`](https://crates.io/crates/weirdnet-device-link), [`weirdnet-relay`](https://crates.io/crates/weirdnet-relay), [`weirdnet-net`](https://crates.io/crates/weirdnet-net), [`weirdnet-agent`](https://crates.io/crates/weirdnet-agent), [`weirdnet-backends`](https://crates.io/crates/weirdnet-backends).

## M5 Context sync and failover (Done)

Conversations are kept in sync across inference nodes, so a reply can resume mid-answer after a model node or an inference node dies, with the full context and no repeated text.

Weirdnet crates: [`weirdnet-agent`](https://crates.io/crates/weirdnet-agent), [`weirdnet-sync`](https://crates.io/crates/weirdnet-sync).

## M6 Workload spreading (In progress)

Load balancing across replicas, moving models around the cloud, fair sharing between devices, priority classes, batch jobs, a shared answer cache, and several inference nodes active at once. Balancing, the per-device cap, priorities and batch fan-out are built; placement and the shared cache are not.

Weirdnet crates: [`weirdnet-agent`](https://crates.io/crates/weirdnet-agent), [`weirdnet-backends`](https://crates.io/crates/weirdnet-backends), [`weirdnet-semcache`](https://crates.io/crates/weirdnet-semcache).

## M7 Split models (In progress)

A model too big for one machine is split across several (vLLM with Ray, or llama.cpp RPC), reproducing the big mode on two large-memory machines. The llama.cpp RPC path is built and tested on one machine over loopback, with a real model: opt-in pool policy, private-link-only planning, group up and down, teardown on member loss.

Weirdnet crates: [`weirdnet-backends`](https://crates.io/crates/weirdnet-backends).

## M8 Move existing servers onto Ymir (Planned)

The servers that serve models today move to Ymir, one step at a time, with the owner approving each step.

Weirdnet crates: none new.

## M9 Breadth and hardening (In progress)

Built: self-update with rollback on weirdnet-update (pinned root, signed bundles, trial boots), agent tasks (task/1) for Bragi's agent mode, and a security review with checks that run, and a `[model] callers` guard so a model node can refuse app devices. Not built: push wake-ups, the Windows service, BSD packages, a bundle fetcher. The four upstream issues are still open. Revoked model nodes are now refused and `[model] callers` takes peer ids only.

Weirdnet crates: [`weirdnet-update`](https://crates.io/crates/weirdnet-update), `weirdnet-push`, [`weirdnet-agent`](https://crates.io/crates/weirdnet-agent).

## M10 Cert-only server link (Planned)

Service nodes get their own certificate and sync keys without holding the root seed, so one node can be revoked without rotating your identity. m3talsmith approved the design on 2026-11-20. Most of the crates are now published (weirdnet-sync 0.7.0, weirdnet-device-link 0.4.0, weirdnet-agent 0.4.4, weirdnet-backends 0.1.2); the Weirdnet lead is still verifying them, and weirdnet-update 0.2.1 and the adoption notes come later. Ymir wires M10 up after that. A majority-vote lease follows as weirdnet-sync 0.8.0 after M10.

Weirdnet crates: [`weirdnet-sync`](https://crates.io/crates/weirdnet-sync), [`weirdnet-device-link`](https://crates.io/crates/weirdnet-device-link).
