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

## M5 Context sync and failover (Done)

Conversations are kept in sync across inference nodes, so a reply can resume mid-answer after a model node or an inference node dies, with the full context and no repeated text.

Weirdnet crates: `weirdnet-agent`, `weirdnet-sync`.

Known gaps:
- No eviction or working sets yet: inference nodes keep every conversation, until `weirdnet-sync` can drop a local copy.
- Job checkpoints, attachments, retrieved context and prefix hints are not built. The conversation store lives in Ymir until `weirdnet-agent` has a context module.
- Tested on one computer with a mock engine; no cross-machine run yet.

## M6 Workload spreading (In progress)

Load balancing across replicas, moving models around the cloud, fair sharing between devices, priority classes, batch jobs, a shared answer cache, and several inference nodes active at once. Balancing, the per-device cap, priorities and batch fan-out are built; placement and the shared cache are not.

Weirdnet crates: `weirdnet-agent`, `weirdnet-backends`, `weirdnet-semcache`.

Known gaps:
- No scheduler lease, placement plan, model swapping across nodes, replication or prefix warming yet: they need a lease and a synced plan from weirdnet-sync.
- No shared (semantic) answer cache across nodes yet.
- A batch job lives on the node that submitted it; it is not synced, so another node can't pick it up.
- Every answer waits 3 s per linked device the app can't reach (12 to 15 s in the multi-node test) before it starts: a `weirdnet-agent` 0.4.2 bug in finding inference nodes, reported to the Weirdnet lead. The first-token test waits on that fix.
- Tested on one computer with a mock engine.

## M7 Split models (In progress)

A model too big for one machine is split across several (vLLM with Ray, or llama.cpp RPC), reproducing the big mode on the two DGX Sparks. The llama.cpp RPC path is built and tested on one machine over loopback, with a real model: opt-in pool policy, private-link-only planning, group up and down, teardown on member loss.

Weirdnet crates: `weirdnet-backends`.

Known gaps:
- vLLM/Ray (tensor parallel over RoCE) groups are not built: they need the DGX Sparks, which were off limits.
- The big mode has not been reproduced as a group job on server-a and server-b; that needs m3talsmith's go-ahead.
- Capability records carry no link information, so private links, ports and memory are declared in the pool policy and link speed is not measured (upstream request).
- A crashed worker on another node is noticed only when that node leaves the pool or a request fails; alerts for a lost group are logged, not delivered.
- A group is started by the operator (ymirctl group up); placement and leases wait on sync 0.7.0.

## M8 Move server-a and server-b onto Ymir (Planned)

The two DGX Sparks that serve models today move to Ymir, one step at a time, with the owner approving each step.

Weirdnet crates: none new.

## M9 Breadth and hardening (In progress)

Built: self-update with rollback on weirdnet-update (pinned root, signed bundles, trial boots), agent tasks (task/1) for Bragi's agent mode, and a security review with checks that run, and a `[model] callers` guard so a model node can refuse app devices. Not built: push wake-ups, the Windows service, BSD packages, a bundle fetcher. The four upstream issues are still open.

Weirdnet crates: `weirdnet-update`, `weirdnet-push`, `weirdnet-agent`.

Known gaps:
- Push wake-ups wait on the Weirdnet front end of `weirdnet-push` (does not exist) and on Apple and Google credentials from m3talsmith.
- Windows service, named-pipe control and BSD packages are not built or tried.
- Self-update takes a bundle directory: no fetcher or relay announcements yet, nothing signed with a real org key, and it was only run on one box with test keys.
- Upstream, still open: weirdnet-openai #1 and #2, weirdnet-agent #1, weirdnet-backends #1. New asks written up: a `tasks` field in `Serving`, roles in authorization, and an unsigned signing identity for plain file sets.
- The security review is the author's. Role boundaries are not enforced: a model node accepts any linked device.
- No run across real machines, so the 1.0 criteria in DESIGN.md are mostly open.

## M10 Cert-only server link (Planned)

Service nodes get their own certificate and sync keys without holding the root seed, so one node can be revoked without rotating your identity. Not built: it needs a cert-only mode in weirdnet-sync and weirdnet-device-link, which the published crates do not have.

Weirdnet crates: `weirdnet-sync`, `weirdnet-device-link`.

Known gaps:
- Blocked on the Weirdnet lead: no cert-only link in weirdnet-sync 0.6.0 or weirdnet-device-link 0.3.0, and no issue open for it. The ask is written up in DESIGN.md (where the seed is used, what a link must send instead).
- Until then every model and inference node holds the root seed in a 0600 file.
