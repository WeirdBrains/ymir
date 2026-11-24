# Ymir roadmap

<!-- Generated from roadmap.toml by site/tool/gen_roadmap.py. Edit roadmap.toml, not this file. -->

Ymir is built in milestones. Each one lands its Weirdnet crate work first, published on crates.io, and then the Ymir side. Status comes from the design doc. Training work follows a separate plan: P1, P4a and P2 are built, and P3a is built and tested with a mock trainer.

Updated 2026-11-24. Design: [docs/DESIGN.md](https://github.com/WeirdBrains/ymir/blob/main/docs/DESIGN.md).

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

Load balancing across replicas, fair sharing between devices, priority classes, batch jobs that fan out across nodes, and the pool benchmark are built. Next: moving models between nodes, placement and replication, and a semantic cache.

Weirdnet crates: `weirdnet-agent`, `weirdnet-backends`, `weirdnet-semcache`.

## M7 Split models (In progress)

A model too big for one machine is split across several with llama.cpp RPC: an opt-in pool policy, private-link-only planning, group up and down, and teardown when a member is lost. Next: vLLM with Ray groups.

Weirdnet crates: `weirdnet-backends`.

## M8 Move existing servers onto Ymir (Done)

The servers that served models before now run on Ymir. The move was made one step at a time, with the owner approving each step.

Weirdnet crates: none new.

## M9 Breadth and hardening (In progress)

Built: self-update with rollback (pinned root, signed bundles, trial boots), agent tasks (task/1) for Bragi's agent mode, the Windows service, a security review with checks that run, a `[model] callers` guard so a model node can refuse app devices, Inkling 2 Mini, automatic context sizing (fit fixes in 0.3.0 and 0.3.1), MLX mock end-to-end tests, and outside MCP servers for tasks. Next: push wake-ups, BSD packages, a bundle fetcher, and MLX-LM for Inkling 2 Mini once mlx-lm#2086 is in a release (vLLM and llama.cpp already serve it).

Weirdnet crates: `weirdnet-update`, `weirdnet-push`, `weirdnet-agent`.

## M10 Cert-only server link (Done)

Servers can be linked cert-only: a node certificate and a signed service grant, never the root seed, so a stolen server is never you and one node can be revoked without rotating your identity. `ymirctl pool add --service inference|model` sends such a link, `pool link --cert-only` takes only that, and `pool cert-only` moves a node that holds the seed and wipes its copy.

Weirdnet crates: `weirdnet-sync`, `weirdnet-device-link`.

## P1 Data generation and evals (Done)

Built: `ymirctl batch --jsonl` and `ymirctl eval` (teacher integrity, manifests, resume, bench scorers and an optional judge). Left: the teacher license read from the Hugging Face Hub, turning sources into seed prompts, and a job record that stays in sync across nodes.

Weirdnet crates: `weirdnet-agent`.

## P4a LoRA adapters on a loaded base (Done)

Built: LoRA adapters served on a loaded base, on llama.cpp and vLLM, each advertised in the catalog as its own model with its base. `ymirctl models adapter` adds, lists, serves and rescales them, `models pull --adapter-for` fetches them, a vLLM base can keep LoRA slots free to take adapters without a restart, and an adapter trained on another revision of its base is not served. Versioned rollout and canaries (P4b) come later.

Weirdnet crates: `weirdnet-backends`, `weirdnet-agent`, `weirdnet-openai`, `weirdnet-alerts`.

## P2 Training data from conversations (Done)

Built: a per-conversation training consent that is off unless you turn it on and syncs between your devices, `ymirctl context train` to set or withdraw it, `context export --training` to write only the consented conversations as chat-format JSONL with system prompts, tools, attachments and reasoning stripped and patterns redacted, owner-only and never synced, and `context withdrawn` to list what an earlier export should drop.

Weirdnet crates: `weirdnet-agent`, `weirdnet-sync`.

## P3 Fine-tuning as a pool job (In progress)

P3a is built: `ymirctl train run` fine-tunes a llama.cpp base on one node with the pinned upstream llama.cpp build (a full fine-tune, then a LoRA adapter extracted from it), from a dataset with a recorded source and license. The node's GPU is held for the run with `weirdnet-backends`' new reservation, so its models unload and nothing loads there until the run ends. It is tested with a mock trainer; no real run yet. Next: the llama.cpp wrapper and a first real run. A synced job record and resume on another node come later (P3b).

Weirdnet crates: `weirdnet-backends`, `weirdnet-agent`.
