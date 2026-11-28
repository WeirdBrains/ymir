# Ymir roadmap

<!-- Generated from roadmap.toml by site/tool/gen_roadmap.py. Edit roadmap.toml, not this file. -->

Ymir is built in milestones. Each one lands its Weirdnet crate work first, published on crates.io, and then the Ymir side. Status comes from the design doc. Training work follows a separate plan: P1, P4a and P2 are built; P3a and P4b are built and tested with a mock trainer and mock models.

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

Load balancing across replicas, fair sharing between devices, priority classes, batch jobs that fan out across nodes, the pool benchmark, and moving a model between nodes (`ymirctl move`, with the old node finishing first) and placement with replica counts (`[inference.placement]`, one inference node placing under the `ymir.scheduler` lease, its plan published as a signed `ymir.plan` and fenced by term) are built, and so are cloud-wide swap (cold loads where nothing is evicted, idle models unloaded after `idle_unload_secs`) and a semantic answer cache shared across the pool's inference nodes (`[inference.cache] embedding_model`, `mode = "pool"`). Model nodes refuse a stale holder's loads and unloads (`weirdnet-backends` 0.4.11 fenced commands). Next: replication on queue wait and prefix warming.

Weirdnet crates: `weirdnet-agent`, `weirdnet-backends`, `weirdnet-semcache`.

## M7 Split models (In progress)

A model too big for one machine is split across several, with llama.cpp RPC or with vLLM on a Ray cluster (`vllm-ray`): an opt-in pool policy, private-link-only planning and binds, group up and down, and teardown when a member is lost. Next: a real GPU run of `vllm-ray` and the big-mode reproduction on server-a and server-b.

Weirdnet crates: `weirdnet-backends`.

## M8 Move existing servers onto Ymir (Done)

The servers that served models before now run on Ymir. The move was made one step at a time, with the owner approving each step.

Weirdnet crates: none new.

## M9 Breadth and hardening (In progress)

Built: self-update with rollback (pinned root, signed bundles, trial boots), agent tasks (task/1) for Bragi's agent mode, the Windows service, a security review with checks that run, a `[model] callers` guard so a model node can refuse app devices, Inkling 2 Mini, automatic context sizing (fit fixes in 0.3.0 and 0.3.1), MLX mock end-to-end tests, outside MCP servers for tasks, and a bundle fetcher (`ymirctl update fetch` and `update watch`, which hears announcements from a relay). Push wake-ups over the relay wake path: a seed holder syncs Bragi's wake cards (never a cert-only node) and `ymirctl pool wake DEVICE` wakes one. Next: APNs/FCM wake-ups, BSD packages, and MLX-LM for Inkling 2 Mini once mlx-lm#2086 is in a release (vLLM and llama.cpp already serve it).

Weirdnet crates: `weirdnet-update`, `weirdnet-push`, `weirdnet-agent`.

## M10 Cert-only server link (Done)

Servers can be linked cert-only: a node certificate and a signed service grant, never the root seed, so a stolen server is never you and one node can be revoked without rotating your identity. `ymirctl pool add --service inference|model` sends such a link, `pool link --cert-only` takes only that, and `pool cert-only` moves a node that holds the seed and wipes its copy.

Weirdnet crates: `weirdnet-sync`, `weirdnet-device-link`.

## M11 Sharing a pool with other identities (In progress)

Built: `ymirctl pool share USER_ID` records a synced guest grant for another Weirdnet identity (optional pinned devices, model allowlist, scopes and expiry) and prints a `wnet-pool1:` / `weirdnet-pool://share/` link with the pool's inference nodes; `pool unshare` revokes it on every node at once and `pool guests` lists them. Guests get the catalog and inference only, never conversations, wake cards, tasks, admin, models or training. Tested over the in-process transport; next: a real-network run and the open-link screens in Morph and Bragi.

Weirdnet crates: `weirdnet-agent`.

## P1 Data generation and evals (Done)

Built: `ymirctl batch --jsonl` and `ymirctl eval` (teacher integrity, manifests, resume, bench scorers and an optional judge). The teacher's license comes from `--license`, the Hub repo's model card (`--license-from`) or the pool catalog, and `ymirctl train seed` turns source texts into seed prompts in two passes through the teacher. Left: a job record that stays in sync across nodes.

Weirdnet crates: `weirdnet-agent`.

## P4a LoRA adapters on a loaded base (Done)

Built: LoRA adapters served on a loaded base, on llama.cpp and vLLM, each advertised in the catalog as its own model with its base. `ymirctl models adapter` adds, lists, serves and rescales them, `models pull --adapter-for` fetches them, a vLLM base can keep LoRA slots free to take adapters without a restart, and an adapter trained on another revision of its base is not served. Versioned rollout and canaries are P4b.

Weirdnet crates: `weirdnet-backends`, `weirdnet-agent`, `weirdnet-openai`, `weirdnet-alerts`.

## P4b Versions, canaries and rollback (Done)

Built: a rollout name in the pool policy with versions behind it and a traffic weight each. Each conversation stays on one version, `name@N` pins a version, and `ymirctl models rollout` adds versions, changes weights live, rolls back and retires. A version gets up to 10% of new conversations freely; more needs a `ymirctl eval` result at least as good as the current version's. Tested against mock models; not yet with a real fine-tune.

Weirdnet crates: none new.

## P2 Training data from conversations (Done)

Built: a per-conversation training consent that is off unless you turn it on and syncs between your devices, `ymirctl context train` to set or withdraw it, `context export --training` to write only the consented conversations as chat-format JSONL with system prompts, tools, attachments and reasoning stripped and patterns redacted, owner-only and never synced, and `context withdrawn` to list what an earlier export should drop.

Weirdnet crates: `weirdnet-agent`, `weirdnet-sync`.

## P3 Fine-tuning as a pool job (In progress)

P3a is built: `ymirctl train run` fine-tunes a llama.cpp base on one node with the pinned upstream llama.cpp build (a full fine-tune, then a LoRA adapter extracted from it), from a dataset with a recorded source and license. The node's GPU is held for the run with `weirdnet-backends`' new reservation, so its models unload and nothing loads there until the run ends. The llama.cpp trainer (`scripts/trainer.py`: `llama-finetune` at the engine's pinned build, then LoRA extraction by truncated SVD) is built and tested against a fake `llama-finetune`; no real run yet. Next: a first real run on a node the lead clears. P3b's Ymir side is built: `train run --job` checkpoints each epoch and `train resume` carries a job on from its last checkpoint, on a local stand-in for the synced job record that `weirdnet-agent` 0.6.0 will provide.

Weirdnet crates: `weirdnet-backends`, `weirdnet-agent`.
