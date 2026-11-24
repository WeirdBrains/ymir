# Training and fine-tuning around Ymir: implementation plan

Status: plan, updated 2026-11-23. P1 is built (`ymirctl batch --jsonl` and `ymirctl eval`); the Hub
license lookup, seed prompts and a synced job record are still left. P4a (LoRA adapters on a loaded
base) is built. P2 and P3 have not started. This document turns the four training proposals in the
white paper's "Training and fine-tuning (proposal)" section (its LaTeX source is not in this
repository yet) into work items, maps each onto the modules
that exist today, and sets the order. The order below was confirmed by the project lead on 2026-11-22, and
m3talsmith answered the open questions the same day (see "Decisions").

Ymir is designed by m3talsmith and published by [Weird Brains LLC](https://weirdbrains.com).

## Ground rules this plan keeps

- **Training stays out of Ymir.** [DESIGN.md](../DESIGN.md#non-goals) says Ymir feeds, schedules and serves a
  separate trainer, and does not train models itself. Ymir supplies data generation, evals, serving and (later)
  scheduling a job onto a node; the trainer is a separate program that Ymir starts or talks to: llama.cpp's
  finetune for now.
- **One GPU per run.** Multi-machine training over Weirdnet spends most of its time waiting on the network
  (white paper, "Why training stays on one machine or a private link"). Every slice below trains on one node.
  Two-node data-parallel LoRA on a private link is listed last, as optional.
- **Generic code goes to Weirdnet, through the Weirdnet lead.** [AGENTS.md](../../AGENTS.md): job records,
  sync schemas, engine adapters, routing and model download are `weirdnet-*` concerns. This repo never edits a
  `weirdnet-*` crate; each item that needs one is listed under "Needed from Weirdnet" and sent to the lead.
- **Inference nodes serve live traffic.** Any training run, and any large generation or eval run, on an
  inference or model node that serves the pool needs the project lead's OK for that run first. Development and
  tests use `YMIR_HOME` in a temp directory, mock engines and no real models, as every other test here does.
- **Licenses.** A teacher model's license decides whether its outputs may train another model. Every dataset
  carries the teacher's model id and license, and the tools refuse to start a run whose teacher license is
  unknown unless the operator overrides it explicitly.

## What exists today

| Piece | Where | Use for training |
|---|---|---|
| Loopback OpenAI bridge (`weirdnet-openai`) | `ymird`, `127.0.0.1:8484` | Any OpenAI client can generate data and run evals now (white paper listing). |
| `ymirctl batch` | [`crates/ymir-core/src/batch.rs`](../../crates/ymir-core/src/batch.rs), `crates/ymirctl/src/lib.rs` | Fans prompts over the pool at `batch` priority with retries by job id. Input is one plain prompt per line; output is tab-separated text. No system prompt, sampling, seed, JSONL, resume or synced job record. |
| `ymirctl bench` | [`crates/ymirctl/src/bench/`](../../crates/ymirctl/src/bench/mod.rs) | Deterministic scoring (exact, numbers, JSON, lines, sandboxed code) at temperature 0 with per-request seeds and nonces; per node and model. The scorers are the start of an eval harness. |
| Context store | [`crates/ymir-core/src/context.rs`](../../crates/ymir-core/src/context.rs) | Conversations as sealed `weirdnet-sync` records (`{conversation}/meta`, `{conversation}/m/...`), shared with Bragi. Destined for `weirdnet-agent`'s `context` module. No training flag. |
| Fairness and priorities | DESIGN.md, ["Fairness and priorities"](../DESIGN.md#fairness-and-priorities) | `interactive` > `batch` > `background`, deficit round robin per device, so generation does not starve chat. |
| Drain, load, unload | `ymir-core` control and coordinator | Free a node's GPU by hand before a run. |
| `ymirctl models pull --node ... --serve` | [`crates/ymir-core/src/models.rs`](../../crates/ymir-core/src/models.rs), `crates/ymirctl/src/models/` | Rolls a merged fine-tune (GGUF) onto any node without a restart. |
| `LocalModel.args` | [`crates/ymir-core/src/config.rs`](../../crates/ymir-core/src/config.rs) | Extra engine arguments, so `--lora FILE` on llama-server works today by hand; nothing in the catalog knows about it. |

## The proposals

### P1. Data generation, distillation and evals through the bridge and batch (path built, use proposed)

**Status (2026-11-22).** First slice built: `ymirctl batch --jsonl` (rows with system prompt, sampling and seed;
wait-for-model; `usable`, `cache_hit` and `standin` per row; manifest with the operator's license; a sources
manifest with each source's license; `--resume`) and `ymirctl eval` (bench scorers, optional judge). The cache mode is on
the wire (`weirdnet-agent` 0.5.1: batch generation and evals send `bypass` with `wait_for_model`, and the
coordinator honours it), and answer-cache entries are keyed on the weights revision where a node reports one.
Left: the teacher license read from the Hub, turning
fetched sources into seed prompts, and the synced job record.

**Scope.** A teacher model in the pool answers or labels a file of prompts; the answers become a dataset for a
student. The same path scores a candidate model on held-out prompts, optionally with a second model as judge.

**Work (Ymir side).**
- `ymirctl batch --jsonl`: input rows of `{"id", "messages" | "prompt", "system", "temperature", "seed",
  "max_tokens", "response_format"}`; output rows of `{"id", "messages", "completion", "model", "node",
  "attempts", "usage"}`. Plain-text mode stays as it is.
- Teacher integrity: generation requests set `wait_for_model` (never accept a stand-in's answer as teacher
  data), skip the answer cache (`weirdnet-semcache` would otherwise return near-duplicates for similar
  prompts), and record the model that actually answered. Rows answered by any other model are marked and
  excluded by default.
- `--resume OUT.jsonl`: skip rows whose id is already in the output, so a long run survives the submitting
  machine stopping (a stopgap until the synced job record exists).
- A dataset manifest next to the output: teacher model id and revision, license as reported by the Hub,
  prompt file hash, row counts, date. Written by `ymirctl`, read by the trainer.
- `ymirctl eval`: held-out JSONL with expected answers, scored with the `bench` scorers (moved into a shared
  module, not copied); optional `--judge MODEL` with a fixed rubric prompt; a report comparing two models or
  two versions, with the same seeds for both.

**Dependencies.** None on other proposals. Code: `ymir-core::batch`, `ymirctl` batch and bench, `weirdnet-agent`
infer/2 (`Request.seed`, `sampling`, `response_format`, `wait_for_model` exist in 0.5).

**Needed from Weirdnet.** Skipping the answer cache over infer/2: done (`Request::cache`, `weirdnet-agent`
0.5.1; `weirdnet-semcache` 0.2.1 `get_rev`/`put_rev`; `weirdnet-openai` 0.2.1 `cache` / `x-weirdnet-cache`). Later: the synced job record
for batch hand-off (`weirdnet-agent`, already planned), which replaces `--resume`.

**Data and compute.** Seed prompts from the operator. Compute is the existing pool at `batch` priority; no new
hardware. Thousands of prompts are an overnight job on a few nodes.

**Risks.** Teacher licenses (several open-weight licenses restrict training on outputs). Teacher errors become
student errors, so the eval set must be held out and never generated by the same run. A large run still loads
the live pool; runs over a few hundred prompts against serving nodes wait for the lead's OK. LLM judges are
biased toward their own family; report judge and deterministic scores separately.

### P2. Training data from synced conversations, with consent (proposal)

**Scope.** A trainer running on one of the user's linked devices exports only the conversations the user has
marked for training, with chosen parts stripped, as a dataset. Model nodes never hold conversations, and
nothing leaves the user's devices.

**Work.**
- A per-conversation opt-in flag. Keep it in its own last-writer-wins record (`{conversation}/train`), not in
  `meta`: `meta` is one register, so an older client that rewrites the title without knowing the field would
  silently clear the flag. Absent means "not for training"; the default is off and never inferred.
- `ymirctl context export --training OUT.jsonl`: reads the local replica, takes flagged, non-deleted
  conversations, applies strip rules (drop system prompts, drop tool results, drop attachments, redact by
  pattern), and writes chat-format JSONL plus the same manifest as P1 (with conversation ids and record
  times, so a later export can honour a withdrawn flag or a deletion).
- Bragi: a toggle per conversation, in both shells (a Bragi change, coordinated with its owner).

**Dependencies.** Independent of P1, but shares P1's manifest and JSONL format. Code: `ymir-core::context`,
`membership` (`CONTEXT_TYPE`), Bragi's `bragi-core`.

**Needed from Weirdnet.** The context schema is generic and is moving to `weirdnet-agent`'s `context` module;
the training flag record should be added there (or agreed with the lead before Ymir adds it to the interim
module), so Bragi and Ymir read the same record.

**Data and compute.** Only what the user marks. Export is local disk I/O; no GPU.

**Risks.** Privacy is the whole risk: an export is a copy outside the encrypted store, so it is written with
owner-only permissions, never synced, never logged, and the tool prints counts, not content. Withdrawing
consent cannot recall a dataset already used; the manifest makes it possible to rebuild without it. Bridge
conversations are kept 24 hours, so most chats worth training on are Bragi chats.

### P3. Fine-tuning as a pool job (proposal)

**Scope.** A LoRA or QLoRA run on one capable GPU node at background priority, with serving paused or moved off
that node, checkpoints synced, and resume on another capable node if the first goes down.

**Work, in two parts.**
- **P3a, one node, no hand-off.** `ymirctl train reserve --node X` drains the node, unloads its models, and
  marks it unavailable for placement until `release` (or until the run ends); `ymirctl train run` starts the
  separate trainer there with a dataset (P1/P2 manifest), a base model already on that node, and a config;
  progress and logs stream back; the adapter is written into the node's models directory. The trainer is an
  external program (its own repo or package; see open questions), invoked with a documented interface:
  dataset path, base model path, output path, hyperparameters, a cancel signal, progress on stdout as JSON
  lines.
- **P3b, synced job record and resume.** The run's record (dataset content root, base model, config, last
  checkpoint content root) is synced, and another node with enough memory and the same base model resumes from
  the last checkpoint when the first is lost. Builds on the synced job record planned for batch hand-off.

**Dependencies.** P1 or P2 for data (or a dataset supplied by hand). P4a, to serve what it produces.

**Needed from Weirdnet.** A node reservation in the `weirdnet-backends` node protocol and supervisor (today
there are `Drain`, `Load` and `Unload`, but no "hold this GPU, place nothing here"). The synced job record and
hand-off (`weirdnet-agent`). Checkpoint transfer as content roots (`weirdnet-net`, exists).

**Data and compute.** QLoRA of a 7B to 8B model fits in roughly 10 to 16 GB of GPU memory; LoRA in 16-bit needs
about 20 to 24 GB; hours per run for tens of thousands of rows. Disk: the base model plus a few hundred MB of
checkpoints (an adapter is usually well under 1 percent of the model; optimizer state is larger).

**Risks.** It takes a serving node out of the pool: every run on a node that serves live traffic needs the
lead's OK, and reserve refuses a node that is the only one serving a model unless forced. The trainer's Python
and CUDA stack must not be installed system-wide without asking; ship it as a separate, pinned environment.
Resume is only correct if the data order and seeds are recorded, so P3b must checkpoint them.

### P4. Distribution, hot swap and versioned rollout (partly built)

**Status (2026-11-23).** P4a (LoRA adapters on a loaded base, llama.cpp and vLLM) is built. P4b
(versioned rollout and canaries) has not started. Pulling a merged GGUF onto any node and serving it
without a restart is already built.

**Scope.** Built: pull a merged GGUF onto any node and serve it without a restart. Proposed: serve LoRA adapters
on a loaded base, name versions in the catalog, send a small share of traffic to a new version, and roll back by
moving traffic to the previous version, which stays loaded until the new one proves itself.

**Work, in two parts.**
- **P4a, adapters on a base.** A first-class `lora` list on `LocalModel` (adapter path, scale, the catalog id it
  serves as, e.g. `base-model+support-v2`), instead of raw `args`. llama.cpp first (`--lora`, `--lora-scaled`;
  llama-server can also change adapter scale at runtime), then vLLM (`--enable-lora`, `--lora-modules`), where
  each adapter is its own model name. The catalog advertises each adapter as a model with its base, so routing
  and failover need no change. `ymirctl models pull` learns adapter files (a GGUF LoRA next to its base).
- **P4b, versions and canary.** Version names in the catalog (`support@2`, `support@3`), a pool-policy weight
  per version (for example 5 percent to the new one), rollback by setting the weight back, and the previous
  version kept loaded until an operator retires it. `ymirctl eval` (P1) on the same held-out set is the gate.

**P4a design (first slice, 2026-11-23).** The engine work is `weirdnet-backends` 0.4 (`LoraAdapter`,
`ModelLaunch::adapters`, `Supervisor::add_adapter`/`remove_adapter`, a status row per adapter with its `base`,
and `models/1` pulls with `adapter_for`). Ymir adds `[[model.local.lora]]` (`id`, `file`, `scale`,
`revision`), drops adapters that don't match their base at start, counts adapter weights in the base's
context plan, shows adapters with their base in the status, and `ymirctl models adapter add | remove | list`
plus `models pull --adapter-for BASE --node NAME --serve`. Details in
[DESIGN.md](../DESIGN.md#lora-adapters-on-a-loaded-base-training-plan-p4a-2026-11-23). Second slice (same
day): live scale changes, a per-request scale over infer/2, adapters served and unserved on a running base
(vLLM at run time; llama-server only for files it started with, a new file still restarts the base), and the
adapter's base in catalog/1, on `weirdnet-backends` 0.4.1, `weirdnet-agent` 0.5.2 and `weirdnet-openai` 0.2.2.
Third slice (same day, no Weirdnet change), which completes P4a: `lora_slots` on a vLLM `[[model.local]]` starts
it with LoRA on so adapters can be served on it later without a restart (`ymirctl models adapter slots`), and the
base-revision check: adapters record the base `revision` and `quant` they were trained on (`base_revision`,
`base_quant`, filled in by `adapter add`), and one whose base has moved on is not served (logged at start,
flagged by `adapter list`, refused by `adapter add`).

**Dependencies.** P4a needs nothing else and closes the loop with an adapter trained by hand. P4b needs P1's
eval and P4a.

**Needed from Weirdnet.** Engine launch arguments for adapters and their capability record (`weirdnet-backends`
adapters); adapter files in model search and download (`weirdnet-model`); weighted routing between versions of
one model (`weirdnet-agent` pool or Ymir's `balance`; to be agreed with the lead, since routing is listed as
Weirdnet's).

**Data and compute.** None beyond serving; an adapter adds little memory over its base.

**Risks.** A canary sends some real requests to a worse model; keep the share small and make rollback one
command. Answers from different versions must not share answer-cache entries: done for revisions a node reports
(`[[model.local]] revision`, keyed with `get_rev`/`put_rev`); P4 must set it for every fine-tune. Adapters are tied to the exact base revision and quantisation; loading one on another base must be
refused, not attempted (done in P4a for adapters in `node.toml` that record their base).

## Order (confirmed 2026-11-22)

Smallest useful end-to-end slice first. Slices 1 and 2 together give a complete loop with no new GPU
scheduling: generate a dataset through the pool, train an adapter by hand on a machine that is not serving,
serve it as an adapter, and score it against the base through the pool.

1. **P1: JSONL batch, teacher integrity, manifest, resume, and `ymirctl eval`.** Ymir only, no Weirdnet change
   needed to start, usable on day one.
2. **P4a: LoRA adapters on a loaded base (llama.cpp first).** Built (2026-11-23). Small Ymir change; asks the lead for the
   `weirdnet-backends` adapter arguments. Closes the loop.
3. **P2: consented conversation export.** Needs the flag record agreed with the lead and Bragi.
4. **P3a: reserve a node and run the external trainer on it, one node, no hand-off.** Needs the reservation in
   `weirdnet-backends`; each run on a serving node needs the lead's OK.
5. **P4b: versions, canary weights and rollback**, gated by P1's eval.
6. **P3b: synced training job record and checkpoint resume**, once `weirdnet-agent` has the synced job record.
7. Optional, later: two-node data-parallel LoRA over a private link (10 Gb/s or faster), never over Weirdnet.

## Testing

- Unit tests for the JSONL formats, manifest, resume, export strip rules and flag semantics, under a temp
  `YMIR_HOME`.
- Integration tests in `ymir-testkit` against a fake pool with mock engines: a generation run with a stand-in
  answer (excluded), a cached answer (skipped), retries, resume; an eval comparing two mock models; an adapter
  advertised in the catalog.
- No test downloads a model, starts a GPU backend or trains; a real-trainer test is `#[ignore]` and runs only
  on a node the lead has cleared.

## Decisions (2026-11-22, m3talsmith)

1. **Trainer:** llama.cpp finetune for now. It reads GGUF bases and writes GGUF LoRA adapters, which is what P4a
   serves on llama.cpp first; a PEFT/TRL trainer for vLLM can come later behind the same interface (P3a).
2. **Non-goal:** reworded in [DESIGN.md](../DESIGN.md#non-goals) to "Ymir feeds, schedules and serves a
   separate trainer".
3. **First dataset and eval topic:** "Decentralized networks: how they work". Datasets stay configurable and none
   is built in; the tests use a small synthetic fixture on this topic. Sources for the real dataset can be fetched
   by m3talsmith when needed.
4. **Conversation training flag (P2):** the Bragi toggle comes after Bragi v1; the Ymir side of P2 can be built
   before it, but nothing is exported until a conversation can be flagged.
5. **Version traffic weights (P4b):** in Ymir's `balance` for now (see the proposal below).
6. **White paper:** to be committed under `docs/whitepaper/` by m3talsmith. Until it is on main, this plan refers to
   its "Training and fine-tuning (proposal)" section by title; a link to the section's source replaces this note then.

### Proposal: where version weights could live later

Weights in Ymir's `balance` keep P4b inside this repo: the pool policy names versions and a weight each, and
`balance` picks a version before it picks a replica. Two alternatives, for later and only with the Weirdnet lead:

- **In the `weirdnet-agent` pool**, as a weighted choice between catalog entries that share a version family.
  Every Weirdnet app would get canaries, at the cost of a protocol change (catalog entries would need a family
  and a version).
- **On the client**, where the app picks a version per conversation (sticky), so one conversation never mixes
  versions. This is simpler to reason about for evals but needs every client to agree on the rule.

Either way, a conversation should stay on one version once it starts, and answer-cache keys must include the
version.
