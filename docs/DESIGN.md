# Ymir design

Date: 2026-11-19. Status: draft for m3talsmith's review. Nothing is built yet.
Reference build: m3talsmith's two GB10 servers (deployment configs and a plan snapshot are kept off-repo).
This doc follows the collective `weirdnet-agents/AGENTS.md`.

## Vision: a cloud of LLM models

Ymir is a **private, self-hosted model cloud**. You start with one machine. You add more over
Weirdnet. The pool acts as **one endpoint**, and the models are **spread across it**: requests go to
the least-loaded node that has the model warm, popular models get replicas, batch and agent jobs
fan out, and a model too big for any one machine is split across several. Nothing listens on a
public port. Your Weirdnet identity and device roster decide who may use or command the pool.

Two deliverables share one design:

1. **Ymir** (this repo): the node daemon `ymird`, the CLI `ymirctl`, and the local OpenAI-compatible
   bridge, so Jan and any OpenAI client can use the pool.
2. **Bragi** (separate repo, see [Client app](#client-app-bragi)): a Jan-like chat app with
   Weirdnet built in. GPUI on desktop, Flutter on phones, over one Rust core.

## Names

The working name was `weirdnet-llm-node`. That can't stand: `weirdnet-*` is reserved for generic
crates, and this is an app. Candidates (checked 2026-11-19: no `WeirdBrains/<name>` repo yet; the bare
names are taken on crates.io, so app crates use a suffix, as `morph-*` and `mouser-*` do):

| Name | Why | Crates |
|---|---|---|
| **Ymir** (proposed) | In Norse myth the clouds were made from Ymir's brain: a literal "cloud of brains". Fits Muninn and Mimir-style naming. | `ymir-core`, `ymird`, `ymirctl` (free) |
| Smidja | Old Norse "smithy": the place models are forged and served. | `smidja` (free) |
| Brokkr | The dwarf smith who forged Mjölnir; a strong "workhorse" name. | `brokkrd` (free; `brokkr` taken) |

Client app: **Bragi** (Norse god of poetry and speech; `bragi-core` free), or "Ymir Chat" if m3talsmith
prefers one brand. URL schemes: `weirdnet-ymir://` and `weirdnet-bragi://`, registered in
`weirdnet-schemes`. This doc uses Ymir and Bragi throughout.

## Goals

- **One machine first.** One command turns a DGX Spark, a PC with an NVIDIA or AMD GPU, an Apple
  Silicon Mac or a CPU-only box into a working node, with a backend and models picked for its hardware.
- **Grow over Weirdnet.** Add nodes by invite or by linking your own devices. The pool behaves as one
  endpoint, with active load balancing, placement and replication, job splitting, a shared semantic
  cache, and fairness between users and devices.
- **Split models too big for one node** (server-a/server-b big mode, generalized to any group of nodes).
- **Commands over Weirdnet, not HTTPS.** Inference, control (load, unload, big mode, config) and
  health all ride Weirdnet streams, authorized by Weirdnet identity and roster. No Caddy, no API keys
  for Weirdnet callers.
- **Every OpenAI client works** through a local bridge on `127.0.0.1`.
- **Everything else optional:** Redis, Caddy, Tailscale and a second node are never required.
- **Thin app.** Generic parts go in `weirdnet-*` crates on crates.io (BSD-3-Clause); Ymir adds only
  what is specific to running a model cloud.

## Non-goals

- Training or fine-tuning (the separate trainer stays separate).
- A public, multi-tenant cloud for strangers; no billing.
- A WeirdBrains-run gateway or shared domain (forbidden by AGENTS.md).
- Writing our own inference engine: we drive llama.cpp and vLLM, and in-process llama.cpp through
  `weirdnet-model`.
- Carrying NCCL/Ray tensor traffic over Weirdnet. Weirdnet carries control and tokens; tensor
  parallel traffic stays on the fast private link (RoCE, NVLink, 10/25/100 GbE).
- Replacing Tailscale for SSH and general admin. Tailscale stays if you use it; Ymir doesn't need it.

## Minimum hardware

| Tier | Hardware | What you get |
|---|---|---|
| Minimum (CPU) | x86_64 with AVX2, or ARM64; 8 GB RAM; 20 GB free disk | llama.cpp on CPU (`cpu-dispatch`), Gemma 4 E2B at Q4, a few tokens/s. Fine as a pool member for small jobs or embeddings. |
| Entry GPU | NVIDIA with 8 GB VRAM (Turing sm_75 or newer), AMD RDNA2+ with 8 GB (ROCm or Vulkan), or Apple Silicon with 16 GB | llama.cpp GPU offload; Gemma 4 E4B plus a 7-9B Q4 model. |
| Comfortable | 16-24 GB VRAM, or a 32-64 GB Mac | A 27-35B Q4 model (dense, or MoE like `qwen3.6-35b-a3b`) plus a small stand-in and an embedding model. |
| vLLM | Linux, NVIDIA sm_80+ (Ampere or newer, including GB10 sm_121 with CUDA 13 builds) with 24 GB+, or AMD Instinct / RDNA3 on the ROCm list | vLLM as an extra backend: higher concurrency and NVFP4/FP8 checkpoints. |
| DGX Spark | GB10, sm_121, 128 GB unified memory | The server-a/server-b setup on one box: `qwen3.6-35b-a3b` on llama.cpp, a 27B NVFP4 on vLLM, Gemma 4 E4B stand-in, embeddings. |

Disk: models are big. The setup shows the download size and asks before fetching (the
`weirdnet-model` `Consent` rule).

## Single-box quickstart

```sh
# 1. Install (user-level, no root): a prebuilt binary into ~/.local/bin, no Rust needed.
#    Also available as a TAUR package, .deb, .rpm, Nix flake, Homebrew and a Windows installer.
curl -fsSL https://github.com/WeirdBrains/ymir/releases/latest/download/install.sh | sh

# 2. Set up: detects the hardware, picks a backend and models that fit, shows sizes,
#    asks before downloading, writes ~/.config/ymir/node.toml and starts the user service.
ymirctl setup

# 3. Use it. Point Jan (or anything OpenAI-compatible) at the local bridge:
#    Base URL http://127.0.0.1:8484/v1   API key: the value `ymirctl bridge token` prints
ymirctl status
ymirctl chat "Hello"
```

`ymirctl setup --dry-run` prints the hardware report, the chosen backends and models and the
download plan, and changes nothing. `--yes` accepts the defaults for unattended installs, and
`--system` installs a system service instead of a user service (for headless servers like the Sparks).
Adding Redis, LAN HTTPS, alerts or a second node are later, separate `ymirctl` commands.

## Architecture

```
          Jan / scripts / agents                          Bragi (GPUI desktop, Flutter phone)
                  │ OpenAI HTTP (loopback)                          │ in-process
          ┌───────▼────────┐                                ┌───────▼────────┐
          │ weirdnet-openai│  bridge                        │ bragi-core     │
          └───────┬────────┘                                └───────┬────────┘
                  └──────────────┬─────────────────────────────────┘
                         weirdnet-pool (client: discovery, catalog, routing, streaming, cancel)
                                 │  Weirdnet streams (Noise, relays, hole punching)
     ┌───────────────────────────┼──────────────────────────────┐
┌────▼──────────── ymird (node) ─────────────┐        ┌──────────▼──── ymird (node) ──┐
│ weirdnet-serve: control, caps, health      │◄──────►│ ...same...                    │
│ weirdnet-agent remote::infer InferHost     │ caps,  │                               │
│ weirdnet-route: queues, fairness, placement│ jobs,  │                               │
│ weirdnet-semcache: local + pool cache      │ cache  │                               │
│ weirdnet-backends: llama.cpp, vLLM, Ray,   │        │                               │
│   llama.cpp RPC; process supervision       │        │                               │
│ weirdnet-model: hw detect, catalog, sizing │        │                               │
│ weirdnet-alerts: rules, dedupe, sinks      │        │                               │
└────────────────────────────────────────────┘        └───────────────────────────────┘
        tensor-parallel traffic (NCCL over RoCE, llama.cpp RPC) only on the private link
```

### Node daemon (`ymird`)

One long-lived process per machine (systemd user or system unit, launchd on macOS, a Windows
service running as `NT SERVICE\ymird` per AGENTS.md). It owns one `weirdnet-net` `Node` with the
Ymir `Profile` (scheme `weirdnet-ymir://`; it bootstraps from the public WeirdBrains relay by default,
and the operator can override that). It:

- probes the hardware and keeps a live **capability record** (below);
- supervises backend processes through `weirdnet-backends` (start, health check, swap, idle
  unload, restart with backoff); this replaces llama-swap;
- serves inference on `remote::infer` (`/weirdnet-agent/infer/1`) through `InferHost`, whose
  `Admission` hook is wired to the `weirdnet-route` queue (it answers `Busy` when the queue is full);
- serves control and health on `weirdnet-serve` protocols;
- runs the scheduler role when elected (placement, cache shard map);
- runs the semantic cache and the alert rules.

`ymir-core` holds the Ymir-specific glue: config schema, pool policy, the setup wizard, migration
importers (llama-swap YAML, big-mode), and wiring. `ymirctl` is a thin front end over `ymir-core` and
talks to the local daemon over a Unix socket or named pipe, or to any pool node over Weirdnet.

### Command and job protocol

We **extend `weirdnet-agent` remote::infer** rather than invent a parallel inference protocol (the
Weirdnet lead's advice, and it already has streaming tokens, cancel, `Admission`/`Busy` and
`RemoteModel` fallback). Additions proposed for `weirdnet-agent`:

- request fields: `principal` (user and device, filled in by the server from the authenticated
  peer, never trusted from the body), `priority` (`interactive`, `batch`, `background`), `job_id` and
  `idempotency_key`, sampling parameters, `response_format`, tools;
- an `embeddings` call next to `infer`;
- response metadata: served-by node, model revision, cache hit or stand-in flag, token counts and
  timings (for the router's throughput estimates);
- a `Queued { position }` progress frame before the first token.

The 8 MiB frame cap stays; large inputs (documents for batch jobs) travel as `weirdnet-net`
`content` roots and are fetched by the node.

New protocols in `weirdnet-serve` (generic, not Ymir-specific; JSON or CBOR frames over `weirdnet-net`
streams, versioned ids):

| Protocol | Purpose |
|---|---|
| `/weirdnet-serve/caps/1` | push and pull signed capability records |
| `/weirdnet-serve/control/1` | `Load`, `Unload`, `Pin`, `Drain`, `SetPolicy`, `GroupUp`/`GroupDown` (big mode), `Reload` |
| `/weirdnet-serve/health/1` | health snapshot and a metrics stream (subscribe) |

`weirdnet-route` adds `/weirdnet-route/place/1` (placement plans from the scheduler) and
`/weirdnet-route/job/1` (batch jobs: submit, progress, results stream, cancel). Ymir-only
messages, if any, go under `/ymir/...`.

**Streaming tokens:** each call is one stream. Tokens flow as frames with backpressure (the
`weirdnet-net` stream window); dropping the call resets the stream, which cancels generation on
the backend (llama.cpp and vLLM requests are aborted, freeing the slot). Streams survive a
relay-to-direct path move without losing bytes (`weirdnet-net` 0.8).

### Local OpenAI-compatible bridge (`weirdnet-openai`)

An HTTP server (`/v1/models`, `/v1/chat/completions` with SSE, `/v1/completions`,
`/v1/embeddings`) that translates to `weirdnet-pool` calls. It runs inside `ymird` and can also run
alone (`ymirctl bridge`) on a laptop that serves no models, so Jan on any machine uses the pool
through its own local bridge.

- Binds `127.0.0.1:8484` by default. The local token is generated at setup, stored 0600 in the
  config dir and shown with `ymirctl bridge token`; it guards against other local users only.
  Calls leave the machine as this device's Weirdnet identity.
- `/v1/models` lists the **pool's** models (merged catalog), not just this node's.
- Optional LAN or tailnet listener for clients that can't run a bridge (a phone running stock Jan):
  HTTPS with `weirdnet-tls` certificates and `weirdnet-accounts` API keys with per-key rate limits.
  This is the only place API keys remain, and it is off by default. It replaces Caddy for those clients.
- Error mapping: `Busy` → 503 with `Retry-After`; stand-in answers carry `x-ymir-standin: <model>`;
  cache hits carry `x-ymir-cache: hit`.

### Health, metrics and alerts

- `weirdnet-serve` health: backend states (cold, loading, ready, failed), GPU memory, utilization
  and temperature (NVML, amdsmi, IOKit), unified-memory pressure, disk, queue depths, tokens/s,
  time to first token, cache hit rate, peers and paths (direct or relayed).
- Metrics are also exported as Prometheus text on the loopback bridge (`/metrics`), optional.
- `weirdnet-alerts` (new): rules over health events, dedupe ("at most once an hour", plus a
  recovery notice, as the Wi-Fi fallback alert does), and sinks: Weirdnet push to the owner's devices
  (`weirdnet-push`), SMTP (the password lives in the OS keyring, never in config) and webhooks.
- Built-in rules carried from server-a/server-b: backend crash loop, model load failure, peer lost, **network
  degraded** (the primary interface is down and traffic is on a fallback route, i.e. the Wi-Fi
  fallback alert, generalized), relayed-only for too long, disk low, GPU hot.
- Pool-wide: every node watches its peers' capability records, so no separate monitor host
  (the dev box's monitor timer) is needed. The elected scheduler sends "node X gone" once.

### Cold-start fallback

The server-a coordinator answered with `gemma4-e4b` while the 27B loaded. Generalized:

1. Route to a node where the model is **warm** if one exists (the router prefers warm replicas).
2. Otherwise start loading it (the `GroupUp`/`Load` control) and, if the pool policy names a
   `standin` for that model and the load estimate is over `standin_after` (default 10 s), answer
   with the stand-in, flagged. A client can opt out per request (`x-wait: 1` as today, or
   `wait_for_model` in the Weirdnet request).
3. Load time estimates come from measured history per node and model (85 s for the 27B on server-b).

## Single node (primary target)

### Hardware detection

Generic detection goes in `weirdnet-model` (a new `hw` module next to its CPU checks), so Morph and
Mouser can size in-process models the same way.

| Platform | How | Notes |
|---|---|---|
| NVIDIA | NVML (`nvml-wrapper`, loaded at runtime), driver and CUDA version, compute capability | **GB10 (sm_121)**: NVML reports no dedicated memory; treat it as unified and use system memory (`/proc/meminfo`) minus an OS reserve. CUDA 12.8 builds stop at sm_120: require CUDA 13 builds of llama.cpp, torch and vLLM on sm_121 (the a training host lesson). |
| AMD | amdsmi or sysfs (`mem_info_vram_total`), gfx target | ROCm for supported gfx targets, else Vulkan. APUs (Strix Halo) are unified memory. |
| Apple | `sysctl hw.memsize`, Metal `recommendedMaxWorkingSetSize` | Unified memory; Metal llama.cpp. |
| Intel and others | Vulkan device query | llama.cpp Vulkan. |
| CPU | existing `weirdnet-model::cpu` checks | `cpu-dispatch` llama.cpp. |

The report (`ymirctl hw`) lists devices, memory, the usable budget, interconnects (RDMA/RoCE
devices, link speeds; used later for splitting) and what each backend can do here.

### Backend choice

| Backend | Where | Role |
|---|---|---|
| llama.cpp `llama-server` (managed process) | Everywhere: CUDA, ROCm/HIP, Metal, Vulkan, CPU; Linux, macOS, Windows, BSD | Default backend. GGUF models. |
| In-process llama.cpp (`weirdnet-model` `llama`) | Everywhere | Tiny models and embeddings without an extra process; Bragi's offline mode. |
| vLLM (managed process, Rust frontend as on server-a/server-b) | Linux, NVIDIA sm_80+ (sm_121 needs the CUDA 13 build), AMD ROCm on vLLM's supported list | Optional. High concurrency, NVFP4/FP8. Never on macOS, Windows or BSD. |
| vLLM + Ray | Linux nodes with a fast private link | Split models (see big mode). |
| llama.cpp RPC (`rpc-server`) | Everywhere llama.cpp builds | Split GGUF models across heterogeneous nodes. |

`weirdnet-backends` installs pinned, checksummed backend builds per platform into its own
directory (with consent), or uses a system one the operator points to. vLLM gets its own uv-managed
venv (as `/opt/llm/vllm` today).

### Model sizing

`weirdnet-model`'s catalog grows from Gemma 4 only to a tiered catalog with per-quant memory
footprints. Setup picks models whose weights plus a KV budget (context × concurrency) fit within
the usable budget (default 85% of VRAM, or of unified memory after the OS reserve), and proposes:

- one **main** chat model (largest that fits with headroom),
- one **stand-in** (Gemma 4 E2B/E4B),
- one **embedding** model (`qwen3-embedding-0.6b`) if the cache is on.

The operator can change any of them. Swap groups (as in llama-swap) decide what may be resident at
the same time; idle models unload after a TTL.

## Growing the pool

### Joining

Two ways, both built on what Weirdnet has:

1. **Your own device (full trust).** Link it as a device of your Weirdnet identity
   (`weirdnet-device-link`, the QR offer and SAS check; it needs a bump from `weirdnet-net` 0.4 to 0.8
   first). It appears in your `weirdnet-sync` roster, and `Roster::check_caller` admits it as a linked
   device: it can serve, use and command the pool. `ymirctl pool link` on the new machine prints the
   offer; you approve on any existing device.
2. **Someone else's node or a shared user (scoped trust).** You issue an invite
   (`weirdnet-net` `invite`, a `weirdnet-ymir://join/...` link). They send a `JoinRequest`; you approve
   it, which issues a signed `Membership` on resource `pool:<id>` with a role from `weirdnet-access`.
   Revocation is a `MemberRevocation`.

Pool roles (`weirdnet-access` role defaults supplied by Ymir): `owner` (linked devices), `admin`
(policy, placement, members), `operator` (control: load, unload, drain, big mode), `user` (inference,
embeddings, jobs), `compute` (a node that contributes capacity but whose owner can't command yours).

Discovery uses `weirdnet-net` as it is: nodes find each other by key (`request_key`), user record and
`peer_apps` (each node does `serve_app("ymir", …)`), with `PeerHint`s for LAN and fixed addresses.
mDNS finds same-LAN nodes, but mDNS presence never grants trust; the roster and memberships do. No
new discovery mechanism is proposed.

### Capability advertisement

Every node publishes a **capability record** over `/weirdnet-serve/caps/1` to pool members (not to the
DHT): signed by the node key, chained to its user by `NodeCert`, with `seq` and a short expiry (90 s,
refreshed every 15 s and on change), like `weirdnet-net` peer records. Contents:

- hardware: accelerators, memory budget, interconnects and link speeds, backends available;
- models: id, revision, quant, backend, state (cold, loading, ready), context and concurrency
  limits, load-time estimate;
- load: queue depth per priority, active slots, EWMA prefill and decode tokens/s per model, recent
  time to first token, memory free;
- policy flags: draining, `compute`-only, accepts batch, accepts group jobs.

`weirdnet-pool` merges the records it can see into the **pool catalog**: the union of models with
their replicas and current load. That is what `/v1/models` and Bragi's model picker show.

### Leaving and failure

- **Graceful leave:** `ymirctl drain` stops new admissions, finishes or hands back queued work,
  publishes a final record with `draining`, then a withdrawal.
- **Crash or network loss:** its record expires (≤ 90 s; peers also notice the connection drop
  sooner). The scheduler marks its replicas gone and re-places popular models.
- **In-flight requests:** if the node fails before the first token, `RemoteModel` already retries the
  next node in order. After the first token the client gets an error with the partial text; the
  bridge and Bragi can offer "continue on another node", which resends the conversation plus the
  partial answer (open question: make that automatic?).
- **Batch jobs:** per-task retry on another node, by idempotency key.
- **Group (split) jobs:** losing any member tears the group down (NCCL can't survive a member
  loss); the leader reports, members return to their normal models, the router falls back to the
  stand-in or a smaller model, and an alert goes out.
- **Scheduler loss:** the role is a lease (below); another node takes it when the lease lapses.
  Routing keeps working without a scheduler, because placement plans persist and every client
  routes for itself.

## Workload spreading (core)

This is what makes the pool one endpoint rather than a list of servers.

### Who decides what

- **Routing is client-side and per request**, in `weirdnet-pool`, from the capability records the
  client sees: no hop through a central router, and no single point of failure.
- **Admission is node-side:** each node's `weirdnet-route` queue accepts or answers `Busy`.
- **Placement is pool-wide** and done by one **scheduler** at a time: a lease held by an online
  `owner` node, chosen by lowest peer id among eligible nodes (the same tie rule `weirdnet-sync`
  uses for first rotation), renewed in the capability records. Its output is a signed, versioned
  placement plan; nodes apply it with `Load`/`Unload`. If no scheduler is online, nodes keep the last
  plan.

### Load balancing across replicas

For a model with several warm replicas, the client scores each candidate by predicted completion:

```
wait  = queue_depth(priority and above) × mean_service_time
cost  = wait + prompt_tokens / prefill_tps + expected_tokens / decode_tps + rtt
       + (cold ? load_estimate : 0) + (relayed ? relay_penalty : 0)
```

- **Power of two choices:** score two random candidates (or all, if few) and take the better one,
  so stale records don't make every client pile onto the same "least loaded" node.
- **Affinity:** a conversation's follow-up turns prefer the node that served the previous turn
  (prefix and KV cache reuse), unless that node's cost exceeds the best by a set margin.
- **Busy feedback:** a `Busy` answer moves on to the next candidate, and the node's score is
  penalized briefly. A node close to full reports its queue in its record, so clients back off
  before hitting `Busy`.
- **Local first, on ties:** the local node wins ties (no network), which is also the single-box path.

### Placement and replication

The scheduler decides which models live where from: the pool policy (models wanted, `min_replicas`,
`max_replicas`, pins such as "27B NVFP4 only on Spark-class nodes"), each node's memory budget and
backends, and demand (EWMA requests/min and tokens/min per model from the records).

- Bin-pack by memory with headroom; prefer nodes where the backend is fastest for that format
  (NVFP4 → vLLM on Blackwell; GGUF → llama.cpp).
- **Replicate popular models:** when a model's queue wait stays over a target (default 2 s for
  interactive), add a replica on the node with the most free budget; drop extra replicas after
  sustained low demand, keeping `min_replicas`.
- Keep the stand-in and the embedding model warm on at least one node each, always.
- Changes are rate-limited (no flapping) and never evict a model with active requests; `Drain`
  first.

### Splitting batch and agent jobs

`/weirdnet-route/job/1` accepts a job: a list of independent tasks (prompts, embedding chunks over
a corpus given as a content root, or `weirdnet-agent` `remote::task` subtasks with their
approvals). The submitting client's `weirdnet-pool` (or a node on its behalf, for long jobs) fans
tasks out with the same scoring, at `batch` priority, with a concurrency cap per node, per-task
retries by idempotency key, and results streamed back as they finish (or in order, on request).
Cancel stops all of them. Agent subtasks keep `weirdnet-agent`'s approval and budget rules: a
subtask never gains tools the parent didn't have.

### Shared semantic cache

`weirdnet-semcache` (new) generalizes the RedisVL `rvl-cache`:

- **Rules carried over from `rvl-app.py` `cacheable()`:** single user turn, text only, no tools, one
  choice; the key context is a hash of the system prompt and sampling parameters; models can opt out.
  Cosine-distance threshold **0.12** (server-a's deployed value), TTL 24 h.
- **Local mode (default, single box):** an embedded vector index (HNSW) in an encrypted
  `weirdnet-storage` database. Redis/RedisVL is an optional backend behind the same trait.
- **Pool mode:** entries are sharded by rendezvous hashing on `(scope, model)` to a home node plus
  one replica, so a lookup is one call. All nodes must use the same embedding model and revision,
  which the pool policy pins; a change starts a fresh cache.
- **Scope:** a cache entry is visible only within its scope: per user by default, or pool-wide if
  the owner turns that on. Cached answers never cross users without that setting.
- Hits are flagged to the client and can be skipped per request (`cache: false`).

### Fairness and priorities

Replaces Caddy's per-key `rate_limit` blocks (`@m3talsmith`, `@ci`):

- Each node queues per principal (user, then device), served by **deficit round robin** weighted by
  role and owner-set weights, inside three priority classes: `interactive` > `batch` > `background`.
  Batch work yields between requests, never mid-stream.
- Quotas per role or principal: requests/min and tokens/min, enforced on each node and summed
  pool-wide by the scheduler (soft, eventually consistent).
- Owner traffic can be guaranteed a share (for example 50% of each owner node) so shared users can
  never starve the owner's own machines.
- A `compute` node's own owner always gets priority on their own hardware.

## Splitting models across nodes (big mode, generalized)

On server-a/server-b, `big-mode on` stopped llama-swap on both boxes, started a Ray head and worker over the
ConnectX-7 link (NCCL over RoCE, Ray ports 23001 and 23100-23999) and served one big model with
vLLM. Generalized as a **group job**:

1. A model in the pool policy is marked `split` with allowed backends (`vllm-ray` for TP/PP across
   Linux NVIDIA nodes; `llamacpp-rpc` for GGUF across any nodes).
2. The scheduler picks a group whose combined budget fits, preferring nodes with a measured fast
   link between them (RDMA/RoCE ≥ 100 Gb/s for tensor parallel; pipeline parallel or llama.cpp RPC can
   live with 10 GbE; it refuses over Wi-Fi and warns over relays).
3. `GroupUp` goes to every member: each drains the models the plan says to drop, starts its worker
   (Ray worker or `rpc-server`) bound **only to the private-link address**, and reports ready. The
   leader starts the serving process (vLLM with Ray, or `llama-server --rpc`) and advertises the model.
4. Inference reaches the leader over Weirdnet like any model. Tensor traffic stays on the link.
5. `GroupDown` (or a member loss) reverses it, and normal placement resumes.

Big mode can be triggered manually (`ymirctl group up qwen-big`), on a schedule, or by the
scheduler when demand for the split model crosses a threshold. The NCCL settings in
the deployed `nccl.env` become backend defaults for RoCE groups (set by detected
hardware, not hard-coded per host).

## Mapping from server-a/server-b to Ymir

| server-a/server-b piece | New home |
|---|---|
| llama-swap + `config.normal.yaml` (model processes, swap groups, TTL) | `weirdnet-backends` supervisor; importer `ymirctl import llama-swap` in `ymir-core` |
| llama.cpp builds in `/opt/llm/llama.cpp` | `weirdnet-backends` pinned builds (CUDA 13 for sm_121) |
| vLLM 0.34 + Rust frontend, qwen3 reasoning parser, FlashInfer prebuilt, 1800 s startup timeout | `weirdnet-backends` vLLM adapter (options in pool policy) |
| `config.big.yaml`, `config.big.<host>.yaml`, `/opt/llm/bin/big-mode` | group jobs in `weirdnet-route` + `weirdnet-serve` `GroupUp`/`GroupDown`; importer in `ymir-core` |
| `ray-head`/`ray-worker` units, Ray ports 23001/23100-23999, memory monitor off | `weirdnet-backends` Ray adapter |
| `nccl.env` (NCCL over RoCE, NET/IB) | `weirdnet-backends` RoCE defaults from detected hardware |
| `rvl-cache.service` / `rvl-app.py` semantic cache, threshold 0.12 | `weirdnet-semcache` (embedded default, Redis optional) |
| rvl-app cold-start coordinator (`gemma4-e4b` while the 27B loads, `/rvl/status`, `/rvl/warm`) | stand-in policy in `weirdnet-route`; warm/status via `weirdnet-serve` control and health |
| Redis 8 | optional `weirdnet-semcache` backend |
| Caddy, `tls internal`, ports 443/9443, per-tenant keys in `keys.env`, `rate_limit` | Weirdnet identity + roster + memberships; `weirdnet-route` fairness and quotas; for non-Weirdnet clients, the optional `weirdnet-openai` LAN listener with `weirdnet-tls` + `weirdnet-accounts` keys |
| `server-b/<model>` name prefixes for models on server-b | the pool catalog (one name per model; routing picks the node) |
| `ts-watchdog`, Tailscale hardening | stays as host ops, outside Ymir (Tailscale becomes optional); a `tailscale` health probe can feed `weirdnet-alerts` |
| monitor timer on the dev box | peer watching via capability records + `weirdnet-alerts` |
| msmtp Gmail alerts, `ts-alert` | `weirdnet-alerts` SMTP and push sinks (secret in the keyring) |
| Wi-Fi fallback alert (the old route-state and netcheck timers) | `weirdnet-alerts` "network degraded" rule from `weirdnet-serve` health (default-route interface) |
| server-a NAT, MSS clamp, subnet route for server-b | host networking, out of scope; Weirdnet relays and hole punching reach server-b without it |
| `hfget`, `hfcli` | `weirdnet-model` download with consent |
| `reclaim.sh` | out of scope (host maintenance) |
| `e2e*.sh` smoke tests | Ymir integration tests (`ymirctl selftest`) |

## Crates

### Existing, reused (versions as of 2026-11-19; all 0.x and unaudited)

| Crate | Used for |
|---|---|
| `weirdnet-net` 0.8.0 | node, streams, `serve_app`/`peer_apps`/`resolve_app`, key-only dialing, relays, hole punching, invites, `member`/`join`, `content` for large inputs, `events`/`network_changed` |
| `weirdnet-keys` 0.1.1 | node and user keys, sealing |
| `weirdnet-sync` 0.5.0 | device roster, `Roster::check_caller`, synced pool policy and Bragi history (LWW, OR-set). Its next release moves to net 0.8. |
| `weirdnet-access` 0.1.1 | pool roles and enforcement order |
| `weirdnet-device-link` 0.1.0 | linking your own machines (needs a bump to net 0.8) |
| `weirdnet-agent` 0.2.0 | `remote::infer` (`InferHost`, `Admission`, `RemoteModel`), `remote::task`, `ModelBackend` |
| `weirdnet-model` 0.2.1 | catalog, consent-gated download, in-process llama.cpp, CPU checks |
| `weirdnet-tls` 0.2.0 | optional LAN HTTPS for the bridge |
| `weirdnet-accounts` 0.1.0 | API keys and rate limits for the optional LAN listener only |
| `weirdnet-storage` 0.1.0 | encrypted local databases (cache, history, metrics) |
| `weirdnet-push` | alert and wake delivery to phones |
| `weirdnet-schemes`, `weirdnetctl` | registering `weirdnet-ymir://` and `weirdnet-bragi://` |
| `weirdnet-update` | signed update metadata, if Ymir self-updates outside package managers |
| `weirdnet-mcp` 0.3.0 | optional `ymirctl mcp` tools (status, load, chat) |
| `weirdnet-relay` 0.5.0 | optional own relay (needs a bump to net 0.8) |

### Extensions to existing crates

- `weirdnet-agent`: the `remote::infer` additions listed above (principal, priority, job ids,
  embeddings, metadata, `Queued`).
- `weirdnet-model`: `hw` detection, tiered catalog with memory footprints, a sizing function.
- `weirdnet-device-link`, `weirdnet-relay`, `weirdnet-sync`: move to `weirdnet-net` 0.8.

### New crates proposed (names are suggestions; m3talsmith decides)

| Crate | Holds |
|---|---|
| `weirdnet-serve` | node control, capability records, health and metrics protocols; the server side a node runs |
| `weirdnet-backends` | external backend adapters and process supervision: llama.cpp server, vLLM, Ray, llama.cpp RPC; pinned build installs (could instead be features of `weirdnet-model`, open question) |
| `weirdnet-route` | node admission queues, fair queuing and quotas, scoring, placement and replication, scheduler lease, group jobs, batch jobs |
| `weirdnet-pool` | the client library: discovery of serving nodes, merged model catalog, per-request routing, streaming, cancel, retries, local fallback. Used by `weirdnet-openai`, `bragi-core` and `ymird` |
| `weirdnet-openai` | the OpenAI-compatible HTTP bridge over `weirdnet-pool` |
| `weirdnet-semcache` | semantic cache: rules, embedded index, optional Redis backend, pool sharding |
| `weirdnet-alerts` | alert rules, dedupe and recovery notices, push/SMTP/webhook sinks |

Ymir-specific crates: `ymir-core` (config, pool policy, setup wizard, importers, wiring), `ymird`,
`ymirctl`.

## Security and auth

- **Identity:** every node and client speaks as a Weirdnet node with a `NodeCert` chained to a user.
  Noise authenticates every stream end to end, relays included.
- **Authorization** happens on the serving node, per call, in this order: is the caller a linked
  device of the owner (`Roster::check_caller`)? Otherwise does it present a valid, unrevoked
  `Membership` for `pool:<id>` (`check_presented`)? Then the role from `weirdnet-access` decides the
  verb: `infer`, `embed`, `job`, `control`, `admin`. `InferHost::serve` is only reached after this
  check (the rule `weirdnet-agent` already states).
- **No API keys between Weirdnet peers.** Revoking a device or membership cuts access everywhere
  as soon as nodes see the revocation.
- **Prompts and answers** cross only Noise streams. Nodes don't log prompt or answer text by
  default; metrics carry counts and timings only. Cache entries are scope-bound and encrypted at rest.
- **Unauthenticated backend ports** (llama-server, vLLM, Ray, NCCL, llama.cpp RPC) bind to loopback,
  or for groups to the private-link address only, never to a LAN or public interface. Ray and RPC
  have no auth of their own; the doc treats the private link as trusted, so it must be a direct cable
  or an isolated VLAN.
- **Private by default:** a pool is Private (not visible outside its members). Capability records
  go only to members, never to the DHT.
- **Secrets** (SMTP password, LAN API keys, bridge token) never appear in config files, logs or
  docs. Config references keyring entries by name.
- Downloads of backends and models are checksummed and pinned; model downloads need consent.

## Config format (one file per node)

`node.toml` (in `~/.config/ymir/` for a user install, `/etc/ymir/` for a system install; `YMIR_HOME`
overrides). Pool-wide settings (models wanted, replicas, stand-ins, cache, quotas) live in the
**pool policy**, synced among owner devices with `weirdnet-sync` and edited with `ymirctl policy`;
`node.toml` may seed it on the first node.

```toml
# ~/.config/ymir/node.toml (example; no secrets belong here)
[node]
name = "server-a"
roles = ["serve", "bridge"]          # serve, bridge, scheduler-eligible, compute-only

[network]
bootstrap = "default"                # the public WeirdBrains relay; or a list, or []
listen = ["/ip4/0.0.0.0/tcp/4747", "/ip4/0.0.0.0/udp/4747/quic-v1"]
private_link = "10.0.0.1"         # optional: address for group jobs (RoCE/NCCL, RPC)

[hardware]
memory_budget = "auto"               # or e.g. "100GiB"
reserve = "12GiB"                    # unified memory kept for the OS

[backends]
llamacpp = "managed"                 # managed | system:/path | off
vllm = "auto"                        # auto (only where supported) | off

[[models]]
id = "qwen3.6-35b-a3b"
backend = "llamacpp"
quant = "Q4_K_M"
swap_group = "main"

[[models]]
id = "gemma4-e4b"
role = "standin"

[[models]]
id = "qwen3-embedding-0.6b"
role = "embedding"

[bridge]
listen = "127.0.0.1:8484"
lan = false                          # true enables HTTPS + API keys for non-Weirdnet clients

[cache]
mode = "local"                       # off | local | pool
backend = "embedded"                 # embedded | redis
redis_url = "redis://127.0.0.1:6379" # used only when backend = "redis"
threshold = 0.12

[alerts]
push = true                          # Weirdnet push to the owner's devices
smtp = { to = "owner@example.com", server = "smtp.gmail.com:587", user = "owner@example.com", password_ref = "keyring:ymir-smtp" }
```

## Install and deploy

- **Packages** per the platform matrix: Linux `tar.gz`, `deb`, `rpm`, `pkg.tar.zst` (TAUR), Nix flake
  (each with and without a `.desktop` entry where relevant; the daemon has none), x86_64 and aarch64;
  macOS (Apple Silicon Metal; Intel CPU) as `zip` and Homebrew; Windows `zip`, standalone `exe` and a
  WiX 7 installer (`-acceptEula wix7`), service as `NT SERVICE\ymird`; FreeBSD `pkg`/`tar.gz`,
  OpenBSD `tar.gz`, GhostBSD and DragonFly BSD after them (CPU and Vulkan; no vLLM).
- **Android and iOS: opt out for `ymird`** with the reason "a serving daemon can't run in the
  background on mobile (iOS forbids it; Android needs a foreground service and thermals make it
  impractical)". Phones are clients through Bragi.
- **User-level by default** (systemd user unit with lingering, launchd agent); `--system` for headless
  servers where it is genuinely system-level.
- Prebuilt binaries only; installing never needs Rust. Release files are signed with minisign.
- `install.sh` detects OS and arch, downloads and verifies the signed archive, installs to
  `~/.local/bin`, then runs `ymirctl setup` (skippable).
- Backends are fetched by `weirdnet-backends` with consent (vLLM's venv is large).

### Releases

- The shared release script: `scripts/weirdnet_release.py` vendored from `weirdnet-project-helpers`,
  configured in `release.toml` with the `[[platforms]]` matrix (including the mobile `opt_out`s and
  the BSD rows once the script supports their os values).
- Every public crate (`ymir-core`, `ymird`, `ymirctl`, and the new `weirdnet-*` crates in their own
  repos) is published to crates.io on each release that affects it. Apps track Weirdnet main during
  development and pin published versions for releases (`[[requires]]` for lockstep crates).
- After the GitHub release: `weirdnet_release.py assets`, then the TAUR recipe with
  `weirdnet_release.py taur X.Y.Z --dry-run` and then for real (as Morph does: GitHub, TAUR and Nix
  channels on Linux).
- Repos are public (BSD-3-Clause proposed).

## Migration plan for server-a/server-b

Nothing here runs until m3talsmith approves each step. Every step is reversible: the current units
are stopped, not removed, until the final step.

1. **Shadow:** install `ymird` on server-a and server-b with the bridge on a different port, `serve` off. Import
   `config.normal.yaml` and big-mode into a pool policy and diff it against the live configs.
2. **Second backend owner:** on server-b, let `ymird` manage the 27B NVFP4 vLLM and embeddings while
   llama-swap is stopped (rollback: start llama-swap). Join server-a and server-b as linked devices.
3. **server-a cut-over:** `ymird` takes over llama.cpp models and the stand-in on server-a. Caddy now proxies to
   the `weirdnet-openai` bridge on server-a, so existing clients and keys keep working during the switch.
4. **Clients:** Jan on the dev box and other machines move to their own local bridge (or Bragi). CI
   gets a node identity with the `user` role, or a LAN API key on the optional listener.
5. **Cache:** switch to `weirdnet-semcache` (start cold; RedisVL entries aren't migrated). Redis can
   stay as its backend if wanted.
6. **Big mode:** replace `/opt/llm/bin/big-mode` with the group job over the same ConnectX-7 link
   and NCCL settings. Verify with the existing `e2e-big.sh` prompts.
7. **Alerts:** move the Wi-Fi fallback and other alerts to `weirdnet-alerts`, keep `ts-watchdog` (host
   ops), retire `monitor timer` once peer watching proves itself for a week.
8. **Retire:** after a quiet week, disable Caddy, rvl-cache and llama-swap units, and remove
   them only with m3talsmith's explicit approval.

## Client app (Bragi)

A Jan-like chat app with Weirdnet built in. It is a pool client: it never needs the bridge.

**One app, two shells.** The desktop app (GPUI) and the phone app (Flutter) work exactly the same
way. Both (1) get their models from the Ymir inference nodes and (2) send prompts across the cloud.
All of that behaviour lives in one crate, `bragi-core`: the same discovery, the same merged model
list, the same routing, the same streaming and cancel, the same history and pairing. Only the UI
shell differs. Neither platform is a special case: there is no desktop-only or phone-only way of
finding nodes, picking models or sending prompts. Anything that exists on only some devices is an
optional extra, offered the same way on both and marked as such below.

### Same repo or its own?

**Its own repo, `WeirdBrains/bragi`.** Reasons:

- Different platform matrix: Bragi ships desktop and mobile (apk, aab, F-Droid, ipa); `ymird`
  opts out of mobile. One `release.toml` per repo keeps each matrix honest.
- Different toolchains and cadence: GPUI, Flutter, Xcode and Android SDKs for Bragi; a Rust daemon
  and GPU backends for Ymir. Store reviews shouldn't hold up server fixes, or the reverse.
- Everything they share is generic and lives in `weirdnet-*` crates (`weirdnet-pool`,
  `weirdnet-agent`, `weirdnet-sync`), so neither repo depends on the other. Bragi works with any
  `weirdnet-serve` node, not only Ymir.
- It matches Morph and Muninn: one product per repo, with a Rust core and thin shells.

### What it does (identical on desktop and phone)

1. **Finds nodes and models:** `weirdnet-pool` discovers serving nodes the user can reach (own
   linked devices, pools they're a member of), reads capability records and merges them into **one
   model picker** with live status (warm, cold, loading, busy), the best node's estimated speed and
   whether it's local.
2. **Sends prompts over Weirdnet** to the node the router picks, with streaming tokens, a stop
   button (cancel resets the stream), retry before the first token, and "continue on another node"
   after a failure.
3. **Optional extra, same on both: offline on-device model.** If the user turns it on, with no
   reachable node `bragi-core` runs a small model in-process through `weirdnet-model` (`llama`;
   Metal on macOS and iOS, CUDA, Vulkan or CPU elsewhere), downloaded only with consent. It is off by
   default on every platform and shows in the same picker as "On this device". It is not how the
   app normally works: the normal path is always the cloud.
4. **History:** chats are stored in an encrypted `weirdnet-storage` database and synced between the
   user's linked devices with `weirdnet-sync` (an op log; messages are append-only, titles and
   settings LWW, folders and tags as OR-sets). Sync is per device-roster, end to end encrypted.
5. **Pairing:** the first launch either links to an existing Weirdnet identity on another device
   (`weirdnet-device-link`, the QR and SAS flow) or accepts a pool invite link (`weirdnet-bragi://join/...`
   or a `weirdnet-ymir://` pool invite).
6. Extras later, on both shells at once: attachments (sent as content roots), agent mode through `remote::task` with
   `weirdnet-agent` approvals, MCP tools through `weirdnet-mcp`.

### Stack (decided by m3talsmith): GPUI desktop, Flutter phones, one Rust core

The same shape Morph uses (`morph-core`, `morph-desktop` on GPUI, `morph-bridge` with
`flutter_rust_bridge` 2.13.0 for Flutter on Android and iOS):

```
┌── bragi-desktop (GPUI, Rust) ──┐   ┌── app/ (Flutter, Dart) ─────────────┐
│ views over bragi-core directly  │   │ widgets; Dart streams               │
└──────────────┬─────────────────┘   ├── bragi-bridge (flutter_rust_bridge) ┤
               │                     └──────────────┬──────────────────────┘
               └──────────── bragi-core (one Rust crate) ──┘
                 conversations, history, settings, view models, actions
                 └── weirdnet-pool, weirdnet-sync, weirdnet-storage, weirdnet-model, weirdnet-agent
```

- **`bragi-core`** is the one client core and holds all behaviour; the shells only draw it: it owns the tokio runtime (one per process: the bridge
  starts it on mobile; `bragi-desktop` starts it beside GPUI's executor, as Morph does), the Weirdnet
  node, discovery, the merged catalog, routing, streaming, history and settings. It exposes named
  actions and observable state, so menus, keybindings and both shells dispatch the same actions
  (AGENTS.md: no business logic in front ends).
- The two shells differ only in how they reach the core: GPUI calls it in-process, Flutter through
  the bridge. The calls, events and results are the same on both (the bridge API below is the same
  action set GPUI dispatches).
- **GPUI desktop** (`bragi-desktop`) links `bragi-core` directly; token streams are `futures`
  streams polled on GPUI's executor, which updates the view entity per batch of tokens.
- **Flutter mobile** (`bragi-bridge`, thin): `flutter_rust_bridge` v2, pinned to the version Morph
  uses (2.13.0) to share tooling.

### The FFI boundary and streaming tokens

- The bridge exposes a small, coarse API: `start(config_dir)`, `catalog_stream(sink)`,
  `send(conversation_id, prompt, model_choice) -> request_id`, `cancel(request_id)`,
  `history_page(...)`, `pair_offer()`, `pair_accept(link)`, and settings calls. Plain data structs
  cross (frb generates Dart classes); no Weirdnet types leak to Dart.
- **Tokens cross as `StreamSink<ChatEvent>`** (the pattern `morph-bridge` uses for its control
  events): `ChatEvent` is `Queued{position}`, `Started{node, model, standin}`, `Delta{text}`,
  `Usage{...}`, `Done`, or `Error{kind, partial}`. Rust coalesces deltas every ~16 ms or 32 tokens
  so the Dart isolate isn't flooded; backpressure is the coalescing, not dropped tokens.
- **Cancel** from Dart calls `cancel(request_id)`, which drops the Weirdnet call (stream reset, the
  node aborts generation). Closing the Dart stream subscription does the same.
- Lifecycle is one `bragi-core` code path for both shells: when the OS backgrounds or suspends the
  app (minimised or sleeping laptop, backgrounded phone) the core calls `Node::pause`, on return
  `Node::resume`, and on a network change `Node::network_changed`. If an answer is cut off, both apps
  offer the same "continue". The OS may suspend phones sooner; that changes timing, not behaviour.
- Optional extra on both: push wake-ups (`weirdnet-push`) to say a long job finished, off by default.
- Errors cross as typed enums, never panics; the core catches panics at the boundary.

### Platform targets

Per the AGENTS.md matrix, as Morph's `release.toml` does:

| OS | Shell | Formats | Channels |
|---|---|---|---|
| Linux x86_64/aarch64 (Wayland first) | GPUI | `tar.gz`, `zip`, `deb`, `rpm`, `pkg.tar.zst`, Nix (with and without `.desktop`) | GitHub, TAUR, Nix |
| macOS | GPUI | `zip`, `dmg`, `app` | GitHub (Homebrew later) |
| Windows | GPUI | `zip`, standalone `exe`, WiX 7 installer | GitHub, winget later |
| Android | Flutter | `apk`, `aab` | GitHub, `apk-sideload`, F-Droid, Google Play |
| iOS | Flutter | `ipa` | TestFlight, App Store, AltStore, EU marketplace, ad hoc |
| FreeBSD, OpenBSD, GhostBSD, DragonFly | GPUI if it builds there; else opt out with reason | `pkg`/`tar.gz` | GitHub |

- Desktop windows are frameless with in-app controls (AGENTS.md), close-only under tiling WMs.
- Flutter iOS uses Swift Package Manager, no Podfile.
- F-Droid builds are reproducible with FOSS-only dependencies, no FCM (push off unless the user
  opts in to the relay wake path). The Rust core is built per ABI (`aarch64-linux-android`,
  `x86_64-linux-android`) with cargo-ndk; iOS as an XCFramework (`aarch64-apple-ios`, simulator
  targets). The optional on-device model is included in every build where llama.cpp builds; a
  slimmer build without it is allowed on any platform, and the cloud path is the same with or without.
- Updates: Salty Gull (`salty-client`) for asset and full updates where the platform allows;
  F-Droid off by default.

### Bragi milestones

Every milestone ships its features in `bragi-core` for both shells; a shell lagging a milestone is
a bug, not a design difference. The shared test suite runs against `bragi-core`, and both shells
run the same end-to-end scenario (pair, list models, chat, cancel, continue).

- **B1 Core and both shells:** `bragi-core` + `bragi-desktop` (Linux) + `bragi-bridge` and the
  Flutter app (Android): pair, catalog from one node, streaming chat with cancel, local history.
  Tests over `weirdnet-net` in-memory transports with a fake `InferHost`.
- **B2 Pool client:** multi-node catalog, routing via `weirdnet-pool`, retries, "continue on another
  node", stand-in and cache flags, in both shells.
- **B3 Platforms:** iOS (TestFlight), macOS and Windows; Android apk, aab and F-Droid packaging.
- **B4 Sync and optional extras:** history sync across linked devices; the optional on-device model
  and push wake-ups, on both shells.
- **B5 Breadth:** BSDs where GPUI allows, agent mode and attachments, on both shells.

## Milestones (Ymir)

Each milestone lands its `weirdnet-*` crate work first, published, then the Ymir side.

- **M0 Design (this doc).** Repo, names decided, Weirdnet lead review, crate owners agreed.
- **M1 Single box.** `weirdnet-model` `hw` + sizing; `weirdnet-backends` with llama.cpp; `ymird` with
  `weirdnet-serve` health; `weirdnet-openai` bridge on loopback; `ymirctl setup` and the quickstart
  working on a Spark, an NVIDIA PC, a Mac and a CPU box; embedded `weirdnet-semcache`.
- **M2 Backend parity with server-a/server-b.** vLLM adapter (CUDA 13 on sm_121), swap groups and idle unload,
  stand-in cold start, `weirdnet-alerts` (push, SMTP, network-degraded rule), packages and the
  release script.
- **M3 Weirdnet control.** `remote::infer` extensions; caller auth by roster and membership;
  control protocol; linked devices command a node; Jan on another machine through its local bridge.
- **M4 Pool basics.** Joining (link, invite), capability records, the merged catalog, failover,
  drain.
- **M5 Workload spreading (core).** Load-balanced routing across replicas, scheduler lease,
  placement and replication, fair queuing, quotas and priorities, batch and agent job fan-out,
  pool-mode semantic cache. Exit test: three nodes, two replicas of one model, mixed interactive and
  batch load; interactive p95 wait stays under target and no principal starves.
- **M6 Split models.** Group jobs: vLLM + Ray over RoCE, llama.cpp RPC over Ethernet; big mode on
  server-a/server-b reproduced as a group job.
- **M7 server-a/server-b migration** (steps above, each approved by m3talsmith).
- **M8 Breadth and hardening.** Windows service, BSD packages, a security review of the auth and
  backend-port boundaries, 1.0 criteria.

## Open questions for m3talsmith

1. **Names:** Ymir for the node app and Bragi for the client? Or Smidja/Brokkr, or "Ymir Chat"?
2. **Ownership:** the Weirdnet lead offered to review this doc and own the generic crates
   (`weirdnet-serve`, `-backends`, `-route`, `-pool`, `-openai`, `-semcache`, `-alerts`). Accept, and
   are those names right? Should backend adapters be `weirdnet-model` features instead of a new crate?
3. **llama-swap:** replace it with our own supervisor (proposed), or wrap llama-swap as one backend?
4. **Clients without Weirdnet** (stock Jan on a phone, CI runners): keep an optional LAN/tailnet
   HTTPS listener with API keys, or require a bridge or Bragi everywhere?
5. **Shared users:** will anyone besides your own devices use the pool (CI, teammates, a training host at
   work)? That decides how much of the membership, quota and cache-scope work is in M5.
6. **Mid-stream failover:** automatic "continue on another node", or always ask?
7. **Pool-wide cache sharing** across users: allowed when the owner turns it on, or never?
8. **Private link trust:** Ray and llama.cpp RPC have no auth. Is "direct cable or isolated VLAN
   only" acceptable, or do we need a tunnel (with the bandwidth cost)?
9. **Updates:** package managers only, or `weirdnet-update` signed self-updates for the daemon?
10. **License:** BSD-3-Clause for Ymir and Bragi like Morph and the crates?
