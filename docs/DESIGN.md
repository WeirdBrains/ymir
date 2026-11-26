# Ymir design

Date: 2026-11-19. Status: design settled; every open question was answered on 2026-11-19 (see
[Decisions](#decisions-2026-11-19)). **M1 is built** (single box, all three roles in one `ymird`;
see [M1 status](#m1-status-built-2026-11-19)).
Reference build: a two-server deployment (deployment configs and a plan snapshot are kept off-repo).
This doc follows the collective `weirdnet-agents/AGENTS.md`.

## Vision: a cloud of LLM models

Ymir is a **private, self-hosted model cloud**. You start with one machine. You add more over
Weirdnet. The pool acts as **one endpoint**, and the models are **spread across it**: requests go to
the least-loaded node that has the model warm, popular models get replicas, models swap across
nodes as demand moves, batch and agent jobs fan out, and a model too big for any one machine is split
across several. Conversations and their context are synced across nodes, so any node can carry on any
conversation. Nothing listens on a network port except Weirdnet. The pool is m3talsmith's own linked
devices, and only they can use or command it.

Three roles make up the cloud: **model nodes** run the models, **inference nodes** coordinate
(catalog, load balancing, placement, failover, context), and **app nodes** (Bragi and any
`weirdnet-agent` app) ask for models and send prompts. One machine can hold all three, or they can
be spread over any number of devices (see [Roles](#roles)).

Two deliverables share one design:

1. **Ymir** (this repo): the daemon `ymird` (model, inference and app roles as toggles), the CLI
   `ymirctl`, and a loopback-only OpenAI-compatible bridge for local tools such as Jan.
2. **Bragi** (separate repo, see [Client app](#client-app-bragi)): a Jan-like chat app with
   Weirdnet built in. GPUI on desktop, Flutter on phones, one shared Rust core; both apps behave
   the same.

## Names (confirmed 2026-11-19)

- **Ymir**: the node app. In Norse myth the clouds were made from Ymir's brain. Crates `ymir-core`,
  `ymird`, `ymirctl` (the bare `ymir` name is taken on crates.io). Scheme `weirdnet-ymir://`.
- **Bragi**: the client app (Norse god of poetry and speech). Crates `bragi-core`, `bragi-desktop`,
  `bragi-bridge`. Scheme `weirdnet-bragi://`.

Both schemes are registered in `weirdnet-schemes`.

## Ownership

| Area | Owner | Notes |
|---|---|---|
| Generic `weirdnet-*` crates (new ones and the extensions below) | Weirdnet Lead Developer | Also reviews this doc. Builds and publishes the crates; Ymir and Bragi consume released versions. |
| Ymir (`ymir-core`, `ymird`, `ymirctl`), this repo | m3talsmith and this team | The inference-node coordinator (routing, cloud-wide swap and placement, failover, context ownership), role wiring, control, setup, migration. |
| Bragi (`bragi-core`, `bragi-desktop`, `bragi-bridge`, Flutter app), `WeirdBrains/bragi` | m3talsmith and this team | The chat client on desktop and phone. |
| Design decisions and naming | m3talsmith | |

When Ymir or Bragi needs something generic, we file it with the Weirdnet lead and build against
Weirdnet main until it ships (AGENTS.md); we don't build it inside the app.

## Goals

- **One machine first.** One command runs all three roles on one box. It turns a DGX Spark, a PC with an NVIDIA or AMD GPU, an Apple
  Silicon Mac or a CPU-only box into a working node, with a backend and models picked for its hardware.
- **Grow over Weirdnet.** Add more of your own machines by linking them. The pool behaves as one
  endpoint, with active load balancing, cloud-wide model swapping and replication, job splitting,
  synced conversation context, a shared semantic cache, and fairness between devices.
- **Split models too big for one node** (server-a/server-b big mode, generalized to any group of nodes).
- **Weirdnet is the only remote path.** Inference, control (load, unload, big mode, config) and
  health all ride Weirdnet streams, authorized by the device roster. No Caddy, no HTTPS endpoint,
  no API keys.
- **Seamless failover:** if a node dies mid-answer, the reply continues on another node with no
  duplicate text.
- **Local OpenAI clients work** (Jan, scripts) through a bridge on `127.0.0.1` on the same machine.
- **Everything else optional:** Redis, Tailscale and a second node are never required.
- **Any Weirdnet agent app can use the cloud:** pool discovery and remote inference live in
  `weirdnet-agent`, not in Ymir or Bragi.
- **Thin apps.** Generic parts go in `weirdnet-*` crates on crates.io (BSD-3-Clause).

## Non-goals

- Training or fine-tuning inside Ymir: Ymir feeds, schedules and serves a separate trainer (decided by m3talsmith 2026-11-22; see [docs/design/training.md](design/training.md)).
- Other users: no CI tenants, teammates or shared pools; no invites, membership roles or quotas.
- Any HTTPS or API-key endpoint, LAN or public. No [WeirdBrains](https://weirdbrains.com)-run gateway (forbidden by AGENTS.md).
- Writing our own inference engine: we drive llama.cpp and vLLM, and in-process llama.cpp through
  `weirdnet-model`.
- llama-swap: replaced by Ymir's own process manager.
- Carrying NCCL/Ray tensor traffic over Weirdnet. Weirdnet carries control, context and tokens;
  tensor-parallel traffic stays on the fast private link (RoCE, NVLink, 10/25/100 GbE).
- Syncing KV caches between nodes (they stay local; see [Context sync](#context-sync)).
- Replacing Tailscale for SSH and general admin. Tailscale stays if you use it; Ymir doesn't need it.

## Minimum hardware

| Tier | Hardware | What you get |
|---|---|---|
| Minimum (CPU) | x86_64 with AVX2, or ARM64; 8 GB RAM; 20 GB free disk | llama.cpp on CPU (`cpu-dispatch`), Gemma 4 E2B at Q4, a few tokens/s. Fine as a pool member for small jobs or embeddings. |
| Entry GPU | NVIDIA with 8 GB VRAM (Turing sm_75 or newer), AMD RDNA2+ with 8 GB (ROCm or Vulkan), or Apple Silicon with 16 GB | llama.cpp GPU offload; Gemma 4 E4B plus a 7-9B Q4 model. |
| Comfortable | 16-24 GB VRAM, or a 32-64 GB Mac | A 27-35B Q4 model (dense, or MoE like `qwen3.6-35b-a3b`) plus a small stand-in and an embedding model. |
| vLLM | Linux, NVIDIA sm_80+ (Ampere or newer, including GB10 sm_121 with CUDA 13 builds) with 24 GB+, or AMD Instinct / RDNA3 on the ROCm list | vLLM as an extra backend: higher concurrency and NVFP4/FP8 checkpoints. |
| DGX Spark | GB10, sm_121, 128 GB unified memory | The two-server setup on one box: `qwen3.6-35b-a3b` on llama.cpp, a 27B NVFP4 on vLLM, Gemma 4 E4B stand-in, embeddings. |

Disk: models are big. The setup shows the download size and asks before fetching (the
`weirdnet-model` `Consent` rule).

## Single-box quickstart

```sh
# 1. Install (user-level, no root): a prebuilt binary into ~/.local/bin, no Rust needed.
#    Also available as a TAUR package, .deb, .rpm, Nix flake, Homebrew and a Windows installer.
curl -fsSL https://github.com/WeirdBrains/ymir/releases/latest/download/install.sh | sh

# 2. Set up: detects the hardware, picks a backend and models that fit, shows sizes,
#    asks before downloading, writes ~/.config/ymir/node.toml with the model, inference and
#    app roles all on, and starts one ymird user service that runs all three.
ymirctl setup

# 3. Use it on this machine. Point Jan (or anything OpenAI-compatible) at the loopback bridge:
#    Base URL http://127.0.0.1:8484/v1   API key: the value `ymirctl bridge token` prints
ymirctl status
ymirctl chat "Hello"
```

`ymirctl setup --dry-run` prints the hardware report, the chosen backends and models and the
download plan, and changes nothing. `--yes` accepts the defaults for unattended installs, and
`--system` installs a system service instead of a user service (for headless servers like the Sparks).
Adding Redis, alerts or a second node are later, separate `ymirctl` commands.

## Roles

Ymir has three roles. Each is an independent toggle in the per-node config, and they can be placed
freely: all three on one device, or spread across devices in any combination, with any number of
each.

| Role | What it does | What it never does |
|---|---|---|
| **Model node** | Runs models and takes requests, like a vLLM or llama.cpp server (it can literally be stock `vllm serve` or `llama-server`). A thin Weirdnet agent wrapper advertises its hardware and its loaded and available models, accepts `Load`/`Unload`, and serves inference. Its local supervisor starts and stops engine processes when told. | Routing, load balancing, placement decisions, conversation state, caching, talking to apps. |
| **Inference node** | The coordinator. Takes prompts from apps, keeps the pool's model catalog, picks model nodes (load balancing), decides placement and cloud-wide swapping, runs the semantic cache and cold-start stand-ins, resumes mid-answer failover, owns context sync through `weirdnet-sync`, and aggregates health and alerts. | Running models itself (unless the same device is also a model node). |
| **App node** | Bragi (desktop and phone), the loopback OpenAI bridge, or any `weirdnet-agent` app. Queries the catalog and sends prompts, to inference nodes only. | Talking to model nodes directly. |

**Same protocol, local or remote.** Roles always talk through the same protocols. When two roles
share a device they use an in-process fast path (a direct call through the same trait, with no
serialization) or, across processes, the same frames over a local socket. Behaviour, auth and
failure handling don't change with placement.

### Protocols between roles

| Pair | Protocol | Crate (owner) | Messages |
|---|---|---|---|
| App → Inference | `/weirdnet-agent/infer/2` (new version; `/1` still served) and `/weirdnet-agent/catalog/1` | `weirdnet-agent` `pool` (app side, decided 2026-11-19) and `serve` host traits (inference side); Weirdnet lead | catalog query and subscribe (models, state, which are warm), infer with streaming tokens, cancel, `Queued`, embeddings, jobs, `context_ref` |
| Inference → Model | `/weirdnet-agent/infer/2` (raw mode: fully assembled messages, `assistant_prefix`) and `/weirdnet-backends/node/1` | `weirdnet-agent` (inference), `weirdnet-backends` `node` (caps and control); Weirdnet lead | signed capability records (hardware, models, state, queue, tokens/s, warm prefixes), `Load`, `Unload`, `Drain`, `GroupUp`/`GroupDown`, health |
| Inference ↔ Inference | `/weirdnet/sync/2` (the `weirdnet-sync` pull protocol), `/ymir/coord/1` | `weirdnet-sync` (context, pool policy, leases); Ymir (leases and placement plans) | synced context and policy; scheduler lease; placement plans |
| Any → any (admin) | `/ymir/control/1` | Ymir | `ymirctl` commands: status, policy edits, drain, group up/down |

The model-node wrapper is generic (`weirdnet-backends::node` plus `weirdnet-agent` `InferHost`), so
any Weirdnet app could run a model node; the coordinator behaviour is Ymir's, built on generic
pieces.

### Binaries and role flags

One daemon, **`ymird`**, with roles switched on in `node.toml` (`[roles] model = true`,
`inference = true`, `app = true`) or on the command line (`ymird --roles model,inference`). Each
enabled role runs as a component in the same process (see [Weirdnet nodes per process](#weirdnet-nodes-per-process)
for how many `weirdnet-net` nodes that takes today). The app role inside `ymird` is the loopback
OpenAI bridge. `ymirctl` is the CLI for any of them. Bragi is a
separate app and is always an app node. The single-box quickstart enables all three roles in one
`ymird`.

### Deployment topologies

1. **One box, all roles.** A DGX Spark, gaming PC or Mac runs `ymird` with model, inference and app
   (bridge) on. Jan on the same machine uses `127.0.0.1:8484`; Bragi on a phone talks to the box over
   Weirdnet.
2. **Model + inference on a server, apps elsewhere.** A GPU box runs model and inference; laptops run
   Bragi or `ymirctl bridge` (app only); phones run Bragi. Nothing but Weirdnet crosses the network.
3. **Several model nodes, one inference node.** The server-a/server-b layout below: server-a is model + inference, server-b
   is model only, the dev box is an app node. Adding a third GPU box means linking it with
   `model = true`.
4. **Redundant coordinators.** Two or more inference nodes (say on server-a and server-b, or on a small always-on
   box with no GPU) in front of any number of model nodes. Apps fail over between inference nodes;
   if one dies, the others carry on with the same synced state.

## Architecture

```
  App nodes:  Bragi (GPUI desktop, Flutter phone) · loopback OpenAI bridge (Jan, scripts) · any weirdnet-agent app
                 └──── weirdnet-agent pool: find inference nodes, catalog, infer, cancel, resume ────┐
                                                                                                      │ /weirdnet-agent/infer/2
                                                                                                      │ /weirdnet-agent/catalog/1
  Inference nodes (one or more, active-active):                                                       ▼
   ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
   │ ymir-core coordinator: catalog, routing and load balancing, scheduler lease, cloud-wide swap │
   │   and placement, cold-start stand-ins, mid-answer resume, fairness, group jobs               │
   │ weirdnet-agent context + weirdnet-sync (context, pool policy) · weirdnet-semcache · alerts   │
   └───────────────┬──────────────────────────────────────────────────────────────┬──────────────┘
                   │ /weirdnet-agent/infer/2 (raw) · /weirdnet-backends/node/1     │
  Model nodes:     ▼                                                               ▼
   ┌──────────────────────────────────┐                             ┌──────────────────────────────┐
   │ weirdnet-backends node wrapper:  │                             │ ...same...                   │
   │  caps, Load/Unload, supervisor   │                             │                              │
   │ llama-server · vLLM · Ray · RPC  │◄═══ private link (NCCL/RPC)══►                             │
   │ weirdnet-model: hw, sizing       │                             │                              │
   └──────────────────────────────────┘                             └──────────────────────────────┘
```

Every device runs `ymird` as a systemd user or system unit, a launchd agent, or a Windows service
running as `NT SERVICE\ymird` (AGENTS.md). It bootstraps from the public [WeirdBrains](https://weirdbrains.com) relay by
default, and the operator can override that.

### Profiles

The Ymir and Bragi `Profile`s are copies of `weirdnet_net::profile::WEIRDNET` that change **only**
`join_link`, `answer_link` and `link_mark` (to `weirdnet-ymir://` and `weirdnet-bragi://` links).
Every other identifier (protocol, names, certificate and offer labels, port) stays as in `WEIRDNET`.
That is required, not a style choice: `Roster::check_caller` verifies a caller's certificate with
`profile.verify_cert` (weirdnet-sync 0.5, `src/roster.rs`), so a device whose certs were issued under a
different profile would not be admitted, and the apps would stop sharing one network (AGENTS.md:
apps share one Weirdnet network and use their own protocol names).

### Weirdnet nodes per process

The goal is one `weirdnet-net` `Node` per `ymird`. That isn't possible with released crates:
`weirdnet_sync::start_node` starts its own `Node` on `SYNC_PROFILE` (weirdnet-sync 0.5, `src/net.rs`),
so context sync can't yet ride an app's node.

**Decided 2026-11-19 (option A):** for now a `ymird` process (and Bragi) may run **two or more**
`weirdnet-net` nodes under the same device identity: the app node, the sync node, and more if a
component needs its own. It works with what is published, costs extra connections, and doesn't block
M5. A model-only `ymird`, which doesn't sync, normally has one.

**Later (option B):** when `weirdnet-sync` can run its handler on an app's existing `Node` (accepting
`/weirdnet/sync/2` there; Weirdnet lead work), Ymir and Bragi move to one shared node per process,
with no protocol change.

### Model node

- Probes the hardware (`weirdnet-model` `hw`) and publishes a signed capability record to inference
  nodes on `/weirdnet-backends/node/1`.
- Its supervisor (`weirdnet-backends`) starts, health-checks, stops and restarts engine processes
  (llama-server, vLLM, Ray worker, `rpc-server`) when told. Within the limits it is given (swap
  groups, idle TTL), it can swap locally, but **what** to load is the inference node's call.
- Serves `remote::infer` in raw mode through `InferHost`: fully assembled messages in, tokens out,
  cancel aborts the engine request. Its `Admission` hook answers `Busy` when engine slots and the
  queue are full.
- Holds no conversation state; its KV/prefix cache stays local and is reported as prefix hashes.
- Can wrap a stock server: point it at an existing `vllm serve` or `llama-server` URL on loopback
  (`managed = false`) and it only advertises and forwards.

### Inference node

- Builds the **pool catalog** from model nodes' capability records and answers
  `/weirdnet-agent/catalog/1`. Each entry carries the model's origin (`hub`, `upload`, `train`) and license
  when the first node that knows says them (U8; `ModelEntry::origin`/`license`, weirdnet-backends 0.4.5).
- For each prompt: checks the semantic cache, assembles the prompt from the synced context, picks a
  model node by load and capability, streams tokens back to the app, writes the answer into the
  context, and resumes on another model node if one dies.
- Decides **placement and cloud-wide swapping** while it holds the scheduler lease.
- Owns **context sync** with the other inference nodes and app nodes.
- Aggregates health from model nodes and sends alerts.

**Redundancy: active-active, with one lease.** Any number of inference nodes run at once and every
one serves app requests. They share state through `weirdnet-sync`: context, pool policy and recent
placement plans. Each builds its own catalog view from the capability records it receives (model
nodes push to all inference nodes). Load balancing stays correct with several coordinators because
model nodes report their own queue depth across all callers, and each coordinator also counts its
own in-flight requests; power-of-two choices absorbs the staleness. Only **placement** needs a
single writer: a scheduler lease held by one inference node (lowest peer id among those online,
renewed every 15 s through `/ymir/coord/1`, taken over when it lapses). If no lease holder is
reachable, model nodes keep what they have loaded and every coordinator keeps routing.

### App node

- Uses the `weirdnet-agent` pool to find inference nodes among m3talsmith's linked devices (local one
  first, then the lowest-latency one), query the catalog and send prompts.
- If an inference node fails, the pool moves to another one; mid-answer it resumes there from the
  text it has already shown (see [Mid-answer failover](#mid-answer-failover-decided-2026-11-19-automatic)).
- Never discovers or calls model nodes.

### Inference protocol details

`remote::infer` (streaming tokens, cancel, `Admission`/`Busy`, `RemoteModel` fallback) gets a new
protocol version, **`/weirdnet-agent/infer/2`** (Weirdnet lead). It can't be an in-place extension
of `/1`: a 0.2 server would ignore `assistant_prefix` and generate the prefix again (duplicate text),
and a 0.2 client fails on an unknown `Queued` frame. Servers serve `/1` and `/2` side by side; a
client resumes only when the server's `Accepted` says `resume: true`.

**Approved for M3 (2026-11-19):** `infer/2` is CBOR throughout (requests and every frame), while
`infer/1` stays JSON as it is. `weirdnet-agent`'s core is transport-neutral: a `Caller` (the
authenticated peer and its cert, or "local" for the in-process path) plus an `authorize` hook the
host implements (Ymir's runs `Roster::check_caller`), so the same server code runs over Weirdnet and
in process. Ymir moves from `ymir-infer`'s request types to agent's in M3.

What `/2` adds:

- request fields: `priority` (`interactive`, `batch`, `background`), a stable `job_id` plus an
  `attempt` number, a `context_ref` (conversation id and head, see [Context sync](#context-sync)) or
  inline messages, an optional `assistant_prefix` with its length for resume, sampling parameters and
  seed, `response_format`, tools;
- the caller's identity: `serve()` and `InferHost::open` receive the transport-authenticated peer
  and its presented cert (today `open` gets only the model name), so hosts can run
  `Roster::check_caller`, per-device fairness and role checks. Nothing about the caller is taken from
  the body;
- `Accepted { id, model, resume }`: `resume: true` only from servers that honour `assistant_prefix`;
- frames: `Delta { offset, text }` (offset = characters of the answer before this delta), new
  `Reasoning { offset, text }` and `ToolCall { index, complete call }` frames, `Usage`, `Done`;
- `Queued { position }` before the first token, from a real priority queue on the server; each
  `Queued` frame resets the client's accept timeout (10 s, `DEFAULT_ACCEPT_TIMEOUT`), so a queued
  request isn't mistaken for a dead node;
- a raw mode for inference → model calls (assembled messages only, no context lookups);
- an `embeddings` call: vectors as CBOR byte strings of little-endian `f32`, streamed one item per
  frame;
- response metadata: served-by model node, model revision, cache hit or stand-in flag, resume
  offsets, token counts and timings.

The 8 MiB frame cap stays as a guard; anything bigger (attachments, documents for batch jobs, large
embedding inputs) travels as `weirdnet-net` `content` roots fetched by whoever needs them.

`RemoteModel` keeps its current default (give up once tokens have streamed). Resume is opt-in: the
`pool` turns it on, other apps can.

**Streaming tokens:** each call is one stream, app → inference and inference → model. Tokens flow as
frames with backpressure (the `weirdnet-net` stream window); the inference node forwards frames as
they arrive. Dropping the app's call resets its stream, the inference node drops its call to the
model node, and the engine request is aborted, freeing the slot. Streams survive a relay-to-direct
path move without losing bytes (`weirdnet-net` 0.8).

### Local OpenAI-compatible bridge (`weirdnet-openai`), loopback only

An app node for local tools that only speak the OpenAI API (Jan, editors, scripts). It is
**loopback-only: it binds `127.0.0.1:8484` and has no network listener of any kind.** There is no
option to bind it to a LAN, tailnet or public address. Remote use always goes over Weirdnet, from
that machine's own bridge or Bragi.

- Endpoints: `/v1/models`, `/v1/chat/completions` with SSE, `/v1/completions`, `/v1/embeddings`,
  translated to `weirdnet-agent` pool calls to inference nodes.
- It runs inside `ymird` (app role) or alone (`ymirctl bridge`) on a laptop that runs no other role.
- A local token, generated at setup, kept in the OS keyring (not a file) and shown by `ymirctl bridge
  token`, is required on **every** endpoint, `/v1/models` and `/metrics` included. It guards against
  other local OS users and local web pages. Calls leave the machine as this device's identity.
- DNS-rebinding guard: requests whose `Host` header isn't `127.0.0.1:<port>` or `localhost:<port>`
  are refused. No CORS headers are sent, so browsers can't read responses cross-origin.
- `/v1/models` lists the **pool's** models (the merged catalog).
- Stateless OpenAI requests are mapped onto synced contexts by message-prefix hash, so failover works
  for them too (ephemeral retention, see Context sync).
- Error mapping: `Busy` → 503 with `Retry-After`; stand-in answers carry `x-weirdnet-standin: <model>`;
  cache hits `x-weirdnet-cache: hit`; resumed answers `x-weirdnet-resumed: <node>` (generic
  `x-weirdnet-*` headers, since the bridge is a Weirdnet crate).

### Health, metrics and alerts

- Model nodes report backend states (cold, loading, ready, failed), GPU memory, utilization and
  temperature (NVML, amdsmi, IOKit), unified-memory pressure, disk, queue depths, tokens/s and time to
  first token. Inference nodes add cache hit rate, context-sync lag, failovers, and peers and paths
  (direct or relayed). `ymirctl status` shows the whole pool.
- Metrics are also available as Prometheus text on the loopback bridge (`/metrics`, token required), optional.
- `weirdnet-alerts` (new) runs on inference nodes (the lease holder sends, to avoid duplicates). It is
  sinks plus a health rule pack built on `weirdnet-rules` 0.1.0 (conditions over an event history,
  loop protection, schedules), not a rule engine of its own. The rules add dedupe ("at most once an
  hour", plus a recovery notice, as the Wi-Fi fallback alert does). Sinks: SMTP (the password lives
  in the OS keyring, never in config), webhooks, and push (below).

- Built-in rules carried from server-a/server-b: backend crash loop, model load failure, node lost, **network
  degraded** (the primary interface is down and traffic is on a fallback route, i.e. the Wi-Fi
  fallback alert, generalized), relayed-only for too long, disk low, GPU hot, failover happened.
- Peer watching replaces a separate monitor host (the dev box's monitor timer).

### Push (decided 2026-11-19: `weirdnet-push`, owned by the inference node)

The inference node owns push. It runs the push sender and holds the APNs and FCM credentials. To
alert a phone (or say a long job finished), it sends `weirdnet-push`'s 96-byte sealed wake payload
with a reason code; Apple and Google see only ciphertext. The phone wakes, and Bragi fetches the
alert or result itself over Weirdnet from an inference node. With several inference nodes, the
lease holder sends, so a phone isn't woken twice.

What `weirdnet-push` 0.1.0 already has (checked in its source): the wake payload, handles and the
device-side `ReplayGuard`; a client; and a **server component** (feature `server`, the
`weirdnet-push` binary) with an **APNs sender** (token auth with the `.p8` key, team id, alert push
with `mutable-content`) and an **FCM HTTP v1 sender** (service-account JSON), plus the relay wake path
for F-Droid builds. The senders sit behind a library `Service` (`register`, `rotate`, `unregister`,
`wake`), so the inference node can embed it in-process.

What is missing: the server's only front end is an HTTP router (`server::serve` binds `listen` and
expects TLS in front), and Ymir has no HTTP or HTTPS endpoints. So **Weirdnet-lead work:** a
Weirdnet front end for `Service` (device registration, rotation, unregistration and wake requests over
a `weirdnet-net` protocol, authorized by the roster) so the inference node embeds the senders with no
network listener. Until then, push to phones is off (alerts go by SMTP and to apps that are open).

What m3talsmith has to provide: an **Apple developer account key** (an APNs `.p8` auth key and team id)
for iOS, and a **Firebase project** (a service-account JSON) for FCM on Play builds. F-Droid builds use
the relay wake path and need neither. These credentials live only on inference nodes, in the keyring
or 0600 files referenced by name from config, never in the repo. A model node with no inference
  node in reach can still send its own SMTP alert if configured.

### Cold-start fallback

The server-a coordinator answered with `gemma4-e4b` while the 27B loaded. On the inference node:

1. Route to a model node where the model is **warm** if one exists.
2. Otherwise load it on the best model node for it (placement), and, if the pool policy names a
   `standin` and the load estimate is over `standin_after` (default 10 s), answer with the stand-in,
   flagged. A client can opt out per request (`x-wait: 1` as today, or `wait_for_model`).
3. Load time estimates come from measured history per model node and model (85 s for the 27B on server-b).

## Single node (primary target)

On one box all three roles run in one `ymird` and talk through the in-process fast path. The rest of
this section is mostly the model role: what hardware it finds, which engines it runs and which
models it picks.

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
devices, link speeds; used for splitting) and what each backend can do here.

### Backends

| Backend | Where | Role |
|---|---|---|
| llama.cpp `llama-server` (managed by Ymir) | Everywhere: CUDA, ROCm/HIP, Metal, Vulkan, CPU; Linux, macOS, Windows, BSD | Default backend. GGUF models. |
| MLX-LM `mlx_lm.server` (managed venv, `weirdnet-backends` `EngineKind::Mlx`) | macOS on Apple Silicon | Default for chat models there. MLX weights. |
| In-process llama.cpp (`weirdnet-model` `llama`) | Everywhere | Tiny models and embeddings without an extra process; Bragi's optional offline model. |
| vLLM (managed by Ymir, Rust frontend as on server-a/server-b) | Linux, NVIDIA sm_80+ (sm_121 needs the CUDA 13 build), AMD ROCm on vLLM's supported list | Optional. High concurrency, NVFP4/FP8. Never on macOS, Windows or BSD. |
| vLLM + Ray | Linux nodes with a fast private link | Split models (see big mode). |
| llama.cpp RPC (`rpc-server`) | Everywhere llama.cpp builds | Split GGUF models across mixed nodes. |

There is no llama-swap backend. `weirdnet-backends` holds the per-engine adapters (launch arguments,
readiness and health probes, request abort, slot save and restore, and pinned checksummed engine
builds per platform, fetched with `weirdnet-model`'s download and `Consent` rather than a second
downloader; vLLM in its own uv venv as `/opt/llm/vllm` today). Hardware detection in `weirdnet-model`
`hw` doesn't depend on `llama-cpp-2`, so it works without the `llama` feature. Ymir's process
manager decides what runs when.

### The bundled engine (decided 2026-11-20)

m3talsmith asked for a binary install with minimal setup on the server: no separate llama.cpp
install. Every Ymir release file now carries a pinned llama.cpp build next to `ymird`, and `ymirctl
setup` uses it by default (`llamacpp = "managed"`). A server install is: download or install the
package, `ymirctl setup --download --yes --service`.

Three ways were weighed:

| Option | For | Against |
|---|---|---|
| (a) llama.cpp in-process (`llama-cpp-2`, GPU backends as cargo features) | One binary, no IPC | A model crash (CUDA error, `GGML_ASSERT`, out-of-memory abort) kills `ymird` and every other model and the coordinator with it; a server would have to be rewritten on top of the bindings (OpenAI API, chat templates, slots, embeddings, tool calls, RPC); one build per GPU backend for `ymird` itself; CUDA builds of the whole daemon in CI |
| **(b) bundle upstream's `llama-server` in the release files** (chosen) | Crash isolation for free (the engine is a child process; the supervisor restarts it and the coordinator fails over, as before); the supervisor, adapters and tests stay the same; llama.cpp's own server, which is what the pool already runs; upstream publishes checksummed builds for every target we ship (Linux x86_64/ARM64 CPU+Vulkan and CUDA 13, macOS Metal, Windows), so CI never compiles CUDA | Larger downloads, the CUDA variant especially; two variants per Linux target |
| (c) download a pinned llama.cpp during `ymirctl setup` | Small packages | Setup needs the internet and a second trust path; an air-gapped or locked-down server can't install; the package isn't self-contained |

Why (b): crash isolation and failover are requirements, and (b) keeps the engine a separate process
without anything new to install. In-process llama.cpp stays where it fits: tiny models and
embeddings in `weirdnet-model` (`llama`), and Bragi's offline model.

How it works:

- `packaging/engines/llama.cpp.toml` pins one upstream build (`b12628`, first Inkling-capable pin; was `b12590`) and, per target and
  variant, the upstream release archives with their SHA-256. `scripts/engine.py fetch` downloads and
  checks them and keeps only `llama-server`, `rpc-server`, their libraries and GPU backends, the
  licenses and an `engine.toml` (build, variant, backends). `scripts/engine.py build` compiles the
  same set from source for targets upstream doesn't publish (the BSDs, distributions that build from
  source). `scripts/engine.py pin bNNNN` moves the pin.
- Variants: `ymir` = llama.cpp with CPU (one backend per CPU generation, picked at run time) and
  Vulkan (any GPU), Metal on Apple Silicon; `ymir-cuda` = CPU and CUDA 13 with the CUDA runtime
  libraries (NVIDIA redistributables), for Linux x86_64 and ARM64 (and Windows x86_64). CUDA 13's
  default architectures cover sm_75 through sm_120 and sm_121a (GB10), so one build serves
  everything from a Turing desktop card to a DGX Spark; it needs driver 580 or newer. ROCm can be added as a third variant from
  upstream's ROCm builds; until then AMD runs through Vulkan.
- Layout: `lib/ymir/llama.cpp/` beside `bin/` (tar.gz, deb, rpm, Arch, Homebrew, Nix), or
  `llama.cpp\` next to `ymird.exe` (Windows zip, update bundles). `ymir_core::engine` finds it
  relative to the running executable; `$YMIR_ENGINE_DIR` overrides. `managed` resolves at every start
  in this order: `<home>/engines/llama.cpp/`, the bundled engine, `PATH`. So a package upgrade
  upgrades the engine, and `--llama-server PATH` (`system:PATH`) still picks an external build.
- `ymirctl setup` names the engine it found and warns when the machine has an NVIDIA GPU but the
  installed build has no CUDA.
- vLLM stays an external engine (`vllm = "auto"` or `system:PATH`, `image:registry/name:tag`, or a wrapped `vllm serve`).
- Tests: `ymir-testkit/tests/bundled_engine.rs` (setup and `ymird` pick the bundled engine, with the
  mock engine laid out as a package installs it) and `real_bundle.rs` (an unpacked release archive,
  its own `ymirctl` and `ymird` processes, a real tiny GGUF streamed through the bridge, and a killed
  engine that `ymird` survives and restarts). The Release workflow runs `real_bundle` on every
  default Linux and macOS archive.

Generic part, proposed to the Weirdnet lead for `weirdnet-backends` (the design already gives it
"pinned checksummed engine builds per platform"): locating an engine shipped with an app plus its
`engine.toml`, the pin manifest format and fetch with `weirdnet-model`'s download and consent, and the
engine build script, so Morph, Bragi and other apps reuse them. Until then the code is in
`ymir_core::engine` and `scripts/engine.py`, marked to be extracted.


### Inkling 2 Mini (2026-11-22)

Thinking Machines' Inkling 2 Mini (`thinkingmachines/Inkling-2-Mini`, Apache 2.0) is a looped
multimodal MoE (`model_type: inkling2_loop`, 26.4B / 3.4B active, NVFP4+MXFP8, 1M context, text and
image). It is small enough for a DGX Spark. Ymir serves it through vLLM or llama.cpp; MLX is not
ready yet.

- **vLLM (preferred on Linux NVIDIA, including Spark).** `ymirctl models pull thinkingmachines/Inkling-2-Mini`
  records the Hub id (vLLM fetches weights into the Hugging Face cache on first load) and writes a
  `[[model.local]]` with `backend = "vllm"`, the Inkling reasoning and tool-call parsers, DSpark
  speculative decoding, and `image = "vllm/vllm-openai:inkling-2-mini"`, and saves the repo's
  `config.json` so ymird can size `--max-model-len` (see "Context length" below; the vLLM recipe's
  262144 cap for a Spark is no longer written: only 8 of its 48 unrolled attention layers are global,
  so the full 1M fits on 128 GB). Support needs vLLM
  0.35.0 or a newer nightly, or that image; it is not in 0.34.0 and is not in a vLLM release yet
  ([recipe](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini),
  [vllm#69028](https://github.com/vllm-project/vllm/pull/69028)). If the node's `vllm` is too old,
  ymird leaves the model unloaded and logs how to point `[model] vllm` at `image:vllm/vllm-openai:inkling-2-mini`
  or wrap a container the operator started:

      docker run --gpus all --ipc=host -p 8000:8000 \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        vllm/vllm-openai:inkling-2-mini thinkingmachines/Inkling-2-Mini \
        --reasoning-parser inkling --tool-call-parser inkling --enable-auto-tool-choice \
        --max-model-len 262144

      # node.toml
      wrap = "http://127.0.0.1:8000/v1"
      context = 262144   # a wrapped server's context is whatever it was started with

  `image:` writes a `vllm-image` shim under the engines dir (`docker` or `podman`, host network,
  `--entrypoint vllm`) so the supervisor can still run `vllm serve MODEL --host 127.0.0.1 --port N`.
  `ymirctl models search` lists safetensors repos on vLLM-capable nodes (`--format vllm` or
  `--format all`). A safetensors pull from another machine over `--node` is not in the models/1
  protocol (that path is GGUF); run `ymirctl models pull` on the node itself.

- **llama.cpp.** The bundled engine is llama.cpp **b12628** (GGUF architecture `inkling2_loop` from
  [llama.cpp#32694](https://github.com/ggml-org/llama.cpp/pull/32694) in b12600, plus the NVFP4
  loop-scale fix in [llama.cpp#32704](https://github.com/ggml-org/llama.cpp/pull/32704)).
  `ymirctl models pull ggml-org/Inkling-2-Mini-GGUF` picks a fitting quant and also downloads a
  `mmproj-*.gguf` from the same repo (preferring Q8_0, e.g. `mmproj-Inkling-2-Mini-Q8_0.gguf`) and
  sets `mmproj` on the stanza so images work. That auto-fetch is generic: any GGUF repo with an
  `mmproj-*.gguf` gets the same treatment. Use ggml-org or bartowski GGUFs on this pin; Unsloth
  `Mainline_Support` files need the #32704 fix (included in b12628), and Unsloth's root files need
  Unsloth's own llama.cpp, which Ymir does not ship.

- **MLX (not yet).** mlx-lm support is still an open draft
  ([mlx-lm#2086](https://github.com/ml-explore/mlx-lm/pull/2086)). `ymirctl models pull` of an
  Inkling MLX repo fails with that pointer. Use vLLM or a GGUF until the PR is in a released
  mlx-lm; even then Ymir's MLX engine is text-only today (see above), so images would still need
  more work (`mlx-vlm` is not adopted).

### Engines per platform: MLX on Macs, llama.cpp everywhere else (2026-11-20)

| Platform | Chat models | Vision models, embeddings | Opt-in |
|---|---|---|---|
| macOS, Apple Silicon | **MLX-LM** (`backend = "mlx"`), MLX weights from `mlx-community` | llama.cpp (bundled, Metal) | `--backend llamacpp` |
| macOS, Intel | llama.cpp (bundled, Metal/CPU) | llama.cpp | — |
| Linux, Windows | llama.cpp (bundled; `ymir-cuda` for NVIDIA) | llama.cpp | vLLM (Linux, NVIDIA) |

**MLX (default on Apple Silicon).** m3talsmith: "run MLX/MLX-LM on Macs instead of llama.cpp".
`ymirctl setup` on an arm64 Mac proposes the same catalog models, served by `mlx_lm.server` from
MLX-format weights (`ymir_core::mlx::WEIGHTS`: each catalog chat model mapped to an
`mlx-community` 4-bit repo, MXFP4 for gpt-oss, pinned to a commit). `setup --download` installs
`mlx-lm` into `<home>/engines/mlx` with `weirdnet_backends::mlx::install` (the version
`weirdnet-backends` pins, 0.32.2 in 0.1.3; `uv` when present, else `python3 -m venv`), and
downloads the weights to `<models_dir>/mlx/<name>`, asking first for each, so loads need no
network. MLX-LM's server is text-only and has no embeddings endpoint
(`EngineKind::Mlx.capabilities()` is chat only), so models that take images (Gemma 4, with their
projector) and the embedding model stay on llama.cpp: setup decides from those capabilities,
config validation rejects `mlx` with `role = "embedding"`, and the supervisor answers 400 to image
or audio parts sent to an MLX model. (`mlx-vlm` would serve images on MLX; not
adopted yet.) `mlx = "managed" | "system:PATH" | "off"` in `[model]`.

**How MLX runs: `EngineKind::Mlx`.** Since `weirdnet-backends` 0.1.3 MLX-LM is an engine kind
of its own. When a node has an `mlx` model, `NodeConfig::engines` adds `EngineConfig::mlx` with the
resolved `mlx_lm.server`, and the model launches as `EngineKind::Mlx` with its weights directory as
the source. The supervisor starts `mlx_lm.server --model <dir> --max-tokens <context>` on a
private loopback port behind its own proxy: readiness only after the model has produced a token
(MLX-LM answers `/health` before its weights are in), the API key checked as the bearer token,
the request's `model` rewritten to `default_model`, media parts and embeddings refused, responses
streamed through. The engine is still its own process tree, so a crash stays out of `ymird`, and
restart and failover work as before. Each model's capabilities (chat, embeddings, images, audio)
come from the supervisor (`ModelStatus::capabilities`: a llama.cpp model with a projector takes
images and audio, an MLX model text only); `ymird` uses them to tell embedding models from chat
ones, falling back to the older `embedding` flag for nodes that don't report them, and `ymirctl
status` shows what each model takes. (Until 0.1.3 Ymir shipped its own `ymir-engine` wrapper for
this; it is gone.) Tests: `ymir-testkit/tests/mlx.rs` (a llama.cpp and an MLX model on one node,
mock engines, capabilities) and unit tests in `mlx` and `setup`; the macOS workflow runs a setup
dry run on Apple Silicon (expects MLX) and Intel (expects llama.cpp), and an Apple Silicon job
that installs mlx-lm, downloads SmolLM2 135M's MLX weights and chats with it (reported but not
required, as hosted runners may not expose Metal).

**mistral.rs was tried and dropped.** Measured on a GPU node against llama.cpp with the same GGUF and
projector (8k context, one sequence, ~1,450-token prompt, 256 tokens out): Gemma 4 E4B decoded at
40.7 tok/s on mistral.rs vs 64.3 on llama.cpp, Qwen3.6-35B-A3B at 70.0 vs 66.9; prefill and time
to first token were about even; loads took 11–36 s vs 3 s, and mistral.rs fetched the base model's
config and tokenizer from Hugging Face at load. llama.cpp with the projector files matched it on
images (see Vision), so m3talsmith chose llama.cpp alone: "keep llama.cpp and include the
projector files".

### Vision (projectors)

llama.cpp reads images through a separate projector GGUF (`--mmproj`). `ymir_core::vision` pins
one per vision-capable catalog model, from the same upstream repo as the weights, with size and
SHA-256: `gemma-4-e2b` → `ggml-org/gemma-4-E2B-it-GGUF` `mmproj-gemma-4-E2B-it-Q8_0.gguf`
(557 MB), `gemma-4-e4b` → `ggml-org/gemma-4-E4B-it-GGUF` `mmproj-gemma-4-E4B-it-Q8_0.gguf`
(560 MB). The other catalog models are text-only. `ymirctl setup --download` fetches the projector
with its model (checksummed, with consent, through `weirdnet-model`), and `ymird` passes
`--mmproj` whenever it is on disk; `mmproj = "path"` or `"off"` in `[[model.local]]` overrides, as
does an explicit `--mmproj` in `args`. The model budget does not count the projector (about
0.5 GB) yet.

Checked on a GPU node against mistral.rs 0.9.5, same GGUF and projector, OpenAI chat requests straight to
each engine (2026-11-20; the pinned Gemma 4 E4B Q8_0 projector, checksum verified, answered the
same as the F16 one):

| Request | Gemma 4 E4B: llama.cpp | mistral.rs | Qwen3.6-35B-A3B: llama.cpp | mistral.rs |
|---|---|---|---|---|
| One image (64 px) | ✓ | ✓ | ✓ | ✓ |
| Two images in one message | ✓ | ✗ (400, "0 image placeholders but 2 image inputs") | ✓ | ✓ |
| Tiny image (8 px) | ✓ | ✓ | ✓ | ✗ (400, Qwen needs ≥ 32 px) |
| Large image (1920×1080) | ✓ 0.6 s | ✓ 0.3 s | ✓ 1.7 s | ✓ 1.3 s |
| Audio (`input_audio`, WAV) | ✓ | ✗ (only its own `audio_url` part) | ✗ (no audio in the projector) | ✗ |
| Remote `https://` image URL | ✗ (data URLs only) | ✗ | ✗ | ✗ |

So llama.cpp with projectors covers everything mistral.rs did here and more. mistral.rs's extra
content parts (`video_url`, `audio_url`, `file`) have no llama.cpp counterpart; nothing in Ymir
sends them.

**Images and audio through the pool.** With `weirdnet-agent` 0.5 a message's content is text or
parts (text, `image_url` with a data URL, `input_audio`), infer/2 carries them, `weirdnet-openai`
0.2 passes them through the bridge instead of refusing them, and `weirdnet-backends` 0.2 forwards
them to `llama-server` unchanged. Routing goes by capability: each model's `Capabilities` (from its
node's record) say what it reads, the catalog/1 entry lists them (`CatalogModel::with_inputs`, the
union over the nodes that have the model), and the coordinator sends a request with images or audio
only to a node whose copy of the model takes them. With no model named it uses the main model if
that takes them, else the best chat model in the pool that does (`Catalog::route`); a named model
that no node reads them with is a 400 ("does not take images"). A node whose record has no
capabilities (before `weirdnet-backends` 0.1.3) counts as text only. Answers to requests with media
are not cached. `tests/vision.rs` checks the bridge end to end (images reach the model with a
projector, one without is refused) and `tests/nodes.rs` the routing. Remote `https://` image URLs
are not fetched; data URLs only.

**Limits for media, reported, never defaulted (video milestone 1, LF-43, started 2026-11-26).** Clients
such as Bragi size video frames from what the model reports and refuse when it reports nothing
([docs/design/video.md](design/video.md)). Each catalog/1 entry now carries `context`: the context
window a request can count on wherever it is routed, the smallest `request_context` of the nodes that
have the model (the configured context, split over the parallel slots for llama.cpp). When any node
with the model sets no context the field is absent, because the request may be routed there; Ymir
never fills it in. `ymirctl status` shows `context N per request` or `context not reported`. Media limits
(`weirdnet-agent` 0.5.7 `MediaLimits`: images per request, per-image token cost or patch size, image
edge, audio length and tokens per minute, request bytes) come from the engine (`Capabilities::limits`,
`weirdnet-backends` 0.4.8) or from the operator per model in `node.toml`
(`limits = { max_images = 8, image_tokens = 256, image_edge = 896 }` on a `[model.local]`). catalog/1
carries the strictest value across the nodes that have the model, field by field (`MediaLimits::tightest`);
a field nobody reports stays absent. `[bridge] max_request_bytes` sets the OpenAI bridge's body limit
(`weirdnet-openai` 0.2.4: a 413 names it in `x-weirdnet-max-request-bytes`, and `/v1/models` lists it
with each model's `weirdnet.limits`); unset keeps the crate default. `ymirctl status` shows
`limits: ...` or `media limits not reported`.

### Model sizing

`weirdnet-model`'s catalog grows from Gemma 4 only to a tiered catalog with per-quant memory
footprints. Setup picks models whose weights plus a KV budget (context × concurrency) fit within
the usable budget (default 85% of VRAM, or of unified memory after the OS reserve), and proposes:

- one **main** chat model (largest that fits with headroom),
- one **stand-in** (Gemma 4 E2B/E4B),
- one **embedding** model (`qwen3-embedding-0.6b`) if the cache is on.

The operator can change any of them.

### Searching and pulling models (`ymirctl models`)

Beyond the catalog, `ymirctl models search` queries the Hugging Face Hub for GGUF (and MLX) repos
and gives a per-repo fit verdict for the node it runs on: file size, plus KV cache at the context
length (read from the GGUF header with a range request, or estimated when that fails), plus 10%
for compute buffers (`weirdnet_model::fit`, the same sum every node uses), against GPU memory or
the unified budget. `pull` takes the best quant that fits (highest quality, full precision only as
a last resort), handles split GGUFs, resumes, verifies SHA-256, and reads an optional Hub token from
the node's own environment or files. It prints the `[[model.local]]` entry; `--register` appends it
to `node.toml`, otherwise the file is never touched. `list` shows what is on disk and whether it is
registered; `remove` deletes files only. Without `--node` these run on the machine itself and need no
daemon.

#### Context length (2026-11-22)

A model runs with as much context as it can take. `pull` (and `setup`, for llama.cpp) writes no
`context`; at start ymird plans one per model (`ymir_core::kvfit::plan_launches`) and passes it to the
engine (`--ctx-size` for llama.cpp, `--max-model-len` for vLLM), so the catalog advertises exactly what
the server runs with:

- **Length.** The trained context: `<arch>.context_length` from the GGUF header, or `config.json`
  (the text config's `max_position_embeddings` or other length key, times the RoPE scaling factor
  the way vLLM derives it; yarn from `original_max_position_embeddings`). Unknown: 32768.
- **KV cache.** Per token: K and V heads times head size, summed over attention layers. Loop passes
  count once each (Inkling's `loop.*` / `loop_config`), sliding-window layers hold one window per
  server slot plus llama.cpp's 512-token micro-batch (llama-server runs 4 slots with a unified cache
  unless `slots` / `--parallel` says otherwise; pattern from `sliding_window_pattern`, `layer_types`
  or `local_layer_ids`, and a window without a pattern counts every layer as global), Gemma's
  `shared_kv_layers` hold none, linear or recurrent layers hold none, and a vLLM MTP head adds its
  layers. Element size from `-ctk`/`-ctv` or `--kv-cache-dtype`, else 16-bit.
- **Score buffers.** Attention with a per-head score bias (Inkling's relative positions, T5) cannot
  use flash attention in llama.cpp (`build_attn_mha`, b12628), so the compute buffer holds f32
  score matrices of context x micro-batch (`-ub`, 512) x heads. With a bias there are two (the
  gathered bias and the scores are both live at the add), plus the f32 mask and i32 relative
  positions: 97 GiB for Inkling at 1M tokens. Under plain `-fa off` there is one, plus the mask.
  These match llama.cpp's own reservation (`llama-fit-params -fitp on` on the ggml-org
  Inkling-2-Mini IQ4_NL header: 68,002 MiB of compute at 701,440 tokens, 101,563 MiB at 1M; Qwen3
  0.6B under `-fa off` at 32,768: 1,110 MiB against a 1,024 MiB matrix). v0.2.0 left the scores
  out; v0.3.0 counted one matrix, calibrated on an RSS read after load that did not hold the whole
  compute buffer, and planned Inkling at 701,440 tokens (70.9 GiB) beside a busy gemma4-e4b on node-a,
  where llama-server mapped 108 GiB setting up its context and the node ran out of memory. The plan
  now gives Inkling 437,248 tokens alone on a Spark (397,312 beside a busy gemma4-e4b at its full
  131,072, or with 93 GiB measured free as on node-a).
- **Fit.** Weights + KV + score buffers + 10% (as `weirdnet_model::fit`) against the node's model
  budget (`[model] memory_budget` / `reserve`), with 20% of the memory free for the model kept free.
  Models that can be resident together share it: each other swap group
  counts with its largest member's weights (and KV for a fixed context), and what is left is split
  evenly among the automatic models. vLLM is also held to `--gpu-memory-utilization` (default 0.9)
  of the pool. When the full length does not fit, the context is cut to what does, in 1024-token
  steps (never below 4096), and ymird logs the KV size, the free memory and the budget. MLX-LM
  grows its cache as it goes and wrapped servers set their own, so neither is planned.
- **Overrides.** `context = N`, or `--ctx-size` / `-c` / `--max-model-len` in `args` (moved into the
  launch's context so the flag appears once), is kept as set; ymird warns if its KV cache may not fit.
- **Where the shape comes from.** The GGUF file itself (the reader in `kvfit` also reads the
  sliding-window and loop keys that `weirdnet_model::gguf::GgufInfo` lacks; it is free of Ymir types
  and meant to move there). For a vLLM Hub id: `models_dir/vllm/<owner>--<name>/config.json` and
  `weights`, saved by `pull`, else the Hugging Face cache's snapshot.
- **Older entries.** Pulls before this wrote `context = 32768` (and `--max-model-len 262144` for
  Inkling on a Spark). That cannot be told apart from a chosen value, so ymird keeps it and logs a
  hint when the model could take more. `ymirctl models context` lists every model's planned context
  and why: with ymird running, as the load-time planner below would size each one if it were asked
  for now (beside what runs, on measured free memory, a loaded model's own estimate counted back, and
  the idle models it would unload named); without ymird, the start-up plan; `--old-defaults` (or `--auto ID...`) removes those values from `node.toml` (keeping
  `node.toml.bak`, fetching a vLLM model's `config.json` if it is missing); restart ymird to apply.
- **At load time** (`ymir_core::ctxplan`). The start-up plan is what a model gets while it could
  share the node with every other model. When one is asked for, ymird sizes it beside what actually
  runs: free memory is the smaller of the budget less the running models' estimates and the measured
  free memory (`MemAvailable` less the OS reserve on unified memory and CPU nodes, `nvidia-smi` free
  memory on a discrete NVIDIA GPU). If its full trained length (or explicit context) does not fit,
  running models with no request in flight and none for the last 30 s are unloaded first, longest
  idle first, until it does; a model with a request in flight or loading is never unloaded, and a
  model that fits whole unloads nothing, so two models don't evict each other on every turn. It then
  gets the largest context that fits, with 20% kept free, and ymird logs what it unloaded, the
  context and why. The context is set on the supervisor's launch (removed and re-added while not
  loaded; the supervisor has no setter), so the capability record and every catalog built from it,
  Bragi's budget included, report the value the engine runs with. This runs for loads through the
  node's own inference role (`LocalModelNode::load`). A keeper task re-sizes every automatic model
  that is not loaded whenever what runs changes (and every 30 s), sharing the room among those that
  could load together, so a `Load` from another device's inference node (node/1 goes straight to
  the supervisor), a crash restart, or a model added by a `--node` pull starts with a context that
  fits beside what runs; those loads do not unload idle models (node/1 has no hook before a load;
  that needs `weirdnet-backends`). `[[model.split]]` keeps its set context. A model that is
  itself loaded (asked about by `ymirctl models context`) is sized with its own estimate added back
  to the measured free memory, so it is never sized against itself; one still loading is not. The
  plan's log line gives the whole estimate (KV and buffers, weights, and the 10% that holds the
  engine's fixed-size buffers), which is what a measurement such as `VmHWM` is held to.

Hub search, repo files, GGUF headers, the downloader, the model store and the fit are
`weirdnet-model`'s (`hf::Hub`, `fit`, `download_parts`, `ModelStore`); `crates/ymirctl/src/models/`
keeps what the crate does not cover: MLX repos, the quality-first quant choice, the `fits` /
`partial` / `too big` verdict, `node.toml` handling and the output.

#### Over Weirdnet (`--node`)

Each model node serves `/weirdnet-backends/models/1` (`weirdnet-backends` 0.3) next to the node
protocol, to the owner only: the same roster check as every other Ymir protocol
(`remote::roster_auth`), so only the user's linked, unrevoked devices are answered and a stranger
gets a refusal before anything runs. `[model] remote_manage = false` turns it off on a node.

`ymirctl models search | pull | list | remove --node <name|peer-id|auto|all>` sends the request to
the `ymird` on the machine it runs on (`ControlRequest::Models`, answered as a stream of lines), and
that `ymird` (`crates/ymir-core/src/models.rs`, `ModelsHub`) resolves the node and runs it with a
`ModelsClient` that presents this device's certificate. `--node` takes a device label from
`ymirctl pool list` (any case), a peer id (or its start), `all` (every model node) or
`auto`. A pull streams its progress back (`ModelsReply::Progress`), one line per node.

- `auto` (pulls only) asks every model node how the repo fits it (`Files`: the node ranks the repo's
  models against its own memory budget with `fit::rank`) and takes the node that can hold the biggest
  model, the larger budget on a tie. It says which node it picked and why. `--quant` / `--file` narrow
  the model first.
- `pull --serve` (or `--load`) adds the model to the node's supervisor when the download finishes
  (`Supervisor::add_model`), so it starts serving without a restart. It lives in memory only: nothing
  edits another node's `node.toml`. The pull prints the `[[model.local]]` entry to add to keep it
  across a restart (`file` is relative to that node's models directory). `--register` is local only.
- `remove --node` stops serving the model (`Supervisor::remove_model`) and deletes its files.
- The Hub token is read on the node that downloads (`HF_TOKEN`, `YMIR_HF_TOKEN_FILE`,
  `state/hf-token`, ...) and is not part of any message.
- An older `ymird` (without models/1) answers "does not take model management"; the node then needs
  an upgrade, or the commands run on that machine without `--node`.
- A node with the model role off is not a candidate; the machine running the command can manage its
  own model role in process (`--node <its name>`).

Gaps are listed in HANDOFF.md ("Gaps for the Weirdnet Lead"): launch settings for a served pull
(`-fa`, `--jinja`, `gpu_layers`, `idle_ttl`), MLX in `weirdnet-model`, `SearchHit` parameter counts,
and a per-device owner role.

#### Uploading a model trained elsewhere (`ymirctl models upload`, 2026-11-24)

Slice U1 of [docs/design/model-upload.md](design/model-upload.md), local only. `ymirctl models upload FILE [--name ID]
[--adapter-for BASE] [--license L] [--role R] [--no-register]` takes one GGUF model or GGUF LoRA adapter on this
machine (`ymir_core::upload` on `weirdnet-model` 0.5.3's `artifact::inspect`, `Provenance` and `ModelStore::import`):
the header must read as a model or a LoRA (projectors, split shards, symlinks and other files are refused), an
adapter's base must be a llama.cpp `[[model.local]]` here with the same architecture, and at least 10 percent of the
disk or 20 GiB (whichever is more) must stay free. The file is copied into a staging directory outside the store
while it is hashed, then placed whole: models to `models_dir/ID/`, adapters to `models_dir/adapters/ID/` (with
`adapter.json`, so `ModelStore::adapters` lists them); models uploaded before 2026-11-24 stay in
`models_dir/uploads/ID/`. `provenance.json` (`weirdnet-provenance/1`) beside it records
the files and SHA-256, the revision (`sha256:` and 16 hex digits over the files), the license when given (not
required), the architecture and, for adapters, the base's id, revision and quantisation. The entry is added to
node.toml by default with that revision (adapters with `base_revision` and `base_quant`), so the answer cache and
the base-revision check use it; `--no-register` only prints it. A llama.cpp adapter serves after ymird restarts or
the base next loads.

With `--node NAME` (slice U2) the file goes to another of your model nodes instead: this machine's `ymird` sends it
over models/1 (`ModelsClient::upload`: 48 KiB chunks, checked on the node, resumable after Ctrl-C or a dropped link,
at least `[model] upload_keep_free` left free there, by default the same 10 percent or 20 GiB), the node serves it,
and over Ymir's `/ymir/uploads/1` (`ymir_core::upload_net`) it records the revision for the pool
(`NodeService::set_model_info`, and its origin and license with `set_model_origin`) and adds the entry to its own node.toml, unless `--no-persist`. Any linked device may
upload; refusals name their reason (`exists`, `no-space`, `wrong-arch` and so on). `[model] upload = false` on a node refuses uploads (`not-allowed`) and keeps pulls, listing and removals (U10).

With `--node all` or `--node A,B` (slice U6) it goes to a pool: each target's free space is read first (a node
without room after `upload_keep_free` is left out), the file is sent once, to the node with the most room or
`--stage NODE`, and the others copy it from that node over models/1 `Fetch` (`ModelsClient::replicate`,
`PullSource::Peer`; every model node's models/1 service has a peer dialer and its certificate for this). Each copy is
recorded and saved like an upload. A node that can't copy from the stage gets the file from this device instead.

With `--node` an upload also takes the other formats `weirdnet-model` reads (slice U7): a directory of safetensors
(for vLLM) or MLX weights, a PEFT LoRA adapter directory (`--adapter-for` a vLLM base) and a GGUF vision projector
(`--projector-for MODEL`, a llama.cpp model). `ymir_core::upload::inspect_outgoing` checks and hashes it here
(pickles, code and symlinks are refused); the node's models/1 checks the engine (a full model must suit its
`[model] upload_engine`: llama.cpp, unless `llamacpp` is `off`, then vLLM, then MLX), places it under
`safetensors/`, `mlx/`, `adapters/` or `projectors/`, and `/ymir/uploads/1` saves a `backend = "vllm"` or `"mlx"`
entry, a `[[model.local.lora]]` on the base, or the base's `mmproj`. On the machine itself an upload stays GGUF
models and GGUF LoRA.

Training outputs take the same path (slice U5): `ymirctl train run` imports its adapter with
`ymir_core::upload::import` into `models_dir/adapters/ID/ID.gguf` with `provenance.json` (origin `train`: base,
dataset record, trainer, run) and registers it with its content revision, and `ymirctl models upload [FILE]
--from-run RUN_DIR` uploads a run's adapter with its run record, checked against the record's SHA-256 and base
revision.

`ymirctl train run --node X` (P3a option C, 2026-11-24) runs on another model node over Ymir's own protocol
`/ymir/train/1` (`ymir_core::train_net`, on `weirdnet-net`'s generic streams): the asking device's `ymird` resolves the
node on the roster and sends the dataset, manifest and settings with its certificate; the node serves it only with
`[train] remote = true`, to the owner's devices (`[train] callers` narrows them), runs `train::run` in process and
streams its lines back. Closing the stream cancels the run; node.toml is never edited from the network.

Every model node on Weirdnet also answers `/ymir/train-status/1` (2026-11-25), read-only, for any of the owner's
linked devices: `TrainStatus` with `[train] remote`, its newest 50 runs (`RunSummary`: status, base, adapter id and
revision, dataset name and license, teacher, rows, epochs, steps, loss, times, error, checkpoints) and its jobs
(`JobSummary`: state, holder, attempts, last checkpoint). No local paths leave the node. `ymirctl train status
[--node X] [--json]` shows it, and Bragi's training view reads it with the rollouts (`/ymir/rollouts/1`) to show runs,
versions and canary shares.

`ymirctl train run --job` (P3b, 2026-11-25) makes the run a resumable job: trainer interface 2 checkpoints after each
epoch, and the job record (`ymir_core::train_job`: a lease with a fencing term, attempts, back-off, the last
checkpoint's content root) lets `ymirctl train resume JOB` carry it on from there, on this node or, with `[train]
resume = true`, on another with the same base revision and trainer build. The record is kept on the node until
`weirdnet-agent` 0.6.0's synced job record replaces the stand-in; see [training.md](design/training.md).

### LoRA adapters on a loaded base (training plan P4a, 2026-11-23)

An adapter is served as a model of its own, on its base's engine. The engine side is
`weirdnet-backends` 0.4 (`LoraAdapter`, `ModelLaunch::adapters`, `Supervisor::add_adapter`): llama.cpp
starts with one `--lora FILE` per adapter and `--lora-init-without-apply`, and each request names its
adapter and scale in the body (`"lora": [{"id": i, "scale": s}]`; requests for the base send an empty
list), so the base and every adapter share one llama-server and one KV cache. vLLM gets
`--enable-lora`, one `--lora-modules` list and `--max-loras`, and each adapter is its own served model
name. MLX, wrapped servers and embedding models take none.

- **Config.** `[[model.local.lora]]` under its base: `id` (the pool id, unique among the node's models
  and adapters, e.g. `qwen3-8b+support-v2`), `file` (a LoRA GGUF for llama.cpp, absolute or under
  `models_dir`; a PEFT directory or Hub id for vLLM), `scale` (llama.cpp, default 1.0) and `revision`
  (keys the answer cache like a model's, so answers from an older adapter under the same id are never
  replayed). At start ymird checks each adapter against its base (`ModelLaunch::check_adapters`: a
  LoRA GGUF whose architecture is the base's) and drops, with an error in the log, any that fail; the
  base and the other adapters still serve.
- **Catalog and routing.** The supervisor reports a status row per adapter with `base` set and the
  base's engine, state and capabilities. Capability records carry those rows, so the pool catalog lists
  each adapter as a model and routing, failover and the stand-in need no change: a request for the
  adapter's id goes to a node that has it, like any model. `ymirctl status` shows "LoRA adapter on
  BASE". Loading an adapter loads its base; unloading an adapter is refused (unload the base).
- **Context planner.** An adapter's weights count in its base's footprint (`kvfit::resolve` adds
  `adapters::bytes`: each GGUF's size, or a PEFT directory's weight files), so the start-up plan and
  the load-time plan size the base's context with its adapters loaded. Adapter rows are not residents
  of their own (they mirror the base), a request for an adapter keeps its base busy, a load of an
  adapter is planned as its base, and an adapter added or removed at run time re-reads its base.
- **ymirctl.** `ymirctl models adapter add BASE FILE [--id ID] [--scale S] [--revision R]` adds a
  `[[model.local.lora]]` under its base in `node.toml` (keeping `node.toml.bak` and every other byte;
  refused unless the result validates and, for llama.cpp, the adapter matches the base file here);
  `adapter remove ID` takes it out (the file stays); `adapter list` shows them. Restart ymird to apply.
  Over Weirdnet, `ymirctl models pull REPO --node NAME --adapter-for BASE [--file F] --serve` has the
  node download the repo's adapter into `models_dir/adapters/ID/`, check it against BASE, and serve it
  without a restart (in memory; the output gives the `adapter add` command that keeps it). `models list
  --node` shows adapters with their base.
- **Alerts.** Engine events go to the alerter through `weirdnet-alerts`' own engine monitor again
  (`weirdnet-alerts` 0.2.2 takes `weirdnet-backends` 0.4); the interim copy in Ymir is gone.

**Live changes (second slice, 2026-11-23).** `weirdnet-backends` 0.4.1, `weirdnet-agent` 0.5.2 and
`weirdnet-openai` 0.2.2 change adapters while the base keeps running, where the engine allows:

- **Scale.** `ymirctl models adapter scale ID S|--reset --node NAME` sets an adapter's scale from its
  next request (`Supervisor::set_adapter_scale`; llama.cpp takes the scale per request, so nothing
  restarts). One request can set its own: `ymirctl chat --model ID --adapter-scale S`, or
  `"adapter_scale"` in an OpenAI body to the bridge, carried as infer/2 `Request::adapter_scale` to the
  model node. vLLM applies adapters at full strength and refuses both. Scales must be finite and within
  ±16.
- **Serve and unserve.** `ymirctl models adapter serve ID --base BASE --node NAME [--scale S]` serves an
  adapter already on the node's disk (`models list --node`) on BASE; `adapter unserve ID --node NAME`
  stops serving it and keeps its files (models/1 `ServeAdapter`, `UnserveAdapter`, `AdapterScale`).
  Pulling with `--adapter-for ... --serve` and `models remove` of an adapter use the same live calls.
- **What each engine allows.** vLLM: the supervisor starts it with
  `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` whenever LoRA is on (the base has adapters, or
  `ModelLaunch::lora_slots`), and adds and removes adapters with `/v1/load_lora_adapter` and
  `/v1/unload_lora_adapter` with no restart. llama-server (b12628) loads adapter files only at start
  (`POST /lora-adapters` rescales only those): an adapter whose file the running engine already has is
  live at once (taken off and served again, under any id), and a new file still serves after the base's
  next load, which the first request for it triggers. ymirctl says which happened.
- **Catalog.** catalog/1 entries carry the adapter's `base` (`CatalogModel::base`), so apps such as
  Bragi can label adapters; the bridge's `/v1/models` shows it as `weirdnet.base`.
- These changes are in memory only: `node.toml` keeps what `models adapter add | remove` wrote, and a
  restart of ymird goes back to it.

**LoRA slots and the base-revision check (third slice, 2026-11-23).** No Weirdnet change; P4a is
complete with this.

- **`lora_slots`.** `[[model.local]] lora_slots = N` (vLLM only) starts the base with LoRA on even with
  no adapters listed (`ModelLaunch::lora_slots`: `--enable-lora`, `--max-loras` of at least N, and
  runtime loading allowed), so `adapter serve` and `models pull --adapter-for --serve` add adapters to
  it with no restart. Without it, a vLLM base started with no adapters runs without LoRA and the first
  adapter restarts it, as with llama-server; ymirctl's "after the next load" message says so. Refused
  for llama.cpp (adapter files load only at start: list them under `[[model.local.lora]]`), wrapped
  servers, embedding models and 0. `ymirctl models adapter slots BASE N|--off` sets it in `node.toml`;
  `adapter list` shows it. Restart ymird to apply.
- **Base-revision check.** `[[model.local.lora]]` gains `base_revision` and `base_quant`: the base's
  `revision` and `quant` the adapter was trained on. When both sides are set and differ (trimmed,
  case-insensitive), the adapter is not served (`LocalAdapter::base_mismatch`): ymird drops it at start
  with an error in the log naming both revisions (`adapters::drop_wrong_base`), the base and its other
  adapters still serve, and `adapter list` marks it "NOT SERVED" with the reason. `adapter add` records
  the base's current revision and quantisation by default (`--base-revision`, `--base-quant` to say
  otherwise) and refuses an adapter that wouldn't serve. Nothing recorded, or a base without a
  revision, means nothing to check. Adapters served live over Weirdnet (`serve`, `pull --serve`) are
  not in `node.toml` and are not checked; the GGUF architecture check still applies to them.

### Process manager (replaces llama-swap)

Ymir's process manager has two halves, because models swap across nodes in the cloud:

- **On each model node, the supervisor** (`weirdnet-backends`) does what llama-swap did locally:
  swap groups decide which models may be resident together, idle models unload after a TTL, crashed
  engines restart with backoff, and every start waits for the adapter's readiness probe. It obeys
  `Load`/`Unload`, reports every state change in its capability record, and never kills a model with
  requests in flight (it drains first).
- **On the inference node, cloud-wide swap and placement** (`ymir-core`) decides which model goes on
  which model node (see [Cloud-wide swap and placement](#cloud-wide-swap-and-placement)).

On a single box both halves run in the same `ymird`.

### MCP server (`ymirctl mcp`)

`ymirctl mcp` serves the node to MCP clients. Code: `crates/ymirctl/src/mcp.rs` (tool table, tools, resources and the
loopback bind check). JSON-RPC, the tool table and its schemas, annotations, the stdio loop, the HTTP route and guard, and
resources come from the published `weirdnet-mcp` 0.6 crate (features `stdio-server` and `axum`, so no second
`weirdnet-agent`). It is stateless. It speaks MCP 2026-07-28 and still answers older `initialize` clients.

- **Transports.** stdio (one JSON message per line; requests run concurrently), or `--http [ADDR]`: `POST /mcp`, JSON replies,
  `Authorization: Bearer <bridge token>`, `Host` and `Origin` limited to this machine (`Guard::loopback`). The guard does not
  look at the bind address, so `ymirctl` refuses non-loopback addresses itself (`loopback_addr`, and again on the listener). The `ymird`-hosted variant is not built; the control socket and bridge are the only
  backends, so the client side adds nothing a local `ymirctl` could not do.
- **Tools.** Read: `ymir_status`, `ymir_nodes`, `ymir_devices`, `ymir_models`, `ymir_chat`, `ymir_complete`, `ymir_embed`,
  `ymir_service_status`, `ymir_models_search`, `ymir_models_local`. Mutating: `ymir_load_model`, `ymir_unload_model`,
  `ymir_move_model`, `ymir_pull_model`, and (only with `--allow-remove`) `ymir_remove_model`. `--read-only` hides all five.
- **Safety.** Mutating tools say `MUTATES THE POOL` (or `DELETES FILES`) in their description, carry `readOnlyHint: false`
  (`destructiveHint: true` for unload and remove) and refuse unless the arguments hold `confirm: true` (a string `"true"` does
  not count). Pull and remove without it return a preview. Nothing edits `node.toml` (`--register` is not exposed). Media files
  for chat are accepted by extension only, checked before they are read, within 5 MB (image) and 10 MB (audio).
- **Chat.** `prompt` plus `images` and `audio` build OpenAI content parts (`image_url` data URL, `input_audio`); `messages` takes
  parts as they are. The pool refuses media for a model that does not take it. `route: weirdnet` sends text only, through the
  node's synced conversations.
- **Resources (read-only).** `ymir://status`, `ymir://nodes`, `ymir://models` and `ymir://models/local`, each the same JSON
  as the matching read tool; an unknown URI is a not-found error.
- **Schemas.** Chat `messages` (array of `{role, content}`), `route` and search `format` (enums), `embed` `input` (a string or a
  list) and `max_tokens >= 1` are in the tool table itself (`ParamType::Array`/`Object`/`Enum`/`AnyOf`/`Json`); a unit test
  runs `ToolSpec::check` on it. Tools are async (`call_async`), so nothing blocks a thread.

## Growing the pool

### Joining: your own devices only

The pool is m3talsmith's own devices. A machine joins, in any role, by being **linked as a device of
his Weirdnet identity** (`weirdnet-device-link`, the QR offer and SAS check; it needs a bump from
`weirdnet-net` 0.4 to 0.8 first). It then appears in the `weirdnet-sync` roster, and every node
admits a caller only if `Roster::check_caller` says it is a linked device. `ymirctl pool link` on the
new machine prints the offer; **m3talsmith admits it from an existing roster device** (comparing the SAS
code), then turns on the roles it should run. Until a roster member admits it,
`Roster::check_caller` answers `NotAdmitted` and every node refuses it. Linking sends the identity
seed to the new device (`weirdnet-device-link`), and `SyncState::new` needs that seed, so model and
inference nodes hold m3talsmith's root seed (decided 2026-11-19; see [Root seed on service
nodes](#root-seed-on-service-nodes)). Since M10 (2026-11-24) a server can instead be linked **cert-only**:
it gets a node certificate and a user-signed service grant, never the seed (see [M10 status](#m10-status-built-2026-11-24)). Unlinking (revocation in the roster) cuts the device off everywhere as soon as peers see
the roster update; `weirdnet-sync` rotates its epoch key at the same time.

There are no invites, memberships or permission roles. The role toggles say what a device does,
not what it may do. As defence in depth only, a model node serves only callers that advertise the
inference role in `peer_apps`, so apps can't bypass the coordinator by mistake. That role is
self-declared, so it is never the security boundary; the roster check is.

Discovery uses `weirdnet-net` as it is: nodes find each other by key (`request_key`), user record and
`peer_apps` (each node does `serve_app("ymir", …)` and lists its roles), with `PeerHint`s for LAN and
fixed addresses. mDNS finds same-LAN nodes, but presence never grants anything; the roster does. No
new discovery mechanism is proposed.

### Capability advertisement

Two layers:

- **Model node → inference nodes:** a **capability record** on `/weirdnet-backends/node/1`, pushed to
  every inference node it can reach (never to the DHT): signed by the node key, chained to the user by
  `NodeCert`, with `seq` and a short expiry (90 s, refreshed every 15 s and on change), like
  `weirdnet-net` peer records. Contents: hardware (accelerators, memory budget and free memory,
  interconnects and link speeds, backends); models (id, revision, quant, backend, state, context and
  concurrency limits, load-time estimate; plus models it has on disk and could load); load (queue
  depth per priority across all callers, active slots, EWMA prefill and decode tokens/s, recent time
  to first token); warm prefix hashes; flags (draining, accepts group jobs).
- **Inference node → apps:** the merged **pool catalog** on `/weirdnet-agent/catalog/1`: one entry
  per model with its state (warm, loadable, loading, busy), estimated speed and replica count, but no
  model-node addresses. That is what `/v1/models` and Bragi's model picker show.

### Leaving and failure

- **Model node, graceful:** `ymirctl drain` stops new admissions, finishes in-flight work, publishes
  a final record with `draining`, then a withdrawal. The lease holder re-places its models.
- **Model node crash or network loss:** its record expires (≤ 90 s; inference nodes notice the
  dropped stream sooner). Requests in flight resume on another model node (below).
- **Inference node failure:** apps' pools switch to another inference node. Context is synced, so the
  new one has the conversation. Answers in flight resume from the app's buffer. If it held the
  scheduler lease, another inference node takes it when the lease lapses.
- **Batch jobs:** per-task retry on another model node by idempotency key; a job whose inference node
  dies is picked up by another inference node from the synced job record.
- **Group (split) jobs:** losing any member tears the group down (NCCL can't survive a member
  loss); members return to their normal models, the inference node falls back to the stand-in or a
  smaller model, and an alert goes out.
- **No inference node at all:** apps show the pool as unreachable; Bragi can use its optional
  on-device model.

### Mid-answer failover (decided 2026-11-19: automatic)

Two levels, one mechanism: whoever is receiving the stream keeps a buffer of exactly what it has
passed on and resumes from it.

1. **Buffer what was shown.** As frames arrive, the receiver (the inference node for model-node
   streams, the app's `weirdnet-agent` pool for inference-node streams) appends their text (and
   reasoning text) to a buffer and only then forwards them. Tool-call frames are held until the call
   is complete and never forwarded half-built (`ToolCall` frames carry only complete calls).
2. **Detect failure:** the stream resets, the connection drops, or no frame arrives within the stall
   timeout (default 15 s, or 3× the recent inter-token time, whichever is larger). The stall timeout
   is shorter than `weirdnet-net`'s 30 s stream resume window, so on a stall the receiver sends
   `Cancel` and resets the stream itself rather than waiting for the path to come back.
3. **Model node died** (the common case): the inference node reruns its routing with that node
   excluded, preferring the same model and revision, then the same model at another quant, then the
   stand-in only if the policy allows (flagged), and a node that has the prefix warm. It sends the
   same assembled messages, sampling parameters and seed, the same `job_id` with `attempt` + 1, plus
   `assistant_prefix` = its buffer and its length. It only resumes on a server whose `Accepted` says
   `resume: true`; otherwise it picks another replica. The
   model node renders the chat template with the final assistant message left open (vLLM
   `continue_final_message`, llama.cpp assistant prefill) and generates only what comes after it.
   The app sees one unbroken stream.
4. **Inference node died:** the app's pool sends the request to another inference node with the same
   `context_ref` and `assistant_prefix` = the app's buffer. That node reads the context (synced), and
   continues as in step 3.
5. **No duplicate tokens, by offsets:** every `Delta` and `Reasoning` frame carries its offset in the
   answer. The resumed stream starts at offset = prefix length, and the receiver appends a frame only
   if its offset equals its buffer length (it drops anything at a lower offset, and treats a gap as a
   failure). Frames from an old attempt are recognised by `attempt` and dropped. If a partial tool call
   was held back, the prefix ends before it and the call is generated again whole.
6. **Limits and signalling:** up to two resumes per answer per level; the final frame's metadata
   records each resume offset and node (`x-weirdnet-resumed` on the bridge), and Bragi shows a small
   "continued on server-b" note. If nothing can take it, the app gets the partial answer with a clear error.
   Sampling after a resume is a fresh draw, so the continuation is valid but not bit-identical.

## Context sync

Decided 2026-11-19: nodes sync context. Conversation state is replicated with `weirdnet-sync`
among **inference nodes and app nodes** (Bragi), so any inference node can continue any conversation,
failover resumes with the full context, and routing never depends on a sticky node. Inference nodes
own it: they write answers and checkpoints, apply retention, and serve pulls. Model nodes are not in
the sync set and hold no conversation state; they get fully assembled messages per request. One store
serves inference nodes, Bragi and the bridge.

### Where it lives

All of it is generic and goes in Weirdnet crates, not Ymir:

- **`weirdnet-sync`** 0.5 (already on `weirdnet-net` 0.8): HLC, the append-only op log, LWW registers
  and OR-sets, tombstone GC, sealed pages under the roster's `SyncKey` with epoch rotation, chunked
  values, and the `/weirdnet/sync/2` pull protocol. **Gap:** 0.5 has no way to drop a local copy without
  a tombstone and no on-demand working sets; that needs a `weirdnet-sync` design (Weirdnet lead)
  before M5. Until then inference nodes hold the full context set.
- **`weirdnet-agent` `context` module (new):** the conversation schema as sync records, the
  `context_ref` that requests carry, assembly of a prompt from a context, the storage trait (backed
  by `weirdnet-storage` in apps) and retention rules. `weirdnet-agent` Companions (Morph) can use the
  same store.

### What is synced and what is recomputed

| Item | Synced? | How |
|---|---|---|
| Messages (user, assistant, tool results) | Yes, text | Append-only op per message; an assistant message is written when it completes. Jobs also checkpoint per finished task. Mid-answer resume doesn't need a checkpoint: the receiver's buffer is the source. |
| System prompt, model choice, sampling settings, title | Yes | LWW registers per conversation. |
| Folders, tags, pins | Yes | OR-sets. |
| Attachments | Reference synced, bytes fetched on demand | A `weirdnet-net` content root (BLAKE3) and metadata in the record; chunks are fetched by a node only when a prompt needs them, and cached locally. |
| Retrieved context (RAG snippets, tool outputs) | Yes, as text with its source reference | Stored as sync values (chunked by `weirdnet-sync` up to 16 MiB); anything larger becomes a content root. |
| Prefix-cache hints | Yes, as small hints | Per model and tokenizer revision: block hashes of the token prefix (e.g. a hash per 256 tokens) and which nodes reported them warm. Advisory only. |
| KV cache | **No** | Stays local on the node that computed it: it is large, and tied to the model, quant, backend and hardware. |
| Semantic cache entries | No (separate) | Pooled by `weirdnet-semcache`, not by context sync. |

**KV is rebuilt or warmed, never copied.** A model node that receives a conversation it hasn't
seen prefills it. The router uses the synced prefix hints as a cost reduction (a node with the prefix
warm saves the prefill time) but never as a requirement. To make failover and moves cheap, a node
may warm in the background: when the scheduler places a replica, or a conversation has been active
in the last few minutes, the inference node asks the likely failover target to prefill the hot prefix at `background`
priority (vLLM automatic prefix caching; llama.cpp slot save and restore on the same node across
restarts).

### Size limits

AGENTS.md rules out transfer size caps, so the limits are about what each device keeps, not what
can move:

- Long values need no splitting of our own: `weirdnet-sync` sends pages of up to 112 KiB
  (`PAGE_BUDGET`) and already splits values up to 16 MiB into BLAKE3-checked chunks. Anything bigger
  is a content root.
- **Clients (Bragi) keep everything** they have history for.
- **Inference nodes keep a working set** (once `weirdnet-sync` supports it, see the gap above):
  conversations active in the last 30 days, within a disk budget (default 2 GiB of context per
  inference node), least recently used first out. Evicting drops only the local copy (no tombstone);
  if it's needed again, the inference node pulls it from any device that has it.
- A request whose context the inference node lacks carries the `context_ref`; it pulls missing ops
  from the caller (which always has them) before serving. Small one-off requests can carry their
  messages inline instead.
- What a model node receives is bounded by the model's context window: the inference node assembles
  the prompt (truncating or summarizing per the conversation's policy) before sending it.

### Encryption

- **In transit:** Noise on every `weirdnet-net` stream, and sync pages sealed under the roster's
  `SyncKey` (XChaCha20-Poly1305), rotated when a device is unlinked, so a revoked device can't read
  new context even through a relay.
- **At rest:** every replica lives in an encrypted `weirdnet-storage` (SQLCipher) database, and cached
  attachment chunks in an encrypted chunk store; the key comes from `weirdnet-keys` and the OS
  keyring, never from config. Epoch keys are stored encrypted (`EpochKeys::export`).
- Nodes never log context text; metrics count tokens, not content.

### Retention

- Bragi conversations are kept until deleted (the user's history).
- Bridge conversations (stateless OpenAI calls, mapped by prefix hash) are **ephemeral**: kept 24 h
  by default for failover and follow-ups, configurable, or off.
- Deleting a conversation writes a tombstone; every device removes the text, attachment references
  and prefix hints, and drops KV slots and semantic-cache entries that came from it (cache entries
  record their source conversation). Tombstones are garbage-collected by `weirdnet-sync`'s rules.
- "Delete everything older than N days" is a pool policy setting applied on every device.

### Shared roster and sync group with Morph (decided 2026-11-19)

Ymir and Bragi share Morph's roster and sync group; there is no separate Ymir device list.

- **Same devices, same admission.** One roster of m3talsmith's linked devices, one `weirdnet-device-link`
  flow, one `Roster::check_caller`. A device linked for Morph is already a Ymir device and the other way
  round; unlinking it anywhere removes it everywhere, and the sync epoch key rotates once for all.
  Being in the roster doesn't make a device a Ymir node: its role toggles do.
- **Own namespace.** Ymir records use their own sync types with a `ymir.` prefix (`ymir.conversation`,
  `ymir.message`, `ymir.attachment`, `ymir.prefix_hint`, `ymir.job`, `ymir.policy`, `ymir.placement`),
  next to Morph's (`bookmarks`, `history`, `chats` and the rest). `weirdnet-sync` keeps per-type options
  (`SyncState::configure`: sealing and limits), and every Ymir type is sealed. Neither app reads or writes
  the other's types; merging Morph's `chats` with `ymir.conversation` is a possible later change, not
  part of this design.
- **Opt-in per device.** Each device syncs only the types it enables (as Morph's type opt-ins already
  work). Model nodes enable none; inference nodes enable the Ymir types and not Morph's, so a GPU server
  never holds m3talsmith's bookmarks or browsing history; Bragi devices enable the Ymir types; a desktop
  with both apps enables both sets.
- **Size.** Ymir context can be far larger than Morph's data. Each device's context budget
  (`context_disk_budget`) applies to `ymir.*` types only, and Morph's own data isn't counted against it.
  Big payloads stay out of the op log as content roots.
- **Retention and eviction.** Ymir's retention (24 h for bridge conversations, delete-older-than in the
  pool policy) applies only to `ymir.*` records, and its tombstones only touch Ymir records. Local
  eviction of Ymir context (once `weirdnet-sync` supports it, before M5) never evicts Morph records.
- **Revocation rules apply to Ymir types too:** `weirdnet-sync`'s per-type revocation rules cover
  `ymir.*` records like Morph's, so ops written by a revoked device are voided.
- **Compatibility:** Ymir needs **Morph v0.3 or later** to share a sync group. Morph main already pins
  `weirdnet-sync =0.5.0` and `weirdnet-net =0.8.0` (since `857c3ea`; and `weirdnet-agent =0.4.0`,
  `weirdnet-model =0.4.0`, `weirdnet-mcp =0.5.0` since `6206257`) and ships that in v0.3; the
  released Morph 0.2.x line stays on sync 0.2 / net 0.6 and can't talk to it. All of a user's devices
  move to Morph v0.3 together.

### One store with Bragi

Bragi's chat history **is** this context store: `bragi-core` reads and writes conversations through
the `weirdnet-agent` `context` module, and there is no separate Bragi history database or sync
protocol. A chat started in Bragi on the phone can continue in Bragi on the desktop, through Jan on
the bridge (if it was a bridge conversation) or on any node, because all of them read the same
replicated records.

## Workload spreading (core)

This is what makes the pool one endpoint rather than a list of servers.

### Who decides what

- **Routing and load balancing: inference nodes**, per request, from the capability records they
  receive. With several inference nodes, each routes independently (active-active).
- **Admission: model nodes.** Each model node's `Admission` hook accepts or answers `Busy`, so a
  model node is never overcommitted whichever coordinator sends work.
- **Placement and cloud-wide swapping: the scheduler lease holder** (one inference node at a time,
  see [Inference node](#inference-node)). Its output is a signed, versioned placement plan, synced to
  the other inference nodes and applied with `Load`/`Unload` on model nodes. If no lease holder is
  online, model nodes keep what they have and coordinators keep routing.
- **Apps decide nothing about placement:** they pick an inference node (local first) and a model
  from the catalog.

### Load balancing across replicas

For a model with several warm replicas, the inference node scores each model node by predicted completion:

```
wait  = queue_depth(priority and above) × mean_service_time
cost  = wait + uncached_prompt_tokens / prefill_tps + expected_tokens / decode_tps + rtt
       + (cold ? load_estimate : 0) + (relayed ? relay_penalty : 0)
```

- `uncached_prompt_tokens` uses the synced prefix hints: a node with the conversation's prefix warm
  only prefills the new part, which gives natural (not required) affinity.
- **Power of two choices:** score two random candidates (or all, if few) and take the better one, so
  stale records don't make every client pile onto the same "least loaded" node.
- **Busy feedback:** a `Busy` answer moves on to the next candidate and briefly penalizes that
  node's score.
- `queue_depth` is what the model node reports across all coordinators, plus this coordinator's own
  requests sent since that report, so several inference nodes don't oversubscribe one model node.
- **Local first, on ties:** a model node on the same device wins ties, which is also the single-box
  path.

### Cloud-wide swap and placement

Each model node's supervisor swaps models in and out of its own memory; the scheduler (the lease
holder) does the same across the cloud. It decides which models live where from the pool policy (models wanted,
`min_replicas`, `max_replicas`, pins such as "27B NVFP4 only on Spark-class nodes"), each node's
memory and backends, and demand (EWMA requests/min and tokens/min per model).

- **Swap across nodes:** a request for a model that isn't resident anywhere loads it on the model
  node where it fits best and costs the least to evict for (least-used models, idle nodes first), not
  necessarily the node the client asked. When demand for a model fades, it is unloaded there and the
  memory goes to whatever is in demand.
- Bin-pack by memory with headroom; prefer the fastest backend for the format (NVFP4 → vLLM on
  Blackwell; GGUF → llama.cpp).
- **Replicate popular models:** when a model's queue wait stays over a target (default 2 s for
  interactive), add a replica on the node with the most free budget; drop extra replicas after
  sustained low demand, keeping `min_replicas`.
- Keep the stand-in and the embedding model warm on at least one node each, always.
- Changes are rate-limited (no flapping), never evict a model with requests in flight (drain first),
  and ask the target to warm hot prefixes before traffic moves.

### Splitting batch and agent jobs

An inference node accepts a job from an app (through the `weirdnet-agent` pool): a list of independent tasks (prompts, embedding chunks
over a corpus given as a content root, or `remote::task` subtasks with their approvals). It fans
them out with the same scoring at `batch` priority, with a concurrency cap per node, per-task retries
by idempotency key, and results streamed back as they finish (or in order, on request). Cancel stops
all of them. The job record is synced, so if its inference node dies another picks it up, and the
laptop that submitted it can sleep while it runs.

Approvals for agent subtasks: `remote::task` normally asks the submitting device, which may be
asleep. Rule: a job declares **pre-approved tool scopes** when it is submitted (approved then, on
that device, under `weirdnet-agent`'s policy); a subtask that needs anything outside them waits for
approval from the submitting device up to a timeout (default 10 minutes) and is then **denied**, and
the job carries on without it. A subtask never gains tools the parent didn't have.

### Data generation and evals (training plan P1)

Built 2026-11-22 (see [docs/design/training.md](design/training.md)). `ymirctl batch --jsonl` sends rows with their own
messages, sampling and seed (`ControlRequest::BatchTasks`, `ymir_core::batch::BatchTask`) in chunks of 32, waits for the
teacher model (no stand-in), and writes JSONL rows plus `OUT.manifest.json` (`ymir_core::dataset`): teacher, the
teacher's license (`--license`, else `--license-from` a Hub repo's model card, else the pool catalog's; `unknown`
only with `--allow-unknown-license`), input hash, counts and the models that
answered. A row is `usable` only when it answered fresh from the teacher; cache hits and stand-ins are read from the
answer's `Meta`. `--resume` skips rows already answered and keeps the job id. `ymirctl eval` scores models on rows with
`expect` (the bench scorers) and/or `rubric` (graded by `--judge`), at temperature 0 with a fixed seed.

Sources: `--sources FILE` takes a sources manifest (`dataset::Sources`: a topic, and each source's id, title, URL and
license; Ymir fetches nothing). Every row then names its `source`, unknown ids are refused before anything runs, and
the dataset's manifest records the sources, their hash and rows per source.

Answer cache: requests carry infer/2 `Request::cache` (`weirdnet-agent` 0.5.1; `batch::CacheMode` re-exports it).
`batch --jsonl` defaults to `bypass` (`--cache refresh|use` to change it) and `eval` always bypasses; both set
`wait_for_model`. The coordinator honours the mode itself: `prepare` reads the cache only when `cache.may_read()`, and
a finished answer is offered only when `cache.may_store()` (`bypass` neither reads nor stores, `refresh` stores
without reading). The bridge takes the same mode from the body's `cache` or `x-weirdnet-cache` (`weirdnet-openai`
0.2.1). Rows still record `Meta::cache_hit`, so a `--cache use` run marks replayed rows unusable.

Weights revisions: `[[model.local]] revision` (a Hub commit, file hash or fine-tune id) goes in the node's capability
record (`ModelEntry::revision`) and `ModelNode::revision`. Answers are stored with `put_rev` under the revision of the
node that answered and read with `get_rev` under the model's revision when every node with the model reports the same
one (none reporting one is plain `get`/`put`); when replicas differ the cache is not read, since the answering weights
aren't known in advance. Answers carry the revision in `Meta::revision`. This is what keeps a fine-tuned model under
the same id (P4) from replaying answers of its old weights.

### Training data from conversations, with consent (training plan P2)

Built 2026-11-24, Ymir side (see [docs/design/training.md](design/training.md)). Consent is its own record in
`ymir.context`: `{conversation}/train`, a last-writer-wins register `{"on": bool, "at_ms": ms}`
(`ymir_core::context::Train`), not a field in `meta`, so an older client that rewrites `meta` never clears it.
Absent means off; it is never inferred. `ymirctl context train ID [--off]` sets it (the conversation must exist here;
it syncs like any context record) and `context show` marks consented ones `[training]`. Bragi writes the same record
(a cap button in the Flutter app bar, a Training button in the GPUI shell, `/train` in the terminal app). Messages
deleted in Bragi (`{"deleted": true}` over the record) are skipped.

`ymirctl context export --training OUT.jsonl` asks the local ymird for the consented, live conversations with their raw
message values (`ControlRequest::ContextForTraining`, local transport) and writes them with `ymir_core::convexport`:
one chat-format row per conversation (`{"id", "messages": [{"role", "content"}], "models"}`). Strip rules: system
prompts dropped unless `--keep-system`; tool calls and results, attachments and reasoning dropped (only message text
is kept); the conversation is cut at the first answer that is not complete (stopped, failed, streaming) or came from
a stand-in, and trailing user turns go, so a row ends on an answer; `--redact REGEX` (repeatable) replaces matches with
`[redacted]`. The output and `OUT.manifest.json` (`ymir-conversations/1`: conversation ids with consent and update
times, kept turns, what was dropped by kind, redaction count, the number and SHA-256 of the patterns but never the
patterns) are written owner-only (0600) through a temporary file, are never synced, and the command prints counts
only. `ymirctl context withdrawn MANIFEST` lists the conversations in an earlier export that are no longer consented
or were deleted, so a dataset can be rebuilt without them.

### Versioned rollouts (training plan P4b)

Built 2026-11-24 (see [docs/design/training.md](design/training.md), "P4b design"). The pool policy on an inference
node names a rollout and its versions, each a model id in the pool with a traffic weight:

```toml
[[inference.rollout]]
name = "support"
[[inference.rollout.version]]
version = 2
model = "llama-8b+support-r2"
weight = 95
[[inference.rollout.version]]
version = 3
model = "llama-8b+support-r3"
weight = 5
```

The coordinator resolves `support` to one version per conversation (weighted, from a hash of the conversation's
synced id or its opening messages) and `support@3` to version 3. The catalog lists `support` and every `support@N`,
`Accepted` names what was asked for, and answers are cached under the version's model id. `ymirctl models rollout`
changes the weights live and in node.toml. A raw weight that would give a version more than a 10 percent share
needs an eval at least as good as the current main version's (`--eval`), or `--force`. A share named outright,
`--canary PCT` (any whole percentage from 0 to 100), is set exactly and is not capped: the version's weight becomes
PCT and the other versions share the rest in their present proportions; `--eval`, when given, must pass first
(decision of 2026-11-25). Rollback moves the traffic back, and the old version stays loaded
until retired and unloaded.

`ymirctl models upload FILE --node NODE --rollout NAME` (model-upload slice U3) feeds a rollout from an upload: the
upload is named `<id>-r<N>` for the next version N, added as `NAME@N` at weight 0 on this machine once the node
serves it, and `--canary PCT` gives it exactly that share (with `--eval ROWS [--judge MODEL]`, which runs `ymirctl
eval` on it and the version with the most traffic, only if it does at least as well). `models rollout list` shows each version's origin, license, base and
dataset hash from its provenance.

Another inference node's rollouts are read and changed over Weirdnet with `/ymir/rollouts/1` (`ymirctl models
rollout ... --inference NODE`, and Bragi): the same rules (`ymir_core::rollout_edit`, shared with the local command:
a raw weight above the canary share carries the scored eval rows or `force`; `Change::Share` sets a percentage), applied live and saved in that node's
node.toml unless `--no-persist`. Any of the owner's linked devices may, narrowed by `[inference] rollout_callers`
(peer ids); `[inference] remote_rollouts = false` turns it off. See
[docs/design/model-upload.md](design/model-upload.md#rollouts-over-weirdnet-ymirrollouts1-planned-2026-11-25-follow-up-to-u4).

### Shared semantic cache

`weirdnet-semcache` (new) generalizes the RedisVL `rvl-cache`:

- **Rules carried over from `rvl-app.py` `cacheable()`:** single user turn, text only, no tools, one
  choice; the key context is a hash of the system prompt and sampling parameters; models can opt out.
  Cosine-distance threshold **0.12** (server-a's deployed value), TTL 24 h.
- **Storage (AGENTS.md storage rule):** the crate defines a storage trait; the app owns the schema and
  migrations. Ymir backs it with an embedded vector index (HNSW) in its own encrypted
  `weirdnet-storage` database by default, or Redis/RedisVL.
- Runs on inference nodes, before routing (a hit never reaches a model node).
- **Pool mode** (several inference nodes): entries are sharded by rendezvous hashing on the model plus the
  key-context hash (system prompt and sampling parameters) to a home inference node plus one replica, so a lookup is one call. All nodes use the same embedding model and revision, pinned by
  the pool policy; a change starts a fresh cache. The pool is one owner, so entries are shared across
  all linked devices.
- Hits are flagged to the client and can be skipped per request (`cache: false`).
- **Built (2026-11-26, local mode):** `[inference.cache] embedding_model = "ID"` turns on `weirdnet-semcache`
  0.2.2's `SemanticCache` behind the exact cache, in `Coordinator::run`
  ([crates/ymir-core/src/semantic.rs](../crates/ymir-core/src/semantic.rs)). A prompt is cacheable when it is one user
  turn after any system messages, text only, with no tools, no prefix to continue and no stand-in answering. The
  user turn is embedded with that model, only while it is loaded on some node (the cache never cold-starts it, and
  a failed embedding just skips the cache). The scope is the model, its weights' revision, the embedding model and
  its revision, and a SHA-256 of the system prompt, sampling, seed, response format and adapter scale. A hit needs
  cosine similarity `1 - threshold` (default distance 0.12, so 0.88), within `ttl` (default `24h`), among at most
  `entries` (default 1024, least recently used first). A hit answers as `served_by = "semantic-cache"`.
  `cache: bypass` neither reads nor stores, `refresh` stores without reading. Tests:
  [crates/ymir-core/tests/semantic.rs](../crates/ymir-core/tests/semantic.rs).

### Fairness and priorities

The pool has one user, so fairness is between **devices** and between **priority classes**:

- Three classes: `interactive` > `batch` > `background` (prefix warming, re-placement). Batch work
  yields between requests, never mid-stream. Interactive requests may take any free slot first.
- Within a class, each inference node queues by **deficit round robin per calling app device**, so
  a laptop running a big batch job can't starve the phone's chat. Weights per device are optional.
  Model nodes order their queue by priority class only.
- A device that is also a model node can reserve a small share of its own slots for its own apps, so
  a machine you're sitting at stays usable while the cloud is busy.
- No quotas or rate limits (they replaced Caddy's `rate_limit` for other tenants, which no longer
  exist).

## Splitting models across nodes (big mode, generalized)

**Opt-in, not the normal way to scale** (decided 2026-11-19). Cross-machine splitting needs a private
link: a direct cable or an isolated VLAN, like the server-a/server-b ConnectX-7 cable. That is allowed, but most
layouts won't have one, so it is off unless a pool policy marks a model `split` and the nodes report a
private link. The normal way to scale is routing and replication across nodes: more model nodes, more
replicas of popular models, and cloud-wide swapping. Ray and llama.cpp RPC never run over a LAN shared
with other devices, Wi-Fi, a tailnet or relays.

On server-a/server-b, `big-mode on` stopped llama-swap on both boxes, started a Ray head and worker over the
ConnectX-7 link (NCCL over RoCE, Ray ports 23001 and 23100-23999) and served one big model with
vLLM. Generalized as a **group job** decided by the scheduler (the lease-holding inference node) and run by model nodes:

1. A model in the pool policy is marked `split` with allowed backends (`vllm-ray` for TP/PP across
   Linux NVIDIA nodes; `llamacpp-rpc` for GGUF across any nodes).
2. The scheduler picks a group of model nodes whose combined memory fits, preferring nodes with a measured fast
   link between them (RDMA/RoCE ≥ 100 Gb/s for tensor parallel; pipeline parallel or llama.cpp RPC can
   live with 10 GbE; it refuses over Wi-Fi and relays).
3. `GroupUp` goes to every member: each model node's supervisor drains the models the plan says to drop,
   starts its worker (Ray worker or `rpc-server`) bound **only to the private-link address**, and
   reports ready. The leader starts the serving process (vLLM with Ray, or `llama-server --rpc`) and
   advertises the model.
4. Inference nodes reach the group's leader model node over Weirdnet like any model node. Tensor traffic stays on the link.
5. `GroupDown` (or a member loss) reverses it, and normal placement resumes.

Big mode can be triggered manually (`ymirctl group up qwen-big`), on a schedule, or by the
scheduler when demand for the split model crosses a threshold. The NCCL settings in
the deployed `nccl.env` become backend defaults for RoCE groups (set from detected
hardware, not hard-coded per host).

## Mapping from server-a/server-b to Ymir

| server-a/server-b piece | New home |
|---|---|
| llama-swap + `config.normal.yaml` (model processes, swap groups, TTL) | model-node supervisor in `weirdnet-backends` (local swap) + inference-node scheduler in `ymir-core` (cloud-wide swap); importer `ymirctl import llama-swap` |
| llama.cpp builds in `/opt/llm/llama.cpp` | `weirdnet-backends` pinned builds (CUDA 13 for sm_121) |
| vLLM 0.34 + Rust frontend, qwen3 reasoning parser, FlashInfer prebuilt, 1800 s startup timeout | `weirdnet-backends` vLLM adapter (options in the pool policy) |
| `config.big.yaml`, `config.big.<host>.yaml`, `/opt/llm/bin/big-mode` | group jobs: decided in `ymir-core` (inference), run via `GroupUp`/`GroupDown` on `/weirdnet-backends/node/1` (model); importer in `ymir-core` |
| `ray-head`/`ray-worker` units, Ray ports 23001/23100-23999, memory monitor off | `weirdnet-backends` Ray adapter, started by the model-node supervisor |
| `nccl.env` (NCCL over RoCE, NET/IB) | `weirdnet-backends` RoCE defaults from detected hardware |
| `rvl-cache.service` / `rvl-app.py` semantic cache, threshold 0.12 | `weirdnet-semcache` on the inference node (embedded default, Redis optional) |
| rvl-app cold-start coordinator (`gemma4-e4b` while the 27B loads, `/rvl/status`, `/rvl/warm`) | the inference node (it is the coordinator now): stand-in policy and warm-on-demand; status in the catalog |
| Redis 8 | optional `weirdnet-semcache` backend |
| Caddy, `tls internal`, ports 443/9443, per-tenant keys in `keys.env`, `rate_limit` | removed: apps reach inference nodes over Weirdnet (`Roster::check_caller`); local tools use the loopback bridge (an app node); device fairness on inference nodes |
| `server-b/<model>` name prefixes for models on server-b | the inference node's pool catalog (one name per model; routing picks the model node) |
| `ts-watchdog`, Tailscale hardening | stays as host ops, outside Ymir (Tailscale becomes optional); a `tailscale` probe can feed `weirdnet-alerts` |
| monitor timer on the dev box | inference nodes watching model-node records + `weirdnet-alerts` |
| msmtp Gmail alerts, `ts-alert` | `weirdnet-alerts` SMTP and push sinks (secret in the keyring) |
| Wi-Fi fallback alert (the old route-state and netcheck timers) | `weirdnet-alerts` "network degraded" rule from each node's health report (default-route interface) |
| server-a NAT, MSS clamp, subnet route for server-b | host networking, out of scope; Weirdnet relays and hole punching reach server-b without it |
| `hfget`, `hfcli` | `weirdnet-model` download with consent |
| `reclaim.sh` | out of scope (host maintenance) |
| `e2e*.sh` smoke tests | Ymir integration tests (`ymirctl selftest`) |
| (new) conversation state | `weirdnet-agent` `context` over `weirdnet-sync`, owned by inference nodes |
| (new) app-side model discovery and prompts | `weirdnet-agent` `pool` on app nodes (Bragi, bridge, `ymirctl`) |

## Crates

### Existing, reused (versions as of 2026-11-19, rechecked after M2; all 0.x and unaudited)

| Crate | Used for |
|---|---|
| `weirdnet-net` 0.8.0 | node, streams, `serve_app`/`peer_apps`/`resolve_app`, key-only dialing, relays, hole punching, `content` for attachments and large inputs, `events`/`network_changed` |
| `weirdnet-keys` 0.1.1 | node and user keys, sealing, storage keys |
| `weirdnet-sync` 0.5.0 | device roster and `Roster::check_caller` (already in 0.5.0; the only authorization), synced pool policy, and context replication (HLC, op log, LWW, OR-set, sealed pages, chunked values). Already on net 0.8. |
| `weirdnet-device-link` 0.3.0 | linking a machine to m3talsmith's identity. On `weirdnet-net =0.8.0` (the M3/M4/B1 gate's device-link half is released). |
| `weirdnet-agent` 0.3.0 | `remote::infer` on `/weirdnet-agent/infer/1` (`InferHost`, `Admission`, `RemoteModel`), `remote::task`, `ModelBackend`. 0.4.0 adds `infer/2`, `pool` and `context` (below) and is the M3 gate. |
| `weirdnet-model` 0.2.1 (Ymir M1/M2 pin; 0.3.0 is out on agent =0.3.0, 0.4.0 comes with agent 0.4.0) | catalog, consent-gated download, in-process llama.cpp, CPU checks |
| `weirdnet-storage` 0.1.0 | encrypted local databases (context replicas, cache, metrics) |
| `weirdnet-push` 0.1.0 | sealed 96-byte wake payloads, `ReplayGuard`, and the `server` feature's APNs and FCM senders, embedded in the inference node |
| `weirdnet-rules` 0.1.0 | the rule engine under `weirdnet-alerts` |
| `weirdnet-schemes`, `weirdnetctl` | registering `weirdnet-ymir://` and `weirdnet-bragi://` |
| `weirdnet-update` | signed update metadata for `ymird` self-update (decision 14) |
| `weirdnet-mcp` 0.4.0 (on agent =0.3.0; 0.5.0 comes with agent 0.4.0) | optional `ymirctl mcp` tools (status, load, chat) |
| `weirdnet-relay` 0.6.0 | the public [WeirdBrains](https://weirdbrains.com) relay and an optional own relay. On `weirdnet-net =0.8.0`; the public relay runs 0.6.0 (weirdnet-relay `07f1162`). |

No longer used by this design: `weirdnet-access`, `weirdnet-accounts` and `weirdnet-tls` (no
roles, no API keys, no HTTPS endpoint).

### Extensions to existing crates (Weirdnet lead)

- **`weirdnet-agent`**:
  - `/weirdnet-agent/infer/2` beside `/1` (see [Inference protocol details](#inference-protocol-details)):
    priority, `job_id` + `attempt`, `context_ref`, `assistant_prefix`, `Accepted.resume`, offsets on
    deltas, reasoning and tool-call frames, `Queued` with a real priority queue, raw mode, embeddings,
    caller peer and cert passed to `serve()` and `InferHost::open`. `RemoteModel`'s default stays
    "give up once tokens have streamed"; resume is opt-in.
  - **`pool` module, app side** (decided 2026-11-19, replaces the earlier `weirdnet-pool` proposal):
    find inference nodes among linked devices, query and subscribe to the catalog
    (`/weirdnet-agent/catalog/1`), send prompts with streaming and cancel, fail over between inference
    nodes, resume from the app's buffer, submit jobs. It sits next to `remote::infer` and
    `RemoteModel` (`RemoteModel` becomes a thin `ModelBackend` over the pool), so any Weirdnet agent
    app can use the cloud. Used by `bragi-core`, `weirdnet-openai` and `ymirctl`.
  - **`serve` host traits, inference side:** the server half of the catalog and infer protocols, so a
    coordinator (Ymir's, or another app's) plugs in its routing behind them.
  - **`context` module**: the conversation schema over `weirdnet-sync`, `context_ref`, prompt
    assembly, job checkpoints, storage trait and retention.
  - `pool` and `context` are **cargo features, off by default**, so `weirdnet-sync`, `weirdnet-net` and
    libp2p don't land on Morph, `weirdnet-model` and `weirdnet-mcp` unless they ask.
- **`weirdnet-model`**: `hw` detection, tiered catalog with memory footprints, a sizing function.
- **`weirdnet-device-link`, `weirdnet-relay`**: releases on `weirdnet-net` 0.8, and a redeploy of the
  public relay (gate for M3, M4 and B1).
- **`weirdnet-sync`**: running on an app's `Node` (option B above) and local eviction / on-demand
  working sets (before M5); later, a cert-only server link (M10).
- **`weirdnet-push`**: a Weirdnet (non-HTTP) front end for the server `Service`, so it can be embedded
  with no listener.
- **`weirdnet-update`**: the self-update flow for `ymird` (signed full packages, boot-health rollback)
  uses what 0.1.0 already has; any gaps found in M9 go to the Weirdnet lead.

### New crates proposed (Weirdnet lead)

| Crate | Holds |
|---|---|
| `weirdnet-backends` | the **model-node wrapper**: engine adapters for llama.cpp server, vLLM, Ray and llama.cpp RPC (launch, readiness, abort, slot save and restore, pinned builds), the local supervisor (swap groups, TTL, restarts, drain), wrapping a stock server by URL, and the `/weirdnet-backends/node/1` protocol behind a `node` feature (capability records, `Load`/`Unload`/`Drain`/`GroupUp`/`GroupDown`, health). Engine builds come through `weirdnet-model`'s download and consent. It decides nothing cloud-wide. |
| `weirdnet-openai` | the loopback-only OpenAI-compatible bridge (an app node) over the `weirdnet-agent` pool: token on every endpoint (kept in the keyring), `Host` check, no CORS, `x-weirdnet-*` headers |
| `weirdnet-semcache` | semantic cache: rules, a storage trait (the app owns the schema), optional Redis backend, sharding by model + key-context hash across inference nodes |
| `weirdnet-alerts` | sinks (SMTP, webhook, push wake) and a health rule pack with dedupe and recovery notices, on `weirdnet-rules` |

### Interim crates in this repo (to be extracted)

m3talsmith's rule for the build: no `weirdnet-*` crate is created or changed from this repo; the
published ones are used as they are (`weirdnet-model` =0.2.1 for its catalog and consent-gated
download). Generic code that M1 needs lives in separate, self-contained crates here, each marked
"to be extracted" in its README and `Cargo.toml` description, with no Ymir types in it. Moving one is
a rename plus a dependency swap. All are `publish = false`.

| Interim crate | To be extracted into | Holds |
|---|---|---|
| `ymir-hw` (retired in M3: now `weirdnet-model` 0.4 `hw`) | `weirdnet-model` (`hw` module, tiered catalog, sizing) | detection (NVIDIA via `nvidia-smi`, AMD sysfs, Apple `sysctl`, `/proc/meminfo`; GB10 as unified), the usable budget, a tiered catalog of `weirdnet_model::ModelSpec`s with footprints, `plan()` |
| `ymir-backends` (retired in M3: now `weirdnet-backends` 0.1.0) | `weirdnet-backends` (the Weirdnet lead extracts it into a new repo after M2) | llama-server and vLLM adapters, wrapping a stock server on loopback, the local supervisor (load, readiness, drain, unload, swap groups, idle TTL, crash backoff and crash loops, `SupervisorEvent`s, logs), chat/completions/embeddings forwarding, SSE decoding. M2 changes are done. |
| `ymir-openai` (retired in M3: now `weirdnet-openai` 0.1.0) | `weirdnet-openai` | the loopback-only bridge (token on every endpoint, `Host` check, no CORS, `x-weirdnet-*` headers) |
| `ymir-semcache` (retired in M3: now `weirdnet-semcache` 0.1.0) | `weirdnet-semcache` | the `AnswerCache` seam and an exact-match placeholder |
| `ymir-infer` (retired in M3: now `weirdnet-agent` 0.4 request types) | `weirdnet-agent` (infer/2 frames, `pool`, serve host traits); Ymir moves to agent's request types in M3 | `ChatRequest`, `CompletionRequest`, `EmbedRequest`, `Event` (offsets, reasoning, tool calls, `Queued`), `Pool`, `CatalogModel` |
| `ymir-alerts` (retired after M3: now `weirdnet-alerts` 0.1.0) | `weirdnet-alerts` | observations, `weirdnet-rules` `RuleSpec`s with `fail`/`ok`/`event` actions, the server-a/server-b rule pack, hourly dedupe with recovery notices, SMTP (lettre), webhook, push (off) and log sinks with dry-run, route/disk/GPU probes |

Also generic but small, in `ymir-core` for now: `service` (systemd units, launchd agents and
daemons, the Windows logon task and `sc.exe` service registration, as testable plans; a candidate for a
`weirdnet-*` service crate, noted for the Weirdnet lead) and `engine` (finding the llama.cpp build shipped next to the executable, for
`weirdnet-backends`). `health` moved to `weirdnet_alerts::engine`.

Trait seams that stay in `ymir-core` until their protocol exists: `ModelNode` (becomes the
`/weirdnet-backends/node/1` client plus raw infer/2 in M3) and the control socket (the local transport
of `/ymir/control/1`).

### App crates (m3talsmith and this team)

- `ymir-core`: the **inference-node coordinator** (catalog, routing and load balancing, scheduler
  lease, cloud-wide swap and placement, group-job decisions, cold-start stand-ins, mid-answer resume
  of model-node failures, device fairness, `/ymir/coord/1` and `/ymir/control/1`), role wiring, config
  and pool policy, setup wizard, importers.
- `ymird`: one daemon; enables the model, inference and app (bridge) roles from `node.toml`.
- `ymirctl`: the CLI (an app node for `chat`, and the admin front end).
- In the Bragi repo: `bragi-core`, `bragi-desktop`, `bragi-bridge`.

All crates set `license = "BSD-3-Clause"`.

## Security and auth

- **Identity:** every node and client is a Weirdnet node with a `NodeCert` chained to m3talsmith's
  user. Noise authenticates every stream end to end, relays included.
- **Authorization** is one check on the serving node, per call: `Roster::check_caller` must say the
  caller is a linked device. Anything else is refused before `InferHost::serve` or a control handler
  runs; `serve()` and `InferHost::open` get the authenticated peer and cert to do it. There are no
  permission roles, memberships, invites or API keys. On top of that, as defence in depth only, model
  nodes serve only callers advertising the inference role, and inference nodes answer `/ymir/coord/1` only
  from other inference nodes. Co-located roles use the same identity and the same check on the
  in-process path.
- **Revocation:** unlinking a device in the roster cuts it off everywhere as soon as peers see the
  update, and rotates the sync key so it can't read new context.

### Root seed on service nodes

Decided 2026-11-19: for now, model and inference nodes get m3talsmith's root identity seed through
`weirdnet-device-link`, like any linked device. They are his own machines on his own network.

- **Risk:** anyone who copies the seed off a node *is* m3talsmith on Weirdnet, in every app, not just
  Ymir. Unlinking that node in the roster stops it as a device, but doesn't help against a stolen
  seed: recovering from that means rotating the root identity on every device. A headless server is
  also more exposed than a phone (remote shells, backups, disk images).
- **Mitigation now:** the seed is stored in the OS keyring where one exists, otherwise in a file with
  mode 0600 owned by the service account (system installs) or the user, never in `node.toml`, logs,
  backups made by Ymir, or crash reports. `ymirctl` never prints it. Disk images and backups of
  service nodes should be treated as holding the identity.
- **Built in M10 (2026-11-24):** a cert-only server link (`weirdnet-sync` 0.7, `weirdnet-device-link`
  0.4), so a service node gets a node certificate and the sync keys it needs but not the seed, and can be
  revoked without rotating the root identity. `ymirctl pool cert-only` moves an existing node over and
  wipes its seed copy. Nodes linked the old way keep their seed until that is run on them.
- **No network listeners besides Weirdnet.** The OpenAI bridge binds `127.0.0.1` only and can't be
  configured otherwise. Backend ports (llama-server, vLLM) bind to loopback. Ray workers and
  llama.cpp RPC servers for groups bind to the private-link address only; they have no auth of their
  own, so the private link must be a direct cable or an isolated VLAN (decided 2026-11-19).
- **Prompts, answers and context** cross only Noise streams, with sync pages sealed under the
  roster's key; at rest they are in encrypted databases. Nodes don't log prompt or answer text;
  metrics carry counts and timings only.
- **Least data per role:** model nodes see only the assembled prompt for the request in hand and keep
  nothing after it (beyond their local KV cache); context lives on inference and app nodes.
- **Private by default:** capability records and context go only to linked devices, never to the DHT.
- **Secrets** (SMTP password, bridge token, storage keys) never appear in config files, logs or docs.
  Config references keyring entries by name.
- Downloads of backends and models are checksummed and pinned; model downloads need consent.

## Config format (one file per node)

`node.toml` (in `~/.config/ymir/` for a user install, `/etc/ymir/` for a system install; `YMIR_HOME`
overrides). The three roles are independent toggles; each role reads only its own section, and a
section for a disabled role is ignored. Pool-wide settings (models wanted, replicas, stand-ins,
cache, context retention, fairness weights) live in the **pool policy**, synced among inference and
app nodes with `weirdnet-sync` and edited with `ymirctl policy`; `node.toml` may seed it on the first
inference node.

```toml
# ~/.config/ymir/node.toml (example: server-a = model + inference; no secrets belong here)
[node]
name = "server-a"

[roles]
model = true                         # run models and take requests from inference nodes
inference = true                     # coordinate: catalog, routing, placement, context sync
app = false                          # the loopback OpenAI bridge (Bragi is a separate app)

[network]
bootstrap = "default"                # the public WeirdBrains relay; or a list, or []
listen = ["/ip4/0.0.0.0/tcp/4747", "/ip4/0.0.0.0/udp/4747/quic-v1"]   # Weirdnet only

# ---- model role ----
[model]
memory_budget = "auto"               # or e.g. "100GiB"
reserve = "12GiB"                    # unified memory kept for the OS
private_link = "10.0.0.1"            # optional: private address for group jobs (RoCE/NCCL, RPC)
llamacpp = "managed"                 # managed | system:/path | off
vllm = "auto"                        # auto (only where supported) | off
# wrap = "http://127.0.0.1:8000"     # or wrap a stock vllm/llama-server already running here

[[model.local]]                      # models this node keeps on disk and may be told to load
id = "qwen3.6-35b-a3b"
backend = "llamacpp"
quant = "Q4_K_M"
swap_group = "main"
# revision = "<hub commit or fine-tune id>"  # optional: keys the answer cache on these weights

[[model.local]]
id = "gemma4-e4b"

[[model.local]]
id = "qwen3-embedding-0.6b"

# ---- inference role ----
[inference]
scheduler_eligible = true            # may hold the placement lease
context_disk_budget = "2GiB"         # working set of synced conversations

[inference.placement]                # optional (M6): models kept loaded on a number of model nodes
replicas = { "qwen3.6-35b-a3b" = 2 }
every_secs = 15                      # how often the scheduler looks

[inference.cache]
mode = "local"                       # off | local | pool
backend = "embedded"                 # embedded | redis
redis_url = "redis://127.0.0.1:6379" # used only when backend = "redis"
threshold = 0.12                     # cosine distance for a semantic hit
embedding_model = "bge-m3"           # optional: turns on the semantic cache (used only while loaded)
ttl = "24h"
entries = 1024

[inference.alerts]
push = true                          # Weirdnet push to m3talsmith's devices
smtp = { to = "owner@example.com", server = "smtp.gmail.com:587", user = "owner@example.com", password_ref = "keyring:ymir-smtp" }

# ---- app role ----
[app.bridge]
port = 8484                          # always 127.0.0.1; there is no address setting

# ---- fine-tuning (training plan P3a; optional) ----
# [train]
# trainer = "/path/to/scripts/trainer.py"  # speaks Ymir's trainer interface (docs/design/training.md)
# build = "b12628"                   # the llama.cpp build it must report; any other is refused
# resume = false                     # carry on other nodes' training jobs from their checkpoints (P3b)
# [train.env]
# YMIR_TRAINER_PYTHON = "/opt/ymir-trainer/bin/python"  # a venv with scripts/trainer-requirements.txt
# YMIR_LLAMA_FINETUNE = "/opt/ymir-trainer/llama-finetune"  # scripts/engine.py build --trainer
```

The single-box quickstart writes all three toggles as `true`. A laptop that only runs the bridge
has `[roles] app = true` and nothing else.

## Install and deploy

- **Packages** per the platform matrix: Linux `tar.gz`, `deb`, `rpm`, `pkg.tar.zst` (TAUR), Nix flake,
  x86_64 and aarch64 (the daemon has no `.desktop` entry); macOS (Apple Silicon Metal; Intel CPU) as
  `zip` and Homebrew; Windows `zip`, standalone `exe` and a WiX 7 installer (`-acceptEula wix7`),
  service as `NT SERVICE\ymird`; FreeBSD `pkg`/`tar.gz`, OpenBSD `tar.gz`, GhostBSD and DragonFly
  BSD after them (CPU and Vulkan; no vLLM).
- **Android and iOS: opt out for `ymird`** with the reason "a serving daemon can't run in the
  background on mobile (iOS forbids it; Android needs a foreground service and thermals make it
  impractical)". Phones are clients through Bragi.
- **User-level by default** (systemd user unit with lingering, launchd agent, Windows logon task);
  `--system` for headless servers where it is genuinely system-level. See
  [Running as a service](#running-as-a-service).
- Prebuilt binaries only; installing never needs Rust. Release files are signed with minisign.
- **llama.cpp is bundled** in every release file, in two variants (`ymir`: CPU and Vulkan, Metal on
  macOS; `ymir-cuda`: CUDA 13 and its runtime). See [The bundled engine](#the-bundled-engine-decided-2026-11-20).

### Running as a service

`ymirctl service install|start|stop|status|uninstall` (and `setup --service`) run `ymird` in the
background. `ymir_core::service` builds each action as a plan (files to write, commands to run) for an
explicit OS, so the unit, plist, task XML and command lines are unit-tested on every OS, and CI runs a
real install, chat, stop, start and uninstall on macOS and Windows.

| | User level (default) | `--system` |
|---|---|---|
| Linux | systemd user unit, `Restart=on-failure`; lingering keeps it up after logout | system unit with `User=` (`--run-as`, never root), node folder `/var/lib/ymir` |
| macOS | launchd agent in `gui/<uid>`, `RunAtLoad`, `KeepAlive` on failure, log in `state/logs/ymird.log` | launchd daemon with `UserName`, node folder `/Library/Application Support/Ymir` |
| Windows | Task Scheduler task `\Ymir\ymird`: logon trigger for this user, `conhost.exe --headless` (no console window), no time limit, restart on failure, runs on battery, `--log-file` | service `ymird` as `NT SERVICE\ymird` (SID type unrestricted), auto start, restart after 60 s (also on an error exit), node folder `%ProgramData%\Ymir` with a protected DACL for SYSTEM, Administrators and the service account |

How `ymird` stops cleanly, engines included: SIGTERM (systemd, launchd); the service control
manager's stop, shutdown and preshutdown (`ymird --windows-service`, through the `windows-service`
crate); console close, logoff and shutdown events; and a `shutdown` request on the local control
transport, which `ymirctl service stop` sends to a Windows task before `schtasks /End` (a task can
only be ended hard). On Windows the control transport is a local-only named pipe,
`\\.\pipe\ymird-<FNV-1a of the socket path>`, whose default DACL lets only its owner, SYSTEM and
Administrators write; `ymirctl` must use the same `--home`, elevated for the Windows service.

The WiX 7 MSI (`packaging/windows/ymir.wxs`, built by `make-installer.ps1` from the release zip) is
per machine: Program Files\Ymir, the system `PATH`, the `ymird` service registered for manual start
(it needs a `node.toml` first; `ymirctl service install --system` then sets it to start at boot) and
`%ProgramData%\Ymir` with the ACL above, kept on uninstall. An upgrade restarts the service if it was
running. `YmirSetup-VERSION.exe` is a WiX bundle that installs the MSI for the machine's architecture.
User-level Windows installs use the zip and `ymirctl setup --service`. Not yet: code signing for the
installers and binaries, and `--managed-updates` for the Windows service.

### Updates (decided 2026-11-19: both)

- **Package managers are the primary path** (TAUR/pacman, apt, dnf, Nix, Homebrew, winget).
- **`ymird` can also update itself** from signed releases through `weirdnet-update`: org root pinned at
  build time, signed manifests, anti-rollback state, and the boot-health state machine, so an update
  that doesn't come up healthy rolls back to the previous version automatically. Each release file is
  also minisign-signed and checked before install.
- **Off when a package manager installed it:** the package marks the install as managed, and
  self-update is then disabled automatically unless the user opts in (`ymirctl update --self on`),
  so the package database and the binary never disagree silently.
- Updates are never forced (AGENTS.md): self-update defaults to "ask" for each release; automatic
  install is opt-in. On a multi-node pool, `ymirctl update --pool` rolls model nodes one at a time,
  draining each first.
- `install.sh` detects OS and arch, downloads and verifies the signed archive, installs to
  `~/.local/bin`, then runs `ymirctl setup` (skippable).
- Backends are fetched by `weirdnet-backends` with consent (vLLM's venv is large).

### Releases

- The shared release script: `scripts/weirdnet_release.py` vendored from `weirdnet-project-helpers`,
  configured in `release.toml` with the `[[platforms]]` matrix (including the mobile `opt_out`s and
  the BSD rows once the script supports their os values).
- Every public crate (`ymir-core`, `ymird`, `ymirctl`, and the `weirdnet-*` crates in their own repos)
  is published to crates.io on each release that affects it. Apps track Weirdnet main during
  development and pin published versions for releases (`[[requires]]` for lockstep crates).
- After the GitHub release: `weirdnet_release.py assets`, then the TAUR recipe with
  `weirdnet_release.py taur X.Y.Z --dry-run` and then for real (as Morph does: GitHub, TAUR and Nix
  channels on Linux).
- Repos are public, licensed BSD-3-Clause (`LICENSE`, same text and holder as Morph).
- **Coordinated Weirdnet releases** (Weirdnet lead), in order:
  1. `weirdnet-device-link` and `weirdnet-relay` on `weirdnet-net` 0.8, then redeploy the public relay
     (it must answer `/weirdnet/names/2`). Gates M3, M4 and B1. Done: `weirdnet-device-link` 0.3.0 and `weirdnet-relay` 0.6.0, both on `weirdnet-net =0.8.0`, and the public relay runs 0.6.0.
  2. `weirdnet-agent` 0.4.0 (`infer/2`, `pool`, `context`, caller identity), released with
     `weirdnet-model` 0.4.0 and `weirdnet-mcp` 0.5.0, and Morph moves with them. Done: all three
     are out, and Morph main pins `weirdnet-agent =0.4.0`, `weirdnet-model =0.4.0` and
     `weirdnet-mcp =0.5.0` (since `6206257`; it serves `infer/2` beside `infer/1`).
     `weirdnet-model` 0.4.1 and `weirdnet-agent` 0.4.1 followed.
  3. The new crates (`weirdnet-backends`, `-openai`, `-semcache`, `-alerts`) as each milestone needs them.

## Migration plan for server-a/server-b

Target layout (topology 3, with the option of topology 4):

| Device | Roles |
|---|---|
| server-a | model + inference (scheduler-eligible) |
| server-b | model (optionally also inference later, for a redundant coordinator) |
| the dev box | app: `ymirctl bridge` for Jan and scripts, Bragi desktop |
| m3talsmith's phone | app: Bragi |

Nothing here runs until m3talsmith approves each step. Every step is reversible: the current units
are stopped, not removed, until the final step.

1. **Shadow:** install `ymird` on server-a and server-b with all roles off, link both and the dev box to m3talsmith's
   identity, import `config.normal.yaml` and big-mode into a pool policy and the `[[model.local]]`
   lists, and diff them against the live configs.
2. **server-b as a model node:** turn on `model` on server-b; its supervisor manages the 27B NVFP4 vLLM and the
   embedding model while llama-swap is stopped there (rollback: start llama-swap).
3. **server-a as model + inference:** turn on `model` and `inference` on server-a. server-a's supervisor takes over the
   llama.cpp models and the stand-in; llama-swap stops. server-a's inference node now routes to both model
   nodes and holds the scheduler lease. During this step Caddy keeps serving existing HTTPS clients
   through a loopback bridge on server-a (an app role on server-a, temporary), only until step 4 finishes.
4. **the dev box as an app node:** `ymirctl bridge` on the dev box; Jan points at `127.0.0.1:8484`.
   Bragi on the dev box and the phone once B1 ships. The `ci` key in Caddy loses access when Caddy goes
   (CI isn't part of the pool, decided 2026-11-19); if CI still needs a model, it runs one itself.
5. **Cache:** switch to `weirdnet-semcache` on server-a's inference node (start cold; RedisVL entries aren't
   migrated). Redis can stay as its backend if wanted.
6. **Big mode:** replace `/opt/llm/bin/big-mode` with the group job over the same ConnectX-7 link
   and NCCL settings, decided by server-a's inference node. Verify with the existing `e2e-big.sh` prompts.
7. **Alerts:** move the Wi-Fi fallback and other alerts to `weirdnet-alerts` on server-a's inference node,
   keep the host-ops watchdog, retire the old monitor timer once peer watching proves itself for a week.
8. **Optional redundancy:** turn on `inference` on server-b too, so the pool keeps coordinating if server-a goes
   down (server-b's models stay reachable).
9. **Retire:** after a quiet week, disable Caddy, rvl-cache and llama-swap units, and remove
   them only with m3talsmith's explicit approval.

## Client app (Bragi)

A Jan-like chat app with Weirdnet built in. It is an **app node**: it talks to inference nodes
through the `weirdnet-agent` pool, never to model nodes, and never needs the bridge.

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
- Everything they share is generic and lives in `weirdnet-*` crates (the `weirdnet-agent` pool and
  context modules, `weirdnet-sync`), so neither repo depends on the other. Bragi works with any
  inference node that serves the `weirdnet-agent` catalog and infer protocols, not only Ymir's.
- It matches Morph and Muninn: one product per repo, with a Rust core and thin shells.

### What it does (identical on desktop and phone)

1. **Finds models:** the `weirdnet-agent` pool finds the inference nodes among m3talsmith's linked
   devices and subscribes to the pool catalog, shown as **one model picker** with live status (warm,
   loadable, loading, busy) and estimated speed.
2. **Sends prompts over Weirdnet** to an inference node (the local one first), which picks the
   model node, with streaming tokens, a stop
   button (cancel resets the stream), retry before the first token, and automatic mid-answer
   failover: if a node dies, the answer continues on another node with no duplicate text (see
   [Mid-answer failover](#mid-answer-failover-decided-2026-11-19-automatic)), with a small
   "continued on server-b" note.
3. **Optional extra, same on both: offline on-device model.** If the user turns it on, with no
   reachable node `bragi-core` runs a small model in-process through `weirdnet-model` (`llama`;
   Metal on macOS and iOS, CUDA, Vulkan or CPU elsewhere), downloaded only with consent. It is off by
   default on every platform and shows in the same picker as "On this device". It is not how the
   app normally works: the normal path is always the cloud.
4. **History is the synced context store:** Bragi has no history database of its own. It reads and
   writes conversations through the `weirdnet-agent` `context` module (see
   [Context sync](#context-sync)), replicated with `weirdnet-sync` across all of m3talsmith's devices,
   nodes included, encrypted in transit and at rest. A chat can continue on any device or node.
5. **Pairing:** the first launch links the device to m3talsmith's Weirdnet identity
   (`weirdnet-device-link`, the QR and SAS flow), which puts it in the roster. That is the only way
   in; there are no invites.
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
                 └── weirdnet-agent (pool, context), weirdnet-sync, weirdnet-storage, weirdnet-model
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
  events): `ChatEvent` is `Queued{position}`, `Started{node, model, standin}`,
  `Delta{offset, text}`, `Reasoning{offset, text}`, `ToolCall{...}`, `Resumed{node, offset}`,
  `Usage{...}`, `Done`, or `Error{kind, partial}`, mirroring the `infer/2` frames. Rust coalesces deltas every ~16 ms or 32 tokens
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

Gate: B1 needs `weirdnet-device-link` and `weirdnet-relay` on `weirdnet-net` 0.8 and the redeployed
public relay, like M3 and M4. Approved 2026-11-19; released: `weirdnet-device-link` 0.3.0 and `weirdnet-relay` 0.6.0, both on `weirdnet-net =0.8.0`, and the public relay runs 0.6.0.

- **B1 Core and both shells** (built 2026-11-19; see [B1 status](#b1-status-built-2026-11-19)): `bragi-core` + `bragi-desktop` (Linux) + `bragi-bridge` and the
  Flutter app (Android): pair, catalog from one node, streaming chat with cancel, local history.
  Tests over `weirdnet-net` in-memory transports with a fake `InferHost`.
- **B2 Pool client** (built 2026-11-19; see [B2 status](#b2-status-built-2026-11-19)): multi-node catalog, routing via
  the `weirdnet-agent` pool, retries, automatic mid-answer failover, stand-in and cache flags, in both shells. A rollout (P4b) is one row with its versions under it.
- **B3 Platforms** (in progress 2026-11-19; see [B3 status](#b3-status-in-progress-2026-11-19)): iOS (TestFlight), macOS and Windows; Android apk, aab and F-Droid packaging.
- **B4 Sync and optional extras:** history on the shared context store (after M5), and a per-chat training consent
  (the `{conversation}/train` record of training plan P2, built in every shell 2026-11-24); push wake-ups (Bragi side
  built 2026-11-26, [Bragi v0.15.0](https://github.com/WeirdBrains/bragi/releases/tag/v0.15.0), see [B4 status](#b4-status-in-progress-2026-11-19));
  the optional on-device model, on both shells.
- **B5 Breadth** (in progress; see [B5 status](#b5-status-in-progress-2026-11-20)): attachments (text, image and audio) and agent mode are built on both shells. The BSD desktop window does not build yet, so on BSD it is the terminal app only.
- **B6 Your own endpoints** (built 2026-11-21; see [B6 status](#b6-status-built-2026-11-21)): any OpenAI-compatible server, with an optional API key, next to the Weirdnet pool or as a fallback. Built in the core, the Flutter app and the GPUI desktop shell.
- **B7 Models on your pool** (in progress; see [B7 status](#b7-status-in-progress)): a Models screen to search, pull and remove models on your Ymir nodes, and (desktop, Bragi 0.7.0) upload one from this device. Built in both shells and tried against demo nodes and a mock Hub. Next: the upload dialog in the Flutter app, rollouts from Bragi, and a check against a real Ymir node.
- **v0.2 Chat, code and the desktop link** (shipped in Bragi v0.2.1): chat and code as modes of one conversation, a desktop linker with QR pairing over Weirdnet so the phone's code mode works against a paired desktop, project scaffolding, a shell-out command, prompt history, a notification log, mouse support in the terminal app, and thinking blocks that stay collapsed and can be expanded.
- **v0.4 History bar and more than one pool** (shipped in Bragi v0.4.0): the history sidebar can be hidden and the choice is remembered, in the terminal app, the desktop window and a wide Flutter layout; a device can be linked to more than one Ymir pool and switch which one is active, each with its own history.

## Milestones (Ymir)

Each milestone lands its `weirdnet-*` crate work first (Weirdnet lead), published, then the Ymir side.

**Dependencies on Weirdnet work:**
- **Gate for M3, M4 and B1 (approved 2026-11-19):** `weirdnet-device-link` and `weirdnet-relay`
  released on `weirdnet-net` 0.8 and the public relay redeployed (net 0.8 uses `/weirdnet/names/2`).
  Now released: `weirdnet-device-link` 0.3.0 and `weirdnet-relay` 0.6.0, both on `weirdnet-net =0.8.0`, and the public relay runs 0.6.0.
- **Gate for M3:** `weirdnet-agent` 0.4.0 with `/weirdnet-agent/infer/2`, released with
  `weirdnet-model` 0.4.0, `weirdnet-mcp` 0.5.0 and Morph (see Releases). Done: Morph main is on
  agent `=0.4.0`, model `=0.4.0` and mcp `=0.5.0` (`6206257`).
- **Gate for M5:** a `weirdnet-sync` design for local eviction and on-demand working sets; and, if
  option B is chosen, sync running on an app's `Node`.

- **M0 Design (this doc).** Repo, names, decisions and roles done (2026-11-19); Weirdnet lead review.
- **M1 Single box, all three roles in one `ymird`** (built 2026-11-19; see
  [M1 status](#m1-status-built-2026-11-19)). `weirdnet-model` `hw` + sizing; `weirdnet-backends`
  llama.cpp adapter and supervisor (model role); a first coordinator with catalog and routing to the
  local model node (inference role); `weirdnet-openai` loopback bridge (app role); the in-process fast
  path between them; `ymirctl setup` and the quickstart on a Spark, an NVIDIA PC, a Mac and a CPU box;
  embedded `weirdnet-semcache`.
- **M2 Model-node parity with server-a/server-b** (built 2026-11-19; see [M2 status](#m2-status-built-2026-11-19)). vLLM adapter (CUDA 13 on sm_121), wrapping a stock server,
  swap groups and idle unload, stand-in cold start, `weirdnet-alerts` (push, SMTP, network-degraded
  rule), packages and the release script.
- **Order after M2 (approved 2026-11-19):** M3, then Bragi's B1, then Ymir and Bragi milestones
  alternate.
- **M3 Roles over Weirdnet** (built 2026-11-19; see [M3 status](#m3-status-built-2026-11-19)). The same protocols across devices: app → inference
  (`weirdnet-agent` pool, catalog, infer extensions) and inference → model (`/weirdnet-backends/node/1`,
  raw infer); `Roster::check_caller`; `ymirctl` and Jan on another machine through its own bridge.
- **M4 Several model nodes** (built 2026-11-19; see [M4 status](#m4-status-built-2026-11-19)). Linking, capability records, the merged
  catalog, failover before the first token, drain.
- **M5 Context sync and failover** (built 2026-11-19 with gaps; see [M5 status](#m5-status-built-2026-11-19-with-gaps); eviction waits on the `weirdnet-sync` design). `weirdnet-agent`
  `context` over `weirdnet-sync` 0.5: messages written at completion, job checkpoints, system prompts, attachments as content roots, retrieved
  context, prefix hints; inference-node working sets and pull-on-demand; encryption at rest;
  retention and tombstones; bridge ephemeral contexts. Mid-answer resume at both levels. Exit test:
  kill a model node mid-answer, then an inference node mid-answer; each reply continues with the full
  context and no duplicate text.
- **M6 Workload spreading (core)** (built 2026-11-19, partly; see [M6 status](#m6-status-built-2026-11-19-partly)). Built: load-balanced routing across replicas,
  device fairness and priority classes, batch job fan-out, and the pool benchmark; several inference
  nodes active-active. Moving a model between nodes, and placement with replica counts under the scheduler lease, are built
  (`ymirctl move`, `[inference.placement]`, 2026-11-26). A semantic answer cache on each inference node is built (`[inference.cache] embedding_model`, `weirdnet-semcache` 0.2.2, 2026-11-26). Still to do: a published placement plan with fencing, cloud-wide swap, prefix warming, and the pool-mode (shared) semantic cache. Exit
  test: two inference nodes, three model nodes, two replicas of one model, mixed interactive and batch
  load from two app devices; interactive p95 wait stays under target and neither device starves.
- **M7 Split models** (in progress 2026-11-20; see [M7 status](#m7-status-built-2026-11-20-llamacpp-rpc-only)). llama.cpp RPC groups are built (opt-in pool policy, private-link-only planning, group up and down, teardown when a member is lost). Still to do: vLLM with Ray groups, and the big-mode reproduction on server-a/server-b.
- **M8 server-a/server-b migration** (done 2026-11-20; each step approved by m3talsmith).
- **M9 Breadth and hardening** (in progress; see [M9 status](#m9-status-in-progress-2026-11-20)). Built: self-update via `weirdnet-update` with rollback; agent tasks; the Windows service (2026-11-20, see [Running as a service](#running-as-a-service)); a security review of the roster check, role boundaries and backend ports; Inkling 2 Mini; automatic context sizing (fit fixes in 0.3.0 and 0.3.1); MLX mock end-to-end tests; outside MCP servers for tasks; and a bundle fetcher (`update fetch`, `update watch`). Still to do: push wake-ups (once the `weirdnet-push` front end exists), BSD packages, and MLX-LM for Inkling 2 Mini once mlx-lm#2086 is in a release.
- **M10 Cert-only server link** (built 2026-11-24; see [M10 status](#m10-status-built-2026-11-24)).
  `weirdnet-sync` 0.7 and `weirdnet-device-link` 0.4 give service nodes a node certificate and sync keys
  without the root seed; `ymirctl pool cert-only` moves an existing node and wipes its seed copy; a node
  can be revoked without rotating the root identity. Bragi links a server cert-only with
  `bragi-tui link CODE --server inference|model` (Bragi v0.5.0).

## M1 status (built 2026-11-19)

**Works, on one box, in one `ymird`:**

- `ymirctl hw` and `ymirctl setup`: detect the hardware, compute the budget, propose a main model, a
  stand-in and an embedding model (or serve your own GGUFs with `--model`), find `llama-server`, and
  write `node.toml` with all three roles on plus a bridge token. `--download` fetches catalog models
  through `weirdnet-model`'s consent gate (asks first; checks size and SHA-256).
- Model role: the supervisor starts `llama-server` on a random loopback port with a per-process API
  key passed by environment, waits for `/health`, streams answers, drains and stops it (SIGTERM to
  its process group, SIGKILL after 10 s), enforces swap groups, notices crashes, and keeps per-model
  logs in `state/logs/`. Engines die with `ymird` (`PR_SET_PDEATHSIG` on Linux).
- Inference role: the merged catalog, routing to the least-loaded ready node, cold-start load on
  first use, one reload and retry if the engine died before the first token, the embedded cache tee.
- App role: the loopback bridge (`/v1/models`, `/v1/chat/completions` streaming and plain,
  `/metrics`), and `ymirctl chat`, which goes through the bridge like any OpenAI client.
- Admin: `ymirctl status`, `load`, `unload`, `models`, `bridge token [--rotate]` over the control
  socket (`state/ymird.sock`, 0600 in a 0700 directory).
- Tests: unit tests per crate; `ymir-testkit/tests/e2e.rs` runs setup → `ymird` → chat, swap
  groups, unload/load, crash recovery, the cache and the bridge guards against a mock llama-server;
  `tests/real_llama.rs` (ignored by default) runs the same path with a real `llama-server` and GGUF
  (passes on the dev box with the b12532 CPU build and SmolLM2-135M).

**How M1 differs from the milestone text above:**

- The generic pieces are interim crates in this repo (table in [Crates](#interim-crates-in-this-repo-to-be-extracted)),
  not `weirdnet-*` releases, by m3talsmith's rule for this build.
- No Weirdnet networking yet: roles talk only through the in-process fast path (`Pool`,
  `ModelNode`), which is the M3 boundary.
- The quickstart ran on one NVIDIA PC (the dev box, NVIDIA GPU, with a CPU llama.cpp build). Not yet run
  on a Spark, a Mac or a CPU-only box. `ymirctl setup` does not install or start a user service; it
  prints the `ymird` command.
- The semantic cache is per inference node and in memory (see [Shared semantic cache](#shared-semantic-cache)); pool mode and Redis are not built.

**Stubbed or not yet built:**

- Bridge token in `state/bridge.token` (0600) instead of the OS keyring.
- `llamacpp = "managed"` has no pinned engine downloads yet; it uses `<home>/engines/llama.cpp/`
  or `PATH`.
- vLLM adapter: launch arguments and probes only; not run (no vLLM install was available).
- Hardware: `nvidia-smi` instead of NVML; no Metal working-set query, Vulkan device query or
  interconnect report.
- No prefix-hash mapping onto synced contexts (M5).
- Idle TTL, restart with backoff, stand-in cold start, alerts and push (M2 and later).
- Windows: the control socket and process-group handling are Unix only.

## M2 status (built 2026-11-19)

**Works, on one box, in one `ymird`:**

- **vLLM adapter**, managed and wrapped. Managed: the supervisor runs `vllm serve` on a random
  loopback port with a per-process API key, waits for `/health`, streams, drains and stops it. Wrapped:
  `wrap = "http://127.0.0.1:8000/v1"` attaches to a stock server Ymir didn't start (vLLM,
  llama-server or anything OpenAI-compatible), loopback only, with an optional `wrap_api_key`
  (`env:`/`file:` reference). Unload only detaches; Ymir never stops a wrapped server.
  Tested on an NVIDIA GPU with vLLM 0.34.0 (torch 2.14.0+cu130) in a venv under
  a scratch directory, `Qwen/Qwen2.5-0.5B-Instruct`, `--gpu-memory-utilization 0.5`: managed launch ready in
  72.8 s, **176.6 tok/s** (256 tokens) and **183.1 tok/s** (512) streamed through the bridge; wrapped
  stock server **172.2 tok/s**; GPU memory freed after unload.
- **Swap groups and idle unload.** `idle_ttl = "10m"` per model; a reaper (every 2 s) unloads models
  idle past their TTL with nothing in flight. Swap groups from M1 are unchanged.
- **Crash handling.** A crash while serving restarts with backoff (1 s doubling to 60 s); five crashes
  in ten minutes is a crash loop and the model stays down until an operator `ymirctl load` (forced,
  clears the history). A load waits out a pending backoff instead of racing it.
- **Stand-in cold start.** `standin = "<model>"`: when the requested model isn't warm and its
  estimated load time (last measured load, else size/1 GB/s + 3 s) is over `standin_after` (10 s), the
  stand-in answers now while the big model loads in the background. Replies are flagged:
  `x-weirdnet-standin: <id>` header, `weirdnet.standin` in the body, `model` set to the stand-in, and
  `weirdnet_bridge_standin_total` in `/metrics`. `x-wait: true` (or `ymirctl chat --wait`) waits for
  the real model instead. The stand-in is skipped when loading would evict it (same swap group on the
  same node). `standin_warm = true` loads it at start.
- **`/v1/completions`** (plain and streaming `text_completion`) and **`/v1/embeddings`** (default
  model: the first embedding model in the catalog). A chat GGUF used for embeddings needs
  `args = ["--pooling", "mean"]` on llama-server (it refuses with pooling `none`); this isn't forced,
  so real embedding models keep their own pooling.
- **Alerts** (`ymir-alerts`, on `weirdnet-rules` 0.1.0 specs): the server-a/server-b rule pack (engine crash
  loop, load failure, crash notices; network degraded when the default route is on Wi-Fi, no route;
  disk free below 10 %; GPU at 85 °C or more), with recovery notices and an hourly repeat.
  Sinks: log, SMTP (lettre; STARTTLS, TLS, or plain to loopback; password as `env:`/`file:`), webhook
  (https, or http to loopback) and push (off: logs "would push" until `weirdnet-push` is embedded).
  `dry_run = true` formats and logs without sending. `ymirctl status` lists failing alerts. All
  tests use dry-run and in-memory sinks; nothing was ever sent.
- **User service.** `ymirctl service install|uninstall|status` and `ymirctl setup --service` write a
  systemd user unit (Linux) or a launchd agent (macOS, `com.weirdbrains.ymird`), with `--dry-run`,
  `--no-start` and `--unit-dir`. `systemd-analyze --user verify` passes on the generated unit; nothing
  was enabled on the dev box. Linger isn't changed; `status` says whether it's on.
- **Packages** (`scripts/package.py`, built locally, not released): `ymir-0.1.0-{x86_64,aarch64}-unknown-linux-gnu.{tar.gz,zip}`,
  `ymir_0.1.0-1_{amd64,arm64}.deb` and `ymir-bin-0.1.0-1-{x86_64,aarch64}.pkg.tar.zst` from the
  TAUR recipe `packaging/arch/ymir-bin`, plus `SHA256SUMS`. Checked with `--version` (aarch64 under
  qemu), `dpkg` metadata, `pacman -Qip` and namcap. `release.toml` declares the platform matrix for
  the shared release script (vendored `weirdnet-project-helpers` 0.3.2); `validate` passes and
  `taur-check 0.1.0` says "ready for TAUR". `ymird` and `ymirctl` also `cargo check` for
  `x86_64-unknown-freebsd`.
- Tests: 64 pass, 3 ignored (fmt and clippy clean). `ymir-testkit/tests/m2.rs` (7) covers the
  stand-in, completions/embeddings/idle unload, wrapping a stock server, crash backoff and crash loop,
  wrap refusals, deduped alerts and the user unit; `tests/real_vllm.rs` (ignored) runs managed and
  wrapped vLLM for real.

**Not done, stubbed or untested:**

- **vLLM on GB10 (CUDA 13, sm_121) is untested.** server-a/server-b are live and weren't touched; only an
  older NVIDIA GPU was tested.
- Packages: no `.rpm` (no `rpmbuild` or `nfpm` on the build box), no minisign signatures, the Nix
  flake and the Homebrew formula are written but untested, no macOS or Windows builds. Windows can't
  build yet: the control socket is a Unix socket. Linux binaries built on Arch need glibc 2.39, so
  Debian 12 and Ubuntu 22.04 need an older-glibc release build.
- Push is a stub sink; secrets from the OS keyring (`keyring:`) aren't supported yet.
- The webhook's TLS path (hyper-rustls with ring) is built but only the loopback http path is tested.
- Big-mode and Ray/RPC groups are M7; GPU temperature comes from `nvidia-smi` only.

**`ymir-backends` M2 changes are done** and ready for the Weirdnet lead to extract into
`weirdnet-backends`. Also generic and to be extracted later: `ymir-core`'s `service` module (unit/agent install) if another app wants it.

## M3 status (built 2026-11-19)

Built on the published crates, unchanged, now on their patch releases (caret ranges, no `=` pins):
`weirdnet-agent` 0.4.1 (`infer/2`, `catalog/1`, `pool`, `RosterAuth`, raw prompts), `weirdnet-model`
0.4.2 (`hw`), `weirdnet-backends` 0.1.1 (`node` feature), `weirdnet-openai` 0.1.1,
`weirdnet-semcache` 0.1.1, `weirdnet-alerts` 0.1.0 (`backends` feature), `weirdnet-net` 0.8,
`weirdnet-sync` 0.5.1 and `weirdnet-device-link` 0.3.0. The interim crates `ymir-hw`, `ymir-backends`, `ymir-openai`,
`ymir-semcache`, `ymir-infer` and `ymir-alerts` are gone (alerts subjects keep `[ymir <host>]` through
`.with_product("ymir")`; push is behind weirdnet-alerts' `push` feature, which Ymir doesn't enable).

**Works:**

- **One request type end to end.** The bridge parses OpenAI JSON into agent's `v2::Request`
  (`weirdnet_openai::request::chat`); the pool, the inference role and model nodes all speak it, and
  answers stream back as `Chunk`s. The local box uses the same path in process (`host::local_pool`, a
  memory transport), so local and remote differ only in the transport.
- **Inference role over Weirdnet** (`remote::serve`): `infer/2` and `catalog/1` (plus `infer/1` for
  older agents) for the user's devices, with `RosterAuth` (`check_caller`): another user's node, or a
  device with a valid cert that isn't in this roster, is refused. Admission is `[network] slots` and
  `queue` (4 and 32); Busy answers `Retry-After: 1`. The catalog is fresh on every request and
  republished when it changes.
- **App role on another machine** (`remote::connect_pool`): an app-only `ymird` reaches the
  inference nodes in its roster through agent's `Pool` (failover between inference nodes before the
  first token), and serves its own loopback bridge, so `ymirctl` and Jan work there unchanged.
- **Model role on another machine**: a model-only `ymird` serves `weirdnet-backends`' node protocol
  (`NodeServer`, `/weirdnet-backends/node/1`, signed capability records, admission, hardware and
  budget). Inference nodes find model nodes in the roster that announce `weirdnet-backends-node`,
  follow their capability records, and route, load and unload through them (`RemoteModelNode`;
  `ymirctl load` on the inference node loads on the remote box). Model nodes that lapse or are
  revoked leave the catalog.
- **Identity and roster** (`membership`, under `state/pool/`, 0700, secrets 0600): `ymirctl pool link`
  shows a `weirdnet-ymir://link/…` code and takes the identity after the 6-digit check; `pool add
  <code>` links another device from here; `pool devices`, `pool revoke <peer>`, and `pool init
  --standalone` for a cloud of its own. With `ymird` running, add and revoke go through its control
  socket. `ymirctl status` shows the Weirdnet identity, listeners and linked devices. Roster changes
  spread by `weirdnet-sync` passes every 30 s; a device linked elsewhere is answered after the next
  passes, and revocations stick.
- **`[network]`** in node.toml: `enabled`, `port` (4848), `listen`, `mdns`, `discovery`, `relays`
  (default: the [Weird Brains](https://weirdbrains.com) relay), `peers` (`/ip4/…/tcp/…/p2p/PEER`; dialing grants nothing),
  `slots`, `queue`. Weirdnet starts only when the node has an identity.
- Tests (M3 count; M4 below): 32 pass, 3 ignored (the 9 `ymir-alerts` tests now live in `weirdnet-alerts`); fmt and clippy (`-D warnings`) are clean.
  - `ymir-core/tests/remote.rs` (5): several devices in one process on the memory transport. They
    cover chat with tools, unknown models, cache hits, the catalog, embeddings and revocation;
    this box's in-process pool; strangers refused (wrong user, unknown device); failover between
    inference nodes; and roster changes reaching other devices.
  - `ymir-testkit/tests/m3.rs` (3): a combined node (model and inference roles on one box) serving
    both, with a second inference node using its model role; and three `ymird`s over TCP on 127.0.0.1 (model-only,
    inference-only, app-only; no mDNS, discovery or relays). The app bridge goes over Weirdnet to
    inference, then over Weirdnet to the model node and the mock engine. It also covers a remote
    `ymirctl load`, the cache across nodes, `ymirctl chat`, refusing a stranger's identity, and
    cutting off a revoked device. `pool link` and `pool add` move an identity between two homes
    through device-link on loopback.

**M3 gaps closed by the patch releases (agent 0.4.1, backends/openai/semcache 0.1.1):**
`/v1/completions` works again (agent's raw `Input::Prompt`; the 501 and its tests are gone, the M2
completions tests are back), streamed answers carry `x-weirdnet-cache` early, embedding usage is
real, and an embeddings `BadRequest` is a 400 again (client errors pass through the pool without
failover; a stranger's catalog request now fails with `Refused("wrong-user")`, not `Unavailable`).
A box with both the inference and model roles now serves both: `NodeServer::start_with(...
.with_legacy_infer2(false))` leaves `infer/2` to the inference role, and the inference node's
pool of model nodes dials through `RawInferDialer` (raw inference on
`/weirdnet-backends/infer/1`, falling back to `infer/2` for nodes on backends 0.1.0). The
membership sync pass is `weirdnet_sync::pull_types_shared` (0.8.5: snapshot, every type due in one session, `merge_back`).

**Open gaps (upstream, parked for M9): `weirdnet-openai` #1 and #2, `weirdnet-agent` #1 and
`weirdnet-backends` #1.**

- **Content arrays are text-only** (image parts are refused by the request parser), and unknown
  OpenAI parameters are dropped.
- **Streamed answers report the stand-in in the final chunk**; `BridgeClient` reads it there.
- `Busy` always answers `Retry-After: 1`; Windows kills the engine outright; vLLM is Linux only.
- **The exact answer cache is for temperature 0 only** (no tools, inline messages), per
  `weirdnet-semcache`. With `[inference.cache] embedding_model` set, a semantic cache behind it answers
  single-turn prompts that mean the same (see [Shared semantic cache](#shared-semantic-cache)); other
  requests always go to a model. Since `weirdnet-agent` 0.5.1 a request can
  also skip it (`cache: bypass`) or refresh it (`cache: refresh`), and entries are keyed on the weights
  revision where a node reports one.
- No `GroupWorkers` in M3 (group jobs answered Unsupported); M7 adds them for llama.cpp RPC.

**Not done or untested:**

- **Across real machines is untested.** All multi-node tests run on the dev box (memory transport
  and loopback TCP). The laptop has Rust 1.99, but its load average was 17 on 8 GB, so a build
  there would have been disruptive. server-a and server-b weren't touched.
- **Linking m3talsmith's real identity needs him**: `ymirctl pool link` on a node, then the code opened
  in Morph. Morph main now accepts `weirdnet-ymir://link/…` codes (Identity → Add device), and `pool add`
  takes Morph's `weirdnet-morph://link/` and Bragi's `weirdnet-bragi://link/` codes too (same payload; `normalize_link` rewrites the scheme, and Bragi does the same for Ymir's and Morph's codes). `bragi/scripts/ymir-link-e2e.sh` links a headless Bragi core and real `ymird` nodes both ways over loopback. The signed roster label reads
  "Ymir on <name>" (the device-link hint). Linking the real identity still waits for m3talsmith.
- Seeds are kept in 0600 files, as decision 12 says, not in the OS keyring. M10's cert-only link
  (built 2026-11-24) keeps them off servers linked that way.
- **Flaky upstream roster propagation (found on the dev box):** `ymir-core/tests/remote.rs`
  `roster_changes_reach_other_devices` fails about half the runs, on the pre-bump versions too: when
  an app device links a third device, the inference node's pass against the app can finish with no
  error and still not learn the new cert (a pass that gets `needs-admission`/`unknown-key`
  first), and four rounds don't always fix it. For the Weirdnet lead (`weirdnet-sync`). **Fixed in `weirdnet-sync` 0.5.2** (see the M5 status): 12 of 12 runs pass.

## M4 status (built 2026-11-19)

Several model nodes behind one inference role. Linking, capability records and the merged catalog
were already in place from M3 (every model node is a linked device that announces
`/weirdnet-backends/node/1`; the inference node follows each node's signed capability record with
`NodeClient`). M4 adds routing that survives a node going wrong, drain, and the commands to see it.

- **Merged catalog.** `Coordinator::merged_catalog` joins every node's models by id; a row lists
  each node and its state, and which of those are draining. `catalog/1` reports a model as warm,
  loading or loadable from nodes that take new requests only, with `replicas` the number of ready
  ones. Nodes may hold different models; one model on several nodes is one catalog entry.
- **Routing.** A ready node with the fewest requests in flight; otherwise the first node that has
  the model (it loads there). Draining nodes and nodes already tried for this request are skipped.
  A model that exists only on draining nodes answers `unavailable` ("only on nodes that are
  draining"), never a silent hang.
- **Failover before the first token.** `Coordinator::run` tries one node at a time. If the node
  fails before anything has streamed (unreachable, busy, refused, engine unavailable, stream broke
  before the first chunk, cold load failed, stalled, timed out), the request moves to the next node that has the
  model. If no other node has it and the failure was `Unavailable`, it reloads on the same node once
  (an engine that just died). Once a token is out, the error goes to the asker and the app's pool
  resumes from what it received (M5 does the same thing across inference nodes). `BadRequest`,
  `Unsupported` and cancels never move. Embeddings fail over the same way.
- **Drain.** `ModelNode::drain(on)` and `draining()`. For a remote node it sends the node's own
  `Drain` command (the node refuses new raw infer and loads, waits for requests under way, and its
  capability record says `draining`); for this box's model role it flags the local node and the
  service's `ModelHost`, so a combined node drains for other inference nodes too. `ymirctl drain
  NODE [--resume]`, run against any inference node; `ymirctl nodes` (and the "Model nodes" block of
  `ymirctl status`) lists nodes as serving or draining with ready/total models and in-flight.
- **Nodes joining and leaving.** Discovery adds devices that announce the model-node app and drops
  ones whose records lapse. New: when a node's record stream ends and it can't be reached again, its
  record is dropped at once, instead of staying until it expires (90 s), so a dead node leaves the
  list and the catalog within one discovery round (about 10 s) and the router stops trying it.
  Revoking a device removes it as before.

Tests:

- `ymir-core/tests/nodes.rs` (7, one process, scripted nodes): merged catalog with different model
  sets and routing by model; a dead node fails over before the first token (and both dead returns
  the error); a cold load that fails moves on; no move after the first token or for a bad request;
  drain skips a node, hides it from `catalog/1` and resumes; nodes join, are replaced by name and leave;
  embeddings fail over and respect drain. Remote embeddings now report real prompt tokens (the pool's
  answer carries `usage` since agent 0.4.1; Ymir was returning 0).
- `ymir-testkit/tests/m4.rs` (1, five daemons on loopback with the mock engine): two model nodes with
  different model sets (server-a: alpha; server-b: alpha, beta) behind one inference node, an app node, and a third model
  node (gamma) that joins later. Checks the merged catalog through `ymirctl models`; beta served only
  by server-b; `ymirctl drain server-a` sends every alpha request to server-b and `ymirctl nodes` shows it draining, then
  resume; server-a is shut down and alpha requests still succeed from server-b (whether or not the dead node was
  tried first), and server-a leaves `ymirctl nodes`; m3 joins and answers gamma.
  The test retries a `needs-admission` refusal for up to 100 s (observed: the inference node refuses the app for
  tens of seconds after the first requests, and it clears after a roster pass): that is the upstream roster flake
  below, which shows up here as the app or a model node not yet admitted. It retries nothing else.

For the Weirdnet lead (generic, so it belongs in `weirdnet-*`; Ymir carries its own versions until then):

- **`weirdnet-backends`: a routing/failover helper over several `NodeClient` records.** The pick
  (ready, fewest in flight, skip draining and already-tried), the retryable-error set and the
  failover-before-first-chunk loop are not Ymir-specific. A `NodePool` next to `NodeClient` would let
  any app do this.
- **`weirdnet-backends`: drop a node from a `NodeClient` follower when it can't be reached,** instead
  of holding the last record until `expires_ms`. Ymir does this in `RemoteModelNode::connect`.
- **`weirdnet-sync`: the roster flake** (above) now also shows in M4: a daemon pair's first calls
  after linking can get `needs-admission` for several seconds.

Totals after M4: 39 pass, 1 fail (the known flaky `roster_changes_reach_other_devices`), 3 ignored; fmt and
clippy (`-D warnings`) clean. M4 added 8 tests (7 in-process, 1 five-daemon).

Gaps: no capability-aware placement beyond "ready, then fewest in flight" (no hardware, speed or
warm-prefix scoring; that is M6/M7); drain on a remote node needs
that node to run the model role of this version or later (an older node answers `unsupported`); the
laptop cross-machine run did not happen (it was busy when tried).

## M5 status (built 2026-11-19, with gaps)

Answers survive a node dying mid-answer at both levels, and conversations live on the inference
nodes and sync among them. On published crates only: `weirdnet-agent` 0.4.2, `weirdnet-openai`
0.1.2, `weirdnet-sync` 0.5.2 (not 0.6; see below).

**What works:**

- **Resume after the first token, model level.** When a model node dies mid-answer, the inference node
  continues on another node that serves the model: it sends what has streamed so far as an
  `assistant_prefix` (up to twice per answer), and the asker sees one whole answer with no repeated
  text. A request with a prefix skips the answer cache, waits for its own model (no stand-in), and
  a resumed answer is never cached as whole. Over Weirdnet this goes through `Pool::infer`, which
  needs 0.4.2 (0.4.1 failed with "characters missing" on a request that carries a prefix at the
  start). A model node that goes quiet is given up on after 5 to 12 s (the asker's pool waits 15 s or more),
  so the inference node moves first.
- **Cold loads keep the asker waiting, not timing out.** A request that waits for its model to load
  (`x-wait`, or no stand-in) gets an empty `Meta` frame every 5 s while the load runs, so the asker's
  stall timer (15 to 25 s) doesn't give up on a vLLM load that takes a minute or more. Before this, such
  a request ended with a stall error after 25 s while the model went on loading.
- **Long prompts are not stalls.** A model node sends nothing while its engine processes the prompt
  (`weirdnet-backends` streams only tokens), so the inference node sends the same `Meta` keepalive
  every 5 s from the moment it routes a request until its first text or reasoning goes out (a tool
  call still being collected counts as quiet too), within an allowance that grows with the prompt
  (`coordinator::prefill_allowance`: 60 s plus the request at 3 bytes a token and 50 tokens/s, at
  most 2 h). Past it the keepalive stops and the asker's own stall timer applies again. The pool to
  a remote model node waits as long for its first frame (`remote::model_node_stall`), and 5 to 12 s
  between frames as before; a node or engine that dies still ends the stream at once. Before this,
  Inkling's 62 s on a 17.9k-token prompt on a 128 GB Spark ended with "stalled: no frame for 25000 ms", and the
  node cancelled the answer once the prompt was done. llama.cpp's `return_progress` would make the
  keepalive evidence of progress, but the request body is built in `weirdnet-backends`.
- **Resume at the inference level.** The inference host now says `resume: true`, so an app's pool whose
  inference node dies mid-answer continues on the other one, from the text it already shows. It is
  also what `continue_final_message` needs: the bridge continues the last assistant message through
  the pool, the reply holds only new text (tested through the bridge, and the 400 when
  `add_generation_prompt` is also set).
- **Conversations.** `ymir-core::context` keeps each conversation as sealed `weirdnet-sync`
  records of type `ymir.context`: a meta register (title, system prompt, model, a tombstone flag) and one
  append-only record per message. An assistant message is written when its answer completes,
  under the job's id, so a retry or a resume writes the same record: one answer, never two. Tombstones
  and age-based retention (`expire`) hide a conversation everywhere. The op log is saved to `log.json`
  (sealed values) and reloaded, so sequence numbers carry on after a restart.
- **`context_ref` requests.** The inference node assembles the prompt (system prompt, messages through
  `head`) from its store. A request that names a conversation pulls from the caller first, so the newest turn
  is there before the periodic pass; the cache is never used for these. `ymirctl context show|add|ask|delete|train|export|withdrawn`
  drives it; `ask` adds a user turn on the app node and has the pool answer it.
- **Sync.** Inference and app nodes pull `ymir.context` on every sync pass (`[network] sync_secs`, default 30);
  a model-only node doesn't carry conversations. A lone device starts its own key chain; with several,
  the first sealed write waits for the first rotation.
- **`Daemon::shutdown`** now halts the Weirdnet node even while requests hold it, so an asker sees the
  node go (this is what the kill tests rely on).

**Exit test, on separate daemons over loopback with a slow mock engine** (`ymir-testkit/tests/m5.rs`): kill a
model node mid-answer: the answer continues on the other with the exact text (5 of 5 runs). Kill an
inference node mid-answer: the app's pool resumes on the other inference node, which gets the
conversation by pull-on-demand, continues from the shown text, writes the answer once, and every
node's copy ends the same (5 of 5). A third test checks conversations syncing among two inference
nodes and the app, the answer on every copy, and a tombstone reaching all.

**Tests:** `cargo test --workspace` all pass; the previously flaky `roster_changes_reach_other_devices` passes
12 of 12 on sync 0.5.2, and the 100 s retry window in the M4 daemon test is gone.

**`weirdnet-sync` 0.5.2 (roster fix):** a device linked on one device now reaches the others in one
pass. It fixes the flake above and the `needs-admission` refusals. Ymir moved to it. Not moved
to `weirdnet-sync` 0.6 (the Identity store), which would put two sync versions in the tree.

**Gaps:**

- **Not in M5 yet:** job checkpoints per finished task, attachments as content roots, retrieved context, prefix
  hints, and the bridge's ephemeral contexts as a named feature (the bridge sends inline messages, which
  are ephemeral already). Retention is a function, not yet a scheduled setting.
- **No eviction or working sets.** `weirdnet-sync` can't drop a local copy without a tombstone, so inference
  nodes hold every conversation. This is the sync design the milestone was gated on (Weirdnet lead).
- **Generic code in Ymir.** The conversation schema, store, assembly, retention and the saved log belong in
  `weirdnet-agent`'s `context` module (and the log's persistence in `weirdnet-sync` or its storage trait);
  they are `ymir-core::context` and `membership.rs` until then, behind one interface.
- **Bragi uses the store for history since B4, but not yet for requests**, and the mock engine stands in for llama.cpp; the real-engine
  resume is the same prefix path the engines already take.
- **Cross-machine** runs are still pending (the laptop).

## M6 status (built 2026-11-19, partly)

The pieces of workload spreading that need only published crates (`weirdnet-agent` 0.4.2,
`weirdnet-backends` 0.1.1) are built. Placement, replication and the shared cache are not.

**What works:**

- **Load balancing across replicas** (`ymir-core::balance`). Among the model nodes with a model ready,
  the inference node scores each by predicted completion (`wait + service + load + penalty`, with
  `wait = queue_depth / slots * mean_service`) and takes the cheaper of two random candidates (power of
  two choices). `queue_depth` is the larger of the node's reported in-flight count and what this inference
  node has itself sent it that hasn't finished; `mean_service` is measured here per node and model (2 s
  until one request finishes); a `Busy` answer adds a penalty that fades over 5 s; a node on this device wins
  ties. Replaces "fewest in flight".
- **Device fairness** (`ymir-core::fair`). A device may have at most `[network] batch_per_device` batch or
  background requests running or waiting on an inference node (default half of `slots + queue`); past
  that the answer is `Busy`. Interactive requests are never held back, and the admission queue already
  serves them before batch work, so one device's batch can't fill the queue and leave another with `Busy`.
- **Priority classes.** `weirdnet-agent`'s admission queue (interactive, then batch, then background) is in
  use at every inference node; the bridge asks at interactive, `ymirctl batch` at batch.
- **Batch fan-out** (`ymir-core::batch`, `ymirctl batch FILE`). One prompt per line, spread over the pool at
  batch priority with a concurrency cap, per-task retries under one job id (`Busy` included, so the fan-out
  slows to what the pool takes), results in prompt order.
- **Pool benchmark** (`ymirctl bench`, `crates/ymirctl/src/bench.rs`). A client of the loopback bridge, so
  every request takes the normal pool route. It lists the pool's models, then takes them one at a time
  (a busy or loading model is re-checked, then skipped; a model that is not loaded is skipped unless
  `--include-cold`). Per model: warm-up, then N repeats of three fixed prompts (short, a 200-word story,
  a long prompt) with time to first token, decode tok/s, prompt tok/s and total latency as median and p90,
  then eight deterministic checks at temperature 0 (arithmetic, JSON format, two Python functions run
  with CPU, memory and time limits and no network where `unshare` allows; on Windows the limits come from a
  Job Object and the network is not cut off; on macOS, which refuses memory limits, they are left
  unscored; exact-match facts).
  Every request has its own seed and nonce so neither the answer cache nor a prefix cache is hit. The bridge
  cannot pin a node, so each request records the node that answered (`served_by`) and a model on several
  nodes gets a row per node. Output is a timestamped `.json` and `.md`; nodes are labelled node-a, node-b
  unless `--real-names`. Embedding models get a latency check instead.
- **Several inference nodes active-active.** Each routes on its own; the app's pool already spreads and fails
  over between them (M4/M5).

**Exit test** (`ymir-testkit/tests/m6.rs`, separate daemons over loopback, mock engine): two inference
nodes, three model nodes (two replicas of `alpha`, one `beta`), two app devices. One device fans out 24
prompts (8 at a time, `batch_per_device = 2`) while the other asks ten questions one after another.
Result: all 24 batch tasks answered in order with 0 failures, and the interactive p95 with the batch
running was within noise of the p95 without it (the test bounds it to 2 s plus twice the quiet p95).
Unit tests cover the scoring (shorter queue wins, local first on ties, own sends counted, busy
penalty, slower and cold nodes, spread over equal nodes) and the fairness cap; an in-process test shows
three replicas sharing a burst and a slow one getting the least.

**The first-token delay (diagnosed 2026-11-19, upstream).** In this topology every request's first token
arrived after 12 to 15 s, then the whole answer in 0.4 s, also on the M5 commit. It is not a stall and
not Ymir's routing: `weirdnet-agent` 0.4.2's `LinkedNodes::nodes()`, which `Pool::infer` and
`Pool::embed` call on every request before the first dial, checks each linked device in turn and, for
each one it can't identify, waits the whole `with_wait` (Ymir's `find_wait`, 3 s) even when `connect`
failed at once, with nothing cached between calls. The app can't reach model nodes or other apps
(by design), so the M6 app pays 4 × 3 s = 12 s per request (three model nodes and the other device).
Evidence: `tests/linked_nodes.rs` (ignored, measures) times `nodes()` at 9.1 s with three offline
devices and a 3 s wait, 1.58 s with a 0.5 s wait, the same on every call; in `tests/latency.rs` (one
inference node, one model node) the first token takes 3.1 s, and 0.08 to 0.11 s once the app is also
given the model node's address. Real pools hit this too: every phone or laptop that is off adds 3 s to
every answer. Reported to the Weirdnet lead; `tests/latency.rs` (first token under 1 s) is ignored until
that release, and the M6 p95 comparison is between two equally delayed runs until then.

- **Moving a model between nodes** (built 2026-11-26, `Coordinator::move_model`, `ymirctl move MODEL --to NODE`).
  The model loads on the target first (it needs its own copy of the weights; the move refuses otherwise),
  the `(source, model)` pair is taken out of routing so new requests go to the target, the source finishes
  what it reports in flight for the model (up to a grace time, 60 s by default) and then unloads it. A load
  that fails on the target leaves the source serving. Copying the weights over is not part of it (pull them
  on the target first). Tests: `crates/ymir-core/tests/move_model.rs` (fake nodes) and
  `crates/ymir-testkit/tests/m6_move.rs` (daemons over Weirdnet: a slow answer on the source finishes whole
  before it unloads; the MCP tool `ymir_move_model` does the same with `confirm: true`).
- **Placement and replication** (built 2026-11-26, `ymir_core::placement`). `[inference.placement] replicas =
  { MODEL = N }` keeps each listed model loading or ready on N model nodes that take new requests; models not
  listed are left alone. Every `every_secs` (15 s) the placing inference node takes one step per model: load
  it on the node with its own copy and the fewest loaded models (never a draining or reserved node, one where
  it failed to load, or one where it would evict another ready model of its swap group), or unload it from an
  idle extra (never one with answers in flight, never below N). Draining a replica brings one up elsewhere;
  resuming it unloads an idle extra.
  **Who places:** the holder of the `ymir.scheduler` lease (`weirdnet-sync` 0.8 majority-vote lease; voters and
  candidates are the devices whose grant has `ymir.inference`, that is cert-only inference nodes, each voting
  with its terms in `voter.cbor`). An inference node that holds the seed has no such grant, so it places only
  while no cert-only inference node exists; `scheduler_eligible = false` never places. Tests:
  `placement::tests` (the rules), `crates/ymir-testkit/tests/m6_placement.rs` (daemons: two replicas come up,
  stay at two, follow a drain and settle back) and `m6_scheduler_lease.rs` (two cert-only inference nodes:
  exactly one holds the lease and the other takes over with a higher term).

**Gaps:**

- **Not built: a published placement plan, fencing, cloud-wide swap, replication on queue wait, prefix warming.**
  The lease and the replica counts are built (above). Still to do: publishing the plan (`SyncState::publish_plan`
  under a sealed `ymir.plan` type that cert-only inference grants would need to list), model nodes refusing a
  stale holder's load or unload (`FenceGuard`; today a stale holder can at worst add or drop one replica for a
  round, which the next round undoes), several seed-holding inference nodes taking part in the lease (they would
  need an `ymir.inference` grant of their own), and warm-prefix data in the capability record. The scorer ignores prefill and decode speed, uncached prompt length, round-trip time
  and relays for the same reason.
- **Not built: the pool-mode semantic cache.** Each inference node has its own semantic cache (in memory, lost on restart); entries are not sharded or shared between inference nodes, and there is no Redis backend.
- **Batch jobs are not synced.** The submitting node runs the fan-out, so if it goes down the job stops
  (the design wants the job record synced so another node picks it up). No tasks with tools or approvals.
- **Fairness is a cap, not a weighted share.** Devices are held to the same cap; there are no per-device
  weights or quotas.
- **One computer, a mock engine,** and the delay above.

## M7 status (built 2026-11-20, llama.cpp RPC only)

A model too big for one node can be served by a group of model nodes over a private link, using
`weirdnet-backends` 0.1.1's group jobs (`GroupPlan`, `GroupWorkers`, `GroupUp`/`GroupDown`). Built and
tested on the dev box only, over loopback, with mocks and with a real model. No other machine was touched.

**What works:**

- **Opt-in by pool policy.** `[[inference.split]]` (on inference nodes) names the model, its size, the
  leader and the allowed nodes (`[[inference.split.node]]`: node name or peer id, `link`, rpc `port`,
  `memory` the group may use there, `gbps`, models to `drop` first). A model not listed is never split.
  `min_link_gbps` defaults to 10 (llama.cpp RPC wants 10 GbE or better).
- **Private links only.** Loopback, RFC 1918, link-local and ULA addresses are accepted; public addresses
  and CGNAT (100.64/10, which is what tailnets use) are refused. The config refuses a public link at load,
  the planner refuses a non-private or too-slow link, and a worker binds only its own
  `[model] private_link` and refuses a plan for any other address. Speed is declared in the policy, not
  measured; a non-loopback link with no `gbps` is refused.
- **Planner** (`ymir-core::group::plan_group`): the leader is the named node (or the largest); members
  are added largest first until memory covers the model; nodes that answer `group_jobs: false`, are
  draining, or lack a private fast link are skipped. A model that fits on the leader alone is refused as
  not needing a group. Errors say which rule failed.
- **Group lifecycle** (`Coordinator::group_up` / `group_down`, `ymirctl group up|down|status MODEL`):
  workers start first (`rpc-server -H <private ip> -p <port>`, waited on until it listens), then the
  leader runs `llama-server --rpc a:p,b:p` on loopback with a key kept in the state dir; the leader's
  node advertises the model like any other (a wrapped launch), so the pool, cache and balancer need no
  special case. A failed step rolls back what had started. `group down` unloads at the leader and stops
  every member's processes.
- **Member loss.** A watcher (every 5 s) takes a group down when a member leaves the pool, starts
  draining, or a local group process exits; the loss and its reason are kept for `ymirctl group status`
  and logged as a warning. The model then goes back to normal placement (the stand-in, if configured).
- **Leader and worker config:** `[model] private_link`, `rpc_server` (`managed`, `system:` path, `off`),
  `rpc_args`; the leader's `[[model.split]]` (`id`, `file`, `serve_port`, `context`, `slots`,
  `gpu_layers`, `args`).

**Tested:**

- Unit tests for the link rules, planner, plan and leader arguments, worker refusals, key file.
- `m7.rs`: three daemons over Weirdnet with a mock `rpc-server` and the mock engine: a slow link and an
  unlisted model are refused before anything starts; `group up` makes the worker listen and the group
  answer a chat; `group down` stops the worker; after another `up`, shutting the worker's node down takes
  the group down and `group status` shows why.
- `real_split.rs` (ignored; backs off if the GPU is busy or has under 7 GiB free): Qwen3-8B Q4_K_M, a
  GPU `llama-server` leader and a CPU `rpc-server` worker (`--tensor-split 3,1`) as two model nodes over
  loopback on an NVIDIA GPU. It came up in 20 s, answered through the pool, and ran at **8.3 tokens/s**
  against **63.1 tokens/s** for the same model unsplit on the GPU. The split is far slower because a
  quarter of the layers run on the CPU and every token crosses the RPC link; splitting is for models that
  do not fit, not for speed.

**Not done or untested:**

- **vLLM with Ray (and tensor parallel over RoCE) groups.** Not built. `RpcWorkers` answers Unsupported
  for any worker kind except `rpc-server`. They need a Ray cluster over 100 Gb/s links; the only such
  machines are server-a and server-b, which were off limits.
- **The big-mode reproduction on server-a and server-b** (a group job across the two servers, replacing the hand-run
  big mode) has not been run. It needs m3talsmith's go-ahead and a plan for the private link between them.
- **Capability records carry no link information**, only `group_jobs` and `groups`; so links, ports and
  memory are declared in Ymir's pool policy. Upstream request: let a node advertise its private links
  (address, measured speed) so the planner can discover them.
- **Link speed is declared, not measured.**
- **A crashed worker on another node is only noticed when that node leaves the pool or a request fails**
  (node/1 has no question to ask a remote node about a group); a crashed local process is noticed in 5 s.
- **Alerts for a lost group are logged and kept in status; not delivered** to an app device or channel.
- **Scheduling is local.** The group is started by the operator or by `ymirctl`; nothing starts one on
  demand or by placement, because the lease and synced plan are waiting on sync 0.7.0.
- The group's leader serves through a loopback `llama-server` the node did not start through the
  Supervisor, so engine args cannot be changed at runtime; the leader's args come from `[[model.split]]`.

## B1 status (built 2026-11-19)

Bragi lives in its own public repo, [`WeirdBrains/bragi`](https://github.com/WeirdBrains/bragi)
(BSD-3-Clause), on `weirdnet-agent` 0.4.1 (pool client and types), `weirdnet-device-link =0.3.0`,
`weirdnet-sync` 0.5.1 and `weirdnet-net =0.8.0`. Nothing is released or published to a store.

**What works, in both shells:** pair, list models, chat, stop, continue, and history.

- **`bragi-core`** holds all of it. Pairing goes through `weirdnet-device-link` with
  `weirdnet-bragi://` codes and the 6-digit check. Bragi either shows a code for an already-linked
  device, or as a linked device it opens another device's code. The catalog comes from one node:
  the `Pool` catalog watch, chat models only, warm first. Chat streams through `Pool::infer` as
  offset-addressed deltas, coalesced every 16 ms. Stop cancels the request and keeps the partial
  answer. Continue picks up a stopped or cut-off answer (see the gaps). History is local JSON, one
  file per conversation, with atomic writes. Lifecycle covers pause, resume and network changes.
  `bragi_core::demo` is an in-process inference node with a test identity: a fake `InferHost`, a
  `CatalogFeed` with two chat models and one embedding model, served by `serve::Server` over the
  memory transport. It runs the tests and both shells' demo profile.
- **`bragi-desktop`** is the GPUI shell, Linux only, frameless with in-app controls (close only
  under tiling window managers). It has a pairing card with a QR code and the check code, model
  chips, history, a "link a device" field, and streaming bubbles with Stop and Continue. `--demo`
  runs it against the demo node. `--scenario` plays the shared scenario through the view's own
  actions and exits 0 only if the continued answer is exactly the full answer.
- **`bragi-bridge`** is the flutter_rust_bridge 2.13 API, built by the Native Assets hook. The
  **Flutter app** (Android) has the same screens over a thin controller; all behaviour stays in
  the core. `BRAGI_DEMO=true` selects the demo profile.

**Tests:**

- `cargo test --workspace` passes 16 tests: `bragi-core` has 3 unit tests and 3 scenario tests
  (the full scenario with a restart into history, codes that differ link nothing, and a stranger
  refused while a linked device links another), and `bragi-desktop` has 10 (9 for the text field
  copied from Morph, plus delta placement).
- `flutter test` passes 2 tests: the scenario through the real bridge and core, and delta
  placement.
- `cargo clippy --workspace --all-targets -D warnings` and `flutter analyze` are clean.
- Both shells were also played end to end in virtual workspaces only. The desktop ran in Xvfb with
  `env -i` and a private `XDG_RUNTIME_DIR` (`scripts/desktop-scenario.sh`). Android ran on a
  windowless emulator with a throwaway AVD, a private adb port and `-wipe-data`
  (`scripts/android-scenario.sh`). Each saved 8 screenshots, and both passed.

**Builds (the dev box):**

- The desktop debug binary builds (it isn't packaged yet).
- The demo APK (debug, x86_64, 57 MB) builds and runs on the emulator.
- The release APK (arm64, real network, debug-key signed, 13 MB) was built earlier on agent 0.4.0 and not
  rebuilt after the 0.4.1 bump; the demo debug APK was rebuilt on 0.4.1 and passed the emulator scenario.

**Gaps (for the Weirdnet lead where generic):**

- **`Pool::infer` can't continue from a caller-supplied prefix.** It splices node offsets onto an
  empty buffer, so a prefix gives a gap error, or `NoResume` from a node that doesn't resume.
  Continue therefore sends the partial answer plus an instruction turn ("continue from exactly
  where it stopped") and keeps the offsets running on the client. That's exact on the demo node,
  but a real model may rephrase the seam. Asked for: a prefix-aware `Pool::infer`, or a
  `Prefix` passthrough.
- **Copies of generic code:**
  - the identity store (`bragi-core/src/identity.rs`, a copy of Ymir's `membership`) belongs in
    `weirdnet-sync`
  - the GPUI `TextField`/`TextInput` (from Morph) belongs in a shared `weirdnet-*` UI crate
  - the Android `ndk-context` init (`bragi-bridge/src/android.rs`, from Morph) belongs in
    `weirdnet-net`
  - `third_party/libp2p-swarm` (Morph's patch relaxing the wasm-only `js-sys` pin so GPUI resolves)
    goes away when libp2p or `weirdnet-net` moves on
- **Not in B1:** a multi-node catalog and failover (built in B2), synced history (B4), Android
  `weirdnet-bragi://` intent handling and a QR scanner (codes are pasted for now), a copy button
  for the code on desktop, and a release signing key.
- **Untested against a real Ymir node or across machines.** Every run used the demo node over the
  memory transport. A run with Ymir's mock engine on a second machine comes with the first
  cross-machine Ymir test.
- **Android screenshot artifacts:** `adb input` leaves the keyboard up (harness, not the app). Gboard
  also autocorrected the first typed word ("What" became "Chat"; the chat title took it from the prompt).
  That was the keyboard, not the scenario's typing: the text field handed the app what the keyboard
  committed. Fixed in the scripted demo build only (`BRAGI_DEMO`): the prompt field sets
  `autocorrect: false` and `enableSuggestions: false`. Real builds keep the keyboard's own settings.
  The Android screenshot set was retaken and `android-05-answer.png` reads "What does Bragi do?".

## B2 status (built 2026-11-19)

Bragi now uses every inference node among the user's devices at once. All behaviour is in
`bragi-core`; both shells only draw it. Still on `weirdnet-agent` 0.4.1; nothing released.

**What works, in both shells:**

- **One merged model list.** `bragi-core` follows each inference node's catalog (one `Pool::subscribe`
  watcher per node the roster lists, found by polling every 2 s) and merges them: one row per
  chat model, best status across nodes (warm, busy, loading, loadable), replicas added, the
  largest speed and context, the nodes that serve it (warm first), and a stand-in if a node
  names one. A node that goes away drops out of the list within a few seconds. The chips say
  "2 of 2 nodes" (desktop) or "2/2 nodes" (Android).
- **Routing and failover.** A request goes to a pool over just the nodes that serve the chosen
  model, warm first. `Pool` moves on to the next node when one is unreachable, refuses or is busy
  before any text, and, during the answer, resumes on the next node that says it can (text is
  placed by offset, so nothing repeats).
- **Retries.** If every node fails before any text and the error could pass (unreachable, busy,
  failed, timed out), the core waits 0.4 s, then 1.2 s, rebuilds the node order and tries again, up
  to twice, sending a `retry` event the shells show as a notice. Cancel ends it at once; "no such
  model" and refusals aren't retried.
- **Flags.** The cache hit and stand-in from the answer's metadata arrive as a `meta` event while
  it streams, and with the number of times it moved nodes (`resumed`) are kept in the history
  message, so reopening a chat still shows "cached", "stand-in gemma-4-e4b" or "moved to another
  node". The message keeps the model that was asked for.
- **Demo.** `DemoNode::start_cloud` runs two inference nodes of one test user (the second is
  certified by the first, so an app linked through the first is admitted by both): `gemma-4-e4b`
  warm on both, `llama-3.3-70b` warm on the second, `qwen3-32b` loadable on the first, which a
  stand-in answers. A repeated question is a cache hit. Scripted `Faults` make the next requests
  be refused or the next answer's node die after N words; a prompt containing "even if a node
  dies" does the latter, so the Android scenario can show it. The single-node demo is unchanged.

**Tests:** all pass.

- `cargo test --workspace`: 22 tests. `bragi-core` has 12: 3 unit tests, the 3 B1 scenario tests and
  6 new ones in `tests/pool.rs` (the merged catalog and routing to the node that has the model, a node
  refusing before the first token, every node failing and the core retrying then failing with
  "failed" in the history, an answer that moves to the other node when its node dies and arrives
  exact, the cache and stand-in flags in the events and the history, and a node leaving the catalog).
  `bragi-desktop` has 10.
- `flutter test` passes 2: the B1 scenario, extended with the merged list and the same three flags
  through the real bridge, plus delta placement. `flutter analyze` and `cargo clippy -D warnings` are clean.
- Both shells were played end to end in virtual workspaces, 11 screenshots each, both passed. The
  desktop ran in Xvfb with `env -i` (`scripts/desktop-scenario.sh`). Android ran on a windowless
  `google_apis` (API 35) emulator with a throwaway AVD, a separate SDK root and `-wipe-data`
  (`scripts/android-scenario.sh`); the SDK, AVD and scratch were deleted afterwards. Screenshots
  `*-09-standin`, `*-10-cached` and `*-11-failover` show the three flags.
  They are in `/workspace/bragi-b2/` on the bot's box.

**Gaps:**

- **Only simulated nodes.** Every run used the two in-process demo nodes over the memory transport,
  not Ymir's M4 daemons, and no run crossed machines. That run comes with the first cross-machine test.
- **A cut-off answer can't yet move on a real Ymir node.** The pool resumes only on a node that
  accepts an `assistant_prefix`. The demo nodes do; Ymir's don't (the `Request::continuing` and
  `RoutedBackend` resume work, which depends on an unreleased `weirdnet-agent`). Until then a real
  node that dies mid-answer leaves a partial answer. Pending, not started.
- **Node order is plain.** Warm first, in catalog order. It doesn't weigh speed, distance or load,
  and the user can't pin a node. Bragi's `Pool` has no per-node view beyond the catalog; a per-node
  health or latency feed from `weirdnet-agent` would let it.
- **Continue** now sends the partial answer as an `assistant_prefix` (`Request::continuing`, agent 0.4.2), no longer an instruction turn.
- **Copies of generic code**, as in B1; the catalog merge itself (`CatalogView::merge`) is generic and
  belongs next to `Pool::subscribe` in `weirdnet-agent` (a merged `CatalogWatch` over all nodes).

## B3 status (in progress 2026-11-19)

Packages for Linux, macOS, Windows, Android and unsigned iOS shipped in v0.1.0 (and later in v0.2.1).
Store listings, signing with a release key and a finished F-Droid recipe are still left. The notes
below are from the first packaging pass on a Linux box.

**Built and run:**

- **Android release builds.** `app/android/app/build.gradle.kts` signs `release` with
  `app/android/key.properties` when it exists (git-ignored, with `*.jks`); without it the debug
  key signs and the build says so. `scripts/package-android.sh` writes one apk per ABI
  (`arm64-v8a` 13.6 MB, `armeabi-v7a` 12.3 MB, `x86_64` 14.4 MB) and the aab (70 MB) to
  `dist/android` with `SHA256SUMS`. Both built here (debug-signed, demo mode on).
- **F-Droid.** Fastlane metadata (title, descriptions, 512 px icon, changelog) under
  `app/android/fastlane/metadata/android/en-US` and a draft recipe `packaging/fdroid/org.weirdbrains.bragi.yml`.
  The recipe is valid YAML but was never run through `fdroid build` (no fdroidserver here) and it
  names a tag `v0.1.0` that does not exist.
- **Linux desktop.** `scripts/package-linux.sh` makes `dist/bragi-0.1.0-linux-x86_64.tar.gz`: binary, `.desktop`
  file and hicolor icons in a `/usr` layout.
- **Windows (partly).** `cargo check --target x86_64-pc-windows-gnu -p bragi-desktop` passes, so the
  shell compiles for Windows. A cross-link does not work from Linux: `gpui_windows` needs the
  DirectX shader compiler at build time. A Windows runner is needed.
- **iOS scaffold.** `app/ios` from `flutter create` (bundle id `org.weirdbrains.bragi`, display name
  Bragi, the brand AppIcon set). Not built.
- **CI (ran 2026-11-19/20, no secrets, nothing published).** `.github/workflows/platforms.yml` (manual or on shell
  changes) on GitHub Actions: `bragi-desktop` builds and its tests pass on `macos-15` and `windows-2025`
  (run 44676967359), and `flutter build ios --release --no-codesign` succeeds (run 44684531804). Fixes it
  needed: `macos-14` runners no longer exist (moved to `macos-15`); the iOS Rust targets in the bridge's
  `rust-toolchain.toml`; and linking the bridge for iOS 15 (`.cargo/config.toml`: rustc's default minimum left
  `___chkstk_darwin`, used by `aws-lc-sys`, unresolved). The Native Assets hook needed no change.

**Tests:** `cargo fmt --check`, `clippy -D warnings` and `cargo test --workspace` (core 3 unit + 6 pool + 3 scenario, desktop 3) pass. Both scenarios passed, 11 screenshots each: desktop under Xvfb with `env -i`, Android on a headless google_apis (API 35) emulator with a throwaway AVD and SDK root, deleted afterwards. The Android run used the release
x86_64 apk (debug-signed, demo mode). Screenshots are in `~/bragi-shots/b3/` on the dev box (the bot box could not be reached for a copy).

**Gaps:**

- **macOS, Windows and iOS build in CI but have never been run.** The desktop tests pass on both, but no scenario
  was played on macOS or Windows and the iOS app was never launched (no Mac or device here); the Android-only
  pieces are cfg-gated, so the iOS build should have no `RustContext` equivalent to supply, unchecked. TestFlight needs Apple Developer
  membership and signing, which are m3talsmith's.
- **No release keys and no upload.** Play and F-Droid need a real keystore (m3talsmith) and a tagged
  release, and F-Droid needs a merge request to `fdroiddata`.
- **The net 0.8.1 Android init swap waits** on a `weirdnet-device-link` release on a looser
  `weirdnet-net` range (it pins `=0.8.0`).
- **The aab is 70 MB** because it carries all ABIs, and the apks are 12 to 14 MB each; no size work was done.
- **No macOS bundle, Windows installer or notarization.**

## B4 status (in progress 2026-11-19)

History moved onto the shared context store. The optional parts are not built.

**Built and run:**

- **Conversations are sealed `ymir.context` records.** `bragi-core`'s `context` module copies Ymir's M5
  store layout (`{conversation}/meta`, `{conversation}/m/{id}`, sealed `weirdnet-sync` records), so a
  conversation written by Bragi can be read by a Ymir pool and the other way round; Bragi's own fields (how an
  answer ended, its node, the cache and stand-in flags, resumes) ride along as extra JSON keys, which Ymir
  ignores. The generic part belongs in `weirdnet-agent` and is copied here until a release has it.
- **`Identity` carries the op log and writes records.** It keeps `log.json` (reloaded on start so sequence numbers
  carry on), configures the sealed type, and pulls it from every other device on each sync pass. A lone device
  starts the key chain itself. With other devices the first rotation comes from a sync pass, so a write
  before then waits: it is kept in `identity/pending.json`, shows in reads, and goes out once the key is there.
- **History is the store.** `History` keeps its API (`load`, `save`, `page`, `delete`); `save` writes only the
  messages that changed, a streaming answer updates one record in place, and a delete is a tombstone that
  reaches the other devices. B1's local files are imported once on the first linked start and moved to
  `history.imported`. Sync passes now run right after going online and on resume (`Core::sync_now`); both
  shells redraw when a pass changed history.
- **Test.** `tests/sync.rs`: two cores linked to one demo node; one chats, the other reads it after sync passes,
  answers from the second device, and a delete on one disappears from the other. 6 of 6 runs passed after a fix to
  the harness (below).
- **Scenarios.** Both shells now end with a "second laptop": a second core on the demo identity asks a question and
  syncs it over, and it appears in the history list and opens (desktop screenshots 12 and 13, Android 12 and 13).

**Tests:** `cargo fmt --check`, `clippy -D warnings` and `cargo test --workspace` pass (core 7 unit + 6 pool + 3
scenario + 1 sync, desktop 3). Desktop scenario under Xvfb with `env -i` passed (13 screenshots), Android on a headless
google_apis (API 35) emulator with a throwaway AVD and SDK root, deleted afterwards, passed (13 screenshots, release x86_64
apk, debug-signed, demo mode). Screenshots: `~/bragi-shots/b4/` on the dev box.

**Not built, and why:**

- **Push wake-ups: Bragi side built (2026-11-26, Bragi v0.15.0, `weirdnet-push` 0.1.1).** Each Bragi device makes a wake
  key and handle per pool and shares its wake card as a sealed record of its own type, `bragi.wake`, pulled only between
  full devices (a cert-only server granted `ymir.context` can't read it). Wakes carry no content; a replay guard (600 s)
  refuses repeats, stale wakes and other handles. Settings: wake-ups on by default with an off switch (off withdraws the
  card and forgets the key); the relay wake path (`/weirdnet/wake/1` to the home relay) is opt-in, off by default, and every
  shell shows its battery cost before turning it on; on Android it runs as a foreground service, so F-Droid builds have a
  wake path without FCM. Cards are pulled every 2 minutes, not every sync pass (each record type pulled is a sync
  session of its own). **Not built:** APNs and FCM delivery (needs push accounts and a push server), and Ymir's side:
  `ymird` sending a wake to a linked Bragi when an answer or job is waiting, and keeping `bragi.wake` out of cert-only
  grants. Not checked against the public relay (does it serve `/weirdnet/wake/1`?).
- **On-device model: not built.** It needs a llama.cpp build and model packaging for Android and the desktop, and a
  place in the pool as a local node; that is its own piece of work, not started.
- **Not run across real machines or with Morph or Ymir as the other device** (test identities, memory transport):
  needs m3talsmith to link two real devices. Ymir's own sealed-context read of a Bragi record is checked only for the JSON fields it
  reads, not against a running Ymir pool.
- **Requests still send whole messages.** Bragi does not send `context_ref` yet, so a Ymir node reads no history
  from the store for Bragi's chats; that follows once the node side of M5 is in use.
- **Keys:** a conversation written on a device that has not yet received the first epoch key waits (see above); it is
  unsealed on that device's disk until then, like B1's history was.
- **Test harness finding:** devices on the memory transport find each other by address only, and the key chain's first
  rotation is made by the pulling device with the lowest peer id, so a stand-in device that never pulls can stall it.
  The demo node now pulls too (`DemoNode::sync_with`). Real devices run sync passes and are not affected.
- **No merge of two devices editing one message** beyond last-writer-wins per record, and no retention or expiry yet.

## B5 status (in progress 2026-11-20)

Attachments and agent mode are built on both shells. BSD support is written but could not be built here.

**Built and run:**

- **Text attachments.** `bragi-core::attach`: a text file up to 256 KiB, up to 5 per message, is inlined into the prompt as
  a fenced block and kept (name and text) with the message in history, so a reopened chat still shows it. Binary
  and larger files are refused with a reason, because `infer/2` carries text only. Desktop: a path box and chips; Flutter:
  the system file picker and chips (the demo build adds a sample file instead of opening the picker).
- **Agent mode.** `Core::send_agent` runs a task on a node over `weirdnet-agent`'s `remote::task` (task/1). Tool calls
  and their outcome are logged on the message, a tool that needs approval shows a card (allow once, allow for this task,
  deny) in both shells, and stopping at a card cancels the task. A node that runs no tasks says so. The demo node hosts a
  small notes tool so the scenarios can run it.
- **Tests.** `bragi-core/tests/breadth.rs`: attachment reaches the model and stays with the message; a task asks, is allowed
  and runs its tool on the node; a declined card stops the tool and stopping at a card cancels; a node without tasks says so.
  Both scenarios gained attach and agent steps (desktop 14 to 18, Android 14 to 17).
- **m5 flake (Ymir).** Not a load timing issue but a race: the mock engine's word delay is a process-wide env var and the
  tests in `m5.rs` set it in parallel, so one test could speed up another's answer so it finished before the node was killed.
  Fixed by making the tests that set it take turns (`73e740e`).

**Not built or not verified, and why:**

- **BSDs: the desktop window does not build on FreeBSD (found 2026-11-20), so on BSD it is the terminal app only.** The manual `freebsd` CI job in `platforms.yml` (a FreeBSD 14.2 VM,
  rustup stable, git CLI, CA bundle, a stub `bzip2.pc`) got through every dependency and then failed in `gpui_linux`: GPUI gates
  `layer_shell`, `PriorityQueueReceiver` and `PriorityQueueSender` to linux, windows and wasm, while `gpui_linux` uses them on FreeBSD.
  It fails the same way at the pinned rev and on Zed's main, so it is a Zed bug, not Bragi's; fixing it means patching or forking GPUI.
  The job requires `bragi-core` to pass its tests there, builds `bragi-desktop` without the window (the terminal app), and only warns for the desktop window. OpenBSD and DragonFly are not GPUI targets.
- **Real nodes served no `task/1` in B5**, so agent mode ran only against the demo node. Ymir M9 adds it (see the M9 status);
  Bragi's own agent mode has still not been run against a real `ymird`, only the two sides separately.
- Attachments are text only (no images, PDFs or binary), and the Flutter file picker was only exercised in its scripted demo path.

## B6 status (built 2026-11-21)

Any OpenAI-compatible server, with an optional API key, can sit next to the Weirdnet pool or act as a
fallback when it is down. Settings add, edit, delete, test and reorder both Weirdnet links and
endpoints; keys are sealed on the device and never synced. Each chat is bound to one target and keeps
its own history, and chats nest in folders. This is built in `bragi-core` (`endpoints`), the Flutter
app (desktop and phone) and the GPUI desktop shell (`view/endpoints.rs`). The older "GPUI shell is
next" note is stale.

## B7 status (in progress)

A Models screen on both shells searches the Hugging Face Hub with a fit badge for each of the user's
Ymir nodes, pulls a model to one node, the best-fitting one or all of them with progress and cancel,
lists what each node has, and removes with a confirm. It runs over Weirdnet with this device's own
identity; nodes accept it only from devices linked to their owner. Built in the core, the Flutter app
and the GPUI desktop shell, and tried against demo nodes and a mock Hub (`models_scenario.rs`). The
desktop also uploads a GGUF model or LoRA adapter from this device to one node (model-upload slice U4,
Bragi 0.7.0, on Ymir 0.12.0): a preview before sending, progress, Cancel and Resume, an optional
license, the node's refusals such as too little free space, and the llama.cpp restart note for an
adapter; the result shows the `ymirctl` commands for a rollout ([U4 as built](design/model-upload.md#u4-as-built-2026-11-25-bragi-070)).
Next: the Flutter upload dialog, `/ymir/rollouts/1` so Bragi can add an upload to a rollout, and a
check against a real Ymir node.

## v0.2 status (shipped in Bragi v0.2.1)

Chat and code are modes of one conversation (Tab, Shift-Tab, `/chat`, `/code`, `/mode`). A desktop
linker (`bragi linker pair`) shows a QR code and a typed code so a phone can pair over the LAN or
through the Weirdnet relay and use code mode against shared folders on the desktop. Also in v0.2.1:
project scaffolding (`/root`), a shell-out command (`!cmd` or `/shell`), prompt history, a
notification log, mouse support in the terminal app, and thinking blocks that stay collapsed and can
be expanded.

## M9 status (in progress 2026-11-20)

The parts of "breadth and hardening" that can be built and tested on one machine are built: self-update with rollback,
agent tasks for Bragi, the Windows service, a security review with checks that run, Inkling 2 Mini, automatic context
sizing (fit fixes in 0.3.0 and 0.3.1), MLX mock end-to-end tests, outside MCP servers for tasks, and a bundle fetcher
(`update fetch`, `update watch`). Push wake-ups and BSD packages are not; MLX-LM for Inkling waits on mlx-lm#2086.

**Built and run:**

- **Agent tasks (`task/1`), for Bragi's agent mode.** `ymir-core::tasks`: the inference role serves `/weirdnet-agent/task/1`
  to the user's linked devices (the same roster check as every other protocol; a stranger, another identity's device or a
  revoked device gets nothing). The model is the pool's own, through the coordinator. With no workspace set, a task has **no
  tools**. With `[inference.agent] workspace = "/some/dir"` it has one, `files.read`: it asks the asker for approval on every
  call (a card in Bragi), reads one UTF-8 file of up to `max_file_kib` (64) inside that directory, and refuses `..`, absolute
  paths and links that point out of it. `enabled = false` turns tasks off. Companions `assistant` (Bragi's default) and
  `ymir` run here; others are refused. Tests (`ymir-core/tests/tasks.rs`, in-memory transport, test identities): approved
  card reads the file and the model repeats it, a declined card never runs the tool, `../` and absolute paths error, a stranger
  and a revoked device are refused, no workspace means no tools; a unit test covers path resolution.
- **Self-update with rollback (`ymir-core::update`, `ymirctl update`).** On `weirdnet-update` 0.2.0 (0.2.2 with `wire` since the fetcher). `update trust --pin HEX
  --root FILE` pins the org root (the first root must match the pin; a pin is never replaced). `update stage DIR` checks a bundle
  directory (`manifest.env`, optional `freshness.env`, `roots/*.env`, the files): k-of-n root chain, release-key delegation for
  `com.weirdbrains.ymir`/`stable`, expiry, replay and equivocation, no downgrade below the running version, a known-bad update,
  this OS and CPU, a plain file-set format (deb/rpm are refused, they need a package manager), `ymird` present, and every file's
  signed size and BLAKE3, then moves it into `updates/releases/<id>/` in one step. The next `update boot` runs it on trial; `ymird`
  commits it once serving; a start that never gets there counts as failed, and after two the previous good release returns
  and the update is marked bad for good (re-staging it is refused). `ymirctl service install --managed-updates` starts the
  service through `ymirctl update boot --exec`. Tests: five unit tests (happy path and restart, crash loop and rollback, wrong
  signer, tampered file, other OS, replay, downgrade, wrong pin, nothing pinned) and `m9.rs` through `ymirctl`.
- **Security review of the roster check, role boundaries and backend ports** (below), with checks that run in `m9.rs`: a running
  combined node (bridge, Weirdnet listener, an engine) has only loopback listeners, the state dir is 0700, the control socket,
  bridge token and identity seed 0600, the bridge refuses a missing or wrong token and a foreign `Host`.

**Security review (by the author, not audited).**

- *Roster check:* infer/1, infer/2, catalog/1, node/1 and now task/1 all authorize through `RosterAuth` before anything is
  read; capability records must carry the user's certificate. Covered by existing tests for strangers and revocation
  (`ymir-core/tests/remote.rs`, `m3.rs`) and the new task tests. Nothing admits a device that is not in the user's roster.
- *Role boundaries (guard added after the first review pass, tightened in the second):* `[model] callers = ["<peer id>"]` narrows a model
  node to the named linked devices through Ymir's own `Authorize`; `m9.rs` shows an app device refused and the inference node admitted, and
  that with no `callers` an app still gets in (the default, kept for compatibility; the daemon logs a warning). **Peer ids only:** a
  `NodeCert` label is not covered by the cert signature, so a label is a weak identity. Config validation rejects a label entry with an
  error that says to use the peer id (`callers_must_be_peer_ids_not_labels`), and a non-peer-id entry matches nothing at the check either.
  There is no cert fingerprint in the published crates to offer as a second form. The transport still carries no role. Original finding:
  **not enforced by the transport.** A model node authorizes any linked device, so an app device can dial it
  directly and skip the inference node's cache, batch caps and fairness. It cannot reach anything outside the user's own
  devices, so this is a bypass of Ymir's scheduling, not of confidentiality (read from the code; not demonstrated). Fixing it needs the
  caller's role in the authorization decision (upstream request below). **Becomes an upstream check in `weirdnet-agent` 0.4.4**; the
  peer-id guard stays as defence in depth.
- *Revoked model nodes (second review pass):* `weirdnet-backends` 0.1.1's `Verifier` checks a capability record's cert signature but
  not the roster, so a revoked model node's old cert still verified and `RemoteModelNode` trusted it. Ymir now refuses a model node that
  is revoked or missing from the active roster on connect, on every request (load, unload, drain, group up and down, chat, embed), on
  every capability record update (the record is dropped and the follower stops) and on every read of the record, so the next catalog
  refresh and routing pass see it gone (`a_revoked_model_node_is_refused_on_connect_and_on_each_request` in `m9.rs`). **Becomes an
  upstream check in `weirdnet-backends` 0.1.2 (`Verifier::with_roster`)**; the Ymir check stays as defence in depth.
- *Backend ports:* engines bind `127.0.0.1` (`--host 127.0.0.1` in `weirdnet-backends`; the test checks the live sockets) and
  each launch gets a random API key (`LLAMA_API_KEY` or `VLLM_API_KEY`, in the environment), which Ymir sends as a bearer token. The bridge is loopback with a
  token on every endpoint and a `Host` check. Another local user on the same box can still connect to a port but not use it
  without the key. The Weirdnet listener is the one socket meant to be reachable from the network (default `0.0.0.0`); it
  accepts only Noise-authenticated peers and then the roster check.
- *Group workers (M7):* llama.cpp `rpc-server` has no authentication, so Ymir binds it to a declared private-link address only and refuses
  any address that is not loopback, RFC 1918 or link-local (tests in `m7.rs`). Anything else on that network can talk to the worker, so a
  "private link" must be a cable or an isolated VLAN, as the design already says. The M9 check does not start a group.
- *Secrets:* the bridge token and seed are files (0600), not in the OS keyring; the keyring is still open work. A cert-only
  node (M10) has `credential.json` (the user's public key) instead of a seed. The unit file
  carries no secret. Update state is 0700.
- *Task tools:* an agent task has no tool unless the operator sets a workspace, the one tool is read-only and asks every time, and
  "always allow" answers last only for that task's session (the node builds a fresh policy per task).
- *Not reviewed:* `weirdnet-*` internals (Weirdnet lead), the engines themselves, the semantic cache's privacy, Windows named-pipe
  control (does not exist yet).

**Not built, and why:**

- **Push wake-ups:** `weirdnet-push` 0.1.0 has sealed payloads and APNs/FCM senders, but its Weirdnet front end (so the inference node
  can embed it with no listener) does not exist; tracked below. It also needs Apple and Google credentials from m3talsmith.
- **BSD packages: FreeBSD in progress (2026-11-26).** `ymirctl service install --system --run-as USER` writes an rc.d script
  (`/usr/local/etc/rc.d/ymird`, `daemon(8)` running `ymird` as that account, `sysrc ymird_enable=YES`; FreeBSD has no per-user
  services, so user scope is refused), and `scripts/package.py --target x86_64-unknown-freebsd` writes a `tar.gz` and, on a FreeBSD
  host, a `pkg` (`pkg create`; no bundled llama.cpp, since upstream publishes no BSD builds: the `misc/llama-cpp` port, named in the
  package message). [.github/workflows/freebsd.yml](../.github/workflows/freebsd.yml) builds and tests in a FreeBSD 14.2 VM,
  installs the pkg and runs the rc.d service through a chat, a stop and an uninstall. The Release workflow builds the
  x86_64 tar.gz and pkg in the same VM and they go on the GitHub release (from 0.21.0); FreeBSD is not a `release.toml`
  `[[platforms]]` entry yet, since the shared release script (weirdnet-project-helpers 0.4.0) knows no `freebsd` OS. No aarch64; OpenBSD (`tar.gz`), GhostBSD and DragonFly not started. (Bragi's FreeBSD CI job is a separate item,
  see the Bragi B5 status.)
- **The bundle fetcher is built (2026-11-24); publishing is not.**
  - `update fetch SOURCE [--announcement FILE]` reads a bundle directory over HTTPS or from disk (`Updater::fetch` with a
    `Source`). The manifest is verified against the pinned root first, on a copy of the client state so nothing is
    recorded, and only then is anything else downloaded. Each file is read up to its signed size. An announced manifest
    hash must match. Then the usual `stage` runs.
  - `update watch --relay ADDR` subscribes to Ymir's app and channel on `/weirdnet/updates/1` (`weirdnet-update` 0.2.2
    `wire`, as `weirdnet-relay` 0.6.1 `[updates]` serves it) from a dial-only node with a fresh key, and with `--fetch`
    stages what it hears.
  - Tests: `ymir-core` unit tests, and `ymir-testkit/tests/update_fetch.rs` against a local HTTP server and an in-test
    carrier built on `weirdnet-update`'s `Carrier` (not a running `weirdnet-relay`).
  - Publishing (0.15.2): `ymirctl update sign-bundle DIR --key FILE --root FILE [--pin HEX] --arch A` (`ymir_core::update_publish`)
    signs a plain file set with a release key: it checks that the key is delegated by the root for
    `com.weirdbrains.ymir`/`stable`, writes `manifest.env` (each file's size and BLAKE3, kind `full`/`tar-gz`, update id
    `com.weirdbrains.ymir-stable-SEQ`, seq `MAJOR*10^6 + MINOR*10^3 + PATCH`, valid 90 days for staging) and the root as
    `roots/1.env`, and verifies the result as a node would before writing. The key file is the 32-byte seed in hex and
    is never printed. `scripts/weirdnet_release.py updates VERSION --upload` (weirdnet-project-helpers 0.4.0, config in
    `release.toml [updates]`) builds `ymir[-cuda]-VERSION-linux-ARCH.update.tar.gz` and `ymir-VERSION-macos-ARCH.update.tar.gz`
    from the release archives, signs them on the release machine with the release's own `ymirctl` (the key path in
    `WEIRDNET_UPDATE_KEY`; CI holds no key and only checks the plan), and uploads them with a sha256 list. The updater
    tells bundles apart only by OS and CPU, so the CUDA bundles are separate and a node is given one on purpose. The
    org root is `packaging/updates/root.env`.
  - A node takes one with: download the `.update.tar.gz`, unpack it, `ymirctl update stage DIR`, restart the service
    (the trial start and rollback do the rest). GitHub release assets can't hold the bundle's `llama.cpp/` paths, so
    `update fetch https://...` needs an HTTPS host serving the unpacked directory. Not built: that host, announcements
    and a `weirdnet:` content source.
  - Hands-off updates (0.15.3, `ymir_core::autoupdate`): `ymird` checks `[update] feed` (default
    `github:WeirdBrains/ymir`, GitHub's latest release; any `https://` URL with the same JSON works) two to five
    minutes after it starts and then every `[update] interval` (default `6h`, at least `5m`). A release newer than
    the running one, not rolled back here before, with an asset `ymir[-cuda]-VERSION-OS-ARCH.update.tar.gz` for this
    machine (`[update] variant`: `auto` keeps the build that runs, by its `llama.cpp/engine.toml`; or `plain`,
    `cuda`), is downloaded whole (up to the announced size), unpacked into `updates/incoming/` (only plain files and
    directories under one top directory) and staged with the usual checks; the feed itself is not trusted. Under a
    service manager (systemd's `INVOCATION_ID`, launchd, or `YMIR_SERVICE=1`) and when started through `update boot`
    (the `YMIR_UPDATE_BOOT` it now sets, or a `ymird` inside `updates/releases/`), it shuts down cleanly and exits 75;
    the service restarts it (the shipped units use `Restart=on-failure`) and `update boot` runs the release on trial.
    A trial that isn't serving in 15 minutes exits 1, and the second failed start rolls back to the last good release
    and marks the update bad. Elsewhere it only stages. Settings: `[update] auto = true`, `channel = "stable"`,
    `interval = "6h"`, `feed`, `variant = "auto"`, `restart = true`; nothing happens without a pinned root.
    `ymirctl update check` does one check by hand. The restart doesn't wait for idle: requests in flight see the node
    go and resume elsewhere, as at any restart.
- **Updating `ymirctl` and the engines with `ymird`:** the release directory may hold `ymirctl`, but only `ymird` is started from
  it; the service still calls the installed `ymirctl`.
- **Self-update on a real machine** was not exercised: the tests run on one box with fake binaries and test keys.

**Upstream issues (Weirdnet lead; no `weirdnet-*` edits made here).** All four are still open and unchanged on 2026-11-20:
`weirdnet-openai` #1 (images, content arrays, unknown parameters) and #2 (Retry-After from Busy), `weirdnet-agent` #1
(Busy retry delay), `weirdnet-backends` #1 (Windows graceful stop, vLLM on Windows). New, found in M9 (to file; drafts):

1. **`weirdnet-agent`: `Serving` has no `tasks` field.** `task/1` can only be served by calling `node.accept_streams(task::PROTOCOL, ..)`
   and `task::serve` by hand, and redoing the roster check. Ask: `Serving { tasks: Option<Arc<dyn TaskHost>> }` with the same
   authorization as infer; and `TaskHost::session` should receive the `Caller`, so a host can scope tools per device.
2. **`weirdnet-agent`/`weirdnet-backends`: authorization has no role.** `Authorize::authorize(&Caller)` and `RosterAuth` cannot tell an
   app device from an inference node, so a model node cannot refuse app devices. Ask: let a roster entry or the announced app
   (`INFERENCE_APP`, `MODEL_NODE_APP`) reach the authorization hook.
3. **`weirdnet-update`: `signing_identity` is required (1..128 chars) for plain file-set packages.** A Linux tarball has no platform
   signing identity; Ymir writes `unsigned`. Ask: allow empty for `InstallMode::Swap` formats, or document the value.
4. **`weirdnet-push`:** the Weirdnet front end for the server `Service` (already listed above, needed for push wake-ups).

**1.0 criteria (proposed).** Ymir is 1.0 when all of these hold; each has its state today.

| Criterion | State |
| --- | --- |
| M1 to M7 built, M8 done with m3talsmith's approval at each step | M1 to M6 and M8 done, M7 llama.cpp only |
| Full workspace tests pass on a clean checkout, and the real-engine tests have run on the target hardware | passes here; real engines run on one NVIDIA box only |
| A run across at least two real machines: link, serve, fail over mid-answer, update | not done (all tests are one box, memory or loopback transport) |
| The four upstream issues are closed or knowingly accepted | all open |
| Push wake-ups work end to end on a phone | not built |
| Signed releases from the real org root, installed by `ymirctl update` and rolled back when broken | mechanism built and tested with test keys; no real keys, no release |
| Packages for Linux, macOS and Windows, plus BSD where GPUI/Rust allow | tarball and service files only; nothing signed |
| A review by someone other than the author | the review above is the author's |

## M10 status (built 2026-11-24)

m3talsmith approved the M10 design on 2026-11-20; the Weirdnet lead published the crates (`weirdnet-sync` 0.7,
`weirdnet-device-link` 0.4). Ymir is on `weirdnet-sync` 0.7.1 and `weirdnet-device-link` 0.4.2. Not 0.8 yet: `weirdnet-agent`
0.5.2 and `weirdnet-backends` 0.4.2 accept sync below 0.8, and 0.8's lease is for M6's scheduler, not for this.

**What a cert-only node holds.** A node certificate and a user-signed `DeviceGrant` (class `Service`, its role, the sync types
it may hold keys for) in the roster, the user's *public* key in `credential.json` (0600, the layout of
`weirdnet_sync::Identity`), its node key, and, for an inference node, the epoch keys. Never the seed. Serving, the roster
check and sync use only the public key and the epoch keys, so it serves and syncs like any device.

- **Roles.** `--service inference` grants `ymir.inference` with the types `ymir.pool` and `ymir.context` (it reads and writes
  sealed conversations); `--service model` grants `ymir.model` roster-only: no data keys at all, and its roster arrives in
  clear roster swaps (`Envelope::clear`), which carry certs, revocations and grants, never records.
- **What it cannot do.** Link, approve or revoke devices, sign grants or rotate keys (`weirdnet-sync` answers
  `NotPermitted`; Ymir refuses first with a message naming cert-only): `pool add`, `pool revoke` and the daemon's
  `PoolApprove`, `PoolRevoke` and `PoolServiceBundle` fail there.
- **Revoking it.** On a device with the seed, `ymirctl pool revoke PEER` as for any device: the roster revokes the cert and
  `weirdnet-sync` rotates, so the node cannot open anything written after. The root identity is unchanged.

**Commands.**

- `ymirctl pool add CODE --service inference|model` on a device that holds the seed: signs the grant (in the running
  `ymird` when it is on Weirdnet, so a first rotation lands in the live roster, `ControlRequest::PoolServiceBundle`, local
  socket only), makes the bundle (`SyncState::service_bundle`) and sends it with a certificate over
  `Pairing::send_credential`; then approves the certificate as before.
- `ymirctl pool link --cert-only` on the new server: takes only a cert-only link (`HostNode::take_any`). If the other side sends
  the seed anyway, nothing is stored and the link fails; the seed did cross the encrypted link to this process, so on a
  server, send with `--service` in the first place. Without `--cert-only`, `pool link` takes either kind.
- `ymirctl pool cert-only` (the migration) on a node that holds the seed, with `ymird` stopped: shows a link code for its
  *own* node key, takes a cert-only link from a seed holder, keeps the roster, op log and older epoch keys it had, writes
  `credential.json`, then overwrites the `seed` file with zeros, syncs and deletes it. The peer id stays the same, so
  nothing else re-links. Refused for another identity or another node key. Copy-on-write filesystems, snapshots and
  backups may still hold the old blocks.
- `ymirctl pool devices` marks a cert-only node (`cert-only ymir.inference`) from its signed grant, and says when this node
  is one.

**Tested** (`ymir-testkit/tests/m10.rs`, over Weirdnet on 127.0.0.1): a real `pool link --cert-only` / `pool add --service
inference` with the 6-digit check (no `seed` file, `credential.json` 0600, the grant shown on the seed holder); `pool add`,
`pool revoke` and `pool cert-only` refused on the cert-only node; a `--cert-only` link refusing the seed and storing
nothing; a node linked the old way moved with `pool cert-only` (`seed` gone, same peer id and identity, grant `ymir.model`);
revoking from the seed holder with the identity unchanged. With daemons: a cert-only model node and a cert-only inference
node serve an app's chat; a conversation written on the app reaches the cert-only inference node; that node's daemon refuses
`PoolRevoke` and `PoolServiceBundle`; after it is revoked, a later conversation never reaches it.

**Not done or untested.** No real machine was moved: every existing server still holds the seed until `pool cert-only` is run
on it with m3talsmith's go-ahead. Bragi and Morph send only full links until they learn `send_credential` (Bragi is next).
Secrets are still 0600 files, not the OS keyring (`SecretStore` makes that possible later). The `[model] callers` guard
still uses the announced app, not grant roles.

## Decisions (2026-11-19)

m3talsmith answered the first round of open questions:

1. **Names:** Ymir (node app) and Bragi (client) confirmed.
2. **Ownership:** the Weirdnet Lead Developer reviews this doc and owns only the generic
   `weirdnet-*` crates; m3talsmith and this team build Ymir and Bragi (see [Ownership](#ownership)).
3. **Pool discovery and remote inference belong in `weirdnet-agent`** (a `pool` module next to
   `remote::infer`/`RemoteModel`), so any Weirdnet agent app can use the cloud. The separate
   `weirdnet-pool` crate is dropped.
4. **llama-swap is replaced** by Ymir's own process manager, because models swap across nodes in the
   cloud.
5. **No HTTPS or API-key endpoint.** Weirdnet is the only remote path; the OpenAI bridge is
   loopback-only.
6. **The pool is m3talsmith's own devices only** (no CI, teammates or other training hosts): linked devices via
   `Roster::check_caller`; no invites, roles or quotas.
7. **Mid-answer failover is automatic**, resuming from the streamed tokens on a replica.
8. **License:** BSD-3-Clause for Ymir and Bragi.
9. **Nodes sync context** (conversations, prompts, attachments, retrieved context, prefix hints)
   with `weirdnet-sync`; KV caches stay local. Bragi's history uses the same store.
10. **Three roles:** model node (runs models, thin wrapper), inference node (the coordinator: catalog,
    load balancing, placement, failover, context sync) and app node (Bragi and any `weirdnet-agent`
    app, talking only to inference nodes). Apps never call model nodes directly.
11. **Roles are placed freely:** independent toggles per node, any combination on any device, any
    number of each; the protocol is the same local or remote, with an in-process fast path.
12. **Root seed on service nodes (resolved 2026-11-19):** model and inference nodes get the root seed
    through device-link for now, stored in the keyring or a 0600 file; a cert-only server link follows
    as M10 (Weirdnet lead).
13. **Push (resolved 2026-11-19):** `weirdnet-push`, owned by the inference node, which runs the sender
    and holds the APNs and FCM credentials m3talsmith provides.
14. **Updates (resolved 2026-11-19):** both. Package managers first; signed self-update with rollback
    via `weirdnet-update`, off by default for package-manager installs unless the user opts in.
15. **Nodes per process (resolved 2026-11-19):** option A. A process may run two or more Weirdnet
    nodes for now (app node, sync node, more if needed); one shared node once `weirdnet-sync` can run
    on an app's node.
16. **Private link (resolved 2026-11-19):** Ray and llama.cpp RPC run only over a private link (direct
    cable or isolated VLAN). Allowed but not expected: cross-machine splitting is opt-in, and routing
    and replication across nodes is the normal way to scale.
17. **Relay and device-link bump (resolved 2026-11-19):** approved. The Weirdnet lead moves both to
    `weirdnet-net` 0.8 and redeploys the public relay. Done (device-link 0.3.0, relay 0.6.0).
18. **Roster (resolved 2026-11-19):** Ymir shares Morph's roster and sync group, with its own `ymir.*`
    record types.
19. **M3 wire format (approved 2026-11-19):** `infer/2` is CBOR throughout; `infer/1` stays JSON.
20. **Agent core (approved 2026-11-19):** `weirdnet-agent`'s core is transport-neutral, a `Caller`
    plus an `authorize` hook; Ymir moves to agent's request types in M3, and the M3 gate is
    `weirdnet-agent` 0.4.0 (with `weirdnet-model` 0.4.0 and `weirdnet-mcp` 0.5.0).
21. **Order (approved 2026-11-19):** M3, then B1, then Ymir and Bragi milestones alternate. Alerts
    stay in `ymir-alerts` until the Weirdnet lead extracts them; `ymir-backends` moves to a new
    `weirdnet-backends` repo after M2.

## Review by the Weirdnet lead (2026-11-19)

The Weirdnet lead reviewed the doc at `cd4fe63`. This revision applies it: Profiles copy `WEIRDNET`;
two nodes per process for now; `infer/2` beside `/1` with opt-in resume and offset-based dedupe;
reasoning and tool-call frames; caller identity in `serve()`/`open`; a real `Queued` queue; stall
handling vs net's 30 s resume; feature-gated `pool`/`context`; the device-link/relay gate; the
coordinated `weirdnet-agent` release (0.3 at the time; the `infer/2` release is now 0.4.0); `weirdnet-sync` facts (`/weirdnet/sync/2`, 112 KiB pages,
16 MiB chunked values, already on net 0.8) and its eviction gap; alerts on `weirdnet-rules` with push
as a wake only; backend downloads through `weirdnet-model`; the semcache storage trait and shard key;
bridge hardening; who admits a device; and the job approval rule.

## Open questions for m3talsmith

The design questions were resolved on 2026-11-19 (decisions 1-21 above). Found while building M2:

1. Install `rpmbuild` (or `nfpm`) and `minisign` on the build box, or build those in release CI?
2. A release build on an older glibc (Debian 12 / Ubuntu 22.04 need glibc 2.36 / 2.35; local builds
   need 2.39)?
3. Windows: a named-pipe control transport before Windows packages, or leave Windows to M9?
4. macOS builds and the Homebrew tap: which machine builds and tests them?
5. Embeddings from a chat GGUF: keep `--pooling mean` as a documented setting, or have setup add it?
6. On the dev box the default route is Wi-Fi, so the network-degraded alert fires there. Keep the rule
   on for every node, or only on nodes with `alerts.enabled`?
7. Enable linger and the user service on any machine (nothing is enabled now)?
