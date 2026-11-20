# Ymir design

Date: 2026-11-19. Status: design settled; every open question was answered on 2026-11-19 (see
[Decisions](#decisions-2026-11-19)). **M1 is built** (single box, all three roles in one `ymird`;
see [M1 status](#m1-status-built-2026-11-19)).
Reference build: m3talsmith's two GB10 servers (deployment configs and a plan snapshot are kept off-repo).
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

- Training or fine-tuning (the separate trainer stays separate).
- Other users: no CI tenants, teammates or shared pools; no invites, membership roles or quotas.
- Any HTTPS or API-key endpoint, LAN or public. No WeirdBrains-run gateway (forbidden by AGENTS.md).
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
| DGX Spark | GB10, sm_121, 128 GB unified memory | The server-a/server-b setup on one box: `qwen3.6-35b-a3b` on llama.cpp, a 27B NVFP4 on vLLM, Gemma 4 E4B stand-in, embeddings. |

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
running as `NT SERVICE\ymird` (AGENTS.md). It bootstraps from the public WeirdBrains relay by
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
  `/weirdnet-agent/catalog/1`.
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

### Model sizing

`weirdnet-model`'s catalog grows from Gemma 4 only to a tiered catalog with per-quant memory
footprints. Setup picks models whose weights plus a KV budget (context × concurrency) fit within
the usable budget (default 85% of VRAM, or of unified memory after the OS reserve), and proposes:

- one **main** chat model (largest that fits with headroom),
- one **stand-in** (Gemma 4 E2B/E4B),
- one **embedding** model (`qwen3-embedding-0.6b`) if the cache is on.

The operator can change any of them.

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
nodes](#root-seed-on-service-nodes)). Unlinking (revocation in the roster) cuts the device off everywhere as soon as peers see
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
| `weirdnet-relay` 0.6.0 | the public WeirdBrains relay and an optional own relay. On `weirdnet-net =0.8.0`; the public relay runs 0.6.0 (weirdnet-relay `07f1162`). |

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

Also generic but small, in `ymir-core` for now: `service` (systemd user unit and launchd agent
install). `health` moved to `weirdnet_alerts::engine`.

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
- **Follow-up (later milestone M10, owned by the Weirdnet lead):** a cert-only server link in
  `weirdnet-sync`/`weirdnet-device-link`, so a service node gets a node certificate and the sync keys
  it needs but not the seed, and can be revoked without rotating the root identity. Existing nodes
  then re-link and their seed copies are wiped.
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
private_link = "10.0.0.1"         # optional: address for group jobs (RoCE/NCCL, RPC)
llamacpp = "managed"                 # managed | system:/path | off
vllm = "auto"                        # auto (only where supported) | off
# wrap = "http://127.0.0.1:8000"     # or wrap a stock vllm/llama-server already running here

[[model.local]]                      # models this node keeps on disk and may be told to load
id = "qwen3.6-35b-a3b"
backend = "llamacpp"
quant = "Q4_K_M"
swap_group = "main"

[[model.local]]
id = "gemma4-e4b"

[[model.local]]
id = "qwen3-embedding-0.6b"

# ---- inference role ----
[inference]
scheduler_eligible = true            # may hold the placement lease
context_disk_budget = "2GiB"         # working set of synced conversations

[inference.cache]
mode = "local"                       # off | local | pool
backend = "embedded"                 # embedded | redis
redis_url = "redis://127.0.0.1:6379" # used only when backend = "redis"
threshold = 0.12

[inference.alerts]
push = true                          # Weirdnet push to m3talsmith's devices
smtp = { to = "owner@example.com", server = "smtp.gmail.com:587", user = "owner@example.com", password_ref = "keyring:ymir-smtp" }

# ---- app role ----
[app.bridge]
port = 8484                          # always 127.0.0.1; there is no address setting
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
- **User-level by default** (systemd user unit with lingering, launchd agent); `--system` for headless
  servers where it is genuinely system-level.
- Prebuilt binaries only; installing never needs Rust. Release files are signed with minisign.

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
   keep `ts-watchdog` (host ops), retire `monitor timer` once peer watching proves itself for a week.
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
  the `weirdnet-agent` pool, retries, automatic mid-answer failover, stand-in and cache flags, in both shells.
- **B3 Platforms** (in progress 2026-11-19; see [B3 status](#b3-status-in-progress-2026-11-19)): iOS (TestFlight), macOS and Windows; Android apk, aab and F-Droid packaging.
- **B4 Sync and optional extras:** history on the shared context store (after M5); the optional on-device model
  and push wake-ups, on both shells.
- **B5 Breadth:** BSDs where GPUI allows, agent mode and attachments, on both shells.

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
- **M6 Workload spreading (core)** (built 2026-11-19, partly; see [M6 status](#m6-status-built-2026-11-19-partly)). Load-balanced routing across replicas, scheduler lease,
  cloud-wide swap, placement and replication, prefix warming, device fairness and priority classes,
  batch and agent job fan-out, pool-mode semantic cache; several inference nodes active-active. Exit
  test: two inference nodes, three model nodes, two replicas of one model, mixed interactive and batch
  load from two app devices; interactive p95 wait stays under target and neither device starves.
- **M7 Split models.** Group jobs: vLLM + Ray over RoCE, llama.cpp RPC over Ethernet; big mode on
  server-a/server-b reproduced as a group job.
- **M8 server-a/server-b migration** (steps above, each approved by m3talsmith).
- **M9 Breadth and hardening.** Self-update via `weirdnet-update` with rollback; push via the
  embedded `weirdnet-push` senders (once its Weirdnet front end exists); Windows service, BSD packages, a security review of the roster check,
  role boundaries and backend ports, 1.0 criteria.
- **M10 Cert-only server link** (Weirdnet lead). `weirdnet-sync`/`weirdnet-device-link` give service
  nodes a node certificate and sync keys without the root seed; Ymir nodes re-link and wipe their seed
  copies; a node can then be revoked without rotating the root identity.

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
- The semantic cache is an exact-match placeholder (deterministic requests only).

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
  Tested on the dev box's NVIDIA GPU (sm_86) with vLLM 0.34.0 (torch 2.14.0+cu130) in a venv under
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

- **vLLM on the Spark (CUDA 13, sm_121) is untested.** server-a/server-b are live and weren't touched; only
  sm_86 was run.
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
  (default: the Weird Brains relay), `peers` (`/ip4/…/tcp/…/p2p/PEER`; dialing grants nothing),
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
membership sync pass is `weirdnet_sync::pull_from_shared` (snapshot, pull, `merge_back`).

**Open gaps (upstream, parked for M9): `weirdnet-openai` #1 and #2, `weirdnet-agent` #1 and
`weirdnet-backends` #1.**

- **Content arrays are text-only** (image parts are refused by the request parser), and unknown
  OpenAI parameters are dropped.
- **Streamed answers report the stand-in in the final chunk**; `BridgeClient` reads it there.
- `Busy` always answers `Retry-After: 1`; Windows kills the engine outright; vLLM is Linux only.
- **The answer cache is exact-match for temperature 0 only** (no tools, inline messages), per
  `weirdnet-semcache`; other requests always go to a model.
- No `GroupWorkers` (group jobs answer Unsupported until M7).

**Not done or untested:**

- **Across real machines is untested.** All multi-node tests run on the dev box (memory transport
  and loopback TCP). The laptop has Rust 1.99, but its load average was 17 on 8 GB, so a build
  there would have been disruptive. server-a and server-b weren't touched.
- **Linking m3talsmith's real identity needs him**: `ymirctl pool link` on a node, then the code opened
  in Morph. Morph main now accepts `weirdnet-ymir://link/…` codes (Identity → Add device), and `pool add`
  takes Morph's `weirdnet-morph://link/` codes too (same payload). The signed roster label reads
  "Ymir on <name>" (the device-link hint). Linking the real identity still waits for m3talsmith.
- Seeds are kept in 0600 files, as decision 12 says, not in the OS keyring. M10's cert-only link
  removes the seed copies.
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
  is there before the periodic pass; the cache is never used for these. `ymirctl context show|add|ask|delete`
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
- **Bragi doesn't use the store yet** (B4), and the mock engine stands in for llama.cpp; the real-engine
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

**Gaps:**

- **Not built: scheduler lease, placement plan, cloud-wide swap, replication on queue wait, prefix warming.**
  They need a lease and a signed, synced plan (`weirdnet-sync` has no lease primitive) and warm-prefix data in the
  capability record. The scorer ignores prefill and decode speed, uncached prompt length, round-trip time
  and relays for the same reason.
- **Not built: the pool-mode semantic cache.** `weirdnet-semcache` still has only the exact cache here.
- **Batch jobs are not synced.** The submitting node runs the fan-out, so if it goes down the job stops
  (the design wants the job record synced so another node picks it up). No tasks with tools or approvals.
- **Fairness is a cap, not a weighted share.** Devices are held to the same cap; there are no per-device
  weights or quotas.
- **One computer, a mock engine,** and the delay above.

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

What a Linux box with no Mac and no Windows can build and check. Nothing is released, signed with a
real key, or uploaded anywhere.

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
