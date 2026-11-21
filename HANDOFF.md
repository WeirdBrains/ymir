# Handoff

A running log of finished work, newest first. Each entry says what changed, the commits, the checks run and what was not tested, what is left, and what is next.

## 2026-11-21: `ymirctl bench` over Weirdnet, a node×model matrix

**What changed**

- `ymirctl bench` now talks to each inference node directly over Weirdnet (`ymir_core::direct::Direct`): a probe node with this device's identity, ephemeral ports, no discovery or relays, then a `Pool` whose node list is the one peer (catalog/1 for models, infer/2 for answers). That is a real node pin, with no `weirdnet-*` change. `--via bridge` keeps the old loopback path.
- New: `--node` and `-m` (repeatable or `all`), `--suite quick|standard|hard`, `--context`, `--max-tokens`, `--quality-tokens`, `--cold` (warns, asks unless `--yes`), `bench list`, `bench compare A.json B.json`, CSV next to JSON and Markdown, side-by-side matrices.
- Still strictly one request at a time, busy or loading models skipped.

- A device with a model role (`/weirdnet-backends/node/1`) is measured on that hardware through `RemoteModelNode`; one with only an inference role goes through a one-peer `Pool` (catalog/1, infer/2). `bench list` shows the role of each node.
- Needs `--hint` for nodes that are not in `[network] peers` (tailnet addresses from `ymirctl pool addresses` on the node).

**Tested for real** (hard suite, warm models only, 2 repeats, 1 warm-up, one request at a time): two nodes reached directly, one model on both. Every model scored 15/15, so the hard suite still does not separate these models; speed does. A node whose model role only allows listed callers refuses this device (`not-an-allowed-caller`); the node's `[model] callers` would need this device, a config change not made here. A node placing a model on another node reports `served_by`.

**Not done:** pinning that node (needs its `[model] callers` to list this device), `--cold` runs against live nodes, harder suites.

## 2026-11-21: `ymirctl bench`, a benchmark that compares every model on the pool

**What changed**

- New command `ymirctl bench` ([bench.rs](crates/ymirctl/src/bench.rs)) and a timed streaming client in `ymir-core` (`BridgeClient::chat_timed`, `embed_timed`). It goes through the loopback bridge, so every request takes the normal pool route. It lists the pool's models, then runs them one at a time: a busy or loading model is re-checked and then skipped, and a model that is not loaded is skipped unless `--include-cold`.
- Per model: one warm-up, then N repeats (default 3) of three fixed prompts (short, a 200-word story, a long prompt) measuring time to first token, decode tok/s, prompt tok/s and total latency (median and p90). Then eight deterministic checks at temperature 0: two arithmetic, JSON format, two Python functions that are executed (CPU, memory and time limits, empty environment, no network where `unshare` allows; `--no-code-exec` skips them) and three exact-match facts. Embedding models get a latency check.
- Each request has its own seed and a nonce, so neither the answer cache nor a node's prefix cache flatters the numbers. Requests send `x-wait: 1`, so a cold model loads instead of a stand-in answering (a stand-in answer counts as an error).
- Output is a timestamped `bench-<time>.json` and `.md` in `--out-dir`. Nodes are labelled node-a, node-b, ... unless `--real-names`. Docs: [DESIGN.md](docs/DESIGN.md) and the [README](README.md).

**How to run:** `ymirctl bench` on a node with the app role (needs the bridge); `--include-cold` to load unloaded models, `-m MODEL` to pick some, `--repeats N`.

**First run** (warm-up 1, 2 repeats, one model at a time).

| model | node | TTFT ms (median) | decode tok/s | prompt tok/s | quality |
|---|---|---|---|---|---|
| gemma-4-e2b | node-a | 307 | 202.0 | 4245 | 8/8 |
| gemma4-e4b | node-b | 293 | 64.9 | 2645 | 8/8 |
| qwen3-14b | node-a | 353 | 46.8 | 1361 | 8/8 |
| qwen3.6-35b-a3b | node-b | 290 | 63.1 | 1865 | 8/8 |
| qwen3.8-27b-nvfp4 | node-c | 284 | 12.5 | 2551 | 8/8 |
| qwen3-embedding-0.6b | not recorded | embedding, 1024 dims, 172 ms for 3 inputs | | | |

(Node labels are per run, not stable names.)

**Checks run:** `cargo fmt --all --check`, `cargo clippy --workspace --all-targets --locked -- -D warnings` and `cargo test --workspace --locked` all pass. New tests: scoring and statistics units, the Python sandbox (a runaway loop is killed), and an end-to-end run against a mock bridge (anonymised names, cold model skipped, embedding row).

**Not tested / limits:** the bridge cannot pin a node, so a model on several nodes is only measured on whichever node routing picks (rows are split by the node that answered). Thinking models spend tokens on reasoning, so their quality score uses 2048 tokens per check. With two repeats the p90 is the larger of two samples. The embedding path does not record which node answered. Running the benchmark loads cold models, which can evict others in the same swap group.

**What's left:** a node pin (a header or a `ymirctl bench --node`) needs support in the bridge, which is a `weirdnet-*` change; embeddings throughput by batch size; image and audio models once they exist.

**What's next:** run it on a schedule and keep the JSON to spot regressions after a node or engine upgrade.

## 2026-11-21: A new node joined the pool

**What changed**

- Installed Ymir (built-in backends binary, user-level service) on a new GPU-equipped Linux machine with the [documented setup](docs/DESIGN.md): `ymirctl setup --roles model --download --service`. It serves a 14B chat model, a small stand-in model and an embedding model.
- Paired it with the existing Weirdnet using a matched link code (`ymirctl pool link` on the new node, `ymirctl pool add` on an existing one). The first pairing did not route: the inference node only dials the peers in its `[network] peers` list and the mDNS/relay paths, and it never learns the address of a node paired after it started. The node was revoked, its pool state reset, and it was paired again; routing worked once the new node was given the inference node's address in its own `[network] peers`, so it dials in.

**Checks run:** the pool's node list shows the new node serving; the bridge `/v1/models` lists its models; a small chat completion for the 14B model was served by the new node (about 40 tokens/s, so on the GPU or a fast path) and the stand-in answered while the model loaded.

**Not tested:** the machine's NVIDIA driver and library versions mismatch until it reboots, so GPU use is inferred from speed only.

**What's left:** pairing does not share the new node's address with the other nodes, a running node does not hot-reload the roster, and nodes do not discover each other on their own. All three are future work; until then a new node must dial an existing one through `[network] peers`.

**What's next:** restrict `[model] callers` to the inference nodes, and fix the three gaps above.

## 2026-11-20: Site and docs cleanup (status copy and clickable references)

**What changed**

- Site copy and READMEs state plainly what Ymir does. Hedging lines ("so far tested on one computer", "not yet tested", "nothing is released yet" on the home page, and similar) are gone from the [home page](https://ymir.weirdbrains.com/), [quickstart](https://ymir.weirdbrains.com/quickstart), [hardware](https://ymir.weirdbrains.com/hardware), the [deployment guide](https://ymir.weirdbrains.com/docs/guides/deployment/) and the [README](README.md).
- [roadmap.toml](roadmap.toml) no longer carries per-milestone gap lists, and the [roadmap page](https://ymir.weirdbrains.com/roadmap) never showed them. Statuses: M8 (move the existing servers onto Ymir) planned → done. M6, M7 and M9 stay in progress, M10 stays planned; their summaries now say what is built and what is next, and M9 now lists the Windows service as built. [ROADMAP.md](ROADMAP.md) and the site data are regenerated. The M8 line in [docs/DESIGN.md](docs/DESIGN.md) says done.
- Platform claims: the hardware page says Ymir builds for Linux, macOS and Windows (BSD packages planned); the deployment guide describes Bragi for desktops and phones on all five platforms and links Bragi 0.1.0.
- Every reference to another document is a link, for example "Read [DESIGN.md](docs/DESIGN.md) for the details" in the [developer guide](https://ymir.weirdbrains.com/docs/develop/).
- New CI check [site/tool/check_links.py](site/tool/check_links.py), run by the [Pages workflow](.github/workflows/pages.yml): it fails on a `.md` file name or a bare URL that is not a link in the page sources, the README, [ROADMAP.md](ROADMAP.md) and [HANDOFF.md](HANDOFF.md).

**Commits:** `028298c` on main.

**Checks run:** `site/tool/gen_roadmap.py --check`, `site/tool/check_links.py`, the live pages fetched and searched for the removed phrases. The Pages workflow builds the site and runs both checks on the push.

**Not tested:** no local `dart analyze` or site build (the local Dart tooling could not resolve the site's packages); CI is the check for those. [docs/DESIGN.md](docs/DESIGN.md) was not swept for caveat wording beyond the M8 line.

**What's left:** the status sections in [DESIGN.md](docs/DESIGN.md) still use "partly" and "with gaps" in a few headings.

**What's next:** M6, M7 and M9 work, then wiring up M10 once its Weirdnet crates are verified.

## 2026-11-21: Pool nodes upgraded to main b7e8ade

**What changed**

- The two aarch64 GPU nodes of the pool now run `ymird` and `ymirctl` built from main at `b7e8ade` (`cargo zigbuild --release --locked`, glibc 2.39). They ran the earlier `99fb30f` build before. Each node was upgraded in turn: old binaries kept as `*.0.1.0.bak` next to the new ones, only the `ymird` service restarted, no change to `node.toml`, the node identity, the unit files or the reverse proxy.
- Both binaries still report `0.1.0`; tell builds apart by SHA-256 (`ymird` `d35eeb77…`, `ymirctl` `b48f20d0…`).

**Checks run:** `cargo fmt --check`, `cargo clippy --workspace --all-targets -D warnings` and `cargo test --workspace` on main before building. After each upgrade: service active, no warnings in the log. On the main node: `ymirctl nodes` lists all three nodes, `ymirctl pool addresses` works, the bridge `/v1/models` returns 200, one completion served locally and one served by the third node (`qwen3-14b`).

**Not tested:** no rollback was needed, so the `*.bak` restore path was not exercised.

**What's left:** `ymird --version` should carry the commit so builds are identifiable without hashing.

**What's next:** the same upgrade for any new node; then the Weirdnet 0.5 work.
