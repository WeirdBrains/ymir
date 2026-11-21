# Handoff

A running log of finished work, newest first. Each entry says what changed, the commits, the checks run and what was not tested, what is left, and what is next.

## 2026-11-21: `ymirctl models search | pull | list | remove`, part 1 (LF-30 M1, branch only)

**What changed**

- New `ymirctl models` subcommands, local to the node they run on (plain `ymirctl models` still lists the pool's models):
  - `search QUERY...`: Hub search for GGUF (and MLX on Apple Silicon, or `--format all`), downloads, likes, and a fit verdict for this node. The verdict sizes the best quant as weights + KV cache at the context (default 32768, capped at the trained length, `--context`) + 0.5 GiB overhead against GPU memory or the unified budget: `fits`, `partial` (spills to RAM) or `too big`. The KV figure comes from the GGUF header (a range request, stopped before the tokenizer), or is estimated when that fails, and the output says so.
  - `pull REPO [--quant Q] [--file F] [--context N] [--id ID] [--role R] [--register] [--dry-run] [--force] [--dir D]`: picks the highest-quality quant that fits (full precision only as a last resort), downloads every shard of a split GGUF, resumes `.part` files, verifies SHA-256 and size, and checks free disk space first. MLX repos download as a folder under `mlx/`. It prints a `[[model.local]]` entry in the shape `setup` writes; `node.toml` is only touched with `--register`, which appends the entry (a copy is kept as `node.toml.bak`, a duplicate id or an unparsable result writes nothing).
  - `list`: GGUF files (shards as one entry), MLX folders and unfinished downloads, and whether `node.toml` uses each.
  - `remove ID`: asks first (`--yes` skips), deletes files only, warns if the model is still registered, and never follows symlinks.
- The Hub token is read on the node only (`HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `YMIR_HF_TOKEN_FILE`, `state/hf-token`, then Hugging Face's own token file), sent only to the Hub host, prints as `<redacted>` and is never put in an error.
- The code is in `crates/ymirctl/src/models/` (`hub.rs` Hub client and downloader, `gguf.rs` header reader, `plan.rs` quants, fit and stanza, `local.rs` disk and `node.toml`) so it can be swapped for `weirdnet-model` later. `ymirctl` gained direct `ureq`, `sha2`, `toml`, `serde`, `libc` and `dirs` dependencies (all already in the lock file); no `weirdnet-*` crate was touched and `weirdnet-model` stays on 0.4.3.
- Docs: the ymirctl guide page, [DESIGN.md](docs/DESIGN.md) ("Searching and pulling models") and the changelog.

**Checks run:** `cargo fmt --check`, `cargo build`, `cargo clippy --workspace --all-targets -D warnings` and `cargo test --workspace` all pass. New unit tests use a mocked Hub (a local socket server with `Range` support): search parsing and encoding, token only as a header, gated, rate-limit and missing-repo errors, header reading, download, resume after a cut connection, checksum mismatch, quant grouping and choice, split GGUFs, `--register`, `list` and `remove`. One real pull of a small GGUF (about 400 MB) into a throwaway directory on the dev box: downloaded, checksum verified, listed, removed.

**Not tested:** gated repos with a real token; multi-GB and split GGUF downloads against the real Hub (split handling is covered by the mock); MLX pulls on Apple Silicon; Windows free-space check (it reports "unknown" there); the fit verdict on a multi-GPU node.

**What's left (M2, M3)**

- M2: pull onto another pool node (a node/1 download request, so a pool node can fetch for a peer), and a pool-wide search that shows the verdict per node.
- M3: swap to `weirdnet-model` once `weirdnet-model` has hub search, a quant chooser, token support, remove, split GGUF and a KV-aware fit; vision projector (`mmproj`) download alongside a model; skipping non-chat repos (ASR, embedding) in search; `--json` output.
- `ymird --version` still reports 0.1.0 for every build.

**What's next:** LF-16, then the rest of the image and audio work.

## 2026-11-21: Weirdnet 0.5 crates, images and audio through the pool, MLX on `weirdnet-backends` 0.2 (LF-18, branch only)

**What changed**

- Crates: `weirdnet-agent` 0.5.0, `weirdnet-backends` 0.2.0, `weirdnet-openai` 0.2.0, `weirdnet-semcache` 0.2.0. `weirdnet-model` stays on 0.4.3 and `weirdnet-alerts` on 0.1.0 (see "Gaps for the Weirdnet Lead").
- The catalog advertises inputs: `agent_catalog` calls `CatalogModel::with_inputs` with what the nodes' copies of each model take.
- Routing by capability in the coordinator: a request with images or audio goes only to nodes whose copy of the model takes them; with no model named it picks the main model or the best pool model that reads them; a model nobody reads them with is a 400. Media answers skip the cache.
- `tests/vision.rs` now expects success through the bridge (the `the_bridge_does_not_carry_images_yet` test became `the_bridge_carries_images_to_models_that_take_them`), plus a new routing test in `tests/nodes.rs`.
- MLX runs as `EngineKind::Mlx` from `weirdnet-backends` 0.2.0; the `ymir-engine` shim (binary, `shim.rs`, its test binary and tests) is removed. The OpenAI bridge keeps accepting content parts (`image_url`, `input_audio`) through `weirdnet-openai` 0.2.
- The earlier interrupted executor's MLX work (kept as a patch in scratch, written against `weirdnet-backends` 0.1.3) was reused: shim removal, `EngineConfig::mlx`, packaging and docs changes.
- [DESIGN.md](docs/DESIGN.md): the "pool does not carry images" gap is replaced by how it works now.

**Gaps for the Weirdnet Lead** (no `weirdnet-*` crate was touched)

1. `weirdnet-backends` 0.2.0 depends on `weirdnet-model` `0.4`, not 0.5, and its `node` API takes model 0.4's `Hardware` and `Budget`. An app on `weirdnet-model` 0.5.0 gets two copies and a type mismatch in `with_hardware`, so Ymir stays on 0.4.3. Needs backends 0.2.1 on model 0.5.
2. `weirdnet-alerts` 0.1.0's `backends` feature needs `weirdnet-backends` 0.1, which pulls in `weirdnet-agent` 0.4 as a second copy. Ymir drops the feature and carries the engine-event monitor in `ymir-core/src/engine_alerts.rs` (a copy of `weirdnet_alerts::engine`). Needs alerts 0.2 on backends 0.2; then delete that module.

**CI follow-up:** the push token has no `workflow` scope, so `.github/workflows/macos.yml` is unchanged on this branch. It still runs `ymir-testkit --test engine_shim`, which this branch deletes. Before the macOS job can pass, change that step to `cargo test --locked -p ymir-testkit --test mlx` (named "MLX and llama.cpp on one node (mock engines)"). Someone with workflow scope must push that one change.

**Checks run:** see the commit's push; `cargo fmt --check`, build, clippy `-D warnings` and tests on the whole workspace.

**Not tested:** a real GPU node with a real Gemma projector (mock engines only); an image across two real machines; MLX on Apple Silicon (no Mac here; the macOS workflow covers it once the branch is pushed through CI); Bragi sending images (see Bragi's HANDOFF).

**What's left:** the two crate gaps above; `weirdnet-model` 0.5 once backends follows; remote `https://` image URLs are not fetched.

**What's next:** LF-16, the Ymir MCP server.

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
