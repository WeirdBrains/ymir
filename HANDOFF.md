# Handoff

A running log of finished work, newest first. Each entry says what changed, the commits, the checks run and what was not tested, what is left, and what is next.

## 2026-11-27 10:37 EST: M11 sharing a pool with other identities (on main)

Michael (11-27, t65u/t66u): share pools with other Weirdnet identities (Morph's m3talsmith profile `5e452f1b…` vs the Ymir pool `08e2b657…`), and have the Weirdnet lead implement the crate side. Built on `weirdnet-agent` 0.5.9 (`fa038ee`). Built on the dev box with mocks; no live node touched. Design: [docs/design/pool-sharing.md](docs/design/pool-sharing.md).

- **Grants** ([guests.rs](crates/ymir-core/src/guests.rs)): `GuestGrant` JSON in `ymir.pool` as `guest/<user_id>` (label, pinned `peers`, `models` allowlist, `scopes`, `expires_at`, `revoked_at`); `share`/`unshare`/`all_grants`, `policy` (a `GuestPolicy` that reads grants live), `pool_share` (a `PoolShare`).
- **Host** ([host.rs](crates/ymir-core/src/host.rs), [remote.rs](crates/ymir-core/src/remote.rs)): `remote::guests(m)` = `SharedPoolAuth(RosterAuth).with_guests(GuestAuth(policy))`; `InferenceHost::with_guests` uses `authorize_for(Infer)`, refuses guests `Input::Context` (`guest-context`) and models outside the allowlist (`guest-model`); the catalog feed uses `scoped(Catalog)` and `CatalogFeed::rechecking()` (`LiveCatalog::recheck`). `roster_auth`/`role_auth` (models, train, tasks, wake, admin) unchanged.
- **CLI** ([pool.rs](crates/ymirctl/src/pool.rs), [control.rs](crates/ymir-core/src/control.rs)): `ymirctl pool share|unshare|guests`, through `ymird` (`PoolShare`/`PoolUnshare` control requests) when it is on Weirdnet.
- **Tests:** guests unit tests 7/7; [pool_sharing.rs](crates/ymir-core/tests/pool_sharing.rs) 2/2 (second identity via `connect_shared`: refused before the grant, catalog + chat after, context refused, roster-only auth refuses the guest, allowlist, peer pinning with a second guest device, unshare refuses the next chat and closes a live catalog subscription within 5 s, expiry); existing remote.rs tests; fmt, clippy `-D warnings`, link check.

**Not tested:** real networks and relays; Morph/Bragi "open share link" screens (their side). **Next:** ships in v0.23.0 with video Milestone 2; then a real two-identity run over real networks.

## 2026-11-27 10:40 EST: v0.22.1 released

**Release:** [v0.22.1](https://github.com/WeirdBrains/ymir/releases/tag/v0.22.1) (release commit fac5d21 on 49f61fd; Release run 46009619619 green, CI 46009618777, macOS 46009618590 and Pages 46009619015 green on fac5d21, FreeBSD 46002234634 green on 49f61fd): the M7 fixes from the first real GPU run. The `vllm-ray` listener check follows Ray's whole process tree, Ray and vLLM workers are pinned to the private link (`VLLM_HOST_IP`, `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`), trusted all-interface listeners are logged at info, and `ymirctl engines install vllm` picks a PyTorch CUDA build the driver can run (`--torch-backend` overrides), checks it after installing, and accepts `Inkling2LoopForConditionalGeneration`. 44 files on GitHub: leak scan 33 files, 0 hits (sums ok 32), then 4 Arch `.sig`s and 6 signed update bundles plus their sums (`sign-release.sh 0.22.1` EXIT 0); `weirdnet_release.py assets 0.22.1` ok for linux, macos, windows and both FreeBSD files. Arch recipes on main; TAUR ymir-bin and ymir-cuda-bin 0.22.1-1 (dry-run first). Committed straight to main (no PRs). No live nodes touched.

- On the way: Windows CI failed twice on test-only code (a loopback-name assertion, then `std::os::unix` in `ymir-mock-ray`), fixed in c5b5bf3 and 49f61fd; the first cut attempt stopped on a flaky `m3` network-timing test (passes alone) before committing; alien-blue's artifact download ran at about 0.57 MB/s, so `pub.sh` took about an hour.

**Next:** re-verify on m1 + m2 with 0.22.1: the driver-matched install (PyPI's CUDA 13.0 PyTorch on driver 580), `vllm-ray` with Inkling 2 Mini, text and vision with the Python frontend, no LAN or Tailscale listeners under the new check, clean teardown, `node.toml` restored.

## 2026-11-27 07:00 EST: v0.22.1 fixes from the M7 real run (on main)

Michael (11-27, t64u): "Let's fix the bugs you found." Built on the dev box with mocks; no live node touched for this commit. Details in [DESIGN](docs/DESIGN.md#m7-vllm-with-ray-groups-built-2026-11-26-mocks-only) ("Security boundary", "Installing vLLM and Ray").

- **Listener check follows the process tree** ([ray.rs](crates/ymir-core/src/ray.rs) `tree_in`/`tree_of`/`listeners_of_tree`, [group.rs](crates/ymir-core/src/group.rs)): every member of the started process's group plus, transitively, every child in any process group or session (Ray's workers, vLLM's workers, NCCL, torch.distributed). The leader's `vllm serve` tree is checked after ready and on watch ticks. `stop` kills what is left of the tree, checking each pid's start time.
- **Link pinning:** Ray head, member `ray start` and `vllm serve` get `VLLM_HOST_IP`, `NCCL_SOCKET_IFNAME==IFACE` and `GLOO_SOCKET_IFNAME=IFACE` (interface from the link address via `if-addrs`).
- **Trusted all-interface listeners** are logged at info, once per (role, ports).
- **Installer** ([vllm_install.rs](crates/ymir-core/src/vllm_install.rs), [engines.rs](crates/ymirctl/src/engines.rs)): driver CUDA from `nvidia-smi`; CUDA 13+ → PyPI PyTorch (cu130 build, no `--torch-backend`); 12.6–12.9 → `cu126`/`cu128`/`cu129` with `--version` (nightly refused); older refused; `--torch-backend` overrides. Post-install `torch.version.cuda` ≤ driver CUDA or the install fails. Record gains `driver_cuda`, `torch_backend`, `torch`, `torch_cuda`. Inkling check accepts `Inkling2LoopForConditionalGeneration`.
- **Tests:** [m7_ray.rs](crates/ymir-testkit/tests/m7_ray.rs) 3/3 (new: a mock worker in its own process group listening on 0.0.0.0 is refused and stopped with the group; the head gets the link env), ray/vllm_install unit tests (tree following, link env, driver parsing, backend choice, CUDA mismatch), [engines.rs tests](crates/ymirctl/tests/engines.rs) 2/2 (dry run shows driver/PyTorch, record, mismatch failure, override). fmt, clippy `-D warnings`, workspace tests, link check.

**Next:** cut v0.22.1, then re-verify on m1 + m2 (driver-matched install, text and vision with the Python frontend, no stray listeners).

## 2026-11-27 06:29 EST: M7 `vllm-ray` with Inkling 2 Mini on real GPUs (m1 + m2)

Michael (11-27) approved a live GPU run. It worked, and it found a security gap in the listener check (open; see below). Results and settings are in [DESIGN](docs/DESIGN.md#m7-vllm-with-ray-groups-built-2026-11-26-mocks-only) ("Real GPU run").

- **Nodes:** m1 and m2 (DGX Spark GB10, 128 GB unified, driver 580 / CUDA 13.0, ymird 0.22.0 via self-update), link 10.100.40.0/24 on `enp1s0f1np1`. starbase was left out (NVML driver/library mismatch, 60 GB).
- **Backups first:** `node.toml.bak-20261127-0606-pre-m7` on both, `/root/nft-ruleset-20261127-0606-pre-m7.nft` and `/root/iptables-20261127-0606-pre-m7.rules`. Firewall: `table inet ymir_ray` (copy in `/root/ymir-ray.nft`) drops TCP/UDP 6390-6399 and 10000-10999 except on `lo` and the two private links. It was applied with a 5-minute automatic rollback, which was cancelled once SSH, Tailscale, the link and Weirdnet were checked. It is **not persistent** (gone at reboot).
- **Installer, for real:** `ymirctl engines install vllm --uv ~llm/.local/bin/uv` installed vLLM `0.34.1rc1.dev273+gef706a1b8` and Ray 2.60.1 into `/var/lib/llm/ymir/engines/vllm` on both; the record says nightly and Inkling. Inkling 2 Mini weights: `hf download thinkingmachines/Inkling-2-Mini` into `/srv/models/huggingface` on both (19 GB each, about 2.5 min).
- **Run:** `group up` at 06:20:51, up at 06:23:51. `ray status`: 2 active nodes, 2/2 GPU in the placement group. Chats through the pool (`ymirctl chat --model inkling-2-mini-ray`): "pong" in 7.0 s, a 3-sentence answer in 11.0 s, about 400 words in 20.5 s; vLLM reported 52.5 tokens/s generation. `group down`: 20.6 s, then no Ray or vLLM process on either node; only m1's stand-in `llama-server` was still on the GPU.
- **What it took (all in DESIGN):** tp 2 / pp 1 (no PP for Inkling), `size` above one node's `memory`, a Python-frontend wrapper (the Rust frontend fails on Inkling), text only (torch cu132 PTX vs driver 580), `ray_port` 6390 (Redis has 6379 on m1).
- **Security gap (open):** the listener check sees only the `ray start` process group. Ray's workers (vLLM, NCCL, torch.distributed) listened on all interfaces outside the pinned ports, on m1's LAN address (192.168.1.218) and on m2's Tailscale address (100.92.9.99). The group was not refused. The run was kept short and torn down. Fix next: check Ray's workers by session or process tree, pin them to the link (`NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `VLLM_HOST_IP`), log trusted listeners at info. Until then don't use `ray_trust_firewall`.
- **Installer follow-ups:** choose a torch CUDA build the driver can run (580 → cu130), not the newest; check the checkpoint's own architecture (`Inkling2LoopForConditionalGeneration`) as well.
- **Left on the nodes:** both `node.toml` files restored from the backups (the test config is in `node.toml.m7-test-20261127`); the managed venv, uv in `~llm/.local/bin`, the Inkling weights, the `vllm-pyfrontend` wrapper and the runtime nft table stay. m2's `ymird.service` reports its unit file changed on disk; that was there before this run.

## 2026-11-26 18:38 EST: v0.22.0 released

**Release:** [v0.22.0](https://github.com/WeirdBrains/ymir/releases/tag/v0.22.0) (release commit 70ff1e9 on e704ad8; Release run 45905897897 green, CI, macOS, Pages and FreeBSD green on e704ad8): M7 vLLM with Ray groups (`backend = "vllm-ray"`, private link only, mocks only), `ymirctl engines install vllm` (a managed vLLM nightly that runs Inkling 2 Mini, plus Ray), and video limits in catalog/1 (LF-43). 44 files on GitHub: leak scan 33 files, 0 hits, then 4 Arch `.sig`s and 6 signed update bundles plus their sums (`sign-release.sh 0.22.0` EXIT 0); `weirdnet_release.py assets 0.22.0` ok for linux, macos, windows and both FreeBSD files. Arch recipes 1847744 on main; TAUR ymir-bin and ymir-cuda-bin 0.22.0-1 (dry-run first). Committed straight to main (Michael: no PRs). No live nodes touched. The dev box was replaced mid-release (about 17:20 EST) and its alien-blue key had to be re-authorized; nothing was lost.

**Next:** a real GPU test of `vllm-ray` and of the vLLM nightly with Inkling 2 Mini waits on Michael picking hardware. When vLLM 0.35.0 is on PyPI, the installer can drop the nightly default.

## 2026-11-26 16:59 EST: vLLM installer defaults to a nightly that runs Inkling 2 Mini (on main)

Michael (11-26): "Make sure it is a version of vLLM that can run Inkling 2 mini". No live node touched.

- Requirement, verified 2026-11-26: Inkling 2 Mini needs vLLM with [vllm#69028](https://github.com/vllm-project/vllm/pull/69028) (merged 2026-11-21, `ab18334`). No release has it: PyPI's newest is 0.34.0 (2026-11-16), which lacks it, and the [recipe](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini) says "vLLM 0.35.0+ nightly" (nightly wheel or `vllm/vllm-openai:nightly`). The nightly index then served `0.34.1rc1.dev231+g43108ffcc`, 341 commits past `ab18334`, whose registry has `InklingForCausalLM` and `InklingForConditionalGeneration`. The recipe pins no `transformers` or Ray version.
- [vllm_install.rs](crates/ymir-core/src/vllm_install.rs): the default install adds `--extra-index-url https://wheels.vllm.ai/nightly` (uv `--prerelease=allow`, pip `--pre`); `--version V` installs a PyPI release and `--index-url` replaces the nightly. After installing it runs the venv's Python against vLLM's `ModelRegistry` and records `inkling` and `nightly` in `ymir-install.json`; a nightly install without Inkling fails, a pinned release without it says so.
- [vllm.rs](crates/ymir-core/src/vllm.rs): `managed_with_inkling`, so ymird's Inkling preflight accepts the managed venv's `vllm` (numbered below 0.35.0) when the record says Inkling. `INKLING_MIN_VERSION` stays 0.35.0 for other binaries.
- Docs: [DESIGN](docs/DESIGN.md) ("Inkling 2 Mini" and "Installing vLLM and Ray"), site ymirctl docs `#engines`, README, CHANGELOG.

**Checks:** tests/engines.rs 2/2 (pinned 0.34.0 records no Inkling; default nightly records Inkling and passes `ensure_inkling`; status), vllm unit tests, fmt, clippy `-D warnings`, workspace tests, link check.

**Not tested:** a real nightly download (PyTorch-sized; not run in tests or CI), the ROCm nightly index.

## 2026-11-26 16:30 EST: `ymirctl engines install vllm` (on main)

Michael (11-26): "Also, make installing vLLM easy with Ymir". Built on the dev box; no live node touched.

- [vllm_install.rs](crates/ymir-core/src/vllm_install.rs) and [engines.rs](crates/ymirctl/src/engines.rs): `ymirctl engines install|upgrade|status|uninstall vllm`. It makes a venv at `<home>/engines/vllm` (uv `venv --python 3.12` plus `pip install --torch-backend=auto|cpu`, else `python3 -m venv` plus pip, refusing Python outside 3.10 to 3.13) and installs `vllm[==V] ray`. The accelerator is detected (nvidia-smi, rocminfo or /opt/rocm, else CPU) or set with `--accel`; ROCm and CPU need `--index-url`, since PyPI's vLLM wheels are CUDA builds. Flags: `--version`, `--uv`, `--python`, `--reinstall`, `--force` (non-Linux), `--dry-run`. After installing it checks `vllm --version` and `ray --version` and writes `ymir-install.json`. Uninstall removes the venv only when that record is there.
- Discovery was wired in 04ef62d: `vllm = "auto"` and `ray = "auto"` use `engines/vllm/bin/` before `PATH`. `ymirctl setup` now points at the installer when vLLM is missing.
- Docs: README, site ymirctl docs (`#engines`, and the group section names `vllm-ray`), DESIGN, CHANGELOG.

**Checks:** [tests/engines.rs](crates/ymirctl/tests/engines.rs) 2/2 (fake `uv` and `python3`: dry run makes nothing; install, record, discovery by `resolve_vllm`/`resolve_ray`/`worker_config`, explicit settings win, status, upgrade without a new venv, uninstall, a foreign dir left alone, Python 3.8 refused, CPU without an index refused), 8 unit tests, fmt, clippy `-D warnings`, workspace tests (63 suites green), link check.

**Not tested:** a real install (it downloads PyTorch; not run in tests or CI), the ROCm and CPU wheel indexes.

## 2026-11-26 16:14 EST: M7 vLLM with Ray groups (on main, mocks only)

Michael (11-26): "Let's start on the Ray groups then!" Built on the dev box only; m1, m2 and starbase not touched.

- [ray.rs](crates/ymir-core/src/ray.rs) and `RpcWorkers::up_ray` in [group.rs](crates/ymir-core/src/group.rs): `[[inference.split]] backend = "vllm-ray"` plans worker kind `ray`, with `ray.port`, `vllm.tp` and `vllm.pp` in the plan. All members get `GroupUp` at once. The leader runs `ray start --head --block` on its `private_link`, waits for every member's `ray start --block --address` (`ray status`), then runs `vllm serve --distributed-executor-backend ray` on loopback (key in `VLLM_API_KEY`). Sizing: tensor parallel is the fewest GPUs of any member (`gpus` per node, default 1), pipeline parallel is the member count; both can be overridden, and the planner refuses a `TP x PP` that does not fit.
- Security: Ray binds the link only, `ray_args` may not touch the bind flags, and Ymir lists each Ray process group's TCP listeners (`/proc`) after start, join and vLLM ready, and on every watch tick. Anything off the link or loopback refuses the group and tears it down. Real Ray's GCS listens on all interfaces anyway, so a real cluster needs a host firewall plus `[model] ray_trust_firewall = true` (off by default). **Michael decides** on that flag and the firewall when he picks hardware. Linux only.
- Teardown: every group process leads its own process group; stop = SIGTERM to the group, 10 s, SIGKILL; dropping a handle kills the group; `ymird` shutdown stops all groups (and a start still waiting gives up). If the leader fails, the coordinator stops waiting on members and takes down any that come up later.
- `[model] vllm = "auto"` and new `ray = "auto"` prefer the managed venv `engines/vllm/bin/` before `PATH`. The `ymirctl` installer for that venv comes next.
- Mocks: [ymir-mock-ray](crates/ymir-testkit/src/bin/ymir-mock-ray.rs) (head with a child in its process group; `--mock-bind-any` listens on 0.0.0.0 like the GCS); the mock engine now also acts as `vllm serve` (Ray executor: refuses when the cluster has fewer nodes than `--pipeline-parallel-size`).

**Checks:** fmt, clippy `-D warnings`, `ymir-core` lib 201, [m7_ray.rs](crates/ymir-testkit/tests/m7_ray.rs) 2/2 (up, chat, down, member lost, all-interface bind refused; the head's child gone each time), `m7.rs` 1/1, full workspace run (60 of 61 suites; `m5` `an_inference_node_dying_mid_answer_is_resumed_with_the_full_context` failed once under load with `no-epoch-key`, then m5 passed 4/4), link check. Docs: DESIGN (new "M7 vLLM with Ray groups" section, security), CHANGELOG, ROADMAP/roadmap.toml/site roadmap, whitepaper.

**Not tested:** real Ray and vLLM (versions, NCCL over RoCE, real `ray start --block` under SIGTERM), UDP listeners.

## 2026-11-26 15:10 EST: video milestone 1, context and media limits in catalog/1 (merged `feature/video-upload`, LF-43)

Michael (11-26): "Let's do the video uploads". Bragi samples clips into frames sized from the limits Ymir reports and refuses when they are missing (Bragi merged the same day), so Ymir's part is milestone 1 of [docs/design/video.md](docs/design/video.md): report limits, never invent them. Weirdnet Lead published weirdnet-agent 0.5.7 (`MediaLimits`, `CatalogModel::limits`), weirdnet-backends 0.4.8 (`Capabilities::limits`, `ModelLaunch::limits`, `ModelStatus::limits`) and weirdnet-openai 0.2.4 (`Bridge::with_max_request_bytes`, 413 with `x-weirdnet-max-request-bytes`, `/v1/models` `weirdnet.limits`) for this; Cargo.toml and release.toml `[[requires]]` bumped.

- [model_node.rs](crates/ymir-core/src/model_node.rs): `request_context(engine, context, slots)`, the context one request can count on (llama.cpp's `--ctx-size` split over its slots), `None` when none is set; `limits_line` for status text.
- [coordinator.rs](crates/ymir-core/src/coordinator.rs): `CatalogRow::context` is the smallest across the nodes that have the model, `None` when any of them doesn't say; `CatalogRow::limits` folds every node's `ModelStatus::limits()` with `MediaLimits::tightest` from `UNKNOWN`, so a field nobody reports stays absent. catalog/1 entries carry both (`with_context`, `with_limits`).
- [config.rs](crates/ymir-core/src/config.rs): `[model.local] limits = { max_images = 8, image_tokens = 256, image_edge = 896 }` goes into `ModelLaunch::limits`; `[bridge] max_request_bytes` feeds `Bridge::with_max_request_bytes` in [daemon.rs](crates/ymir-core/src/daemon.rs) (unset keeps the crate default).
- [ymirctl status](crates/ymirctl/src/lib.rs): `context N per request` / `context not reported`, `limits: ...` / `media limits not reported` for chat models.
- Tests in [tests/nodes.rs](crates/ymir-core/tests/nodes.rs): `the_catalog_reports_the_context_a_request_can_count_on_and_never_invents_one`, `the_catalog_reports_the_strictest_media_limits_and_leaves_unknown_ones_out`. Docs: DESIGN "Limits for media", [video.md](docs/design/video.md) status, CHANGELOG.

**Checks:** on alien-blue, after merging main v0.21.0: `cargo fmt --all --check`; clippy `--workspace --all-targets -D warnings`; `cargo test --workspace --no-fail-fast` 61 suites, 410 passed, 1 failed, 7 ignored. The failure was `m5` `an_inference_node_dying_mid_answer_is_resumed_with_the_full_context` (`refused: no-epoch-key` while warming up) with the box at load 60 to 97 from other work; `m5` alone then passed 4/4. The run before had `conversations_sync_and_any_inference_node_answers_from_them` fail the same way once and pass alone; the same m5 load flake as on 10:45. The new catalog tests pass. `release_requires` caught release.toml still asking for agent 0.5.6 and backends 0.4.7; both are bumped.

**Next:** transcription route (milestone 2); the remaining Weirdnet asks (`weirdnet-media`, a video part and modality, a transcription call, `weirdnet-mcp` content blocks, an ffmpeg recipe).

## 2026-11-26 15:05 EST: v0.21.0 released

**Release:** [v0.21.0](https://github.com/WeirdBrains/ymir/releases/tag/v0.21.0) (release commit 6f087b3 on 5e2b602; Release run 45870102685, green, the first with the FreeBSD job): M6 semantic answer cache (`weirdnet-semcache` 0.2.2), one sync session per device (`weirdnet-sync` 0.8.5 `pull_types_shared`), M9 FreeBSD rc.d service and `x86_64-unknown-freebsd` `.tar.gz` and `.pkg` (helpers 0.4.1 FreeBSD `[[platforms]]`). 44 files on GitHub: leak scan 33 files, 0 hits, then 4 Arch `.sig`s and 6 signed update bundles plus their sums (`sign-release.sh 0.21.0` EXIT 0; no FreeBSD update bundle, FreeBSD is GitHub-only); `weirdnet_release.py assets 0.21.0` ok for linux, macos, windows and both FreeBSD files. Arch recipes 614d11b on main; TAUR ymir-bin and ymir-cuda-bin 0.21.0-1 (dry-run first). Committed straight to main (Michael: no PRs). No live nodes touched. `pub.sh`'s download ran at ~0.5 MB/s on alien-blue for most of the afternoon, so the publish took about 2 h.

**Next:** M7 vLLM with Ray groups waits on Michael (GPU hardware; Ray has no auth). M9: OpenBSD (opted out in `release.toml`), GhostBSD and DragonFly not started; a FreeBSD self-update bundle and aarch64 FreeBSD are not built. M6: pool-mode semantic cache, Redis backend, persisting entries.

## 2026-11-26 10:45 EST: M6 semantic answer cache; one sync session per device (merged `feature/semcache`)

- [semantic.rs](crates/ymir-core/src/semantic.rs) on `weirdnet-semcache` 0.2.2 `SemanticCache`: `[inference.cache] embedding_model = "ID"` (plus `threshold`, a cosine distance, default 0.12; `ttl` `24h`; `entries` 1024) adds a semantic layer behind the exact cache in `Coordinator::run`. Cacheable: one user turn after any system messages, text only, no tools, no prefix, no stand-in. Scope: model + weights revision + embedding model + its revision + SHA-256 of system prompt, sampling, seed, response format, adapter scale. The embedding model is used only while loaded (never cold-started); a failed embedding skips the cache. Hits are `served_by = "semantic-cache"`; `cache: bypass` / `refresh` apply. `[inference.cache]` is now validated. Tests: [tests/semantic.rs](crates/ymir-core/tests/semantic.rs) (4) and 3 unit tests.
- [membership.rs](crates/ymir-core/src/membership.rs): `tick` pulls roster, conversations, wake cards and training jobs in one session per device with `weirdnet_sync::pull_types_shared` (0.8.5), not one `pull_from_shared` per type. `release.toml` `[[requires]]`: weirdnet-sync 0.8.5, weirdnet-semcache 0.2.2 (the `release_requires` test caught the stale one).
- Bragi main b4383e7 has the same sync change (no release cut).

**Checks:** fmt, clippy `-D warnings`, `cargo test --workspace --no-fail-fast`: 59 of 61 suites ok in the full parallel run; `m5` `conversations_sync_and_any_inference_node_answers_from_them` and `p3b` `on_the_synced_board_a_run_that_loses_its_lease_stops_and_writes_nothing_more` failed once there under load, then passed alone (2/2 each) and as whole files (m5 4/4, p3b 8/8 twice). Watch them on CI. Docs: CHANGELOG, DESIGN (cache section, config example, M6 status, gaps), ROADMAP/roadmap.toml/site roadmap, site docs `#semantic-cache`.

**Not built:** the pool-mode (shared, sharded) semantic cache, the Redis backend, persisting entries across restarts.

## 2026-11-26 10:30 EST: M9 FreeBSD service and packages (merged `wip/freebsd`)

- [service.rs](crates/ymir-core/src/service.rs): `Os::FreeBsd`. `service install --system --run-as USER` writes `/usr/local/etc/rc.d/ymird` (rc.subr; `daemon(8) -r -R 5 -u USER` runs `ymird --home DIR`, output to `state/logs/ymird.log`), runs `sysrc ymird_enable=YES` and `service ymird restart`; stop/start/status go through `service ymird`; uninstall stops it, `sysrc -x ymird_enable` and removes the script. User scope is refused (FreeBSD has no per-user services). The system node folder defaults to `/var/db/ymir`. Unit test checks the script with `sh -n`.
- [scripts/package.py](scripts/package.py): `--target x86_64-unknown-freebsd` (or aarch64) writes `ymir-V-TRIPLE.tar.gz`, and with `pkg` on the host a `ymir-V-TRIPLE.pkg` (`pkg create`, message pointing at `misc/llama-cpp`). No bundled engine for FreeBSD.
- [.github/workflows/freebsd.yml](.github/workflows/freebsd.yml): FreeBSD 14.2 VM (cross-platform-actions 0.29.0): build, test `ymir-core` lib and `ymirctl`, package, `pkg add`, then the rc.d service as user `ymir` through a chat with the mock engine, stop (no engine left), uninstall, `pkg delete`. Runs on dispatch and on pushes to main that touch those files.

**Checks:** fmt, clippy `-D warnings`, `ymir-core` lib (188) and `ymirctl` tests on the box; `package.py` with fake binaries (tar.gz made, pkg skipped: no `pkg` on Linux). A FreeBSD build cannot be cross-checked from Linux (aws-lc-sys needs a FreeBSD sysroot), so the VM job is the first real FreeBSD build. Docs: CHANGELOG, DESIGN BSD gap, README, site docs.

**Next:** the FreeBSD job's first run; then FreeBSD in the Release workflow and `release.toml` platforms. OpenBSD, GhostBSD and DragonFly not started.

## 2026-11-26 10:25 EST: v0.20.0 released

**Release:** [v0.20.0](https://github.com/WeirdBrains/ymir/releases/tag/v0.20.0) (release commit 7cec694 on 1a934ac, the `--no-ff` merges of `docs/bragi-status-0.15` ddb2210, `feature/push-wake` 77d35dc and `feature/m6-placement` 1a934ac; Release run 45838670704, green): M6 `ymirctl move` (also the `ymir_move_model` MCP tool), placement with replica counts under the `ymir.scheduler` lease (`[inference.placement]`), and push wake-ups of Bragi devices from seed holders. 42 files on GitHub: leak scan 31 files, 0 hits, then 4 Arch `.sig`s and 6 signed update bundles plus their sums (`sign-release.sh 0.20.0` EXIT 0); `weirdnet_release.py assets 0.20.0` ok. Arch recipes f9bf1cd on main; TAUR ymir-bin and ymir-cuda-bin 0.20.0-1 (dry-run first). Committed straight to main (Michael: no PRs). No live nodes touched.

**Next:** M9 FreeBSD (rc.d service, `package.py` tar.gz/pkg, FreeBSD VM CI job) and the M6 semantic cache (`weirdnet-semcache` 0.2.2) with one sync session per device (`weirdnet-sync` 0.8.5 `pull_types_shared`), merging to main next.

## 2026-11-26 09:00 EST: v0.19.0 released

**Release:** [v0.19.0](https://github.com/WeirdBrains/ymir/releases/tag/v0.19.0) (release commit c72762d on 0a3e0b0, the `--no-ff` merge of `feature/blobs`; Release run 45824407253, green): training files over Weirdnet (P3b blob transfer, `/ymir/blobs/1`): another node resumes a board job with its dataset and checkpoint fetched from the devices that hold them, each file checked against its manifest, and registers itself as a holder. The cut ran every check before pushing (fmt, clippy, 55 test suites). The leak scan of dist found nothing (31 files, 0 hits); the GitHub release has 31 files plus the four gpg signatures of the Arch packages and the six signed update bundles with their sums (42 files); the assets check passes. TAUR has ymir-bin and ymir-cuda-bin 0.19.0-1; their recipes are on this branch (`chore/release-0.19.0`).

**Known flake:** `train_board.rs` `a_lapsed_board_claim_is_taken_over_after_the_wait` failed once on CI (45795286711, add1902, Linux): the cut-off device kept renewing its lease. Likely cause: a paused weirdnet-net node still dials out to addresses it remembers, and a resumed `/tcp/0` listener can get its old port back, so the test's `isolate()` did not cut it off. Fix on `fix/board-isolate` (875955a): `isolate()` brings the other devices back on addresses the cut-off one never saw; train_board 3/3 locally, going through CI after this release. Separately, main's push CI 45815845665 (79b55b6) failed once on Windows only with the runner's disk full in kvfit's sparse test files (NTFS does not keep `set_len` files sparse); the rerun passed.

**Next:** merge `fix/board-isolate` once green. Two-node training on real nodes waits for Michael.

## 2026-11-26 08:50 EST: M6 placement and replication, `ymir_move_model`, a daemon test for `ymirctl move` (branch `feature/m6-placement`, LD-3)

- [placement.rs](crates/ymir-core/src/placement.rs): `[inference.placement] replicas = { MODEL = N }` (`every_secs`, default 15). `plan` takes one step per model per round: load on the node with its own copy and the fewest loaded models (skipping draining, reserved, being-moved, failed and swap-group-evicting nodes), or unload an idle extra (never one with answers in flight, never below N). `Coordinator::place_once` / `spawn_placement`. Counts of 0 or over 64 are refused in `NodeConfig::validate`.
- Who places ([daemon.rs](crates/ymir-core/src/daemon.rs) `placement_leader`): a cert-only inference node (`ymir.inference` grant) votes on and stands for the `ymir.scheduler` lease (`Membership::enable_scheduler_voter`, terms in `voter.cbor`; a service grant has one role, so it never needs the training voter too) and places while `Elector::handle()` is valid; a seed holder places only while `scheduler_voters() == 0`; `scheduler_eligible = false` never places. Without Weirdnet it places alone.
- [mcp.rs](crates/ymirctl/src/mcp.rs): `ymir_move_model` (MUTATES THE POOL, `confirm: true`, `grace` capped at 600 s), hidden by `--read-only`.
- **Tests:** `placement::tests` (5); [m6_move.rs](crates/ymir-testkit/tests/m6_move.rs) (daemons: the move waits about 2.9 s for a slow answer on the source, which finishes whole; refusals change nothing; the MCP tool refuses without confirm and moves it back); [m6_placement.rs](crates/ymir-testkit/tests/m6_placement.rs) (three model nodes: two replicas come up, stay at two for 8 s, a drain brings up the third, a resume settles back to two; 3 of 3 runs); [m6_scheduler_lease.rs](crates/ymir-testkit/tests/m6_scheduler_lease.rs) (two cert-only inference nodes: one holder, the other takes over with a higher term). The daemon helpers from `m6.rs` moved to [tests/common/pool.rs](crates/ymir-testkit/tests/common/pool.rs). Checks: fmt, clippy `-D warnings`, `cargo test --workspace` (59 suites green), links.
- **Not done (M6):** publishing the plan (`publish_plan` under a sealed `ymir.plan` type; cert-only inference grants would need to list it), fencing loads/unloads on model nodes (`FenceGuard`), lease participation for seed-holding inference nodes (they need an `ymir.inference` grant), cloud-wide swap, prefix warming, the semantic cache (needs `weirdnet-semcache`; asked of the Weirdnet lead).
- **M7 audit (vLLM with Ray groups):** the protocol is ready (`GroupUp`/`GroupDown` in weirdnet-backends 0.4.7 carry a worker kind string; `RpcWorkers::up` answers `Unsupported` for `ray`), so no crate change is needed for the plumbing. To build: a Ray worker kind (`ray start --head` on the leader, `ray start --address` on members, bound to the private link), a vLLM leader (`vllm serve --distributed-executor-backend ray` with tensor/pipeline sizes from `plan_group`), teardown like RPC groups, and a mock `ray`/`vllm` for the testkit. Risk: Ray has no authentication of its own (GCS, dashboard and worker ports would let anyone on the link run code), so it must stay private-link-only with the ports fenced, which needs an owner decision. A real check needs two GPU machines on a fast private link (server-a/server-b or a RoCE pair), so it waits for Michael.
- **M9 audit (BSD packages):** not built. Needs a FreeBSD CI job (a VM, as Bragi's `platforms.yml` `freebsd` job), `scripts/package.py` support for a FreeBSD `tar.gz` and a `pkg` (`+MANIFEST`), an `rc.d` script in `ymirctl service install` (systemd, launchd and Windows only today), and `llamacpp = "system:..."` from FreeBSD's `misc/llama-cpp` port (upstream publishes no BSD builds; no vLLM). OpenBSD: `tar.gz` only. A cross `cargo check --target x86_64-unknown-freebsd` from Linux stops at `aws-lc-sys` (C, via `weirdnet-net` → libp2p-quic → quinn → aws-lc-rs) without a FreeBSD sysroot, so the build check belongs in the FreeBSD VM job; Bragi's FreeBSD job already builds the same `weirdnet-net` stack natively.
- **Not merged:** Ymir main is LD-1's until v0.19.0 is DONE.

## 2026-11-26 08:10 EST: moving a model between nodes, M6 (branch `feature/move-model`, LD-3)

- [coordinator.rs](crates/ymir-core/src/coordinator.rs): `Coordinator::move_model(model, from, to, grace)`: loads on `to` (refused when `to` has no copy), takes `(source, model)` out of `pick` (`leaving`), waits until the source reports nothing in flight for the model (at most `grace`), unloads it there. `ControlRequest::Move` and `ymirctl move MODEL --to NODE [--from NODE] [--grace SECS]`.
- **Test:** [crates/ymir-core/tests/move_model.rs](crates/ymir-core/tests/move_model.rs): a request running on the source finishes there, a request during the move goes to the target, the source unloads after; refused for a target without a copy, a model loaded nowhere, the same node, an unknown node, and a failed load on the target (the source keeps serving). clippy `-D warnings` on the workspace, ymirctl tests.
- **M6 audit (rest):** placement and replication need a lease and a synced plan; weirdnet-sync 0.8.4 has leases now (the training board uses them), so a placement plan is buildable without a crate change. The semantic cache needs an embedding-keyed cache in `weirdnet-semcache` (0.2.1 has only the exact cache): a Weirdnet crate change.
- **Not done:** a daemon-level test of `ymirctl move` over loopback; an MCP tool for it. Not merged: Ymir main is LD-1's for v0.19.0.

## 2026-11-26 07:45 EST: push wake-ups, Ymir side (branch `feature/push-wake`, LD-3)

- [membership.rs](crates/ymir-core/src/membership.rs): `WAKE_TYPE = "bragi.wake"`, configured (sealed) on full devices only; a sync pass pulls it from full devices every `WAKE_PULL_EVERY` (2 min: each type pulled is a sync session of its own), and `refresh_wake_cards` makes the next pass pull again. `ServiceRole::types` never lists it and `service_bundle` refuses a grant that would (unit test `no_service_grant_lists_the_wake_cards`).
- [wake.rs](crates/ymir-core/src/wake.rs): `DeviceWake` (the record Bragi 0.15.0 writes), `wakeable` (active full devices' cards; empty on a cert-only node), `wake` (seals with the card's key, delivers through the device's relay on `/weirdnet/wake/1`), `find` (peer id or label).
- `ymirctl pool wake [DEVICE] [--connection]`: lists offline from the identity; sending goes through the running node (`ControlRequest::PoolWake`).
- **Test:** [crates/ymir-testkit/tests/wake.rs](crates/ymir-testkit/tests/wake.rs): a seed holder, a full "Bragi" device and a cert-only inference node on loopback with a stand-in wake relay: the card reaches the holder by `tick`, the inference grant has no `bragi.wake`, its pull is refused and it holds none, a wake before the device subscribes is `Offline`, after it is `Delivered` and opens with the device's key, a withdrawn card leaves nothing to wake. clippy `-D warnings` on the workspace.
- **Not done:** `ymird` waking a device on its own (when an agent task or a training job finishes for it) — needs a decision on which events; APNs/FCM; a check against the public relay (does it serve `/weirdnet/wake/1`?). Not merged: Ymir main is LD-1's for v0.19.0.

## 2026-11-26 07:05 EST: training files over Weirdnet, P3b blob transfer (branch `feature/blobs`)

- [blobs_net.rs](crates/ymir-core/src/blobs_net.rs): `/ymir/blobs/1` (CBOR): `BlobAsk { cert, root }`; the holder answers `Bundle { manifest_name, manifest }`, each listed file's bytes as 48 KiB `Chunk`s in manifest order, `Done` (or `Missing` / `Refused`). `BlobsService` (roster auth: the owner's linked devices) and `BlobsServer`; `BlobsClient::ensure(root, holders)` checks the manifest against the root, takes the file list from the manifest (paths inside the bundle only), checks each file's size and sha256 while writing into a staging folder in the store, gives up on a holder after 60 s of silence and tries the next, and stores only a whole bundle (`LocalBlobs::put_checked`); staging is removed either way.
- [train_job.rs](crates/ymir-core/src/train_job.rs): `Ctx.fetch` (`BlobFetch`); before a run, the dataset from the job's owner or its checkpoint's holders and the last checkpoint from its holders; a fetch failure fails the attempt as retryable. A node that resumes from a checkpoint it did not hold adds itself to the holders (`JobStore::set_holders`: file store, board via `Claim::set_holders`, control socket). `LocalBlobs::bundle` / `staging` / `put_checked`.
- [daemon.rs](crates/ymir-core/src/daemon.rs), [remote.rs](crates/ymir-core/src/remote.rs), [control.rs](crates/ymir-core/src/control.rs), [train_jobs_ctl.rs](crates/ymir-core/src/train_jobs_ctl.rs): every networked `ymird` serves its training store on `/ymir/blobs/1` and holds a `BlobsClient` into it; `JobCall::Fetch` / `SetHolders`; `cli_ctx` gives `ymirctl train` the daemon as its fetcher.
- Tests: blobs_net unit tests (a lying holder: flipped byte, long file, short file, a manifest for another root; refusal, unknown root, fallback past an unreachable holder and a liar to a good one, no staging left); [tests/train_blobs.rs](crates/ymir-testkit/tests/train_blobs.rs): three ymirds on 127.0.0.1 with separate homes and stores (a seed holder and two cert-only model nodes): gpu1 posts a 3-epoch job and loses its GPU after epoch 1; gpu2 resumes it through its ymird, which fetches the dataset and the epoch-1 checkpoint from gpu1 (gpu2's store had neither), registers gpu2 as a holder, and finishes it; the holder sees it done (~66 s, mostly the voters' 60 s restart grace).

**Checks:** workspace clippy `-D warnings`; blobs_net 3/3, train_blobs 1/1, p3b, train_daemon, train_isolation, train_board (see the CI runs below).

**Not done:** a broken transfer starts the bundle again (no partial resume); no bandwidth cap or progress events; fetched bundles stay in the store until it is cleaned like any other; nodes on 0.18.0 or older don't serve `/ymir/blobs/1` (the fetch fails over to the next holder, else the attempt is retried). Two-node training on real nodes waits for Michael. No live nodes were touched.

## 2026-11-26 06:45 EST: v0.18.0 released

**Release:** [v0.18.0](https://github.com/WeirdBrains/ymir/releases/tag/v0.18.0) (release commit 35188d7 on add1902; Release run 45802019717, green): training jobs on the synced job board between nodes (P3b), `ymird` holding the board for `ymirctl train`, model nodes granted `["ymir.pool", "ymir.train"]` voting on and claiming jobs without access to conversations, and the fix for concurrent saves of a device's Weirdnet state (the `p2.rs` failure under load), on weirdnet-sync 0.8.4. The cut ran every check before pushing (54 test suites); its first try stopped before the commit because `release.toml`'s default `CARGO_TARGET_DIR` was not writable on the build machine, and the second set its own. The leak scan of dist found nothing (31 files); the GitHub release has 31 files plus the four gpg signatures of the Arch packages and the six signed update bundles with their sums (42 files); the assets check passes. TAUR has ymir-bin and ymir-cuda-bin 0.18.0-1; their recipes are on this branch (`chore/release-0.18.0`).

**Next:** P3b blob transfer: a job's dataset and checkpoints fetched over Weirdnet from the devices that hold them, so another node resumes a board job without copying files by hand.

## 2026-11-26 04:05 EST: v0.17.0 released

**Release:** [v0.17.0](https://github.com/WeirdBrains/ymir/releases/tag/v0.17.0) (release commit 90f71fb on 8bfe741; Release run 45776523009, green): model-upload U10, `[model] upload = false`, on weirdnet-backends 0.4.7 and weirdnet-net 0.8.3. The jobboard work waits for the next release. The cut ran its checks before pushing; its first try stopped on one failure of `p2.rs` (`only_marked_conversations_are_exported_and_withdrawal_is_reported`: the pool roster's directory missing) while the build machine was heavily loaded, which passed three times alone and on the second try. The leak scan of dist found nothing (31 files); the GitHub release has 31 files plus the four gpg signatures of the Arch packages and the signed update bundles with their sums; the assets check passes. TAUR has ymir-bin and ymir-cuda-bin 0.17.0-1; their recipes are on this branch (`chore/release-0.17.0`).

**Next:** v0.18.0 with the training lead's jobboard (weirdnet-sync 0.8.4).

## 2026-11-26 03:35 EST: model nodes on the job board, ymird holds it (branch `feature/jobboard`)

- weirdnet-sync 0.8.4 (Cargo.toml, Cargo.lock, release.toml `[[requires]]`): per-type keys for service devices (0.8.3) and type-scoped sealed records (0.8.4). On 0.8.3 alone m10 failed: the model node's grant lacked `ymir.pool` (refused `type-not-granted`, fixed here) and a cert-only inference node could not open sealed `ymir.context` records (fixed in weirdnet-sync 0.8.4).
- [membership.rs](crates/ymir-core/src/membership.rs): `TRAIN_TYPE` (`ymir.train`, unsealed); `ServiceRole::Model` grants `["ymir.pool", "ymir.train"]`; a cert-only device syncs the pool and every type in its grant; full devices serve `ymir.train`; the sync pass pulls it with peers that hold it; `enable_train_voter(_with)` (model role and `ymir.train` in the grant; limits from the job policy; terms in `voter.cbor`).
- [daemon.rs](crates/ymir-core/src/daemon.rs): the model role's voter at network start; `Network.jobs`, a `BoardJobStore` with hint-addressed leases, for the control socket and `/ymir/train-status/1`.
- [train_jobs_ctl.rs](crates/ymir-core/src/train_jobs_ctl.rs): `ControlRequest::Jobs { call }` / `ControlResponse.job`; `ControlJobStore` (the `JobStore` over the socket; claims only as the daemon's own peer id, renewing in ymird); `cli_ctx` for `ymirctl train` (board when ymird is up on Weirdnet, else the file store). `TrainStatus::read_with` in [train_net.rs](crates/ymir-core/src/train_net.rs).
- Tests: [tests/train_isolation.rs](crates/ymir-testkit/tests/train_isolation.rs) (a cert-only model node syncs, votes on and claims a job the seed holder posted; no epoch key and no `ymir.context` key; its context pull fails; a sealed context entry handed to it does not open), [tests/train_daemon.rs](crates/ymir-testkit/tests/train_daemon.rs) (two ymirds on 127.0.0.1: post on the holder, claim, renew, checkpoint and finish on the cert-only model node over the control socket, `train status` lists it; ~64 s, mostly the voter's 60 s restart grace).

**Checks:** workspace clippy `-D warnings`; m10 2/2, p3b 8/8, p3a_remote 2/2, train_board 3/3, train_isolation 1/1 (3 runs), train_daemon 1/1; CI 45782530375 and macOS 45782535575 green on 93414c4, Pages 45784606192 green on cc4252b (a rustdoc link fix).

**Not done:** datasets and checkpoints stay in the writing node's `LocalBlobs`, so another node resumes a board job only if it has those files; blob transfer over Weirdnet is the follow-up after 0.18.0. Already linked cert-only model nodes keep their old roster-only grant until re-linked from a seed holder. No live nodes were touched.

## 2026-11-26 01:45 EST: training jobs on the synced job board (branch `feature/jobboard`)

- [crates/ymir-core/src/train_board.rs](crates/ymir-core/src/train_board.rs): `BoardJobStore`, Ymir's `JobStore` over `weirdnet-agent` 0.5.6's `JobBoard<TrainJob>` (records, policies, checkpoint pointers and claim errors mapped; digests as plain hex; live claims kept by job id, renewing themselves).
- [crates/ymir-core/src/train_job.rs](crates/ymir-core/src/train_job.rs): `claim`, `finish`, `fail` and `release` are async; `ClaimError::NoQuorum`, retried for `CLAIM_QUORUM_WAIT` (30 s) when a run claims its job.
- weirdnet-agent 0.5.6 (`jobs`), weirdnet-sync 0.8.2, on top of weirdnet-backends 0.4.7 and weirdnet-net 0.8.3 (one copy of each); `SyncState.voter` in [membership.rs](crates/ymir-core/src/membership.rs).
- Tests: [tests/common/board.rs](crates/ymir-testkit/tests/common/board.rs), three linked devices on in-memory nodes with the model role and voters; [tests/train_board.rs](crates/ymir-testkit/tests/train_board.rs) (a claim held elsewhere, a retryable failure and its back-off, the next device resuming from the last checkpoint, release and the owner's resume, a takeover after a lapsed lease with the stale holder fenced; the same moves on `FileJobStore`); [tests/p3b.rs](crates/ymir-testkit/tests/p3b.rs) runs `train_job` with the mock trainer on the board: a job lost after an epoch carried on by another device, Ctrl-C pausing, a run that loses its lease stopping.

**Checks:** workspace clippy `-D warnings`; `cargo fmt --check`; p3b 8/8 (7 s), train_board 3/3 (17 s), each three times; `train_job` unit tests 7/7; p3a_remote 2/2.

**Not done:** model nodes don't use the board. Their grant is roster only (no sync types), and with weirdnet-sync 0.8.2 a grant with `ymir.train` would hand them the epoch keys that also open `ymir.context`. weirdnet-sync 0.8.3 (per-type keys) comes first; then the grant, the voter on model nodes in the daemon and `train run --job` on the board. The blobs (datasets, checkpoints) still move by the node's own store.

## 2026-11-26 01:10 EST: model-upload U10: `[model] upload = false` on weirdnet-backends 0.4.7 (branch `feature/upload-switch`)

**What changed:** weirdnet-backends 0.4.7 and weirdnet-net 0.8.3 (919204a, the bump alone, which `feature/jobboard` builds on). `[model] upload` (default on; `false` passes `WritePolicy::with_uploads(false)`, so uploads are refused with `not-allowed` while pulls, listing and removals work). The refusal reads "that node has `[model] upload = false`" (`upload_net::UPLOADS_OFF`); `--node auto` tries the nodes with room from the most room down and passes over one with uploads off. `upload::local::entries` uses `StoredEntry::list` (no more serde). `/ymir/uploads/1` takes 16 streams (was 64). Docs: CHANGELOG, the plan ("U10"), DESIGN, the site's ymirctl page.

**Checks:** fmt; clippy `--workspace --all-targets -D warnings`; `ymir-core` unit tests (`upload_net`, including the setting's round trip); `models_pool.rs` (a node with `upload = false` searches, pulls, serves and removes, and refuses an upload with the hint), `models_upload.rs` (`--node auto` passes over that node), `models_upload_formats.rs`, `models_upload_pool.rs`, `rollouts_net.rs`; the full workspace in CI.

**Not tested:** across real machines.

**Next:** merge when CI, macOS and Pages are green; v0.17.0 with `feature/jobboard` (agreed with the training lead).

## 2026-11-26 00:43 EST: v0.16.0 released

**Release:** [v0.16.0](https://github.com/WeirdBrains/ymir/releases/tag/v0.16.0) (release commit 9c6c86b on 149dfe9; Release run 45758480177, green). It folds in what would have been 0.15.5: model-upload U9 (directories and projectors without `--node`, `--node auto`, `--rollout --inference`), resumable training jobs (P3b), the teacher license from the Hub or the pool catalog, `train seed questions|prompts` and training status over `/ymir/train-status/1`. The cut ran its checks (fmt, clippy `-D warnings`, `cargo test --workspace --locked`) before pushing. The leak scan of dist found nothing (31 files); the GitHub release has 31 files plus the four gpg signatures of the Arch packages and the six signed update bundles (`com.weirdbrains.ymir-stable-16000`) with their sums; the assets check passes. TAUR has ymir-bin and ymir-cuda-bin 0.16.0-1; their recipes are on this branch (`chore/release-0.16.0`).

**Next:** adopt weirdnet-backends 0.4.7 and weirdnet-net 0.8.3: `[model] upload = false` with `WritePolicy::with_uploads`, `StoredEntry` built with its constructors instead of serde, and the streams limit back down now that dropped streams are released promptly.

## 2026-11-25 22:15 EST: training status over Weirdnet (branch `feature/train-status`)

- `/ymir/train-status/1` (`ymir_core::train_net::{TrainStatus, RunSummary, JobSummary, StatusReply}`), served by every
  model node on Weirdnet for any linked device (`remote::serve_train_status`), and `TrainHub::status`;
  `ControlRequest::TrainStatus`; `ymirctl train status [--node X] [--json]`. For Bragi's training view.
- Tests: `p3a_remote.rs` (status from an app-only device, of a node with and without `[train] remote`), `p3b.rs` (jobs).

## 2026-11-25 22:05 EST: teacher license lookup and seed prompts (branch `feature/teacher-license`)

- `batch --jsonl` without `--license`: `--license-from OWNER/NAME` reads the Hub repo's model card
  (`MlxHub::license`, `ymir_core::dataset::hub_license`; the token goes to the public Hub only), else the pool
  catalog's `weirdnet.license` (`catalog_license`). `Generate.license_source` records `operator`, `hub:REPO` or `pool`.
- `ymirctl train seed questions|prompts` (`ymir_core::seed`): source texts to question rows, answers to prompt rows.
- Tests: `dataset.rs` (pool-catalog license, both seed passes through the mock pool), unit tests in `seed.rs`,
  `dataset.rs`, `models/hub.rs`. Built on `feature/p3b-resume`.

## 2026-11-25 22:02 EST: model-upload U9: directories and projectors without --node, `--node auto`, rollouts on another inference node (branch `feature/upload-auto-node`)

Michael asked at 9:33 PM ET for `ymirctl models upload` of folders and vision projectors without `--node`, with the node picked automatically (this machine if it serves models, else the pool node with room after 10% or 20 GiB free), and for the `--rollout --inference` gap to be closed.

**What changed**
- `ymirctl models upload DIR|PROJECTOR` without `--node`: with the model role here, placed in this machine's models directory (`ymir_core::upload::local::place`: staged beside the store, hash-checked, moved in with its provenance, keeping `[model] upload_keep_free` free as the node does) and saved in node.toml by the code `/ymir/uploads/1` uses (`upload_net::save_entry`, split out of `UploadsService`); it serves after a restart. Without the model role it goes to `--node auto`. `--projector-for` no longer needs `--node`.
- `--node auto` for uploads (`ModelsHub::upload_auto`): the other model node with the most room after what it keeps free; none with room is an error with each node's numbers. The choice is printed.
- `--rollout NAME --inference NODE` (`RolloutArgs::inference`): version number, incumbent, add and share all on that inference node over `/ymir/rollouts/1`, saved there unless `--no-persist`.
- `/ymir/uploads/1` accepts 64 streams at once (was 8): a finished stream can stay counted for a while, and nine uploads in a row hit the limit in the tests.
- Docs: CHANGELOG, the plan ("U9 as built"), the site's ymirctl page.

**Checks:** `cargo fmt --check`; clippy `--workspace --all-targets -D warnings`; `cargo test --workspace --locked` (CHECKS). End to end: `models_upload_formats.rs` (a directory placed and saved on a model node without `--node`; the same projector again already there; from a machine without the model role, sent to the node with the most room), `models_upload.rs` (`--node auto`), `models_upload_pool.rs` (`--node auto --stage` refused), `rollouts_net.rs` (an upload joins `support` on `inf` as `support@2` at 20%, saved there and not here).

**Not tested:** across real machines; MLX directories (macOS nodes).

**Not built:** `[model] upload = false` (uploads off, everything else on): `WritePolicy` in weirdnet-backends 0.4.5 has no switch for uploads alone (its `auth` also covers pulls and removals, `max_upload = 0` would read as `too-big`), so this needs a small Weirdnet request; `--serve` as a flag (uploads always serve; `--store-only` would be the useful opposite). `--node auto` does not look at engines: a directory sent to a llama.cpp node is refused there as before.

**Next:** merge after CI; release 0.16.0.

## 2026-11-25 21:50 EST: P3b Ymir side, resumable training jobs (branch `feature/p3b-resume`)

**What changed:** trainer interface 2 (`train --checkpoint-dir DIR` writes `epoch-NNNN/` with `checkpoint.json`, `ymir-train-checkpoint/1`; `--resume CKPT`), spoken by the mock trainer and `scripts/trainer.py` (one `llama-finetune --epochs 1` per epoch: the pinned build saves weights only at the end and has no optimizer state, so AdamW restarts each epoch; no shuffle and a constant learning rate keep the data order and schedule the same). `ymir_core::train_job`: the job record, lease with fencing term, attempts, back-off and checkpoint pointer as the `JobStore` trait asked of `weirdnet-agent` 0.6.0, with `FileJobStore` and `LocalBlobs` standing in on one node; `start`/`resume` drive a run under a claim (renew, publish checkpoints, finish, fail with or without retry, pause on Ctrl-C, stop on a lost lease) after `check_resumable` (trainer build, base revision, `--from`, adapter id, `[train] resume`). `ymirctl train run --job`, `train jobs`, `train resume JOB`, `train job pause|unpause|cancel JOB`. Docs: [training.md](docs/design/training.md) (P3b), [DESIGN.md](docs/DESIGN.md), [CHANGELOG.md](CHANGELOG.md), site.

**Checks:** `cargo clippy -p ymir-core -p ymirctl -p ymir-testkit --all-targets -D warnings`; `cargo test -p ymir-core --lib train` (7 new store tests); `tests/p3b.rs` (5, three runs in a row), `p3a`, `p3a_remote`, `p3a_llamacpp` (with the trainer venv); `scripts/test_trainer.py` (28). Not tested: a real `llama-finetune` with checkpoints (no run cleared), Windows/macOS (CI).

**Left:** swap the stand-ins for `weirdnet-agent` 0.6.0's `JobBoard<TrainJob>` and a blob fetch from the holders over Weirdnet; let a `[train] resume = true` node take over stalled jobs by itself; enforce `needs`. The request went to the lead on 2026-11-25 and is summarised, with the lead's plan, in [training.md](docs/design/training.md)'s P3b section.

## 2026-11-25 18:40 EST: v0.15.4 released; first hands-off self-update confirmed

**Release:** [v0.15.4](https://github.com/WeirdBrains/ymir/releases/tag/v0.15.4) (Release run 45711728332, green). `ymirctl update status` now shows the self-update settings and the last feed check. The assets check passes, including the six signed update bundles (`com.weirdbrains.ymir-stable-15004`) and the gpg-signed Arch packages. The leak scan of dist found nothing.

**Self-update test:** a node on 0.15.3 with `[update] interval = "10m"` found 0.15.4 on the GitHub feed about 7 minutes after the bundles were uploaded. It staged the bundle, exited for the update, started 0.15.4 on trial and committed it 5 seconds later, with no manual steps. The test interval was then removed, so the node is back on the 6h default. Arch recipes are on this branch; TAUR follows.

## 2026-11-25 18:00 EST: v0.15.3 released (self-updating ymird)

**Release:** [v0.15.3](https://github.com/WeirdBrains/ymir/releases/tag/v0.15.3) (Release run 45700519266, green). The assets check passes, including the six signed `.update.tar.gz` bundles (update id `com.weirdbrains.ymir-stable-15003`) and the gpg-signed Arch packages. The leak scan of dist found nothing. ymird now checks its feed (default `github:WeirdBrains/ymir`, every 6h) and stages and restarts onto a newer signed release by itself; see `[update]` in DESIGN. Arch recipes are on this branch; TAUR follows.

## 2026-11-25 15:20 EST: hands-off self-update (branch `feature/auto-update`)

**What changed:** `ymir_core::autoupdate`, `[update]` in node.toml (`auto = true`, `channel = "stable"`, `interval = "6h"`, `feed = "github:WeirdBrains/ymir"`, `variant = "auto"`, `restart = true`).
- `ymird` checks the feed two to five minutes after it starts, then every interval. When it finds a newer release it downloads this machine's `.update.tar.gz` (plain or CUDA, matching the running engine) and unpacks it safely into `updates/incoming/`.
- It stages the bundle through `Updater::stage`, which does all the signature checks. Then, under a service manager and when started through `update boot`, it shuts down and exits 75 so the service restarts it into the trial.
- A trial that isn't serving within 15 minutes exits 1, and the second failed start rolls back.
- `ymirctl update boot` now sets `YMIR_UPDATE_BOOT`. `ymirctl update check` runs one check by hand.
- DESIGN and CHANGELOG are updated.

**Checks:** 7 new `autoupdate` tests: settings, feed URLs, asset names, CUDA detection, unsafe archives, a local feed staging a signed bundle end to end, and a foreign-signed bundle refused. CI is green on all three OSes (run 45684801350, before merging main).

**Left:** release it, then confirm on a real node that a later release is taken without a hand.

## 2026-11-25 15:00 EST: v0.15.2 released with signed update bundles

**Release:** [v0.15.2](https://github.com/WeirdBrains/ymir/releases/tag/v0.15.2) (Release run 45681998536, all green). The assets check passes, including the six `.update.tar.gz` bundles (plain and CUDA Linux x86_64/aarch64, macOS x86_64/aarch64, update id `com.weirdbrains.ymir-stable-15002`) and `ymir-0.15.2-updates.sha256`. The Arch packages are gpg-signed. The leak scan of dist found nothing. Arch recipes are on this branch; TAUR follows.

## 2026-11-25 14:56 EST: model upload U8, origin and license in the pool catalog (branch `feature/catalog-origin`)

**What changed:** Weirdnet request D shipped (weirdnet-agent 0.5.4, weirdnet-backends 0.4.5, weirdnet-openai 0.2.3), so U8 is built. Model nodes put each model's origin (`hub`, `upload`, `train`) and license in their capability record: at start from node.toml (`NodeConfig::origins`: a `provenance.json` beside the weights, else `hub` for weights under `models_dir` or a Hub id), and on every upload, peer copy and Hub pull through `ModelsService::with_node_info` plus a hook for this machine's own model node (`LocalModelNode::set_origin`). `/ymir/uploads/1` also sets them and stays for Bragi 0.10 and older `ymirctl`. A placement stored but not served (the projector edge the lead named) is logged as a warning; the uploader already gets the error and the files stay. Inference nodes copy origin and license into catalog/1 (`ModelNode::origin`, `CatalogRow`), so `/v1/models` shows `weirdnet.origin`/`license`; `ymirctl models` prints them after the state and the MCP `ymir_models` tool returns them. `release.toml` requires weirdnet-backends 0.4.5 and weirdnet-agent 0.5.4. Docs: the plan ("U8 as built"), DESIGN, CHANGELOG, the site's ymirctl page. Bragi tags uploaded and trained models in its picker (Bragi branch `feature/catalog-origin`).

**Checks:** fmt; clippy `-D warnings`; `cargo test --workspace --locked` 347 passed, 0 failed, 7 ignored. New tests: `NodeConfig::origins` (an upload, a trained adapter, a pulled file, a Hub id, a file elsewhere, a wrapped server), the merged catalog across two nodes (`the_catalog_says_where_each_model_came_from`), end to end in `models_upload` (an upload to another node shows `upload`/`mit` in the inference node's catalog; its configured file shows `hub`), and `ymirctl models` lines. Not tested: a real Hub pull firing the hook (covered by weirdnet-backends' own tests), and MLX.

**Note:** the v0.15.2 release flow branched `chore/release-0.15.2` in the same checkout while this work was there, so its merge (c9c9078) brought the U8 commits (0ab6e67, e462558) to main before this branch's CI finished; this branch (rebuilt in its own worktree, same content) adds this entry and links the v0.15.2 entry's release URL, which the Pages link check refused. U8 ships in the next Ymir release after 0.15.2.

## 2026-11-25 14:05 EST: signed update bundles in the release flow (branch `feature/update-bundles`)

**What changed:** `ymirctl update sign-bundle` (`ymir_core::update_publish`) signs a bundle directory with a release key. It refuses a key the root doesn't delegate for `com.weirdbrains.ymir`/`stable` or a root with the wrong pin, and verifies the result as a node would before writing `manifest.env` and `roots/1.env`. It never prints the key. Helpers 0.4.0 is vendored; it adds `weirdnet_release.py updates VERSION [--upload]`, configured in `release.toml [updates]`, which builds the plain and CUDA Linux bundles and the macOS bundles from the release archives. The org root is `packaging/updates/root.env`. The release workflow only checks the plan (the key is not a CI secret). DESIGN and CHANGELOG are updated.

**Release flow addition:** after `github VERSION`, on the release machine: `WEIRDNET_UPDATE_KEY=<release key file> python3 scripts/weirdnet_release.py updates VERSION --dist dist --upload`, then `assets VERSION`. The signer is the release's own x86_64 `ymirctl`. A node takes a bundle with: download, unpack, `ymirctl update stage DIR`, restart the service.

**Checks:** `ymir-core` update tests (14, 4 new) and clippy pass on Linux. An end-to-end run with the real key signed 0.15.1 archives (seq 15001), `update trust` + `update stage` accepted it in a scratch home, and the old 0.1.0 `ymirctl update boot` picked the staged release. Helpers: 50 tests.

## 2026-11-25 12:00 EST: v0.15.1 released, signed and on TAUR

**v0.15.1** (release commit c0b5f6b, tag v0.15.1, cut with `weirdnet_release.py cut --patch` on main cc9f216 after CI was green there; the cut's checks: fmt, clippy `-D warnings`, `cargo test --workspace --locked` 340 passed, 0 failed, 7 ignored): the trainer refuses a dataset too short for llama-finetune, `build --trainer` adds llama-tokenize, the larger synthetic fixture, and the fast `extract-lora` (randomized SVD, CUDA through torch, a thread pool). The Release workflow passed on the tag; its artifacts were checked against `SHA256SUMS` (30 files) and scanned unpacked: no host names, tailnet names, private IPs or home paths. The [release](https://github.com/WeirdBrains/ymir/releases/tag/v0.15.1) has 35 files (`assets 0.15.1` ok, latest), with detached `.sig` files for the four Arch packages (WeirdBrains Muninn packages key, each "Good signature"). `weirdnet_release.py taur 0.15.1 --yes` published `ymir-bin` and `ymir-cuda-bin` 0.15.1-1. The Arch recipes carry the v0.15.1 tarball hashes (branch `chore/release-0.15.1`).

## 2026-11-25 08:35 EST: v0.15.0 released, signed and on TAUR

**v0.15.0** (release commit 6b042db, tag v0.15.0, cut with `weirdnet_release.py cut --minor` on main 9010783 after CI, macOS and Pages were green there; the cut's checks: fmt, clippy `-D warnings`, `cargo test --workspace --locked` 339 passed, 0 failed, 7 ignored): canary percentages (`ymirctl models rollout add --canary PCT`, `rollout weight NAME@N --canary PCT`, `models upload --rollout --canary PCT`, any 0 to 100, not capped, decision 8) and U7 (more upload formats). The Release workflow passed on the tag. Artifacts were downloaded into `dist/` (the full `SHA256SUMS` from its own artifact), checked against those sums (30 files) and scanned unpacked: no host names, tailnet names, private IPs or home paths. The [release](https://github.com/WeirdBrains/ymir/releases/tag/v0.15.0) has 35 files (`assets 0.15.0` ok): the 30 release files, `SHA256SUMS`, and detached `.sig` files for the four Arch packages, signed with the WeirdBrains Muninn packages key (each "Good signature"; sums matching). `weirdnet_release.py taur 0.15.0 --yes` published `ymir-bin` 0.15.0-1 and `ymir-cuda-bin` 0.15.0-1. The Arch recipes carry the v0.15.0 tarball hashes (this branch, `chore/release-0.15.0`). Bragi 0.10.0 has the matching percentage field in its upload dialog.

**Next:** U8, once Weirdnet request D lands.

## 2026-11-25 10:20 EST: fast LoRA extraction (branch `feature/fast-lora-extract`)

The first real run's `extract-lora` (Qwen2.5-0.5B, rank 16, 168 matrices) took 79 minutes on a machine at load ~40. Measured there: a 4864 x 896 float64 SVD took 29.7 s with OpenBLAS's default 16 threads under that load, 4.8 s with 8, 0.86 s with 4 and 1.0 s with 1, so most of the time was BLAS threads spinning, not the SVD.

**What changed** ([scripts/trainer.py](scripts/trainer.py))
- `top_svd`: a randomized, truncated SVD (sketch of rank + 16, 5 power iterations, float32; seeded per matrix, so runs repeat); exact for matrices whose smaller side is at most 4 sketches wide. On CUDA through PyTorch when the venv has a CUDA build (`svd_device`, `YMIR_TRAINER_SVD` = auto, cuda, cpu, exact).
- `lora_tensors` runs `YMIR_TRAINER_THREADS` matrices at a time (default min(4, CPUs)) with one BLAS thread each (`single_threaded_blas`, set before numpy loads), in the base's order, a bounded window in memory.
- The final log line gives the extraction time; [trainer-requirements.txt](scripts/trainer-requirements.txt) names torch as optional.
- Docs: training plan (trainer section), CHANGELOG (Unreleased, Changed).

**Timings** (the run's full model was not kept, so a stand-in: the f32 base plus the run's adapter plus full-rank noise of the same size, the real shapes; load ~30 from other jobs on the machine; rank 16):
- before (the old code), exact float64 SVD, 16 BLAS threads: 1704 s (28 min; the real run took 79 min at load ~40)
- after, exact SVD on 4 threads: 45.2 s
- after, CPU (randomized, 4 threads): 8.2 s (17.4 s wall with loading and writing)
- after, GPU (torch 2.11 cu128, RTX 3060): 4.4 s (21.5 s wall)
- quality: the kept share 53.0% in every mode; the randomized adapter's residual is within 0.03% of the exact one's on the worst matrices.

**Checks:** `python3 scripts/test_trainer.py` 26 passed, 1 skipped (the CUDA comparison, run with `YMIR_TEST_CUDA=1` on a GPU machine: 4 passed on the test machine's RTX 3060 with torch 2.11 cu128); new tests: randomized vs exact at the model's shapes, exact on low rank and repeatable, device choice and refusals, modes agree through the CLI, threaded order.

**Next:** merge after CI; a patch release.

## 2026-11-25 08:20 EST: the trainer refuses a too-short dataset; a larger fixture (branch `feature/trainer-short-corpus`)

The first real `train run` (context 256, then 64) crashed in upstream `llama-finetune` with `GGML_ASSERT(INT64_MAX / ne[i] > current_nelements)` in `ggml_opt_dataset_init`: the small fixture renders to 256 tokens, and upstream needs 2n + 1 for one step, n being the context rounded up to a multiple of 256 (`common_opt_dataset_init` computes `(tokens - n - 1) / (n/2)` in size_t, which underflows on less).

**What changed**
- [scripts/trainer.py](scripts/trainer.py): `trained_context`, `min_tokens`, `count_tokens` (llama-tokenize `--stdin --ids` on the base's vocabulary, the trailing newline dropped as llama-finetune's `-f` does) and `short_text`; `train` counts before starting llama-finetune and refuses with the count and the shortfall. Without `llama-tokenize` next to llama-finetune (or `$YMIR_LLAMA_TOKENIZE`) it logs that and turns upstream's crash into the same refusal.
- [scripts/engine.py](scripts/engine.py): `build --trainer` also builds and keeps `llama-tokenize`.
- New fixture: [decentralized-networks-large.jsonl](crates/ymir-testkit/tests/fixtures/decentralized-networks-large.jsonl) (batch input), [decentralized-networks-large.dataset.jsonl](crates/ymir-testkit/tests/fixtures/decentralized-networks-large.dataset.jsonl) and its manifest: 16 hand-written conversations, CC0-1.0, the same sources file. With Qwen2.5-0.5B-Instruct's chat template it is 1297 tokens (the small one 255), counted with a real `llama-tokenize` at b12628 and the new `count_tokens`.
- Docs: training plan (trainer section, the fixtures in decision 3), CHANGELOG (Unreleased).

**Checks:** `python3 scripts/test_trainer.py` 20 passed (new: `test_min_tokens`, `test_short_text_is_refused_before_training`, `test_enough_text_trains`, `test_context_is_rounded_up_to_256`, `test_tokenize_failure_is_refused`, `test_short_text_crash_without_tokenize`; new fake `scripts/testdata/fake_llama_tokenize.py`); `python3 scripts/test_engine.py` 6 passed; `cargo fmt --check`; clippy `--workspace --all-targets -D warnings`; `cargo test --workspace --locked` 340 passed, 0 failed, 7 ignored (new: testkit `the_large_fixture_dataset_is_accepted_for_training`). `p3a_llamacpp` with the trainer venv is left to CI.

**Not tested:** a training run with the large fixture (not rerun yet, waiting for Michael's OK).

**Next:** merge after CI; then a first real run on the large fixture once approved.

## 2026-11-25 07:31 EST: canary percentages (branch `feature/canary-percent`); the 0.15.0 cut was stopped for it

Michael asked at 7:24 AM ET for a canary percentage such as 20 or 50 (decision 8 in [docs/design/model-upload.md](docs/design/model-upload.md)). The 0.15.0 cut that was running then was stopped before it committed, tagged or pushed anything (its version edits were reverted), so the release includes this.

**What changed**
- [crates/ymir-core/src/rollout_edit.rs](crates/ymir-core/src/rollout_edit.rs): `Change::Share { name, version, percent }`: the version gets exactly `percent` (0 to 100), the other versions with traffic the rest by their present weights (largest remainder, adding up to 100). Not capped at `CANARY_MAX`; an eval in the gate, when given, must pass (`force` skips it). Raw weights keep the gate.
- ymirctl: `models rollout add ... --canary PCT` (an add at weight 0, then the share) and `models rollout weight NAME@N --canary PCT`, both also with `--inference NODE` (one `/ymir/rollouts/1` request per change); `models upload ... --rollout NAME --canary PCT` is a percentage now.
- Docs: decision 8, DESIGN, training plan, CHANGELOG (Added and Changed), the site's ymirctl page.

**Checks:** `cargo fmt --check`; clippy `--workspace --all-targets -D warnings`; `cargo test --workspace --locked` 339 passed, 0 failed, 7 ignored (new: `rollout_edit` share tests, ymirctl `add_and_weight_take_a_named_percentage`; testkit `models_upload` and `rollouts_net` use `--canary 20`/`50` and check the split).

**Next:** full checks, merge, CI on main, then cut 0.15.0 (with U7 and this) and Bragi 0.10.0 (percentage field), sign, TAUR.

## 2026-11-25 05:55 EST: model upload U7, more formats (branch `feature/upload-formats`); U8 blocked on Weirdnet request D

Slice U7 of [docs/design/model-upload.md](docs/design/model-upload.md): over `--node`, an upload also takes a safetensors directory (for vLLM), an MLX directory, a PEFT LoRA directory (on a vLLM base) and a GGUF vision projector (`--projector-for MODEL`). Weirdnet already reads, places and serves these (weirdnet-model 0.5.3, weirdnet-backends 0.4.4, locked); no Weirdnet crate edited. Details in the plan's "U7 as built".

**What changed**
- [crates/ymir-core/src/upload.rs](crates/ymir-core/src/upload.rs): `inspect_outgoing` and `Outgoing`; `provenance_for`; local refusals now point at `--node`.
- [crates/ymir-core/src/config.rs](crates/ymir-core/src/config.rs): `[model] upload_engine` (validated), used by [daemon.rs](crates/ymir-core/src/daemon.rs) for models/1's engine.
- [crates/ymir-core/src/upload_net.rs](crates/ymir-core/src/upload_net.rs): `Register.projector_for` (optional); `UploadsService` saves `backend = "vllm"` (absolute path) and `"mlx"` entries, PEFT adapters on the base, and a projector as the model's `mmproj` ([config_edit.rs](crates/ymir-core/src/config_edit.rs) `set_mmproj`).
- ymirctl: `models upload PATH --node X [--projector-for MODEL]`, the checks before sending, the format in the first line (`GGUF model (llama)`).
- Docs: the plan (U7 section and as built, U8 blocked), DESIGN, CHANGELOG, the site's ymirctl page.

**Checks:** `cargo fmt`; clippy `--workspace --all-targets -D warnings`; `cargo test --workspace --locked` 336 passed, 0 failed, 7 ignored (new: testkit `models_upload_formats`, unit tests for `inspect_outgoing`, `upload_engine`, `set_mmproj`).

**Not tested:** real vLLM, MLX or llama.cpp with the uploaded files (mock engine); MLX end to end (MLX nodes are macOS); local uploads of directories and projectors (not built).

**U8:** blocked. It needs request D (`CatalogModel::origin` and `license` in weirdnet-agent catalog/1), which weirdnet-agent 0.5.2 does not have. The request is drafted for Michael to send; nothing changed for U8.

**Next:** merge, release 0.15.0; Bragi's copy fallback (branch `feature/upload-copy-fallback`) and Bragi 0.10.0.

## 2026-11-25 05:36 EST: v0.14.0 released, signed and on TAUR

**v0.14.0** (release commit 560283b, tag v0.14.0, cut with `weirdnet_release.py cut --minor` on main 42707ce after CI, macOS and Pages were green there; the cut's checks: fmt, clippy `-D warnings`, `cargo test --workspace --locked` 332 passed, 0 failed): uploads to a pool (U6: `ymirctl models upload FILE --node all|A,B [--stage NODE]`, sent once and copied node to node). The Release workflow passed on the tag. Artifacts were downloaded into `dist/` (the full `SHA256SUMS` from its own artifact), checked against those sums (30 files) and scanned unpacked: no host names, tailnet names, private IPs or home paths. The [release](https://github.com/WeirdBrains/ymir/releases/tag/v0.14.0) has 35 files (`assets 0.14.0` ok, marked latest): the 30 release files, `SHA256SUMS`, and detached `.sig` files for the four Arch packages, signed with the WeirdBrains Muninn packages key (each "Good signature"; sums matching). `weirdnet_release.py taur 0.14.0 --yes` published `ymir-bin` 0.14.0-1 and `ymir-cuda-bin` 0.14.0-1. The Arch recipes carry the v0.14.0 tarball hashes (this branch). Bragi 0.9.0's desktop upload dialog copies uploads on to other nodes with the same Weirdnet call.

## 2026-11-25 04:33 EST: model upload U6, pools (branch `feature/upload-pools`)

Slice U6 of [docs/design/model-upload.md](docs/design/model-upload.md): upload once, the other nodes copy it from that node, on Weirdnet's `ModelsClient::replicate` and `Pull::from_peer` (weirdnet-backends 0.4.4, already locked); no Weirdnet crate edited. Details in the plan's "U6 as built".

**What changed**
- [crates/ymir-core/src/models.rs](crates/ymir-core/src/models.rs): `ModelsHub::upload_pool` (targets, the space check first, the stage, `replicate`, each copy recorded, the fallback upload from this device); `register` and `has` split out of `upload`.
- [crates/ymir-core/src/upload_net.rs](crates/ymir-core/src/upload_net.rs): `UploadOp.stage`, `UploadReport.copied_from` and `sent_instead` (serde defaults).
- [crates/ymir-core/src/remote.rs](crates/ymir-core/src/remote.rs): `serve_models` takes the `Network` and sets the peer dialer.
- ymirctl: `models upload --node all|A,B [--stage NODE]` and its output.
- Docs: the plan (U6 section and as built), DESIGN, CHANGELOG, the site's ymirctl page.

**Checks:** `cargo fmt`; clippy `--workspace --all-targets -D warnings`; `cargo test --workspace --locked` 332 passed, 0 failed, 7 ignored (new: `models_upload_pool`, also run 4 more times on its own, all passed).

**Not tested:** against real nodes on different machines; a stage on an older Ymir (the fallback is tested with a node that can't reach the stage).

**Next:** merge, release 0.14.0; a pool target in Bragi if small.

## 2026-11-25 04:02 EST: v0.13.0 released, signed and on TAUR

**v0.13.0** (release commit 2ef42b8, tag v0.13.0, cut with `weirdnet_release.py cut --minor` on main 33764a5 after CI, macOS and Pages were green there; the cut's checks: fmt, clippy `-D warnings`, `cargo test --workspace --locked` 331 passed, 0 failed): rollouts over Weirdnet (`/ymir/rollouts/1`, `ymirctl models rollout ... --inference NODE [--no-persist]`). CI, macOS, Pages and the Release workflow passed on the tag. Artifacts were downloaded into `dist/` (the full `SHA256SUMS` is the `SHA256SUMS` artifact's; each platform artifact carries a partial one), checked against those sums (30 files) and scanned unpacked: no host names, tailnet names, private IPs or home paths. The [release](https://github.com/WeirdBrains/ymir/releases/tag/v0.13.0) has 35 files (`assets 0.13.0` ok, marked latest): the 30 release files, `SHA256SUMS`, and detached `.sig` files for the four Arch packages, signed with the WeirdBrains Muninn packages key (each "Good signature"; sums matching). `weirdnet_release.py taur 0.13.0 --yes` published `ymir-bin` 0.13.0-1 and `ymir-cuda-bin` 0.13.0-1. The Arch recipes carry the v0.13.0 tarball hashes (this branch). Bragi 0.8.0's desktop upload dialog uses the new protocol (noted in the plan's "as built").

## 2026-11-25 03:03 EST: rollouts over Weirdnet, `/ymir/rollouts/1` (branch `feature/rollouts-net`)

The follow-up to model-upload U4, so Bragi and `ymirctl` on another device can put an upload into an inference node's rollout. Decisions of 2026-11-25 2:51 AM ET (made by llm dev from Michael's upload answers; he can override them) are in [docs/design/model-upload.md](docs/design/model-upload.md): any linked device, narrowed by `[inference] rollout_callers`; above the canary share the request carries eval rows or `force`; every operation remotely, rollback included; saved in node.toml by default, with an opt-out. Details in the plan's "`/ymir/rollouts/1` as built".

**Changed:** `ymir_core::rollout_edit` (the rules, moved from `ymirctl`, behaviour unchanged), `ymir_core::rollouts_net` (service, server, client), `[inference] remote_rollouts` and `rollout_callers`, `ControlRequest::RemoteRollouts`, `ymirctl models rollout ... --inference NODE [--no-persist]`. Docs: CHANGELOG, DESIGN, the plan, the site's ymirctl page.

**Checks:** fmt; clippy `--workspace --all-targets -D warnings`; `cargo test --workspace --locked` 331 passed, 0 failed, 7 ignored, including the new testkit `rollouts_net` (three daemons over 127.0.0.1).

**Not tested:** across real machines; `models upload --rollout` with `--inference` (not built: `--rollout` still applies on this machine).

**Next:** the rollout section in Bragi's upload dialog; release Ymir 0.13.0 and Bragi.

## 2026-11-25 02:40 EST: model upload U4 in Bragi 0.7.0 (docs, branch `docs/u4-bragi-upload`)

Correction: the entry below headed 01:43 EST was written at 01:37 EST.

CI and Pages are green on main 377ccb3 (the `fix/no-wixpdb` merge). The Linux CI job first failed in `ymir-testkit --test ctxplan` (`a_load_from_elsewhere_is_followed...`, unchanged since it last passed); a rerun of the failed job passed, and the test passed six times in a row on the box, so it is recorded as a flake, not fixed.

Slice U4 of [docs/design/model-upload.md](docs/design/model-upload.md) is built in Bragi 0.7.0: an upload dialog on the desktop Models screen over models/1, with Ymir 0.12.0's `/ymir/uploads/1` record. Details in the plan's "U4 as built"; the B7 line and status in [docs/DESIGN.md](docs/DESIGN.md) are updated. Rollouts are not set from Bragi yet: the result shows the `ymirctl ... --rollout` commands.

**Not tested:** Bragi against a real Ymir node over the network.

**Next:** `/ymir/rollouts/1` so Bragi can add an upload to a rollout; then a rollout section in Bragi's dialog.

## 2026-11-25 01:43 EST: v0.12.0 released, signed and on TAUR

**v0.12.0** (release commit d5db03d, tag v0.12.0, cut with `weirdnet_release.py cut --minor` on main ba00557 after CI, macOS and Pages were green there; the cut's own checks: fmt, clippy `-D warnings`, `cargo test --workspace --locked` 327 passed, 0 failed): model uploads join rollouts (U3). CI, macOS and the Release workflow passed on the tag. Artifacts were downloaded into `dist/`, checked against the workflow's sums and scanned (Linux, CUDA Linux, macOS, Windows: no host names or home paths). The Windows installer job also produced `YmirSetup-0.12.0.wixpdb` (a WiX debug database), which was left out; `fix/no-wixpdb` stops the workflow producing it. The first `github 0.12.0` upload timed out (HTTP 408) and was re-run with `--clobber`. The [release](https://github.com/WeirdBrains/ymir/releases/tag/v0.12.0) has 35 files (`assets 0.12.0` ok): the 31 release files with `SHA256SUMS`, plus detached `.sig` files for the four Arch packages, signed with the WeirdBrains Muninn packages key (each "Good signature"; the packages' SHA-256s match the workflow's). `weirdnet_release.py taur 0.12.0 --yes` published `ymir-bin` 0.12.0-1 and `ymir-cuda-bin` 0.12.0-1. The Arch recipes carry the v0.12.0 tarball hashes (this branch).

## 2026-11-25 00:21 EST: model upload U3, `ymirctl models upload ... --rollout NAME` (branch `feature/model-upload-rollout`)

Slice U3 of [docs/design/model-upload.md](docs/design/model-upload.md), on P4b's rollouts; no Weirdnet crate edited. Details in the plan's "U3 as built".

**What changed**
- [crates/ymirctl/src/models/upload_rollout.rs](crates/ymirctl/src/models/upload_rollout.rs) (new): `--rollout NAME`, `--canary W`, `--eval ROWS`, `--judge MODEL`, `--force` on `models upload --node`. The next version N comes from this machine's rollouts (live, else node.toml); without `--name` the id becomes `ID-rN`. After the node serves the upload, `rollout add NAME@N --weight 0`; an adapter loaded on its base's next start gets the "waits at weight 0" note. `--canary` runs `rollout weight` through the P4b gate, after `ymirctl eval` on the new model and the incumbent's when `--eval` is given (skipped when nothing has traffic). A refusal leaves the version at weight 0 and says so. `--rollout` without `--node` is refused.
- [crates/ymirctl/src/models/rollout.rs](crates/ymirctl/src/models/rollout.rs): `rollout list` prints each version's origin, license, base and dataset hash from the pool's `models list --node all` (best effort, 10 s).
- [crates/ymir-core/src/control.rs](crates/ymir-core/src/control.rs): status rows (`NodeModel`) carry `revision`, the weights' revision the node reports.
- Tests: `models_upload.rs` now reads the uploaded revision back from the pool (closes the U2 gap) and runs three `--rollout support` uploads (first `--canary 100` with no incumbent; `--canary 50` refused, stays at 0; `--canary 5` passes; `rollout list` shows provenance; no `--node` refused). Unit tests for next version, incumbent, ids and the provenance line.
- Docs: plan "U3 as built", [docs/DESIGN.md](docs/DESIGN.md), [CHANGELOG.md](CHANGELOG.md), the site's ymirctl page.

**Checks**: `cargo fmt --all --check`; `cargo clippy --workspace --all-targets --locked -- -D warnings`; `cargo test --workspace --locked --no-fail-fast` 327 passed, 0 failed, 7 ignored; `site/tool/check_links.py`, `gen_roadmap.py --check`.

**Not tested**: `--eval` end to end (no testkit model answers; the gate itself has P4b's tests); the adapter "waits at weight 0" note; `rollout list` provenance for trained adapters (base and dataset: unit test only).

**Not built**: `--inference NAME|all` (rollouts stay per inference node, agreed 2026-11-25); `--replace` (U2 refuses other files under a taken id, so a version's model cannot be swapped).

**Next**: U4 (Bragi desktop upload, decision 6) or U6 (pools, waits on Weirdnet requests B and C).

## 2026-11-24 23:58 EST: v0.11.0 released, signed and on TAUR

**v0.11.0** (release commit 8ff1654, tag v0.11.0, cut with `weirdnet_release.py cut --minor` on main bc665df after CI and Pages were green there; the cut's own checks: fmt, clippy `-D warnings`, `cargo test --workspace --locked` 325 passed, 0 failed): model uploads to other nodes (U2) on `weirdnet-model` 0.5.3 / `weirdnet-backends` 0.4.4. CI, macOS and the Release workflow passed on the tag. Artifacts were downloaded into `dist/`, checked against the workflow's sums and scanned (Linux, macOS, Windows archives: no host names or home paths). The [release](https://github.com/WeirdBrains/ymir/releases/tag/v0.11.0) has 35 files (`assets 0.11.0` ok): the 31 release files with `SHA256SUMS`, plus detached `.sig` files for the four Arch packages, signed with the WeirdBrains Muninn packages key as v0.8.0/v0.9.0 (each "Good signature"; the packages' SHA-256s match the workflow's). `weirdnet_release.py taur 0.11.0` published `ymir-bin` 0.11.0-1 and `ymir-cuda-bin` 0.11.0-1. The Arch recipes carry the v0.11.0 tarball hashes (this branch).

## 2026-11-24 23:05 EST: model upload U2, `ymirctl models upload FILE --node X` (branch `feature/model-upload-remote`)

Slice U2 of [docs/design/model-upload.md](docs/design/model-upload.md) on `weirdnet-backends` 0.4.4's models/1 upload; no Weirdnet crate edited. Details in the plan's "U2 as built".

**What changed**
- [crates/ymirctl/src/models/upload.rs](crates/ymirctl/src/models/upload.rs) `remote_op`: with `--node`, checks and hashes the file (or a run's adapter, `--from-run`, found in RUN_DIR when FILE is left out), builds its `Provenance` and an `Upload` that serves. `--no-persist` is an alias of `--no-register`. Output in [remote.rs](crates/ymirctl/src/models/remote.rs) (`upload`, `uploaded`, `adapter_uploaded`: a llama.cpp adapter is used from its base's next start; never "live").
- [crates/ymir-core/src/models.rs](crates/ymir-core/src/models.rs): `ModelsOp::Upload`, `Outcome::Uploaded`, `ModelsHub::upload` (one named remote node; resumes; progress never goes back; `exists` with the same revision counts as already there; then registers). `ModelsHub::with_uploads`.
- [crates/ymir-core/src/upload_net.rs](crates/ymir-core/src/upload_net.rs) (new): `/ymir/uploads/1`. `UploadsService` (on model nodes with `remote_manage`, any linked device) checks the stored entry's revision, calls `NodeService::set_model_info(id, Some(revision), quant)` and saves the entry in node.toml unless `--no-persist`; `UploadsClient`; refusal wording.
- [crates/ymir-core/src/config_edit.rs](crates/ymir-core/src/config_edit.rs) (new): node.toml edits moved from `ymirctl` (`stanza`, `local_model`, `append_model`, `add_adapter`, `save`); `ymirctl` calls them.
- `[model] upload_keep_free` (default: the larger of 10% and 20 GiB) feeds the node's `WritePolicy`; `serve_models` takes the policy. `Daemon::start_at(cfg, paths, config)` so a saved upload edits the file ymird was started with (`ymird --config`).
- Tests: [crates/ymir-testkit/tests/models_upload.rs](crates/ymir-testkit/tests/models_upload.rs) (three real daemons on 127.0.0.1: laptop and two model nodes) covers a model sent, stored with provenance (`uploaded_by` the laptop) and saved with its revision; the same files again; other files under a taken id (`exists`); a cut-off upload resumed; an adapter with `--no-persist` (node.toml byte-identical) and saved (`base_revision` from the base); a wrong-architecture adapter; `no-space`; `--node` naming this device or `all`. Unit tests: `remote_op` with `--from-run`, `upload_keep_free`, refusal wording, JSON round trips.
- Docs: plan "U2 as built", [docs/DESIGN.md](docs/DESIGN.md), [CHANGELOG.md](CHANGELOG.md), the site's ymirctl page.

**Checks**: `cargo fmt --all --check`; `cargo clippy --workspace --all-targets --locked -- -D warnings`; `cargo test --workspace --locked --no-fail-fast` 325 passed, 0 failed, 7 ignored; `RUSTDOCFLAGS=-D warnings cargo doc --workspace --no-deps --locked`; `site/tool/check_links.py`, `gen_roadmap.py --check`.

**Not tested**: the pool seeing the recorded revision (`set_model_info` is called; no test reads the capability record back); multi-GB uploads and real links; Ctrl-C from a terminal (the test cuts the stream by sending a short local file); a duplicate upload at the same time (`Busy`); a node with an older ymird (no `/ymir/uploads/1`); `--from-run --node` end to end (unit test only); a vLLM node (uploads are GGUF only).

**For Weirdnet (requests through m3talsmith)**: an "upload placed" hook on `ModelsService` (or `Upload { set_info, persist }` handled in the crate) would let apps drop their own register protocol.

**Next**: U3 (`--rollout`), U4 (Bragi dialog), U6 (pools, now unblocked by `replicate`).

## 2026-11-24 23:03 EST: v0.10.0 released; the first real P3a run's node is configured, the run waits on a restart

**v0.10.0** (release commit 8721d41, tag v0.10.0): `train run --node` and the trainable dataset fixture. CI, macOS and the Release workflow passed on 8721d41; the [release](https://github.com/WeirdBrains/ymir/releases/tag/v0.10.0) has 31 files (`assets` ok). The Arch recipes carry the v0.10.0 tarball hashes (this commit). The upload was interrupted once and left a draft with 25 files; the 6 missing files were uploaded and the draft published, nothing re-cut. Package signatures and TAUR: see the next entry or the report (the GPU node downloads the packages slowly).

**First real P3a run (approved once by m3talsmith, as was adding the model role and `[train]` to one GPU node):**
- That node's node.toml has a timestamped backup and two changes only: `model = true`, and a `[train]` section (the v0.9.0 `scripts/trainer.py`, `build = "b12628"`, `[train.env]` with the trainer's venv python, the patched `llama-finetune` built with `engine.py build --trainer`, CPU only since that node has no CUDA toolkit or Vulkan headers, and `YMIR_TRAINER_NGL = "0"`). `ymirctl train check` (0.8.0) passes: b12628, patch `parse-special`. A throwaway copy of the config starts ymird 0.8.0 with the model role and no models, so a later restart is safe.
- A base is in that node's models directory: Qwen2.5-0.5B-Instruct F16 GGUF (Apache-2.0, sha256 checked against the Hub) and an F32 copy of it for `--from` (every tensor F32; a test chat on it answered).
- Not run, two steps outside what was approved: ymird there is a systemd user service, so loading the model role needs `systemctl --user restart ymird`; and `train run --base` takes a `[[model.local]]` id, so node.toml also needs a `[[model.local]]` entry for the F16 base (not approved yet).

## 2026-11-24 22:37 EST: weirdnet-model 0.5.3 / weirdnet-backends 0.4.4; upload code on weirdnet-model (branch `feature/weirdnet-model-0.5.3`)

The Weirdnet lead released `weirdnet-model` 0.5.3 and `weirdnet-backends` 0.4.4 (crates.io) with the model upload pieces Ymir asked for. This moves U1/U5's interim code onto them; behaviour is kept.

**What changed**
- `Cargo.toml`/`Cargo.lock`: `weirdnet-model` 0.5.3, `weirdnet-backends` 0.4.4; [release.toml](release.toml) `[[requires]]` for both.
- [crates/ymir-core/src/upload.rs](crates/ymir-core/src/upload.rs): `inspect` checks Ymir's rules (one GGUF model or LoRA; symlinks, split shards, projectors and other adapter types refused) and hashes with `artifact::inspect_with`; `Inspected` holds the `Artifact`. `import` copies into a staging directory beside the models directory (`<models_dir>.importing/`, outside the store; the system temp directory if that can't be made), then `ModelStore::import` places it whole with `provenance.json` and `adapter.json`. The record is `weirdnet_model::artifact::Provenance` (`weirdnet-provenance/1`); `architecture` goes in its extra fields. `provenance(item, spec)` builds the record (also for uploads over Weirdnet next). `keep_free` is `KeepFree::default().of(total)`. Ymir's own `Provenance`, `BaseRecord`, `RecordKind`, `ArtifactKind`, `revision_of` and `adapter.json` writer are gone.
- [crates/ymir-core/src/train.rs](crates/ymir-core/src/train.rs): `import_spec` maps a run onto `ProvenanceBase`, `ProvenanceDataset` (format, hash, rows used, `rows_in`/`rows_used` counts, the license) and `ProvenanceTrainer` (name, build, run); the rest of the run (dataset name, teacher, node, settings, steps, loss) goes in the `ymir_train` extra field.
- One layout change: an uploaded model goes to `models_dir/<id>/` (the store's layout, as uploads over Weirdnet) instead of `models_dir/uploads/<id>/`. Models uploaded there before stay, and uploading the same bytes again still answers "already there". Old `ymir-provenance/1` records read as `Provenance` (tested).
- Tests: core upload tests on the new types (the source left alone, no staging left, the store lists the entry with its record, the legacy layout); ymirctl and `p3a` assertions on the new record fields.
- Docs: the plan ("Switched to Weirdnet"), [docs/DESIGN.md](docs/DESIGN.md), [CHANGELOG.md](CHANGELOG.md).

**Checks**: `cargo fmt --all --check`; `cargo clippy --workspace --all-targets --locked -- -D warnings`; `cargo test --workspace --locked --no-fail-fast` 320 passed, 0 failed, 7 ignored; `p3a_llamacpp` with `YMIR_TEST_TRAINER_PYTHON` (the real trainer script's adapter imports through `weirdnet-model`); `site/tool/check_links.py`.

**Not tested**: multi-GB files (three reads: hash, copy, the store's check); a staging directory on another filesystem than the models directory (the store copies then).

**Next**: U2, `ymirctl models upload FILE --node X` over models/1.

## 2026-11-24 21:45 EST: `train run --node` (P3a option C) and a trainable dataset fixture

**Dataset fixture (merged d871bc9):** `crates/ymir-testkit/tests/fixtures/decentralized-networks.dataset.jsonl` (5 synthetic rows: 4 usable, 1 cached duplicate marked unusable) with its `ymir-dataset/1` manifest (teacher `synthetic-fixture`, CC0-1.0, input and sources hashes of the existing fixtures). `tests/dataset.rs` checks that `train::prepare_dataset` takes it (4 rows trained) and that the manifest's hashes and counts match. The 21:45 entry's run command named the batch prompt input, which has no manifest; it now names this file.

**`ymirctl train run --node X`:** a run on another of the owner's model nodes from any linked device (an app-only one in the test), on published crates only (`weirdnet-net` 0.8.2 generic streams, `weirdnet-agent` 0.5.2 channels and roster auth; no Weirdnet crate edited). Ymir's own protocol `/ymir/train/1` (`crates/ymir-core/src/train_net.rs`), served by the model role only with the new `[train] remote = true` (default off), to linked, unrevoked devices, narrowed by `[train] callers`. The asking `ymird` resolves the node on the roster and sends the dataset and manifest inline (24 MiB cap); the node runs `train::run` in process (so it reserves itself through its own `ymird`) and streams lines back over the new streamed control request `Train`. Closing the stream (Ctrl-C) cancels the run. `--add` is refused with `--node`; `--from` must be inside the node's models directory; the adapter is served with `models adapter serve --node`.
- Tests: `tests/p3a_remote.rs` (three real daemons on 127.0.0.1 with the mock trainer: an opted-out node refuses, the opted-in one trains and reserves in its ymird, node.toml unchanged, the sent dataset removed, refusals for a reused id, `--add`, a bad `--from`, an unknown and the local device; dropping the caller cancels the run and frees the GPU), plus unit tests in `train_net` (refused caller, file names and `--from` paths).
- Not tested: a real node and trainer; the cap near 24 MiB; a revoked device (covered by `RosterAuth`, as models/1).
- Missing for the rest (P3b): the run lives only as long as the caller's stream. A run that outlives the caller, shows up on every device and resumes on another node needs `weirdnet-agent`'s synced job record, which is not published.

**Left:** the first real P3a run still needs a GPU model node with `[train]` (and `remote = true` to start it from another device) and the patched trainer; the agent may not edit node.toml.

## 2026-11-24 21:45 EST: v0.9.0 signatures and TAUR; first real P3a run blocked

**v0.9.0, finished:** the four Arch packages are signed with the WeirdBrains Muninn packages key (detached `.sig`, verified "Good signature") and uploaded to the [v0.9.0 release](https://github.com/WeirdBrains/ymir/releases/tag/v0.9.0) next to their packages (now 35 files); `weirdnet_release.py taur 0.9.0` published `ymir-bin` 0.9.0-1 and `ymir-cuda-bin` 0.9.0-1. This closes the "Left" items of the v0.9.0 entry below.

**First real P3a run (approved once by m3talsmith, 2026-11-24): not run, blocked.**
- `ymirctl train run` runs on the node it is called on; there is no `--node` for it yet (only `train reserve` takes one). The only machine the agent can reach is an app-only node: no model or inference role, no `[train]` in node.toml, so `train check` says "no trainer is set", and `ymirctl nodes` there lists no model nodes. Its own GPU (12 GB) isn't a Ymir model node.
- The agent may not edit node.toml. A run needs either shell access to one GPU model node, or that node's owner to add `[train]` (`trainer` = `scripts/trainer.py` from v0.9.0 in a venv with `scripts/trainer-requirements.txt`, `build = "b12628"`) and build the patched trainer there (`python3 scripts/engine.py build --trainer --out DIR`, then `YMIR_LLAMA_FINETUNE` under `[train.env]`), and an F32 GGUF of a small base on that node (`--from`). Then: `ymirctl train check`, `ymirctl train run --base ID --dataset crates/ymir-testkit/tests/fixtures/decentralized-networks.dataset.jsonl --epochs 1 --context 256` (corrected 2026-11-24: `decentralized-networks.jsonl` is the batch prompt input and has no manifest; the trainable fixture is `decentralized-networks.dataset.jsonl` with its `.manifest.json`, added after this entry), and `ymirctl nodes` plus a chat to confirm serving came back.
- Alternative for later: `train run --node X`, so a run can be started from any linked device (training plan "Next").

## 2026-11-24 16:45 EST: v0.9.0 released

`release: v0.9.0` (`94c1c29`, tag [`v0.9.0`](https://github.com/WeirdBrains/ymir/releases/tag/v0.9.0)) under "Just release", cut with `scripts/weirdnet_release.py` on main `536f4ec` after CI was green there (Linux, macOS, Windows; Windows needed a second attempt in the MSI step) and the release checks passed locally (fmt, clippy `-D warnings`, `cargo test --workspace` 314 passed, 0 failed). It carries the trainer's parse-special patch and build-number fix (`0b4cf61`), model upload U1 and U5, the M9 bundle fetcher and the Windows bench sandbox. The tag's CI, macOS and Release workflows passed ([Release run 45483965006](https://github.com/WeirdBrains/ymir/actions/runs/45483965006)). The GitHub release has 31 files with `SHA256SUMS`; `weirdnet_release.py assets 0.9.0` is ok. Artifacts scanned (Linux, macOS and Windows archives): only CI runner paths, no host names or home paths.

This branch records the v0.9.0 tarball hashes in `packaging/arch/ymir-bin` and `ymir-cuda-bin` (`taur-check 0.9.0` ready for both).

**Left, not possible from the dev box:** publishing to TAUR (`weirdnet_release.py taur 0.9.0`; `taurctl` and its login are on the maintainer's machine) and the detached `.pkg.tar.zst.sig` signatures v0.8.0 had (the signing key is local there). crates.io does not apply (`[crates] publish = []`). The Homebrew formula still has placeholder sums, as before.

## 2026-11-24 15:18 EST: model upload U5, trainer outputs on the upload path (branch `feature/model-upload-trainer-output`)

Slice U5 of [docs/design/model-upload.md](docs/design/model-upload.md), next after U1 in the approved order.

**What changed**
- [crates/ymir-core/src/train.rs](crates/ymir-core/src/train.rs): `train run` imports its adapter with `upload::import` (origin `train`) into `models_dir/adapters/ID/ID.gguf` with `provenance.json` and `adapter.json`, instead of moving it to `adapters/ID.gguf`; the run's copy is removed. New `train::import_spec(record, adapter, ms)` builds the record (base id/revision/quant, the dataset record, the trainer with run, node, hyperparameters, steps and loss). `AdapterRecord` gains `revision` (serde default, so older `run.json` files still read). The adapter id must now pass `upload::check_id`; an old flat `adapters/ID.gguf` still blocks the id.
- [crates/ymir-core/src/upload.rs](crates/ymir-core/src/upload.rs): `ImportSpec` gains `dataset` and `trainer` and derives `Default`.
- [crates/ymirctl/src/train.rs](crates/ymirctl/src/train.rs): `--add` (and the printed `adapter add` line) use the content revision, falling back to the run id for older records; prints the provenance path.
- [crates/ymirctl/src/models/upload.rs](crates/ymirctl/src/models/upload.rs): `--from-run RUN_DIR`, FILE optional with it. Reads `run.json` (format, `done`, an adapter), checks the file's SHA-256 against the record, defaults the base and id from the run, refuses a base here at another revision than the run trained on, and imports with the run's record.
- [crates/ymir-testkit/src/bin/ymir-mock-trainer.rs](crates/ymir-testkit/src/bin/ymir-mock-trainer.rs): `extract-lora` writes a real (tensorless) GGUF LoRA header, since the import reads it.
- Tests: `p3a` checks the new layout, the provenance record and the content revision in node.toml; `p3a_llamacpp` paths; two new ymirctl tests (`upload_from_a_run_*`: a copied run with its record, the same bytes already here, a wrong SHA-256, another base revision, an unfinished run, a missing run.json).
- Docs: the plan ("U5 as built", status), [docs/design/training.md](docs/design/training.md), [docs/DESIGN.md](docs/DESIGN.md), [CHANGELOG.md](CHANGELOG.md).

**Checks**: `cargo fmt --all --check`; `cargo clippy --workspace --all-targets --locked -- -D warnings`; `cargo test --workspace --locked` 314 passed, 0 failed, 7 ignored; `p3a_llamacpp` with `YMIR_TEST_TRAINER_PYTHON` set to a venv with `scripts/trainer-requirements.txt` (the real trainer script's adapter imports); `site/tool/check_links.py`.

**Not tested**: a real training run on a GPU; a run copied between two real nodes.

**Behaviour change**: new runs register the content revision, not the run id. Existing node.toml entries and adapters are untouched.

**Left**: `train run --add --rollout NAME` waits for U3. Nothing in U5 needs `weirdnet-model` 0.5.3 / `weirdnet-backends` 0.4.4; at the switch-over `import_spec` maps onto their `Provenance`. Not released: Ymir's release is signed on the release machine, which is offline.

**Next**: U2 (upload to one node over Weirdnet), once Weirdnet requests A and B ship.

## 2026-11-24 15:00 EST: llama-finetune parses special tokens (branch `feature/trainer-parse-special`)

m3talsmith's updated decision 7: patch `llama-finetune` in Ymir's trainer build to tokenize with `parse_special = true`, stay on b12628, and drop the patch when upstream releases the fix (tracked: draft ggml-org/llama.cpp#31258). This replaces the "no local patch" answer in the entry below.

**What changed**
- [packaging/engines/patches/llama-finetune-parse-special.patch](packaging/engines/patches/llama-finetune-parse-special.patch): `examples/training/finetune.cpp` only. `common_tokenize(ctx, params.prompt, true, /*parse_special =*/ true)`, and `llama-finetune --ymir-patches` prints `parse-special` (upstream refuses that argument, so an unpatched build is told apart).
- [scripts/engine.py](scripts/engine.py): `TRAINER_PATCHES`, `build_plan` (the steps of `build`: the patches, with `--trainer` only, before cmake), `apply_patch` (`git apply --check` first; a reused `--work` tree that has it is left alone; exits if it doesn't apply), `engine.toml` gets `trainer_patches`. A build without `--trainer` is unchanged, and `llama-server`/`rpc-server` compile from unpatched sources either way.
- [scripts/trainer.py](scripts/trainer.py): `REQUIRED_PATCHES = ("parse-special",)`; `check_pin` (train and extract-lora) refuses a `llama-finetune` at the pin without it; `version` adds `"patches"` (Ymir's `TrainerVersion` ignores the extra field; `build` stays `b12628`, so `[train] build` in node.toml is unchanged).
- Found with a real build: at b12628 `--version` prints `version: 1.0.0-dev (build N, commit X)`, and N counts commits, so `engine.py build`'s shallow clone reported `build 1` and `trainer.py` read `version: 1` as b1: a real `build --trainer` could never pass the pin check. `engine.py build` now passes `-DLLAMA_BUILD_NUMBER` from the pin (every build, serving too, which only changes the version line), and `trainer.py parse_build` reads both the 1.0 format and the older `version: 12628 (X)`. The fake prints the 1.0 format.
- The fake `llama-finetune` answers `--ymir-patches` (`FAKE_FINETUNE_PATCHES`, empty acts unpatched) and records the base's control tokens it tokenized as specials. The test base has `<|im_start|>`/`<|im_end|>` as control tokens.
- Tests: new [scripts/test_engine.py](scripts/test_engine.py) (stdlib: the trainer build applies the patch before cmake, a serving build applies none, the patch touches only `finetune.cpp`, `engine.toml` names it, and the patch applies to the pinned upstream `finetune.cpp` read from GitHub, then is seen as applied on a reused tree). [scripts/test_trainer.py](scripts/test_trainer.py) adds: required patches match engine.py's and each patch names its id; an unpatched build is refused for train and extract-lora (nothing runs); the special tokens of every turn train as single tokens. CI runs `test_engine.py` in the trainer step.
- Docs: [docs/design/training.md](docs/design/training.md) (decision 7 updated, the special-token bullet, tests), [CHANGELOG.md](CHANGELOG.md) (Changed).

**Checks**: `cargo fmt --all --check`; `cargo clippy --workspace --all-targets --locked -D warnings`; `cargo test --workspace --locked`; `p3a_llamacpp` with `YMIR_TEST_TRAINER_PYTHON`; `python3 scripts/test_engine.py` 5 passed; `scripts/test_trainer.py` 14 passed; the patch applies with `git apply --check` to a b12628 checkout (8a97fa9), and the patched `finetune.cpp` passes `clang++ -std=c++17 -fsyntax-only` against b12628's headers; `site/tool/check_links.py`.

**Real build** (box, CPU only, clang, `--no-cpu-variants`): `engine.py build --trainer` cloned b12628, applied the patch (and said "already applied" on the reused `--work`), built `llama-server`, `rpc-server` and `llama-finetune`; `engine.toml` has `trainer_patches = ["parse-special"]`; `llama-finetune --ymir-patches` prints `parse-special`; `trainer.py version` gives `{"interface": 1, "engine": "llama.cpp", "build": "b12628", "patches": ["parse-special"]}`. The same binary rebuilt without the patch exits 1 on `--ymir-patches`, and `trainer.py` refuses it ("without Ymir's parse-special patch").

**Not tested**: a real training run (no model downloaded); tokenization of specials was checked with the fake, not a real vocabulary; GPU builds.

**Next**: when upstream releases a `llama-finetune` that parses specials, move the pin (`scripts/engine.py pin bNNNN`, `PINNED_BUILD`), delete the patch, its `TRAINER_PATCHES`/`REQUIRED_PATCHES` entries and the fake's `--ymir-patches`, and redo the Inkling checks for serving.

## 2026-11-24 14:50 EST: model upload U1, local `ymirctl models upload` (branch `feature/model-upload-local`)

The plan ([docs/design/model-upload.md](docs/design/model-upload.md)) was approved by the lead (order U1, U5, U2, U3,
U4, U6, U7, U8; merged to main in 5ad1242) and m3talsmith answered its seven questions at 2:22 PM ET; the plan now
records them under "Decisions" (every linked device may upload, license optional, no caps but 10 percent or 20 GiB
kept free, uploads saved to node.toml by default, Bragi desktop first, private Hub repo as the stopgap), the Weirdnet
lead's accepted shapes for A to C (`weirdnet-model` 0.5.3, `weirdnet-backends` 0.4.4, not built yet) and the
switch-over to them.

**What changed**
- `ymir_core::upload` (interim, marked to move to `weirdnet-model`): `inspect` a GGUF model or LoRA (refuses
  projectors, split shards, symlinks, non-GGUF, other adapter types), `import` (hash while copying into a hidden
  temp file, sync, hard-link into place without replacing; `provenance.json`, and `adapter.json` for adapters in
  `weirdnet-model`'s shape), `revision_of`, `check_adapter_base`, `check_license` (optional), `keep_free`.
- `ymirctl models upload FILE [--name] [--adapter-for BASE] [--license] [--role] [--no-register] [--dir]`: adds the
  entry to node.toml by default with the upload's revision (adapters through the same validated edit as `models adapter
  add`, with `base_revision`/`base_quant`); says a llama.cpp adapter serves after a restart or the base's next load;
  `--node` refused until U2. `hub::disk_space` (free and total) backs `free_bytes`.
- Docs: DESIGN ("Uploading a model trained elsewhere"), the plan ("U1 as built", "Decisions", "Stopgap until U2"),
  CHANGELOG.

**Checks run**: `cargo fmt --all --check`; `cargo clippy --workspace --all-targets --locked -- -D warnings`; `cargo
test --workspace --locked --no-fail-fast`: 312 passed, 0 failed, 7 ignored. New tests: 11 in `upload::tests`
(ymir-core) and 5 `models::tests::upload_*` (ymirctl), all on synthetic GGUF headers in temp dirs; the CLI tests pass
the disk figures in, so they don't depend on the machine's free space.

**Not tested**: a real fine-tuned model or adapter served after upload; Windows and macOS (CI runs on push to main).

**What's left / next**: U5 (`--from-run`, `train run` writing `provenance.json`), then U2 once `weirdnet-model` 0.5.3
and `weirdnet-backends` 0.4.4 are published. Roadmap entry not added yet.

## 2026-11-24 14:30 EST: model upload plan (branch `feature/model-upload-plan`, not merged)

m3talsmith asked for uploading models to Ymir through ymirctl and Bragi; the main use case is fine-tuning off the
network and putting the result into the pool. Plan only, for the project lead's review before any code:
[docs/design/model-upload.md](docs/design/model-upload.md). It covers what can be uploaded (GGUF models and splits,
mmproj projectors, GGUF LoRA, PEFT adapters, safetensors for vLLM, MLX directories), a provenance record (base,
dataset manifest, license, revision from the content hash), a chunked, resumable, SHA-256-checked `Upload` command on
models/1, staging plus node-to-node replication for pools, a write grant for uploads, validation, quotas and disk
checks, how an upload joins a P4b rollout at weight 0 behind the eval gate, P3a outputs on the same path
(`--from-run`), the ymirctl and Bragi UX, requests A to D for the Weirdnet lead, risks and the slice order (U1 local
upload with provenance first).

**Checks run**: docs only; no code changed, no build. Surveyed `weirdnet-model` 0.5.2 (`ModelStore`, adapters, GGUF
headers, download), `weirdnet-backends` 0.4.3 (models/1, `StoredEntry`, `ModelsService`), `weirdnet-agent` 0.5.2
(`RosterAuth::require_roles`), `weirdnet-net` 0.8.2 stream limits, and Bragi main (Models screen, pools, rollouts,
cert-only links).

**What's left / next**: the lead's review and m3talsmith's answers to the plan's open questions; then U1. Requests A and
B go to the Weirdnet lead once the plan is confirmed.

## 2026-11-24 14:30 EST: trainer pin decision, white paper link (branch `feature/trainer-pin-decision`)

**What changed** (docs only): [docs/design/training.md](docs/design/training.md) links the white paper's training section ([docs/whitepaper/sections/10-training.tex](docs/whitepaper/sections/10-training.tex)) instead of saying it is not in the repository, and records m3talsmith's decisions of 2026-11-24: 7, move the llama.cpp pin to a build whose `llama-finetune` parses special tokens, with no local patch; 8, "Just release".

**Upstream check (2026-11-24):** no llama.cpp build fixes it. `examples/training/finetune.cpp` calls `common_tokenize(ctx, params.prompt, true)` with `parse_special` left at its default `false` at b12628, at the latest build b12680 and at `v1.0.0`; the file's last change is ggml-org/llama.cpp#20834 (2026-07-23). The only fix in flight is draft ggml-org/llama.cpp#31258 (its chat path tokenizes with `parse_special = true`), not merged. **The pin stays at b12628**; nothing in `scripts/trainer.py`, `packaging/engines/llama.cpp.toml`, CI or packaging changed.

**Coupling:** the trainer and the bundled serving engine share the one pin in `packaging/engines/llama.cpp.toml` (`engine.py build --trainer` builds `llama-finetune` there; `trainer.py` refuses another build), so the move also moves serving and needs the Inkling checks redone. `weirdnet-backends` has no llama.cpp pin (only b12628 in comments and its HANDOFF), so no Weirdnet request.

**Checks:** `site/tool/check_links.py`, `site/tool/gen_roadmap.py --check`, and every relative link in the training plan resolves. Also reworded the usage-stop entry's local-state line, which named a machine and a home path (check_links failed on it). No code changed, so no cargo run.

**Not done:** a test that the trainer's text tokenizes specials as specials: it needs a `llama-finetune` that parses them (or a real tokenizer), which no upstream build has. When #31258 or another fix is released: `scripts/engine.py pin bNNNN`, update `PINNED_BUILD`, the fake `llama-finetune`, the tests and docs that say b12628, and add a tokenization check against a tiny vocab fixture.

## 2026-11-24 13:25 EST: state at a usage stop (open branches, what's left)

**On main (ac8aa63, CI green on Linux, macOS, Windows):** the llama.cpp trainer wrapper (feafb07), the Windows bench sandbox (e8923bf), the M9 bundle fetcher (b202a74), Infra's v0.8.0 PKGBUILD hashes (b99aabc). Release-ready: ac8aa63.

**Open branches, all pushed:**
- `fix/rollout-list-no-inference` (1 commit on ac8aa63): `ymirctl models rollout list` on a running node without the inference role said "ymird is not running here"; it now says the node has no inference role (the ControlResponse refusal is told apart from a dead socket). Unit test + `p4b.rs` e2e against a model-only ymird. CI 45455967977 green on all three OSes. **Left: merge (`--no-ff`) and push.**
- `chore/helpers-0.3.3` (1 commit): vendors weirdnet-project-helpers 0.3.3 (validate and `--dry-run` compare `[[requires]]` versions offline) and pins `helpers = "0.3.3"`. `validate` passes, and fails with requires 0.4.2 against Cargo.toml 0.4.3. CI 45456634158 (Linux, macOS green; Windows pending at writing). **Left: merge once Windows is green.**
- `feat/m9-bsd-packages`: FreeBSD `.pkg` + rc.d service, FreeBSD/OpenBSD `tar.gz`, `engine.py` without ggml-rpc RDMA on BSD, manual `bsd.yml`. Already merged in: main up to ac8aa63 and the fetcher. In the FreeBSD VM: cargo build, engine from source, package, `pkg add`, binaries, user and home all work. **Left: `service ymird start` did not come up** (run 45451131242: no `ymirctl status` in 60 s). The rc.d pidfile fix and failure diagnostics (syslog, ps, foreground ymird) are in run 45454646555, still running at writing; read its log, fix, re-run. **Before merging: remove the TEMPORARY `push:` trigger at the top of `.github/workflows/bsd.yml`.** HANDOFF/DESIGN/CHANGELOG entries for BSD are on the branch (the HANDOFF "RESULT" placeholder needs the final run result).
- Older local branches pushed so nothing is only on a machine (already merged or superseded, kept for reference): `feat/m10-cert-only`, `feat/p2-training-export`, `feat/p3a-trainer`, `fix/p2-tombstones`, `fix/post-031`, `deps/backends-0.4.2`, `fix/pages-doc-links`, `fix/plan-total-and-requires`, `fix/test-port-race`, `p4a-slots-base-revision`, `weirdnet-live-adapters`.

**Blocked / decisions:**
- OpenBSD: its rust package is 1.81 (7.6), the workspace needs 1.99, and rustup has no OpenBSD.
- MLX-LM for Inkling 2 Mini: mlx-lm#2086 is still an open draft (checked 2026-11-24); latest mlx-lm 0.32.2.
- P3a trainer: upstream `llama-finetune` tokenizes with `parse_special=false`, so chat-template special tokens train as text. Decided 2026-11-24: patched in the trainer build (`feature/trainer-parse-special`) until upstream releases a fix (draft ggml-org/llama.cpp#31258).
- Bench on macOS: macOS refuses both memory limits, so code answers are left unscored there. Allow scoring with CPU, file and process limits only?
- Publishing update bundles and announcements needs the release script and the org keys.
- The whitepaper roadmap line (docs/whitepaper/sections/12-roadmap.tex) still says the fetcher is next; it is Michael's file.

**Local state:** the dev box's own Ymir checkout has a stash "local: release.yml aarch64 on 24.04 + CUDA check in docker" (main already has the 24.04-arm runner; compare before dropping). The phase log is kept off-repo on the dev box.

## 2026-11-24: M9 bundle fetcher (branch `feat/m9-bundle-fetcher`)

**What changed**
- `ymir_core::update`:
  - `Updater::fetch(Source, announced, ...)`: the manifest is verified first, on a copy of the client state so nothing is recorded, and only then are files read, each up to its signed size; then `stage`.
  - `check_announcement`, a `Source` trait with `DirSource`, and `pin()`.
  - Root rotations in a published bundle are `roots/1.env`, `roots/2.env`, ...
- ymirctl:
  - `update fetch SOURCE [--announcement FILE]` (an `https://` bundle directory through ureq, `http://` only to loopback, or a directory).
  - `update watch --relay ADDR [--fetch] [--once] [--wait-secs N]` (`/weirdnet/updates/1` from a dial-only node with a fresh key).
- `weirdnet-update` 0.2.0 -> 0.2.2 with `wire` (Cargo.lock: that crate only).
- Docs: [docs/DESIGN.md](docs/DESIGN.md) (M9 status), the site's `#updates`, [roadmap.toml](roadmap.toml), [CHANGELOG.md](CHANGELOG.md).

**Checks**: `ymir-core` `update::` 8 passed (3 new: fetch happy path; unsigned, unannounced and oversized refused with nothing downloaded; announcements). `ymir-testkit/tests/update_fetch.rs` 2 passed, against a loopback HTTP server and an in-test carrier on `weirdnet-update`'s `Carrier`. `m9.rs` and `release_requires` still pass; clippy `-D warnings` on ymir-core, ymirctl and ymir-testkit.

**Not tested / not built**: a running `weirdnet-relay` (the test carrier uses the same crate code, but not the relay binary), real HTTPS, and real keys. Nothing publishes bundles or announcements yet; that belongs in the release script and needs the org keys. There is no watcher in `ymird`, and no `weirdnet:` sources.

## 2026-11-24: the llama.cpp trainer for P3a (branch `feat/p3a-llamacpp-trainer`)

**What changed**
- [scripts/trainer.py](scripts/trainer.py) speaks trainer interface 1 around upstream `llama-finetune`, pinned to the engine's build (b12628, [packaging/engines/llama.cpp.toml](packaging/engines/llama.cpp.toml)). `train` needs an F32 base (upstream's limit), renders conversations with the base's chat template, relays progress bars as events, and removes its 0600 training text. `extract-lora` writes a GGUF LoRA (truncated SVD of each `blk.*` matrix's change, `alpha = rank`, llama.cpp's `lora_a`/`lora_b` layout). `YMIR_TRAINER_PYTHON` re-runs it in a venv.
- `scripts/engine.py build --trainer` also builds `llama-finetune` (upstream's archives don't ship it).
- Docs: [docs/design/training.md](docs/design/training.md) ("The llama.cpp trainer"), [docs/DESIGN.md](docs/DESIGN.md) (`[train]` sample), the site's `#train`, `--from` help (F32, not F16), [roadmap.toml](roadmap.toml), [CHANGELOG.md](CHANGELOG.md).

**Checks**: `python scripts/test_trainer.py` 10 passed (fake `llama-finetune` in [scripts/testdata](scripts/testdata/fake_llama_finetune.py)); `crates/ymir-testkit/tests/p3a_llamacpp.rs` passed with `YMIR_TEST_TRAINER_PYTHON` (it skips without it); clippy `-p ymir-testkit --all-targets -D warnings`; `check_links.py`, `gen_roadmap.py --check`. Linux CI runs both trainer tests in a venv.

**Not tested**: no real `llama-finetune` run, no `engine.py build --trainer` build, and no adapter loaded by a real llama-server. Known limit: at b12628 `llama-finetune` tokenizes with `parse_special = false`, so chat-template special tokens are trained as text. That needs a decision: a one-line patch at build, or a later pin.

## 2026-11-24: training plan P4b, versioned rollouts (branch `feat/p4b-rollouts`)

**What changed**
- `ymir_core::rollout`: `[[inference.rollout]]` (`name`, `[[inference.rollout.version]]` with `version`, `model`, `weight`), validated at load. `Rollout::pick` makes a weighted, sticky choice from a SHA-256 of the conversation key: the synced conversation id, else the system prompt and first user message, else the job id. `name@N` is parsed by `split_versioned`.
- Coordinator:
  - `versioned()` in `prepare`. A version whose model no node has is passed over for one with traffic, and the name is refused when every weight is 0.
  - `rollouts()`/`set_rollouts()` (live), and catalog entries for the name and each `name@N`.
  - `Prepared::asked` and `answering_name()`, so `Accepted` names what the app asked for. weirdnet-openai would otherwise flag a version as a stand-in, because it compares the requested model with the answering one.
  - The host's resume check and conversation record use the version's model.
- Control: `Rollouts { set }`.
- ymirctl `models rollout list | add | weight | rollback | retire`. It writes node.toml through toml_edit, keeping the rest of the file, then applies to a running ymird. The eval gate: above 10% a version needs `--eval` (a `ymirctl eval --out` file in which it passes at least the incumbent's share of checks, and of judge rows when both were judged) or `--force`.
- Docs: [docs/design/training.md](docs/design/training.md) (P4b design, status, order), [docs/DESIGN.md](docs/DESIGN.md) (Versioned rollouts), [roadmap.toml](roadmap.toml) (P4b done; [ROADMAP.md](ROADMAP.md) and the site regenerated), the site's ymirctl page (`#rollout`), [CHANGELOG.md](CHANGELOG.md).

**Checks**: `cargo fmt --check`; clippy `--workspace --all-targets -D warnings`; `cargo test --workspace` 290 passed, 0 failed, 7 ignored. New tests:
- `rollout` unit tests: shares, stickiness, rollback moves only the canary's conversations, keys, validation.
- `tests/nodes.rs` `a_rollout_sends_a_canary_share_and_rolls_back_by_weight`.
- ymirctl `models::rollout` tests: the gate, rollback and retire, and node.toml kept.
- `crates/ymir-testkit/tests/p4b.rs`, against a running ymird with two mock models: canary, gate, eval, live weights, pin, rollback saved, retire, and no stand-in label.

**Not tested / not built**: a real fine-tune behind a rollout. Several inference nodes don't share weights: each reads its own node.toml, so set the same rollout on each. There's no automatic rollback on errors. Bragi does not show versions yet (it sees `support` and `support@N` as catalog models).

## 2026-11-24: a test that keeps `release.toml` `[[requires]]` in step (branch `test/release-requires`)

Infra's cut was blocked: `release.toml` still required `weirdnet-backends` 0.4.2 after Cargo.toml moved to 0.4.3 (fixed in `f062f64`). The release script compares the lock only at cut time. Now `crates/ymir-testkit/tests/release_requires.rs` (in `cargo test --workspace`, so Linux CI) fails when a `[[requires]]` version differs from its manifest's requirement or from the one Cargo.lock entry, or when that entry isn't from crates.io. Checked by reverting the version to 0.4.2: the test fails with "release.toml [[requires]] weirdnet-backends = 0.4.2, but Cargo.toml asks for 0.4.3".

## 2026-11-24: Windows CI green again (branches `fix/p3a-windows-lock`, `fix/bench-windows`)

**What changed**
- Windows CI had failed on main since at least `6d21e7c`, before the v0.7.0 cut. The reason: `ymirctl bench` scored code answers through a unix-only sandbox (`sh`, `ulimit`, `unshare`), yet `python_available()` said yes on Windows. Now `bench::python_available()` (public) is false off unix, so code questions are left unscored there. The unit and end-to-end bench tests expect what is scored.
- P3a's stale-lock unit test checked a dead pid, which Windows doesn't (a lock there holds until `ymirctl train release`). That half of the test is unix-only now.
- [CHANGELOG.md](CHANGELOG.md) `[Unreleased]` / Fixed.

**Checks**: CI on the branch through `workflow_dispatch` ([run 45423042502](https://github.com/WeirdBrains/ymir/actions/runs/45423042502)): Linux, macOS and Windows all passed, with the Windows service, MSI and setup exe steps.

**Not done**: a sandbox for code answers on Windows (a Job Object with limits would be the way).

## 2026-11-24: training plan P3a, fine-tuning on one node (branch `feat/p3a-trainer`)

**What changed**
- Michael's decisions (2026-11-24). The trainer is upstream llama.cpp full fine-tuning plus a LoRA extraction step, pinned to one build ([docs/design/training.md](docs/design/training.md), decision 1 updated). The reserve command is approved, shipped as [`weirdnet-backends` 0.4.3](https://crates.io/crates/weirdnet-backends/0.4.3): `Supervisor::reserve`/`release`, node/1 `Reserve`/`Release`, `Capability::reserved`.
- `ymir_core::train`:
  - `[train]` in node.toml: `trainer`, `build`, `work_dir`, `env`.
  - The trainer interface: `version`, `train`, `extract-lora`, JSON-line events, SIGTERM to cancel. `check_trainer` refuses another interface or build.
  - `prepare_dataset` takes only P1 (`ymir-dataset/1`, the license checked) or P2 (`ymir-conversations/1`, the hash checked) datasets and writes the usable rows as chat JSONL.
  - The GPU lock `state/train/lock.json` (stale once its pid is gone; `pid: None` for a hold taken by hand), and `reserve_running`/`release_running` through the control socket.
  - `run`: lock, reserve, train, extract, install the adapter to `models_dir/adapters/ID.gguf`, write `run.json`, and release however it ends. `runs` lists them.
- Nodes:
  - `ModelNode::reserved`/`reserve`/`release` (local: the supervisor; remote: node/1). A reserved node reports `draining()`, so routing and placement skip it.
  - The coordinator names the holder when a model is only on reserved nodes.
  - `NodeSummary::reserved`; `ymirctl nodes` shows `reserved (held by …)`.
- Control: `Reserve { node, holder, force }` and `Release { node, holder }`. Reserving is refused when the node is the only one with a loaded model ready, unless `force`; without the inference role this box can't see other nodes, so it asks for `--force`.
- ymird: at start, a live lock reserves the supervisor, so a hold survives a restart.
- ymirctl `train run | reserve | release | check | list`. `run --add` edits node.toml through the existing adapter code (revision = run id, plus the base's revision and quantisation when trained on the served file).
- Testkit: the `ymir-mock-trainer` bin.
- Docs: the training plan (status, interface, decision 1, order), [ROADMAP.md](ROADMAP.md) and the site roadmap (P3 in progress), the site's ymirctl page, [DESIGN.md](docs/DESIGN.md)'s node.toml example, [CHANGELOG.md](CHANGELOG.md).

**Checks**: `cargo fmt`; clippy `--workspace --all-targets -D warnings`; `cargo test --workspace` (see the phase report); new `crates/ymir-testkit/tests/p3a.rs` (4 tests: a run on a generated and on a conversation dataset with `--add`; refusals for the license, a missing or changed manifest, ids, another build and a hold by hand; failed and cancelled runs; a running ymird reserved for a run, a refused sole-server reserve, a load refused meanwhile, and a hold surviving a restart), plus unit tests for the lock, events and run ids.

**Not tested**: no real trainer exists yet. A wrapper around the pinned `llama-finetune` and the extraction tool is still to write, with the build to pin. No real node, GPU or model was touched. On Windows the lock is not checked for a dead process: it holds until `ymirctl train release`. Reserving a remote node over Weirdnet was not run end to end here: `weirdnet-backends` covers node/1 `Reserve` in its own tests.

**Next**: the llama.cpp trainer wrapper (its own package, pinned build), and a first real run on a node the lead clears. Then `train run --node` for a remote node, and P3b.

## 2026-11-24: M10, cert-only server links (branch `feat/m10-cert-only`)

**What changed**
- Dependencies: `weirdnet-sync` 0.7.1 (service devices, grants), `weirdnet-device-link` 0.4.2 (credential links); `hex` and `zeroize` added. Not sync 0.8: `weirdnet-agent` 0.5.2 and `weirdnet-backends` 0.4.2 accept below 0.8.
- `ymir_core::membership`: `credential.json` instead of `seed` on a cert-only node (`weirdnet_sync::Identity`'s layout); `adopt_service`, `convert_to_service` (keeps roster, op log and older keys, writes the credential, then zeroes, syncs and deletes `seed`), `is_service`, `user_keypair`, `service_bundle(peer, ServiceRole)`, `grant_of`; `seed`, `approve`, `revoke` refuse on a cert-only node; `ServiceRole` (`inference`: roles `ymir.inference`, types `ymir.pool` + `ymir.context`; `model`: `ymir.model`, roster-only).
- Control: `PoolServiceBundle { peer, role }` (local socket), so the grant and a first rotation land in the running ymird.
- ymirctl: `pool add CODE --service inference|model`, `pool link --cert-only`, `pool cert-only` (migration, ymird stopped), `pool devices` shows grants.
- Docs: [DESIGN.md](docs/DESIGN.md) (M10 status rewritten, root-seed section, milestone list, secrets), [roadmap.toml](roadmap.toml) (M10 done; [ROADMAP.md](ROADMAP.md) and the site regenerated), site ymirctl reference and guide, [CHANGELOG.md](CHANGELOG.md).

**Checks run**: fmt; clippy `--workspace --all-targets --locked -D warnings`; `cargo test --workspace --locked --no-fail-fast`: 273 passed, 0 failed, 7 ignored; `gen_roadmap.py --check`; link check. New: `ymir-testkit/tests/m10.rs` (real `pool link --cert-only` / `pool add --service` and `pool cert-only` over loopback; daemons with a cert-only model node and inference node serving, syncing, refusing seed-only requests, and cut off by revocation).

**Not tested**: on real machines; no existing server was moved (each needs m3talsmith's go-ahead to run `pool cert-only`). Bragi and Morph still send full links only.

**What's left / next**: Bragi sends cert-only links to Ymir servers (`send_credential`), on `weirdnet-sync` 0.7 and `weirdnet-device-link` 0.4.

## 2026-11-24: P2 export skips deleted messages; Bragi's toggle is in (branch `fix/p2-tombstones`)

**What changed**: `convexport` skips a message whose value is Bragi's delete tombstone (`{"deleted": true}`, no role) instead of counting it as a dropped tool turn. Docs: [DESIGN.md](docs/DESIGN.md) (P2 and B4), [training.md](docs/design/training.md), [CHANGELOG.md](CHANGELOG.md): Bragi's per-chat toggle (Bragi main `72e7729`) writes the same `{conversation}/train` record.

**Checks run**: fmt; clippy `--workspace --all-targets --locked -D warnings`; `cargo test -p ymir-core` and the `p2` end-to-end test; new `convexport::tests::a_deleted_message_is_skipped_not_counted`.

**What's left / next**: P3a (reserve a node, run the external trainer) needs a `weirdnet-backends` node reservation.

## 2026-11-24: training plan P2, consented conversation export (branch `feat/p2-training-export`)

**What changed**
- `ymir_core::context`: `{conversation}/train` record (`Train { on, at_ms }`), its own LWW register so a `meta` rewrite never clears it; `set_train` (conversation must exist), `train`, `for_training` (consented, live, raw message values with other apps' fields).
- `ymir_core::convexport` (new): strip rules (system prompts unless `--keep-system`; tools, attachments, reasoning; cut at incomplete or stand-in answers; trailing user turns; `--redact REGEX`), chat-format rows, `ymir-conversations/1` manifest (ids, consent/update times, drop counts, pattern count and hash, never patterns or content), owner-only atomic writes, `withdrawn`.
- Control: `ContextTrain`, `ContextForTraining` (local socket); `ConversationView::training`.
- ymirctl: `context train ID [--off]`, `context export --training OUT [--keep-system] [--redact RE]...`, `context withdrawn MANIFEST`; `context show` marks `[training]`.
- New dependency: `regex` 1.
- Docs: [DESIGN.md](docs/DESIGN.md) (P2 section), [training.md](docs/design/training.md), [roadmap.toml](roadmap.toml) (P2 done; [ROADMAP.md](ROADMAP.md) and the site regenerated), [CHANGELOG.md](CHANGELOG.md).

**Checks run**: fmt; clippy `--workspace --all-targets --locked -D warnings`; `cargo test --workspace --locked --no-fail-fast`: 270 passed, 0 failed, 7 ignored; `gen_roadmap.py --check`; link check. New tests: `context::training_consent_is_its_own_record_and_off_by_default`, five in `convexport`, and end to end `p2::only_marked_conversations_are_exported_and_withdrawal_is_reported` (two daemons over Weirdnet on loopback: consent set on the app node syncs, export on the inference node, withdrawal reported).

**Not tested**: on real nodes with real Bragi history (only Ymir-written and synthetic Bragi-shaped records); training on an export (P3).

**What's left / next**: Bragi's per-conversation toggle writing the same record (Bragi side). Next in the plan: P3a (reserve a node and run the external trainer), then P4b.

## 2026-11-24: weirdnet-backends 0.4.2, engine port race (branch `deps/backends-0.4.2`)

**What changed**
- `weirdnet-backends` 0.4.2 (released for this, approved by Michael): before an engine is ready the supervisor checks `/v1/models` with the launch's key lists the model; if another server answers on the port, or the engine exits while loading because the port was taken, the launch is retried on a new port (3 tries) and isn't counted as a crash.
- Ymir: `weirdnet-backends` 0.4.2 in `Cargo.toml`/`Cargo.lock`; `release.toml` `[[requires]]` 0.4.2; the mock engine's `/v1/models` checks the key, as real engines do.

**Checks run**: fmt; clippy `--workspace --all-targets --locked -D warnings`; `cargo test --workspace --locked --no-fail-fast`: 263 passed, 0 failed, 7 ignored. The `adapters` end-to-end test binary 200 times, 24 at a time, with 24 busy loops (load average about 30): 0 failures.

**Not tested**: real llama-server and vLLM losing a port (the crate's tests use a mock that impersonates a stranger, logs a bind failure, or has another process hold the port).

## 2026-11-23: test free-port race (branch `fix/test-port-race`)

**What changed**
- `mcp_http` tests start `ymirctl mcp --http 127.0.0.1:0` and read the bound port from its stderr line (it was: bind port 0, drop it, pass the number, so under load another test's server got the port: 401s from the wrong server, refused connections).
- `m2::wraps_a_stock_server_without_owning_it` starts the mock engine with `--port 0` and reads the port it prints (the mock now prints `local_addr`; so does the mock rpc-server).
- `m7`, `real_split`, `real_bundle` need ports written into node.toml or a command line before the server binds, so they use `ymir_testkit::config_port()`: a free port in 20000..32000, below the ephemeral ranges that port-0 binds and outgoing connections use.

**Checks run**: fmt; clippy `-D warnings`; `cargo test --workspace --locked --no-fail-fast`: 263 passed, 0 failed, 7 ignored. Stress, 5000 runs of the `mcp_http` test binary, 48 at a time, with 48 busy loops (load average 88 to 140): new 0 failures; the old binary in the same conditions 510 failures (mostly 401 from another test's server, some refused/reset). `m2`'s wrap test 100 runs, 16 at a time: 0 failures.

**Not changed**: `weirdnet-backends`' supervisor still picks a free port for each engine it starts (the engine has to be told its port); that is crate code and not touched here.

## 2026-11-23: `lora_slots` and the adapter base-revision check; P4a done (branch `p4a-slots-base-revision`)

**What changed** (no Weirdnet change: `weirdnet-backends` 0.4.1 already has `ModelLaunch::lora_slots`)
- `[[model.local]] lora_slots = N` (vLLM only; refused for llama.cpp, wrapped, embedding, 0) passes `ModelLaunch::lora_slots`, so a vLLM base with no adapters at start runs with LoRA on and takes adapters live (`adapter serve`, `pull --adapter-for --serve`). `ymirctl models adapter slots BASE N|--off` edits `node.toml`; `adapter list` shows it; the NextLoad message mentions vLLM without slots.
- Base-revision check: `[[model.local.lora]] base_revision`, `base_quant`. `LocalAdapter::base_mismatch` (both sides set, trimmed, case-insensitive); ymird drops a mismatched adapter at start with an error log (`adapters::drop_wrong_base`, before `keep_servable`), the base still serves. `adapter add` records the base's current `revision`/`quant` (`--base-revision`/`--base-quant` override) and refuses a mismatch; `adapter list` marks NOT SERVED.
- P4a marked done: [roadmap.toml](roadmap.toml) ([ROADMAP.md](ROADMAP.md) and the site regenerated; its crate list corrected to the crates actually used), [training.md](docs/design/training.md), [DESIGN.md](docs/DESIGN.md) ("LoRA slots and the base-revision check"), [CHANGELOG.md](CHANGELOG.md).

**Checks run**: fmt; clippy `--workspace --all-targets --locked -D warnings`; `cargo test --workspace --locked --no-fail-fast`: 263 passed, 0 failed, 7 ignored; link check; `gen_roadmap.py --check`. New tests: `config::lora_slots_and_the_base_revision_check`, `slots_are_set_and_cleared_in_place` (ymirctl), end to end `an_adapter_trained_on_another_base_revision_is_not_served`, and the opt-in `real_vllm_runtime_lora` (ignored; needs a GPU and `YMIR_TEST_VLLM_LORA`).

**Not tested**: real vLLM with `lora_slots` and run-time adapter loading (no vLLM mock; the launch flags are unit-tested in `weirdnet-backends`); real llama-server and vLLM adapters in general; the revision check on real trained adapters (the check compares config strings only; adapters served live over Weirdnet aren't in `node.toml` and aren't checked).

**What's left / next**: P4a is complete. Next in the training plan's order: P2 (consented conversation export), then P3a and P4b.

## 2026-11-23: plan total for a fixed context, `[[requires]]` for weirdnet-backends (branch `fix/plan-total-and-requires`)

**What changed** (flagged by Infra after the 0.4.0 release)
- The context plan note for a context set in `node.toml` now gives the whole estimate like the automatic one: KV and buffers, and the total with weights and the 10% (`ctxplan::decide` at load time, `kvfit::plan_launches` at start-up).
- `release.toml` has `[[requires]]` for `weirdnet-backends` 0.4.1 (`pin = "none"`: checked on crates.io and in `Cargo.lock`, the manifest entry with its features is left as written).
- Also merged before this: `fafb6bf`, Pages' `cargo doc -D warnings` failed on doc links to private consts in `coordinator.rs` (since `be81b8e`); Pages is green again.

**Checks run**: fmt; clippy `--workspace --all-targets --locked -D warnings`; `cargo test --workspace --locked --no-fail-fast`: 260 passed, 0 failed, 6 ignored; `weirdnet_release.py validate`.

**Not tested**: the new note on real hardware (unit and CLI tests only).

## 2026-11-23: live LoRA adapter changes on new Weirdnet crates (branch `weirdnet-live-adapters`)

**What changed**
- Weirdnet crates released for it (approved by Michael): `weirdnet-agent` 0.5.2 (`Request::adapter_scale`, `CatalogModel::base`), `weirdnet-backends` 0.4.1 (live adapter add/remove on vLLM, live scale on llama.cpp, per-request scale, models/1 `ServeAdapter`/`UnserveAdapter`/`AdapterScale`), `weirdnet-alerts` 0.2.2 (takes backends 0.4), `weirdnet-openai` 0.2.2 (`adapter_scale` body field, `weirdnet.base`).
- Ymir: those versions; `ymir_core::engine_alerts` removed (ymird uses `weirdnet_alerts::engine::monitor` again); `agent_catalog` sets each adapter's `base`; `ModelsOp::{ServeAdapter, UnserveAdapter, AdapterScale}` in process and over models/1; `ymirctl models adapter serve | unserve | scale --node`; `ymirctl chat --adapter-scale`.
- Docs: [DESIGN.md](docs/DESIGN.md) (P4a "Live changes"), [training.md](docs/design/training.md) P4a paragraph, [CHANGELOG.md](CHANGELOG.md).

**Checks run**: `cargo fmt`, `cargo clippy --workspace --all-targets -D warnings`, `cargo test --workspace --no-fail-fast`: 260 passed, 0 failed, 6 ignored (`mcp_http::token_host_and_origin_are_checked` failed once with a connection refused and passed on its own and in the full rerun). New assertions: the over-Weirdnet adapter test (`models_pool.rs`) covers `chat --adapter-scale`, `adapter scale`, `unserve` and `serve` on a running base with no restart; `adapters.rs` checks `weirdnet.base` in `/v1/models`.

**Not tested**: real llama-server and vLLM with adapters (mock engines only). vLLM run-time `load_lora_adapter`/`unload_lora_adapter` follow vLLM's docs, not run on a GPU here. llama-server cannot load a new adapter file at run time (checked in the b12628 source), so that still restarts the base.

**What's left / next**: `lora_slots` in `node.toml` for a vLLM base with no adapters at start; the base-revision check.

## 2026-11-23: three fixes after the 0.3.1 rollout (branch `fix/post-031`)

**What changed**
- `ymirctl models context` agrees with the load-time planner. It used to print the start-up plan (`kvfit::plan_launches`), which keeps room for every model in `node.toml` that could run beside each one (on a 128 GB Spark: about 33 GiB for a vLLM 27B at 131,072 plus the embedding model, so Inkling showed 321,536 tokens / 41.2 GiB free and a "may not fit" for the 382,976 it ran with). With ymird answering on the control socket it now sizes each model with `ctxplan::preview`: beside what runs (`ymirctl status` rows now carry `context` and `slots`), on measured free memory (`SystemProbe`), with a loaded model's own estimate counted back (`ctxplan::free_beside`; not for one still loading), naming the idle models it would unload. Without ymird it prints the start-up plan and says so.
- Long prompts are not stalls. `Coordinator::attempt` sends the cold-load keepalive (empty `Meta` every 5 s) from routing until the first text or reasoning goes out, within `coordinator::prefill_allowance` (60 s + request bytes / 3 tokens at 50 tok/s, at most 2 h); the pool to a remote model node waits as long for the first frame (`remote::model_node_stall`), 5-12 s between frames as before. `ymirctl chat` had failed with `stalled: no frame for 25000 ms` during a 62 s prefill (17.9k tokens, Inkling at 479k context).
- The load-time plan note gives the whole estimate: "61.0 GiB for KV and buffers; 85.5 GiB in all with 16.7 GiB of weights and 10% for the engine's other buffers".
- Docs: [CHANGELOG.md](CHANGELOG.md), [DESIGN.md](docs/DESIGN.md) (load-time `models context`, long prompts). Host names in older test comments and docs replaced with `node-a` / `node-b`.

**Why 479,232 and not 437,248:** 437,248 is the test fixture's budget alone ((128 - 16 GiB) x 85% = 95.2 GiB). The real node had `budget 108.0 GiB` and 102.3 GiB measured free with nothing else running; that gives 479,232 (pinned in `ctxplan` tests).

**The 4% "over plan":** not over the estimate. 77.7 GiB was weights + KV and buffers; the plan's whole estimate adds 10% for the engine's fixed-size buffers: 85.5 GiB. Measured VmHWM 79.4 GiB loaded and 81.0 GiB after a 17.9k-token prompt, 4.5 GiB under it, with the 20% headroom untouched. `llama-fit-params -fitp on` (b12628, CPU build, Inkling-2-Mini IQ4_NL header) at 479,232: 15,219 MiB context + 46,519 MiB compute = 60.3 GiB, so Ymir's 61.0 GiB is 0.7 GiB above llama.cpp; with 16.7 GiB of weights llama.cpp's own total is 77.0 GiB. The other ~2.4 GiB at load is what `llama-fit-params` does not count (mmproj/CLIP compute, host staging and output buffers, process heap; not separated here: no Vulkan device or mmproj file on the build box). The +1.6 GiB after the prompt fits llama-server's context checkpoints for a sliding-window model (each holds one sequence's SWA cache, ~240 MiB for Inkling; up to 32 per slot, 8192 tokens apart) and its host prompt cache (`--cache-ram`, 8 GiB by default). Both live outside the planner's numbers and grow with use; the 10% covers what was measured. Estimate not changed.

**Checks:** `cargo fmt --all --check`; `cargo clippy --workspace --all-targets --locked -- -D warnings` (old unused workspace-dependency warnings only); `cargo test --workspace --locked`. New tests: `ctxplan` (the 479,232 from measured free; a loaded model counted back, not one loading; preview wording; llama-fit-params points at 382,976 and 479,232), `ymirctl` `models context` with and without ymird, `nodes` (12 keepalives over a 62 s prefill, none past the allowance, the allowance's growth, and a 62 s prefill through `local_pool` with weirdnet-agent's default stall: it failed with `Stalled { ms: 25000 }` before the fix).

**Not tested:** on a Spark (Infra verifies after the next release); a remote model node's long prefill end to end over Weirdnet (unit-tested stall only).

**What's left:** keepalives that prove progress need llama.cpp `return_progress` in the request body and a progress chunk in `weirdnet-backends` (a wish for the Weirdnet lead); measuring the fixed buffers on a Spark (`-lv 4`); counting llama-server's checkpoints and prompt cache if they ever outgrow the 10%.


## 2026-11-23: LoRA adapters on a loaded base, training plan P4a first slice (branch `feat/p4a-lora-adapters`)

**What changed**
- `weirdnet-backends` 0.3 → 0.4 (its LoRA adapter release; `weirdnet-model` 0.5.2 with it). `weirdnet-alerts` 0.2.1's `backends` feature only takes `weirdnet-backends` < 0.4, so the feature is off and ymird forwards supervisor events to the alerter itself (`ymir_core::engine_alerts`, interim, same observations and probes).
- `[[model.local.lora]]` (`id`, `file`, `scale`, `revision`) → `ModelLaunch::adapters`. Validation: unique ids across models and adapters, finite scale, llama.cpp or vLLM, not wrapped, not embedding. At start `adapters::keep_servable` drops (with an error) adapters that fail `check_adapters` (missing, not a LoRA, other architecture); the base still serves. Adapter revisions key the answer cache.
- Context planner: adapter weights count in the base (`kvfit::resolve` + `adapters::bytes`); adapter status rows are not residents; touch/load/prepare map an adapter to its base; adapter Added/Removed re-reads the base (its explicit flag kept).
- Catalog: `CatalogRow::base`, `NodeModel::base`; `ymirctl status` prints "LoRA adapter on BASE". Routing needed no change (adapter rows are in capability records).
- ymirctl: `models adapter add BASE FILE [--id] [--scale] [--revision]`, `adapter remove ID`, `adapter list` (toml_edit, keeps `node.toml.bak`; checks the GGUF against the base); `models pull REPO --node NAME --adapter-for BASE [--file] --serve`; `models list --node` shows adapters.
- Mock engine: `--lora` and per-request `"lora"`.
- Docs: [DESIGN.md](docs/DESIGN.md) ("LoRA adapters on a loaded base"), the P4a design paragraph in [docs/design/training.md](docs/design/training.md), [CHANGELOG.md](CHANGELOG.md).

**Checks:** `cargo fmt --all --check`; `cargo clippy --workspace --all-targets --locked -- -D warnings` (clean apart from the old unused workspace-dependency warnings); `cargo test --workspace --locked`: **252 passed, 0 failed, 6 ignored**. New: config, `adapters` unit tests, `ymirctl` node.toml edits, `ymir-testkit` `adapters` (add/list/remove, start-up drop, chat by adapter id with its scale, base on the same engine, unload refused), `ctxplan` (adapter weights cut the base's context; adapter loads its base; base evicted with its adapter), `models_pool` (pull `--adapter-for --serve` over Weirdnet, routed chat). `ctxplan` 40/40 and adapters+models_pool 10/10 repeated runs green.

**Not tested:** a real llama-server or vLLM with a real LoRA (no GPU here); vLLM adapter paths only in unit tests.

**What's left (next P4a slice):** runtime scale changes (llama-server `POST /lora-adapters`) and adding an adapter to a running base without restarting it (llama.cpp, vLLM `load_lora_adapter`/`unload_lora_adapter`): both in `weirdnet-backends`; a per-request scale over infer/2 (`weirdnet-agent`); refusing an adapter trained on another base revision/quant once adapters record it; adapters in the catalog/1 entry (a `base` field in `weirdnet-agent`'s `CatalogModel`, so `/v1/models` and Bragi can show them); `weirdnet-alerts` taking backends 0.4 (then drop `engine_alerts`); `ymirctl models pull --adapter-for` without `--node`; the site's `ymirctl` page.


## 2026-11-22: models run with as much context as they can take (branch `feat/full-context`)

**What changed**
- ymird plans each model's context at start ([`ymir_core::kvfit`](crates/ymir-core/src/kvfit.rs)): the full trained length (GGUF `<arch>.context_length`, or `config.json` with RoPE scaling for vLLM), cut in 1024-token steps only where the KV cache would not fit beside the weights and the node's other co-resident models; the log says why. KV sizing counts loop passes, sliding-window layers (window only), linear/recurrent layers (none), a vLLM MTP head, and `-ctk`/`-ctv` / `--kv-cache-dtype`. vLLM is also held to `--gpu-memory-utilization`. MLX and wrapped servers are not planned.
- The launch carries the planned context, so `--ctx-size` / `--max-model-len` and the catalog's advertised context are the same number. A context flag in `args` becomes the launch's context and appears once.
- `ymirctl models pull` writes no `context` unless `--context` is given, prints the context the model will get on this node, drops Inkling's `--max-model-len 262144`, and saves a vLLM repo's `config.json` and weight size under `models_dir/vllm/`. `setup` leaves llama.cpp contexts to ymird.
- `ymirctl models context` lists each model's planned context and why; `--auto ID...` / `--old-defaults` remove a fixed context (and its flags) from `node.toml`, keeping `node.toml.bak`. ymird logs a hint for entries still at an old pull default (32768, or 262144 for Inkling on vLLM).
- Docs: DESIGN ("Context length"), site `ymirctl` page, CHANGELOG.

**Checks:** `cargo fmt --all --check`; `cargo clippy --workspace --all-targets --locked --offline -- -D warnings` (clean); `cargo test --workspace --locked --offline`: **223 passed, 0 failed, 6 ignored**. New tests: Inkling GGUF header and config.json shapes (8 of 48 unrolled layers global, 32 GiB of f16 KV at 1M), RoPE/yarn and `layer_types`, cache types and flag handling, fit and cut, shared memory across swap groups, explicit contexts kept, vLLM sizing from the saved config, `pull` output, and `models context --old-defaults`.

**Not tested:** a real llama.cpp or vLLM launch at the planned context (no GPU here); real Hub fetch of `config.json`. Estimates are Ymir's, not the engine's: llama.cpp compute buffers and vLLM's CUDA graphs are covered only by the 10% margin.

**What's left:** move the GGUF sliding-window/loop reader into `weirdnet-model` (`GgufInfo` lacks those fields); `--node` pulls served in memory (`weirdnet-backends` models/1) still use the pull's context.

**What's next:** release, then on each node: `ymirctl models context --old-defaults` and restart ymird.

## 2026-11-22: Inkling 2 Mini, first-class support (branch `feature/inkling`)

**What changed**
- Bundled llama.cpp pin moved from b12590 to **b12628** (`packaging/engines/llama.cpp.toml`, `scripts/engine.py pin b12628`). b12600 added `inkling2_loop` (ggml-org/llama.cpp#32694); b12628 includes the NVFP4 loop-scale fix (#32704). `engine.py fetch --target x86_64-unknown-linux-gnu --variant default --check` reports `build 12628`.
- GGUF pull auto-fetches a `mmproj-*.gguf` in the same repo (prefers Q8_0) and sets `mmproj` on the `[[model.local]]` stanza, so `ggml-org/Inkling-2-Mini-GGUF` gets `mmproj-Inkling-2-Mini-Q8_0.gguf` the way catalog Gemma projectors work.
- `ymirctl models pull` of a safetensors/transformers repo (e.g. `thinkingmachines/Inkling-2-Mini`) records the Hub id for `backend = "vllm"` (vLLM fetches into the HF cache on first load), honors `--register`, and for Inkling adds the reasoning/tool parsers, DSpark, `image = "vllm/vllm-openai:inkling-2-mini"`, and `--max-model-len 262144` on GB10 / Spark. Search lists safetensors repos on vLLM-capable nodes (`--format vllm` or `--format all`).
- `[model] vllm = "image:..."` and `[[model.local]] image` write a docker/podman shim. ymird preflights Inkling launches: a vLLM older than 0.35.0 without the image or a wrap is left unloaded with a clear error (0.34.0 cannot load `inkling2_loop`; support is not in a vLLM release yet).
- MLX Inkling is refused with a pointer at mlx-lm#2086 (still an open draft). Roadmap/DESIGN record that follow-up.

**Checks:** On this box (`grok-bot-vm-541285893`, rustc 1.100.0), with `CARGO_BUILD_JOBS=4` and `--locked --offline`: `cargo fmt --all --check`, `cargo clippy --workspace --all-targets --locked --offline -- -D warnings` (clean aside from pre-existing unused workspace-dependency warnings: chrono, lettre, thiserror, weirdnet-rules, ymird), `cargo test --workspace --locked --offline`: **213 passed, 0 failed, 6 ignored**. New unit tests: version/Spark/args/wrapper (`ymir-core` `vllm`), config validation, GGUF+mmproj pick and pull, safetensors→vLLM stanza (including Spark context cap), MLX refusal, vLLM search.

**Not tested end to end:** no real Inkling weights, no real vLLM, no DGX Spark, no docker image run, no llama.cpp load of an `inkling2_loop` GGUF (this box has no GPU and must not download the model). The b12628 default linux engine archive was fetched and `llama-server --version` checked. CUDA variant fetch was not run here (large). `--node` remote pull of safetensors is still the GGUF-only models/1 path.

**What's next:** Infra cuts the release and rolls it out. On a Spark node: pull, register, load, and a test chat (commands in the merge notes). MLX once mlx-lm#2086 is released.


## 2026-11-22: MLX end-to-end mock test recovered (branch `feature/mlx`)

**What changed:** `crates/ymir-testkit/tests/mlx.rs` is back: one node serves a GGUF on llama.cpp (with a projector) and an MLX model, both on the mock engine; it checks the engine kinds and launches `NodeConfig` builds, chats with each model through `ymird`, and checks the capabilities each reports (the GGUF takes images and audio, the MLX model text only). It was written during the LF-18 MLX work but never committed (it lived untracked in a scratch clone and a local stash), so [DESIGN.md](docs/DESIGN.md)'s MLX section pointed at a file that did not exist and the macOS workflow had dropped its MLX mock step. The macOS workflow runs it again after the `ymir-core` unit tests. The rest of that stash (MLX as `EngineKind::Mlx`, the `ymir-engine` shim removal, packaging and docs) had already landed on main through LF-18 and later commits, so nothing else was carried over; main's newer Weirdnet pins stay.

**Checks (Linux x86_64):** `cargo check --workspace --all-targets --locked` clean; `cargo test --workspace --locked`: 178 passed, 0 failed, 6 ignored across 34 test binaries, including `llamacpp_and_mlx_models_on_one_node`; `cargo clippy --workspace --all-targets -- -D warnings` clean; `cargo fmt --check` clean.

**Not tested:** MLX on Apple Silicon (no Mac here); the macOS workflow change needs a push with a token that has the `workflow` scope.

## 2026-11-22: training plan P1, `ymirctl batch --jsonl` and `ymirctl eval` (branch `feature/batch-jsonl`)

**What changed**
- `ymirctl batch FILE --jsonl` ([`ymir-core::dataset`](crates/ymir-core/src/dataset.rs), [`batch::run_tasks`](crates/ymir-core/src/batch.rs), control request `BatchTasks`): rows with their own messages or prompt, system prompt, sampling and seed; the teacher waits for its model (no stand-in); output rows marked `usable`, `cache_hit` and `standin` from the answer's `Meta`; a `*.manifest.json` with the teacher model and the operator's `--license` (required), input hash, counts and the cache mode; `--resume`; `--sources` for a sources manifest (topic, URL and license per source; nothing is fetched).
- `ymirctl eval FILE --model M [--model M2] [--judge J]` ([`ymirctl/src/eval.rs`](crates/ymirctl/src/eval.rs)): bench scorers for `expect`, an optional judge for `rubric`, temperature 0 and a fixed seed, per-model totals.
- `weirdnet-agent` 0.5.1, `weirdnet-semcache` 0.2.1 and `weirdnet-openai` 0.2.1: `batch::CacheMode` is now the infer/2 `CacheMode`; batch generation (default `--cache bypass`) and evals send it with `wait_for_model`. The coordinator checks `Request::cache` itself (`may_read` in `prepare`, `may_store` before `put`). Weights revisions: `[[model.local]] revision` goes in the capability record and `ModelNode::revision`; the cache uses `get_rev`/`put_rev` (no read when replicas report different revisions), and `Meta::revision` says which weights answered.
- Docs: DESIGN (data generation and evals, answer cache, `revision` in the example `node.toml`), training plan status, CHANGELOG, site `ymirctl` page.

**Checks:** `cargo fmt --check`, `cargo clippy --workspace --all-targets -D warnings`, `cargo test --workspace` (see the merge entry for results). New tests: coordinator cache modes and revisions (`ymir-core/tests/nodes.rs`), JSONL generation, sources and eval against a mock pool (`ymir-testkit/tests/dataset.rs`), request shape (`batch` unit tests). Synthetic fixtures only; no live inference nodes, no real model.

**Not done:** the teacher license read from the Hub, turning fetched sources into seed prompts, the synced job record (`--resume` is the stopgap), a real dataset.

## 2026-11-22: LF-37 outside MCP servers for agent tasks

**What changed**

- A node can use outside MCP servers (stdio or streamable HTTP) in its agent tasks. They are listed as `[[inference.agent.mcp]]` in `node.toml` ([config and validation](crates/ymir-core/src/mcp_servers.rs)); the tools join the task's registry as `<id>.<tool>` in [`tasks::serve`](crates/ymir-core/src/tasks.rs). Every call asks for approval on the asking device; only linked devices can run tasks (unchanged). A server that fails to start or list its tools is left out with a warning.
- Secrets are references only (`env:NAME` or `file:/path`); a literal secret or a plain-http address (other than loopback) is refused by validation.
- `ymirctl mcp-servers list | add | remove | enable | disable | test` ([source](crates/ymirctl/src/mcp_servers.rs)). A local command asks before it is added (it runs as the user ymird runs as). The commands edit `node.toml` in place with `toml_edit`, so comments and layout survive; a change applies on the next ymird start.
- Built on `weirdnet-mcp` 0.6.1 (features `client`, `agent`) and `weirdnet-agent` 0.5.0's `AgentSession`; no client code was written in Ymir. The wish list for the Weirdnet Lead is in Bragi's [mcp-client design](https://github.com/WeirdBrains/bragi/blob/main/docs/design/mcp-client.md).

**Checks run:** see the merge entry.

**Not tested:** a real stdio server with a real model on a node; Windows.

**What's left:** a pool-wide server list (each node lists its own); `ymirctl chat --tools`; resources and prompts (not in the client yet).

**What's next:** Bragi's MCP client settings.


## 2026-11-22: training plan, white paper reference fixed (branch `feature/training-plan-whitepaper-ref`)

**What changed:** the white paper is not on main yet, so [`docs/design/training.md`](docs/design/training.md) refers to its "Training and fine-tuning (proposal)" section by title instead of linking a file that isn't there (the previous entry's links would not have resolved). Docs only.

**Checks:** none needed (docs only).

**What's next:** link the section's source once the white paper is on main.


## 2026-11-22: training plan, decisions (branch `feature/training-plan-answers`)

**What changed:** [`docs/design/training.md`](docs/design/training.md) records m3talsmith's answers to the open questions (llama.cpp finetune as the trainer; first topic "Decentralized networks: how they work"; Bragi's training toggle after v1; version weights in Ymir's `balance`, with alternatives proposed) and links the white paper's training section. [DESIGN.md](docs/DESIGN.md#non-goals): the non-goal now reads "Ymir feeds, schedules and serves a separate trainer". Docs only.

**Checks:** none needed (docs only).

## 2026-11-22: training plan (branch `feature/training-plan`)

**What changed:** new [`docs/design/training.md`](docs/design/training.md), the implementation plan for the white paper's four training proposals (data generation and evals through batch and the bridge, consented conversation export, fine-tuning as a pool job, adapter serving and versioned rollout). Each has its scope, the modules it maps onto, what it needs from Weirdnet crates, data and compute, risks, and a suggested order with the smallest end-to-end slice first (JSONL batch and `ymirctl eval`, then LoRA adapters on a loaded base). Docs only: no code. Training stays a separate program; the plan proposes rewording the non-goal once approved.

**Checks:** none needed (docs only).

**What's next:** the order was confirmed by the project lead (2026-11-22) and the branch merged into main with `--no-ff`; open questions stay open in the doc. Next: slice 1 (P1) on its own branch, after the Weirdnet requests for P1 and P4a are sent.

## 2026-11-22: LF-38 follow-up, video design corrected (branch `feature/video-design`)

**What changed** in [`docs/design/video.md`](docs/design/video.md): no default frame count or size anywhere; Ymir reports per-model limits (context in use, `max_images`, image token cost and edge, audio length, later native video limits) as reported-or-absent, measured at model load where possible, with operator overrides in `node.toml`. That is now milestone 1 and the prerequisite for Bragi's video. Wish list updated. Docs only.


## 2026-11-22: LF-38, post-v1 video design, server and node side (branch `feature/video-design`)

**What changed**
- New [`docs/design/video.md`](docs/design/video.md): what Ymir does for video. Phase 1 needs nothing for Bragi (frames and sound travel as image and audio parts); suggested small additions are a request-size setting, `max_images` and per-image cost in `Capabilities`, and a transcription role. Phase 2 is a `video` modality and capability flag, routing with a frames fallback hint, backend pass-through, and content roots for large clips. Also opt-in server-side expansion for OpenAI-only clients, `ymirctl mcp` video inputs, privacy, testing, milestones and the Weirdnet wish list. Cross-linked with Bragi's doc.
- Docs only: no code.

**Checks:** none needed (docs only).


## 2026-11-22: `feature/pool-bench` merged into main (`ymirctl bench`)

**What changed**

- Merged `feature/pool-bench` (a0f93de) into main with `--no-ff`: `ymirctl bench` (node×model matrix over Weirdnet, suites quick/standard/hard, `bench list`, `bench compare`, `bench hints`, hints from `--hint @FILE` or the `bench-hints` file) next to main's `ymirctl mcp` and `models`.
- Conflicts resolved by keeping both sides: [`README.md`](README.md), [`HANDOFF.md`](HANDOFF.md), `crates/ymirctl/Cargo.toml`, `crates/ymirctl/src/lib.rs` (modules, the `Mcp` and `Bench` commands, `bridge` stays `pub(crate)`), `crates/ymir-core/src/client.rs` (`post` and the timed chat/embedding calls) and `Cargo.lock` (regenerated).

**Checks run:** `cargo fmt --all --check`, `cargo build --workspace --locked`, `cargo clippy --workspace --all-targets --locked -- -D warnings`, `cargo test --workspace --locked` (results in the merge commit message of the push).

**Not tested:** nothing new beyond the branch's own tests; no release, tag or publish.

## 2026-11-21: `ymirctl mcp` moves to `weirdnet-mcp` 0.6 (LF-16 follow-up, branch `feature/mcp-weirdnet-0.6`, based on main 0536020)

**What changed**

- `weirdnet-mcp = { version = "0.6", features = ["stdio-server", "axum"] }`. `crates/ymirctl/src/mcp/rpc.rs` (205 lines) is deleted: tools are `call_async`, the stdio loop is `weirdnet_mcp::stdio::serve`, HTTP is `weirdnet_mcp::http::router("/mcp", .., Guard::loopback(Auth::token(bridge_token)))`. The schema patching is gone; `messages`, `route`, `format`, `input` and `max_tokens >= 1` live in the tool table.
- Kept: every tool, the `confirm` gates, `--read-only`, `--allow-remove`, images and audio in chat, and our own loopback bind check (the guard checks `Host` and `Origin`, not the bind address). HTTP error bodies are now JSON-RPC errors, and a GET is no longer our hand-made 405.
- New: read-only resources `ymir://status`, `ymir://nodes`, `ymir://models`, `ymir://models/local`.
- Tests: `ToolSpec::check` on the table; scripted stdio session and resources (`crates/ymirctl/tests/mcp_stdio.rs`); HTTP with token, Host, Origin and loopback refusal against the real binary (`tests/mcp_http.rs`); the fake-pool tests in `ymir-testkit/tests/mcp.rs` moved to `handle_async` and gained a resources test.

**Checks run:** fmt, build, clippy -D warnings and the workspace tests; a read-only stdio smoke test against the real pool (tools/list, nodes, resources, models and one chat).

**Not tested:** a real MCP client (Cursor, Claude, VS Code) and a 2026-07-28 client; Windows (CI on the branch is the check).

**What's next:** merge on Michael's OK; a bench tool after `feature/pool-bench` merges.

## 2026-11-21: `ymirctl mcp`, an MCP server (LF-16, branch `feature/mcp-v2`, based on `feature/model-search-pull` b0a425a)

**What changed**

- `ymirctl mcp` (stdio) and `ymirctl mcp --http [ADDR]` (loopback streamable HTTP with the bridge token). Ten read tools and the mutating `ymir_load_model`, `ymir_unload_model`, `ymir_pull_model`; `ymir_remove_model` only with `--allow-remove`; `--read-only` hides every mutating tool. Mutating tools need `confirm: true`. Chat takes images and audio (files or content parts). See [DESIGN.md](docs/DESIGN.md) ("MCP server") and the [ymirctl guide](site/lib/pages/docs_ymirctl.dart) for tools and client config.
- Built on `weirdnet-mcp` 0.5.3 (default features; one `weirdnet-agent` in the lock). The protocol front end is `crates/ymirctl/src/mcp/rpc.rs`, kept small so it can be replaced. Replaces the earlier interrupted `rmcp` attempt (kept as a patch in the dev box scratch folder, not used).
- `BridgeClient::post` (non-streaming OpenAI calls); `service_lines`, `load_cfg` and `bridge` in `ymirctl` are `pub(crate)`.

**Needed from `weirdnet-mcp` (for the Weirdnet Lead):** (1) async `Tools::call`, now each call blocks a thread; (2) `ParamType::Object`, array-of-objects, `enum` and `anyOf`, now patched into `tools/list` here; (3) a server stdio loop and an HTTP route helper with the loopback and bearer rules, now in `rpc.rs`; (4) resources (`resources/list`, `read`), which the earlier attempt had for models and nodes; (5) `image` and `audio` content in tool results.

**Checks run:** see the commit message; unit tests, a scripted stdio client against the real `ymirctl` binary (`crates/ymirctl/tests/mcp_stdio.rs`), and an integration test against a fake pool with mock engines (`crates/ymir-testkit/tests/mcp.rs`): read tools, images, confirm gating, remove, the stdio loop with concurrent requests and batches, and HTTP (token, host, origin).

**Not tested:** `ymir_models_search` and `ymir_pull_model` through MCP against the real Hub (the `models` code has mock-Hub tests); real Cursor, Claude or VS Code clients; audio with a real audio model; `ymirctl bench` as a tool (it is on `feature/pool-bench`, not in this branch's base); a `ymird`-hosted MCP endpoint (not built).

**What's left:** the tools above once `weirdnet-mcp` has them. Already rebased onto the Weirdnet crate bump (`feature/mcp` is the earlier, superseded push; `feature/mcp-v2` is the one to use).

**What's next:** a bench tool after `feature/pool-bench` merges.

## 2026-11-21: Pool-wide model search and download over Weirdnet (LF-30 M2/M3, branch only)

**What changed**

- Moved to `weirdnet-backends` 0.3, `weirdnet-model` 0.5.1 and `weirdnet-alerts` 0.2 together. `engine_alerts.rs` is gone (the crate's `engine.removed` rule covers it). `ymirctl models` now runs on `weirdnet-model`'s `hf::Hub`, `fit::choose`/`rank`, `HfToken` (read on the downloading node only, never sent), `download_parts` and `ModelStore::list`/`remove_local`; the local Hub client, GGUF reader and fit code were deleted. CLI behaviour is unchanged.
- `ymird` serves the owner-only `/weirdnet-backends/models/1` (`ModelsServer` with `RosterAuth`) and uses `Supervisor::add_model`/`remove_model`, so `pull --serve` serves a model without a restart (in memory only). `[model] remote_manage = false` turns it off. `node.toml` is never written remotely; `--register` (with its `.bak`) stays local.
- `ymirctl models search|pull|list|remove --node <name|peer-id|auto|all>` over Weirdnet via `ModelsClient`, with streamed progress. `auto` ranks nodes with `fit::rank` and says why it picked one. A device that is not the owner gets a clear refusal.
- Code: `crates/ymir-core/src/models.rs`, `crates/ymirctl/src/models/remote.rs`; docs in [DESIGN.md](docs/DESIGN.md), the ymirctl site page and the changelog.

**Checks run:** fmt, build, clippy `-D warnings`, `cargo test --workspace`; new tests with mock Hub and throwaway daemons on one machine: owner-only refusal, remote pull progress, `auto` choice, `add_model` without restart, remove, and `node.toml` untouched.

**Not tested:** real pulls on the live pool nodes (deliberately not run), real Hub gated repos, MLX pulls, and Windows.

**What's left:** Bragi Models screen (M4); `--json` output; mmproj download.

**What's next:** the Ymir MCP server.

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

**CI follow-up:** `.github/workflows/macos.yml` no longer runs the removed `engine_shim` test; the MLX unit tests in `ymir-core --lib` still run in the step before. There is no replacement end-to-end mock test for MLX plus llama.cpp on one node yet. The "setup --dry-run picks mlx" job (macos-15) had been failing on main too: a 6 GiB budget picks the Gemma models, which take images, so they correctly stay on llama.cpp (MLX-LM's server is text-only) and the config never has `backend = "mlx"`. The code was right and the grep was stale; the job now checks the MLX-LM line and the llama.cpp picks, and a unit test (`a_small_mac_keeps_image_models_on_llamacpp_and_names_mlx_the_backend`) pins the behaviour. Main's macos.yml needs the same grep change (and the `engine_shim` step stays there while the shim does).

**Checks run:** see the commit's push; `cargo fmt --check`, build, clippy `-D warnings` and tests on the whole workspace.

**Not tested:** a real GPU node with a real Gemma projector (mock engines only); an image across two real machines; MLX on Apple Silicon (no Mac here; the macOS workflow covers it once the branch is pushed through CI); Bragi sending images (see Bragi's HANDOFF).

**What's left:** the two crate gaps above; `weirdnet-model` 0.5 once backends follows; remote `https://` image URLs are not fetched.

**What's next:** LF-16, the Ymir MCP server.

## 2026-11-21: `ymirctl bench`, a harder `hard` suite and hints from a file

**What changed**

- `--suite hard` now separates models: multi-step math and logic, Python functions whose output is run and compared, strict-format instructions, long-context retrieval (sized by `--context`) and tool-call checks. Every check is scored by exact match, parsed JSON or code run in the sandbox, at temperature 0, so a rerun gives the same score for the same answers. `quick` and `standard` are unchanged. Results show a pass rate per category.
- Hints no longer have to be typed. `--hint ADDR` still repeats; `--hint @FILE` reads one address per line (`#` comments); and the file `bench-hints` in the Ymir home is read when it exists. `ymirctl pool addresses` on another node, piped into `ymirctl bench hints add`, fills that file (only dialable addresses are kept: no loopback, link-local or relay circuits). `ymirctl bench hints list` shows what will be dialled. The file lives in the user's home, never in the repo.
- A node whose `[model] callers` is set only serves the devices listed. The benchmark's own peer id is the one `ymirctl pool devices` marks `this node` on the machine running it; add that id to the node's `callers` list.

**Checks run:** `cargo fmt --all --check`, `cargo clippy --workspace --all-targets --locked -- -D warnings`, `cargo build --workspace --locked`, `cargo test --workspace --locked`. New tests cover the hint parser, the file and default-file merge, and the scoring of each new hard check.

**Not tested / limits:** the hard suite's difficulty is only judged against the models on one pool; a stronger model may still score near the top. `--cold` runs against live nodes were not tried.

**What's next:** keep the JSON from each run to spot regressions after an engine upgrade.

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

## 2026-11-21: Weirdnet 0.5 image and audio work merged to main

**What changed**

- `feature/weirdnet-0.5-content-parts` (head `1d08971`) is merged into main with a merge commit: Weirdnet 0.5 crates, capability routing, images and audio through the pool, MLX on weirdnet-backends 0.2, the engine shim removed, and the macOS workflow fix. The branch is kept.

**Checks run:** `cargo fmt --check`, `cargo build --workspace`, `cargo clippy --workspace --all-targets -D warnings` and `cargo test --workspace` on the merge result, all passing. CI and macOS workflow runs on the branch head were green.

**Not tested:** real GPU vision, MLX on Apple Silicon, phone.

**What's left:** stacked branches (model search, MCP, pool bench, OpenAI endpoints) should merge main before they land.

**What's next:** those stacked branches.

## 2026-11-21: Model search and pull merged to main

**What changed**

- `feature/model-search-pull` (head `ce167d8`) is merged into main with a merge commit: `ymirctl models search/pull/list/remove`, the Hub client, and the move to weirdnet-backends 0.3, weirdnet-model 0.5.1 and weirdnet-alerts 0.2. The branch is kept.

**Checks run:** `cargo fmt --check`, `cargo build --workspace`, `cargo clippy --workspace --all-targets -D warnings` and `cargo test --workspace` on the merge result, all passing.

**Not tested:** downloads from the real Hub in CI.

**What's left:** `feature/mcp-v2` needs this merged in before it lands.

**What's next:** merge the MCP server.

## 2026-11-21: MCP server merged to main

**What changed**

- `feature/mcp-v2` (head `52c540f`) is merged into main with a merge commit: `ymirctl mcp` (stdio and loopback HTTP). Before the merge, main was merged into the branch and `mcp.rs` was adapted to the async `models::run` and the new `--node` args from model search. The branch is kept.

**Checks run:** `cargo fmt --check`, `cargo build --workspace`, `cargo clippy --workspace --all-targets -D warnings` and `cargo test --workspace` on the branch head, whose tree equals the merge result.

**Not tested:** `ymir_models_search` and `ymir_pull_model` through MCP against the real Hub.

**What's left:** `feature/pool-bench` conflicts with main in [HANDOFF.md](HANDOFF.md) only.

**What's next:** the pool bench merge.

## 2026-11-21: Yanked Weirdnet crate versions replaced

**What changed**

- Exact pins moved to the patch releases that replace the yanked ones: `weirdnet-device-link` =0.3.0 → =0.3.2, `weirdnet-net` =0.8.0 → =0.8.2, `weirdnet-rules` =0.1.0 → =0.1.1. `weirdnet-alerts` moved 0.2.0 → 0.2.1 in Cargo.lock only, because 0.2.0 required the yanked `weirdnet-rules` =0.1.0. No source changes.

**Checks run:** `cargo fmt --check`, `cargo build --workspace`, `cargo clippy --workspace --all-targets -D warnings` and `cargo test --workspace`, all passing.

**Not tested:** the live pool on the new crate versions.

**What's left:** other branches carry the old pins until they merge main.

**What's next:** none.

## 2026-11-21: Windows fixes for model search

**What changed**

- `models::hub::free_bytes` now works on Windows (`GetDiskFreeSpaceExW`); it returned `None` there before, so the fit check never saw free disk space.
- `plan::id_for_file` takes the file name after either `/` or a backslash; on Windows it used the whole path as the model id, so scans and registered-model matching broke. A new unit test covers both path styles.
- The split-quant pull test accepts the TOML literal-string quote that Windows paths produce.

**Checks run:** `cargo fmt --check`, `cargo build --workspace`, `cargo clippy --workspace --all-targets -D warnings`, `cargo test --workspace`, and `cargo clippy -p ymir-core -p ymirctl -p ymird --all-targets --target x86_64-pc-windows-gnu -D warnings`, all passing.

**Not tested:** the Windows tests were not run locally; the Windows CI job is the check.

**What's left:** none.

**What's next:** none.

## 2026-11-21: Ymir MCP server on weirdnet-mcp 0.6 merged

**What changed**

- Merged `feature/mcp-weirdnet-0.6` into main with a merge commit. `ymirctl mcp` now uses the published `weirdnet-mcp` 0.6 and the local JSON-RPC module is gone.

**Checks run:** `cargo fmt --check`, `cargo build --workspace`, `cargo clippy --workspace --all-targets -D warnings` and `cargo test --workspace` on the merged tree.

**Not tested:** Windows locally; main's CI is the check.

**What's left:** none.

**What's next:** none.

