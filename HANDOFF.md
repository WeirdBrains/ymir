# Handoff

A running log of finished work, newest first. Each entry says what changed, the commits, the checks run and what was not tested, what is left, and what is next.


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

