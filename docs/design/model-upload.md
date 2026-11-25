# Uploading models to Ymir: implementation plan

Status: plan, 2026-11-24. Approved by the project lead on 2026-11-24 with two changes (U5 moved up to follow U1;
U6 waits on Weirdnet requests B and C, with no interim peer copy in Ymir). m3talsmith answered the open questions the
same day (see "Decisions"). U1 (local upload), U5 (trainer outputs on the same path), U2 (upload to another node)
and U3 (`--rollout`) are built; see "U1 as built", "U5 as built", "U2 as built" and "U3 as built".

Ymir is designed by m3talsmith and published by [Weird Brains LLC](https://weirdbrains.com).

## Why

The main use case: **fine-tune a model off the network, on your own machine, then put the result into the
pool.** Today a model reaches a node only from the Hugging Face Hub (`ymirctl models pull`, Bragi's Models
screen) or from a run on the node itself (`ymirctl train run`, P3a). A model trained elsewhere has to be pushed
to a Hub repo first, or copied onto the node by hand and written into `node.toml`.

This plan adds `ymirctl models upload` and an Upload dialog in Bragi that send local files to a node or a pool
over Weirdnet, check them, record where they came from, serve them, and hand them to a versioned rollout
(P4b) whose eval decides how much traffic they get. Full fine-tuned models and LoRA adapters are both
first-class. P3a's trainer outputs use the same path, so a run on one node can reach the others.

## Ground rules this plan keeps

- **The node protocol stays the only door.** Uploads go over Weirdnet streams, owner-authenticated like every
  other Ymir protocol. No new listener, no HTTP upload endpoint, nothing on the loopback bridge.
- **Generic code goes to Weirdnet, through the Weirdnet lead.** Reading and hashing model artifacts, the store
  layout, the models/1 wire protocol and its client are `weirdnet-model` and `weirdnet-backends`. This repo
  never edits a `weirdnet-*` crate; each need is listed under "Requests for the Weirdnet lead".
- **Weights are data, never code.** Pickled weights (`.bin`, `.pt`, `.pth`, `.ckpt`) and Python files are refused;
  safetensors and GGUF are parsed by header only. Nothing uploaded is ever run.
- **Provenance is recorded, not demanded.** Every upload gets a record of its files, hashes, revision and base;
  the license and dataset are recorded when given and are not required (decision 2).
- **Live nodes.** Uploading onto a node that serves the pool is ordinary model management (like `pull --node`)
  but large: tests use `YMIR_HOME` in a temp directory, mock engines and tiny synthetic GGUFs, never a real model.

## What exists today (surveyed 2026-11-24)

| Piece | Where | What upload reuses |
|---|---|---|
| `ymirctl models search \| pull \| list \| remove [--node]` | `crates/ymirctl/src/models/`, `crates/ymir-core/src/models.rs` (`ModelsHub`) | The `--node NAME\|peer\|auto\|all` resolution, progress lines, the printed `[[model.local]]` entry and `--register` (local only). |
| `/weirdnet-backends/models/1` | `weirdnet-backends` 0.4.3 `models` (`ModelsCommand`, `Pull`, `Pulled`, `StoredEntry`, `ModelsService`, `ModelsClient`) | One CBOR request per stream with the caller's cert, streamed `Progress`, cancel by closing the stream, `Pull::serving` to add to the supervisor live, `adapter_for` to serve an adapter on a base. No way to send bytes to the node. |
| Model store | `weirdnet-model` 0.5.2 `ModelStore`, `StoredModel`, `StoredAdapter`, `adapter.json`, `download_parts`, `download_adapter`, `Consent`, `sha256_of` | Resumable `.part` files with a size and SHA-256 check before the rename; adapters under `adapters/ID/` with `adapter.json` (format, base model, architecture, repo, rank, alpha). |
| GGUF headers | `weirdnet-model` `gguf::read_header` (`GgufHeader`: kind model/adapter, architecture, base model, repo, KV shape) | Telling a model from a LoRA from a projector, the architecture check, the context plan. |
| Adapter checks | `weirdnet-backends` `ModelLaunch::check_adapters`; Ymir `adapters::drop_wrong_base`, `base_revision` / `base_quant` | Refusing an adapter whose base architecture, revision or quantisation differs. |
| Fit | `weirdnet_model::fit`, Ymir `kvfit` / `ctxplan` | Fit verdict per node; adapter bytes counted in the base's footprint. |
| Catalog | supervisor status rows -> capability record (`ModelEntry::revision`, capabilities, adapter `base`) -> catalog/1 (`CatalogModel::base`, `with_inputs`) | An uploaded model or adapter appears in the pool the moment the supervisor serves it; no catalog change. |
| Revisions and cache | `[[model.local]] revision`, `put_rev` / `get_rev` | An upload's content hash becomes its revision, so answers from older weights under the same id are never replayed. |
| Rollouts (P4b) | `[[inference.rollout]]`, `ymirctl models rollout add \| weight \| rollback \| retire`, `--eval` gate above 10 percent | An upload lands as a new version with weight 0, then earns traffic through the eval. |
| Training runs (P3a) | `ymirctl train run`, `run.json` (`RunRecord`: base, dataset, hyperparameters, trainer build, adapter hash) | The provenance of a trainer output; `train run --node` for a remote node is not built. |
| Auth | `remote::roster_auth` (`RosterAuth`, no `require_roles`); device grants `ymir.inference`, `ymir.model` (`weirdnet-sync` 0.7 roster); cert-only service links (M10) | models/1 admits any linked, unrevoked device of the owner. `RosterAuth::require_roles` can narrow it to a granted role. |
| Bragi | `bragi-core` `ModelsManager` (`models.rs`, `models_core.rs`) over `ModelsClient` with the device's cert; Models screen in Flutter and GPUI (search, fit badges, target auto/node/all, progress, cancel, remove); several pools (`pools.rs`); rollout versions folded in the model list (v0.6.0); cert-only server links (v0.5.0, default in progress) | The upload dialog sits on the Models screen and reuses its node picker, progress and cancel. |

Gaps that matter here: models/1 frames are at most 64 KiB (`weirdnet-net` `Limits::max_frame`) with a 1 MiB
window, so bytes must be chunked; nothing reports a node's free disk; `StoredModel` keeps no origin, license or
revision; the store has no layout for safetensors or MLX directories (Ymir keeps them under `models_dir/vllm/` and
`models_dir/mlx/` itself); projectors are pinned per catalog model in `ymir_core::vision` only.

## What can be uploaded

| Kind | Format | Engine | Lands in | Checks |
|---|---|---|---|---|
| Full model | GGUF, one file or split (`-00001-of-0000N`) | llama.cpp | the store's GGUF layout | header readable, `general.type` model, architecture known to the pinned engine, split parts complete, fit |
| Full model | safetensors directory (`config.json`, tokenizer files, `*.safetensors`, optional `*.safetensors.index.json`) | vLLM | `models_dir/vllm/<id>/` | safetensors headers parse, `config.json` has `architectures`, index covers every shard, no `.py`, no pickles, fit from `config.json` |
| Full model | MLX directory (`config.json`, `*.safetensors`, tokenizer) | MLX-LM | `models_dir/mlx/<id>/` | as safetensors; only to nodes whose engine is MLX |
| Projector | mmproj GGUF (`clip` architecture) | llama.cpp `--mmproj` | beside its model | header says projector; `--projector-for BASE` required; checked against the base's vision tower size where the header says it |
| LoRA adapter | GGUF LoRA (`general.type` adapter, `adapter.type` lora) | llama.cpp | `adapters/<id>/` | architecture equals the base's; base revision and quant recorded; base must be served on llama.cpp |
| LoRA adapter | PEFT directory (`adapter_config.json`, `adapter_model.safetensors`) | vLLM | `adapters/<id>/` | `peft_type` LORA, `base_model_name_or_path` matches the base's Hub id or `--base-matches` override, `r` within the base's `--max-lora-rank`; base served on vLLM |

A directory is sent as its list of files (never a tarball), so each file is checked, resumed and hashed on
its own. A PEFT adapter for a llama.cpp base is refused with the command that converts it
(`convert_lora_to_gguf.py`); a GGUF for a vLLM-only node likewise. Merging an adapter into its base, quantising
and converting formats stay off-node, in the user's own toolchain.

## Provenance

Every upload carries a provenance record, stored beside the files (`provenance.json`, next to `adapter.json`
for adapters) and returned by `models list`:

```json
{
  "format": "ymir-provenance/1",
  "kind": "model | adapter | projector",
  "id": "qwen3-8b-support-r3",
  "files": [{"path": "qwen3-8b-support-r3.Q4_K_M.gguf", "size": 4920000000, "sha256": "..."}],
  "revision": "sha256:3f2a9c0d41b7e8a5",
  "origin": "upload | train | hub",
  "license": "apache-2.0",
  "base": {"id": "qwen3-8b", "hub": "Qwen/Qwen3-8B", "revision": "...", "quant": "Q4_K_M", "sha256": "..."},
  "dataset": {"format": "ymir-dataset/1", "sha256": "...", "rows": 1200, "teacher_license": "apache-2.0"},
  "trainer": {"name": "llama.cpp finetune", "build": "b12628", "run": "optional run id"},
  "uploaded_by": "device label from the roster",
  "uploaded_ms": 1795550000000
}
```

- **License** is optional (`--license`, decision 2) and recorded when given; with none the record has no
  `license` field. When the base's license or the dataset's teacher license is known and differs from the declared
  one, ymirctl warns.
- **Base** is required for adapters and optional for full fine-tunes (strongly suggested: it is what lets the eval
  compare like with like). Filled from the GGUF header (`general.base_model.*`) or `config.json` when present;
  `--base` names a pool model and records its revision, quant and hash.
- **Dataset** is optional: `--dataset-manifest FILE` takes a P1 `ymir-dataset/1` or P2 `ymir-conversations/1`
  manifest (only its hash, counts and licenses are recorded, never rows). A conversations manifest stays useful
  later: `ymirctl context withdrawn` can name uploads trained on conversations whose consent was withdrawn.
- **Revision** is the first 16 hex digits of the SHA-256 of the sorted `path:sha256` list. It goes in the
  capability record as the model's revision, so the answer cache and the base-revision check already use it.
- The record is signed by nobody; it is what the uploader said, checked against the bytes where it can be
  (hashes, header architecture, base hash). It is metadata for people and for the rollout, not a trust anchor.

## The upload path

### One node (the first slices)

A new models/1 command, `Upload`, in one stream:

1. **Offer.** The client sends `Upload { id, kind, files: [{path, size, sha256}], provenance, adapter_for,
   projector_for, serve, replace }` with its cert.
2. **Admission and pre-checks, before a byte of weights moves.** The node authorizes the caller (see "Auth"),
   checks names (no `/`, no `..`, no hidden files, allowed extensions), the id is free (or `replace`), the total size
   against free disk (keeping 10 percent or 20 GiB free, decision 4), and, for adapters, that the base is served on an
   engine that takes this format. It answers `UploadReady { upload, have: [bytes already held per file] }` or a
   refusal with a reason: `exists`, `no-space`, `wrong-engine`, `no-base`, `bad-name`.
3. **Header first.** The client sends the first chunk of each GGUF or safetensors file before the rest; the node
   reads the header as soon as it has it and refuses early (`bad-format`, `wrong-arch`, `not-adapter`) rather than
   after gigabytes.
4. **Chunks.** `UploadChunk { file, offset, data }` with at most 48 KiB of data, under the 64 KiB frame limit; the
   stream's window gives back-pressure. The node appends to `uploads/<upload>/<path>.part`, where `upload` is the
   hash of the offer's file list, so the same files resume whoever sends them. It reports `Progress` every 250 ms as
   pulls do.
5. **Verify and place.** On `UploadDone` the node hashes each part, compares size and SHA-256, re-reads headers,
   runs the format checks above and the fit, then renames into the store (atomic per file, the directory last),
   writes `provenance.json` (and `adapter.json` for an adapter), and, with `serve`, adds it to the supervisor as
   `pull --serve` does (an adapter on its base with `Supervisor::add_adapter_live`). It answers `Uploaded(Pulled)`
   with the files, revision, whether it serves, and for an adapter when it serves (`AdapterChange`). **Only vLLM
   loads a new adapter live.** On a llama.cpp base a new adapter file is `AdapterChange::NextLoad`: it serves after
   the base restarts or next loads, and ymirctl and Bragi say exactly that, never that it is live.
6. **Cancel and resume.** Closing the stream cancels; parts stay. Offering the same files again resumes from
   `have`. Abandoned parts are removed after a TTL (default 7 days) or by `models upload --abandon`.

**Persistence (decision 5).** An upload is saved into the target node's `node.toml` by default (the
`[[model.local]]` entry, or `[[model.local.lora]]` under its base, keeping `node.toml.bak`), so it and any rollout
version that names it survive a restart; `--no-persist` keeps it in memory only and prints the entry. This is a
deliberate exception to `pull --node`, which never edits another node's `node.toml`. Locally (U1) the same default
applies: `--no-register` only prints the entry.

### A pool

`--pool` (every model node whose engine takes the format and that it fits) or `--node all` / a list. Pushing the
same 5 to 70 GB from a laptop once per node wastes its uplink, often through a relay. So:

- **Stage once, nodes pull from each other.** The client uploads to one node (the best-connected fitting node, or
  `--stage NODE`); then every other target pulls the files from that node over a models/1 `Fetch` read command,
  checked against the same hashes, node to node. Progress for every node streams back to the client.
- **Fallback: push to each.** When no node can stage (older nodes, `--no-stage`), the client pushes to each target
  in turn; resume keeps a broken one cheap.
- **Hub as staging**: see "Stopgap until U2" below.

### Stopgap until U2: a private Hub repo (decision 7)

Until uploads over Weirdnet exist, a model trained elsewhere reaches a remote node through the Hub, with no change to
Ymir:

1. Push the GGUF (or adapter) to a private Hugging Face repo with your own Hub tools.
2. Give each node that should pull it a read token for that repo, in its own environment or files (`HF_TOKEN`,
   `YMIR_HF_TOKEN_FILE`, `state/hf-token`); the token never travels in Ymir's messages.
3. `ymirctl models pull OWNER/REPO --node NAME --serve` (a model), or `ymirctl models pull OWNER/REPO --node NAME
   --adapter-for BASE --serve` (an adapter; on llama.cpp it serves after the base's next load).

It puts unreleased weights on a third-party host and records no provenance beyond the repo, so it stays a stopgap.
On the node itself, `ymirctl models upload` (U1) works today.

## Auth and permissions

- **Who (decision 1).** Every device linked to the owner's roster may upload, as models/1 admits today
  (`RosterAuth` without roles). There is no `ymir.models.write` grant. Revoking a device in the roster stops its
  uploads, like everything else it does.
- **Node to node.** Staged replication (U6) needs the target node to read from the staging node: `Fetch` admits a
  linked device, and only for an upload the owner started (the staging node keeps the offer and the target names it
  by its upload id).
- **Identity.** The caller's `NodeCert` travels in the request as for every models/1 command; Noise authenticates
  the stream. The uploading device's roster label goes in the provenance record.
- **Disabling.** `[model] remote_manage = false` turns off all of models/1, uploads included; `[model] upload =
  false` turns off uploads only.

## Validation and disk space

- **Names and types**: as above; refuse symlinks, device files, pickles and `.py`.
- **Headers**: GGUF via `weirdnet_model::gguf::read_header` (kind, architecture, base, split counts); safetensors
  via its 8-byte length plus JSON header (dtype, shapes, total bytes must equal the file size); `config.json` and
  `adapter_config.json` parsed with size limits. Anything unreadable is refused.
- **Adapters match their base**: GGUF architecture equals the base's; PEFT `base_model_name_or_path` equals the
  base's Hub id (or `--base-matches` records an override); `base_revision` and `base_quant` recorded so
  `drop_wrong_base` keeps catching drift.
- **Fit**: full models against the node's model budget with `weirdnet_model::fit` (the same verdict `models search`
  shows). `fits` serves; `partial` serves with a warning; `too big` stores only with `--store-only` and never serves.
  Adapters add their bytes to the base's footprint.
- **Disk (decision 4)**: no size cap per upload and no total per node. The one limit: 10 percent of the models
  disk or 20 GiB, whichever is more, always stays free. The node reports free and total bytes in `UploadReady`,
  refuses an upload that would go below that, and rechecks before each rename. U1 applies the same rule locally.
- **Concurrency**: one upload per id at a time; a second offer for the same files joins the first (same `.part`).

## Catalog, versions and rollouts

- **Catalog.** Once served, an upload is a model like any other: the supervisor's status row feeds the capability
  record (with the new revision), catalog/1 lists it, adapters with their `base`. Bragi shows it at once.
- **Naming for versions.** Rollouts want a new id per version (`qwen3-8b+support-r3`) so the old one keeps serving.
  ymirctl suggests `<name>-r<N>` from the rollout's next version when `--rollout` is given and `--name` is not.
  `--replace` (same id, new revision) is allowed for a model outside any rollout; for a model that a rollout version
  names, it is refused.
- **Feeding a rollout.** `--rollout NAME` makes the upload the next version of NAME on the inference nodes:
  after the model node serves it (an adapter on a llama.cpp base only after the base's next load: ymirctl says so
  and the version waits at weight 0 until it answers), ymirctl asks the inference node (its local ymird, `ControlRequest::Rollouts`) to
  `rollout add NAME@N --model ID --weight 0`. Weight 0 means it is pinnable as `NAME@N` and takes no traffic.
  `--eval ROWS.jsonl [--judge MODEL]` then runs `ymirctl eval` on `NAME@N` and on the version with the most traffic,
  and `--canary W` sets the weight: up to 10 percent freely, more only with the eval passing the existing P4b gate
  (or `--force`). Rollback and retire stay `models rollout rollback | retire`. Weights shared by several inference
  nodes are still per node (a P4b gap), so `--rollout` applies on every inference node the user names
  (`--inference all`) and reports each.
- **Provenance in the rollout.** `models rollout list` shows each version's origin, license, base and dataset hash
  from the model's provenance, so the person moving traffic sees what they are promoting.

## Training outputs (P3a) on the same path

- `ymirctl models upload --from-run RUN_DIR` reads `run.json` (`RunRecord`): the adapter file, base and revision,
  dataset record and license, trainer build, run id. It builds the provenance record (origin `train`) and sends
  the adapter like any other upload, to other nodes or a pool.
- A run on a node writes the same `provenance.json` beside its adapter (origin `train`), so `models list` and
  rollouts treat local runs and uploads alike, and `train run --add --rollout NAME` can join a rollout through the
  same code.
- When `train run --node` (remote) is built, its adapter reaches the other nodes by staged replication from the
  training node: no new path.
- Off-network fine-tunes are the reverse direction of the same record: the user trains with their own tools on a
  dataset made by `ymirctl batch --jsonl` or `context export --training`, and passes its manifest with
  `--dataset-manifest`.

## ymirctl UX

```text
ymirctl models upload <FILE|DIR>...
    [--node NAME|peer|auto | --pool | --node all]   # default: this machine, no daemon needed
    [--adapter-for BASE] [--projector-for BASE]     # kind otherwise read from the files
    [--name ID] [--license SPDX]
    [--base MODEL] [--dataset-manifest FILE] [--from-run RUN_DIR]
    [--serve] [--no-persist | --no-register] [--store-only] [--replace]
    [--rollout NAME [--eval ROWS.jsonl [--judge MODEL]] [--canary W] [--inference NAME|all]]
    [--stage NODE | --no-stage] [--dry-run] [--abandon]
```

- `--dry-run` reads and hashes locally, prints kind, format, architecture, base, revision, size, the per-node fit
  and free disk, and the provenance it would send; nothing moves.
- Progress is one line per node (`staging 41% 2.0 GB/4.9 GB 38 MB/s via relay`), as pulls show today, then the
  served id, revision, what was saved to node.toml, and for a llama.cpp adapter that it serves after the base's next
  load.
- Exit codes and refusal reasons are the protocol's (`no-space`, `wrong-arch`, ...), worded with the fix.
- `ymirctl models list [--node]` gains `ORIGIN` and `LICENSE` columns; `--provenance ID` prints the record.

## Bragi UX

- **Upload** button on the Models screen (Flutter and GPUI, same `bragi-core` state; the terminal app gets
  `/models upload PATH`). Desktop first (decision 6): GPUI, then the Flutter desktop builds; phones later.
- **Dialog**: file or folder picker; Bragi reads the files locally (same `weirdnet-model` inspection) and shows
  kind, format, architecture and size before anything is sent; target picker (auto, a node, the pool) with each
  node's fit and free disk; for adapters a base picker filtered to served bases of the matching architecture and
  engine; name; license (optional); optional dataset manifest; an "Add
  to rollout" section (rollout name, starting weight 0, optional canary with the eval file).
- **Progress**: one bar per node with speed and path (direct or relay), Cancel (keeps parts), Resume after a restart
  of Bragi (the upload id is the file list's hash, kept in Bragi's state), and the result row linking to the model in
  the catalog and to its rollout version.
- **Pools**: the dialog uploads to the active pool (`pools.rs`); switching pools mid-upload is refused until it is
  cancelled or done.
- **Permissions**: any linked device may upload (decision 1); an unlinked or revoked one sees the controls off with
  the reason, as the Models screen does today.
- Logic in `bragi-core` (`ModelsManager::upload`, over the `weirdnet-backends` client); the shells only draw it.

## Requests for the Weirdnet lead

A, B and C were sent to the Weirdnet lead on 2026-11-24; D is held. The lead accepted A to C the same day as
`weirdnet-model` 0.5.3 and `weirdnet-backends` 0.4.4 (released 2026-11-24), with these changes to the shapes below:

- `ArtifactKind::Adapter { format: AdapterFormat }`, and `ArtifactFormat::{Gguf { parts }, Safetensors, Mlx}` with no
  `Peft` variant (PEFT is an adapter format). Projectors are `GgufKind::Projector`.
- `inspect` reuses `gguf::read_header`, `hf::peft_lora` and `ModelLaunch::check_adapters`.
- Importing an adapter also writes `adapter.json`; adapter sizes and `remove_adapter` leave `provenance.json` out
  of the count.
- The partial-upload staging folder lives outside the models tree.
- Refusals are `Error::Model` with stable reason strings. There is no `StoredEntryKind` type.
- Live serving uses `add_adapter_live`, and `ModelLaunch::mmproj` for projectors. Only vLLM loads a new adapter
  live; on llama.cpp a new adapter file is `NextLoad` (the base must restart or reload).
- C: with no new role (decision 1), the "any of these roles or a full-identity device" helper may not be needed;
  uploads use plain `RosterAuth`. The lead may drop that part.

None blocks U1 or U5; U2 needs A and B; U6 needs B's peer pull (`Fetch`, `PullSource::Peer`).

**A. `weirdnet-model`: inspect, hash and import local artifacts, with provenance.**
- `artifact::inspect(paths: &[PathBuf]) -> Result<Artifact>`: `Artifact { kind: ArtifactKind (Model, Adapter
  { format: AdapterFormat }, Projector), format: ArtifactFormat (Gguf { parts }, Safetensors, Mlx, Peft),
  architecture: Option<String>, base_model: Option<String>, info: Option<GgufInfo>, files: Vec<ArtifactFile { path,
  size, sha256 }>, revision: String }`. Reads GGUF headers (projectors as their own kind), safetensors headers
  (length and dtype/shape totals checked), `config.json` and `adapter_config.json` with size caps; refuses
  pickles, `.py`, symlinks. Streaming SHA-256 with a progress callback.
- `Provenance` (serde, `format = "weirdnet-provenance/1"` or app-named): kind, files, revision, origin, license,
  base (id, hub, revision, quant, sha256), dataset (format, sha256, counts, licenses), trainer, uploaded_by, time,
  plus an `extra` map for app fields.
- `ModelStore::import(staged: &Path, artifact: &Artifact, provenance: &Provenance, id: &str) -> Result<StoredEntryKind>`:
  verifies hashes, places files atomically in the store layout, including new `safetensors/<id>/` and `mlx/<id>/`
  directory layouts so apps stop keeping their own; writes `provenance.json`. `StoredModel::provenance` and
  `StoredAdapter::provenance: Option<Provenance>`, read back by `list()` / `adapters()`.
- `ModelStore::free_bytes() -> u64` (statvfs / `GetDiskFreeSpaceEx`) and `ModelStore::bytes_by_origin(origin)`.
- Why: every app that lets a person add their own weights (Ymir, Bragi, Mouser, Morph) needs the same parsing,
  hashing and layout, and the store is already this crate's. Provenance next to the files is what lets a node
  answer "where did this come from" after the fact.

**B. `weirdnet-backends` models/1: upload, fetch, disk and a write policy.** (As sent; after decisions 1 and 4 the
write policy needs only the free-space floor and the part TTL, no caps, quotas or separate authorizer.)
- `ModelsCommand::Upload(Upload)`: `Upload { id, files: Vec<ArtifactFile>, provenance: Provenance, adapter_for:
  Option<String>, projector_for: Option<String>, serve: bool, replace: bool, store_only: bool }`.
  Replies: `ModelsReply::UploadReady { upload: String, have: Vec<u64>, free: u64 }`, then `Progress`, then
  `ModelsReply::Uploaded(Pulled)` (`Pulled` gains `revision: Option<String>`). Client frames after the offer:
  `UploadFrame::Chunk { file: u32, offset: u64, data: ByteBuf }` (≤ 48 KiB) and `UploadFrame::Done`. Header-first
  early refusal. Refusals as `RemoteError::Refused { reason }` with stable reasons: `exists`, `too-big`, `no-space`,
  `quota`, `wrong-engine`, `wrong-arch`, `no-base`, `bad-name`, `bad-format`, `not-allowed`. Parts under
  `uploads/<upload>/`, resumable by the offer hash, abandoned after a TTL.
- `ModelsCommand::Fetch { upload_or_id: String, file: u32, offset: u64 }`, streaming chunks, so a node can pull a
  staged upload from another node; and `Pull { source: PullSource::Hub | PullSource::Peer { node, upload } }` on the
  target, verified against the offer's hashes.
- `ModelsClient::upload(node, Upload, progress) -> Result<Pulled, RemoteError>` reading local files (resume from
  `have`), and `ModelsClient::replicate(stage, targets, upload, progress)`.
- `ModelsService::with_write_policy(WritePolicy { auth: Option<Arc<dyn Authorize>>, max_upload: Option<u64>,
  quota: Option<u64>, keep_free: KeepFree, part_ttl: Duration })`: a separate authorizer for `Upload`, `Fetch` and
  `Remove`, so apps can require a role (`RosterAuth::require_roles`) for writes while reads stay owner-wide.
- `ModelsReply::List` entries carry provenance (from A), and `ModelsCommand::Disk` answers free bytes, quota and
  usage, so clients can show them before an offer.
- `Supervisor::add_model` for safetensors and MLX directories from the store, and a projector attached to a served
  model (`ModelLaunch::mmproj` set live, restart on next load as adapters do).
- Why: models/1 is the generic owner-only model-management protocol Ymir and Bragi already share; Bragi only gets
  uploads if the client is in this crate, not in Ymir. The 64 KiB frame limit and resumable parts are protocol
  concerns.

**C. `weirdnet-sync` / `weirdnet-agent`: a model-write device grant (likely not needed after decision 1).** Confirm that `DeviceGrant` roles are free-form
app strings that the owner can grant to personal (non-service) devices, so Ymir can define `ymir.models.write`; if
not, add that. And a `RosterAuth` helper that admits a caller holding any of a role list *or* being a full-identity
device, so `[model] upload = "linked"` and `"granted"` are one line each. Why: uploads are the first write that
should not be open to every linked device, and the 2026-11-20 rule says to authorize on grants, not cert labels.

**D. Later, optional: `weirdnet-agent` catalog/1 origin.** `CatalogModel::origin` (`hub | upload | train`) and
`license`, from the capability record, so Bragi and other apps can label pool models without a models/1 call.

## Risks

- **Disk and bandwidth.** Full models are 5 to 70+ GB; through a relay that is hours. Mitigations: resume,
  staging and node-to-node replication, early header refusal, the free-disk check before sending, a warning
  when the path is relayed.
- **Bad or hostile files.** A malformed GGUF or safetensors header reaching a parser or the engine. Mitigations:
  header parsing with size caps in one tested crate, no pickles, no code, engines never run with remote code.
  Uploads are open to every linked device (decision 1), so a linked device that is compromised can add weights;
  revoking it in the roster is the answer, as for everything else it can do.
- **Wrong base.** Adapters trained on another revision or quant serve badly. Mitigations: architecture, PEFT base and
  revision/quant checks already in P4a, recorded in provenance.
- **Licenses.** Fine-tunes inherit the base's terms and may carry a teacher's output restrictions. Mitigations:
  the license recorded when given (not required, decision 2), base and dataset licenses recorded, warnings on
  mismatch; Ymir cannot judge a license, only record it.
- **Rollout mistakes.** Promoting an untested upload. Mitigations: weight 0 on arrival, the existing 10 percent eval
  gate, rollback.
- **Live nodes.** Uploading and loading onto a node that serves traffic competes for disk, I/O and memory.
  Mitigations: `nice`d hashing, loads planned by `ctxplan` like any model, `--store-only`.
- **Dependency on the Weirdnet lead** for A and B; slice 1 does not wait on them.
- **Consent drift.** A model trained on conversations whose consent is later withdrawn stays in the pool.
  Mitigation: the dataset manifest hash in provenance lets `context withdrawn` name affected uploads; acting on it
  stays the owner's call.

## Suggested order (smallest end-to-end slice first)

1. **U1: local upload, GGUF model and GGUF LoRA, with provenance.** Built 2026-11-24. `ymirctl models upload FILE
   [--name ID] [--adapter-for BASE] [--license L] [--no-register]` on the machine itself (no daemon, no network):
   read the header, hash, import into `models_dir/uploads/ID/` / `adapters/ID/` atomically, write `provenance.json`
   (and `adapter.json`), set `revision` and add the entry to node.toml. Interim inspection and provenance code in `ymir-core`, marked "to move
   to weirdnet-model" until A ships. Covers the main use case end to end when the fine-tune is made on, or copied to,
   a node machine, and fixes the record format.
2. **U5: P3a outputs on the same path.** `--from-run RUN_DIR` reads `run.json`; `train run` writes the same
   `provenance.json` (origin `train`) beside its adapter; `train run --rollout` later uses U3's step. Moved up by the
   lead (2026-11-24): it needs only U1.
3. **U2: upload to one node over Weirdnet** (`--node NAME|auto`, `--serve`, saved to the node's node.toml unless
   `--no-persist`), chunked, resumable, sha256-checked, with the free-disk check. Needs A and B. This is the main use case from a laptop.
4. **U3: into a rollout.** `--rollout NAME [--eval ROWS] [--canary W]`, provenance in `rollout list`.
5. **U4: Bragi upload dialog** (desktop first, decision 6), on the Models screen, with progress, cancel, resume
   and "add to rollout". Built without "add to rollout" in Bragi 0.7.0 (see "U4 as built").
6. **U6: pools.** `--pool` / `--node all` with staging and node-to-node `Fetch`; push-to-each fallback. Waits on
   Weirdnet B's peer pull; Ymir does not stub a peer copy of its own (lead, 2026-11-24).
7. **U7: more formats.** safetensors for vLLM, MLX directories, PEFT adapters, mmproj projectors.
8. **U8 (optional): catalog origin** in Bragi's model list once D exists.

Tests for each slice use temp `YMIR_HOME`s, tiny synthetic GGUF and safetensors files written by the test, mock
engines and the in-memory or loopback Weirdnet transport (two daemons for U2, U3 and U6), including: a hash
mismatch, a truncated part resumed, a wrong-architecture adapter refused before the body, the free-space floor, an
unlinked device refused, a pickle refused, and a rollout version arriving at weight 0.

## U1 as built (2026-11-24)

- `ymir_core::upload`: `inspect` (a regular `.gguf` file, not a symlink or a split shard; `read_header` says model or
  LoRA; projectors and other adapter types refused), `import` (copies into a hidden temporary file in the target
  directory while hashing, syncs, links it into place without replacing anything, then writes `provenance.json`
  and, for adapters, `adapter.json` in `weirdnet-model`'s shape so `ModelStore::adapters` lists it like a pulled
  one), `revision_of`, `check_adapter_base`, `check_license`, `keep_free`. The same bytes uploaded again under the same
  id are reported and not copied; other bytes under a taken id are refused.
- `ymirctl models upload FILE [--name ID] [--adapter-for BASE] [--license L] [--role R] [--no-register] [--dir D]`:
  an adapter's base must be a llama.cpp `[[model.local]]` whose file is on this machine with the same
  architecture; its `base_revision` and `base_quant` are recorded. Entries are added to node.toml by default
  (`local::register` for models, `models adapter add`'s validated edit for adapters); the adapter message says it
  serves after a restart or the base's next load. `--node` came with U2 (below).
- **Switched to Weirdnet (2026-11-24, `weirdnet-model` 0.5.3 / `weirdnet-backends` 0.4.4):** `inspect` hashes with
  `artifact::inspect`, `import` stages a copy outside the store (`<models_dir>.importing/`) and places it with
  `ModelStore::import` (which writes `adapter.json` itself), the record is `weirdnet-model`'s `Provenance`
  (`weirdnet-provenance/1`; the architecture and a run's settings go in its extra fields), and the free-space floor is
  `KeepFree::default()`. Ymir keeps its own rules on top (one GGUF model or LoRA, ids, optional license, the "same
  bytes again" answer). One change: a model now goes to `models_dir/<id>/` (the store's layout, the same as an
  upload over Weirdnet) instead of `models_dir/uploads/<id>/`; models uploaded there before stay, and uploading the
  same bytes again still finds them. Old `ymir-provenance/1` records read as `Provenance`.

## U5 as built (2026-11-24)

- **`train run` imports its own adapter** with `ymir_core::upload::import` (origin `train`): checked as a GGUF LoRA,
  hashed, linked into `models_dir/adapters/ID/ID.gguf` (was `adapters/ID.gguf`) with `provenance.json` and
  `adapter.json` beside it, and the run's copy removed. `train::import_spec(record, adapter)` builds the record from
  the run: base (id, revision, quant), the dataset record (name, hash, format, license, teacher, rows) and the
  trainer (`name`, `build`, `interface`, `run`, `node`, `trained_from`, `hyper`, `steps`, `final_loss`). The weights'
  `license` is left empty: the dataset's license is the rows', not the adapter's.
- **One revision for the same bytes everywhere.** `run.json`'s adapter gains `revision` (the content revision), and
  `train run --add` registers that instead of the run id, so the adapter has the same revision on every node it
  reaches and in every rollout. Runs recorded before this keep their run-id revisions; nothing on disk moves.
- **`ymirctl models upload [FILE] --from-run RUN_DIR`** reads `run.json` (format `ymir-train-run/1`, status `done`,
  an adapter) and uploads its adapter with the same record: FILE defaults to the run's adapter in this node's models
  directory; the file's SHA-256 must be the record's; `--adapter-for` defaults to the run's base and the id to the
  run's adapter id; a base here at another revision than the run trained on is refused. Locally this takes a run
  copied from another node; with U2's `--node` it sends a run's adapter over Weirdnet with no new code.
- **Left for U3:** `train run --add --rollout NAME` joining a rollout goes through U3's rollout step.
- **Nothing here needs Weirdnet 0.5.3 / 0.4.4.** At the switch-over, `train::import_spec` maps onto
  `weirdnet-model`'s `Provenance` the same way U1's record does.

## U2 as built (2026-11-24)

- **`ymirctl models upload FILE --node NAME`** (also `--from-run RUN_DIR`, `--name`, `--adapter-for`, `--license`,
  `--role`, `--no-persist`, an alias of `--no-register`). `ymirctl` checks and hashes the file as U1 does and builds
  its `Provenance` (`ymirctl::models::upload::remote_op`); FILE with `--from-run` defaults to the run's adapter in
  RUN_DIR. Its `ymird` sends it with `ModelsClient::upload` (48 KiB chunks over models/1, progress streamed back, no
  more than 100% shown). It always asks the node to serve it (`Upload::serving`): a model is added to that node's
  supervisor and loads on its first request; a llama.cpp adapter is added to its base and the message says when it
  is used (on the base's next start, and if the base is running the first request for the adapter restarts it). It
  never says "live".
- **Resume.** Ctrl-C or a dropped link leaves the parts staged on the node (`<models_dir>.uploads/`, swept after 7
  days); sending the same files again resumes and says how much was already there.
- **Already there.** An `exists` refusal for the same id whose stored revision is the files' revision counts as
  "already has it; nothing was sent", and the next step still runs. Other files under a taken id are refused.
- **Saved on the node (decision 5), `/ymir/uploads/1`.** `weirdnet-backends` has no hook when an upload completes,
  so Ymir adds its own small protocol (`ymir_core::upload_net`), served next to models/1 when `[model]
  remote_manage` is on, to any linked device (`roster_auth`). After the upload the uploader's `ymird` sends `Register
  { id, revision, adapter_for, persist, role }`. The node checks that the stored entry's provenance has that
  revision, calls `NodeService::set_model_info(id, Some(revision), quant)` (as the lead asked; an adapter's quant is
  its base's), and unless `--no-persist` saves it in the node.toml `ymird` was started with (`Daemon::start_at`):
  a model as `[[model.local]]` (role `main` when there is no main model, else `chat`; GPU layers when the node has a
  GPU; its `revision`), an adapter as `[[model.local.lora]]` under its base with `revision`, `base_revision` and
  `base_quant` from the base's entry. The edit keeps comments (`ymir_core::config_edit`, moved from `ymirctl`), is
  validated before it is written, keeps `node.toml.bak`, and is serialised per node. It is skipped, with the reason,
  when node.toml already has that id (the same revision says so), the base is not in node.toml, or the adapter was
  trained on another revision of the base. `--no-persist` leaves node.toml alone and says the entry is gone after a
  restart. A node with an older `ymird` stores the upload but cannot record it, and the message says so.
- **Free space (decision 4).** The node's `WritePolicy` keeps `[model] upload_keep_free` free, by default
  `KeepFree::default()` (the larger of 10% and 20 GiB); refusals read as `NODE refused the upload (REASON): ...` with
  what to do (`exists`, `no-space`, `too-big`, `quota`, `wrong-arch`, `no-base`, `not-adapter`, `wrong-engine`,
  `checksum`, `not-allowed`). A duplicate upload in progress answers `Busy`.
- **One node.** `--node all` and `auto` are refused for uploads (pools are U6); `--node` naming this device says to
  run it without `--node`.
- **Not built from the plan:** `--node auto`, `--serve` as a flag (it always serves) and `[model] upload = false`.
- **Tests:** `crates/ymir-testkit/tests/models_upload.rs` runs three real daemons on 127.0.0.1 (a laptop and two
  model nodes, one keeping more space free than it has): a model sent, placed with provenance (`uploaded_by` the
  laptop) and saved; the same files again; other files under the id; a cut-off upload resumed; an adapter with and
  without `--no-persist`; a wrong-architecture adapter; `no-space`.

## U3 as built (2026-11-25)

- **`ymirctl models upload FILE --node NODE --rollout NAME [--canary W [--eval ROWS.jsonl [--judge MODEL]] [--force]]`**
  (`ymirctl::models::upload_rollout`). The version number N is one more than the highest version of NAME on this
  machine (ymird's live rollouts, else node.toml), 1 for a new rollout. Without `--name` the upload's id (and its
  provenance id) becomes `<id>-r<N>`, and the output says `As ID, version NAME@N`.
- **After the node stores and serves it**, ymirctl runs P4b's `models rollout add NAME@N --model ID --weight 0` on
  this machine: node.toml's `[[inference.rollout]]` and, when ymird runs, `ControlRequest::Rollouts` at once. An
  adapter the node will load on its base's next start (`AdapterChange::NextLoad`) gets the note that NAME@N waits at
  weight 0 until it answers.
- **`--canary W`** then runs `models rollout weight NAME@N W` through P4b's gate: up to the 10 percent canary share
  freely; above it only with an eval that passes or `--force`. With `--eval ROWS`, ymirctl first runs `ymirctl eval`
  (seed 0) on the new version's model and on the model of the version with the most traffic, writes the scored rows
  to a temporary file, passes it as the gate's `--eval` and removes it after. When no other version has traffic
  there is nothing to protect: the eval is skipped and the weight is set. A refusal leaves the version added at
  weight 0 and says so (`NAME@N stays at weight 0`).
- **`models rollout list`** prints each version's provenance under it (`origin`, `license` or `not given`, `base`,
  dataset hash), best effort from `models list --node all` on the pool's model nodes (10 s), keyed by served id.
- **`--replace`** was not built (U2 refuses other files under a taken id with `exists`), so the model a rollout
  version names cannot be swapped under it: a new version always gets a new id.
- **Not built:** `--inference NAME|all` (rollouts stay per inference node; `--rollout` applies on this machine's
  inference role only, as agreed on 2026-11-25). `--rollout` without `--node` is refused.
- **Tests:** `upload_rollout` unit tests (next version, incumbent, ids), `rollout` (provenance line and list), and in
  `crates/ymir-testkit/tests/models_upload.rs` three uploads to `support` on a running inference node: the first
  takes `--canary 100` with no eval, the second's `--canary 50` is refused and stays at weight 0, the third's
  `--canary 5` passes; `rollout list` shows origin and license. The `--eval` path is covered by P4b's gate tests;
  no testkit model can answer an eval.

## U4 as built (2026-11-25, Bragi 0.7.0)

- **Where:** Bragi's desktop (GPUI) Models screen, `Upload…` (decision 6); the logic is in `bragi-core`
  (`models/upload.rs`: `inspect`, `UploadSpec`, `ModelsManager::upload`; `Core::models_inspect_upload`,
  `models_upload`, `models_pending_uploads`, `models_forget_upload`). Bragi uses only published Weirdnet crates
  (`weirdnet-backends` 0.4.4, `weirdnet-model` 0.5.3, `weirdnet-agent` 0.5.2).
- **Dialog:** a file picker; Bragi inspects and hashes the file here (`weirdnet_model::artifact::inspect`) and
  shows kind, format, architecture, size and, for an adapter, the base it was made from, before anything is sent.
  Then one target node (the ready model nodes), for an adapter a base picker of the models that node serves, the
  name (suggested from the file), the license (optional, decision 2) and "keep it after the node restarts"
  (persist into the node's node.toml, on by default, decision 5).
- **Sending:** over models/1 (`ModelsClient::upload`, as `ymirctl models upload --node`), one progress bar that never
  goes back, and Cancel, which keeps the parts on the node. Bragi keeps unfinished uploads per pool in
  `uploads.json`; the screen lists them with Resume and Forget, also after Bragi restarts. `exists` with the same
  revision counts as "already there". The node's refusals are shown with what to do: `no-space` (after
  `[model] upload_keep_free`, decision 4), `exists`, `wrong-arch`/`no-base` for an adapter, an older Ymir without
  `/ymir/uploads/1`. An adapter the node loads at its base's next start (`AdapterChange::NextLoad`, llama.cpp) gets
  the restart note.
- **Recorded with the node** over `/ymir/uploads/1` like ymirctl does. Bragi can't depend on `ymir-core` (not
  published), so it carries a small copy of the wire types marked temporary (`models/ymir_uploads.rs`); it goes when
  Weirdnet's "upload placed" hook on `ModelsService` lands.
- **Rollouts:** not set from Bragi (decided 2026-11-25): the result shows the `ymirctl models upload ... --rollout
  NAME` and `models rollout add NAME@N --model ID --weight 0` commands for the inference node. A Ymir
  `/ymir/rollouts/1` for remote rollout changes is the follow-up.
- **Not built yet:** the Flutter dialog (the bridge passes the `uploaded` event on); the terminal app's
  `/models upload PATH`; "auto" and pool targets (U6); each node's fit and free disk in the target picker; filtering
  bases by architecture and engine; a dataset manifest; speed and direct-or-relay path; refusing a pool switch
  mid-upload.
- **Tests:** `bragi-core` `models/upload_tests.rs` (five, over the in-memory transport: a model uploaded and saved,
  uploaded twice, `exists`; refusals for no-space, an older Ymir, an adapter's missing base, a node not on the
  roster; pending uploads). The desktop scenario (`scripts/desktop-scenario.sh`, Xvfb, the demo profile) covers the
  preview, progress, Cancel, the unfinished list, Resume, a no-space refusal from a demo node that keeps more free
  than its disk has, and an adapter. The demo's engine takes adapters live, so the restart note's shot is a seeded
  view state; not tested against a real Ymir node.

## Rollouts over Weirdnet, `/ymir/rollouts/1` (planned 2026-11-25, follow-up to U4)

So that Bragi (or `ymirctl` on a laptop) can put an upload into a rollout on an inference node it is not
running on. Rollouts stay per inference node, as in U3.

- **Protocol:** `/ymir/rollouts/1`, JSON request and reply on one stream, served by `ymird` when it has the
  inference role, next to `/ymir/uploads/1`. Requests (tag `op`): `list`; `add {name, version?, model,
  weight}` (no `version`: one more than the highest, as U3 picks it); `weight {name, version, weight}`;
  `rollback {name}`; `retire {name, version, force}`. Each change may carry `force` and `eval` (the scored
  rows of a `ymirctl eval --out` file, inline). Replies: `rollouts {rollouts, said}` (the rollouts in
  force after the change and what changed, as `ymirctl models rollout` says it) or `failed {message}`.
- **The same rules as `ymirctl models rollout`:** the edit logic moves from `ymirctl` into `ymir-core`
  (`apply`, the canary gate at 10 percent, `eval_gate`, `write_rollouts`, and the eval's `Scored` row), so both
  paths share it. A change is validated, written to node.toml (`[[inference.rollout]]`, other content
  kept, one edit at a time) and set live through the coordinator, so new conversations follow at once.
- **Who:** the owner's linked devices, the same callers as `/ymir/uploads/1` (roster auth).
- **Clients:** `ymir_core::rollouts_net::RolloutsClient`; `ymirctl models rollout ... --inference NODE` runs a
  change on another inference node. Bragi gets the same small temporary copy of the wire types as for
  uploads, and a rollout section in the upload dialog (rollout name, weight 0 by default, a canary share
  up to 10 percent).
- **Tests:** unit tests for the moved logic (unchanged behaviour), and a testkit test with two daemons:
  an `add` over the network lands in the inference node's node.toml and in its live rollouts; a weight
  above the canary share without an eval is refused; an unlinked caller is refused.

**Decisions (2026-11-25 2:51 AM ET, made by llm dev from Michael's upload answers; Michael can override them):**

1. Any linked device may change rollouts on an inference node, as for uploads; `[inference]
   rollout_callers` (peer ids, like `[train] callers`) narrows the list.
2. A remote change above the 10 percent canary share carries the scored eval rows or `force`. Bragi
   offers 0 to 10 percent only, for now.
3. Every operation works remotely, rollback and retire included: rollback is the safety valve.
4. A remote change is written to the node's node.toml by default, as uploads are; a request with
   `persist: false` changes only the live rollouts (until ymird restarts).

### `/ymir/rollouts/1` as built (2026-11-25)

- `ymir_core::rollout_edit`: `Change` (`add`, `weight`, `rollback`, `retire`), `Gate` (scored rows, their source,
  `force`), `apply`, `eval_gate`, `write_rollouts`, `next_version`, and the eval's `Scored` row (re-exported by
  `ymirctl::eval`). `ymirctl models rollout` converts its arguments to a `Change` and reads `--eval FILE` into rows.
- `ymir_core::rollouts_net`: `Request {cert, change, gate, persist}` (no change: a read), `Reply::{Rollouts {rollouts,
  said, saved, note}, Failed}`, `RolloutsService` (changes the coordinator's rollouts in force, one edit at a time,
  then node.toml when `persist`; a node.toml that can't be written leaves the change live and says so),
  `RolloutsServer`, `RolloutsClient`. `ymird` serves it with the inference role when `[inference] remote_rollouts`
  (default on); callers are the roster narrowed by `[inference] rollout_callers` (validated as peer ids).
- `ymirctl models rollout ... --inference NODE [--no-persist]` goes through this machine's `ymird`
  (`ControlRequest::RemoteRollouts`, the `ModelsHub`'s `RolloutsClient`). `--inference` naming this device is
  refused. `models upload --rollout` still applies on this machine only.
- Tests: `rollout_edit` and `rollouts_net` unit tests; the moved logic's existing `ymirctl` tests unchanged;
  `crates/ymir-testkit/tests/rollouts_net.rs` (three daemons: add with all the traffic, saved and live; the canary
  gate refusing 50 and taking 5; `--no-persist` live only; list; rollback; retire; a linked device outside
  `rollout_callers` refused; `--inference` at this device refused).

## Decisions (m3talsmith, 2026-11-24 2:22 PM ET)

1. **Who may upload:** every linked device. No `ymir.models.write` grant.
2. **License:** not required. Recorded when given; no `--allow-unknown-license` flag.
3. **Pools:** upload once to one node; the other nodes copy from it (U6).
4. **Limits:** no size cap and no per-node total; always keep 10 percent of the disk or 20 GiB free, whichever is
   more.
5. **Persistence:** uploads are saved into the target node's `node.toml` by default so they survive a restart;
   `--no-persist` (remote) or `--no-register` (local) opts out.
6. **Bragi:** desktop first.
7. **Stopgap:** a private Hugging Face repo plus `ymirctl models pull` is fine until U2 (see "Stopgap until U2").
