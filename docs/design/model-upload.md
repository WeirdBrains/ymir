# Uploading models to Ymir: implementation plan

Status: plan, 2026-11-24. Approved by the project lead on 2026-11-24 with two changes (U5 moved up to follow U1;
U6 waits on Weirdnet requests B and C, with no interim peer copy in Ymir). U1 is in progress. m3talsmith's open
questions below are still open.

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
- **Provenance is required, as for datasets.** P1 refuses a dataset whose teacher license is unknown unless the
  operator says so; an upload is held to the same rule for its own license and its base model.
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

- **License** is required (`--license`). `unknown` only with `--allow-unknown-license`, and the record says so.
  When the base's license or the dataset's teacher license is more restrictive than the declared one, ymirctl
  warns before sending.
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
   against the per-upload limit, quota and free disk, and, for adapters, that the base is served on an engine that
   takes this format. It answers `UploadReady { upload, have: [bytes already held per file] }` or a refusal with a
   reason: `exists`, `too-big`, `no-space`, `quota`, `wrong-engine`, `no-base`, `bad-name`.
3. **Header first.** The client sends the first chunk of each GGUF or safetensors file before the rest; the node
   reads the header as soon as it has it and refuses early (`bad-format`, `wrong-arch`, `not-adapter`) rather than
   after gigabytes.
4. **Chunks.** `UploadChunk { file, offset, data }` with at most 48 KiB of data, under the 64 KiB frame limit; the
   stream's window gives back-pressure. The node appends to `uploads/<upload>/<path>.part`, where `upload` is the
   hash of the offer's file list, so the same files resume whoever sends them. It reports `Progress` every 250 ms as
   pulls do.
5. **Verify and place.** On `UploadDone` the node hashes each part, compares size and SHA-256, re-reads headers,
   runs the format checks above and the fit, then renames into the store (atomic per file, the directory last),
   writes `provenance.json`, and, with `serve`, adds it to the supervisor as `pull --serve` does (an adapter on its
   base with `Supervisor::add_adapter_live`). It answers `Uploaded(Pulled)` with the files, revision, whether it
   serves, and for an adapter when it serves (`AdapterChange`).
6. **Cancel and resume.** Closing the stream cancels; parts stay. Offering the same files again resumes from
   `have`. Abandoned parts are removed after a TTL (default 7 days) or by `models upload --abandon`.

A node never edits its own `node.toml` because of an upload (as with `pull --node`): the served model lives in
memory and the output prints the `[[model.local]]` / `adapter add` entry to keep it across a restart. `--register`
writes it when the target is the machine ymirctl runs on. (Open question 5 asks whether uploads should persist on
remote nodes.)

### A pool

`--pool` (every model node whose engine takes the format and that it fits) or `--node all` / a list. Pushing the
same 5 to 70 GB from a laptop once per node wastes its uplink, often through a relay. So:

- **Stage once, nodes pull from each other.** The client uploads to one node (the best-connected fitting node, or
  `--stage NODE`); then every other target pulls the files from that node over a models/1 `Fetch` read command,
  checked against the same hashes, node to node. Progress for every node streams back to the client.
- **Fallback: push to each.** When no node can stage (older nodes, `--no-stage`), the client pushes to each target
  in turn; resume keeps a broken one cheap.
- **Hub as staging** works today with no change: push to a private Hub repo, then `models pull --node all`. It stays
  documented as the interim path, but it puts unreleased weights on a third-party host and loses the provenance
  record, so it is not the main path.

## Auth and permissions

- **Who.** Today models/1 admits any linked, unrevoked device of the owner (`RosterAuth` without roles). Writing
  tens of gigabytes and new weights into the pool is more than search or pull, so `Upload` (and `Fetch` to a
  node) gets its own check: the caller must hold a user-signed grant `ymir.models.write` in the `weirdnet-sync`
  roster (`RosterAuth::require_roles`), which the owner grants to personal devices (`ymirctl pool grant DEVICE
  models.write`, Bragi's device list). Cert-only service nodes (M10) hold `ymir.inference` / `ymir.model` only, so a
  compromised server cannot push weights into the rest of the pool. `[model] upload = "granted" | "linked" | "off"`
  (default `granted`; `linked` is today's models/1 rule). The default is open question 1.
- **Node to node.** Staged replication needs the target node to read from the staging node: `Fetch` admits a caller
  with the `ymir.model` grant, and only for an upload the owner started (the staging node keeps the offer, signed
  into the target's request as the owner's upload id).
- **Identity.** The caller's `NodeCert` travels in the request as for every models/1 command; Noise authenticates
  the stream. The uploading device's roster label goes in the provenance record.
- **Disabling.** `[model] remote_manage = false` turns off all of models/1, uploads included; `upload = "off"` turns
  off uploads only.

## Validation, quotas and disk space

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
- **Disk**: the node reports free bytes on the models filesystem in `UploadReady`; it refuses unless free space
  after the upload stays above `[model] keep_free` (default 10 percent of the filesystem or 20 GiB, the larger)
  and rechecks before each rename.
- **Quotas**: `[model] max_upload = "100GiB"` per upload and `upload_quota = "500GiB"` for everything with
  `origin = upload` on that node; `models list` shows both. Defaults are open question 4.
- **Concurrency**: one upload per id at a time; a second offer for the same files joins the first (same `.part`).

## Catalog, versions and rollouts

- **Catalog.** Once served, an upload is a model like any other: the supervisor's status row feeds the capability
  record (with the new revision), catalog/1 lists it, adapters with their `base`. Bragi shows it at once.
- **Naming for versions.** Rollouts want a new id per version (`qwen3-8b+support-r3`) so the old one keeps serving.
  ymirctl suggests `<name>-r<N>` from the rollout's next version when `--rollout` is given and `--name` is not.
  `--replace` (same id, new revision) is allowed for a model outside any rollout; for a model that a rollout version
  names, it is refused.
- **Feeding a rollout.** `--rollout NAME` makes the upload the next version of NAME on the inference nodes:
  after the model node serves it, ymirctl asks the inference node (its local ymird, `ControlRequest::Rollouts`) to
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
    [--name ID] [--license SPDX | --allow-unknown-license]
    [--base MODEL] [--dataset-manifest FILE] [--from-run RUN_DIR]
    [--serve] [--register] [--store-only] [--replace]
    [--rollout NAME [--eval ROWS.jsonl [--judge MODEL]] [--canary W] [--inference NAME|all]]
    [--stage NODE | --no-stage] [--dry-run] [--abandon]
```

- `--dry-run` reads and hashes locally, prints kind, format, architecture, base, revision, size, the per-node fit
  and free disk, and the provenance it would send; nothing moves.
- Progress is one line per node (`staging 41% 2.0 GB/4.9 GB 38 MB/s via relay`), as pulls show today, then the
  served id, revision, and the `[[model.local]]` or `adapter add` line to keep it.
- Exit codes and refusal reasons are the protocol's (`no-space`, `wrong-arch`, ...), worded with the fix.
- `ymirctl models list [--node]` gains `ORIGIN` and `LICENSE` columns; `--provenance ID` prints the record.

## Bragi UX

- **Upload** button on the Models screen (Flutter and GPUI, same `bragi-core` state; the terminal app gets
  `/models upload PATH`). Desktop first; on phones the system picker works for small adapters but full models are
  not a phone job (open question 6).
- **Dialog**: file or folder picker; Bragi reads the files locally (same `weirdnet-model` inspection) and shows
  kind, format, architecture and size before anything is sent; target picker (auto, a node, the pool) with each
  node's fit and free disk; for adapters a base picker filtered to served bases of the matching architecture and
  engine; name; license (required, with the "unknown" override behind a confirm); optional dataset manifest; an "Add
  to rollout" section (rollout name, starting weight 0, optional canary with the eval file).
- **Progress**: one bar per node with speed and path (direct or relay), Cancel (keeps parts), Resume after a restart
  of Bragi (the upload id is the file list's hash, kept in Bragi's state), and the result row linking to the model in
  the catalog and to its rollout version.
- **Pools**: the dialog uploads to the active pool (`pools.rs`); switching pools mid-upload is refused until it is
  cancelled or done.
- **Permissions**: a device without the upload grant sees the button disabled with the reason ("This device can't
  add models to the pool. Grant it on a device that can."), as the Models screen does today for unlinked devices.
- Logic in `bragi-core` (`ModelsManager::upload`, over the `weirdnet-backends` client); the shells only draw it.

## Requests for the Weirdnet lead

A, B and C were sent to the Weirdnet lead on 2026-11-24; D is held. None blocks U1 or U5; U2 needs A and B; U6
needs B's peer pull (`Fetch`, `PullSource::Peer`) and C's grant.

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

**B. `weirdnet-backends` models/1: upload, fetch, disk and a write policy.**
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

**C. `weirdnet-sync` / `weirdnet-agent`: a model-write device grant.** Confirm that `DeviceGrant` roles are free-form
app strings that the owner can grant to personal (non-service) devices, so Ymir can define `ymir.models.write`; if
not, add that. And a `RosterAuth` helper that admits a caller holding any of a role list *or* being a full-identity
device, so `[model] upload = "linked"` and `"granted"` are one line each. Why: uploads are the first write that
should not be open to every linked device, and the 2026-11-20 rule says to authorize on grants, not cert labels.

**D. Later, optional: `weirdnet-agent` catalog/1 origin.** `CatalogModel::origin` (`hub | upload | train`) and
`license`, from the capability record, so Bragi and other apps can label pool models without a models/1 call.

## Risks

- **Disk and bandwidth.** Full models are 5 to 70+ GB; through a relay that is hours. Mitigations: resume,
  staging and node-to-node replication, early header refusal, free-disk and quota checks before sending, a warning
  when the path is relayed.
- **Bad or hostile files.** A malformed GGUF or safetensors header reaching a parser or the engine. Mitigations:
  header parsing with size caps in one tested crate, no pickles, no code, engines never run with remote code, and
  a write grant narrower than "any linked device".
- **Wrong base.** Adapters trained on another revision or quant serve badly. Mitigations: architecture, PEFT base and
  revision/quant checks already in P4a, recorded in provenance.
- **Licenses.** Fine-tunes inherit the base's terms and may carry a teacher's output restrictions. Mitigations:
  license required, base and dataset licenses recorded, warnings on mismatch; Ymir cannot judge a license, only
  record it.
- **Rollout mistakes.** Promoting an untested upload. Mitigations: weight 0 on arrival, the existing 10 percent eval
  gate, rollback.
- **Live nodes.** Uploading and loading onto a node that serves traffic competes for disk, I/O and memory.
  Mitigations: `nice`d hashing, loads planned by `ctxplan` like any model, `--store-only`.
- **Dependency on the Weirdnet lead** for A and B; slice 1 does not wait on them.
- **Consent drift.** A model trained on conversations whose consent is later withdrawn stays in the pool.
  Mitigation: the dataset manifest hash in provenance lets `context withdrawn` name affected uploads; acting on it
  stays the owner's call.

## Suggested order (smallest end-to-end slice first)

1. **U1: local upload, GGUF model and GGUF LoRA, with provenance.** `ymirctl models upload FILE [--name ID]
   [--adapter-for BASE] --license L [--allow-unknown-license] [--register]` on the machine itself (no daemon, no
   network): read the header, hash, import into `models_dir` / `adapters/ID/` atomically, write `provenance.json`, set
   `revision` and print or register the entry. Interim inspection and provenance code in `ymir-core`, marked "to move
   to weirdnet-model" until A ships. Covers the main use case end to end when the fine-tune is made on, or copied to,
   a node machine, and fixes the record format.
2. **U5: P3a outputs on the same path.** `--from-run RUN_DIR` reads `run.json`; `train run` writes the same
   `provenance.json` (origin `train`) beside its adapter; `train run --rollout` later uses U3's step. Moved up by the
   lead (2026-11-24): it needs only U1.
3. **U2: upload to one node over Weirdnet** (`--node NAME|auto`, `--serve`), chunked, resumable, sha256-checked,
   with the write grant, free disk and quota checks. Needs A and B. This is the main use case from a laptop.
4. **U3: into a rollout.** `--rollout NAME [--eval ROWS] [--canary W]`, provenance in `rollout list`.
5. **U4: Bragi upload dialog** (desktop first, then Flutter), on the Models screen, with progress, cancel, resume
   and "add to rollout".
6. **U6: pools.** `--pool` / `--node all` with staging and node-to-node `Fetch`; push-to-each fallback. Waits on
   Weirdnet B's peer pull and C's grant; Ymir does not stub a peer copy of its own (lead, 2026-11-24).
7. **U7: more formats.** safetensors for vLLM, MLX directories, PEFT adapters, mmproj projectors.
8. **U8 (optional): catalog origin** in Bragi's model list once D exists.

Tests for each slice use temp `YMIR_HOME`s, tiny synthetic GGUF and safetensors files written by the test, mock
engines and the in-memory or loopback Weirdnet transport (two daemons for U2, U3 and U6), including: a hash
mismatch, a truncated part resumed, a wrong-architecture adapter refused before the body, no space, over quota, a
device without the grant refused, a pickle refused, and a rollout version arriving at weight 0.

## Open questions for m3talsmith

All open as of 2026-11-24.

1. **Who may upload:** only devices holding a new `ymir.models.write` grant (proposed default), or any linked
   device as models/1 allows today? Service nodes are excluded either way.
2. **License:** required for every upload, with `unknown` only behind an explicit flag, as for datasets?
3. **Pools:** stage on one node and replicate node to node (proposed), or push from the client to each node?
4. **Quota defaults:** per-upload cap, per-node total for uploads, and the free space to keep (proposed: none, none,
   10 percent or 20 GiB)?
5. **Persistence on remote nodes:** keep today's rule (a remote node never edits its `node.toml`; served in memory,
   entry printed), or let an upload persist on the target (`--persist`) so a rollout version survives a restart?
6. **Phones:** Bragi upload on desktop only at first, or phones too (adapters only)?
7. **Interim path:** is a private Hub repo plus `models pull` acceptable as the documented stopgap until U2?
