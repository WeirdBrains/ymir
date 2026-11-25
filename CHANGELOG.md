# Changelog

All notable changes to Ymir are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- A larger synthetic training fixture, `crates/ymir-testkit/tests/fixtures/decentralized-networks-large.dataset.jsonl` with its `ymir-dataset/1` manifest (16 conversations, 1297 tokens with Qwen2.5's chat template), long enough for a real `llama-finetune` step at context 256; the small fixture renders to 256 tokens.
- `scripts/engine.py build --trainer` also builds `llama-tokenize`, which the trainer uses to count the training text.

### Changed

- `scripts/trainer.py extract-lora` is fast: a randomized, truncated SVD, on the GPU through PyTorch when the trainer's venv has a CUDA build of it, else on the CPU several matrices at a time with one BLAS thread each. A rank 16 adapter for Qwen2.5-0.5B took 79 minutes on a busy machine before, and now 8 s on the CPU or 4 s on the GPU. `YMIR_TRAINER_SVD` (`auto`, `cuda`, `cpu`, `exact`) and `YMIR_TRAINER_THREADS` choose where and how wide.

### Fixed

- `scripts/trainer.py train` refuses a dataset too short to train on, with its token count and how many it needs, instead of letting `llama-finetune` crash (`GGML_ASSERT ... ggml_opt_dataset_init`). Upstream needs at least 2 x context + 1 tokens, with the context rounded up to a multiple of 256 as llama.cpp does, so at least 513. Without `llama-tokenize` next to `llama-finetune` (an older trainer build), the crash is turned into the same refusal.

## [0.15.0] - 2026-11-25

### Added

- More upload formats (model-upload plan, U7): `ymirctl models upload DIR --node NAME` sends a directory of safetensors weights to a vLLM node, or MLX weights to an MLX node, and a PEFT LoRA adapter directory with `--adapter-for` a vLLM model; `ymirctl models upload mmproj.gguf --node NAME --projector-for MODEL` sends a vision projector for a llama.cpp model. The directory is checked and hashed here (pickles, code and symlinks are refused) and again on the node, which places it, serves it and saves it in its `node.toml`: a `backend = "vllm"` or `"mlx"` entry, a LoRA on the base, or the model's `mmproj`. On the machine itself an upload is still a GGUF model or GGUF LoRA.
- Canary percentages (decision 8 in the model-upload plan): `ymirctl models rollout add NAME@N --model ID --canary PCT`, `ymirctl models rollout weight NAME@N --canary PCT` (also with `--inference NODE`, as `/ymir/rollouts/1`'s new `share` change) and `ymirctl models upload ... --rollout NAME --canary PCT` give a version exactly PCT percent of new conversations, any of 0 to 100 (such as 20 or 50); the other versions share the rest in their present proportions. A percentage named this way is not capped at the 10% canary share and needs no eval; an `--eval` given with it must pass.
- `[model] upload_engine` (`llamacpp`, `vllm` or `mlx`): the engine an uploaded full model must suit on that node. Unset, it is llama.cpp, unless `llamacpp = "off"`; then vLLM, then MLX. Before, every node took GGUF only.
- More upload formats and canary percentages

### Changed

- `ymirctl models upload ... --rollout NAME --canary` takes a percentage now (it was a raw weight, gated above 10%), and `--force` there only skips a failing `--eval`. Raising a version with a raw weight, `ymirctl models rollout weight NAME@N W`, keeps the canary share and the eval gate.

## [0.14.0] - 2026-11-25

### Added

- Upload to a pool (model-upload plan, U6): `ymirctl models upload FILE --node all` (every model node but this one) or `--node A,B` sends the file once, to the node with the most free space (or `--stage NODE`), and the other nodes copy it from that node, node to node (Weirdnet's `ModelsClient::replicate`), so this machine's uplink carries it once. Each node's free space is read first: a node without room after what it keeps free (`[model] upload_keep_free`, by default the larger of 10% and 20 GiB) is left out before anything is sent. Each copy is recorded and saved in that node's `node.toml` unless `--no-persist`. A node that can't copy from the other (it can't reach it, or runs an older Ymir) gets the file from this machine instead, and the output says so.
- Every model node's models/1 service can copy uploads from the owner's other nodes (a peer dialer with the node's certificate), and answers their copies of its own uploads.
- Upload to a pool: send once, the other nodes copy it node to node

## [0.13.0] - 2026-11-25

### Added

- Rollouts over Weirdnet (`/ymir/rollouts/1`, the follow-up to model-upload U4): `ymirctl models rollout list|add|weight|rollback|retire ... --inference NODE` reads and changes another inference node's rollouts by that node's rules (above the 10 percent canary share the request carries the eval rows from `--eval FILE`, or `--force`). A change applies at once and is saved in that node's node.toml; `--no-persist` changes only its live rollouts. Any of the owner's linked devices may, narrowed by `[inference] rollout_callers` (peer ids); `[inference] remote_rollouts = false` turns it off.
- Rollouts over Weirdnet: /ymir/rollouts/1 and ymirctl models rollout --inference

### Changed

- The rollout rules (`apply`, the canary gate, the eval gate, the node.toml rewrite) and the eval's scored row moved from `ymirctl` into `ymir_core::rollout_edit`, shared by both paths; behaviour unchanged.

### Fixed

- The Release workflow no longer puts the WiX debug database (`YmirSetup-VERSION.wixpdb`) among the release files or in `SHA256SUMS` (the bundle is built with `-pdbtype none`, as the MSIs were).

## [0.12.0] - 2026-11-25

### Added

- `ymirctl models upload FILE --node NAME --rollout ROLLOUT` (model-upload slice U3): the upload becomes the next version of a rollout on this machine, named `ID-rN`, at weight 0 once the node serves it. `--canary W` gives it traffic through the rollout gate (up to 10% freely; more with `--eval ROWS [--judge MODEL]`, which scores it against the version with the most traffic, or `--force`).
- `ymirctl models rollout list` shows each version's origin, license, base and dataset hash.
- `ymirctl status` rows carry the weights' revision each node reports.
- Model uploads join rollouts: ymirctl models upload FILE --node X --rollout NAME adds the next version at weight 0, --canary W goes through the eval gate, and rollout list shows provenance.

## [0.11.0] - 2026-11-24

### Added

- `ymirctl models upload FILE --node NAME` (model-upload slice U2): send a GGUF model or LoRA adapter (or a training run's adapter, `--from-run`) from this device to another of your model nodes over Weirdnet. It goes in 48 KiB chunks checked on the node, resumes after Ctrl-C or a dropped link, keeps `[model] upload_keep_free` free on the node (default: the larger of 10% and 20 GiB), is served by the node's `ymird`, gets its revision recorded for the pool, and is saved in that node's node.toml unless `--no-persist` (Ymir's `/ymir/uploads/1`). Any linked device may upload. Refusals say why (`exists`, `no-space`, `wrong-arch`, ...). A llama.cpp adapter is used from its base's next start, and the message says so.
- Model uploads to other nodes: ymirctl models upload FILE --node X (resumable, saved in the node's node.toml), on weirdnet-model 0.5.3 and weirdnet-backends 0.4.4.

### Changed

- `weirdnet-model` 0.5.3 and `weirdnet-backends` 0.4.4. `ymirctl models upload` and `train run` now use `weirdnet-model`'s artifact inspection, `Provenance` (`weirdnet-provenance/1`) and `ModelStore::import`; an uploaded model goes to `models_dir/ID/` (was `models_dir/uploads/ID/`, where earlier uploads stay). Behaviour is otherwise the same.

## [0.10.0] - 2026-11-24

### Added

- A trainable dataset fixture, `crates/ymir-testkit/tests/fixtures/decentralized-networks.dataset.jsonl` with its `ymir-dataset/1` manifest (synthetic, CC0-1.0), which `ymirctl train run --dataset` accepts; a test checks the manifest's hashes and counts against the fixtures.
- `ymirctl train run --node NAME`: start a training run on another of your model nodes from any linked device, app-only ones included, over Weirdnet (`/ymir/train/1`). The node takes runs only with `[train] remote = true` (off by default), from your linked devices (`[train] callers` narrows them); the dataset and its manifest go with the run (up to 24 MiB), the node reserves itself for the run as a local run does, progress streams back, and Ctrl-C cancels it there. `--add` is refused with `--node` (node.toml is never edited over the network), and `--from` names a GGUF in the node's models directory.
- Train run --node over Weirdnet; trainable dataset fixture

## [0.9.0] - 2026-11-24

### Added

- `ymirctl models upload FILE [--name ID] [--adapter-for BASE] [--license L] [--no-register]`: put a GGUF model or GGUF LoRA adapter trained elsewhere into this node's models directory. The header is checked (an adapter against its base's architecture), the file is hashed while it is copied in atomically, a `provenance.json` beside it records the files, hashes, revision, license (optional) and base, and the entry is added to node.toml with that revision (`--no-register` only prints it). At least 10 percent of the disk or 20 GiB stays free. Local only; uploading to other nodes comes next ([plan](docs/design/model-upload.md)).
- `ymirctl models upload [FILE] --from-run RUN_DIR`: upload a `ymirctl train run` adapter with its run record as the provenance (origin `train`: base, dataset record, trainer, run). The file must match the record's SHA-256 and the base here the revision it was trained on.
- Training: llama-finetune parses chat-template special tokens; model upload (U1, U5); M9 bundle fetcher; Windows bench sandbox

### Changed

- `ymirctl train run` puts its adapter in `models_dir/adapters/ID/ID.gguf` (was `adapters/ID.gguf`) with a `provenance.json` and `adapter.json` beside it, through the same import as uploads, and `--add` registers it with its content revision (`sha256:...`, the same on every node) instead of the run id. Adapters from earlier runs stay where they are.
- The llama.cpp trainer for `ymirctl train`, `scripts/trainer.py`, pinned to the engine's llama.cpp build (b12628). `train` runs upstream `llama-finetune` on an F32 GGUF base with conversations rendered in the base's chat template, and `extract-lora` keeps the top `rank` singular directions of each weight's change as a GGUF LoRA adapter. `scripts/engine.py build --trainer` builds `llama-finetune`, which upstream's release files don't include, and `scripts/trainer-requirements.txt` lists its Python packages. Tested against a fake `llama-finetune` only.
- M9 bundle fetcher: `ymirctl update fetch SOURCE [--announcement FILE]` downloads a signed bundle directory over HTTPS (or from a directory). Its manifest is checked against the pinned root before anything else is downloaded, each file is read up to its signed size, and an announced manifest must match. `ymirctl update watch --relay ADDR [--fetch] [--once]` hears Ymir's update announcements from a Weirdnet relay that carries them (`/weirdnet/updates/1`) and can stage what it hears. Moves to `weirdnet-update` 0.2.2 with its `wire` feature.
- The llama.cpp trainer's `llama-finetune` now tokenizes the training text with `parse_special = true`, so the chat template's special tokens (`<|im_start|>` and the like) train as the tokens the model emits instead of as text. Upstream b12628 has no such option, so `scripts/engine.py build --trainer` applies `packaging/engines/patches/llama-finetune-parse-special.patch` to `llama-finetune`'s source only (`llama-server` and `rpc-server` stay upstream's), and `scripts/trainer.py` refuses a `llama-finetune` without it. `trainer.py version` also lists the patches. The patch goes when upstream releases a fix (tracked: draft ggml-org/llama.cpp#31258).
- `ymirctl bench` scores code answers on Windows too: Python runs in a Job Object (5 s of CPU per process, 2 GB of memory, at most 8 processes, all killed afterwards) in an empty directory with an empty environment. The network is not cut off there, unlike `unshare` on Linux. On macOS code questions stay unscored, now said plainly and checked in CI: macOS refuses both memory limits (`ulimit -v` and `-d`), so the sandbox never ran code there.

### Fixed

- `scripts/engine.py build` passes the pinned build number to llama.cpp, whose shallow clone otherwise reported `build 1`, and `scripts/trainer.py` reads llama.cpp 1.0's version line (`version: 1.0.0-dev (build 12628, ...)`). Before this, a `llama-finetune` from `build --trainer` was always refused as the wrong build.
- `ymirctl bench`'s code sandbox now applies its limits where `sh` is dash (Debian, Ubuntu). It used to set every limit in one `ulimit` call, which dash refuses, with the error hidden, so code ran with no CPU, memory, file-size or process limit, only the 10 s wall clock. Each limit is now set on its own, with each shell's spelling. If one can't be set, code questions are left unscored.

## [0.8.0] - 2026-11-24

### Added

- M10, cert-only server links: a server can join with a node certificate and a user-signed service grant, never the identity seed. `ymirctl pool add CODE --service inference|model` sends one (an inference node gets the pool's and conversations' keys, a model node none), `pool link --cert-only` accepts only that, and `pool cert-only` moves a node linked with the seed (same peer id; the `seed` file is overwritten and deleted). A cert-only node keeps `credential.json` (the user's public key) instead of `seed` and cannot link, approve or revoke devices; revoking it from a seed holder leaves the identity unchanged. `pool devices` shows cert-only nodes and their roles. Moves to `weirdnet-sync` 0.7.1 and `weirdnet-device-link` 0.4.2.
- Training plan P3a, fine-tuning on one node: `ymirctl train run --base ID --dataset FILE` trains a LoRA adapter with the trainer in `[train]`. Michael's choice is upstream llama.cpp's full fine-tuning plus a LoRA extraction step, pinned to one build, and a trainer reporting another build is refused. Only datasets with a manifest are taken (a generated dataset's teacher license must be known). The node's GPU is held for the run: a running ymird unloads its models and loads none until the run ends, refused when the node is the only one serving a loaded model unless `--force`, and a ymird started meanwhile loads nothing. The adapter goes to `adapters/` in the models directory with a `run.json` record, and `--add` puts it in node.toml. `train reserve`/`release` hold and free a node by hand, `train check` tests the trainer, `train list` shows runs. `ymirctl nodes` shows reserved nodes. Moves to `weirdnet-backends` 0.4.3 (node reservations).
- Training plan P4b, versioned rollouts: `[[inference.rollout]]` gives a model name versions with traffic weights. Each conversation stays on one version, `NAME@N` pins one, and the catalog lists both. `ymirctl models rollout list|add|weight|rollback|retire` changes them in node.toml and in the running ymird at once. A version gets up to 10% of new conversations freely; more needs `--eval FILE` (a `ymirctl eval --out` file in which it does at least as well as the version with the most traffic) or `--force`. Answers are cached per version, and apps are told the name they asked for.
- `crates/ymir-testkit/tests/release_requires.rs`: CI fails when `release.toml` `[[requires]]` no longer matches Cargo.toml and Cargo.lock.
- Cert-only server links, ymirctl train with node GPU reservations (mock trainer only), weighted model rollouts with canary, eval gate and rollback, and a green Windows CI.

### Fixed

- `ymirctl bench` scores code answers only on Linux, macOS and BSD, where they run in the sandbox (`ulimit`, and `unshare` when it works). On Windows, code questions are left unscored instead of failing, and Windows CI passes again.
- The training lock is not checked for a dead process on Windows: a hold there lasts until `ymirctl train release`.

## [0.7.0] - 2026-11-24

### Added

- Training plan P2, consented conversation export: `ymirctl context train ID [--off]` gives or withdraws a conversation's training consent (its own `{conversation}/train` record, off unless set, synced); `context show` marks `[training]`. `ymirctl context export --training OUT.jsonl [--keep-system] [--redact REGEX]...` writes only consented, live conversations as chat-format JSONL, with system prompts, tool calls, attachments and reasoning stripped, cut at incomplete or stand-in answers, plus `OUT.manifest.json` (`ymir-conversations/1`); both owner-only, and only counts are printed. `ymirctl context withdrawn MANIFEST` lists what an earlier export should drop. Messages deleted in Bragi are skipped.
- Consented conversation export for training (skips messages deleted in Bragi), and engine launches that survive a taken port.

### Fixed

- A model's engine could lose the port ymird told it to another process before binding it, and then fail to start, or another server answered in its place. `weirdnet-backends` 0.4.2 checks that the server on the port is the launched engine (its key, its model) and tries again on a new port, without counting a crash. `release.toml` requires 0.4.2. The mock engine's `/v1/models` is behind its key, like llama-server and vLLM.
- Tests: a free-port race made `mcp_http::token_host_and_origin_are_checked` flaky under load (it picked a free port, let it go, and the server bound it later, by which time another test could have it). `ymirctl mcp --http` and the mock engine now bind port 0 in tests and the test reads the port they print; tests that must write a port into a config pick it outside the OS's ephemeral range (`ymir_testkit::config_port`). The mock engine and mock rpc-server print the port they actually bound.

## [0.6.0] - 2026-11-23

### Added

- `lora_slots = N` on a vLLM `[[model.local]]`: the base starts with LoRA on even with no adapters listed, so adapters can be served on it later without a restart. `ymirctl models adapter slots BASE N|--off` sets it; `adapter list` shows it.
- The base-revision check: `[[model.local.lora]]` takes `base_revision` and `base_quant`, the base's revision and quantisation the adapter was trained on. If the base has moved on, ymird doesn't serve the adapter (an error in the log names both revisions; the base still serves) and `adapter list` marks it NOT SERVED. `ymirctl models adapter add` records the base's current ones (`--base-revision`, `--base-quant` to override) and refuses an adapter that wouldn't serve.
- An opt-in real-vLLM test of runtime LoRA with `lora_slots` (`YMIR_TEST_VLLM_LORA`).
- LoRA slots for vLLM bases, ymirctl models adapter slots, and the adapter base-revision check.

## [0.5.0] - 2026-11-23

### Added

- Live LoRA adapter changes, without restarting the base where the engine allows: `ymirctl models adapter scale ID S|--reset --node NAME` (llama.cpp, from the next request), `adapter serve ID --base BASE --node NAME` (an adapter already on the node's disk) and `adapter unserve ID --node NAME` (its files stay). vLLM loads and unloads adapters at run time; llama-server only re-enables files it started with, so a new adapter file still serves after the base's next load. In memory only; `node.toml` is unchanged.
- `ymirctl chat --adapter-scale S`, and `"adapter_scale"` in a bridge request body: how strongly a LoRA adapter applies to one answer (llama.cpp).
- The catalog names each adapter's base (catalog/1 `base`; `weirdnet.base` in the bridge's `/v1/models`), so apps can label adapters.
- Live LoRA adapter changes and per-request scale, on the current Weirdnet crates.

### Changed

- `release.toml` requires `weirdnet-backends` 0.4.1 from crates.io (`[[requires]]`), so a release cannot go out against an unpublished or overridden engine crate.
- `weirdnet-agent` 0.5.2, `weirdnet-backends` 0.4.1, `weirdnet-openai` 0.2.2 and `weirdnet-alerts` 0.2.2 (with `backends`). Engine alerts come from `weirdnet-alerts`' monitor again; Ymir's interim copy (`ymir_core::engine_alerts`) is removed.

### Fixed

- The context plan's log line gives the whole estimate for a context set in `node.toml` too (KV and buffers, and the total with weights), at start-up and at load time.

## [0.4.0] - 2026-11-23

### Added

- LoRA adapters on a loaded base (training plan P4a): `[[model.local.lora]]` (`id`, `file`, `scale`, `revision`) serves each adapter as its own model id on its base's engine (llama.cpp `--lora` with the adapter and scale per request; vLLM `--enable-lora`). The catalog and `ymirctl status` list adapters with their base, requests are routed by the adapter's id like any model, and the context planner counts adapter weights in the base's memory. `ymirctl models adapter add | remove | list` edits `node.toml`; `ymirctl models pull REPO --node NAME --adapter-for BASE --serve` pulls an adapter onto a node and serves it without a restart. Adapters that don't match their base are dropped at start with an error, and the base still serves.
- LoRA adapters on a loaded base model, and context sizing that matches what is actually loaded.

### Fixed

- `ymirctl models context` agrees with the load-time planner: with ymird running it sizes each model as ymird would if it were asked for now, beside what runs and on measured free memory, with a loaded model's own memory counted back. On a 128 GB Spark it had shown Inkling at 321,536 tokens with 41.2 GiB free and warned that the 382,976 it ran with might not fit: it showed the start-up plan, which keeps room for every model in `node.toml` that could run beside it (there qwen3.8-27b-nvfp4 at 131,072 and the embedding model, about 33 GiB). Without ymird it still shows the start-up plan, and says so.
- A long prompt is no longer taken for a stalled node. The inference node keeps its asker alive (a `Meta` frame every 5 s) until the first token, within an allowance that grows with the prompt, and waits as long for a remote model node's first frame. `ymirctl chat` on a 128 GB Spark had failed with "stalled: no frame for 25000 ms" during Inkling's 62 s on a 17.9k-token prompt.
- The load-time plan's log line gives the whole estimate (KV and buffers, weights and the 10% for the engine's other buffers): 85.5 GiB for Inkling at 479,232 tokens on a 128 GB Spark, which measured 79.4 GiB loaded and 81.0 GiB after a 17.9k-token prompt.

### Changed

- `weirdnet-backends` 0.4 (LoRA adapters). Engine alerts are forwarded by ymird until `weirdnet-alerts` accepts 0.4.

## [0.3.1] - 2026-11-23

### Fixed

- The context plan counts both f32 score matrices llama.cpp keeps for attention with a score bias (Inkling 2 Mini, T5), plus the mask and relative positions, as llama.cpp's own reservation shows (`llama-fit-params`: 66.4 GiB of compute at 701,440 tokens). v0.3.0 counted one and planned Inkling at 701,440 tokens (70.9 GiB) beside a busy gemma4-e4b on a 128 GB Spark; llama-server mapped 108 GiB while setting up and the node ran out of memory. Inkling now gets 437,248 tokens alone on a Spark and 397,312 beside a busy gemma4-e4b. Plain `-fa off` still counts one matrix, now with its mask.
- Inkling 2 Mini's context plan counts both attention score matrices, so it no longer runs a node out of memory.

## [0.3.0] - 2026-11-23

### Fixed

- The context plan no longer under-counts models that llama.cpp can't flash-attend: Inkling 2 Mini (and T5, or any model under `-fa off`) keeps a context x micro-batch x heads f32 score buffer, 48 GiB for Inkling at 1M tokens, which v0.2.0 left out (it planned 1M on a 128 GB Spark from a 53.9 GiB estimate; llama-server used 104.6 GiB). Server slots (4 by default) now widen sliding-window caches, Gemma's shared-KV layers are not counted, and 20% of the free memory is kept free. Inkling 2 Mini now gets 721,920 tokens alone on a Spark.
- The updater installs bundled programs executable: `llama-server`, `rpc-server` and any other native binary or script in a package (ELF, Mach-O, `#!`, or executable in the bundle) get mode 755, not just `ymird` and `ymirctl`.

### Changed

- Contexts follow the models that run beside them: when a model is asked for, ymird sizes its context against the memory actually free (measured, and estimated from the running models), unloading idle models first (no request in flight, none for 30 s) when the requested model would otherwise be cut; a busy model is never unloaded. ymird logs what it unloaded and the context it chose, and the catalog reports that context. Models not loaded are re-sized as what runs changes.

### Added

- Inkling 2 Mini gets the largest context that really fits beside the node's other models.

## [0.2.0] - 2026-11-23

### Changed

- Models run with as much context as they can take: ymird gives each model its full trained context (GGUF header, or `config.json` with RoPE scaling for vLLM) and cuts it only where the KV cache would not fit beside the weights and the node's other models, saying why in the log. Sliding-window layers, looped layers and the KV cache type are counted. An explicit `context` is kept. `ymirctl models pull` no longer writes `context = 32768` (or `--max-model-len 262144` for Inkling on a Spark), and setup leaves llama.cpp contexts to ymird. The catalog advertises the context each server is started with.

### Added

- `ymirctl models context` shows each model's context and why; `--auto ID` or `--old-defaults` lifts contexts that older pulls wrote by default.
- Second public release: models get the largest context that fits.

## [0.1.0] - 2026-11-22

### Added

- First-class Inkling 2 Mini support: `ymirctl models pull thinkingmachines/Inkling-2-Mini` registers a vLLM Hub id (with Inkling reasoning/tool parsers, DSpark, the `vllm/vllm-openai:inkling-2-mini` image, and a 262144 context cap on DGX Spark / GB10); `pull ggml-org/Inkling-2-Mini-GGUF` also fetches `mmproj-Inkling-2-Mini-Q8_0.gguf` (any `mmproj-*.gguf` in the same repo is pulled this way). `ymirctl models search` lists safetensors/vLLM repos on vLLM-capable nodes (`--format vllm` or `--format all`). ymird refuses to launch Inkling on a vLLM older than 0.35.0 unless the node uses that image or a wrapped server. MLX is not supported until mlx-lm#2086 lands in a release; pull says so.
- `[model] vllm = "image:registry/name:tag"` and `[[model.local]] image` run vLLM through docker or podman (host networking, `--entrypoint vllm` so the supervisor's `vllm serve` still works).
- Bundled llama.cpp is pinned at b12628 (Inkling 2 Mini GGUF support, including the NVFP4 loop-scale fix from ggml-org/llama.cpp#32704).
- One `ymird` for the model, inference and app roles, with `ymirctl` setup, status, load, unload and chat.
- Loopback OpenAI bridge: `/v1/models`, `/v1/chat/completions`, `/v1/completions` and `/v1/embeddings`.
- llama.cpp and vLLM engines, swap groups, idle unload, and crash restarts with backoff.
- Wrapping a stock OpenAI-compatible server, such as `vllm serve`.
- Stand-in answers, flagged, while a large model loads.
- Health alerts by email, webhook and log, including the default route on Wi-Fi.
- User-level service install (systemd user unit, launchd agent).
- `ymird` runs on Windows, as a logon task (user level) or as the Windows service `NT SERVICE\ymird` (`--system`).
- `ymirctl service start` and `stop`, and `--system` for a systemd system unit or launchd daemon.
- Windows MSI and `YmirSetup` installers that register the Windows service.
- Linux tar.gz, zip, deb and pkg.tar.zst packages, a TAUR recipe and a Nix flake.
- llama.cpp bundled in every release file, used by `ymirctl setup` by default: `ymir` (CPU and Vulkan) and `ymir-cuda` (CUDA 13 with its runtime).
- rpm packages, a release workflow for Linux, macOS and Windows, and a smoke test that runs a real model from an unpacked archive.
- `ymirctl models search`, `pull`, `list` and `remove`: find a model on Hugging Face with a fit verdict for this node, download it (resumable, checksum-verified, split GGUFs, optional token), and print or register its `[[model.local]]` entry.
- `ymirctl batch --jsonl`: data generation through the pool. JSONL rows with their own messages, system prompt, sampling and seed; the teacher model waits to load (no stand-in); answers go to `--out` with `usable`, `cache_hit` and `standin` per row and a manifest with the teacher's license (`--license` required); `--resume` continues an interrupted run; `--sources` records a sources manifest (topic, URLs and licenses) and the source of every row.
- Requests can skip or refresh the answer cache (infer/2 `cache`, `weirdnet-agent` 0.5.1): `batch --jsonl` bypasses it by default (`--cache refresh|use`) and `eval` always does; the bridge takes `"cache": false|"refresh"` or `x-weirdnet-cache: off|refresh`.
- `[[model.local]] revision`: the weights revision goes in the capability record and keys the answer cache, so new weights under the same model id never replay old answers; answers report it in their metadata.
- `ymirctl eval`: score models on a held-out JSONL file with the bench scorers (`expect`) and an optional `--judge` model for rubrics.
- `ymirctl mcp`: an MCP server for Cursor, Claude Desktop and VS Code over stdio or loopback streamable HTTP. Tools for status, nodes, models, chat with images and audio, completions, embeddings, model search and the service; load, unload and pull need `confirm: true`, and `ymir_remove_model` needs `--allow-remove` too. Built on `weirdnet-mcp`.
- `ymirctl models search | pull | list | remove --node <name|peer-id|auto|all>`: the same on the model nodes among your devices over Weirdnet (`/weirdnet-backends/models/1`, owner only). `auto` picks the node that fits best, a pull streams its progress, `pull --serve` serves the model from the node's running `ymird` without a restart (in memory; `node.toml` is not edited remotely), and `remove` stops serving and deletes. `[model] remote_manage = false` turns it off on a node.
- First public 0.1.0 release.
