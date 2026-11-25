//! End-to-end tests of the four commands against a mocked Hub and a temporary model directory.

use serde_json::json;
use sha2::{Digest, Sha256};
use weirdnet_model::hw::Hardware;
use ymir_core::NodeConfig;

use super::hub::mock::{MockHub, sample_header};
use super::*;

fn sha(b: &[u8]) -> String {
    Sha256::digest(b).iter().map(|x| format!("{x:02x}")).collect()
}

fn gguf_bytes(len: usize, seed: u8) -> Vec<u8> {
    let mut b = sample_header(32, 8);
    b.resize(len, seed);
    b
}

struct Fixture {
    hub: MockHub,
    dir: tempfile::TempDir,
}

fn hardware(ram_gib: u64) -> Hardware {
    Hardware { os: "linux".into(), arch: "x86_64".into(), cpu_threads: 8, ram_bytes: ram_gib << 30, gpus: Vec::new() }
}

impl Fixture {
    fn new() -> Self {
        let hub = MockHub::start();
        // A repo with a single Q8_0, a single Q4_K_M and a two-shard Q6_K.
        let files = [
            ("tiny-Q8_0.gguf", gguf_bytes(400_000, 1)),
            ("tiny-Q4_K_M.gguf", gguf_bytes(250_000, 2)),
            ("tiny-Q6_K-00001-of-00002.gguf", gguf_bytes(200_000, 3)),
            ("tiny-Q6_K-00002-of-00002.gguf", gguf_bytes(200_000, 4)),
        ];
        let siblings: Vec<_> = files.iter().map(|(n, b)| json!({"rfilename": n, "size": b.len(), "lfs": {"sha256": sha(b)}})).collect();
        let mut siblings = siblings;
        let mmproj = b"MMPROJ".to_vec();
        siblings.push(json!({"rfilename": "mmproj-tiny-f16.gguf", "size": mmproj.len(), "lfs": {"sha256": sha(&mmproj)}}));
        siblings.push(json!({"rfilename": "README.md", "size": 5}));
        hub.route("/api/models/acme/Tiny-GGUF?blobs=true", 200, json!({"gated": false, "tags": ["gguf"], "siblings": siblings}).to_string());
        for (n, b) in &files {
            hub.route(&format!("/acme/Tiny-GGUF/resolve/main/{n}"), 200, b.clone());
        }
        hub.route("/acme/Tiny-GGUF/resolve/main/mmproj-tiny-f16.gguf", 200, mmproj);
        Self { hub, dir: tempfile::tempdir().unwrap() }
    }

    fn ctx(&self, ram_gib: u64, cfg_text: Option<&str>) -> Ctx {
        let config = self.dir.path().join("node.toml");
        if let Some(t) = cfg_text {
            std::fs::write(&config, t).unwrap();
        }
        Ctx {
            store: ModelStore::new(self.dir.path().join("models")).with_hub(&self.hub.url),
            mlx: MlxHub::new(&self.hub.url, None),
            hw: hardware(ram_gib),
            state: self.dir.path().join("state"),
            cfg: NodeConfig::load(&config).ok(),
            config,
            tty: false,
            live: None,
        }
    }
}

fn pull_args(repo: &str) -> PullArgs {
    PullArgs {
        repo: repo.into(),
        quant: None,
        file: None,
        context: None,
        id: None,
        role: None,
        register: false,
        dry_run: false,
        force: false,
        serve: false,
        load: false,
        adapter_for: None,
        dir: DirArgs { dir: None },
        node: NodeArgs::default(),
    }
}

fn text(out: Vec<u8>) -> String {
    String::from_utf8(out).unwrap()
}

#[test]
fn pull_picks_the_best_quant_that_fits_and_prints_a_stanza_without_touching_node_toml() {
    let fx = Fixture::new();
    let ctx = fx.ctx(32, Some("[node]\nname = \"n\"\n"));
    let mut out = Vec::new();
    pull(&ctx, &mut out, &pull_args("acme/Tiny-GGUF")).unwrap();
    let s = text(out);
    assert!(s.contains("acme/Tiny-GGUF: Q8_0, 2 file(s)"), "{s}");
    assert!(s.contains("downloaded, checksum verified"), "{s}");
    assert!(s.contains("[[model.local]]\nid = \"tiny\"") && s.contains("role = \"main\"") && s.contains("tiny-Q8_0.gguf"), "{s}");
    assert!(s.contains("mmproj = ") && s.contains("mmproj-tiny-f16.gguf"), "{s}");
    assert!(s.contains("node.toml was not changed"));
    // No fixed context in the entry: ymird runs it at the trained 131072, which fits here.
    assert!(!s.contains("context = "), "{s}");
    assert!(s.contains("Context on this node: context 131072 (the trained length is 131072"), "{s}");
    assert_eq!(std::fs::read(ctx.dir().join("tiny-Q8_0.gguf")).unwrap(), gguf_bytes(400_000, 1));
    assert_eq!(std::fs::read(ctx.dir().join("mmproj-tiny-f16.gguf")).unwrap(), b"MMPROJ");
    assert_eq!(std::fs::read_to_string(&ctx.config).unwrap(), "[node]\nname = \"n\"\n");
    assert!(!ctx.dir().join("tiny-Q4_K_M.gguf").exists());
    // The next run is a no-op download.
    let mut out = Vec::new();
    pull(&ctx, &mut out, &pull_args("acme/Tiny-GGUF")).unwrap();
    assert!(text(out).contains("already here, verified"));
}

#[test]
fn pull_a_split_quant_downloads_every_shard_and_points_at_the_first() {
    let fx = Fixture::new();
    let ctx = fx.ctx(32, None);
    let mut out = Vec::new();
    pull(&ctx, &mut out, &PullArgs { quant: Some("q6_k".into()), ..pull_args("acme/Tiny-GGUF") }).unwrap();
    let s = text(out);
    assert!(s.contains("Q6_K, 3 file(s)"), "{s}"); // two shards + mmproj
    assert!(ctx.dir().join("tiny-Q6_K-00002-of-00002.gguf").exists());
    // The path is a TOML string: basic (`"`) on Unix, literal (`'`) when it holds Windows backslashes.
    assert!(s.contains("tiny-Q6_K-00001-of-00002.gguf\"") || s.contains("tiny-Q6_K-00001-of-00002.gguf'"), "{s}");
}

#[test]
fn a_smaller_node_gets_a_smaller_quant_and_a_tiny_one_is_told_nothing_fits() {
    let fx = Fixture::new();
    // 400 KB of weights + 4 GiB of KV at 32k + 0.5 GiB overhead: 8 GiB of RAM leaves 3.4 GiB usable, so a lower context fits.
    let ctx = fx.ctx(8, None);
    let mut out = Vec::new();
    let err = pull(&ctx, &mut out, &pull_args("acme/Tiny-GGUF")).unwrap_err().to_string();
    assert!(err.contains("no quant of acme/Tiny-GGUF fits") && err.contains("--force"), "{err}");
    assert!(!ctx.dir().exists() || std::fs::read_dir(ctx.dir()).unwrap().next().is_none());
    let mut out = Vec::new();
    pull(&ctx, &mut out, &PullArgs { context: Some(4096), ..pull_args("acme/Tiny-GGUF") }).unwrap();
    let s = text(out);
    // An asked-for context is the operator's and goes in the entry.
    assert!(s.contains("Q8_0") && s.contains("context = 4096") && s.contains("Context on this node: context 4096 (set in node.toml)"), "{s}");
    assert!(s.contains("KV cache and buffers; ") && s.contains(" in all with "), "a fixed context gives its total: {s}");
}

#[test]
fn dry_run_downloads_nothing() {
    let fx = Fixture::new();
    let ctx = fx.ctx(32, None);
    let mut out = Vec::new();
    pull(&ctx, &mut out, &PullArgs { dry_run: true, ..pull_args("acme/Tiny-GGUF") }).unwrap();
    assert!(text(out).contains("Dry run"));
    assert!(!ctx.dir().exists());
}

#[test]
fn unknown_quant_and_non_model_repos_say_what_is_available() {
    let fx = Fixture::new();
    let ctx = fx.ctx(32, None);
    let err = pull(&ctx, &mut Vec::new(), &PullArgs { quant: Some("Q2_K".into()), ..pull_args("acme/Tiny-GGUF") }).unwrap_err().to_string();
    assert!(err.contains("no `Q2_K` quant") && err.contains("Q4_K_M"), "{err}");
    fx.hub.route("/api/models/acme/Plain?blobs=true", 200, json!({"siblings": [{"rfilename": "model.bin", "size": 1}]}).to_string());
    let err = pull(&ctx, &mut Vec::new(), &pull_args("acme/Plain")).unwrap_err().to_string();
    assert!(err.contains("no GGUF files") && err.contains("safetensors"), "{err}");
}

#[test]
fn register_appends_to_node_toml_and_the_role_is_chat_when_a_main_model_exists() {
    let fx = Fixture::new();
    let ctx = fx.ctx(32, Some("[node]\nname = \"n\"\n\n[[model.local]]\nid = \"old\"\nrole = \"main\"\nfile = \"/x/old.gguf\"\n"));
    let mut out = Vec::new();
    pull(&ctx, &mut out, &PullArgs { register: true, id: Some("tiny-chat".into()), ..pull_args("acme/Tiny-GGUF") }).unwrap();
    let s = text(out);
    assert!(s.contains("role = \"chat\"") && s.contains("Added `tiny-chat`"), "{s}");
    let cfg = NodeConfig::load(&ctx.config).unwrap();
    assert_eq!(cfg.model.local.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(), ["old", "tiny-chat"]);
    assert!(ctx.config.with_extension("toml.bak").exists());
}

#[test]
fn list_shows_registered_and_unregistered_files_and_remove_asks_first() {
    let fx = Fixture::new();
    let ctx = fx.ctx(32, Some("[node]\nname = \"n\"\n"));
    pull(&ctx, &mut Vec::new(), &PullArgs { register: true, ..pull_args("acme/Tiny-GGUF") }).unwrap();
    pull(&ctx, &mut Vec::new(), &PullArgs { quant: Some("Q4_K_M".into()), ..pull_args("acme/Tiny-GGUF") }).unwrap();
    let ctx = Ctx { cfg: NodeConfig::load(&ctx.config).ok(), ..ctx };
    let mut out = Vec::new();
    list(&ctx, &mut out).unwrap();
    let s = text(out);
    assert!(s.contains("tiny-q8_0") && s.contains("registered as `tiny`"), "{s}");
    assert!(s.lines().any(|l| l.contains("tiny-q4_k_m") && l.contains("not registered")), "{s}");

    // Declining keeps the file; `yes` deletes it and warns about the stanza for a registered one.
    let mut out = Vec::new();
    remove(
        &ctx,
        &mut out,
        &mut &b"no\n"[..],
        &RemoveArgs { id: "tiny-q4_k_m".into(), yes: false, dir: DirArgs { dir: None }, node: NodeArgs::default() },
    )
    .unwrap();
    assert!(text(out).contains("Nothing deleted") && ctx.dir().join("tiny-Q4_K_M.gguf").exists());
    let mut out = Vec::new();
    remove(
        &ctx,
        &mut out,
        &mut &b"yes\n"[..],
        &RemoveArgs { id: "tiny-q4_k_m".into(), yes: false, dir: DirArgs { dir: None }, node: NodeArgs::default() },
    )
    .unwrap();
    assert!(!ctx.dir().join("tiny-Q4_K_M.gguf").exists());
    let mut out = Vec::new();
    remove(
        &ctx,
        &mut out,
        &mut std::io::empty(),
        &RemoveArgs { id: "tiny".into(), yes: true, dir: DirArgs { dir: None }, node: NodeArgs::default() },
    )
    .unwrap();
    let s = text(out);
    assert!(s.contains("node.toml still lists `tiny`") && s.contains("Deleted"), "{s}");
    assert!(!ctx.dir().join("tiny-Q8_0.gguf").exists());
    assert!(std::fs::read_to_string(&ctx.config).unwrap().contains("id = \"tiny\""), "node.toml is never edited by remove");
}

#[test]
fn search_shows_downloads_a_fit_verdict_and_the_other_quants() {
    let fx = Fixture::new();
    fx.hub.route("/api/models", 200, json!([{"id": "acme/Tiny-GGUF", "downloads": 12_345, "likes": 7, "gated": false, "gguf": {"total": 600_000_000u64}}, {"id": "acme/Gone-GGUF", "downloads": 3}]).to_string());
    let ctx = fx.ctx(32, None);
    let mut out = Vec::new();
    search(&ctx, &mut out, &SearchArgs { query: vec!["tiny".into()], format: None, limit: 5, context: None, node: NodeArgs::default() }).unwrap();
    let s = text(out);
    assert!(s.contains("acme/Tiny-GGUF  (GGUF, 12.3k downloads, 7 likes)"), "{s}");
    assert!(s.contains("best: Q8_0") && s.contains(": fits") && !s.contains("estimated"), "{s}");
    assert!(s.contains("other quants:") && s.contains("Q4_K_M") && s.contains("Q6_K"), "{s}");
    assert!(s.contains("acme/Gone-GGUF") && s.contains("could not read the files"), "{s}");
}

#[test]
fn a_gated_repo_names_the_token_and_the_token_is_sent_only_when_set() {
    let fx = Fixture::new();
    fx.hub.route("/api/models/acme/Locked?blobs=true", 403, "no");
    let ctx = fx.ctx(32, None);
    let err = pull(&ctx, &mut Vec::new(), &pull_args("acme/Locked")).unwrap_err().to_string();
    assert!(err.contains("HF_TOKEN"), "{err}");
    assert!(fx.hub.state.lock().unwrap().auth.iter().all(Option::is_none));
}

#[test]
fn mlx_repos_download_as_a_directory() {
    let fx = Fixture::new();
    let w = vec![9u8; 5000];
    fx.hub.route(
        "/api/models/mlx-community/Tiny-4bit?blobs=true",
        200,
        json!({"tags": ["mlx"], "siblings": [{"rfilename": "config.json", "size": 2}, {"rfilename": "model.safetensors", "size": 5000, "lfs": {"sha256": sha(&w)}}, {"rfilename": ".gitattributes", "size": 1}]}).to_string(),
    );
    fx.hub.route("/mlx-community/Tiny-4bit/resolve/main/config.json", 200, "{}");
    fx.hub.route("/mlx-community/Tiny-4bit/resolve/main/model.safetensors", 200, w);
    let ctx = fx.ctx(32, None);
    let mut out = Vec::new();
    pull(&ctx, &mut out, &pull_args("mlx-community/Tiny-4bit")).unwrap();
    let s = text(out);
    assert!(s.contains("backend = \"mlx\"") && s.contains("Apple Silicon only"), "{s}");
    assert!(ctx.dir().join("mlx/Tiny-4bit/model.safetensors").exists() && !ctx.dir().join("mlx/Tiny-4bit/.gitattributes").exists());
    let mut out = Vec::new();
    list(&ctx, &mut out).unwrap();
    assert!(text(out).contains("tiny-4bit") && true);
}

#[test]
fn ids_come_from_the_file_name_for_unix_and_windows_paths() {
    assert_eq!(plan::id_for_file("/srv/m/Tiny-Q8_0.gguf"), "tiny-q8_0");
    assert_eq!(plan::id_for_file(r"C:\models\Tiny-Q8_0.gguf"), "tiny-q8_0");
    assert_eq!(plan::id_for_file(r"C:\models\big-Q4_K_M-00001-of-00002.gguf"), "big-q4_k_m");
}

fn inkling_repo() -> serde_json::Value {
    json!({
        "tags": ["transformers", "safetensors"],
        "siblings": [
            {"rfilename": "config.json", "size": 2048},
            {"rfilename": "model.safetensors", "size": 20_000_000u64, "lfs": {"sha256": "ab"}}
        ]
    })
}

#[test]
fn pull_of_a_safetensors_repo_prints_a_vllm_stanza_and_downloads_nothing() {
    let fx = Fixture::new();
    fx.hub.route("/api/models/thinkingmachines/Inkling-2-Mini?blobs=true", 200, inkling_repo().to_string());
    let ctx = fx.ctx(32, Some("[node]\nname = \"n\"\n"));
    let mut out = Vec::new();
    pull(&ctx, &mut out, &PullArgs { register: true, ..pull_args("thinkingmachines/Inkling-2-Mini") }).unwrap();
    let s = text(out);
    assert!(s.contains("vLLM Hub id"), "{s}");
    assert!(s.contains("backend = \"vllm\""), "{s}");
    assert!(s.contains("file = \"thinkingmachines/Inkling-2-Mini\""), "{s}");
    assert!(s.contains("reasoning-parser") && s.contains("inkling"), "{s}");
    assert!(s.contains("tool-call-parser") && s.contains("enable-auto-tool-choice"), "{s}");
    assert!(s.contains("dspark"), "{s}");
    assert!(s.contains("image = \"vllm/vllm-openai:inkling-2-mini\""), "{s}");
    assert!(s.contains("Added `inkling-2-mini`"), "{s}");
    // Not Spark-class in this fixture: no 262144 cap.
    assert!(!s.contains("max-model-len"), "{s}");
    assert!(!ctx.dir().join("model.safetensors").exists());
    let cfg = NodeConfig::load(&ctx.config).unwrap();
    assert_eq!(cfg.model.local[0].backend.as_deref(), Some("vllm"));
    assert_eq!(cfg.model.local[0].image.as_deref(), Some(ymir_core::vllm::INKLING_IMAGE));
}

fn inkling_config() -> serde_json::Value {
    json!({
        "model_type": "inkling2_loop",
        "text_config": {
            "model_max_length": 1_048_576, "hidden_size": 2304, "num_hidden_layers": 24,
            "num_attention_heads": 24, "num_key_value_heads": 8, "head_dim": 128,
            "local_layer_ids": [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22],
            "sliding_window_size": 1024, "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3}
        }
    })
}

#[test]
fn pull_of_inkling_on_spark_gets_the_full_context_and_mlx_is_refused() {
    use weirdnet_model::hw::{Accel, Gpu};
    let fx = Fixture::new();
    fx.hub.route("/api/models/thinkingmachines/Inkling-2-Mini?blobs=true", 200, inkling_repo().to_string());
    let mut ctx = fx.ctx(128, None);
    ctx.hw = Hardware {
        os: "linux".into(),
        arch: "aarch64".into(),
        cpu_threads: 20,
        ram_bytes: 128 << 30,
        gpus: vec![Gpu {
            accel: Accel::Nvidia,
            name: "NVIDIA GB10".into(),
            memory_bytes: 128 << 30,
            unified: true,
            compute_capability: Some((12, 1)),
            driver: None,
        }],
    };
    fx.hub.route("/thinkingmachines/Inkling-2-Mini/resolve/main/config.json", 200, inkling_config().to_string());
    let mut out = Vec::new();
    pull(&ctx, &mut out, &pull_args("thinkingmachines/Inkling-2-Mini")).unwrap();
    let s = text(out);
    // No recipe cap any more: ymird sizes --max-model-len, and on 128 GiB that is the full 1M.
    assert!(!s.contains("max-model-len") && !s.contains("context = "), "{s}");
    assert!(s.contains("Context on this node: context 1048576"), "{s}");
    assert!(ymir_core::kvfit::vllm_meta_dir(ctx.dir(), "thinkingmachines/Inkling-2-Mini").join("config.json").is_file());

    fx.hub.route(
        "/api/models/mlx-community/Inkling-2-Mini-4bit?blobs=true",
        200,
        json!({"tags": ["mlx"], "siblings": [{"rfilename": "config.json", "size": 2}, {"rfilename": "model.safetensors", "size": 8}]}).to_string(),
    );
    let err = pull(&ctx, &mut Vec::new(), &pull_args("mlx-community/Inkling-2-Mini-4bit")).unwrap_err().to_string();
    assert!(err.contains("mlx-lm") && err.contains("2086"), "{err}");
}

#[test]
fn search_lists_vllm_safetensors_repos() {
    let fx = Fixture::new();
    fx.hub.route(
        "/api/models",
        200,
        json!([{"id": "thinkingmachines/Inkling-2-Mini", "downloads": 5000, "likes": 80, "gated": false, "safetensors": {"total": 20_000_000_000u64}}]).to_string(),
    );
    fx.hub.route("/api/models/thinkingmachines/Inkling-2-Mini?blobs=true", 200, inkling_repo().to_string());
    let ctx = fx.ctx(32, None);
    let mut out = Vec::new();
    search(
        &ctx,
        &mut out,
        &SearchArgs { query: vec!["inkling".into()], format: Some(Format::Vllm), limit: 5, context: None, node: NodeArgs::default() },
    )
    .unwrap();
    let s = text(out);
    assert!(s.contains("thinkingmachines/Inkling-2-Mini") && s.contains("vLLM"), "{s}");
    assert!(s.contains("Inkling 2 Mini") && s.contains("0.35.0"), "{s}");
}

#[test]
fn context_lists_each_model_and_lifts_old_pull_defaults() {
    let fx = Fixture::new();
    let ctx = fx.ctx(32, Some("[node]\nname = \"n\"\n"));
    pull(&ctx, &mut Vec::new(), &PullArgs { context: Some(32_768), register: true, ..pull_args("acme/Tiny-GGUF") }).unwrap();
    let ctx = fx.ctx(32, None);
    let mut out = Vec::new();
    ctxcmd::run(&ctx, &mut out, &ContextArgs { ids: vec![], auto: false, old_defaults: false, dir: DirArgs { dir: None } }).unwrap();
    let s = text(out);
    assert!(s.contains("context = 32768") && s.contains("--old-defaults"), "{s}");
    let mut out = Vec::new();
    ctxcmd::run(&ctx, &mut out, &ContextArgs { ids: vec![], auto: false, old_defaults: true, dir: DirArgs { dir: None } }).unwrap();
    let s = text(out);
    assert!(s.contains("Put `tiny` on the automatic context") && s.contains("auto") && s.contains("context 131072"), "{s}");
    let cfg = NodeConfig::load(&ctx.config).unwrap();
    assert_eq!(cfg.model.local[0].context, None);
    assert!(ctx.config.with_extension("toml.bak").exists());
    let err = ctxcmd::run(
        &fx.ctx(32, None),
        &mut Vec::new(),
        &ContextArgs { ids: vec!["nope".into()], auto: true, old_defaults: false, dir: DirArgs { dir: None } },
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("no [[model.local]] with id `nope`"), "{err}");
}

/// A status row for `models context`'s view of what runs.
fn running(id: &str, state: ymir_core::control::ModelState, context: Option<u32>) -> ymir_core::control::NodeModel {
    serde_json::from_value(json!({
        "node": "n", "id": id, "engine": "llamacpp", "state": state, "in_flight": 0, "swap_group": null,
        "purpose": "chat", "pid": 1, "context": context
    }))
    .unwrap()
}

#[test]
fn context_with_ymird_running_sizes_a_loaded_model_with_its_own_memory_counted_back() {
    use ymir_core::control::ModelState;
    let fx = Fixture::new();
    let models = fx.dir.path().join("models");
    std::fs::create_dir_all(&models).unwrap();
    // `tiny`: 128 KiB of f16 KV per token, so 8 GiB at its 65,536. `big`: 14 GiB of weights,
    // in node.toml but not loaded (a sparse file: only the size counts).
    std::fs::write(models.join("tiny.gguf"), gguf_bytes(400_000, 1)).unwrap();
    let big = std::fs::File::create(models.join("big.gguf")).unwrap();
    {
        use std::io::Write as _;
        (&big).write_all(&gguf_bytes(4096, 2)).unwrap();
    }
    big.set_len(14 << 30).unwrap();
    let toml = format!(
        "[node]\nname = \"n\"\n\n[model]\nmodels_dir = {:?}\n\n[[model.local]]\nid = \"tiny\"\nfile = \"tiny.gguf\"\ncontext = 65536\n\n[[model.local]]\nid = \"big\"\nfile = \"big.gguf\"\ncontext = 8192\n",
        models.display().to_string()
    );
    // 32 GiB of RAM: (32 - 4) x 85% = 23.8 GiB for models.
    let ctx = fx.ctx(32, Some(&toml));
    let args = ContextArgs { ids: vec![], auto: false, old_defaults: false, dir: DirArgs { dir: None } };

    // ymird not running: the start-up plan keeps room for `big` beside `tiny`, so 65,536 looks short.
    let mut out = Vec::new();
    ctxcmd::run(&ctx, &mut out, &args).unwrap();
    let s = text(out);
    assert!(s.contains("ymird is not running here"), "{s}");
    assert!(s.contains("context 65536 (set in node.toml): its 8.0 GiB f16 KV cache and score buffers may not fit"), "{s}");

    // ymird running with `tiny` loaded, 8 GiB measured free (tiny's own 8.8 GiB already taken):
    // what the load-time planner would give it, with its own memory counted back. It fits.
    let live = Live {
        models: vec![running("tiny", ModelState::Ready, Some(65_536)), running("big", ModelState::Unloaded, Some(8192))],
        measured: Some(8 << 30),
    };
    let ctx = Ctx { live: Some(live), ..ctx };
    let mut out = Vec::new();
    ctxcmd::run(&ctx, &mut out, &args).unwrap();
    let s = text(out);
    assert!(s.contains("8.0 GiB measured free") && s.contains("if it were asked for now"), "{s}");
    let tiny = s.lines().find(|l| l.trim_start().starts_with("tiny")).unwrap();
    assert!(tiny.contains("context 65536 (set in node.toml)") && tiny.contains("with nothing else running"), "{tiny}");
    assert!(tiny.contains("its own 8.8 GiB is counted back") && tiny.contains("runs at 65536 now"), "{tiny}");
    assert!(!tiny.contains("may not fit"), "{tiny}");
    // `big` does not fit beside `tiny` as it runs (8 GiB free), so the planner would unload the
    // idle `tiny` for it, as it would at load.
    let big = s.lines().find(|l| l.trim_start().starts_with("big")).unwrap();
    assert!(big.contains("it would unload idle tiny") && !big.contains("may not fit"), "{big}");
}

// ---- `models upload` (docs/design/model-upload.md, U1) ----

/// A LoRA adapter GGUF header for `arch`, plus some weight bytes.
fn lora_bytes(arch: &str) -> Vec<u8> {
    let mut o = b"GGUF".to_vec();
    o.extend(3u32.to_le_bytes());
    o.extend(0u64.to_le_bytes());
    o.extend(3u64.to_le_bytes());
    for (k, v) in [("general.architecture", arch), ("general.type", "adapter"), ("adapter.type", "lora")] {
        o.extend((k.len() as u64).to_le_bytes());
        o.extend(k.as_bytes());
        o.extend(8u32.to_le_bytes());
        o.extend((v.len() as u64).to_le_bytes());
        o.extend(v.as_bytes());
    }
    o.resize(o.len() + 2048, 5);
    o
}

/// Uploads without looking at the real disk's free space (the box or CI runner may be nearly full).
fn run_upload(ctx: &Ctx, out: &mut dyn Write, a: &upload::UploadArgs) -> Result<()> {
    upload::run_with(ctx, out, a, None)
}

fn upload_args(file: &std::path::Path) -> upload::UploadArgs {
    upload::UploadArgs {
        file: Some(file.to_path_buf()),
        from_run: None,
        name: None,
        adapter_for: None,
        license: Some("apache-2.0".into()),
        role: None,
        no_register: false,
        dir: DirArgs { dir: None },
        node: None,
    }
}

/// A node.toml with a llama.cpp base `base` (file `base.gguf` in the fixture, revision r1).
fn with_base(fx: &Fixture) -> (Ctx, std::path::PathBuf) {
    let base = fx.dir.path().join("base.gguf");
    std::fs::write(&base, gguf_bytes(10_000, 9)).unwrap();
    let cfg = format!(
        // A literal TOML string, so a Windows path's backslashes are not read as escapes.
        "[node]\nname = \"n\"\n\n[[model.local]]\nid = \"base\"\nrole = \"main\"\nfile = '{}'\nrevision = \"r1\"\nquant = \"Q4_K_M\"\n",
        base.display()
    );
    (fx.ctx(32, Some(&cfg)), base)
}

#[test]
fn upload_of_a_model_imports_it_with_provenance_and_prints_its_entry() {
    let fx = Fixture::new();
    let ctx = fx.ctx(32, Some("[node]\nname = \"n\"\n"));
    let src = fx.dir.path().join("my-tune-Q4_K_M.gguf");
    let bytes = gguf_bytes(50_000, 6);
    std::fs::write(&src, &bytes).unwrap();
    let mut out = Vec::new();
    run_upload(&ctx, &mut out, &upload::UploadArgs { no_register: true, ..upload_args(&src) }).unwrap();
    let s = text(out);
    let dest = ctx.dir().join("my-tune-q4_k_m/my-tune-Q4_K_M.gguf");
    assert_eq!(std::fs::read(&dest).unwrap(), bytes);
    let p = ymir_core::upload::read_provenance(dest.parent().unwrap()).unwrap();
    assert_eq!((p.license.as_deref(), p.files[0].sha256.as_str()), (Some("apache-2.0"), sha(&bytes).as_str()));
    assert!(s.contains("model (llama)") && s.contains(&format!("revision {}", p.revision)), "{s}");
    assert!(s.contains("[[model.local]]\nid = \"my-tune-q4_k_m\"") && s.contains(&format!("revision = \"{}\"", p.revision)), "{s}");
    assert!(s.contains("node.toml was not changed"), "{s}");
    assert_eq!(std::fs::read_to_string(&ctx.config).unwrap(), "[node]\nname = \"n\"\n");
    // It lists like any downloaded model, and uploading it again copies nothing.
    let mut out = Vec::new();
    list(&ctx, &mut out).unwrap();
    assert!(text(out).contains("my-tune-q4_k_m"));
    let mut out = Vec::new();
    run_upload(&ctx, &mut out, &upload::UploadArgs { name: Some("my-tune-q4_k_m".into()), no_register: true, ..upload_args(&src) }).unwrap();
    assert!(text(out).contains("nothing was copied"));
}

#[test]
fn upload_adds_the_model_to_node_toml_by_default_and_refuses_a_taken_id() {
    let fx = Fixture::new();
    let (ctx, _) = with_base(&fx);
    let src = fx.dir.path().join("tuned.gguf");
    std::fs::write(&src, gguf_bytes(20_000, 7)).unwrap();
    let mut out = Vec::new();
    run_upload(&ctx, &mut out, &upload_args(&src)).unwrap();
    assert!(text(out).contains("Added `tuned`"));
    let cfg = NodeConfig::load(&ctx.config).unwrap();
    let m = cfg.model.local.iter().find(|m| m.id == "tuned").unwrap();
    assert_eq!(m.role.as_deref(), Some("chat"));
    assert!(m.revision.as_deref().is_some_and(|r| r.starts_with("sha256:")));
    let ctx = fx.ctx(32, None);
    let err = run_upload(&ctx, &mut Vec::new(), &upload::UploadArgs { name: Some("base".into()), ..upload_args(&src) }).unwrap_err();
    assert!(err.to_string().contains("already has a model or adapter `base`"), "{err}");
    // Without node.toml there is nothing to add it to: it imports and says so.
    let bare = Fixture::new();
    let ctx = bare.ctx(32, None);
    let mut out = Vec::new();
    run_upload(&ctx, &mut out, &upload_args(&src)).unwrap();
    assert!(text(out).contains("nothing was registered"));
}

#[test]
fn upload_records_a_license_when_given_and_does_not_need_one() {
    let fx = Fixture::new();
    let ctx = fx.ctx(32, Some("[node]\nname = \"n\"\n"));
    let src = fx.dir.path().join("m.gguf");
    std::fs::write(&src, gguf_bytes(5_000, 1)).unwrap();
    let err = run_upload(&ctx, &mut Vec::new(), &upload::UploadArgs { license: Some("bad/license".into()), ..upload_args(&src) }).unwrap_err();
    assert!(err.to_string().contains("license id"), "{err}");
    assert!(!ctx.dir().join("m").exists(), "nothing copied before the checks pass");
    let mut out = Vec::new();
    run_upload(&ctx, &mut out, &upload::UploadArgs { license: None, no_register: true, ..upload_args(&src) }).unwrap();
    assert!(!text(out).contains("license"));
    assert_eq!(ymir_core::upload::read_provenance(&ctx.dir().join("m")).unwrap().license, None);
}

#[test]
fn upload_keeps_ten_percent_or_twenty_gib_of_the_disk_free() {
    let fx = Fixture::new();
    let ctx = fx.ctx(32, Some("[node]\nname = \"n\"\n"));
    let src = fx.dir.path().join("m.gguf");
    std::fs::write(&src, gguf_bytes(5_000, 1)).unwrap();
    let err = upload::run_with(&ctx, &mut Vec::new(), &upload_args(&src), Some((20 << 30, 100 << 30))).unwrap_err();
    assert!(err.to_string().contains("must stay free"), "{err}");
    upload::run_with(&ctx, &mut Vec::new(), &upload_args(&src), Some((21 << 30, 100 << 30))).unwrap();
}

#[test]
fn upload_of_an_adapter_checks_its_base_and_says_llama_cpp_needs_a_restart() {
    let fx = Fixture::new();
    let (ctx, _) = with_base(&fx);
    let src = fx.dir.path().join("support.gguf");
    std::fs::write(&src, lora_bytes("llama")).unwrap();
    // An adapter needs its base named, and a model must not name one.
    let err = run_upload(&ctx, &mut Vec::new(), &upload_args(&src)).unwrap_err();
    assert!(err.to_string().contains("--adapter-for"), "{err}");
    let model = fx.dir.path().join("full.gguf");
    std::fs::write(&model, gguf_bytes(5_000, 2)).unwrap();
    let err = run_upload(&ctx, &mut Vec::new(), &upload::UploadArgs { adapter_for: Some("base".into()), ..upload_args(&model) }).unwrap_err();
    assert!(err.to_string().contains("full model"), "{err}");
    // A base that isn't in node.toml, and one of another architecture, are refused before copying.
    let err = run_upload(&ctx, &mut Vec::new(), &upload::UploadArgs { adapter_for: Some("nope".into()), ..upload_args(&src) }).unwrap_err();
    assert!(err.to_string().contains("no [[model.local]] with id `nope`"), "{err}");
    let other = fx.dir.path().join("other.gguf");
    std::fs::write(&other, lora_bytes("qwen3")).unwrap();
    let err = run_upload(&ctx, &mut Vec::new(), &upload::UploadArgs { adapter_for: Some("base".into()), ..upload_args(&other) }).unwrap_err();
    assert!(err.to_string().contains("different model"), "{err}");
    assert!(!ctx.dir().join("adapters").exists());
    // The right one imports, registers with its revision and the base's, and says when it serves.
    let mut out = Vec::new();
    run_upload(&ctx, &mut out, &upload::UploadArgs { adapter_for: Some("base".into()), ..upload_args(&src) }).unwrap();
    let s = text(out);
    assert!(s.contains("Added LoRA adapter `base+support`") && s.contains("llama.cpp loads a new adapter file only when its base starts"), "{s}");
    assert!(!s.to_ascii_lowercase().contains("live"), "never claims the adapter is live: {s}");
    let cfg = NodeConfig::load(&ctx.config).unwrap();
    let a = &cfg.model.local[0].lora[0];
    assert_eq!((a.id.as_str(), a.base_revision.as_deref(), a.base_quant.as_deref()), ("base+support", Some("r1"), Some("Q4_K_M")));
    assert!(a.revision.as_deref().is_some_and(|r| r.starts_with("sha256:")));
    let dir = ctx.dir().join("adapters/base+support");
    let p = ymir_core::upload::read_provenance(&dir).unwrap();
    let base = p.base.unwrap();
    assert_eq!((base.id.as_deref(), base.revision.as_deref()), (Some("base"), Some("r1")));
    assert!(dir.join("adapter.json").is_file());
    let stored = ctx.store.adapters();
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0].id, "base+support");
}

// ---- `models upload --from-run` (docs/design/model-upload.md, U5) ----

/// A finished `train run` directory whose adapter is `bytes` at `file` (relative to models_dir),
/// trained on revision `base_revision` of `base`.
fn finished_run(fx: &Fixture, bytes: &[u8], file: &str, base_revision: &str) -> std::path::PathBuf {
    let dir = fx.dir.path().join("runs/20261124-150000-abcd");
    std::fs::create_dir_all(&dir).unwrap();
    let record = serde_json::json!({
        "format": ymir_core::train::RUN_FORMAT, "id": "20261124-150000-abcd", "node": "gpu1", "status": "done",
        "base": "base", "trained_from": "base.gguf",
        "dataset": {"name": "kad.jsonl", "sha256": "ab", "format": "ymir-dataset/1", "license": "mit", "teacher": "teacher-8b", "rows_in": 4, "rows_used": 2},
        "hyper": {"epochs": 2, "learning_rate": 1e-5, "context": 2048, "batch": 4, "seed": 0, "rank": 8},
        "trainer": {"interface": 1, "engine": "llama.cpp", "build": "b12628"},
        "steps": 4, "final_loss": 0.4,
        "adapter": {"id": "base+kad", "file": file, "sha256": sha(bytes), "bytes": bytes.len(), "base_revision": base_revision, "base_quant": "Q4_K_M"},
        "reserved_in_ymird": false, "started_ms": 1, "ended_ms": 2
    });
    std::fs::write(dir.join("run.json"), serde_json::to_vec_pretty(&record).unwrap()).unwrap();
    dir
}

fn from_run_args(run: &std::path::Path, file: Option<&std::path::Path>) -> upload::UploadArgs {
    upload::UploadArgs { file: file.map(std::path::Path::to_path_buf), from_run: Some(run.to_path_buf()), license: None, ..upload_args(run) }
}

#[test]
fn upload_from_a_run_carries_the_run_record_as_provenance() {
    let fx = Fixture::new();
    let (ctx, _) = with_base(&fx);
    let bytes = lora_bytes("llama");
    // A run made on another node: its adapter was copied here with its run directory.
    let copied = fx.dir.path().join("base+kad.gguf");
    std::fs::write(&copied, &bytes).unwrap();
    let run = finished_run(&fx, &bytes, "adapters/base+kad/base+kad.gguf", "r1");
    // Without the file, the run's adapter must be in this node's models directory.
    let err = run_upload(&ctx, &mut Vec::new(), &from_run_args(&run, None)).unwrap_err();
    assert!(err.to_string().contains("not in this node's models directory"), "{err}");
    let mut out = Vec::new();
    run_upload(&ctx, &mut out, &from_run_args(&run, Some(&copied))).unwrap();
    let s = text(out);
    assert!(s.contains("From run 20261124-150000-abcd on gpu1 (origin train)"), "{s}");
    assert!(s.contains("Added LoRA adapter `base+kad` on `base`") && s.contains("only when its base starts"), "{s}");
    let dir = ctx.dir().join("adapters/base+kad");
    let p = ymir_core::upload::read_provenance(&dir).unwrap();
    assert_eq!((p.origin.as_str(), p.license.as_deref(), p.files[0].sha256.as_str()), ("train", None, sha(&bytes).as_str()));
    let base = p.base.clone().unwrap();
    assert_eq!((base.id.as_deref(), base.revision.as_deref(), base.quant.as_deref()), (Some("base"), Some("r1"), Some("Q4_K_M")));
    let d = p.dataset.clone().unwrap();
    assert_eq!((d.licenses, d.rows, d.format.as_deref()), (vec!["mit".to_string()], Some(2), Some("ymir-dataset/1")));
    assert_eq!(p.extra["ymir_train"]["teacher"], "teacher-8b");
    let t = p.trainer.clone().unwrap();
    assert_eq!((t.name.as_deref(), t.build.as_deref(), t.run.as_deref()), (Some("llama.cpp"), Some("b12628"), Some("20261124-150000-abcd")));
    let cfg = NodeConfig::load(&ctx.config).unwrap();
    let a = &cfg.model.local[0].lora[0];
    assert_eq!((a.id.as_str(), a.revision.as_deref(), a.base_revision.as_deref()), ("base+kad", Some(p.revision.as_str()), Some("r1")));
    // Now on this node, the file can be left out: the same bytes are already there.
    let mut out = Vec::new();
    run_upload(&ctx, &mut out, &upload::UploadArgs { no_register: true, ..from_run_args(&run, None) }).unwrap();
    assert!(text(out).contains("nothing was copied"));
}

#[test]
fn upload_from_a_run_refuses_other_bytes_another_base_revision_and_unfinished_runs() {
    let fx = Fixture::new();
    let (ctx, _) = with_base(&fx);
    let bytes = lora_bytes("llama");
    let copied = fx.dir.path().join("a.gguf");
    std::fs::write(&copied, &bytes).unwrap();
    // Another file than the run made.
    let mut other = bytes.clone();
    other.push(1);
    let run = finished_run(&fx, &other, "adapters/base+kad/base+kad.gguf", "r1");
    let err = run_upload(&ctx, &mut Vec::new(), &from_run_args(&run, Some(&copied))).unwrap_err();
    assert!(err.to_string().contains("SHA-256 differs"), "{err}");
    // Trained on another revision of the base than this node serves.
    let run = finished_run(&fx, &bytes, "adapters/base+kad/base+kad.gguf", "r0");
    let err = run_upload(&ctx, &mut Vec::new(), &from_run_args(&run, Some(&copied))).unwrap_err();
    assert!(err.to_string().contains("trained on revision r0 of `base`"), "{err}");
    // A run that failed has no adapter.
    let path = run.join("run.json");
    let t = std::fs::read_to_string(&path).unwrap().replace("\"done\"", "\"failed\"");
    std::fs::write(&path, t).unwrap();
    let err = run_upload(&ctx, &mut Vec::new(), &from_run_args(&run, Some(&copied))).unwrap_err();
    assert!(err.to_string().contains("did not finish"), "{err}");
    let err = run_upload(&ctx, &mut Vec::new(), &from_run_args(&fx.dir.path().join("nope"), Some(&copied))).unwrap_err();
    assert!(err.to_string().contains("train run"), "{err:#}");
    assert!(!ctx.dir().join("adapters").exists(), "nothing copied before the checks pass");
}

// ---- `models upload --node` (docs/design/model-upload.md, U2): what is sent ----

#[test]
fn an_upload_to_another_node_carries_the_files_and_the_run_record() {
    let fx = Fixture::new();
    let bytes = lora_bytes("llama");
    let run = finished_run(&fx, &bytes, "adapters/base+kad/base+kad.gguf", "r1");
    // The adapter is found in the run directory when FILE is left out.
    std::fs::write(run.join("base+kad.gguf"), &bytes).unwrap();
    let a = upload::UploadArgs { node: Some("gpu".into()), ..from_run_args(&run, None) };
    let mut out = Vec::new();
    let op = upload::remote_op(&mut out, &a).unwrap();
    assert!(text(out).contains("LoRA adapter (llama)") && op.persist && op.role.is_none());
    let up = &op.upload;
    assert_eq!((up.id.as_str(), up.adapter_for.as_deref(), up.serve), ("base+kad", Some("base"), true));
    assert_eq!(up.files.len(), 1);
    assert_eq!(
        (up.files[0].path.as_str(), up.files[0].sha256.trim_start_matches("sha256:")),
        ("base+kad.gguf", sha(&bytes).trim_start_matches("sha256:"))
    );
    assert_eq!(op.local, vec![std::fs::canonicalize(run.join("base+kad.gguf")).unwrap()]);
    let p = &up.provenance;
    assert_eq!(p.origin, "train");
    let base = p.base.clone().unwrap();
    assert_eq!((base.id.as_deref(), base.revision.as_deref(), base.quant.as_deref()), (Some("base"), Some("r1"), Some("Q4_K_M")));
    assert_eq!(p.trainer.as_ref().and_then(|t| t.run.as_deref()), Some("20261124-150000-abcd"));
    // --no-persist; a model with --adapter-for and an adapter without a base are refused here.
    let op = upload::remote_op(&mut Vec::new(), &upload::UploadArgs { no_register: true, ..a }).unwrap();
    assert!(!op.persist);
    let lone = fx.dir.path().join("lone.gguf");
    std::fs::write(&lone, &bytes).unwrap();
    let err = upload::remote_op(&mut Vec::new(), &upload::UploadArgs { node: Some("gpu".into()), ..upload_args(&lone) }).unwrap_err();
    assert!(err.to_string().contains("--adapter-for BASE"), "{err}");
}
