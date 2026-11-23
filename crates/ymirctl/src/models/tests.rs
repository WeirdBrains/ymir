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
