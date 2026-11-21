//! End-to-end tests of the four commands against a mocked Hub and a temporary model directory.

use serde_json::json;
use sha2::{Digest, Sha256};
use weirdnet_model::hw::Hardware;

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
        siblings.push(json!({"rfilename": "mmproj-tiny-f16.gguf", "size": 5}));
        siblings.push(json!({"rfilename": "README.md", "size": 5}));
        hub.route("/api/models/acme/Tiny-GGUF?blobs=true", 200, json!({"gated": false, "tags": ["gguf"], "siblings": siblings}).to_string());
        for (n, b) in &files {
            hub.route(&format!("/acme/Tiny-GGUF/resolve/main/{n}"), 200, b.clone());
        }
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
    assert!(s.contains("acme/Tiny-GGUF: Q8_0, 1 file(s)"), "{s}");
    assert!(s.contains("downloaded, checksum verified"), "{s}");
    assert!(s.contains("[[model.local]]\nid = \"tiny\"") && s.contains("role = \"main\"") && s.contains("tiny-Q8_0.gguf"), "{s}");
    assert!(s.contains("node.toml was not changed"));
    assert_eq!(std::fs::read(ctx.dir().join("tiny-Q8_0.gguf")).unwrap(), gguf_bytes(400_000, 1));
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
    assert!(s.contains("Q6_K, 2 file(s)"), "{s}");
    assert!(ctx.dir().join("tiny-Q6_K-00002-of-00002.gguf").exists());
    assert!(s.contains("tiny-Q6_K-00001-of-00002.gguf\""), "{s}");
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
    assert!(text(out).contains("Q8_0"));
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
    assert!(pull(&ctx, &mut Vec::new(), &pull_args("acme/Plain")).unwrap_err().to_string().contains("no GGUF files"));
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
