//! Training plan P3a: a fine-tuning run on one node with the mock trainer (no model is read or
//! trained, nothing leaves the box). The run checks the trainer's pinned build and the
//! dataset's manifest and license, holds the node's GPU (a lock file, and a `weirdnet-backends`
//! reservation in a running ymird, which unloads its models and loads none until the run ends),
//! and leaves a LoRA adapter in the models directory and a run record. Failed and cancelled runs
//! release the GPU too; a hold taken by hand survives a ymird restart.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Result;
use clap::Parser as _;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use ymir_core::control::{self, ControlRequest};
use ymir_core::train::{self, RunSpec, RunStatus};
use ymir_core::{Daemon, NodeConfig, Paths};

const MOCK: &str = env!("CARGO_BIN_EXE_ymir-mock-engine");
const TRAINER: &str = env!("CARGO_BIN_EXE_ymir-mock-trainer");

async fn ctl(home: &Path, args: &[&str]) -> Result<String> {
    let mut argv = vec!["ymirctl".to_string(), "--home".into(), home.display().to_string()];
    argv.extend(args.iter().map(|a| a.to_string()));
    let cli = ymirctl::Cli::try_parse_from(argv)?;
    let mut out = Vec::new();
    let mut input = std::io::Cursor::new(Vec::new());
    ymirctl::run(cli, &mut out, &mut input).await?;
    Ok(String::from_utf8(out)?)
}

struct Home {
    root: PathBuf,
    home: PathBuf,
    paths: Paths,
}

impl Home {
    fn models(&self) -> PathBuf {
        self.root.join("models")
    }

    /// node.toml: a model and an inference role (no identity, so no Weirdnet), base `alpha`, and
    /// the mock trainer pinned at `build` with `env`.
    fn write(&self, build: &str, env: &[(&str, &str)]) -> Result<()> {
        let env: String = env.iter().map(|(k, v)| format!("{k} = \"{v}\"\n")).collect();
        let record = self.root.join("trainer.calls");
        let toml = format!(
            "[node]\nname = \"gpu1\"\n[roles]\nmodel = true\ninference = true\napp = false\n[model]\nllamacpp = \"system:{MOCK}\"\nvllm = \"off\"\nmodels_dir = \"{}\"\n[[model.local]]\nid = \"alpha\"\nfile = \"alpha.gguf\"\nrole = \"main\"\nrevision = \"r1\"\nquant = \"Q4_K_M\"\n[inference.alerts]\nprobe_secs = 3600\n[train]\ntrainer = \"{TRAINER}\"\nbuild = \"{build}\"\n[train.env]\nYMIR_MOCK_TRAINER_RECORD = \"{}\"\n{env}",
            self.models().display(),
            record.display()
        );
        std::fs::write(self.home.join("node.toml"), toml)?;
        Ok(())
    }
}

fn home(root: &Path) -> Result<Home> {
    let home = root.join("ymir");
    std::fs::create_dir_all(&home)?;
    std::fs::create_dir_all(root.join("models"))?;
    std::fs::write(root.join("models/alpha.gguf"), b"GGUF mock base")?;
    let h = Home { root: root.into(), paths: Paths::new(&home), home };
    h.write("b0000", &[])?;
    Ok(h)
}

/// A generated dataset (P1): two usable rows, a stand-in's answer and a failed row, with its
/// manifest; the teacher's license is `license`.
fn generated(dir: &Path, name: &str, license: &str) -> Result<PathBuf> {
    let path = dir.join(name);
    let row = |id: &str, ok: bool, usable: bool, q: &str, a: &str| json!({"id": id, "ok": ok, "usable": usable, "messages": [{"role": "user", "content": q}], "completion": a, "attempts": 1});
    let rows = [
        row("1", true, true, "What is a DHT?", "A distributed hash table."),
        row("2", true, true, "What is Kademlia?", "A DHT with an XOR metric."),
        row("3", true, false, "What is gossip?", "Stand-in answer."),
        row("4", false, false, "What is a relay?", ""),
    ];
    let body: String = rows.iter().map(|r| format!("{r}\n")).collect();
    std::fs::write(&path, &body)?;
    let manifest = ymir_core::dataset::Manifest {
        format: ymir_core::dataset::MANIFEST_FORMAT.into(),
        job: "dataset-1".into(),
        teacher: ymir_core::dataset::Teacher { model: "teacher-8b".into(), license: license.into(), license_source: "operator".into() },
        input_name: "prompts.jsonl".into(),
        input_sha256: "00".into(),
        output_name: name.into(),
        defaults: Default::default(),
        sources: None,
        cache: ymir_core::batch::CacheMode::Bypass,
        rows_in_input: 4,
        rows_answered: 3,
        rows_usable: 2,
        rows_failed: 1,
        cache_hits: 0,
        standins: 1,
        answered_by: Default::default(),
        created_ms: 0,
        updated_ms: 0,
    };
    std::fs::write(ymir_core::dataset::manifest_path(&path), serde_json::to_vec(&manifest)?)?;
    Ok(path)
}

/// An exported conversation dataset (P2) with its manifest.
fn conversations(dir: &Path) -> Result<PathBuf> {
    let path = dir.join("chats.jsonl");
    let body = format!("{}\n", json!({"id": "c1", "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}));
    std::fs::write(&path, &body)?;
    let manifest = json!({
        "format": ymir_core::convexport::FORMAT, "output_name": "chats.jsonl", "output_sha256": hex(&Sha256::digest(body.as_bytes())),
        "rules": {"keep_system": false, "redact_patterns": 0}, "conversations": [], "conversations_flagged": 1,
        "conversations_exported": 1, "turns_exported": 2,
        "dropped": {"system": 0, "tool": 0, "attachments": 0, "reasoning": 0, "incomplete": 0, "standin": 0, "after_cut": 0},
        "redactions": 0, "created_ms": 0
    });
    std::fs::write(ymir_core::dataset::manifest_path(&path), serde_json::to_vec(&manifest)?)?;
    Ok(path)
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn run_dir(h: &Home, out: &str) -> PathBuf {
    let run = out.split_whitespace().nth(1).expect("the run id").to_string();
    h.paths.state().join("train").join(run)
}

fn record(dir: &Path) -> Result<train::RunRecord> {
    Ok(serde_json::from_str(&std::fs::read_to_string(dir.join("run.json"))?)?)
}

#[tokio::test(flavor = "multi_thread")]
async fn a_run_trains_an_adapter_with_the_pinned_trainer_and_adds_it() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let h = home(tmp.path())?;
    let data = generated(tmp.path(), "kad.jsonl", "apache-2.0")?;

    let out = ctl(&h.home, &["train", "check"]).await?;
    assert!(out.contains("mock build b0000 (interface 1), as pinned"), "{out}");

    let d = data.display().to_string();
    let out =
        ctl(&h.home, &["train", "run", "--base", "alpha", "--dataset", &d, "--epochs", "2", "--rank", "8", "--id", "alpha+kad", "--add"]).await?;
    assert!(out.contains("for `alpha+kad`"), "{out}");
    assert!(out.contains("ymird isn't running; the run holds the GPU"), "{out}");
    assert!(out.contains("train: 4/4") && out.contains("extract-lora: 1/1"), "{out}");
    assert!(out.contains("Done: 4 steps, final loss 0.4000. Adapter `alpha+kad` is adapters/alpha+kad/alpha+kad.gguf"), "{out}");
    assert!(out.contains("Added it to node.toml on `alpha`"), "{out}");

    assert!(out.contains("Provenance: adapters/alpha+kad/provenance.json (origin train, run "), "{out}");
    let stored = h.models().join("adapters/alpha+kad");
    let bytes = std::fs::read(stored.join("alpha+kad.gguf"))?;
    assert!(bytes.windows(16).any(|w| w == b"mock lora rank 8"), "the trainer's adapter, moved in");
    let dir = run_dir(&h, &out);
    assert!(!dir.join("adapter.gguf").exists(), "the run's copy is removed once imported");
    let r = record(&dir)?;
    assert_eq!((r.status, r.steps, r.base.as_str(), r.trained_from.as_str()), (RunStatus::Done, 4, "alpha", "alpha.gguf"));
    assert_eq!((r.dataset.rows_in, r.dataset.rows_used, r.dataset.license.as_str()), (4, 2, "apache-2.0"));
    assert_eq!(r.dataset.teacher.as_deref(), Some("teacher-8b"));
    assert_eq!((r.trainer.build.as_str(), r.hyper.epochs, r.hyper.rank, r.reserved_in_ymird), ("b0000", 2, 8, false));
    let a = r.adapter.clone().unwrap();
    assert_eq!((a.base_revision.as_deref(), a.base_quant.as_deref(), a.bytes), (Some("r1"), Some("Q4_K_M"), bytes.len() as u64));
    assert_eq!(a.file, "adapters/alpha+kad/alpha+kad.gguf");
    // The same provenance.json an upload writes (origin train), with the run's record in it.
    let p = ymir_core::upload::read_provenance(&stored).expect("provenance.json");
    assert_eq!((p.origin.as_str(), p.id.as_str(), p.files[0].sha256.as_str()), ("train", "alpha+kad", a.sha256.as_str()));
    assert_eq!(Some(p.revision.clone()), a.revision);
    let base = p.base.clone().unwrap();
    assert_eq!((base.id.as_deref(), base.revision.as_deref(), base.quant.as_deref()), (Some("alpha"), Some("r1"), Some("Q4_K_M")));
    assert_eq!(p.dataset.as_ref().unwrap().licenses, ["apache-2.0"]);
    let t = p.trainer.clone().unwrap();
    assert_eq!((t.name.as_deref(), t.build.as_deref(), t.run.as_deref()), (Some("mock"), Some("b0000"), Some(r.id.as_str())));
    assert_eq!(p.extra["ymir_train"]["hyper"]["rank"], 8);
    assert!(stored.join("adapter.json").is_file(), "listed like a pulled adapter");
    assert!(!dir.join("full.gguf").exists(), "the full model is dropped unless --keep-full");
    // The trainer got the usable rows only, each ending in the teacher's answer.
    let rows: Vec<Value> = std::fs::read_to_string(dir.join("data.jsonl"))?.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[1]["messages"][1], json!({"role": "assistant", "content": "A DHT with an XOR metric."}));
    let calls = std::fs::read_to_string(h.root.join("trainer.calls"))?;
    assert!(calls.contains(&format!("train --base {}", h.models().join("alpha.gguf").display())), "{calls}");
    assert!(calls.contains("extract-lora --base") && calls.contains("--rank 8"), "{calls}");
    assert!(train::held(&h.paths).is_none(), "the GPU is free again");

    // node.toml serves it on alpha, keyed to this run and the base it was trained on.
    let cfg = NodeConfig::load(&h.paths.config())?;
    let lora = &cfg.model.local[0].lora;
    assert_eq!(lora.len(), 1);
    assert_eq!(
        (lora[0].id.as_str(), lora[0].file.as_str(), lora[0].revision.as_deref()),
        ("alpha+kad", "adapters/alpha+kad/alpha+kad.gguf", a.revision.as_deref())
    );
    assert_eq!((lora[0].base_revision.as_deref(), lora[0].base_quant.as_deref()), (Some("r1"), Some("Q4_K_M")));

    // A consented conversation export trains too; without --add the command to serve it is printed.
    let chats = conversations(tmp.path())?;
    let out = ctl(&h.home, &["train", "run", "--base", "alpha", "--dataset", &chats.display().to_string(), "--keep-full"]).await?;
    assert!(out.contains("To serve it: ymirctl models adapter add alpha adapters/alpha+ft-"), "{out}");
    assert!(out.contains("--revision sha256:"), "{out}");
    let r2 = record(&run_dir(&h, &out))?;
    assert_eq!((r2.dataset.rows_used, r2.dataset.teacher.clone()), (1, None));
    assert!(run_dir(&h, &out).join("full.gguf").exists(), "--keep-full");

    let list = ctl(&h.home, &["train", "list"]).await?;
    assert!(
        list.contains("done") && list.contains("-> alpha+kad (adapters/alpha+kad/alpha+kad.gguf)") && list.contains("chats.jsonl (1 rows)"),
        "{list}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_run_refuses_another_build_an_unknown_license_and_unrecorded_data() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let h = home(tmp.path())?;
    let unknown = generated(tmp.path(), "unknown.jsonl", "unknown")?;
    let run = |d: &Path| vec!["train".to_string(), "run".into(), "--base".into(), "alpha".into(), "--dataset".into(), d.display().to_string()];
    let ctl_v = |args: Vec<String>| {
        let home = h.home.clone();
        async move { ctl(&home, &args.iter().map(String::as_str).collect::<Vec<_>>()).await }
    };

    let e = ctl_v(run(&unknown)).await.unwrap_err();
    assert!(format!("{e:#}").contains("the teacher's license is unknown"), "{e:#}");
    let mut allowed = run(&unknown);
    allowed.push("--allow-unknown-license".into());
    ctl_v(allowed).await?;

    let bare = tmp.path().join("bare.jsonl");
    std::fs::write(&bare, "{\"messages\": []}\n")?;
    let e = ctl_v(run(&bare)).await.unwrap_err();
    assert!(format!("{e:#}").contains("needs the manifest"), "{e:#}");

    let chats = conversations(tmp.path())?;
    let mut f = std::fs::OpenOptions::new().append(true).open(&chats)?;
    writeln!(f, "{}", json!({"id": "c2", "messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]}))?;
    let e = ctl_v(run(&chats)).await.unwrap_err();
    assert!(format!("{e:#}").contains("changed since its manifest"), "{e:#}");

    let good = generated(tmp.path(), "good.jsonl", "mit")?;
    let mut dup = run(&good);
    dup.extend(["--id".into(), "alpha".into()]);
    assert!(format!("{:#}", ctl_v(dup).await.unwrap_err()).contains("already a model or adapter"));
    let mut other = run(&good);
    other[3] = "beta".into();
    assert!(format!("{:#}", ctl_v(other).await.unwrap_err()).contains("no [[model.local]] with id `beta`"));

    // Another build than the pin is refused before anything runs.
    h.write("b9999", &[])?;
    let e = ctl(&h.home, &["train", "check"]).await.unwrap_err();
    assert!(format!("{e:#}").contains("reports build b0000 (mock), and [train] build pins b9999"), "{e:#}");
    assert!(format!("{:#}", ctl_v(run(&good)).await.unwrap_err()).contains("pins b9999"));
    h.write("b0000", &[])?;

    // A hold taken by hand keeps runs off the GPU until it is released.
    let out = ctl(&h.home, &["train", "reserve"]).await?;
    assert!(out.contains("won't load a model when it starts"), "{out}");
    let e = ctl_v(run(&good)).await.unwrap_err();
    assert!(format!("{e:#}").contains("held by operator"), "{e:#}");
    assert!(ctl(&h.home, &["train", "list"]).await?.contains("held by operator"));
    assert!(ctl(&h.home, &["train", "release"]).await?.contains("released"));
    assert!(ctl(&h.home, &["train", "release"]).await?.contains("was not reserved"));
    ctl_v(run(&good)).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn failed_and_cancelled_runs_release_the_gpu() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let h = home(tmp.path())?;
    let data = generated(tmp.path(), "kad.jsonl", "apache-2.0")?;
    let d = data.display().to_string();

    for (step, why) in [("train", "out of memory"), ("extract-lora", "the tensors differ in shape")] {
        h.write("b0000", &[("YMIR_MOCK_TRAINER_FAIL", step)])?;
        let e = ctl(&h.home, &["train", "run", "--base", "alpha", "--dataset", &d]).await.unwrap_err();
        assert!(format!("{e:#}").contains(why), "{e:#}");
        assert!(train::held(&h.paths).is_none(), "released after a failed {step}");
    }
    let runs = train::runs(&h.paths, &NodeConfig::load(&h.paths.config())?.train);
    assert_eq!(runs.iter().map(|r| r.status).collect::<Vec<_>>(), [RunStatus::Failed, RunStatus::Failed]);
    assert!(runs.iter().all(|r| r.adapter.is_none() && r.error.is_some()));
    assert!(!h.models().join("adapters").exists(), "no adapter from a failed run");

    // Cancel mid-run: the trainer is stopped and the record says so.
    h.write("b0000", &[("YMIR_MOCK_TRAINER_STEP_MS", "400")])?;
    let cfg = NodeConfig::load(&h.paths.config())?;
    let spec =
        RunSpec { base: "alpha".into(), dataset: data.clone(), hyper: train::Hyper { epochs: 20, ..Default::default() }, ..Default::default() };
    let mut cancel = Box::pin(tokio::time::sleep(Duration::from_millis(1200)));
    let mut steps = 0;
    let e = train::run(&h.paths, &cfg, &spec, &mut cancel, &mut |u| {
        if matches!(u, train::Update::Event(train::Stage::Train, train::Event::Progress { .. })) {
            steps += 1;
        }
    })
    .await
    .unwrap_err();
    assert!(format!("{e:#}").contains("cancelled"), "{e:#}");
    assert!((1..40).contains(&steps), "{steps}");
    let runs = train::runs(&h.paths, &cfg.train);
    assert_eq!(runs[0].status, RunStatus::Cancelled);
    assert!(train::held(&h.paths).is_none());
    Ok(())
}

async fn status(h: &Home) -> Result<ymir_core::control::StatusReport> {
    Ok(control::request(&h.paths.socket(), &ControlRequest::Status).await?.status.unwrap())
}

#[tokio::test(flavor = "multi_thread")]
async fn ymird_unloads_for_a_run_loads_nothing_meanwhile_and_keeps_a_hold_across_a_restart() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let h = home(tmp.path())?;
    h.write("b0000", &[("YMIR_MOCK_TRAINER_STEP_MS", "300")])?;
    let data = generated(tmp.path(), "kad.jsonl", "apache-2.0")?;
    let d = data.display().to_string();
    let daemon = Daemon::start(NodeConfig::load(&h.paths.config())?, h.paths.clone()).await?;
    ctl(&h.home, &["load", "alpha"]).await?;

    // The only node serving alpha: refused unless forced.
    let e = ctl(&h.home, &["train", "run", "--base", "alpha", "--dataset", &d]).await.unwrap_err();
    assert!(format!("{e:#}").contains("gpu1 is the only node serving alpha"), "{e:#}");
    assert!(train::held(&h.paths).is_none(), "a refused reservation leaves no lock");

    let watch = async {
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        loop {
            let s = status(&h).await?;
            if s.nodes[0].reserved.is_some() {
                assert!(s.nodes[0].reserved.as_deref().unwrap().starts_with("ymir-train "));
                assert!(s.models.iter().all(|m| m.state != weirdnet_backends::ModelState::Ready), "alpha is unloaded for the run");
                break;
            }
            assert!(std::time::Instant::now() < deadline, "never reserved");
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        let nodes = ctl(&h.home, &["nodes"]).await?;
        assert!(nodes.contains("reserved") && nodes.contains("held by ymir-train"), "{nodes}");
        let e = ctl(&h.home, &["load", "alpha"]).await.unwrap_err();
        assert!(format!("{e:#}").contains("reserved (held by ymir-train"), "{e:#}");
        anyhow::Ok(())
    };
    let args = ["train", "run", "--base", "alpha", "--dataset", &d, "--epochs", "3", "--force"];
    let (out, watched) = tokio::join!(ctl(&h.home, &args), watch);
    watched?;
    let out = out?;
    assert!(out.contains("Reserved this node in ymird"), "{out}");
    assert!(record(&run_dir(&h, &out))?.reserved_in_ymird);
    assert!(status(&h).await?.nodes[0].reserved.is_none(), "released after the run");
    ctl(&h.home, &["load", "alpha"]).await?;

    // A hold by hand unloads now and outlives a restart of ymird.
    assert!(ctl(&h.home, &["train", "reserve", "--force"]).await?.contains("its models are unloaded"));
    assert!(ctl(&h.home, &["load", "alpha"]).await.is_err());
    daemon.shutdown().await;
    let daemon = Daemon::start(NodeConfig::load(&h.paths.config())?, h.paths.clone()).await?;
    assert_eq!(status(&h).await?.nodes[0].reserved.as_deref(), Some("operator"));
    assert!(ctl(&h.home, &["load", "alpha"]).await.is_err(), "still held after the restart");
    assert!(ctl(&h.home, &["train", "release"]).await?.contains("released"));
    ctl(&h.home, &["load", "alpha"]).await?;
    assert!(ctl(&h.home, &["nodes"]).await?.contains("serving"));
    daemon.shutdown().await;
    Ok(())
}
