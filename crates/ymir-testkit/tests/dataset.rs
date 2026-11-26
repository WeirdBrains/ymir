//! `ymirctl batch --jsonl` (training plan P1) against a fake pool: a real `ymird` in this process
//! with mock engines, the synthetic fixture in `tests/fixtures/`, and no network or real model.
//! Checks the rows, the manifest, the cache-hit marking, the license gate and `--resume`.

use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::Parser as _;
use serde_json::Value;
use ymir_core::{Daemon, NodeConfig, Paths};

const MOCK: &str = env!("CARGO_BIN_EXE_ymir-mock-engine");
const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/decentralized-networks.jsonl");

/// A node serving `teacher` and `other`; with `licensed`, `teacher`'s weights carry a provenance
/// record with a license (as an upload does), so the pool's catalog reports it.
async fn node_with(dir: &Path, licensed: bool) -> Result<(PathBuf, Daemon)> {
    let home = dir.join("home");
    let mdir = dir.join("models");
    std::fs::create_dir_all(&home)?;
    std::fs::create_dir_all(&mdir)?;
    std::fs::write(mdir.join("other.gguf"), b"GGUF mock")?;
    let teacher = if licensed {
        let up = mdir.join("uploads/teacher");
        std::fs::create_dir_all(&up)?;
        std::fs::write(up.join("teacher.gguf"), b"GGUF mock")?;
        let rec = serde_json::json!({"format": "weirdnet-provenance/1", "kind": "model", "id": "teacher", "origin": "upload",
            "revision": "sha256:00", "files": [], "license": "apache-2.0"});
        std::fs::write(up.join(weirdnet_model::artifact::PROVENANCE_FILE), serde_json::to_vec(&rec)?)?;
        "uploads/teacher/teacher.gguf"
    } else {
        std::fs::write(mdir.join("teacher.gguf"), b"GGUF mock")?;
        "teacher.gguf"
    };
    let toml = format!(
        "[node]\nname = \"dataset-node\"\n[roles]\nmodel = true\ninference = true\napp = true\n[model]\nllamacpp = \"system:{MOCK}\"\nvllm = \"off\"\nmodels_dir = \"{}\"\n\
         [[model.local]]\nid = \"other\"\nfile = \"other.gguf\"\nrole = \"main\"\n[[model.local]]\nid = \"teacher\"\nfile = \"{teacher}\"\n\
         [inference]\n[inference.alerts]\nprobe_secs = 3600\n[app.bridge]\nport = 0\n",
        mdir.display()
    );
    std::fs::write(home.join("node.toml"), toml)?;
    let paths = Paths::new(&home);
    let daemon = Daemon::start(NodeConfig::load(&paths.config())?, paths).await?;
    Ok((home, daemon))
}

async fn node(dir: &Path) -> Result<(PathBuf, Daemon)> {
    node_with(dir, false).await
}

/// Runs `ymirctl` on a blocking thread (it isn't Send) and returns its output or error.
async fn ctl(home: &Path, args: &[&str]) -> Result<String> {
    let mut argv = vec!["ymirctl".to_string(), "--home".into(), home.display().to_string()];
    argv.extend(args.iter().map(|a| a.to_string()));
    tokio::task::spawn_blocking(move || {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
        rt.block_on(async {
            let cli = ymirctl::Cli::try_parse_from(argv)?;
            let mut out = Vec::new();
            let mut input = std::io::Cursor::new(Vec::new());
            ymirctl::run(cli, &mut out, &mut input).await?;
            Ok(String::from_utf8(out)?)
        })
    })
    .await?
}

fn rows(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn jsonl_generation_marks_rows_writes_a_manifest_and_resumes() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let (home, daemon) = node(dir.path()).await?;
    let input = dir.path().join("prompts.jsonl");
    // The first three rows only, then all of them on resume.
    let fixture = std::fs::read_to_string(FIXTURE)?;
    let first: String = fixture.lines().take(4).map(|l| format!("{l}\n")).collect();
    std::fs::write(&input, &first)?;
    let out = dir.path().join("data.jsonl");
    let (i, o) = (input.display().to_string(), out.display().to_string());
    let base = ["batch", &i, "--jsonl", "--model", "teacher", "--out", &o, "--concurrency", "1", "--max-tokens", "64"];

    // No license: refused before anything runs.
    let e = ctl(&home, &base).await.unwrap_err();
    assert!(format!("{e:#}").contains("--license"), "{e:#}");
    let e = ctl(&home, &[&base[..], &["--license", "unknown"]].concat()).await.unwrap_err();
    assert!(format!("{e:#}").contains("--allow-unknown-license"), "{e:#}");
    assert!(!out.exists());

    let with_license = [&base[..], &["--license", "apache-2.0", "--system", "Answer briefly."]].concat();
    let said = ctl(&home, &with_license).await?;
    assert!(said.contains("3 sent, 0 skipped") && said.contains("3 usable"), "{said}");
    let r = rows(&out);
    assert_eq!(r.iter().map(|x| x["id"].as_str().unwrap()).collect::<Vec<_>>(), ["dn-1", "dn-2", "dn-3"]);
    for x in &r {
        assert_eq!((x["ok"].as_bool(), x["usable"].as_bool(), x["cache_hit"].as_bool()), (Some(true), Some(true), Some(false)), "{x}");
        assert_eq!(x["model"], "teacher", "{x}");
        assert!(x["completion"].as_str().unwrap().starts_with("Hello from teacher!"), "{x}");
    }
    // The default system prompt goes on rows without one; dn-3 keeps its own.
    assert_eq!(r[0]["messages"][0]["role"], "system");
    assert_eq!(r[0]["messages"][0]["content"], "Answer briefly.");
    assert_eq!(r[2]["messages"][0]["content"], "You explain networking to beginners.");
    let manifest: Value = serde_json::from_str(&std::fs::read_to_string(dir.path().join("data.jsonl.manifest.json"))?)?;
    assert_eq!(manifest["format"], "ymir-dataset/1");
    assert_eq!(manifest["teacher"]["model"], "teacher");
    assert_eq!(manifest["teacher"]["license"], "apache-2.0");
    assert_eq!(manifest["input_name"], "prompts.jsonl");
    assert_eq!((manifest["rows_answered"].as_u64(), manifest["answered_by"]["teacher"].as_u64()), (Some(3), Some(3)));
    assert_eq!(manifest["cache"], "bypass");
    assert!(manifest.get("sources").is_none());
    let job = manifest["job"].as_str().unwrap().to_string();

    // The output exists: refused without --resume.
    let e = ctl(&home, &with_license).await.unwrap_err();
    assert!(format!("{e:#}").contains("--resume"), "{e:#}");

    // Resume over the whole fixture: only the new rows run. dn-4-again repeats dn-4 at
    // temperature 0, but the default `--cache bypass` goes on the wire, so it is a fresh answer.
    std::fs::write(&input, &fixture)?;
    let said = ctl(&home, &[&with_license[..], &["--resume"]].concat()).await?;
    assert!(said.contains("2 sent, 3 skipped") && said.contains("5 rows answered, 5 usable") && said.contains("0 from the cache"), "{said}");
    let r = rows(&out);
    assert_eq!(r.len(), 5, "answered rows are not run again");
    let again = r.iter().find(|x| x["id"] == "dn-4-again").unwrap();
    assert_eq!((again["ok"].as_bool(), again["cache_hit"].as_bool(), again["usable"].as_bool()), (Some(true), Some(false), Some(true)), "{again}");
    let manifest: Value = serde_json::from_str(&std::fs::read_to_string(dir.path().join("data.jsonl.manifest.json"))?)?;
    assert_eq!(manifest["job"], job.as_str(), "a resumed run keeps its job id");
    assert_eq!((manifest["rows_in_input"].as_u64(), manifest["rows_usable"].as_u64(), manifest["cache_hits"].as_u64()), (Some(5), Some(5), Some(0)));

    // With `--cache use` the node's cache may answer: bypass stored nothing above, so dn-4 runs
    // and is stored, and dn-4-again (same question, temperature 0) is a hit, marked unusable.
    let used = dir.path().join("used.jsonl").display().to_string();
    let with_use =
        with_license.iter().copied().map(|a| if a == o.as_str() { used.as_str() } else { a }).chain(["--cache", "use"]).collect::<Vec<_>>();
    let said = ctl(&home, &with_use).await?;
    assert!(said.contains("5 rows answered, 4 usable") && said.contains("1 from the cache"), "{said}");
    let r = rows(Path::new(&used));
    let again = r.iter().find(|x| x["id"] == "dn-4-again").unwrap();
    assert_eq!((again["cache_hit"].as_bool(), again["usable"].as_bool()), (Some(true), Some(false)), "{again}");

    // Another teacher can't continue this output.
    let other = [&base[..], &["--license", "apache-2.0", "--resume"]]
        .concat()
        .into_iter()
        .map(|a| if a == "teacher" { "other" } else { a })
        .collect::<Vec<_>>();
    let e = ctl(&home, &other).await.unwrap_err();
    assert!(format!("{e:#}").contains("teacher"), "{e:#}");

    daemon.shutdown().await;
    Ok(())
}

const SOURCES_FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/decentralized-networks.sources.json");

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn jsonl_generation_records_its_sources() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let (home, daemon) = node(dir.path()).await?;
    let input = dir.path().join("rows.jsonl");
    std::fs::write(
        &input,
        "{\"id\": \"s1\", \"prompt\": \"How does a DHT find a key?\", \"source\": \"dht-notes\"}\n\
         {\"id\": \"s2\", \"prompt\": \"Why do peers use relays?\", \"source\": \"relay-notes\"}\n\
         {\"id\": \"s3\", \"prompt\": \"What is a k-bucket?\", \"source\": \"dht-notes\"}\n",
    )?;
    let out = dir.path().join("data.jsonl");
    let (i, o) = (input.display().to_string(), out.display().to_string());
    let args = ["batch", &i, "--jsonl", "--model", "teacher", "--out", &o, "--license", "apache-2.0", "--sources", SOURCES_FIXTURE];
    let said = ctl(&home, &args).await?;
    assert!(said.contains("3 usable"), "{said}");
    let r = rows(&out);
    assert_eq!(r.iter().map(|x| x["source"].as_str().unwrap()).collect::<Vec<_>>(), ["dht-notes", "relay-notes", "dht-notes"]);
    let m: Value = serde_json::from_str(&std::fs::read_to_string(dir.path().join("data.jsonl.manifest.json"))?)?;
    let s = &m["sources"];
    assert_eq!(s["name"], "decentralized-networks.sources.json");
    assert_eq!(s["topic"], "Decentralized networks: how they work");
    assert_eq!(s["sources"][1]["license"], "CC-BY-4.0");
    assert_eq!((s["rows"]["dht-notes"].as_u64(), s["rows"]["relay-notes"].as_u64()), (Some(2), Some(1)));
    assert_eq!(s["sha256"].as_str().unwrap().len(), 64);

    // A row naming a source that isn't listed is refused before anything runs.
    std::fs::write(&input, "{\"id\": \"x\", \"prompt\": \"q\", \"source\": \"nowhere\"}\n")?;
    let o2 = dir.path().join("bad.jsonl").display().to_string();
    let e = ctl(&home, &["batch", &i, "--jsonl", "--model", "teacher", "--out", &o2, "--license", "apache-2.0", "--sources", SOURCES_FIXTURE])
        .await
        .unwrap_err();
    assert!(format!("{e:#}").contains("not in the sources manifest"), "{e:#}");
    assert!(!dir.path().join("bad.jsonl").exists());
    daemon.shutdown().await;
    Ok(())
}

const EVAL_FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/decentralized-networks-eval.jsonl");

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn eval_scores_models_with_checks_and_a_judge() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let (home, daemon) = node(dir.path()).await?;
    let out = dir.path().join("eval.jsonl");
    let o = out.display().to_string();
    let args = ["eval", EVAL_FIXTURE, "--model", "teacher", "--model", "other", "--judge", "other", "--out", &o, "--concurrency", "1"];
    let said = ctl(&home, &args).await?;
    // The mock echoes the prompt: ev-1 contains its word, ev-2 is not the exact answer, and the
    // mock judge never says PASS or FAIL.
    let line = |m: &str| said.lines().find(|l| l.starts_with(m)).unwrap_or_default().split_whitespace().map(str::to_string).collect::<Vec<_>>();
    assert_eq!(line("teacher "), ["teacher", "1/2", "-", "0", "0", "0"], "{said}");
    assert_eq!(line("other "), ["other", "1/2", "-", "0", "0", "0"], "{said}");
    assert!(said.contains("is also a model under test"), "{said}");
    assert!(said.contains("teacher: the judge gave no clear PASS/FAIL on 1 row(s)"), "{said}");
    let rows: Vec<Value> = rows(&out);
    assert_eq!(rows.len(), 6);
    let ev1 = rows.iter().find(|r| r["model"] == "teacher" && r["id"] == "ev-1").unwrap();
    assert_eq!((ev1["check"].as_bool(), ev1["cache_hit"].as_bool()), (Some(true), Some(false)), "{ev1}");
    let ev3 = rows.iter().find(|r| r["model"] == "teacher" && r["id"] == "ev-3").unwrap();
    assert!(
        ev3.get("check").is_none() && ev3.get("judge").is_none() && ev3["judge_reply"].as_str().unwrap().starts_with("Hello from other!"),
        "{ev3}"
    );

    // Run again: every answer is deterministic, but evals bypass the answer cache, so none is
    // replayed.
    let said = ctl(&home, &args).await?;
    assert_eq!(line_of(&said, "teacher "), ["teacher", "1/2", "-", "0", "0", "0"], "{said}");

    // No judge: rubric rows are noted and scored by `expect` only.
    let said = ctl(&home, &["eval", EVAL_FIXTURE, "--model", "teacher"]).await?;
    assert!(said.contains("have a rubric but no --judge"), "{said}");
    daemon.shutdown().await;
    Ok(())
}

fn line_of(said: &str, m: &str) -> Vec<String> {
    said.lines().find(|l| l.starts_with(m)).unwrap_or_default().split_whitespace().map(str::to_string).collect()
}

const DATASET_FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/decentralized-networks.dataset.jsonl");

/// The synthetic training dataset next to the prompt fixture: hand-written answers in
/// `ymirctl batch --jsonl`'s output format with its `ymir-dataset/1` manifest, so `ymirctl train
/// run --dataset` takes it as it is (training plan P3a's first run).
#[test]
fn the_fixture_dataset_is_accepted_for_training() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let out = dir.path().join("DATA.jsonl");
    let rec = ymir_core::train::prepare_dataset(Path::new(DATASET_FIXTURE), false, &out)?;
    assert_eq!(rec.format, ymir_core::dataset::MANIFEST_FORMAT);
    assert_eq!((rec.rows_in, rec.rows_used), (5, 4), "the cached repeat is not trained on");
    assert_eq!(rec.license, "CC0-1.0");
    assert_eq!(rec.teacher.as_deref(), Some("synthetic-fixture"));
    let train: Vec<Value> = rows(&out);
    assert_eq!(train.len(), 4);
    for r in &train {
        let turns = r["messages"].as_array().expect("messages");
        assert_eq!(turns.last().expect("a turn")["role"], "assistant");
        assert!(turns.iter().any(|t| t["role"] == "user"));
    }
    assert_eq!(train[2]["messages"][0]["role"], "system", "a row's system prompt is kept");

    // The manifest stays true to the files next to it.
    let m: ymir_core::dataset::Manifest =
        serde_json::from_str(&std::fs::read_to_string(ymir_core::dataset::manifest_path(Path::new(DATASET_FIXTURE)))?)?;
    let sha =
        |p: &str| -> Result<String> { Ok(<sha2::Sha256 as sha2::Digest>::digest(std::fs::read(p)?).iter().map(|b| format!("{b:02x}")).collect()) };
    assert_eq!(m.input_sha256, sha(FIXTURE)?, "the prompt fixture changed: update input_sha256 in the dataset manifest");
    assert_eq!(m.sources.as_ref().expect("sources").sha256, sha(SOURCES_FIXTURE)?, "the sources fixture changed: update the dataset manifest");
    assert_eq!((m.rows_in_input, m.rows_usable, m.cache_hits), (5, 4, 1));
    Ok(())
}

const LARGE_INPUT_FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/decentralized-networks-large.jsonl");
const LARGE_DATASET_FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/decentralized-networks-large.dataset.jsonl");

/// The larger synthetic dataset: the small one renders to about 256 tokens, under the 513
/// (2 x 256 + 1) llama-finetune needs for one step at its smallest context, so a real first run
/// takes this one (1297 tokens with Qwen2.5's chat template, counted by llama-tokenize).
#[test]
fn the_large_fixture_dataset_is_accepted_for_training() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let out = dir.path().join("DATA.jsonl");
    let rec = ymir_core::train::prepare_dataset(Path::new(LARGE_DATASET_FIXTURE), false, &out)?;
    assert_eq!((rec.rows_in, rec.rows_used), (16, 16));
    assert_eq!(rec.license, "CC0-1.0");
    let train: Vec<Value> = rows(&out);
    let chars: usize =
        train.iter().flat_map(|r| r["messages"].as_array().expect("messages").iter()).map(|t| t["content"].as_str().expect("content").len()).sum();
    assert!(chars > 3 * 1295, "well over the small fixture's text ({chars} characters)");
    assert!(train.iter().any(|r| r["messages"][0]["role"] == "system"));

    let m: ymir_core::dataset::Manifest =
        serde_json::from_str(&std::fs::read_to_string(ymir_core::dataset::manifest_path(Path::new(LARGE_DATASET_FIXTURE)))?)?;
    let sha =
        |p: &str| -> Result<String> { Ok(<sha2::Sha256 as sha2::Digest>::digest(std::fs::read(p)?).iter().map(|b| format!("{b:02x}")).collect()) };
    assert_eq!(m.input_sha256, sha(LARGE_INPUT_FIXTURE)?, "the large prompt fixture changed: update input_sha256 in its manifest");
    assert_eq!(m.sources.as_ref().expect("sources").sha256, sha(SOURCES_FIXTURE)?, "the sources fixture changed: update the large manifest");
    assert_eq!((m.rows_in_input, m.rows_usable, m.cache_hits), (16, 16, 0));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_teacher_license_comes_from_the_pool_catalog_when_not_given() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let (home, daemon) = node_with(dir.path(), true).await?;
    let input = dir.path().join("prompts.jsonl");
    let first: String = std::fs::read_to_string(FIXTURE)?.lines().take(2).map(|l| format!("{l}\n")).collect();
    std::fs::write(&input, &first)?;
    let i = input.display().to_string();
    let o = dir.path().join("data.jsonl").display().to_string();
    let said = ctl(&home, &["batch", &i, "--jsonl", "--model", "teacher", "--out", &o, "--max-tokens", "64"]).await?;
    assert!(said.contains("Teacher license: apache-2.0 (the pool's catalog, upload)"), "{said}");
    let m: Value = serde_json::from_str(&std::fs::read_to_string(dir.path().join("data.jsonl.manifest.json"))?)?;
    assert_eq!(m["teacher"]["license"], "apache-2.0");
    assert_eq!(m["teacher"]["license_source"], "pool");
    // `other` has no recorded license: still refused, naming both ways to give one.
    let o2 = dir.path().join("other.jsonl").display().to_string();
    let e = ctl(&home, &["batch", &i, "--jsonl", "--model", "other", "--out", &o2]).await.unwrap_err();
    assert!(format!("{e:#}").contains("--license-from"), "{e:#}");
    // --license still wins, recorded as the operator's.
    let o3 = dir.path().join("given.jsonl").display().to_string();
    ctl(&home, &["batch", &i, "--jsonl", "--model", "teacher", "--out", &o3, "--license", "mit", "--max-tokens", "64"]).await?;
    let m: Value = serde_json::from_str(&std::fs::read_to_string(dir.path().join("given.jsonl.manifest.json"))?)?;
    assert_eq!((m["teacher"]["license"].as_str(), m["teacher"]["license_source"].as_str()), (Some("mit"), Some("operator")));
    daemon.shutdown().await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn seed_prompts_come_from_source_texts_in_two_passes() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let (home, daemon) = node(dir.path()).await?;
    let sources = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/decentralized-networks.sources.json");
    let texts = dir.path().join("texts");
    std::fs::create_dir_all(&texts)?;
    let para = "A distributed hash table spreads keys over many peers. Each peer keeps a small table of others, sorted by distance, and a lookup asks ever closer peers until it reaches the key's owner. Peers that leave are replaced as others notice them gone.";
    std::fs::write(texts.join("dht-notes.txt"), format!("{para}\n\n{para}\n\n{para}"))?;
    let (t, q) = (texts.display().to_string(), dir.path().join("q.jsonl").display().to_string());
    let base = ["train", "seed", "questions", "--sources", sources, "--texts", &t, "--out", &q];
    let e = ctl(&home, &base).await.unwrap_err();
    assert!(format!("{e:#}").contains("no text for source(s) relay-notes"), "{e:#}");
    std::fs::write(texts.join("relay-notes.md"), format!("# Relays\n\n{para}"))?;
    let said = ctl(&home, &[&base[..], &["--chunk-chars", "600"]].concat()).await?;
    assert!(said.contains("Wrote 3 question rows") && said.contains("dht-notes 2, relay-notes 1"), "{said}");
    // The rows go through the teacher like any other, sources checked.
    let qa = dir.path().join("qa.jsonl").display().to_string();
    let said = ctl(
        &home,
        &["batch", &q, "--jsonl", "--model", "teacher", "--license", "apache-2.0", "--sources", sources, "--out", &qa, "--max-tokens", "64"],
    )
    .await?;
    assert!(said.contains("3 sent"), "{said}");
    assert!(ctl(&home, &["train", "seed", "questions", "--sources", sources, "--texts", &t, "--out", &q]).await.is_err(), "overwrote");

    // Pass 2 on answers that ask questions (the mock engine doesn't).
    let answers = dir.path().join("answers.jsonl");
    std::fs::write(
        &answers,
        [
            r#"{"id": "q-dht-notes-001", "source": "dht-notes", "usable": true, "completion": "Question: How does a DHT lookup find a key?\nQuestion: What does each peer keep?"}"#,
            r#"{"id": "q-relay-notes-001", "source": "relay-notes", "usable": true, "completion": "Question: How does a DHT lookup find a key?"}"#,
        ]
        .join("\n"),
    )?;
    let p = dir.path().join("prompts.jsonl").display().to_string();
    let said = ctl(&home, &["train", "seed", "prompts", &answers.display().to_string(), "--out", &p]).await?;
    assert!(said.contains("Wrote 2 prompt rows") && said.contains("1 duplicate"), "{said}");
    let out = dir.path().join("data.jsonl").display().to_string();
    let said = ctl(
        &home,
        &["batch", &p, "--jsonl", "--model", "teacher", "--license", "apache-2.0", "--sources", sources, "--out", &out, "--max-tokens", "64"],
    )
    .await?;
    assert!(said.contains("2 sent"), "{said}");
    assert!(rows(Path::new(&out)).iter().all(|r| r["source"] == "dht-notes"));
    daemon.shutdown().await;
    Ok(())
}
