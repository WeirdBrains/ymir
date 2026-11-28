//! Training plan P4b against a running ymird with two mock models: a rollout name answered by
//! its versions by weight, the canary share and the eval gate in `ymirctl models rollout`, live
//! weight changes (no restart), a pinned `name@N`, rollback, and node.toml kept in step.

use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::Parser as _;
use ymir_core::{Daemon, NodeConfig, Paths};

const MOCK: &str = env!("CARGO_BIN_EXE_ymir-mock-engine");

async fn ctl(home: &Path, args: &[&str]) -> Result<String> {
    let mut argv = vec!["ymirctl".to_string(), "--home".into(), home.display().to_string()];
    argv.extend(args.iter().map(|a| a.to_string()));
    let cli = ymirctl::Cli::try_parse_from(argv)?;
    let mut out = Vec::new();
    let mut input = std::io::Cursor::new(Vec::new());
    ymirctl::run(cli, &mut out, &mut input).await?;
    Ok(String::from_utf8(out)?)
}

fn setup(root: &Path) -> Result<(PathBuf, Paths)> {
    let home = root.join("ymir");
    let models = root.join("models");
    std::fs::create_dir_all(&home)?;
    std::fs::create_dir_all(&models)?;
    for m in ["s2", "s3"] {
        std::fs::write(models.join(format!("{m}.gguf")), b"GGUF mock")?;
    }
    let toml = format!(
        "[node]\nname = \"gpu1\"\n[roles]\nmodel = true\ninference = true\napp = true\n[model]\nllamacpp = \"system:{MOCK}\"\nvllm = \"off\"\nmodels_dir = \"{}\"\n\
         [[model.local]]\nid = \"s2\"\nfile = \"s2.gguf\"\nrole = \"main\"\n[[model.local]]\nid = \"s3\"\nfile = \"s3.gguf\"\n\
         [inference.alerts]\nprobe_secs = 3600\n[app.bridge]\nport = 0\n\
         [[inference.rollout]]\nname = \"support\"\n[[inference.rollout.version]]\nversion = 2\nmodel = \"s2\"\nweight = 100\n",
        models.display()
    );
    std::fs::write(home.join("node.toml"), toml)?;
    Ok((home.clone(), Paths::new(&home)))
}

/// Who answered `n` different conversations asked for `model`: (by s2, by s3).
async fn spread(home: &Path, model: &str, n: usize) -> Result<(usize, usize)> {
    let (mut s2, mut s3) = (0, 0);
    for i in 0..n {
        let reply = ctl(home, &["chat", "--wait", "--model", model, &format!("question {i}")]).await?;
        assert!(!reply.contains("stand-in"), "a version is not a stand-in: {reply}");
        s2 += usize::from(reply.contains("from s2!"));
        s3 += usize::from(reply.contains("from s3!"));
    }
    Ok((s2, s3))
}

#[tokio::test(flavor = "multi_thread")]
async fn a_rollout_takes_a_canary_needs_an_eval_for_more_and_rolls_back_live() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let (home, paths) = setup(tmp.path())?;
    let daemon = Daemon::start(NodeConfig::load(&paths.config())?, paths.clone()).await?;
    ctl(&home, &["load", "s2"]).await?;
    ctl(&home, &["load", "s3"]).await?;

    // From node.toml at start: all of `support` is v2.
    assert_eq!(spread(&home, "support", 6).await?, (6, 0));
    let list = ctl(&home, &["models", "rollout", "list"]).await?;
    assert!(list.contains("support@2") && list.contains("100.0%"), "{list}");

    // A canary within the share needs nothing more, and applies without a restart.
    let out = ctl(&home, &["models", "rollout", "add", "support@3", "--model", "s3", "--weight", "100"]).await.unwrap_err();
    assert!(format!("{out:#}").contains("canary share"), "{out:#}");
    let out = ctl(&home, &["models", "rollout", "add", "support@3", "--model", "s3", "--weight", "10"]).await?;
    assert!(out.contains("Applied to the running ymird"), "{out}");
    let (a, b) = spread(&home, "support", 40).await?;
    assert_eq!(a + b, 40);
    assert!(b <= 15, "about 9% on the canary, got {b}/40");
    // A pin reaches v3 whatever its weight.
    assert_eq!(spread(&home, "support@3", 2).await?, (0, 2));

    // More than the canary share: an eval in which v3 does at least as well.
    let eval = tmp.path().join("eval.jsonl");
    let row = |m: &str, id: &str, check: bool| {
        serde_json::json!({"model": m, "id": id, "ok": true, "answer": "", "check": check, "cache_hit": false}).to_string()
    };
    std::fs::write(&eval, [row("s3", "a", true), row("s2", "a", false)].join("\n"))?;
    let e = eval.display().to_string();
    ctl(&home, &["models", "rollout", "weight", "support@2", "0"]).await?;
    assert!(ctl(&home, &["models", "rollout", "weight", "support@3", "100", "--eval", &e]).await?.contains("100.0%"));
    assert_eq!(spread(&home, "support", 5).await?, (0, 5));

    // Rollback: v3's traffic back to v2, live, and in node.toml for the next start.
    let out = ctl(&home, &["models", "rollout", "rollback", "support"]).await?;
    assert!(out.contains("support@3 has no traffic now"), "{out}");
    assert_eq!(spread(&home, "support", 5).await?, (5, 0));
    let saved = NodeConfig::load(&paths.config())?.inference.rollout;
    assert_eq!(saved[0].version(2).unwrap().weight, 100);
    assert_eq!(saved[0].version(3).unwrap().weight, 0);

    // Retiring keeps the model loaded and drops the version.
    let out = ctl(&home, &["models", "rollout", "retire", "support@3"]).await?;
    assert!(out.contains("still loaded"), "{out}");
    assert!(ctl(&home, &["chat", "--wait", "--model", "support@3", "hi"]).await.is_err());
    daemon.shutdown().await;
    Ok(())
}

/// On a running node without the inference role, `rollout list` says that, not that ymird is
/// down (found on the nodes after 0.8.0).
#[tokio::test(flavor = "multi_thread")]
async fn rollout_list_on_a_node_without_inference_says_so() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let (home, paths) = setup(tmp.path())?;
    let toml = std::fs::read_to_string(paths.config())?.replace("inference = true\napp = true", "inference = false\napp = false");
    std::fs::write(paths.config(), toml)?;
    let daemon = Daemon::start(NodeConfig::load(&paths.config())?, paths.clone()).await?;
    let s = ctl(&home, &["models", "rollout", "list"]).await?;
    assert!(s.contains("no inference role"), "{s}");
    assert!(!s.contains("not running"), "{s}");
    daemon.shutdown().await;
    // Stopped: now it is the node.toml view.
    let s = ctl(&home, &["models", "rollout", "list"]).await?;
    assert!(s.contains("ymird is not running here"), "{s}");
    Ok(())
}
