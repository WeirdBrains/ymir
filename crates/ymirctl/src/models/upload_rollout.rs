//! `ymirctl models upload FILE --node X --rollout NAME` (docs/design/model-upload.md, slice U3):
//! after the upload, the model becomes the next version of rollout NAME on this machine's
//! inference role (P4b's `[[inference.rollout]]`), at weight 0. `--canary PCT` then gives it
//! exactly PCT percent (0 to 100) of new conversations, the other versions sharing the rest; with
//! `--eval ROWS` (run here on the new version's model and the one with the most traffic, on the
//! same rows) only when it passes P4b's gate. A named percentage is not capped (m3talsmith,
//! 2026-11-25). With `--inference NODE` the rollout is that inference node's, changed over
//! Weirdnet (`/ymir/rollouts/1`) as `ymirctl models rollout ... --inference NODE` does.

use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};
use ymir_core::Paths;
use ymir_core::config::NodeConfig;
use ymir_core::rollout::Rollout;

use super::rollout::{self, Gate, RolloutCmd};

/// What `--rollout` asks for.
#[derive(clap::Args, Debug, Clone, Default)]
pub struct RolloutArgs {
    /// Make the upload the next version of this rollout on this machine's inference role, at
    /// weight 0 (pinnable as NAME@N). With --node only. Without --name the id is `ID-rN`.
    #[arg(long, value_name = "NAME")]
    pub rollout: Option<String>,
    /// The inference node whose rollout it joins (a device name or peer id), over Weirdnet,
    /// instead of this machine's inference role. It saves the change in its node.toml unless
    /// --no-persist is given.
    #[arg(long, value_name = "NODE", requires = "rollout")]
    pub inference: Option<String>,
    /// Give the new version exactly this percentage (0 to 100) of new conversations once it is
    /// added, such as 20 or 50; the other versions share the rest.
    #[arg(long, value_name = "PCT", requires = "rollout", value_parser = clap::value_parser!(u32).range(0..=100))]
    pub canary: Option<u32>,
    /// Eval rows (`ymirctl eval` format) to score the new version against the version with the
    /// most traffic before --canary; the share is set only if it does at least as well.
    #[arg(long, value_name = "ROWS.jsonl", requires = "canary")]
    pub eval: Option<PathBuf>,
    /// A model that grades the eval's rubric rows.
    #[arg(long, requires = "eval")]
    pub judge: Option<String>,
    /// Set the --canary share even if the --eval check fails.
    #[arg(long, requires = "canary")]
    pub force: bool,
}

/// The rollouts on `inference` (over Weirdnet), or this machine's: ymird's live ones, else node.toml's.
async fn rollouts(paths: &Paths, inference: Option<&str>) -> Result<Vec<Rollout>> {
    if let Some(node) = inference {
        return rollout::remote_list(paths, node).await;
    }
    if let Ok(r) = ymir_core::control::request(&paths.socket(), &ymir_core::control::ControlRequest::Rollouts { set: None }).await
        && let Some(live) = r.rollouts
    {
        return Ok(live);
    }
    let config = paths.config();
    let text = std::fs::read_to_string(&config).with_context(|| format!("reading {} (run `ymirctl setup` first)", config.display()))?;
    let cfg: NodeConfig = toml::from_str(&text).with_context(|| format!("parsing {}", config.display()))?;
    Ok(cfg.inference.rollout)
}

/// The version number the upload gets in rollout `name`.
pub async fn next_version(paths: &Paths, name: &str, inference: Option<&str>) -> Result<u32> {
    let all = rollouts(paths, inference).await?;
    Ok(all.iter().find(|r| r.name == name).map_or(1, |r| r.versions.iter().map(|v| v.version).max().unwrap_or(0) + 1))
}

/// The id a rollout version's upload gets when `--name` is not given.
pub fn versioned_id(id: &str, n: u32) -> String {
    format!("{id}-r{n}")
}

/// One rollout change, here or on `a.inference` (saved there when `persist`).
async fn change(paths: &Paths, out: &mut dyn Write, a: &RolloutArgs, cmd: RolloutCmd, persist: bool) -> Result<()> {
    match &a.inference {
        Some(node) => rollout::run_remote(paths, out, node, cmd, persist).await,
        None => rollout::run(paths, out, cmd).await,
    }
}

/// Adds `model` as NAME@N at weight 0, then (with --canary) evaluates and sets the weight.
/// `persist`: an inference node named with --inference saves the change in its node.toml.
pub async fn join(paths: &Paths, out: &mut dyn Write, a: &RolloutArgs, n: u32, model: &str, adapter_waits: bool, persist: bool) -> Result<()> {
    let Some(name) = &a.rollout else { return Ok(()) };
    let version = format!("{name}@{n}");
    let add = RolloutCmd::Add { version: version.clone(), model: model.to_string(), weight: 0, canary: None, gate: Gate::default() };
    change(paths, out, a, add, persist).await?;
    if adapter_waits {
        writeln!(out, "{version} waits at weight 0 until `{model}` answers (its base loads it on its next start).")?;
    }
    let Some(w) = a.canary else {
        let on = a.inference.as_deref().map(|n| format!(" --inference {n}")).unwrap_or_default();
        writeln!(out, "Give it traffic with `ymirctl models rollout weight {version} --canary PCT{on}` (a percentage, such as 20).")?;
        return Ok(());
    };
    let mut gate = Gate { eval: None, force: a.force };
    if let Some(rows) = &a.eval {
        match incumbent(paths, name, n, a.inference.as_deref()).await? {
            None => writeln!(out, "No other version of `{name}` has traffic, so there is nothing to compare with; the eval is skipped.")?,
            Some(incumbent) => gate.eval = Some(evaluate(paths, out, a, rows, &version, model, &incumbent).await?),
        }
    }
    let set =
        change(paths, out, a, RolloutCmd::Weight { version: version.clone(), weight: None, canary: Some(w), gate: gate.clone() }, persist).await;
    if let Some(f) = &gate.eval {
        let _ = std::fs::remove_file(f);
    }
    set.with_context(|| format!("{version} stays at weight 0"))
}

/// Runs `ymirctl eval` on `rows` for the new version's model and the incumbent's; returns the
/// `--out` file the gate reads.
async fn evaluate(
    paths: &Paths,
    out: &mut dyn Write,
    a: &RolloutArgs,
    rows: &std::path::Path,
    version: &str,
    model: &str,
    incumbent: &str,
) -> Result<PathBuf> {
    let (name, n) = version.split_once('@').unwrap_or((version, ""));
    {
        let file = std::env::temp_dir().join(format!("ymir-eval-{name}-{n}-{}.jsonl", std::process::id()));
        writeln!(out, "Evaluating `{model}` against `{incumbent}` on {}...", rows.display())?;
        let args = crate::eval::EvalArgs {
            file: rows.to_path_buf(),
            model: vec![model.to_string(), incumbent.to_string()],
            judge: a.judge.clone(),
            out: Some(file.clone()),
            seed: 0,
            max_tokens: None,
            system: None,
            concurrency: None,
            retries: None,
        };
        crate::eval::run(paths, args, out).await.with_context(|| format!("{version} was added at weight 0; the eval did not run"))?;
        Ok(file)
    }
}

/// The model of the version with the most traffic, other than `n`.
async fn incumbent(paths: &Paths, name: &str, n: u32, inference: Option<&str>) -> Result<Option<String>> {
    let all = rollouts(paths, inference).await?;
    Ok(all
        .iter()
        .find(|r| r.name == name)
        .and_then(|r| r.versions.iter().filter(|v| v.version != n && v.weight > 0).max_by_key(|v| v.weight))
        .map(|v| v.model.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn the_next_version_follows_node_toml_and_ids_carry_it() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());
        std::fs::create_dir_all(paths.config().parent().unwrap()).unwrap();
        std::fs::write(paths.config(), "[node]\nname = \"n\"\n").unwrap();
        assert_eq!(next_version(&paths, "support", None).await.unwrap(), 1, "a new rollout starts at 1");
        let text = "[node]\nname = \"n\"\n\n[[inference.rollout]]\nname = \"support\"\n\n[[inference.rollout.version]]\nversion = 1\nmodel = \"a\"\nweight = 90\n\n[[inference.rollout.version]]\nversion = 4\nmodel = \"b\"\nweight = 10\n";
        std::fs::write(paths.config(), text).unwrap();
        assert_eq!(next_version(&paths, "support", None).await.unwrap(), 5, "after the highest, gaps kept");
        assert_eq!(next_version(&paths, "other", None).await.unwrap(), 1);
        assert_eq!(incumbent(&paths, "support", 5, None).await.unwrap().as_deref(), Some("a"));
        assert_eq!(incumbent(&paths, "other", 1, None).await.unwrap(), None);
        assert_eq!(versioned_id("llama-8b+support", 5), "llama-8b+support-r5");
    }
}
