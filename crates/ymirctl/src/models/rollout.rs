//! `ymirctl models rollout`: versioned rollouts (training plan P4b). Changes are written to
//! node.toml (`[[inference.rollout]]`) and, when ymird runs here, applied to it at once.
//!
//! Raising a version above [`CANARY_MAX`] percent of the traffic needs an eval (`ymirctl eval
//! --out FILE` with both models on the same rows) in which it does at least as well as the
//! version with the most traffic, or `--force`. Lowering a weight (rollback) is never gated.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use toml_edit::DocumentMut;
use ymir_core::Paths;
use ymir_core::config::NodeConfig;
use ymir_core::control::{self, ControlRequest};
use ymir_core::rollout::{Rollout, Version, split_versioned};

/// The largest share (percent) a version may get without an eval.
pub const CANARY_MAX: f64 = 10.0;

#[derive(clap::Subcommand, Debug, Clone)]
pub enum RolloutCmd {
    /// The rollouts, their versions, weights and traffic shares (live when ymird runs here).
    List,
    /// Add a version (and the rollout, if new): `support@3 --model llama-8b+support-r3`.
    Add {
        /// NAME@N
        version: String,
        /// The model id in the pool that serves it.
        #[arg(long)]
        model: String,
        /// Its traffic weight (default 0: pinnable as NAME@N, no traffic yet).
        #[arg(long, default_value_t = 0)]
        weight: u32,
        #[command(flatten)]
        gate: Gate,
    },
    /// Set a version's traffic weight.
    Weight {
        /// NAME@N
        version: String,
        weight: u32,
        #[command(flatten)]
        gate: Gate,
    },
    /// Move the newest version's traffic back to the version before it.
    Rollback { name: String },
    /// Remove a version (its model stays loaded until you unload it).
    Retire {
        /// NAME@N
        version: String,
        /// Retire it even though it still has traffic.
        #[arg(long)]
        force: bool,
    },
}

#[derive(clap::Args, Debug, Clone, Default)]
pub struct Gate {
    /// A `ymirctl eval --out` file scoring this version's model and the current one's on the
    /// same rows (needed above the canary share).
    #[arg(long)]
    pub eval: Option<PathBuf>,
    /// Skip the eval gate.
    #[arg(long)]
    pub force: bool,
}

fn parse_version(s: &str) -> Result<(&str, u32)> {
    split_versioned(s).with_context(|| format!("`{s}`: name a version as NAME@N, e.g. support@3"))
}

pub async fn run(paths: &Paths, out: &mut dyn Write, cmd: RolloutCmd) -> Result<()> {
    let config = paths.config();
    let text = std::fs::read_to_string(&config).with_context(|| format!("reading {} (run `ymirctl setup` first)", config.display()))?;
    let cfg: NodeConfig = toml::from_str(&text).with_context(|| format!("parsing {}", config.display()))?;
    let mut rollouts = cfg.inference.rollout.clone();
    if let RolloutCmd::List = cmd {
        let live = live(paths).await;
        let (shown, note) = match live {
            Some(l) if l != rollouts => (l, Some("ymird's live rollouts differ from node.toml (shown: live)")),
            Some(l) => (l, None),
            None => (rollouts, Some("ymird is not running here: shown from node.toml")),
        };
        return list(out, &shown, note);
    }
    let said = apply(&mut rollouts, &cmd)?;
    ymir_core::rollout::validate(&rollouts)?;
    let new = write_rollouts(&text, &rollouts)?;
    toml::from_str::<NodeConfig>(&new).context("the edited node.toml does not parse")?.validate()?;
    crate::models::adapters_save(&config, &text, &new)?;
    writeln!(out, "{said}")?;
    match control::request(&paths.socket(), &ControlRequest::Rollouts { set: Some(rollouts) }).await {
        Ok(r) if r.ok => writeln!(out, "Applied to the running ymird: new conversations follow now.")?,
        Ok(r) => writeln!(out, "Saved to node.toml; ymird did not take it: {}", r.message.unwrap_or_default())?,
        Err(_) => writeln!(out, "Saved to node.toml; ymird is not running here, and uses it when it starts.")?,
    }
    Ok(())
}

async fn live(paths: &Paths) -> Option<Vec<Rollout>> {
    let r = control::request(&paths.socket(), &ControlRequest::Rollouts { set: None }).await.ok()?;
    r.rollouts
}

fn list(out: &mut dyn Write, rollouts: &[Rollout], note: Option<&str>) -> Result<()> {
    if rollouts.is_empty() {
        writeln!(out, "No rollouts. Add one with `ymirctl models rollout add NAME@1 --model ID --weight 100`.")?;
    }
    for r in rollouts {
        writeln!(out, "{}", r.name)?;
        let shares = r.shares();
        let mut vs: Vec<&Version> = r.versions.iter().collect();
        vs.sort_by_key(|v| v.version);
        for v in vs {
            let share = shares.iter().find(|(n, _)| *n == v.version).map_or(0.0, |s| s.1);
            writeln!(out, "  {}@{:<4} {:<40} weight {:<5} {:>5.1}%", r.name, v.version, v.model, v.weight, share)?;
        }
        if r.total_weight() == 0 {
            writeln!(out, "  (no version has traffic: requests for `{}` are refused)", r.name)?;
        }
    }
    if let Some(n) = note {
        writeln!(out, "{n}")?;
    }
    Ok(())
}

/// Applies a change to `rollouts`; returns what to say.
pub fn apply(rollouts: &mut Vec<Rollout>, cmd: &RolloutCmd) -> Result<String> {
    match cmd {
        RolloutCmd::List => Ok(String::new()),
        RolloutCmd::Add { version, model, weight, gate } => {
            let (name, n) = parse_version(version)?;
            if !rollouts.iter().any(|r| r.name == name) {
                rollouts.push(Rollout { name: name.into(), versions: Vec::new() });
            }
            let r = rollouts.iter_mut().find(|r| r.name == name).expect("just added");
            if r.version(n).is_some() {
                bail!("{name}@{n} exists; change its weight with `ymirctl models rollout weight {name}@{n} W`");
            }
            r.versions.push(Version { version: n, model: model.clone(), weight: 0 });
            gated(r, n, *weight, gate)?;
            r.versions.iter_mut().find(|v| v.version == n).expect("added").weight = *weight;
            Ok(format!("Added {name}@{n} (`{model}`) with weight {weight} ({:.1}% of new conversations).", share(r, n)))
        }
        RolloutCmd::Weight { version, weight, gate } => {
            let (name, n) = parse_version(version)?;
            let r = find(rollouts, name)?;
            if r.version(n).is_none() {
                bail!("rollout `{name}` has no version {n}");
            }
            gated(r, n, *weight, gate)?;
            r.versions.iter_mut().find(|v| v.version == n).expect("checked").weight = *weight;
            Ok(format!("{name}@{n} now has weight {weight} ({:.1}% of new conversations).", share(r, n)))
        }
        RolloutCmd::Rollback { name } => {
            let r = find(rollouts, name)?;
            let newest =
                r.versions.iter().filter(|v| v.weight > 0).map(|v| v.version).max().with_context(|| format!("no version of `{name}` has traffic"))?;
            let prev = r
                .versions
                .iter()
                .filter(|v| v.version < newest)
                .map(|v| v.version)
                .max()
                .with_context(|| format!("{name}@{newest} is the oldest version; there is nothing to roll back to"))?;
            let moved = r.version(newest).expect("found").weight;
            for v in &mut r.versions {
                if v.version == newest {
                    v.weight = 0;
                } else if v.version == prev {
                    v.weight += moved;
                }
            }
            Ok(format!(
                "Rolled back `{name}`: {name}@{newest} has no traffic now and {name}@{prev} has {:.1}%. {name}@{newest} stays pinnable until you retire it.",
                share(r, prev)
            ))
        }
        RolloutCmd::Retire { version, force } => {
            let (name, n) = parse_version(version)?;
            let r = find(rollouts, name)?;
            let v = r.version(n).with_context(|| format!("rollout `{name}` has no version {n}"))?.clone();
            if v.weight > 0 && !force {
                bail!("{name}@{n} still has traffic (weight {}); set it to 0 or roll back first, or pass --force", v.weight);
            }
            r.versions.retain(|x| x.version != n);
            let gone = r.versions.is_empty();
            if gone {
                rollouts.retain(|r| r.name != name);
            }
            Ok(format!(
                "Retired {name}@{n}{}. Its model `{}` is still loaded where it was: `ymirctl unload {}` frees it.",
                if gone { format!(" and the rollout `{name}` (it had no other version)") } else { String::new() },
                v.model,
                v.model
            ))
        }
    }
}

fn find<'a>(rollouts: &'a mut [Rollout], name: &str) -> Result<&'a mut Rollout> {
    rollouts.iter_mut().find(|r| r.name == name).with_context(|| format!("no rollout `{name}` (see `ymirctl models rollout list`)"))
}

fn share(r: &Rollout, n: u32) -> f64 {
    r.shares().into_iter().find(|(v, _)| *v == n).map_or(0.0, |s| s.1)
}

/// The eval gate: refuses `weight` for version `n` when it would take more than the canary
/// share from a version that has traffic, unless an eval shows it does as well, or `--force`.
fn gated(r: &Rollout, n: u32, weight: u32, gate: &Gate) -> Result<()> {
    let mut after = r.clone();
    after.versions.iter_mut().find(|v| v.version == n).expect("present").weight = weight;
    let new_share = share(&after, n);
    let Some(incumbent) = r.versions.iter().filter(|v| v.version != n && v.weight > 0).max_by_key(|v| v.weight) else {
        return Ok(()); // nothing serves the name yet: no traffic to protect
    };
    if new_share <= CANARY_MAX || gate.force {
        return Ok(());
    }
    let model = &r.version(n).expect("present").model;
    let Some(file) = &gate.eval else {
        bail!(
            "{}@{n} would get {new_share:.1}% of new conversations, more than the {CANARY_MAX}% canary share: pass --eval FILE (`ymirctl eval ROWS --model {model} --model {} --out FILE`), or --force",
            r.name,
            incumbent.model
        );
    };
    eval_gate(file, model, &incumbent.model)
}

/// Passes when `candidate` scored at least as well as `incumbent` on the rows both answered,
/// for the deterministic checks and (when both were judged) the judge.
pub fn eval_gate(file: &Path, candidate: &str, incumbent: &str) -> Result<()> {
    let text = std::fs::read_to_string(file).with_context(|| format!("reading {}", file.display()))?;
    let mut rows: Vec<crate::eval::Scored> = Vec::new();
    for (i, l) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        rows.push(serde_json::from_str(l).with_context(|| format!("{} line {}: not a `ymirctl eval --out` row", file.display(), i + 1))?);
    }
    let of = |m: &str| -> std::collections::BTreeMap<String, crate::eval::Scored> {
        rows.iter().filter(|s| s.model == m).map(|s| (s.id.clone(), s.clone())).collect()
    };
    let (c, inc) = (of(candidate), of(incumbent));
    if c.is_empty() || inc.is_empty() {
        bail!(
            "{} has no rows for {}; run `ymirctl eval` with both --model {candidate} and --model {incumbent}",
            file.display(),
            if c.is_empty() { candidate } else { incumbent }
        );
    }
    let both: Vec<&String> = c.keys().filter(|k| inc.contains_key(*k)).collect();
    let rate = |m: &std::collections::BTreeMap<String, crate::eval::Scored>, f: fn(&crate::eval::Scored) -> Option<bool>| {
        let v: Vec<bool> = both.iter().filter_map(|k| f(&m[*k])).collect();
        (v.iter().filter(|b| **b).count(), v.len())
    };
    let mut compared = false;
    for (what, f) in
        [("checks", (|s: &crate::eval::Scored| s.check) as fn(&crate::eval::Scored) -> Option<bool>), ("judge", |s: &crate::eval::Scored| s.judge)]
    {
        let ((cp, cn), (ip, inn)) = (rate(&c, f), rate(&inc, f));
        if cn == 0 || inn == 0 {
            continue;
        }
        compared = true;
        if (cp as f64 / cn as f64) < (ip as f64 / inn as f64) {
            bail!(
                "the eval gate failed on {what}: {candidate} passed {cp}/{cn}, {incumbent} passed {ip}/{inn}; keep it at the canary share or retrain"
            );
        }
    }
    if !compared {
        bail!("{} has no checked or judged rows that both models answered", file.display());
    }
    Ok(())
}

/// node.toml with `[[inference.rollout]]` replaced by `rollouts` (other content kept).
pub fn write_rollouts(text: &str, rollouts: &[Rollout]) -> Result<String> {
    #[derive(serde::Serialize)]
    struct W<'a> {
        rollout: &'a [Rollout],
    }
    let mut doc: DocumentMut = text.parse().context("node.toml")?;
    let fresh: DocumentMut = toml_edit::ser::to_document(&W { rollout: rollouts })?;
    if doc.get("inference").is_none() {
        doc["inference"] = toml_edit::table();
    }
    let inf = doc["inference"].as_table_mut().context("[inference] is not a table")?;
    inf.remove("rollout");
    if !rollouts.is_empty()
        && let Some(item) = fresh.get("rollout")
    {
        let mut item = item.clone();
        if let Some(a) = item.as_array_mut() {
            // to_document writes an inline array; node.toml uses [[inference.rollout]] tables.
            let mut tables = toml_edit::ArrayOfTables::new();
            for v in a.iter() {
                let t: toml_edit::Table = v.as_inline_table().context("rollout")?.clone().into_table();
                tables.push(t);
            }
            item = toml_edit::Item::ArrayOfTables(tables);
        }
        inf.insert("rollout", item);
    }
    let mut out = doc.to_string();
    // Nested version tables come out inline; make them [[inference.rollout.version]] tables.
    let mut doc2: DocumentMut = out.parse()?;
    if let Some(aot) = doc2["inference"].get_mut("rollout").and_then(|i| i.as_array_of_tables_mut()) {
        for t in aot.iter_mut() {
            if let Some(arr) = t.get("version").and_then(|v| v.as_array()).cloned() {
                let mut vt = toml_edit::ArrayOfTables::new();
                for v in arr.iter() {
                    vt.push(v.as_inline_table().context("version")?.clone().into_table());
                }
                t.insert("version", toml_edit::Item::ArrayOfTables(vt));
            }
        }
    }
    out = doc2.to_string();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ro() -> Vec<Rollout> {
        vec![Rollout { name: "support".into(), versions: vec![Version { version: 2, model: "s2".into(), weight: 100 }] }]
    }

    fn add(v: &str, m: &str, w: u32, gate: Gate) -> RolloutCmd {
        RolloutCmd::Add { version: v.into(), model: m.into(), weight: w, gate }
    }

    #[test]
    fn a_canary_is_free_more_needs_an_eval_and_rollback_moves_traffic_back() {
        let mut r = ro();
        apply(&mut r, &add("support@3", "s3", 5, Gate::default())).unwrap();
        let e = apply(&mut r, &RolloutCmd::Weight { version: "support@3".into(), weight: 100, gate: Gate::default() }).unwrap_err();
        assert!(e.to_string().contains("canary share"), "{e}");
        apply(&mut r, &RolloutCmd::Weight { version: "support@3".into(), weight: 100, gate: Gate { eval: None, force: true } }).unwrap();
        assert_eq!(r[0].version(3).unwrap().weight, 100);
        apply(&mut r, &RolloutCmd::Rollback { name: "support".into() }).unwrap();
        assert_eq!((r[0].version(2).unwrap().weight, r[0].version(3).unwrap().weight), (200, 0));
        assert!(apply(&mut r, &RolloutCmd::Retire { version: "support@2".into(), force: false }).is_err(), "has traffic");
        apply(&mut r, &RolloutCmd::Retire { version: "support@3".into(), force: false }).unwrap();
        assert_eq!(r[0].versions.len(), 1);
        assert!(apply(&mut r, &RolloutCmd::Rollback { name: "support".into() }).is_err(), "nothing older");
        // A first version needs no eval: nothing else has traffic.
        let mut fresh = Vec::new();
        apply(&mut fresh, &add("new@1", "n1", 100, Gate::default())).unwrap();
    }

    #[test]
    fn the_eval_gate_compares_the_same_rows() {
        let dir = tempfile::tempdir().unwrap();
        let row = |m: &str, id: &str, check: bool| {
            serde_json::json!({"model": m, "id": id, "ok": true, "answer": "", "check": check, "cache_hit": false}).to_string()
        };
        let f = dir.path().join("e.jsonl");
        std::fs::write(&f, [row("s3", "a", true), row("s3", "b", true), row("s2", "a", true), row("s2", "b", false)].join("\n")).unwrap();
        eval_gate(&f, "s3", "s2").unwrap();
        let e = eval_gate(&f, "s2", "s3").unwrap_err();
        assert!(e.to_string().contains("1/2"), "{e}");
        assert!(eval_gate(&f, "s9", "s2").is_err());
        let mut r = ro();
        apply(&mut r, &add("support@3", "s3", 0, Gate::default())).unwrap();
        apply(&mut r, &RolloutCmd::Weight { version: "support@3".into(), weight: 100, gate: Gate { eval: Some(f), force: false } }).unwrap();
    }

    #[test]
    fn node_toml_keeps_its_other_content() {
        let text = "# my node\n[node]\nname = \"n\" # keep\n\n[inference]\nstandin = \"tiny\"\n";
        let mut r = ro();
        r[0].versions.push(Version { version: 3, model: "s3".into(), weight: 5 });
        let new = write_rollouts(text, &r).unwrap();
        assert!(new.contains("# my node") && new.contains("name = \"n\" # keep") && new.contains("standin = \"tiny\""), "{new}");
        assert!(new.contains("[[inference.rollout]]") && new.contains("[[inference.rollout.version]]"), "{new}");
        let back: NodeConfig = toml::from_str(&new).unwrap();
        assert_eq!(back.inference.rollout, r);
        let gone = write_rollouts(&new, &[]).unwrap();
        assert!(!gone.contains("rollout"), "{gone}");
        assert!(toml::from_str::<NodeConfig>(&gone).unwrap().inference.rollout.is_empty());
    }
}
