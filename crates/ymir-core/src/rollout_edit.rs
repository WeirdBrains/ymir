//! Changing versioned rollouts (training plan P4b), shared by `ymirctl models rollout` on this
//! machine and `/ymir/rollouts/1` ([`crate::rollouts_net`]) from the owner's other devices.
//!
//! Raising a version above [`CANARY_MAX`] percent of the traffic needs an eval (the scored rows of
//! `ymirctl eval --out` with both models on the same rows) in which it does at least as well as the
//! version with the most traffic, or `force`. Lowering a weight (rollback) is never gated.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use toml_edit::DocumentMut;

use crate::rollout::{Rollout, Version};

/// The largest share (percent) a version may get without an eval.
pub const CANARY_MAX: f64 = 10.0;

/// One (model, row) result of `ymirctl eval`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scored {
    pub model: String,
    pub id: String,
    pub ok: bool,
    pub answer: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub check: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_reply: Option<String>,
    #[serde(default)]
    pub cache_hit: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub standin: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// A change to the rollouts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Change {
    /// Add `name@version` (and the rollout, if new) served by `model`.
    Add { name: String, version: u32, model: String, weight: u32 },
    /// Set a version's traffic weight.
    Weight { name: String, version: u32, weight: u32 },
    /// Move the newest version's traffic back to the version before it.
    Rollback { name: String },
    /// Remove a version (its model stays loaded until it is unloaded).
    Retire {
        name: String,
        version: u32,
        #[serde(default)]
        force: bool,
    },
}

/// What lets a version past the canary share.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Gate {
    /// Scored rows for this version's model and the current one's on the same rows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval: Option<Vec<Scored>>,
    /// Where the rows came from, for messages (a file name; default "the eval").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Skip the eval gate.
    #[serde(default)]
    pub force: bool,
}

/// One more than the highest version of `name` (1 for a new rollout).
pub fn next_version(rollouts: &[Rollout], name: &str) -> u32 {
    rollouts.iter().filter(|r| r.name == name).flat_map(|r| r.versions.iter().map(|v| v.version)).max().unwrap_or(0) + 1
}

/// Applies a change to `rollouts`; returns what to say.
pub fn apply(rollouts: &mut Vec<Rollout>, change: &Change, gate: &Gate) -> Result<String> {
    match change {
        Change::Add { name, version, model, weight } => {
            let (name, n) = (name.as_str(), *version);
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
        Change::Weight { name, version, weight } => {
            let (name, n) = (name.as_str(), *version);
            let r = find(rollouts, name)?;
            if r.version(n).is_none() {
                bail!("rollout `{name}` has no version {n}");
            }
            gated(r, n, *weight, gate)?;
            r.versions.iter_mut().find(|v| v.version == n).expect("checked").weight = *weight;
            Ok(format!("{name}@{n} now has weight {weight} ({:.1}% of new conversations).", share(r, n)))
        }
        Change::Rollback { name } => {
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
        Change::Retire { name, version, force } => {
            let (name, n) = (name.as_str(), *version);
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
    let Some(rows) = &gate.eval else {
        bail!(
            "{}@{n} would get {new_share:.1}% of new conversations, more than the {CANARY_MAX}% canary share: pass --eval FILE (`ymirctl eval ROWS --model {model} --model {} --out FILE`), or --force",
            r.name,
            incumbent.model
        );
    };
    eval_gate(rows, gate.source.as_deref().unwrap_or("the eval"), model, &incumbent.model)
}

/// Passes when `candidate` scored at least as well as `incumbent` on the rows both answered,
/// for the deterministic checks and (when both were judged) the judge. `source` names the rows.
pub fn eval_gate(rows: &[Scored], source: &str, candidate: &str, incumbent: &str) -> Result<()> {
    let of =
        |m: &str| -> std::collections::BTreeMap<String, Scored> { rows.iter().filter(|s| s.model == m).map(|s| (s.id.clone(), s.clone())).collect() };
    let (c, inc) = (of(candidate), of(incumbent));
    if c.is_empty() || inc.is_empty() {
        bail!(
            "{} has no rows for {}; run `ymirctl eval` with both --model {candidate} and --model {incumbent}",
            source,
            if c.is_empty() { candidate } else { incumbent }
        );
    }
    let both: Vec<&String> = c.keys().filter(|k| inc.contains_key(*k)).collect();
    let rate = |m: &std::collections::BTreeMap<String, Scored>, f: fn(&Scored) -> Option<bool>| {
        let v: Vec<bool> = both.iter().filter_map(|k| f(&m[*k])).collect();
        (v.iter().filter(|b| **b).count(), v.len())
    };
    let mut compared = false;
    for (what, f) in [("checks", (|s: &Scored| s.check) as fn(&Scored) -> Option<bool>), ("judge", |s: &Scored| s.judge)] {
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
        bail!("{} has no checked or judged rows that both models answered", source);
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

    #[test]
    fn the_next_version_follows_the_highest() {
        let r = vec![Rollout { name: "s".into(), versions: vec![Version { version: 4, model: "m".into(), weight: 0 }] }];
        assert_eq!((next_version(&r, "s"), next_version(&r, "new")), (5, 1));
    }

    #[test]
    fn a_change_reads_as_json_with_its_op() {
        let c: Change = serde_json::from_str(r#"{"op":"retire","name":"s","version":2}"#).unwrap();
        assert_eq!(c, Change::Retire { name: "s".into(), version: 2, force: false });
    }
}
