//! Changing versioned rollouts (training plan P4b), shared by `ymirctl models rollout` on this
//! machine and `/ymir/rollouts/1` ([`crate::rollouts_net`]) from the owner's other devices.
//!
//! Raising a version above [`CANARY_MAX`] percent of the traffic needs an eval (the scored rows of
//! `ymirctl eval --out` with both models on the same rows) in which it does at least as well as the
//! version with the most traffic, or `force`. Lowering a weight (rollback) is never gated.
//!
//! A share named outright ([`Change::Share`], `--canary PCT`) is not capped: any percentage from 0
//! to 100 is set as asked, and an eval is checked only when one is given (m3talsmith, 2026-11-25).

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
    /// Give a version exactly `percent` (0 to 100) of new conversations: its weight becomes
    /// `percent` and the other versions with traffic share the rest in their present proportions.
    /// Not capped by the canary share; an eval in the gate, when given, must pass.
    Share { name: String, version: u32, percent: u32 },
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
        Change::Share { name, version, percent } => {
            let (name, n, p) = (name.as_str(), *version, *percent);
            if p > 100 {
                bail!("{p}% is not a share: give a percentage from 0 to 100");
            }
            let r = find(rollouts, name)?;
            if r.version(n).is_none() {
                bail!("rollout `{name}` has no version {n}");
            }
            if gate.eval.is_some()
                && !gate.force
                && let Some(incumbent) = r.versions.iter().filter(|v| v.version != n && v.weight > 0).max_by_key(|v| v.weight)
            {
                let model = &r.version(n).expect("checked").model;
                eval_gate(gate.eval.as_deref().unwrap_or_default(), gate.source.as_deref().unwrap_or("the eval"), model, &incumbent.model)?;
            }
            set_share(r, n, p);
            let mut said = format!("{name}@{n} now gets {:.1}% of new conversations", share(r, n));
            let others: Vec<String> = r
                .versions
                .iter()
                .filter(|v| v.version != n && v.weight > 0)
                .map(|v| format!("{name}@{} {:.1}%", v.version, share(r, v.version)))
                .collect();
            if others.is_empty() && p > 0 && p < 100 {
                said.push_str(&format!(" (no other version has traffic, so {p}% is all of it)"));
            } else if !others.is_empty() {
                said.push_str(&format!("; {}", others.join(", ")));
            }
            said.push('.');
            Ok(said)
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

/// Weights summing to 100: `p` for version `n`, the rest split over the other versions with
/// traffic by their present weights (largest remainder, so the shares add up exactly).
fn set_share(r: &mut Rollout, n: u32, p: u32) {
    let rest = u64::from(100 - p);
    let total: u64 = r.versions.iter().filter(|v| v.version != n).map(|v| u64::from(v.weight)).sum();
    let mut parts: Vec<(u32, u64, u64)> = r
        .versions
        .iter()
        .filter(|v| v.version != n && v.weight > 0)
        .map(|v| {
            let x = rest * u64::from(v.weight);
            (v.version, x / total, x % total)
        })
        .collect();
    let mut left = rest - parts.iter().map(|x| x.1).sum::<u64>();
    let mut order: Vec<usize> = (0..parts.len()).collect();
    order.sort_by(|a, b| parts[*b].2.cmp(&parts[*a].2).then(parts[*a].0.cmp(&parts[*b].0)));
    for i in order {
        if left == 0 {
            break;
        }
        parts[i].1 += 1;
        left -= 1;
    }
    for v in &mut r.versions {
        if v.version == n {
            v.weight = p;
        } else if let Some(x) = parts.iter().find(|x| x.0 == v.version) {
            v.weight = x.1 as u32;
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

    fn ro(ws: &[(u32, u32)]) -> Vec<Rollout> {
        vec![Rollout { name: "s".into(), versions: ws.iter().map(|(n, w)| Version { version: *n, model: format!("m{n}"), weight: *w }).collect() }]
    }

    fn weights(r: &[Rollout]) -> Vec<(u32, u32)> {
        r[0].versions.iter().map(|v| (v.version, v.weight)).collect()
    }

    #[test]
    fn a_named_share_is_set_exactly_without_the_canary_cap() {
        let share = |n, p| Change::Share { name: "s".into(), version: n, percent: p };
        let mut r = ro(&[(1, 100), (2, 0)]);
        let said = apply(&mut r, &share(2, 20), &Gate::default()).unwrap();
        assert_eq!(weights(&r), vec![(1, 80), (2, 20)]);
        assert!(said.contains("s@2 now gets 20.0%") && said.contains("s@1 80.0%"), "{said}");
        apply(&mut r, &share(2, 50), &Gate::default()).unwrap();
        assert_eq!(weights(&r), vec![(1, 50), (2, 50)]);
        // Three versions: the rest keeps their proportions and adds up to 100.
        let mut r = ro(&[(1, 2), (2, 1), (3, 0)]);
        apply(&mut r, &share(3, 1), &Gate::default()).unwrap();
        assert_eq!(weights(&r), vec![(1, 66), (2, 33), (3, 1)]);
        apply(&mut r, &share(3, 100), &Gate::default()).unwrap();
        assert_eq!(weights(&r), vec![(1, 0), (2, 0), (3, 100)]);
        apply(&mut r, &share(3, 0), &Gate::default()).unwrap();
        assert_eq!(weights(&r), vec![(1, 0), (2, 0), (3, 0)]);
        assert!(apply(&mut r, &share(3, 101), &Gate::default()).unwrap_err().to_string().contains("0 to 100"));
        assert!(apply(&mut r, &share(9, 5), &Gate::default()).is_err());
        // Alone, any share is all of it.
        let mut r = ro(&[(1, 0)]);
        assert!(apply(&mut r, &share(1, 20), &Gate::default()).unwrap().contains("is all of it"));
        // A weight above the canary share still needs an eval.
        let mut r = ro(&[(1, 100), (2, 0)]);
        assert!(apply(&mut r, &Change::Weight { name: "s".into(), version: 2, weight: 50 }, &Gate::default()).is_err());
    }

    #[test]
    fn a_named_share_checks_an_eval_when_one_is_given() {
        let row = |m: &str, id: &str, ok: bool| Scored {
            model: m.into(),
            id: id.into(),
            ok: true,
            answer: String::new(),
            check: Some(ok),
            judge: None,
            judge_reply: None,
            cache_hit: false,
            standin: None,
            error: None,
        };
        let worse = Gate { eval: Some(vec![row("m2", "a", false), row("m1", "a", true)]), source: None, force: false };
        let mut r = ro(&[(1, 100), (2, 0)]);
        let share = Change::Share { name: "s".into(), version: 2, percent: 20 };
        assert!(apply(&mut r, &share, &worse).unwrap_err().to_string().contains("eval gate failed"));
        assert_eq!(weights(&r), vec![(1, 100), (2, 0)]);
        apply(&mut r, &share, &Gate { force: true, ..worse }).unwrap();
        assert_eq!(weights(&r), vec![(1, 80), (2, 20)]);
        let c: Change = serde_json::from_str(r#"{"op":"share","name":"s","version":2,"percent":20}"#).unwrap();
        assert_eq!(c, share);
    }

    #[test]
    fn a_change_reads_as_json_with_its_op() {
        let c: Change = serde_json::from_str(r#"{"op":"retire","name":"s","version":2}"#).unwrap();
        assert_eq!(c, Change::Retire { name: "s".into(), version: 2, force: false });
    }
}
