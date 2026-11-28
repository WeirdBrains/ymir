//! `ymirctl models rollout`: versioned rollouts (training plan P4b). Changes are written to
//! node.toml (`[[inference.rollout]]`) and, when ymird runs here, applied to it at once.
//!
//! Raising a version above [`CANARY_MAX`] percent of the traffic needs an eval (`ymirctl eval
//! --out FILE` with both models on the same rows) in which it does at least as well as the
//! version with the most traffic, or `--force`. Lowering a weight (rollback) is never gated.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use weirdnet_backends::models::ModelsReply;
use ymir_core::Paths;
use ymir_core::config::NodeConfig;
use ymir_core::control::{self, ControlRequest};
use ymir_core::models::{ModelsOp, Outcome};
use ymir_core::rollout::{Rollout, Version, split_versioned};
use ymir_core::upload::Provenance;

pub use ymir_core::rollout_edit::{CANARY_MAX, write_rollouts};
use ymir_core::rollout_edit::{Change, Gate as CoreGate, Scored};

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
        #[arg(long, default_value_t = 0, conflicts_with = "canary")]
        weight: u32,
        /// Give it exactly this percentage (0 to 100) of new conversations; the other versions
        /// share the rest. Not capped at the canary share; --eval, when given, must pass.
        #[arg(long, value_name = "PCT", value_parser = clap::value_parser!(u32).range(0..=100))]
        canary: Option<u32>,
        #[command(flatten)]
        gate: Gate,
    },
    /// Set a version's traffic weight (`weight support@3 20`), or its share in percent
    /// (`weight support@3 --canary 20`).
    Weight {
        /// NAME@N
        version: String,
        /// The relative weight (above the canary share only with --eval or --force).
        #[arg(required_unless_present = "canary", conflicts_with = "canary")]
        weight: Option<u32>,
        /// Give it exactly this percentage (0 to 100) of new conversations; the other versions
        /// share the rest. Not capped at the canary share; --eval, when given, must pass.
        #[arg(long, value_name = "PCT", value_parser = clap::value_parser!(u32).range(0..=100))]
        canary: Option<u32>,
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
        // Suggest `add` only where rollouts are routed (or could be once ymird starts).
        let hint = !matches!(live, Live::NoInference);
        let (shown, note) = list_source(live, rollouts);
        let prov = if shown.is_empty() { Default::default() } else { provenance(paths).await };
        return list(out, &shown, note, hint, &prov);
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

/// What the local ymird says about rollouts.
#[derive(Debug)]
pub enum Live {
    /// The rollouts in force.
    Rollouts(Vec<Rollout>),
    /// ymird is running but routes no rollouts: no inference role here.
    NoInference,
    /// No ymird answered.
    NotRunning,
}

async fn live(paths: &Paths) -> Live {
    match control::request(&paths.socket(), &ControlRequest::Rollouts { set: None }).await {
        Ok(r) => match r.rollouts {
            Some(l) if r.ok => Live::Rollouts(l),
            // A running ymird without a coordinator (the only refusal of a plain read).
            _ => Live::NoInference,
        },
        Err(_) => Live::NotRunning,
    }
}

/// Which rollouts `list` shows (live or node.toml's) and the note under them.
pub fn list_source(live: Live, file: Vec<Rollout>) -> (Vec<Rollout>, Option<&'static str>) {
    match live {
        Live::Rollouts(l) if l != file => (l, Some("ymird's live rollouts differ from node.toml (shown: live)")),
        Live::Rollouts(l) => (l, None),
        Live::NoInference if file.is_empty() => {
            (file, Some("This node has no inference role, so no rollouts are routed here (they belong in an inference node's node.toml)."))
        }
        Live::NoInference => (
            file,
            Some(
                "This node has no inference role, so ymird routes none of these: shown from node.toml, used only if the node gets the inference role.",
            ),
        ),
        Live::NotRunning => (file, Some("ymird is not running here: shown from node.toml")),
    }
}

/// Where each served model came from, as the pool's model nodes record it (best effort: empty
/// when ymird is not running here or no node answers in time).
async fn provenance(paths: &Paths) -> HashMap<String, Provenance> {
    let socket = paths.socket();
    let ask = control::request_models(&socket, "all", ModelsOp::List, |_| {});
    let Ok(Ok(done)) = tokio::time::timeout(Duration::from_secs(10), ask).await else { return HashMap::new() };
    let mut by_id = HashMap::new();
    for r in done.nodes {
        let Outcome::Ok { reply } = r.outcome else { continue };
        let ModelsReply::List(entries) = *reply else { continue };
        for e in entries {
            if let Some(p) = e.provenance() {
                by_id.entry(e.served_as.clone().unwrap_or_else(|| e.id().to_string())).or_insert_with(|| p.clone());
            }
        }
    }
    by_id
}

/// One line on a version's model: origin, license, base and dataset hash.
pub fn provenance_line(p: &Provenance) -> String {
    let mut parts = vec![format!("origin {}", p.origin)];
    parts.push(format!("license {}", p.license.as_deref().unwrap_or("not given")));
    if let Some(b) = p.base.as_ref().and_then(|b| b.hub.clone().or(b.id.clone())) {
        parts.push(format!("base {b}"));
    }
    if let Some(d) = p.dataset.as_ref().and_then(|d| d.sha256.as_deref()) {
        parts.push(format!("dataset {}", d.get(..19).unwrap_or(d)));
    }
    parts.join(", ")
}

fn list(out: &mut dyn Write, rollouts: &[Rollout], note: Option<&str>, hint: bool, prov: &HashMap<String, Provenance>) -> Result<()> {
    if rollouts.is_empty() && hint {
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
            if let Some(p) = prov.get(&v.model) {
                writeln!(out, "        {}", provenance_line(p))?;
            }
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

/// `cmd` as changes, in order, and their gate (empty: a list). `add --canary PCT` is an add at
/// weight 0, then the share.
fn to_changes(cmd: &RolloutCmd) -> Result<Vec<(Change, CoreGate)>> {
    let share = |name: &str, n: u32, percent: u32| Change::Share { name: name.into(), version: n, percent };
    let changes = match cmd {
        RolloutCmd::List => return Ok(Vec::new()),
        RolloutCmd::Add { version, model, weight, canary, gate } => {
            let (name, n) = parse_version(version)?;
            let add = Change::Add { name: name.into(), version: n, model: model.clone(), weight: if canary.is_some() { 0 } else { *weight } };
            match canary {
                Some(p) => vec![(add, CoreGate::default()), (share(name, n, *p), gate.rows()?)],
                None => vec![(add, gate.rows()?)],
            }
        }
        RolloutCmd::Weight { version, weight, canary, gate } => {
            let (name, n) = parse_version(version)?;
            match (canary, weight) {
                (Some(p), _) => vec![(share(name, n, *p), gate.rows()?)],
                (None, Some(w)) => vec![(Change::Weight { name: name.into(), version: n, weight: *w }, gate.rows()?)],
                (None, None) => bail!("give the weight, or --canary PCT"),
            }
        }
        RolloutCmd::Rollback { name } => vec![(Change::Rollback { name: name.clone() }, CoreGate::default())],
        RolloutCmd::Retire { version, force } => {
            let (name, n) = parse_version(version)?;
            vec![(Change::Retire { name: name.into(), version: n, force: *force }, CoreGate::default())]
        }
    };
    Ok(changes)
}

/// Applies a change to `rollouts`; returns what to say (the rules are in
/// [`ymir_core::rollout_edit`], shared with `/ymir/rollouts/1`).
pub fn apply(rollouts: &mut Vec<Rollout>, cmd: &RolloutCmd) -> Result<String> {
    let mut said = Vec::new();
    for (change, gate) in to_changes(cmd)? {
        said.push(ymir_core::rollout_edit::apply(rollouts, &change, &gate)?);
    }
    Ok(said.join("\n"))
}

/// The rollouts live on the inference node `node`, through this machine's ymird and `/ymir/rollouts/1`.
pub async fn remote_list(paths: &Paths, node: &str) -> Result<Vec<Rollout>> {
    let req = ControlRequest::RemoteRollouts { node: node.to_string(), request: ymir_core::rollouts_net::Request::list() };
    let r = control::request(&paths.socket(), &req).await.context("asking this machine's ymird (is it running? `ymirctl service start`)")?;
    if !r.ok {
        bail!("{}", r.message.unwrap_or_default());
    }
    Ok(r.remote_rollouts.context("ymird answered without the rollouts")?.rollouts)
}

/// `ymirctl models rollout ... --inference NODE`: the same on another inference node, through
/// this machine's ymird and `/ymir/rollouts/1`. `persist`: that node saves it in its node.toml.
pub async fn run_remote(paths: &Paths, out: &mut dyn Write, node: &str, cmd: RolloutCmd, persist: bool) -> Result<()> {
    let changes = to_changes(&cmd)?;
    let requests: Vec<ymir_core::rollouts_net::Request> = if changes.is_empty() {
        vec![ymir_core::rollouts_net::Request::list()]
    } else {
        changes
            .into_iter()
            .map(|(change, gate)| ymir_core::rollouts_net::Request { gate, ..ymir_core::rollouts_net::Request::change(change) })
            .collect()
    };
    for mut request in requests {
        request.persist = persist;
        let listing = request.change.is_none();
        let req = ControlRequest::RemoteRollouts { node: node.to_string(), request };
        let r = control::request(&paths.socket(), &req).await.context("asking this machine's ymird (is it running? `ymirctl service start`)")?;
        if !r.ok {
            bail!("{}", r.message.unwrap_or_default());
        }
        let label = r.message.unwrap_or_else(|| node.to_string());
        let got = r.remote_rollouts.context("ymird answered without the rollouts")?;
        if listing {
            let prov = if got.rollouts.is_empty() { Default::default() } else { provenance(paths).await };
            let note = format!("on {label}, live");
            return list(out, &got.rollouts, Some(note.as_str()), true, &prov);
        }
        writeln!(out, "{}", got.said.unwrap_or_default())?;
        match (got.saved, got.note) {
            (true, _) => writeln!(out, "Applied on {label} and saved in its node.toml: new conversations follow now.")?,
            (false, Some(n)) => writeln!(out, "Applied on {label} (live only): {n}")?,
            (false, None) => writeln!(out, "Applied on {label}, live only (--no-persist): gone when its ymird restarts.")?,
        }
    }
    Ok(())
}

impl Gate {
    /// The gate with the eval file's rows read in.
    pub fn rows(&self) -> Result<CoreGate> {
        let eval = match &self.eval {
            Some(f) => Some(read_scored(f)?),
            None => None,
        };
        Ok(CoreGate { eval, source: self.eval.as_ref().map(|f| f.display().to_string()), force: self.force })
    }
}

/// The rows of a `ymirctl eval --out` file.
pub fn read_scored(file: &Path) -> Result<Vec<Scored>> {
    let text = std::fs::read_to_string(file).with_context(|| format!("reading {}", file.display()))?;
    let mut rows = Vec::new();
    for (i, l) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        rows.push(serde_json::from_str(l).with_context(|| format!("{} line {}: not a `ymirctl eval --out` row", file.display(), i + 1))?);
    }
    Ok(rows)
}

/// [`ymir_core::rollout_edit::eval_gate`] on a `ymirctl eval --out` file.
pub fn eval_gate(file: &Path, candidate: &str, incumbent: &str) -> Result<()> {
    ymir_core::rollout_edit::eval_gate(&read_scored(file)?, &file.display().to_string(), candidate, incumbent)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ro() -> Vec<Rollout> {
        vec![Rollout { name: "support".into(), versions: vec![Version { version: 2, model: "s2".into(), weight: 100 }] }]
    }

    fn add(v: &str, m: &str, w: u32, gate: Gate) -> RolloutCmd {
        RolloutCmd::Add { version: v.into(), model: m.into(), weight: w, canary: None, gate }
    }

    #[test]
    fn add_and_weight_take_a_named_percentage() {
        let mut r = ro();
        let cmd = RolloutCmd::Add { version: "support@3".into(), model: "s3".into(), weight: 0, canary: Some(20), gate: Gate::default() };
        let said = apply(&mut r, &cmd).unwrap();
        assert!(said.contains("Added support@3") && said.contains("support@3 now gets 20.0%"), "{said}");
        let w: Vec<u32> = r[0].versions.iter().map(|v| v.weight).collect();
        assert_eq!(w, vec![80, 20]);
        apply(&mut r, &RolloutCmd::Weight { version: "support@3".into(), weight: None, canary: Some(50), gate: Gate::default() }).unwrap();
        let w: Vec<u32> = r[0].versions.iter().map(|v| v.weight).collect();
        assert_eq!(w, vec![50, 50]);
        use clap::Parser as _;
        #[derive(clap::Parser)]
        struct T {
            #[command(subcommand)]
            cmd: RolloutCmd,
        }
        assert!(T::try_parse_from(["t", "weight", "support@3", "--canary", "101"]).is_err());
        assert!(T::try_parse_from(["t", "weight", "support@3", "5", "--canary", "20"]).is_err());
        assert!(T::try_parse_from(["t", "weight", "support@3"]).is_err());
        assert!(T::try_parse_from(["t", "add", "support@4", "--model", "m", "--canary", "100"]).is_ok());
    }

    #[test]
    fn list_says_why_it_shows_node_toml() {
        let say = |live, file: Vec<Rollout>, hint| {
            let (shown, note) = list_source(live, file);
            let mut out = Vec::new();
            list(&mut out, &shown, note, hint, &HashMap::new()).unwrap();
            String::from_utf8(out).unwrap()
        };
        let s = say(Live::NoInference, vec![], false);
        assert!(s.contains("no inference role, so no rollouts are routed here") && !s.contains("not running") && !s.contains("Add one"), "{s}");
        let s = say(Live::NoInference, ro(), false);
        assert!(s.contains("support@2") && s.contains("no inference role") && !s.contains("not running"), "{s}");
        let s = say(Live::NotRunning, vec![], true);
        assert!(s.contains("Add one") && s.contains("ymird is not running here"), "{s}");
        let s = say(Live::Rollouts(ro()), ro(), true);
        assert!(s.contains("support@2") && !s.contains("node.toml"), "{s}");
        let s = say(Live::Rollouts(vec![]), ro(), true);
        assert!(s.contains("differ from node.toml"), "{s}");
    }

    #[test]
    fn a_canary_is_free_more_needs_an_eval_and_rollback_moves_traffic_back() {
        let mut r = ro();
        apply(&mut r, &add("support@3", "s3", 5, Gate::default())).unwrap();
        let e =
            apply(&mut r, &RolloutCmd::Weight { version: "support@3".into(), weight: Some(100), canary: None, gate: Gate::default() }).unwrap_err();
        assert!(e.to_string().contains("canary share"), "{e}");
        apply(&mut r, &RolloutCmd::Weight { version: "support@3".into(), weight: Some(100), canary: None, gate: Gate { eval: None, force: true } })
            .unwrap();
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
        apply(
            &mut r,
            &RolloutCmd::Weight { version: "support@3".into(), weight: Some(100), canary: None, gate: Gate { eval: Some(f), force: false } },
        )
        .unwrap();
    }

    #[test]
    fn provenance_lines_name_origin_license_base_and_dataset() {
        let p: Provenance = serde_json::from_value(serde_json::json!({
            "format": "weirdnet-provenance/1", "kind": "adapter", "id": "b+t-r2", "files": [], "revision": "sha256:00",
            "origin": "train", "license": "apache-2.0", "base": {"id": "b", "hub": "org/base"},
            "dataset": {"sha256": "sha256:0123456789abcdef0123456789"}
        }))
        .unwrap();
        assert_eq!(provenance_line(&p), "origin train, license apache-2.0, base org/base, dataset sha256:0123456789ab");
        let mut list_out = Vec::new();
        let r = vec![Rollout { name: "support".into(), versions: vec![Version { version: 2, model: "b+t-r2".into(), weight: 0 }] }];
        list(&mut list_out, &r, None, true, &HashMap::from([("b+t-r2".to_string(), p)])).unwrap();
        let s = String::from_utf8(list_out).unwrap();
        assert!(s.contains("support@2") && s.contains("origin train, license apache-2.0"), "{s}");
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
