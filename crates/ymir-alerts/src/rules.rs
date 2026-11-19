//! The default rule pack: what server-a/server-b's ts-alert scripts checked, as `weirdnet-rules` specs.

use serde::{Deserialize, Serialize};
use weirdnet_rules::{Condition, Op, RuleSpec};

use crate::{Action, AlertRule};

/// Limits for the built-in rules.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Thresholds {
    /// Alert when a watched filesystem has less free space than this, in percent.
    #[serde(default = "disk_min_free_pct")]
    pub disk_min_free_pct: f64,
    /// Alert at this GPU temperature; clear 5 degrees below.
    #[serde(default = "gpu_max_c")]
    pub gpu_max_c: f64,
}

fn disk_min_free_pct() -> f64 {
    10.0
}

fn gpu_max_c() -> f64 {
    85.0
}

impl Default for Thresholds {
    fn default() -> Self {
        Self { disk_min_free_pct: disk_min_free_pct(), gpu_max_c: gpu_max_c() }
    }
}

fn cond(field: &str, op: Op, value: impl ToString) -> Condition {
    Condition { field: field.into(), op, value: value.to_string() }
}

fn rule(on: &str, when: Vec<Condition>, then: Action) -> AlertRule {
    RuleSpec { on: vec![on.into()], when, then: vec![then], ..Default::default() }
}

fn fail(key: &str, title: &str) -> Action {
    Action::Fail { key: key.into(), title: title.into(), repeat_secs: None }
}

fn ok(key: &str, title: &str) -> Action {
    Action::Ok { key: key.into(), title: title.into() }
}

/// Engine health (from the supervisor), the network-degraded rule, disk and GPU.
pub fn default_rules(t: &Thresholds) -> Vec<AlertRule> {
    vec![
        // Engines. One key per model: a crash loop or a failed load raises it, ready clears it.
        rule(
            "engine.crash_loop",
            vec![],
            fail("engine.{model}", "{model} on {host} is in a crash loop ({crashes} crashes); restarts stopped until `ymirctl load {model}`"),
        ),
        rule("engine.load_failed", vec![], fail("engine.{model}", "{model} on {host} failed to load: {why}")),
        rule("engine.ready", vec![], ok("engine.{model}", "{model} on {host} is serving again")),
        rule(
            "engine.crashed",
            vec![],
            Action::Event {
                key: "engine.{model}.crash".into(),
                title: "{model} on {host} crashed ({why}); restarting in {restart_in_secs}s".into(),
                repeat_secs: None,
            },
        ),
        // The server-a/server-b route-state rule: the default route must not be on Wi-Fi.
        rule("net.route", vec![cond("state", Op::Eq, "degraded")], fail("net.route", "DEGRADED but up: {host} default route uses Wi-Fi ({dev})")),
        rule("net.route", vec![cond("state", Op::Eq, "noroute")], fail("net.route", "{host} has no default route")),
        rule("net.route", vec![cond("state", Op::Eq, "ok")], ok("net.route", "{host} default route back on {dev}")),
        rule("disk", vec![cond("free_pct", Op::Lt, t.disk_min_free_pct)], fail("disk.{path}", "{host}: {path} has {free_pct}% free")),
        rule(
            "disk",
            vec![cond("free_pct", Op::Ge, t.disk_min_free_pct + 1.0)],
            ok("disk.{path}", "{host}: {path} has room again ({free_pct}% free)"),
        ),
        rule("gpu", vec![cond("temp_c", Op::Ge, t.gpu_max_c)], fail("gpu.{index}", "{host}: GPU {index} is at {temp_c} C")),
        rule("gpu", vec![cond("temp_c", Op::Lt, t.gpu_max_c - 5.0)], ok("gpu.{index}", "{host}: GPU {index} cooled to {temp_c} C")),
    ]
}
