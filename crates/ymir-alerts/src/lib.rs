//! Node health alerts: rule packs over observations, dedupe with recovery notices, and sinks.
//!
//! To be extracted into `weirdnet-alerts`. Rules are [`weirdnet_rules::RuleSpec`]s; only the
//! spec, matching and comparison parts of `weirdnet-rules` are used, so the alerter is `Send` and
//! can live in an async daemon.

pub mod config;
pub mod probe;
pub mod rules;
pub mod sinks;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use weirdnet_rules::{RuleSpec, ThenAction, compare, syntax};

pub use config::AlertsConfig;
pub use sinks::{LogSink, MemorySink, PushSink, Sink, SmtpSink, WebhookSink};

/// Something a node saw: `kind` (`engine.crash_loop`, `net.route`, `disk`) and its fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    pub kind: String,
    #[serde(default)]
    pub fields: Value,
}

impl Observation {
    pub fn new(kind: impl Into<String>, fields: Value) -> Self {
        Self { kind: kind.into(), fields }
    }
}

/// What a matching rule does. `key` and `title` are templates: `{field}` from the observation
/// (`{host}` is always there), plus `{now}` and `{date}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "do", rename_all = "snake_case")]
pub enum Action {
    /// Raise (or keep raising) the alert `key`.
    Fail {
        key: String,
        title: String,
        /// Repeat at most this often while it keeps failing; default the alerter's repeat.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        repeat_secs: Option<u64>,
    },
    /// Clear `key`, with a recovery notice if it was raised.
    Ok { key: String, title: String },
    /// A one-off notice, deduped like a failure.
    Event {
        key: String,
        title: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        repeat_secs: Option<u64>,
    },
}

impl ThenAction for Action {
    fn verb(&self) -> &'static str {
        match self {
            Action::Fail { .. } => "fail",
            Action::Ok { .. } => "ok",
            Action::Event { .. } => "event",
        }
    }
}

pub type AlertRule = RuleSpec<Action>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlertStatus {
    Failing,
    Recovered,
    Event,
}

/// One message to send.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Alert {
    pub key: String,
    pub status: AlertStatus,
    pub host: String,
    pub title: String,
    /// The observation that raised it.
    pub observation: Observation,
    /// Local time, `2026-11-19 14:05`.
    pub at: String,
}

impl Alert {
    /// The subject line: `[ymir server-a] DOWN ...`, like ts-alert.
    pub fn subject(&self, product: &str) -> String {
        let tag = match self.status {
            AlertStatus::Failing => "ALERT",
            AlertStatus::Recovered => "OK",
            AlertStatus::Event => "NOTE",
        };
        format!("[{product} {}] {tag}: {}", self.host, self.title)
    }

    pub fn body(&self) -> String {
        let fields = serde_json::to_string_pretty(&self.observation.fields).unwrap_or_default();
        format!("{}\n\nHost: {}\nWhen: {}\nKey: {}\nObserved: {}\n{}\n", self.title, self.host, self.at, self.key, self.observation.kind, fields)
    }
}

#[derive(Debug, Default)]
struct KeyState {
    failing: bool,
    last_sent: Option<Instant>,
}

/// Matches observations against rules, dedupes, and hands alerts to the sinks.
pub struct Alerter {
    host: String,
    product: String,
    rules: Vec<AlertRule>,
    sinks: Vec<Arc<dyn Sink>>,
    repeat: Duration,
    state: Mutex<HashMap<String, KeyState>>,
}

impl Alerter {
    pub fn new(host: impl Into<String>, rules: Vec<AlertRule>, sinks: Vec<Arc<dyn Sink>>) -> Self {
        Self { host: host.into(), product: "ymir".into(), rules, sinks, repeat: Duration::from_secs(3600), state: Mutex::new(HashMap::new()) }
    }

    /// How often a key that keeps failing alerts again (default an hour).
    pub fn with_repeat(mut self, repeat: Duration) -> Self {
        self.repeat = repeat;
        self
    }

    /// The name in subject lines (default `ymir`).
    pub fn with_product(mut self, product: impl Into<String>) -> Self {
        self.product = product.into();
        self
    }

    pub fn product(&self) -> &str {
        &self.product
    }

    /// Keys currently failing.
    pub fn failing(&self) -> Vec<String> {
        let mut v: Vec<String> =
            self.state.lock().unwrap_or_else(|e| e.into_inner()).iter().filter(|(_, s)| s.failing).map(|(k, _)| k.clone()).collect();
        v.sort();
        v
    }

    /// The alerts `obs` raises now, after dedupe. Does not send.
    pub fn evaluate(&self, obs: &Observation, now: Instant) -> Vec<Alert> {
        let mut fields = obs.fields.clone();
        if let Value::Object(m) = &mut fields {
            m.entry("host").or_insert_with(|| Value::String(self.host.clone()));
        } else {
            fields = serde_json::json!({"host": self.host, "value": fields});
        }
        let local = chrono::Local::now();
        let mut out = Vec::new();
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        for rule in &self.rules {
            if !rule.on.iter().any(|p| syntax::matches_kind(p, &obs.kind)) {
                continue;
            }
            if !rule.when.iter().all(|c| compare::compare(compare::path(&fields, &c.field), c.op, &c.value)) {
                continue;
            }
            for action in &rule.then {
                let fill = |t: &str| weirdnet_rules::template::expand(t, &local, &obs.kind, |k| compare::path(&fields, k).cloned());
                let (key, title, status, repeat) = match action {
                    Action::Fail { key, title, repeat_secs } => {
                        (fill(key), fill(title), AlertStatus::Failing, repeat_secs.map_or(self.repeat, Duration::from_secs))
                    }
                    Action::Event { key, title, repeat_secs } => {
                        (fill(key), fill(title), AlertStatus::Event, repeat_secs.map_or(self.repeat, Duration::from_secs))
                    }
                    Action::Ok { key, title } => (fill(key), fill(title), AlertStatus::Recovered, Duration::ZERO),
                };
                let s = state.entry(key.clone()).or_default();
                let send = match status {
                    AlertStatus::Failing => {
                        let due = !s.failing || s.last_sent.is_none_or(|t| now.duration_since(t) >= repeat);
                        s.failing = true;
                        due
                    }
                    AlertStatus::Event => s.last_sent.is_none_or(|t| now.duration_since(t) >= repeat),
                    AlertStatus::Recovered => std::mem::take(&mut s.failing),
                };
                if send {
                    s.last_sent = Some(now);
                    out.push(Alert {
                        key,
                        status,
                        host: self.host.clone(),
                        title,
                        observation: Observation { kind: obs.kind.clone(), fields: fields.clone() },
                        at: local.format("%Y-%m-%d %H:%M %Z").to_string(),
                    });
                }
            }
        }
        out
    }

    /// Evaluates `obs` and sends what it raises to every sink. Sink failures are logged, not
    /// returned: alerting must never take the node down.
    pub async fn observe(&self, obs: &Observation) -> Vec<Alert> {
        let alerts = self.evaluate(obs, Instant::now());
        for a in &alerts {
            for sink in &self.sinks {
                if let Err(e) = sink.send(&self.product, a).await {
                    tracing::warn!(sink = sink.name(), key = %a.key, "alert not sent: {e}");
                }
            }
        }
        alerts
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn alerter() -> Alerter {
        Alerter::new("m9", rules::default_rules(&rules::Thresholds::default()), Vec::new())
    }

    #[test]
    fn network_degraded_alerts_once_then_recovers() {
        let a = alerter();
        let t0 = Instant::now();
        let degraded = Observation::new("net.route", json!({"state": "degraded", "dev": "wlan0"}));
        let ok = Observation::new("net.route", json!({"state": "ok", "dev": "eth0"}));
        let first = a.evaluate(&degraded, t0);
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].status, AlertStatus::Failing);
        assert_eq!(first[0].title, "DEGRADED but up: m9 default route uses Wi-Fi (wlan0)");
        assert_eq!(first[0].subject("ymir"), "[ymir m9] ALERT: DEGRADED but up: m9 default route uses Wi-Fi (wlan0)");
        assert!(a.evaluate(&degraded, t0 + Duration::from_secs(60)).is_empty(), "deduped within the hour");
        assert_eq!(a.evaluate(&degraded, t0 + Duration::from_secs(3601)).len(), 1, "repeats hourly");
        assert_eq!(a.failing(), vec!["net.route".to_string()]);
        let back = a.evaluate(&ok, t0 + Duration::from_secs(3700));
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].status, AlertStatus::Recovered);
        assert_eq!(back[0].title, "m9 default route back on eth0");
        assert!(a.evaluate(&ok, t0 + Duration::from_secs(3800)).is_empty(), "one recovery notice");
        assert!(a.failing().is_empty());
    }

    #[test]
    fn no_route_and_engine_rules() {
        let a = alerter();
        let t = Instant::now();
        let r = a.evaluate(&Observation::new("net.route", json!({"state": "noroute", "dev": "-"})), t);
        assert_eq!(r[0].title, "m9 has no default route");
        let r = a.evaluate(&Observation::new("engine.crash_loop", json!({"model": "qwen", "crashes": 5})), t);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].key, "engine.qwen");
        assert!(r[0].title.contains("crash loop"), "{}", r[0].title);
        let r = a.evaluate(&Observation::new("engine.ready", json!({"model": "qwen", "load_secs": 3.0})), t);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].status, AlertStatus::Recovered);
        assert!(a.evaluate(&Observation::new("engine.ready", json!({"model": "other"})), t).is_empty(), "nothing to recover");
    }

    #[test]
    fn disk_and_gpu_thresholds() {
        let a = alerter();
        let t = Instant::now();
        assert!(a.evaluate(&Observation::new("disk", json!({"path": "/", "free_pct": 40.0})), t).is_empty());
        assert_eq!(a.evaluate(&Observation::new("disk", json!({"path": "/", "free_pct": 4.5})), t).len(), 1);
        assert_eq!(a.evaluate(&Observation::new("disk", json!({"path": "/", "free_pct": 30.0})), t)[0].status, AlertStatus::Recovered);
        assert_eq!(a.evaluate(&Observation::new("gpu", json!({"index": 0, "temp_c": 91})), t).len(), 1);
        assert!(a.evaluate(&Observation::new("gpu", json!({"index": 0, "temp_c": 83})), t).is_empty(), "hysteresis: not yet cool");
        assert_eq!(a.evaluate(&Observation::new("gpu", json!({"index": 0, "temp_c": 70})), t)[0].status, AlertStatus::Recovered);
    }

    #[test]
    fn rules_round_trip_as_toml() {
        let rules = rules::default_rules(&rules::Thresholds::default());
        #[derive(Serialize, Deserialize)]
        struct Pack {
            rule: Vec<AlertRule>,
        }
        let text = toml::to_string(&Pack { rule: rules.clone() }).unwrap();
        let back: Pack = toml::from_str(&text).unwrap();
        assert_eq!(back.rule, rules);
    }

    #[tokio::test]
    async fn observe_sends_to_sinks() {
        let mem = Arc::new(MemorySink::default());
        let a = Alerter::new("m9", rules::default_rules(&rules::Thresholds::default()), vec![mem.clone()]);
        a.observe(&Observation::new("engine.load_failed", json!({"model": "big", "why": "out of memory"}))).await;
        let sent = mem.sent();
        assert_eq!(sent.len(), 1);
        assert!(sent[0].title.contains("big") && sent[0].title.contains("out of memory"));
    }
}
