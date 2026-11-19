//! `[inference.alerts]` in a node config.

use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::rules::Thresholds;
use crate::sinks::{LogSink, PushSink, Sink, SmtpSink, WebhookSink};
use crate::{AlertRule, Alerter};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlertsConfig {
    /// Watch and log alerts (sinks below decide where else they go).
    #[serde(default = "yes")]
    pub enabled: bool,
    /// Build every message but send nothing; sinks log what they would have sent.
    #[serde(default)]
    pub dry_run: bool,
    /// A failing alert repeats at most this often, in seconds (default an hour).
    #[serde(default = "hour")]
    pub repeat_secs: u64,
    /// How often probes run, in seconds.
    #[serde(default = "every")]
    pub probe_secs: u64,
    /// Filesystems to watch for free space; default the Ymir home and the models dir.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub disks: Vec<PathBuf>,
    #[serde(default, flatten)]
    pub thresholds: Thresholds,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub smtp: Option<SmtpConfig>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub webhook: Vec<WebhookConfig>,
    #[serde(default)]
    pub push: PushConfig,
    /// Extra rules, `weirdnet-rules` specs with `fail`/`ok`/`event` actions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rule: Vec<AlertRule>,
}

fn yes() -> bool {
    true
}

fn hour() -> u64 {
    3600
}

fn every() -> u64 {
    30
}

impl Default for AlertsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            dry_run: false,
            repeat_secs: hour(),
            probe_secs: every(),
            disks: Vec::new(),
            thresholds: Thresholds::default(),
            smtp: None,
            webhook: Vec::new(),
            push: PushConfig::default(),
            rule: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SmtpConfig {
    pub server: String,
    #[serde(default = "smtp_port")]
    pub port: u16,
    /// `starttls` (default), `tls` or `none` (loopback relays only).
    #[serde(default = "starttls")]
    pub security: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Where the password comes from: `env:NAME` or `file:/path`. Never the password itself.
    /// `keyring:` is planned (it needs the platform keychain crate).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub password: Option<String>,
    pub from: String,
    pub to: Vec<String>,
}

fn smtp_port() -> u16 {
    587
}

fn starttls() -> String {
    "starttls".into()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WebhookConfig {
    pub url: String,
    /// Bearer token source, `env:NAME` or `file:/path`, if the endpoint wants one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PushConfig {
    /// Push to the owner's phones through `weirdnet-push`. Off until it has a Weirdnet front
    /// end; turning it on only logs that.
    #[serde(default)]
    pub enabled: bool,
}

/// Reads a secret from `env:NAME` or `file:/path`. The value is never logged.
pub fn resolve_secret(source: &str) -> Result<String, String> {
    if let Some(name) = source.strip_prefix("env:") {
        return std::env::var(name).map_err(|_| format!("environment variable {name} is not set"));
    }
    if let Some(path) = source.strip_prefix("file:") {
        return std::fs::read_to_string(path).map(|s| s.trim_end_matches(['\n', '\r']).to_string()).map_err(|e| format!("reading {path}: {e}"));
    }
    if source.starts_with("keyring:") {
        return Err("keyring: secrets are not supported yet; use env:NAME or file:/path".into());
    }
    Err("a secret must be a reference (env:NAME or file:/path), not the secret itself".into())
}

impl AlertsConfig {
    /// The sinks this config asks for. The log sink is always there.
    pub fn sinks(&self) -> Result<Vec<Arc<dyn Sink>>, String> {
        let mut out: Vec<Arc<dyn Sink>> = vec![Arc::new(LogSink)];
        if let Some(s) = &self.smtp {
            out.push(Arc::new(SmtpSink::new(s, self.dry_run)?));
        }
        for w in &self.webhook {
            out.push(Arc::new(WebhookSink::new(w, self.dry_run)?));
        }
        if self.push.enabled {
            out.push(Arc::new(PushSink));
        }
        Ok(out)
    }

    /// An alerter for `host` with the default rules plus `rule`.
    pub fn alerter(&self, host: &str) -> Result<Alerter, String> {
        let mut rules = crate::rules::default_rules(&self.thresholds);
        rules.extend(self.rule.iter().cloned());
        Ok(Alerter::new(host, rules, self.sinks()?).with_repeat(std::time::Duration::from_secs(self.repeat_secs.max(1))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_refuses_inline_secrets() {
        let cfg: AlertsConfig = toml::from_str(
            r#"
            dry_run = true
            disk_min_free_pct = 5
            [smtp]
            server = "smtp.example.com"
            user = "alerts@example.com"
            password = "env:YMIR_TEST_SMTP_PASSWORD_UNSET"
            from = "alerts@example.com"
            to = ["owner@example.com"]
            [[webhook]]
            url = "https://hooks.example.com/ymir"
            [[rule]]
            on = ["engine.unloaded"]
            then = [{ do = "event", key = "u.{model}", title = "{model} unloaded" }]
            "#,
        )
        .unwrap();
        assert_eq!(cfg.thresholds.disk_min_free_pct, 5.0);
        assert_eq!(cfg.smtp.as_ref().unwrap().port, 587);
        assert_eq!(cfg.rule.len(), 1);
        assert!(resolve_secret("hunter2").is_err());
        assert!(resolve_secret("keyring:ymir/smtp").is_err());
        assert!(resolve_secret("env:YMIR_TEST_SMTP_PASSWORD_UNSET").is_err());
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("pw");
        std::fs::write(&f, "s3cret\n").unwrap();
        assert_eq!(resolve_secret(&format!("file:{}", f.display())).unwrap(), "s3cret");
        // Dry run needs no password.
        assert_eq!(cfg.sinks().unwrap().len(), 3);
    }
}
