//! Where alerts go.

use std::sync::Mutex;
use std::time::Duration;

use futures::future::BoxFuture;
use lettre::message::{Mailbox, header::ContentType};
use lettre::transport::smtp::authentication::Credentials;
use lettre::{AsyncSmtpTransport, AsyncTransport, Message, Tokio1Executor};
use serde_json::json;

use crate::Alert;
use crate::config::{SmtpConfig, WebhookConfig, resolve_secret};

pub trait Sink: Send + Sync {
    fn name(&self) -> &'static str;
    /// Delivers one alert. `product` names the sender in subjects (`ymir`).
    fn send<'a>(&'a self, product: &'a str, alert: &'a Alert) -> BoxFuture<'a, Result<(), String>>;
}

/// Writes alerts to the daemon log.
pub struct LogSink;

impl Sink for LogSink {
    fn name(&self) -> &'static str {
        "log"
    }

    fn send<'a>(&'a self, product: &'a str, alert: &'a Alert) -> BoxFuture<'a, Result<(), String>> {
        Box::pin(async move {
            tracing::warn!(key = %alert.key, status = ?alert.status, "{}", alert.subject(product));
            Ok(())
        })
    }
}

/// Keeps alerts in memory (tests, and `ymirctl status`).
#[derive(Default)]
pub struct MemorySink {
    sent: Mutex<Vec<Alert>>,
}

impl MemorySink {
    pub fn sent(&self) -> Vec<Alert> {
        self.sent.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

impl Sink for MemorySink {
    fn name(&self) -> &'static str {
        "memory"
    }

    fn send<'a>(&'a self, _product: &'a str, alert: &'a Alert) -> BoxFuture<'a, Result<(), String>> {
        self.sent.lock().unwrap_or_else(|e| e.into_inner()).push(alert.clone());
        Box::pin(async { Ok(()) })
    }
}

/// Push to the owner's phones. `weirdnet-push` has no Weirdnet front end yet, so this only notes
/// that the alert would have been pushed.
pub struct PushSink;

impl Sink for PushSink {
    fn name(&self) -> &'static str {
        "push"
    }

    fn send<'a>(&'a self, product: &'a str, alert: &'a Alert) -> BoxFuture<'a, Result<(), String>> {
        Box::pin(async move {
            tracing::info!(key = %alert.key, "push is off until weirdnet-push has a Weirdnet front end; would push: {}", alert.subject(product));
            Ok(())
        })
    }
}

/// Email over SMTP (STARTTLS or TLS). In dry-run mode the message is built and kept, not sent.
pub struct SmtpSink {
    cfg: SmtpConfig,
    from: Mailbox,
    to: Vec<Mailbox>,
    transport: Option<AsyncSmtpTransport<Tokio1Executor>>,
    dry: Option<Mutex<Vec<String>>>,
}

impl SmtpSink {
    pub fn new(cfg: &SmtpConfig, dry_run: bool) -> Result<Self, String> {
        let from: Mailbox = cfg.from.parse().map_err(|e| format!("smtp from `{}`: {e}", cfg.from))?;
        let to = cfg.to.iter().map(|t| t.parse::<Mailbox>().map_err(|e| format!("smtp to `{t}`: {e}"))).collect::<Result<Vec<_>, _>>()?;
        if to.is_empty() {
            return Err("smtp: `to` is empty".into());
        }
        let transport = if dry_run {
            None
        } else {
            let builder = match cfg.security.as_str() {
                "starttls" => AsyncSmtpTransport::<Tokio1Executor>::starttls_relay(&cfg.server).map_err(|e| format!("smtp {}: {e}", cfg.server))?,
                "tls" => AsyncSmtpTransport::<Tokio1Executor>::relay(&cfg.server).map_err(|e| format!("smtp {}: {e}", cfg.server))?,
                "none" if matches!(cfg.server.as_str(), "127.0.0.1" | "localhost" | "::1") => {
                    AsyncSmtpTransport::<Tokio1Executor>::builder_dangerous(&cfg.server)
                }
                "none" => return Err("smtp: security = \"none\" is only allowed for a loopback relay".into()),
                other => return Err(format!("smtp: unknown security `{other}` (starttls, tls or none)")),
            };
            let mut builder = builder.port(cfg.port).timeout(Some(Duration::from_secs(20)));
            if let Some(user) = &cfg.user {
                let source = cfg.password.as_deref().ok_or("smtp: `user` is set but `password` (env:NAME or file:/path) is not")?;
                let password = resolve_secret(source).map_err(|e| format!("smtp password: {e}"))?;
                builder = builder.credentials(Credentials::new(user.clone(), password));
            }
            Some(builder.build())
        };
        Ok(Self { cfg: cfg.clone(), from, to, transport, dry: dry_run.then(|| Mutex::new(Vec::new())) })
    }

    pub fn message(&self, product: &str, alert: &Alert) -> Result<Message, String> {
        let mut b = Message::builder().from(self.from.clone()).subject(alert.subject(product)).header(ContentType::TEXT_PLAIN);
        for t in &self.to {
            b = b.to(t.clone());
        }
        b.body(alert.body()).map_err(|e| format!("building the email: {e}"))
    }

    /// Messages a dry-run sink built, as RFC 5322 text.
    pub fn dry_sent(&self) -> Vec<String> {
        self.dry.as_ref().map(|d| d.lock().unwrap_or_else(|e| e.into_inner()).clone()).unwrap_or_default()
    }
}

impl Sink for SmtpSink {
    fn name(&self) -> &'static str {
        "smtp"
    }

    fn send<'a>(&'a self, product: &'a str, alert: &'a Alert) -> BoxFuture<'a, Result<(), String>> {
        Box::pin(async move {
            let msg = self.message(product, alert)?;
            if let Some(dry) = &self.dry {
                let text = String::from_utf8_lossy(&msg.formatted()).into_owned();
                tracing::info!(server = %self.cfg.server, to = ?self.cfg.to, "dry run: would email: {}", alert.subject(product));
                dry.lock().unwrap_or_else(|e| e.into_inner()).push(text);
                return Ok(());
            }
            let t = self.transport.as_ref().ok_or("smtp transport missing")?;
            t.send(msg).await.map(|_| ()).map_err(|e| format!("smtp {}: {e}", self.cfg.server))
        })
    }
}

/// POSTs alerts as JSON. In dry-run mode the payload is kept, not sent.
pub struct WebhookSink {
    url: String,
    token: Option<String>,
    http: reqwest::Client,
    dry: Option<Mutex<Vec<serde_json::Value>>>,
}

impl WebhookSink {
    pub fn new(cfg: &WebhookConfig, dry_run: bool) -> Result<Self, String> {
        let loopback = ["http://127.0.0.1", "http://localhost", "http://[::1]"].iter().any(|p| cfg.url.starts_with(p));
        if !cfg.url.starts_with("https://") && !loopback {
            return Err(format!("webhook {}: use https (plain http only to loopback)", cfg.url));
        }
        let token = match (&cfg.token, dry_run) {
            (Some(src), false) => Some(resolve_secret(src).map_err(|e| format!("webhook token: {e}"))?),
            _ => None,
        };
        // ring, like lettre, so no aws-lc C build (it doesn't cross-build for the BSDs).
        let _ = rustls::crypto::ring::default_provider().install_default();
        let http = reqwest::Client::builder().timeout(Duration::from_secs(15)).build().map_err(|e| format!("webhook client: {e}"))?;
        Ok(Self { url: cfg.url.clone(), token, http, dry: dry_run.then(|| Mutex::new(Vec::new())) })
    }

    pub fn payload(product: &str, alert: &Alert) -> serde_json::Value {
        json!({
            "source": product,
            "key": alert.key,
            "status": alert.status,
            "host": alert.host,
            "title": alert.title,
            "subject": alert.subject(product),
            "at": alert.at,
            "observation": alert.observation,
        })
    }

    pub fn dry_sent(&self) -> Vec<serde_json::Value> {
        self.dry.as_ref().map(|d| d.lock().unwrap_or_else(|e| e.into_inner()).clone()).unwrap_or_default()
    }
}

impl Sink for WebhookSink {
    fn name(&self) -> &'static str {
        "webhook"
    }

    fn send<'a>(&'a self, product: &'a str, alert: &'a Alert) -> BoxFuture<'a, Result<(), String>> {
        Box::pin(async move {
            let body = Self::payload(product, alert);
            if let Some(dry) = &self.dry {
                tracing::info!(url = %self.url, "dry run: would post: {}", alert.subject(product));
                dry.lock().unwrap_or_else(|e| e.into_inner()).push(body);
                return Ok(());
            }
            let mut req = self.http.post(&self.url).json(&body);
            if let Some(t) = &self.token {
                req = req.bearer_auth(t);
            }
            let resp = req.send().await.map_err(|e| format!("webhook {}: {e}", self.url))?;
            if resp.status().is_success() { Ok(()) } else { Err(format!("webhook {}: {}", self.url, resp.status())) }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use super::*;
    use crate::{Alerter, Observation, rules};

    fn alert() -> Alert {
        let a = Alerter::new("m9", rules::default_rules(&rules::Thresholds::default()), Vec::new());
        a.evaluate(&Observation::new("net.route", json!({"state": "degraded", "dev": "wlan0"})), Instant::now()).remove(0)
    }

    #[tokio::test]
    async fn smtp_dry_run_builds_the_email() {
        let cfg = SmtpConfig {
            server: "smtp.example.com".into(),
            port: 587,
            security: "starttls".into(),
            user: Some("alerts@example.com".into()),
            password: Some("env:YMIR_TEST_NEVER_SET".into()),
            from: "Ymir <alerts@example.com>".into(),
            to: vec!["owner@example.com".into()],
        };
        assert!(SmtpSink::new(&cfg, false).is_err(), "a real sink needs the password");
        let s = SmtpSink::new(&cfg, true).unwrap();
        s.send("ymir", &alert()).await.unwrap();
        let sent = s.dry_sent();
        assert_eq!(sent.len(), 1);
        let unfolded = sent[0].replace("\r\n ", " ");
        assert!(unfolded.contains("Subject: [ymir m9] ALERT: DEGRADED but up: m9 default route uses Wi-Fi (wlan0)"), "{unfolded}");
        assert!(sent[0].contains("To: owner@example.com"));
        let insecure = SmtpConfig { security: "none".into(), ..cfg };
        assert!(SmtpSink::new(&insecure, false).is_err(), "plain SMTP only to loopback");
    }

    #[tokio::test]
    async fn webhook_posts_json_to_a_mock() {
        assert!(WebhookSink::new(&WebhookConfig { url: "http://hooks.example.com/x".into(), token: None }, false).is_err());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(async move {
            let (mut s, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 16384];
            let mut got = Vec::new();
            loop {
                let n = s.read(&mut buf).await.unwrap();
                got.extend_from_slice(&buf[..n]);
                let text = String::from_utf8_lossy(&got).to_string();
                if let Some(i) = text.find("\r\n\r\n") {
                    let len: usize =
                        text.lines().find_map(|l| l.to_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse().unwrap())).unwrap_or(0);
                    if got.len() >= i + 4 + len {
                        break;
                    }
                }
            }
            s.write_all(b"HTTP/1.1 204 No Content\r\ncontent-length: 0\r\n\r\n").await.unwrap();
            String::from_utf8_lossy(&got).to_string()
        });
        let w = WebhookSink::new(&WebhookConfig { url: format!("http://127.0.0.1:{port}/hook"), token: None }, false).unwrap();
        w.send("ymir", &alert()).await.unwrap();
        let req = server.await.unwrap();
        assert!(req.starts_with("POST /hook"));
        assert!(req.contains("\"key\":\"net.route\"") && req.contains("\"status\":\"failing\""), "{req}");
        let dry = WebhookSink::new(&WebhookConfig { url: "https://hooks.example.com/x".into(), token: Some("env:NOPE".into()) }, true).unwrap();
        dry.send("ymir", &alert()).await.unwrap();
        assert_eq!(dry.dry_sent()[0]["host"], "m9");
    }
}
