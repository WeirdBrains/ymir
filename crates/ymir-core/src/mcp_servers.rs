//! Outside MCP servers for agent tasks (`[[inference.agent.mcp]]` in `node.toml`).
//!
//! The protocol client and the `ToolSource` adapter are `weirdnet-mcp`'s (features `client` and
//! `agent`); this module is the node's configuration of them: the config shape, its validation,
//! secret references (`env:NAME` or `file:/path`, never the secret itself) and connecting.
//!
//! A stdio server runs as the user `ymird` runs as. Only the operator adds one (at `ymirctl` or in
//! `node.toml`); a remote request can't. Every call from these tools asks the asking device for
//! approval (the tier floor is `Research`, so no outside tool is ever read-only-and-trusted).

use std::collections::{BTreeMap, HashSet};
use std::time::Duration;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use weirdnet_mcp::agent::{McpSource, SourceOptions};
use weirdnet_mcp::client::{ClientOptions, HttpConfig, Secret, StdioCommand};

/// How a server is reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum McpTransport {
    /// A local command speaking MCP on stdin and stdout.
    Stdio,
    /// Streamable HTTP.
    Http,
}

/// One `[[inference.agent.mcp]]` entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Names the server; its tools are offered as `<id>.<tool>`. No spaces or dots.
    pub id: String,
    #[serde(default = "yes")]
    pub enabled: bool,
    pub transport: McpTransport,
    /// stdio: the program.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    /// stdio: environment variable name to a secret reference (`env:NAME` or `file:/path`).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
    /// http: the MCP endpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// http: the bearer token, as a secret reference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bearer: Option<String>,
    /// http: header name to a secret reference.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
    /// Per-request timeout in seconds (default 60).
    #[serde(default = "timeout_secs")]
    pub timeout_secs: u64,
}

fn yes() -> bool {
    true
}

fn timeout_secs() -> u64 {
    60
}

fn is_secret_ref(s: &str) -> bool {
    s.starts_with("env:") || s.starts_with("file:")
}

fn loopback(url: &str) -> bool {
    let host = url.strip_prefix("http://").unwrap_or_default();
    let host = host.split(['/', '?']).next().unwrap_or_default();
    let name =
        if let Some(rest) = host.strip_prefix('[') { rest.split(']').next().unwrap_or_default() } else { host.split(':').next().unwrap_or_default() };
    name == "localhost" || name == "::1" || name.parse::<std::net::Ipv4Addr>().is_ok_and(|a| a.is_loopback())
}

impl McpServerConfig {
    /// Checks one entry (nothing is started or read).
    pub fn validate(&self) -> Result<()> {
        let id = &self.id;
        if id.is_empty() || id.contains(|c: char| c.is_whitespace() || c == '.') {
            bail!("mcp server id `{id}` must be non-empty with no spaces or dots");
        }
        if self.timeout_secs == 0 {
            bail!("mcp server `{id}`: timeout_secs must be at least 1");
        }
        let refs = self.env.values().chain(self.bearer.iter()).chain(self.headers.values());
        for r in refs {
            if !is_secret_ref(r) {
                bail!("mcp server `{id}`: env, bearer and header values must be env:NAME or file:/path, not the secret itself");
            }
        }
        match self.transport {
            McpTransport::Stdio => {
                if self.command.as_deref().is_none_or(str::is_empty) {
                    bail!("mcp server `{id}`: a stdio server needs `command`");
                }
                if self.url.is_some() || self.bearer.is_some() || !self.headers.is_empty() {
                    bail!("mcp server `{id}`: url, bearer and headers are for http servers");
                }
            }
            McpTransport::Http => {
                let Some(url) = self.url.as_deref() else { bail!("mcp server `{id}`: an http server needs `url`") };
                if !(url.starts_with("https://") || (url.starts_with("http://") && loopback(url))) {
                    bail!("mcp server `{id}`: the url must be https:// (plain http only for localhost)");
                }
                if self.command.is_some() || !self.args.is_empty() || self.cwd.is_some() || !self.env.is_empty() {
                    bail!("mcp server `{id}`: command, args, cwd and env are for stdio servers");
                }
            }
        }
        Ok(())
    }

    /// What to show the operator before this server is started: the command line and that it runs
    /// as them. `None` for http.
    pub fn warning(&self) -> Option<String> {
        (self.transport == McpTransport::Stdio).then(|| self.stdio_command(false).ok().map(|c| c.warning())).flatten()
    }

    fn stdio_command(&self, resolve: bool) -> Result<StdioCommand> {
        let mut c = StdioCommand::new(self.command.clone().unwrap_or_default());
        for a in &self.args {
            c = c.arg(a.clone());
        }
        if let Some(d) = &self.cwd {
            c = c.cwd(d);
        }
        for (k, v) in &self.env {
            let value = if resolve { secret(v).map_err(|e| anyhow::anyhow!("mcp server `{}`: env {k}: {e}", self.id))? } else { String::new() };
            c = c.env(k.clone(), Secret::new(value));
        }
        Ok(c)
    }

    /// Starts or connects to the server and lists nothing yet. The tools are offered as `<id>.<tool>`.
    pub async fn connect(&self) -> Result<McpSource> {
        self.validate()?;
        let options = ClientOptions { timeout: Duration::from_secs(self.timeout_secs), client_name: "ymir".into(), ..ClientOptions::default() };
        let source = match self.transport {
            McpTransport::Stdio => McpSource::connect_stdio(self.id.clone(), &self.stdio_command(true)?, options).await,
            McpTransport::Http => {
                let mut h = HttpConfig::new(self.url.clone().unwrap_or_default());
                if let Some(b) = &self.bearer {
                    h = h.bearer(Secret::new(secret(b).map_err(|e| anyhow::anyhow!("mcp server `{}`: bearer: {e}", self.id))?));
                }
                for (k, v) in &self.headers {
                    h = h.header(k.clone(), Secret::new(secret(v).map_err(|e| anyhow::anyhow!("mcp server `{}`: header {k}: {e}", self.id))?));
                }
                McpSource::connect_http(self.id.clone(), &h, options).await
            }
        }
        .map_err(|e| anyhow::anyhow!("mcp server `{}`: {e:?}", self.id))?;
        Ok(source.with_options(SourceOptions { name_prefix: Some(format!("{}.", self.id)), ..SourceOptions::default() }))
    }
}

fn secret(source: &str) -> Result<String, String> {
    weirdnet_alerts::config::resolve_secret(source)
}

/// Checks the list: each entry, and no id twice.
pub fn validate(servers: &[McpServerConfig]) -> Result<()> {
    let mut seen = HashSet::new();
    for s in servers {
        s.validate()?;
        if !seen.insert(&s.id) {
            bail!("mcp server `{}` is listed twice", s.id);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stdio(id: &str) -> McpServerConfig {
        McpServerConfig {
            id: id.into(),
            enabled: true,
            transport: McpTransport::Stdio,
            command: Some("/bin/true".into()),
            args: vec![],
            cwd: None,
            env: BTreeMap::new(),
            url: None,
            bearer: None,
            headers: BTreeMap::new(),
            timeout_secs: 60,
        }
    }

    fn http(url: &str) -> McpServerConfig {
        McpServerConfig { transport: McpTransport::Http, command: None, url: Some(url.into()), ..stdio("h") }
    }

    #[test]
    fn validation() {
        assert!(stdio("files").validate().is_ok());
        assert!(stdio("a b").validate().is_err());
        assert!(stdio("a.b").validate().is_err());
        assert!(McpServerConfig { command: None, ..stdio("x") }.validate().is_err());
        let mut s = stdio("x");
        s.env.insert("TOKEN".into(), "sk-123".into());
        assert!(s.validate().unwrap_err().to_string().contains("not the secret itself"));
        s.env.insert("TOKEN".into(), "env:MY_TOKEN".into());
        assert!(s.validate().is_ok());
        assert!(http("https://example.com/mcp").validate().is_ok());
        assert!(http("http://example.com/mcp").validate().is_err());
        assert!(http("http://127.0.0.1:9/mcp").validate().is_ok());
        assert!(http("http://localhost:9/mcp").validate().is_ok());
        assert!(http("http://[::1]:9/mcp").validate().is_ok());
        assert!(http("http://127.0.0.1.evil.com/mcp").validate().is_err());
        assert!(McpServerConfig { bearer: Some("tok".into()), ..http("https://e.com/m") }.validate().is_err());
        assert!(validate(&[stdio("a"), stdio("a")]).is_err());
    }

    #[test]
    fn warning_shows_the_command_without_secrets() {
        let mut s = stdio("x");
        s.env.insert("TOKEN".into(), "env:MY_TOKEN_UNSET".into());
        let w = s.warning().unwrap();
        assert!(w.contains("/bin/true"));
        assert!(http("https://e.com/m").warning().is_none());
    }

    #[test]
    fn toml_round_trip_keeps_references_only() {
        let mut s = http("https://e.com/m");
        s.bearer = Some("env:E_TOKEN".into());
        let t = toml::to_string(&s).unwrap();
        assert!(t.contains("env:E_TOKEN"));
        assert_eq!(toml::from_str::<McpServerConfig>(&t).unwrap(), s);
    }
}
