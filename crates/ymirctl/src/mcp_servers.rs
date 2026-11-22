//! `ymirctl mcp-servers`: the outside MCP servers this node's agent tasks may use
//! (`[[inference.agent.mcp]]` in `node.toml`). Secrets are only ever references (`env:NAME` or
//! `file:/path`); nothing here prints a secret. A change takes effect when `ymird` restarts.

use std::collections::BTreeMap;
use std::io::{BufRead, Write};

use anyhow::{Context, Result, bail};
use clap::{Args, Subcommand};
use ymir_core::Paths;
use ymir_core::mcp_servers::{McpServerConfig, McpTransport};

#[derive(Args, Debug)]
pub struct McpServersCmd {
    #[command(subcommand)]
    pub sub: McpServersSub,
}

#[derive(Subcommand, Debug)]
pub enum McpServersSub {
    /// List the servers (no secrets are shown).
    List {
        #[arg(long)]
        json: bool,
    },
    /// Add a server: a local command (`--command`) or a streamable HTTP endpoint (`--url`).
    Add(AddArgs),
    /// Remove a server.
    Remove { id: String },
    /// Turn a server on.
    Enable { id: String },
    /// Turn a server off without removing it.
    Disable { id: String },
    /// Connect to a server now and list its tools (starts a stdio server's command).
    Test { id: String },
}

#[derive(Args, Debug)]
pub struct AddArgs {
    /// Names the server; its tools are offered as `<id>.<tool>`.
    pub id: String,
    /// stdio: the program to run. It runs as the user ymird runs as.
    #[arg(long, conflicts_with = "url")]
    pub command: Option<String>,
    /// stdio: an argument (repeat for more).
    #[arg(long = "arg", allow_hyphen_values = true)]
    pub args: Vec<String>,
    /// stdio: the working directory.
    #[arg(long)]
    pub cwd: Option<String>,
    /// stdio: NAME=env:VAR or NAME=file:/path, the value the command gets as NAME (repeat).
    #[arg(long = "env")]
    pub env: Vec<String>,
    /// http: the MCP endpoint (https://, or http:// on localhost).
    #[arg(long)]
    pub url: Option<String>,
    /// http: the bearer token as env:VAR or file:/path.
    #[arg(long)]
    pub bearer: Option<String>,
    /// http: NAME=env:VAR or NAME=file:/path, an extra header (repeat).
    #[arg(long = "header")]
    pub headers: Vec<String>,
    /// Per-request timeout in seconds.
    #[arg(long, default_value_t = 60)]
    pub timeout: u64,
    /// Don't ask before adding a local command.
    #[arg(long, short)]
    pub yes: bool,
}

fn pairs(items: &[String], what: &str) -> Result<BTreeMap<String, String>> {
    let mut m = BTreeMap::new();
    for i in items {
        let (k, v) = i.split_once('=').with_context(|| format!("{what} `{i}` must be NAME=env:VAR or NAME=file:/path"))?;
        if k.is_empty() {
            bail!("{what} `{i}` has no name");
        }
        m.insert(k.to_string(), v.to_string());
    }
    Ok(m)
}

pub async fn run(paths: &Paths, cmd: McpServersCmd, out: &mut dyn Write, input: &mut dyn BufRead) -> Result<()> {
    let path = paths.config();
    let mut cfg = crate::load_cfg(paths)?;
    let list = &mut cfg.inference.agent.mcp;
    match cmd.sub {
        McpServersSub::List { json } => {
            if json {
                writeln!(out, "{}", serde_json::to_string_pretty(&*list)?)?;
            } else if list.is_empty() {
                writeln!(out, "No MCP servers. Add one with `ymirctl mcp-servers add`.")?;
            }
            if !json {
                for s in list.iter() {
                    let target = match s.transport {
                        McpTransport::Stdio => format!("stdio: {} {}", s.command.as_deref().unwrap_or_default(), s.args.join(" ")),
                        McpTransport::Http => format!("http: {}", s.url.as_deref().unwrap_or_default()),
                    };
                    let secrets = s.env.len() + s.headers.len() + usize::from(s.bearer.is_some());
                    writeln!(
                        out,
                        "{}\t{}\t{}\t({secrets} secret reference{})",
                        s.id,
                        if s.enabled { "on" } else { "off" },
                        target.trim_end(),
                        if secrets == 1 { "" } else { "s" }
                    )?;
                }
            }
            Ok(())
        }
        McpServersSub::Add(a) => {
            if list.iter().any(|s| s.id == a.id) {
                bail!("`{}` is already listed; remove it first", a.id);
            }
            let transport = match (&a.command, &a.url) {
                (Some(_), None) => McpTransport::Stdio,
                (None, Some(_)) => McpTransport::Http,
                _ => bail!("give either --command (a local server) or --url (an HTTP server)"),
            };
            let s = McpServerConfig {
                id: a.id.clone(),
                enabled: true,
                transport,
                command: a.command,
                args: a.args,
                cwd: a.cwd,
                env: pairs(&a.env, "--env")?,
                url: a.url,
                bearer: a.bearer,
                headers: pairs(&a.headers, "--header")?,
                timeout_secs: a.timeout,
            };
            s.validate()?;
            if let Some(w) = s.warning()
                && !a.yes
                && !crate::confirm(out, input, &format!("{w}\nAdd it? [y/N] "))?
            {
                writeln!(out, "Not added.")?;
                return Ok(());
            }
            list.push(s);
            cfg.save(&path)?;
            writeln!(out, "Added `{}`. Restart ymird to use it; every tool call will ask for approval on the asking device.", a.id)?;
            Ok(())
        }
        McpServersSub::Remove { id } => {
            let before = list.len();
            list.retain(|s| s.id != id);
            if list.len() == before {
                bail!("no MCP server `{id}`");
            }
            cfg.save(&path)?;
            writeln!(out, "Removed `{id}`.")?;
            Ok(())
        }
        McpServersSub::Enable { id } => set_enabled(&mut cfg, &path, &id, true, out),
        McpServersSub::Disable { id } => set_enabled(&mut cfg, &path, &id, false, out),
        McpServersSub::Test { id } => {
            let s = list.iter().find(|s| s.id == id).with_context(|| format!("no MCP server `{id}`"))?;
            let source = s.connect().await?;
            let tools = source.client().list_tools().await.map_err(|e| anyhow::anyhow!("listing tools: {e:?}"))?;
            writeln!(out, "`{id}` answers with {} tool{}:", tools.len(), if tools.len() == 1 { "" } else { "s" })?;
            for t in tools {
                writeln!(out, "  {}.{}", id, t.name)?;
            }
            source.client().close().await;
            Ok(())
        }
    }
}

fn set_enabled(cfg: &mut ymir_core::config::NodeConfig, path: &std::path::Path, id: &str, on: bool, out: &mut dyn Write) -> Result<()> {
    let s = cfg.inference.agent.mcp.iter_mut().find(|s| s.id == id).with_context(|| format!("no MCP server `{id}`"))?;
    s.enabled = on;
    cfg.save(path)?;
    writeln!(out, "`{id}` is {}. Restart ymird to apply.", if on { "on" } else { "off" })?;
    Ok(())
}
