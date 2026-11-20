//! `ymird`: one daemon for any mix of the model, inference and app roles.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use ymir_core::config::Roles;
use ymir_core::{Daemon, NodeConfig, Paths};

#[derive(Parser)]
#[command(version, about = "Ymir node daemon: the model, inference and app roles that node.toml turns on")]
struct Args {
    /// Config and state directory (default ~/.config/ymir).
    #[arg(long, env = "YMIR_HOME")]
    home: Option<PathBuf>,
    /// Config file (default `<home>/node.toml`).
    #[arg(long)]
    config: Option<PathBuf>,
    /// Override `[roles]`, such as `model,inference,app` or `app`.
    #[arg(long)]
    roles: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_env("YMIR_LOG").unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")))
        .with_writer(std::io::stderr)
        .init();
    let args = Args::parse();
    let paths = args.home.map(Paths::new).unwrap_or_else(Paths::from_env);
    let config = args.config.unwrap_or_else(|| paths.config());
    let mut cfg =
        NodeConfig::load(&config).with_context(|| format!("no usable config; run `ymirctl setup` first (looked at {})", config.display()))?;
    if let Some(r) = args.roles {
        cfg.roles = Roles::parse(&r)?;
    }
    let daemon = Daemon::start(cfg, paths.clone()).await?;
    // Serving: an update on trial passes its health check (see `ymirctl update boot`).
    ymir_core::update::commit_if_on_trial(&paths);
    wait_for_signal().await;
    daemon.shutdown().await;
    Ok(())
}

async fn wait_for_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut term = signal(SignalKind::terminate()).expect("SIGTERM handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
