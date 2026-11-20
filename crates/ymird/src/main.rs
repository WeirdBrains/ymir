//! `ymird`: one daemon for any mix of the model, inference and app roles.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use ymir_core::config::Roles;
use ymir_core::{Daemon, NodeConfig, Paths};

#[cfg(windows)]
mod winservice;

#[derive(Parser, Debug, Clone)]
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
    /// Append the log here instead of writing it to stderr (services without a journal).
    #[arg(long)]
    log_file: Option<PathBuf>,
    /// Run under the Windows service control manager (how the `ymird` service starts it).
    #[cfg(windows)]
    #[arg(long, hide = true)]
    windows_service: bool,
}

fn init_logging(log_file: Option<&std::path::Path>) -> Result<()> {
    let filter = || tracing_subscriber::EnvFilter::try_from_env("YMIR_LOG").unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    match log_file {
        Some(path) => {
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
            }
            let file = std::fs::OpenOptions::new().create(true).append(true).open(path).with_context(|| format!("opening {}", path.display()))?;
            tracing_subscriber::fmt().with_env_filter(filter()).with_ansi(false).with_writer(std::sync::Mutex::new(file)).init();
        }
        // No colour codes when stderr is a file (launchd's StandardErrorPath, a redirect).
        None => tracing_subscriber::fmt()
            .with_env_filter(filter())
            .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
            .with_writer(std::io::stderr)
            .init(),
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    init_logging(args.log_file.as_deref())?;
    #[cfg(windows)]
    if args.windows_service {
        return winservice::run(args);
    }
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let daemon = start(&args).await?;
        tokio::select! {
            _ = wait_for_signal() => {}
            _ = daemon.exit_requested() => {}
        }
        daemon.shutdown().await;
        Ok(())
    })
}

/// Loads the config and starts the roles; an update on trial passes its health check once they serve.
async fn start(args: &Args) -> Result<Daemon> {
    let paths = args.home.clone().map(Paths::new).unwrap_or_else(Paths::from_env);
    let config = args.config.clone().unwrap_or_else(|| paths.config());
    let mut cfg =
        NodeConfig::load(&config).with_context(|| format!("no usable config; run `ymirctl setup` first (looked at {})", config.display()))?;
    if let Some(r) = &args.roles {
        cfg.roles = Roles::parse(r)?;
    }
    let daemon = Daemon::start(cfg, paths.clone()).await?;
    // Serving: an update on trial passes its health check (see `ymirctl update boot`).
    ymir_core::update::commit_if_on_trial(&paths);
    Ok(daemon)
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
    #[cfg(windows)]
    {
        // Ctrl+C or Ctrl+Break in a console, the console closing, logoff and shutdown.
        use tokio::signal::windows;
        let (mut brk, mut close, mut logoff, mut shutdown) = (
            windows::ctrl_break().expect("Ctrl+Break handler"),
            windows::ctrl_close().expect("console close handler"),
            windows::ctrl_logoff().expect("logoff handler"),
            windows::ctrl_shutdown().expect("shutdown handler"),
        );
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = brk.recv() => {}
            _ = close.recv() => {}
            _ = logoff.recv() => {}
            _ = shutdown.recv() => {}
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
