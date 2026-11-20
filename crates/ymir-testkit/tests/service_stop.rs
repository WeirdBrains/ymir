//! How a service stops `ymird` where there is no signal to send (a Windows logon task): the
//! `shutdown` control request wakes `Daemon::exit_requested`, and after `shutdown` the control
//! socket (a named pipe on Windows) is gone. Runs on every OS.

use std::path::Path;
use std::time::Duration;

use anyhow::Result;
use clap::Parser as _;
use ymir_core::control::{self, ControlRequest};
use ymir_core::{Daemon, NodeConfig, Paths};

const MOCK: &str = env!("CARGO_BIN_EXE_ymir-mock-engine");

async fn ctl(home: &Path, args: &[&str]) -> Result<String> {
    let mut argv = vec!["ymirctl".to_string(), "--home".into(), home.display().to_string()];
    argv.extend(args.iter().map(|a| a.to_string()));
    let mut out = Vec::new();
    ymirctl::run(ymirctl::Cli::try_parse_from(argv)?, &mut out, &mut std::io::empty()).await?;
    Ok(String::from_utf8(out)?)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn shutdown_request_stops_ymird() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let home = dir.path().join("home");
    let gguf = dir.path().join("tiny.gguf");
    std::fs::write(&gguf, b"GGUF mock")?;
    ctl(&home, &["setup", "--name", "svc", "--llama-server", MOCK, "--model", gguf.to_str().unwrap(), "--port", "0"]).await?;
    let paths = Paths::new(&home);
    let daemon = Daemon::start(NodeConfig::load(&paths.config())?, paths.clone()).await?;
    assert!(control::reachable(&paths.socket()).await);

    // `ymirctl service status --unit-dir` only looks at the file; with nothing installed it says so.
    let units = dir.path().join("units");
    let out = ctl(&home, &["service", "status", "--unit-dir", units.to_str().unwrap()]).await?;
    assert!(out.starts_with("not installed"), "{out}");

    let r = control::request(&paths.socket(), &ControlRequest::Shutdown).await?;
    assert!(r.ok, "{r:?}");
    tokio::time::timeout(Duration::from_secs(5), daemon.exit_requested()).await?;
    daemon.shutdown().await;
    assert!(!control::reachable(&paths.socket()).await, "the control endpoint closes with ymird");
    Ok(())
}

#[test]
fn pipe_names_are_per_home_and_stable() {
    let a = control::pipe_name(Path::new("/x/a/state/ymird.sock"));
    assert!(a.starts_with(r"\\.\pipe\ymird-") && a.len() == r"\\.\pipe\ymird-".len() + 16, "{a}");
    assert_eq!(a, control::pipe_name(Path::new("/x/a/state/ymird.sock")));
    assert_ne!(a, control::pipe_name(Path::new("/x/b/state/ymird.sock")));
    // Windows spellings of one path name one pipe: case, separators and the `\\?\` prefix.
    #[cfg(windows)]
    {
        assert_eq!(control::pipe_name(Path::new(r"C:\Y\state\ymird.sock")), control::pipe_name(Path::new(r"\\?\c:\y\state\ymird.sock")));
        assert_eq!(control::pipe_name(Path::new(r"C:\Y\state\ymird.sock")), control::pipe_name(Path::new("C:/Y/state/ymird.sock")));
    }
}
