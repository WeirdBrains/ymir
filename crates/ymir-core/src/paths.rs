//! Where a node keeps its config and state.

use std::path::PathBuf;

/// `$YMIR_HOME`, else `~/.config/ymir` (user install). Holds `node.toml` and `state/`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Paths {
    pub home: PathBuf,
}

impl Paths {
    pub fn new(home: impl Into<PathBuf>) -> Self {
        Self { home: home.into() }
    }

    pub fn from_env() -> Self {
        let home = std::env::var_os("YMIR_HOME")
            .filter(|h| !h.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| dirs::config_dir().unwrap_or_else(|| PathBuf::from(".")).join("ymir"));
        Self { home }
    }

    pub fn config(&self) -> PathBuf {
        self.home.join("node.toml")
    }

    /// Private state: the bridge token, the control socket and engine logs (mode 0700).
    pub fn state(&self) -> PathBuf {
        self.home.join("state")
    }

    pub fn socket(&self) -> PathBuf {
        self.state().join("ymird.sock")
    }

    pub fn token(&self) -> PathBuf {
        self.state().join("bridge.token")
    }

    /// This device's Weirdnet identity and roster (mode 0700, secrets 0600).
    pub fn pool(&self) -> PathBuf {
        self.state().join("pool")
    }

    pub fn logs(&self) -> PathBuf {
        self.state().join("logs")
    }

    /// Engine builds Ymir manages (`engines/llama.cpp/llama-server`).
    pub fn engines(&self) -> PathBuf {
        self.home.join("engines")
    }

    /// Creates `state/` with mode 0700.
    pub fn ensure_state(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(self.state())?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(self.state(), std::fs::Permissions::from_mode(0o700))?;
        }
        Ok(())
    }
}
