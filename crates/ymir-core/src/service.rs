//! Runs `ymird` as a user-level service: a systemd user unit on Linux, a launchd agent on macOS.
//! No root, no system units. The Windows service (`NT SERVICE\ymird`) comes with the WiX
//! installer.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

pub const SYSTEMD_UNIT: &str = "ymird.service";
pub const LAUNCHD_LABEL: &str = "com.weirdbrains.ymird";

/// What the service runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceSpec {
    pub ymird: PathBuf,
    pub home: PathBuf,
    /// Start through `ymirctl update boot --exec`, so a staged update runs on trial and rolls back
    /// if it does not come up. `None`: run `ymird` directly.
    pub ymirctl: Option<PathBuf>,
}

/// The command line the service runs.
pub fn command(spec: &ServiceSpec) -> Vec<String> {
    let s = |p: &Path| p.to_string_lossy().into_owned();
    match &spec.ymirctl {
        None => vec![s(&spec.ymird), "--home".into(), s(&spec.home)],
        Some(ctl) => vec![
            s(ctl),
            "--home".into(),
            s(&spec.home),
            "update".into(),
            "boot".into(),
            "--exec".into(),
            "--installed".into(),
            s(&spec.ymird),
            "--".into(),
            "--home".into(),
            s(&spec.home),
        ],
    }
}

/// Quotes a path for a systemd `ExecStart=` line.
fn sd_quote(p: &Path) -> String {
    let s = p.to_string_lossy();
    if s.chars().any(|c| c.is_whitespace() || c == '"' || c == '\\' || c == '%') {
        format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"").replace('%', "%%"))
    } else {
        s.into_owned()
    }
}

/// The systemd user unit.
pub fn systemd_unit(spec: &ServiceSpec) -> String {
    format!(
        "# Written by `ymirctl service install`. Ymir node daemon, user level.\n\
[Unit]\n\
Description=Ymir node (model, inference and app roles)\n\
Documentation=https://github.com/WeirdBrains/ymir\n\
StartLimitIntervalSec=600\n\
StartLimitBurst=5\n\
\n\
[Service]\n\
Type=simple\n\
ExecStart={}\n\
Restart=on-failure\n\
RestartSec=5\n\
KillMode=mixed\n\
TimeoutStopSec=30\n\
Environment=YMIR_LOG=info\n\
NoNewPrivileges=yes\n\
\n\
[Install]\n\
WantedBy=default.target\n",
        command(spec).iter().map(|a| sd_quote(Path::new(a))).collect::<Vec<_>>().join(" ")
    )
}

fn xml(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// The launchd agent plist.
pub fn launchd_plist(spec: &ServiceSpec) -> String {
    let log = spec.home.join("state").join("logs").join("ymird.log");
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{LAUNCHD_LABEL}</string>
  <key>ProgramArguments</key>
  <array>{}</array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>ProcessType</key><string>Background</string>
  <key>StandardErrorPath</key><string>{}</string>
  <key>StandardOutPath</key><string>{}</string>
</dict>
</plist>
"#,
        command(spec).iter().map(|a| format!("<string>{}</string>", xml(a))).collect::<String>(),
        xml(&log.to_string_lossy()),
        xml(&log.to_string_lossy())
    )
}

/// Where this OS keeps user services.
pub fn default_dir() -> Option<PathBuf> {
    if cfg!(target_os = "macos") {
        dirs::home_dir().map(|h| h.join("Library").join("LaunchAgents"))
    } else if cfg!(target_os = "linux") {
        dirs::config_dir().map(|c| c.join("systemd").join("user"))
    } else {
        None
    }
}

fn file_name() -> &'static str {
    if cfg!(target_os = "macos") { "com.weirdbrains.ymird.plist" } else { SYSTEMD_UNIT }
}

/// The service file for this OS.
pub fn render(spec: &ServiceSpec) -> Result<String> {
    if cfg!(target_os = "macos") {
        Ok(launchd_plist(spec))
    } else if cfg!(target_os = "linux") {
        Ok(systemd_unit(spec))
    } else {
        bail!("user services are set up by the installer on this OS (Windows: the WiX installer registers NT SERVICE\\ymird)")
    }
}

/// `ymird` next to the running executable, else on `PATH`.
pub fn find_ymird() -> Option<PathBuf> {
    let exe = if cfg!(windows) { "ymird.exe" } else { "ymird" };
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join(exe)))
        .filter(|p| p.is_file())
        .or_else(|| weirdnet_backends::find_in_path(exe))
}

#[derive(Debug, Clone, Default)]
pub struct InstallOptions {
    /// Write here instead of the OS's user-service directory (and never start it).
    pub dir: Option<PathBuf>,
    /// Enable and start it now.
    pub start: bool,
    /// Print what would happen; change nothing.
    pub dry_run: bool,
}

/// Writes the user service and, with `start` (and no custom `dir`), enables and starts it.
/// Returns what it did, line by line.
pub fn install(spec: &ServiceSpec, opts: &InstallOptions) -> Result<Vec<String>> {
    let text = render(spec)?;
    let custom = opts.dir.is_some();
    let dir = opts.dir.clone().or_else(default_dir).context("no user-service directory on this OS")?;
    let path = dir.join(file_name());
    let mut log = Vec::new();
    if opts.dry_run {
        log.push(format!("would write {}:", path.display()));
        log.extend(text.lines().map(|l| format!("  {l}")));
        if opts.start && !custom {
            log.extend(start_commands().into_iter().map(|c| format!("would run: {}", c.join(" "))));
        }
        return Ok(log);
    }
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, &text)?;
    std::fs::rename(&tmp, &path)?;
    log.push(format!("wrote {}", path.display()));
    if opts.start && !custom {
        for c in start_commands() {
            run(&c)?;
            log.push(format!("ran: {}", c.join(" ")));
        }
    } else if !custom {
        log.push(format!("start it with: {}", start_commands().iter().map(|c| c.join(" ")).collect::<Vec<_>>().join(" && ")));
    }
    if cfg!(target_os = "linux") && !lingering() {
        log.push("note: to keep ymird running after you log out (and start it at boot), run `loginctl enable-linger $USER` once".into());
    }
    Ok(log)
}

fn start_commands() -> Vec<Vec<String>> {
    let v = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    if cfg!(target_os = "macos") {
        let plist = default_dir().map(|d| d.join(file_name()).to_string_lossy().into_owned()).unwrap_or_default();
        vec![v(&["launchctl", "bootstrap", &format!("gui/{}", uid()), &plist])]
    } else {
        vec![v(&["systemctl", "--user", "daemon-reload"]), v(&["systemctl", "--user", "enable", "--now", SYSTEMD_UNIT])]
    }
}

fn stop_commands() -> Vec<Vec<String>> {
    let v = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    if cfg!(target_os = "macos") {
        vec![v(&["launchctl", "bootout", &format!("gui/{}/{LAUNCHD_LABEL}", uid())])]
    } else {
        vec![v(&["systemctl", "--user", "disable", "--now", SYSTEMD_UNIT])]
    }
}

#[cfg(unix)]
fn uid() -> u32 {
    // SAFETY: getuid has no preconditions.
    unsafe { libc::getuid() }
}

#[cfg(not(unix))]
fn uid() -> u32 {
    0
}

fn run(cmd: &[String]) -> Result<()> {
    let status = Command::new(&cmd[0]).args(&cmd[1..]).status().with_context(|| format!("running {}", cmd[0]))?;
    if !status.success() {
        bail!("`{}` failed ({status})", cmd.join(" "));
    }
    Ok(())
}

/// Whether systemd keeps this user's services running without a login session.
pub fn lingering() -> bool {
    let user = std::env::var("USER").unwrap_or_default();
    Command::new("loginctl")
        .args(["show-user", &user, "-p", "Linger"])
        .output()
        .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).trim() == "Linger=yes")
}

/// Stops, disables and removes the user service.
pub fn uninstall(dir: Option<&Path>, dry_run: bool) -> Result<Vec<String>> {
    let custom = dir.is_some();
    let dir = dir.map(Path::to_path_buf).or_else(default_dir).context("no user-service directory on this OS")?;
    let path = dir.join(file_name());
    let mut log = Vec::new();
    if !custom {
        for c in stop_commands() {
            if dry_run {
                log.push(format!("would run: {}", c.join(" ")));
            } else if run(&c).is_ok() {
                log.push(format!("ran: {}", c.join(" ")));
            }
        }
    }
    if path.exists() {
        if dry_run {
            log.push(format!("would remove {}", path.display()));
        } else {
            std::fs::remove_file(&path)?;
            log.push(format!("removed {}", path.display()));
        }
    } else {
        log.push(format!("{} is not installed", path.display()));
    }
    Ok(log)
}

/// Where the service file is, and what the service manager says.
pub fn status(dir: Option<&Path>) -> Result<Vec<String>> {
    let custom = dir.is_some();
    let dir = dir.map(Path::to_path_buf).or_else(default_dir).context("no user-service directory on this OS")?;
    let path = dir.join(file_name());
    let mut log =
        vec![if path.exists() { format!("installed: {}", path.display()) } else { format!("not installed ({} is missing)", path.display()) }];
    if !custom && cfg!(target_os = "linux") {
        let out = Command::new("systemctl").args(["--user", "is-active", SYSTEMD_UNIT]).output();
        if let Ok(o) = out {
            log.push(format!("systemd: {}", String::from_utf8_lossy(&o.stdout).trim()));
        }
        log.push(format!("lingering: {}", if lingering() { "yes" } else { "no (`loginctl enable-linger $USER` keeps it running after logout)" }));
    }
    Ok(log)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> ServiceSpec {
        ServiceSpec { ymird: "/opt/ymir/bin/ymird".into(), home: "/home/u/My Config/ymir".into(), ymirctl: None }
    }

    #[test]
    fn managed_updates_start_through_ymirctl_boot() {
        let mut sp = spec();
        sp.ymirctl = Some("/opt/ymir/bin/ymirctl".into());
        let unit = systemd_unit(&sp);
        assert!(
            unit.contains(
                "ExecStart=/opt/ymir/bin/ymirctl --home \"/home/u/My Config/ymir\" update boot --exec --installed /opt/ymir/bin/ymird -- --home \"/home/u/My Config/ymir\"\n"
            ),
            "{unit}"
        );
        assert!(launchd_plist(&sp).contains("<string>update</string><string>boot</string><string>--exec</string>"));
    }

    #[test]
    fn unit_text() {
        let u = systemd_unit(&spec());
        assert!(u.contains("ExecStart=/opt/ymir/bin/ymird --home \"/home/u/My Config/ymir\"\n"), "{u}");
        assert!(u.contains("Restart=on-failure") && u.contains("WantedBy=default.target"));
        let p = launchd_plist(&ServiceSpec { ymird: "/opt/a&b/ymird".into(), home: "/h".into(), ymirctl: None });
        assert!(p.contains("<string>/opt/a&amp;b/ymird</string>"));
        assert!(p.contains(LAUNCHD_LABEL));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn install_into_a_custom_dir_never_starts() {
        let dir = tempfile::tempdir().unwrap();
        let opts = InstallOptions { dir: Some(dir.path().into()), start: true, dry_run: true };
        let log = install(&spec(), &opts).unwrap();
        assert!(log.iter().all(|l| !l.contains("systemctl")), "{log:?}");
        assert!(!dir.path().join(SYSTEMD_UNIT).exists(), "dry run writes nothing");
        let log = install(&spec(), &InstallOptions { dry_run: false, ..opts }).unwrap();
        assert!(log[0].starts_with("wrote "));
        assert!(dir.path().join(SYSTEMD_UNIT).is_file());
        assert!(status(Some(dir.path())).unwrap()[0].starts_with("installed"));
        uninstall(Some(dir.path()), false).unwrap();
        assert!(!dir.path().join(SYSTEMD_UNIT).exists());
    }
}
