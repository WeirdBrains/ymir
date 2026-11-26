//! Runs `ymird` as a background service on Linux, macOS, Windows and FreeBSD.
//!
//! User level by default (no root, no admin):
//! - Linux: a systemd user unit (`~/.config/systemd/user/ymird.service`).
//! - macOS: a launchd agent (`~/Library/LaunchAgents/com.weirdbrains.ymird.plist`), loaded in
//!   the login session's `gui/<uid>` domain.
//! - Windows: a Task Scheduler task (`\Ymir\ymird`) that starts at logon, runs `ymird` under a
//!   headless console host (no window) and restarts after a failure.
//!
//! `--system` (root or an elevated prompt) installs a machine service instead: a systemd system
//! unit with `User=`, a launchd daemon with `UserName`, or a Windows service running as its own
//! virtual account `NT SERVICE\ymird` (the WiX installer registers the same service).
//!
//! FreeBSD has no per-user services, so there it is `--system` only: an `rc.d` script
//! (`/usr/local/etc/rc.d/ymird`) that runs `ymird` as `--run-as` under `daemon(8)`, which restarts
//! it after a crash, enabled with `sysrc ymird_enable=YES`.
//!
//! Everything here is a [`Plan`] (files to write, commands to run) built for an explicit [`Os`],
//! so the units, plists, task XML and command lines are tested on every OS.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

pub const SYSTEMD_UNIT: &str = "ymird.service";
pub const LAUNCHD_LABEL: &str = "com.weirdbrains.ymird";
pub const PLIST_FILE: &str = "com.weirdbrains.ymird.plist";
/// The Task Scheduler task (user level, Windows).
pub const TASK_NAME: &str = r"\Ymir\ymird";
pub const TASK_FILE: &str = "ymird-task.xml";
/// The Windows service (system level) and its virtual account.
pub const WINDOWS_SERVICE: &str = "ymird";
pub const WINDOWS_ACCOUNT: &str = r"NT SERVICE\ymird";
/// What `--system` with `--unit-dir` writes on Windows: the `sc.exe` commands, to run elevated.
pub const WINDOWS_SCRIPT: &str = "ymird-service.cmd";
const DESCRIPTION: &str = "Ymir node (model, inference and app roles)";
/// The FreeBSD rc.d script (and service name).
pub const RC_SCRIPT: &str = "ymird";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Os {
    Linux,
    Macos,
    Windows,
    FreeBsd,
}

impl Os {
    /// This build's OS. Other Unixes use the Linux (systemd) rules.
    pub fn current() -> Self {
        if cfg!(target_os = "macos") {
            Os::Macos
        } else if cfg!(windows) {
            Os::Windows
        } else if cfg!(target_os = "freebsd") {
            Os::FreeBsd
        } else {
            Os::Linux
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Scope {
    /// This user's service manager (the default).
    #[default]
    User,
    /// The machine's: root or an elevated prompt.
    System,
}

/// What the service runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceSpec {
    pub ymird: PathBuf,
    pub home: PathBuf,
    /// Start through `ymirctl update boot --exec`, so a staged update runs on trial and rolls back
    /// if it does not come up. `None`: run `ymird` directly.
    pub ymirctl: Option<PathBuf>,
    pub scope: Scope,
    /// System scope on Linux and macOS: the account `ymird` runs as (never root).
    pub run_as: Option<String>,
}

impl ServiceSpec {
    pub fn user(ymird: impl Into<PathBuf>, home: impl Into<PathBuf>) -> Self {
        Self { ymird: ymird.into(), home: home.into(), ymirctl: None, scope: Scope::User, run_as: None }
    }

    fn log_file(&self, os: Os) -> PathBuf {
        join(os, &join(os, &join(os, &self.home, "state"), "logs"), "ymird.log")
    }
}

/// Joins with the target OS's separator (paths for Windows are built on Linux in tests).
fn join(os: Os, base: &Path, name: &str) -> PathBuf {
    let b = base.to_string_lossy();
    let sep = if os == Os::Windows { '\\' } else { '/' };
    if b.ends_with(['/', '\\']) { PathBuf::from(format!("{b}{name}")) } else { PathBuf::from(format!("{b}{sep}{name}")) }
}

fn s(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

/// The command line the service runs (program first).
pub fn command(spec: &ServiceSpec) -> Vec<String> {
    command_for(Os::current(), spec)
}

/// The command line for `os`. Windows adds `--log-file` (no journal there) and, for the Windows
/// service, `--windows-service` so `ymird` talks to the service control manager.
pub fn command_for(os: Os, spec: &ServiceSpec) -> Vec<String> {
    let mut daemon_args = vec!["--home".to_string(), s(&spec.home)];
    if os == Os::Windows {
        if spec.scope == Scope::System {
            daemon_args.insert(0, "--windows-service".into());
        }
        daemon_args.extend(["--log-file".into(), s(&spec.log_file(os))]);
    }
    match &spec.ymirctl {
        None => [vec![s(&spec.ymird)], daemon_args].concat(),
        Some(ctl) => [
            vec![
                s(ctl),
                "--home".into(),
                s(&spec.home),
                "update".into(),
                "boot".into(),
                "--exec".into(),
                "--installed".into(),
                s(&spec.ymird),
                "--".into(),
            ],
            daemon_args,
        ]
        .concat(),
    }
}

/// Quotes a path for a systemd `ExecStart=` line.
fn sd_quote(p: &str) -> String {
    if p.chars().any(|c| c.is_whitespace() || c == '"' || c == '\\' || c == '%') {
        format!("\"{}\"", p.replace('\\', "\\\\").replace('"', "\\\"").replace('%', "%%"))
    } else {
        p.to_string()
    }
}

/// Quotes one argument the way `CommandLineToArgvW` (and the MSVC runtime) splits it.
pub fn win_quote(a: &str) -> String {
    if !a.is_empty() && !a.chars().any(|c| c == ' ' || c == '\t' || c == '"') {
        return a.to_string();
    }
    let mut out = String::from("\"");
    let mut backslashes = 0;
    for c in a.chars() {
        match c {
            '\\' => backslashes += 1,
            '"' => {
                out.push_str(&"\\".repeat(backslashes * 2 + 1));
                out.push('"');
                backslashes = 0;
            }
            _ => {
                out.push_str(&"\\".repeat(backslashes));
                out.push(c);
                backslashes = 0;
            }
        }
    }
    out.push_str(&"\\".repeat(backslashes * 2));
    out.push('"');
    out
}

/// A Windows command line from arguments.
pub fn win_command_line(args: &[String]) -> String {
    args.iter().map(|a| win_quote(a)).collect::<Vec<_>>().join(" ")
}

/// The systemd unit: a user unit, or with system scope a system unit running as `run_as`.
pub fn systemd_unit(spec: &ServiceSpec) -> String {
    let (user, wanted, hardening) = match spec.scope {
        Scope::User => (String::new(), "default.target", ""),
        Scope::System => {
            (format!("User={}\n", spec.run_as.as_deref().unwrap_or("ymir")), "multi-user.target", "ProtectSystem=full\nPrivateTmp=yes\n")
        }
    };
    let after = if spec.scope == Scope::System { "After=network-online.target\nWants=network-online.target\n" } else { "" };
    let level = if spec.scope == Scope::System { "system level" } else { "user level" };
    format!(
        "# Written by `ymirctl service install`. Ymir node daemon, {level}.\n\
[Unit]\n\
Description={DESCRIPTION}\n\
Documentation=https://github.com/WeirdBrains/ymir\n\
{after}\
StartLimitIntervalSec=600\n\
StartLimitBurst=5\n\
\n\
[Service]\n\
Type=simple\n\
{user}\
ExecStart={}\n\
Restart=on-failure\n\
RestartSec=5\n\
KillMode=mixed\n\
TimeoutStopSec=30\n\
Environment=YMIR_LOG=info\n\
NoNewPrivileges=yes\n\
{hardening}\
\n\
[Install]\n\
WantedBy={wanted}\n",
        command_for(Os::Linux, spec).iter().map(|a| sd_quote(a)).collect::<Vec<_>>().join(" ")
    )
}

fn xml(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;").replace('\'', "&apos;")
}

/// launchd's default `PATH` is only the system directories: add Homebrew's, so engines found by
/// name (vLLM, a Homebrew llama.cpp) resolve the same as in a shell.
const LAUNCHD_PATH: &str = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin";

/// The launchd plist: a LaunchAgent (user), or a LaunchDaemon running as `run_as` (system).
pub fn launchd_plist(spec: &ServiceSpec) -> String {
    let log = s(&spec.log_file(Os::Macos));
    let user = match (spec.scope, &spec.run_as) {
        (Scope::System, Some(u)) => format!("  <key>UserName</key><string>{}</string>\n", xml(u)),
        _ => String::new(),
    };
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<!-- Written by `ymirctl service install`. Ymir node daemon. -->
<plist version="1.0">
<dict>
  <key>Label</key><string>{LAUNCHD_LABEL}</string>
  <key>ProgramArguments</key>
  <array>{}</array>
{user}  <key>WorkingDirectory</key><string>{}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>YMIR_LOG</key><string>info</string>
    <key>PATH</key><string>{LAUNCHD_PATH}</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>ExitTimeOut</key><integer>30</integer>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardErrorPath</key><string>{}</string>
  <key>StandardOutPath</key><string>{}</string>
</dict>
</plist>
"#,
        command_for(Os::Macos, spec).iter().map(|a| format!("<string>{}</string>", xml(a))).collect::<String>(),
        xml(&s(&spec.home)),
        xml(&log),
        xml(&log)
    )
}

/// Quotes an argument for `sh` (the rc.d script).
fn sh_quote(a: &str) -> String {
    if !a.is_empty() && a.chars().all(|c| c.is_ascii_alphanumeric() || "/._-=:+,@%".contains(c)) {
        a.to_string()
    } else {
        format!("'{}'", a.replace('\'', r"'\''"))
    }
}

/// The FreeBSD rc.d script (system scope): `daemon(8)` runs `ymird` as `run_as`, restarts it 5 s
/// after it exits and writes its output to the node's log; `service ymird stop` stops both.
pub fn rcd_script(spec: &ServiceSpec) -> String {
    let user = spec.run_as.as_deref().unwrap_or("ymir");
    let cmd = command_for(Os::FreeBsd, spec).iter().map(|a| sh_quote(a)).collect::<Vec<_>>().join(" ");
    format!(
        "#!/bin/sh\n\
# Written by `ymirctl service install --system`. {DESCRIPTION}.\n\
#\n\
# PROVIDE: {RC_SCRIPT}\n\
# REQUIRE: LOGIN NETWORKING\n\
# KEYWORD: shutdown\n\
#\n\
# Add ymird_enable=\"YES\" to /etc/rc.conf (sysrc ymird_enable=YES) to start it at boot.\n\
\n\
. /etc/rc.subr\n\
\n\
name=\"{RC_SCRIPT}\"\n\
rcvar=\"ymird_enable\"\n\
load_rc_config $name\n\
: ${{ymird_enable:=\"NO\"}}\n\
\n\
pidfile=\"/var/run/${{name}}.pid\"\n\
procname=\"/usr/sbin/daemon\"\n\
command=\"/usr/sbin/daemon\"\n\
command_args=\"-r -R 5 -P ${{pidfile}} -o {} -u {} -- /usr/bin/env YMIR_LOG=info {}\"\n\
\n\
run_rc_command \"$1\"\n",
        sh_quote(&s(&spec.log_file(Os::FreeBsd))),
        sh_quote(user),
        cmd.replace('"', "\\\""),
    )
}

/// The Task Scheduler definition (user level, Windows): at `user`'s logon, run `ymird` through
/// `conhost.exe --headless` so no console window opens, never time out, restart after a failure,
/// and keep running on battery.
pub fn task_xml(spec: &ServiceSpec, user: &str, system_root: &str) -> String {
    let cmd = command_for(Os::Windows, spec);
    let conhost = format!(r"{}\System32\conhost.exe", system_root.trim_end_matches('\\'));
    let args = format!("--headless {}", win_command_line(&cmd));
    format!(
        r#"<?xml version="1.0" encoding="UTF-16"?>
<!-- Written by `ymirctl service install`. Ymir node daemon, user level. -->
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>{DESCRIPTION}. https://github.com/WeirdBrains/ymir</Description>
    <URI>{}</URI>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
      <UserId>{user}</UserId>
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>{user}</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>5</Priority>
    <RestartOnFailure>
      <Interval>PT1M</Interval>
      <Count>999</Count>
    </RestartOnFailure>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{}</Command>
      <Arguments>{}</Arguments>
      <WorkingDirectory>{}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"#,
        xml(TASK_NAME),
        xml(&conhost),
        xml(&args),
        xml(&s(&spec.home)),
        user = xml(user),
    )
}

/// UTF-16LE with a byte-order mark: what `schtasks /Create /XML` reads reliably.
pub fn utf16_with_bom(text: &str) -> Vec<u8> {
    let mut out = vec![0xFF, 0xFE];
    for u in text.replace("\r\n", "\n").replace('\n', "\r\n").encode_utf16() {
        out.extend(u.to_le_bytes());
    }
    out
}

/// One step of a plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    pub cmd: Vec<String>,
    /// Failure is expected sometimes (stopping what isn't running) and is not an error.
    pub may_fail: bool,
}

impl Step {
    fn run(a: &[&str]) -> Self {
        Self { cmd: a.iter().map(|s| s.to_string()).collect(), may_fail: false }
    }
    fn try_run(a: &[&str]) -> Self {
        Self { may_fail: true, ..Self::run(a) }
    }
    fn owned(cmd: Vec<String>) -> Self {
        Self { cmd, may_fail: false }
    }
    pub fn line(&self) -> String {
        self.cmd.iter().map(|a| if a.contains(' ') { format!("\"{a}\"") } else { a.clone() }).collect::<Vec<_>>().join(" ")
    }
}

/// What a service action does: directories to create, files to write, then commands to run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Plan {
    pub dirs: Vec<PathBuf>,
    pub files: Vec<(PathBuf, Vec<u8>)>,
    pub remove: Vec<PathBuf>,
    pub steps: Vec<Step>,
    pub notes: Vec<String>,
}

/// Where the service lives and how to address it, for one OS and scope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Target {
    pub os: Os,
    pub scope: Scope,
    /// The service file's directory (`--unit-dir`, else the OS default). Windows user scope keeps
    /// the task XML in the node's state directory; Windows system scope has no file.
    pub dir: Option<PathBuf>,
    /// `--unit-dir` was given: write files only, never touch the service manager.
    pub custom: bool,
    /// macOS: the numeric user id for the `gui/<uid>` domain.
    pub uid: u32,
}

impl Target {
    /// The default for this machine.
    pub fn current(scope: Scope, unit_dir: Option<&Path>, home: &Path) -> Result<Self> {
        let os = Os::current();
        let custom = unit_dir.is_some();
        let dir = match unit_dir {
            Some(d) => Some(d.to_path_buf()),
            None => default_dir(os, scope, home),
        };
        if os == Os::FreeBsd && scope == Scope::User && !custom {
            bail!("FreeBSD has no per-user services: install it with --system --run-as USER (an rc.d script)");
        }
        if dir.is_none() && !(os == Os::Windows && scope == Scope::System) {
            bail!("no service directory for this OS and user; pass --unit-dir");
        }
        Ok(Self { os, scope, dir, custom, uid: uid() })
    }

    pub fn file(&self) -> Option<PathBuf> {
        let name = match (self.os, self.scope) {
            (Os::Linux, _) => SYSTEMD_UNIT,
            (Os::Macos, _) => PLIST_FILE,
            (Os::Windows, Scope::User) => TASK_FILE,
            (Os::Windows, Scope::System) if self.custom => WINDOWS_SCRIPT,
            (Os::Windows, Scope::System) => return None,
            (Os::FreeBsd, _) => RC_SCRIPT,
        };
        self.dir.as_ref().map(|d| join(self.os, d, name))
    }

    fn launchd_domain(&self) -> String {
        if self.scope == Scope::System { "system".into() } else { format!("gui/{}", self.uid) }
    }

    fn systemctl(&self, args: &[&str]) -> Step {
        let mut v = vec!["systemctl"];
        if self.scope == Scope::User {
            v.push("--user");
        }
        v.extend(args);
        Step::run(&v)
    }
}

/// Where this OS keeps services of `scope` (`home` is the node's directory: the Windows task XML
/// is kept in its state directory).
pub fn default_dir(os: Os, scope: Scope, home: &Path) -> Option<PathBuf> {
    match (os, scope) {
        (Os::Linux, Scope::User) => dirs::config_dir().map(|c| c.join("systemd").join("user")),
        (Os::Linux, Scope::System) => Some("/etc/systemd/system".into()),
        (Os::Macos, Scope::User) => dirs::home_dir().map(|h| h.join("Library").join("LaunchAgents")),
        (Os::Macos, Scope::System) => Some("/Library/LaunchDaemons".into()),
        (Os::Windows, Scope::User) => Some(join(os, home, "state")),
        (Os::Windows, Scope::System) => None,
        (Os::FreeBsd, Scope::User) => None,
        (Os::FreeBsd, Scope::System) => Some("/usr/local/etc/rc.d".into()),
    }
}

/// The default node directory for a system service: `%ProgramData%\Ymir`,
/// `/Library/Application Support/Ymir`, `/var/lib/ymir` or (FreeBSD) `/var/db/ymir`.
pub fn system_home(os: Os) -> PathBuf {
    match os {
        Os::Windows => {
            let pd = std::env::var("ProgramData").unwrap_or_else(|_| r"C:\ProgramData".into());
            join(os, Path::new(&pd), "Ymir")
        }
        Os::Macos => "/Library/Application Support/Ymir".into(),
        Os::Linux => "/var/lib/ymir".into(),
        Os::FreeBsd => "/var/db/ymir".into(),
    }
}

/// Facts about the machine an install plan needs, gathered at run time (fixed in tests).
#[derive(Debug, Clone, Default)]
pub struct Host {
    /// Windows: `DOMAIN\user` for the task's logon trigger.
    pub windows_user: String,
    /// Windows: `%SystemRoot%`.
    pub system_root: String,
    /// Windows system scope: the service is already registered (reconfigure, don't create).
    pub service_exists: bool,
    /// Linux user scope: systemd keeps this user's services running after logout.
    pub lingering: bool,
}

impl Host {
    pub fn detect(target: &Target) -> Self {
        let env = |k: &str| std::env::var(k).unwrap_or_default();
        let user = match (env("USERDOMAIN"), env("USERNAME")) {
            (d, u) if !d.is_empty() && !u.is_empty() => format!(r"{d}\{u}"),
            (_, u) => u,
        };
        let root = std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".into());
        let exists = target.os == Os::Windows
            && target.scope == Scope::System
            && !target.custom
            && Command::new("sc.exe").args(["query", WINDOWS_SERVICE]).output().is_ok_and(|o| o.status.success());
        let lingering = target.os == Os::Linux && target.scope == Scope::User && lingering();
        Self { windows_user: user, system_root: root, service_exists: exists, lingering }
    }
}

/// The `sc.exe` commands that register (or reconfigure) the Windows service.
fn windows_service_steps(spec: &ServiceSpec, exists: bool) -> Vec<Step> {
    let bin = win_command_line(&command_for(Os::Windows, spec));
    let verb = if exists { "config" } else { "create" };
    let mut v: Vec<Step> = vec![Step::owned(
        ["sc.exe", verb, WINDOWS_SERVICE, "binPath=", &bin, "start=", "auto", "obj=", WINDOWS_ACCOUNT, "DisplayName=", "Ymir node"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
    )];
    v.push(Step::run(&["sc.exe", "sidtype", WINDOWS_SERVICE, "unrestricted"]));
    v.push(Step::run(&["sc.exe", "description", WINDOWS_SERVICE, DESCRIPTION]));
    // Restart after a crash or an error exit: 60 s, every time; the count resets after a day.
    v.push(Step::run(&["sc.exe", "failure", WINDOWS_SERVICE, "reset=", "86400", "actions=", "restart/60000/restart/60000/restart/60000"]));
    v.push(Step::run(&["sc.exe", "failureflag", WINDOWS_SERVICE, "1"]));
    // The node directory: protected DACL for SYSTEM, Administrators and the service account only.
    let home = s(&spec.home);
    v.push(Step::owned(
        [
            "icacls",
            &home,
            "/inheritance:r",
            "/grant:r",
            "*S-1-5-18:(OI)(CI)F",
            "*S-1-5-32-544:(OI)(CI)F",
            &format!("{WINDOWS_ACCOUNT}:(OI)(CI)F"),
            "/Q",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect(),
    ));
    v.push(Step { cmd: ["icacls", &format!(r"{home}\*"), "/reset", "/T", "/C", "/Q"].iter().map(|s| s.to_string()).collect(), may_fail: true });
    v
}

/// Writes the service and, with `start`, enables and starts it.
pub fn install_plan(spec: &ServiceSpec, t: &Target, host: &Host, start: bool) -> Result<Plan> {
    if spec.scope == Scope::System && t.os != Os::Windows && spec.run_as.as_deref().is_none_or(|u| u.is_empty() || u == "root") {
        bail!("a system service needs --run-as USER (an account that owns the node directory; never root)");
    }
    if t.os == Os::Windows && spec.scope == Scope::System && spec.ymirctl.is_some() {
        bail!("--managed-updates isn't available for the Windows service yet (the service control manager starts ymird itself)");
    }
    let mut p = Plan::default();
    let logs = join(t.os, &join(t.os, &spec.home, "state"), "logs");
    p.dirs.push(logs);
    let file = t.file();
    match (t.os, spec.scope) {
        (Os::Linux, _) => p.files.push((file.clone().unwrap(), systemd_unit(spec).into_bytes())),
        (Os::Macos, _) => p.files.push((file.clone().unwrap(), launchd_plist(spec).into_bytes())),
        (Os::FreeBsd, Scope::User) => bail!("FreeBSD has no per-user services: install it with --system --run-as USER (an rc.d script)"),
        (Os::FreeBsd, Scope::System) => p.files.push((file.clone().unwrap(), rcd_script(spec).into_bytes())),
        (Os::Windows, Scope::User) => {
            if host.windows_user.is_empty() {
                bail!("can't tell the current Windows user (USERNAME is not set)");
            }
            p.files.push((file.clone().unwrap(), utf16_with_bom(&task_xml(spec, &host.windows_user, &host.system_root))))
        }
        (Os::Windows, Scope::System) => {
            if let Some(f) = &file {
                let mut script = String::from("@echo off\r\nrem Written by `ymirctl service install --system`. Run from an elevated prompt.\r\n");
                for st in windows_service_steps(spec, false) {
                    script.push_str(&win_command_line(&st.cmd));
                    script.push_str("\r\n");
                }
                script.push_str(&format!("sc.exe start {WINDOWS_SERVICE}\r\n"));
                p.files.push((f.clone(), script.into_bytes()));
            }
        }
    }
    if t.custom {
        p.notes.push("written to --unit-dir only: nothing was registered or started".into());
        return Ok(p);
    }
    if spec.scope == Scope::System && t.os != Os::Windows {
        let owner = spec.run_as.clone().unwrap_or_default();
        p.steps.push(Step::owned(vec!["chown".into(), "-R".into(), owner, s(&spec.home)]));
    }
    match t.os {
        Os::Linux => {
            p.steps.push(t.systemctl(&["daemon-reload"]));
            if start {
                p.steps.push(t.systemctl(&["enable", SYSTEMD_UNIT]));
                p.steps.push(t.systemctl(&["restart", SYSTEMD_UNIT]));
            }
        }
        Os::Macos => {
            // Reinstall: unload the old definition first (fails harmlessly when none is loaded).
            p.steps.push(Step::try_run(&["launchctl", "bootout", &format!("{}/{LAUNCHD_LABEL}", t.launchd_domain())]));
            if start {
                p.steps.push(Step::run(&["launchctl", "bootstrap", &t.launchd_domain(), &s(file.as_ref().unwrap())]));
            }
        }
        Os::FreeBsd => {
            p.steps.push(Step::run(&["chmod", "555", &s(file.as_ref().unwrap())]));
            if start {
                p.steps.push(Step::run(&["sysrc", "ymird_enable=YES"]));
                p.steps.push(Step::run(&["service", RC_SCRIPT, "restart"]));
            }
        }
        Os::Windows if spec.scope == Scope::User => {
            p.steps.push(Step::run(&["schtasks", "/Create", "/TN", TASK_NAME, "/XML", &s(file.as_ref().unwrap()), "/F"]));
            if start {
                p.steps.push(Step::run(&["schtasks", "/Run", "/TN", TASK_NAME]));
            }
        }
        Os::Windows => {
            p.steps.extend(windows_service_steps(spec, host.service_exists));
            if start {
                p.steps.push(Step::run(&["sc.exe", "start", WINDOWS_SERVICE]));
            }
        }
    }
    if !start {
        p.notes.push("start it with `ymirctl service start`".into());
    }
    if t.os == Os::Linux && spec.scope == Scope::User && !host.lingering {
        p.notes.push("to keep ymird running after you log out (and start it at boot), run `loginctl enable-linger $USER` once".into());
    }
    if t.os == Os::Windows && spec.scope == Scope::User {
        p.notes.push("the task starts ymird at each logon; `ymirctl service install --system` (elevated) runs it at boot instead".into());
    }
    Ok(p)
}

/// Starts the installed service.
pub fn start_plan(t: &Target) -> Plan {
    let steps = match (t.os, t.scope) {
        (Os::Linux, _) => vec![t.systemctl(&["start", SYSTEMD_UNIT])],
        (Os::Macos, _) => vec![Step::run(&["launchctl", "bootstrap", &t.launchd_domain(), &t.file().map(|f| s(&f)).unwrap_or_default()])],
        (Os::Windows, Scope::User) => vec![Step::run(&["schtasks", "/Run", "/TN", TASK_NAME])],
        (Os::Windows, Scope::System) => vec![Step::run(&["sc.exe", "start", WINDOWS_SERVICE])],
        (Os::FreeBsd, _) => vec![Step::run(&["service", RC_SCRIPT, "start"])],
    };
    Plan { steps, ..Default::default() }
}

/// Stops the service (it starts again at the next boot or login; `uninstall` removes it).
/// On Windows user scope the caller first asks `ymird` to stop over the control pipe, since the
/// task can only be ended hard; the `schtasks /End` here is the fallback.
pub fn stop_plan(t: &Target) -> Plan {
    let steps = match (t.os, t.scope) {
        (Os::Linux, _) => vec![t.systemctl(&["stop", SYSTEMD_UNIT])],
        (Os::Macos, _) => vec![Step::try_run(&["launchctl", "bootout", &format!("{}/{LAUNCHD_LABEL}", t.launchd_domain())])],
        (Os::Windows, Scope::User) => vec![Step::try_run(&["schtasks", "/End", "/TN", TASK_NAME])],
        (Os::Windows, Scope::System) => vec![Step::try_run(&["sc.exe", "stop", WINDOWS_SERVICE])],
        (Os::FreeBsd, _) => vec![Step::try_run(&["service", RC_SCRIPT, "stop"])],
    };
    Plan { steps, ..Default::default() }
}

/// Stops, disables and removes the service.
pub fn uninstall_plan(t: &Target) -> Plan {
    let mut p = Plan::default();
    if let Some(f) = t.file() {
        p.remove.push(f);
    }
    if t.custom {
        return p;
    }
    match (t.os, t.scope) {
        (Os::Linux, _) => {
            p.steps.push(Step { may_fail: true, ..t.systemctl(&["disable", "--now", SYSTEMD_UNIT]) });
        }
        (Os::Macos, _) => p.steps.extend(stop_plan(t).steps),
        (Os::Windows, Scope::User) => {
            p.steps.extend(stop_plan(t).steps);
            p.steps.push(Step::try_run(&["schtasks", "/Delete", "/TN", TASK_NAME, "/F"]));
        }
        (Os::Windows, Scope::System) => {
            p.steps.extend(stop_plan(t).steps);
            p.steps.push(Step::try_run(&["sc.exe", "delete", WINDOWS_SERVICE]));
        }
        (Os::FreeBsd, _) => {
            p.steps.extend(stop_plan(t).steps);
            p.steps.push(Step::try_run(&["sysrc", "-x", "ymird_enable"]));
        }
    }
    p
}

/// Carries out a plan (or with `dry_run` describes it). Returns what it did, line by line.
pub fn execute(plan: &Plan, dry_run: bool) -> Result<Vec<String>> {
    let mut log = Vec::new();
    for d in &plan.dirs {
        if dry_run {
            log.push(format!("would create {}", d.display()));
        } else {
            std::fs::create_dir_all(d).with_context(|| format!("creating {}", d.display()))?;
        }
    }
    for (path, bytes) in &plan.files {
        if dry_run {
            log.push(format!("would write {}:", path.display()));
            let text = if bytes.starts_with(&[0xFF, 0xFE]) {
                let units: Vec<u16> = bytes[2..].as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes(*c)).collect();
                String::from_utf16_lossy(&units)
            } else {
                String::from_utf8_lossy(bytes).into_owned()
            };
            log.extend(text.lines().map(|l| format!("  {}", l.trim_end_matches('\r'))));
            continue;
        }
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o644))?;
        }
        std::fs::rename(&tmp, path).with_context(|| format!("writing {}", path.display()))?;
        log.push(format!("wrote {}", path.display()));
    }
    for step in &plan.steps {
        if dry_run {
            log.push(format!("would run: {}", step.line()));
            continue;
        }
        match run(&step.cmd) {
            Ok(()) => log.push(format!("ran: {}", step.line())),
            Err(e) if step.may_fail => log.push(format!("skipped: {} ({e:#})", step.line())),
            Err(e) => return Err(e),
        }
    }
    for path in &plan.remove {
        if !path.exists() {
            log.push(format!("{} is not there", path.display()));
        } else if dry_run {
            log.push(format!("would remove {}", path.display()));
        } else {
            std::fs::remove_file(path).with_context(|| format!("removing {}", path.display()))?;
            log.push(format!("removed {}", path.display()));
        }
    }
    log.extend(plan.notes.iter().map(|n| format!("note: {n}")));
    Ok(log)
}

fn run(cmd: &[String]) -> Result<()> {
    let out = Command::new(&cmd[0]).args(&cmd[1..]).output().with_context(|| format!("running {}", cmd[0]))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let msg = String::from_utf8_lossy(&out.stdout);
        let detail = [err.trim(), msg.trim()].into_iter().find(|t| !t.is_empty()).unwrap_or("");
        bail!("`{}` failed ({}){}", cmd.join(" "), out.status, if detail.is_empty() { String::new() } else { format!(": {detail}") });
    }
    Ok(())
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

/// Whether systemd keeps this user's services running without a login session.
pub fn lingering() -> bool {
    let user = std::env::var("USER").unwrap_or_default();
    Command::new("loginctl")
        .args(["show-user", &user, "-p", "Linger"])
        .output()
        .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).trim() == "Linger=yes")
}

/// Where the service file is, and what the service manager says.
pub fn status(t: &Target) -> Vec<String> {
    let mut log = Vec::new();
    match t.file() {
        Some(f) if f.exists() => log.push(format!("installed: {}", f.display())),
        Some(f) => log.push(format!("not installed ({} is missing)", f.display())),
        None => {}
    }
    if t.custom {
        return log;
    }
    let query = |cmd: &[&str]| Command::new(cmd[0]).args(&cmd[1..]).output().ok();
    match (t.os, t.scope) {
        (Os::Linux, _) => {
            let mut c = vec!["systemctl"];
            if t.scope == Scope::User {
                c.push("--user");
            }
            c.extend(["is-active", SYSTEMD_UNIT]);
            if let Some(o) = query(&c) {
                log.push(format!("systemd: {}", String::from_utf8_lossy(&o.stdout).trim()));
            }
            if t.scope == Scope::User {
                log.push(format!(
                    "lingering: {}",
                    if lingering() { "yes" } else { "no (`loginctl enable-linger $USER` keeps it running after logout)" }
                ));
            }
        }
        (Os::Macos, _) => {
            let target = format!("{}/{LAUNCHD_LABEL}", t.launchd_domain());
            match query(&["launchctl", "print", &target]) {
                Some(o) if o.status.success() => log.push(format!("launchd: {}", launchd_state(&String::from_utf8_lossy(&o.stdout)))),
                _ => log.push(format!("launchd: not loaded ({target})")),
            }
        }
        (Os::Windows, Scope::User) => match query(&["schtasks", "/Query", "/TN", TASK_NAME, "/FO", "CSV", "/NH"]) {
            Some(o) if o.status.success() => log.push(format!("task {TASK_NAME}: {}", task_state(&String::from_utf8_lossy(&o.stdout)))),
            _ => log.push(format!("task {TASK_NAME}: not registered")),
        },
        (Os::Windows, Scope::System) => match query(&["sc.exe", "query", WINDOWS_SERVICE]) {
            Some(o) if o.status.success() => log.push(format!("service {WINDOWS_SERVICE}: {}", sc_state(&String::from_utf8_lossy(&o.stdout)))),
            _ => log.push(format!("service {WINDOWS_SERVICE}: not registered")),
        },
        (Os::FreeBsd, _) => match query(&["service", RC_SCRIPT, "status"]) {
            Some(o) => log.push(format!("rc.d: {}", String::from_utf8_lossy(&o.stdout).trim())),
            None => log.push("rc.d: `service` is not available".into()),
        },
    }
    log
}

/// `state = running` and `pid = 123` from `launchctl print`.
pub fn launchd_state(out: &str) -> String {
    let field = |k: &str| out.lines().map(str::trim).find_map(|l| l.strip_prefix(k).map(|v| v.trim().to_string()));
    match (field("state = "), field("pid = ")) {
        (Some(s), Some(p)) => format!("{s} (pid {p})"),
        (Some(s), None) => s,
        _ => "loaded".into(),
    }
}

/// The status column of `schtasks /Query /FO CSV /NH` ("TaskName","Next Run Time","Status").
pub fn task_state(out: &str) -> String {
    out.lines()
        .find(|l| !l.trim().is_empty())
        .and_then(|l| l.trim().trim_matches('"').rsplit("\",\"").next().map(str::to_string))
        .unwrap_or_else(|| "registered".into())
}

/// `STATE : 4  RUNNING` from `sc.exe query`.
pub fn sc_state(out: &str) -> String {
    out.lines()
        .map(str::trim)
        .find(|l| l.starts_with("STATE"))
        .and_then(|l| l.split_whitespace().last())
        .map(str::to_lowercase)
        .unwrap_or_else(|| "registered".into())
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

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> ServiceSpec {
        ServiceSpec::user("/opt/ymir/bin/ymird", "/home/u/My Config/ymir")
    }

    fn win_spec() -> ServiceSpec {
        ServiceSpec::user(r"C:\Program Files\Ymir\ymird.exe", r"C:\Users\u\AppData\Roaming\ymir")
    }

    fn target(os: Os, scope: Scope) -> Target {
        let dir = match (os, scope) {
            (Os::Linux, Scope::User) => Some("/home/u/.config/systemd/user".into()),
            (Os::Windows, Scope::System) => None,
            (os, scope) => default_dir(os, scope, Path::new(r"C:\Users\u\AppData\Roaming\ymir")),
        };
        Target { os, scope, dir, custom: false, uid: 501 }
    }

    fn host() -> Host {
        Host { windows_user: r"PC\u".into(), system_root: r"C:\Windows".into(), service_exists: false, lingering: true }
    }

    fn lines(p: &Plan) -> Vec<String> {
        p.steps.iter().map(Step::line).collect()
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
        assert!(u.contains("Restart=on-failure") && u.contains("WantedBy=default.target") && !u.contains("User="));
        let mut sys = spec();
        sys.scope = Scope::System;
        sys.run_as = Some("llm".into());
        let u = systemd_unit(&sys);
        assert!(u.contains("User=llm\n") && u.contains("WantedBy=multi-user.target") && u.contains("After=network-online.target"), "{u}");
    }

    #[test]
    fn launchd_agent_and_daemon() {
        let p = launchd_plist(&ServiceSpec::user("/opt/a&b/ymird", "/Users/u/Library/Ymir"));
        assert!(p.contains("<string>/opt/a&amp;b/ymird</string><string>--home</string><string>/Users/u/Library/Ymir</string>"), "{p}");
        assert!(p.contains(&format!("<key>Label</key><string>{LAUNCHD_LABEL}</string>")));
        assert!(p.contains("<key>RunAtLoad</key><true/>") && p.contains("<key>SuccessfulExit</key><false/>"));
        assert!(p.contains("<key>StandardErrorPath</key><string>/Users/u/Library/Ymir/state/logs/ymird.log</string>"));
        assert!(p.contains("/opt/homebrew/bin") && !p.contains("UserName"));
        let mut d = ServiceSpec::user("/usr/local/bin/ymird", "/Library/Application Support/Ymir");
        d.scope = Scope::System;
        d.run_as = Some("_ymir".into());
        assert!(launchd_plist(&d).contains("<key>UserName</key><string>_ymir</string>"));
    }

    #[test]
    fn launchd_commands() {
        let t = target(Os::Macos, Scope::User);
        let plist = s(&t.file().unwrap());
        let p = install_plan(&spec(), &t, &host(), true).unwrap();
        assert_eq!(lines(&p), vec![format!("launchctl bootout gui/501/{LAUNCHD_LABEL}"), format!("launchctl bootstrap gui/501 {plist}")]);
        assert!(p.steps[0].may_fail && !p.steps[1].may_fail);
        assert!(p.dirs[0].ends_with("state/logs"), "launchd needs the log directory to exist: {:?}", p.dirs);
        assert_eq!(lines(&start_plan(&t)), vec![format!("launchctl bootstrap gui/501 {plist}")]);
        assert_eq!(lines(&stop_plan(&t)), vec![format!("launchctl bootout gui/501/{LAUNCHD_LABEL}")]);
        let u = uninstall_plan(&t);
        assert_eq!(u.remove, vec![PathBuf::from(&plist)]);
        let sys = Target { scope: Scope::System, dir: Some("/Library/LaunchDaemons".into()), ..t };
        let mut d = spec();
        d.scope = Scope::System;
        assert!(install_plan(&d, &sys, &host(), true).is_err(), "a LaunchDaemon needs --run-as");
        d.run_as = Some("_ymir".into());
        let p = install_plan(&d, &sys, &host(), true).unwrap();
        assert!(lines(&p).contains(&"launchctl bootstrap system /Library/LaunchDaemons/com.weirdbrains.ymird.plist".to_string()), "{p:?}");
        assert!(lines(&p)[0].starts_with("chown -R _ymir "));
    }

    #[test]
    fn freebsd_rc_script_and_commands() {
        let mut d = ServiceSpec::user("/usr/local/bin/ymird", "/var/db/ymir node");
        d.scope = Scope::System;
        let t = target(Os::FreeBsd, Scope::System);
        assert_eq!(t.file(), Some(PathBuf::from("/usr/local/etc/rc.d/ymird")));
        assert!(install_plan(&d, &t, &host(), true).is_err(), "an rc.d service needs --run-as");
        d.run_as = Some("ymir".into());
        let rc = rcd_script(&d);
        assert!(rc.starts_with("#!/bin/sh\n") && rc.contains("# PROVIDE: ymird\n") && rc.contains("rcvar=\"ymird_enable\"\n"), "{rc}");
        assert!(
            rc.contains(
                "command_args=\"-r -R 5 -P ${pidfile} -o '/var/db/ymir node/state/logs/ymird.log' -u ymir -- /usr/bin/env YMIR_LOG=info /usr/local/bin/ymird --home '/var/db/ymir node'\"\n"
            ),
            "{rc}"
        );
        assert!(rc.ends_with("run_rc_command \"$1\"\n"));
        // It parses as sh.
        #[cfg(unix)]
        {
            let f = std::env::temp_dir().join(format!("ymird-rc-{}", std::process::id()));
            std::fs::write(&f, &rc).unwrap();
            let ok = Command::new("sh").arg("-n").arg(&f).status().unwrap().success();
            std::fs::remove_file(&f).unwrap();
            assert!(ok, "sh -n refused the script:\n{rc}");
        }
        let p = install_plan(&d, &t, &host(), true).unwrap();
        assert_eq!(p.files[0].0, PathBuf::from("/usr/local/etc/rc.d/ymird"));
        assert_eq!(
            lines(&p),
            vec![
                "chown -R ymir \"/var/db/ymir node\"".to_string(),
                "chmod 555 /usr/local/etc/rc.d/ymird".into(),
                "sysrc ymird_enable=YES".into(),
                "service ymird restart".into()
            ]
        );
        assert_eq!(lines(&start_plan(&t)), vec!["service ymird start"]);
        assert_eq!(lines(&stop_plan(&t)), vec!["service ymird stop"]);
        let u = uninstall_plan(&t);
        assert_eq!(lines(&u), vec!["service ymird stop", "sysrc -x ymird_enable"]);
        assert_eq!(u.remove, vec![PathBuf::from("/usr/local/etc/rc.d/ymird")]);
        // No per-user services there.
        assert!(install_plan(&spec(), &target(Os::FreeBsd, Scope::User), &host(), true).is_err());
        assert_eq!(system_home(Os::FreeBsd), PathBuf::from("/var/db/ymir"));
        // Quoting a single quote for sh.
        assert_eq!(sh_quote("it's"), r"'it'\''s'");
    }

    #[test]
    fn systemd_commands() {
        let t = target(Os::Linux, Scope::User);
        let p = install_plan(&spec(), &t, &host(), true).unwrap();
        assert_eq!(
            lines(&p),
            vec!["systemctl --user daemon-reload", "systemctl --user enable ymird.service", "systemctl --user restart ymird.service"]
        );
        assert_eq!(lines(&install_plan(&spec(), &t, &host(), false).unwrap()), vec!["systemctl --user daemon-reload"]);
        assert_eq!(lines(&start_plan(&t)), vec!["systemctl --user start ymird.service"]);
        assert_eq!(lines(&stop_plan(&t)), vec!["systemctl --user stop ymird.service"]);
        let sys = Target { scope: Scope::System, dir: Some("/etc/systemd/system".into()), ..t };
        assert_eq!(lines(&stop_plan(&sys)), vec!["systemctl stop ymird.service"]);
    }

    #[test]
    fn windows_quoting() {
        assert_eq!(win_quote(r"C:\Program Files\Ymir\ymird.exe"), r#""C:\Program Files\Ymir\ymird.exe""#);
        assert_eq!(win_quote(r"C:\Ymir\ymird.exe"), r"C:\Ymir\ymird.exe");
        assert_eq!(win_quote(r"C:\My Dir\"), r#""C:\My Dir\\""#);
        assert_eq!(win_quote(r#"a"b"#), r#""a\"b""#);
        assert_eq!(win_quote(""), r#""""#);
    }

    #[test]
    fn windows_command_lines() {
        assert_eq!(
            command_for(Os::Windows, &win_spec()),
            vec![
                r"C:\Program Files\Ymir\ymird.exe",
                "--home",
                r"C:\Users\u\AppData\Roaming\ymir",
                "--log-file",
                r"C:\Users\u\AppData\Roaming\ymir\state\logs\ymird.log"
            ]
        );
        let mut sys = win_spec();
        sys.scope = Scope::System;
        assert_eq!(command_for(Os::Windows, &sys)[1], "--windows-service");
        // Linux and macOS log through the journal or launchd, and never pass --windows-service.
        assert_eq!(command_for(Os::Linux, &spec()), vec!["/opt/ymir/bin/ymird", "--home", "/home/u/My Config/ymir"]);
    }

    #[test]
    fn windows_task() {
        let x = task_xml(&win_spec(), r"PC\u", r"C:\Windows");
        assert!(x.starts_with("<?xml version=\"1.0\" encoding=\"UTF-16\"?>"));
        assert!(x.contains(r"<URI>\Ymir\ymird</URI>"));
        assert!(x.contains(r"<LogonTrigger>") && x.contains(r"<UserId>PC\u</UserId>") && x.contains("<LogonType>InteractiveToken</LogonType>"));
        assert!(x.contains("<ExecutionTimeLimit>PT0S</ExecutionTimeLimit>") && x.contains("<RestartOnFailure>"));
        assert!(x.contains("<DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>"));
        assert!(x.contains(r"<Command>C:\Windows\System32\conhost.exe</Command>"), "{x}");
        assert!(
            x.contains(r#"<Arguments>--headless &quot;C:\Program Files\Ymir\ymird.exe&quot; --home C:\Users\u\AppData\Roaming\ymir --log-file C:\Users\u\AppData\Roaming\ymir\state\logs\ymird.log</Arguments>"#),
            "{x}"
        );
        let b = utf16_with_bom("a\nb");
        assert_eq!(b, vec![0xFF, 0xFE, b'a', 0, b'\r', 0, b'\n', 0, b'b', 0]);

        let t = target(Os::Windows, Scope::User);
        let file = t.file().unwrap();
        assert_eq!(s(&file), r"C:\Users\u\AppData\Roaming\ymir\state\ymird-task.xml");
        let p = install_plan(&win_spec(), &t, &host(), true).unwrap();
        assert_eq!(lines(&p), vec![format!(r"schtasks /Create /TN \Ymir\ymird /XML {} /F", s(&file)), r"schtasks /Run /TN \Ymir\ymird".to_string()]);
        assert_eq!(lines(&stop_plan(&t)), vec![r"schtasks /End /TN \Ymir\ymird"]);
        assert!(stop_plan(&t).steps[0].may_fail);
        assert_eq!(lines(&uninstall_plan(&t)), vec![r"schtasks /End /TN \Ymir\ymird", r"schtasks /Delete /TN \Ymir\ymird /F"]);
        assert!(install_plan(&win_spec(), &t, &Host::default(), true).is_err(), "needs the user for the logon trigger");
    }

    #[test]
    fn windows_service() {
        let mut sp = ServiceSpec::user(r"C:\Program Files\Ymir\ymird.exe", r"C:\ProgramData\Ymir");
        sp.scope = Scope::System;
        let t = target(Os::Windows, Scope::System);
        assert_eq!(t.file(), None);
        let p = install_plan(&sp, &t, &host(), true).unwrap();
        let cmds: Vec<Vec<String>> = p.steps.iter().map(|s| s.cmd.clone()).collect();
        assert_eq!(
            cmds[0],
            vec![
                "sc.exe",
                "create",
                "ymird",
                "binPath=",
                r#""C:\Program Files\Ymir\ymird.exe" --windows-service --home C:\ProgramData\Ymir --log-file C:\ProgramData\Ymir\state\logs\ymird.log"#,
                "start=",
                "auto",
                "obj=",
                r"NT SERVICE\ymird",
                "DisplayName=",
                "Ymir node"
            ]
        );
        let l = lines(&p);
        assert!(l.contains(&"sc.exe sidtype ymird unrestricted".to_string()), "{l:?}");
        assert!(l.contains(&"sc.exe failure ymird reset= 86400 actions= restart/60000/restart/60000/restart/60000".to_string()));
        assert!(l.iter().any(|c| c.starts_with(r"icacls C:\ProgramData\Ymir /inheritance:r /grant:r *S-1-5-18:(OI)(CI)F *S-1-5-32-544:(OI)(CI)F")));
        assert_eq!(l.last().unwrap(), "sc.exe start ymird");
        let again = install_plan(&sp, &t, &Host { service_exists: true, ..host() }, false).unwrap();
        assert_eq!(again.steps[0].cmd[1], "config", "an existing service is reconfigured");
        assert_eq!(lines(&stop_plan(&t)), vec!["sc.exe stop ymird"]);
        assert_eq!(lines(&uninstall_plan(&t)), vec!["sc.exe stop ymird", "sc.exe delete ymird"]);
        sp.ymirctl = Some(r"C:\Program Files\Ymir\ymirctl.exe".into());
        assert!(install_plan(&sp, &t, &host(), true).is_err());
        // --unit-dir: a script to run elevated, nothing registered.
        let custom = Target { dir: Some(r"C:\tmp".into()), custom: true, ..t };
        sp.ymirctl = None;
        let p = install_plan(&sp, &custom, &host(), true).unwrap();
        assert!(p.steps.is_empty());
        assert_eq!(s(&p.files[0].0), r"C:\tmp\ymird-service.cmd");
        assert!(String::from_utf8_lossy(&p.files[0].1).contains("sc.exe start ymird\r\n"));
    }

    #[test]
    fn state_parsers() {
        assert_eq!(launchd_state("com.weirdbrains.ymird = {\n\tactive count = 1\n\tstate = running\n\tpid = 4242\n}"), "running (pid 4242)");
        assert_eq!(task_state("\"\\Ymir\\ymird\",\"N/A\",\"Running\"\r\n"), "Running");
        assert_eq!(
            sc_state("SERVICE_NAME: ymird\n        TYPE               : 10  WIN32_OWN_PROCESS\n        STATE              : 4  RUNNING\n"),
            "running"
        );
    }

    #[test]
    fn install_into_a_custom_dir_never_starts() {
        let dir = tempfile::tempdir().unwrap();
        let home = tempfile::tempdir().unwrap();
        let sp = ServiceSpec::user("/opt/ymir/bin/ymird", home.path());
        let t = Target { os: Os::current(), scope: Scope::User, dir: Some(dir.path().into()), custom: true, uid: 501 };
        let p = install_plan(&sp, &t, &host(), true).unwrap();
        assert!(p.steps.is_empty(), "{p:?}");
        let log = execute(&p, true).unwrap();
        assert!(log.iter().any(|l| l.starts_with("would write")), "{log:?}");
        let file = t.file().unwrap();
        assert!(!file.exists(), "dry run writes nothing");
        execute(&p, false).unwrap();
        assert!(file.is_file());
        assert!(status(&t)[0].starts_with("installed"));
        execute(&uninstall_plan(&t), false).unwrap();
        assert!(!file.exists());
    }
}
