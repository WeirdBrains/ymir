//! Scoring: judging answers (exact match, numbers, JSON, lines, code run in a sandbox) and the
//! statistics over samples.

use std::time::Duration;

use anyhow::Result;
use serde_json::{Value, json};

use super::suites::{Check, Kind, expected_json};

/// The answer without a reasoning block.
pub fn strip_think(s: &str) -> String {
    let mut out = s.to_string();
    while let Some(a) = out.find("<think>") {
        match out[a..].find("</think>") {
            Some(b) => out.replace_range(a..a + b + "</think>".len(), ""),
            None => out.truncate(a),
        }
    }
    if let Some(b) = out.find("</think>") {
        out = out[b + "</think>".len()..].to_string();
    }
    out.trim().to_string()
}

pub fn normalize(s: &str) -> String {
    strip_think(s).trim_matches(|c: char| c.is_whitespace() || ".,!?\"'`*_:;".contains(c)).to_lowercase()
}

pub fn last_number(s: &str) -> Option<i64> {
    let s = strip_think(s).replace(',', "");
    let mut last = None;
    let mut cur = String::new();
    for c in s.chars().chain(std::iter::once(' ')) {
        if c.is_ascii_digit() || (c == '-' && cur.is_empty()) {
            cur.push(c);
        } else {
            if let Ok(n) = cur.parse::<i64>() {
                last = Some(n);
            }
            cur.clear();
        }
    }
    last
}

/// The first fenced block, else the whole text.
pub fn extract_block(s: &str) -> String {
    let s = strip_think(s);
    if let Some(a) = s.find("```") {
        let rest = &s[a + 3..];
        let rest = rest.split_once('\n').map_or(rest, |(_, r)| r);
        return rest.split("```").next().unwrap_or(rest).trim().to_string();
    }
    s
}

pub fn json_ok(s: &str) -> bool {
    let Ok(v) = serde_json::from_str::<Value>(&extract_block(s)) else { return false };
    v["name"] == "Ada" && v["age"].as_f64() == Some(36.0) && v["tags"] == json!(["x", "y"]) && v.as_object().is_some_and(|o| o.len() == 3)
}

/// Runs `code` plus `tests` with Python in an empty temp directory with an empty environment,
/// CPU, memory and process limits and a wall-clock limit. On unix also a file-size limit and,
/// where the machine allows it, no network (`unshare`); on Windows the limits are a Job Object's
/// and the network is not cut off. This limits accidents; it is not a security boundary.
pub fn run_python(code: &str, tests: &str) -> Result<bool> {
    let dir = tempfile::tempdir()?;
    std::fs::write(dir.path().join("t.py"), format!("{code}\n\n{tests}"))?;
    let mut sandboxed = sandbox::spawn(dir.path())?;
    let child = &mut sandboxed.child;
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        if child.try_wait()?.is_some() {
            break;
        }
        if std::time::Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Ok(false);
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    let out = sandboxed.child.wait_with_output()?;
    Ok(out.status.success() && String::from_utf8_lossy(&out.stdout).trim() == "OK")
}

/// Whether code answers can be run: Python 3 for `run_python`'s sandbox.
pub fn python_available() -> bool {
    sandbox::python().is_some()
}

#[cfg(unix)]
mod sandbox {
    use std::path::{Path, PathBuf};
    use std::process::{Child, Command, Stdio};
    use std::sync::OnceLock;

    use anyhow::{Context, Result};

    /// One limit per `ulimit` (dash, Debian's and Ubuntu's sh, takes one at a time), with each
    /// shell's spelling: processes are `-u` in bash and zsh, `-p` in dash; memory is `-v`, or
    /// `-d` where the address space can't be limited.
    const LIMITS: &str =
        "ulimit -t 5 && ulimit -f 1024 && { ulimit -v 2000000 2>/dev/null || ulimit -d 2000000; } && { ulimit -u 64 2>/dev/null || ulimit -p 64; }";

    pub struct Sandboxed {
        pub child: Child,
    }

    /// python3, if it runs and `sh` can set every limit in [`LIMITS`] (checked once).
    pub fn python() -> Option<PathBuf> {
        static OK: OnceLock<bool> = OnceLock::new();
        let ok = *OK.get_or_init(|| {
            let quiet = |c: &mut Command| c.stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok_and(|s| s.success());
            quiet(Command::new("python3").arg("--version")) && quiet(Command::new("sh").args(["-c", LIMITS]))
        });
        ok.then(|| "python3".into())
    }

    /// Whether network isolation is available for running model code.
    fn can_unshare() -> bool {
        Command::new("unshare")
            .args(["--net", "--map-root-user", "true"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    }

    pub fn spawn(dir: &Path) -> Result<Sandboxed> {
        // A limit that can't be set stops the run before Python starts.
        let run = format!("{{ {LIMITS}; }} 2>/dev/null || exit 1; exec python3 -B -I t.py");
        let mut cmd = if can_unshare() {
            let mut c = Command::new("unshare");
            c.args(["--net", "--map-root-user", "sh", "-c", &run]);
            c
        } else {
            let mut c = Command::new("sh");
            c.args(["-c", &run]);
            c
        };
        let child = cmd
            .current_dir(dir)
            .env_clear()
            .env("PATH", "/usr/local/bin:/usr/bin:/bin")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .context("starting python3")?;
        Ok(Sandboxed { child })
    }
}

#[cfg(windows)]
mod sandbox {
    //! A Job Object: 5 s of CPU per process, 2 GB of memory for the job, at most 8 processes,
    //! and everything in it killed when the job's handle closes (after the run, or a timeout).
    //! Python waits for one byte on stdin before it runs `t.py`, and the byte is sent once it
    //! is in the job, so no model code runs outside it.

    use std::io::Write;
    use std::os::windows::io::AsRawHandle;
    use std::os::windows::process::CommandExt;
    use std::path::{Path, PathBuf};
    use std::process::{Child, Command, Stdio};

    use anyhow::{Context, Result, bail};
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_ACTIVE_PROCESS, JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION,
        JOB_OBJECT_LIMIT_JOB_MEMORY, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, JOB_OBJECT_LIMIT_PROCESS_TIME, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JobObjectExtendedLimitInformation, SetInformationJobObject,
    };

    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    /// Runs `t.py` as `__main__` once a byte arrives on stdin.
    const BOOT: &str = "import sys, runpy; sys.stdin.read(1); sys.stdin.close(); runpy.run_path('t.py', run_name='__main__')";

    struct Job(HANDLE);

    impl Drop for Job {
        fn drop(&mut self) {
            // SAFETY: the handle came from CreateJobObjectW and is closed once; closing it kills
            // what is left in the job (KILL_ON_JOB_CLOSE).
            unsafe { CloseHandle(self.0) };
        }
    }

    pub struct Sandboxed {
        pub child: Child,
        _job: Job,
    }

    /// `python.exe` on PATH, skipping the Microsoft Store alias in WindowsApps, if it is Python 3.
    pub fn python() -> Option<PathBuf> {
        let path = std::env::var_os("PATH")?;
        std::env::split_paths(&path)
            .filter(|d| !d.to_string_lossy().to_ascii_lowercase().contains("windowsapps"))
            .map(|d| d.join("python.exe"))
            .filter(|p| p.is_file())
            .find(|p| {
                Command::new(p)
                    .arg("--version")
                    .creation_flags(CREATE_NO_WINDOW)
                    .output()
                    .is_ok_and(|o| o.status.success() && String::from_utf8_lossy(&o.stdout).starts_with("Python 3"))
            })
    }

    fn job() -> Result<Job> {
        // SAFETY: plain Win32 calls on a struct this function owns; the handle is checked and
        // owned by `Job` from here on.
        unsafe {
            let h = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if h.is_null() {
                bail!("CreateJobObjectW failed: {}", std::io::Error::last_os_error());
            }
            let job = Job(h);
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_TIME
                | JOB_OBJECT_LIMIT_JOB_MEMORY
                | JOB_OBJECT_LIMIT_ACTIVE_PROCESS
                | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION;
            // 100 ns units.
            info.BasicLimitInformation.PerProcessUserTimeLimit = 5 * 10_000_000;
            info.BasicLimitInformation.ActiveProcessLimit = 8;
            info.JobMemoryLimit = 2_000_000_000;
            let ok = SetInformationJobObject(
                job.0,
                JobObjectExtendedLimitInformation,
                (&raw const info).cast(),
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );
            if ok == 0 {
                bail!("SetInformationJobObject failed: {}", std::io::Error::last_os_error());
            }
            Ok(job)
        }
    }

    pub fn spawn(dir: &Path) -> Result<Sandboxed> {
        let py = python().context("no Python 3 (python.exe) on PATH")?;
        let job = job()?;
        let mut cmd = Command::new(&py);
        cmd.args(["-B", "-I", "-c", BOOT]).current_dir(dir).env_clear().creation_flags(CREATE_NO_WINDOW);
        // Python needs SYSTEMROOT (for its random source and DLLs) and nothing else from here.
        if let Some(root) = std::env::var_os("SYSTEMROOT") {
            cmd.env("SYSTEMROOT", root);
        }
        if let Some(d) = py.parent() {
            cmd.env("PATH", d);
        }
        let mut child = cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().context("starting python")?;
        // SAFETY: both handles are live: the job's is owned by `job`, the process's by `child`.
        let ok = unsafe { AssignProcessToJobObject(job.0, child.as_raw_handle() as HANDLE) };
        if ok == 0 {
            let err = std::io::Error::last_os_error();
            let _ = child.kill();
            let _ = child.wait();
            bail!("AssignProcessToJobObject failed: {err}");
        }
        let mut stdin = child.stdin.take().context("python's stdin")?;
        stdin.write_all(b"\n")?;
        drop(stdin);
        Ok(Sandboxed { child, _job: job })
    }
}

/// `Some(passed)`, or `None` when the check could not run.
pub fn judge(check: &Check, answer: &str, code_ok: bool) -> Option<bool> {
    match &check.kind {
        Kind::JsonEq(j) => Some(serde_json::from_str::<Value>(&extract_block(answer)).is_ok_and(|v| v == expected_json(j))),
        Kind::Lines(l) => {
            let t = strip_think(answer);
            let got: Vec<&str> = t.lines().map(str::trim).filter(|x| !x.is_empty()).collect();
            Some(got == *l)
        }
        Kind::Func(f) => Some(f(answer)),
        Kind::Has(w) => Some(strip_think(answer).to_lowercase().contains(&w.to_lowercase())),
        Kind::Exact(ok) => Some(ok.contains(&normalize(answer).as_str())),
        Kind::Number(n) => Some(last_number(answer) == Some(*n)),
        Kind::Json => Some(json_ok(answer)),
        Kind::Code(tests) => code_ok.then(|| run_python(&extract_block(answer), tests).unwrap_or(false)),
    }
}

// ---- statistics ----

pub fn median(v: &[f64]) -> Option<f64> {
    percentile(v, 50.0)
}

/// Nearest-rank percentile.
pub fn percentile(v: &[f64], p: f64) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let rank = ((p / 100.0) * s.len() as f64).ceil().max(1.0) as usize;
    Some(s[rank.min(s.len()) - 1])
}

pub fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

#[cfg(test)]
/// The letters-only words of the answer, lower case, joined by single spaces.
pub fn words(s: &str) -> String {
    strip_think(s).split(|c: char| !c.is_alphabetic()).filter(|w| !w.is_empty()).map(str::to_lowercase).collect::<Vec<_>>().join(" ")
}
