//! `ymirctl engines install vllm`: vLLM and Ray in a managed venv under the engines dir
//! (`<home>/engines/vllm`), where `[model] vllm = "auto"` and `ray = "auto"` find them before
//! `PATH` (see [`crate::config::resolve_ray`]). So one command makes single-node vLLM and
//! `vllm-ray` groups work with no `PATH` or `node.toml` edit.
//!
//! The venv is made with `uv` when it is on `PATH` (`uv venv --python 3.12`, then `uv pip install
//! --torch-backend ...`), else `python3 -m venv` and `pip` (Python 3.10 to 3.13, what vLLM
//! supports). The accelerator is detected (`nvidia-smi`: CUDA; `rocminfo` or `/opt/rocm`: ROCm;
//! else CPU) or given. PyPI's vLLM wheels are CUDA builds, so ROCm and CPU need `--index-url`, the
//! vLLM wheel index for that platform (or a container image: `vllm = "image:..."`). After
//! installing, `vllm --version` and `ray --version` must run, and `ymir-install.json` records
//! what was installed. Uninstall removes the venv only when that record is there.
//!
//! **Inkling 2 Mini.** By default this installs vLLM's nightly (from [`NIGHTLY_INDEX`]), because
//! Inkling 2 Mini (`inkling2_loop`) is in vLLM main since vllm-project/vllm#69028 (merged
//! 2026-11-21) but in no release yet: 0.34.0, the newest on PyPI, lacks it, and the recipe says
//! "vLLM 0.35.0+ nightly". Nightlies are numbered `0.34.1rc1.devN+gSHA`, below 0.35.0, so the
//! installer asks the installed vLLM's model registry for [`crate::vllm::INKLING_ARCH`] and records
//! the answer; ymird trusts that record ([`crate::vllm::managed_with_inkling`]). A default install
//! whose vLLM lacks Inkling fails. `--version V` installs that release from PyPI instead, and says
//! when it cannot run Inkling.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

/// The Python `uv` makes the venv with.
pub const UV_PYTHON: &str = "3.12";
/// What `ymir-install.json` is called, inside the venv.
pub const RECORD: &str = "ymir-install.json";
/// vLLM's nightly wheels (CUDA, x86_64 and aarch64), built from main.
pub const NIGHTLY_INDEX: &str = "https://wheels.vllm.ai/nightly";
/// Asks vLLM's model registry whether Inkling 2 Mini is there (prints `True` or `False`).
pub const INKLING_CHECK: &str =
    "from vllm.model_executor.models.registry import ModelRegistry as R; print('InklingForConditionalGeneration' in R.get_supported_archs())";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Accel {
    Cuda,
    Rocm,
    Cpu,
}

impl Accel {
    pub fn name(self) -> &'static str {
        match self {
            Accel::Cuda => "cuda",
            Accel::Rocm => "rocm",
            Accel::Cpu => "cpu",
        }
    }

    pub fn parse(s: &str) -> Result<Option<Accel>> {
        Ok(match s {
            "auto" => None,
            "cuda" => Some(Accel::Cuda),
            "rocm" => Some(Accel::Rocm),
            "cpu" => Some(Accel::Cpu),
            other => bail!("--accel `{other}`: auto, cuda, rocm or cpu"),
        })
    }
}

/// What the installer looks at on this machine (a trait so tests can fake it).
pub trait Probe {
    fn which(&self, program: &str) -> Option<PathBuf>;
    fn exists(&self, path: &Path) -> bool;
    fn os(&self) -> &'static str;
}

/// The real machine.
pub struct System;

impl Probe for System {
    fn which(&self, program: &str) -> Option<PathBuf> {
        weirdnet_backends::find_in_path(program)
    }
    fn exists(&self, path: &Path) -> bool {
        path.exists()
    }
    fn os(&self) -> &'static str {
        std::env::consts::OS
    }
}

/// CUDA when `nvidia-smi` is on `PATH`, ROCm when `rocminfo` is or `/opt/rocm` exists, else CPU.
pub fn detect(p: &dyn Probe) -> Accel {
    if p.which("nvidia-smi").is_some() {
        Accel::Cuda
    } else if p.which("rocminfo").is_some() || p.exists(Path::new("/opt/rocm")) {
        Accel::Rocm
    } else {
        Accel::Cpu
    }
}

#[derive(Debug, Clone, Default)]
pub struct Options {
    /// A vLLM release from PyPI (`0.34.0`); default the nightly (for Inkling 2 Mini).
    pub version: Option<String>,
    /// `None`: detect.
    pub accel: Option<Accel>,
    /// A wheel index for vLLM (needed for ROCm and CPU).
    pub index_url: Option<String>,
    /// `uv` to use; default the one on `PATH`.
    pub uv: Option<PathBuf>,
    /// The Python for `python3 -m venv`; default `python3` on `PATH`. Given, `uv` is not used.
    pub python: Option<PathBuf>,
    /// Upgrade what is installed (`pip install --upgrade`).
    pub upgrade: bool,
    /// Make the venv afresh.
    pub reinstall: bool,
    /// Install on an OS vLLM does not support (it runs on Linux).
    pub force: bool,
}

/// One command of the plan.
#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    pub program: PathBuf,
    pub args: Vec<String>,
}

impl std::fmt::Display for Step {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.program.display())?;
        for a in &self.args {
            if a.contains(' ') { write!(f, " '{a}'")? } else { write!(f, " {a}")? }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    pub venv: PathBuf,
    pub accel: Accel,
    /// `uv` or `pip`.
    pub installer: &'static str,
    /// The venv exists and is kept (an upgrade, or a second install).
    pub keep_venv: bool,
    /// vLLM's nightly (the default; it has Inkling 2 Mini).
    pub nightly: bool,
    pub steps: Vec<Step>,
}

fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| x.to_string()).collect()
}

/// The commands that install vLLM and Ray into `venv`.
pub fn plan(venv: &Path, o: &Options, p: &dyn Probe) -> Result<Plan> {
    if p.os() != "linux" && !o.force {
        bail!("vLLM runs on Linux (NVIDIA or AMD GPUs); this is {}. Use llama.cpp or MLX here, or pass --force", p.os());
    }
    let accel = o.accel.unwrap_or_else(|| detect(p));
    if accel != Accel::Cuda && o.index_url.is_none() {
        bail!(
            "PyPI's vLLM wheels are CUDA builds; for {} pass --index-url (the vLLM wheel index for {}), or run vLLM from a container image (`[model] vllm = \"image:...\"`)",
            accel.name(),
            accel.name()
        );
    }
    if let Some(v) = &o.version
        && (v.is_empty() || !v.chars().all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '+' || c == '-'))
    {
        bail!("--version `{v}` is not a version like 0.11.0");
    }
    let spec = match &o.version {
        Some(v) => format!("vllm=={v}"),
        None => "vllm".into(),
    };
    // The nightly, unless a release is pinned or the operator gave an index (ROCm, CPU).
    let nightly = o.version.is_none() && o.index_url.is_none();
    let py = crate::vllm::bin_in(venv, "python");
    let keep_venv = !o.reinstall && p.exists(&py);
    let uv = if o.python.is_some() { None } else { o.uv.clone().or_else(|| p.which("uv")) };
    let mut steps = Vec::new();
    let installer;
    match uv {
        Some(uv) => {
            installer = "uv";
            if !keep_venv {
                let mut a = s(&["venv", "--python", UV_PYTHON]);
                if o.reinstall {
                    a.push("--clear".into());
                }
                a.push(venv.to_string_lossy().into_owned());
                steps.push(Step { program: uv.clone(), args: a });
            }
            let mut a = s(&["pip", "install", "--python"]);
            a.push(py.to_string_lossy().into_owned());
            if o.upgrade {
                a.push("--upgrade".into());
            }
            a.push(format!("--torch-backend={}", if accel == Accel::Cpu { "cpu" } else { "auto" }));
            if nightly {
                a.extend(["--prerelease=allow".into(), "--extra-index-url".into(), NIGHTLY_INDEX.into()]);
            }
            if let Some(u) = &o.index_url {
                a.extend(["--extra-index-url".into(), u.clone()]);
            }
            a.extend([spec, "ray".into()]);
            steps.push(Step { program: uv, args: a });
        }
        None => {
            installer = "pip";
            let python = o
                .python
                .clone()
                .or_else(|| p.which("python3"))
                .context("installing vLLM needs `uv` (https://docs.astral.sh/uv/) or `python3` (3.10 to 3.13) on PATH")?;
            if !keep_venv {
                let mut a = s(&["-m", "venv"]);
                if o.reinstall {
                    a.push("--clear".into());
                }
                a.push(venv.to_string_lossy().into_owned());
                steps.push(Step { program: python, args: a });
            }
            steps.push(Step { program: py.clone(), args: s(&["-m", "pip", "install", "--upgrade", "pip"]) });
            let mut a = s(&["-m", "pip", "install"]);
            if o.upgrade {
                a.push("--upgrade".into());
            }
            if nightly {
                a.extend(["--pre".into(), "--extra-index-url".into(), NIGHTLY_INDEX.into()]);
            }
            if let Some(u) = &o.index_url {
                a.extend(["--extra-index-url".into(), u.clone()]);
            }
            a.extend([spec, "ray".into()]);
            steps.push(Step { program: py, args: a });
        }
    }
    Ok(Plan { venv: venv.to_path_buf(), accel, installer, keep_venv, nightly, steps })
}

/// `Python 3.12.3` -> `(3, 12)`.
pub fn python_version(out: &str) -> Option<(u32, u32)> {
    let v = out.split_whitespace().find(|w| w.chars().next().is_some_and(|c| c.is_ascii_digit()))?;
    let mut it = v.split('.');
    Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
}

/// The first version-looking word (`0.11.0`, `ray, version 2.49.1` -> `2.49.1`).
pub fn version_in(out: &str) -> Option<String> {
    out.split(|c: char| c.is_whitespace() || c == ',')
        .find(|w| w.contains('.') && w.chars().next().is_some_and(|c| c.is_ascii_digit()))
        .map(|w| w.trim_end_matches('.').to_string())
}

/// What `ymir-install.json` keeps.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Record {
    pub accel: Accel,
    pub installer: String,
    /// What was asked for (`None`: the newest).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requested: Option<String>,
    pub vllm: String,
    pub ray: String,
    /// From vLLM's nightly index.
    #[serde(default)]
    pub nightly: bool,
    /// vLLM's model registry has Inkling 2 Mini ([`INKLING_CHECK`]).
    #[serde(default)]
    pub inkling: bool,
    pub at_ms: u64,
}

pub fn read_record(venv: &Path) -> Option<Record> {
    serde_json::from_slice(&std::fs::read(venv.join(RECORD)).ok()?).ok()
}

/// Runs `program --version` and returns the version it prints.
pub fn program_version(program: &Path) -> Result<String> {
    let out = std::process::Command::new(program)
        .arg("--version")
        .stdin(std::process::Stdio::null())
        .output()
        .with_context(|| format!("running {} --version", program.display()))?;
    if !out.status.success() {
        bail!("{} --version failed ({})", program.display(), out.status);
    }
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    version_in(&text).with_context(|| format!("{} --version printed no version", program.display()))
}

/// Runs the plan (blocking; it downloads), checks the programs and writes the record. `log`
/// sees each command before it runs.
pub fn install(plan: &Plan, o: &Options, log: &mut dyn FnMut(&str)) -> Result<Record> {
    if let Some(parent) = plan.venv.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    if plan.installer == "pip"
        && let Some(first) = plan.steps.first()
        && !plan.keep_venv
    {
        let out =
            std::process::Command::new(&first.program).arg("--version").output().with_context(|| format!("running {}", first.program.display()))?;
        let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        match python_version(&text) {
            Some((3, m)) if (10..=13).contains(&m) => {}
            v => bail!(
                "{} is Python {}; vLLM needs 3.10 to 3.13 (install `uv`, or pass --python)",
                first.program.display(),
                v.map_or("?".into(), |(a, b)| format!("{a}.{b}"))
            ),
        }
    }
    for step in &plan.steps {
        log(&step.to_string());
        let status = std::process::Command::new(&step.program)
            .args(&step.args)
            .stdin(std::process::Stdio::null())
            .status()
            .with_context(|| format!("running {step}"))?;
        if !status.success() {
            bail!("{step} failed ({status})");
        }
    }
    let vllm = program_version(&crate::vllm::bin_in(&plan.venv, "vllm")).context("vLLM was installed but does not run")?;
    let ray = program_version(&crate::vllm::bin_in(&plan.venv, "ray")).context("Ray was installed but does not run")?;
    let inkling = has_inkling(&crate::vllm::bin_in(&plan.venv, "python"));
    let at_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64);
    let rec = Record {
        accel: plan.accel,
        installer: plan.installer.into(),
        requested: o.version.clone(),
        vllm,
        ray,
        nightly: plan.nightly,
        inkling,
        at_ms,
    };
    std::fs::write(plan.venv.join(RECORD), serde_json::to_vec_pretty(&rec)?)?;
    if plan.nightly && !inkling {
        bail!(
            "vLLM {} from the nightly index does not register Inkling 2 Mini ({}); it is kept in {} (see `ymirctl engines status`)",
            rec.vllm,
            crate::vllm::INKLING_ARCH,
            plan.venv.display()
        );
    }
    Ok(rec)
}

/// Whether the venv's vLLM registers Inkling 2 Mini (runs [`INKLING_CHECK`] with its Python).
pub fn has_inkling(python: &Path) -> bool {
    std::process::Command::new(python)
        .args(["-c", INKLING_CHECK])
        .stdin(std::process::Stdio::null())
        .output()
        .is_ok_and(|o| o.status.success() && String::from_utf8_lossy(&o.stdout).trim() == "True")
}

/// Removes the managed venv, only when Ymir made it (its record is there).
pub fn uninstall(venv: &Path) -> Result<bool> {
    if !venv.exists() {
        return Ok(false);
    }
    if !venv.join(RECORD).is_file() {
        bail!("{} was not made by `ymirctl engines install vllm` (no {RECORD}); not removing it", venv.display());
    }
    std::fs::remove_dir_all(venv).with_context(|| format!("removing {}", venv.display()))?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake {
        on_path: Vec<&'static str>,
        files: Vec<PathBuf>,
        os: &'static str,
    }

    impl Probe for Fake {
        fn which(&self, program: &str) -> Option<PathBuf> {
            self.on_path.contains(&program).then(|| PathBuf::from(format!("/usr/bin/{program}")))
        }
        fn exists(&self, path: &Path) -> bool {
            self.files.iter().any(|f| f == path)
        }
        fn os(&self) -> &'static str {
            self.os
        }
    }

    fn fake(on_path: &[&'static str]) -> Fake {
        Fake { on_path: on_path.to_vec(), files: vec![], os: "linux" }
    }

    #[test]
    fn detects_the_accelerator() {
        assert_eq!(detect(&fake(&["nvidia-smi", "rocminfo"])), Accel::Cuda);
        assert_eq!(detect(&fake(&["rocminfo"])), Accel::Rocm);
        let mut f = fake(&[]);
        assert_eq!(detect(&f), Accel::Cpu);
        f.files.push("/opt/rocm".into());
        assert_eq!(detect(&f), Accel::Rocm);
        assert_eq!(Accel::parse("auto").unwrap(), None);
        assert!(Accel::parse("tpu").is_err());
    }

    // Unix paths in the expected commands (vLLM runs on Linux).
    #[cfg(unix)]
    #[test]
    fn with_uv_one_venv_and_one_install_of_vllm_and_ray() {
        let v = Path::new("/h/engines/vllm");
        let p = plan(v, &Options { version: Some("0.34.0".into()), ..Default::default() }, &fake(&["uv", "nvidia-smi"])).unwrap();
        assert_eq!((p.installer, p.accel, p.keep_venv, p.nightly), ("uv", Accel::Cuda, false, false));
        assert_eq!(p.steps[0].to_string(), "/usr/bin/uv venv --python 3.12 /h/engines/vllm");
        assert_eq!(p.steps[1].to_string(), "/usr/bin/uv pip install --python /h/engines/vllm/bin/python --torch-backend=auto vllm==0.34.0 ray");
        // The default is the nightly, for Inkling 2 Mini.
        let p = plan(v, &Options::default(), &fake(&["uv", "nvidia-smi"])).unwrap();
        assert!(p.nightly);
        assert_eq!(
            p.steps[1].to_string(),
            "/usr/bin/uv pip install --python /h/engines/vllm/bin/python --torch-backend=auto --prerelease=allow --extra-index-url https://wheels.vllm.ai/nightly vllm ray"
        );
    }

    // Unix paths in the expected commands (vLLM runs on Linux).
    #[cfg(unix)]
    #[test]
    fn without_uv_python3_venv_and_pip() {
        let v = Path::new("/h/engines/vllm");
        let p = plan(v, &Options::default(), &fake(&["python3", "nvidia-smi"])).unwrap();
        assert_eq!(p.installer, "pip");
        let shown: Vec<String> = p.steps.iter().map(|s| s.to_string()).collect();
        assert_eq!(
            shown,
            [
                "/usr/bin/python3 -m venv /h/engines/vllm",
                "/h/engines/vllm/bin/python -m pip install --upgrade pip",
                "/h/engines/vllm/bin/python -m pip install --pre --extra-index-url https://wheels.vllm.ai/nightly vllm ray"
            ]
        );
        assert!(plan(v, &Options::default(), &fake(&["nvidia-smi"])).unwrap_err().to_string().contains("needs `uv`"));
    }

    #[test]
    fn rocm_and_cpu_need_a_wheel_index() {
        let v = Path::new("/h/engines/vllm");
        let e = plan(v, &Options::default(), &fake(&["uv"])).unwrap_err().to_string();
        assert!(e.contains("cpu") && e.contains("--index-url"), "{e}");
        let o = Options { accel: Some(Accel::Cpu), index_url: Some("https://wheels.example/cpu".into()), ..Default::default() };
        let p = plan(v, &o, &fake(&["uv"])).unwrap();
        let last = p.steps.last().unwrap().to_string();
        assert!(last.contains("--torch-backend=cpu") && last.contains("--extra-index-url https://wheels.example/cpu"), "{last}");
        assert!(!p.nightly && !last.contains("wheels.vllm.ai/nightly"), "the operator's index replaces the nightly: {last}");
        let o = Options { accel: Some(Accel::Rocm), index_url: Some("https://wheels.example/rocm".into()), ..Default::default() };
        assert_eq!(plan(v, &o, &fake(&["uv", "nvidia-smi"])).unwrap().accel, Accel::Rocm, "the override wins");
    }

    // Unix paths in the expected commands (vLLM runs on Linux).
    #[cfg(unix)]
    #[test]
    fn upgrade_keeps_the_venv_and_reinstall_clears_it() {
        let v = Path::new("/h/engines/vllm");
        let mut f = fake(&["uv", "nvidia-smi"]);
        f.files.push(v.join("bin/python"));
        let p = plan(v, &Options { upgrade: true, ..Default::default() }, &f).unwrap();
        assert!(p.keep_venv);
        assert_eq!(p.steps.len(), 1);
        assert!(p.steps[0].args.contains(&"--upgrade".to_string()));
        let p = plan(v, &Options { reinstall: true, ..Default::default() }, &f).unwrap();
        assert_eq!(p.steps[0].to_string(), "/usr/bin/uv venv --python 3.12 --clear /h/engines/vllm");
    }

    #[test]
    fn refuses_other_oses_and_odd_versions() {
        let v = Path::new("/h/engines/vllm");
        let mut f = fake(&["uv", "nvidia-smi"]);
        f.os = "macos";
        assert!(plan(v, &Options::default(), &f).unwrap_err().to_string().contains("Linux"));
        assert!(plan(v, &Options { force: true, ..Default::default() }, &f).is_ok());
        let o = Options { version: Some("1.0; rm -rf /".into()), ..Default::default() };
        assert!(plan(v, &o, &fake(&["uv", "nvidia-smi"])).is_err());
    }

    #[test]
    fn reads_versions() {
        assert_eq!(python_version("Python 3.12.3\n"), Some((3, 12)));
        assert_eq!(python_version("nothing"), None);
        assert_eq!(version_in("0.11.0\n").as_deref(), Some("0.11.0"));
        assert_eq!(version_in("ray, version 2.49.1\n").as_deref(), Some("2.49.1"));
        assert_eq!(version_in("INFO 11-26 [__init__.py:216] Automatically detected platform cuda.\n0.11.0").as_deref(), Some("0.11.0"));
    }

    #[test]
    fn uninstall_removes_only_a_venv_ymir_made() {
        let d = tempfile::tempdir().unwrap();
        let v = d.path().join("vllm");
        assert!(!uninstall(&v).unwrap());
        std::fs::create_dir_all(v.join("bin")).unwrap();
        assert!(uninstall(&v).is_err(), "no record: not ours");
        std::fs::write(v.join(RECORD), "{}").unwrap();
        assert!(uninstall(&v).unwrap());
        assert!(!v.exists());
    }
}
