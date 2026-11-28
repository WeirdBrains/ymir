//! `ymirctl engines install vllm-omni`: vLLM-Omni (vLLM's speech, image and video serving) in its
//! own managed venv under the engines dir (`<home>/engines/vllm-omni`), next to the vLLM one
//! ([`crate::vllm_install`]) and never mixed with it: vLLM-Omni needs the vLLM *release* of its
//! own major and minor version (it does not declare that dependency, so both are pinned here),
//! while the managed vLLM is a nightly for Inkling 2 Mini.
//!
//! It is the same installer in every other way: `uv venv --python 3.12` and `uv pip install
//! --torch-backend ...` when `uv` is on `PATH`, else `python3 -m venv` and pip (Python 3.11 to
//! 3.13, what vLLM-Omni supports); CUDA, ROCm or CPU detected or given, with `--index-url` needed
//! off CUDA; and the same driver checks. The driver's CUDA ([`vllm_install::driver_cuda`]) picks
//! PyTorch's build ([`vllm_install::torch_backend`]), PyTorch built for a newer CUDA than the
//! driver fails the install, and vLLM's own wheel built for a newer CUDA than the driver is a
//! warning, before installing and after (PyPI's vLLM release wheels are CUDA 13.0 builds,
//! [`RELEASE_CUDA`]). After installing, `vllm --version` must run and `import vllm_omni` must
//! print its version ([`OMNI_CHECK`]); `ymir-install.json` records what was installed, and
//! uninstall removes the venv only when that record is there.
//!
//! Nothing serves from this venv yet: the Weirdnet engine kind for vLLM-Omni comes with the
//! speech milestones (docs/design/audio.md).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::vllm_install::{self, Accel, Probe, RECORD, Step, UV_PYTHON};

/// The vLLM-Omni (and vLLM) release installed by default: the newest on PyPI on 2026-11-27.
pub const DEFAULT_VERSION: &str = "0.34.0";
/// The CUDA PyPI's vLLM release wheels are built with (0.34.0: CUDA 13.0, PyTorch 2.14.0).
pub const RELEASE_CUDA: (u32, u32) = (13, 0);
/// Prints vLLM-Omni's version (`0.34.0`).
pub const OMNI_CHECK: &str = "import importlib.metadata as m; print(m.version('vllm-omni'))";

/// The managed vLLM-Omni venv in the engines dir.
pub fn venv(engines_dir: &Path) -> PathBuf {
    engines_dir.join("vllm-omni")
}

#[derive(Debug, Clone, Default)]
pub struct Options {
    /// A vLLM-Omni release (`0.32.0`); vLLM is pinned to the same. Default [`DEFAULT_VERSION`].
    pub version: Option<String>,
    /// `None`: detect.
    pub accel: Option<Accel>,
    /// A wheel index for vLLM (needed for ROCm and CPU).
    pub index_url: Option<String>,
    /// `uv` to use; default the one on `PATH`.
    pub uv: Option<PathBuf>,
    /// The Python for `python3 -m venv`; default `python3` on `PATH`. Given, `uv` is not used.
    pub python: Option<PathBuf>,
    pub upgrade: bool,
    pub reinstall: bool,
    /// Install on an OS vLLM does not support (it runs on Linux).
    pub force: bool,
    /// `uv --torch-backend` instead of the driver-matched one.
    pub torch_backend: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    pub venv: PathBuf,
    pub accel: Accel,
    /// `uv` or `pip`.
    pub installer: &'static str,
    pub keep_venv: bool,
    /// The vLLM-Omni and vLLM release.
    pub version: String,
    /// The newest CUDA the driver runs (CUDA only).
    pub driver_cuda: Option<(u32, u32)>,
    /// The `--torch-backend` given to uv; `None`: PyPI's PyTorch.
    pub torch_backend: Option<String>,
    /// The CUDA vLLM's wheel is expected to be built with (PyPI's release: [`RELEASE_CUDA`]);
    /// `None` off CUDA or from an operator's index.
    pub vllm_cuda: Option<(u32, u32)>,
    pub steps: Vec<Step>,
}

impl Plan {
    /// The warning for a vLLM wheel built for a newer CUDA than the driver runs.
    pub fn cuda_warning(&self) -> Option<String> {
        newer_than_driver("vLLM's wheel", self.vllm_cuda?, self.driver_cuda?)
    }
}

fn newer_than_driver(what: &str, built: (u32, u32), driver: (u32, u32)) -> Option<String> {
    (built > driver).then(|| {
        format!(
            "{what} is a CUDA {}.{} build and the GPU driver runs CUDA {}.{}: its kernels may fail (\"the provided PTX was compiled with an unsupported toolchain\"). Update the NVIDIA driver, or install vLLM built for CUDA {}.{} from --index-url",
            built.0, built.1, driver.0, driver.1, driver.0, driver.1
        )
    })
}

fn checked_version(v: &str) -> Result<()> {
    if v.is_empty() || !v.chars().all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '+' || c == '-') {
        bail!("--version `{v}` is not a version like 0.34.0");
    }
    Ok(())
}

/// The commands that install vLLM-Omni and its vLLM into `venv`.
pub fn plan(venv: &Path, o: &Options, p: &dyn Probe) -> Result<Plan> {
    if p.os() != "linux" && !o.force {
        bail!("vLLM-Omni runs on Linux (NVIDIA or AMD GPUs); this is {}. Pass --force to try anyway", p.os());
    }
    let accel = o.accel.unwrap_or_else(|| vllm_install::detect(p));
    if accel != Accel::Cuda && o.index_url.is_none() {
        bail!("PyPI's vLLM wheels are CUDA builds; for {} pass --index-url (the vLLM wheel index for {})", accel.name(), accel.name());
    }
    let version = o.version.clone().unwrap_or_else(|| DEFAULT_VERSION.into());
    checked_version(&version)?;
    if let Some(b) = &o.torch_backend
        && (b.is_empty() || !b.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '.'))
    {
        bail!("--torch-backend `{b}` is not a backend like cu130, cu128, cpu or auto");
    }
    let driver_cuda = if accel == Accel::Cuda { p.driver_cuda() } else { None };
    let torch_backend = match (&o.torch_backend, accel) {
        (Some(b), _) => Some(b.clone()),
        (None, Accel::Cpu) => Some("cpu".into()),
        (None, Accel::Rocm) => Some("auto".into()),
        (None, Accel::Cuda) => vllm_install::torch_backend(driver_cuda, false)?,
    };
    let vllm_cuda = (accel == Accel::Cuda && o.index_url.is_none()).then_some(RELEASE_CUDA);
    let specs = [format!("vllm=={version}"), format!("vllm-omni=={version}")];
    let py = crate::vllm::bin_in(venv, "python");
    let keep_venv = !o.reinstall && p.exists(&py);
    let uv = if o.python.is_some() { None } else { o.uv.clone().or_else(|| p.which("uv")) };
    let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    let mut steps = Vec::new();
    let installer = match uv {
        Some(uv) => {
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
            if let Some(b) = &torch_backend {
                a.push(format!("--torch-backend={b}"));
            }
            if let Some(u) = &o.index_url {
                a.extend(["--extra-index-url".into(), u.clone()]);
            }
            a.extend(specs);
            steps.push(Step { program: uv, args: a });
            "uv"
        }
        None => {
            if let Some(b) = torch_backend.as_ref().filter(|b| *b != "cpu" && *b != "auto") {
                bail!("PyTorch {b} (for this driver) needs `uv` (https://docs.astral.sh/uv/); pip installs PyPI's CUDA 13.0 build");
            }
            let python = o
                .python
                .clone()
                .or_else(|| p.which("python3"))
                .context("installing vLLM-Omni needs `uv` (https://docs.astral.sh/uv/) or `python3` (3.11 to 3.13) on PATH")?;
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
            if let Some(u) = &o.index_url {
                a.extend(["--extra-index-url".into(), u.clone()]);
            }
            a.extend(specs);
            steps.push(Step { program: py, args: a });
            "pip"
        }
    };
    let torch_backend = if installer == "uv" { torch_backend } else { None };
    Ok(Plan { venv: venv.to_path_buf(), accel, installer, keep_venv, version, driver_cuda, torch_backend, vllm_cuda, steps })
}

/// What `ymir-install.json` keeps for vLLM-Omni.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Record {
    pub engine: String,
    pub accel: Accel,
    pub installer: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requested: Option<String>,
    pub omni: String,
    pub vllm: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub driver_cuda: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub torch_backend: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub torch: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub torch_cuda: Option<String>,
    /// The CUDA vLLM's wheel is built with (its version's `cuNNN` tag, else the plan's).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vllm_cuda: Option<String>,
    pub at_ms: u64,
}

impl Record {
    /// PyTorch built for a newer CUDA than the driver: GPU kernels fail, so the install fails.
    pub fn torch_mismatch(&self) -> Option<String> {
        let driver = vllm_install::cuda_version(self.driver_cuda.as_deref()?)?;
        let built = vllm_install::cuda_version(self.torch_cuda.as_deref()?)?;
        (self.accel == Accel::Cuda).then(|| newer_than_driver(&format!("PyTorch {}", self.torch.as_deref().unwrap_or("?")), built, driver))?
    }

    /// vLLM's wheel built for a newer CUDA than the driver: a warning (text may still run).
    pub fn vllm_mismatch(&self) -> Option<String> {
        let driver = vllm_install::cuda_version(self.driver_cuda.as_deref()?)?;
        let built = vllm_install::cuda_version(self.vllm_cuda.as_deref()?)?;
        (self.accel == Accel::Cuda).then(|| newer_than_driver(&format!("vLLM {}", self.vllm), built, driver))?
    }
}

pub fn read_record(venv: &Path) -> Option<Record> {
    serde_json::from_slice(&std::fs::read(venv.join(RECORD)).ok()?).ok()
}

/// `cu130` (alone or ending a version such as `0.34.0+cu129`) -> `(13, 0)`.
pub fn cuda_tag(s: &str) -> Option<(u32, u32)> {
    let i = s.rfind("cu")?;
    let d: String = s[i + 2..].chars().take_while(|c| c.is_ascii_digit()).collect();
    if !(3..=4).contains(&d.len()) {
        return None;
    }
    let (major, minor) = d.split_at(d.len() - 1);
    Some((major.parse().ok()?, minor.parse().ok()?))
}

/// Runs the plan (blocking; it downloads), checks vLLM and vLLM-Omni, and writes the record.
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
        match vllm_install::python_version(&text) {
            Some((3, m)) if (11..=13).contains(&m) => {}
            v => bail!(
                "{} is Python {}; vLLM-Omni needs 3.11 to 3.13 (install `uv`, or pass --python)",
                first.program.display(),
                v.map_or("?".into(), |(a, b)| format!("{a}.{b}"))
            ),
        }
    }
    for step in &plan.steps {
        log(&step.to_string());
        let mut cmd = std::process::Command::new(&step.program);
        cmd.args(&step.args).stdin(std::process::Stdio::null());
        // As for vLLM: run from the engines dir, away from an operator's `uv.toml`.
        if let Some(parent) = plan.venv.parent().filter(|_| plan.venv.is_absolute()) {
            cmd.current_dir(parent);
        }
        let status = cmd.status().with_context(|| format!("running {step}"))?;
        if !status.success() {
            bail!("{step} failed ({status})");
        }
    }
    let vllm = vllm_install::program_version(&crate::vllm::bin_in(&plan.venv, "vllm")).context("vLLM was installed but does not run")?;
    let python = crate::vllm::bin_in(&plan.venv, "python");
    let omni = omni_version(&python).context("vLLM-Omni was installed but `import vllm_omni` failed")?;
    let (torch, torch_cuda) = vllm_install::torch_build(&python).map_or((None, None), |(v, c)| (Some(v), Some(c)));
    let at_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64);
    let rec = Record {
        engine: "vllm-omni".into(),
        accel: plan.accel,
        installer: plan.installer.into(),
        requested: o.version.clone(),
        omni,
        vllm_cuda: cuda_tag(&vllm).or(plan.vllm_cuda).map(|(a, b)| format!("{a}.{b}")),
        vllm,
        driver_cuda: plan.driver_cuda.map(|(a, b)| format!("{a}.{b}")),
        torch_backend: plan.torch_backend.clone(),
        torch,
        torch_cuda,
        at_ms,
    };
    std::fs::write(plan.venv.join(RECORD), serde_json::to_vec_pretty(&rec)?)?;
    if let Some(e) = rec.torch_mismatch() {
        bail!("{e}; it is kept in {} (see `ymirctl engines status`)", plan.venv.display());
    }
    Ok(rec)
}

/// vLLM-Omni's version in the venv ([`OMNI_CHECK`]).
fn omni_version(python: &Path) -> Result<String> {
    let o = std::process::Command::new(python).args(["-c", OMNI_CHECK]).stdin(std::process::Stdio::null()).output()?;
    let v = String::from_utf8_lossy(&o.stdout).trim().to_string();
    if !o.status.success() || vllm_install::version_in(&v).is_none() {
        bail!("{} -c ... printed `{v}` ({})", python.display(), o.status);
    }
    Ok(v)
}

/// Removes the managed venv, only when Ymir made it (its record is there).
pub fn uninstall(venv: &Path) -> Result<bool> {
    if !venv.exists() {
        return Ok(false);
    }
    if !venv.join(RECORD).is_file() {
        bail!("{} was not made by `ymirctl engines install vllm-omni` (no {RECORD}); not removing it", venv.display());
    }
    std::fs::remove_dir_all(venv).with_context(|| format!("removing {}", venv.display()))?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    struct Fake {
        on_path: Vec<&'static str>,
        os: &'static str,
        cuda: Option<(u32, u32)>,
    }

    #[cfg(unix)]
    impl Probe for Fake {
        fn which(&self, program: &str) -> Option<PathBuf> {
            self.on_path.contains(&program).then(|| PathBuf::from(format!("/usr/bin/{program}")))
        }
        fn exists(&self, _: &Path) -> bool {
            false
        }
        fn os(&self) -> &'static str {
            self.os
        }
        fn driver_cuda(&self) -> Option<(u32, u32)> {
            self.cuda
        }
    }

    #[cfg(unix)]
    fn fake(on_path: &[&'static str], cuda: Option<(u32, u32)>) -> Fake {
        Fake { on_path: on_path.to_vec(), os: "linux", cuda }
    }

    #[cfg(unix)]
    #[test]
    fn plans_a_pinned_pair_in_its_own_venv() {
        let v = venv(Path::new("/h/engines"));
        assert_eq!(v, Path::new("/h/engines/vllm-omni"));
        let p = plan(&v, &Options::default(), &fake(&["uv", "nvidia-smi"], Some((13, 0)))).unwrap();
        assert_eq!((p.installer, p.accel, p.version.as_str()), ("uv", Accel::Cuda, DEFAULT_VERSION));
        assert_eq!(p.steps[0].to_string(), "/usr/bin/uv venv --python 3.12 /h/engines/vllm-omni");
        assert_eq!(p.steps[1].to_string(), "/usr/bin/uv pip install --python /h/engines/vllm-omni/bin/python vllm==0.34.0 vllm-omni==0.34.0");
        assert_eq!((p.torch_backend.clone(), p.vllm_cuda, p.cuda_warning()), (None, Some(RELEASE_CUDA), None));

        let o = Options { version: Some("0.32.0".into()), upgrade: true, ..Default::default() };
        let p = plan(&v, &o, &fake(&["uv", "nvidia-smi"], Some((13, 0)))).unwrap();
        assert!(p.steps.last().unwrap().to_string().ends_with("--upgrade vllm==0.32.0 vllm-omni==0.32.0"), "{}", p.steps[1]);

        let p = plan(&v, &Options::default(), &fake(&["python3", "nvidia-smi"], Some((13, 0)))).unwrap();
        assert_eq!(p.installer, "pip");
        assert_eq!(p.steps.len(), 3);
        assert!(p.steps[2].to_string().ends_with("-m pip install vllm==0.34.0 vllm-omni==0.34.0"));
    }

    #[cfg(unix)]
    #[test]
    fn the_driver_checks_match_vllms() {
        let v = Path::new("/h/engines/vllm-omni");
        // A CUDA 12.8 driver: PyTorch's cu128, and a warning that vLLM's release wheel is 13.0.
        let p = plan(v, &Options::default(), &fake(&["uv", "nvidia-smi"], Some((12, 8)))).unwrap();
        assert_eq!(p.torch_backend.as_deref(), Some("cu128"));
        assert!(p.steps[1].to_string().contains("--torch-backend=cu128"));
        let w = p.cuda_warning().unwrap();
        assert!(w.contains("CUDA 13.0 build") && w.contains("runs CUDA 12.8"), "{w}");
        // Too old a driver is refused, as for vLLM.
        let e = plan(v, &Options::default(), &fake(&["uv", "nvidia-smi"], Some((12, 2)))).unwrap_err().to_string();
        assert!(e.contains("CUDA 12.2"), "{e}");
        // An operator's index: the wheel's CUDA is unknown until installed.
        let o = Options { index_url: Some("https://example.invalid/cu128".into()), ..Default::default() };
        assert_eq!(plan(v, &o, &fake(&["uv", "nvidia-smi"], Some((12, 8)))).unwrap().vllm_cuda, None);
        // Off CUDA an index is needed; off Linux --force is.
        assert!(plan(v, &Options::default(), &fake(&["uv"], None)).unwrap_err().to_string().contains("--index-url"));
        let mac = Fake { on_path: vec!["uv"], os: "macos", cuda: None };
        assert!(plan(v, &Options::default(), &mac).unwrap_err().to_string().contains("--force"));
        for bad in ["", "0.34; rm -rf /", "0.34 0"] {
            assert!(plan(v, &Options { version: Some(bad.into()), ..Default::default() }, &fake(&["uv", "nvidia-smi"], None)).is_err(), "{bad}");
        }
        assert!(plan(v, &Options { torch_backend: Some("cu1 30".into()), ..Default::default() }, &fake(&["uv", "nvidia-smi"], None)).is_err());
    }

    #[test]
    fn records_flag_a_newer_cuda_than_the_driver() {
        assert_eq!(cuda_tag("cu130"), Some((13, 0)));
        assert_eq!(cuda_tag("0.34.0+cu129"), Some((12, 9)));
        assert_eq!(cuda_tag("0.34.0"), None);
        let mut r = Record {
            engine: "vllm-omni".into(),
            accel: Accel::Cuda,
            installer: "uv".into(),
            requested: None,
            omni: "0.34.0".into(),
            vllm: "0.34.0".into(),
            driver_cuda: Some("12.8".into()),
            torch_backend: Some("cu128".into()),
            torch: Some("2.14.0+cu128".into()),
            torch_cuda: Some("12.8".into()),
            vllm_cuda: Some("13.0".into()),
            at_ms: 0,
        };
        assert!(r.torch_mismatch().is_none());
        assert!(r.vllm_mismatch().unwrap().contains("vLLM 0.34.0 is a CUDA 13.0 build"));
        r.torch_cuda = Some("13.0".into());
        assert!(r.torch_mismatch().unwrap().contains("PyTorch 2.14.0+cu128"));
        r.accel = Accel::Rocm;
        assert!(r.torch_mismatch().is_none() && r.vllm_mismatch().is_none());
    }
}
