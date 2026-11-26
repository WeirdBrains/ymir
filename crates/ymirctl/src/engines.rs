//! `ymirctl engines`: install, upgrade, check and remove the engines Ymir manages in its engines
//! dir. Today that is vLLM with Ray (`engines install vllm`, see
//! [`ymir_core::vllm_install`]): `ymird` finds the venv's `vllm` and `ray` on its own (`[model]
//! vllm = "auto"`, `ray = "auto"`), so nothing else needs editing. Restart `ymird` after an
//! install so it picks them up.

use std::io::Write;
use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Subcommand, ValueEnum};
use ymir_core::Paths;
use ymir_core::vllm_install::{self, Accel, Options, Probe, System};

#[derive(Subcommand, Debug)]
pub enum EnginesCmd {
    /// Which vLLM and Ray `ymird` would use (managed venv, PATH or node.toml), and their versions.
    Status,
    /// Install vLLM and Ray into a managed venv (`<home>/engines/vllm`), with `uv` when it is on
    /// PATH, else `python3 -m venv` and pip. Detects CUDA, ROCm or CPU.
    Install(InstallArgs),
    /// Upgrade the managed vLLM and Ray (to the newest, or to --version).
    Upgrade(InstallArgs),
    /// Remove the managed venv (only one this command made).
    Uninstall {
        #[arg(value_enum)]
        engine: Engine,
    },
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    /// vLLM with Ray (for `vllm-ray` split models).
    Vllm,
}

#[derive(Args, Debug)]
pub struct InstallArgs {
    #[arg(value_enum)]
    pub engine: Engine,
    /// A vLLM release from PyPI, such as 0.34.0. Default: vLLM's nightly, which has Inkling 2
    /// Mini (no release has it yet; 0.35.0 will).
    #[arg(long)]
    pub version: Option<String>,
    /// auto (nvidia-smi: cuda; rocminfo or /opt/rocm: rocm; else cpu), cuda, rocm or cpu.
    #[arg(long, default_value = "auto")]
    pub accel: String,
    /// A vLLM wheel index (needed for rocm and cpu: PyPI's wheels are CUDA builds).
    #[arg(long)]
    pub index_url: Option<String>,
    /// The `uv` to use (default the one on PATH).
    #[arg(long)]
    pub uv: Option<PathBuf>,
    /// Use this Python's `-m venv` and pip instead of uv.
    #[arg(long)]
    pub python: Option<PathBuf>,
    /// Make the venv afresh.
    #[arg(long)]
    pub reinstall: bool,
    /// Install on an OS vLLM does not support.
    #[arg(long)]
    pub force: bool,
    /// Print the commands, run nothing.
    #[arg(long)]
    pub dry_run: bool,
}

pub fn run(paths: &Paths, cmd: EnginesCmd, out: &mut dyn Write) -> Result<()> {
    run_with(paths, cmd, out, &System)
}

pub fn run_with(paths: &Paths, cmd: EnginesCmd, out: &mut dyn Write, probe: &dyn Probe) -> Result<()> {
    let venv = ymir_core::vllm::venv(&paths.engines());
    match cmd {
        EnginesCmd::Status => status(paths, out),
        EnginesCmd::Install(a) => install(&venv, a, false, out, probe),
        EnginesCmd::Upgrade(a) => install(&venv, a, true, out, probe),
        EnginesCmd::Uninstall { engine: Engine::Vllm } => {
            if vllm_install::uninstall(&venv)? {
                writeln!(out, "removed {}", venv.display())?;
                writeln!(out, "restart ymird (`ymirctl service stop` and `start`) so it stops using it")?;
            } else {
                writeln!(out, "no managed vLLM at {}", venv.display())?;
            }
            Ok(())
        }
    }
}

fn install(venv: &std::path::Path, a: InstallArgs, upgrade: bool, out: &mut dyn Write, probe: &dyn Probe) -> Result<()> {
    let Engine::Vllm = a.engine;
    let o = Options {
        version: a.version,
        accel: Accel::parse(&a.accel)?,
        index_url: a.index_url,
        uv: a.uv,
        python: a.python,
        upgrade,
        reinstall: a.reinstall,
        force: a.force,
    };
    if upgrade && vllm_install::read_record(venv).is_none() {
        anyhow::bail!("no managed vLLM at {}; run `ymirctl engines install vllm` first", venv.display());
    }
    let plan = vllm_install::plan(venv, &o, probe)?;
    let what = if plan.nightly { "vLLM nightly" } else { "vLLM" };
    writeln!(out, "{what} and Ray for {} into {} (with {})", plan.accel.name(), plan.venv.display(), plan.installer)?;
    if a.dry_run {
        for s in &plan.steps {
            writeln!(out, "  {s}")?;
        }
        writeln!(out, "  then check `vllm --version`, `ray --version` and that vLLM has Inkling 2 Mini (dry run: nothing ran)")?;
        return Ok(());
    }
    let mut lines = Vec::new();
    let rec = vllm_install::install(&plan, &o, &mut |l| lines.push(l.to_string()))?;
    for l in lines {
        writeln!(out, "  ran {l}")?;
    }
    writeln!(out, "installed vLLM {} and Ray {} ({})", rec.vllm, rec.ray, rec.accel.name())?;
    if rec.inkling {
        writeln!(out, "Inkling 2 Mini: supported (vLLM registers {})", ymir_core::vllm::INKLING_ARCH)?;
    } else {
        writeln!(
            out,
            "Inkling 2 Mini: not in this vLLM (it needs 0.35.0 or a nightly: run `ymirctl engines install vllm --reinstall` without --version)"
        )?;
    }
    writeln!(out, "ymird uses them with `[model] vllm = \"auto\"` and `ray = \"auto\"` (the defaults); restart ymird to pick them up")?;
    Ok(())
}

fn status(paths: &Paths, out: &mut dyn Write) -> Result<()> {
    let model = ymir_core::NodeConfig::load(&paths.config()).map(|c| c.model).unwrap_or_default();
    let venv = ymir_core::vllm::venv(&paths.engines());
    match vllm_install::read_record(&venv) {
        Some(r) => writeln!(
            out,
            "managed: {} (vLLM {}{}, Ray {}, {}, by {}, Inkling 2 Mini {}{})",
            venv.display(),
            r.vllm,
            if r.nightly { " nightly" } else { "" },
            r.ray,
            r.accel.name(),
            r.installer,
            if r.inkling { "yes" } else { "no" },
            r.requested.map(|v| format!(", asked for {v}")).unwrap_or_default()
        )?,
        None => writeln!(out, "managed: none (`ymirctl engines install vllm`)")?,
    }
    let show = |out: &mut dyn Write, name: &str, setting: &str, p: Option<PathBuf>| -> Result<()> {
        match p {
            Some(p) => {
                let from = if p.starts_with(&venv) { "managed" } else { "node.toml or PATH" };
                let v = vllm_install::program_version(&p).unwrap_or_else(|e| format!("does not run: {e}"));
                writeln!(out, "{name}: {} ({from}; {v}) [model] {name} = \"{setting}\"", p.display())?;
            }
            None => writeln!(out, "{name}: not found ([model] {name} = \"{setting}\")")?,
        }
        Ok(())
    };
    show(out, "vllm", &model.vllm, ymir_core::config::resolve_vllm(&model, paths))?;
    show(out, "ray", &model.ray, ymir_core::config::resolve_ray(&model, paths))?;
    Ok(())
}
