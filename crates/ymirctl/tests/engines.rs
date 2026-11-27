//! `ymirctl engines`: install, status, upgrade and uninstall of the managed vLLM + Ray venv, with
//! fake `uv` and `python3` scripts that make a venv with stub `vllm` and `ray` (nothing is
//! downloaded), and `ymird`'s discovery of them with `[model] vllm = "auto"`, `ray = "auto"`.
#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use clap::Parser as _;
use ymir_core::Paths;
use ymir_core::vllm_install::Probe;

/// Writes the stub `vllm` and `ray` a real install would leave in `bin`.
const STUBS: &str = r#"
stubs() {
  case "$2" in *vllm==*) v=0.34.0 ink=False ;; *) v=0.34.1rc1.dev231+g43108ffcc ink=True ;; esac
  printf '#!/bin/sh\necho %s\n' "$v" > "$1/vllm"
  printf '#!/bin/sh\necho "ray, version 2.49.1"\n' > "$1/ray"
  # The venv's python answers the registry check and PyTorch's CUDA (bin/torch_cuda, default 13.0).
  [ -s "$1/python" ] || printf '#!/bin/sh\ncase "$2" in *torch.version*) echo "torch 2.14.0 $(cat "$(dirname "$0")/torch_cuda" 2>/dev/null || echo 13.0)" ;; *) echo %s ;; esac\n' "$ink" > "$1/python"
  chmod +x "$1/vllm" "$1/ray" "$1/python"
}
"#;

fn script(path: &Path, body: &str) {
    std::fs::write(path, format!("#!/bin/sh\n{STUBS}\necho \"$0 $*\" >> \"$(dirname \"$0\")/calls.log\"\n{body}")).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

/// `uv venv --python V [--clear] DIR` makes DIR/bin/python; `uv pip install --python PY ...`
/// puts the stubs next to PY.
fn fake_uv(dir: &Path) -> PathBuf {
    let uv = dir.join("uv");
    script(
        &uv,
        r#"if [ "$1" = venv ]; then for a; do d="$a"; done; mkdir -p "$d/bin"; : > "$d/bin/python"; exit 0; fi
if [ "$1" = pip ]; then stubs "$(dirname "$4")" "$*"; exit 0; fi
exit 2
"#,
    );
    uv
}

/// `python3 --version`, and `python3 -m venv DIR` making a venv whose python's `-m pip install`
/// puts the stubs in.
fn fake_python(dir: &Path, version: &str) -> PathBuf {
    let py = dir.join("python3");
    script(
        &py,
        &format!(
            r#"if [ "$1" = --version ]; then echo "Python {version}"; exit 0; fi
if [ "$1" = -m ] && [ "$2" = venv ]; then for a; do d="$a"; done; mkdir -p "$d/bin"; cp "$0" "$d/bin/python"; exit 0; fi
if [ "$1" = -m ] && [ "$2" = pip ]; then stubs "$(dirname "$0")" "$*"; exit 0; fi
if [ "$1" = -c ]; then echo True; exit 0; fi
exit 2
"#
        ),
    );
    py
}

struct Linux {
    uv: Option<PathBuf>,
}

impl Probe for Linux {
    fn which(&self, program: &str) -> Option<PathBuf> {
        match program {
            "uv" => self.uv.clone(),
            "nvidia-smi" => Some("/usr/bin/nvidia-smi".into()),
            _ => None,
        }
    }
    fn exists(&self, path: &Path) -> bool {
        path.exists()
    }
    fn os(&self) -> &'static str {
        "linux"
    }
    fn driver_cuda(&self) -> Option<(u32, u32)> {
        Some((13, 0))
    }
}

fn engines(home: &Path, args: &[&str], probe: &dyn Probe) -> anyhow::Result<String> {
    let mut argv = vec!["ymirctl".to_string(), "--home".into(), home.display().to_string(), "engines".into()];
    argv.extend(args.iter().map(|a| a.to_string()));
    let cli = ymirctl::Cli::try_parse_from(argv)?;
    let ymirctl::Cmd::Engines { cmd } = cli.cmd else { unreachable!() };
    let mut out = Vec::new();
    ymirctl::engines::run_with(&Paths::new(home), cmd, &mut out, probe)?;
    Ok(String::from_utf8(out)?)
}

#[test]
fn install_with_uv_then_ymird_finds_vllm_and_ray_then_upgrade_and_uninstall() {
    let d = tempfile::tempdir().unwrap();
    let (home, tools) = (d.path().join("home"), d.path().join("tools"));
    std::fs::create_dir_all(&tools).unwrap();
    let probe = Linux { uv: Some(fake_uv(&tools)) };
    let paths = Paths::new(&home);
    let venv = paths.engines().join("vllm");

    // Dry run: the plan, and nothing made.
    let o = engines(&home, &["install", "vllm", "--dry-run"], &probe).unwrap();
    assert!(o.contains("vLLM nightly") && o.contains("for cuda") && o.contains("with uv") && o.contains("wheels.vllm.ai/nightly vllm ray"), "{o}");
    assert!(o.contains("Inkling 2 Mini") && o.contains("dry run"), "{o}");
    assert!(o.contains("GPU driver: CUDA 13.0; PyTorch: PyPI's (CUDA 13.0 build)") && !o.contains("--torch-backend"), "{o}");
    assert!(
        o.contains("vLLM's kernels: CUDA 13.2 builds") && o.contains("warning: newer than the driver's CUDA 13.0") && o.contains("--vllm-cuda cu130"),
        "{o}"
    );
    let o = engines(&home, &["install", "vllm", "--vllm-cuda", "cu130", "--dry-run"], &probe).unwrap();
    assert!(o.contains("wheels.vllm.ai/nightly/cu130 vllm ray") && !o.contains("warning"), "{o}");
    assert!(!venv.exists() && !tools.join("calls.log").exists());
    assert!(engines(&home, &["upgrade", "vllm"], &probe).unwrap_err().to_string().contains("install vllm` first"));

    // A pinned release without Inkling installs, and says so.
    let o = engines(&home, &["install", "vllm", "--version", "0.34.0"], &probe).unwrap();
    assert!(o.contains("installed vLLM 0.34.0") && o.contains("Inkling 2 Mini: not in this vLLM"), "{o}");
    let rec = ymir_core::vllm_install::read_record(&venv).unwrap();
    assert_eq!((rec.requested.as_deref(), rec.nightly, rec.inkling), (Some("0.34.0"), false, false));
    assert!(!ymir_core::vllm::managed_with_inkling(&venv.join("bin/vllm")));

    // The default: the nightly, which has Inkling, and ymird's preflight trusts the record.
    let o = engines(&home, &["install", "vllm", "--reinstall"], &probe).unwrap();
    assert!(o.contains("installed vLLM 0.34.1rc1.dev231+g43108ffcc and Ray 2.49.1 (cuda)") && o.contains("Inkling 2 Mini: supported"), "{o}");
    let rec = ymir_core::vllm_install::read_record(&venv).unwrap();
    assert_eq!((rec.requested.as_deref(), rec.installer.as_str(), rec.nightly, rec.inkling), (None, "uv", true, true));
    assert!(ymir_core::vllm::managed_with_inkling(&venv.join("bin/vllm")));
    assert_eq!((rec.driver_cuda.as_deref(), rec.torch_cuda.as_deref(), rec.torch_backend.as_deref()), (Some("13.0"), Some("13.0"), None));
    assert!(o.contains("PyTorch 2.14.0 (CUDA 13.0; the driver runs 13.0)"), "{o}");
    assert_eq!(rec.vllm_cuda.as_deref(), Some("13.2"), "the default nightly's kernels");
    assert!(o.contains("warning: vLLM 0.34.1rc1.dev231+g43108ffcc's kernels are CUDA 13.2 builds"), "{o}");
    assert!(ymir_core::vllm::ensure_inkling(&venv.join("bin/vllm"), false).is_ok(), "a nightly numbered below 0.35.0 is accepted by its record");

    // Auto-discovery: no node.toml edit, no PATH.
    let model = ymir_core::config::ModelSection::default();
    assert_eq!(ymir_core::config::resolve_vllm(&model, &paths), Some(venv.join("bin/vllm")));
    assert_eq!(ymir_core::config::resolve_ray(&model, &paths), Some(venv.join("bin/ray")));
    let cfg: ymir_core::NodeConfig = toml::from_str("[node]\nname = \"n\"\n").unwrap();
    let w = cfg.worker_config(&paths, "k".into());
    assert_eq!((w.vllm, w.ray), (Some(venv.join("bin/vllm")), Some(venv.join("bin/ray"))));
    // An explicit setting still wins.
    let off = ymir_core::config::ModelSection { ray: "off".into(), vllm: "system:/opt/vllm".into(), ..Default::default() };
    assert_eq!(ymir_core::config::resolve_ray(&off, &paths), None);
    assert_eq!(ymir_core::config::resolve_vllm(&off, &paths), Some("/opt/vllm".into()));

    let s = engines(&home, &["status"], &probe).unwrap();
    assert!(s.contains("managed:") && s.contains("nightly, Ray 2.49.1") && s.contains("Inkling 2 Mini yes"), "{s}");
    assert!(s.contains("(managed; 0.34.1rc1.dev231+g43108ffcc)") && s.contains("(managed; 2.49.1)"), "{s}");

    // Upgrade keeps the venv and asks for --upgrade.
    let calls_before = std::fs::read_to_string(tools.join("calls.log")).unwrap().lines().count();
    engines(&home, &["upgrade", "vllm"], &probe).unwrap();
    let calls = std::fs::read_to_string(tools.join("calls.log")).unwrap();
    let new: Vec<&str> = calls.lines().skip(calls_before).collect();
    assert_eq!(new.len(), 1, "no new venv: {new:?}");
    assert!(new[0].contains("pip install") && new[0].contains("--upgrade") && new[0].contains("nightly vllm ray"), "{new:?}");

    // A PyTorch built for a newer CUDA than the driver's fails the install, and says what to do.
    std::fs::write(venv.join("bin/torch_cuda"), "13.2").unwrap();
    let e = engines(&home, &["upgrade", "vllm"], &probe).unwrap_err().to_string();
    assert!(e.contains("CUDA 13.2 build") && e.contains("driver runs CUDA 13.0"), "{e}");
    let o = engines(&home, &["install", "vllm", "--torch-backend", "cu130", "--dry-run"], &probe).unwrap();
    assert!(o.contains("--torch-backend=cu130") && o.contains("PyTorch: cu130"), "{o}");

    // Uninstall removes it; a directory Ymir did not make is left alone.
    assert!(engines(&home, &["uninstall", "vllm"], &probe).unwrap().contains("removed"));
    assert!(!venv.exists());
    assert!(engines(&home, &["status"], &probe).unwrap().contains("managed: none"));
    std::fs::create_dir_all(venv.join("bin")).unwrap();
    assert!(engines(&home, &["uninstall", "vllm"], &probe).is_err());
    assert!(venv.exists());
}

#[test]
fn install_with_python_venv_and_pip_needs_a_python_vllm_supports() {
    let d = tempfile::tempdir().unwrap();
    let (home, tools) = (d.path().join("home"), d.path().join("tools"));
    std::fs::create_dir_all(&tools).unwrap();
    let probe = Linux { uv: None };
    let old = fake_python(&tools, "3.8.10");
    let e = engines(&home, &["install", "vllm", "--python", old.to_str().unwrap()], &probe).unwrap_err().to_string();
    assert!(e.contains("3.10 to 3.13"), "{e}");
    let py = fake_python(&tools, "3.12.3");
    let o = engines(&home, &["install", "vllm", "--python", py.to_str().unwrap()], &probe).unwrap();
    assert!(o.contains("with pip") && o.contains("installed vLLM 0.34.1rc1") && o.contains("Inkling 2 Mini: supported"), "{o}");
    assert_eq!(ymir_core::vllm_install::read_record(&Paths::new(&home).engines().join("vllm")).unwrap().installer, "pip");
    // CPU needs a wheel index.
    let e = engines(&home, &["install", "vllm", "--accel", "cpu", "--reinstall", "--dry-run"], &probe).unwrap_err().to_string();
    assert!(e.contains("--index-url"), "{e}");
}

/// `uv` for vLLM-Omni: the venv's python answers `import vllm_omni` (its version), PyTorch's
/// CUDA (bin/torch_cuda, default 13.0); `vllm` prints the pinned release.
fn fake_omni_uv(dir: &Path) -> PathBuf {
    let uv = dir.join("uv-omni");
    std::fs::write(
        &uv,
        r#"#!/bin/sh
echo "$0 $*" >> "$(dirname "$0")/omni-calls.log"
if [ "$1" = venv ]; then for a; do d="$a"; done; mkdir -p "$d/bin"; exit 0; fi
if [ "$1" = pip ]; then
  b="$(dirname "$4")"
  for a; do case "$a" in vllm-omni==*) v="${a#vllm-omni==}" ;; esac; done
  printf '#!/bin/sh\necho %s\n' "$v" > "$b/vllm"
  printf '#!/bin/sh\ncase "$2" in *torch.version*) echo "torch 2.14.0 $(cat "$(dirname "$0")/torch_cuda" 2>/dev/null || echo 13.0)" ;; *vllm-omni*) echo %s ;; *) exit 1 ;; esac\n' "$v" > "$b/python"
  chmod +x "$b/vllm" "$b/python"; exit 0
fi
exit 2
"#,
    )
    .unwrap();
    std::fs::set_permissions(&uv, std::fs::Permissions::from_mode(0o755)).unwrap();
    uv
}

#[test]
fn vllm_omni_installs_in_its_own_venv_with_the_driver_checks() {
    let d = tempfile::tempdir().unwrap();
    let (home, tools) = (d.path().join("home"), d.path().join("tools"));
    std::fs::create_dir_all(&tools).unwrap();
    let probe = Linux { uv: Some(fake_omni_uv(&tools)) };
    let paths = Paths::new(&home);
    let venv = paths.engines().join("vllm-omni");

    let o = engines(&home, &["install", "vllm-omni", "--dry-run"], &probe).unwrap();
    assert!(o.contains("vLLM-Omni 0.34.0 and vLLM 0.34.0 for cuda") && o.contains("vllm==0.34.0 vllm-omni==0.34.0"), "{o}");
    assert!(o.contains("GPU driver: CUDA 13.0") && o.contains("vLLM's kernels: CUDA 13.0 builds") && !o.contains("warning"), "{o}");
    assert!(o.contains("dry run") && !venv.exists() && !tools.join("omni-calls.log").exists(), "{o}");
    assert!(engines(&home, &["upgrade", "vllm-omni"], &probe).unwrap_err().to_string().contains("install vllm-omni` first"));

    let o = engines(&home, &["install", "vllm-omni", "--version", "0.32.0"], &probe).unwrap();
    assert!(o.contains("installed vLLM-Omni 0.32.0 and vLLM 0.32.0 (cuda)") && o.contains("PyTorch 2.14.0 (CUDA 13.0"), "{o}");
    let rec = ymir_core::vllm_omni_install::read_record(&venv).unwrap();
    assert_eq!((rec.omni.as_str(), rec.vllm_cuda.as_deref(), rec.requested.as_deref()), ("0.32.0", Some("13.0"), Some("0.32.0")));
    assert!(!paths.engines().join("vllm").exists(), "the vLLM venv is untouched");
    let s = engines(&home, &["status"], &probe).unwrap();
    assert!(s.contains("vllm-omni: ") && s.contains("vLLM-Omni 0.32.0"), "{s}");

    // PyTorch built for a newer CUDA than the driver fails the install, and keeps the record.
    std::fs::write(venv.join("bin/torch_cuda"), "13.2").unwrap();
    let e = engines(&home, &["upgrade", "vllm-omni"], &probe).unwrap_err().to_string();
    assert!(e.contains("CUDA 13.2 build") && e.contains("runs CUDA 13.0"), "{e}");

    assert!(engines(&home, &["uninstall", "vllm-omni"], &probe).unwrap().contains("removed"));
    assert!(!venv.exists());
    std::fs::create_dir_all(&venv).unwrap();
    assert!(engines(&home, &["uninstall", "vllm-omni"], &probe).unwrap_err().to_string().contains("not removing"));
}
