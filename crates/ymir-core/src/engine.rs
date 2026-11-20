//! The llama.cpp engine that ships inside Ymir's release files, and where `llamacpp = "managed"`
//! finds an engine.
//!
//! Every release file (tar.gz, zip, deb, rpm, Arch package, Homebrew) carries a pinned llama.cpp
//! build next to `ymird`: `llama-server`, `rpc-server` and their libraries, plus an `engine.toml`
//! that says which build and which GPU backends it holds. Nothing else has to be installed for a
//! model node to run GGUF models. The engine still runs as its own process under the supervisor,
//! so a model that crashes (a CUDA error, an out-of-memory abort) takes down only that engine,
//! never `ymird`, and the supervisor's restart and the coordinator's failover work as before.
//!
//! Where the bundled engine is looked for, relative to the running executable (symlinks resolved):
//!
//! | Layout | Directory |
//! |---|---|
//! | `$YMIR_ENGINE_DIR` (override) | that directory |
//! | Flat (Windows zip, update bundles) | `<bin>/llama.cpp/` |
//! | Prefix (tar.gz, packages, Homebrew) | `<bin>/../lib/ymir/llama.cpp/` |
//! | Prefix, libexec style | `<bin>/../libexec/ymir/llama.cpp/` |
//!
//! To be extracted: locating an engine shipped with an app and reading its `engine.toml` is generic
//! and belongs in `weirdnet-backends` (Weirdnet lead), next to its engine adapters.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use weirdnet_model::hw::{Accel, Hardware};

/// Overrides where the bundled engine is looked for.
pub const ENGINE_DIR_ENV: &str = "YMIR_ENGINE_DIR";
/// The file in the engine directory that describes the build.
pub const INFO_FILE: &str = "engine.toml";

/// What `engine.toml` says about a bundled build.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineInfo {
    /// `llama.cpp`.
    pub engine: String,
    /// The upstream build, such as `b12590`.
    pub build: String,
    /// The release variant: `default` (CPU and Vulkan, or Metal on macOS) or `cuda`.
    pub variant: String,
    /// GPU and CPU backends in this build: `cpu`, `vulkan`, `cuda`, `metal`, `rocm`.
    #[serde(default)]
    pub backends: Vec<String>,
    /// Where the build came from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

impl EngineInfo {
    pub fn has(&self, backend: &str) -> bool {
        self.backends.iter().any(|b| b == backend)
    }
}

/// An engine directory shipped with this install.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bundled {
    pub dir: PathBuf,
    /// `None` if `engine.toml` is missing or unreadable (the binaries still work).
    pub info: Option<EngineInfo>,
}

impl Bundled {
    /// Reads `dir`, if it holds a `llama-server`.
    pub fn at(dir: impl Into<PathBuf>) -> Option<Self> {
        let dir = dir.into();
        if !program(&dir, "llama-server").is_file() {
            return None;
        }
        let info = std::fs::read_to_string(dir.join(INFO_FILE)).ok().and_then(|t| toml::from_str(&t).ok());
        Some(Self { dir, info })
    }

    pub fn llama_server(&self) -> PathBuf {
        program(&self.dir, "llama-server")
    }

    pub fn rpc_server(&self) -> Option<PathBuf> {
        Some(program(&self.dir, "rpc-server")).filter(|p| p.is_file())
    }

    /// One line for `ymirctl setup`: `llama.cpp b12590 (cuda: cpu, cuda)`.
    pub fn describe(&self) -> String {
        match &self.info {
            Some(i) => format!("{} {} ({}: {})", i.engine, i.build, i.variant, i.backends.join(", ")),
            None => "llama.cpp (no engine.toml)".into(),
        }
    }
}

/// `dir/name`, with `.exe` on Windows.
pub fn program(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{name}{}", std::env::consts::EXE_SUFFIX))
}

/// The directories a bundled engine may be in, for an executable at `exe`, in search order.
pub fn candidates(exe: &Path) -> Vec<PathBuf> {
    let Some(bin) = exe.parent() else { return Vec::new() };
    let mut out = vec![bin.join("llama.cpp")];
    if let Some(prefix) = bin.parent() {
        out.push(prefix.join("lib").join("ymir").join("llama.cpp"));
        out.push(prefix.join("libexec").join("ymir").join("llama.cpp"));
    }
    out
}

/// The bundled engine for an executable at `exe`.
pub fn bundled_from(exe: &Path) -> Option<Bundled> {
    candidates(exe).into_iter().find_map(Bundled::at)
}

/// The engine shipped with the running `ymird` or `ymirctl` (or in `$YMIR_ENGINE_DIR`).
pub fn bundled() -> Option<Bundled> {
    if let Some(dir) = std::env::var_os(ENGINE_DIR_ENV).filter(|d| !d.is_empty()) {
        return Bundled::at(PathBuf::from(dir));
    }
    let exe = std::env::current_exe().ok()?;
    let exe = std::fs::canonicalize(&exe).unwrap_or(exe);
    bundled_from(&exe)
}

/// A hint when the bundled build can't use this machine's accelerator well.
pub fn advice(hw: &Hardware, info: &EngineInfo) -> Option<String> {
    let nvidia = hw.gpus.iter().any(|g| g.accel == Accel::Nvidia);
    let amd = hw.gpus.iter().any(|g| g.accel == Accel::Amd);
    if nvidia && !info.has("cuda") {
        let how = if info.has("vulkan") { "it runs on the GPU through Vulkan, which is slower than CUDA" } else { "it runs on the CPU only" };
        return Some(format!(
            "this is the `{}` build of llama.cpp and this machine has an NVIDIA GPU: {how}. Install the CUDA build of Ymir (the `ymir-cuda` package, or the `-cuda` archive) for full speed",
            info.variant
        ));
    }
    if amd && !info.has("vulkan") && !info.has("rocm") {
        return Some(format!(
            "this is the `{}` build of llama.cpp, which can't use this machine's AMD GPU; install the default build of Ymir (Vulkan)",
            info.variant
        ));
    }
    None
}

#[cfg(test)]
mod tests {
    use weirdnet_model::hw::Gpu;

    use super::*;

    fn touch_exe(p: &Path) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, b"#!/bin/sh\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
    }

    #[test]
    fn finds_the_engine_in_each_layout() {
        let t = tempfile::tempdir().unwrap();
        // Prefix layout: bin/ymird and lib/ymir/llama.cpp/.
        let prefix = t.path().join("prefix");
        let dir = prefix.join("lib/ymir/llama.cpp");
        touch_exe(&program(&dir, "llama-server"));
        std::fs::write(dir.join(INFO_FILE), "engine = \"llama.cpp\"\nbuild = \"b1\"\nvariant = \"cuda\"\nbackends = [\"cpu\", \"cuda\"]\n").unwrap();
        let b = bundled_from(&prefix.join("bin/ymird")).unwrap();
        assert_eq!(b.dir, dir);
        assert_eq!(b.llama_server(), program(&dir, "llama-server"));
        assert!(b.rpc_server().is_none());
        assert_eq!(b.describe(), "llama.cpp b1 (cuda: cpu, cuda)");

        // Flat layout wins over the prefix one: an update bundle carries its own engine.
        let flat = prefix.join("bin/llama.cpp");
        touch_exe(&program(&flat, "llama-server"));
        touch_exe(&program(&flat, "rpc-server"));
        let b = bundled_from(&prefix.join("bin/ymird")).unwrap();
        assert_eq!(b.dir, flat);
        assert!(b.info.is_none());
        assert!(b.rpc_server().is_some());

        // Nothing there.
        assert!(bundled_from(&t.path().join("elsewhere/bin/ymird")).is_none());
    }

    #[test]
    fn advises_the_cuda_build_on_nvidia() {
        let gpu = |accel| Gpu { accel, name: "gpu".into(), memory_bytes: 12 << 30, unified: false, compute_capability: Some((8, 6)), driver: None };
        let hw = |gpus| Hardware { os: "linux".into(), arch: "x86_64".into(), cpu_threads: 8, ram_bytes: 32 << 30, gpus };
        let info = |variant: &str, b: &[&str]| EngineInfo {
            engine: "llama.cpp".into(),
            build: "b1".into(),
            variant: variant.into(),
            backends: b.iter().map(|s| s.to_string()).collect(),
            source: None,
        };
        let default = info("default", &["cpu", "vulkan"]);
        let cuda = info("cuda", &["cpu", "cuda"]);
        assert!(advice(&hw(vec![gpu(Accel::Nvidia)]), &default).unwrap().contains("ymir-cuda"));
        assert!(advice(&hw(vec![gpu(Accel::Nvidia)]), &cuda).is_none());
        assert!(advice(&hw(vec![]), &default).is_none());
        assert!(advice(&hw(vec![gpu(Accel::Amd)]), &default).is_none());
        assert!(advice(&hw(vec![gpu(Accel::Amd)]), &cuda).unwrap().contains("Vulkan"));
    }
}
