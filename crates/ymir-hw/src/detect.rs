//! What accelerators and memory this machine has.

use std::process::Command;

use serde::{Deserialize, Serialize};

/// The accelerator family that decides which engine build runs here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Accel {
    Nvidia,
    Amd,
    Apple,
    Cpu,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gpu {
    pub accel: Accel,
    pub name: String,
    /// Dedicated memory; for unified-memory devices, the system memory.
    pub memory_bytes: u64,
    /// The device shares system memory (Apple, GB10, AMD APUs).
    pub unified: bool,
    /// NVIDIA compute capability, such as `(8, 6)`.
    pub compute_capability: Option<(u32, u32)>,
    pub driver: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hardware {
    pub os: String,
    pub arch: String,
    pub cpu_threads: usize,
    pub ram_bytes: u64,
    pub gpus: Vec<Gpu>,
}

impl Hardware {
    /// The accelerator of the first GPU, else CPU.
    pub fn accel(&self) -> Accel {
        self.gpus.first().map(|g| g.accel).unwrap_or(Accel::Cpu)
    }

    /// Models live in system memory (unified devices and CPU-only machines).
    pub fn unified(&self) -> bool {
        self.gpus.first().is_none_or(|g| g.unified)
    }

    /// Whether vLLM can run here: Linux on NVIDIA sm_80 or newer. (AMD ROCm support depends on
    /// vLLM's list of gfx targets and is not detected yet.)
    pub fn vllm_supported(&self) -> bool {
        self.os == "linux" && self.gpus.iter().any(|g| g.accel == Accel::Nvidia && g.compute_capability.is_some_and(|(major, _)| major >= 8))
    }

    /// NVIDIA sm_121 (GB10) needs CUDA 13 engine builds.
    pub fn needs_cuda13(&self) -> bool {
        self.gpus.iter().any(|g| g.accel == Accel::Nvidia && g.compute_capability.is_some_and(|(major, minor)| major >= 12 && minor >= 1))
    }
}

/// Probes this machine. Never fails: missing tools just mean fewer devices.
pub fn detect() -> Hardware {
    let ram_bytes = system_memory().unwrap_or(0);
    let mut gpus = Vec::new();
    if let Some(out) = run("nvidia-smi", &["--query-gpu=name,memory.total,compute_cap,driver_version", "--format=csv,noheader,nounits"]) {
        gpus.extend(parse_nvidia_smi(&out, ram_bytes));
    }
    if gpus.is_empty() {
        gpus.extend(amd_sysfs(ram_bytes));
    }
    if gpus.is_empty() && cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") {
        gpus.push(Gpu {
            accel: Accel::Apple,
            name: apple_chip().unwrap_or_else(|| "Apple silicon".into()),
            memory_bytes: ram_bytes,
            unified: true,
            compute_capability: None,
            driver: None,
        });
    }
    Hardware {
        os: std::env::consts::OS.into(),
        arch: std::env::consts::ARCH.into(),
        cpu_threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        ram_bytes,
        gpus,
    }
}

fn run(program: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(program).args(args).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

fn system_memory() -> Option<u64> {
    if cfg!(target_os = "linux") {
        return parse_meminfo_total(&std::fs::read_to_string("/proc/meminfo").ok()?);
    }
    if cfg!(any(target_os = "macos", target_os = "freebsd", target_os = "openbsd", target_os = "netbsd")) {
        let key = if cfg!(target_os = "macos") { "hw.memsize" } else { "hw.physmem" };
        return run("sysctl", &["-n", key])?.trim().parse().ok();
    }
    None
}

fn apple_chip() -> Option<String> {
    run("sysctl", &["-n", "machdep.cpu.brand_string"]).map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

/// `MemTotal` from `/proc/meminfo`, in bytes.
pub fn parse_meminfo_total(meminfo: &str) -> Option<u64> {
    let line = meminfo.lines().find(|l| l.starts_with("MemTotal:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib * 1024)
}

/// Parses `nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader,nounits`.
/// A device without dedicated memory (`[N/A]`, as on GB10) is unified and gets `ram_bytes`.
pub fn parse_nvidia_smi(csv: &str, ram_bytes: u64) -> Vec<Gpu> {
    csv.lines()
        .filter_map(|line| {
            let cols: Vec<&str> = line.split(',').map(str::trim).collect();
            if cols.len() < 4 || cols[0].is_empty() {
                return None;
            }
            let mib = cols[1].parse::<u64>().ok();
            let cc = cols[2].split_once('.').and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)));
            let unified = mib.is_none() || cols[0].contains("GB10");
            Some(Gpu {
                accel: Accel::Nvidia,
                name: cols[0].to_string(),
                memory_bytes: if unified { ram_bytes } else { mib.unwrap_or(0) * 1024 * 1024 },
                unified,
                compute_capability: cc,
                driver: Some(cols[3].to_string()).filter(|d| !d.is_empty()),
            })
        })
        .collect()
}

fn amd_sysfs(ram_bytes: u64) -> Vec<Gpu> {
    let Ok(entries) = std::fs::read_dir("/sys/class/drm") else { return Vec::new() };
    let mut gpus = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let dev = entry.path().join("device");
        if std::fs::read_to_string(dev.join("vendor")).map(|v| v.trim() != "0x1002").unwrap_or(true) {
            continue;
        }
        let vram: u64 = std::fs::read_to_string(dev.join("mem_info_vram_total")).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(0);
        // APUs report a small carve-out; treat anything under 4 GiB as unified memory.
        let unified = vram < 4 * crate::GIB;
        gpus.push(Gpu {
            accel: Accel::Amd,
            name: "AMD GPU".into(),
            memory_bytes: if unified { ram_bytes } else { vram },
            unified,
            compute_capability: None,
            driver: None,
        });
    }
    gpus
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvidia_discrete() {
        let gpus = parse_nvidia_smi("NVIDIA GeForce GPU, 12288, 8.6, 615.78.08\n", 64 << 30);
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].memory_bytes, 12 << 30);
        assert!(!gpus[0].unified);
        assert_eq!(gpus[0].compute_capability, Some((8, 6)));
    }

    #[test]
    fn gb10_is_unified() {
        let gpus = parse_nvidia_smi("NVIDIA GB10, [N/A], 12.1, 580.95.05\n", 128 << 30);
        assert!(gpus[0].unified);
        assert_eq!(gpus[0].memory_bytes, 128 << 30);
        let hw = Hardware { os: "linux".into(), arch: "aarch64".into(), cpu_threads: 20, ram_bytes: 128 << 30, gpus };
        assert!(hw.vllm_supported());
        assert!(hw.needs_cuda13());
    }

    #[test]
    fn meminfo() {
        assert_eq!(parse_meminfo_total("MemTotal:       65536 kB\nMemFree: 1 kB\n"), Some(65536 * 1024));
    }

    #[test]
    fn detect_does_not_panic() {
        let hw = detect();
        assert!(hw.cpu_threads >= 1);
    }
}
