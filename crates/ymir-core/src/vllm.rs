//! vLLM version checks, Inkling 2 Mini launch flags, and a container-image wrapper.
//!
//! vLLM stays an external engine (`auto`, `system:PATH`, a wrapped `vllm serve`, or
//! `image:registry/name:tag`). Inkling 2 Mini (`inkling2_loop`) needs vLLM 0.35.0 or a newer
//! nightly, or the `vllm/vllm-openai:inkling-2-mini` image, until that support is in a release
//! (<https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini>). A too-old binary gets a clear error
//! rather than a load failure from inside vLLM.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Result, bail};
use weirdnet_model::hw::{Accel, Hardware};

/// First vLLM version whose recipe includes Inkling 2 Mini. 0.34.0 does not. Support landed on
/// main in vllm-project/vllm#69028 and is also published as the `inkling-2-mini` image until a
/// release has it (`nightly_required` on 0.35.0). Nightlies built after it (merge commit
/// ab18334, 2026-11-21) carry it but are numbered below this (`0.34.1rc1.devN+gSHA`), so the
/// managed install ([`crate::vllm_install`]) checks vLLM's model registry instead and records it.
pub const INKLING_MIN_VERSION: (u64, u64, u64) = (0, 35, 0);

/// The architecture vLLM registers for Inkling 2 Mini (`vllm/model_executor/models/registry.py`).
pub const INKLING_ARCH: &str = "InklingForConditionalGeneration";
/// The class vLLM main registers Inkling 2 Mini (`inkling2_loop`) as.
pub const INKLING2_LOOP_ARCH: &str = "Inkling2LoopForConditionalGeneration";

/// Day-0 image with Inkling 2 Mini, including linux/arm64 for DGX Spark.
pub const INKLING_IMAGE: &str = "vllm/vllm-openai:inkling-2-mini";

/// The context the vLLM recipe caps Inkling at on a DGX Spark. Ymir no longer sets it itself (ymird
/// sizes `--max-model-len` from the model and the node, see [`crate::kvfit`]); it is only the
/// example for a container started by hand and wrapped.
pub const SPARK_MAX_MODEL_LEN: u32 = 262_144;

/// The managed vLLM venv (`ymirctl engines install vllm`): vLLM and Ray, under the engines dir.
pub fn venv(engines_dir: &Path) -> PathBuf {
    engines_dir.join("vllm")
}

/// A program in a venv: `bin/NAME` (`Scripts\\NAME.exe` on Windows).
pub fn bin_in(venv: &Path, name: &str) -> PathBuf {
    if cfg!(windows) { venv.join("Scripts").join(format!("{name}.exe")) } else { venv.join("bin").join(name) }
}

/// A Hugging Face repo, file path or Hub id that is Inkling 2 Mini (or its varloop research
/// checkpoint).
pub fn is_inkling(source: &str) -> bool {
    let s = source.to_ascii_lowercase();
    s.contains("inkling-2-mini") || s.contains("inkling2_loop") || s.contains("inkling2-mini") || s.contains("tml-d24-varloop")
}

/// NVIDIA GB10 / DGX Spark (sm_121) or a GPU whose name says so.
pub fn is_spark_class(hw: &Hardware) -> bool {
    hw.needs_cuda13()
        || hw.gpus.iter().any(|g| {
            g.accel == Accel::Nvidia && {
                let n = g.name.to_ascii_lowercase();
                n.contains("gb10") || n.contains("spark")
            }
        })
}

/// Extra `vllm serve` arguments for Inkling 2 Mini. The context is not among them: ymird sets
/// `--max-model-len` from the model's trained length and this node's memory ([`crate::kvfit`]).
pub fn inkling_args() -> Vec<String> {
    vec![
        "--reasoning-parser".into(),
        "inkling".into(),
        "--tool-call-parser".into(),
        "inkling".into(),
        "--enable-auto-tool-choice".into(),
        "--speculative-config".into(),
        r#"{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}"#.into(),
    ]
}

/// The official image to use for Inkling, if `source` is that model.
pub fn image_for(source: &str) -> Option<&'static str> {
    is_inkling(source).then_some(INKLING_IMAGE)
}

/// Why a node's vLLM cannot serve Inkling 2 Mini, with the image and wrap path to use instead.
pub fn inkling_unsupported_message(found: Option<(u64, u64, u64)>) -> String {
    let have = match found {
        Some((a, b, c)) => format!("{a}.{b}.{c}"),
        None => "unknown".into(),
    };
    format!(
        "this node's vLLM ({have}) cannot load Inkling 2 Mini (inkling2_loop). It needs vLLM {}.{}.{} or a nightly built since 2026-11-21 (`ymirctl engines install vllm` installs one and checks it), or the {INKLING_IMAGE} image (support is not in 0.34.0, and is not in a vLLM release yet). Point `[model] vllm` at `image:{INKLING_IMAGE}`, or start the image yourself and wrap it: `docker run --gpus all --ipc=host -p 8000:8000 -v ~/.cache/huggingface:/root/.cache/huggingface {INKLING_IMAGE} thinkingmachines/Inkling-2-Mini --reasoning-parser inkling --tool-call-parser inkling --enable-auto-tool-choice --max-model-len {SPARK_MAX_MODEL_LEN}` then set `wrap = \"http://127.0.0.1:8000/v1\"` and `context = {SPARK_MAX_MODEL_LEN}` (the same length) on the model",
        INKLING_MIN_VERSION.0, INKLING_MIN_VERSION.1, INKLING_MIN_VERSION.2
    )
}

/// Whether a probed vLLM can load `inkling2_loop`. `via_image` or a wrap skips the check (the
/// operator picked a known-good runtime). A version at or above [`INKLING_MIN_VERSION`], or help
/// text that mentions `inkling`, is enough.
pub fn inkling_supported(version: Option<(u64, u64, u64)>, mentions_inkling: bool, via_image: bool) -> bool {
    via_image || mentions_inkling || version.is_some_and(|v| v >= INKLING_MIN_VERSION)
}

/// The first `X.Y.Z` triple in `s` (so `vLLM 0.35.0.dev+...` and `0.34.0` both work).
pub fn parse_version(s: &str) -> Option<(u64, u64, u64)> {
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_digit() {
            if let Some((ver, _end)) = triple_at(s, i) {
                return Some(ver);
            }
            while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'.') {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    None
}

fn triple_at(s: &str, start: usize) -> Option<((u64, u64, u64), usize)> {
    let rest = &s[start..];
    let (a, pos) = version_component(rest, 0)?;
    let (b, pos) = version_dot_component(rest, pos)?;
    let (c, pos) = version_dot_component(rest, pos)?;
    Some(((a, b, c), start + pos))
}

fn version_dot_component(s: &str, pos: usize) -> Option<(u64, usize)> {
    if s.as_bytes().get(pos).copied() != Some(b'.') {
        return None;
    }
    version_component(s, pos + 1)
}

fn version_component(s: &str, start: usize) -> Option<(u64, usize)> {
    let mut pos = start;
    while pos < s.len() && s.as_bytes()[pos].is_ascii_digit() {
        pos += 1;
    }
    if pos == start {
        return None;
    }
    Some((s[start..pos].parse().ok()?, pos))
}

/// `true` if `s` talks about the inkling parser or `inkling2_loop` architecture.
pub fn mentions_inkling(s: &str) -> bool {
    let l = s.to_ascii_lowercase();
    l.contains("inkling2_loop") || l.split(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-').any(|w| w == "inkling")
}

/// A docker/podman image reference (`registry/name:tag`).
pub fn valid_image(s: &str) -> bool {
    let s = s.trim();
    !s.is_empty()
        && s.len() < 256
        && !s.contains("..")
        && !s.contains(char::is_whitespace)
        && s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '/' | ':' | '-' | '_' | '@'))
}

/// What a `vllm` program reported.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Probe {
    pub version: Option<(u64, u64, u64)>,
    pub mentions_inkling: bool,
}

impl Probe {
    pub fn from_output(version_text: &str, help_text: &str) -> Self {
        Self { version: parse_version(version_text), mentions_inkling: mentions_inkling(version_text) || mentions_inkling(help_text) }
    }

    pub fn supports_inkling(&self, via_image: bool) -> bool {
        inkling_supported(self.version, self.mentions_inkling, via_image)
    }
}

/// Runs `program --version` and `program serve --help` (best effort) to see if Inkling works.
pub fn probe(program: &Path) -> Probe {
    let version_text = run(program, &["--version"]);
    let help_text = run(program, &["serve", "--help"]);
    Probe::from_output(&version_text, &help_text)
}

fn run(program: &Path, args: &[&str]) -> String {
    Command::new(program)
        .args(args)
        .output()
        .ok()
        .map(|o| {
            let mut s = String::from_utf8_lossy(&o.stdout).into_owned();
            s.push_str(&String::from_utf8_lossy(&o.stderr));
            s
        })
        .unwrap_or_default()
}

/// Check a resolved vLLM program before launching an Inkling model. `via_image` is set when the
/// program is our image wrapper (or the operator set `vllm = "image:..."`).
pub fn ensure_inkling(program: &Path, via_image: bool) -> Result<()> {
    if via_image {
        return Ok(());
    }
    let p = probe(program);
    if p.supports_inkling(false) || managed_with_inkling(program) {
        return Ok(());
    }
    bail!("{}", inkling_unsupported_message(p.version));
}

/// Whether `program` is the managed venv's `vllm` and its install record says the registry has
/// Inkling (`ymirctl engines install vllm` checked it).
pub fn managed_with_inkling(program: &Path) -> bool {
    program
        .parent()
        .and_then(Path::parent)
        .and_then(crate::vllm_install::read_record)
        .is_some_and(|r| r.inkling && program.file_name() == Some(std::ffi::OsStr::new(if cfg!(windows) { "vllm.exe" } else { "vllm" })))
}

/// A small `vllm` shim that runs `docker` or `podman` with `--entrypoint vllm`, so the supervisor
/// can still invoke `vllm serve MODEL --host 127.0.0.1 --port N ...`. Host networking makes the
/// loopback port visible to ymird. Written under `engines_dir` as `vllm-image`.
pub fn write_image_wrapper(engines_dir: &Path, image: &str) -> Result<PathBuf> {
    if !valid_image(image) {
        bail!("vLLM image `{image}` is not a container image name (registry/name:tag)");
    }
    std::fs::create_dir_all(engines_dir)?;
    let path = engines_dir.join("vllm-image");
    let body = wrapper_script(image);
    if path.is_file()
        && let Ok(existing) = std::fs::read_to_string(&path)
        && existing == body
    {
        return Ok(path);
    }
    let tmp = engines_dir.join(".vllm-image.tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(body.as_bytes())?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o755))?;
    }
    std::fs::rename(&tmp, &path)?;
    Ok(path)
}

fn wrapper_script(image: &str) -> String {
    // The official OpenAI image already enters at `vllm serve`; the supervisor also prefixes
    // `serve`, so we force --entrypoint vllm and pass the argv through unchanged.
    format!(
        r#"#!/bin/sh
set -e
IMAGE={image}
if command -v docker >/dev/null 2>&1; then
  RUNTIME=docker
  GPU="--gpus all"
elif command -v podman >/dev/null 2>&1; then
  RUNTIME=podman
  GPU="--gpus all"
else
  echo "ymir: docker or podman is required to run $IMAGE" >&2
  exit 1
fi
HF="${{HF_HOME:-${{XDG_CACHE_HOME:-$HOME/.cache}}/huggingface}}"
mkdir -p "$HF"
# host network: ymird health-checks 127.0.0.1:port on this machine.
exec "$RUNTIME" run --rm --ipc=host --network=host $GPU \
  -v "$HF:/root/.cache/huggingface" \
  -e HF_HOME=/root/.cache/huggingface \
  --entrypoint vllm \
  "$IMAGE" \
  "$@"
"#
    )
}

/// MLX-LM cannot serve Inkling until mlx-lm#2086 lands in a release.
pub fn mlx_not_supported_message() -> &'static str {
    "Inkling 2 Mini is not supported on a released mlx-lm yet (https://github.com/ml-explore/mlx-lm/pull/2086 is still an open draft). Pull thinkingmachines/Inkling-2-Mini for vLLM, or ggml-org/Inkling-2-Mini-GGUF for llama.cpp, instead"
}

#[cfg(test)]
mod tests {
    use super::*;
    use weirdnet_model::hw::{Accel, Gpu, Hardware};

    fn hw(name: &str, cc: Option<(u32, u32)>, unified: bool) -> Hardware {
        Hardware {
            os: "linux".into(),
            arch: "aarch64".into(),
            cpu_threads: 20,
            ram_bytes: 128 << 30,
            gpus: vec![Gpu { accel: Accel::Nvidia, name: name.into(), memory_bytes: 128 << 30, unified, compute_capability: cc, driver: None }],
        }
    }

    #[test]
    fn detects_inkling_sources_and_spark_hardware() {
        assert!(is_inkling("thinkingmachines/Inkling-2-Mini"));
        assert!(is_inkling("ggml-org/Inkling-2-Mini-GGUF"));
        assert!(is_inkling("mlx-community/Inkling-2-Mini-4bit"));
        assert!(is_inkling("thinkingmachines/TML-d24-varloop-research"));
        assert!(!is_inkling("Qwen/Qwen3-8B"));
        assert!(is_spark_class(&hw("NVIDIA GB10", Some((12, 1)), true)));
        assert!(is_spark_class(&hw("DGX Spark", None, true)));
        assert!(!is_spark_class(&hw("NVIDIA H100", Some((9, 0)), false)));
        let cpu = Hardware { os: "linux".into(), arch: "x86_64".into(), cpu_threads: 8, ram_bytes: 32 << 30, gpus: vec![] };
        assert!(!is_spark_class(&cpu));
    }

    #[test]
    fn parses_vllm_versions_and_decides_support() {
        assert_eq!(parse_version("0.34.0"), Some((0, 34, 0)));
        assert_eq!(parse_version("vLLM 0.35.0.dev+abc"), Some((0, 35, 0)));
        assert_eq!(parse_version("version: 0.35.0.post1"), Some((0, 35, 0)));
        assert_eq!(parse_version("not a version"), None);
        assert!(!inkling_supported(Some((0, 34, 0)), false, false));
        assert!(inkling_supported(Some((0, 35, 0)), false, false));
        assert!(inkling_supported(Some((0, 34, 0)), true, false));
        assert!(inkling_supported(Some((0, 34, 0)), false, true));
        assert!(mentions_inkling("Available reasoning parsers: inkling, deepseek_r1"));
        assert!(mentions_inkling("Inkling2LoopForConditionalGeneration / inkling2_loop"));
        assert!(!mentions_inkling("no such parser on this build"));
        assert!(!mentions_inkling("twinkling"));
        let msg = inkling_unsupported_message(Some((0, 34, 0)));
        assert!(msg.contains("0.34.0") && msg.contains("0.35.0") && msg.contains(INKLING_IMAGE) && msg.contains("wrap"), "{msg}");
    }

    #[test]
    fn inkling_args_leave_the_context_to_ymird() {
        let a = inkling_args();
        assert!(a.windows(2).any(|w| w == ["--reasoning-parser", "inkling"]));
        assert!(a.windows(2).any(|w| w == ["--tool-call-parser", "inkling"]));
        assert!(a.iter().any(|s| s == "--enable-auto-tool-choice"));
        assert!(a.windows(2).any(|w| w[0] == "--speculative-config" && w[1].contains("dspark")));
        assert!(!a.contains(&"--max-model-len".to_string()));
        assert_eq!(image_for("thinkingmachines/Inkling-2-Mini"), Some(INKLING_IMAGE));
        assert!(image_for("Qwen/Qwen3-8B").is_none());
    }

    #[test]
    fn writes_an_idempotent_wrapper_for_a_valid_image() {
        let d = tempfile::tempdir().unwrap();
        let p = write_image_wrapper(d.path(), INKLING_IMAGE).unwrap();
        let t = std::fs::read_to_string(&p).unwrap();
        assert!(t.contains(INKLING_IMAGE) && t.contains("--entrypoint vllm") && t.contains("--network=host"));
        let p2 = write_image_wrapper(d.path(), INKLING_IMAGE).unwrap();
        assert_eq!(p, p2);
        assert!(write_image_wrapper(d.path(), "not an image; rm -rf /").is_err());
        assert!(valid_image(INKLING_IMAGE) && !valid_image("vllm/vllm openai") && !valid_image("../escape"));
    }

    #[test]
    fn probe_from_output_does_not_need_a_binary() {
        let old = Probe::from_output("0.34.0", "usage: vllm serve");
        assert_eq!(old.version, Some((0, 34, 0)));
        assert!(!old.supports_inkling(false));
        assert!(old.supports_inkling(true));
        let new = Probe::from_output("0.35.0", "");
        assert!(new.supports_inkling(false));
    }
}
