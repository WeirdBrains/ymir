#!/usr/bin/env python3
"""The llama.cpp engine that ships inside Ymir's release files. Standard library only.

Ymir runs GGUF models with llama.cpp's `llama-server`, as a separate process under ymird's
supervisor (a crashed model never takes ymird down). Every release file carries a pinned build of it
next to ymird, so a server install needs nothing else. The pin lives in
packaging/engines/llama.cpp.toml: one upstream llama.cpp build, and for each target and variant the
upstream release archives (with their SHA-256) that make up the engine.

    python3 scripts/engine.py list [--json]
    python3 scripts/engine.py fetch --target TRIPLE [--variant default|cuda] --out DIR [--check]
    python3 scripts/engine.py build --out DIR [--backend vulkan] [--backend cuda] [--jobs N] [--trainer] [--check]
    python3 scripts/engine.py pin BUILD          # rewrite the pin for llama.cpp build BUILD (bNNNN)

`fetch` downloads (cached in $YMIR_ENGINE_CACHE, default ~/.cache/ymir-engines), checks every
SHA-256, and writes DIR with only what the engine needs: llama-server, rpc-server, their libraries
and GPU backends, LICENSE files and engine.toml. Release files put DIR at lib/ymir/llama.cpp/
(beside bin/), or at llama.cpp\\ next to ymird.exe on Windows.

`build` compiles the same set from source at the pinned build, for targets upstream doesn't publish
binaries for (FreeBSD, OpenBSD and other BSDs, or distribution builds that must build from source).
`--trainer` adds `llama-finetune` at the same build, for scripts/trainer.py (fine-tuning, training
plan P3a); upstream's release archives don't carry it. Only that build applies the patches in
packaging/engines/patches/ (TRAINER_PATCHES), and they touch only llama-finetune's source, so
llama-server and rpc-server stay upstream's.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIN = ROOT / "packaging/engines/llama.cpp.toml"
UPSTREAM = "ggml-org/llama.cpp"
PATCHES = ROOT / "packaging/engines/patches"
# Applied by `build --trainer`, in order. Each touches only examples/training/finetune.cpp, and each
# id is what the patched `llama-finetune --ymir-patches` prints (scripts/trainer.py REQUIRED_PATCHES).
#   parse-special: tokenize the training text with parse_special = true, so the chat template's
#   special tokens train as the tokens the model emits. Drop it once upstream releases the fix
#   (tracked: draft ggml-org/llama.cpp#31258) and the pin moves past it.
TRAINER_PATCHES = {"parse-special": "llama-finetune-parse-special.patch"}

# Which upstream archives make up each (target, variant). `{b}` is the build. The CUDA builds use
# CUDA 13, whose default architectures cover Turing (sm_75) through Blackwell, GB10 (sm_121) included.
LAYOUT = {
    ("x86_64-unknown-linux-gnu", "default"): (["cpu", "vulkan"], ["llama-{b}-bin-ubuntu-vulkan-x64.tar.gz"]),
    ("x86_64-unknown-linux-gnu", "cuda"): (["cpu", "cuda"], ["llama-{b}-bin-ubuntu-cuda-13.5-x64.tar.gz", "cudart-llama-{b}-bin-ubuntu-cuda-13.5-x64.tar.gz"]),
    ("aarch64-unknown-linux-gnu", "default"): (["cpu", "vulkan"], ["llama-{b}-bin-ubuntu-vulkan-arm64.tar.gz"]),
    ("aarch64-unknown-linux-gnu", "cuda"): (["cpu", "cuda"], ["llama-{b}-bin-ubuntu-cuda-13.5-arm64.tar.gz", "cudart-llama-{b}-bin-ubuntu-cuda-13.5-arm64.tar.gz"]),
    ("aarch64-apple-darwin", "default"): (["cpu", "metal"], ["llama-{b}-bin-macos-arm64.tar.gz"]),
    ("x86_64-apple-darwin", "default"): (["cpu"], ["llama-{b}-bin-macos-x64.tar.gz"]),
    ("x86_64-pc-windows-msvc", "default"): (["cpu", "vulkan"], ["llama-{b}-bin-win-vulkan-x64.zip"]),
    ("x86_64-pc-windows-msvc", "cuda"): (["cpu", "cuda"], ["llama-{b}-bin-win-cuda-13.5-x64.zip", "cudart-llama-bin-win-cuda-13.5-x64.zip"]),
    ("aarch64-pc-windows-msvc", "default"): (["cpu", "vulkan"], ["llama-{b}-bin-win-vulkan-arm64.zip"]),
}

PROGRAMS = {"llama-server": "llama-server", "ggml-rpc-server": "rpc-server", "rpc-server": "rpc-server"}
# With `build --trainer`.
TRAINER_PROGRAMS = {"llama-finetune": "llama-finetune"}
LIB_PREFIXES = ("ggml", "llama", "mtmd", "cudart", "cublas")
CUDA_NOTICE = """The CUDA runtime libraries in this directory (cudart, cuBLAS, cuBLASLt) are NVIDIA CUDA
Toolkit redistributables, shipped unmodified under the NVIDIA CUDA Toolkit End User License
Agreement, Attachment A (https://docs.nvidia.com/cuda/eula/). They come from llama.cpp's
official release archives.
"""


def load_pin():
    if not PIN.exists():
        sys.exit(f"{PIN} is missing; run: python3 scripts/engine.py pin BUILD")
    with open(PIN, "rb") as f:
        return tomllib.load(f)


def find_variant(pin, target, variant):
    for v in pin["variant"]:
        if v["target"] == target and v["variant"] == variant:
            return v
    have = ", ".join(f"{v['target']}/{v['variant']}" for v in pin["variant"])
    sys.exit(f"no {variant} engine for {target} in {PIN.name} (have: {have}); use `build` for other targets")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_dir():
    d = Path(os.environ.get("YMIR_ENGINE_CACHE") or Path.home() / ".cache/ymir-engines")
    d.mkdir(parents=True, exist_ok=True)
    return d


def download(url, dest, want):
    if dest.exists() and sha256(dest) == want:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}", file=sys.stderr)
    req = urllib.request.Request(url, headers={"User-Agent": "ymir-engine-fetch"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    got = sha256(tmp)
    if got != want:
        tmp.unlink()
        sys.exit(f"{dest.name}: sha256 {got} does not match the pinned {want}")
    tmp.replace(dest)
    return dest


def extract(archive, into):
    """Unpacks `archive` into `into`, dropping a single top-level directory."""
    stage = Path(tempfile.mkdtemp(dir=into.parent))
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for m in z.infolist():
                if m.filename.startswith("/") or ".." in Path(m.filename).parts:
                    sys.exit(f"{archive.name}: unsafe path {m.filename}")
            z.extractall(stage)
    else:
        with tarfile.open(archive) as t:
            t.extractall(stage, filter="data")
    entries = list(stage.iterdir())
    top = entries[0] if len(entries) == 1 and entries[0].is_dir() and not entries[0].is_symlink() else stage
    into.mkdir(parents=True, exist_ok=True)
    for p in top.iterdir():
        dest = into / p.name
        if dest.exists() or dest.is_symlink():
            if dest.is_dir() and not dest.is_symlink():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(p), dest)
    shutil.rmtree(stage, ignore_errors=True)


def keep(name, trainer=False):
    """The engine's files among an upstream build's: the two programs (and llama-finetune with
    `trainer`), their libraries, licenses."""
    base = name[:-4] if name.endswith(".exe") else name
    programs = {**PROGRAMS, **TRAINER_PROGRAMS} if trainer else PROGRAMS
    if base in programs:
        return programs[base] + (".exe" if name.endswith(".exe") else "")
    if name.upper().startswith("LICENSE"):
        return name
    m = re.match(r"^(?:lib)?([A-Za-z0-9_.+-]+?)(?:\.so(?:\.[0-9.]+)?|\.dylib|\.dll)$", name)
    if not m:
        return None
    stem = m.group(1)
    if not stem.startswith(LIB_PREFIXES):
        return None
    # Each llama.cpp tool has an -impl library; only the server's is needed.
    if stem.endswith("-impl") and not stem.startswith("llama-server-impl") and not (trainer and stem.startswith("llama-finetune-impl")):
        return None
    return name


def select(src, out, trainer=False):
    """Copies what `keep` wants from `src` into `out`, keeping symlinks as symlinks."""
    out.mkdir(parents=True, exist_ok=True)
    kept = []
    for p in sorted(src.iterdir()):
        if p.is_dir() and not p.is_symlink():
            continue
        name = keep(p.name, trainer)
        if not name:
            continue
        dest = out / name
        if p.is_symlink():
            os.symlink(os.readlink(p), dest)
        else:
            shutil.copy2(p, dest)
            if name.removesuffix(".exe") in ("llama-server", "rpc-server", "llama-finetune") or ".so" in name or name.endswith(".dylib"):
                dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        kept.append(name)
    # Drop symlinks whose target was not kept.
    for p in out.iterdir():
        if p.is_symlink() and not p.exists():
            p.unlink()
            kept.remove(p.name)
    if not any(k.startswith("llama-server") for k in kept):
        sys.exit(f"no llama-server in {src}")
    if trainer and not any(k.startswith("llama-finetune") for k in kept):
        sys.exit(f"no llama-finetune in {src}")
    return kept


def trainer_patches():
    return [PATCHES / name for name in TRAINER_PATCHES.values()]


def apply_patch(src, patch):
    """Applies `patch` to the llama.cpp tree `src` with git; a tree that already has it (a reused
    --work) is left as it is. Exits if it does not apply cleanly."""
    git = ["git", "-C", str(src), "apply", "--whitespace=nowarn"]
    if subprocess.run([*git, "--reverse", "--check", str(patch)], capture_output=True).returncode == 0:
        print(f"{patch.name}: already applied", file=sys.stderr)
        return
    r = subprocess.run([*git, "--check", str(patch)], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"{patch.name} does not apply to llama.cpp {load_pin()['build']}: {r.stderr.strip()}")
    subprocess.run([*git, str(patch)], check=True)
    print(f"{patch.name}: applied", file=sys.stderr)


def build_plan(a, src, build):
    """The steps of `build`, in order: ("patch", PATH) for each trainer patch (with --trainer only),
    then ("run", ARGV) for cmake's configure and build."""
    rpath = "@loader_path" if sys.platform == "darwin" else "$ORIGIN"
    defs = [
        "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=" + ("ON" if a.trainer else "OFF"),
        "-DLLAMA_BUILD_SERVER=ON", "-DGGML_NATIVE=OFF", "-DGGML_BACKEND_DL=ON", "-DGGML_RPC=ON",
        f"-DCMAKE_INSTALL_RPATH={rpath}", "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
    ]
    if platform.machine() in ("x86_64", "AMD64", "amd64", "aarch64", "arm64") and not a.no_cpu_variants:
        defs.append("-DGGML_CPU_ALL_VARIANTS=ON")
    for b in a.backend or []:
        defs.append({"vulkan": "-DGGML_VULKAN=ON", "cuda": "-DGGML_CUDA=ON", "metal": "-DGGML_METAL=ON", "rocm": "-DGGML_HIP=ON"}[b])
    steps = [("patch", p) for p in trainer_patches()] if a.trainer else []
    steps.append(("run", ["cmake", "-S", str(src), "-B", str(build), "-G", "Ninja" if shutil.which("ninja") else "Unix Makefiles", *defs]))
    steps.append(("run", ["cmake", "--build", str(build), "--config", "Release", "-j", str(a.jobs), "--target", "llama-server", "ggml-rpc-server", *(["llama-finetune"] if a.trainer else [])]))
    return steps


def write_info(out, build, variant, backends, source, patches=()):
    lines = [
        "# The llama.cpp build shipped with Ymir (scripts/engine.py). ymirctl setup reads this.",
        'engine = "llama.cpp"',
        f'build = "{build}"',
        f'variant = "{variant}"',
        "backends = [" + ", ".join(f'"{b}"' for b in backends) + "]",
        f'source = "{source}"',
    ]
    if patches:
        lines.append("trainer_patches = [" + ", ".join(f'"{p}"' for p in patches) + "]")
    (out / "engine.toml").write_text("\n".join(lines) + "\n")


def check(out):
    """Runs the engine once (`--version`), as a release would, with a clean environment."""
    exe = out / ("llama-server.exe" if os.name == "nt" else "llama-server")
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    if os.name == "nt":
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
    r = subprocess.run([str(exe), "--version"], capture_output=True, text=True, env=env, timeout=60)
    text = (r.stdout + r.stderr).strip()
    print(text, file=sys.stderr)
    if r.returncode != 0 or "version" not in text:
        sys.exit(f"{exe} --version failed ({r.returncode})")


def cmd_fetch(a):
    pin = load_pin()
    v = find_variant(pin, a.target, a.variant)
    out = Path(a.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    work = Path(tempfile.mkdtemp(prefix="ymir-engine-"))
    try:
        unpacked = work / "unpacked"
        for asset in v["assets"]:
            f = download(f"{pin['release']}/{asset['name']}", cache_dir() / asset["name"], asset["sha256"])
            extract(f, unpacked)
        kept = select(unpacked, out)
        lic = out / "LICENSE"
        if lic.exists():
            lic.rename(out / "LICENSE-llama.cpp")
        if "cuda" in v["backends"]:
            (out / "NOTICE-CUDA").write_text(CUDA_NOTICE)
        write_info(out, pin["build"], a.variant, v["backends"], f"https://github.com/{UPSTREAM}/releases/tag/{pin['build']}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"{out}: llama.cpp {pin['build']} {a.target}/{a.variant} ({len(kept)} files, {dir_size(out) // (1 << 20)} MiB)")
    if a.check:
        check(out)


def cmd_build(a):
    pin = load_pin()
    out = Path(a.out).resolve()
    work = Path(a.work or tempfile.mkdtemp(prefix="ymir-engine-src-"))
    src = work / "llama.cpp"
    if not src.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", pin["build"], f"https://github.com/{UPSTREAM}", str(src)], check=True)
    backends = ["cpu"] + list(a.backend or [])
    build = work / "build"
    for kind, step in build_plan(a, src, build):
        if kind == "patch":
            apply_patch(src, step)
        else:
            subprocess.run(step, check=True)
    if out.exists():
        shutil.rmtree(out)
    select(build / "bin", out, a.trainer)
    shutil.copy(src / "LICENSE", out / "LICENSE-llama.cpp")
    write_info(out, pin["build"], a.variant, backends, f"https://github.com/{UPSTREAM}/tree/{pin['build']} (built from source)", list(TRAINER_PATCHES) if a.trainer else ())
    print(f"{out}: llama.cpp {pin['build']} built from source ({', '.join(backends)}, {dir_size(out) // (1 << 20)} MiB)")
    if a.check:
        check(out)


def dir_size(d):
    return sum(p.stat().st_size for p in Path(d).iterdir() if p.is_file() and not p.is_symlink())


def cmd_list(a):
    pin = load_pin()
    rows = [{"target": v["target"], "variant": v["variant"], "backends": v["backends"]} for v in pin["variant"]]
    if a.json:
        print(json.dumps(rows))
    else:
        print(f"llama.cpp {pin['build']}")
        for r in rows:
            print(f"  {r['target']:<28} {r['variant']:<8} {', '.join(r['backends'])}")


def cmd_pin(a):
    url = f"https://api.github.com/repos/{UPSTREAM}/releases/tags/{a.build}"
    req = urllib.request.Request(url, headers={"User-Agent": "ymir-engine-pin", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        rel = json.load(r)
    assets = {x["name"]: x for x in rel["assets"]}
    out = [
        "# The llama.cpp build that ships inside Ymir's release files, as upstream's own release archives.",
        "# Generated by `python3 scripts/engine.py pin BUILD`; `scripts/engine.py fetch` checks every sha256.",
        f'build = "{a.build}"',
        f'release = "https://github.com/{UPSTREAM}/releases/download/{a.build}"',
    ]
    for (target, variant), (backends, names) in LAYOUT.items():
        out += ["", "[[variant]]", f'target = "{target}"', f'variant = "{variant}"', "backends = [" + ", ".join(f'"{b}"' for b in backends) + "]", "assets = ["]
        for n in names:
            name = n.format(b=a.build)
            x = assets.get(name)
            if not x:
                sys.exit(f"{a.build} has no asset {name}")
            digest = (x.get("digest") or "").removeprefix("sha256:")
            if not digest:
                sys.exit(f"{name}: the release API gave no sha256 digest; download it and hash it by hand")
            out.append(f'  {{ name = "{name}", sha256 = "{digest}", size = {x["size"]} }},')
        out.append("]")
    PIN.write_text("\n".join(out) + "\n")
    print(f"wrote {PIN.relative_to(ROOT)} for {a.build}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("fetch")
    p.add_argument("--target", required=True)
    p.add_argument("--variant", default="default")
    p.add_argument("--out", required=True)
    p.add_argument("--check", action="store_true", help="run llama-server --version afterwards")
    p = sub.add_parser("build")
    p.add_argument("--out", required=True)
    p.add_argument("--backend", action="append", choices=["vulkan", "cuda", "metal", "rocm"])
    p.add_argument("--variant", default="default")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    p.add_argument("--work", help="source and build directory (default: a temp dir)")
    p.add_argument("--trainer", action="store_true", help="also build llama-finetune, for scripts/trainer.py")
    p.add_argument("--no-cpu-variants", action="store_true", help="one CPU backend instead of one per CPU generation")
    p.add_argument("--check", action="store_true")
    p = sub.add_parser("pin")
    p.add_argument("build")
    a = ap.parse_args()
    {"list": cmd_list, "fetch": cmd_fetch, "build": cmd_build, "pin": cmd_pin}[a.cmd](a)


if __name__ == "__main__":
    main()
