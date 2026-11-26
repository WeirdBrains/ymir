#!/usr/bin/env python3
"""Build Ymir's release files from already-built binaries. Standard library only.

    python3 scripts/package.py [--target TRIPLE ...] [--variant default|cuda] [--out dist] [--recipe]
                               [--engine DIR | --no-engine]
    python3 scripts/package.py --pacman-from dist/ymir-VERSION-TRIPLE.tar.gz ...

For each target (default: the host's x86_64 or aarch64 GNU triple) it reads
$CARGO_TARGET_DIR/<triple>/release/{ymird,ymirctl} (or target/release for the host), adds the
llama.cpp engine for that target and variant (scripts/engine.py fetch, pinned in
packaging/engines/llama.cpp.toml; --engine uses a directory you prepared, --no-engine leaves it out)
and writes, with PKG = ymir for the default variant and ymir-cuda for the CUDA one:

  Linux:
  PKG-VERSION-TRIPLE.tar.gz, .zip        user-level install: bin/, lib/ymir/llama.cpp/, share/doc, LICENSE
  PKG_DEBVER-1_ARCH.deb                  built with `ar` (no dpkg-deb needed)
  PKG-bin-PKGVER-1-ARCH.pkg.tar.zst      makepkg on the TAUR recipe with local sources (Arch only)
  PKG-VERSION-1.ARCH.rpm                 only when rpmbuild is installed (skipped with a notice)
  macOS: PKG-VERSION-TRIPLE.tar.gz, .zip (same layout as Linux; Homebrew installs from the tar.gz)
  Windows: PKG-VERSION-TRIPLE.zip        ymird.exe, ymirctl.exe and llama.cpp\\ side by side
  FreeBSD: PKG-VERSION-TRIPLE.tar.gz     no engine (upstream publishes no BSD builds; use the
                                         misc/llama-cpp port), and PKG-VERSION-TRIPLE.pkg when
                                         `pkg` is installed (a FreeBSD host; else skipped)
  SHA256SUMS, and .minisig files when minisign and MINISIGN_KEY are set (else skipped)

ymird finds the engine relative to its own executable, so every layout works without
configuration. --recipe rewrites packaging/arch/PKG-bin/PKGBUILD for VERSION with the sha256 of the
tarballs built here (the release workflow does this with the release's own files). --pacman-from
makes only the Arch packages, from tarballs built elsewhere (the release workflow's Arch job).

The daemon has no .desktop entry, so there is only the standalone form of each package.
"""

import argparse
import gzip
import hashlib
import io
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BINS = ["ymird", "ymirctl"]
MAINTAINER = "Weird Brains LLC <m3talsmith@gmail.com>"
URL = "https://github.com/WeirdBrains/ymir"
SUMMARY = "Private cloud of LLM models on your own machines (node daemon and CLI)"
ARCH = {"x86_64": ("amd64", "x86_64"), "aarch64": ("arm64", "aarch64")}
VARIANTS = ["default", "cuda"]
ENGINE_DIR = "lib/ymir/llama.cpp"
# What the bundled llama-server links against (besides glibc), per package format.
DEB_DEPENDS = "libstdc++6, libgcc-s1, libgomp1, libssl3 | libssl3t64, libbrotli1, zlib1g, libzstd1"
ARCH_DEPENDS = "'glibc' 'gcc-libs' 'openssl' 'brotli' 'zlib' 'zstd'"
RPM_REQUIRES = "glibc, libstdc++, libgomp, openssl-libs, libbrotli, zlib, libzstd"


def version():
    text = (ROOT / "Cargo.toml").read_text()
    m = re.search(r'\[workspace\.package\][^\[]*?\nversion\s*=\s*"([^"]+)"', text, re.S)
    if not m:
        sys.exit("no [workspace.package] version in Cargo.toml")
    return m.group(1)


def debver(v):
    return v.replace("-", "~")


def pkgver(v):
    return v.replace("-", "")


def pkg_name(variant):
    return "ymir" if variant == "default" else f"ymir-{variant}"


def other_pkgs(variant):
    return [pkg_name(x) for x in VARIANTS if x != variant]


def target_dir():
    return Path(os.environ.get("CARGO_TARGET_DIR") or ROOT / "target")


def host_triple():
    machine = {"amd64": "x86_64", "arm64": "aarch64"}.get(platform.machine(), platform.machine())
    return f"{machine}-unknown-freebsd" if sys.platform.startswith("freebsd") else f"{machine}-unknown-linux-gnu"


def exe_suffix(triple):
    return ".exe" if "windows" in triple else ""


def bin_dir(triple):
    d = target_dir() / triple / "release"
    sfx = exe_suffix(triple)
    if not (d / f"ymird{sfx}").exists() and triple == host_triple():
        d = target_dir() / "release"
    for b in BINS:
        if not (d / f"{b}{sfx}").is_file():
            sys.exit(f"{d / (b + sfx)} is missing; build first: cargo build --release --locked --target {triple} -p ymird -p ymirctl")
    return d


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mtime():
    return int(os.environ.get("SOURCE_DATE_EPOCH", time.time()))


def _info(name, size=0, mode=0o644, kind=tarfile.REGTYPE, link=None):
    ti = tarfile.TarInfo(name)
    ti.size, ti.mode, ti.type = size, mode, kind
    if link is not None:
        ti.type, ti.linkname, ti.size, ti.mode = tarfile.SYMTYPE, link, 0, 0o777
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = "root"
    ti.mtime = mtime()
    return ti


def add_entry(tar, name, data, mode, link=None):
    if link is not None:
        tar.addfile(_info(name, link=link))
    else:
        tar.addfile(_info(name, len(data), mode), io.BytesIO(data))


def add_dirs(tar, names):
    for d in names:
        tar.addfile(_info(d, mode=0o755, kind=tarfile.DIRTYPE))


def parent_dirs(names):
    return sorted({str(Path(n).parents[i]) for n in names for i in range(len(Path(n).parents) - 1)})


# An entry is (path, bytes or None, mode, symlink target or None).


def engine_entries(engine, prefix):
    """The engine directory's files under `prefix`, symlinks kept."""
    if engine is None:
        return []
    out = []
    for p in sorted(engine.iterdir()):
        name = f"{prefix}/{p.name}"
        if p.is_symlink():
            out.append((name, None, 0o777, os.readlink(p)))
        elif p.is_file():
            mode = 0o755 if os.access(p, os.X_OK) else 0o644
            out.append((name, p.read_bytes(), mode, None))
    return out


def archive_files(bdir, triple, engine):
    """Entries of the user-level tar.gz/zip."""
    sfx = exe_suffix(triple)
    if sfx:  # Windows: flat, the engine in llama.cpp\ next to ymird.exe.
        return [
            (f"ymird{sfx}", (bdir / f"ymird{sfx}").read_bytes(), 0o755, None),
            (f"ymirctl{sfx}", (bdir / f"ymirctl{sfx}").read_bytes(), 0o755, None),
            ("README.md", (ROOT / "README.md").read_bytes(), 0o644, None),
            ("LICENSE", (ROOT / "LICENSE").read_bytes(), 0o644, None),
        ] + engine_entries(engine, "llama.cpp")
    return [
        ("bin/ymird", (bdir / "ymird").read_bytes(), 0o755, None),
        ("bin/ymirctl", (bdir / "ymirctl").read_bytes(), 0o755, None),
        ("share/doc/ymir/README.md", (ROOT / "README.md").read_bytes(), 0o644, None),
        ("share/icons/hicolor/scalable/apps/ymir.svg", (ROOT / "brand/png/hicolor/scalable/apps/ymir.svg").read_bytes(), 0o644, None),
        ("LICENSE", (ROOT / "LICENSE").read_bytes(), 0o644, None),
    ] + engine_entries(engine, ENGINE_DIR)


def system_files(bdir, engine):
    """Entries under / for deb/rpm packages."""
    return [
        ("usr/bin/ymird", (bdir / "ymird").read_bytes(), 0o755, None),
        ("usr/bin/ymirctl", (bdir / "ymirctl").read_bytes(), 0o755, None),
        ("usr/lib/systemd/user/ymird.service", (ROOT / "packaging/systemd/ymird.service").read_bytes(), 0o644, None),
        ("usr/share/doc/ymir/README.md", (ROOT / "README.md").read_bytes(), 0o644, None),
        ("usr/share/doc/ymir/copyright", (ROOT / "LICENSE").read_bytes(), 0o644, None),
        ("usr/share/icons/hicolor/scalable/apps/ymir.svg", (ROOT / "brand/png/hicolor/scalable/apps/ymir.svg").read_bytes(), 0o644, None),
    ] + engine_entries(engine, f"usr/{ENGINE_DIR}")


def resolve(entries, name, link):
    """A symlink's target data, for formats without symlinks (zip)."""
    by_name = {n: (d, l) for n, d, _, l in entries}
    seen = 0
    while link is not None and seen < 10:
        name = str(Path(name).parent / link)
        data, link = by_name[name]
        seen += 1
    return data


def build_tar_zip(out, v, triple, bdir, engine, variant):
    top = f"{pkg_name(variant)}-{v}-{triple}"
    entries = archive_files(bdir, triple, engine)
    made = []
    if "windows" not in triple:
        tgz = out / f"{top}.tar.gz"
        with tarfile.open(tgz, "w:gz") as tar:
            add_dirs(tar, [top] + [f"{top}/{d}" for d in parent_dirs(n for n, *_ in entries)])
            for name, data, mode, link in entries:
                add_entry(tar, f"{top}/{name}", data, mode, link)
        made.append(tgz)
    if "freebsd" in triple:
        return made
    z = out / f"{top}.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data, mode, link in entries:
            if link is not None:
                # No symlinks in zip: only the soname the libraries load needs a copy.
                if not re.search(r"\.so\.\d+$", name):
                    continue
                data, mode = resolve(entries, name, link), 0o755
            info = zipfile.ZipInfo(f"{top}/{name}", date_time=time.gmtime(mtime())[:6])
            info.external_attr = (0o100000 | mode) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)
    made.append(z)
    return made


def glibc_needed(binaries):
    """The newest GLIBC_x.y symbol version the binaries use, if objdump can tell."""
    best = None
    for b in binaries:
        r = subprocess.run(["objdump", "-T", str(b)], capture_output=True, text=True) if shutil.which("objdump") else None
        for m in re.finditer(r"GLIBC_(\d+)\.(\d+)", r.stdout if r else ""):
            ver = (int(m.group(1)), int(m.group(2)))
            best = max(best or ver, ver)
    return f"{best[0]}.{best[1]}" if best else None


def elf_files(bdir, engine):
    files = [bdir / b for b in BINS]
    if engine is not None:
        files += [p for p in engine.iterdir() if p.is_file() and not p.is_symlink() and p.read_bytes()[:4] == b"\x7fELF"]
    return files


FREEBSD_MESSAGE = """Ymir has no bundled llama.cpp on FreeBSD. Install the port and point ymird at it:
  pkg install llama-cpp
  ymirctl setup --llama-server /usr/local/bin/llama-server --model /path/to/model.gguf
Run it at boot with an rc.d script (FreeBSD has no per-user services):
  ymirctl service install --system --run-as USER
"""


def build_freebsd_pkg(out, v, triple, bdir, variant):
    """A FreeBSD package (`pkg create`), only on a host that has `pkg`."""
    if not shutil.which("pkg"):
        print(f"notice: pkg not found; {triple} .pkg skipped (build it on FreeBSD)")
        return []
    import json

    name = pkg_name(variant)
    files = [
        ("bin/ymird", (bdir / "ymird").read_bytes(), 0o755),
        ("bin/ymirctl", (bdir / "ymirctl").read_bytes(), 0o755),
        ("share/doc/ymir/README.md", (ROOT / "README.md").read_bytes(), 0o644),
        ("share/licenses/ymir/LICENSE", (ROOT / "LICENSE").read_bytes(), 0o644),
    ]
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as t:
        t = Path(t)
        stage = t / "stage"
        for rel, data, mode in files:
            f = stage / "usr/local" / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(data)
            f.chmod(mode)
        manifest = {
            "name": name,
            "version": pkgver(v),
            "origin": f"net/{name}",
            "comment": SUMMARY,
            "desc": SUMMARY + ". " + URL,
            "maintainer": "m3talsmith@gmail.com",
            "www": URL,
            "prefix": "/usr/local",
            "licenselogic": "single",
            "licenses": ["BSD3CLAUSE"],
            "categories": ["net", "misc"],
            "messages": [{"message": FREEBSD_MESSAGE}],
        }
        (t / "+MANIFEST").write_text(json.dumps(manifest, indent=2))
        (t / "plist").write_text("".join(f"{rel}\n" for rel, *_ in files))
        built = t / "out"
        built.mkdir()
        subprocess.run(["pkg", "create", "-M", str(t / "+MANIFEST"), "-p", str(t / "plist"), "-r", str(stage), "-o", str(built)], check=True)
        made = sorted(built.glob(f"{name}-*.pkg"))
        if len(made) != 1:
            sys.exit(f"pkg create made {[p.name for p in built.iterdir()]}, not one {name}-*.pkg")
        dest = out / f"{name}-{v}-{triple}.pkg"
        shutil.move(made[0], dest)
    return [dest]


def build_deb(out, v, arch, bdir, engine, variant):
    deb_arch = ARCH[arch][0]
    name = pkg_name(variant)
    files = system_files(bdir, engine)
    data_buf = io.BytesIO()
    with tarfile.open(fileobj=data_buf, mode="w:gz") as tar:
        add_dirs(tar, ["./"] + [f"./{d}/" for d in parent_dirs(n for n, *_ in files)])
        for n, data, mode, link in files:
            add_entry(tar, f"./{n}", data, mode, link)
    glibc = glibc_needed(elf_files(bdir, engine))
    depends = f"libc6 (>= {glibc})" if glibc else "libc6"
    if engine is not None:
        depends += f", {DEB_DEPENDS}"
    others = ", ".join(other_pkgs(variant))
    installed_kb = sum(len(d) for _, d, _, l in files if l is None) // 1024 + 1
    engine_line = (
        f" Includes llama.cpp's llama-server ({'CUDA 13' if variant == 'cuda' else 'CPU and Vulkan'}) in /usr/{ENGINE_DIR}.\n" if engine else ""
    )
    control = (
        f"Package: {name}\nVersion: {debver(v)}-1\nArchitecture: {deb_arch}\nMaintainer: {MAINTAINER}\n"
        f"Installed-Size: {installed_kb}\nDepends: {depends}\n"
        + ("Recommends: libvulkan1\n" if engine is not None and variant == "default" else "")
        + (f"Conflicts: {others}\nReplaces: {others}\nProvides: ymir\n" if variant != "default" else f"Conflicts: {others}\n")
        + f"Section: net\nPriority: optional\nHomepage: {URL}\n"
        f"Description: {SUMMARY}\n ymird runs the model, inference and app roles; ymirctl sets up and talks to it.\n"
        f"{engine_line}"
        " Run it per user: systemctl --user enable --now ymird\n"
    )
    md5 = "".join(f"{hashlib.md5(d).hexdigest()}  {n}\n" for n, d, _, l in files if l is None)
    ctl_buf = io.BytesIO()
    with tarfile.open(fileobj=ctl_buf, mode="w:gz") as tar:
        add_dirs(tar, ["./"])
        add_entry(tar, "./control", control.encode(), 0o644)
        add_entry(tar, "./md5sums", md5.encode(), 0o644)
    deb = out / f"{name}_{debver(v)}-1_{deb_arch}.deb"
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        (t / "debian-binary").write_bytes(b"2.0\n")
        (t / "control.tar.gz").write_bytes(ctl_buf.getvalue())
        (t / "data.tar.gz").write_bytes(data_buf.getvalue())
        deb.unlink(missing_ok=True)
        # Order matters: debian-binary first.
        subprocess.run(["ar", "rcD" if _ar_has_D() else "rc", str(deb), "debian-binary", "control.tar.gz", "data.tar.gz"], cwd=t, check=True)
    return [deb]


def _ar_has_D():
    r = subprocess.run(["ar", "--help"], capture_output=True, text=True)
    return "[D]" in r.stdout or " D " in r.stdout


def pkgbuild(v, sums, variant="default", local=False):
    """The PKG-bin recipe. `sums` maps arch -> sha256 of the release tarball."""
    pkg = pkg_name(variant)
    if local:
        src = {a: f'("{pkg}-{v}-{a}-unknown-linux-gnu.tar.gz")' for a in ARCH}
    else:
        src = {a: f'("{pkg}-${{_ver}}-{a}-unknown-linux-gnu.tar.gz::${{url}}/releases/download/v${{_ver}}/{pkg}-${{_ver}}-{a}-unknown-linux-gnu.tar.gz")' for a in ARCH}
    engine_desc = "llama.cpp with CUDA 13 included" if variant == "cuda" else "llama.cpp included"
    names = ["ymir", "ymir-git"] + [n for p in other_pkgs(variant) for n in (p, f"{p}-bin")]
    conflicts = " ".join(f"'{n}'" for n in dict.fromkeys(names))
    optdepends = (
        "optdepends=('vulkan-icd-loader: GPU offload through Vulkan'\n            'python-vllm: the vLLM engine (NVIDIA)')"
        if variant == "default"
        else "optdepends=('python-vllm: the vLLM engine (NVIDIA)')"
    )
    return f"""# Maintainer: {MAINTAINER}
# Prebuilt binaries from the GitHub release. scripts/package.py --recipe (and the release workflow)
# fill in pkgver and the sha256sums; scripts/weirdnet_release.py taur publishes this recipe to TAUR.
pkgname={pkg}-bin
_ver={v}
pkgver={pkgver(v)}
pkgrel=1
pkgdesc="{SUMMARY}, {engine_desc} (prebuilt binary)"
arch=('x86_64' 'aarch64')
url="{URL}"
license=('BSD-3-Clause' 'MIT')
depends=({ARCH_DEPENDS})
{optdepends}
provides=('ymir')
conflicts=({conflicts})
options=('!strip' '!debug')
source_x86_64={src['x86_64']}
source_aarch64={src['aarch64']}
sha256sums_x86_64=('{sums.get('x86_64', 'SKIP')}')
sha256sums_aarch64=('{sums.get('aarch64', 'SKIP')}')

package() {{
  cd "{pkg}-${{_ver}}-${{CARCH}}-unknown-linux-gnu"
  install -Dm755 bin/ymird "${{pkgdir}}/usr/bin/ymird"
  install -Dm755 bin/ymirctl "${{pkgdir}}/usr/bin/ymirctl"
  # The bundled llama.cpp engine; ymird finds it at ../lib/ymir/llama.cpp from /usr/bin.
  install -d "${{pkgdir}}/usr/lib/ymir"
  cp -a {ENGINE_DIR} "${{pkgdir}}/usr/lib/ymir/"
  install -Dm644 share/doc/ymir/README.md "${{pkgdir}}/usr/share/doc/ymir/README.md"
  install -Dm644 share/icons/hicolor/scalable/apps/ymir.svg "${{pkgdir}}/usr/share/icons/hicolor/scalable/apps/ymir.svg"
  install -Dm644 LICENSE "${{pkgdir}}/usr/share/licenses/${{pkgname}}/LICENSE"
  install -Dm644 /dev/stdin "${{pkgdir}}/usr/lib/systemd/user/ymird.service" <<'UNIT'
{(ROOT / "packaging/systemd/ymird.service").read_text().rstrip()}
UNIT
}}
"""


def build_pacman(out, v, arch, tarball, variant):
    if not shutil.which("makepkg"):
        print("notice: makepkg not found; skipping pkg.tar.zst (the release workflow's Arch job builds it)")
        return []
    pkg = pkg_name(variant)
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as t:
        t = Path(t)
        shutil.copy(tarball, t / tarball.name)
        (t / "PKGBUILD").write_text(pkgbuild(v, {arch: sha256(tarball)}, variant, local=True))
        # The system makepkg.conf, with CARCH set so the aarch64 package can be made on x86_64.
        conf = t / "makepkg.conf"
        conf.write_text(f'source /etc/makepkg.conf\nCARCH="{arch}"\nPKGEXT=".pkg.tar.zst"\nPKGDEST="{out}"\nSRCDEST="{t}"\nBUILDDIR="{t / "build"}"\n')
        r = subprocess.run(["makepkg", "--config", str(conf), "--nodeps", "--noconfirm", "--force"], cwd=t, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
            sys.exit(f"makepkg failed for {arch}")
    return sorted(out.glob(f"{pkg}-bin-{pkgver(v)}-1-{arch}.pkg.tar.zst"))


def build_rpm(out, v, arch, bdir, engine, variant):
    name = pkg_name(variant)
    if not shutil.which("rpmbuild"):
        print(f"notice: rpmbuild not found; skipping {name}-{v}-1.{arch}.rpm (the release workflow builds it)")
        return []
    files = system_files(bdir, engine)
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        root = t / "root"
        for n, data, mode, link in files:
            p = root / n
            p.parent.mkdir(parents=True, exist_ok=True)
            if link is not None:
                os.symlink(link, p)
            else:
                p.write_bytes(data)
                p.chmod(mode)
        listed = "\n".join(f"/{n}" for n, *_ in files)
        if engine is not None:
            listed += f"\n%dir /usr/lib/ymir\n%dir /usr/{ENGINE_DIR}"
        others = " ".join(other_pkgs(variant))
        spec = t / "ymir.spec"
        spec.write_text(
            f"Name: {name}\nVersion: {v.replace('-', '~')}\nRelease: 1\nSummary: {SUMMARY}\nLicense: BSD-3-Clause AND MIT\nURL: {URL}\nBuildArch: {arch}\n"
            # The bundled libraries are private to the engine: don't provide or require them system-wide.
            f"AutoReqProv: no\nRequires: {RPM_REQUIRES if engine is not None else 'glibc'}\nConflicts: {others}\n"
            + ("Provides: ymir\n" if variant != "default" else "")
            + "%global __os_install_post %{nil}\n%global debug_package %{nil}\n"
            f"%description\n{SUMMARY}.\n%install\ncp -a {root}/. %{{buildroot}}/\n%files\n{listed}\n"
        )
        subprocess.run(["rpmbuild", "-bb", "--target", arch, "--define", f"_topdir {t}", "--define", f"_rpmdir {out}", "--define", "_build_name_fmt %%{NAME}-%%{VERSION}-%%{RELEASE}.%%{ARCH}.rpm", str(spec)], check=True)
    return sorted(out.glob(f"{name}-{v.replace('-', '~')}-1.{arch}.rpm"))


def sign(files):
    key = os.environ.get("MINISIGN_KEY")
    if not (shutil.which("minisign") and key and Path(key).is_file()):
        print("notice: minisign or MINISIGN_KEY not set; release files are not signed")
        return []
    sigs = []
    for f in files:
        subprocess.run(["minisign", "-S", "-s", key, "-m", str(f)], check=True, stdin=subprocess.DEVNULL)
        sigs.append(Path(f"{f}.minisig"))
    return sigs


def fetch_engine(triple, variant, into):
    subprocess.run([sys.executable, str(ROOT / "scripts/engine.py"), "fetch", "--target", triple, "--variant", variant, "--out", str(into)], check=True)
    return into


def write_sums(out):
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != "SHA256SUMS" and not p.name.endswith((".minisig", ".part")))
    (out / "SHA256SUMS").write_text("".join(f"{sha256(f)}  {f.name}\n" for f in files))


def pacman_from(out, v, tarballs):
    made = []
    for tb in tarballs:
        m = re.match(r"^(ymir(?:-([a-z0-9]+))?)-" + re.escape(v) + r"-(x86_64|aarch64)-unknown-linux-gnu\.tar\.gz$", tb.name)
        if not m:
            sys.exit(f"{tb.name}: not a Linux release tarball for {v}")
        made += build_pacman(out, v, m.group(3), tb, m.group(2) or "default")
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", action="append", help="target triple (repeatable)")
    ap.add_argument("--variant", default="default", choices=VARIANTS, help="engine variant (default: CPU+Vulkan, or Metal on macOS)")
    ap.add_argument("--engine", help="a prepared engine directory (scripts/engine.py fetch/build) instead of fetching one")
    ap.add_argument("--no-engine", action="store_true", help="leave the llama.cpp engine out (it must then be installed separately)")
    ap.add_argument("--out", default="dist")
    ap.add_argument("--recipe", action="store_true", help="rewrite packaging/arch/PKG-bin/PKGBUILD with these tarballs' sums")
    ap.add_argument("--pacman-from", nargs="+", type=Path, metavar="TARBALL", help="only build Arch packages from these tarballs")
    a = ap.parse_args()
    v = version()
    out = (ROOT / a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if a.pacman_from:
        made = pacman_from(out, v, a.pacman_from)
        write_sums(out)
        for f in made:
            print(f"{f.stat().st_size:>10}  {f.name}")
        return
    if a.engine and len(a.target or [host_triple()]) > 1:
        sys.exit("--engine works with one --target")
    made, sums = [], {}
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as scratch:
        for triple in a.target or [host_triple()]:
            arch = triple.split("-")[0]
            linux = triple.endswith("-linux-gnu")
            freebsd = triple.endswith("-unknown-freebsd")
            if arch not in ARCH or not (linux or freebsd or triple.endswith("-apple-darwin") or triple.endswith("-pc-windows-msvc")):
                sys.exit(f"{triple}: only x86_64/aarch64 Linux GNU, macOS, Windows MSVC and FreeBSD targets are packaged here")
            if freebsd and a.variant != "default":
                sys.exit(f"{triple}: only the default variant (no CUDA on FreeBSD)")
            bdir = bin_dir(triple)
            engine = None
            if a.engine:
                engine = Path(a.engine).resolve()
            elif freebsd:
                print(f"notice: {triple} has no bundled llama.cpp (no upstream FreeBSD builds); the package points at the misc/llama-cpp port")
            elif not a.no_engine:
                engine = fetch_engine(triple, a.variant, Path(scratch) / f"{triple}-{a.variant}")
            if engine is not None and not (engine / f"llama-server{exe_suffix(triple)}").is_file():
                sys.exit(f"{engine} has no llama-server")
            tz = build_tar_zip(out, v, triple, bdir, engine, a.variant)
            made += tz
            if freebsd:
                made += build_freebsd_pkg(out, v, triple, bdir, a.variant)
            if not linux:
                continue
            sums[arch] = sha256(tz[0])
            made += build_deb(out, v, arch, bdir, engine, a.variant)
            made += build_pacman(out, v, arch, tz[0], a.variant)
            made += build_rpm(out, v, arch, bdir, engine, a.variant)
    made += sign(made)
    write_sums(out)
    if a.recipe:
        path = ROOT / f"packaging/arch/{pkg_name(a.variant)}-bin/PKGBUILD"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(pkgbuild(v, sums, a.variant))
        print(f"wrote {path.relative_to(ROOT)}")
    for f in made:
        print(f"{f.stat().st_size:>10}  {f.relative_to(ROOT) if f.is_relative_to(ROOT) else f}")


if __name__ == "__main__":
    main()
