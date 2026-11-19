#!/usr/bin/env python3
"""Build Ymir's Linux release files from already-built binaries. Standard library only.

    python3 scripts/package.py [--target TRIPLE ...] [--out dist] [--recipe]

For each target (default: the host's x86_64 or aarch64 GNU triple) it reads
$CARGO_TARGET_DIR/<triple>/release/{ymird,ymirctl} (or target/release for the host) and writes:

  ymir-VERSION-TRIPLE.tar.gz, .zip       user-level install: bin/, share/doc, LICENSE
  ymir_DEBVER-1_ARCH.deb                 built with `ar` (no dpkg-deb needed)
  ymir-bin-PKGVER-1-ARCH.pkg.tar.zst     makepkg on the TAUR recipe with local sources
  ymir-VERSION-1.ARCH.rpm                only when rpmbuild is installed (skipped with a notice)
  SHA256SUMS, and .minisig files when minisign and MINISIGN_KEY are set (else skipped)

--recipe rewrites packaging/arch/ymir-bin/PKGBUILD for VERSION with the sha256 of the tarballs
built here (the release workflow does this with the release's own files).

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
MAINTAINER = "m3talsmith <m3talsmith@gmail.com>"
URL = "https://github.com/WeirdBrains/ymir"
SUMMARY = "Private cloud of LLM models on your own machines (node daemon and CLI)"
ARCH = {"x86_64": ("amd64", "x86_64"), "aarch64": ("arm64", "aarch64")}


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


def target_dir():
    return Path(os.environ.get("CARGO_TARGET_DIR") or ROOT / "target")


def host_triple():
    return f"{platform.machine()}-unknown-linux-gnu"


def bin_dir(triple):
    d = target_dir() / triple / "release"
    if not (d / "ymird").exists() and triple == host_triple():
        d = target_dir() / "release"
    for b in BINS:
        if not (d / b).is_file():
            sys.exit(f"{d / b} is missing; build first: cargo build --release --locked --target {triple} -p ymird -p ymirctl")
    return d


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _info(name, size=0, mode=0o644, kind=tarfile.REGTYPE, mtime=None):
    ti = tarfile.TarInfo(name)
    ti.size, ti.mode, ti.type = size, mode, kind
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = "root"
    ti.mtime = mtime if mtime is not None else int(os.environ.get("SOURCE_DATE_EPOCH", time.time()))
    return ti


def add_file(tar, name, data, mode=0o644):
    tar.addfile(_info(name, len(data), mode), io.BytesIO(data))


def add_dirs(tar, names):
    for d in names:
        tar.addfile(_info(d, mode=0o755, kind=tarfile.DIRTYPE))


def archive_files(bdir):
    """(path in the archive, bytes, mode) for the user-level tar.gz/zip."""
    return [
        ("bin/ymird", (bdir / "ymird").read_bytes(), 0o755),
        ("bin/ymirctl", (bdir / "ymirctl").read_bytes(), 0o755),
        ("share/doc/ymir/README.md", (ROOT / "README.md").read_bytes(), 0o644),
        ("LICENSE", (ROOT / "LICENSE").read_bytes(), 0o644),
    ]


def system_files(bdir):
    """(path under /, bytes, mode) for deb/rpm/pacman packages."""
    return [
        ("usr/bin/ymird", (bdir / "ymird").read_bytes(), 0o755),
        ("usr/bin/ymirctl", (bdir / "ymirctl").read_bytes(), 0o755),
        ("usr/lib/systemd/user/ymird.service", (ROOT / "packaging/systemd/ymird.service").read_bytes(), 0o644),
        ("usr/share/doc/ymir/README.md", (ROOT / "README.md").read_bytes(), 0o644),
        ("usr/share/doc/ymir/copyright", (ROOT / "LICENSE").read_bytes(), 0o644),
    ]


def build_tar_zip(out, v, triple, bdir):
    top = f"ymir-{v}-{triple}"
    tgz = out / f"{top}.tar.gz"
    with tarfile.open(tgz, "w:gz") as tar:
        add_dirs(tar, [top, f"{top}/bin", f"{top}/share", f"{top}/share/doc", f"{top}/share/doc/ymir"])
        for name, data, mode in archive_files(bdir):
            add_file(tar, f"{top}/{name}", data, mode)
    z = out / f"{top}.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data, mode in archive_files(bdir):
            info = zipfile.ZipInfo(f"{top}/{name}", date_time=time.gmtime(int(os.environ.get("SOURCE_DATE_EPOCH", time.time())))[:6])
            info.external_attr = (0o100000 | mode) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data)
    return [tgz, z]


def glibc_needed(binaries):
    """The newest GLIBC_x.y symbol version the binaries use, if objdump can tell."""
    best = None
    for b in binaries:
        r = subprocess.run(["objdump", "-T", str(b)], capture_output=True, text=True) if shutil.which("objdump") else None
        for m in re.finditer(r"GLIBC_(\d+)\.(\d+)", r.stdout if r else ""):
            ver = (int(m.group(1)), int(m.group(2)))
            best = max(best or ver, ver)
    return f"{best[0]}.{best[1]}" if best else None


def build_deb(out, v, arch, bdir):
    deb_arch = ARCH[arch][0]
    files = system_files(bdir)
    data_buf = io.BytesIO()
    with tarfile.open(fileobj=data_buf, mode="w:gz") as tar:
        dirs = sorted({str(Path(n).parents[i]) for n, _, _ in files for i in range(len(Path(n).parents) - 1)})
        add_dirs(tar, ["./"] + [f"./{d}/" for d in dirs])
        for name, data, mode in files:
            add_file(tar, f"./{name}", data, mode)
    glibc = glibc_needed([bdir / b for b in BINS])
    depends = f"libc6 (>= {glibc})" if glibc else "libc6"
    installed_kb = sum(len(d) for _, d, _ in files) // 1024 + 1
    control = (
        f"Package: ymir\nVersion: {debver(v)}-1\nArchitecture: {deb_arch}\nMaintainer: {MAINTAINER}\n"
        f"Installed-Size: {installed_kb}\nDepends: {depends}\nSection: net\nPriority: optional\nHomepage: {URL}\n"
        f"Description: {SUMMARY}\n ymird runs the model, inference and app roles; ymirctl sets up and talks to it.\n"
        " Run it per user: systemctl --user enable --now ymird\n"
    )
    md5 = "".join(f"{hashlib.md5(d).hexdigest()}  {n}\n" for n, d, _ in files)
    ctl_buf = io.BytesIO()
    with tarfile.open(fileobj=ctl_buf, mode="w:gz") as tar:
        add_dirs(tar, ["./"])
        add_file(tar, "./control", control.encode())
        add_file(tar, "./md5sums", md5.encode())
    deb = out / f"ymir_{debver(v)}-1_{deb_arch}.deb"
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


def pkgbuild(v, sums, local=False):
    """The ymir-bin recipe. `sums` maps arch -> sha256 of the release tarball."""
    if local:
        src = {a: f'("ymir-{v}-{a}-unknown-linux-gnu.tar.gz")' for a in ARCH}
    else:
        src = {a: f'("ymir-${{_ver}}-{a}-unknown-linux-gnu.tar.gz::${{url}}/releases/download/v${{_ver}}/ymir-${{_ver}}-{a}-unknown-linux-gnu.tar.gz")' for a in ARCH}
    return f"""# Maintainer: {MAINTAINER}
# Prebuilt binaries from the GitHub release. scripts/package.py --recipe (and the release workflow)
# fill in pkgver and the sha256sums; scripts/weirdnet_release.py taur publishes this recipe to TAUR.
pkgname=ymir-bin
_ver={v}
pkgver={pkgver(v)}
pkgrel=1
pkgdesc="{SUMMARY} (prebuilt binary)"
arch=('x86_64' 'aarch64')
url="{URL}"
license=('BSD-3-Clause')
depends=('glibc' 'libgcc')
optdepends=('llama.cpp: the llama-server engine for GGUF models'
            'python-vllm: the vLLM engine (NVIDIA)')
provides=('ymir')
conflicts=('ymir' 'ymir-git')
options=('!strip' '!debug')
source_x86_64={src['x86_64']}
source_aarch64={src['aarch64']}
sha256sums_x86_64=('{sums.get('x86_64', 'SKIP')}')
sha256sums_aarch64=('{sums.get('aarch64', 'SKIP')}')

package() {{
  cd "ymir-${{_ver}}-${{CARCH}}-unknown-linux-gnu"
  install -Dm755 bin/ymird "${{pkgdir}}/usr/bin/ymird"
  install -Dm755 bin/ymirctl "${{pkgdir}}/usr/bin/ymirctl"
  install -Dm644 share/doc/ymir/README.md "${{pkgdir}}/usr/share/doc/ymir/README.md"
  install -Dm644 LICENSE "${{pkgdir}}/usr/share/licenses/${{pkgname}}/LICENSE"
  install -Dm644 /dev/stdin "${{pkgdir}}/usr/lib/systemd/user/ymird.service" <<'UNIT'
{(ROOT / "packaging/systemd/ymird.service").read_text().rstrip()}
UNIT
}}
"""


def build_pacman(out, v, arch, tarball):
    if not shutil.which("makepkg"):
        print("notice: makepkg not found; skipping pkg.tar.zst")
        return []
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as t:
        t = Path(t)
        shutil.copy(tarball, t / tarball.name)
        (t / "PKGBUILD").write_text(pkgbuild(v, {arch: sha256(tarball)}, local=True))
        # The system makepkg.conf, with CARCH set so the aarch64 package can be made on x86_64.
        conf = t / "makepkg.conf"
        conf.write_text(f'source /etc/makepkg.conf\nCARCH="{arch}"\nPKGEXT=".pkg.tar.zst"\nPKGDEST="{out}"\nSRCDEST="{t}"\nBUILDDIR="{t / "build"}"\n')
        r = subprocess.run(["makepkg", "--config", str(conf), "--nodeps", "--noconfirm", "--force"], cwd=t, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
            sys.exit(f"makepkg failed for {arch}")
    return sorted(out.glob(f"ymir-bin-{pkgver(v)}-1-{arch}.pkg.tar.zst"))


def build_rpm(out, v, arch, bdir):
    if not shutil.which("rpmbuild"):
        print(f"notice: rpmbuild not found; skipping ymir-{v}-1.{arch}.rpm (the release workflow builds it)")
        return []
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        root = t / "root"
        for name, data, mode in system_files(bdir):
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            p.chmod(mode)
        files = "\n".join(f"/{n}" for n, _, _ in system_files(bdir))
        spec = t / "ymir.spec"
        spec.write_text(
            f"Name: ymir\nVersion: {v.replace('-', '~')}\nRelease: 1\nSummary: {SUMMARY}\nLicense: BSD-3-Clause\nURL: {URL}\nBuildArch: {arch}\n"
            f"%description\n{SUMMARY}.\n%install\ncp -a {root}/. %{{buildroot}}/\n%files\n{files}\n"
        )
        subprocess.run(["rpmbuild", "-bb", "--target", arch, "--define", f"_topdir {t}", "--define", f"_rpmdir {out}", "--define", "_build_name_fmt %%{NAME}-%%{VERSION}-%%{RELEASE}.%%{ARCH}.rpm", str(spec)], check=True)
    return sorted(out.glob(f"ymir-*.{arch}.rpm"))


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", action="append", help="target triple (repeatable)")
    ap.add_argument("--out", default="dist")
    ap.add_argument("--recipe", action="store_true", help="rewrite packaging/arch/ymir-bin/PKGBUILD with these tarballs' sums")
    a = ap.parse_args()
    v = version()
    out = (ROOT / a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    made, sums = [], {}
    for triple in a.target or [host_triple()]:
        arch = triple.split("-")[0]
        if arch not in ARCH or not triple.endswith("-linux-gnu"):
            sys.exit(f"{triple}: only x86_64/aarch64 Linux GNU targets are packaged here")
        bdir = bin_dir(triple)
        tz = build_tar_zip(out, v, triple, bdir)
        sums[arch] = sha256(tz[0])
        made += tz
        made += build_deb(out, v, arch, bdir)
        made += build_pacman(out, v, arch, tz[0])
        made += build_rpm(out, v, arch, bdir)
    made += sign(made)
    (out / "SHA256SUMS").write_text("".join(f"{sha256(f)}  {f.name}\n" for f in made if not f.name.endswith(".minisig")))
    if a.recipe:
        (ROOT / "packaging/arch/ymir-bin/PKGBUILD").write_text(pkgbuild(v, sums))
        print("wrote packaging/arch/ymir-bin/PKGBUILD")
    for f in made:
        print(f"{f.stat().st_size:>10}  {f.relative_to(ROOT) if f.is_relative_to(ROOT) else f}")


if __name__ == "__main__":
    main()
