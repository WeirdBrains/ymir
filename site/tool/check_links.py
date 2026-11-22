#!/usr/bin/env python3
"""Flag references in the site content that should be clickable links but are not.

Scans site/lib/**/*.dart (the page HTML lives there). Text inside <a>...</a> (or a Jaspr a(...) call) counts as linked; <pre> is ignored
(commands and listings may show bare names and URLs); loopback and example hosts are placeholders. Flags:
  * bare http(s) URLs,
  * names of documents that have a home on the site or GitHub (DESIGN.md, AGENTS.md, CHANGELOG, LICENSE,
    README, ROADMAP.md, BRAND.md, node.toml, release.toml, roadmap.toml),
  * crates.io / docs.rs / TAUR / F-Droid / Flathub / Homebrew names written as plain words are covered by the
    URL rule when they carry a host; add names to DOC_NAMES below to cover more.
Then, with --dist DIR (a built site), checks every internal href/src resolves to a file.
Usage: check_links.py [--dist build/jaspr] [--base /ymir/]
"""
import re, sys, pathlib, argparse

DOC_NAMES = [
    r"DESIGN\.md", r"AGENTS\.md", r"CHANGELOG(?:\.md)?", r"LICENSE", r"README(?:\.md)?", r"ROADMAP\.md",
    r"BRAND\.md", r"node\.toml", r"release\.toml", r"roadmap\.toml", r"peers\.txt",
]

CRATE = r"weirdnet-(?:agent|backends|openai|model|mcp|semcache|schemes|alerts)"
NAME_RE = re.compile(r"(?<![\w./:~-])(" + "|".join(DOC_NAMES + [CRATE]) + r")(?![\w-])")
URL_RE = re.compile(r"https?://[^\s<>\"')]+")
EXAMPLE_HOST = re.compile(r"https?://(?:127\.0\.0\.1|localhost|\[::1\]|[\w.-]*example[\w.-]*|server-[ab])")
SPANS = [  # (kind, regex): linked or ignored regions
    ("skip", re.compile(r"<pre\b.*?</pre>", re.S)),
    ("link", re.compile(r"<a\b.*?</a>", re.S)),
    ("link", re.compile(r"\ba\(\s*href:.*?\]\s*\)", re.S)),  # Jaspr a(href: ..., [text])
    ("skip", re.compile(r"<[^>\n]+>")),                          # tag attributes
    ("skip", re.compile(r"^\s*(?:import|//).*$", re.M)),
    ("skip", re.compile(r"^(?:const|final|String)\b[^;]*;", re.M)),  # Dart constants holding link targets
]


def scan_file(f: pathlib.Path):
    raw = f.read_text()
    kinds = [None] * len(raw)
    for kind, rx in SPANS:
        for m in rx.finditer(raw):
            for i in range(m.start(), m.end()):
                if kinds[i] is None:
                    kinds[i] = kind
    seen = set()
    for m in sorted(list(NAME_RE.finditer(raw)) + list(URL_RE.finditer(raw)), key=lambda m: m.start()):
        k = kinds[m.start()]
        if k == "skip":
            continue
        text = m.group(0)
        line = raw.count("\n", 0, m.start()) + 1
        if m.re is URL_RE:
            if k != "link" and not EXAMPLE_HOST.match(text):
                yield line, text
        else:
            # A name needs a link at its first mention on the page; later mentions may stay plain.
            key = text
            if key not in seen and k != "link":
                yield line, text
            seen.add(key)


def scan_sources(root: pathlib.Path) -> int:
    bad = 0
    for f in sorted((root / "lib").rglob("*.dart")):
        if f.name.startswith("main.server") or f.as_posix().endswith("content/roadmap.dart"):
            # content/roadmap.dart is generated from roadmap.toml; its crate names are rendered as crates.io chips.
            continue
        for line, text in scan_file(f):
            print(f"{f.relative_to(root)}:{line}: not a link: {text}")
            bad += 1
    return bad


def check_dist(dist: pathlib.Path, base: str) -> int:
    bad = 0
    for page in sorted(dist.rglob("*.html")):
        if "docs/api" in page.as_posix() or "docs/reference" in page.as_posix() and False:
            continue
        for m in re.finditer(r'(?:href|src)="([^"]+)"', page.read_text(errors="ignore")):
            u = m.group(1).split("#")[0].split("?")[0]
            if not u or re.match(r"[a-z][a-z0-9+.-]*:", u) or u.startswith("//"):
                continue
            if u.startswith("/"):
                if not u.startswith(base):
                    continue
                t = dist / u[len(base):]
            else:
                t = (page.parent / u)
            if not (t.is_file() or (t / "index.html").is_file()):
                print(f"{page.relative_to(dist)}: dead internal link {m.group(1)}")
                bad += 1
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist")
    ap.add_argument("--base", default="/")
    a = ap.parse_args()
    site = pathlib.Path(__file__).resolve().parent.parent
    bad = scan_sources(site)
    if a.dist:
        bad += check_dist(pathlib.Path(a.dist), a.base)
    print("links: ok" if not bad else f"links: {bad} problem(s)")
    sys.exit(1 if bad else 0)
