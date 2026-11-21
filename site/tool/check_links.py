#!/usr/bin/env python3
"""Fail when site prose or a README refers to a document or URL without a link.

    python3 site/tool/check_links.py

Rule (weirdnet-agents AGENTS.md, Public content, 7): every reference to another URL or document is a clickable
link. This flags, outside links and code blocks, (1) file names ending in .md and (2) bare http(s) URLs. It
checks the Dart page sources under site/lib (text inside HTML strings and Component.text calls), README.md,
ROADMAP.md, HANDOFF.md and site/README.md. Loopback and documentation-range hosts are examples, not references.
"""
import pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
MD_NAME = re.compile(r'(?<![\w/.-])[\w./-]*\w\.md\b')
URL = re.compile(r'https?://[^\s<>"\')\]`]+')
EXAMPLE = re.compile(r'https?://(127\.0\.0\.1|localhost|0\.0\.0\.0|192\.0\.2\.|\[::1\]|example\.|[\w.-]*\.example|server-)')

def scan(text):
    """Yield offending snippets from HTML-ish text."""
    text = re.sub(r'<pre\b.*?</pre>', ' ', text, flags=re.S)
    text = re.sub(r'<a\b.*?</a>', ' ', text, flags=re.S)
    text = re.sub(r'<!--.*?-->', ' ', text, flags=re.S)
    text = re.sub(r'<[^>]+>', ' ', text)  # remaining tags, with their href and src attributes
    for m in MD_NAME.finditer(text):
        yield m.group(0)
    for m in URL.finditer(text):
        if not EXAMPLE.match(m.group(0)):
            yield m.group(0)

def scan_markdown(text):
    text = re.sub(r'```.*?```', ' ', text, flags=re.S)
    text = re.sub(r'<!--.*?-->', ' ', text, flags=re.S)
    text = re.sub(r'\[[^\]]*\]\([^)]*\)', ' ', text)         # [text](target)
    text = re.sub(r'<https?://[^>]+>', ' ', text)            # <autolink>
    text = re.sub(r'^\[[^\]]+\]:\s*\S+.*$', ' ', text, flags=re.M)  # reference definitions
    text = re.sub(r'`([^`]*)`', lambda m: m.group(0) if '.md' in m.group(1) else ' ', text)
    return scan(text)

DART_LINK = re.compile(r"\ba\(\s*(?:classes:[^,]*,\s*)?href:[^\[]*\[\s*(?:Component\.text\((?:'(?:[^'\\]|\\.)*'\s*)+\)[,\s]*)+\]")

def dart_strings(src):
    src_links_removed = DART_LINK.sub(' ', src)
    for m in re.finditer(r"r?'''(.*?)'''", src, flags=re.S):
        yield m.group(1)
    for m in re.finditer(r"Component\.text\(\s*((?:'(?:[^'\\]|\\.)*'\s*)+)", src_links_removed):
        yield ''.join(re.findall(r"'((?:[^'\\]|\\.)*)'", m.group(1)))

def main():
    bad = []
    for f in sorted((ROOT / 'site/lib').rglob('*.dart')):
        if 'content/' in str(f):
            continue
        for s in dart_strings(f.read_text()):
            bad += [(f, x) for x in scan(s)]
    for name in ('README.md', 'ROADMAP.md', 'HANDOFF.md', 'site/README.md'):
        f = ROOT / name
        if f.exists():
            bad += [(f, x) for x in scan_markdown(f.read_text())]
    for f, x in bad:
        print(f"{f.relative_to(ROOT)}: unlinked reference: {x}")
    if bad:
        sys.exit(f"{len(bad)} unlinked reference(s): make each one a link (see AGENTS.md, Public content, rule 7)")

main()
