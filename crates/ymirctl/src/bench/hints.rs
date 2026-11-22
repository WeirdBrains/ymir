//! Where to dial the nodes: the addresses a benchmark is given besides `[network] peers`.
//!
//! They come from `--hint ADDR`, `--hint @FILE` (one address per line, `#` comments), and the
//! file `bench-hints` in the Ymir home, which is read when it exists. `ymirctl bench hints add`
//! fills that file from the output of `ymirctl pool addresses` on another node (which prints
//! the addresses that node can be reached at), so nobody types a multiaddr:
//!
//! ```text
//! ssh m2 ymirctl pool addresses | ymirctl bench hints add
//! ```
//!
//! Only dialable, stable-looking addresses are kept (no loopback, link-local or relay circuits).
//! The file lives in the user's home, never in a repository.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use ymir_core::Paths;

/// The default hints file.
pub fn default_file(paths: &Paths) -> PathBuf {
    paths.home.join("bench-hints")
}

/// The usable addresses in `text`: one multiaddr ending in `/p2p/PEER` per line.
pub fn parse(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for line in text.lines() {
        let l = line.split('#').next().unwrap_or("").trim();
        if l.is_empty() || !usable(l) || out.iter().any(|x| x == l) {
            continue;
        }
        out.push(l.to_string());
    }
    out
}

fn usable(a: &str) -> bool {
    a.starts_with('/')
        && a.contains("/p2p/")
        && !a.contains("/p2p-circuit")
        && !a.starts_with("/ip4/127.")
        && !a.starts_with("/ip6/::1")
        && !a.starts_with("/ip6/fe80")
        && !a.starts_with("/ip4/169.254.")
}

/// Every hint to dial: the flags (an `@file` is read), then the default file.
pub fn resolve(paths: &Paths, flags: &[String]) -> Result<Vec<String>> {
    let mut all = Vec::new();
    for f in flags {
        match f.strip_prefix('@') {
            Some(file) => {
                let text = std::fs::read_to_string(file).with_context(|| format!("reading the hints file {file}"))?;
                all.extend(parse(&text));
            }
            None => all.push(f.clone()),
        }
    }
    if let Ok(text) = std::fs::read_to_string(default_file(paths)) {
        all.extend(parse(&text));
    }
    let mut seen = std::collections::HashSet::new();
    all.retain(|a| seen.insert(a.clone()));
    Ok(all)
}

/// Adds the usable addresses in `text` to `file`; returns how many were new.
pub fn add(file: &Path, text: &str) -> Result<usize> {
    let have = std::fs::read_to_string(file).unwrap_or_default();
    let known = parse(&have);
    let new: Vec<String> = parse(text).into_iter().filter(|a| !known.contains(a)).collect();
    if new.is_empty() {
        if parse(text).is_empty() {
            bail!("no dialable `/ip4/…/p2p/PEER` addresses in the input (loopback, link-local and relay addresses are ignored)");
        }
        return Ok(0);
    }
    let mut out = have;
    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
    }
    for a in &new {
        out.push_str(a);
        out.push('\n');
    }
    if let Some(dir) = file.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(file, out).with_context(|| format!("writing {}", file.display()))?;
    Ok(new.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "/dns4/relay.example/tcp/4747/p2p/12D3KooWRelay/p2p-circuit/p2p/12D3KooWNode\n\
/ip4/192.0.2.9/tcp/4848/p2p/12D3KooWNode\n/ip4/127.0.0.1/tcp/4848/p2p/12D3KooWNode\n/ip6/fe80::1/tcp/4848/p2p/12D3KooWNode\n\
# a comment\n/ip4/192.0.2.9/tcp/4848/p2p/12D3KooWNode\n";

    #[test]
    fn only_dialable_addresses_are_kept_once() {
        assert_eq!(parse(SAMPLE), vec!["/ip4/192.0.2.9/tcp/4848/p2p/12D3KooWNode".to_string()]);
    }

    #[test]
    fn add_appends_new_addresses_and_flags_read_files() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("bench-hints");
        assert_eq!(add(&file, SAMPLE).unwrap(), 1);
        assert_eq!(add(&file, SAMPLE).unwrap(), 0);
        assert!(add(&file, "nothing here").is_err());
        let paths = Paths { home: dir.path().to_path_buf() };
        let flag = format!("@{}", file.display());
        let got = resolve(&paths, &["/ip4/198.51.100.1/tcp/1/p2p/X".into(), flag]).unwrap();
        assert_eq!(got.len(), 2, "the file is read once even when it is also the default");
        assert!(got[0].contains("198.51.100.1"));
    }
}
