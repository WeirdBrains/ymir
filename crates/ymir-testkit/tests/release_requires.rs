//! `release.toml` `[[requires]]` must name the versions the workspace builds with. A
//! `[[requires]]` left behind after a dependency bump (0.4.2 against a Cargo.toml at 0.4.3) once
//! blocked a cut, because the release script checked the lock only then. Since helpers 0.3.3
//! `validate` and `--dry-run` compare them too; this test keeps the check in `cargo test`.

use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn read_toml(p: &Path) -> toml::Table {
    std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display())).parse().unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// Every version requirement for `name` in a manifest's dependency tables (workspace, target,
/// dev and build ones included).
fn requirements(manifest: &toml::Table, name: &str) -> Vec<String> {
    fn dep_tables<'a>(t: &'a toml::Table, out: &mut Vec<&'a toml::Table>) {
        for (k, v) in t {
            if let Some(sub) = v.as_table() {
                if matches!(k.as_str(), "dependencies" | "dev-dependencies" | "build-dependencies") {
                    out.push(sub);
                } else {
                    dep_tables(sub, out);
                }
            }
        }
    }
    let mut tables = Vec::new();
    dep_tables(manifest, &mut tables);
    tables
        .into_iter()
        .filter_map(|t| t.get(name))
        .filter_map(|d| match d {
            toml::Value::String(s) => Some(s.clone()),
            toml::Value::Table(t) => t.get("version").and_then(|v| v.as_str()).map(str::to_string),
            _ => None,
        })
        .collect()
}

#[test]
fn release_requires_match_the_manifest_and_the_lock() {
    let root = root();
    let release = read_toml(&root.join("release.toml"));
    let reqs = release.get("requires").and_then(|r| r.as_array()).cloned().unwrap_or_default();
    let own = read_toml(&root.join("Cargo.toml"))["workspace"]["package"]["version"].as_str().unwrap().to_string();
    for r in reqs {
        let name = r["crate"].as_str().expect("[[requires]] crate");
        let version = r.get("version").and_then(|v| v.as_str()).unwrap_or("{version}").replace("{version}", &own);
        if let Some(rel) = r.get("manifest").and_then(|m| m.as_str()) {
            let found = requirements(&read_toml(&root.join(rel)), name);
            assert!(!found.is_empty(), "{rel} has no dependency on {name}");
            for req in found {
                let bare = req.trim_start_matches(['=', '^', '~', ' ']);
                assert_eq!(bare, version, "release.toml [[requires]] {name} = {version}, but {rel} asks for {req}");
            }
        }
        if let Some(rel) = r.get("lock").and_then(|l| l.as_str()) {
            let lock = read_toml(&root.join(rel));
            let got: Vec<_> = lock["package"].as_array().unwrap().iter().filter(|p| p["name"].as_str() == Some(name)).collect();
            assert_eq!(got.len(), 1, "{rel}: expected one {name} entry");
            assert_eq!(
                got[0]["version"].as_str(),
                Some(version.as_str()),
                "release.toml [[requires]] {name} = {version}, but {rel} has another version"
            );
            let source = got[0].get("source").and_then(|s| s.as_str()).unwrap_or("");
            assert!(source.starts_with("registry+"), "{rel}: {name} must come from crates.io for a release, not {source:?}");
        }
    }
}
