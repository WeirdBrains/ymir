//! Publishing: sign a bundle directory for [`crate::update`] (`ymirctl update sign-bundle`).
//!
//! The release flow unpacks a release tarball into a plain file set (`ymird`, `ymirctl`,
//! `llama.cpp/*`), then this writes the signed `manifest.env` (every file's size and BLAKE3)
//! and the org root as `roots/1.env`. Before writing, it checks that the release key is
//! delegated by the root for [`APP`]/[`CHANNEL`], and afterwards it verifies the result as a
//! node would (pin, chain, signature, scope, expiry), so a bad key or root fails here and not
//! on the nodes. The key is read from a file holding the 32-byte seed as 64 hex characters;
//! it is never printed.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use semver::Version;
use weirdnet_update::envelope::Domain;
use weirdnet_update::manifest::{Arch, FullPackage, Labels, PackageFormat, Platform};
use weirdnet_update::{Client, FileEntry, Root, Signed, SigningKey, Timestamp, UpdateBundle, UpdateKind, UpdateManifest, VerifyOptions};
use zeroize::Zeroizing;

use crate::update::{APP, CHANNEL, DAEMON_FILE};

/// What to sign.
#[derive(Debug, Clone)]
pub struct BundleSpec {
    pub version: Version,
    /// Must be higher than any update already published on the channel.
    pub seq: u64,
    pub platform: Platform,
    pub arch: Arch,
    pub notes: String,
    /// How long the manifest is accepted for staging.
    pub valid_secs: u64,
}

/// The update id a bundle is published under.
pub fn update_id(seq: u64) -> String {
    format!("{APP}-{CHANNEL}-{seq}")
}

/// The seq for a release version, so seqs rise with versions: `MAJOR*10^6 + MINOR*10^3 + PATCH`.
pub fn seq_for(v: &Version) -> Result<u64> {
    if !v.pre.is_empty() {
        bail!("pre-releases are not published as updates on {CHANNEL}");
    }
    if v.minor >= 1000 || v.patch >= 1000 {
        bail!("{v}: minor and patch must be below 1000 for the seq scheme");
    }
    Ok(v.major * 1_000_000 + v.minor * 1_000 + v.patch)
}

/// Parses an OS name as manifests use it (`linux`, `macos`, ...).
pub fn platform(s: &str) -> Result<Platform> {
    match s {
        "linux" => Ok(Platform::Linux),
        "macos" => Ok(Platform::Macos),
        "freebsd" => Ok(Platform::Freebsd),
        _ => bail!("unknown OS {s:?} (linux, macos, freebsd)"),
    }
}

/// Parses a CPU name (`x86_64`/`x86-64`, `aarch64`/`arm64`).
pub fn arch(s: &str) -> Result<Arch> {
    match s {
        "x86_64" | "x86-64" | "amd64" => Ok(Arch::X86_64),
        "aarch64" | "arm64" => Ok(Arch::Aarch64),
        _ => bail!("unknown CPU {s:?} (x86_64, aarch64)"),
    }
}

/// Reads a release key file (64 hex characters, the seed). Errors never include the contents.
pub fn read_key(path: &Path) -> Result<SigningKey> {
    let text = Zeroizing::new(std::fs::read_to_string(path).with_context(|| format!("reading the key file {}", path.display()))?);
    let hex = text.trim();
    if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        bail!("{}: not a key file (expected 64 hex characters)", path.display());
    }
    let mut seed = Zeroizing::new([0u8; 32]);
    for (i, out) in seed.iter_mut().enumerate() {
        *out = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).map_err(|_| anyhow!("{}: not a key file", path.display()))?;
    }
    Ok(SigningKey::from_seed(&seed))
}

/// What [`sign_bundle`] wrote.
#[derive(Debug)]
pub struct SignedBundle {
    pub update_id: String,
    pub files: usize,
    pub bytes: u64,
    pub pin: String,
}

/// The files of a bundle directory, relative, sorted, without the envelopes.
fn payload_files(dir: &Path) -> Result<Vec<String>> {
    fn walk(base: &Path, dir: &Path, out: &mut Vec<String>) -> Result<()> {
        for e in std::fs::read_dir(dir).with_context(|| dir.display().to_string())? {
            let e = e?;
            let p = e.path();
            let meta = std::fs::symlink_metadata(&p)?;
            let rel = p.strip_prefix(base)?.to_string_lossy().replace('\\', "/");
            if meta.is_dir() {
                if rel != "roots" {
                    walk(base, &p, out)?;
                }
            } else if meta.is_file() {
                if rel != "manifest.env" && rel != "freshness.env" {
                    out.push(rel);
                }
            } else {
                bail!("{rel}: a bundle holds only regular files (no symlinks)");
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    walk(dir, dir, &mut out)?;
    out.sort();
    Ok(out)
}

/// Signs the file set in `dir` with `key` under the org root in `root_env` (a signed root
/// envelope), writing `dir/manifest.env` and `dir/roots/1.env`. With `pin`, the root must have
/// that fingerprint.
pub fn sign_bundle(dir: &Path, key: &SigningKey, root_env: &[u8], pin: Option<&str>, spec: &BundleSpec, now: Timestamp) -> Result<SignedBundle> {
    let root_signed = Signed::from_bytes(root_env).map_err(|e| anyhow!("the root file: {e}"))?;
    let root: Root = root_signed.decode().map_err(|e| anyhow!("the root file: {e}"))?;
    let root_pin = root.pin().to_string();
    if let Some(p) = pin
        && !p.eq_ignore_ascii_case(&root_pin)
    {
        bail!("the root file's fingerprint is {root_pin}, not the expected {p}");
    }
    let public = key.public();
    let delegated = root.release.iter().any(|d| {
        d.key == public
            && (d.apps.is_empty() || d.apps.iter().any(|a| a == APP))
            && (d.channels.is_empty() || d.channels.iter().any(|c| c == CHANNEL))
    });
    if !delegated {
        bail!("this release key is not delegated by the root for {APP}/{CHANNEL}");
    }
    let names = payload_files(dir)?;
    if !names.iter().any(|n| n == DAEMON_FILE) {
        bail!("{}: no `{DAEMON_FILE}` at the top of the bundle", dir.display());
    }
    let mut files = Vec::with_capacity(names.len());
    let mut bytes = 0;
    for n in &names {
        let data = std::fs::read(dir.join(n)).with_context(|| n.clone())?;
        bytes += data.len() as u64;
        files.push(FileEntry::describe(n, &data));
    }
    let id = update_id(spec.seq);
    let m = UpdateManifest {
        app: APP.into(),
        channel: CHANNEL.into(),
        update_id: id.clone(),
        seq: spec.seq,
        kind: UpdateKind::Full(FullPackage {
            platform: spec.platform,
            arch: spec.arch,
            format: PackageFormat::TarGz,
            signing_identity: "unsigned".into(),
        }),
        version: spec.version.clone(),
        base_releases: vec![],
        files,
        min_client: None,
        max_client: None,
        capabilities: vec![],
        labels: Labels::default(),
        targeting: None,
        rollout: None,
        rollback_of: None,
        notes: spec.notes.clone(),
        created: now,
        expires: now.plus(spec.valid_secs),
    };
    let manifest = Signed::sign(Domain::Manifest, &m, &[key]).map_err(|e| anyhow!("signing: {e}"))?;

    // Check it as a node would before writing anything.
    let mut client = Client::new(root.pin(), &root_signed, now).map_err(|e| anyhow!("the root does not verify: {e}"))?;
    let bundle = UpdateBundle { roots: vec![root_signed.clone()], manifest: Some(manifest.clone()), freshness: None };
    let below = Version::new(0, 0, 0);
    client.verify(APP, CHANNEL, &bundle, &VerifyOptions::new(now, below)).map_err(|e| anyhow!("the signed bundle does not verify: {e}"))?;

    let roots: PathBuf = dir.join("roots");
    if roots.exists() {
        std::fs::remove_dir_all(&roots)?;
    }
    std::fs::create_dir_all(&roots)?;
    std::fs::write(roots.join("1.env"), root_env)?;
    std::fs::write(dir.join("manifest.env"), manifest.to_bytes().map_err(|e| anyhow!("{e}"))?)?;
    Ok(SignedBundle { update_id: id, files: names.len(), bytes, pin: root_pin })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::update::this_target;
    use weirdnet_update::Delegation;

    fn key(n: u8) -> SigningKey {
        SigningKey::from_seed(&[n; 32])
    }

    fn root(release: &SigningKey) -> (Vec<u8>, String) {
        let r = key(1);
        let root = Root {
            version: 1,
            org: "weirdbrains".into(),
            keys: vec![r.public()],
            threshold: 1,
            release: vec![Delegation::any(release.public())],
            release_threshold: 1,
            timestamp: vec![],
            timestamp_threshold: 0,
            expires: None,
        };
        (Signed::sign(Domain::Root, &root, &[&r]).unwrap().to_bytes().unwrap(), root.pin().to_string())
    }

    fn spec(seq: u64) -> BundleSpec {
        let (platform, arch) = this_target().unwrap();
        BundleSpec { version: Version::new(0, 15, 2), seq, platform, arch, notes: "test".into(), valid_secs: 86_400 }
    }

    fn payload(dir: &Path) {
        std::fs::create_dir_all(dir.join("llama.cpp")).unwrap();
        std::fs::write(dir.join("ymird"), b"#!/bin/sh\necho ymird\n").unwrap();
        std::fs::write(dir.join("ymirctl"), b"#!/bin/sh\n").unwrap();
        std::fs::write(dir.join("llama.cpp/engine.toml"), b"engine = \"llama.cpp\"\n").unwrap();
    }

    #[test]
    fn seq_rises_with_versions() {
        assert_eq!(seq_for(&Version::new(0, 15, 2)).unwrap(), 15_002);
        assert!(seq_for(&Version::new(0, 15, 2)).unwrap() > seq_for(&Version::new(0, 14, 999)).unwrap());
        assert!(seq_for(&Version::new(1, 0, 0)).unwrap() > seq_for(&Version::new(0, 999, 999)).unwrap());
        assert!(seq_for(&Version::parse("0.16.0-rc.1").unwrap()).is_err());
        assert_eq!(update_id(15_002), "com.weirdbrains.ymir-stable-15002");
    }

    #[test]
    fn key_files() {
        let t = tempfile::tempdir().unwrap();
        let good = t.path().join("k");
        std::fs::write(&good, format!("{}\n", "07".repeat(32))).unwrap();
        assert_eq!(read_key(&good).unwrap().public(), key(7).public());
        let bad = t.path().join("bad");
        std::fs::write(&bad, "secret-but-not-hex").unwrap();
        let e = read_key(&bad).unwrap_err().to_string();
        assert!(!e.contains("secret"), "{e}");
    }

    #[cfg(unix)]
    #[test]
    fn signed_bundle_stages_on_a_node() {
        use crate::Paths;
        use crate::update::Updater;
        let t = tempfile::tempdir().unwrap();
        let rel = key(10);
        let (root_env, pin) = root(&rel);
        let b = t.path().join("bundle");
        payload(&b);
        let now = Timestamp::now();
        let s = sign_bundle(&b, &rel, &root_env, Some(&pin), &spec(15_002), now).unwrap();
        assert_eq!(s.files, 3);
        assert!(b.join("roots/1.env").is_file() && b.join("manifest.env").is_file());
        // Signing again (e.g. a re-run) replaces the envelopes and keeps them out of the file set.
        let s = sign_bundle(&b, &rel, &root_env, Some(&pin), &spec(15_002), now).unwrap();
        assert_eq!(s.files, 3);

        let paths = Paths::new(t.path().join("home"));
        let mut u = Updater::open(&paths, now).unwrap();
        u.trust(&pin, &root_env, now).unwrap();
        let v = u.stage(&b, &Version::new(0, 8, 0), now).unwrap();
        assert_eq!(v.manifest.update_id, "com.weirdbrains.ymir-stable-15002");
        assert_eq!(v.manifest.version, Version::new(0, 15, 2));
    }

    #[test]
    fn refuses_wrong_key_or_root() {
        let t = tempfile::tempdir().unwrap();
        let (root_env, pin) = root(&key(10));
        let b = t.path().join("bundle");
        payload(&b);
        let now = Timestamp::now();
        let e = sign_bundle(&b, &key(11), &root_env, None, &spec(1), now).unwrap_err().to_string();
        assert!(e.contains("not delegated"), "{e}");
        let e = sign_bundle(&b, &key(10), &root_env, Some(&"0".repeat(pin.len())), &spec(1), now).unwrap_err().to_string();
        assert!(e.contains("fingerprint"), "{e}");
        assert!(!b.join("manifest.env").exists());
        std::fs::remove_file(b.join("ymird")).unwrap();
        let e = sign_bundle(&b, &key(10), &root_env, None, &spec(1), now).unwrap_err().to_string();
        assert!(e.contains("ymird"), "{e}");
    }
}
