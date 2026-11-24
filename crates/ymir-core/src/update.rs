//! Self-update with rollback, on `weirdnet-update`.
//!
//! What this does: checks a signed update bundle against a pinned org root (threshold signatures,
//! scope, expiry, anti-rollback, no downgrade), checks every file's size and BLAKE3, stages the
//! release under `updates/releases/<id>/`, and runs it on trial: a start that does not reach
//! "ready" counts as failed, and after two the previous good release comes back and the update is
//! marked bad for good.
//!
//! Fetching ([`Updater::fetch`], `ymirctl update fetch`): a bundle is read from a [`Source`], a
//! bundle directory served over HTTPS or on disk, with the same layout as below (root rotations
//! as `roots/1.env`, `roots/2.env`, ... in order). The manifest comes first and is verified
//! against the pinned root before anything else is downloaded, and each file is read up to its
//! signed size. A signed announcement (`weirdnet-update`'s, as relays carry them) can name the
//! manifest the source must serve. It does not install system packages (`.deb`/`.rpm` are
//! refused: only the `swap` formats, a set of files, are taken). Nothing is installed without
//! `ymirctl update stage` or `update fetch`.
//!
//! Layout under `<home>/updates/`:
//! `trust.json` (the pinned root fingerprint and the first root), `client.state` (anti-rollback
//! state), `health.json` (boot trial), `active.json` (the good release), `releases/<id>/`.
//! A bundle directory holds `manifest.env`, optional `freshness.env` and `roots/*.env` (signed
//! envelopes, `Signed::to_bytes`), and the files the manifest lists at their paths.

use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use semver::Version;
use serde::{Deserialize, Serialize};
use weirdnet_update::health::{BootChoice, BootHealth, HealthState};
use weirdnet_update::manifest::{Arch, InstallMode, Platform};
use weirdnet_update::{
    Announcement, Client, ClientState, Hash32, RootPin, Signed, Timestamp, UpdateBundle, UpdateKind, UpdateManifest, VerifiedUpdate, VerifyOptions,
};

use crate::Paths;

/// The app id and default channel the manifests are signed for.
pub const APP: &str = "com.weirdbrains.ymir";
pub const CHANNEL: &str = "stable";
/// The program the daemon runs from a release.
pub const DAEMON_FILE: &str = "ymird";

/// The programs a package carries besides the daemon: `ymirctl` and the bundled llama.cpp
/// servers ymird launches.
#[cfg_attr(not(unix), allow(dead_code))]
const PROGRAMS: &[&str] = &[DAEMON_FILE, "ymirctl", "llama-server", "rpc-server"];

/// Whether an installed package file gets the executable bit: a known program, a file that was
/// executable in the bundle, or a native binary or script (ELF, Mach-O, `#!`), wherever it sits
/// in the package. Everything else is installed read-only data.
#[cfg_attr(not(unix), allow(dead_code))]
fn executable(path: &str, data: &[u8], bundle_mode: u32) -> bool {
    let name = path.rsplit(['/', '\\']).next().unwrap_or(path);
    let magic = data.get(..4).unwrap_or(data);
    PROGRAMS.contains(&name)
        || bundle_mode & 0o111 != 0
        || magic == b"\x7fELF"
        || magic == [0xcf, 0xfa, 0xed, 0xfe]
        || magic == [0xca, 0xfe, 0xba, 0xbe]
        || data.starts_with(b"#!")
}

/// Where the updater keeps its files.
pub fn dir(paths: &Paths) -> PathBuf {
    paths.home.join("updates")
}

#[derive(Serialize, Deserialize)]
struct Trust {
    pin: String,
    root: Vec<u8>,
}

#[derive(Default, Serialize, Deserialize)]
struct Active {
    /// The update id of the release that last passed its trial, if any (else the installed binary).
    good: Option<String>,
    /// The update id on trial or staged, with its manifest (for recording the install on commit).
    pending: Option<Pending>,
}

#[derive(Serialize, Deserialize)]
struct Pending {
    id: String,
    manifest: weirdnet_update::UpdateManifest,
    hash: String,
}

/// What `boot` tells the starter to run.
#[derive(Debug, PartialEq, Eq)]
pub enum Boot {
    /// The installed binary (nothing staged, or nothing good yet).
    Installed,
    /// A release directory's daemon: the good one, or the candidate on trial.
    Run { id: String, path: PathBuf, trial: bool },
    /// The candidate failed twice and was set aside; this is what runs now.
    RolledBack { failed: String, run: Option<(String, PathBuf)> },
}

/// What `status` shows.
#[derive(Debug, PartialEq, Eq)]
pub struct Status {
    pub trusted: bool,
    pub good: Option<String>,
    pub health: HealthState,
    pub bad: Vec<String>,
}

pub struct Updater {
    dir: PathBuf,
    client: Option<Client>,
    health: BootHealth,
    active: Active,
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("replacing {}", path.display()))
}

fn read_opt(path: &Path) -> Result<Option<Vec<u8>>> {
    match std::fs::read(path) {
        Ok(b) => Ok(Some(b)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

/// The largest manifest, freshness or root envelope a source may send.
const MAX_ENVELOPE: u64 = 1 << 20;
/// The most root rotations read from a source.
const MAX_ROOTS: usize = 16;

/// Where [`Updater::fetch`] reads a bundle: the bundle directory's files by relative path.
pub trait Source {
    /// Opens `path` (`manifest.env`, `freshness.env`, `roots/N.env`, or a file the manifest
    /// lists); `None` when the source does not have it.
    fn open(&self, path: &str) -> Result<Option<Box<dyn std::io::Read + '_>>>;
    /// The source, for messages.
    fn describe(&self) -> String;
}

/// A bundle directory on disk as a [`Source`].
pub struct DirSource(pub PathBuf);

impl Source for DirSource {
    fn open(&self, path: &str) -> Result<Option<Box<dyn std::io::Read + '_>>> {
        match std::fs::File::open(self.0.join(path)) {
            Ok(f) => Ok(Some(Box::new(f))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e).with_context(|| format!("{}", self.0.join(path).display())),
        }
    }

    fn describe(&self) -> String {
        self.0.display().to_string()
    }
}

/// What [`Updater::fetch`] staged.
#[derive(Debug)]
pub struct Fetched {
    pub update: VerifiedUpdate,
    pub files: usize,
    pub bytes: u64,
}

fn read_capped(src: &dyn Source, path: &str, max: u64) -> Result<Option<Vec<u8>>> {
    let Some(r) = src.open(path)? else { return Ok(None) };
    let mut b = Vec::new();
    r.take(max + 1).read_to_end(&mut b).with_context(|| format!("reading {path} from {}", src.describe()))?;
    if b.len() as u64 > max {
        bail!("{path} from {} is larger than {max} bytes", src.describe());
    }
    Ok(Some(b))
}

/// What Ymir installs: a full package of plain files for this machine, with the daemon in it.
fn check_package(m: &UpdateManifest) -> Result<()> {
    let UpdateKind::Full(pkg) = &m.kind else { bail!("only full packages are taken here, not {}", m.kind.name()) };
    if Some((pkg.platform, pkg.arch)) != this_target() {
        bail!("the package is for {}/{:?}, not this machine", pkg.platform.as_str(), pkg.arch);
    }
    if pkg.format.install_mode() != InstallMode::Swap {
        bail!("a {:?} package needs a package manager; Ymir takes only plain file sets", pkg.format);
    }
    if !m.files.iter().any(|f| f.path == DAEMON_FILE) {
        bail!("the package has no `{DAEMON_FILE}`");
    }
    Ok(())
}

/// The OS and CPU this build runs on, as manifests name them.
pub fn this_target() -> Option<(Platform, Arch)> {
    let os = if cfg!(target_os = "linux") {
        Platform::Linux
    } else if cfg!(target_os = "macos") {
        Platform::Macos
    } else if cfg!(target_os = "windows") {
        Platform::Windows
    } else if cfg!(target_os = "freebsd") {
        Platform::Freebsd
    } else {
        return None;
    };
    let arch = if cfg!(target_arch = "x86_64") {
        Arch::X86_64
    } else if cfg!(target_arch = "aarch64") {
        Arch::Aarch64
    } else {
        return None;
    };
    Some((os, arch))
}

impl Updater {
    /// Opens the updater's state under `<home>/updates`, creating it (mode 0700) on first use.
    pub fn open(paths: &Paths, now: Timestamp) -> Result<Self> {
        let dir = dir(paths);
        std::fs::create_dir_all(dir.join("releases")).with_context(|| format!("creating {}", dir.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))?;
        }
        let client = match read_opt(&dir.join("trust.json"))? {
            None => None,
            Some(b) => {
                let t: Trust = serde_json::from_slice(&b).context("trust.json")?;
                let pin = RootPin::from_hex(&t.pin).map_err(|e| anyhow!("trust.json pin: {e}"))?;
                Some(match read_opt(&dir.join("client.state"))? {
                    Some(state) => {
                        let st = ClientState::from_bytes(&state).map_err(|e| anyhow!("client.state: {e}"))?;
                        Client::from_state(pin, st, now).map_err(|e| anyhow!("client.state: {e}"))?
                    }
                    None => {
                        let root = Signed::from_bytes(&t.root).map_err(|e| anyhow!("trust.json root: {e}"))?;
                        Client::new(pin, &root, now).map_err(|e| anyhow!("the first root does not match the pin: {e}"))?
                    }
                })
            }
        };
        let health = match read_opt(&dir.join("health.json"))? {
            Some(b) => serde_json::from_slice(&b).context("health.json")?,
            None => BootHealth::default(),
        };
        let active = match read_opt(&dir.join("active.json"))? {
            Some(b) => serde_json::from_slice(&b).context("active.json")?,
            None => Active::default(),
        };
        Ok(Self { dir, client, health, active })
    }

    /// Pins the org root: `pin` (hex fingerprint) and the first signed root. Refuses a root that
    /// does not match the pin, and refuses to change a pin already set.
    pub fn trust(&mut self, pin_hex: &str, root: &[u8], now: Timestamp) -> Result<()> {
        if self.client.is_some() {
            bail!("an update root is already pinned; remove {} to start over", self.dir.join("trust.json").display());
        }
        let pin = RootPin::from_hex(pin_hex).map_err(|e| anyhow!("the pin: {e}"))?;
        let env = Signed::from_bytes(root).map_err(|e| anyhow!("the root: {e}"))?;
        let client = Client::new(pin, &env, now).map_err(|e| anyhow!("that root does not match the pin: {e}"))?;
        let t = Trust { pin: pin.to_string(), root: root.to_vec() };
        write_atomic(&self.dir.join("trust.json"), &serde_json::to_vec(&t)?)?;
        self.client = Some(client);
        self.save_client()
    }

    fn save_client(&self) -> Result<()> {
        if let Some(c) = &self.client {
            write_atomic(&self.dir.join("client.state"), &c.state().to_bytes().map_err(|e| anyhow!("{e}"))?)?;
        }
        Ok(())
    }

    fn save_health(&self) -> Result<()> {
        write_atomic(&self.dir.join("health.json"), &serde_json::to_vec(&self.health)?)
    }

    fn save_active(&self) -> Result<()> {
        write_atomic(&self.dir.join("active.json"), &serde_json::to_vec(&self.active)?)
    }

    fn release(&self, id: &str) -> PathBuf {
        self.dir.join("releases").join(id)
    }

    pub fn status(&self) -> Status {
        Status {
            trusted: self.client.is_some(),
            good: self.active.good.clone(),
            health: self.health.state.clone(),
            bad: self.client.as_ref().map(|c| c.state().bad.iter().cloned().collect()).unwrap_or_default(),
        }
    }

    /// Verifies the bundle in `bundle`, checks its files and stages the release for the next start.
    /// `running` is this build's version (a Full update may not go below it).
    pub fn stage(&mut self, bundle: &Path, running: &Version, now: Timestamp) -> Result<VerifiedUpdate> {
        self.not_busy()?;
        let client = self.client.as_mut().ok_or_else(|| anyhow!("no update root is pinned (`ymirctl update trust`)"))?;
        let mut roots = Vec::new();
        if let Ok(rd) = std::fs::read_dir(bundle.join("roots")) {
            let mut names: Vec<_> = rd.filter_map(|e| e.ok()).map(|e| e.path()).collect();
            names.sort();
            for p in names {
                roots.push(Signed::from_bytes(&std::fs::read(&p)?).map_err(|e| anyhow!("{}: {e}", p.display()))?);
            }
        }
        let manifest =
            Signed::from_bytes(&std::fs::read(bundle.join("manifest.env")).context("manifest.env")?).map_err(|e| anyhow!("manifest.env: {e}"))?;
        let freshness = match read_opt(&bundle.join("freshness.env"))? {
            Some(b) => Some(Signed::from_bytes(&b).map_err(|e| anyhow!("freshness.env: {e}"))?),
            None => None,
        };
        let opts = VerifyOptions::new(now, running.clone());
        let verified = client
            .verify(APP, CHANNEL, &UpdateBundle { roots, manifest: Some(manifest), freshness }, &opts)
            .map_err(|e| anyhow!("the update is refused: {e}"));
        // The accepted seq and any root rotation are recorded even if what follows fails.
        let saved = self.save_client();
        let verified = verified?;
        saved?;

        let m = &verified.manifest;
        check_package(m)?;
        // Copy into a scratch directory, checking each file against the signed size and hash, then
        // move it into place in one step.
        let id = &m.update_id;
        let scratch = self.dir.join("releases").join(format!(".{id}.partial"));
        let _ = std::fs::remove_dir_all(&scratch);
        std::fs::create_dir_all(&scratch)?;
        for f in &m.files {
            let src = bundle.join(&f.path);
            let meta = std::fs::symlink_metadata(&src).with_context(|| f.path.clone())?;
            if !meta.is_file() || meta.len() != f.size {
                let _ = std::fs::remove_dir_all(&scratch);
                bail!("{}: not a regular file of the signed size", f.path);
            }
            let data = std::fs::read(&src)?;
            if let Err(e) = f.verify(&data) {
                let _ = std::fs::remove_dir_all(&scratch);
                bail!("{}: {e}", f.path);
            }
            let dst = scratch.join(&f.path);
            if let Some(parent) = dst.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&dst, &data)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mode = if executable(&f.path, &data, meta.permissions().mode()) { 0o755 } else { 0o644 };
                std::fs::set_permissions(&dst, std::fs::Permissions::from_mode(mode))?;
            }
        }
        let target = self.release(id);
        let _ = std::fs::remove_dir_all(&target);
        std::fs::rename(&scratch, &target)?;
        self.active.pending = Some(Pending { id: id.clone(), manifest: m.clone(), hash: verified.manifest_hash.to_hex() });
        self.health.stage(id);
        self.save_active()?;
        self.save_health()?;
        Ok(verified)
    }

    fn not_busy(&self) -> Result<()> {
        if matches!(self.health.state, HealthState::Staged { .. } | HealthState::Trial { .. }) {
            bail!(
                "an update is already {}; wait for it to commit or roll back",
                if matches!(self.health.state, HealthState::Staged { .. }) { "staged" } else { "on trial" }
            );
        }
        Ok(())
    }

    /// Checks a signed announcement (`Signed::to_bytes`) against the pinned root: signed by
    /// release keys delegated for Ymir's app and channel, current at `now`. `None` when it names
    /// nothing newer than what was already taken.
    pub fn check_announcement(&self, bytes: &[u8], now: Timestamp) -> Result<Option<Announcement>> {
        let client = self.client.as_ref().ok_or_else(|| anyhow!("no update root is pinned (`ymirctl update trust`)"))?;
        let env = Signed::from_bytes(bytes).map_err(|e| anyhow!("the announcement: {e}"))?;
        let a = client.verify_announcement(&env, now).map_err(|e| anyhow!("the announcement is refused: {e}"))?;
        if let Some(a) = &a
            && (a.app != APP || a.channel != CHANNEL)
        {
            bail!("the announcement is for {} on {}, not {APP} on {CHANNEL}", a.app, a.channel);
        }
        Ok(a)
    }

    /// Downloads a bundle from `src` and stages it ([`Updater::stage`]). The manifest is verified
    /// first (without recording anything), so nothing it does not sign is downloaded, and each
    /// file is read up to its signed size. With `announced`, the manifest must be the one an
    /// announcement named (its payload hash).
    pub fn fetch(&mut self, src: &dyn Source, announced: Option<&Hash32>, running: &Version, now: Timestamp) -> Result<Fetched> {
        self.not_busy()?;
        let client = self.client.as_ref().ok_or_else(|| anyhow!("no update root is pinned (`ymirctl update trust`)"))?;
        let what = src.describe();
        let manifest_bytes = read_capped(src, "manifest.env", MAX_ENVELOPE)?.ok_or_else(|| anyhow!("{what} has no manifest.env"))?;
        let manifest = Signed::from_bytes(&manifest_bytes).map_err(|e| anyhow!("manifest.env from {what}: {e}"))?;
        if let Some(h) = announced
            && manifest.payload_hash() != *h
        {
            bail!("{what} serves another manifest than the announced one ({} instead of {})", manifest.payload_hash().to_hex(), h.to_hex());
        }
        let mut roots = Vec::new();
        for n in 1..=MAX_ROOTS {
            let Some(b) = read_capped(src, &format!("roots/{n}.env"), MAX_ENVELOPE)? else { break };
            roots.push(b);
        }
        let freshness = read_capped(src, "freshness.env", MAX_ENVELOPE)?;
        let parse = |b: &[u8], name: &str| Signed::from_bytes(b).map_err(|e| anyhow!("{name} from {what}: {e}"));
        let bundle = UpdateBundle {
            roots: roots.iter().enumerate().map(|(i, b)| parse(b, &format!("roots/{}.env", i + 1))).collect::<Result<_>>()?,
            manifest: Some(manifest),
            freshness: freshness.as_deref().map(|b| parse(b, "freshness.env")).transpose()?,
        };
        let mut probe = Client::from_state(client.pin(), client.state().clone(), now).map_err(|e| anyhow!("{e}"))?;
        let verified =
            probe.verify(APP, CHANNEL, &bundle, &VerifyOptions::new(now, running.clone())).map_err(|e| anyhow!("the update is refused: {e}"))?;
        let m = &verified.manifest;
        check_package(m)?;

        let scratch = self.dir.join("downloads").join(&m.update_id);
        let _ = std::fs::remove_dir_all(&scratch);
        let result = (|| {
            std::fs::create_dir_all(scratch.join("roots"))?;
            std::fs::write(scratch.join("manifest.env"), &manifest_bytes)?;
            for (i, b) in roots.iter().enumerate() {
                std::fs::write(scratch.join("roots").join(format!("{:04}.env", i + 1)), b)?;
            }
            if let Some(b) = &freshness {
                std::fs::write(scratch.join("freshness.env"), b)?;
            }
            let mut bytes = 0u64;
            for f in &m.files {
                let mut r = src.open(&f.path)?.ok_or_else(|| anyhow!("{what} has no {}", f.path))?;
                let dst = scratch.join(&f.path);
                if let Some(parent) = dst.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let mut out = std::fs::File::create(&dst)?;
                let n = std::io::copy(&mut (&mut r).take(f.size + 1), &mut out).with_context(|| format!("downloading {}", f.path))?;
                if n != f.size {
                    bail!("{}: {what} sent {}{n} bytes, and the manifest signs {}", f.path, if n > f.size { "more than " } else { "" }, f.size);
                }
                bytes += n;
            }
            let staged = self.stage(&scratch, running, now)?;
            Ok(Fetched { update: staged, files: m.files.len(), bytes })
        })();
        let _ = std::fs::remove_dir_all(&scratch);
        result
    }

    /// Call at the very start of each launch; the state is saved before the answer is acted on.
    pub fn boot(&mut self) -> Result<Boot> {
        let choice = self.health.begin_boot();
        self.save_health()?;
        let good = |s: &Self| s.active.good.as_ref().map(|id| (id.clone(), s.release(id).join(DAEMON_FILE)));
        Ok(match choice {
            BootChoice::Current => match good(self) {
                Some((id, path)) if path.is_file() => Boot::Run { id, path, trial: false },
                _ => Boot::Installed,
            },
            BootChoice::Candidate(id) => Boot::Run { path: self.release(&id).join(DAEMON_FILE), id, trial: true },
            BootChoice::RollBack(failed) => {
                if let Some(c) = self.client.as_mut() {
                    c.mark_bad(&failed);
                }
                self.save_client()?;
                self.active.pending = None;
                self.save_active()?;
                let _ = std::fs::remove_dir_all(self.release(&failed));
                Boot::RolledBack { failed, run: good(self).filter(|(_, p)| p.is_file()) }
            }
        })
    }

    /// The daemon is up (first control request answerable): a trial release becomes the good one.
    pub fn ready(&mut self) -> Result<Option<String>> {
        let Some(id) = self.health.mark_ready() else { return Ok(None) };
        self.save_health()?;
        if let Some(p) = self.active.pending.take().filter(|p| p.id == id) {
            let verified = VerifiedUpdate {
                manifest: p.manifest,
                manifest_hash: weirdnet_update::Hash32::from_hex(&p.hash).map_err(|e| anyhow!("{e}"))?,
                freshness: None,
            };
            if let Some(c) = self.client.as_mut() {
                c.mark_installed(&verified);
            }
            self.save_client()?;
            let previous = self.active.good.replace(id.clone());
            self.save_active()?;
            if let Some(old) = previous.filter(|o| *o != id) {
                let _ = std::fs::remove_dir_all(self.release(&old));
            }
        }
        Ok(Some(id))
    }
}

/// Called by `ymird` once it is serving: commits a release on trial. Does nothing when no update
/// is on trial or the updater has no state; a failure is logged, not fatal.
pub fn commit_if_on_trial(paths: &Paths) {
    if !dir(paths).join("health.json").exists() {
        return;
    }
    match Updater::open(paths, Timestamp::now()).and_then(|mut u| u.ready()) {
        Ok(Some(id)) => tracing::info!(update = %id, "update committed after a healthy start"),
        Ok(None) => {}
        Err(e) => tracing::warn!("update commit: {e:#}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use weirdnet_update::envelope::Domain;
    use weirdnet_update::manifest::{FullPackage, Labels, PackageFormat};
    use weirdnet_update::{Delegation, FileEntry, Root, SigningKey, UpdateManifest};

    const NOW: Timestamp = Timestamp(1_800_000_000);

    fn key(n: u8) -> SigningKey {
        SigningKey::from_seed(&[n; 32])
    }

    struct Org {
        release: SigningKey,
        root_bytes: Vec<u8>,
        pin: String,
    }

    fn org() -> Org {
        let (r1, r2, release) = (key(1), key(2), key(10));
        let root = Root {
            version: 1,
            org: "weirdbrains".into(),
            keys: vec![r1.public(), r2.public()],
            threshold: 2,
            release: vec![Delegation::any(release.public())],
            release_threshold: 1,
            timestamp: vec![],
            timestamp_threshold: 0,
            expires: None,
        };
        let env = Signed::sign(Domain::Root, &root, &[&r1, &r2]).unwrap();
        Org { release, root_bytes: env.to_bytes().unwrap(), pin: root.pin().to_string() }
    }

    fn v(s: &str) -> Version {
        Version::parse(s).unwrap()
    }

    /// Writes a bundle for `version` into `dir`: a `ymird` with `body`, signed by `signer`.
    fn bundle(dir: &Path, signer: &SigningKey, id: &str, seq: u64, version: &str, body: &[u8], platform: Platform) {
        bundle_with(dir, signer, id, seq, version, &[("ymird", body)], platform);
    }

    /// [`bundle`] with these files (each written read-only, as a tarball unpacked by a tool that
    /// drops modes would leave them).
    fn bundle_with(dir: &Path, signer: &SigningKey, id: &str, seq: u64, version: &str, files: &[(&str, &[u8])], platform: Platform) {
        for (name, body) in files {
            let p = dir.join(name);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, body).unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o644)).unwrap();
            }
        }
        let (_, arch) = this_target().unwrap();
        let m = UpdateManifest {
            app: APP.into(),
            channel: CHANNEL.into(),
            update_id: id.into(),
            seq,
            kind: UpdateKind::Full(FullPackage { platform, arch, format: PackageFormat::TarGz, signing_identity: "unsigned".into() }),
            version: v(version),
            base_releases: vec![],
            files: files.iter().map(|(name, body)| FileEntry::describe(name, body)).collect(),
            min_client: None,
            max_client: None,
            capabilities: vec![],
            labels: Labels::default(),
            targeting: None,
            rollout: None,
            rollback_of: None,
            notes: "test".into(),
            created: NOW,
            expires: NOW.plus(86_400),
        };
        let env = Signed::sign(Domain::Manifest, &m, &[signer]).unwrap();
        std::fs::write(dir.join("manifest.env"), env.to_bytes().unwrap()).unwrap();
    }

    fn open(tmp: &Path, o: &Org) -> Updater {
        let paths = Paths::new(tmp.join("home"));
        let mut u = Updater::open(&paths, NOW).unwrap();
        u.trust(&o.pin, &o.root_bytes, NOW).unwrap();
        u
    }

    fn here() -> Platform {
        this_target().unwrap().0
    }

    #[cfg(unix)]
    #[test]
    fn bundled_programs_are_installed_executable_and_data_is_not() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let o = org();
        let mut u = open(tmp.path(), &o);
        let elf: &[u8] = b"\x7fELF\x02\x01\x01 a server";
        let files: &[(&str, &[u8])] = &[
            ("ymird", b"daemon"),
            ("ymirctl", b"ctl"),
            ("llama/llama-server", b"server"),
            ("llama/rpc-server", b"rpc"),
            ("llama/llama-quantize", elf),
            ("llama/libggml.so", elf),
            ("scripts/engine.py", b"#!/usr/bin/env python3\n"),
            ("share/README.md", b"docs"),
        ];
        bundle_with(&tmp.path().join("b1"), &o.release, "u1", 1, "0.2.0", files, here());
        u.stage(&tmp.path().join("b1"), &v("0.1.0"), NOW).unwrap();
        let release = dir(&Paths::new(tmp.path().join("home"))).join("releases/u1");
        let mode = |p: &str| std::fs::metadata(release.join(p)).unwrap().permissions().mode() & 0o777;
        for p in ["ymird", "ymirctl", "llama/llama-server", "llama/rpc-server", "llama/llama-quantize", "llama/libggml.so", "scripts/engine.py"] {
            assert_eq!(mode(p), 0o755, "{p}");
        }
        assert_eq!(mode("share/README.md"), 0o644);
        // A file that was executable in the bundle stays so, whatever its name.
        assert!(executable("tools/thing", b"data", 0o755));
        assert!(!executable("tools/thing", b"data", 0o644));
    }

    #[test]
    fn a_signed_update_is_staged_tried_and_committed() {
        let tmp = tempfile::tempdir().unwrap();
        let o = org();
        let mut u = open(tmp.path(), &o);
        bundle(&tmp.path().join("b1"), &o.release, "u1", 1, "0.2.0", b"daemon one", here());
        u.stage(&tmp.path().join("b1"), &v("0.1.0"), NOW).unwrap();
        assert!(matches!(u.status().health, HealthState::Staged { .. }));

        // The next start runs it on trial; reaching ready commits it, and a restart keeps it.
        let Boot::Run { id, path, trial: true } = u.boot().unwrap() else { panic!("wanted a trial") };
        assert_eq!(id, "u1");
        assert_eq!(std::fs::read(&path).unwrap(), b"daemon one");
        assert_eq!(u.ready().unwrap().as_deref(), Some("u1"));
        let mut again = Updater::open(&Paths::new(tmp.path().join("home")), NOW).unwrap();
        assert!(matches!(again.boot().unwrap(), Boot::Run { trial: false, .. }));
        assert_eq!(again.status().good.as_deref(), Some("u1"));
        // Going back to an older version is a downgrade, refused (a Full update may not go below the running build).
        bundle(&tmp.path().join("b0"), &o.release, "u0", 2, "0.0.9", b"old", here());
        assert!(again.stage(&tmp.path().join("b0"), &v("0.2.0"), NOW).unwrap_err().to_string().contains("refused"));
    }

    #[test]
    fn a_release_that_never_gets_ready_is_rolled_back_and_stays_out() {
        let tmp = tempfile::tempdir().unwrap();
        let o = org();
        let mut u = open(tmp.path(), &o);
        // A good release first.
        bundle(&tmp.path().join("b1"), &o.release, "u1", 1, "0.2.0", b"good", here());
        u.stage(&tmp.path().join("b1"), &v("0.1.0"), NOW).unwrap();
        u.boot().unwrap();
        u.ready().unwrap();
        // A bad one: two starts that never reach ready, then it is set aside.
        bundle(&tmp.path().join("b2"), &o.release, "u2", 2, "0.3.0", b"broken", here());
        u.stage(&tmp.path().join("b2"), &v("0.2.0"), NOW).unwrap();
        assert!(matches!(u.boot().unwrap(), Boot::Run { trial: true, .. }));
        assert!(matches!(u.boot().unwrap(), Boot::Run { trial: true, .. }));
        let Boot::RolledBack { failed, run } = u.boot().unwrap() else { panic!("wanted a rollback") };
        assert_eq!(failed, "u2");
        assert_eq!(std::fs::read(run.unwrap().1).unwrap(), b"good", "the previous release runs again");
        assert_eq!(u.status().bad, ["u2"]);
        assert!(matches!(u.boot().unwrap(), Boot::Run { trial: false, .. }));
        // It cannot come back, even re-signed under the same id.
        let e = u.stage(&tmp.path().join("b2"), &v("0.2.0"), NOW).unwrap_err().to_string();
        assert!(e.contains("refused"), "{e}");
    }

    #[test]
    fn what_is_not_signed_by_the_pinned_org_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let o = org();
        let mut u = open(tmp.path(), &o);
        let b = tmp.path().join("b");
        let refused = |u: &mut Updater, b: &Path| u.stage(b, &v("0.1.0"), NOW).unwrap_err().to_string();

        // Signed by a key the root does not delegate.
        bundle(&b, &key(99), "x1", 1, "0.2.0", b"evil", here());
        assert!(refused(&mut u, &b).contains("refused"));
        // Signed properly, but a file changed after signing.
        bundle(&b, &o.release, "x2", 2, "0.2.0", b"daemon", here());
        std::fs::write(b.join("ymird"), b"daemon!").unwrap();
        let e = refused(&mut u, &b);
        assert!(e.contains("ymird"), "{e}");
        assert!(!dir(&Paths::new(tmp.path().join("home"))).join("releases/x2").exists(), "nothing half-installed");
        // A package for another OS.
        let other = if here() == Platform::Linux { Platform::Windows } else { Platform::Linux };
        bundle(&b, &o.release, "x3", 3, "0.2.0", b"daemon", other);
        assert!(refused(&mut u, &b).contains("not this machine"));
        // An older seq than one already accepted is a replay.
        bundle(&b, &o.release, "x4", 5, "0.2.0", b"daemon", here());
        u.stage(&b, &v("0.1.0"), NOW).unwrap();
        u.boot().unwrap();
        u.ready().unwrap();
        bundle(&b, &o.release, "x5", 4, "0.2.1", b"daemon", here());
        assert!(refused(&mut u, &b).contains("refused"));
        // A different root than the pinned one cannot be trusted.
        let mut fresh = Updater::open(&Paths::new(tmp.path().join("other")), NOW).unwrap();
        let rogue = org_with(key(7), key(8));
        assert!(fresh.trust(&o.pin, &rogue, NOW).is_err());
    }

    /// A root made by other keys, as bytes.
    fn org_with(a: SigningKey, b: SigningKey) -> Vec<u8> {
        let root = Root {
            version: 1,
            org: "weirdbrains".into(),
            keys: vec![a.public(), b.public()],
            threshold: 2,
            release: vec![Delegation::any(key(10).public())],
            release_threshold: 1,
            timestamp: vec![],
            timestamp_threshold: 0,
            expires: None,
        };
        Signed::sign(Domain::Root, &root, &[&a, &b]).unwrap().to_bytes().unwrap()
    }

    #[test]
    fn without_a_pinned_root_nothing_is_taken() {
        let tmp = tempfile::tempdir().unwrap();
        let mut u = Updater::open(&Paths::new(tmp.path().join("home")), NOW).unwrap();
        let e = u.stage(&tmp.path().join("nothing"), &v("0.1.0"), NOW).unwrap_err().to_string();
        assert!(e.contains("no update root"), "{e}");
        assert!(!u.status().trusted);
    }

    /// A source that counts what was opened, and can serve too much of one file.
    struct Counting {
        dir: PathBuf,
        opened: std::cell::RefCell<Vec<String>>,
        pad: Option<&'static str>,
    }

    impl Source for Counting {
        fn open(&self, path: &str) -> Result<Option<Box<dyn std::io::Read + '_>>> {
            self.opened.borrow_mut().push(path.to_string());
            let Ok(mut b) = std::fs::read(self.dir.join(path)) else { return Ok(None) };
            if self.pad == Some(path) {
                b.extend_from_slice(b"and more");
            }
            Ok(Some(Box::new(std::io::Cursor::new(b))))
        }
        fn describe(&self) -> String {
            "the test source".into()
        }
    }

    fn counting(dir: PathBuf) -> Counting {
        Counting { dir, opened: Default::default(), pad: None }
    }

    #[test]
    fn a_fetched_bundle_is_verified_first_then_downloaded_and_staged() {
        let tmp = tempfile::tempdir().unwrap();
        let o = org();
        let mut u = open(tmp.path(), &o);
        let files: &[(&str, &[u8])] = &[("ymird", b"daemon two"), ("lib/ymir/llama.cpp/llama-server", b"server")];
        bundle_with(&tmp.path().join("b1"), &o.release, "u1", 1, "0.2.0", files, here());
        let src = counting(tmp.path().join("b1"));
        let f = u.fetch(&src, None, &v("0.1.0"), NOW).unwrap();
        assert_eq!((f.update.manifest.update_id.as_str(), f.files, f.bytes), ("u1", 2, 16));
        assert_eq!(*src.opened.borrow(), ["manifest.env", "roots/1.env", "freshness.env", "ymird", "lib/ymir/llama.cpp/llama-server"]);
        assert!(matches!(u.status().health, HealthState::Staged { .. }));
        let Boot::Run { path, .. } = u.boot().unwrap() else { panic!("wanted a trial") };
        assert_eq!(std::fs::read(path).unwrap(), b"daemon two");
        assert!(!dir(&Paths::new(tmp.path().join("home"))).join("downloads/u1").exists(), "the download directory is removed");
        // Staged or on trial: no second fetch.
        assert!(u.fetch(&src, None, &v("0.1.0"), NOW).unwrap_err().to_string().contains("on trial"));
    }

    #[test]
    fn a_fetch_downloads_nothing_unsigned_or_unannounced_or_oversized() {
        let tmp = tempfile::tempdir().unwrap();
        let o = org();
        let mut u = open(tmp.path(), &o);
        // Signed by a key the root does not delegate: refused after the manifest alone.
        bundle(&tmp.path().join("bad"), &key(99), "u1", 1, "0.2.0", b"evil", here());
        let src = counting(tmp.path().join("bad"));
        assert!(u.fetch(&src, None, &v("0.1.0"), NOW).unwrap_err().to_string().contains("refused"));
        assert!(!src.opened.borrow().iter().any(|p| p == "ymird"), "{:?}", src.opened.borrow());
        // Nothing was recorded by the refused attempt: the good one still goes through.
        bundle(&tmp.path().join("b1"), &o.release, "u1", 1, "0.2.0", b"daemon", here());
        // Another manifest than the announced one: refused before anything else is read.
        let src = counting(tmp.path().join("b1"));
        let e = u.fetch(&src, Some(&Hash32([7; 32])), &v("0.1.0"), NOW).unwrap_err().to_string();
        assert!(e.contains("another manifest than the announced one"), "{e}");
        assert_eq!(*src.opened.borrow(), ["manifest.env"]);
        // More bytes than signed: refused, nothing staged.
        let src = Counting { pad: Some("ymird"), ..counting(tmp.path().join("b1")) };
        let e = u.fetch(&src, None, &v("0.1.0"), NOW).unwrap_err().to_string();
        assert!(e.contains("sent more than 7 bytes, and the manifest signs 6"), "{e}");
        assert!(matches!(u.status().health, HealthState::Idle));
        // The real one.
        u.fetch(&counting(tmp.path().join("b1")), None, &v("0.1.0"), NOW).unwrap();
    }

    #[test]
    fn announcements_are_checked_against_the_pinned_root() {
        use weirdnet_update::Announcement;
        let tmp = tempfile::tempdir().unwrap();
        let o = org();
        let mut u = open(tmp.path(), &o);
        bundle(&tmp.path().join("b1"), &o.release, "u1", 1, "0.2.0", b"daemon", here());
        let manifest = Signed::from_bytes(&std::fs::read(tmp.path().join("b1/manifest.env")).unwrap()).unwrap();
        let pin = RootPin::from_hex(&o.pin).unwrap();
        let ann = |m: &Signed, signer: &SigningKey| {
            let a = Announcement::for_manifest("weirdbrains", pin, m, NOW, 3600, vec!["https://updates.example/ymir/u1".into()]).unwrap();
            a.sign(&[signer]).unwrap().to_bytes().unwrap()
        };
        let a = u.check_announcement(&ann(&manifest, &o.release), NOW).unwrap().expect("news");
        assert_eq!((a.update_id.as_str(), a.seq, a.sources.len()), ("u1", 1, 1));
        assert!(u.check_announcement(&ann(&manifest, &key(99)), NOW).unwrap_err().to_string().contains("refused"));
        assert!(u.check_announcement(&ann(&manifest, &o.release), NOW.plus(7200)).is_err(), "expired");
        // The announced manifest is fetched; then the same announcement is old news.
        u.fetch(&DirSource(tmp.path().join("b1")), Some(&a.manifest_hash), &v("0.1.0"), NOW).unwrap();
        assert!(u.check_announcement(&ann(&manifest, &o.release), NOW).unwrap().is_none());
    }
}
