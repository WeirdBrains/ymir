//! Hands-off self-update: `ymird` checks a feed on a schedule, and stages and starts new releases
//! itself (`[update]` in node.toml, on by default).
//!
//! The feed is a GitHub repository's latest release (`github:OWNER/REPO`, the default
//! `github:WeirdBrains/ymir`) or any `https://` URL that answers the same JSON (`tag_name`,
//! `assets[].name`, `browser_download_url`, `size`). Release assets can't hold the bundle's
//! `llama.cpp/` paths, so a release carries each bundle as one `.update.tar.gz`
//! (`scripts/weirdnet_release.py updates`): it is downloaded whole, unpacked into
//! `updates/incoming/`, and handed to [`Updater::stage`], which checks it against the pinned root
//! like any other bundle (signatures, scope, expiry, anti-rollback, no downgrade, OS and CPU, each
//! file's size and BLAKE3). Nothing from the feed is trusted: a feed can only offer bytes that the
//! org's release key signed.
//!
//! After staging, a daemon started by `ymirctl update boot` under a service manager (systemd,
//! launchd) shuts down cleanly and exits with [`RESTART_EXIT_CODE`]; the service restarts it, and
//! `update boot` runs the staged release on trial. A trial that isn't serving within
//! [`TRIAL_TIMEOUT`] exits with an error (see [`trial_watchdog`]), and after two failed starts the
//! previous good release comes back and the update is marked bad for good. Elsewhere (a daemon in
//! a terminal, or one not started through `update boot`) the update is only staged and logged.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use semver::Version;
use serde::{Deserialize, Serialize};
use weirdnet_update::Timestamp;
use weirdnet_update::health::HealthState;

use crate::Paths;
use crate::update::{self, Updater};
use crate::update_publish;

/// `ymird` exits with this after staging an update, so the service manager restarts it
/// (systemd `Restart=on-failure` counts it as a failure; launchd `KeepAlive` restarts anyway).
pub const RESTART_EXIT_CODE: i32 = 75;
/// A release on trial that isn't serving by then exits, so the next start can roll it back.
pub const TRIAL_TIMEOUT: Duration = Duration::from_secs(15 * 60);
/// The largest bundle a feed may send.
const MAX_BUNDLE: u64 = 4 << 30;
const DEFAULT_FEED: &str = "github:WeirdBrains/ymir";

/// `[update]` in node.toml.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UpdateSection {
    /// Check the feed and take new releases without being asked. Needs a pinned update root
    /// (`ymirctl update trust`); without one nothing is checked.
    #[serde(default = "yes")]
    pub auto: bool,
    /// The release channel; only `stable` is published.
    #[serde(default = "stable")]
    pub channel: String,
    /// How often to check, such as `6h`, `30m` or `1d` (at least `5m`).
    #[serde(default = "six_hours")]
    pub interval: String,
    /// `github:OWNER/REPO`, or an `https://` URL answering a GitHub-style latest-release JSON.
    #[serde(default = "default_feed")]
    pub feed: String,
    /// Which build to take: `auto` (the one running now), `plain` or `cuda`.
    #[serde(default = "auto")]
    pub variant: String,
    /// Restart into a staged update (under a service manager). Off: stage only.
    #[serde(default = "yes")]
    pub restart: bool,
}

fn yes() -> bool {
    true
}
fn stable() -> String {
    update::CHANNEL.into()
}
fn six_hours() -> String {
    "6h".into()
}
fn default_feed() -> String {
    DEFAULT_FEED.into()
}
fn auto() -> String {
    "auto".into()
}

impl Default for UpdateSection {
    fn default() -> Self {
        Self { auto: true, channel: stable(), interval: six_hours(), feed: default_feed(), variant: auto(), restart: true }
    }
}

impl UpdateSection {
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// Checks every setting; node.toml loading calls it.
    pub fn validate(&self) -> Result<()> {
        if self.channel != update::CHANNEL {
            bail!("[update] channel {:?}: only {:?} is published", self.channel, update::CHANNEL);
        }
        self.interval()?;
        Feed::parse(&self.feed)?;
        if !matches!(self.variant.as_str(), "auto" | "plain" | "cuda") {
            bail!("[update] variant {:?}: auto, plain or cuda", self.variant);
        }
        Ok(())
    }

    pub fn interval(&self) -> Result<Duration> {
        let d = parse_duration(&self.interval).with_context(|| format!("[update] interval {:?}", self.interval))?;
        if d < Duration::from_secs(300) {
            bail!("[update] interval {:?} is below 5m", self.interval);
        }
        Ok(d)
    }
}

/// `90s`, `30m`, `6h`, `1d`, or plain seconds.
pub fn parse_duration(s: &str) -> Result<Duration> {
    let s = s.trim();
    let (num, mult) = match s.char_indices().last() {
        Some((i, 's')) => (&s[..i], 1),
        Some((i, 'm')) => (&s[..i], 60),
        Some((i, 'h')) => (&s[..i], 3600),
        Some((i, 'd')) => (&s[..i], 86_400),
        _ => (s, 1),
    };
    let n: u64 = num.trim().parse().map_err(|_| anyhow!("not a duration (such as 30m, 6h, 1d)"))?;
    Ok(Duration::from_secs(n.checked_mul(mult).ok_or_else(|| anyhow!("too long"))?))
}

/// `http://` to this machine only (a local mirror, or a test).
fn loopback(s: &str) -> bool {
    ["http://127.0.0.1", "http://localhost", "http://[::1]"]
        .iter()
        .any(|p| s.strip_prefix(p).is_some_and(|rest| rest.is_empty() || rest.starts_with(':') || rest.starts_with('/')))
}

/// Where releases are announced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Feed {
    GitHub { owner: String, repo: String },
    Url(String),
}

impl Feed {
    pub fn parse(s: &str) -> Result<Self> {
        if let Some(rest) = s.strip_prefix("github:") {
            let (owner, repo) = rest.split_once('/').ok_or_else(|| anyhow!("[update] feed {s:?}: github:OWNER/REPO"))?;
            let ok = |x: &str| !x.is_empty() && x.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c));
            if !ok(owner) || !ok(repo) {
                bail!("[update] feed {s:?}: github:OWNER/REPO");
            }
            return Ok(Feed::GitHub { owner: owner.into(), repo: repo.into() });
        }
        if s.starts_with("https://") || loopback(s) {
            return Ok(Feed::Url(s.into()));
        }
        bail!("[update] feed {s:?}: github:OWNER/REPO or an https:// URL")
    }

    /// The URL of the latest-release JSON.
    pub fn latest_url(&self) -> String {
        match self {
            Feed::GitHub { owner, repo } => format!("https://api.github.com/repos/{owner}/{repo}/releases/latest"),
            Feed::Url(u) => u.clone(),
        }
    }
}

/// A release as the feed describes it (GitHub's release JSON, the fields used here).
#[derive(Debug, Clone, Deserialize)]
pub struct FeedRelease {
    pub tag_name: String,
    #[serde(default)]
    pub draft: bool,
    #[serde(default)]
    pub prerelease: bool,
    #[serde(default)]
    pub assets: Vec<FeedAsset>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FeedAsset {
    pub name: String,
    pub browser_download_url: String,
    #[serde(default)]
    pub size: u64,
}

/// The bundle asset for this machine: `ymir[-cuda]-VERSION-OS-ARCH.update.tar.gz`.
pub fn asset_name(cuda: bool, version: &Version) -> Option<String> {
    let (os, arch) = update::this_target()?;
    let arch = match arch {
        weirdnet_update::manifest::Arch::X86_64 => "x86_64",
        weirdnet_update::manifest::Arch::Aarch64 => "aarch64",
    };
    Some(format!("ymir{}-{version}-{}-{arch}.update.tar.gz", if cuda { "-cuda" } else { "" }, os.as_str()))
}

/// Whether the running build is the CUDA one: its `llama.cpp/engine.toml` (next to a release's
/// `ymird`, or under `../lib/ymir/` for an installed one) names a CUDA variant or backend.
pub fn running_cuda(exe: &Path) -> bool {
    let Some(dir) = exe.parent() else { return false };
    for p in [dir.join("llama.cpp/engine.toml"), dir.join("../lib/ymir/llama.cpp/engine.toml")] {
        if let Ok(text) = std::fs::read_to_string(&p) {
            return text.lines().any(|l| {
                let l = l.trim();
                (l.starts_with("variant") || l.starts_with("backends")) && l.contains("cuda")
            });
        }
    }
    false
}

/// What one check did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// The feed's latest release is not newer than the running one.
    UpToDate { latest: Version },
    /// Nothing done, and why (no root pinned, an update already staged or on trial, the release
    /// was rolled back before, no bundle for this machine).
    Skipped(String),
    /// A new release is staged for the next start.
    Staged { update_id: String, version: Version },
}

/// Reads the feed, and stages a newer release's bundle for this machine.
pub async fn check_once(paths: &Paths, cfg: &UpdateSection, running: &Version, exe: &Path, http: &reqwest::Client) -> Result<Outcome> {
    {
        let u = Updater::open(paths, Timestamp::now())?;
        let s = u.status();
        if !s.trusted {
            return Ok(Outcome::Skipped("no update root is pinned (`ymirctl update trust`)".into()));
        }
        if let HealthState::Staged { update_id } | HealthState::Trial { update_id, .. } = &s.health {
            return Ok(Outcome::Skipped(format!("{update_id} is already staged or on trial")));
        }
    }
    let url = Feed::parse(&cfg.feed)?.latest_url();
    let resp = http.get(&url).header("Accept", "application/vnd.github+json").send().await.with_context(|| format!("asking {url}"))?;
    if !resp.status().is_success() {
        bail!("{url}: the feed answered {}", resp.status());
    }
    let rel: FeedRelease = resp.json().await.with_context(|| format!("{url}: not a release JSON"))?;
    if rel.draft || rel.prerelease {
        return Ok(Outcome::Skipped(format!("{} is a draft or pre-release", rel.tag_name)));
    }
    let latest = Version::parse(rel.tag_name.trim_start_matches('v')).with_context(|| format!("release tag {:?}", rel.tag_name))?;
    if latest <= *running {
        return Ok(Outcome::UpToDate { latest });
    }
    let id = update_publish::update_id(update_publish::seq_for(&latest)?);
    {
        let u = Updater::open(paths, Timestamp::now())?;
        if u.status().bad.contains(&id) {
            return Ok(Outcome::Skipped(format!("{id} ({latest}) was rolled back here before")));
        }
    }
    let cuda = match cfg.variant.as_str() {
        "cuda" => true,
        "plain" => false,
        _ => running_cuda(exe),
    };
    let Some(name) = asset_name(cuda, &latest) else { return Ok(Outcome::Skipped("no bundles are published for this OS or CPU".into())) };
    let Some(asset) = rel.assets.iter().find(|a| a.name == name) else {
        return Ok(Outcome::Skipped(format!("{} has no {name}", rel.tag_name)));
    };
    if asset.size > MAX_BUNDLE {
        bail!("{name}: {} bytes is more than a bundle may be", asset.size);
    }
    let incoming = update::dir(paths).join("incoming");
    let _ = std::fs::remove_dir_all(&incoming);
    std::fs::create_dir_all(&incoming)?;
    let result = async {
        let file = incoming.join(&name);
        download(http, &asset.browser_download_url, &file, asset.size).await?;
        let (paths, running, incoming) = (paths.clone(), running.clone(), incoming.clone());
        tokio::task::spawn_blocking(move || -> Result<Outcome> {
            let dir = unpack(&file, &incoming.join("bundle"))?;
            let mut u = Updater::open(&paths, Timestamp::now())?;
            let v = u.stage(&dir, &running, Timestamp::now())?;
            Ok(Outcome::Staged { update_id: v.manifest.update_id.clone(), version: v.manifest.version.clone() })
        })
        .await?
    }
    .await;
    let _ = std::fs::remove_dir_all(&incoming);
    result
}

async fn download(http: &reqwest::Client, url: &str, dest: &Path, size: u64) -> Result<()> {
    use futures::StreamExt;
    use tokio::io::AsyncWriteExt;
    let resp = http.get(url).send().await.with_context(|| format!("downloading {url}"))?;
    if !resp.status().is_success() {
        bail!("{url}: the server answered {}", resp.status());
    }
    let mut out = tokio::fs::File::create(dest).await?;
    let mut n = 0u64;
    let mut body = resp.bytes_stream();
    while let Some(chunk) = body.next().await {
        let chunk = chunk.with_context(|| format!("downloading {url}"))?;
        n += chunk.len() as u64;
        if n > MAX_BUNDLE || (size > 0 && n > size) {
            bail!("{url}: more bytes than the feed announced");
        }
        out.write_all(&chunk).await?;
    }
    out.flush().await?;
    if size > 0 && n != size {
        bail!("{url}: {n} bytes, the feed announced {size}");
    }
    Ok(())
}

/// Unpacks a bundle `.tar.gz` (one top directory of plain files) into `dest` and returns that
/// directory. Links, devices, absolute paths and `..` are refused.
pub fn unpack(tar_gz: &Path, dest: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(dest)?;
    let f = std::fs::File::open(tar_gz).with_context(|| tar_gz.display().to_string())?;
    let mut ar = tar::Archive::new(flate2::read::GzDecoder::new(std::io::BufReader::new(f)));
    ar.set_preserve_permissions(false);
    ar.set_overwrite(true);
    let mut top: Option<PathBuf> = None;
    for e in ar.entries().context("reading the bundle archive")? {
        let mut e = e.context("reading the bundle archive")?;
        let path = e.path()?.into_owned();
        if path.is_absolute() || path.components().any(|c| !matches!(c, std::path::Component::Normal(_) | std::path::Component::CurDir)) {
            bail!("the bundle archive has an unsafe path: {}", path.display());
        }
        match e.header().entry_type() {
            tar::EntryType::Regular | tar::EntryType::Directory => {}
            t => bail!("the bundle archive has a {t:?} entry ({}); only files and directories are taken", path.display()),
        }
        let first = path.components().find(|c| matches!(c, std::path::Component::Normal(_)));
        if let Some(std::path::Component::Normal(first)) = first {
            match &top {
                None => top = Some(PathBuf::from(first)),
                Some(t) if t.as_os_str() != first => bail!("the bundle archive has more than one top directory"),
                _ => {}
            }
        }
        if !e.unpack_in(dest)? {
            bail!("the bundle archive has an unsafe path: {}", path.display());
        }
    }
    let top = top.ok_or_else(|| anyhow!("the bundle archive is empty"))?;
    let dir = dest.join(top);
    if !dir.join("manifest.env").is_file() {
        bail!("the bundle archive has no manifest.env at the top of {}", dir.display());
    }
    Ok(dir)
}

/// Whether a service manager restarts this process when it exits with an error.
pub fn under_service_manager() -> bool {
    let set = |k: &str| std::env::var_os(k).is_some_and(|v| !v.is_empty() && v != "0");
    set("INVOCATION_ID") || set("YMIR_SERVICE") || (cfg!(target_os = "macos") && set("XPC_SERVICE_NAME") && set("LAUNCH_JOB_LABEL"))
}

/// Whether this process was started through `ymirctl update boot`, so a restart picks up a
/// staged release: the env it sets, or this binary living in a staged release directory.
pub fn booted_by_updater(paths: &Paths, exe: &Path) -> bool {
    std::env::var_os(BOOT_ENV).is_some() || exe.starts_with(update::dir(paths).join("releases"))
}

/// Set by `ymirctl update boot --exec` for the daemon it starts.
pub const BOOT_ENV: &str = "YMIR_UPDATE_BOOT";

/// The schedule: the first check a few minutes after the start, then every `interval`. When an
/// update is staged and the daemon can be restarted into it, `restart` is notified and the
/// loop ends.
pub async fn run(paths: Paths, cfg: UpdateSection, running: Version, restart: Arc<tokio::sync::Notify>) {
    let interval = match cfg.validate().and_then(|_| cfg.interval()) {
        Ok(d) => d,
        Err(e) => {
            tracing::warn!("self-update off: {e:#}");
            return;
        }
    };
    let exe = std::env::current_exe().unwrap_or_default();
    let can_restart = cfg.restart && under_service_manager() && booted_by_updater(&paths, &exe);
    let http = match reqwest::Client::builder()
        .user_agent(concat!("ymird/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(30))
        .read_timeout(Duration::from_secs(120))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!("self-update off: no HTTP client: {e}");
            return;
        }
    };
    // Spread the first check so a pool restarted together doesn't ask at once.
    let spread = u64::from(std::process::id()) % 180;
    tracing::info!(feed = %cfg.feed, every = %cfg.interval, restart = can_restart, "self-update on");
    tokio::time::sleep(Duration::from_secs(120 + spread)).await;
    loop {
        match check_once(&paths, &cfg, &running, &exe, &http).await {
            Ok(Outcome::UpToDate { latest }) => tracing::debug!(%latest, "self-update: up to date"),
            Ok(Outcome::Skipped(why)) => tracing::info!("self-update: {why}"),
            Ok(Outcome::Staged { update_id, version }) => {
                if can_restart {
                    tracing::info!(update = %update_id, %version, "self-update: staged; restarting into it (rolls back if it doesn't come up)");
                    restart.notify_one();
                    return;
                }
                tracing::info!(update = %update_id, %version, "self-update: staged; it runs at the next start of the service");
            }
            Err(e) => tracing::warn!("self-update check failed: {e:#}"),
        }
        tokio::time::sleep(interval).await;
    }
}

/// For a release on trial: if the daemon isn't serving within `timeout`, exit with an error so
/// the service manager restarts it and `update boot` counts the failed start (and rolls back
/// after two). `serving` is set once the daemon serves.
pub fn trial_watchdog(paths: &Paths, timeout: Duration, serving: Arc<std::sync::atomic::AtomicBool>) {
    let Ok(u) = Updater::open(paths, Timestamp::now()) else { return };
    let HealthState::Trial { update_id, .. } = u.status().health else { return };
    std::thread::spawn(move || {
        std::thread::sleep(timeout);
        if !serving.load(std::sync::atomic::Ordering::SeqCst) {
            tracing::error!(update = %update_id, "the release on trial is not serving after {}s; exiting so it can be rolled back", timeout.as_secs());
            std::process::exit(1);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_default_and_validate() {
        let d = UpdateSection::default();
        assert!(d.auto && d.restart);
        assert_eq!(d.interval().unwrap(), Duration::from_secs(6 * 3600));
        d.validate().unwrap();
        let c: crate::NodeConfig = toml::from_str("").unwrap();
        assert_eq!(c.update, d);
        let c: crate::NodeConfig = toml::from_str("[update]\nauto = false\ninterval = \"30m\"\nvariant = \"cuda\"\n").unwrap();
        assert!(!c.update.auto);
        assert_eq!(c.update.interval().unwrap(), Duration::from_secs(1800));
        for bad in [
            UpdateSection { channel: "beta".into(), ..d.clone() },
            UpdateSection { interval: "1m".into(), ..d.clone() },
            UpdateSection { interval: "soon".into(), ..d.clone() },
            UpdateSection { feed: "http://example.com/x".into(), ..d.clone() },
            UpdateSection { feed: "github:nope".into(), ..d.clone() },
            UpdateSection { variant: "rocm".into(), ..d.clone() },
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
        assert_eq!(parse_duration("1d").unwrap(), Duration::from_secs(86_400));
        assert_eq!(parse_duration("90").unwrap(), Duration::from_secs(90));
    }

    #[test]
    fn feeds() {
        assert_eq!(Feed::parse(DEFAULT_FEED).unwrap().latest_url(), "https://api.github.com/repos/WeirdBrains/ymir/releases/latest");
        assert_eq!(Feed::parse("https://example.com/latest.json").unwrap().latest_url(), "https://example.com/latest.json");
    }

    #[test]
    fn asset_names() {
        let v = Version::new(0, 15, 3);
        let n = asset_name(false, &v).unwrap();
        assert!(n.starts_with("ymir-0.15.3-") && n.ends_with(".update.tar.gz"), "{n}");
        assert!(asset_name(true, &v).unwrap().starts_with("ymir-cuda-0.15.3-"));
    }

    #[test]
    fn cuda_is_read_from_the_engine() {
        let t = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(t.path().join("llama.cpp")).unwrap();
        let exe = t.path().join("ymird");
        std::fs::write(t.path().join("llama.cpp/engine.toml"), "variant = \"default\"\nbackends = [\"cpu\", \"vulkan\"]\n").unwrap();
        assert!(!running_cuda(&exe));
        std::fs::write(t.path().join("llama.cpp/engine.toml"), "variant = \"cuda\"\nbackends = [\"cpu\", \"cuda\"]\n").unwrap();
        assert!(running_cuda(&exe));
    }

    /// A test org, a signed bundle of `version` packed as the release flow packs it, and a local
    /// feed serving it.
    #[cfg(unix)]
    mod feed {
        use super::*;
        use weirdnet_update::envelope::Domain;
        use weirdnet_update::{Delegation, Root, Signed, SigningKey};

        fn key(n: u8) -> SigningKey {
            SigningKey::from_seed(&[n; 32])
        }

        fn org() -> (SigningKey, Vec<u8>, String) {
            let (r, rel) = (key(1), key(10));
            let root = Root {
                version: 1,
                org: "weirdbrains".into(),
                keys: vec![r.public()],
                threshold: 1,
                release: vec![Delegation::any(rel.public())],
                release_threshold: 1,
                timestamp: vec![],
                timestamp_threshold: 0,
                expires: None,
            };
            (rel, Signed::sign(Domain::Root, &root, &[&r]).unwrap().to_bytes().unwrap(), root.pin().to_string())
        }

        fn packed(dir: &Path, rel: &SigningKey, root: &[u8], version: &Version) -> (String, Vec<u8>) {
            let name = asset_name(false, version).unwrap();
            let top = name.trim_end_matches(".tar.gz").to_string();
            let b = dir.join(&top);
            std::fs::create_dir_all(b.join("llama.cpp")).unwrap();
            std::fs::write(b.join("ymird"), b"#!/bin/sh\n").unwrap();
            std::fs::write(b.join("llama.cpp/engine.toml"), b"variant = \"default\"\n").unwrap();
            let (platform, arch) = update::this_target().unwrap();
            let spec = update_publish::BundleSpec {
                version: version.clone(),
                seq: update_publish::seq_for(version).unwrap(),
                platform,
                arch,
                notes: String::new(),
                valid_secs: 86_400,
            };
            update_publish::sign_bundle(&b, rel, root, None, &spec, Timestamp::now()).unwrap();
            let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
            {
                let mut t = tar::Builder::new(&mut gz);
                t.append_dir_all(&top, &b).unwrap();
                t.finish().unwrap();
            }
            (name, gz.finish().unwrap())
        }

        async fn serve(tag: &str, name: &str, body: Vec<u8>) -> String {
            use axum::{Router, routing::get};
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let base = format!("http://{}", listener.local_addr().unwrap());
            let json = serde_json::json!({
                "tag_name": tag,
                "assets": [{"name": name, "browser_download_url": format!("{base}/dl/{name}"), "size": body.len()}],
            });
            let body = Arc::new(body);
            let app = Router::new()
                .route("/latest", get(move || async move { axum::Json(json.clone()) }))
                .route("/dl/{name}", get(move || async move { (*body).clone() }));
            tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            format!("{base}/latest")
        }

        #[tokio::test]
        async fn a_newer_release_on_the_feed_is_downloaded_verified_and_staged() {
            let t = tempfile::tempdir().unwrap();
            let (rel, root, pin) = org();
            let v = Version::new(0, 15, 3);
            let (name, body) = packed(t.path(), &rel, &root, &v);
            let feed = serve("v0.15.3", &name, body).await;
            let paths = Paths::new(t.path().join("home"));
            let cfg = UpdateSection { feed, ..UpdateSection::default() };
            let http = reqwest::Client::new();
            let exe = t.path().join("bin/ymird");

            // No root pinned: nothing is taken.
            let o = check_once(&paths, &cfg, &Version::new(0, 15, 2), &exe, &http).await.unwrap();
            assert!(matches!(o, Outcome::Skipped(ref w) if w.contains("pinned")), "{o:?}");

            Updater::open(&paths, Timestamp::now()).unwrap().trust(&pin, &root, Timestamp::now()).unwrap();
            // Not newer than what runs.
            let o = check_once(&paths, &cfg, &Version::new(0, 15, 3), &exe, &http).await.unwrap();
            assert_eq!(o, Outcome::UpToDate { latest: v.clone() });

            let o = check_once(&paths, &cfg, &Version::new(0, 15, 2), &exe, &http).await.unwrap();
            assert_eq!(o, Outcome::Staged { update_id: "com.weirdbrains.ymir-stable-15003".into(), version: v.clone() });
            assert!(!update::dir(&paths).join("incoming").exists());
            // Staged once; the next check leaves it alone until it runs.
            let o = check_once(&paths, &cfg, &Version::new(0, 15, 2), &exe, &http).await.unwrap();
            assert!(matches!(o, Outcome::Skipped(ref w) if w.contains("already staged")), "{o:?}");
            let mut u = Updater::open(&paths, Timestamp::now()).unwrap();
            assert!(matches!(u.boot().unwrap(), update::Boot::Run { trial: true, .. }));
        }

        #[tokio::test]
        async fn a_bundle_not_signed_by_the_pinned_org_is_refused() {
            let t = tempfile::tempdir().unwrap();
            let (_, root, pin) = org();
            let (other_rel, other_root, _) = {
                let (r, rel) = (key(2), key(11));
                let root = Root {
                    version: 1,
                    org: "evil".into(),
                    keys: vec![r.public()],
                    threshold: 1,
                    release: vec![Delegation::any(rel.public())],
                    release_threshold: 1,
                    timestamp: vec![],
                    timestamp_threshold: 0,
                    expires: None,
                };
                (rel, Signed::sign(Domain::Root, &root, &[&r]).unwrap().to_bytes().unwrap(), ())
            };
            let v = Version::new(0, 15, 3);
            let (name, body) = packed(t.path(), &other_rel, &other_root, &v);
            let feed = serve("v0.15.3", &name, body).await;
            let paths = Paths::new(t.path().join("home"));
            Updater::open(&paths, Timestamp::now()).unwrap().trust(&pin, &root, Timestamp::now()).unwrap();
            let cfg = UpdateSection { feed, ..UpdateSection::default() };
            let e = check_once(&paths, &cfg, &Version::new(0, 15, 2), &t.path().join("ymird"), &reqwest::Client::new()).await.unwrap_err();
            assert!(format!("{e:#}").contains("refused"), "{e:#}");
            assert!(matches!(Updater::open(&paths, Timestamp::now()).unwrap().status().health, HealthState::Idle));
        }

        type Tb<'a> = tar::Builder<&'a mut flate2::write::GzEncoder<Vec<u8>>>;

        #[test]
        fn unsafe_archives_are_refused() {
            let t = tempfile::tempdir().unwrap();
            let mk = |entries: &dyn Fn(&mut Tb)| {
                let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
                {
                    let mut b = tar::Builder::new(&mut gz);
                    entries(&mut b);
                    b.finish().unwrap();
                }
                gz.finish().unwrap()
            };
            let link = mk(&|b| {
                let mut h = tar::Header::new_gnu();
                h.set_entry_type(tar::EntryType::Symlink);
                h.set_size(0);
                b.append_link(&mut h, "x/ymird", "/etc/passwd").unwrap();
            });
            let p = t.path().join("link.tar.gz");
            std::fs::write(&p, link).unwrap();
            assert!(unpack(&p, &t.path().join("out")).is_err());
            let two = mk(&|b| {
                for n in ["a/manifest.env", "b/ymird"] {
                    let mut h = tar::Header::new_gnu();
                    h.set_size(1);
                    h.set_mode(0o644);
                    h.set_cksum();
                    b.append_data(&mut h, n, &b"x"[..]).unwrap();
                }
            });
            let p = t.path().join("two.tar.gz");
            std::fs::write(&p, two).unwrap();
            let e = unpack(&p, &t.path().join("out2")).unwrap_err().to_string();
            assert!(e.contains("more than one top"), "{e}");
        }
    }
}
