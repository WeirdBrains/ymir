//! `ymirctl update fetch`: a bundle directory over HTTPS (or on disk) as an update [`Source`].

use std::io::Read;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use ymir_core::update::{DirSource, Source};

/// A bundle directory served over HTTPS: `BASE/manifest.env`, `BASE/<file path>`, ...
pub struct HttpSource {
    base: String,
    agent: ureq::Agent,
}

impl HttpSource {
    /// `https://` anywhere; `http://` only to this machine (a local mirror, or a test).
    pub fn new(base: &str) -> Result<Self> {
        let loopback = ["http://127.0.0.1", "http://localhost", "http://[::1]"]
            .iter()
            .any(|p| base.strip_prefix(p).is_some_and(|rest| rest.is_empty() || rest.starts_with([':', '/'])));
        if !base.starts_with("https://") && !loopback {
            bail!("an update source must be an https:// URL (http:// only to 127.0.0.1 or localhost), or a directory");
        }
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_connect(Some(Duration::from_secs(30)))
            .timeout_recv_response(Some(Duration::from_secs(60)))
            .build()
            .into();
        Ok(Self { base: base.trim_end_matches('/').to_string(), agent })
    }
}

/// Path segments as URL path characters (manifest paths are already checked: relative, `/`
/// separated, no `..`).
fn encode(path: &str) -> String {
    let mut s = String::new();
    for b in path.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~/+".contains(&b) {
            s.push(b as char);
        } else {
            s.push_str(&format!("%{b:02X}"));
        }
    }
    s
}

impl Source for HttpSource {
    fn open(&self, path: &str) -> Result<Option<Box<dyn Read + '_>>> {
        let url = format!("{}/{}", self.base, encode(path));
        let resp =
            self.agent.get(&url).header("User-Agent", concat!("ymirctl/", env!("CARGO_PKG_VERSION"))).call().map_err(|e| anyhow!("{url}: {e}"))?;
        match resp.status().as_u16() {
            200 => Ok(Some(Box::new(resp.into_body().into_reader()))),
            404 | 410 => Ok(None),
            s => bail!("{url}: the server answered {s}"),
        }
    }

    fn describe(&self) -> String {
        self.base.clone()
    }
}

/// The source for `update fetch SOURCE`: a URL or a directory.
pub fn source(s: &str) -> Result<Box<dyn Source>> {
    if s.contains("://") {
        return Ok(Box::new(HttpSource::new(s)?));
    }
    let dir = PathBuf::from(s);
    if !dir.is_dir() {
        bail!("{s} is neither a URL nor a directory");
    }
    Ok(Box::new(DirSource(dir)))
}

/// `ymirctl update watch`: hears Ymir's update announcements from a carrier (a Weirdnet relay
/// with `[updates]`, protocol `/weirdnet/updates/1`), checks each against the pinned root, and
/// with `fetch` downloads and stages a new one from its first https:// source.
pub mod watch {
    use std::io::Write;
    use std::sync::Arc;
    use std::time::Duration;

    use anyhow::{Context, Result, anyhow, bail};
    use weirdnet_net::{DiscoveryMode, Keypair, Limits, Multiaddr, Node, NodeConfig, PeerId, WEIRDNET};
    use weirdnet_update::announce::Topic;
    use weirdnet_update::announce::wire::{PROTOCOL, Subscription};
    use weirdnet_update::envelope::Domain;
    use ymir_core::Paths;
    use ymir_core::update::{APP, CHANNEL, Updater};

    pub struct Options {
        pub relay: String,
        pub fetch: bool,
        /// Stop after the first announcement (or after `wait` with none).
        pub once: bool,
        pub wait: Duration,
    }

    struct NoRequests;
    impl weirdnet_net::Handler<(), ()> for NoRequests {
        fn request(&self, _from: &PeerId, _req: ()) {}
    }

    fn peer_of(a: &Multiaddr) -> Option<PeerId> {
        a.to_string().rsplit_once("/p2p/").and_then(|(_, id)| id.split('/').next()?.parse().ok())
    }

    pub async fn run(paths: &Paths, out: &mut dyn Write, o: Options) -> Result<()> {
        let pin = Updater::open(paths, weirdnet_update::Timestamp::now())?
            .pin()
            .ok_or_else(|| anyhow!("no update root is pinned (`ymirctl update trust`)"))?;
        let addr: Multiaddr = o.relay.parse().with_context(|| format!("--relay {}", o.relay))?;
        let peer = peer_of(&addr).ok_or_else(|| anyhow!("--relay needs the relay's peer id (…/p2p/ID)"))?;
        // A dial-only node with a fresh key: no listener, no discovery, nothing that names this machine.
        let cfg = NodeConfig { discovery: DiscoveryMode::Disabled, ..Default::default() };
        let node = Arc::new(Node::<(), ()>::start(&WEIRDNET, Keypair::generate_ed25519(), cfg, Arc::new(NoRequests)).map_err(|e| anyhow!("{e}"))?);
        let stream =
            node.open_stream_async(peer, std::slice::from_ref(&addr), PROTOCOL, Limits::default()).await.map_err(|e| anyhow!("{}: {e}", o.relay))?;
        let mut sub = Subscription::start(stream, vec![Topic { pin, app: APP.into(), channel: CHANNEL.into() }], false)
            .await
            .map_err(|e| anyhow!("{}: {e}", o.relay))?;
        writeln!(out, "subscribed to {APP} on {CHANNEL} at {}", o.relay)?;
        out.flush()?;
        let idle = if o.once { o.wait } else { Duration::from_secs(u64::from(sub.keepalive_s.max(10)) * 3) };
        loop {
            let env = match tokio::time::timeout(idle, sub.next()).await {
                Err(_) if o.once => {
                    writeln!(out, "no announcement within {} s", o.wait.as_secs())?;
                    return Ok(());
                }
                Err(_) => bail!("{} went quiet (no keepalive)", o.relay),
                Ok(Ok(Some(env))) => env,
                Ok(Ok(None)) => bail!("{} closed the subscription", o.relay),
                Ok(Err(e)) => bail!("{}: {e}", o.relay),
            };
            if env.domain != Domain::Announcement {
                continue; // root rotations arrive with the bundle too
            }
            let now = weirdnet_update::Timestamp::now();
            let mut u = Updater::open(paths, now)?;
            let bytes = env.to_bytes().map_err(|e| anyhow!("{e}"))?;
            match u.check_announcement(&bytes, now) {
                Err(e) => writeln!(out, "ignored an announcement: {e:#}")?,
                Ok(None) => writeln!(out, "heard an announcement that is not newer than the update already taken")?,
                Ok(Some(a)) => {
                    writeln!(out, "announced: {} (seq {}){}", a.update_id, a.seq, if a.security { ", a security update" } else { "" })?;
                    let source = a.sources.iter().find(|s| s.starts_with("https://")).cloned();
                    match (o.fetch, source) {
                        (false, _) => writeln!(out, "not fetched (run with --fetch, or `ymirctl update fetch` with the announcement)")?,
                        (true, None) => writeln!(out, "not fetched: the announcement has no https:// source")?,
                        (true, Some(src)) => {
                            let running = semver::Version::parse(env!("CARGO_PKG_VERSION"))?;
                            let hash = a.manifest_hash;
                            let r = tokio::task::block_in_place(|| -> Result<_> {
                                let s = super::source(&src)?;
                                u.fetch(s.as_ref(), Some(&hash), &running, now)
                            });
                            match r {
                                Ok(f) => writeln!(
                                    out,
                                    "fetched {} files ({} bytes) from {src}; staged {} {}; it runs on trial at the next start",
                                    f.files, f.bytes, f.update.manifest.update_id, f.update.manifest.version
                                )?,
                                Err(e) => writeln!(out, "fetching {src} failed: {e:#}")?,
                            }
                        }
                    }
                }
            }
            out.flush()?;
            if o.once {
                return Ok(());
            }
        }
    }
}
