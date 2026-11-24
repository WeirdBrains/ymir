//! M9 bundle fetcher: `ymirctl update fetch` downloads a signed bundle directory over HTTP from
//! a server on this machine (https:// in real use), checks an announcement first, and stages
//! the update. Test keys only; nothing leaves the box.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use clap::Parser as _;
use weirdnet_update::envelope::Domain;
use weirdnet_update::manifest::{FullPackage, Labels, PackageFormat};
use weirdnet_update::{Announcement, Delegation, FileEntry, Root, Signed, SigningKey, Timestamp, UpdateKind, UpdateManifest};

async fn ctl(home: &Path, args: &[&str]) -> Result<String> {
    let mut argv = vec!["ymirctl".to_string(), "--home".into(), home.display().to_string()];
    argv.extend(args.iter().map(|a| a.to_string()));
    let cli = ymirctl::Cli::try_parse_from(argv)?;
    let mut out = Vec::new();
    let mut input = std::io::Cursor::new(Vec::new());
    ymirctl::run(cli, &mut out, &mut input).await?;
    Ok(String::from_utf8(out)?)
}

/// Serves `dir` on 127.0.0.1 (GET only, 404 for what is missing) and records each path asked for.
fn serve(dir: PathBuf) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let log = Arc::new(Mutex::new(Vec::new()));
    let l = log.clone();
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(mut s) = conn else { continue };
            let mut r = BufReader::new(s.try_clone().unwrap());
            let mut line = String::new();
            if r.read_line(&mut line).is_err() {
                continue;
            }
            loop {
                let mut h = String::new();
                if r.read_line(&mut h).is_err() || h == "\r\n" || h.is_empty() {
                    break;
                }
            }
            let path = line.split_whitespace().nth(1).unwrap_or("/").trim_start_matches('/').to_string();
            l.lock().unwrap().push(path.clone());
            let file = dir.join(path.replace("%20", " "));
            let resp = match std::fs::read(&file) {
                Ok(body) if file.is_file() => {
                    let mut b = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).into_bytes();
                    b.extend(body);
                    b
                }
                _ => b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec(),
            };
            let _ = s.write_all(&resp);
        }
    });
    (url, log)
}

fn key(n: u8) -> SigningKey {
    SigningKey::from_seed(&[n; 32])
}

struct Fx {
    _tmp: tempfile::TempDir,
    dir: PathBuf,
    home: PathBuf,
    root: Root,
    root_env: Signed,
    root_file: PathBuf,
    rel: SigningKey,
    manifest: Signed,
    site: PathBuf,
    now: Timestamp,
}

impl Fx {
    /// A signed announcement of the bundle's manifest, written to a file.
    fn announce(&self, sources: Vec<String>, signer: &SigningKey) -> Result<(PathBuf, Signed)> {
        let a = Announcement::for_manifest("weirdbrains", self.root.pin(), &self.manifest, self.now.minus(10), 3600, sources)?;
        let env = a.sign(&[signer])?;
        let f = self.dir.join("announcement.env");
        std::fs::write(&f, env.to_bytes()?)?;
        Ok((f, env))
    }

    async fn trust(&self) -> Result<()> {
        ctl(&self.home, &["update", "trust", "--pin", &self.root.pin().to_string(), "--root", self.root_file.to_str().unwrap()]).await?;
        Ok(())
    }
}

/// An org root (test keys), and a signed bundle `fetch-1` (ymird and an engine file) under
/// `site/ymir/fetch-1`.
fn fixture() -> Result<Fx> {
    let tmp = tempfile::tempdir()?;
    let dir = tmp.path().to_path_buf();
    let home = dir.join("home");
    let (r1, r2, rel) = (key(1), key(2), key(10));
    let root = Root {
        version: 1,
        org: "weirdbrains".into(),
        keys: vec![r1.public(), r2.public()],
        threshold: 2,
        release: vec![Delegation::any(rel.public())],
        release_threshold: 1,
        timestamp: vec![],
        timestamp_threshold: 0,
        expires: None,
    };
    let root_file = dir.join("root.env");
    let root_env = Signed::sign(Domain::Root, &root, &[&r1, &r2]).unwrap();
    std::fs::write(&root_file, root_env.to_bytes().unwrap())?;
    let (platform, arch) = ymir_core::update::this_target().expect("a supported platform");
    let now = Timestamp::now();
    let site = dir.join("site");
    let bundle = site.join("ymir/fetch-1");
    let files: [(&str, &[u8]); 2] = [("ymird", b"#!/bin/sh\nexit 0\n"), ("lib/ymir/llama.cpp/engine.toml", b"engine = \"llama.cpp\"\n")];
    for (p, body) in files {
        std::fs::create_dir_all(bundle.join(p).parent().unwrap())?;
        std::fs::write(bundle.join(p), body)?;
    }
    let m = UpdateManifest {
        app: ymir_core::update::APP.into(),
        channel: "stable".into(),
        update_id: "fetch-1".into(),
        seq: 1,
        kind: UpdateKind::Full(FullPackage { platform, arch, format: PackageFormat::TarGz, signing_identity: "unsigned".into() }),
        version: semver::Version::new(99, 0, 0),
        base_releases: vec![],
        files: files.iter().map(|(p, b)| FileEntry::describe(p, b)).collect(),
        min_client: None,
        max_client: None,
        capabilities: vec![],
        labels: Labels::default(),
        targeting: None,
        rollout: None,
        rollback_of: None,
        notes: "test".into(),
        created: now.minus(60),
        expires: now.plus(3600),
    };
    let manifest = Signed::sign(Domain::Manifest, &m, &[&rel]).unwrap();
    std::fs::write(bundle.join("manifest.env"), manifest.to_bytes().unwrap())?;
    Ok(Fx { _tmp: tmp, dir, home, root, root_env, root_file, rel, manifest, site, now })
}

#[tokio::test(flavor = "multi_thread")]
async fn ymirctl_fetches_an_announced_bundle_over_http_and_stages_it() -> Result<()> {
    let fx = fixture()?;
    let home = fx.home.clone();
    let (url, log) = serve(fx.site.clone());
    let src = format!("{url}/ymir/fetch-1");

    // Plain http to another machine is refused; nothing is pinned yet, so nothing is taken.
    let e = ctl(&home, &["update", "fetch", "http://updates.example/ymir"]).await.unwrap_err().to_string();
    assert!(e.contains("must be an https:// URL"), "{e}");
    let e = ctl(&home, &["update", "fetch", &src]).await.unwrap_err().to_string();
    assert!(e.contains("no update root"), "{e}");
    fx.trust().await?;

    // An announcement signed by someone else is refused before any download.
    let (bad, _) = fx.announce(vec!["https://updates.example/ymir/fetch-1".into()], &key(99))?;
    let e = ctl(&home, &["update", "fetch", "--announcement", bad.to_str().unwrap(), &src]).await.unwrap_err().to_string();
    assert!(e.contains("announcement is refused"), "{e}");
    assert!(log.lock().unwrap().is_empty());

    // The announced manifest, fetched from the local mirror.
    let (good, _) = fx.announce(vec!["https://updates.example/ymir/fetch-1".into()], &fx.rel)?;
    let out = ctl(&home, &["update", "fetch", "--announcement", good.to_str().unwrap(), &src]).await?;
    assert!(out.contains("announced: fetch-1 (seq 1)"), "{out}");
    assert!(out.contains("fetched 2 files (38 bytes) from") && out.contains("staged fetch-1 99.0.0"), "{out}");
    assert_eq!(
        *log.lock().unwrap(),
        [
            "ymir/fetch-1/manifest.env",
            "ymir/fetch-1/roots/1.env",
            "ymir/fetch-1/freshness.env",
            "ymir/fetch-1/ymird",
            "ymir/fetch-1/lib/ymir/llama.cpp/engine.toml"
        ]
    );
    let path = ctl(&home, &["update", "boot"]).await?;
    assert!(path.trim().ends_with("releases/fetch-1/ymird"), "{path}");
    assert_eq!(std::fs::read(path.trim())?, b"#!/bin/sh\nexit 0\n");
    ymir_core::update::commit_if_on_trial(&ymir_core::Paths::new(&home));

    // Heard again, the same announcement is old news.
    let out = ctl(&home, &["update", "fetch", "--announcement", good.to_str().unwrap()]).await?;
    assert!(out.contains("nothing new"), "{out}");
    Ok(())
}

struct NoRequests;
impl weirdnet_net::Handler<(), ()> for NoRequests {
    fn request(&self, _from: &weirdnet_net::PeerId, _req: ()) {}
}

/// A carrier on 127.0.0.1 speaking `/weirdnet/updates/1` with weirdnet-update's own `Carrier`
/// (what a relay with `[updates]` runs): it trusts `root` and holds `held`. Returns its address.
async fn carrier(root: &Root, root_env: &Signed, held: &[Signed], now: Timestamp) -> Result<(Arc<weirdnet_net::Node<(), ()>>, String)> {
    use futures::{AsyncReadExt, AsyncWriteExt};
    use weirdnet_net::{DiscoveryMode, Keypair, Limits, Node, NodeConfig, WEIRDNET};
    use weirdnet_update::announce::Carrier;
    use weirdnet_update::announce::wire::{self, FromCarrier, PROTOCOL, ToCarrier};

    let mut c = Carrier::new(16);
    c.trust(root.pin(), std::slice::from_ref(root_env), now)?;
    for e in held {
        c.offer(e, now)?;
    }
    let c = Arc::new(Mutex::new(c));
    let cfg = NodeConfig { listen: vec!["/ip4/127.0.0.1/tcp/0".parse()?], discovery: DiscoveryMode::Disabled, ..Default::default() };
    let node =
        Arc::new(Node::<(), ()>::start(&WEIRDNET, Keypair::generate_ed25519(), cfg, Arc::new(NoRequests)).map_err(|e| anyhow::anyhow!("{e}"))?);
    let mut incoming = node.accept_streams(PROTOCOL, Limits::default()).map_err(|e| anyhow::anyhow!("{e}"))?;
    tokio::spawn(async move {
        while let Some(s) = incoming.next().await {
            let c = c.clone();
            tokio::spawn(async move {
                let (mut r, mut w) = s.split();
                if let Ok(Some(ToCarrier::Subscribe { topics, .. })) = wire::read::<_, ToCarrier>(&mut r).await {
                    let held = c.lock().unwrap().matching(&topics, now);
                    let _ = wire::write(&mut w, &FromCarrier::Subscribed { keepalive_s: 10 }).await;
                    for env in held {
                        let _ = wire::write(&mut w, &FromCarrier::Envelope(env)).await;
                    }
                    let mut b = [0u8; 64];
                    while matches!(r.read(&mut b).await, Ok(n) if n > 0) {}
                    let _ = w.close().await;
                }
            });
        }
    });
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        if let Some(a) = node.status().listening.into_iter().find(|a| a.starts_with("/ip4/127.0.0.1/tcp/")) {
            let addr = if a.contains("/p2p/") { a } else { format!("{a}/p2p/{}", node.peer_id()) };
            return Ok((node, addr));
        }
        anyhow::ensure!(std::time::Instant::now() < deadline, "the carrier never listened");
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn ymirctl_watch_hears_an_announcement_from_a_carrier_and_fetches_it() -> Result<()> {
    let fx = fixture()?;
    fx.trust().await?;
    let (url, log) = serve(fx.site.clone());
    // An announcement names https:// sources only; this one points at a port on this machine
    // where nothing answers, so `--fetch` fails without leaving the box.
    let (_, env) = fx.announce(vec!["https://127.0.0.1:9/ymir/fetch-1".into()], &fx.rel)?;
    let (_carrier, addr) = carrier(&fx.root, &fx.root_env, &[env], fx.now).await?;

    let out = ctl(&fx.home, &["update", "watch", "--relay", &addr, "--once", "--wait-secs", "20"]).await?;
    assert!(out.contains("subscribed to com.weirdbrains.ymir on stable"), "{out}");
    assert!(out.contains("announced: fetch-1 (seq 1)") && out.contains("not fetched (run with --fetch"), "{out}");
    assert!(log.lock().unwrap().is_empty(), "nothing was downloaded");

    // With --fetch, the https:// source can't be reached here, and that is reported, not staged.
    let out = ctl(&fx.home, &["update", "watch", "--relay", &addr, "--once", "--fetch", "--wait-secs", "20"]).await?;
    assert!(out.contains("fetching https://127.0.0.1:9/ymir/fetch-1 failed"), "{out}");
    let st = ctl(&fx.home, &["update", "status"]).await?;
    assert!(st.contains("Idle"), "{st}");
    // Fetched by hand from the mirror, after which the carrier's announcement is old news.
    ctl(&fx.home, &["update", "fetch", &format!("{url}/ymir/fetch-1")]).await?;
    let out = ctl(&fx.home, &["update", "watch", "--relay", &addr, "--once", "--wait-secs", "20"]).await?;
    assert!(out.contains("not newer than the update already taken"), "{out}");
    Ok(())
}
