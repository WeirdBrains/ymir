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

#[tokio::test(flavor = "multi_thread")]
async fn ymirctl_fetches_an_announced_bundle_over_http_and_stages_it() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let home = tmp.path().join("home");
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
    let root_file = tmp.path().join("root.env");
    std::fs::write(&root_file, Signed::sign(Domain::Root, &root, &[&r1, &r2]).unwrap().to_bytes().unwrap())?;
    let (platform, arch) = ymir_core::update::this_target().expect("a supported platform");
    let now = Timestamp::now();
    let site = tmp.path().join("site");
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
    let announce = |sources: Vec<String>, signer: &SigningKey| -> Result<PathBuf> {
        let a = Announcement::for_manifest("weirdbrains", root.pin(), &manifest, now.minus(10), 3600, sources)?;
        let f = tmp.path().join("announcement.env");
        std::fs::write(&f, a.sign(&[signer])?.to_bytes()?)?;
        Ok(f)
    };
    let (url, log) = serve(site.clone());
    let src = format!("{url}/ymir/fetch-1");

    // Plain http to another machine is refused; nothing is pinned yet, so nothing is taken.
    let e = ctl(&home, &["update", "fetch", "http://updates.example/ymir"]).await.unwrap_err().to_string();
    assert!(e.contains("must be an https:// URL"), "{e}");
    let e = ctl(&home, &["update", "fetch", &src]).await.unwrap_err().to_string();
    assert!(e.contains("no update root"), "{e}");
    ctl(&home, &["update", "trust", "--pin", &root.pin().to_string(), "--root", root_file.to_str().unwrap()]).await?;

    // An announcement signed by someone else is refused before any download.
    let bad = announce(vec!["https://updates.example/ymir/fetch-1".into()], &key(99))?;
    let e = ctl(&home, &["update", "fetch", "--announcement", bad.to_str().unwrap(), &src]).await.unwrap_err().to_string();
    assert!(e.contains("announcement is refused"), "{e}");
    assert!(log.lock().unwrap().is_empty());

    // The announced manifest, fetched from the local mirror.
    let good = announce(vec!["https://updates.example/ymir/fetch-1".into()], &rel)?;
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
