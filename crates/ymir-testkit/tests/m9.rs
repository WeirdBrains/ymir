//! M9: hardening checks that run for real. A combined node (model, inference and app roles, one
//! mock engine) is started and inspected: every socket it and its engine open is on loopback, the
//! secrets and the control socket are private, and nothing but the bridge token opens the bridge.
//! Then the update flow through `ymirctl`: pin a root, stage a signed bundle, start it on trial.

use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use clap::Parser as _;
use serde_json::{Value, json};
use weirdnet_update::envelope::Domain;
use weirdnet_update::manifest::{FullPackage, Labels, PackageFormat};
use weirdnet_update::{Delegation, FileEntry, Root, Signed, SigningKey, Timestamp, UpdateKind, UpdateManifest};
use ymir_core::control::{self, ControlRequest};
use ymir_core::membership::Membership;
use ymir_core::{Daemon, NodeConfig, Paths};

const MOCK: &str = env!("CARGO_BIN_EXE_ymir-mock-engine");

async fn ctl_with(home: &Path, args: &[&str], out: &mut dyn Write, input: &str) -> Result<()> {
    let mut argv = vec!["ymirctl".to_string(), "--home".into(), home.display().to_string()];
    argv.extend(args.iter().map(|a| a.to_string()));
    let cli = ymirctl::Cli::try_parse_from(argv)?;
    let mut input = std::io::Cursor::new(input.as_bytes().to_vec());
    ymirctl::run(cli, out, &mut input).await
}

async fn ctl(home: &Path, args: &[&str]) -> Result<String> {
    let mut out = Vec::new();
    ctl_with(home, args, &mut out, "").await?;
    Ok(String::from_utf8(out)?)
}

/// A daemon's home in `root/name`.
struct Home {
    paths: Paths,
}

/// node.toml for `roles`, on Weirdnet over 127.0.0.1 only, dialing `peers`.
fn home(root: &Path, name: &str, roles: &str, peers: &[String], models: &[&str]) -> Result<Home> {
    let home = root.join(name);
    let models_dir = root.join("models");
    std::fs::create_dir_all(&home)?;
    std::fs::create_dir_all(&models_dir)?;
    let mut local = String::new();
    for (i, id) in models.iter().enumerate() {
        std::fs::write(models_dir.join(format!("{id}.gguf")), b"GGUF mock")?;
        let role = if i == 0 { "role = \"main\"\n" } else { "" };
        local.push_str(&format!("[[model.local]]\nid = \"{id}\"\nfile = \"{id}.gguf\"\n{role}"));
    }
    let peers = peers.iter().map(|p| format!("\"{p}\"")).collect::<Vec<_>>().join(", ");
    let (m, i, a) = (roles.contains("model"), roles.contains("inference"), roles.contains("app"));
    let toml = format!(
        "[node]\nname = \"{name}\"\n[roles]\nmodel = {m}\ninference = {i}\napp = {a}\n[network]\nlisten = [\"/ip4/127.0.0.1/tcp/0\"]\nmdns = false\ndiscovery = false\nrelays = []\npeers = [{peers}]\n[model]\nllamacpp = \"system:{MOCK}\"\nvllm = \"off\"\nmodels_dir = \"{}\"\n{local}[inference.alerts]\nprobe_secs = 3600\n[app.bridge]\nport = 0\n",
        models_dir.display()
    );
    std::fs::write(home.join("node.toml"), toml)?;
    Ok(Home { paths: Paths::new(&home) })
}

async fn start(h: &Home) -> Result<Daemon> {
    Daemon::start(NodeConfig::load(&h.paths.config())?, h.paths.clone()).await
}

fn bridge(d: &Daemon, h: &Home) -> (reqwest::Client, String, String) {
    let port = d.bridge_addr.unwrap().port();
    let token = ymir_core::token::load(&h.paths).unwrap().unwrap();
    (reqwest::Client::builder().no_proxy().build().unwrap(), format!("http://127.0.0.1:{port}/v1"), token)
}

async fn chat_model(d: &Daemon, h: &Home, model: &str, text: &str) -> Result<(u16, Value)> {
    let (http, url, token) = bridge(d, h);
    let r = http
        .post(format!("{url}/chat/completions"))
        .bearer_auth(&token)
        .json(&json!({"model": model, "temperature": 1, "messages": [{"role": "user", "content": text}]}))
        .send()
        .await?;
    Ok((r.status().as_u16(), r.json().await.unwrap_or(Value::Null)))
}

/// Retries `f` until it answers 200 or `secs` pass (discovery and dialing take a moment).
async fn until_ok(secs: u64, mut f: impl AsyncFnMut() -> Result<(u16, Value)>) -> Result<Value> {
    let deadline = Instant::now() + Duration::from_secs(secs);
    loop {
        let (status, v) = f().await?;
        if status == 200 {
            return Ok(v);
        }
        if Instant::now() > deadline {
            bail!("still {status}: {v}");
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

/// Sockets in LISTEN state owned by this process or its children, as (address, port).
fn listeners() -> Vec<(String, u16)> {
    let me = std::process::id();
    let mut family = std::collections::HashMap::<u32, u32>::new();
    for e in std::fs::read_dir("/proc").unwrap().flatten() {
        let Ok(pid) = e.file_name().to_string_lossy().parse::<u32>() else { continue };
        let Ok(stat) = std::fs::read_to_string(e.path().join("stat")) else { continue };
        // pid (comm) state ppid ...: comm may hold spaces, so split after the last ')'.
        let after = stat.rsplit_once(')').map(|x| x.1).unwrap_or("");
        let ppid: u32 = after.split_whitespace().nth(1).and_then(|p| p.parse().ok()).unwrap_or(0);
        family.insert(pid, ppid);
    }
    let mut mine = std::collections::HashSet::from([me]);
    loop {
        let before = mine.len();
        for (pid, ppid) in &family {
            if mine.contains(ppid) {
                mine.insert(*pid);
            }
        }
        if mine.len() == before {
            break;
        }
    }
    let mut inodes = std::collections::HashSet::new();
    for pid in &mine {
        let Ok(rd) = std::fs::read_dir(format!("/proc/{pid}/fd")) else { continue };
        for fd in rd.flatten() {
            if let Ok(t) = std::fs::read_link(fd.path())
                && let Some(i) = t.to_string_lossy().strip_prefix("socket:[").and_then(|x| x.strip_suffix(']'))
            {
                inodes.insert(i.to_string());
            }
        }
    }
    let mut out = Vec::new();
    for table in ["/proc/net/tcp", "/proc/net/tcp6"] {
        let text = std::fs::read_to_string(table).unwrap_or_default();
        for line in text.lines().skip(1) {
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() > 9 && f[3] == "0A" && inodes.contains(f[9]) {
                let (addr, port) = f[1].split_once(':').unwrap();
                out.push((addr.to_string(), u16::from_str_radix(port, 16).unwrap()));
            }
        }
    }
    out
}

fn is_loopback(hex: &str) -> bool {
    // 127.0.0.1 in /proc/net/tcp is 0100007F; ::1 in tcp6 is 00000000000000000000000001000000;
    // an IPv4-mapped loopback is 0000000000000000FFFF00000100007F.
    matches!(hex, "0100007F" | "00000000000000000000000001000000" | "0000000000000000FFFF00000100007F")
}

#[cfg(unix)]
fn mode(p: &Path) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(p).unwrap().permissions().mode() & 0o777
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_running_node_listens_only_on_loopback_and_keeps_its_secrets_private() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let h = home(tmp.path(), "box", "model,inference,app", &[], &["alpha"])?;
    Membership::create(&h.paths.pool(), "box")?;
    let d = start(&h).await?;
    // Load the model so an engine process is running with its own port, and ask once through the bridge.
    let r = control::request(&h.paths.socket(), &ControlRequest::Load { model: "alpha".into(), node: None }).await?;
    assert!(r.ok, "{:?}", r.message);
    let v = until_ok(30, async || chat_model(&d, &h, "alpha", "hello").await).await?;
    assert!(v["choices"][0]["message"]["content"].as_str().unwrap_or("").contains("hello"));

    let ls = listeners();
    assert!(ls.len() >= 3, "expected the bridge, the Weirdnet listener and an engine: {ls:?}");
    for (addr, port) in &ls {
        assert!(is_loopback(addr), "port {port} listens on {addr}, not loopback: {ls:?}");
    }
    let bridge_port = d.bridge_addr.unwrap().port();
    assert!(ls.iter().any(|(_, p)| *p == bridge_port), "the bridge is listening");

    // Private files: state dir, control socket, bridge token, the identity and its seed.
    assert_eq!(mode(&h.paths.state()), 0o700);
    assert_eq!(mode(&h.paths.socket()), 0o600);
    assert_eq!(mode(&h.paths.token()), 0o600);
    assert_eq!(mode(&h.paths.pool()), 0o700);
    assert_eq!(mode(&h.paths.pool().join("seed")), 0o600);

    // The bridge answers only with the token, and rejects a foreign Host header.
    let http = reqwest::Client::builder().no_proxy().build()?;
    let url = format!("http://127.0.0.1:{bridge_port}/v1/models");
    assert_eq!(http.get(&url).send().await?.status().as_u16(), 401);
    assert_eq!(http.get(&url).bearer_auth("wrong").send().await?.status().as_u16(), 401);
    let token = ymir_core::token::load(&h.paths)?.unwrap();
    assert_eq!(http.get(&url).bearer_auth(&token).send().await?.status().as_u16(), 200);
    let rebind = http.get(&url).bearer_auth(&token).header("Host", "evil.example").send().await?;
    assert!(rebind.status().is_client_error(), "a DNS-rebinding Host is refused, got {}", rebind.status());
    d.shutdown().await;
    Ok(())
}

fn key(n: u8) -> SigningKey {
    SigningKey::from_seed(&[n; 32])
}

#[tokio::test]
async fn ymirctl_updates_stage_a_signed_bundle_and_start_it_on_trial() -> Result<()> {
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
    let bundle = tmp.path().join("bundle");
    std::fs::create_dir_all(&bundle)?;
    std::fs::write(bundle.join("ymird"), b"#!/bin/sh\nexit 0\n")?;
    let m = UpdateManifest {
        app: ymir_core::update::APP.into(),
        channel: "stable".into(),
        update_id: "m9-test".into(),
        seq: 1,
        kind: UpdateKind::Full(FullPackage { platform, arch, format: PackageFormat::TarGz, signing_identity: "unsigned".into() }),
        version: semver::Version::new(99, 0, 0),
        base_releases: vec![],
        files: vec![FileEntry::describe("ymird", b"#!/bin/sh\nexit 0\n")],
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
    std::fs::write(bundle.join("manifest.env"), Signed::sign(Domain::Manifest, &m, &[&rel]).unwrap().to_bytes().unwrap())?;

    // Not pinned yet: nothing is taken.
    let e = ctl(&home, &["update", "stage", bundle.to_str().unwrap()]).await.unwrap_err().to_string();
    assert!(e.contains("no update root"), "{e}");
    let pin = root.pin().to_string();
    ctl(&home, &["update", "trust", "--pin", &pin, "--root", root_file.to_str().unwrap()]).await?;
    // A wrong pin is refused before anything is stored.
    assert!(ctl(&tmp.path().join("other"), &["update", "trust", "--pin", &"00".repeat(32), "--root", root_file.to_str().unwrap()]).await.is_err());

    let out = ctl(&home, &["update", "stage", bundle.to_str().unwrap()]).await?;
    assert!(out.contains("staged m9-test 99.0.0"), "{out}");
    // The next start picks the release on trial; reaching ready (what ymird does once serving) commits it.
    let path = ctl(&home, &["update", "boot"]).await?;
    assert!(path.trim().ends_with("releases/m9-test/ymird"), "{path}");
    assert_eq!(std::fs::read(path.trim())?, b"#!/bin/sh\nexit 0\n");
    ymir_core::update::commit_if_on_trial(&Paths::new(&home));
    let st = ctl(&home, &["update", "status"]).await?;
    assert!(st.contains("good release: m9-test") && st.contains("pinned"), "{st}");
    Ok(())
}
