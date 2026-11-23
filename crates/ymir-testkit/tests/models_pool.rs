//! `ymirctl models --node`: model search, pull, serve and remove over Weirdnet on 127.0.0.1
//! (nothing leaves the box). Real daemons with throwaway homes and identities, a mock engine and a
//! mock Hub.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use clap::Parser as _;
use weirdnet_net::{Keypair, WEIRDNET};
use weirdnet_sync::Roster;
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
    home: PathBuf,
    paths: Paths,
}

/// node.toml for `roles`, on Weirdnet over 127.0.0.1 only, dialing `peers`.
fn home(root: &Path, name: &str, roles: &str, peers: &[String], models: &[&str]) -> Result<Home> {
    let home = root.join(name);
    let models_dir = root.join(format!("{name}-models"));
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
    Ok(Home { paths: Paths::new(&home), home })
}

/// What `pool link` and `pool add` do, without the network: `from` certifies a new node for
/// `to` and the identity lands there.
fn prelink(from: &Membership, to: &Home, label: &str) -> Membership {
    let key = Keypair::generate_ed25519();
    let mut seed = from.seed().unwrap().0;
    let user = Keypair::ed25519_from_bytes(&mut seed).unwrap();
    let cert = Roster::self_certify(&WEIRDNET, &user, &key.public().to_peer_id().to_string(), label, 1).unwrap();
    from.approve(cert.clone()).unwrap();
    Membership::adopt(&to.paths.pool(), key, &from.seed().unwrap().0, cert, &from.link_settings()).unwrap()
}

async fn start(h: &Home) -> Result<Daemon> {
    Daemon::start(NodeConfig::load(&h.paths.config())?, h.paths.clone()).await
}

/// `/ip4/127.0.0.1/tcp/PORT/p2p/PEER` for a running daemon, from its control status.
async fn addr(h: &Home) -> Result<String> {
    let s = control::request(&h.paths.socket(), &ControlRequest::Status).await?.status.unwrap();
    let n = s.network.expect("on Weirdnet");
    let a = n.listening.iter().find(|a| a.starts_with("/ip4/127.0.0.1/tcp/")).expect("a loopback listener");
    let a = a.split("/p2p/").next().unwrap();
    Ok(format!("{a}/p2p/{}", n.peer))
}

async fn peer_of(h: &Home) -> Result<String> {
    Ok(control::request(&h.paths.socket(), &ControlRequest::Status).await?.status.unwrap().network.expect("on Weirdnet").peer)
}

// ---- models over Weirdnet ----

/// A Hub with repo `o/n`: a 4000-byte Q4_0 and a 6000-byte Q8_0, and a search for it.
fn hub() -> String {
    use std::io::BufRead;
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let info = r#"{"id":"o/n","siblings":[{"rfilename":"m-Q4_0.gguf","size":4000},{"rfilename":"m-Q8_0.gguf","size":6000}]}"#;
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { break };
            let mut reader = std::io::BufReader::new(stream.try_clone().unwrap());
            let mut first = String::new();
            reader.read_line(&mut first).unwrap();
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 || line.trim().is_empty() {
                    break;
                }
            }
            let path = first.split_whitespace().nth(1).unwrap_or("/").split('?').next().unwrap_or("").to_string();
            let body: Option<Vec<u8>> = match path.as_str() {
                "/api/models" => Some(br#"[{"id":"o/n","downloads":9,"likes":1}]"#.to_vec()),
                "/api/models/o/n" => Some(info.into()),
                "/o/n/resolve/main/m-Q4_0.gguf" => Some(gguf(4000)),
                "/o/n/resolve/main/m-Q8_0.gguf" => Some(gguf(6000)),
                "/api/models/o/tune" => {
                    Some(br#"{"id":"o/tune","tags":["gguf"],"siblings":[{"rfilename":"tune-LoRA-F16.gguf","size":600}]}"#.to_vec())
                }
                "/o/tune/resolve/main/tune-LoRA-F16.gguf" => Some(lora(600)),
                _ => None,
            };
            let _ = match body {
                Some(b) => {
                    let _ = stream.write_all(format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", b.len()).as_bytes());
                    stream.write_all(&b)
                }
                None => stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"),
            };
        }
    });
    url
}

/// A GGUF header (llama, one block, one KV head of size 1) padded to `len` bytes.
fn gguf(len: usize) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend(0u64.to_le_bytes());
    let kv: [(&str, Option<u32>); 5] = [
        ("general.architecture", None),
        ("llama.block_count", Some(1)),
        ("llama.attention.head_count", Some(1)),
        ("llama.attention.head_count_kv", Some(1)),
        ("llama.attention.key_length", Some(1)),
    ];
    out.extend((kv.len() as u64).to_le_bytes());
    for (k, v) in kv {
        out.extend((k.len() as u64).to_le_bytes());
        out.extend(k.as_bytes());
        match v {
            None => {
                out.extend(8u32.to_le_bytes());
                out.extend(5u64.to_le_bytes());
                out.extend(b"llama");
            }
            Some(n) => {
                out.extend(4u32.to_le_bytes());
                out.extend(n.to_le_bytes());
            }
        }
    }
    out.resize(len, 0);
    out
}

/// A LoRA adapter GGUF header for a llama base, padded to `len` bytes.
fn lora(len: usize) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend(0u64.to_le_bytes());
    out.extend(3u64.to_le_bytes());
    for (k, v) in [("general.architecture", "llama"), ("general.type", "adapter"), ("adapter.type", "lora")] {
        out.extend((k.len() as u64).to_le_bytes());
        out.extend(k.as_bytes());
        out.extend(8u32.to_le_bytes());
        out.extend((v.len() as u64).to_le_bytes());
        out.extend(v.as_bytes());
    }
    out.resize(len, 0);
    out
}

/// Retries `f` until it is true or `secs` pass.
async fn eventually(secs: u64, what: &str, mut f: impl AsyncFnMut() -> Result<bool>) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(secs);
    loop {
        if f().await? {
            return Ok(());
        }
        if Instant::now() > deadline {
            bail!("gave up waiting for: {what}");
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
}

/// `ymirctl models --node` from one machine reaches the model nodes among the user's devices:
/// `auto` picks the node with room, a pull streams progress and (with `--serve`) the model is served
/// by that node's running ymird without a restart, `remove` stops serving it, and node.toml on the
/// node is never edited.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_pool_searches_pulls_serves_and_removes_models_over_weirdnet() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();
    // The Hub is a local mock, and the token (if this machine has one) is only sent to the public Hub.
    unsafe { std::env::set_var("WEIRDNET_HF_URL", hub()) };

    let ctl_h = home(root, "ctl", "inference", &[], &[])?;
    let first = Membership::create(&ctl_h.paths.pool(), "ctl")?;
    let big_h = home(root, "big", "model", &[], &[])?;
    let tiny_h = home(root, "tiny", "model", &[], &[])?;
    for (h, label) in [(&big_h, "big"), (&tiny_h, "tiny")] {
        prelink(&first, h, label);
    }
    drop(first);
    // `tiny` has almost no model memory, so nothing fits there.
    let toml = std::fs::read_to_string(tiny_h.paths.config())?.replace("[model]\n", "[model]\nmemory_budget = \"1KiB\"\n");
    std::fs::write(tiny_h.paths.config(), toml)?;
    let big_toml_before = std::fs::read_to_string(big_h.paths.config())?;

    let big = start(&big_h).await?;
    let tiny = start(&tiny_h).await?;
    let peers = vec![addr(&big_h).await?, addr(&tiny_h).await?];
    let ctl_h = home(root, "ctl", "inference", &peers, &[])?;
    let ctl_d = start(&ctl_h).await?;
    let big_peer = peer_of(&big_h).await?;

    // Both model nodes are found, and each says what it holds.
    eventually(40, "both model nodes answer models/1", async || {
        let out = ctl(&ctl_h.home, &["models", "list", "--node", "all"]).await.unwrap_or_default();
        Ok(out.contains("Models on big:") && out.contains("Models on tiny:"))
    })
    .await?;

    // Search shows each node's own verdict.
    let out = ctl(&ctl_h.home, &["models", "search", "m", "--node", "all"]).await?;
    assert!(out.contains("o/n  (GGUF, 9 downloads, 1 likes)"), "{out}");
    assert!(out.contains("big: best: Q8_0") && out.contains(": fits"), "{out}");
    assert!(out.contains("tiny: nothing fits"), "{out}");

    // `auto` picks the node with room and says why; the progress streams.
    let out = ctl(&ctl_h.home, &["models", "pull", "o/n", "--node", "auto"]).await?;
    assert!(out.contains("Picked big: it can hold m-Q8_0"), "{out}");
    assert!(out.contains("big: 100% (6000 B") || out.contains("big: 100%"), "{out}");
    assert!(out.contains("big: pulled m-Q8_0") && out.contains("not serving yet"), "{out}");
    assert!(out.contains("[[model.local]]") && out.contains("file = \"m-Q8_0.gguf\""), "{out}");
    assert!(root.join("big-models/m-Q8_0.gguf").is_file());

    // `--serve`: the running ymird serves it right away, no restart.
    let out = ctl(&ctl_h.home, &["models", "pull", "o/n", "--file", "m-Q4_0.gguf", "--node", &big_peer, "--serve"]).await?;
    assert!(out.contains("big: serving `m-Q4_0` now"), "{out}");
    let out = ctl(&ctl_h.home, &["models", "list", "--node", "big"]).await?;
    assert!(out.lines().any(|l| l.contains("m-Q4_0") && l.contains("serving as `m-Q4_0`")), "{out}");
    assert!(out.lines().any(|l| l.contains("m-Q8_0") && l.contains("not served")), "{out}");
    let status = ctl(&big_h.home, &["status"]).await?;
    assert!(status.contains("m-Q4_0"), "the node's own status lists the new model: {status}");

    // Only a node that fits is asked to take a model it cannot hold.
    let err = format!("{:#}", ctl(&ctl_h.home, &["models", "pull", "o/n", "--node", "tiny"]).await.unwrap_err());
    assert!(err.contains("could not do it"), "{err}");

    // A name that is not on the roster, and options that only make sense locally.
    let err = format!("{:#}", ctl(&ctl_h.home, &["models", "list", "--node", "nobody"]).await.unwrap_err());
    assert!(err.contains("no device named `nobody`"), "{err}");
    let err = format!("{:#}", ctl(&ctl_h.home, &["models", "pull", "o/n", "--node", "big", "--register"]).await.unwrap_err());
    assert!(err.contains("--register edits node.toml on the machine you run it on"), "{err}");
    let err = format!("{:#}", ctl(&ctl_h.home, &["models", "pull", "o/n", "--serve"]).await.unwrap_err());
    assert!(err.contains("--serve and --load need --node"), "{err}");

    // Remove stops serving and deletes the files.
    let out = ctl(&ctl_h.home, &["models", "remove", "m-Q4_0", "--node", "big", "--yes"]).await?;
    assert!(out.contains("big: deleted m-Q4_0"), "{out}");
    assert!(!root.join("big-models/m-Q4_0.gguf").exists());
    assert!(!ctl(&big_h.home, &["status"]).await?.contains("m-Q4_0"));

    // node.toml on the node was never edited from afar.
    assert_eq!(std::fs::read_to_string(big_h.paths.config())?, big_toml_before);

    ctl_d.shutdown().await;
    tiny.shutdown().await;
    big.shutdown().await;
    Ok(())
}

/// A LoRA adapter pulled from the Hub onto a base another device serves (training plan P4a): the
/// node checks it against the base, serves it as its own model without a restart, lists it with
/// its base, and a chat on this machine for the adapter's id runs on that base with it applied.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_adapter_is_pulled_onto_a_served_base_and_routed_by_its_name() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();
    unsafe { std::env::set_var("WEIRDNET_HF_URL", hub()) };

    let ctl_h = home(root, "ctl", "inference,app", &[], &[])?;
    let first = Membership::create(&ctl_h.paths.pool(), "ctl")?;
    let big_h = home(root, "big", "model", &[], &[])?;
    prelink(&first, &big_h, "big");
    drop(first);
    let big = start(&big_h).await?;
    let ctl_h = home(root, "ctl", "inference,app", &[addr(&big_h).await?], &[])?;
    let ctl_d = start(&ctl_h).await?;
    eventually(40, "big answers models/1", async || Ok(ctl(&ctl_h.home, &["models", "list", "--node", "big"]).await.is_ok())).await?;

    // The base first, then the adapter for it.
    let out = ctl(&ctl_h.home, &["models", "pull", "o/n", "--file", "m-Q4_0.gguf", "--node", "big", "--serve"]).await?;
    assert!(out.contains("big: serving `m-Q4_0` now"), "{out}");
    let err = format!("{:#}", ctl(&ctl_h.home, &["models", "pull", "o/tune", "--node", "auto", "--adapter-for", "m-Q4_0"]).await.unwrap_err());
    assert!(err.contains("needs the node that serves the base"), "{err}");
    let err = format!("{:#}", ctl(&ctl_h.home, &["models", "pull", "o/tune", "--adapter-for", "m-Q4_0"]).await.unwrap_err());
    assert!(err.contains("--adapter-for needs --node"), "{err}");
    let out = ctl(&ctl_h.home, &["models", "pull", "o/tune", "--node", "big", "--adapter-for", "m-Q4_0", "--serve"]).await?;
    assert!(out.contains("big: serving `tune-LoRA-F16` now, on `m-Q4_0`"), "{out}");
    assert!(out.contains("ymirctl models adapter add m-Q4_0 adapters/tune-LoRA-F16/tune-LoRA-F16.gguf --id tune-LoRA-F16"), "{out}");
    assert!(root.join("big-models/adapters/tune-LoRA-F16/tune-LoRA-F16.gguf").is_file());
    let out = ctl(&ctl_h.home, &["models", "list", "--node", "big"]).await?;
    assert!(out.lines().any(|l| l.contains("tune-LoRA-F16") && l.contains("adapter `tune-LoRA-F16` on `m-Q4_0`")), "{out}");

    // Routed by its name from this machine, through the pool, to the base's engine on big.
    let mut reply = String::new();
    eventually(40, "the adapter answers through the pool", async || {
        reply = ctl(&ctl_h.home, &["chat", "--model", "tune-LoRA-F16", "hi"]).await.unwrap_or_default();
        Ok(!reply.is_empty())
    })
    .await?;
    assert_eq!(reply.trim(), "Hello from m-Q4_0 with tune-LoRA-F16 at 1! You said: hi");
    let reply = ctl(&ctl_h.home, &["chat", "--model", "m-Q4_0", "hi"]).await?;
    assert_eq!(reply.trim(), "Hello from m-Q4_0! You said: hi");

    // Live changes, with the base running: a scale for one answer, then for the adapter, then
    // off and back on (llama-server still has the file, so no restart).
    let reply = ctl(&ctl_h.home, &["chat", "--model", "tune-LoRA-F16", "--adapter-scale", "0.5", "hi"]).await?;
    assert_eq!(reply.trim(), "Hello from m-Q4_0 with tune-LoRA-F16 at 0.5! You said: hi");
    let err = format!("{:#}", ctl(&ctl_h.home, &["models", "adapter", "scale", "tune-LoRA-F16", "0.25"]).await.unwrap_err());
    assert!(err.contains("name it with --node"), "{err}");
    let out = ctl(&ctl_h.home, &["models", "adapter", "scale", "tune-LoRA-F16", "0.25", "--node", "big"]).await?;
    assert!(out.contains("big: `tune-LoRA-F16` now applies at scale 0.25"), "{out}");
    let reply = ctl(&ctl_h.home, &["chat", "--model", "tune-LoRA-F16", "hi"]).await?;
    assert_eq!(reply.trim(), "Hello from m-Q4_0 with tune-LoRA-F16 at 0.25! You said: hi");
    let out = ctl(&ctl_h.home, &["models", "adapter", "unserve", "tune-LoRA-F16", "--node", "big"]).await?;
    assert!(out.contains("big: stopped serving `tune-LoRA-F16` on `m-Q4_0`; its files are kept"), "{out}");
    assert!(root.join("big-models/adapters/tune-LoRA-F16/tune-LoRA-F16.gguf").is_file());
    let out = ctl(&ctl_h.home, &["models", "adapter", "serve", "tune-LoRA-F16", "--base", "m-Q4_0", "--node", "big"]).await?;
    assert!(out.contains("big: serving `tune-LoRA-F16` now, on `m-Q4_0` (no restart)"), "{out}");
    let mut reply = String::new();
    eventually(40, "the adapter is back in the pool", async || {
        reply = ctl(&ctl_h.home, &["chat", "--model", "tune-LoRA-F16", "hi"]).await.unwrap_or_default();
        Ok(reply.contains("with tune-LoRA-F16"))
    })
    .await?;
    assert_eq!(reply.trim(), "Hello from m-Q4_0 with tune-LoRA-F16 at 1! You said: hi");

    ctl_d.shutdown().await;
    big.shutdown().await;
    Ok(())
}
