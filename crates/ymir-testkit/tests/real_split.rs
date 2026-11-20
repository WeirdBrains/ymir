//! M7 with the real thing: llama.cpp's `rpc-server` as a CPU worker and `llama-server` as the
//! leader on the GPU, splitting a real model between two model nodes over loopback. Ignored by
//! default; it backs off (passes with a note) when the GPU is busy or too full:
//!
//! YMIR_TEST_LLAMA_SERVER=/path/llama-server YMIR_TEST_RPC_SERVER=/path/ggml-rpc-server \
//!   YMIR_TEST_GGUF=/path/model.gguf cargo test -p ymir-testkit --test real_split -- --ignored --nocapture
//!
//! It prints tokens/s for the split run and for the same model unsplit, same server, same prompt.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use clap::Parser as _;
use serde_json::{Value, json};
use weirdnet_net::{Keypair, WEIRDNET};
use weirdnet_sync::Roster;
use ymir_core::control::{self, ControlRequest};
use ymir_core::membership::Membership;
use ymir_core::{Daemon, NodeConfig, Paths};

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
    let llama = std::env::var("YMIR_TEST_LLAMA_SERVER").unwrap();
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
        "[node]\nname = \"{name}\"\n[roles]\nmodel = {m}\ninference = {i}\napp = {a}\n[network]\nlisten = [\"/ip4/127.0.0.1/tcp/0\"]\nmdns = false\ndiscovery = false\nrelays = []\nsync_secs = 1\npeers = [{peers}]\n[model]\nllamacpp = \"system:{llama}\"\nvllm = \"off\"\nmodels_dir = \"{}\"\n{local}[inference.alerts]\nprobe_secs = 3600\n[app.bridge]\nport = 0\n",
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

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

fn append(h: &Home, text: &str) -> Result<()> {
    std::fs::OpenOptions::new().append(true).open(h.home.join("node.toml"))?.write_all(text.as_bytes())?;
    Ok(())
}

async fn peer(h: &Home) -> Result<String> {
    Ok(control::request(&h.paths.socket(), &ControlRequest::Status).await?.status.unwrap().network.unwrap().peer)
}

/// (free MiB, utilization %) of GPU 0, and how many other compute processes are on it.
fn gpu() -> Option<(u64, u64, usize)> {
    let q = |args: &[&str]| std::process::Command::new("nvidia-smi").args(args).output().ok().map(|o| String::from_utf8_lossy(&o.stdout).to_string());
    let a = q(&["--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"])?;
    let mut it = a.trim().split(',').map(|x| x.trim().parse::<u64>().unwrap_or(u64::MAX));
    let (free, util) = (it.next()?, it.next()?);
    let procs = q(&["--query-compute-apps=pid", "--format=csv,noheader"])?.lines().filter(|l| !l.trim().is_empty()).count();
    Some((free, util, procs))
}

/// Completion tokens per second for one fixed prompt against a llama-server.
async fn tokens_per_sec(port: u16, key: Option<&str>) -> Result<(f64, u64)> {
    let http = reqwest::Client::builder().no_proxy().build()?;
    let mut best = (0.0, 0);
    for _ in 0..2 {
        let mut r = http.post(format!("http://127.0.0.1:{port}/v1/chat/completions"));
        if let Some(k) = key {
            r = r.bearer_auth(k);
        }
        let t = Instant::now();
        let v: Value = r
            .json(&json!({"messages": [{"role": "user", "content": "Write a short story about a lighthouse keeper. /no_think"}], "max_tokens": 96, "temperature": 0}))
            .send()
            .await?
            .json()
            .await?;
        let n = v["usage"]["completion_tokens"].as_u64().unwrap_or(0);
        let tps = n as f64 / t.elapsed().as_secs_f64();
        if tps > best.0 {
            best = (tps, n);
        }
    }
    Ok(best)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs YMIR_TEST_LLAMA_SERVER, YMIR_TEST_RPC_SERVER, YMIR_TEST_GGUF and a quiet GPU"]
async fn a_real_model_is_split_between_two_nodes() -> Result<()> {
    let (llama, rpc, gguf) = (std::env::var("YMIR_TEST_LLAMA_SERVER")?, std::env::var("YMIR_TEST_RPC_SERVER")?, std::env::var("YMIR_TEST_GGUF")?);
    match gpu() {
        Some((free, util, procs)) if free >= 7168 && util <= 25 => eprintln!("GPU ok: {free} MiB free, {util}% busy, {procs} other process(es)"),
        other => {
            eprintln!("backing off: the GPU is busy or too full (free MiB, util %, procs): {other:?}");
            return Ok(());
        }
    }
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();
    let inf_tmp = home(root, "inf", "inference app", &[], &[])?;
    let first = Membership::create(&inf_tmp.paths.pool(), "inf")?;
    let (lead_tmp, work_tmp) = (home(root, "lead", "model", &[], &[])?, home(root, "work", "model", &[], &[])?);
    for (h, label) in [(&lead_tmp, "lead"), (&work_tmp, "work")] {
        prelink(&first, h, label);
    }
    drop(first);
    let (serve, rpc_port) = (free_port(), free_port());
    for (h, leader) in [(&lead_tmp, true), (&work_tmp, false)] {
        let p = h.home.join("node.toml");
        let t = std::fs::read_to_string(&p)?;
        std::fs::write(
            &p,
            t.replacen(
                "vllm = \"off\"\n",
                &format!(
                    "vllm = \"off\"\nprivate_link = \"127.0.0.1\"\nrpc_server = \"system:{rpc}\"\nrpc_args = [\"-d\", \"CPU\", \"-t\", \"4\"]\n"
                ),
                1,
            ),
        )?;
        if leader {
            append(
                h,
                &format!(
                    "[[model.split]]\nid = \"big\"\nfile = \"{gguf}\"\nserve_port = {serve}\ncontext = 2048\ngpu_layers = 99\nargs = [\"--tensor-split\", \"3,1\"]\n"
                ),
            )?;
        }
    }
    let (lead, work) = (start(&lead_tmp).await?, start(&work_tmp).await?);
    let (lead_peer, work_peer) = (peer(&lead_tmp).await?, peer(&work_tmp).await?);
    let models = vec![addr(&lead_tmp).await?, addr(&work_tmp).await?];
    let inf_h = home(root, "inf", "inference app", &models, &[])?;
    append(
        &inf_h,
        &format!(
            "[[inference.split]]\nid = \"big\"\nsize = \"9GiB\"\nleader = \"{lead_peer}\"\n\
             [[inference.split.node]]\nnode = \"{lead_peer}\"\nlink = \"127.0.0.1\"\nmemory = \"8GiB\"\n\
             [[inference.split.node]]\nnode = \"{work_peer}\"\nlink = \"127.0.0.1\"\nport = {rpc_port}\nmemory = \"8GiB\"\n"
        ),
    )?;
    let inf = start(&inf_h).await?;
    eventually(60, "both model nodes join", async || Ok(ctl(&inf_h.home, &["nodes"]).await?.matches("serving").count() == 2)).await?;

    let t = Instant::now();
    let up = ctl(&inf_h.home, &["group", "up", "big"]).await?;
    eprintln!("{} (up in {:.1}s)", up.trim(), t.elapsed().as_secs_f64());
    let a =
        ctl(&inf_h.home, &["chat", "--model", "big", "--wait", "--max-tokens", "40", "What is the capital of France? Answer in one word. /no_think"])
            .await?;
    eprintln!("through the pool: {:?}", a.trim());
    assert!(a.to_lowercase().contains("paris"), "{a}");
    let key = std::fs::read_to_string(Paths::new(&lead_tmp.home).state().join("split.key"))?;
    let (split_tps, n) = tokens_per_sec(serve, Some(key.trim())).await?;
    eprintln!("SPLIT   (GPU 3 : CPU worker 1, loopback): {split_tps:.1} tokens/s over {n} tokens");

    ctl(&inf_h.home, &["group", "down", "big"]).await?;
    eventually(30, "the worker's rpc-server exits", async || Ok(std::net::TcpStream::connect(("127.0.0.1", rpc_port)).is_err())).await?;
    inf.shutdown().await;
    lead.shutdown().await;
    work.shutdown().await;

    // Unsplit baseline: the same server and model on the GPU alone (only if it is still quiet).
    if let Some((free, util, _)) = gpu()
        && free >= 7168
        && util <= 25
    {
        let port = free_port();
        let mut child = tokio::process::Command::new(&llama)
            .args(["--model", &gguf, "--host", "127.0.0.1", "--port", &port.to_string(), "--ctx-size", "2048", "--n-gpu-layers", "99", "--no-webui"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true)
            .spawn()?;
        let http = reqwest::Client::builder().no_proxy().build()?;
        eventually(180, "the baseline server loads", async || {
            Ok(http.get(format!("http://127.0.0.1:{port}/health")).send().await.is_ok_and(|r| r.status() == 200))
        })
        .await?;
        let (tps, n) = tokens_per_sec(port, None).await?;
        eprintln!("UNSPLIT (GPU alone):                       {tps:.1} tokens/s over {n} tokens");
        let _ = child.kill().await;
    } else {
        eprintln!("the GPU got busy; no unsplit baseline");
    }
    Ok(())
}
