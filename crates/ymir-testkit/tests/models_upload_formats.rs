//! `ymirctl models upload PATH --node X` for the formats of slice U7 (docs/design/model-upload.md):
//! a safetensors directory to a vLLM node, a PEFT adapter onto a vLLM base, and a GGUF vision
//! projector for a llama.cpp model, each placed by the node and saved in its node.toml. Real
//! daemons on 127.0.0.1 with throwaway homes and identities and a mock engine.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use clap::Parser as _;
use weirdnet_net::{Keypair, WEIRDNET};
use weirdnet_sync::Roster;
use ymir_core::control::{self, ControlRequest};
use ymir_core::membership::Membership;
use ymir_core::upload;
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
fn home(root: &Path, name: &str, roles: &str, peers: &[String], models: &[&str], extra: &str) -> Result<Home> {
    home_with(root, name, roles, peers, models, &format!("llamacpp = \"system:{MOCK}\"\nvllm = \"off\"\n"), extra)
}

/// [`home`] with the `[model]` engine lines given.
fn home_with(root: &Path, name: &str, roles: &str, peers: &[String], models: &[&str], engines: &str, extra: &str) -> Result<Home> {
    let home = root.join(name);
    let models_dir = root.join(format!("{name}-models"));
    std::fs::create_dir_all(&home)?;
    std::fs::create_dir_all(&models_dir)?;
    let mut local = String::new();
    for (i, id) in models.iter().enumerate() {
        std::fs::write(models_dir.join(format!("{id}.gguf")), gguf(4000, "llama"))?;
        let role = if i == 0 { "role = \"main\"\n" } else { "" };
        local.push_str(&format!("[[model.local]]\nid = \"{id}\"\nfile = \"{id}.gguf\"\nrevision = \"r1\"\n{role}"));
    }
    let peers = peers.iter().map(|p| format!("\"{p}\"")).collect::<Vec<_>>().join(", ");
    let (m, i, a) = (roles.contains("model"), roles.contains("inference"), roles.contains("app"));
    let toml = format!(
        "[node]\nname = \"{name}\"\n[roles]\nmodel = {m}\ninference = {i}\napp = {a}\n[network]\nlisten = [\"/ip4/127.0.0.1/tcp/0\"]\nmdns = false\ndiscovery = false\nrelays = []\npeers = [{peers}]\n[model]\n{engines}models_dir = \"{}\"\n{extra}{local}[inference.alerts]\nprobe_secs = 3600\n[app.bridge]\nport = 0\n",
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

/// A GGUF header (llama, one block, one KV head of size 1) padded to `len` bytes.
fn gguf(len: usize, arch: &str) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend(0u64.to_le_bytes());
    let kv: [(&str, Option<u32>); 5] = [
        ("general.architecture", None),
        ("block_count", Some(1)),
        ("attention.head_count", Some(1)),
        ("attention.head_count_kv", Some(1)),
        ("attention.key_length", Some(1)),
    ];
    out.extend((kv.len() as u64).to_le_bytes());
    for (k, v) in kv {
        let k = if v.is_some() { format!("{arch}.{k}") } else { k.to_string() };
        out.extend((k.len() as u64).to_le_bytes());
        out.extend(k.as_bytes());
        match v {
            None => {
                out.extend(8u32.to_le_bytes());
                out.extend((arch.len() as u64).to_le_bytes());
                out.extend(arch.as_bytes());
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

/// A safetensors file with `n` F32 weights.
fn safetensors(n: u64) -> Vec<u8> {
    let header = serde_json::json!({"w": {"dtype": "F32", "shape": [n], "data_offsets": [0, n * 4]}});
    let h = serde_json::to_vec(&header).unwrap();
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(vec![7u8; (n * 4) as usize]);
    out
}

/// A GGUF vision projector header (`clip`, `mmproj`), padded to `len` bytes.
fn mmproj(len: usize) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend(0u64.to_le_bytes());
    out.extend(2u64.to_le_bytes());
    for (k, v) in [("general.architecture", "clip"), ("general.type", "mmproj")] {
        out.extend((k.len() as u64).to_le_bytes());
        out.extend(k.as_bytes());
        out.extend(8u32.to_le_bytes());
        out.extend((v.len() as u64).to_le_bytes());
        out.extend(v.as_bytes());
    }
    out.resize(len, 0);
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn directories_adapters_and_projectors_are_uploaded_and_saved() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();
    let lap_h = home(root, "lap", "inference", &[], &[], "")?;
    let first = Membership::create(&lap_h.paths.pool(), "lap")?;
    let gpu_h = home(root, "gpu", "model", &[], &["base"], "upload_keep_free = \"0\"\n")?;
    let vbase = "[[model.local]]\nid = \"vbase\"\nbackend = \"vllm\"\nfile = \"Qwen/Qwen3-8B\"\nrole = \"main\"\nlora_slots = 2\n";
    let vl_h = home_with(
        root,
        "vl",
        "model",
        &[],
        &[],
        &format!("llamacpp = \"off\"\nvllm = \"system:{MOCK}\"\n"),
        &format!("upload_keep_free = \"0\"\n{vbase}"),
    )?;
    prelink(&first, &gpu_h, "gpu");
    prelink(&first, &vl_h, "vl");
    drop(first);
    let gpu = start(&gpu_h).await?;
    let vl = start(&vl_h).await?;
    let lap_h = home(root, "lap", "inference", &[addr(&gpu_h).await?, addr(&vl_h).await?], &[], "")?;
    let lap = start(&lap_h).await?;
    eventually(40, "gpu answers models/1", async || Ok(ctl(&lap_h.home, &["models", "list", "--node", "gpu"]).await.is_ok())).await?;
    eventually(40, "vl answers models/1", async || Ok(ctl(&lap_h.home, &["models", "list", "--node", "vl"]).await.is_ok())).await?;

    let files = root.join("files");
    let st = files.join("Tune-8B");
    std::fs::create_dir_all(&st)?;
    std::fs::write(st.join("config.json"), br#"{"architectures":["Qwen3ForCausalLM"],"_name_or_path":"Qwen/Qwen3-8B"}"#)?;
    std::fs::write(st.join("tokenizer.json"), b"{}")?;
    std::fs::write(st.join("model.safetensors"), safetensors(20_000))?;
    let st_s = st.display().to_string();

    // A safetensors directory is refused by a llama.cpp node, and placed and saved by a vLLM one.
    let mut out = Vec::new();
    let err = ctl_with(&lap_h.home, &["models", "upload", &st_s, "--node", "gpu"], &mut out, "").await;
    let out = String::from_utf8(out)?;
    assert!(err.is_err(), "{out}");
    assert!(!root.join("gpu-models/safetensors/tune-8b").exists());
    let out = ctl(&lap_h.home, &["models", "upload", &st_s, "--node", "vl", "--license", "apache-2.0"]).await?;
    assert!(out.contains("Tune-8B: safetensors model (Qwen3ForCausalLM)") && out.contains("as `tune-8b`"), "{out}");
    assert!(out.contains("vl: saved in its node.toml"), "{out}");
    let dir = root.join("vl-models/safetensors/tune-8b");
    assert_eq!(std::fs::read(dir.join("model.safetensors"))?, std::fs::read(st.join("model.safetensors"))?);
    let prov = upload::read_provenance(&dir).expect("provenance.json beside it");
    assert_eq!(prov.license.as_deref(), Some("apache-2.0"));
    let cfg = NodeConfig::load(&vl_h.paths.config())?;
    let m = cfg.model.local.iter().find(|m| m.id == "tune-8b").expect("saved in node.toml");
    assert_eq!((m.backend.as_deref(), m.revision.as_deref()), (Some("vllm"), Some(prov.revision.as_str())));
    assert_eq!(m.file.as_deref().map(PathBuf::from), Some(std::fs::canonicalize(&dir)?));

    // A PEFT adapter goes on the vLLM base; without --adapter-for it is refused here.
    let peft = files.join("support-lora");
    std::fs::create_dir_all(&peft)?;
    std::fs::write(peft.join("adapter_config.json"), br#"{"peft_type":"LORA","r":8,"lora_alpha":16,"base_model_name_or_path":"Qwen/Qwen3-8B"}"#)?;
    std::fs::write(peft.join("adapter_model.safetensors"), safetensors(3_000))?;
    let peft_s = peft.display().to_string();
    let err = format!("{:#}", ctl(&lap_h.home, &["models", "upload", &peft_s, "--node", "vl"]).await.unwrap_err());
    assert!(err.contains("--adapter-for BASE"), "{err}");
    let out = ctl(&lap_h.home, &["models", "upload", &peft_s, "--node", "vl", "--adapter-for", "vbase"]).await?;
    assert!(out.contains("support-lora: PEFT LoRA adapter (?)") && out.contains("as `vbase+support-lora`"), "{out}");
    assert!(out.contains("vl: saved in its node.toml"), "{out}");
    let cfg = NodeConfig::load(&vl_h.paths.config())?;
    let b = cfg.model.local.iter().find(|m| m.id == "vbase").unwrap();
    let a = b.lora.iter().find(|a| a.id == "vbase+support-lora").expect("on vbase in node.toml");
    assert!(root.join("vl-models").join(&a.file).join("adapter_config.json").is_file(), "{}", a.file);

    // A vision projector goes with a llama.cpp model and is saved as its mmproj.
    let mm = files.join("mmproj-F16.gguf");
    std::fs::write(&mm, mmproj(50_000))?;
    let mm_s = mm.display().to_string();
    let err = format!("{:#}", ctl(&lap_h.home, &["models", "upload", &mm_s, "--node", "gpu"]).await.unwrap_err());
    assert!(err.contains("--projector-for MODEL"), "{err}");
    let err = format!("{:#}", ctl(&lap_h.home, &["models", "upload", &st_s, "--node", "vl", "--projector-for", "vbase"]).await.unwrap_err());
    assert!(err.contains("not a vision projector"), "{err}");
    let out = ctl(&lap_h.home, &["models", "upload", &mm_s, "--node", "gpu", "--projector-for", "base"]).await?;
    assert!(out.contains("mmproj-F16.gguf: GGUF vision projector (clip)") && out.contains(", for `base`"), "{out}");
    assert!(out.contains("gpu: saved in its node.toml"), "{out}");
    let cfg = NodeConfig::load(&gpu_h.paths.config())?;
    let b = cfg.model.local.iter().find(|m| m.id == "base").unwrap();
    let p = b.mmproj.as_deref().expect("mmproj saved on base");
    assert_eq!(std::fs::read(root.join("gpu-models").join(p))?, std::fs::read(&mm)?);
    // U9: without --node, on a machine that serves models, a directory is placed here and saved
    // in this node.toml (it serves after a restart); the same projector again is already there.
    let out = ctl(&gpu_h.home, &["models", "upload", &st_s, "--name", "tune-local", "--license", "mit"]).await?;
    assert!(out.contains("Placed `tune-local`") && out.contains("Restart ymird to serve it"), "{out}");
    let here = root.join("gpu-models/safetensors/tune-local");
    assert_eq!(std::fs::read(here.join("model.safetensors"))?, std::fs::read(st.join("model.safetensors"))?);
    assert_eq!(upload::read_provenance(&here).and_then(|p| p.license).as_deref(), Some("mit"));
    let cfg = NodeConfig::load(&gpu_h.paths.config())?;
    let m = cfg.model.local.iter().find(|m| m.id == "tune-local").expect("saved in this node.toml");
    assert_eq!(m.backend.as_deref(), Some("vllm"));
    let out = ctl(&gpu_h.home, &["models", "upload", &mm_s, "--projector-for", "base"]).await?;
    assert!(out.contains("nothing was copied") && out.contains("already has"), "{out}");
    // From a machine that serves no models it goes to the model node with the most room.
    // (Both have the same disk here, so either may be chosen; only the vLLM one takes it.)
    let mut buf = Vec::new();
    let sent = ctl_with(&lap_h.home, &["models", "upload", &st_s, "--name", "tune-auto", "--no-persist"], &mut buf, "").await;
    let out = String::from_utf8(buf)?;
    assert!(out.contains("Sending it to the model node with the most room"), "{out}");
    if out.contains("Chose vl") {
        assert!(sent.is_ok() && root.join("vl-models/safetensors/tune-auto/model.safetensors").is_file(), "{out}");
    } else {
        assert!(out.contains("Chose gpu"), "{out}");
    }

    lap.shutdown().await;
    vl.shutdown().await;
    gpu.shutdown().await;
    Ok(())
}
