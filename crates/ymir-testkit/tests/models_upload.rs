//! `ymirctl models upload FILE --node X` (docs/design/model-upload.md, slice U2): a model or LoRA
//! adapter sent from one device to another of the owner's model nodes over Weirdnet on 127.0.0.1
//! (nothing leaves the box). Real daemons with throwaway homes and identities and a mock engine.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use clap::Parser as _;
use weirdnet_net::{Keypair, WEIRDNET};
use weirdnet_sync::Roster;
use ymir_core::control::{self, ControlRequest};
use ymir_core::membership::Membership;
use ymir_core::models::{ModelsOp, Outcome};
use ymir_core::upload::{self, ImportSpec};
use ymir_core::upload_net::UploadOp;
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
        "[node]\nname = \"{name}\"\n[roles]\nmodel = {m}\ninference = {i}\napp = {a}\n[network]\nlisten = [\"/ip4/127.0.0.1/tcp/0\"]\nmdns = false\ndiscovery = false\nrelays = []\npeers = [{peers}]\n[model]\nllamacpp = \"system:{MOCK}\"\nvllm = \"off\"\nmodels_dir = \"{}\"\n{extra}{local}[inference.alerts]\nprobe_secs = 3600\n[app.bridge]\nport = 0\n",
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

/// A LoRA adapter GGUF header for an `arch` base, padded to `len` bytes.
fn lora(len: usize, arch: &str) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend(0u64.to_le_bytes());
    out.extend(3u64.to_le_bytes());
    for (k, v) in [("general.architecture", arch), ("general.type", "adapter"), ("adapter.type", "lora")] {
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

/// Bytes in files under `dir`, recursively (0 when it doesn't exist).
fn bytes_under(dir: &Path) -> u64 {
    let Ok(rd) = std::fs::read_dir(dir) else { return 0 };
    rd.flatten()
        .map(|e| {
            let p = e.path();
            if p.is_dir() { bytes_under(&p) } else { e.metadata().map(|m| m.len()).unwrap_or(0) }
        })
        .sum()
}

/// From a device without a model role to a model node: a model is sent in chunks, placed with its
/// provenance and saved in the node's node.toml (by default); the same files again are "already
/// there"; an interrupted upload resumes; an adapter goes on its base (`--no-persist` leaves
/// node.toml alone) and an adapter for another architecture is refused; a node keeping more free
/// space than it has refuses with `no-space`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn models_and_adapters_are_uploaded_to_another_node_and_saved_there() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();
    let lap_h = home(root, "lap", "inference", &[], &[], "")?;
    let first = Membership::create(&lap_h.paths.pool(), "lap")?;
    let gpu_h = home(root, "gpu", "model", &[], &["base"], "upload_keep_free = \"0\"\n")?;
    let full_h = home(root, "full", "model", &[], &[], "upload_keep_free = \"1000000TiB\"\n")?;
    prelink(&first, &gpu_h, "gpu");
    prelink(&first, &full_h, "full");
    drop(first);
    let gpu = start(&gpu_h).await?;
    let full = start(&full_h).await?;
    let lap_h = home(root, "lap", "inference", &[addr(&gpu_h).await?, addr(&full_h).await?], &[], "")?;
    let lap = start(&lap_h).await?;
    let lap_peer = peer_of(&lap_h).await?;
    eventually(40, "gpu answers models/1", async || Ok(ctl(&lap_h.home, &["models", "list", "--node", "gpu"]).await.is_ok())).await?;

    let files = root.join("files");
    std::fs::create_dir_all(&files)?;
    let model = files.join("My-Model-Q4_0.gguf");
    std::fs::write(&model, gguf(300_000, "llama"))?;
    let model_s = model.display().to_string();

    // Not to this device, and to one node only.
    let err = format!("{:#}", ctl(&lap_h.home, &["models", "upload", &model_s, "--node", "lap"]).await.unwrap_err());
    assert!(err.contains("is this device"), "{err}");
    let err = format!("{:#}", ctl(&lap_h.home, &["models", "upload", &model_s, "--node", "all"]).await.unwrap_err());
    assert!(err.contains("an upload goes to one node"), "{err}");

    // A model: sent, placed with its provenance, saved in gpu's node.toml.
    let out = ctl(&lap_h.home, &["models", "upload", &model_s, "--node", "gpu", "--license", "mit"]).await?;
    assert!(out.contains("My-Model-Q4_0.gguf: model (llama)") && out.contains("as `my-model-q4_0`"), "{out}");
    assert!(out.contains("gpu: 100%"), "{out}");
    assert!(out.contains("gpu: stored `my-model-q4_0`"), "{out}");
    assert!(out.contains("gpu: saved in its node.toml"), "{out}");
    let stored = root.join("gpu-models/my-model-q4_0/My-Model-Q4_0.gguf");
    assert_eq!(std::fs::read(&stored)?, std::fs::read(&model)?);
    let prov = upload::read_provenance(stored.parent().unwrap()).expect("provenance.json beside it");
    assert_eq!((prov.origin.as_str(), prov.license.as_deref()), (upload::ORIGIN_UPLOAD, Some("mit")));
    assert_eq!(prov.uploaded_by.as_deref(), Some(lap_peer.as_str()), "the node records who sent it");
    assert!(out.contains(&format!("revision {}", prov.revision)), "{out}");
    let cfg = NodeConfig::load(&gpu_h.paths.config())?;
    let m = cfg.model.local.iter().find(|m| m.id == "my-model-q4_0").expect("saved in node.toml");
    assert_eq!(m.revision.as_deref(), Some(prov.revision.as_str()));
    assert_eq!(m.role.as_deref(), Some("chat"), "gpu has a main model already");
    assert_eq!(m.file.as_deref(), Some("my-model-q4_0/My-Model-Q4_0.gguf"));
    assert!(gpu_h.paths.config().with_extension("toml.bak").is_file());
    assert!(ctl(&gpu_h.home, &["status"]).await?.contains("my-model-q4_0"), "served by gpu's running ymird");
    // The pool sees the upload's revision: gpu's capability record, as this inference node reads it.
    let want = prov.revision.clone();
    let gpu_peer = peer_of(&gpu_h).await?;
    let mut seen = None;
    eventually(40, "the pool reports the uploaded revision", async || {
        let st = control::request(&lap_h.paths.socket(), &ControlRequest::Status).await?.status.unwrap();
        seen = st.models.iter().find(|m| m.id == "my-model-q4_0" && m.node == gpu_peer).map(|m| m.revision.clone());
        Ok(seen.as_ref().is_some_and(|r| r.as_deref() == Some(want.as_str())))
    })
    .await
    .map_err(|e| anyhow::anyhow!("{e}: last seen {seen:?}"))?;

    // The same files again: nothing is sent, nothing is added twice.
    let out = ctl(&lap_h.home, &["models", "upload", &model_s, "--node", "gpu"]).await?;
    assert!(out.contains("gpu: already has `my-model-q4_0` with the same contents; nothing was sent."), "{out}");
    assert!(out.contains("node.toml already has `my-model-q4_0` at this revision"), "{out}");
    // Other files under that id are refused.
    let other = files.join("other.gguf");
    std::fs::write(&other, gguf(200_000, "llama"))?;
    let mut out = Vec::new();
    let err =
        ctl_with(&lap_h.home, &["models", "upload", &other.display().to_string(), "--node", "gpu", "--name", "my-model-q4_0"], &mut out, "").await;
    let out = String::from_utf8(out)?;
    assert!(err.is_err() && out.contains("gpu refused the upload (exists)") && out.contains("--name"), "{out}");

    // Interrupted: the parts stay on gpu, and the next upload of the same files resumes.
    let big = files.join("Big-Q8_0.gguf");
    std::fs::write(&big, gguf(3 << 20, "llama"))?;
    let cut = root.join("cut");
    std::fs::create_dir_all(&cut)?;
    std::fs::write(cut.join("Big-Q8_0.gguf"), &std::fs::read(&big)?[..(2 << 20)])?;
    let item = upload::inspect(&big)?;
    let spec = ImportSpec { id: "big-q8_0".into(), origin: upload::ORIGIN_UPLOAD.into(), ..Default::default() };
    let up = weirdnet_backends::models::Upload::new("big-q8_0", &item.artifact, upload::provenance(&item, &spec)).serving();
    let op = UploadOp { upload: up, local: vec![cut.join("Big-Q8_0.gguf")], persist: true, role: None };
    let done = control::request_models(&lap_h.paths.socket(), "gpu", ModelsOp::Upload(Box::new(op)), |_| {}).await?;
    assert!(matches!(&done.nodes[0].outcome, Outcome::Failed { .. }), "{done:?}");
    let staging = root.join("gpu-models.uploads");
    eventually(20, "gpu keeps the parts", async || Ok(bytes_under(&staging) > 0)).await?;
    let out = ctl(&lap_h.home, &["models", "upload", &big.display().to_string(), "--node", "gpu", "--no-persist"]).await?;
    assert!(out.contains("gpu: resumed an earlier upload"), "{out}");
    assert!(out.contains("gpu: stored `big-q8_0`") && out.contains("--no-persist"), "{out}");
    assert_eq!(std::fs::read(root.join("gpu-models/big-q8_0/Big-Q8_0.gguf"))?, std::fs::read(&big)?);
    assert!(!NodeConfig::load(&gpu_h.paths.config())?.model.local.iter().any(|m| m.id == "big-q8_0"));

    // An adapter with --no-persist: on its base, node.toml unchanged.
    let before = std::fs::read_to_string(gpu_h.paths.config())?;
    let tune = files.join("tune.gguf");
    std::fs::write(&tune, lora(100_000, "llama"))?;
    let out = ctl(&lap_h.home, &["models", "upload", &tune.display().to_string(), "--node", "gpu", "--adapter-for", "base", "--no-persist"]).await?;
    assert!(out.contains("LoRA adapter (llama)") && out.contains("for `base`"), "{out}");
    assert!(out.contains("is added to `base`"), "{out}");
    assert!(out.contains("node.toml was not changed (--no-persist)"), "{out}");
    assert_eq!(std::fs::read_to_string(gpu_h.paths.config())?, before);

    // An adapter saved: under its base, with the base's revision; node.toml still loads.
    let tune2 = files.join("tune2.gguf");
    std::fs::write(&tune2, lora(120_000, "llama"))?;
    let out = ctl(&lap_h.home, &["models", "upload", &tune2.display().to_string(), "--node", "gpu", "--adapter-for", "base", "--name", "base+tune2"])
        .await?;
    assert!(out.contains("gpu: saved in its node.toml"), "{out}");
    let cfg = NodeConfig::load(&gpu_h.paths.config())?;
    cfg.validate()?;
    let b = cfg.model.local.iter().find(|m| m.id == "base").unwrap();
    let a = b.lora.iter().find(|a| a.id == "base+tune2").expect("saved under its base");
    assert_eq!(a.base_revision.as_deref(), Some("r1"));
    assert!(a.revision.as_deref().is_some_and(|r| r.starts_with("sha256:")), "{a:?}");
    assert!(root.join("gpu-models").join(&a.file).is_file(), "{}", a.file);

    // An adapter for another architecture is refused, in words.
    let qwen = files.join("qwen-tune.gguf");
    std::fs::write(&qwen, lora(90_000, "qwen2"))?;
    let mut out = Vec::new();
    let err = ctl_with(&lap_h.home, &["models", "upload", &qwen.display().to_string(), "--node", "gpu", "--adapter-for", "base"], &mut out, "").await;
    let out = String::from_utf8(out)?;
    assert!(err.is_err() && out.contains("gpu refused the upload (wrong-arch)"), "{out}");

    // --rollout (U3): each upload becomes the next version of `support` on this inference node,
    // as `support-rN`; its first --canary needs no eval, later ones over the canary share do.
    let sup = |v: u32| -> Result<String> {
        let d = files.join(format!("v{v}"));
        std::fs::create_dir_all(&d)?;
        let f = d.join("Support.gguf");
        std::fs::write(&f, gguf(150_000 + v as usize * 1000, "llama"))?;
        Ok(f.display().to_string())
    };
    let s1 = sup(1)?;
    let out = ctl(&lap_h.home, &["models", "upload", &s1, "--node", "gpu", "--license", "mit", "--rollout", "support", "--canary", "100"]).await?;
    assert!(out.contains("As `support-r1`, version support@1") && out.contains("gpu: stored `support-r1`"), "{out}");
    assert!(out.contains("Applied to the running ymird"), "{out}");
    let rollout_of = |h: &Home| -> Result<Vec<(u32, String, u32)>> {
        let cfg = NodeConfig::load(&h.paths.config())?;
        let r = cfg.inference.rollout.iter().find(|r| r.name == "support");
        Ok(r.map(|r| r.versions.iter().map(|v| (v.version, v.model.clone(), v.weight)).collect()).unwrap_or_default())
    };
    assert_eq!(rollout_of(&lap_h)?, vec![(1, "support-r1".to_string(), 100)]);
    let s2 = sup(2)?;
    let mut out = Vec::new();
    let err = ctl_with(&lap_h.home, &["models", "upload", &s2, "--node", "gpu", "--rollout", "support", "--canary", "50"], &mut out, "").await;
    let (out, err) = (String::from_utf8(out)?, format!("{:#}", err.unwrap_err()));
    assert!(out.contains("support@2") && err.contains("canary share") && err.contains("support@2 stays at weight 0"), "{out}\n{err}");
    assert_eq!(rollout_of(&lap_h)?[1], (2, "support-r2".to_string(), 0));
    let s3 = sup(3)?;
    let out = ctl(&lap_h.home, &["models", "upload", &s3, "--node", "gpu", "--rollout", "support", "--canary", "5"]).await?;
    assert!(out.contains("support@3"), "{out}");
    assert_eq!(rollout_of(&lap_h)?[2], (3, "support-r3".to_string(), 5));
    let mut list = String::new();
    eventually(40, "rollout list shows provenance", async || {
        list = ctl(&lap_h.home, &["models", "rollout", "list"]).await?;
        Ok(list.contains("origin upload, license mit"))
    })
    .await
    .map_err(|e| anyhow::anyhow!("{e}: {list}"))?;
    assert!(list.contains("support@1") && list.contains("support-r3") && list.contains("license not given"), "{list}");
    // Without --node there is no rollout to join.
    let err = format!("{:#}", ctl(&lap_h.home, &["models", "upload", &s3, "--rollout", "support"]).await.unwrap_err());
    assert!(err.contains("--rollout goes with --node"), "{err}");

    // A node that must keep more free space than it has refuses with no-space.
    eventually(40, "full answers models/1", async || Ok(ctl(&lap_h.home, &["models", "list", "--node", "full"]).await.is_ok())).await?;
    let mut out = Vec::new();
    let err = ctl_with(&lap_h.home, &["models", "upload", &model_s, "--node", "full"], &mut out, "").await;
    let out = String::from_utf8(out)?;
    assert!(err.is_err() && out.contains("full refused the upload (no-space)"), "{out}");
    assert!(!root.join("full-models/my-model-q4_0").exists());

    lap.shutdown().await;
    full.shutdown().await;
    gpu.shutdown().await;
    Ok(())
}
