//! Training plan P2: conversations the user marked for training, exported with the strip rules.
//! Two daemons over Weirdnet on 127.0.0.1 (nothing leaves the box): the conversations start on
//! the inference node, the consent is set on the app node, syncs to the inference node, and the export there holds only the marked
//! conversation, owner-only; withdrawing it shows in `context withdrawn`.

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

async fn ctl(home: &Path, args: &[&str]) -> Result<String> {
    let mut argv = vec!["ymirctl".to_string(), "--home".into(), home.display().to_string()];
    argv.extend(args.iter().map(|a| a.to_string()));
    let cli = ymirctl::Cli::try_parse_from(argv)?;
    let mut out: Vec<u8> = Vec::new();
    let mut input = std::io::Cursor::new(Vec::new());
    ymirctl::run(cli, &mut out as &mut dyn Write, &mut input).await?;
    Ok(String::from_utf8(out)?)
}

struct Home {
    home: PathBuf,
    paths: Paths,
}

fn home(root: &Path, name: &str, roles: &str, peers: &[String]) -> Result<Home> {
    let home = root.join(name);
    std::fs::create_dir_all(&home)?;
    let peers = peers.iter().map(|p| format!("\"{p}\"")).collect::<Vec<_>>().join(", ");
    let (i, a) = (roles.contains("inference"), roles.contains("app"));
    let toml = format!(
        "[node]\nname = \"{name}\"\n[roles]\nmodel = false\ninference = {i}\napp = {a}\n[network]\nlisten = [\"/ip4/127.0.0.1/tcp/0\"]\nmdns = false\ndiscovery = false\nrelays = []\nsync_secs = 1\npeers = [{peers}]\n[model]\nllamacpp = \"off\"\nvllm = \"off\"\n[inference.alerts]\nprobe_secs = 3600\n[app.bridge]\nport = 0\n"
    );
    std::fs::write(home.join("node.toml"), toml)?;
    Ok(Home { paths: Paths::new(&home), home })
}

fn prelink(from: &Membership, to: &Home, label: &str) {
    let key = Keypair::generate_ed25519();
    let mut seed = from.seed().unwrap().0;
    let user = Keypair::ed25519_from_bytes(&mut seed).unwrap();
    let cert = Roster::self_certify(&WEIRDNET, &user, &key.public().to_peer_id().to_string(), label, 1).unwrap();
    from.approve(cert.clone()).unwrap();
    Membership::adopt(&to.paths.pool(), key, &from.seed().unwrap().0, cert, &from.link_settings()).unwrap();
}

async fn start(h: &Home) -> Result<Daemon> {
    Daemon::start(NodeConfig::load(&h.paths.config())?, h.paths.clone()).await
}

async fn addr(h: &Home) -> Result<String> {
    let s = control::request(&h.paths.socket(), &ControlRequest::Status).await?.status.unwrap();
    let n = s.network.expect("on Weirdnet");
    let a = n.listening.iter().find(|a| a.starts_with("/ip4/127.0.0.1/tcp/")).expect("a loopback listener");
    Ok(format!("{}/p2p/{}", a.split("/p2p/").next().unwrap(), n.peer))
}

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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn only_marked_conversations_are_exported_and_withdrawal_is_reported() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();
    let inf_tmp = home(root, "inf", "inference", &[])?;
    let first = Membership::create(&inf_tmp.paths.pool(), "inf")?;
    let app_tmp = home(root, "app", "app", &[])?;
    prelink(&first, &app_tmp, "app");
    drop(first);
    let inf = start(&inf_tmp).await?;
    let app_h = home(root, "app", "app", &[addr(&inf_tmp).await?])?;
    let app = start(&app_h).await?;

    // The first write waits for the pool's first content key (a sync pass brings it).
    eventually(40, "the content key", async || Ok(ctl(&inf_tmp.home, &["context", "add", "warmup", "x"]).await.is_ok())).await?;
    ctl(&inf_tmp.home, &["context", "delete", "warmup"]).await?;
    for c in ["keep", "private"] {
        ctl(&inf_tmp.home, &["context", "add", c, "my key is sk-12345 ok?"]).await?;
        ctl(&inf_tmp.home, &["context", "add", c, "noted", "--role", "assistant"]).await?;
    }
    eventually(40, "the conversations reach the app node", async || {
        Ok(ctl(&app_h.home, &["context", "show"]).await.unwrap_or_default().lines().filter(|l| l.contains("(2 messages)")).count() == 2)
    })
    .await?;
    // Consent is off until given, and needs a conversation that exists.
    assert!(!ctl(&app_h.home, &["context", "show"]).await?.contains("[training]"));
    assert!(ctl(&app_h.home, &["context", "train", "nope"]).await.is_err());
    ctl(&app_h.home, &["context", "train", "keep"]).await?;

    // It syncs: the inference node sees it and exports only that one.
    eventually(40, "the consent reaches the inference node", async || {
        Ok(ctl(&inf_tmp.home, &["context", "show"]).await.unwrap_or_default().lines().any(|l| l.starts_with("keep ") && l.contains("[training]")))
    })
    .await?;
    let out = root.join("train.jsonl");
    let o = out.display().to_string();
    let said = ctl(&inf_tmp.home, &["context", "export", "--training", "--redact", r"sk-\d+", &o]).await?;
    assert!(said.starts_with("1 of 1 conversation(s)"), "{said}");
    assert!(!said.contains("sk-"), "counts only, never content: {said}");
    let body = std::fs::read_to_string(&out)?;
    assert_eq!(body.lines().count(), 1);
    let row: serde_json::Value = serde_json::from_str(body.lines().next().unwrap())?;
    assert_eq!(row["id"], "keep");
    assert_eq!(row["messages"][0]["content"], "my key is [redacted] ok?");
    assert_eq!(row["messages"][1]["role"], "assistant");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(&out)?.permissions().mode() & 0o777, 0o600);
    }
    // An export with no `--training` is refused.
    assert!(ctl(&inf_tmp.home, &["context", "export", &o]).await.is_err());

    // Withdrawn on the app: the earlier export's manifest names it as gone.
    ctl(&app_h.home, &["context", "train", "keep", "--off"]).await?;
    let manifest = format!("{o}.manifest.json");
    eventually(40, "the withdrawal reaches the inference node", async || {
        Ok(ctl(&inf_tmp.home, &["context", "withdrawn", &manifest]).await.unwrap_or_default().lines().next() == Some("keep"))
    })
    .await?;

    app.shutdown().await;
    inf.shutdown().await;
    Ok(())
}
