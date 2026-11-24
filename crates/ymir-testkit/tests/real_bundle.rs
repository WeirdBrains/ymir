//! A release file as a user installs it, with its real bundled llama.cpp and a real GGUF: the
//! unpacked archive's `bin/ymirctl setup` (no `--llama-server`), then its `bin/ymird` as a separate
//! process, a streamed chat through the bridge, and a killed engine that ymird survives and
//! restarts. Ignored by default:
//!
//! python3 scripts/package.py --variant default   # or download a release archive
//! tar xzf dist/ymir-VERSION-TRIPLE.tar.gz -C /tmp
//! YMIR_TEST_BUNDLE=/tmp/ymir-VERSION-TRIPLE YMIR_TEST_GGUF=/path/tiny.gguf \
//!   cargo test -p ymir-testkit --test real_bundle -- --ignored --nocapture

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde_json::json;
use ymir_core::Paths;
use ymir_core::client::BridgeClient;
use ymir_core::control::{self, ControlRequest, ModelState, NodeModel};

struct Ymird(Child);

impl Drop for Ymird {
    fn drop(&mut self) {
        // SAFETY: our own child; SIGTERM lets it stop its engines.
        unsafe { libc::kill(self.0.id() as i32, libc::SIGTERM) };
        let deadline = Instant::now() + Duration::from_secs(20);
        while Instant::now() < deadline {
            if let Ok(Some(_)) = self.0.try_wait() {
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let _ = self.0.kill();
    }
}

/// A port for a config or command line (see [`ymir_testkit::config_port`]).
fn free_port() -> u16 {
    ymir_testkit::config_port()
}

fn alive(pid: u32) -> bool {
    // SAFETY: signal 0 only checks that the process exists.
    unsafe { libc::kill(pid as i32, 0) == 0 }
}

async fn model(paths: &Paths) -> Option<NodeModel> {
    control::request(&paths.socket(), &ControlRequest::Status).await.ok()?.status?.models.into_iter().next()
}

async fn chat(client: &BridgeClient, prompt: &str) -> Result<(String, usize)> {
    let mut extra = serde_json::Map::new();
    extra.insert("max_tokens".into(), 24.into());
    let mut pieces = 0;
    let out = client.chat_stream(None, vec![json!({"role": "user", "content": prompt})], extra, |_| pieces += 1).await?;
    Ok((out.text, pieces))
}

fn clean_env(cmd: &mut Command, home: &Path) {
    // Nothing from the caller's Ymir setup or PATH: the bundled engine has to be found on its own.
    cmd.env_clear().env("HOME", home).env("PATH", "/usr/bin:/bin").env("YMIR_LOG", "info");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs YMIR_TEST_BUNDLE (an unpacked release archive) and YMIR_TEST_GGUF"]
async fn release_archive_runs_a_real_model() -> Result<()> {
    let bundle = PathBuf::from(std::env::var("YMIR_TEST_BUNDLE").context("YMIR_TEST_BUNDLE")?);
    let gguf = PathBuf::from(std::env::var("YMIR_TEST_GGUF").context("YMIR_TEST_GGUF")?);
    let dir = tempfile::tempdir()?;
    let home = dir.path().join("home");
    let port = free_port();

    let mut setup = Command::new(bundle.join("bin/ymirctl"));
    clean_env(&mut setup, dir.path());
    let out = setup
        .args(["--home", home.to_str().unwrap(), "setup", "--name", "bundle-test", "--port", &port.to_string()])
        .args(["--model", gguf.to_str().unwrap(), "--models-dir", dir.path().to_str().unwrap()])
        .output()?;
    let text = String::from_utf8_lossy(&out.stdout);
    eprintln!("{text}");
    if !out.status.success() {
        bail!("setup failed: {}", String::from_utf8_lossy(&out.stderr));
    }
    assert!(text.contains("shipped with Ymir"), "setup found the bundled engine:\n{text}");
    let cfg = std::fs::read_to_string(home.join("node.toml"))?;
    assert!(cfg.contains("llamacpp = \"managed\""), "{cfg}");
    // Small context: the test must fit next to whatever else runs on this machine.
    std::fs::write(home.join("node.toml"), cfg.replacen("[[model.local]]", "[[model.local]]\ncontext = 2048", 1))?;

    let mut cmd = Command::new(bundle.join("bin/ymird"));
    clean_env(&mut cmd, dir.path());
    let log = std::fs::File::create(dir.path().join("ymird.log"))?;
    let _ymird = Ymird(cmd.args(["--home", home.to_str().unwrap()]).stdout(Stdio::null()).stderr(log).spawn()?);
    let paths = Paths::new(&home);
    let deadline = Instant::now() + Duration::from_secs(30);
    while model(&paths).await.is_none() {
        if Instant::now() > deadline {
            bail!("ymird did not start:\n{}", std::fs::read_to_string(dir.path().join("ymird.log"))?);
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    let client = BridgeClient::new(port, ymir_core::token::load(&paths)?.context("bridge token")?);

    let t = Instant::now();
    let (text, pieces) = chat(&client, "Say hello in five words.").await.with_context(|| engine_log(&paths))?;
    eprintln!("cold answer in {:.1}s ({pieces} pieces): {text:?}", t.elapsed().as_secs_f32());
    assert!(!text.trim().is_empty() && pieces > 1);
    let m = model(&paths).await.unwrap();
    assert_eq!(m.state, ModelState::Ready);
    let pid = m.pid.context("engine pid")?;
    if cfg!(target_os = "linux") {
        let exe = std::fs::read_link(format!("/proc/{pid}/exe"))?;
        eprintln!("engine pid {pid}: {}", exe.display());
        assert!(exe.starts_with(std::fs::canonicalize(&bundle)?), "the engine runs from the bundle: {}", exe.display());
    }

    // The devices the bundled engine can use (Vulkan, CUDA, Metal; none means CPU).
    let server = ymir_core::engine::bundled_from(&bundle.join("bin/ymird")).context("bundled engine")?.llama_server();
    let mut list = Command::new(server);
    clean_env(&mut list, dir.path());
    eprintln!("{}", String::from_utf8_lossy(&list.arg("--list-devices").output()?.stdout).trim());

    // Crash isolation: kill the engine; ymird stays up and the next prompt brings it back.
    // SAFETY: the engine process this test's ymird started.
    unsafe { libc::kill(pid as i32, libc::SIGKILL) };
    let deadline = Instant::now() + Duration::from_secs(15);
    while alive(pid) && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert!(model(&paths).await.is_some(), "ymird survived the engine crash");
    let (text, _) = chat(&client, "And again, in three words.").await.with_context(|| engine_log(&paths))?;
    eprintln!("after the crash: {text:?}");
    assert!(!text.trim().is_empty());
    let pid2 = model(&paths).await.unwrap().pid.unwrap();
    assert_ne!(pid, pid2, "a new engine process");
    Ok(())
}

fn engine_log(paths: &Paths) -> String {
    let logs = std::fs::read_dir(paths.logs()).into_iter().flatten().flatten();
    logs.filter_map(|e| std::fs::read_to_string(e.path()).ok()).collect::<Vec<_>>().join("\n")
}
