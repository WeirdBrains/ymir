//! `ymirctl mcp-servers`: add, list, test, disable and remove, with `ymirctl mcp` (stdio) as the
//! demo server. Secrets stay references.

use clap::Parser;
use ymir_core::config::NodeConfig;

async fn ctl(home: &std::path::Path, args: &[&str], answers: &str) -> (anyhow::Result<()>, String) {
    let mut argv = vec!["ymirctl".to_string(), "--home".into(), home.display().to_string()];
    argv.extend(args.iter().map(|s| s.to_string()));
    let cli = ymirctl::Cli::try_parse_from(argv).unwrap();
    let mut out = Vec::new();
    let mut input = std::io::Cursor::new(answers.to_string());
    let r = ymirctl::run(cli, &mut out, &mut input).await;
    (r, String::from_utf8(out).unwrap())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn servers_are_added_listed_tested_and_removed() {
    let home = tempfile::tempdir().unwrap();
    let cfg_path = home.path().join("node.toml");
    NodeConfig::default().save(&cfg_path).unwrap();
    let with_comment = format!("# my own note: keep me\n{}", std::fs::read_to_string(&cfg_path).unwrap());
    std::fs::write(&cfg_path, with_comment).unwrap();
    let bin = env!("CARGO_BIN_EXE_ymirctl");
    let demo_home = tempfile::tempdir().unwrap();
    let demo_home = demo_home.path().display().to_string();
    let key = home.path().join("demo-key");
    std::fs::write(&key, "hunter2\n").unwrap();
    let key_ref = format!("DEMO_KEY=file:{}", key.display());

    // A local command asks first; "no" adds nothing.
    let add = ["mcp-servers", "add", "demo", "--command", bin, "--arg", "--home", "--arg", &demo_home, "--arg", "mcp", "--env", &key_ref];
    let (r, out) = ctl(home.path(), &add, "n\n").await;
    r.unwrap();
    assert!(out.contains("Not added"), "{out}");
    assert!(NodeConfig::load(&home.path().join("node.toml")).unwrap().inference.agent.mcp.is_empty());

    let (r, out) = ctl(home.path(), &add, "y\n").await;
    r.unwrap();
    assert!(out.contains("Added `demo`"), "{out}");

    let (_, out) = ctl(home.path(), &["mcp-servers", "list"], "").await;
    assert!(out.contains("demo\ton\tstdio:") && out.contains("1 secret reference"), "{out}");
    assert!(!out.contains("DEMO_KEY"), "names of secrets are not needed in the listing: {out}");

    // The same id twice, a literal secret and a plain-http endpoint are refused.
    assert!(ctl(home.path(), &add, "y\n").await.0.is_err());
    let literal = ["mcp-servers", "add", "x", "--url", "https://example.com/mcp", "--bearer", "sk-literal"];
    assert!(ctl(home.path(), &literal, "").await.0.is_err());
    let plain = ["mcp-servers", "add", "y", "--url", "http://example.com/mcp"];
    assert!(ctl(home.path(), &plain, "").await.0.is_err());
    let toml = std::fs::read_to_string(home.path().join("node.toml")).unwrap();
    assert!(!toml.contains("sk-literal") && toml.contains("file:") && !toml.contains("hunter2"));

    // Test connects and lists the demo server's tools as `demo.<tool>`.
    let (r, out) = ctl(home.path(), &["mcp-servers", "test", "demo"], "").await;
    r.unwrap();
    assert!(out.contains("demo.ymir_status"), "{out}");

    let text = std::fs::read_to_string(&cfg_path).unwrap();
    assert!(text.contains("# my own note: keep me"), "comments survive an edit");
    ctl(home.path(), &["mcp-servers", "disable", "demo"], "").await.0.unwrap();
    assert!(!NodeConfig::load(&home.path().join("node.toml")).unwrap().inference.agent.mcp[0].enabled);
    ctl(home.path(), &["mcp-servers", "remove", "demo"], "").await.0.unwrap();
    assert!(ctl(home.path(), &["mcp-servers", "remove", "demo"], "").await.0.is_err());
    let text = std::fs::read_to_string(&cfg_path).unwrap();
    assert!(text.contains("# my own note: keep me") && !text.contains("[[inference.agent.mcp]]"), "{text}");
}
