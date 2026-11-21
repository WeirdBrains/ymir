//! `ymirctl mcp` against a fake pool: a real `ymird` in this process with mock engines (text
//! models, an embedding model and two vision-capable ones), driven with raw MCP JSON-RPC through
//! the protocol layer, through the stdio loop over a pipe, and over loopback HTTP with the bridge
//! token. Nothing here touches the network or a real model.

use std::path::PathBuf;

use anyhow::Result;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
use ymir_core::{Daemon, NodeConfig, Paths, vision};
use ymirctl::mcp::{Policy, Rpc, YmirTools};

const MOCK: &str = env!("CARGO_BIN_EXE_ymir-mock-engine");

struct Node {
    _dir: tempfile::TempDir,
    home: PathBuf,
    models: PathBuf,
    paths: Paths,
    daemon: Option<Daemon>,
}

async fn node() -> Result<Node> {
    let dir = tempfile::tempdir()?;
    let home = dir.path().join("home");
    let mdir = dir.path().join("models");
    std::fs::create_dir_all(&mdir)?;
    std::fs::create_dir_all(&home)?;
    for f in ["alpha.gguf", "beta.gguf", "emb.gguf"] {
        std::fs::write(mdir.join(f), b"GGUF mock")?;
    }
    let gemma = weirdnet_model::hw::find("gemma-4-e4b").unwrap().spec.file.clone();
    std::fs::write(mdir.join(&gemma), b"GGUF mock")?;
    std::fs::write(vision::for_model("gemma-4-e4b").unwrap().path(&mdir), b"GGUF mock projector")?;
    let toml = format!(
        "[node]\nname = \"mcp-node\"\n[roles]\nmodel = true\ninference = true\napp = true\n[model]\nllamacpp = \"system:{MOCK}\"\nvllm = \"off\"\nmodels_dir = \"{}\"\n\
         [[model.local]]\nid = \"alpha\"\nfile = \"alpha.gguf\"\n[[model.local]]\nid = \"beta\"\nfile = \"beta.gguf\"\n\
         [[model.local]]\nid = \"emb\"\nfile = \"emb.gguf\"\nrole = \"embedding\"\n\
         [[model.local]]\nid = \"gemma-4-e4b\"\nfile = \"{gemma}\"\n\
         [[model.local]]\nid = \"gemma-blind\"\nfile = \"{gemma}\"\nmmproj = \"off\"\n\
         [inference]\n[inference.alerts]\nprobe_secs = 3600\n[app.bridge]\nport = 0\n",
        mdir.display()
    );
    std::fs::write(home.join("node.toml"), toml)?;
    let paths = Paths::new(&home);
    let cfg = NodeConfig::load(&paths.config())?;
    let daemon = Daemon::start(cfg, paths.clone()).await?;
    Ok(Node { _dir: dir, home, models: mdir, paths, daemon: Some(daemon) })
}

impl Node {
    fn rpc(&self, policy: Policy) -> Rpc<YmirTools> {
        ymirctl::mcp::server(YmirTools::new(self.paths.clone(), Some(self.home.clone()), policy))
    }

    async fn stop(mut self) {
        self.daemon.take().unwrap().shutdown().await;
    }
}

/// Calls a JSON-RPC method and returns the whole reply.
async fn call_rpc(rpc: &Rpc<YmirTools>, method: &str, params: Value) -> Value {
    rpc.handle(None, json!({"jsonrpc": "2.0", "id": 7, "method": method, "params": params})).await.expect("a reply")
}

/// A tool's structured answer; fails the test on a tool error.
async fn ok(rpc: &Rpc<YmirTools>, name: &str, args: Value) -> Value {
    let r = call_rpc(rpc, "tools/call", json!({"name": name, "arguments": args})).await;
    assert_eq!(r["result"]["isError"], false, "{name} failed: {r}");
    r["result"]["structuredContent"].clone()
}

/// A tool's error text; fails the test when the tool succeeds.
async fn err(rpc: &Rpc<YmirTools>, name: &str, args: Value) -> String {
    let r = call_rpc(rpc, "tools/call", json!({"name": name, "arguments": args})).await;
    assert_eq!(r["result"]["isError"], true, "{name} should fail: {r}");
    r["result"]["content"][0]["text"].as_str().unwrap().to_string()
}

async fn tool_names(rpc: &Rpc<YmirTools>) -> Vec<String> {
    call_rpc(rpc, "tools/list", json!({})).await["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap().to_string())
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn read_tools_against_a_fake_pool() -> Result<()> {
    let n = node().await?;
    let rpc = n.rpc(Policy { read_only: true, allow_remove: false });

    let init =
        call_rpc(&rpc, "initialize", json!({"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}})).await;
    assert_eq!(init["result"]["serverInfo"]["name"], "ymir");

    // Read-only: nothing that changes anything is listed or callable.
    let names = tool_names(&rpc).await;
    for t in [
        "ymir_status",
        "ymir_nodes",
        "ymir_devices",
        "ymir_models",
        "ymir_chat",
        "ymir_complete",
        "ymir_embed",
        "ymir_service_status",
        "ymir_models_search",
        "ymir_models_local",
    ] {
        assert!(names.iter().any(|n| n == t), "{t} missing from {names:?}");
    }
    for t in ymirctl::mcp::mutating_tools() {
        assert!(!names.iter().any(|n| n == t), "{t} listed in read-only mode");
        assert!(call_rpc(&rpc, "tools/call", json!({"name": t, "arguments": {"confirm": true}})).await["error"].is_object());
    }

    let st = ok(&rpc, "ymir_status", json!({})).await;
    assert_eq!(st["node"], "mcp-node");
    assert_eq!(st["roles"], json!(["model", "inference", "app"]));
    let nodes = ok(&rpc, "ymir_nodes", json!({})).await;
    assert_eq!(nodes["nodes"][0]["models"], 5);
    let dev = ok(&rpc, "ymir_devices", json!({})).await;
    assert!(dev["network"].is_null() && dev["message"].as_str().unwrap().contains("pool link"), "{dev}");

    // Chat through the bridge, a completion and embeddings.
    let a = ok(&rpc, "ymir_chat", json!({"prompt": "ping pong"})).await;
    assert_eq!(a["text"], "Hello from alpha! You said: ping pong");
    assert_eq!(a["route"], "bridge");
    let a = ok(&rpc, "ymir_chat", json!({"model": "beta", "system": "be brief", "messages": [{"role": "user", "content": "hi"}], "temperature": 0}))
        .await;
    assert_eq!(a["text"], "Hello from beta! You said: hi");
    let a = ok(&rpc, "ymir_complete", json!({"prompt": "Tell me", "model": "alpha"})).await;
    assert!(a["text"].as_str().unwrap().contains("once upon a time"), "{a}");
    let e = ok(&rpc, "ymir_embed", json!({"input": ["two words", "x"]})).await;
    assert_eq!(e["embeddings"][0], json!([9.0, 2.0, 1.0, 0.0]));
    assert_eq!(e["embeddings"].as_array().unwrap().len(), 2);
    let e = ok(&rpc, "ymir_embed", json!({"input": "one"})).await;
    assert_eq!(e["embeddings"].as_array().unwrap().len(), 1);

    // Images: as content parts in `messages`, and from a file with `images`. Only a model that
    // takes images accepts them.
    let png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
    let parts = json!([{"type": "text", "text": "describe"}, {"type": "image_url", "image_url": {"url": png}}]);
    let a = ok(&rpc, "ymir_chat", json!({"model": "gemma-4-e4b", "messages": [{"role": "user", "content": parts}]})).await;
    assert!(a["text"].as_str().unwrap().contains("describe [1 image(s)]"), "{a}");
    let file = n.home.join("pic.png");
    std::fs::write(&file, b"\x89PNG\r\n\x1a\nfake")?;
    let a = ok(&rpc, "ymir_chat", json!({"model": "gemma-4-e4b", "prompt": "two", "images": [file.to_str().unwrap(), png]})).await;
    assert!(a["text"].as_str().unwrap().contains("two [2 image(s)]"), "{a}");
    let e = err(&rpc, "ymir_chat", json!({"model": "gemma-blind", "prompt": "x", "images": [png]})).await;
    assert!(e.contains("does not take images"), "{e}");
    // A file that is not media is never read.
    let txt = n.home.join("notes.txt");
    std::fs::write(&txt, b"private")?;
    assert!(err(&rpc, "ymir_chat", json!({"prompt": "x", "images": [txt.to_str().unwrap()]})).await.contains("not a image type"));
    // Audio parts reach the pool as input_audio; a model without audio input is refused.
    let wav = n.home.join("a.wav");
    std::fs::write(&wav, b"RIFFfake")?;
    let e = err(&rpc, "ymir_chat", json!({"model": "alpha", "prompt": "hear", "audio": [wav.to_str().unwrap()]})).await;
    assert!(e.contains("audio"), "{e}");

    // Models with replicas, state and what they take; the one that answered is warm.
    let m = ok(&rpc, "ymir_models", json!({})).await;
    assert_eq!(m["source"], "bridge");
    let models = m["models"].as_array().unwrap();
    let beta = models.iter().find(|m| m["id"] == "beta").unwrap();
    assert_eq!(beta["state"], "warm", "{m}");
    assert_eq!(beta["replicas"], 1);
    assert_eq!(beta["placements"][0]["node"], "mcp-node");
    assert!(models.iter().any(|m| m["id"] == "emb" && m["purpose"] == "embedding"), "{m}");
    let takes = |id: &str| models.iter().find(|m| m["id"] == id).unwrap()["takes"]["images"].clone();
    assert_eq!((takes("gemma-4-e4b"), takes("gemma-blind"), takes("alpha")), (json!(true), json!(false), json!(false)), "{m}");

    // Clear errors: unknown models, bad arguments, Weirdnet without an identity.
    assert!(err(&rpc, "ymir_chat", json!({"model": "nope", "prompt": "x"})).await.contains("unknown model"));
    assert!(err(&rpc, "ymir_chat", json!({})).await.contains("required"));
    assert!(err(&rpc, "ymir_chat", json!({"prompt": "x", "route": "weirdnet"})).await.contains("pool link"));
    assert!(err(&rpc, "ymir_embed", json!({})).await.contains("input"));
    assert!(err(&rpc, "ymir_models_search", json!({"query": "x", "format": "zip"})).await.contains("unknown format"));

    // The service (read-only status) and the files on this node.
    let s = ok(&rpc, "ymir_service_status", json!({})).await;
    assert!(s["lines"].as_array().unwrap().iter().any(|l| l.as_str().unwrap().starts_with("ymird: answering")), "{s}");
    let local = ok(&rpc, "ymir_models_local", json!({})).await;
    let text = local["text"].as_str().unwrap();
    assert!(text.contains("alpha") && text.contains("registered as"), "{text}");

    n.stop().await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mutating_tools_need_confirm_and_remove_needs_its_flag() -> Result<()> {
    let n = node().await?;
    let rpc = n.rpc(Policy::default());
    let names = tool_names(&rpc).await;
    for t in ["ymir_load_model", "ymir_unload_model", "ymir_pull_model"] {
        assert!(names.iter().any(|n| n == t), "{t} missing");
    }
    assert!(!names.iter().any(|n| n == "ymir_remove_model"), "remove is off by default");
    assert!(
        call_rpc(&rpc, "tools/call", json!({"name": "ymir_remove_model", "arguments": {"id": "alpha", "confirm": true}})).await["error"].is_object()
    );
    let listed = call_rpc(&rpc, "tools/list", json!({})).await;
    let load = listed["result"]["tools"].as_array().unwrap().iter().find(|t| t["name"] == "ymir_load_model").unwrap().clone();
    assert_eq!(load["annotations"]["readOnlyHint"], false);
    assert!(load["description"].as_str().unwrap().starts_with("MUTATES THE POOL"));

    let state = |st: &Value, id: &str| st["models"].as_array().unwrap().iter().find(|m| m["id"] == id).unwrap()["state"]["state"].clone();
    // Without confirm (absent, false or not a boolean) nothing is loaded.
    for args in [json!({"model": "alpha"}), json!({"model": "alpha", "confirm": false}), json!({"model": "alpha", "confirm": "true"})] {
        assert!(err(&rpc, "ymir_load_model", args).await.contains("confirm=true"));
    }
    assert_ne!(state(&ok(&rpc, "ymir_status", json!({})).await, "alpha"), json!("ready"));
    // With it, load and unload go through the control socket.
    assert!(ok(&rpc, "ymir_load_model", json!({"model": "alpha", "confirm": true})).await["message"].as_str().unwrap().contains("ready"));
    assert_eq!(state(&ok(&rpc, "ymir_status", json!({})).await, "alpha"), json!("ready"));
    assert!(err(&rpc, "ymir_unload_model", json!({"model": "alpha"})).await.contains("confirm=true"));
    assert_eq!(state(&ok(&rpc, "ymir_status", json!({})).await, "alpha"), json!("ready"), "an unconfirmed unload must not run");
    assert!(ok(&rpc, "ymir_unload_model", json!({"model": "alpha", "confirm": true})).await["message"].as_str().unwrap().contains("unloaded"));
    assert!(!err(&rpc, "ymir_load_model", json!({"model": "nope", "confirm": true})).await.is_empty());
    assert!(err(&rpc, "ymir_load_model", json!({"confirm": true})).await.contains("`model` is required"));

    // Remove, with its flag: a preview first, files stay until confirm.
    let rpc = n.rpc(Policy { read_only: false, allow_remove: true });
    assert!(tool_names(&rpc).await.iter().any(|n| n == "ymir_remove_model"));
    let r = ok(&rpc, "ymir_remove_model", json!({"id": "beta"})).await;
    assert_eq!(r["deleted"], false);
    assert!(r["text"].as_str().unwrap().contains("Delete beta"), "{r}");
    assert!(n.models.join("beta.gguf").exists());
    let r = ok(&rpc, "ymir_remove_model", json!({"id": "beta", "confirm": true})).await;
    assert_eq!(r["deleted"], true);
    assert!(!n.models.join("beta.gguf").exists());
    assert!(n.models.join("alpha.gguf").exists());

    n.stop().await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stdio_loop_answers_requests_in_any_order() -> Result<()> {
    let n = node().await?;
    let rpc = n.rpc(Policy::default());
    let (mut cw, sr) = tokio::io::duplex(1 << 20);
    let (sw, cr) = tokio::io::duplex(1 << 20);
    let task = tokio::spawn(ymirctl::mcp::serve_stdio(rpc, sr, sw));
    let mut lines = tokio::io::BufReader::new(cr).lines();
    let send = |v: Value| format!("{v}\n");
    cw.write_all(
        send(json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "ymir_chat", "arguments": {"prompt": "slow"}}})).as_bytes(),
    )
    .await?;
    cw.write_all(send(json!({"jsonrpc": "2.0", "method": "notifications/initialized"})).as_bytes()).await?;
    cw.write_all(send(json!({"jsonrpc": "2.0", "id": 2, "method": "ping"})).as_bytes()).await?;
    cw.write_all(b"\n[{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"ping\"},{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"ping\"}]\n").await?;
    drop(cw);
    let mut seen = Vec::new();
    while let Some(l) = lines.next_line().await? {
        seen.push(serde_json::from_str::<Value>(&l)?);
    }
    task.await??;
    // Three replies and a batch reply; the notification and blank line got none.
    assert_eq!(seen.len(), 3, "{seen:?}");
    let chat = seen.iter().find(|v| v["id"] == 1).unwrap();
    assert_eq!(chat["result"]["structuredContent"]["text"], "Hello from alpha! You said: slow");
    assert!(seen.iter().any(|v| v["id"] == 2));
    assert!(seen.iter().any(|v| v.as_array().is_some_and(|a| a.len() == 2)));
    n.stop().await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn streamable_http_on_loopback_with_the_bridge_token() -> Result<()> {
    let n = node().await?;
    let token = ymir_core::token::load(&n.paths)?.unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let task = tokio::spawn(ymirctl::mcp::serve_http(listener, n.rpc(Policy::default()), token.clone(), async {
        let _ = stop_rx.await;
    }));
    let url = format!("http://127.0.0.1:{port}/mcp");
    let http = reqwest::Client::builder().no_proxy().build()?;
    let init = json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}});
    let post = |tok: Option<&str>, body: &Value| {
        let mut r = http.post(&url).header("accept", "application/json, text/event-stream").json(body);
        if let Some(t) = tok {
            r = r.bearer_auth(t);
        }
        r.send()
    };
    assert_eq!(post(None, &init).await?.status(), 401);
    assert_eq!(post(Some("wrong"), &init).await?.status(), 401);
    let r = post(Some(&token), &init).await?;
    assert_eq!(r.status(), 200);
    let v: Value = r.json().await?;
    assert_eq!(v["result"]["serverInfo"]["name"], "ymir");

    // A tool call, and a notification (202, no body).
    let call = json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "ymir_nodes", "arguments": {}}});
    let v: Value = post(Some(&token), &call).await?.json().await?;
    assert_eq!(v["result"]["structuredContent"]["nodes"][0]["models"], 5);
    let r = post(Some(&token), &json!({"jsonrpc": "2.0", "method": "notifications/initialized"})).await?;
    assert_eq!(r.status(), 202);
    // No server-initiated stream.
    assert_eq!(http.get(&url).bearer_auth(&token).send().await?.status(), 405);

    // A foreign Host or Origin is refused (DNS rebinding, a web page), even with the token.
    let r = http.post(&url).header("host", "evil.example").bearer_auth(&token).json(&init).send().await?;
    assert_eq!(r.status(), 403);
    let r = http.post(&url).header("origin", "https://evil.example").bearer_auth(&token).json(&init).send().await?;
    assert_eq!(r.status(), 403);

    // Only loopback addresses are accepted.
    assert!(ymirctl::mcp::loopback_addr("0.0.0.0:8485").is_err());
    let _ = stop_tx.send(());
    task.await??;
    n.stop().await;
    Ok(())
}
