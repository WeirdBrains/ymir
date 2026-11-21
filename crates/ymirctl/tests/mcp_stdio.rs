//! `ymirctl mcp` as a real MCP client would start it: a child process speaking newline-delimited
//! JSON-RPC on stdin and stdout, with no ymird running.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde_json::{Value, json};

struct Client {
    child: Child,
    stdin: Option<ChildStdin>,
    out: BufReader<ChildStdout>,
    _home: tempfile::TempDir,
    next: u64,
}

impl Client {
    fn start(flags: &[&str]) -> Self {
        let home = tempfile::tempdir().unwrap();
        let mut child = Command::new(env!("CARGO_BIN_EXE_ymirctl"))
            .arg("--home")
            .arg(home.path())
            .arg("mcp")
            .args(flags)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let (stdin, out) = (child.stdin.take(), BufReader::new(child.stdout.take().unwrap()));
        Self { child, stdin, out, _home: home, next: 1 }
    }

    fn send_raw(&mut self, line: &str) {
        let w = self.stdin.as_mut().unwrap();
        writeln!(w, "{line}").unwrap();
        w.flush().unwrap();
    }

    fn read(&mut self) -> Value {
        let mut l = String::new();
        self.out.read_line(&mut l).unwrap();
        serde_json::from_str(&l).unwrap_or_else(|e| panic!("not JSON ({e}): {l:?}"))
    }

    fn rpc(&mut self, method: &str, params: Value) -> Value {
        let id = self.next;
        self.next += 1;
        self.send_raw(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).to_string());
        let r = self.read();
        assert_eq!(r["id"], id, "{r}");
        r
    }

    fn tool(&mut self, name: &str, args: Value) -> Value {
        self.rpc("tools/call", json!({"name": name, "arguments": args}))["result"].clone()
    }

    fn finish(mut self) {
        drop(self.stdin.take());
        let status = self.child.wait().unwrap();
        assert!(status.success(), "{status}");
    }
}

fn init(c: &mut Client) {
    let r = c.rpc("initialize", json!({"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}));
    assert_eq!(r["result"]["serverInfo"]["name"], "ymir");
    assert_eq!(r["result"]["protocolVersion"], "2025-06-18");
    c.send_raw(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"}).to_string());
}

fn names(c: &mut Client) -> Vec<String> {
    let r = c.rpc("tools/list", json!({}));
    r["result"]["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap().to_string()).collect()
}

#[test]
fn handshake_tools_and_errors_over_stdio() {
    let mut c = Client::start(&[]);
    init(&mut c);
    let list = c.rpc("tools/list", json!({}));
    let tools = list["result"]["tools"].as_array().unwrap();
    let find = |n: &str| tools.iter().find(|t| t["name"] == n).unwrap_or_else(|| panic!("{n} missing"));
    for t in [
        "ymir_status",
        "ymir_nodes",
        "ymir_models",
        "ymir_chat",
        "ymir_embed",
        "ymir_models_search",
        "ymir_load_model",
        "ymir_unload_model",
        "ymir_pull_model",
    ] {
        find(t);
    }
    // Removal is not offered unless the server was started with --allow-remove.
    assert!(!tools.iter().any(|t| t["name"] == "ymir_remove_model"));
    assert_eq!(find("ymir_status")["annotations"]["readOnlyHint"], true);
    for t in ["ymir_load_model", "ymir_unload_model", "ymir_pull_model"] {
        assert_eq!(find(t)["annotations"]["readOnlyHint"], false, "{t}");
        assert!(find(t)["description"].as_str().unwrap().contains("confirm=true"), "{t}");
    }
    assert_eq!(find("ymir_chat")["inputSchema"]["properties"]["messages"]["type"], "array");

    // No ymird here: tools fail with a message the model can act on, not a protocol error.
    let r = c.tool("ymir_status", json!({}));
    assert_eq!(r["isError"], true);
    assert!(r["content"][0]["text"].as_str().unwrap().contains("not answering"), "{r}");
    // Without confirm a mutating tool does nothing, even before it needs ymird.
    let r = c.tool("ymir_unload_model", json!({"model": "x"}));
    assert_eq!(r["isError"], true);
    assert!(r["content"][0]["text"].as_str().unwrap().contains("confirm=true"), "{r}");
    // Unknown tools are protocol errors; unknown methods too; ping works.
    assert!(c.rpc("tools/call", json!({"name": "nope"}))["error"]["message"].as_str().unwrap().contains("unknown tool"));
    assert_eq!(c.rpc("nope/nothing", json!({}))["error"]["code"], -32601);
    assert_eq!(c.rpc("ping", json!({}))["result"], json!({}));
    // Garbage gets a parse error and the server keeps going.
    c.send_raw("{not json");
    assert_eq!(c.read()["error"]["code"], -32700);
    assert_eq!(c.rpc("ping", json!({}))["result"], json!({}));
    c.finish();
}

#[test]
fn flags_choose_the_tools() {
    let mut c = Client::start(&["--read-only"]);
    init(&mut c);
    let n = names(&mut c);
    assert!(n.contains(&"ymir_status".to_string()));
    assert!(!n.iter().any(|t| ["ymir_load_model", "ymir_unload_model", "ymir_pull_model", "ymir_remove_model"].contains(&t.as_str())), "{n:?}");
    // A hidden tool is not callable either.
    assert!(c.rpc("tools/call", json!({"name": "ymir_load_model", "arguments": {"model": "x", "confirm": true}}))["error"].is_object());
    c.finish();

    let mut c = Client::start(&["--allow-remove"]);
    init(&mut c);
    assert!(names(&mut c).contains(&"ymir_remove_model".to_string()));
    c.finish();
}

#[test]
fn contradicting_flags_fail() {
    let home = tempfile::tempdir().unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_ymirctl"))
        .arg("--home")
        .arg(home.path())
        .args(["mcp", "--read-only", "--allow-remove"])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("contradict"));
}

#[test]
fn resources_list_and_read() {
    let mut c = Client::start(&[]);
    init(&mut c);
    let list = c.rpc("resources/list", json!({}));
    let uris: Vec<&str> = list["result"]["resources"].as_array().unwrap().iter().map(|r| r["uri"].as_str().unwrap()).collect();
    assert_eq!(uris, ["ymir://status", "ymir://nodes", "ymir://models", "ymir://models/local"]);
    // Unknown URI: not found. Known URI with no ymird: an error that says why, not an empty list.
    assert_eq!(c.rpc("resources/read", json!({"uri": "ymir://nope"}))["error"]["code"], -32002);
    let r = c.rpc("resources/read", json!({"uri": "ymir://status"}));
    assert!(r["error"]["message"].as_str().unwrap().contains("not answering"), "{r}");
    // The model files on this node need no daemon.
    let r = c.rpc("resources/read", json!({"uri": "ymir://models/local"}));
    assert_eq!(r["result"]["contents"][0]["mimeType"], "application/json", "{r}");
    assert!(r["result"]["contents"][0]["text"].as_str().unwrap().contains("text"));
    c.finish();
}
