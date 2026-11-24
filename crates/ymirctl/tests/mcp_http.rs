//! `ymirctl mcp --http` as a client would meet it: loopback only, bearer token, Host and Origin checks.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use serde_json::{Value, json};

struct Server {
    child: Child,
    port: u16,
    token: String,
    _home: tempfile::TempDir,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn ymirctl(home: &std::path::Path) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_ymirctl"));
    c.arg("--home").arg(home);
    c
}

fn start() -> Server {
    let home = tempfile::tempdir().unwrap();
    let out = ymirctl(home.path()).args(["bridge", "token"]).output().unwrap();
    assert!(out.status.success());
    let token = String::from_utf8(out.stdout).unwrap().trim().to_string();
    assert!(!token.is_empty());
    // Port 0: the server binds first and then prints the address it got, so the port can't be
    // taken by another test in between, and it is listening once the line is out.
    let mut child = ymirctl(home.path())
        .args(["mcp", "--read-only", "--http", "127.0.0.1:0"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut lines = BufReader::new(child.stderr.take().unwrap()).lines();
    let line = lines
        .by_ref()
        .map_while(Result::ok)
        .find(|l| l.contains("http://127.0.0.1:"))
        .expect("the MCP server exited without saying where it listens");
    // Keep reading so the server never blocks on a full pipe.
    std::thread::spawn(move || lines.for_each(drop));
    let port = line
        .split("http://127.0.0.1:")
        .nth(1)
        .and_then(|r| r.split('/').next())
        .and_then(|p| p.parse().ok())
        .unwrap_or_else(|| panic!("no address in: {line}"));
    Server { child, port, token, _home: home }
}

/// POSTs `body` to /mcp with the given extra header lines; returns the status and the body.
fn post(s: &Server, host: &str, headers: &[String], body: &Value) -> (u16, String) {
    let body = body.to_string();
    let mut req = format!(
        "POST /mcp HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nAccept: application/json, text/event-stream\r\nConnection: close\r\nContent-Length: {}\r\n",
        body.len()
    );
    for h in headers {
        req.push_str(h);
        req.push_str("\r\n");
    }
    req.push_str("\r\n");
    req.push_str(&body);
    let mut c = TcpStream::connect(("127.0.0.1", s.port)).unwrap();
    c.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
    c.write_all(req.as_bytes()).unwrap();
    let mut raw = String::new();
    c.read_to_string(&mut raw).unwrap();
    let status = raw.split_whitespace().nth(1).unwrap().parse().unwrap();
    (status, raw.split("\r\n\r\n").nth(1).unwrap_or_default().to_string())
}

fn auth(s: &Server) -> String {
    format!("Authorization: Bearer {}", s.token)
}

#[test]
fn token_host_and_origin_are_checked() {
    let s = start();
    let host = format!("127.0.0.1:{}", s.port);
    let ping = json!({"jsonrpc": "2.0", "id": 1, "method": "ping"});

    let (code, body) = post(&s, &host, &[auth(&s)], &ping);
    assert_eq!(code, 200, "{body}");
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["result"], json!({}));

    // No token, or the wrong one.
    assert_eq!(post(&s, &host, &[], &ping).0, 401);
    assert_eq!(post(&s, &host, &["Authorization: Bearer nope".into()], &ping).0, 401);
    // A Host that is not this machine (DNS rebinding) and a foreign browser Origin.
    assert_eq!(post(&s, "evil.example", &[auth(&s)], &ping).0, 403);
    assert_eq!(post(&s, &host, &[auth(&s), "Origin: http://evil.example".into()], &ping).0, 403);
    assert_eq!(post(&s, &host, &[auth(&s), format!("Origin: http://{host}")], &ping).0, 200);

    // Tools and resources work over HTTP; --read-only hides the mutating tools.
    let (code, body) = post(&s, &host, &[auth(&s)], &json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}));
    assert_eq!(code, 200);
    let v: Value = serde_json::from_str(&body).unwrap();
    let names: Vec<&str> = v["result"]["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"ymir_status") && !names.contains(&"ymir_load_model"), "{names:?}");
    let (_, body) = post(&s, &host, &[auth(&s)], &json!({"jsonrpc": "2.0", "id": 3, "method": "resources/list"}));
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["result"]["resources"].as_array().unwrap().len(), 4);
}

#[test]
fn non_loopback_addresses_are_refused() {
    let home = tempfile::tempdir().unwrap();
    assert!(ymirctl(home.path()).args(["bridge", "token"]).output().unwrap().status.success());
    let out = ymirctl(home.path()).args(["mcp", "--http", "0.0.0.0:0"]).stdin(Stdio::null()).output().unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("loopback"));
}
