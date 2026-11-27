//! A stand-in for Ray's `ray` CLI, enough for a `vllm-ray` group's orchestration:
//!
//! - `ray start --head --block --node-ip-address IP --port P`: the "GCS", listening on `IP:P`.
//!   It also starts a child in its process group (as Ray's head starts `gcs_server` and the
//!   raylet) and prints `mock ray: child PID`, so tests can see that stopping the group stops it.
//!   With `--mock-bind-any` the child listens on all interfaces at `P+1`, like Ray's real GCS.
//!   With `--mock-worker-bind-any` it also starts a "worker" in a process group of its own (as the
//!   raylet starts Ray's workers) that listens on all interfaces at `P+2`, and prints
//!   `mock ray: worker PID`. It prints `mock ray env K=V` for the link pinning variables it got.
//! - `ray start --block --address HOST:P --node-ip-address IP`: joins the head and holds the
//!   connection until killed; exits 1 when the head goes away.
//! - `ray status --address HOST:P`: prints Ray's "Active:" block, one `node_` line per node.
//!
//! Protocol with the head: a line `join` (held open) or `status` (answered with the node count).

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |a: &str| args.iter().position(|x| x == a).and_then(|i| args.get(i + 1)).cloned();
    let has = |a: &str| args.iter().any(|x| x == a);
    match args.get(1).map(String::as_str) {
        Some("__child") => {
            if let Some(port) = flag("--listen-any") {
                let _l = TcpListener::bind(("0.0.0.0", port.parse::<u16>().expect("port"))).expect("bind any");
                loop {
                    std::thread::park();
                }
            }
            loop {
                std::thread::park();
            }
        }
        Some("start") if has("--head") => {
            assert!(has("--block"), "Ymir runs ray in the foreground");
            let ip = flag("--node-ip-address").expect("--node-ip-address");
            let port: u16 = flag("--port").and_then(|p| p.parse().ok()).expect("--port");
            let mut child = std::process::Command::new(&args[0]);
            child.arg("__child");
            if has("--mock-bind-any") {
                child.args(["--listen-any", &(port + 1).to_string()]);
            }
            let mut child = child.spawn().expect("child");
            eprintln!("mock ray: child {}", child.id());
            std::thread::spawn(move || child.wait());
            for k in ["VLLM_HOST_IP", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME"] {
                if let Ok(v) = std::env::var(k) {
                    eprintln!("mock ray env {k}={v}");
                }
            }
            if has("--mock-worker-bind-any") {
                let mut w = std::process::Command::new(&args[0]);
                w.args(["__child", "--listen-any", &(port + 2).to_string()]);
                // A process group of its own, as the raylet starts Ray's workers (Unix; the
                // `vllm-ray` tests run on Linux only).
                #[cfg(unix)]
                std::os::unix::process::CommandExt::process_group(&mut w, 0);
                let mut w = w.spawn().expect("worker");
                eprintln!("mock ray: worker {}", w.id());
                std::thread::spawn(move || w.wait());
            }
            let l = TcpListener::bind((ip.as_str(), port)).expect("bind");
            eprintln!("mock ray head on {}", l.local_addr().expect("addr"));
            let nodes = Arc::new(AtomicUsize::new(1));
            for c in l.incoming().flatten() {
                let nodes = nodes.clone();
                std::thread::spawn(move || serve(c, &nodes));
            }
        }
        Some("start") => {
            assert!(has("--block"), "Ymir runs ray in the foreground");
            let addr = flag("--address").expect("--address");
            flag("--node-ip-address").expect("--node-ip-address");
            let Ok(mut s) = TcpStream::connect(&addr) else {
                eprintln!("could not reach the head at {addr}");
                std::process::exit(1);
            };
            s.write_all(b"join\n").expect("join");
            eprintln!("mock ray worker joined {addr}");
            let mut buf = [0u8; 64];
            while matches!(s.read(&mut buf), Ok(n) if n > 0) {}
            eprintln!("the head went away");
            std::process::exit(1);
        }
        Some("status") => {
            let addr = flag("--address").expect("--address");
            let Ok(mut s) = TcpStream::connect(&addr) else {
                eprintln!("could not reach {addr}");
                std::process::exit(1);
            };
            s.write_all(b"status\n").expect("status");
            let mut out = String::new();
            let _ = s.read_to_string(&mut out);
            let n: usize = out.trim().parse().unwrap_or(0);
            println!("======== Autoscaler status ========\nNode status\n---------------\nActive:");
            for i in 0..n {
                println!(" 1 node_{i:040x}");
            }
            println!("Pending:\n (no pending nodes)\nRecent failures:\n (no failures)");
        }
        other => {
            eprintln!("mock ray: unsupported {other:?}");
            std::process::exit(2);
        }
    }
}

fn serve(c: TcpStream, nodes: &AtomicUsize) {
    let mut r = BufReader::new(c.try_clone().expect("clone"));
    let mut line = String::new();
    if r.read_line(&mut line).is_err() {
        return;
    }
    match line.trim() {
        "join" => {
            nodes.fetch_add(1, Ordering::SeqCst);
            let mut buf = [0u8; 64];
            while matches!(r.read(&mut buf), Ok(n) if n > 0) {}
            nodes.fetch_sub(1, Ordering::SeqCst);
        }
        "status" => {
            let mut c = c;
            let _ = write!(c, "{}", nodes.load(Ordering::SeqCst));
        }
        _ => {}
    }
}
