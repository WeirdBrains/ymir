//! A stand-in for llama.cpp's `rpc-server`: takes `-H HOST -p PORT`, listens on TCP and accepts
//! connections until it is killed. Enough for a group job's worker to be started, found
//! listening and torn down.

use std::net::TcpListener;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |a: &str, b: &str| args.iter().position(|x| x == a || x == b).and_then(|i| args.get(i + 1)).cloned();
    let host = flag("-H", "--host").unwrap_or_else(|| "127.0.0.1".into());
    let port: u16 = flag("-p", "--port").and_then(|p| p.parse().ok()).unwrap_or(50052);
    let l = TcpListener::bind((host.as_str(), port)).expect("bind");
    eprintln!("mock rpc-server listening on {host}:{port}");
    for c in l.incoming() {
        drop(c);
    }
}
