//! `ModelsHub` against mock nodes: the owner's requests reach `ModelsService`s over in-memory
//! channels (the same protocol messages Weirdnet carries), a stranger is refused, a pull streams
//! progress, `auto` picks the node that fits, and a pulled model is served without a restart.

use std::io::{BufRead, BufReader, Write};
use std::sync::Arc;

use futures::future::BoxFuture;
use tokio::sync::mpsc::unbounded_channel;
use weirdnet_agent::remote::{Authorize, Caller, Dialer, MemoryDialer, RemoteError};
use weirdnet_backends::models::{MODELS_PROTOCOL, ModelsClient, ModelsReply, ModelsService, Pull};
use weirdnet_backends::{ModelState, Supervisor, SupervisorOptions};
use weirdnet_model::ModelStore;
use weirdnet_model::hw::Budget;

use super::*;

/// A GGUF header (llama, one block, one KV head of size 1: 4 bytes of KV per token) padded to `len`.
fn gguf(len: usize) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend(0u64.to_le_bytes());
    let kv: [(&str, Option<u32>); 5] = [
        ("general.architecture", None),
        ("llama.block_count", Some(1)),
        ("llama.attention.head_count", Some(1)),
        ("llama.attention.head_count_kv", Some(1)),
        ("llama.attention.key_length", Some(1)),
    ];
    out.extend((kv.len() as u64).to_le_bytes());
    for (k, v) in kv {
        out.extend((k.len() as u64).to_le_bytes());
        out.extend(k.as_bytes());
        match v {
            None => {
                out.extend(8u32.to_le_bytes());
                out.extend(5u64.to_le_bytes());
                out.extend(b"llama");
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

/// A Hub with repo `o/n`: a 4000-byte Q4_0 and a 6000-byte Q8_0, and a search for it.
fn hub() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let info = r#"{"id":"o/n","siblings":[{"rfilename":"m-Q4_0.gguf","size":4000},{"rfilename":"m-Q8_0.gguf","size":6000}]}"#;
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { break };
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut first = String::new();
            reader.read_line(&mut first).unwrap();
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 || line.trim().is_empty() {
                    break;
                }
            }
            let path = first.split_whitespace().nth(1).unwrap_or("/").split('?').next().unwrap_or("").to_string();
            let body: Option<Vec<u8>> = match path.as_str() {
                "/api/models" => Some(br#"[{"id":"o/n","downloads":9,"likes":1}]"#.to_vec()),
                "/api/models/o/n" => Some(info.into()),
                "/o/n/resolve/main/m-Q4_0.gguf" => Some(gguf(4000)),
                "/o/n/resolve/main/m-Q8_0.gguf" => Some(gguf(6000)),
                _ => None,
            };
            let _ = match body {
                Some(b) => {
                    let _ = stream.write_all(format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", b.len()).as_bytes());
                    stream.write_all(&b)
                }
                None => stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"),
            };
        }
    });
    url
}

struct Node {
    svc: Arc<ModelsService>,
    sup: Supervisor,
    dir: tempfile::TempDir,
}

/// A model node whose models/1 admits only the peer `owner`, with `usable` bytes of model memory.
fn node(hub: &str, usable: u64) -> Node {
    let dir = tempfile::tempdir().unwrap();
    let sup = Supervisor::new(vec![], vec![], SupervisorOptions::default());
    let auth: Arc<dyn Authorize> = Arc::new(|c: &Caller| if c.peer == "owner" { Ok(()) } else { Err(RemoteError::refused("not-the-owner")) });
    let budget = Budget { pool_bytes: usable, reserve_bytes: 0, usable_bytes: usable };
    let svc = Arc::new(ModelsService::new(ModelStore::new(dir.path()).with_hub(hub), sup.clone(), auth).with_budget(budget));
    Node { svc, sup, dir }
}

fn target(peer: &str, label: &str, local: bool) -> Target {
    Target { peer: peer.into(), label: label.into(), local }
}

struct Fixed(Vec<Target>);

impl Directory for Fixed {
    fn devices(&self) -> Vec<Target> {
        self.0.clone()
    }
    fn model_nodes(&self) -> BoxFuture<'_, Vec<Target>> {
        Box::pin(async move { self.0.iter().filter(|t| !t.local).cloned().collect() })
    }
}

/// A pool: `alpha` (this node's own model role, in process), `beta` and `gamma` (remote, reached over
/// models/1 as `caller`), and `old` (an older ymird that does not take models/1).
struct Pool {
    hub: ModelsHub,
    alpha: Node,
    beta: Node,
    gamma: Node,
}

fn pool(url: &str, caller: &'static str, usable: [u64; 3]) -> Pool {
    let (alpha, beta, gamma) = (node(url, usable[0]), node(url, usable[1]), node(url, usable[2]));
    let dialer = MemoryDialer::new();
    for (peer, svc) in [("peer-beta", beta.svc.clone()), ("peer-gamma", gamma.svc.clone())] {
        dialer.route_only(peer, &[MODELS_PROTOCOL], move |ch, _| {
            let svc = svc.clone();
            tokio::spawn(async move {
                let _ = svc.serve(ch, Caller::new(caller)).await;
            });
        });
    }
    dialer.route_only("peer-old", &["/weirdnet-backends/node/1"], |_, _| {});
    let dialer: Arc<dyn Dialer> = Arc::new(dialer);
    let alpha_target = target("peer-alpha", "alpha", true);
    let dir =
        Fixed(vec![alpha_target.clone(), target("peer-beta", "Beta", false), target("peer-gamma", "gamma", false), target("peer-old", "old", false)]);
    let hub = ModelsHub::new(Some((alpha_target, alpha.svc.clone())), Arc::new(dir), ModelsClient::new(dialer));
    Pool { hub, alpha, beta, gamma }
}

async fn run(p: &Pool, node: &str, op: ModelsOp) -> (Result<ModelsDone, String>, Vec<ModelsLine>) {
    let (tx, mut rx) = unbounded_channel();
    let done = p.hub.run(node, op, &tx).await;
    drop(tx);
    let mut lines = Vec::new();
    while let Some(l) = rx.recv().await {
        lines.push(l);
    }
    (done, lines)
}

fn failed(done: &ModelsDone, label: &str) -> String {
    match &done.nodes.iter().find(|n| n.node.label == label).expect("node answered").outcome {
        Outcome::Failed { message } => message.clone(),
        Outcome::Ok { reply } => panic!("{label} answered: {reply:?}"),
    }
}

fn reply<'a>(done: &'a ModelsDone, label: &str) -> &'a ModelsReply {
    match &done.nodes.iter().find(|n| n.node.label == label).expect("node answered").outcome {
        Outcome::Ok { reply } => reply,
        Outcome::Failed { message } => panic!("{label}: {message}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_device_that_is_not_the_owner_is_refused_and_nothing_runs() {
    let p = pool(&hub(), "stranger", [100_000; 3]);
    for op in [ModelsOp::List, ModelsOp::Search { query: "m".into(), limit: 3 }, ModelsOp::Pull(Pull::new("o/n").with_model("m-Q4_0"))] {
        let (done, lines) = run(&p, "beta", op).await;
        let msg = failed(&done.unwrap(), "Beta");
        assert!(msg.contains("only devices linked to its owner may manage models"), "{msg}");
        assert!(lines.is_empty(), "no progress for a refused pull");
    }
    assert!(!p.beta.dir.path().join("m-Q4_0.gguf").exists(), "nothing was downloaded for the stranger");
    assert!(p.beta.svc.list().is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_remote_pull_streams_progress_and_serves_the_model_without_a_restart() {
    let p = pool(&hub(), "owner", [100_000; 3]);
    let (done, lines) = run(&p, "beta", ModelsOp::Pull(Pull::new("o/n").with_model("m-Q4_0").serving(false))).await;
    let done = done.unwrap();
    let progress: Vec<(u64, u64)> = lines
        .iter()
        .filter_map(|l| match l {
            ModelsLine::Progress { node, done, total } if node == "Beta" => Some((*done, *total)),
            _ => None,
        })
        .collect();
    assert!(!progress.is_empty() && progress.iter().all(|(d, t)| d <= t && *t == 4000), "{progress:?}");
    assert_eq!(progress.last(), Some(&(4000, 4000)));
    let ModelsReply::Pulled(pulled) = reply(&done, "Beta") else { panic!() };
    assert_eq!((pulled.id.as_str(), pulled.size, pulled.served, pulled.loaded), ("m-Q4_0", 4000, true, false));
    assert!(p.beta.dir.path().join("m-Q4_0.gguf").is_file());
    // Served right away: the supervisor on that node has it, unloaded, with no restart.
    assert_eq!(p.beta.sup.state("m-Q4_0"), Some(ModelState::Unloaded));
    let launch = p.beta.sup.launches().into_iter().find(|l| l.id == "m-Q4_0").unwrap();
    assert_eq!(launch.source, p.beta.dir.path().join("m-Q4_0.gguf").display().to_string());

    let (done, _) = run(&p, "beta", ModelsOp::List).await;
    let ModelsReply::List(list) = reply(&done.unwrap(), "Beta").clone() else { panic!() };
    assert_eq!((list[0].model.id.as_str(), list[0].served_as.as_deref()), ("m-Q4_0", Some("m-Q4_0")));

    // Removing stops serving it and deletes the file.
    let (done, _) = run(&p, "BETA", ModelsOp::Remove { id: "m-Q4_0".into() }).await;
    assert!(matches!(reply(&done.unwrap(), "Beta"), ModelsReply::Removed { freed: 4000, .. }));
    assert_eq!(p.beta.sup.state("m-Q4_0"), None);
    assert!(!p.beta.dir.path().join("m-Q4_0.gguf").exists());
}

#[tokio::test(flavor = "multi_thread")]
async fn this_nodes_own_model_role_is_reached_in_process() {
    let p = pool(&hub(), "owner", [100_000; 3]);
    let (done, lines) = run(&p, "alpha", ModelsOp::Pull(Pull::new("o/n").with_model("m-Q8_0").serving(false))).await;
    let done = done.unwrap();
    assert!(matches!(reply(&done, "alpha"), ModelsReply::Pulled(x) if x.served));
    assert!(lines.iter().any(|l| matches!(l, ModelsLine::Progress { node, .. } if node == "alpha")));
    assert_eq!(p.alpha.sup.state("m-Q8_0"), Some(ModelState::Unloaded));
    assert!(done.chosen.is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn auto_picks_the_node_that_can_hold_the_biggest_model() {
    // Q4_0 needs (4000 + 32 KiB of KV) + 10% = 40,444 bytes, Q8_0 42,644.
    let p = pool(&hub(), "owner", [41_000, 100_000, 42_000]);
    let (done, _) = run(&p, "auto", ModelsOp::Pull(Pull::new("o/n").serving(false))).await;
    let done = done.unwrap();
    let c = done.chosen.as_ref().expect("auto says which node it picked");
    assert_eq!((c.node.label.as_str(), c.model.as_str(), c.size, c.usable_bytes), ("Beta", "m-Q8_0", 6000, 100_000));
    assert_eq!(done.nodes.len(), 1, "only the chosen node pulled");
    assert!(p.beta.dir.path().join("m-Q8_0.gguf").is_file());
    assert!(p.alpha.svc.list().is_empty() && p.gamma.svc.list().is_empty());

    // A model that fits more than one node goes to the one with more memory.
    let (done, _) = run(&p, "auto", ModelsOp::Pull(Pull::new("o/n").with_model("m-Q4_0"))).await;
    assert_eq!(done.unwrap().chosen.unwrap().node.label, "Beta");
    // Where only the small one fits anywhere, auto takes it, on the node with the most memory.
    let p = pool(&hub(), "owner", [41_000, 41_000, 41_500]);
    let (done, _) = run(&p, "auto", ModelsOp::Pull(Pull::new("o/n"))).await;
    let c = done.unwrap().chosen.unwrap();
    assert_eq!((c.node.label.as_str(), c.model.as_str()), ("gamma", "m-Q4_0"));
}

#[tokio::test(flavor = "multi_thread")]
async fn auto_says_why_when_nothing_fits_and_all_reports_each_nodes_answer() {
    let p = pool(&hub(), "owner", [1_000, 2_000, 3_000]);
    let (done, _) = run(&p, "auto", ModelsOp::Pull(Pull::new("o/n"))).await;
    let err = done.unwrap_err();
    assert!(err.contains("nothing in o/n fits any node"), "{err}");
    assert!(err.contains("old"), "the node that does not take models/1 is named: {err}");
    assert!(p.beta.svc.list().is_empty(), "nothing was pulled");
    let (done, _) = run(&p, "auto", ModelsOp::List).await;
    assert!(done.unwrap_err().contains("picks where a pull goes"));

    let (done, _) = run(&p, "all", ModelsOp::List).await;
    let done = done.unwrap();
    assert!(matches!(reply(&done, "alpha"), ModelsReply::List(_)) && matches!(reply(&done, "Beta"), ModelsReply::List(_)));
    let old = failed(&done, "old");
    assert!(old.contains("older ymird"), "{old}");
}

#[tokio::test(flavor = "multi_thread")]
async fn nodes_are_named_by_label_or_peer_id() {
    let p = pool(&hub(), "owner", [100_000; 3]);
    for name in ["beta", "BETA", "peer-beta", "peer-bet"] {
        let (done, _) = run(&p, name, ModelsOp::List).await;
        assert_eq!(done.unwrap().nodes[0].node.label, "Beta", "{name}");
    }
    let (done, _) = run(&p, "delta", ModelsOp::List).await;
    let err = done.unwrap_err();
    assert!(err.contains("no device named `delta`") && err.contains("alpha, Beta, gamma, old"), "{err}");
    let (done, _) = run(&p, "peer-", ModelsOp::List).await;
    assert!(done.is_err(), "a short prefix is not enough to name a device");
}
