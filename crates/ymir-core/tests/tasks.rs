//! Agent tasks (task/1) on an inference node: Bragi's agent mode against a Ymir node, in one
//! process on the in-memory transport. The model is a fake node that asks for `files.read` once and
//! then repeats what the tool returned; the approval card is answered here.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::json;
use weirdnet_agent::pool::NetDialer;
use weirdnet_agent::remote::RemoteError;
use weirdnet_agent::remote::infer::v2::{Request, Usage};
use weirdnet_agent::remote::task::{self, TaskRequest};
use weirdnet_agent::{AgentEvent, AgentId, ApprovalAnswer, ApprovalPrompt, Approver, CancelToken, Role, Scope, ToolSpec};
use weirdnet_backends::{Chunk, ChunkStream, ModelStatus};
use weirdnet_net::{Keypair, Multiaddr, PeerId, WEIRDNET};
use weirdnet_sync::Roster;
use ymir_core::config::AgentSection;
use ymir_core::coordinator::Coordinator;
use ymir_core::membership::{Membership, SyncNode};
use ymir_core::model_node::{BoxFuture, Embedded, ModelNode};
use ymir_core::remote;

struct Reader;

impl ModelNode for Reader {
    fn name(&self) -> &str {
        "reader"
    }
    fn status(&self) -> Vec<ModelStatus> {
        vec![
            serde_json::from_value(json!({
                "id": "small", "engine": "llamacpp", "state": {"state": "ready"}, "in_flight": 0, "swap_group": null,
                "embedding": false, "pid": null, "last_load_secs": null, "size_bytes": null, "wrapped": false, "crashes": 0
            }))
            .unwrap(),
        ]
    }
    fn load<'a>(&'a self, _: &'a str, _: bool) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(async { Ok(()) })
    }
    fn unload<'a>(&'a self, _: &'a str) -> BoxFuture<'a, Result<(), RemoteError>> {
        Box::pin(async { Ok(()) })
    }
    fn chat<'a>(&'a self, _model: &'a str, req: &'a Request) -> BoxFuture<'a, Result<ChunkStream, RemoteError>> {
        Box::pin(async move {
            let msgs = req.inline_messages().unwrap_or_default();
            let mut ev = Vec::new();
            match msgs.last() {
                Some(m) if m.role == Role::Tool => {
                    let said: Vec<&str> = m.content.lines().filter(|l| !l.starts_with('<') && !l.starts_with('[')).collect();
                    ev.push(Chunk::Text(format!("The file says: {}", said.join(" "))));
                }
                _ if req.tools.iter().any(|t| t.name == "files.read") => {
                    let path = msgs.iter().rev().find(|m| m.role == Role::User).map(|m| m.content.clone()).unwrap_or_default();
                    ev.push(Chunk::Text("Reading. ".into()));
                    ev.push(Chunk::ToolCall {
                        index: 0,
                        id: Some("c1".into()),
                        name: Some("files.read".into()),
                        arguments: json!({ "path": path }).to_string(),
                    });
                }
                _ => ev.push(Chunk::Text("I have no tools here.".into())),
            }
            ev.push(Chunk::Done { finish_reason: Some("stop".into()), usage: Some(Usage::new(1, 1)) });
            Ok(Box::pin(futures::stream::iter(ev.into_iter().map(Ok))) as ChunkStream)
        })
    }
    fn embed<'a>(&'a self, _: &'a str, _: &'a [String]) -> BoxFuture<'a, Result<Embedded, RemoteError>> {
        Box::pin(async { Err(RemoteError::Unsupported { message: "no embeddings".into() }) })
    }
}

struct Device {
    m: Arc<Membership>,
    node: Arc<SyncNode>,
    addrs: Vec<Multiaddr>,
}

impl Device {
    fn start(m: Membership) -> Self {
        let m = Arc::new(m);
        let node = Arc::new(weirdnet_sync::start_node(m.node_key().clone(), weirdnet_sync::node_config(true), m.state()).unwrap());
        let mut addrs = Vec::new();
        for _ in 0..200 {
            addrs = node.status().listening.iter().filter_map(|a| a.parse().ok()).collect();
            if !addrs.is_empty() {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(!addrs.is_empty());
        Self { m, node, addrs }
    }
    fn peer(&self) -> String {
        self.m.peer_id().to_string()
    }
    fn dialer_to(&self, other: &Device) -> NetDialer<Vec<u8>, Vec<u8>> {
        let d = NetDialer::new(self.node.clone());
        d.hint(other.m.peer_id() as PeerId, other.addrs.clone());
        d
    }
}

fn link(from: &Membership, dir: &std::path::Path, label: &str) -> Membership {
    let key = Keypair::generate_ed25519();
    let mut seed = from.seed().unwrap().0;
    let user = Keypair::ed25519_from_bytes(&mut seed).unwrap();
    let cert = Roster::self_certify(&WEIRDNET, &user, &key.public().to_peer_id().to_string(), label, 1).unwrap();
    from.approve(cert.clone()).unwrap();
    Membership::adopt(dir, key, &from.seed().unwrap().0, cert, &from.link_settings()).unwrap()
}

fn coordinator() -> Arc<Coordinator> {
    Arc::new(Coordinator::new(
        vec![Arc::new(Reader)],
        Some("small".into()),
        Arc::new(weirdnet_semcache::ExactCache::<weirdnet_semcache::MemoryStore>::new(16)),
    ))
}

/// Answers every card the same way and remembers what was asked.
struct Cards {
    answer: ApprovalAnswer,
    asked: Mutex<Vec<String>>,
}

#[async_trait]
impl Approver for Cards {
    async fn decide(&self, _prompt: &ApprovalPrompt, tool: &ToolSpec) -> ApprovalAnswer {
        self.asked.lock().unwrap().push(tool.name.clone());
        self.answer.clone()
    }
}

async fn run(
    app: &Device,
    inf: &Device,
    companion: &str,
    input: &str,
    answer: ApprovalAnswer,
) -> (Result<task::RemoteOutcome, RemoteError>, Vec<String>) {
    let cards = Cards { answer, asked: Mutex::new(Vec::new()) };
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let r = task::run(
        &app.dialer_to(inf),
        &inf.peer(),
        TaskRequest::new(AgentId::new(companion), input, Vec::new()),
        Some(tx),
        &cards,
        &CancelToken::new(),
    )
    .await;
    let mut tools = Vec::new();
    while let Ok(e) = rx.try_recv() {
        if let AgentEvent::ToolFinished { tool, is_error, .. } = e {
            tools.push(format!("{tool}:{}", if is_error { "error" } else { "ok" }));
        }
    }
    let asked = cards.asked.into_inner().unwrap();
    (r.map(|o| o), [asked, tools].concat())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_task_reads_a_workspace_file_after_the_card_is_approved() {
    let tmp = tempfile::tempdir().unwrap();
    let ws = tmp.path().join("ws");
    std::fs::create_dir_all(&ws).unwrap();
    std::fs::write(ws.join("notes.txt"), "buy oat milk").unwrap();
    std::fs::write(tmp.path().join("outside.txt"), "private").unwrap();

    let inf = Device::start(Membership::create(&tmp.path().join("inf"), "inference").unwrap());
    let app = Device::start(link(&inf.m, &tmp.path().join("app"), "app"));
    let cfg = AgentSection { workspace: Some(ws.display().to_string()), ..AgentSection::default() };
    let _t = ymir_core::tasks::serve(&inf.node, coordinator(), remote::roster_auth(&inf.m), "inf", &cfg).await.unwrap();

    // Approved: the node reads the file and the model repeats it.
    let (r, seen) = run(&app, &inf, "assistant", "notes.txt", ApprovalAnswer::Approve { scope: Scope::Once }).await;
    let out = r.unwrap();
    assert!(out.outcome.text.contains("The file says: buy oat milk"), "{}", out.outcome.text);
    assert_eq!(seen, ["files.read", "files.read:ok"]);

    // Declined: the tool does not run.
    let (r, seen) = run(&app, &inf, "assistant", "notes.txt", ApprovalAnswer::Deny).await;
    assert!(!r.unwrap().outcome.text.contains("buy oat milk"));
    assert_eq!(seen, ["files.read"], "asked, never ran");

    // A path out of the workspace is refused by the tool even when approved.
    for bad in ["../outside.txt", "/etc/hostname"] {
        let (r, seen) = run(&app, &inf, "assistant", bad, ApprovalAnswer::Approve { scope: Scope::Once }).await;
        assert!(!r.unwrap().outcome.text.contains("private"), "{bad}");
        assert_eq!(seen.last().map(String::as_str), Some("files.read:error"), "{bad}: {seen:?}");
    }

    // A companion that does not run here is refused.
    let (r, _) = run(&app, &inf, "someone-else", "x", ApprovalAnswer::Deny).await;
    assert!(r.is_err());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn without_a_workspace_the_task_has_no_tools_and_strangers_are_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let inf = Device::start(Membership::create(&tmp.path().join("inf"), "inference").unwrap());
    let app = Device::start(link(&inf.m, &tmp.path().join("app"), "app"));
    let stranger = Device::start(Membership::create(&tmp.path().join("x"), "stranger").unwrap());
    let _t = ymir_core::tasks::serve(&inf.node, coordinator(), remote::roster_auth(&inf.m), "inf", &AgentSection::default()).await.unwrap();

    let (r, seen) = run(&app, &inf, "ymir", "notes.txt", ApprovalAnswer::Approve { scope: Scope::Once }).await;
    assert_eq!(r.unwrap().outcome.text, "I have no tools here.");
    assert!(seen.is_empty());

    let (r, _) = run(&stranger, &inf, "assistant", "hi", ApprovalAnswer::Deny).await;
    assert!(r.is_err(), "another identity's device gets no task");

    // A revoked device is cut off.
    inf.m.revoke(&app.peer()).unwrap();
    let (r, _) = run(&app, &inf, "assistant", "hi", ApprovalAnswer::Deny).await;
    assert!(r.is_err(), "revoked");
}
