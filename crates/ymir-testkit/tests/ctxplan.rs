//! Contexts at load time against the mock engine: an idle model makes room for the one asked
//! for, a just-used one is kept, and the catalog reports the context each engine runs with.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use weirdnet_backends::{EngineConfig, EngineKind, ModelLaunch, ModelState, Supervisor, SupervisorOptions};
use weirdnet_model::hw::Budget;
use ymir_core::ctxplan::{MemProbe, Planner};
use ymir_core::model_node::{LocalModelNode, ModelNode};

const MOCK: &str = env!("CARGO_BIN_EXE_ymir-mock-engine");
const GIB: u64 = 1 << 30;

/// A sparse `size`-byte GGUF whose header is a one-layer llama with a 1M context: 4 KiB of f16
/// KV per token, so 4 GiB at full length.
fn gguf(dir: &Path, name: &str, size: u64) -> String {
    let mut h = b"GGUF".to_vec();
    h.extend(3u32.to_le_bytes());
    h.extend(0u64.to_le_bytes());
    let kv: [(&str, u32); 5] = [
        ("llama.block_count", 1),
        ("llama.context_length", 1_048_576),
        ("llama.embedding_length", 1024),
        ("llama.attention.head_count", 8),
        ("llama.attention.head_count_kv", 8),
    ];
    h.extend((kv.len() as u64 + 1).to_le_bytes());
    let s = |h: &mut Vec<u8>, s: &str| {
        h.extend((s.len() as u64).to_le_bytes());
        h.extend(s.as_bytes());
    };
    s(&mut h, "general.architecture");
    h.extend(8u32.to_le_bytes());
    s(&mut h, "llama");
    for (k, v) in kv {
        s(&mut h, k);
        h.extend(4u32.to_le_bytes());
        h.extend(v.to_le_bytes());
    }
    let p = dir.join(name);
    std::fs::write(&p, &h).unwrap();
    std::fs::OpenOptions::new().write(true).open(&p).unwrap().set_len(size).unwrap();
    p.to_string_lossy().into_owned()
}

struct NoProbe;

impl MemProbe for NoProbe {
    fn free_bytes(&self) -> Option<u64> {
        None
    }
}

/// Two 20 GiB models on a 50 GiB node: either alone gets its full 1M (4 GiB of KV), not both.
fn node(dir: &Path, grace: Duration) -> (LocalModelNode, Supervisor) {
    let (n, s, _) = node_and_planner(dir, grace);
    (n, s)
}

fn node_and_planner(dir: &Path, grace: Duration) -> (LocalModelNode, Supervisor, Arc<Planner>) {
    let mut launches = vec![
        ModelLaunch::new("a", EngineKind::Llamacpp, gguf(dir, "a.gguf", 20 * GIB)),
        ModelLaunch::new("b", EngineKind::Llamacpp, gguf(dir, "b.gguf", 20 * GIB)),
    ];
    for l in &mut launches {
        l.slots = Some(1);
    }
    let budget = Budget { pool_bytes: 64 * GIB, reserve_bytes: 8 * GIB, usable_bytes: 50 * GIB };
    ymir_core::kvfit::plan_launches(&mut launches, &budget, dir);
    let sup = Supervisor::new(
        vec![EngineConfig::llamacpp(MOCK)],
        launches.clone(),
        SupervisorOptions { log_dir: Some(dir.join("logs")), ..Default::default() },
    );
    let planner = Planner::new(sup.clone(), budget, Arc::new(NoProbe), dir.into(), &HashSet::new()).with_grace(grace);
    let planner = Arc::new(planner);
    let node = LocalModelNode::new("n", sup.clone(), &launches);
    node.set_planner(planner.clone());
    (node, sup, planner)
}

fn context(node: &LocalModelNode, id: &str) -> Option<u32> {
    node.status().into_iter().find(|s| s.id == id).unwrap().context
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_idle_model_is_unloaded_so_the_one_asked_for_gets_its_full_context() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let (node, sup) = node(dir.path(), Duration::ZERO);
    // At start the two share the node, so neither has its full length.
    assert!(context(&node, "a").unwrap() < 1_048_576);
    node.load("a", false).await?;
    assert_eq!(context(&node, "a"), Some(1_048_576), "alone, a gets everything");
    // b is asked for while a is idle: a goes, b gets its full length too.
    node.load("b", false).await?;
    assert_eq!(sup.state("a"), Some(ModelState::Unloaded));
    assert_eq!(sup.state("b"), Some(ModelState::Ready));
    assert_eq!(context(&node, "b"), Some(1_048_576));
    sup.shutdown().await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_just_used_model_is_kept_and_the_next_one_is_cut_to_fit_beside_it() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let (node, sup) = node(dir.path(), Duration::from_secs(3600));
    node.load("a", false).await?;
    node.load("b", false).await?;
    assert_eq!(sup.state("a"), Some(ModelState::Ready), "inside the grace period a stays");
    let b = context(&node, "b").unwrap();
    assert!(b < 1_048_576, "b is cut beside a: {b}");
    sup.shutdown().await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_load_from_elsewhere_is_followed_and_what_is_not_loaded_is_resized_beside_it() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let (node, sup, planner) = node_and_planner(dir.path(), Duration::ZERO);
    let keeper = planner.spawn_keeper();
    // Nothing runs: the keeper gives each model not loaded its share (both could load).
    let shared = context(&node, "b").unwrap();
    // A `Load` over the model-node protocol goes straight to the supervisor.
    sup.load("a").await?;
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let b = context(&node, "b").unwrap();
        // a runs at its share; b now gets all that is left beside it (no longer shared with a).
        if b != shared && sup.state("a") == Some(ModelState::Ready) {
            assert!(b > shared && b < 1_048_576, "{b} vs {shared}");
            break;
        }
        assert!(std::time::Instant::now() < deadline, "b was not resized beside a: {b}");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    keeper.abort();
    sup.shutdown().await;
    Ok(())
}

/// A sparse `size`-byte LoRA GGUF for a llama base.
fn lora(dir: &Path, name: &str, size: u64) -> String {
    let mut h = b"GGUF".to_vec();
    h.extend(3u32.to_le_bytes());
    h.extend(0u64.to_le_bytes());
    h.extend(3u64.to_le_bytes());
    for (k, v) in [("general.architecture", "llama"), ("general.type", "adapter"), ("adapter.type", "lora")] {
        h.extend((k.len() as u64).to_le_bytes());
        h.extend(k.as_bytes());
        h.extend(8u32.to_le_bytes());
        h.extend((v.len() as u64).to_le_bytes());
        h.extend(v.as_bytes());
    }
    let p = dir.join(name);
    std::fs::write(&p, &h).unwrap();
    std::fs::OpenOptions::new().write(true).open(&p).unwrap().set_len(size).unwrap();
    p.to_string_lossy().into_owned()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_adapter_counts_in_its_base_and_is_loaded_and_evicted_with_it() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let d = dir.path();
    // a (20 GiB) carries a 22 GiB adapter: alone on the 50 GiB node it no longer gets its full
    // 1M (4 GiB of KV); b (20 GiB) alone does.
    let mut launches = vec![
        ModelLaunch::new("a", EngineKind::Llamacpp, gguf(d, "a.gguf", 20 * GIB))
            .with_adapters(vec![weirdnet_backends::LoraAdapter::new("a+tune", lora(d, "tune.gguf", 22 * GIB))]),
        ModelLaunch::new("b", EngineKind::Llamacpp, gguf(d, "b.gguf", 20 * GIB)),
    ];
    for l in &mut launches {
        l.slots = Some(1);
    }
    let budget = Budget { pool_bytes: 64 * GIB, reserve_bytes: 8 * GIB, usable_bytes: 50 * GIB };
    ymir_core::kvfit::plan_launches(&mut launches, &budget, d);
    let sup = Supervisor::new(
        vec![EngineConfig::llamacpp(MOCK)],
        launches.clone(),
        SupervisorOptions { log_dir: Some(d.join("logs")), ..Default::default() },
    );
    let planner = Arc::new(Planner::new(sup.clone(), budget, Arc::new(NoProbe), d.into(), &HashSet::new()).with_grace(Duration::ZERO));
    let node = LocalModelNode::new("n", sup.clone(), &launches);
    node.set_planner(planner);

    // Asked for by the adapter's id: the base is sized with the adapter's weights and loaded.
    node.load("a+tune", false).await?;
    assert_eq!(sup.state("a"), Some(ModelState::Ready));
    let a = context(&node, "a").unwrap();
    assert!((32_768..1_048_576).contains(&a), "a is cut for its adapter: {a}");
    // b next, a idle: a (and its adapter, which is not a model of its own) makes room, and b
    // gets its full length.
    node.load("b", false).await?;
    assert_eq!(sup.state("a"), Some(ModelState::Unloaded));
    assert_eq!(context(&node, "b"), Some(1_048_576));
    sup.shutdown().await;
    Ok(())
}
