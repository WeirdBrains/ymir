//! M6, `ymirctl move` between real daemons: separate `ymird`s over Weirdnet on 127.0.0.1 with the
//! mock engine (one inference node, two model nodes that both hold `alpha`, one app device).
//! `alpha` is loaded on `ma1`; a slow answer is streaming from it when the move starts. The move
//! loads `alpha` on `ma2`, the answer on `ma1` finishes whole, then `ma1` unloads it, and the
//! next answers come from `ma2`. Moving it again (it is loaded nowhere else) and moving to a
//! node without a copy are refused, with nothing changed; the MCP tool `ymir_move_model` does nothing
//! without confirm=true and then moves it back.

use std::time::Duration;

use anyhow::{Result, bail};
use serde_json::json;
use ymir_core::control::{self, ControlRequest, ModelState, NodeModel};
use ymir_core::membership::Membership;
use ymirctl::mcp::{Policy, YmirTools};

#[path = "common/pool.rs"]
mod pool;
use pool::*;

/// `model`'s state on each node, as the inference node sees it.
async fn states(h: &Home, model: &str) -> Result<Vec<(String, ModelState)>> {
    let s = control::request(&h.paths.socket(), &ControlRequest::Status).await?.status.unwrap();
    let mut v: Vec<(String, ModelState)> =
        s.models.iter().filter(|m: &&NodeModel| m.id == model).map(|m| (m.node.clone(), m.state.clone())).collect();
    v.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(v)
}

/// A daemon's peer id, which is the name the pool knows a remote model node by.
async fn peer(h: &Home) -> Result<String> {
    Ok(control::request(&h.paths.socket(), &ControlRequest::Status).await?.status.unwrap().network.expect("on Weirdnet").peer)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn move_finishes_the_answer_on_the_old_node_then_unloads_it() -> Result<()> {
    // SAFETY: set before any daemon starts in this test.
    unsafe { std::env::set_var("YMIR_MOCK_WORD_MS", "60") };
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();
    let inf_tmp = home(root, "inf", "inference", &[], &[])?;
    let first = Membership::create(&inf_tmp.paths.pool(), "inf")?;
    let (ma1_tmp, ma2_tmp, mb_tmp) =
        (home(root, "ma1", "model", &[], &["alpha"])?, home(root, "ma2", "model", &[], &["alpha"])?, home(root, "mb", "model", &[], &["beta"])?);
    let dev_tmp = home(root, "dev", "app", &[], &[])?;
    for (h, label) in [(&ma1_tmp, "ma1"), (&ma2_tmp, "ma2"), (&mb_tmp, "mb"), (&dev_tmp, "dev")] {
        prelink(&first, h, label);
    }
    drop(first);

    let (ma1, ma2, mb) = (start(&ma1_tmp).await?, start(&ma2_tmp).await?, start(&mb_tmp).await?);
    let (p1, p2, pb) = (peer(&ma1_tmp).await?, peer(&ma2_tmp).await?, peer(&mb_tmp).await?);
    let models = vec![addr(&ma1_tmp).await?, addr(&ma2_tmp).await?, addr(&mb_tmp).await?];
    let inf_h = home(root, "inf", "inference", &models, &[])?;
    let inf = start(&inf_h).await?;
    let dev_h = home(root, "dev", "app", &[addr(&inf_h).await?], &[])?;
    let dev = start(&dev_h).await?;
    eventually(60, "all three model nodes join", async || Ok(ctl(&inf_h.home, &["nodes"]).await?.matches("serving").count() == 3)).await?;

    ctl(&inf_h.home, &["load", "alpha", "--node", &p1]).await?;
    eventually(30, "alpha is ready on ma1 only", async || {
        let s = states(&inf_h, "alpha").await?;
        Ok(s.iter().any(|(n, st)| *n == p1 && *st == ModelState::Ready) && !s.iter().any(|(n, st)| *n == p2 && *st == ModelState::Ready))
    })
    .await?;
    // Admission, and a first answer from ma1 (the only node with alpha ready).
    let deadline = std::time::Instant::now() + Duration::from_secs(100);
    loop {
        match stream_chat(&dev, &dev_h, "alpha", "warm up", async |_| {}).await {
            Ok(_) => break,
            Err(e) if std::time::Instant::now() > deadline => bail!("the device was never admitted: {e:#}"),
            Err(_) => tokio::time::sleep(Duration::from_millis(500)).await,
        }
    }

    // A slow answer (about 40 words at 60 ms each) streams from ma1 while the move runs.
    let q = (0..40).map(|i| format!("w{i}")).collect::<Vec<_>>().join(" ");
    let (started_tx, started_rx) = tokio::sync::oneshot::channel::<()>();
    let mut started_tx = Some(started_tx);
    let chat = stream_chat(&dev, &dev_h, "alpha", &q, async |_| {
        if let Some(tx) = started_tx.take() {
            let _ = tx.send(());
        }
    });
    let moving = async {
        let _ = started_rx.await;
        let t = std::time::Instant::now();
        let out = ctl(&inf_h.home, &["move", "alpha", "--to", &p2, "--grace", "30"]).await;
        (out, t.elapsed())
    };
    let (text, (moved, took)) = tokio::join!(chat, moving);
    let text = text?;
    let moved = moved?;
    assert!(moved.contains(&format!("alpha moved from {p1} to {p2}")), "{moved}");
    assert!(text.contains(&q), "the answer on ma1 was cut short: {text}");
    // It waited for the answer (about 2.4 s of words) instead of unloading under it.
    eprintln!("the move took {took:?}");
    assert!(took > Duration::from_millis(1000), "the move took {took:?}");
    eventually(30, "alpha is ready on ma2 and unloaded on ma1", async || {
        let s = states(&inf_h, "alpha").await?;
        Ok(s.iter().any(|(n, st)| *n == p2 && *st == ModelState::Ready) && s.iter().any(|(n, st)| *n == p1 && *st == ModelState::Unloaded))
    })
    .await?;
    // New answers come from ma2.
    let after = stream_chat(&dev, &dev_h, "alpha", "after the move", async |_| {}).await?;
    assert!(after.contains("after the move"), "{after}");
    let s = states(&inf_h, "alpha").await?;
    assert!(s.iter().any(|(n, st)| *n == p1 && *st == ModelState::Unloaded), "ma1 loaded alpha again: {s:?}");

    // Refusals leave everything as it was.
    let again = ctl(&inf_h.home, &["move", "alpha", "--to", &p2]).await.unwrap_err().to_string();
    assert!(again.contains("not loaded on another node"), "{again}");
    let no_copy = ctl(&inf_h.home, &["move", "alpha", "--to", &pb]).await.unwrap_err().to_string();
    assert!(no_copy.contains(&format!("{pb} has no copy of alpha")), "{no_copy}");
    let s = states(&inf_h, "alpha").await?;
    assert!(s.iter().any(|(n, st)| *n == p2 && *st == ModelState::Ready), "{s:?}");

    // The MCP tool does the same move, only with confirm=true; here it moves alpha back.
    let rpc = ymirctl::mcp::server(YmirTools::new(inf_h.paths.clone(), Some(inf_h.home.clone()), Policy { read_only: false, allow_remove: false }));
    let call = async |args: serde_json::Value| {
        let r = rpc
            .handle_async(None, json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "ymir_move_model", "arguments": args}}))
            .await
            .unwrap();
        (r["result"]["isError"] == true, r["result"]["content"][0]["text"].as_str().unwrap_or_default().to_string())
    };
    let (failed, text) = call(json!({"model": "alpha", "to": p1})).await;
    assert!(failed && text.contains("confirm=true"), "{text}");
    assert!(states(&inf_h, "alpha").await?.iter().any(|(n, st)| *n == p2 && *st == ModelState::Ready), "an unconfirmed move changed something");
    let (failed, text) = call(json!({"model": "alpha", "to": p1, "grace": 5, "confirm": true})).await;
    assert!(!failed && text.contains(&format!("alpha moved from {p2} to {p1}")), "{text}");
    eventually(30, "alpha is back on ma1", async || {
        let s = states(&inf_h, "alpha").await?;
        Ok(s.iter().any(|(n, st)| *n == p1 && *st == ModelState::Ready) && s.iter().any(|(n, st)| *n == p2 && *st == ModelState::Unloaded))
    })
    .await?;

    dev.shutdown().await;
    inf.shutdown().await;
    for m in [ma1, ma2, mb] {
        m.shutdown().await;
    }
    Ok(())
}
