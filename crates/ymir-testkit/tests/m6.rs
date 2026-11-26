//! M6: workload spreading. Separate daemons over Weirdnet on 127.0.0.1 with a mock engine:
//! two inference nodes, three model nodes (two replicas of `alpha`, one `beta`) and two app
//! devices. One device fans a batch out; the other chats interactively at the same time. The
//! interactive answers stay fast (p95 under a target), the batch device is not starved (every
//! task answers), and the per-device cap on batch work (`batch_per_device`) is what keeps one
//! device from filling the inference nodes' queues.

use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use ymir_core::membership::Membership;

#[path = "common/pool.rs"]
mod pool;
use pool::*;

fn p95(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[((v.len() as f64 * 0.95).ceil() as usize).clamp(1, v.len()) - 1]
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn interactive_stays_fast_while_another_device_runs_a_batch() -> Result<()> {
    // SAFETY: set before any daemon starts in this test.
    unsafe { std::env::set_var("YMIR_MOCK_WORD_MS", "40") };
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();
    let i1_tmp = home(root, "inf1", "inference", &[], &[])?;
    let first = Membership::create(&i1_tmp.paths.pool(), "inf1")?;
    let i2_tmp = home(root, "inf2", "inference", &[], &[])?;
    let (ma1_tmp, ma2_tmp, mb_tmp) =
        (home(root, "ma1", "model", &[], &["alpha"])?, home(root, "ma2", "model", &[], &["alpha"])?, home(root, "mb", "model", &[], &["beta"])?);
    let (ap_tmp, bp_tmp) = (home(root, "dev-batch", "app", &[], &[])?, home(root, "dev-chat", "app", &[], &[])?);
    for (h, label) in [(&i2_tmp, "inf2"), (&ma1_tmp, "ma1"), (&ma2_tmp, "ma2"), (&mb_tmp, "mb"), (&ap_tmp, "dev-batch"), (&bp_tmp, "dev-chat")] {
        prelink(&first, h, label);
    }
    drop(first);

    let (ma1, ma2, mb) = (start(&ma1_tmp).await?, start(&ma2_tmp).await?, start(&mb_tmp).await?);
    let models = vec![addr(&ma1_tmp).await?, addr(&ma2_tmp).await?, addr(&mb_tmp).await?];
    let i1_h = home(root, "inf1", "inference", &models, &[])?;
    network_line(&i1_h, "batch_per_device = 2")?;
    let inf1 = start(&i1_h).await?;
    let i1_addr = addr(&i1_h).await?;
    let mut peers = models.clone();
    peers.push(i1_addr.clone());
    let i2_h = home(root, "inf2", "inference", &peers, &[])?;
    network_line(&i2_h, "batch_per_device = 2")?;
    let inf2 = start(&i2_h).await?;
    let i2_addr = addr(&i2_h).await?;
    let ap_h = home(root, "dev-batch", "app", &[i1_addr.clone(), i2_addr.clone()], &[])?;
    let batch_dev = start(&ap_h).await?;
    let bp_h = home(root, "dev-chat", "app", &[i1_addr, i2_addr], &[])?;
    let chat_dev = start(&bp_h).await?;
    for h in [&i1_h, &i2_h] {
        eventually(60, "all three model nodes join", async || Ok(ctl(&h.home, &["nodes"]).await?.matches("serving").count() == 3)).await?;
    }

    // Warm up both devices (admission, and the models load once).
    for (d, h) in [(&batch_dev, &ap_h), (&chat_dev, &bp_h)] {
        let deadline = Instant::now() + Duration::from_secs(100);
        loop {
            match stream_chat(d, h, "alpha", "warm up", async |_| {}).await {
                Ok(_) => break,
                Err(e) if Instant::now() > deadline => bail!("a device was never admitted: {e:#}"),
                Err(_) => tokio::time::sleep(Duration::from_millis(500)).await,
            }
        }
    }
    stream_chat(&chat_dev, &bp_h, "beta", "warm up beta", async |_| {}).await?;

    // Baseline: five questions with nothing else running.
    let mut quiet = Vec::new();
    for i in 0..5 {
        let model = if i % 3 == 2 { "beta" } else { "alpha" };
        let t = Instant::now();
        stream_chat(&chat_dev, &bp_h, model, &format!("quiet {i}"), async |_| {}).await?;
        quiet.push(t.elapsed());
    }

    // The batch device: 24 prompts, 8 at a time, on a blocking thread (`ymirctl` isn't Send).
    let prompts: String = (0..24).map(|i| format!("batch question {i}\n")).collect();
    let file = root.join("prompts.txt");
    std::fs::write(&file, prompts)?;
    let batch = {
        let (home, file) = (ap_h.home.clone(), file.display().to_string());
        let rt = tokio::runtime::Handle::current();
        tokio::task::spawn_blocking(move || rt.block_on(ctl(&home, &["batch", &file, "--model", "alpha", "--concurrency", "8", "--retries", "40"])))
    };
    // Meanwhile the other device chats interactively, one question after another.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let mut waits = Vec::new();
    for i in 0..10 {
        let model = if i % 3 == 2 { "beta" } else { "alpha" };
        let q = format!("interactive {i}");
        let t = Instant::now();
        let text = stream_chat(&chat_dev, &bp_h, model, &q, async |_| {}).await?;
        waits.push(t.elapsed());
        assert!(text.contains(&q), "{text}");
    }
    let out = batch.await??;

    // The batch device is not starved: all 24 answered, in order, each to its own question.
    let lines: Vec<&str> = out.lines().filter(|l| !l.starts_with('#')).collect();
    assert_eq!(lines.len(), 24, "{out}");
    for (i, l) in lines.iter().enumerate() {
        assert!(l.starts_with(&format!("{i}\t")) && l.ends_with(&format!("batch question {i}")), "{l}");
    }
    assert!(out.contains("24 task(s), 0 failed"), "{out}");
    // The interactive device stayed fast while that ran: a whole answer is about half a second.
    let (worst, p) = (waits.iter().max().unwrap(), p95(waits.clone()));
    let base = p95(quiet.clone());
    eprintln!("interactive: quiet p95 {base:?}; with a batch running p95 {p:?}, worst {worst:?} over {} answers; all {waits:?}", waits.len());
    // The target: under a batch, interactive p95 stays within 2 s plus twice the quiet p95 (the
    // box this runs on is shared and noisy, so the bound is relative).
    assert!(p < Duration::from_secs(2) + base * 2, "interactive p95 {p:?} with a batch running, {base:?} without");

    batch_dev.shutdown().await;
    chat_dev.shutdown().await;
    inf1.shutdown().await;
    inf2.shutdown().await;
    for m in [ma1, ma2, mb] {
        m.shutdown().await;
    }
    Ok(())
}
