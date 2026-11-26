//! M6, placement and replication between real daemons: one inference node (it holds the seed and
//! no cert-only inference node exists, so it places alone) with `[inference.placement] replicas =
//! { alpha = 2 }`, and three model nodes that all hold `alpha`. Nothing is loaded by hand:
//! `alpha` comes up on two of them. Draining one brings it up on the third; resuming the drained
//! one leaves three, and an idle extra is unloaded again, back to two.

use std::time::Duration;

use anyhow::{Result, bail};
use ymir_core::control::{self, ControlRequest, ModelState};
use ymir_core::membership::Membership;

#[path = "common/pool.rs"]
mod pool;
use pool::*;

/// The model nodes with `alpha` ready, as the inference node sees them, and the draining ones.
async fn ready(h: &Home) -> Result<(Vec<String>, Vec<String>)> {
    let s = control::request(&h.paths.socket(), &ControlRequest::Status).await?.status.unwrap();
    let mut r: Vec<String> = s.models.iter().filter(|m| m.id == "alpha" && m.state == ModelState::Ready).map(|m| m.node.clone()).collect();
    r.sort();
    let drained = s.nodes.iter().filter(|n| n.draining).map(|n| n.name.clone()).collect();
    Ok((r, drained))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_scheduler_keeps_two_replicas_loaded() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();
    let inf_tmp = home(root, "inf", "inference", &[], &[])?;
    let first = Membership::create(&inf_tmp.paths.pool(), "inf")?;
    let ms: Vec<Home> = (1..=3).map(|i| home(root, &format!("ma{i}"), "model", &[], &["alpha"])).collect::<Result<_>>()?;
    for (i, h) in ms.iter().enumerate() {
        prelink(&first, h, &format!("ma{}", i + 1));
    }
    drop(first);
    let mut daemons = Vec::new();
    let mut addrs = Vec::new();
    for h in &ms {
        daemons.push(start(h).await?);
        addrs.push(addr(h).await?);
    }
    let inf_h = home(root, "inf", "inference", &addrs, &[])?;
    let p = inf_h.home.join("node.toml");
    let t = std::fs::read_to_string(&p)?;
    std::fs::write(&p, t.replacen("[inference.alerts]", "[inference.placement]\nreplicas = { alpha = 2 }\nevery_secs = 2\n[inference.alerts]", 1))?;
    let inf = start(&inf_h).await?;
    eventually(60, "all three model nodes join", async || Ok(ctl(&inf_h.home, &["nodes"]).await?.matches("serving").count() == 3)).await?;

    eventually(60, "alpha ready on two nodes", async || Ok(ready(&inf_h).await?.0.len() == 2)).await?;
    // It stays at two (no overshoot while records catch up).
    for _ in 0..4 {
        tokio::time::sleep(Duration::from_secs(2)).await;
        let (r, _) = ready(&inf_h).await?;
        if r.len() != 2 {
            bail!("alpha on {r:?}, not two nodes");
        }
    }
    let (two, _) = ready(&inf_h).await?;

    // Drain one replica: the scheduler loads alpha on the third node.
    ctl(&inf_h.home, &["drain", &two[0]]).await?;
    eventually(60, "alpha on the third node", async || {
        let (r, d) = ready(&inf_h).await?;
        Ok(d == [two[0].clone()] && r.iter().filter(|n| **n != two[0]).count() == 2)
    })
    .await?;
    // Resume it: three serve, so an idle extra unloads, back to two.
    ctl(&inf_h.home, &["drain", &two[0], "--resume"]).await?;
    eventually(60, "back to two replicas", async || {
        let (r, d) = ready(&inf_h).await?;
        Ok(d.is_empty() && r.len() == 2)
    })
    .await?;

    inf.shutdown().await;
    for d in daemons {
        d.shutdown().await;
    }
    Ok(())
}
