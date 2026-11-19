//! Health monitoring: engine events and probes, fed to the alerter.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{broadcast, watch};
use weirdnet_backends::{Supervisor, SupervisorEvent};
use ymir_alerts::{Alerter, Observation, probe};

/// An engine event as an alert observation: `engine.<kind>` with the event's fields.
pub fn engine_observation(ev: &SupervisorEvent) -> Observation {
    let v = serde_json::to_value(ev).unwrap_or_default();
    let kind = v.get("kind").and_then(|k| k.as_str()).unwrap_or("unknown").to_string();
    Observation::new(format!("engine.{kind}"), v)
}

/// Runs until `stop`: forwards engine events, and probes the route, disks and GPUs every
/// `every`.
pub async fn monitor(alerter: Arc<Alerter>, supervisor: Option<Supervisor>, disks: Vec<PathBuf>, every: Duration, mut stop: watch::Receiver<bool>) {
    let mut events = supervisor.as_ref().map(Supervisor::subscribe);
    let mut tick = tokio::time::interval(every);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            _ = async { let _ = stop.wait_for(|s| *s).await; } => return,
            ev = next_event(&mut events) => match ev {
                Ok(ev) => { alerter.observe(&engine_observation(&ev)).await; }
                Err(broadcast::error::RecvError::Lagged(n)) => tracing::warn!("alerts: skipped {n} engine events"),
                Err(broadcast::error::RecvError::Closed) => events = None,
            },
            _ = tick.tick() => {
                for obs in probes(&disks).await {
                    alerter.observe(&obs).await;
                }
            }
        }
    }
}

async fn next_event(rx: &mut Option<broadcast::Receiver<SupervisorEvent>>) -> Result<SupervisorEvent, broadcast::error::RecvError> {
    match rx {
        Some(r) => r.recv().await,
        None => std::future::pending().await,
    }
}

/// One round of probes.
pub async fn probes(disks: &[PathBuf]) -> Vec<Observation> {
    let mut out = Vec::new();
    if let Some(r) = probe::route().await {
        out.push(r);
    }
    let mut seen = std::collections::HashSet::new();
    for d in disks {
        // Probe each filesystem once, at the first path that exists on it.
        let existing = d.ancestors().find(|p| p.exists()).map(|p| p.to_path_buf());
        if let Some(p) = existing
            && let Some(o) = probe::disk(&p)
            && seen.insert(dev_of(&p))
        {
            out.push(o);
        }
    }
    out.extend(probe::gpus().await);
    out
}

#[cfg(unix)]
fn dev_of(p: &std::path::Path) -> u64 {
    use std::os::unix::fs::MetadataExt;
    std::fs::metadata(p).map(|m| m.dev()).unwrap_or(0)
}

#[cfg(not(unix))]
fn dev_of(_p: &std::path::Path) -> u64 {
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_events_become_observations() {
        let o = engine_observation(&SupervisorEvent::CrashLoop { model: "q".into(), crashes: 5 });
        assert_eq!(o.kind, "engine.crash_loop");
        assert_eq!(o.fields["model"], "q");
    }
}
