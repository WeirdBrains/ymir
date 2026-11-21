//! Engine events as alert observations.
//!
//! Interim: `weirdnet-alerts` 0.1.0 has this as `engine::monitor` behind its `backends` feature, but
//! that feature needs `weirdnet-backends` 0.1, and Ymir is on 0.2 (a different `Supervisor` type).
//! Drop this module and use `weirdnet_alerts::engine` again once `weirdnet-alerts` supports
//! `weirdnet-backends` 0.2 (noted for the Weirdnet Lead in HANDOFF.md).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{broadcast, watch};
use weirdnet_alerts::{Alerter, Observation, probe};
use weirdnet_backends::{Supervisor, SupervisorEvent};

/// A supervisor event as an observation: `engine.<kind>` with the event's fields.
pub fn observation(ev: &SupervisorEvent) -> Observation {
    let v = serde_json::to_value(ev).unwrap_or_default();
    let kind = v.get("kind").and_then(|k| k.as_str()).unwrap_or("unknown").to_string();
    Observation::new(format!("engine.{kind}"), v)
}

async fn probes(disks: &[PathBuf]) -> Vec<Observation> {
    let mut out = Vec::new();
    out.extend(probe::route().await);
    out.extend(disks.iter().filter_map(|d| probe::disk(d)));
    out.extend(probe::gpus().await);
    out
}

/// Runs until `stop` is true: forwards the supervisor's events and runs the probes every `every`.
pub async fn monitor(alerter: Arc<Alerter>, supervisor: Option<Supervisor>, disks: Vec<PathBuf>, every: Duration, mut stop: watch::Receiver<bool>) {
    let mut events = supervisor.as_ref().map(Supervisor::subscribe);
    let mut tick = tokio::time::interval(every);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            _ = async { let _ = stop.wait_for(|s| *s).await; } => return,
            ev = next_event(&mut events) => match ev {
                Ok(ev) => { alerter.observe(&observation(&ev)).await; }
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
