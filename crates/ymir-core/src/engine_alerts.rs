//! Engine events as alerts, interim: `weirdnet-alerts` 0.2.1's `engine` module (feature
//! `backends`) accepts only `weirdnet-backends` below 0.4, and Ymir needs 0.4 for LoRA
//! adapters. This is the same forwarding (an event becomes `engine.<kind>` with its fields)
//! over 0.4's supervisor. To be dropped once `weirdnet-alerts` takes `weirdnet-backends` 0.4
//! (asked of the Weirdnet lead); the rules, dedupe and sinks stay in `weirdnet-alerts`.

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

/// Runs until `stop` is true: forwards the supervisor's events and runs the
/// `weirdnet-alerts` probes (route, `disks`, GPUs) every `every`.
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
                let mut obs = Vec::new();
                obs.extend(probe::route().await);
                obs.extend(disks.iter().filter_map(|d| probe::disk(d)));
                obs.extend(probe::gpus().await);
                for o in obs {
                    alerter.observe(&o).await;
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
