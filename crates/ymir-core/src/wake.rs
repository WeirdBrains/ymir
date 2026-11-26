//! Waking the user's Bragi devices (M9 push wake-ups, Bragi B4). A Bragi device shares a wake
//! card, a sealed [`WAKE_TYPE`] record keyed by its peer id, with the user's full devices only:
//! a Ymir node that holds the seed reads them; a cert-only node never holds their key (no
//! service grant lists the type). A wake is a sealed 96-byte payload with no content; the relay
//! sees only the handle, and the device refuses repeats and stale wakes.
//!
//! Only the relay wake path is here (`/weirdnet/wake/1`, which a Bragi device keeps open when the
//! user turned it on); APNs and FCM need a push service and are not built.

use std::str::FromStr;
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use weirdnet_net::{Limits, Multiaddr, PeerId};
use weirdnet_push::relaywake::{self, Delivery};
use weirdnet_push::{Reason, WakeCard};
use weirdnet_sync::DeviceClass;

pub use crate::membership::WAKE_TYPE;
use crate::membership::{Membership, SyncNode};

/// One Bragi device's wake card record, as Bragi writes it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceWake {
    /// `None`: the device turned wake-ups off.
    pub card: Option<WakeCard>,
    /// The relay it keeps a wake connection to (`/dns4/…/p2p/RELAY`), while relay wake is on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relay: Option<String>,
}

/// A device this node can wake.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wakeable {
    pub peer_id: String,
    pub label: String,
    pub card: WakeCard,
    /// `None`: it shares a card but keeps no relay wake connection (push only, not built here).
    pub relay: Option<String>,
}

/// The active full devices' wake cards (a card from a service device is ignored; none should
/// exist). Empty on a cert-only node, which can't open them.
pub fn wakeable(m: &Membership) -> Vec<Wakeable> {
    if m.is_service() {
        return Vec::new();
    }
    let roster = m.roster();
    let me = m.peer_id().to_string();
    m.records(WAKE_TYPE)
        .into_iter()
        .filter(|(peer, _)| *peer != me)
        .filter_map(|(peer, v)| {
            let cert = roster.active().find(|c| c.peer_id == peer)?.clone();
            if roster.grant_of(&peer).is_some_and(|g| g.class == DeviceClass::Service) {
                return None;
            }
            let w: DeviceWake = serde_json::from_slice(&v).ok()?;
            Some(Wakeable { peer_id: peer, label: cert.label.clone(), card: w.card?, relay: w.relay })
        })
        .collect()
}

/// `/…/p2p/RELAY` as the relay's peer id and its dial address.
fn relay_target(addr: &str) -> Option<(PeerId, Vec<Multiaddr>)> {
    let (base, peer) = addr.rsplit_once("/p2p/")?;
    let peer = PeerId::from_str(peer).ok()?;
    Some((peer, Multiaddr::from_str(base).ok().map(|a| vec![a]).unwrap_or_default()))
}

/// Wakes `device` through its relay so it syncs (`reason`: a message or a connection waits).
pub async fn wake(node: &SyncNode, device: &Wakeable, reason: Reason) -> Result<Delivery> {
    let Some(relay) = &device.relay else {
        bail!("{} keeps no relay wake connection (relay wake is off there)", device.label);
    };
    let (rpeer, addrs) = relay_target(relay).ok_or_else(|| anyhow!("a bad relay address {relay:?}"))?;
    let payload = device.card.seal(reason).map_err(|e| anyhow!("sealing the wake: {e}"))?;
    let io = node.open_stream_async(rpeer, &addrs, relaywake::PROTOCOL, Limits::default()).await.map_err(|e| anyhow!("reaching the relay: {e}"))?;
    tokio::time::timeout(Duration::from_secs(30), relaywake::deliver(io, &device.card.handle, payload))
        .await
        .map_err(|_| anyhow!("the relay did not answer"))?
        .map_err(|e| anyhow!("the relay: {e}"))
}

/// Finds a device by peer id or label (exact, then case-insensitive).
pub fn find<'a>(devices: &'a [Wakeable], who: &str) -> Option<&'a Wakeable> {
    devices.iter().find(|d| d.peer_id == who || d.label == who).or_else(|| devices.iter().find(|d| d.label.eq_ignore_ascii_case(who)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_bragis_card_record() {
        // The shape Bragi 0.15.0 writes (bragi_core::wake::DeviceWake).
        let key = weirdnet_push::PushKey::generate();
        let key_hex: String = key.public().iter().map(|b| format!("{b:02x}")).collect();
        let card = WakeCard { service: "https://push.weirdbrains.com".into(), handle: weirdnet_push::WakeHandle::generate(), key: key_hex };
        let w = DeviceWake {
            card: Some(card.clone()),
            relay: Some("/ip4/127.0.0.1/tcp/1/p2p/12D3KooWDpJ7As7BWAwRMfu1VU2WCqNjvq387JEYKDBj4kx6nXTN".into()),
        };
        let back: DeviceWake = serde_json::from_slice(&serde_json::to_vec(&w).unwrap()).unwrap();
        assert_eq!(back, w);
        let off: DeviceWake = serde_json::from_str(r#"{"card":null}"#).unwrap();
        assert!(off.card.is_none() && off.relay.is_none());
        assert!(relay_target(w.relay.as_deref().unwrap()).is_some());
        assert!(relay_target("nonsense").is_none());
        // What Ymir seals, the device opens.
        let p = back.card.unwrap().seal(Reason::MessageWaiting).unwrap();
        assert_eq!(key.open(&card.handle, &p).unwrap().reason, Reason::MessageWaiting);
    }
}
