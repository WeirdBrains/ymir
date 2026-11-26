//! `ymirctl pool`: this node's place in the user's Weirdnet. Linking moves the identity Morph
//! (or another Ymir node) holds onto this node with a `weirdnet-ymir://` code and a 6-digit
//! check, as Morph links a phone; no recovery words are typed.

use std::io::{BufRead, Write};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use clap::Subcommand;
use weirdnet_device_link::{HostNode, HostStatus, Pairing, Received, with_link_limits};
use weirdnet_net::{Keypair, Multiaddr, NodeConfig as NetConfig, PeerHint};
use ymir_core::control::{self, ControlRequest};
use ymir_core::membership::{LinkSettings, Membership, SYNC_TYPE, ServiceRole};
use ymir_core::{NodeConfig, Paths};

/// The link scheme Ymir shows (`weirdnet-ymir://link/…`). Never bare `weirdnet://`.
pub const SCHEME: &str = "weirdnet-ymir://";

/// Morph's scheme. Its codes carry the same payload, so `pool add` takes them too.
pub const MORPH_SCHEME: &str = "weirdnet-morph://";

/// Bragi's scheme. Its link codes carry the same payload as Ymir's and Morph's.
pub const BRAGI_SCHEME: &str = "weirdnet-bragi://";

/// Reads a pasted code as a Ymir one: a `weirdnet-morph://link/` or `weirdnet-bragi://link/` code
/// is the same payload under another app's scheme. Anything else is left for the link check to refuse.
pub fn normalize_link(code: &str) -> String {
    let code = code.trim();
    for other in [MORPH_SCHEME, BRAGI_SCHEME] {
        if let Some(rest) = code.strip_prefix(other).filter(|r| r.starts_with("link/")) {
            return format!("{SCHEME}{rest}");
        }
    }
    code.to_string()
}

/// The device-link hint, which becomes the label in the signed roster: "Ymir on `<name>`".
pub fn link_hint(cfg: &NodeConfig) -> String {
    format!("Ymir on {}", label(cfg))
}

#[derive(Subcommand, Debug)]
pub enum PoolCmd {
    /// Join the user's Weirdnet: show a link code to open on a device that is already in it.
    Link {
        /// Only accept a cert-only link (a certificate and a service grant, never the seed).
        /// The other device sends one with `ymirctl pool add CODE --service ROLE`.
        #[arg(long)]
        cert_only: bool,
    },
    /// Move this node to cert-only: show a link code for a device that holds the seed
    /// (`ymirctl pool add CODE --service ROLE` there); then this node's `seed` file is
    /// overwritten and deleted. Stop ymird first.
    CertOnly,
    /// Link another device from this one: paste the code it shows.
    Add {
        /// The `weirdnet-ymir://` (or Morph's `weirdnet-morph://`, or Bragi's `weirdnet-bragi://`) code the other device shows.
        link: String,
        /// The name the other device gets for this user.
        #[arg(long, default_value = "Ymir")]
        profile_name: String,
        /// Skip the code check prompt (tests; the codes still have to match).
        #[arg(long)]
        yes: bool,
        /// Link a server cert-only, as `inference` (holds the pool's and conversations' keys)
        /// or `model` (roster only, no data keys): it gets a certificate and a user-signed
        /// service grant, never the seed, and cannot link or revoke devices.
        #[arg(long, value_name = "ROLE")]
        service: Option<String>,
    },
    /// The linked devices.
    Devices,
    /// Where this node (and the devices it dials) can be reached, for a device that can't find
    /// them on its own: paste one into Bragi's Link a device box. `pool add` sends these along.
    Addresses,
    /// Revoke a device everywhere this roster reaches.
    Revoke { peer: String },
    /// Bragi devices this node can wake (push wake-ups), or wake DEVICE (a peer id or label) so
    /// it syncs. Through the relay the device keeps a wake connection to; a seed holder only (a
    /// cert-only node never holds the wake cards). Sending needs the running node.
    Wake {
        device: Option<String>,
        /// Say a connection is waiting, not a message.
        #[arg(long)]
        connection: bool,
    },
    /// Start a new Weirdnet identity on this node, for a cloud of its own.
    Init {
        /// Yes, a new identity, separate from the one Morph holds.
        #[arg(long)]
        standalone: bool,
    },
}

fn label(cfg: &NodeConfig) -> String {
    if cfg.node.name.is_empty() { "ymir".to_string() } else { cfg.node.name.clone() }
}

/// The network settings for a short link session. It listens on a fresh port, so it works while
/// ymird holds the configured one.
fn link_config(cfg: &NodeConfig) -> Result<NetConfig> {
    let mut c = cfg.network.node_config(&label(cfg))?;
    if cfg.network.listen.is_none() {
        c.listen = NetConfig::default_listen(0);
    }
    Ok(with_link_limits(c))
}

/// Whether the running ymird holds the identity (then roster changes go through it).
async fn daemon_on_weirdnet(paths: &Paths) -> bool {
    match control::request(&paths.socket(), &ControlRequest::Status).await {
        Ok(r) => r.status.is_some_and(|s| s.network.is_some()),
        Err(_) => false,
    }
}

/// `addr` as a dialable `…/p2p/PEER` address, or `None` for one nobody else can dial
/// (unspecified, loopback when `keep_loopback` is off, or not an address at all).
fn dialable(addr: &str, peer: &str, keep_loopback: bool) -> Option<String> {
    let m: Multiaddr = addr.parse().ok()?;
    let s = m.to_string();
    let parts: Vec<&str> = s.split('/').collect();
    let mut has_peer = false;
    for w in parts.windows(2) {
        match w[0] {
            "ip4" | "ip6" => {
                let ip: std::net::IpAddr = w[1].parse().ok()?;
                if ip.is_unspecified() || (!keep_loopback && ip.is_loopback()) {
                    return None;
                }
            }
            "p2p" => has_peer = true,
            _ => {}
        }
    }
    Some(if has_peer { s } else { format!("{s}/p2p/{peer}") })
}

/// Where this node and the devices it dials can be reached: the running ymird's listen
/// addresses (or the configured ones), then the configured `[network] peers` that are linked
/// devices. Loopback addresses only when there is nothing else.
pub async fn addresses(paths: &Paths, cfg: &NodeConfig, m: &Membership) -> Vec<String> {
    let me = m.peer_id().to_string();
    let listening: Vec<String> = match control::request(&paths.socket(), &ControlRequest::Status).await {
        Ok(r) => r.status.and_then(|s| s.network).map(|n| n.listening).unwrap_or_default(),
        Err(_) => Vec::new(),
    };
    let listening = if listening.is_empty() { cfg.network.listen.clone().unwrap_or_default() } else { listening };
    let mut out: Vec<String> = listening.iter().filter_map(|a| dialable(a, &me, false)).collect();
    if out.is_empty() {
        out = listening.iter().filter_map(|a| dialable(a, &me, true)).collect();
    }
    let linked: Vec<String> = m.roster().active().map(|c| c.peer_id.clone()).filter(|p| *p != me).collect();
    for p in &cfg.network.peers {
        if let Ok(hint) = PeerHint::from_addr(p)
            && linked.contains(&hint.peer_id)
            && !out.contains(p)
        {
            out.push(p.clone());
        }
    }
    out
}

fn read(out: &mut dyn Write, input: &mut dyn BufRead, prompt: &str) -> Result<bool> {
    write!(out, "{prompt}")?;
    out.flush()?;
    let mut line = String::new();
    input.read_line(&mut line)?;
    Ok(matches!(line.trim(), "y" | "Y" | "yes"))
}

pub async fn run(paths: &Paths, cfg: &NodeConfig, cmd: PoolCmd, out: &mut dyn Write, input: &mut dyn BufRead) -> Result<()> {
    let dir = paths.pool();
    match cmd {
        PoolCmd::Init { standalone } => {
            if !standalone {
                bail!(
                    "this starts a Weirdnet identity of its own, separate from the one Morph holds; to join your devices use `ymirctl pool link`, or pass --standalone"
                );
            }
            let m = Membership::create(&dir, &label(cfg))?;
            writeln!(out, "New identity {} with this node ({}) as its first device", m.user_id(), m.peer_id())?;
            writeln!(out, "Restart ymird to put it on Weirdnet.")?;
            Ok(())
        }
        PoolCmd::Link { cert_only } => {
            if Membership::exists(&dir) {
                bail!("this node is already linked ({}); see `ymirctl pool devices`", dir.display());
            }
            let node_key = Keypair::generate_ed25519();
            let host = HostNode::start(SCHEME, node_key.clone(), link_config(cfg)?, &link_hint(cfg)).map_err(|e| anyhow!("{e}"))?;
            if cert_only {
                writeln!(out, "On a device that holds your identity, run `ymirctl pool add CODE --service inference` (or `--service model`) with:")?;
            } else {
                writeln!(out, "On a device that is already in your Weirdnet (Morph: Identity → Add device, or `ymirctl pool add`), open:")?;
            }
            writeln!(out, "\n  {}\n", host.link())?;
            writeln!(out, "The code is good for two minutes.")?;
            out.flush()?;
            let (m, label, settings) = match wait_for_link(&host, out, input).await? {
                Received::Full(t) if !cert_only => {
                    let m = Membership::adopt(&dir, node_key, &t.seed, t.node_cert.clone(), &t.settings)?;
                    (m, t.node_cert.label.clone(), t.settings.clone())
                }
                Received::Full(_) => bail!("the other device sent the identity seed, and this link asked for cert-only; nothing was stored"),
                Received::Service(c) => {
                    let m = Membership::adopt_service(&dir, node_key, &c.user_key, c.node_cert.clone(), &c.bundle)?;
                    (m, c.node_cert.label.clone(), c.settings.clone())
                }
                _ => bail!("the other device sent a kind of link this version doesn't know; nothing was stored"),
            };
            let kind = if m.is_service() { " cert-only" } else { "" };
            writeln!(out, "Linked{kind} as {label} ({}) to {}.", m.peer_id(), m.user_id())?;
            let peers = LinkSettings::peers_in(&settings);
            if !peers.is_empty() && (!cfg.network.discovery || !cfg.network.mdns) {
                writeln!(out, "With discovery off, add the devices it can reach to [network] peers in node.toml:")?;
                for p in &peers {
                    writeln!(out, "  {p}")?;
                }
            }
            writeln!(out, "Restart ymird to put it on Weirdnet.")?;
            Ok(())
        }
        PoolCmd::CertOnly => {
            let m = Membership::open(&dir)?.context("this node is not linked; run `ymirctl pool link --cert-only`")?;
            if m.is_service() {
                bail!("this node is already cert-only");
            }
            if control::request(&paths.socket(), &ControlRequest::Status).await.is_ok() {
                bail!("stop ymird first: it holds the identity and would write the seed back");
            }
            let node_key = m.node_key().clone();
            drop(m);
            let host = HostNode::start(SCHEME, node_key, link_config(cfg)?, &link_hint(cfg)).map_err(|e| anyhow!("{e}"))?;
            writeln!(out, "On a device that holds your identity, run `ymirctl pool add CODE --service inference` (or `--service model`) with:")?;
            writeln!(out, "\n  {}\n", host.link())?;
            writeln!(out, "The code is good for two minutes. Afterwards this node's seed file is overwritten and deleted.")?;
            out.flush()?;
            match wait_for_link(&host, out, input).await? {
                Received::Service(c) => {
                    let m = Membership::convert_to_service(&dir, &c.user_key, c.node_cert.clone(), &c.bundle)?;
                    writeln!(out, "{} ({}) is cert-only now; the seed file is gone. Start ymird again.", c.node_cert.label, m.peer_id())?;
                    Ok(())
                }
                Received::Full(_) => {
                    bail!("the other device sent the identity seed; run `ymirctl pool add CODE --service ROLE` there. Nothing changed")
                }
                _ => bail!("the other device sent a kind of link this version doesn't know; nothing changed"),
            }
        }
        PoolCmd::Add { link, profile_name, yes, service } => {
            let role = service.as_deref().map(ServiceRole::parse).transpose()?;
            let m = Membership::open(&dir)?.context("this node is not linked yet; run `ymirctl pool link` or `ymirctl pool init`")?;
            if m.is_service() {
                bail!("this node is linked cert-only and holds no seed; link devices from one that does");
            }
            let net = link_config(cfg)?;
            let link = normalize_link(&link);
            let mut pairing = tokio::task::spawn_blocking(move || Pairing::start(SCHEME, &link, net)).await?.map_err(|e| anyhow!("{e}"))?;
            writeln!(out, "Pairing with {}: the code is {}", pairing.device(), pairing.sas())?;
            if !yes && !read(out, input, "Does the other device show the same code? [y/N] ")? {
                let _ = tokio::task::spawn_blocking(move || pairing.cancel()).await;
                bail!("cancelled: the codes did not match");
            }
            // Where the pool's nodes are: a device on another network (Bragi on a laptop, the
            // nodes on a private network with discovery off) has no other way to find them.
            let peers = addresses(paths, cfg, &m).await;
            let told = peers.len();
            let settings = m.link_settings_with(peers);
            let sync = vec![SYNC_TYPE.to_string()];
            let on_weirdnet = daemon_on_weirdnet(paths).await;
            let cert = match (role, service) {
                (Some(role), Some(name)) => {
                    let peer = pairing.peer().to_string();
                    let bundle = if on_weirdnet {
                        // ymird holds the roster: the grant (and a first rotation) happen there.
                        let r = control::request(&paths.socket(), &ControlRequest::PoolServiceBundle { peer, role: name }).await?;
                        if !r.ok {
                            bail!("{}", r.message.unwrap_or_default());
                        }
                        zeroize::Zeroizing::new(hex::decode(r.message.unwrap_or_default()).context("the sync bundle")?)
                    } else {
                        m.service_bundle(&peer, role)?
                    };
                    let user = m.user_keypair()?;
                    tokio::task::spawn_blocking(move || pairing.send_credential(&user, &profile_name, &settings, &sync, &bundle))
                        .await?
                        .map_err(|e| anyhow!("{e}"))?
                }
                _ => {
                    let seed = m.seed()?;
                    tokio::task::spawn_blocking(move || pairing.send_with(&seed, &profile_name, &settings, &sync))
                        .await?
                        .map_err(|e| anyhow!("{e}"))?
                }
            };
            if on_weirdnet {
                let r = control::request(&paths.socket(), &ControlRequest::PoolApprove { cert: cert.clone() }).await?;
                if !r.ok {
                    bail!("{}", r.message.unwrap_or_default());
                }
            } else {
                m.approve(cert.clone())?;
            }
            let kind = if role.is_some() { " cert-only" } else { "" };
            writeln!(out, "Linked {} ({}){kind}.", cert.label, cert.peer_id)?;
            if told > 0 {
                writeln!(out, "Sent it {told} address{} to reach your nodes at (`ymirctl pool addresses`).", if told == 1 { "" } else { "es" })?;
            }
            Ok(())
        }
        PoolCmd::Addresses => {
            let m = Membership::open(&dir)?.context("this node is not linked yet; run `ymirctl pool link`")?;
            let all = addresses(paths, cfg, &m).await;
            if all.is_empty() {
                bail!("no address to share: ymird isn't listening on Weirdnet (start it, or set [network] listen)");
            }
            for a in all {
                writeln!(out, "{a}")?;
            }
            Ok(())
        }
        PoolCmd::Devices => {
            let m = Membership::open(&dir)?.context("this node is not linked yet; run `ymirctl pool link`")?;
            let me = m.peer_id().to_string();
            let kind = if m.is_service() { " (this node is cert-only)" } else { "" };
            writeln!(out, "Identity {}{kind}", m.user_id())?;
            for c in &m.roster().certs {
                let state = if c.peer_id == me {
                    "this node"
                } else if c.active() {
                    "linked"
                } else {
                    "revoked"
                };
                // The class and roles a device's user-signed grant gives it (labels are not signed).
                let grant = match m.grant_of(&c.peer_id) {
                    Some((weirdnet_sync::DeviceClass::Service, roles)) => format!("  cert-only {}", roles.join(",")),
                    Some((_, roles)) if !roles.is_empty() => format!("  {}", roles.join(",")),
                    _ => String::new(),
                };
                writeln!(out, "  {:<20} {:<10} {}{grant}", c.label, state, c.peer_id)?;
            }
            Ok(())
        }
        PoolCmd::Revoke { peer } => {
            if daemon_on_weirdnet(paths).await {
                let r = control::request(&paths.socket(), &ControlRequest::PoolRevoke { peer }).await?;
                if !r.ok {
                    bail!("{}", r.message.unwrap_or_default());
                }
                writeln!(out, "{}", r.message.unwrap_or_default())?;
            } else {
                let m = Membership::open(&dir)?.context("this node is not linked")?;
                let c = m.revoke(&peer)?;
                writeln!(out, "{} ({}) is revoked", c.label, c.peer_id)?;
            }
            Ok(())
        }
        PoolCmd::Wake { device: None, .. } => {
            let m = Membership::open(&dir)?.context("this node is not linked")?;
            if m.is_service() {
                writeln!(out, "This node is linked cert-only: it never holds the wake cards; wake from one that holds the seed.")?;
                return Ok(());
            }
            let devices = ymir_core::wake::wakeable(&m);
            if devices.is_empty() {
                writeln!(out, "No device shares a wake card yet (Bragi 0.15.0 or newer, wake-ups on).")?;
            }
            for d in devices {
                let how = if d.relay.is_some() { "through its relay" } else { "push only (not built here)" };
                writeln!(out, "{}  {}  {how}", d.label, d.peer_id)?;
            }
            Ok(())
        }
        PoolCmd::Wake { device: Some(device), connection } => {
            if !daemon_on_weirdnet(paths).await {
                bail!("waking a device needs the running node (ymird on Weirdnet)");
            }
            let reason = if connection { "connection" } else { "message" }.to_string();
            let r = control::request(&paths.socket(), &ControlRequest::PoolWake { device, reason }).await?;
            if !r.ok {
                bail!("{}", r.message.unwrap_or_default());
            }
            writeln!(out, "{}", r.message.unwrap_or_default())?;
            Ok(())
        }
    }
}

/// Waits for the other device on a link code: asks the user to compare the 6-digit code, then
/// returns what arrived (the seed, or a cert-only credential).
async fn wait_for_link(host: &HostNode, out: &mut dyn Write, input: &mut dyn BufRead) -> Result<Received> {
    let mut asked = false;
    loop {
        match host.status() {
            HostStatus::Waiting => {}
            HostStatus::Paired { sas } | HostStatus::Received { sas } if !asked => {
                asked = true;
                if !read(out, input, &format!("Does the other device show {sas}? [y/N] "))? {
                    host.cancel();
                    bail!("cancelled: the codes did not match");
                }
                host.confirm();
                writeln!(out, "Waiting for the other device to confirm…")?;
            }
            HostStatus::Received { .. } => return host.take_any().map_err(|e| anyhow!("{e}")),
            HostStatus::Paired { .. } => {}
            HostStatus::Done => bail!("the link was already used"),
            HostStatus::Cancelled => bail!("the other device cancelled"),
            HostStatus::Expired => bail!("the code expired; run the command again"),
            HostStatus::Failed(why) => bail!("linking failed: {why}"),
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn morph_codes_read_as_ymir_codes() {
        assert_eq!(normalize_link("  weirdnet-morph://link/abc \n"), "weirdnet-ymir://link/abc");
        assert_eq!(normalize_link("weirdnet-ymir://link/abc"), "weirdnet-ymir://link/abc");
        // Only link codes convert; the rest fails the link check as before.
        assert_eq!(normalize_link("weirdnet-morph://invite/abc"), "weirdnet-morph://invite/abc");
        assert_eq!(normalize_link(" weirdnet-bragi://link/abc\n"), "weirdnet-ymir://link/abc");
        assert_eq!(normalize_link("weirdnet-bragi://invite/abc"), "weirdnet-bragi://invite/abc");
        assert_eq!(normalize_link("weirdnet://link/abc"), "weirdnet://link/abc");
    }

    #[test]
    fn shared_addresses_are_dialable() {
        let peer = "12D3KooWQH96LGJd33JquZXNyrmrs7HAeaeviiFqdwnzwfdmj1VY";
        assert_eq!(dialable("/ip4/192.0.2.7/tcp/4848", peer, false).unwrap(), format!("/ip4/192.0.2.7/tcp/4848/p2p/{peer}"));
        let full = format!("/ip4/192.0.2.7/udp/4848/quic-v1/p2p/{peer}");
        assert_eq!(dialable(&full, peer, false).unwrap(), full);
        assert_eq!(dialable("/ip4/0.0.0.0/tcp/4848", peer, true), None);
        assert_eq!(dialable("/ip4/127.0.0.1/tcp/4848", peer, false), None);
        assert!(dialable("/ip4/127.0.0.1/tcp/4848", peer, true).is_some());
        assert_eq!(dialable("not an address", peer, true), None);
    }

    #[test]
    fn link_settings_carry_addresses_and_older_ones_read_as_none() {
        assert_eq!(
            LinkSettings::peers_in(r#"{"roster":{"epoch":0,"certs":[]},"peers":["/ip4/192.0.2.7/tcp/4848/p2p/x"]}"#),
            vec!["/ip4/192.0.2.7/tcp/4848/p2p/x"]
        );
        assert!(LinkSettings::peers_in(r#"{"roster":{"epoch":0,"certs":[]}}"#).is_empty());
        assert!(LinkSettings::peers_in("").is_empty());
    }

    #[test]
    fn the_hint_reads_ymir_on_the_node_name() {
        let mut cfg = NodeConfig::default();
        cfg.node.name = "dev-box".into();
        assert_eq!(link_hint(&cfg), "Ymir on dev-box");
        cfg.node.name.clear();
        assert_eq!(link_hint(&cfg), "Ymir on ymir");
    }
}
