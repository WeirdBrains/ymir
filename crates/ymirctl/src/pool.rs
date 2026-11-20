//! `ymirctl pool`: this node's place in the user's Weirdnet. Linking moves the identity Morph
//! (or another Ymir node) holds onto this node with a `weirdnet-ymir://` code and a 6-digit
//! check, as Morph links a phone; no recovery words are typed.

use std::io::{BufRead, Write};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use clap::Subcommand;
use weirdnet_device_link::{HostNode, HostStatus, Pairing, with_link_limits};
use weirdnet_net::{Keypair, NodeConfig as NetConfig};
use ymir_core::control::{self, ControlRequest};
use ymir_core::membership::{Membership, SYNC_TYPE};
use ymir_core::{NodeConfig, Paths};

/// The link scheme Ymir shows (`weirdnet-ymir://link/…`). Never bare `weirdnet://`.
pub const SCHEME: &str = "weirdnet-ymir://";

/// Morph's scheme. Its codes carry the same payload, so `pool add` takes them too.
pub const MORPH_SCHEME: &str = "weirdnet-morph://";

/// Reads a pasted code as a Ymir one: a `weirdnet-morph://link/` code is the same payload under
/// Morph's scheme. Anything else is left for the link check to refuse.
pub fn normalize_link(code: &str) -> String {
    let code = code.trim();
    match code.strip_prefix(MORPH_SCHEME) {
        Some(rest) if rest.starts_with("link/") => format!("{SCHEME}{rest}"),
        _ => code.to_string(),
    }
}

/// The device-link hint, which becomes the label in the signed roster: "Ymir on <name>".
pub fn link_hint(cfg: &NodeConfig) -> String {
    format!("Ymir on {}", label(cfg))
}

#[derive(Subcommand, Debug)]
pub enum PoolCmd {
    /// Join the user's Weirdnet: show a link code to open on a device that is already in it.
    Link,
    /// Link another device from this one: paste the code it shows.
    Add {
        /// The `weirdnet-ymir://` (or Morph's `weirdnet-morph://`) code the other device shows.
        link: String,
        /// The name the other device gets for this user.
        #[arg(long, default_value = "Ymir")]
        profile_name: String,
        /// Skip the code check prompt (tests; the codes still have to match).
        #[arg(long)]
        yes: bool,
    },
    /// The linked devices.
    Devices,
    /// Revoke a device everywhere this roster reaches.
    Revoke { peer: String },
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
        PoolCmd::Link => {
            if Membership::exists(&dir) {
                bail!("this node is already linked ({}); see `ymirctl pool devices`", dir.display());
            }
            let node_key = Keypair::generate_ed25519();
            let host = HostNode::start(SCHEME, node_key.clone(), link_config(cfg)?, &link_hint(cfg)).map_err(|e| anyhow!("{e}"))?;
            writeln!(out, "On a device that is already in your Weirdnet (Morph: Identity → Add device, or `ymirctl pool add`), open:")?;
            writeln!(out, "\n  {}\n", host.link())?;
            writeln!(out, "The code is good for two minutes.")?;
            out.flush()?;
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
                    HostStatus::Received { .. } => {
                        let t = host.take().map_err(|e| anyhow!("{e}"))?;
                        let m = Membership::adopt(&dir, node_key, &t.seed, t.node_cert.clone(), &t.settings)?;
                        writeln!(out, "Linked as {} ({}) to {}.", t.node_cert.label, m.peer_id(), m.user_id())?;
                        writeln!(out, "Restart ymird to put it on Weirdnet.")?;
                        return Ok(());
                    }
                    HostStatus::Paired { .. } => {}
                    HostStatus::Done => bail!("the link was already used"),
                    HostStatus::Cancelled => bail!("the other device cancelled"),
                    HostStatus::Expired => bail!("the code expired; run `ymirctl pool link` again"),
                    HostStatus::Failed(why) => bail!("linking failed: {why}"),
                }
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
        }
        PoolCmd::Add { link, profile_name, yes } => {
            let m = Membership::open(&dir)?.context("this node is not linked yet; run `ymirctl pool link` or `ymirctl pool init`")?;
            let net = link_config(cfg)?;
            let link = normalize_link(&link);
            let mut pairing = tokio::task::spawn_blocking(move || Pairing::start(SCHEME, &link, net)).await?.map_err(|e| anyhow!("{e}"))?;
            writeln!(out, "Pairing with {}: the code is {}", pairing.device(), pairing.sas())?;
            if !yes && !read(out, input, "Does the other device show the same code? [y/N] ")? {
                let _ = tokio::task::spawn_blocking(move || pairing.cancel()).await;
                bail!("cancelled: the codes did not match");
            }
            let seed = m.seed()?;
            let settings = m.link_settings();
            let sync = vec![SYNC_TYPE.to_string()];
            let cert =
                tokio::task::spawn_blocking(move || pairing.send_with(&seed, &profile_name, &settings, &sync)).await?.map_err(|e| anyhow!("{e}"))?;
            if daemon_on_weirdnet(paths).await {
                let r = control::request(&paths.socket(), &ControlRequest::PoolApprove { cert: cert.clone() }).await?;
                if !r.ok {
                    bail!("{}", r.message.unwrap_or_default());
                }
            } else {
                m.approve(cert.clone())?;
            }
            writeln!(out, "Linked {} ({}).", cert.label, cert.peer_id)?;
            Ok(())
        }
        PoolCmd::Devices => {
            let m = Membership::open(&dir)?.context("this node is not linked yet; run `ymirctl pool link`")?;
            let me = m.peer_id().to_string();
            writeln!(out, "Identity {}", m.user_id())?;
            for c in &m.roster().certs {
                let state = if c.peer_id == me {
                    "this node"
                } else if c.active() {
                    "linked"
                } else {
                    "revoked"
                };
                writeln!(out, "  {:<20} {:<10} {}", c.label, state, c.peer_id)?;
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
        assert_eq!(normalize_link("weirdnet://link/abc"), "weirdnet://link/abc");
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
