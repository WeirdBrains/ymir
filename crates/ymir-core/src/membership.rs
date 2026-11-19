//! This device's place in the user's Weirdnet: the user identity, its own node key and
//! certificate, and the linked-devices roster it shares with Morph (decision 18).
//!
//! Stored under `<state>/pool/` (0700), secrets in 0600 files: `seed` (the user key seed, as
//! service nodes hold it for now, decision 12), `node.key`, `keys` (epoch keys), and the public
//! `roster.json`. The OS keyring comes later, as for the bridge token.
//!
//! The roster is kept current by `weirdnet-sync` (`/weirdnet/sync/2`): [`Membership::tick`] runs
//! a pass against every linked device, as Morph does, so devices linked or revoked elsewhere reach
//! this node. Ymir has no synced record types yet (context sync is M5), so a pass carries the
//! roster and the key chain only.
//!
//! Generic, and like Morph's own copy: the snapshot and merge-back around [`weirdnet_sync::pull_from`]
//! belong in `weirdnet-sync` (noted for the Weirdnet lead).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use weirdnet_net::user::NodeCert;
use weirdnet_net::{Keypair, Multiaddr, Node, PeerId, PublicKey, WEIRDNET};
use weirdnet_sync::{Device, EpochKeys, Roster, SyncState};

/// The sync type Ymir announces. It carries no records yet; the roster rides along with every
/// pass.
pub const SYNC_TYPE: &str = "ymir.pool";

/// What a link transfer carries besides the identity: the roster, in Morph's format, so a Morph
/// device and a Ymir node can link each other.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LinkSettings {
    #[serde(default)]
    pub roster: Roster,
}

pub type SyncNode = Node<Vec<u8>, Vec<u8>>;

pub struct Membership {
    dir: PathBuf,
    node_key: Keypair,
    state: Arc<Mutex<SyncState>>,
}

fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

/// The user key from its 32-byte seed.
fn user_from_seed(seed: &[u8]) -> Result<Keypair> {
    let mut bytes: [u8; 32] = seed.try_into().map_err(|_| anyhow!("the identity seed is not 32 bytes"))?;
    let key = Keypair::ed25519_from_bytes(&mut bytes).map_err(|e| anyhow!("identity seed: {e}"))?;
    Ok(key)
}

fn write_private(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    {
        let mut o = std::fs::OpenOptions::new();
        o.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            o.mode(0o600);
        }
        let mut f = o.open(&tmp).with_context(|| format!("writing {}", tmp.display()))?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

fn ensure_dir(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

impl Membership {
    /// Whether `dir` holds an identity.
    pub fn exists(dir: &Path) -> bool {
        dir.join("seed").is_file() && dir.join("node.key").is_file()
    }

    /// A new Weirdnet user identity with this device as its first node. For a Ymir cloud of its
    /// own, or tests; to join Morph's devices, link from Morph instead.
    pub fn create(dir: &Path, label: &str) -> Result<Self> {
        if Self::exists(dir) {
            bail!("{} already holds an identity", dir.display());
        }
        let mut seed = [0u8; 32];
        getrandom::fill(&mut seed).map_err(|e| anyhow!("random seed: {e}"))?;
        let user = user_from_seed(&seed)?;
        let node_key = Keypair::generate_ed25519();
        let cert = Roster::self_certify(&WEIRDNET, &user, &node_key.public().to_peer_id().to_string(), label, now_ms())
            .map_err(|e| anyhow!("certifying this node: {e}"))?;
        let mut roster = Roster::new();
        roster.upsert(cert.clone());
        let m = Self::build(dir, seed.to_vec(), node_key, cert, roster, EpochKeys::default())?;
        m.save()?;
        Ok(m)
    }

    /// The identity a device link delivered: `seed`, the node certificate the existing device
    /// made for `node_key`, and its roster.
    pub fn adopt(dir: &Path, node_key: Keypair, seed: &[u8], cert: NodeCert, settings: &str) -> Result<Self> {
        if Self::exists(dir) {
            bail!("{} already holds an identity", dir.display());
        }
        let user = user_from_seed(seed)?;
        let peer = node_key.public().to_peer_id().to_string();
        if cert.peer_id != peer || cert.user_id != weirdnet_net::user::user_id(&user.public()) || !WEIRDNET.verify_cert(&user.public(), &cert) {
            bail!("the link's certificate is not for this node and identity");
        }
        let given: LinkSettings =
            if settings.trim().is_empty() { LinkSettings::default() } else { serde_json::from_str(settings).unwrap_or_default() };
        let mut roster = Roster::new();
        roster.merge(&given.roster, &user.public(), &WEIRDNET);
        roster.upsert(cert.clone());
        let m = Self::build(dir, seed.to_vec(), node_key, cert, roster, EpochKeys::default())?;
        m.save()?;
        Ok(m)
    }

    /// The identity stored in `dir`, if any.
    pub fn open(dir: &Path) -> Result<Option<Self>> {
        if !Self::exists(dir) {
            return Ok(None);
        }
        let seed = std::fs::read(dir.join("seed")).context("reading the identity seed")?;
        let node_key = Keypair::from_protobuf_encoding(&std::fs::read(dir.join("node.key")).context("reading the node key")?)
            .map_err(|e| anyhow!("node key: {e}"))?;
        let roster: Roster = match std::fs::read(dir.join("roster.json")) {
            Ok(b) => serde_json::from_slice(&b).context("reading roster.json")?,
            Err(_) => Roster::new(),
        };
        let keys = std::fs::read(dir.join("keys")).ok().and_then(|b| EpochKeys::import(&b).ok()).unwrap_or_default();
        let peer = node_key.public().to_peer_id().to_string();
        let cert = roster.get(&peer).cloned().ok_or_else(|| anyhow!("the roster has no certificate for this node"))?;
        Self::build(dir, seed, node_key, cert, roster, keys).map(Some)
    }

    fn build(dir: &Path, seed: Vec<u8>, node_key: Keypair, our_cert: NodeCert, roster: Roster, keys: EpochKeys) -> Result<Self> {
        ensure_dir(dir)?;
        let user = user_from_seed(&seed)?;
        let mut st = SyncState {
            user_id: our_cert.user_id.clone(),
            user_key: user.public(),
            our_node: weirdnet_sync::NodeId::new(our_cert.peer_id.clone()),
            our_cert,
            seed,
            types: vec![SYNC_TYPE.to_string()],
            roster,
            log: weirdnet_sync::OpLog::new(),
            clock: None,
            device: Device::from_node_key(&node_key).ok(),
            keys,
            auto_first_rotation: true,
            records: Default::default(),
        };
        st.publish_device_key(now_ms()).map_err(|e| anyhow!("device key: {e}"))?;
        st.refresh_keys();
        Ok(Self { dir: dir.to_path_buf(), node_key, state: Arc::new(Mutex::new(st)) })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, SyncState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Writes the roster and keys (and, the first time, the seed and node key).
    pub fn save(&self) -> Result<()> {
        ensure_dir(&self.dir)?;
        let (seed, roster, keys) = {
            let st = self.lock();
            (st.seed.clone(), serde_json::to_vec_pretty(&st.roster)?, st.keys.export())
        };
        if !self.dir.join("seed").is_file() {
            write_private(&self.dir.join("seed"), &seed)?;
        }
        if !self.dir.join("node.key").is_file() {
            write_private(&self.dir.join("node.key"), &self.node_key.to_protobuf_encoding().map_err(|e| anyhow!("node key: {e}"))?)?;
        }
        write_private(&self.dir.join("roster.json"), &roster)?;
        write_private(&self.dir.join("keys"), &keys)?;
        Ok(())
    }

    pub fn node_key(&self) -> &Keypair {
        &self.node_key
    }

    pub fn peer_id(&self) -> PeerId {
        self.node_key.public().to_peer_id()
    }

    pub fn state(&self) -> Arc<Mutex<SyncState>> {
        self.state.clone()
    }

    pub fn user_key(&self) -> PublicKey {
        self.lock().user_key.clone()
    }

    pub fn user_id(&self) -> String {
        self.lock().user_id.clone()
    }

    pub fn our_cert(&self) -> NodeCert {
        self.lock().our_cert.clone()
    }

    pub fn roster(&self) -> Roster {
        self.lock().roster.clone()
    }

    /// The roster as it is now, for the agent's `RosterAuth` and `LinkedNodes`.
    pub fn roster_view(&self) -> Arc<dyn Fn() -> Roster + Send + Sync> {
        let st = self.state.clone();
        Arc::new(move || st.lock().unwrap_or_else(|e| e.into_inner()).roster.clone())
    }

    /// The transfer payload for a device linked from here.
    pub fn link_settings(&self) -> String {
        serde_json::to_string(&LinkSettings { roster: self.roster() }).unwrap_or_else(|_| "{}".into())
    }

    /// The seed, for a device link sent from here.
    pub fn seed(&self) -> Result<weirdnet_net::Key32> {
        weirdnet_net::Key32::from_slice(&self.lock().seed).map_err(|e| anyhow!("identity seed: {e}"))
    }

    /// The user linked `cert`'s device from here: it joins the roster (and the key chain once
    /// its device key is known).
    pub fn approve(&self, cert: NodeCert) -> Result<()> {
        self.lock().approve(cert, now_ms()).map_err(|e| anyhow!("approving the device: {e}"))?;
        self.save()
    }

    /// Revokes a device everywhere this roster reaches.
    pub fn revoke(&self, peer: &str) -> Result<NodeCert> {
        let cert = {
            let mut st = self.lock();
            if peer == st.our_cert.peer_id {
                bail!("this is this node's own certificate");
            }
            let user = user_from_seed(&st.seed)?;
            st.revoke(&user, peer, now_ms()).map_err(|e| anyhow!("revoking {peer}: {e}"))?
        };
        self.save()?;
        Ok(cert)
    }

    /// One sync pass against every other active device in the roster (blocking; call from a
    /// blocking task). `addrs` gives the addresses to dial a device at. Returns the devices
    /// reached.
    pub fn tick(&self, node: &SyncNode, addrs: &dyn Fn(&NodeCert) -> Vec<Multiaddr>) -> usize {
        let me = node.peer_id().to_string();
        let peers: Vec<NodeCert> = self.lock().roster.active().filter(|c| c.peer_id != me).cloned().collect();
        let mut reached = 0;
        for cert in peers {
            let Ok(peer) = PeerId::from_str(&cert.peer_id) else { continue };
            let a = addrs(&cert);
            // Pull into a copy: our handler needs the live state to answer the peer, which may
            // be pulling from us at the same moment.
            let mut work = snapshot(&self.lock());
            match weirdnet_sync::pull_from(&mut work, node, peer, &a, SYNC_TYPE) {
                Ok(_) => reached += 1,
                Err(e) => tracing::debug!(peer = %cert.peer_id, "sync pass: {e}"),
            }
            merge_back(&mut self.lock(), &work);
        }
        if let Err(e) = self.save() {
            tracing::warn!("saving the roster: {e}");
        }
        reached
    }
}

fn snapshot(st: &SyncState) -> SyncState {
    SyncState {
        seed: st.seed.clone(),
        user_id: st.user_id.clone(),
        user_key: st.user_key.clone(),
        our_cert: st.our_cert.clone(),
        our_node: st.our_node.clone(),
        types: st.types.clone(),
        roster: st.roster.clone(),
        log: st.log.clone(),
        clock: st.clock.clone(),
        device: st.device.clone(),
        keys: st.keys.clone(),
        auto_first_rotation: st.auto_first_rotation,
        records: st.records.clone(),
    }
}

/// Roster changes (a revocation is never undone) and the keys a rotation brought.
fn merge_back(st: &mut SyncState, work: &SyncState) {
    let key = st.user_key.clone();
    st.roster.merge(&work.roster, &key, &WEIRDNET);
    st.take_keys_from(work);
    st.refresh_keys();
}

#[cfg(test)]
mod tests {
    use super::*;
    use weirdnet_sync::CallerCheck;

    #[test]
    fn create_open_link_and_revoke() {
        let tmp = tempfile::tempdir().unwrap();
        let a_dir = tmp.path().join("a");
        let a = Membership::create(&a_dir, "box-a").unwrap();
        assert!(Membership::create(&a_dir, "again").is_err(), "never overwrites an identity");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(a_dir.join("seed")).unwrap().permissions().mode() & 0o777, 0o600);
        }

        // Link a second device the way device-link does: a cert from the seed holder.
        let b_key = Keypair::generate_ed25519();
        let user = user_from_seed(&a.seed().unwrap().0).unwrap();
        let b_cert = Roster::self_certify(&WEIRDNET, &user, &b_key.public().to_peer_id().to_string(), "box-b", now_ms()).unwrap();
        a.approve(b_cert.clone()).unwrap();
        let (seed, settings) = (a.seed().unwrap(), a.link_settings());
        let b = Membership::adopt(&tmp.path().join("b"), b_key, &seed.0, b_cert.clone(), &settings).unwrap();
        assert_eq!(b.user_id(), a.user_id());
        let a_peer = a.peer_id().to_string();
        let b_peer = b.peer_id().to_string();
        assert_eq!(b.roster().check_caller(&WEIRDNET, &b.user_key(), &a_peer, None), CallerCheck::Linked);
        assert_eq!(a.roster().check_caller(&WEIRDNET, &a.user_key(), &b_peer, Some(&b_cert)), CallerCheck::Linked);

        // A stranger with another identity is not admitted.
        let c = Membership::create(&tmp.path().join("c"), "stranger").unwrap();
        assert_eq!(a.roster().check_caller(&WEIRDNET, &a.user_key(), &c.peer_id().to_string(), Some(&c.our_cert())), CallerCheck::WrongUser);

        // Reopening keeps the roster; revoking sticks.
        let again = Membership::open(&a_dir).unwrap().unwrap();
        assert_eq!(again.peer_id(), a.peer_id());
        again.revoke(&b_peer).unwrap();
        let reopened = Membership::open(&a_dir).unwrap().unwrap();
        assert_eq!(reopened.roster().check_caller(&WEIRDNET, &reopened.user_key(), &b_peer, None), CallerCheck::Revoked);
        assert!(reopened.revoke(&a_peer).is_err());
    }
}
