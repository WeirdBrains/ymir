//! This device's place in the user's Weirdnet: the user identity, its own node key and
//! certificate, and the linked-devices roster it shares with Morph (decision 18).
//!
//! Stored under `<state>/pool/` (0700), secrets in 0600 files: `seed` (the user key seed) on a
//! full device, or `credential.json` (the user's public key only) on a service node linked
//! cert-only (M10), `node.key`, `keys` (epoch keys), and the public `roster.json`. The layout is
//! `weirdnet_sync::Identity`'s. The OS keyring comes later, as for the bridge token.
//!
//! A cert-only (service) node holds a certificate and a user-signed service grant with its
//! roles, never the seed: it verifies, serves and syncs, but cannot link, approve or revoke
//! devices. An inference node's grant lists the sync types it holds keys for; a model node's
//! is roster-only (no data keys).
//!
//! The roster is kept current by `weirdnet-sync` (`/weirdnet/sync/2`): [`Membership::tick`] runs
//! a pass against every linked device, as Morph does, so devices linked or revoked elsewhere reach
//! this node. Ymir has no synced record types yet (context sync is M5), so a pass carries the
//! roster and the key chain only.
//!
//! The pull itself (snapshot, pull, merge back) is `weirdnet_sync::pull_from_shared`.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use weirdnet_net::user::NodeCert;
use weirdnet_net::{Keypair, Multiaddr, Node, PeerId, PublicKey, WEIRDNET};
use weirdnet_sync::{Device, DeviceClass, EpochKeys, LinkBundle, LogEntry, LogPage, RecordOptions, RecordValue, Roster, SyncState};

/// The sync type Ymir announces. It carries no records yet; the roster rides along with every
/// pass.
pub const SYNC_TYPE: &str = "ymir.pool";

/// The sync type of conversations (see [`crate::context`]): sealed under the roster's key, so
/// only the user's own devices can read it.
pub const CONTEXT_TYPE: &str = "ymir.context";

/// The role a service grant gives an inference node (`ymir.inference`).
pub const ROLE_INFERENCE: &str = "ymir.inference";

/// The role a service grant gives a model node (`ymir.model`).
pub const ROLE_MODEL: &str = "ymir.model";
/// The synced training jobs (P3b): unsealed records, which a cert-only model node holds the
/// per-type key for (weirdnet-sync 0.8.3), never the epoch key that opens `ymir.context`.
pub const TRAIN_TYPE: &str = crate::train_job::JOB_KIND;
/// Where a model node keeps its lease voter's terms (`Voter::export`), written before each grant.
const VOTER_FILE: &str = "voter.cbor";

/// `credential.json` on a service node: the user's public key (protobuf, hex), as
/// `weirdnet_sync::Identity` writes it.
const CREDENTIAL: &str = "credential.json";

#[derive(Serialize, Deserialize)]
struct Credential {
    user_key: String,
}

/// What a cert-only link makes of a node: an inference node holds the pool's and the
/// conversations' keys; a model node holds no data keys (roster only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceRole {
    Inference,
    Model,
}

impl ServiceRole {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "inference" => Ok(Self::Inference),
            "model" => Ok(Self::Model),
            other => bail!("unknown service role {other:?} (inference or model)"),
        }
    }

    /// The roles its grant carries.
    pub fn roles(self) -> Vec<String> {
        match self {
            Self::Inference => vec![ROLE_INFERENCE.to_string()],
            Self::Model => vec![ROLE_MODEL.to_string()],
        }
    }

    /// The sync types its grant lets it hold keys for: an inference node the pool and context
    /// records, a model node the pool and the training jobs (never the context). Since
    /// weirdnet-sync 0.8.3 a device is refused (`type-not-granted`) for any type it syncs that
    /// its grant does not list, so the pool must be here for both.
    pub fn types(self) -> Vec<String> {
        match self {
            Self::Inference => vec![SYNC_TYPE.to_string(), CONTEXT_TYPE.to_string()],
            Self::Model => vec![SYNC_TYPE.to_string(), TRAIN_TYPE.to_string()],
        }
    }
}

/// What a cert-only device syncs: the pool and every type its grant holds keys for (the
/// context for an inference node, the training jobs for a model node). Since weirdnet-sync
/// 0.8.3 a peer sends only the types both sides list, so a granted type left off here never
/// arrives.
fn service_types(bundle: &LinkBundle) -> Vec<String> {
    let mut types = vec![SYNC_TYPE.to_string()];
    for t in &bundle.grant.types {
        if !types.contains(t) {
            types.push(t.clone());
        }
    }
    types
}

/// Overwrites `path` with zeros, syncs it and deletes it. Gone is gone only as far as the
/// filesystem allows (copy-on-write filesystems and backups may keep the old blocks).
fn shred(path: &Path) -> Result<()> {
    let len = std::fs::metadata(path).with_context(|| format!("reading {}", path.display()))?.len() as usize;
    {
        let mut f = std::fs::OpenOptions::new().write(true).open(path).with_context(|| format!("opening {}", path.display()))?;
        f.write_all(&vec![0u8; len])?;
        f.sync_all()?;
    }
    std::fs::remove_file(path).with_context(|| format!("deleting {}", path.display()))
}

/// What a link transfer carries besides the identity: the roster, in Morph's format, so a Morph
/// device and a Ymir node can link each other.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LinkSettings {
    #[serde(default)]
    pub roster: Roster,
    /// Where the user's other devices can be dialed (`/ip4/…/tcp/…/p2p/PEER`), from the device
    /// that sent the link: a device on another network (no mDNS, no DHT) finds the nodes this way.
    /// Bragi sends and reads the same field; older senders leave it out.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub peers: Vec<String>,
}

impl LinkSettings {
    /// The `peers` a link transfer's settings carry (none when it has no such field or isn't JSON).
    pub fn peers_in(settings: &str) -> Vec<String> {
        serde_json::from_str::<LinkSettings>(settings).map(|s| s.peers).unwrap_or_default()
    }
}

pub type SyncNode = Node<Vec<u8>, Vec<u8>>;

pub struct Membership {
    dir: PathBuf,
    node_key: Keypair,
    state: Arc<Mutex<SyncState>>,
    /// Whether sync passes also carry conversations (inference and app roles; a model-only
    /// node holds no conversation state).
    context_sync: std::sync::atomic::AtomicBool,
    /// Held across [`Membership::save`]: the sync loop, the control socket and link flows all
    /// save, and two saves at once must not interleave (an older snapshot written last).
    saving: Mutex<()>,
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
    // A temporary name of its own per write: two writers of one file sharing `x.tmp` would
    // rename it away from each other ("writing x: No such file or directory", seen in p2).
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let name = path.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
    let tmp = path.with_file_name(format!(".{name}.{}.{n}.tmp", std::process::id()));
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
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e).with_context(|| format!("writing {}", path.display()));
    }
    Ok(())
}

/// Reloads the op log saved by [`Membership::save`], so our own sequence numbers carry on and
/// conversations survive a restart.
fn load_log(dir: &Path, st: &mut SyncState) {
    let Ok(bytes) = std::fs::read(dir.join("log.json")) else { return };
    let Ok(entries) = serde_json::from_slice::<Vec<LogEntry>>(&bytes) else {
        tracing::warn!("log.json is unreadable; starting the op log empty");
        return;
    };
    let mut by_node: std::collections::BTreeMap<String, Vec<LogEntry>> = Default::default();
    for e in entries {
        by_node.entry(e.node_id.0.clone()).or_default().push(e);
    }
    for (node, ops) in by_node {
        let (from_seq, to_seq) = (ops.iter().map(|e| e.seq).min().unwrap_or(0), ops.iter().map(|e| e.seq).max().unwrap_or(0));
        let node_id = weirdnet_sync::NodeId::new(node);
        st.log.integrate(&LogPage { node_id: node_id.clone(), from_seq, to_seq, ops });
        let c = st.log.contiguous(&node_id);
        st.log.set_cursor(&node_id, c);
    }
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
        (dir.join("seed").is_file() || dir.join(CREDENTIAL).is_file()) && dir.join("node.key").is_file()
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
        let m = Self::build(dir, Some(seed.to_vec()), user.public(), node_key, cert, roster, EpochKeys::default())?;
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
        let m = Self::build(dir, Some(seed.to_vec()), user.public(), node_key, cert, roster, EpochKeys::default())?;
        m.save()?;
        Ok(m)
    }

    /// The identity a cert-only (service) link delivered: the user's public key (protobuf), the
    /// certificate the linking device made for `node_key`, and the sync bundle (a
    /// `weirdnet_sync::LinkBundle` with the service grant, the roster and, for an inference node,
    /// the epoch keys). No seed.
    pub fn adopt_service(dir: &Path, node_key: Keypair, user_key: &[u8], cert: NodeCert, bundle: &[u8]) -> Result<Self> {
        if Self::exists(dir) {
            bail!("{} already holds an identity", dir.display());
        }
        Self::take_service(dir, node_key, user_key, cert, bundle)
    }

    fn take_service(dir: &Path, node_key: Keypair, user_key: &[u8], cert: NodeCert, bundle: &[u8]) -> Result<Self> {
        let user_key = PublicKey::try_decode_protobuf(user_key).map_err(|e| anyhow!("user key: {e}"))?;
        if cert.user_id != weirdnet_net::user::user_id(&user_key) {
            bail!("the link's certificate is not for this identity");
        }
        let bundle = LinkBundle::from_bytes(bundle).map_err(|e| anyhow!("the link's sync bundle: {e}"))?;
        let st = SyncState::new_service(user_key, cert, &node_key, service_types(&bundle), &bundle)
            .map_err(|e| anyhow!("the cert-only link did not check out: {e}"))?;
        let m = Self::wrap(dir, node_key, st)?;
        m.save()?;
        Ok(m)
    }

    /// Moves a node that holds the seed to cert-only (M10's migration): `user_key`, `cert` and
    /// `bundle` come from a cert-only link a seed holder sent to this node's own key. The new
    /// state keeps this node's op log; then `seed` is overwritten and deleted, and
    /// `credential.json` takes its place. Refuses a link for another identity or another key.
    pub fn convert_to_service(dir: &Path, user_key: &[u8], cert: NodeCert, bundle: &[u8]) -> Result<Self> {
        let old = Self::open(dir)?.ok_or_else(|| anyhow!("{} holds no identity", dir.display()))?;
        if old.is_service() {
            bail!("this node is already cert-only");
        }
        if cert.peer_id != old.peer_id().to_string() {
            bail!("the link was made for another node key");
        }
        if cert.user_id != old.user_id() {
            bail!("the link is for another identity ({}), not this node's ({})", cert.user_id, old.user_id());
        }
        let user_key = PublicKey::try_decode_protobuf(user_key).map_err(|e| anyhow!("user key: {e}"))?;
        let bundle = LinkBundle::from_bytes(bundle).map_err(|e| anyhow!("the link's sync bundle: {e}"))?;
        let mut st = SyncState::new_service(user_key, cert, old.node_key(), service_types(&bundle), &bundle)
            .map_err(|e| anyhow!("the cert-only link did not check out: {e}"))?;
        {
            let o = old.lock();
            st.roster.merge(&o.roster, &o.user_key, &WEIRDNET);
            st.log = o.log.clone();
            st.clock = o.clock.clone();
            // Older epoch keys this node already holds stay readable for records sealed before.
            if !bundle.grant.roster_only() {
                st.keys.merge(&o.keys);
            }
        }
        let node_key = old.node_key.clone();
        drop(old);
        // credential.json first, so the directory always holds an identity.
        let m = Self::wrap(dir, node_key, st)?;
        m.save()?;
        shred(&dir.join("seed"))?;
        Ok(m)
    }

    fn wrap(dir: &Path, node_key: Keypair, mut st: SyncState) -> Result<Self> {
        ensure_dir(dir)?;
        st.configure(CONTEXT_TYPE, RecordOptions { sealed: true, ..Default::default() }).map_err(|e| anyhow!("context records: {e}"))?;
        Ok(Self {
            dir: dir.to_path_buf(),
            node_key,
            state: Arc::new(Mutex::new(st)),
            context_sync: std::sync::atomic::AtomicBool::new(false),
            saving: Mutex::new(()),
        })
    }

    /// The identity stored in `dir`, if any.
    pub fn open(dir: &Path) -> Result<Option<Self>> {
        if !Self::exists(dir) {
            return Ok(None);
        }
        let seed = match std::fs::read(dir.join("seed")) {
            Ok(s) => Some(s),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e).context("reading the identity seed"),
        };
        let user_key = match &seed {
            Some(s) => user_from_seed(s)?.public(),
            None => {
                let c: Credential = serde_json::from_slice(&std::fs::read(dir.join(CREDENTIAL)).context("reading credential.json")?)
                    .context("reading credential.json")?;
                let bytes = hex::decode(&c.user_key).context("credential.json")?;
                PublicKey::try_decode_protobuf(&bytes).map_err(|e| anyhow!("credential.json: {e}"))?
            }
        };
        let node_key = Keypair::from_protobuf_encoding(&std::fs::read(dir.join("node.key")).context("reading the node key")?)
            .map_err(|e| anyhow!("node key: {e}"))?;
        let roster: Roster = match std::fs::read(dir.join("roster.json")) {
            Ok(b) => serde_json::from_slice(&b).context("reading roster.json")?,
            Err(_) => Roster::new(),
        };
        let keys = std::fs::read(dir.join("keys")).ok().and_then(|b| EpochKeys::import(&b).ok()).unwrap_or_default();
        let peer = node_key.public().to_peer_id().to_string();
        let cert = roster.get(&peer).cloned().ok_or_else(|| anyhow!("the roster has no certificate for this node"))?;
        Self::build(dir, seed, user_key, node_key, cert, roster, keys).map(Some)
    }

    fn build(
        dir: &Path,
        seed: Option<Vec<u8>>,
        user_key: PublicKey,
        node_key: Keypair,
        our_cert: NodeCert,
        roster: Roster,
        keys: EpochKeys,
    ) -> Result<Self> {
        ensure_dir(dir)?;
        let service = seed.is_none();
        let mut st = SyncState {
            user_id: our_cert.user_id.clone(),
            user_key,
            our_node: weirdnet_sync::NodeId::new(our_cert.peer_id.clone()),
            our_cert,
            seed,
            // The pool, and the training jobs (P3b), served to every device granted them.
            types: vec![SYNC_TYPE.to_string(), TRAIN_TYPE.to_string()],
            roster,
            log: weirdnet_sync::OpLog::new(),
            clock: None,
            device: Device::from_node_key(&node_key).ok(),
            keys,
            auto_first_rotation: !service,
            records: Default::default(),
            // A lease voter is set on model nodes for the training job board (`train_board`).
            voter: None,
        };
        st.configure(CONTEXT_TYPE, RecordOptions { sealed: true, ..Default::default() }).map_err(|e| anyhow!("context records: {e}"))?;
        load_log(dir, &mut st);
        st.publish_device_key(now_ms()).map_err(|e| anyhow!("device key: {e}"))?;
        st.refresh_keys();
        Ok(Self {
            dir: dir.to_path_buf(),
            node_key,
            state: Arc::new(Mutex::new(st)),
            context_sync: std::sync::atomic::AtomicBool::new(false),
            saving: Mutex::new(()),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, SyncState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Writes the roster and keys (and, the first time, the seed or credential and the node key).
    pub fn save(&self) -> Result<()> {
        let _saving = self.saving.lock().unwrap_or_else(|e| e.into_inner());
        ensure_dir(&self.dir)?;
        let (seed, user_key, roster, keys, log) = {
            let st = self.lock();
            let entries: Vec<&LogEntry> = st.log.iter().collect();
            (st.seed.clone(), st.user_key.encode_protobuf(), serde_json::to_vec_pretty(&st.roster)?, st.keys.export(), serde_json::to_vec(&entries)?)
        };
        match &seed {
            Some(seed) if !self.dir.join("seed").is_file() => write_private(&self.dir.join("seed"), seed)?,
            Some(_) => {}
            None if !self.dir.join(CREDENTIAL).is_file() => {
                let c = serde_json::to_vec_pretty(&Credential { user_key: hex::encode(user_key) })?;
                write_private(&self.dir.join(CREDENTIAL), &c)?;
            }
            None => {}
        }
        if !self.dir.join("node.key").is_file() {
            write_private(&self.dir.join("node.key"), &self.node_key.to_protobuf_encoding().map_err(|e| anyhow!("node key: {e}"))?)?;
        }
        write_private(&self.dir.join("roster.json"), &roster)?;
        write_private(&self.dir.join("keys"), &keys)?;
        write_private(&self.dir.join("log.json"), &log)?;
        Ok(())
    }

    /// Whether sync passes carry conversations too.
    pub fn set_context_sync(&self, on: bool) {
        self.context_sync.store(on, std::sync::atomic::Ordering::SeqCst);
    }

    /// Writes one record of `typ` (sealed for [`CONTEXT_TYPE`]) and saves.
    pub fn write_record(&self, typ: &str, record_id: &str, value: &[u8]) -> Result<()> {
        {
            let mut st = self.lock();
            let mut r = st.write(now_ms() as u64, typ, record_id, value);
            // Sealing needs an epoch key. A lone device starts the chain itself; with several,
            // the first rotation comes from a sync pass once every device has published its key
            // (`weirdnet-sync` decides who starts it), so the write waits for that.
            if matches!(&r, Err(weirdnet_sync::SyncError::Refused(m)) if m == "no-epoch-key") && st.roster.active().count() == 1 {
                st.rotate(now_ms()).map_err(|e| anyhow!("starting the key chain: {e}"))?;
                r = st.write(now_ms() as u64, typ, record_id, value);
            }
            r.map_err(|e| anyhow!("writing {typ}/{record_id}: {e}"))?;
        }
        self.save()
    }

    /// The latest value of every record of `typ` this node can read, by record id.
    pub fn records(&self, typ: &str) -> Vec<(String, Vec<u8>)> {
        let st = self.lock();
        st.latest_entries(typ)
            .into_iter()
            .filter_map(|e| match st.read(e) {
                Ok(RecordValue::Ready(v)) => Some((e.record_id.clone(), v.to_vec())),
                _ => None,
            })
            .collect()
    }

    /// One pull of conversations from `peer` now (blocking), for a request that names a
    /// conversation this node doesn't hold yet. True when the pass went through.
    pub fn pull_context(&self, node: &SyncNode, peer: &str, addrs: &[Multiaddr]) -> bool {
        let Ok(peer) = PeerId::from_str(peer) else { return false };
        let out = weirdnet_sync::pull_from_shared(&self.state, node, peer, addrs, CONTEXT_TYPE);
        if out.error.is_none() {
            let _ = self.save();
        }
        out.error.is_none()
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
        self.link_settings_with(Vec::new())
    }

    /// [`link_settings`](Self::link_settings) with `peers`: where the new device can dial the
    /// user's other devices.
    pub fn link_settings_with(&self, peers: Vec<String>) -> String {
        serde_json::to_string(&LinkSettings { roster: self.roster(), peers }).unwrap_or_else(|_| "{}".into())
    }

    /// The seed, for a device link sent from here. A cert-only node has none.
    pub fn seed(&self) -> Result<weirdnet_net::Key32> {
        let st = self.lock();
        let seed = st.seed.as_ref().ok_or_else(|| anyhow!("this node is linked cert-only and holds no seed; link devices from one that does"))?;
        weirdnet_net::Key32::from_slice(seed).map_err(|e| anyhow!("identity seed: {e}"))
    }

    /// Whether this node is linked cert-only (no seed).
    pub fn is_service(&self) -> bool {
        self.lock().is_service()
    }

    /// The user keypair, for signing a device's certificate from here (needs the seed).
    pub fn user_keypair(&self) -> Result<Keypair> {
        let st = self.lock();
        let seed = st.seed.as_ref().ok_or_else(|| anyhow!("this node is linked cert-only and holds no seed; link devices from one that does"))?;
        user_from_seed(seed)
    }

    /// The sync bundle for a cert-only link to `peer` as `role` (signs its service grant,
    /// which joins this roster too, and rotates first if an inference node needs keys and
    /// there is no key chain yet). Needs the seed. Saves.
    pub fn service_bundle(&self, peer: &str, role: ServiceRole) -> Result<zeroize::Zeroizing<Vec<u8>>> {
        let b = {
            let mut st = self.lock();
            if st.is_service() {
                bail!("this node is linked cert-only and holds no seed; link devices from one that does");
            }
            st.service_bundle(peer, &role.roles(), &role.types(), now_ms()).map_err(|e| anyhow!("the service grant: {e}"))?
        };
        self.save()?;
        b.to_bytes().map_err(|e| anyhow!("the sync bundle: {e}"))
    }

    /// Votes on the training job board's claims (`ymir.train/` leases) when this device's grant
    /// has the model role and the training jobs: limits from Ymir's job policy, terms kept in
    /// `voter.cbor` and written before each grant (so a restart never hands out a term twice).
    /// Returns whether it votes.
    pub fn enable_train_voter(&self) -> Result<bool> {
        self.enable_train_voter_with(&crate::train_job::JobPolicy::default())
    }

    /// [`Self::enable_train_voter`] with the claim and takeover limits of `p`. After a start
    /// the voter refuses for as long as its longest grant (the claim time), so a restart never
    /// grants a lease another voter may still count as held.
    pub fn enable_train_voter_with(&self, p: &crate::train_job::JobPolicy) -> Result<bool> {
        use weirdnet_sync::{Voter, Voters};
        let mut st = self.lock();
        let me = st.our_cert.peer_id.clone();
        let granted = st.roster.grant_of(&me).map(|g| g.roles.iter().any(|r| r == ROLE_MODEL) && g.types.iter().any(|t| t == TRAIN_TYPE));
        if !granted.unwrap_or(false) {
            st.set_voter(None);
            return Ok(false);
        }
        let candidates = Voters::Role(crate::train_job::JOB_ROLE.to_string());
        let path = self.dir.join(VOTER_FILE);
        let base = match std::fs::read(&path) {
            Ok(b) => Voter::import(&b, candidates).map_err(|e| anyhow!("{}: {e}", path.display()))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Voter::new(candidates),
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        };
        let (claim, takeover) = (std::time::Duration::from_millis(p.claim_ms), std::time::Duration::from_millis(p.takeover_after_ms));
        let voter = base
            .with_max_duration(claim)
            .with_limits(format!("{TRAIN_TYPE}/"), claim, takeover)
            .with_persist(move |b| write_private(&path, b).map_err(std::io::Error::other));
        st.set_voter(Some(voter));
        Ok(true)
    }

    /// Whether this device stands for the placement scheduler's lease
    /// ([`crate::placement::SCHEDULER_LEASE`]): its user-signed grant has the inference role (a
    /// cert-only inference node). A device holding the seed normally has no such grant and does
    /// not stand.
    pub fn scheduler_candidate(&self) -> bool {
        let st = self.lock();
        st.roster.has_role(&st.our_cert.peer_id, ROLE_INFERENCE)
    }

    /// How many active devices stand for (and vote on) the scheduler lease.
    pub fn scheduler_voters(&self) -> usize {
        weirdnet_sync::Voters::Role(ROLE_INFERENCE.to_string()).resolve(&self.lock().roster).len()
    }

    /// Votes on the scheduler lease when this device is a candidate
    /// ([`Self::scheduler_candidate`]); terms kept in `voter.cbor` as for the training board
    /// (a service grant has one role, so a device never needs both voters). Returns whether it votes.
    pub fn enable_scheduler_voter(&self) -> Result<bool> {
        use weirdnet_sync::{Voter, Voters};
        if !self.scheduler_candidate() {
            return Ok(false);
        }
        let candidates = Voters::Role(ROLE_INFERENCE.to_string());
        let path = self.dir.join(VOTER_FILE);
        let base = match std::fs::read(&path) {
            Ok(b) => Voter::import(&b, candidates).map_err(|e| anyhow!("{}: {e}", path.display()))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Voter::new(candidates),
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        };
        let voter = base.with_persist(move |b| write_private(&path, b).map_err(std::io::Error::other));
        self.lock().set_voter(Some(voter));
        Ok(true)
    }

    /// Whether the training jobs pass with `peer`: both ends hold them (a full device, or a
    /// service device whose grant lists them). A cert-only inference node holds no key for them.
    fn syncs_jobs_with(&self, peer: &str) -> bool {
        let st = self.lock();
        let holds = |p: &str| match st.roster.grant_of(p) {
            Some(g) if g.class == DeviceClass::Service => g.types.iter().any(|t| t == TRAIN_TYPE),
            _ => true,
        };
        let me = st.our_cert.peer_id.clone();
        holds(&me) && st.types.iter().any(|t| t == TRAIN_TYPE) && holds(peer)
    }

    /// `peer`'s class and roles as its user-signed grant states them, if it has one.
    pub fn grant_of(&self, peer: &str) -> Option<(DeviceClass, Vec<String>)> {
        let st = self.lock();
        st.roster.grant_of(peer).map(|g| (g.class, g.roles.clone()))
    }

    /// The user linked `cert`'s device from here: it joins the roster (and the key chain once
    /// its device key is known).
    pub fn approve(&self, cert: NodeCert) -> Result<()> {
        if self.is_service() {
            bail!("this node is linked cert-only and cannot approve devices; link from one that holds the seed");
        }
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
            let seed = st
                .seed
                .as_ref()
                .ok_or_else(|| anyhow!("this node is linked cert-only and cannot revoke devices; revoke from one that holds the seed"))?;
            let user = user_from_seed(seed)?;
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
        let context = self.context_sync.load(std::sync::atomic::Ordering::SeqCst);
        for cert in peers {
            let Ok(peer) = PeerId::from_str(&cert.peer_id) else { continue };
            let a = addrs(&cert);
            // Pulls into a snapshot and merges it back, so our handler can answer a peer pulling
            // from us at the same moment.
            let out = weirdnet_sync::pull_from_shared(&self.state, node, peer, &a, SYNC_TYPE);
            match &out.error {
                None => reached += 1,
                Some(e) => tracing::debug!(peer = %cert.peer_id, applied = out.applied, "sync pass: {e}"),
            }
            if context && out.error.is_none() {
                let out = weirdnet_sync::pull_from_shared(&self.state, node, peer, &a, CONTEXT_TYPE);
                if let Some(e) = &out.error {
                    tracing::debug!(peer = %cert.peer_id, applied = out.applied, "context pass: {e}");
                }
            }
            if out.error.is_none() && self.syncs_jobs_with(&cert.peer_id) {
                let out = weirdnet_sync::pull_from_shared(&self.state, node, peer, &a, TRAIN_TYPE);
                if let Some(e) = &out.error {
                    tracing::debug!(peer = %cert.peer_id, applied = out.applied, "training jobs pass: {e}");
                }
            }
        }
        if let Err(e) = self.save() {
            tracing::warn!("saving the roster: {e}");
        }
        reached
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use weirdnet_sync::CallerCheck;

    #[test]
    fn concurrent_saves_do_not_trip_over_each_other() {
        let tmp = tempfile::tempdir().unwrap();
        let m = Arc::new(Membership::create(&tmp.path().join("a"), "box-a").unwrap());
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let m = m.clone();
                std::thread::spawn(move || (0..40).try_for_each(|_| m.save()))
            })
            .collect();
        for t in threads {
            t.join().unwrap().unwrap();
        }
        let left: Vec<_> = std::fs::read_dir(tmp.path().join("a"))
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(left.is_empty(), "no temporary files are left behind");
        assert!(Membership::open(&tmp.path().join("a")).unwrap().is_some());
    }

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
