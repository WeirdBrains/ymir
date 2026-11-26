//! P3b blob transfer: a training job's dataset and checkpoints between the owner's devices over
//! Weirdnet (`/ymir/blobs/1`), so a job resumes on another node without copying files by hand.
//!
//! A bundle is content-addressed: its root is the sha256 of its manifest (`checkpoint.json` or
//! the dataset index), which lists every file with its sha256 and size. The asker names a root;
//! the holder answers with the manifest, then each listed file in order, in chunks. The asker
//! trusts nothing the holder says about the files: it checks the manifest against the root,
//! takes the file list from the manifest, and checks every file's size and hash while it writes
//! it to a staging folder. Only a complete, checked bundle enters its [`LocalBlobs`].
//!
//! Who may fetch: the owner's linked devices (the roster's certificates), like
//! `/ymir/train-status/1`. On Weirdnet `ymird` holds the node and fetches into the node's own
//! store; `ymirctl` asks it to (see [`crate::train_jobs_ctl`]).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_bytes::ByteBuf;
use sha2::{Digest, Sha256};
use tokio::io::AsyncReadExt;
use weirdnet_agent::remote::{Authorize, Caller, Channel, Dialer, RemoteError};
use weirdnet_agent::serve::{channel, stream_limits};
use weirdnet_net::node::{Message, Node};

use crate::train::{self, CheckpointFile};
use crate::train_job::{BlobStore, LocalBlobs};

pub const BLOBS_PROTOCOL: &str = "/ymir/blobs/1";

/// File bytes per message (under the stream's 64 KiB frame).
const CHUNK: usize = 48 * 1024;

/// The largest manifest a holder may send.
const MAX_MANIFEST: usize = 4 << 20;

/// The request (CBOR).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BlobAsk {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cert: Option<ByteBuf>,
    pub root: String,
}

/// The holder's messages (CBOR): `Bundle`, then each file's bytes as `Chunk`s in the order the
/// manifest lists them, then `Done`. Or a single `Missing` or `Refused`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "m", rename_all = "snake_case")]
pub enum BlobMsg {
    Bundle { manifest_name: String, manifest: ByteBuf },
    Chunk { data: ByteBuf },
    Done,
    Missing,
    Refused { message: String },
}

/// The files a manifest lists (both `checkpoint.json` and the dataset index have `files`).
#[derive(Deserialize)]
struct Listed {
    files: Vec<CheckpointFile>,
}

fn listed(manifest_name: &str, manifest: &[u8]) -> Result<Vec<CheckpointFile>> {
    let l: Listed = serde_json::from_slice(manifest).context("the bundle's manifest lists no files")?;
    for f in &l.files {
        if !train::safe_rel(&f.path) || f.path == manifest_name {
            bail!("`{}` is not a file path inside the bundle", f.path);
        }
    }
    Ok(l.files)
}

/// Serves the node's [`LocalBlobs`] to the owner's linked devices.
pub struct BlobsService {
    blobs: Arc<LocalBlobs>,
    auth: Arc<dyn Authorize>,
}

impl BlobsService {
    pub fn new(blobs: Arc<LocalBlobs>, auth: Arc<dyn Authorize>) -> Self {
        Self { blobs, auth }
    }

    pub async fn serve(&self, mut ch: Channel, caller: Caller) -> Result<(), RemoteError> {
        let Some(ask) = ch.recv_cbor::<BlobAsk>().await? else { return Ok(()) };
        let caller = match (caller.cert_bytes.is_some(), ask.cert) {
            (false, Some(cert)) => caller.with_cert(cert.into_vec()),
            _ => caller,
        };
        if let Err(e) = self.auth.authorize(&caller) {
            return ch
                .send_cbor(&BlobMsg::Refused { message: format!("refused: only the owner's linked devices fetch training files here ({e})") })
                .await;
        }
        let Some((dir, manifest_name)) = self.blobs.bundle(&ask.root) else { return ch.send_cbor(&BlobMsg::Missing).await };
        let manifest = match tokio::fs::read(dir.join(&manifest_name)).await {
            Ok(m) if train::checkpoint_root(&m) == ask.root => m,
            _ => return ch.send_cbor(&BlobMsg::Missing).await,
        };
        let files = match listed(&manifest_name, &manifest) {
            Ok(f) => f,
            Err(_) => return ch.send_cbor(&BlobMsg::Missing).await,
        };
        ch.send_cbor(&BlobMsg::Bundle { manifest_name, manifest: ByteBuf::from(manifest) }).await?;
        let mut buf = vec![0u8; CHUNK];
        for f in &files {
            let mut file = tokio::fs::File::open(dir.join(&f.path)).await.map_err(|e| RemoteError::Failed { message: format!("{}: {e}", f.path) })?;
            let mut left = f.bytes;
            while left > 0 {
                let n =
                    file.read(&mut buf[..CHUNK.min(left as usize)]).await.map_err(|e| RemoteError::Failed { message: format!("{}: {e}", f.path) })?;
                if n == 0 {
                    return Err(RemoteError::Failed { message: format!("{} is shorter than its manifest says", f.path) });
                }
                left -= n as u64;
                ch.send_cbor(&BlobMsg::Chunk { data: ByteBuf::from(buf[..n].to_vec()) }).await?;
            }
        }
        ch.send_cbor(&BlobMsg::Done).await
    }
}

/// Accepts `/ymir/blobs/1` streams until dropped.
pub struct BlobsServer {
    task: tokio::task::JoinHandle<()>,
}

impl BlobsServer {
    pub fn start<Q: Message, R: Message>(node: &Arc<Node<Q, R>>, service: Arc<BlobsService>) -> Result<Self, RemoteError> {
        let mut limits = stream_limits();
        limits.max_streams = 8;
        let mut incoming = node.accept_streams(BLOBS_PROTOCOL, limits).map_err(|e| RemoteError::Failed { message: e.to_string() })?;
        let task = tokio::spawn(async move {
            while let Some(stream) = futures::StreamExt::next(&mut incoming).await {
                let caller = Caller::new(stream.peer().to_string());
                let svc = service.clone();
                tokio::spawn(async move {
                    let _ = svc.serve(channel(stream), caller).await;
                });
            }
        });
        Ok(Self { task })
    }
}

impl Drop for BlobsServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// Fetches bundles from the owner's other devices into this node's [`LocalBlobs`].
pub struct BlobsClient {
    dialer: Arc<dyn Dialer>,
    cert: Option<Vec<u8>>,
    blobs: Arc<LocalBlobs>,
    me: String,
}

impl BlobsClient {
    pub fn new(dialer: Arc<dyn Dialer>, cert: Option<Vec<u8>>, blobs: Arc<LocalBlobs>, me: String) -> Self {
        Self { dialer, cert, blobs, me }
    }

    /// Makes bundle `root` local, from the first of `holders` that has it (this node first).
    /// Returns the peer it came from (`None`: it was here already).
    pub async fn ensure(&self, root: &str, holders: &[String]) -> Result<Option<String>> {
        if self.blobs.has(root) {
            return Ok(None);
        }
        let mut tried = Vec::new();
        for peer in holders.iter().filter(|p| **p != self.me) {
            if tried.contains(peer) {
                continue;
            }
            tried.push(peer.clone());
            match self.fetch_from(peer, root).await {
                Ok(()) => return Ok(Some(peer.clone())),
                Err(e) => tracing::debug!(%peer, root, "blob fetch: {e:#}"),
            }
        }
        if tried.is_empty() {
            bail!("bundle {root} is not on this node and no other device is known to hold it");
        }
        bail!("bundle {root}: none of its holders ({}) sent it", tried.join(", "))
    }

    async fn fetch_from(&self, peer: &str, root: &str) -> Result<()> {
        let mut ch = self.dialer.open(peer, BLOBS_PROTOCOL).await.map_err(|e| anyhow!("{e}"))?;
        ch.send_cbor(&BlobAsk { cert: self.cert.clone().map(ByteBuf::from), root: root.to_string() }).await.map_err(|e| anyhow!("{e}"))?;
        let (manifest_name, manifest) = match ch.recv_cbor::<BlobMsg>().await.map_err(|e| anyhow!("{e}"))? {
            Some(BlobMsg::Bundle { manifest_name, manifest }) => (manifest_name, manifest.into_vec()),
            Some(BlobMsg::Missing) => bail!("it doesn't hold it"),
            Some(BlobMsg::Refused { message }) => bail!("{message}"),
            Some(other) => bail!("unexpected {other:?}"),
            None => bail!("the stream closed"),
        };
        if manifest.len() > MAX_MANIFEST || train::checkpoint_root(&manifest) != root {
            bail!("its manifest doesn't hash to {root}");
        }
        if !train::safe_rel(&manifest_name) || manifest_name.contains('/') {
            bail!("`{manifest_name}` is not a manifest name");
        }
        let files = listed(&manifest_name, &manifest)?;
        let staging = self.blobs.staging(root)?;
        let r = receive(&mut ch, &staging, &manifest_name, &manifest, &files).await;
        let r = r.and_then(|()| self.blobs.put_checked(&staging, &manifest_name, &files, root));
        let _ = std::fs::remove_dir_all(&staging);
        r
    }
}

/// Writes the files into `staging`, each checked against the manifest as it arrives.
async fn receive(ch: &mut Channel, staging: &Path, manifest_name: &str, manifest: &[u8], files: &[CheckpointFile]) -> Result<()> {
    for f in files {
        let path = staging.join(&f.path);
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p)?;
        }
        let mut out = std::fs::File::create(&path).with_context(|| format!("writing {}", path.display()))?;
        let (mut got, mut sha) = (0u64, Sha256::new());
        while got < f.bytes {
            match ch.recv_cbor::<BlobMsg>().await.map_err(|e| anyhow!("{e}"))? {
                Some(BlobMsg::Chunk { data }) => {
                    got += data.len() as u64;
                    if got > f.bytes {
                        bail!("{} is longer than its manifest says", f.path);
                    }
                    sha.update(&data);
                    out.write_all(&data)?;
                }
                Some(other) => bail!("{}: unexpected {other:?} after {got} of {} bytes", f.path, f.bytes),
                None => bail!("{}: the stream closed after {got} of {} bytes", f.path, f.bytes),
            }
        }
        out.sync_all()?;
        if hex::encode(sha.finalize()) != f.sha256 {
            bail!("{} doesn't match its hash", f.path);
        }
    }
    match ch.recv_cbor::<BlobMsg>().await.map_err(|e| anyhow!("{e}"))? {
        Some(BlobMsg::Done) => {}
        other => bail!("expected the end of the bundle, got {other:?}"),
    }
    std::fs::write(staging.join(manifest_name), manifest)?;
    Ok(())
}

/// Where a fetch stages its files (inside the store, so the move into it is a rename).
pub fn staging_dir(store: &Path, root: &str) -> PathBuf {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    store.join(format!(".fetch-{root}-{}-{n}", std::process::id()))
}

#[async_trait::async_trait]
impl crate::train_job::BlobFetch for BlobsClient {
    async fn ensure(&self, root: &str, holders: &[String]) -> Result<Option<String>> {
        BlobsClient::ensure(self, root, holders).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use weirdnet_agent::remote::MemoryDialer;

    struct Allow(bool);
    impl Authorize for Allow {
        fn authorize(&self, _: &Caller) -> Result<(), RemoteError> {
            if self.0 { Ok(()) } else { Err(RemoteError::Failed { message: "not linked".into() }) }
        }
    }

    /// A bundle of two files (one in a subfolder, one over a chunk) in a fresh store.
    fn bundle(dir: &Path) -> (Arc<LocalBlobs>, String, Vec<u8>) {
        let src = dir.join("src");
        std::fs::create_dir_all(src.join("sub")).unwrap();
        let big: Vec<u8> = (0..CHUNK * 2 + 17).map(|i| (i % 251) as u8).collect();
        std::fs::write(src.join("model.gguf"), &big).unwrap();
        std::fs::write(src.join("sub/opt.bin"), b"opt").unwrap();
        let mut files = Vec::new();
        for p in ["model.gguf", "sub/opt.bin"] {
            let (sha256, bytes) = train::hash_file(&src.join(p)).unwrap();
            files.push(CheckpointFile { path: p.into(), sha256, bytes });
        }
        std::fs::write(src.join("m.json"), serde_json::to_vec(&serde_json::json!({ "files": files })).unwrap()).unwrap();
        let store = Arc::new(LocalBlobs::open(dir.join("store")).unwrap());
        let root = store.put(&src, "m.json", &files).unwrap();
        (store, root, big)
    }

    fn serving(d: &MemoryDialer, node: &str, svc: Arc<BlobsService>) {
        d.route(node, move |ch, _| {
            let svc = svc.clone();
            tokio::spawn(async move {
                let _ = svc.serve(ch, Caller::new("asker")).await;
            });
        });
    }

    /// A holder that sends `root`'s real manifest, then `mess` instead of the files.
    fn lying(d: &MemoryDialer, node: &str, manifest: Vec<u8>, mess: Vec<BlobMsg>) {
        d.route(node, move |mut ch, _| {
            let (manifest, mess) = (manifest.clone(), mess.clone());
            tokio::spawn(async move {
                let _ = ch.recv_cbor::<BlobAsk>().await;
                let _ = ch.send_cbor(&BlobMsg::Bundle { manifest_name: "m.json".into(), manifest: ByteBuf::from(manifest) }).await;
                for m in mess {
                    let _ = ch.send_cbor(&m).await;
                }
            });
        });
    }

    fn no_staging_left(dir: &Path) -> bool {
        !std::fs::read_dir(dir).unwrap().flatten().any(|e| e.file_name().to_string_lossy().starts_with(".fetch-"))
    }

    #[tokio::test]
    async fn a_bundle_is_fetched_from_the_first_holder_that_sends_it_whole() {
        let t = tempfile::tempdir().unwrap();
        let (theirs, root, big) = bundle(&t.path().join("a"));
        let manifest = std::fs::read(t.path().join("a/src/m.json")).unwrap();
        let d = MemoryDialer::new();
        serving(&d, "good", Arc::new(BlobsService::new(theirs, Arc::new(Allow(true)))));
        // "liar" sends a file that doesn't match its hash; "gone" is unreachable.
        let bad = vec![BlobMsg::Chunk { data: ByteBuf::from(vec![0u8; big.len()]) }];
        lying(&d, "liar", manifest, bad);
        let mine = Arc::new(LocalBlobs::open(t.path().join("b")).unwrap());
        let c = BlobsClient::new(Arc::new(d), None, mine.clone(), "me".into());
        let holders = ["me", "gone", "liar", "good"].map(String::from);
        assert_eq!(c.ensure(&root, &holders).await.unwrap().as_deref(), Some("good"));
        assert!(mine.has(&root));
        let out = t.path().join("out");
        mine.fetch(&root, &out).unwrap();
        assert_eq!(std::fs::read(out.join("model.gguf")).unwrap(), big);
        assert_eq!(std::fs::read(out.join("sub/opt.bin")).unwrap(), b"opt");
        assert!(no_staging_left(&t.path().join("b")));
        // Here already: no fetch.
        assert_eq!(c.ensure(&root, &[]).await.unwrap(), None);
    }

    #[tokio::test]
    async fn nothing_a_holder_gets_wrong_enters_the_store() {
        let t = tempfile::tempdir().unwrap();
        let (_, root, big) = bundle(&t.path().join("a"));
        let manifest = std::fs::read(t.path().join("a/src/m.json")).unwrap();
        let n = big.len();
        let cases: Vec<(&str, Vec<u8>, Vec<BlobMsg>, &str)> = vec![
            (
                "flipped",
                manifest.clone(),
                vec![BlobMsg::Chunk {
                    data: ByteBuf::from({
                        let mut b = big.clone();
                        b[5] ^= 1;
                        b
                    }),
                }],
                "hash",
            ),
            ("long", manifest.clone(), vec![BlobMsg::Chunk { data: ByteBuf::from(vec![0u8; n + 1]) }], "longer"),
            ("short", manifest.clone(), vec![BlobMsg::Chunk { data: ByteBuf::from(big[..10].to_vec()) }, BlobMsg::Done], "unexpected"),
            ("other", b"{\"files\":[]}".to_vec(), vec![BlobMsg::Done], "doesn't hash"),
        ];
        for (name, m, mess, want) in cases {
            let d = MemoryDialer::new();
            lying(&d, name, m, mess);
            let mine = Arc::new(LocalBlobs::open(t.path().join(name)).unwrap());
            let c = BlobsClient::new(Arc::new(d), None, mine.clone(), "me".into());
            let e = c.fetch_from(name, &root).await.unwrap_err();
            assert!(format!("{e:#}").contains(want), "{name}: {e:#}");
            assert!(!mine.has(&root), "{name}");
            assert!(no_staging_left(&t.path().join(name)), "{name}");
        }
    }

    #[tokio::test]
    async fn a_holder_refuses_devices_it_does_not_know_and_paths_out_of_the_bundle() {
        let t = tempfile::tempdir().unwrap();
        let (theirs, root, _) = bundle(&t.path().join("a"));
        let d = MemoryDialer::new();
        serving(&d, "strict", Arc::new(BlobsService::new(theirs.clone(), Arc::new(Allow(false)))));
        serving(&d, "open", Arc::new(BlobsService::new(theirs, Arc::new(Allow(true)))));
        let c = BlobsClient::new(Arc::new(d), None, Arc::new(LocalBlobs::open(t.path().join("b")).unwrap()), "me".into());
        let e = c.fetch_from("strict", &root).await.unwrap_err();
        assert!(format!("{e:#}").contains("refused"), "{e:#}");
        let e = c.fetch_from("open", &"00".repeat(32)).await.unwrap_err();
        assert!(format!("{e:#}").contains("doesn't hold"), "{e:#}");
        assert!(c.fetch_from("open", "../../etc").await.is_err());
        // A manifest that lists a path out of the bundle is refused before anything is written.
        let e = listed("m.json", br#"{"files":[{"path":"../x","sha256":"","bytes":0}]}"#).unwrap_err();
        assert!(format!("{e:#}").contains("not a file path"), "{e:#}");
    }
}
