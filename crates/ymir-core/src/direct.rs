//! Direct, pinned access to the user's inference nodes over Weirdnet, for `ymirctl bench`.
//!
//! The loopback bridge hands a request to the pool, which picks the node. A benchmark has to ask a
//! chosen node, so [`Direct`] starts a Weirdnet node of its own with this device's identity and
//! certificate (the same files `ymird` opens, read only; no roster or key is written) and talks the
//! protocols Ymir already uses between its roles:
//!
//! - a **model node** (the model role, `/weirdnet-backends/node/1` records and raw inference, the way
//!   an inference node reaches it, [`RemoteModelNode`]) is the hardware itself, so this is the pin that
//!   makes the same model comparable across machines;
//! - an **inference node** that has no model role (`/weirdnet-agent/catalog/1` and `infer/2`) is asked
//!   through a [`Pool`] whose node list is that one peer.
//!
//! A device with a model role is always measured through it. Nothing here reimplements routing,
//! failover or the wire formats.
//!
//! The probe node listens on ephemeral ports, with no LAN discovery, DHT or relays, so it neither
//! collides with a running `ymird` of the same identity nor announces itself. It reaches other nodes
//! through the `[network] peers` of `node.toml` and any extra hints it is given.
//!
//! "Node" here is an inference node. An inference node may place a model on a model node of its
//! own, so every answer also reports `served_by`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use futures::StreamExt;
use weirdnet_agent::pool::{EventSink, NetDialer, Pool, PoolEvent, PoolOptions, encode_cert};
use weirdnet_agent::remote::catalog::{Catalog, ModelState, Purpose};
use weirdnet_agent::remote::infer::v2::{EmbedRequest, Request, Sampling};
use weirdnet_agent::{CancelToken, Message};
use weirdnet_backends::{Chunk, ModelState as BackendState};
use weirdnet_net::{Multiaddr, PeerHint, PeerId};

use crate::client::TimedChat;
use crate::daemon::Network;
use crate::model_node::ModelNode;
use crate::remote::{RemoteModelNode, RemoteOptions};
use crate::{NodeConfig, Paths};

/// A linked inference-capable device and what it serves.
#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub peer: String,
    /// `model` (measured on the hardware) or `inference`.
    pub role: &'static str,
    /// The label the device was linked under.
    pub label: String,
    /// The node's models, or why it could not be asked.
    pub models: Result<Vec<ModelInfo>, String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelInfo {
    pub id: String,
    pub embedding: bool,
    /// `warm`, `loadable`, `loading` or `busy`.
    pub state: &'static str,
    pub replicas: u32,
    pub context: Option<u32>,
}

/// What a timed request asks for.
#[derive(Debug, Clone)]
pub struct AskOptions {
    pub max_tokens: u32,
    pub temperature: f32,
    pub seed: u64,
    /// Wait for a cold model to load instead of taking a stand-in.
    pub wait: bool,
}

pub struct Direct {
    net: Network,
    extra: Vec<(PeerId, Vec<Multiaddr>)>,
    accept: Duration,
    /// Model nodes connected so far (a connection follows the node's records until dropped).
    model_nodes: Mutex<HashMap<String, Arc<RemoteModelNode>>>,
}

/// How long a cold model gets to load on a model node before the benchmark gives up on it.
const LOAD_WAIT: Duration = Duration::from_secs(900);

fn state_name(s: ModelState) -> &'static str {
    match s {
        ModelState::Warm => "warm",
        ModelState::Loadable => "loadable",
        ModelState::Loading => "loading",
        ModelState::Busy => "busy",
        _ => "unknown",
    }
}

/// Records the time of the first streamed token.
struct Clock {
    start: Instant,
    first: Mutex<Option<Duration>>,
}

impl EventSink for Clock {
    fn event(&self, event: PoolEvent) {
        if matches!(event, PoolEvent::Text(ref t) | PoolEvent::Reasoning(ref t) if !t.is_empty()) {
            let mut f = self.first.lock().unwrap_or_else(|e| e.into_inner());
            f.get_or_insert_with(|| self.start.elapsed());
        }
    }
}

impl Direct {
    /// Starts the probe node. `hints` are extra `/ip4/…/tcp/…/p2p/PEER` addresses to dial.
    pub async fn start(cfg: &NodeConfig, paths: &Paths, hints: &[String], accept: Duration) -> Result<Self> {
        let mut cfg = cfg.clone();
        cfg.network.enabled = true;
        cfg.network.listen = Some(vec!["/ip4/0.0.0.0/tcp/0".into(), "/ip4/0.0.0.0/udp/0/quic-v1".into()]);
        cfg.network.mdns = false;
        cfg.network.discovery = false;
        cfg.network.relays = Some(Vec::new());
        let mut extra = Vec::new();
        for h in hints {
            let hint = PeerHint::from_addr(h).map_err(|e| anyhow!("hint `{h}`: {e}"))?;
            let peer: PeerId = hint.peer_id.parse().map_err(|e| anyhow!("hint `{h}`: {e}"))?;
            let addrs = hint.addrs.iter().filter_map(|a| a.parse().ok()).collect();
            extra.push((peer, addrs));
        }
        let net = Network::start(&cfg, paths).await?.context("this device has no Weirdnet identity (run `ymirctl pool link`) or [network] is off")?;
        Ok(Self { net, extra, accept, model_nodes: Mutex::new(HashMap::new()) })
    }

    /// This device's own peer id (never benchmarked).
    pub fn own_peer(&self) -> String {
        self.net.membership.peer_id().to_string()
    }

    /// The linked devices other than this one, sorted by peer id.
    pub fn devices(&self) -> Vec<(String, String)> {
        let own = self.own_peer();
        let mut v: Vec<(String, String)> =
            self.net.membership.roster().active().filter(|c| c.peer_id != own).map(|c| (c.peer_id.clone(), c.label.clone())).collect();
        v.sort();
        v
    }

    fn dialer(&self) -> NetDialer<Vec<u8>, Vec<u8>> {
        let d = NetDialer::new(self.net.node.clone());
        for (peer, addrs) in self.net.hints.iter().chain(self.extra.iter()) {
            d.hint(*peer, addrs.clone());
        }
        d
    }

    /// A pool that can only reach `peer`.
    fn pinned(&self, peer: &str) -> Pool {
        let opts =
            PoolOptions::default().with_cert(Some(encode_cert(&self.net.membership.our_cert()))).with_accept_timeout(self.accept).with_resume(false);
        Pool::new(Arc::new(self.dialer()), vec![peer.to_string()]).with_options(opts)
    }

    fn catalog_models(c: &Catalog) -> Vec<ModelInfo> {
        let mut v: Vec<ModelInfo> = c
            .models
            .iter()
            .map(|m| ModelInfo {
                id: m.id.clone(),
                embedding: m.purpose == Purpose::Embedding,
                state: state_name(m.state),
                replicas: m.replicas,
                context: m.context,
            })
            .collect();
        v.sort_by(|a, b| a.id.cmp(&b.id));
        v
    }

    /// `peer`'s models through its inference role.
    pub async fn models(&self, peer: &str) -> Result<Vec<ModelInfo>> {
        let c = self.pinned(peer).catalog().await.map_err(|e| anyhow!("{e}"))?;
        Ok(Self::catalog_models(&c))
    }

    fn model_node_of(&self, peer: &str) -> Option<Arc<RemoteModelNode>> {
        self.model_nodes.lock().unwrap_or_else(|e| e.into_inner()).get(peer).cloned()
    }

    /// The model node at `peer`, connecting on first use.
    async fn connect_model_node(&self, peer: &str) -> Result<Arc<RemoteModelNode>> {
        if let Some(n) = self.model_node_of(peer) {
            return Ok(n);
        }
        let opts = RemoteOptions { accept_timeout: self.accept, ..Default::default() };
        let n = Arc::new(RemoteModelNode::connect(Arc::new(self.dialer()), peer, &self.net.membership, &opts).await.map_err(|e| anyhow!("{e}"))?);
        self.model_nodes.lock().unwrap_or_else(|e| e.into_inner()).insert(peer.to_string(), n.clone());
        Ok(n)
    }

    fn record_models(node: &RemoteModelNode) -> Vec<ModelInfo> {
        let mut v: Vec<ModelInfo> = node
            .status()
            .into_iter()
            .map(|m| ModelInfo {
                id: m.id,
                embedding: m.embedding,
                state: match m.state {
                    BackendState::Ready if m.in_flight > 0 => "busy",
                    BackendState::Ready => "warm",
                    BackendState::Unloaded => "loadable",
                    BackendState::Loading => "loading",
                    BackendState::Draining => "busy",
                    _ => "failed",
                },
                replicas: u32::from(matches!(m.state, BackendState::Ready)),
                context: m.context,
            })
            .collect();
        v.sort_by(|a, b| a.id.cmp(&b.id));
        v
    }

    /// Every linked device with its models (or the reason it did not answer). A device that has a
    /// model role is measured through it; one that only has an inference role through that.
    pub async fn nodes(&self) -> Vec<NodeInfo> {
        let mut out = Vec::new();
        for (peer, label) in self.devices() {
            let (role, models) = match self.connect_model_node(&peer).await {
                Ok(n) => ("model", Ok(Self::record_models(&n))),
                Err(model_err) => match self.models(&peer).await {
                    Ok(m) => ("inference", Ok(m)),
                    Err(inf_err) => ("inference", Err(format!("{inf_err:#} (model role: {model_err:#})"))),
                },
            };
            out.push(NodeInfo { peer, role, label, models });
        }
        out
    }

    /// Waits for `model` to be loaded on a model node, asking it to load when it is not.
    async fn ensure_loaded(&self, node: &RemoteModelNode, model: &str) -> Result<()> {
        let ready = |n: &RemoteModelNode| n.status().iter().any(|m| m.id == model && matches!(m.state, BackendState::Ready));
        if ready(node) {
            return Ok(());
        }
        node.load(model, false).await.map_err(|e| anyhow!("asking the node to load {model}: {e}"))?;
        let deadline = Instant::now() + LOAD_WAIT;
        while Instant::now() < deadline {
            if ready(node) {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
        bail!("{model} did not finish loading in {} s", LOAD_WAIT.as_secs())
    }

    async fn chat_model_node(&self, peer: &str, model: &str, prompt: &str, o: &AskOptions) -> Result<TimedChat> {
        let node = self.connect_model_node(peer).await?;
        if o.wait {
            self.ensure_loaded(&node, model).await?;
        }
        let mut req = Request::messages(vec![Message::user(prompt)]).with_seed(o.seed).with_sampling(Self::sampling(o)?);
        req.raw = true;
        let start = Instant::now();
        let mut stream = node.chat(model, &req).await.map_err(|e| anyhow!("{e}"))?;
        let mut out = TimedChat { served_by: Some(peer.to_string()), ..Default::default() };
        while let Some(chunk) = stream.next().await {
            match chunk.map_err(|e| anyhow!("{e}"))? {
                Chunk::Text(t) => {
                    if !t.is_empty() && out.ttft.is_none() {
                        out.ttft = Some(start.elapsed());
                    }
                    out.text.push_str(&t);
                }
                Chunk::Reasoning(t) => {
                    if !t.is_empty() && out.ttft.is_none() {
                        out.ttft = Some(start.elapsed());
                    }
                }
                Chunk::ToolCall { .. } => {}
                Chunk::Done { finish_reason, usage } => {
                    out.finish_reason = finish_reason;
                    if let Some(u) = usage {
                        out.prompt_tokens = Some(u.prompt_tokens);
                        out.cached_tokens = Some(u.cached_tokens);
                        out.completion_tokens = Some(u.completion_tokens);
                    }
                }
                _ => {}
            }
        }
        out.total = start.elapsed();
        if out.finish_reason.is_none() {
            bail!("the stream ended early");
        }
        Ok(out)
    }

    fn sampling(o: &AskOptions) -> Result<Sampling> {
        Ok(serde_json::from_value(serde_json::json!({"temperature": o.temperature, "max_tokens": o.max_tokens}))?)
    }

    /// One streamed answer from `peer`, timed.
    pub async fn chat_timed(&self, peer: &str, model: &str, prompt: &str, o: &AskOptions) -> Result<TimedChat> {
        if self.model_node_of(peer).is_some() {
            return self.chat_model_node(peer, model, prompt, o).await;
        }
        let mut req = Request::messages(vec![Message::user(prompt)]).with_model(model).with_sampling(Self::sampling(o)?).with_seed(o.seed);
        req.wait_for_model = o.wait;
        let clock = Clock { start: Instant::now(), first: Mutex::new(None) };
        let answer = self.pinned(peer).infer(req, &clock, &CancelToken::new()).await.map_err(|e| anyhow!("{e}"))?;
        let total = clock.start.elapsed();
        let ttft = *clock.first.lock().unwrap_or_else(|e| e.into_inner());
        let u = answer.usage;
        Ok(TimedChat {
            text: answer.text,
            ttft,
            total,
            prompt_tokens: u.as_ref().map(|u| u.prompt_tokens),
            cached_tokens: u.as_ref().map(|u| u.cached_tokens),
            completion_tokens: u.as_ref().map(|u| u.completion_tokens),
            served_by: if answer.meta.cache_hit { Some("cache".into()) } else { answer.meta.served_by.clone().or(Some(answer.node)) },
            standin: answer.meta.standin.clone(),
            finish_reason: answer.finish_reason,
        })
    }

    /// Embeds `inputs` on `peer`; the vector length and how long it took.
    pub async fn embed_timed(&self, peer: &str, model: &str, inputs: &[String]) -> Result<(usize, Duration)> {
        let start = Instant::now();
        let vectors = if let Some(node) = self.model_node_of(peer) {
            self.ensure_loaded(&node, model).await?;
            start_embed(&*node, model, inputs).await?
        } else {
            let req = EmbedRequest::new(inputs.to_vec()).with_model(model);
            self.pinned(peer).embed(req, &CancelToken::new()).await.map_err(|e| anyhow!("{e}"))?.vectors
        };
        let dims = vectors.first().map_or(0, Vec::len);
        if dims == 0 {
            bail!("the embedding answer had no vector");
        }
        Ok((dims, start.elapsed()))
    }
}

async fn start_embed(node: &dyn ModelNode, model: &str, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
    Ok(node.embed(model, inputs).await.map_err(|e| anyhow!("{e}"))?.0)
}
