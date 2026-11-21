//! Direct, pinned access to the user's inference nodes over Weirdnet, for `ymirctl bench`.
//!
//! The loopback bridge hands a request to the pool, which picks the node. A benchmark has to ask a
//! chosen node, so [`Direct`] starts a Weirdnet node of its own with this device's identity and
//! certificate (the same files `ymird` opens, read only; no roster or key is written) and talks the
//! protocols the pool client already uses: `/weirdnet-agent/catalog/1` for a node's models and
//! `/weirdnet-agent/infer/2` for answers. Pinning is a [`Pool`] whose node list is that one peer, so
//! nothing here reimplements routing, failover or the wire formats.
//!
//! The probe node listens on ephemeral ports, with no LAN discovery, DHT or relays, so it neither
//! collides with a running `ymird` of the same identity nor announces itself. It reaches other nodes
//! through the `[network] peers` of `node.toml` and any extra hints it is given.
//!
//! "Node" here is an inference node. An inference node may place a model on a model node of its
//! own, so every answer also reports `served_by`.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use weirdnet_agent::pool::{EventSink, NetDialer, Pool, PoolEvent, PoolOptions, encode_cert};
use weirdnet_agent::remote::catalog::{Catalog, ModelState, Purpose};
use weirdnet_agent::remote::infer::v2::{EmbedRequest, Request, Sampling};
use weirdnet_agent::{CancelToken, Message};
use weirdnet_net::{Multiaddr, PeerHint, PeerId};

use crate::client::TimedChat;
use crate::daemon::Network;
use crate::{NodeConfig, Paths};

/// A linked inference-capable device and what it serves.
#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub peer: String,
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
}

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
        Ok(Self { net, extra, accept })
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

    /// `peer`'s models now.
    pub async fn models(&self, peer: &str) -> Result<Vec<ModelInfo>> {
        let c = self.pinned(peer).catalog().await.map_err(|e| anyhow!("{e}"))?;
        Ok(Self::catalog_models(&c))
    }

    /// Every linked device with its models (or the reason it did not answer).
    pub async fn nodes(&self) -> Vec<NodeInfo> {
        let mut out = Vec::new();
        for (peer, label) in self.devices() {
            let models = self.models(&peer).await.map_err(|e| format!("{e:#}"));
            out.push(NodeInfo { peer, label, models });
        }
        out
    }

    /// One streamed answer from `peer`, timed.
    pub async fn chat_timed(&self, peer: &str, model: &str, prompt: &str, o: &AskOptions) -> Result<TimedChat> {
        let sampling: Sampling = serde_json::from_value(serde_json::json!({"temperature": o.temperature, "max_tokens": o.max_tokens}))?;
        let mut req = Request::messages(vec![Message::user(prompt)]).with_model(model).with_sampling(sampling).with_seed(o.seed);
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
            served_by: answer.meta.served_by.clone().or(Some(answer.node)),
            standin: answer.meta.standin.clone(),
            finish_reason: answer.finish_reason,
        })
    }

    /// Embeds `inputs` on `peer`; the vector length and how long it took.
    pub async fn embed_timed(&self, peer: &str, model: &str, inputs: &[String]) -> Result<(usize, Duration)> {
        let start = Instant::now();
        let req = EmbedRequest::new(inputs.to_vec()).with_model(model);
        let e = self.pinned(peer).embed(req, &CancelToken::new()).await.map_err(|e| anyhow!("{e}"))?;
        let dims = e.vectors.first().map_or(0, Vec::len);
        if dims == 0 {
            bail!("the embedding answer had no vector");
        }
        Ok((dims, start.elapsed()))
    }
}
