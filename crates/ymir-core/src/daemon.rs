//! One `ymird`: whichever of the three roles `node.toml` turns on, wired through the in-process
//! fast path, and over Weirdnet to the user's other devices once this one has an identity.

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use weirdnet_backends::{Supervisor, SupervisorOptions};
use weirdnet_semcache::{AnswerCache, ExactCache, NoCache};

use crate::control::{self, ControlHandler};
use crate::membership::{Membership, SyncNode};
use crate::{Coordinator, LocalModelNode, ModelNode, NodeConfig, Paths, host, remote};

/// This device on Weirdnet.
pub struct Network {
    pub membership: Arc<Membership>,
    pub node: Arc<SyncNode>,
    pub hints: remote::Hints,
}

impl Network {
    /// Starts the Weirdnet node when this device has an identity and `[network]` allows it.
    pub async fn start(cfg: &NodeConfig, paths: &Paths) -> Result<Option<Self>> {
        if !cfg.network.enabled {
            return Ok(None);
        }
        let Some(m) = Membership::open(&paths.pool())? else { return Ok(None) };
        let m = Arc::new(m);
        let ncfg = cfg.network.node_config(&cfg.name())?;
        let hints = cfg.network.hints()?;
        let (key, state) = (m.node_key().clone(), m.state());
        let node =
            tokio::task::spawn_blocking(move || weirdnet_sync::start_node(key, ncfg, state)).await?.map_err(|e| anyhow!("starting Weirdnet: {e}"))?;
        tracing::info!(peer = %node.peer_id(), user = %m.user_id(), devices = m.roster().active().count(), "Weirdnet: on");
        Ok(Some(Self { membership: m, node: Arc::new(node), hints }))
    }

    /// Sync passes against the user's other devices every `every` (roster changes, revocations).
    fn sync_loop(&self, every: std::time::Duration, mut stop: watch::Receiver<bool>) -> JoinHandle<()> {
        let (m, node, hints) = (self.membership.clone(), self.node.clone(), self.hints.clone());
        tokio::spawn(async move {
            loop {
                let (m, node, hints) = (m.clone(), node.clone(), hints.clone());
                let _ = tokio::task::spawn_blocking(move || {
                    m.tick(&node, &|c| hints.iter().find(|(p, _)| p.to_string() == c.peer_id).map(|(_, a)| a.clone()).unwrap_or_default())
                })
                .await;
                tokio::select! {
                    _ = tokio::time::sleep(every) => {}
                    _ = stop.wait_for(|s| *s) => return,
                }
            }
        })
    }
}

pub struct Daemon {
    pub name: String,
    pub bridge_addr: Option<SocketAddr>,
    pub coordinator: Option<Arc<Coordinator>>,
    paths: Paths,
    supervisor: Option<Supervisor>,
    stop: watch::Sender<bool>,
    tasks: Vec<JoinHandle<()>>,
    socket_id: Option<(u64, u64)>,
    reaper: Option<JoinHandle<()>>,
    /// Weirdnet servers (inference, model node), kept until shutdown.
    serving: Vec<Box<dyn std::any::Any + Send + Sync>>,
    network: Option<Network>,
}

/// (device, inode) of a file, to tell our socket from a successor's.
fn file_id(p: &std::path::Path) -> Option<(u64, u64)> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        std::fs::symlink_metadata(p).ok().map(|m| (m.dev(), m.ino()))
    }
    #[cfg(not(unix))]
    {
        std::fs::symlink_metadata(p).ok().map(|_| (0, 0))
    }
}

impl Daemon {
    /// Starts the enabled roles. Returns once the control socket and the bridge are listening;
    /// models load on first use or on `ymirctl load`.
    pub async fn start(cfg: NodeConfig, paths: Paths) -> Result<Self> {
        cfg.validate()?;
        paths.ensure_state().with_context(|| format!("creating {}", paths.state().display()))?;
        let name = cfg.name();
        let roles = cfg.roles;
        if !(roles.model || roles.inference || roles.app) {
            return Err(anyhow!("no roles are on in {}; set [roles] model, inference and/or app", paths.config().display()));
        }

        // Model role.
        let mut model_node = None;
        let mut supervisor = None;
        if roles.model {
            let engines = cfg.engines(&paths);
            if engines.is_empty() {
                tracing::warn!(
                    "model role: no engine found (llama.cpp is `{}`); loads will fail until llama-server is installed",
                    cfg.model.llamacpp
                );
            }
            for e in &engines {
                tracing::info!(engine = e.kind.name(), program = %e.program.display(), "model role: engine");
            }
            let launches = cfg.launches();
            let sup = Supervisor::new(engines, launches.clone(), SupervisorOptions { log_dir: Some(paths.logs()), ..Default::default() });
            supervisor = Some(sup.clone());
            model_node = Some(Arc::new(LocalModelNode::new(&name, sup, &launches)));
        }

        // Inference role.
        let main_model = cfg.main_model().map(|m| m.id.clone());
        let coordinator = if roles.inference {
            let nodes: Vec<Arc<dyn ModelNode>> = model_node.iter().map(|n| n.clone() as Arc<dyn ModelNode>).collect();
            if nodes.is_empty() {
                tracing::info!("inference role without a local model role: it uses the model nodes among the user's devices");
            }
            let cache: Arc<dyn AnswerCache> =
                if cfg.inference.cache.mode == "off" { Arc::new(NoCache) } else { Arc::new(ExactCache::<weirdnet_semcache::MemoryStore>::new(256)) };
            Some(Arc::new(Coordinator::with_options(nodes, main_model.clone(), cache, cfg.coordinator_options())))
        } else {
            None
        };

        let (stop, stop_rx) = watch::channel(false);
        let mut tasks = Vec::new();
        let mut serving: Vec<Box<dyn std::any::Any + Send + Sync>> = Vec::new();

        // Weirdnet: the roles reach the user's other devices.
        let network = Network::start(&cfg, &paths).await?;
        let ropts = remote::RemoteOptions::default();
        if let Some(net) = &network {
            tasks.push(net.sync_loop(std::time::Duration::from_secs(30), stop_rx.clone()));
            if let Some(c) = &coordinator {
                let (server, feed) = remote::serve(&net.node, c.clone(), &net.membership, host::admission(cfg.network.slots, cfg.network.queue))
                    .map_err(|e| anyhow!("serving the inference role on Weirdnet: {e}"))?;
                serving.push(Box::new(server));
                serving.push(Box::new(feed));
                let d = remote::discover_model_nodes(
                    c.clone(),
                    net.node.clone(),
                    net.membership.clone(),
                    net.hints.clone(),
                    ropts.clone(),
                    std::time::Duration::from_secs(10),
                );
                tasks.push(d);
                tracing::info!("inference role: serving the user's devices on Weirdnet");
            }
            if roles.model
                && let Some(sup) = &supervisor
            {
                let hw = weirdnet_model::hw::detect();
                let budget = weirdnet_model::hw::budget(
                    &hw,
                    weirdnet_model::hw::parse_size(&cfg.model.memory_budget),
                    cfg.model.reserve.as_deref().and_then(weirdnet_model::hw::parse_size),
                );
                let (server, service) =
                    remote::serve_model(&net.node, sup.clone(), &net.membership, 4, 32, Some((hw, budget)), coordinator.is_some())
                        .map_err(|e| anyhow!("serving the model role on Weirdnet: {e}"))?;
                if let Some(n) = &model_node {
                    n.attach_host(service.host().clone());
                }
                serving.push(Box::new(server));
                serving.push(Box::new(service));
                tracing::info!(
                    "model role: serving the user's inference nodes on Weirdnet{}",
                    if coordinator.is_some() { " (alongside the inference role)" } else { "" }
                );
            }
        }

        // Idle unload.
        let reaper = supervisor.as_ref().map(|sup| sup.spawn_reaper(std::time::Duration::from_secs(2)));

        // Keep the stand-in warm.
        if let (Some(c), Some(standin), true) = (&coordinator, cfg.standin(), cfg.inference.standin_warm) {
            let c = c.clone();
            tasks.push(tokio::spawn(async move {
                match c.load(&standin, None, false).await {
                    Ok(on) => tracing::info!(model = %standin, node = %on, "stand-in warm"),
                    Err(e) => tracing::warn!(model = %standin, "stand-in not warm: {e}"),
                }
            }));
        }

        let mut alerter = None;
        // Alerts: engine events and health probes, on the inference role (which aggregates
        // health), or on a lone model node for its own engines.
        if (roles.inference || roles.model) && cfg.inference.alerts.enabled {
            match cfg.inference.alerts.alerter(&name) {
                Ok(a) => {
                    let a = Arc::new(a.with_product("ymir"));
                    alerter = Some(a.clone());
                    let mut disks = cfg.inference.alerts.disks.clone();
                    if disks.is_empty() {
                        disks.push(paths.home.clone());
                        if roles.model {
                            disks.push(cfg.models_dir());
                        }
                    }
                    let every = std::time::Duration::from_secs(cfg.inference.alerts.probe_secs.max(5));
                    tasks.push(tokio::spawn(weirdnet_alerts::engine::monitor(a, supervisor.clone(), disks, every, stop_rx.clone())));
                }
                Err(e) => tracing::error!("alerts are off: {e}"),
            }
        }

        // App role: the loopback bridge over the in-process pool.
        let mut bridge_addr = None;
        if roles.app {
            let pool = match (&coordinator, &network) {
                (Some(c), _) => host::local_pool(c.clone(), &name, host::admission(cfg.network.slots, cfg.network.queue)),
                (None, Some(net)) => remote::connect_pool(net.node.clone(), &net.membership, &net.hints, &ropts),
                (None, None) => {
                    return Err(anyhow!(
                        "the app role needs an inference node: turn on the inference role here, or link this device to the user's devices (`ymirctl pool link`)"
                    ));
                }
            };
            let pool = Arc::new(pool);
            let token = crate::token::load_or_create(&paths)?;
            let bridge = weirdnet_openai::Bridge::bind(weirdnet_openai::BridgeOptions::new(cfg.app.bridge.port, token))
                .await
                .with_context(|| format!("binding the bridge on 127.0.0.1:{}", cfg.app.bridge.port))?;
            let addr = bridge.local_addr()?;
            bridge_addr = Some(addr);
            let mut rx = stop_rx.clone();
            tasks.push(tokio::spawn(async move {
                let shutdown = async move {
                    let _ = rx.wait_for(|s| *s).await;
                };
                if let Err(e) = bridge.serve(pool, shutdown).await {
                    tracing::error!("bridge: {e}");
                }
            }));
            tracing::info!("app role: bridge on http://{addr}/v1");
        }

        // Control socket.
        let listener = control::bind(&paths.socket()).await?;
        let socket_id = file_id(&paths.socket());
        let handler = Arc::new(ControlHandler {
            node: name.clone(),
            roles: roles.names().into_iter().map(str::to_string).collect(),
            bridge_port: bridge_addr.map(|a| a.port()),
            main_model,
            coordinator: coordinator.clone(),
            model_node,
            alerter,
            network: network.as_ref().map(|n| (n.membership.clone(), n.node.clone())),
        });
        tasks.push(tokio::spawn(control::serve(listener, handler, stop_rx)));
        tracing::info!(node = %name, roles = ?roles.names(), "ymird started");
        Ok(Self { name, bridge_addr, coordinator, paths, supervisor, stop, tasks, socket_id, reaper, serving, network })
    }

    /// Stops the bridge and control socket, then every engine.
    pub async fn shutdown(self) {
        let _ = self.stop.send(true);
        if let Some(r) = &self.reaper {
            r.abort();
        }
        for t in self.tasks {
            t.abort();
            let _ = tokio::time::timeout(std::time::Duration::from_secs(5), t).await;
        }
        drop(self.serving);
        if let Some(net) = self.network {
            drop(net.membership);
            if let Ok(node) = Arc::try_unwrap(net.node) {
                let _ = tokio::task::spawn_blocking(move || node.stop()).await;
            }
        }
        if let Some(s) = &self.supervisor {
            s.shutdown().await;
        }
        // Only our own socket: a new ymird may already have replaced it while engines stopped.
        if self.socket_id.is_some() && file_id(&self.paths.socket()) == self.socket_id {
            let _ = std::fs::remove_file(self.paths.socket());
        }
        tracing::info!("ymird stopped");
    }
}
