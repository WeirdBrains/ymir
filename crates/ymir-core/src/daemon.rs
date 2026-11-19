//! One `ymird`: whichever of the three roles `node.toml` turns on, wired through the in-process
//! fast path.

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use ymir_backends::{Supervisor, SupervisorOptions};
use ymir_semcache::{AnswerCache, ExactCache, NoCache};

use crate::control::{self, ControlHandler};
use crate::{Coordinator, LocalModelNode, ModelNode, NodeConfig, Paths};

pub struct Daemon {
    pub name: String,
    pub bridge_addr: Option<SocketAddr>,
    pub coordinator: Option<Arc<Coordinator>>,
    paths: Paths,
    supervisor: Option<Supervisor>,
    stop: watch::Sender<bool>,
    tasks: Vec<JoinHandle<()>>,
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
                tracing::warn!("inference role without a local model role: remote model nodes come with roles over Weirdnet (M3)");
            }
            let cache: Arc<dyn AnswerCache> = if cfg.inference.cache.mode == "off" { Arc::new(NoCache) } else { Arc::new(ExactCache::new(256)) };
            Some(Arc::new(Coordinator::new(nodes, main_model.clone(), cache)))
        } else {
            None
        };

        let (stop, stop_rx) = watch::channel(false);
        let mut tasks = Vec::new();

        // App role: the loopback bridge over the in-process pool.
        let mut bridge_addr = None;
        if roles.app {
            let pool = coordinator.clone().ok_or_else(|| {
                anyhow!("the app role needs an inference node; reaching one over Weirdnet comes in M3, so turn on the inference role here")
            })?;
            let token = crate::token::load_or_create(&paths)?;
            let bridge = ymir_openai::Bridge::bind(ymir_openai::BridgeOptions { port: cfg.app.bridge.port, token })
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
        let handler = Arc::new(ControlHandler {
            node: name.clone(),
            roles: roles.names().into_iter().map(str::to_string).collect(),
            bridge_port: bridge_addr.map(|a| a.port()),
            main_model,
            coordinator: coordinator.clone(),
            model_node,
        });
        tasks.push(tokio::spawn(control::serve(listener, handler, stop_rx)));
        tracing::info!(node = %name, roles = ?roles.names(), "ymird started");
        Ok(Self { name, bridge_addr, coordinator, paths, supervisor, stop, tasks })
    }

    /// Stops the bridge and control socket, then every engine.
    pub async fn shutdown(self) {
        let _ = self.stop.send(true);
        for t in self.tasks {
            let _ = tokio::time::timeout(std::time::Duration::from_secs(5), t).await;
        }
        if let Some(s) = &self.supervisor {
            s.shutdown().await;
        }
        let _ = std::fs::remove_file(self.paths.socket());
        tracing::info!("ymird stopped");
    }
}
