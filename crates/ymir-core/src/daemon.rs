//! One `ymird`: whichever of the three roles `node.toml` turns on, wired through the in-process
//! fast path, and over Weirdnet to the user's other devices once this one has an identity.

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use weirdnet_backends::{EngineKind, Supervisor, SupervisorOptions};
use weirdnet_semcache::{AnswerCache, ExactCache, NoCache};

use crate::control::{self, ControlHandler};
use crate::group::{self, RpcWorkers};
use crate::membership::{Membership, SyncNode};
use crate::{Coordinator, LocalModelNode, ModelNode, NodeConfig, Paths, host, remote};

/// This device on Weirdnet.
pub struct Network {
    pub membership: Arc<Membership>,
    pub node: Arc<SyncNode>,
    pub hints: remote::Hints,
    /// The synced training job board (P3b), for `ymirctl train run --job` on this box.
    pub jobs: Option<Arc<crate::train_board::BoardJobStore>>,
    /// This node's training bundles (datasets, checkpoints), served on `/ymir/blobs/1`, and the
    /// client that fetches others' into it.
    pub blobs: Option<(Arc<crate::train_job::LocalBlobs>, Arc<crate::blobs_net::BlobsClient>)>,
}

impl Network {
    /// Starts the Weirdnet node when this device has an identity and `[network]` allows it.
    pub async fn start(cfg: &NodeConfig, paths: &Paths) -> Result<Option<Self>> {
        if !cfg.network.enabled {
            return Ok(None);
        }
        let Some(m) = Membership::open(&paths.pool())? else { return Ok(None) };
        // A model node votes on the training job board's claims once its grant has the jobs.
        if cfg.roles.model {
            match m.enable_train_voter() {
                Ok(true) => tracing::info!("training jobs: voting on {}/ claims", crate::membership::TRAIN_TYPE),
                Ok(false) => tracing::debug!("training jobs: no {} grant, not voting", crate::membership::TRAIN_TYPE),
                Err(e) => tracing::warn!("training jobs: the lease voter: {e:#}"),
            }
        }
        let m = Arc::new(m);
        let ncfg = cfg.network.node_config(&cfg.name())?;
        let hints = cfg.network.hints()?;
        let (key, state) = (m.node_key().clone(), m.state());
        let node =
            tokio::task::spawn_blocking(move || weirdnet_sync::start_node(key, ncfg, state)).await?.map_err(|e| anyhow!("starting Weirdnet: {e}"))?;
        tracing::info!(peer = %node.peer_id(), user = %m.user_id(), devices = m.roster().active().count(), "Weirdnet: on");
        let node = Arc::new(node);
        let jobs = {
            let hints = hints.clone();
            let leases = weirdnet_sync::Leases::new(node.clone(), m.state())
                .with_addrs(move |p| hints.iter().find(|(h, _)| h.to_string() == p).map(|(_, a)| a.clone()).unwrap_or_default());
            match crate::train_board::BoardJobStore::open(&weirdnet_agent::jobs::SyncHandle::from_leases(leases), m.peer_id().to_string()) {
                Ok(b) => Some(Arc::new(b)),
                Err(e) => {
                    tracing::warn!("training jobs: the synced board: {e:#}");
                    None
                }
            }
        };
        let blobs = match crate::train_job::LocalBlobs::for_node(paths, cfg) {
            Ok(b) => {
                let b = Arc::new(b);
                Some((b.clone(), Arc::new(remote::blobs_client(&node, &m, &hints, b))))
            }
            Err(e) => {
                tracing::warn!("training files: {e:#}");
                None
            }
        };
        Ok(Some(Self { membership: m, node, hints, jobs, blobs }))
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

/// Whether this inference node places models now: alone with no Weirdnet; with the
/// `ymir.scheduler` lease held, on a cert-only inference node (which also votes on it); or,
/// on one that holds the seed, while no cert-only inference node exists to hold the lease.
fn placement_leader(network: Option<&Network>, serving: &mut Vec<Box<dyn std::any::Any + Send + Sync>>) -> Box<dyn Fn() -> bool + Send + Sync> {
    let Some(net) = network else { return Box::new(|| true) };
    let m = net.membership.clone();
    if !m.scheduler_candidate() {
        tracing::info!("placement: this node holds the seed, so it places only while no cert-only inference node holds the scheduler lease");
        return Box::new(move || m.scheduler_voters() == 0);
    }
    if let Err(e) = m.enable_scheduler_voter() {
        tracing::warn!("placement: the scheduler lease voter: {e:#}; not placing");
        return Box::new(|| false);
    }
    let Ok(name) = weirdnet_sync::LeaseName::new(crate::placement::SCHEDULER_LEASE) else { return Box::new(|| false) };
    let hints = net.hints.clone();
    let leases = weirdnet_sync::Leases::new(net.node.clone(), m.state())
        .with_addrs(move |p| hints.iter().find(|(h, _)| h.to_string() == p).map(|(_, a)| a.clone()).unwrap_or_default());
    let elector = Arc::new(leases.elect(name, weirdnet_sync::LeaseConfig::for_role(crate::membership::ROLE_INFERENCE)));
    serving.push(Box::new(elector.clone()));
    Box::new(move || elector.handle().is_some())
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
    /// Woken by a `shutdown` control request.
    exit: Arc<tokio::sync::Notify>,
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
        let config = paths.config();
        Self::start_at(cfg, paths, config).await
    }

    /// [`Daemon::start`] with `cfg` read from `config` (`ymird --config`): an upload saved into
    /// this node's config (`/ymir/uploads/1`) edits that file.
    pub async fn start_at(cfg: NodeConfig, paths: Paths, config: std::path::PathBuf) -> Result<Self> {
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
        let mut group_workers: Option<Arc<RpcWorkers>> = None;
        let mut model_hw = None;
        let mut planner_task = None;
        if roles.model {
            let engines = cfg.engines(&paths);
            if engines.is_empty() {
                tracing::warn!(
                    "model role: no engine found (llama.cpp is `{}`, and this install ships no engine); loads will fail until llama-server is installed",
                    cfg.model.llamacpp
                );
            }
            for e in &engines {
                tracing::info!(engine = e.kind.name(), program = %e.program.display(), "model role: engine");
            }
            let mut launches = cfg.launches();
            launches.retain(|l| match preflight_vllm(&cfg, &engines, l) {
                Ok(()) => true,
                Err(e) => {
                    tracing::error!(model = %l.id, "{e}");
                    false
                }
            });
            crate::adapters::drop_wrong_base(&cfg, &mut launches);
            crate::adapters::keep_servable(&mut launches);
            // Contexts: each model's full trained length, cut only where its KV cache would not fit.
            let hw = weirdnet_model::hw::detect();
            let budget = weirdnet_model::hw::budget(
                &hw,
                weirdnet_model::hw::parse_size(&cfg.model.memory_budget),
                cfg.model.reserve.as_deref().and_then(weirdnet_model::hw::parse_size),
            );
            let plans = crate::kvfit::plan_launches(&mut launches, &budget, &cfg.models_dir());
            crate::kvfit::log_plans(&plans);
            let explicit: std::collections::HashSet<String> = plans.iter().filter(|p| p.explicit).map(|p| p.id.clone()).collect();
            model_hw = Some((hw.clone(), budget));
            // Split models (M7): the leader's llama-server is wrapped, and appears when a group is up.
            let key = group::load_or_create_key(&paths.state().join("split.key")).with_context(|| "creating the split-model key")?;
            launches.extend(cfg.split_launches(&key));
            let sup = Supervisor::new(engines, launches.clone(), SupervisorOptions { log_dir: Some(paths.logs()), ..Default::default() });
            supervisor = Some(sup.clone());
            // A training run (or an operator) holds this node's GPU: start no model until it ends.
            if let Some(lock) = crate::train::held(&paths) {
                sup.reserve(&lock.holder).await.map_err(|e| anyhow!("reserving for {}: {e}", lock.holder))?;
                tracing::warn!(holder = %lock.holder, "this node's GPU is held (state/train/lock.json): no model loads until it is released");
            }
            // At load time, contexts follow what runs beside them (`ctxplan`).
            let probe = Arc::new(crate::ctxplan::SystemProbe::new(&hw, &budget));
            let planner = Arc::new(crate::ctxplan::Planner::new(sup.clone(), budget, probe, cfg.models_dir(), &explicit));
            planner_task = Some(planner.spawn_keeper());
            let node = Arc::new(LocalModelNode::new(&name, sup, &launches));
            node.set_planner(planner);
            node.set_revisions(cfg.revisions().into_iter().map(|(id, rev, _)| (id, rev)));
            for (id, origin, license) in cfg.origins() {
                node.set_origin(&id, origin, license);
            }
            let workers: Arc<RpcWorkers> = Arc::new(RpcWorkers::new(cfg.worker_config(&paths, key)));
            node.set_group_workers(workers.clone());
            group_workers = Some(workers);
            model_node = Some(node);
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
            let coord = Coordinator::with_options(nodes, main_model.clone(), cache, cfg.coordinator_options());
            coord.set_batch_cap(cfg.network.batch_per_device.unwrap_or_else(|| ((cfg.network.slots + cfg.network.queue) / 2).max(1)));
            Some(Arc::new(coord))
        } else {
            None
        };

        let (stop, stop_rx) = watch::channel(false);
        let mut tasks = Vec::new();
        tasks.extend(planner_task.take());
        if let (Some(c), false) = (&coordinator, cfg.inference.split.is_empty()) {
            tasks.push(c.spawn_group_watch(std::time::Duration::from_secs(5)));
        }
        let mut serving: Vec<Box<dyn std::any::Any + Send + Sync>> = Vec::new();
        let mut models_service = None;

        // Weirdnet: the roles reach the user's other devices.
        let network = Network::start(&cfg, &paths).await?;
        let ropts = remote::RemoteOptions::default();
        // Conversations: the inference and app roles keep them and sync them among themselves.
        let context = network.as_ref().filter(|_| roles.inference || roles.app).map(|n| {
            n.membership.set_context_sync(true);
            crate::context::ContextStore::new(n.membership.clone())
        });
        if let Some(net) = &network {
            tasks.push(net.sync_loop(std::time::Duration::from_secs(cfg.network.sync_secs.max(1)), stop_rx.clone()));
            if let Some((blobs, _)) = &net.blobs {
                match remote::serve_blobs(&net.node, &net.membership, blobs.clone()) {
                    Ok(server) => serving.push(Box::new(server)),
                    Err(e) => tracing::warn!("serving training files on Weirdnet: {e}"),
                }
            }
            if let Some(c) = &coordinator {
                let (server, feed) = remote::serve(
                    &net.node,
                    c.clone(),
                    &net.membership,
                    net.hints.clone(),
                    context.clone(),
                    host::admission(cfg.network.slots, cfg.network.queue),
                )
                .map_err(|e| anyhow!("serving the inference role on Weirdnet: {e}"))?;
                serving.push(Box::new(server));
                serving.push(Box::new(feed));
                if cfg.inference.remote_rollouts {
                    let svc = crate::rollouts_net::RolloutsService::new(paths.config(), c.clone());
                    let server = remote::serve_rollouts(&net.node, &net.membership, &cfg.inference.rollout_callers, svc)
                        .map_err(|e| anyhow!("serving rollouts on Weirdnet: {e}"))?;
                    serving.push(Box::new(server));
                }
                if cfg.inference.agent.enabled {
                    let auth = remote::roster_auth(&net.membership);
                    let t = crate::tasks::serve(&net.node, c.clone(), auth, &cfg.node.name, &cfg.inference.agent)
                        .await
                        .map_err(|e| anyhow!("serving agent tasks on Weirdnet: {e}"))?;
                    tasks.push(t);
                }
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
                let (hw, budget) = model_hw.clone().unwrap_or_else(|| {
                    let hw = weirdnet_model::hw::detect();
                    let budget = weirdnet_model::hw::budget(
                        &hw,
                        weirdnet_model::hw::parse_size(&cfg.model.memory_budget),
                        cfg.model.reserve.as_deref().and_then(weirdnet_model::hw::parse_size),
                    );
                    (hw, budget)
                });
                let accelerated = hw.gpus.iter().any(|g| g.accel != weirdnet_model::hw::Accel::Cpu);
                let (server, service) = remote::serve_model(
                    &net.node,
                    sup.clone(),
                    &net.membership,
                    4,
                    32,
                    Some((hw, budget)),
                    coordinator.is_some(),
                    group_workers.clone().map(|w| w as Arc<dyn weirdnet_backends::node::GroupWorkers>),
                    &cfg.model.callers,
                )
                .map_err(|e| anyhow!("serving the model role on Weirdnet: {e}"))?;
                if cfg.model.callers.is_empty() {
                    tracing::warn!("model role: any linked device may use this node, apps included; set [model] callers to the inference nodes");
                }
                if let Some(n) = &model_node {
                    n.attach_host(service.host().clone());
                }
                for (id, rev, quant) in cfg.revisions() {
                    service.set_model_info(&id, Some(rev), quant);
                }
                for (id, origin, license) in cfg.origins() {
                    service.set_model_origin(&id, origin, license);
                }
                let node_service = service.clone();
                serving.push(Box::new(server));
                serving.push(Box::new(service));
                if cfg.model.remote_manage {
                    // The owner's devices manage this node's model files (models/1); a pulled model is
                    // added to the supervisor in memory. An upload is also recorded and, unless the
                    // uploader says `--no-persist`, saved in node.toml (/ymir/uploads/1).
                    let store = crate::models::store(cfg.models_dir(), &paths.state());
                    let policy =
                        weirdnet_backends::models::WritePolicy::default().with_keep_free(cfg.model.keep_free()).with_uploads(cfg.model.upload);
                    let (server, svc) = remote::serve_models(
                        net,
                        sup.clone(),
                        store,
                        budget,
                        cfg.model.upload_engine(),
                        policy,
                        remote::PlacedInfo { record: node_service.clone(), local: model_node.clone() },
                    )
                    .map_err(|e| anyhow!("serving model management on Weirdnet: {e}"))?;
                    serving.push(Box::new(server));
                    let uploads = crate::upload_net::UploadsService::new(config.clone(), cfg.models_dir(), svc.clone(), node_service, accelerated);
                    let server =
                        remote::serve_uploads(&net.node, &net.membership, uploads).map_err(|e| anyhow!("serving uploads on Weirdnet: {e}"))?;
                    serving.push(Box::new(server));
                    let server = remote::serve_train_status(
                        &net.node,
                        &net.membership,
                        paths.clone(),
                        cfg.clone(),
                        net.jobs.clone().map(|j| j as Arc<dyn crate::train_job::JobStore>),
                    )
                    .map_err(|e| anyhow!("serving training status on Weirdnet: {e}"))?;
                    serving.push(Box::new(server));
                    models_service = Some(svc);
                }
                if cfg.train.remote {
                    // The owner's devices start runs here (`ymirctl train run --node`); node.toml is
                    // never touched from here.
                    let server = remote::serve_train(&net.node, &net.membership, paths.clone(), cfg.clone())
                        .map_err(|e| anyhow!("serving training runs on Weirdnet: {e}"))?;
                    serving.push(Box::new(server));
                    if cfg.train.callers.is_empty() {
                        tracing::warn!("training: any linked device may start a run on this node; set [train] callers to narrow it");
                    }
                    tracing::info!("training: taking runs from the user's devices on Weirdnet ({})", crate::train_net::TRAIN_PROTOCOL);
                }
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

        // Placement and replication (M6): one inference node at a time keeps the listed models
        // loaded on their number of model nodes (see `placement`).
        if let (Some(c), false, true) = (&coordinator, cfg.inference.placement.is_empty(), cfg.inference.scheduler_eligible) {
            let leads = placement_leader(network.as_ref(), &mut serving);
            tasks.push(c.spawn_placement(cfg.inference.placement.clone(), leads));
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
        let mut app_pool = None;
        if roles.app {
            let pool = match (&coordinator, &network) {
                (Some(c), _) => host::local_pool(c.clone(), &name, host::admission(cfg.network.slots, cfg.network.queue), context.clone()),
                (None, Some(net)) => remote::connect_pool(net.node.clone(), &net.membership, &net.hints, &ropts),
                (None, None) => {
                    return Err(anyhow!(
                        "the app role needs an inference node: turn on the inference role here, or link this device to the user's devices (`ymirctl pool link`)"
                    ));
                }
            };
            let pool = Arc::new(pool);
            app_pool = Some(pool.clone());
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
        let exit = Arc::new(tokio::sync::Notify::new());
        let handler = Arc::new(ControlHandler {
            node: name.clone(),
            roles: roles.names().into_iter().map(str::to_string).collect(),
            bridge_port: bridge_addr.map(|a| a.port()),
            main_model,
            coordinator: coordinator.clone(),
            model_node,
            alerter,
            network: network.as_ref().map(|n| (n.membership.clone(), n.node.clone())),
            models: network.as_ref().map(|n| Arc::new(remote::models_hub(&n.node, &n.membership, &n.hints, &ropts, &name, models_service))),
            train: network.as_ref().map(|n| Arc::new(remote::train_hub(&n.node, &n.membership, &n.hints, &ropts, &name))),
            jobs: network.as_ref().and_then(|n| n.jobs.clone().map(|j| (j, n.membership.peer_id().to_string()))),
            blobs: network.as_ref().and_then(|n| n.blobs.as_ref().map(|(_, c)| c.clone())),
            context,
            pool: app_pool,
            exit: exit.clone(),
        });
        tasks.push(tokio::spawn(control::serve(listener, handler, stop_rx)));
        tracing::info!(node = %name, roles = ?roles.names(), "ymird started");
        Ok(Self { name, bridge_addr, coordinator, paths, supervisor, stop, tasks, socket_id, reaper, serving, network, exit })
    }

    /// Resolves when a `shutdown` control request arrives (`ymirctl service stop` on a Windows
    /// scheduled task). The caller then runs [`Daemon::shutdown`].
    pub async fn exit_requested(&self) {
        self.exit.notified().await;
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
            // Requests still running hold the node (through the model nodes they route to), so
            // ask it to stop whether or not this is the last holder: connections close now, and
            // an asker whose answer was under way sees the node go and resumes elsewhere.
            net.node.halt();
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

/// Drops an Inkling vLLM launch that this node's vLLM cannot load, with a clear error. Wrapped
/// servers and container images are left alone (the operator picked them).
fn preflight_vllm(cfg: &NodeConfig, engines: &[weirdnet_backends::EngineConfig], l: &weirdnet_backends::ModelLaunch) -> Result<()> {
    if l.engine != EngineKind::Vllm || l.wrap.is_some() {
        return Ok(());
    }
    let inkling = crate::vllm::is_inkling(&l.source) || l.extra_args.iter().any(|a| a == "inkling");
    if !inkling {
        return Ok(());
    }
    if crate::config::vllm_via_image(&cfg.model) {
        return Ok(());
    }
    let Some(eng) = engines.iter().find(|e| e.kind == EngineKind::Vllm) else {
        anyhow::bail!("{}", crate::vllm::inkling_unsupported_message(None));
    };
    crate::vllm::ensure_inkling(&eng.program, false)
}
