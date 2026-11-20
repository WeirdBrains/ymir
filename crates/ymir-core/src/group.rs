//! Split models (M7): group jobs over llama.cpp RPC.
//!
//! A model too big for one machine is served by a *group*: workers (`rpc-server`) on the other
//! members, bound to their private link only, and a leader that runs `llama-server --rpc`
//! against them. The inference node decides (this module's [`plan_group`], run by the
//! coordinator), the model nodes run it ([`RpcWorkers`], behind `GroupUp`/`GroupDown` of
//! `/weirdnet-backends/node/1`). Ray and vLLM groups are not here: the `vllm-ray` backend needs
//! a RoCE cluster that was not available to test on.

use std::collections::HashMap;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::process::{Child, Command};
use weirdnet_agent::remote::RemoteError;
use weirdnet_backends::node::{GroupPlan, GroupWorkers};

/// The worker kind the plans name.
pub const WORKER_KIND: &str = "rpc-server";
/// The group backend: GGUF across any nodes.
pub const BACKEND: &str = "llamacpp-rpc";

fn default_rpc_port() -> u16 {
    50052
}

fn default_serve_port() -> u16 {
    18080
}

fn default_backend() -> String {
    BACKEND.into()
}

fn default_min_gbps() -> f64 {
    10.0
}

/// A split model the leader can serve (`[[model.split]]` in `node.toml`). The model node
/// advertises it like any other, wrapping the `llama-server` the group's leader starts on
/// `serve_port` (loopback, with a key from the state dir).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplitModel {
    pub id: String,
    /// GGUF path, absolute or relative to `models_dir`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    /// Where the leader's `llama-server` listens (loopback).
    #[serde(default = "default_serve_port")]
    pub serve_port: u16,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slots: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_layers: Option<i32>,
    /// More `llama-server` arguments (such as `--tensor-split 3,1`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
}

/// One model node a split model may use (`[[inference.split.node]]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplitNode {
    /// The model node: its name (this box) or peer id.
    pub node: String,
    /// Its private-link address.
    pub link: String,
    /// The port its `rpc-server` listens on.
    #[serde(default = "default_rpc_port")]
    pub port: u16,
    /// Memory the group may use there, such as `10GiB`.
    pub memory: String,
    /// The measured or rated link speed in Gb/s; unknown is refused unless the link is loopback.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gbps: Option<f64>,
    /// Models to unload there before it joins.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub drop: Vec<String>,
}

/// The pool policy for one split model (`[[inference.split]]`). Off unless listed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplitPolicy {
    /// The model's pool id (also the leader's `[[model.split]]` id).
    pub id: String,
    #[serde(default = "default_backend")]
    pub backend: String,
    /// Memory the whole model needs, such as `7GiB`.
    pub size: String,
    /// The node that serves it; default the candidate with the most memory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub leader: Option<String>,
    /// Slowest link accepted, in Gb/s.
    #[serde(default = "default_min_gbps")]
    pub min_link_gbps: f64,
    #[serde(default, rename = "node")]
    pub nodes: Vec<SplitNode>,
}

/// A link address that may carry tensor traffic: loopback, a private (RFC 1918) range,
/// link-local, or an IPv6 unique-local address. Public addresses, CGNAT (tailnets) and anything
/// else are refused: `rpc-server` has no authentication of its own.
pub fn is_private_link(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(a) => a.is_loopback() || a.is_private() || a.is_link_local(),
        IpAddr::V6(a) => a.is_loopback() || (a.segments()[0] & 0xfe00) == 0xfc00 || (a.segments()[0] & 0xffc0) == 0xfe80,
    }
}

pub fn parse_link(s: &str) -> Result<IpAddr, String> {
    let ip: IpAddr = s.trim().parse().map_err(|_| format!("`{s}` is not an IP address"))?;
    if is_private_link(ip) {
        Ok(ip)
    } else {
        Err(format!(
            "{ip} is not a private link address (a direct cable or isolated VLAN: loopback, 10/8, 172.16/12, 192.168/16, link-local or fc00::/7)"
        ))
    }
}

/// What the planner knows about a candidate model node.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub name: String,
    pub memory: u64,
    pub link: IpAddr,
    pub port: u16,
    pub gbps: Option<f64>,
    /// Takes `GroupUp` (its record says so).
    pub group_jobs: bool,
    pub draining: bool,
    pub drop: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PlanError {
    NoCandidates(String),
    TooSmall { need: u64, have: u64 },
    UnknownLeader(String),
    BadLink(String, String),
    SlowLink(String, f64, f64),
    UnknownSpeed(String),
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanError::NoCandidates(m) => write!(f, "no model node can take group jobs: {m}"),
            PlanError::TooSmall { need, have } => write!(f, "the model needs {need} bytes and the candidate nodes have {have} between them"),
            PlanError::UnknownLeader(l) => write!(f, "`{l}` is not a candidate node of this model"),
            PlanError::BadLink(n, m) => write!(f, "{n}: link {m}"),
            PlanError::SlowLink(n, g, min) => write!(f, "{n}: its link ({g} Gb/s) is slower than the {min} Gb/s this model asks for"),
            PlanError::UnknownSpeed(n) => write!(f, "{n}: link speed unknown (set `gbps` on the node)"),
        }
    }
}

impl std::error::Error for PlanError {}

/// Picks the group: the leader (the named one, else the candidate with the most memory), then
/// more members by memory until the model fits. Members other than the leader need a private
/// link at least as fast as the policy asks. Returns the leader and the members in order, the
/// leader first.
pub fn plan_group(policy: &SplitPolicy, need: u64, cands: &[Candidate]) -> Result<Vec<Candidate>, PlanError> {
    let mut usable: Vec<&Candidate> = cands.iter().filter(|c| c.group_jobs && !c.draining).collect();
    if usable.is_empty() {
        return Err(PlanError::NoCandidates("none is up, taking group jobs and not draining".into()));
    }
    usable.sort_by(|a, b| b.memory.cmp(&a.memory).then_with(|| a.name.cmp(&b.name)));
    let leader = match &policy.leader {
        Some(l) => *usable.iter().find(|c| &c.name == l).ok_or_else(|| PlanError::UnknownLeader(l.clone()))?,
        None => usable[0],
    };
    let mut group = vec![leader.clone()];
    let mut have = leader.memory;
    for c in usable.into_iter().filter(|c| c.name != leader.name) {
        if have >= need {
            break;
        }
        if !is_private_link(c.link) {
            return Err(PlanError::BadLink(c.name.clone(), format!("{} is not private", c.link)));
        }
        match (c.gbps, c.link.is_loopback()) {
            (_, true) => {}
            (None, false) => return Err(PlanError::UnknownSpeed(c.name.clone())),
            (Some(g), false) if g < policy.min_link_gbps => return Err(PlanError::SlowLink(c.name.clone(), g, policy.min_link_gbps)),
            _ => {}
        }
        have += c.memory;
        group.push(c.clone());
    }
    if have < need {
        return Err(PlanError::TooSmall { need, have });
    }
    Ok(group)
}

/// The plan a member receives: the same group, with that member's own link, port and drop list.
pub fn plan_for(policy: &SplitPolicy, group: &[Candidate], member: &Candidate) -> GroupPlan {
    let mut plan = GroupPlan::new(format!("split-{}", policy.id), group[0].name.clone())
        .with_members(group.iter().map(|c| c.name.clone()).collect())
        .with_drop(member.drop.clone())
        .with_model(policy.id.clone())
        .with_link(member.link.to_string())
        .with_worker(WORKER_KIND);
    for c in group {
        plan.extra.insert(format!("link.{}", c.name), c.link.to_string());
        plan.extra.insert(format!("port.{}", c.name), c.port.to_string());
    }
    plan
}

/// `llama-server`'s arguments for a leader: the model, loopback, and the workers.
pub fn leader_args(split: &SplitModel, model_path: &Path, endpoints: &[String]) -> Vec<String> {
    let mut a: Vec<String> = [
        "--model",
        &model_path.to_string_lossy(),
        "--host",
        "127.0.0.1",
        "--port",
        &split.serve_port.to_string(),
        "--alias",
        &split.id,
        "--no-webui",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    a.extend(["--rpc".into(), endpoints.join(",")]);
    if let Some(c) = split.context {
        a.extend(["--ctx-size".into(), c.to_string()]);
    }
    if let Some(s) = split.slots {
        a.extend(["--parallel".into(), s.to_string()]);
    }
    if let Some(n) = split.gpu_layers {
        a.extend(["--n-gpu-layers".into(), n.to_string()]);
    }
    a.extend(split.args.iter().cloned());
    a
}

/// What [`RpcWorkers`] needs from the node's configuration.
#[derive(Debug, Clone, Default)]
pub struct WorkerConfig {
    /// `rpc-server`, when this node can be a worker.
    pub rpc_server: Option<PathBuf>,
    /// More arguments for `rpc-server`.
    pub rpc_args: Vec<String>,
    /// `llama-server`, when this node can lead.
    pub llama_server: Option<PathBuf>,
    /// This node's private-link address (`[model] private_link`): the only address it binds.
    pub private_link: Option<String>,
    pub splits: Vec<SplitModel>,
    pub models_dir: PathBuf,
    /// The key the leader's `llama-server` wants (the wrapped launch sends it).
    pub key: String,
    pub log_dir: Option<PathBuf>,
}

struct Proc {
    role: &'static str,
    child: Child,
}

/// Starts and stops a node's part of a group.
pub struct RpcWorkers {
    cfg: WorkerConfig,
    groups: tokio::sync::Mutex<HashMap<String, Vec<Proc>>>,
}

fn refused(m: impl Into<String>) -> RemoteError {
    RemoteError::Refused { message: m.into() }
}

fn failed(m: impl Into<String>) -> RemoteError {
    RemoteError::Failed { message: m.into() }
}

impl RpcWorkers {
    pub fn new(cfg: WorkerConfig) -> Self {
        Self { cfg, groups: tokio::sync::Mutex::new(HashMap::new()) }
    }

    /// Whether every process of `group` here is still running.
    pub async fn alive(&self, group: &str) -> bool {
        let mut g = self.groups.lock().await;
        match g.get_mut(group) {
            Some(procs) => procs.iter_mut().all(|p| matches!(p.child.try_wait(), Ok(None))),
            None => false,
        }
    }

    fn spawn(&self, program: &Path, args: &[String], role: &'static str, group: &str, key: Option<&str>) -> Result<Proc, RemoteError> {
        let mut cmd = Command::new(program);
        cmd.args(args).stdin(Stdio::null()).kill_on_drop(true);
        #[cfg(unix)]
        cmd.process_group(0);
        if let Some(k) = key {
            cmd.env("LLAMA_API_KEY", k);
        }
        match &self.cfg.log_dir {
            Some(dir) => {
                let _ = std::fs::create_dir_all(dir);
                let path = dir.join(format!("{group}-{role}.log"));
                let f = std::fs::File::create(&path).map_err(|e| failed(format!("{}: {e}", path.display())))?;
                cmd.stdout(f.try_clone().map_err(|e| failed(e.to_string()))?).stderr(f);
            }
            None => {
                cmd.stdout(Stdio::null()).stderr(Stdio::null());
            }
        }
        let child = cmd.spawn().map_err(|e| failed(format!("starting {}: {e}", program.display())))?;
        Ok(Proc { role, child })
    }

    async fn wait_tcp(addr: (IpAddr, u16), proc: &mut Proc, within: Duration) -> Result<(), RemoteError> {
        let deadline = Instant::now() + within;
        loop {
            if let Ok(Some(status)) = proc.child.try_wait() {
                return Err(failed(format!("the {} exited ({status}) before it was ready", proc.role)));
            }
            if tokio::net::TcpStream::connect(addr).await.is_ok() {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(RemoteError::Timeout);
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    async fn wait_health(port: u16, key: &str, proc: &mut Proc, within: Duration) -> Result<(), RemoteError> {
        let client = reqwest::Client::new();
        let url = format!("http://127.0.0.1:{port}/health");
        let deadline = Instant::now() + within;
        loop {
            if let Ok(Some(status)) = proc.child.try_wait() {
                return Err(failed(format!("the {} exited ({status}) before it was ready", proc.role)));
            }
            if let Ok(r) = client.get(&url).bearer_auth(key).timeout(Duration::from_secs(2)).send().await
                && r.status().is_success()
            {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(RemoteError::Timeout);
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }
}

#[async_trait]
impl GroupWorkers for RpcWorkers {
    async fn up(&self, me: &str, plan: &GroupPlan) -> Result<(), RemoteError> {
        if plan.worker.as_deref().is_some_and(|w| w != WORKER_KIND) {
            return Err(RemoteError::Unsupported {
                message: format!("this node runs `{WORKER_KIND}` groups, not `{}`", plan.worker.as_deref().unwrap_or("")),
            });
        }
        if !plan.members.iter().any(|m| m == me) {
            return Err(refused(format!("{me} is not a member of group {}", plan.group)));
        }
        let mut groups = self.groups.lock().await;
        if let Some(procs) = groups.get_mut(&plan.group) {
            if procs.iter_mut().all(|p| matches!(p.child.try_wait(), Ok(None))) {
                return Ok(());
            }
            // A dead remnant: start over.
            for mut p in groups.remove(&plan.group).unwrap_or_default() {
                let _ = p.child.start_kill();
                let _ = p.child.wait().await;
            }
        }
        if me == plan.leader {
            let model = plan.model.as_deref().ok_or_else(|| refused("the plan names no model"))?;
            let split =
                self.cfg.splits.iter().find(|s| s.id == model).ok_or_else(|| refused(format!("no [[model.split]] for `{model}` on this node")))?;
            let bin = self.cfg.llama_server.as_ref().ok_or_else(|| refused("no llama-server on this node"))?;
            let mut endpoints = Vec::new();
            for m in plan.members.iter().filter(|m| *m != me) {
                let ip = plan.extra.get(&format!("link.{m}")).ok_or_else(|| refused(format!("the plan has no link for {m}")))?;
                let ip = parse_link(ip).map_err(|e| refused(format!("{m}: {e}")))?;
                let port = plan.extra.get(&format!("port.{m}")).and_then(|p| p.parse::<u16>().ok()).unwrap_or(default_rpc_port());
                endpoints.push(if ip.is_ipv6() { format!("[{ip}]:{port}") } else { format!("{ip}:{port}") });
            }
            if endpoints.is_empty() {
                return Err(refused("a group needs at least one worker besides the leader"));
            }
            let file = split.file.clone().unwrap_or_else(|| format!("{}.gguf", split.id));
            let path = if Path::new(&file).is_absolute() { PathBuf::from(&file) } else { self.cfg.models_dir.join(&file) };
            if !path.is_file() {
                return Err(RemoteError::Unavailable { message: format!("{model}: model file {} is missing", path.display()) });
            }
            let mut p = self.spawn(bin, &leader_args(split, &path, &endpoints), "leader", &plan.group, Some(&self.cfg.key))?;
            let ready = Self::wait_health(split.serve_port, &self.cfg.key, &mut p, Duration::from_secs(600)).await;
            if let Err(e) = ready {
                let _ = p.child.start_kill();
                return Err(e);
            }
            groups.insert(plan.group.clone(), vec![p]);
        } else {
            let own = self
                .cfg
                .private_link
                .as_deref()
                .ok_or_else(|| refused("this node has no [model] private_link, so it takes no part in a split model"))?;
            let ip = parse_link(own).map_err(|e| refused(format!("[model] private_link: {e}")))?;
            if let Some(asked) = plan.link.as_deref()
                && asked != own
            {
                return Err(refused(format!("the plan binds {asked}, but this node's private link is {own}")));
            }
            let port = plan.extra.get(&format!("port.{me}")).and_then(|p| p.parse::<u16>().ok()).unwrap_or(default_rpc_port());
            let bin = self.cfg.rpc_server.as_ref().ok_or_else(|| refused("no rpc-server on this node"))?;
            let mut args = vec!["-H".to_string(), ip.to_string(), "-p".to_string(), port.to_string()];
            args.extend(self.cfg.rpc_args.iter().cloned());
            let mut p = self.spawn(bin, &args, "worker", &plan.group, None)?;
            if let Err(e) = Self::wait_tcp((ip, port), &mut p, Duration::from_secs(60)).await {
                let _ = p.child.start_kill();
                return Err(e);
            }
            groups.insert(plan.group.clone(), vec![p]);
        }
        Ok(())
    }

    async fn down(&self, group: &str) -> Result<(), RemoteError> {
        let procs = self.groups.lock().await.remove(group).unwrap_or_default();
        for mut p in procs {
            let _ = p.child.start_kill();
            let _ = tokio::time::timeout(Duration::from_secs(10), p.child.wait()).await;
        }
        Ok(())
    }
}

/// A group that is up.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroupInfo {
    pub model: String,
    pub group: String,
    pub leader: String,
    pub members: Vec<String>,
    pub since_ms: u64,
}

/// Why a group went down by itself.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroupLoss {
    pub model: String,
    pub reason: String,
    pub at_ms: u64,
}

/// The coordinator's record of its groups.
#[derive(Debug, Default)]
pub struct GroupBook {
    up: HashMap<String, (GroupInfo, SplitPolicy, Vec<Candidate>)>,
    lost: Vec<GroupLoss>,
    /// Models being brought up or down now, so two requests don't race.
    busy: std::collections::HashSet<String>,
}

fn now_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

fn unavailable(m: impl Into<String>) -> RemoteError {
    RemoteError::Unavailable { message: m.into() }
}

impl crate::Coordinator {
    fn book(&self) -> std::sync::MutexGuard<'_, GroupBook> {
        self.groups.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The groups that are up.
    pub fn groups(&self) -> Vec<GroupInfo> {
        let mut v: Vec<GroupInfo> = self.book().up.values().map(|(i, ..)| i.clone()).collect();
        v.sort_by(|a, b| a.model.cmp(&b.model));
        v
    }

    /// Groups that went down on their own, newest last.
    pub fn group_losses(&self) -> Vec<GroupLoss> {
        self.book().lost.clone()
    }

    /// Brings up the group for `model`: plans it from the pool policy and the nodes now in the
    /// pool, starts the workers, then the leader, and has the leader's node advertise the model.
    pub async fn group_up(&self, model: &str) -> Result<GroupInfo, RemoteError> {
        let policy =
            self.options().splits.iter().find(|p| p.id == model).cloned().ok_or_else(|| RemoteError::Refused {
                message: format!("`{model}` is not marked `split` in the pool policy ([[inference.split]])"),
            })?;
        {
            let mut b = self.book();
            if let Some((i, ..)) = b.up.get(model) {
                return Ok(i.clone());
            }
            if !b.busy.insert(model.to_string()) {
                return Err(RemoteError::Busy);
            }
        }
        let result = self.bring_up(&policy).await;
        let mut b = self.book();
        b.busy.remove(model);
        let (info, group) = result?;
        b.up.insert(model.to_string(), (info.clone(), policy, group));
        Ok(info)
    }

    async fn bring_up(&self, policy: &SplitPolicy) -> Result<(GroupInfo, Vec<Candidate>), RemoteError> {
        let need =
            weirdnet_model::hw::parse_size(&policy.size).ok_or_else(|| RemoteError::BadRequest { message: format!("size `{}`", policy.size) })?;
        let mut cands = Vec::new();
        for n in &policy.nodes {
            let Some(node) = self.node_named(&n.node) else { continue };
            let (Some(memory), Ok(link)) = (weirdnet_model::hw::parse_size(&n.memory), parse_link(&n.link)) else { continue };
            cands.push(Candidate {
                name: n.node.clone(),
                memory,
                link,
                port: n.port,
                gbps: n.gbps,
                group_jobs: node.group_jobs(),
                draining: node.draining(),
                drop: n.drop.clone(),
            });
        }
        let group = plan_group(policy, need, &cands).map_err(|e| unavailable(format!("{}: {e}", policy.id)))?;
        if group.len() < 2 {
            return Err(RemoteError::Refused { message: format!("{}: it fits on {} alone; no group is needed", policy.id, group[0].name) });
        }
        let name = format!("split-{}", policy.id);
        let mut started: Vec<Arc<dyn crate::ModelNode>> = Vec::new();
        // Workers first (the leader needs them listening), all at once; then the leader.
        let mut order: Vec<&Candidate> = group.iter().skip(1).collect();
        order.push(&group[0]);
        let mut error = None;
        'start: for batch in [&order[..order.len() - 1], &order[order.len() - 1..]] {
            let jobs = batch.iter().map(|c| {
                let node = self.node_named(&c.name);
                let plan = plan_for(policy, &group, c);
                async move {
                    match node {
                        Some(n) => {
                            let r = n.group_up(&plan).await;
                            (n, r)
                        }
                        None => unreachable!("candidates come from nodes in the pool"),
                    }
                }
            });
            for (n, r) in futures::future::join_all(jobs).await {
                match r {
                    Ok(()) => started.push(n),
                    Err(e) => {
                        error = Some(format!("{}: {e}", n.name()));
                    }
                }
            }
            if error.is_some() {
                break 'start;
            }
        }
        if error.is_none() {
            let leader = self.node_named(&group[0].name).ok_or_else(|| unavailable("the leader left the pool"))?;
            if let Err(e) = leader.load(&policy.id, true).await {
                error = Some(format!("{}: {e}", leader.name()));
            }
        }
        if let Some(e) = error {
            for n in started {
                let _ = n.group_down(&name).await;
            }
            return Err(unavailable(format!("group {name} did not come up: {e}")));
        }
        let info = GroupInfo {
            model: policy.id.clone(),
            group: name,
            leader: group[0].name.clone(),
            members: group.iter().map(|c| c.name.clone()).collect(),
            since_ms: now_ms(),
        };
        tracing::info!(model = %info.model, leader = %info.leader, members = ?info.members, "split model group is up");
        Ok((info, group))
    }

    /// Takes the group for `model` down: the leader stops advertising it, then every member stops
    /// its part. Normal placement resumes.
    pub async fn group_down(&self, model: &str) -> Result<GroupInfo, RemoteError> {
        let (info, ..) = self.book().up.remove(model).ok_or_else(|| RemoteError::Refused { message: format!("no group is up for `{model}`") })?;
        self.tear_down(&info).await;
        Ok(info)
    }

    async fn tear_down(&self, info: &GroupInfo) {
        if let Some(leader) = self.node_named(&info.leader) {
            let _ = leader.unload(&info.model).await;
        }
        for m in &info.members {
            if let Some(n) = self.node_named(m) {
                let _ = n.group_down(&info.group).await;
            }
        }
    }

    /// One look at every group that is up: a member that left the pool, started draining or lost
    /// a process takes the whole group down (NCCL, and RPC, can't survive a lost member). The
    /// model then falls back to the stand-in; the loss is kept for `ymirctl group status`.
    pub async fn check_groups(&self) -> Vec<GroupLoss> {
        let snapshot: Vec<GroupInfo> = self.groups();
        let mut lost = Vec::new();
        for info in snapshot {
            let mut reason = None;
            for m in &info.members {
                match self.node_named(m) {
                    None => reason = Some(format!("{m} left the pool")),
                    Some(n) if n.draining() => reason = Some(format!("{m} is draining")),
                    Some(n) if !n.group_alive(&info.group).await => reason = Some(format!("a process of the group on {m} exited")),
                    _ => {}
                }
                if reason.is_some() {
                    break;
                }
            }
            if let Some(reason) = reason {
                if self.book().up.remove(&info.model).is_none() {
                    continue;
                }
                tracing::warn!(model = %info.model, "split model group lost: {reason}; taking it down");
                self.tear_down(&info).await;
                let loss = GroupLoss { model: info.model.clone(), reason, at_ms: now_ms() };
                let mut b = self.book();
                b.lost.push(loss.clone());
                let n = b.lost.len();
                b.lost.drain(..n.saturating_sub(8));
                lost.push(loss);
            }
        }
        lost
    }

    /// Runs [`check_groups`](crate::coordinator::Coordinator::check_groups) every `every`.
    pub fn spawn_group_watch(self: &Arc<Self>, every: Duration) -> tokio::task::JoinHandle<()> {
        let me = Arc::downgrade(self);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(every).await;
                let Some(c) = me.upgrade() else { return };
                c.check_groups().await;
            }
        })
    }
}

/// A random key for the leader's `llama-server`, kept in `path` (mode 0600), reused on restart.
pub fn load_or_create_key(path: &Path) -> std::io::Result<String> {
    if let Ok(k) = std::fs::read_to_string(path) {
        let k = k.trim().to_string();
        if k.len() >= 32 {
            return Ok(k);
        }
    }
    let mut b = [0u8; 24];
    getrandom::fill(&mut b).map_err(std::io::Error::other)?;
    let key: String = b.iter().map(|x| format!("{x:02x}")).collect();
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let mut f = std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).open(path)?;
        f.write_all(key.as_bytes())?;
    }
    #[cfg(not(unix))]
    std::fs::write(path, &key)?;
    Ok(key)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    fn cand(name: &str, gib: u64, link: &str, gbps: Option<f64>) -> Candidate {
        Candidate {
            name: name.into(),
            memory: gib * GIB,
            link: link.parse().unwrap(),
            port: 50052,
            gbps,
            group_jobs: true,
            draining: false,
            drop: vec![],
        }
    }

    fn policy(leader: Option<&str>) -> SplitPolicy {
        SplitPolicy {
            id: "big".into(),
            backend: BACKEND.into(),
            size: "100GiB".into(),
            leader: leader.map(Into::into),
            min_link_gbps: 10.0,
            nodes: vec![],
        }
    }

    #[test]
    fn only_private_links_may_carry_tensors() {
        for ok in ["127.0.0.1", "10.0.0.1", "172.20.0.5", "192.168.0.9", "169.254.3.3", "::1", "fd00::5", "fe80::1"] {
            assert!(parse_link(ok).is_ok(), "{ok}");
        }
        for bad in ["8.8.8.8", "100.64.0.7", "0.0.0.0", "2001:db8::1", "host.local", "172.32.0.1"] {
            assert!(parse_link(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_leader_is_the_biggest_and_members_are_added_until_it_fits() {
        let c = [cand("a", 24, "10.0.0.1", Some(100.0)), cand("b", 96, "10.0.0.2", Some(100.0)), cand("c", 48, "10.0.0.3", Some(100.0))];
        let g = plan_group(&policy(None), 100 * GIB, &c).unwrap();
        assert_eq!(g.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), ["b", "c"]);
        // A named leader leads even when smaller.
        let g = plan_group(&policy(Some("a")), 100 * GIB, &c).unwrap();
        assert_eq!(g.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), ["a", "b"]);
        // It fits on the leader alone: no group needed beyond it.
        assert_eq!(plan_group(&policy(None), 90 * GIB, &c).unwrap().len(), 1);
        assert_eq!(plan_group(&policy(Some("zz")), 1, &c).unwrap_err(), PlanError::UnknownLeader("zz".into()));
    }

    #[test]
    fn refuses_slow_unknown_public_and_too_small() {
        let slow = [cand("a", 96, "10.0.0.1", None), cand("b", 24, "10.0.0.2", Some(1.0))];
        assert!(matches!(plan_group(&policy(None), 100 * GIB, &slow), Err(PlanError::SlowLink(n, ..)) if n == "b"));
        let unknown = [cand("a", 96, "10.0.0.1", None), cand("b", 24, "10.0.0.2", None)];
        assert_eq!(plan_group(&policy(None), 100 * GIB, &unknown).unwrap_err(), PlanError::UnknownSpeed("b".into()));
        let public = [cand("a", 96, "10.0.0.1", None), cand("b", 24, "203.0.113.9", Some(100.0))];
        assert!(matches!(plan_group(&policy(None), 100 * GIB, &public), Err(PlanError::BadLink(n, _)) if n == "b"));
        let small = [cand("a", 8, "127.0.0.1", None), cand("b", 8, "127.0.0.1", None)];
        assert!(matches!(plan_group(&policy(None), 100 * GIB, &small), Err(PlanError::TooSmall { .. })));
        // Loopback needs no speed (tests, one box); a draining or non-group node is not used.
        let mut c = [cand("a", 8, "127.0.0.1", None), cand("b", 8, "127.0.0.1", None), cand("d", 99, "127.0.0.1", None)];
        c[2].draining = true;
        assert_eq!(plan_group(&policy(None), 12 * GIB, &c).unwrap().len(), 2);
        c[0].group_jobs = false;
        assert!(matches!(plan_group(&policy(None), 12 * GIB, &c), Err(PlanError::TooSmall { .. })));
    }

    #[test]
    fn each_member_gets_its_own_link_and_the_leader_the_endpoints() {
        let c = [cand("lead", 96, "10.0.0.1", Some(100.0)), cand("w1", 48, "10.0.0.2", Some(100.0))];
        let p = policy(None);
        let g = plan_group(&p, 100 * GIB, &c).unwrap();
        let plan = plan_for(&p, &g, &g[1]);
        assert_eq!(
            (plan.group.as_str(), plan.leader.as_str(), plan.worker.as_deref(), plan.link.as_deref()),
            ("split-big", "lead", Some("rpc-server"), Some("10.0.0.2"))
        );
        assert_eq!(plan.extra["link.w1"], "10.0.0.2");
        let split = SplitModel {
            id: "big".into(),
            file: None,
            serve_port: 18080,
            context: Some(4096),
            slots: Some(2),
            gpu_layers: Some(99),
            args: vec!["--tensor-split".into(), "3,1".into()],
        };
        let a = leader_args(&split, Path::new("/m/big.gguf"), &["10.0.0.2:50052".into(), "10.0.0.3:50052".into()]);
        let at = |f: &str| a.iter().position(|x| x == f).map(|i| a[i + 1].clone());
        assert_eq!(
            (at("--host").as_deref(), at("--rpc").as_deref(), at("--ctx-size").as_deref()),
            (Some("127.0.0.1"), Some("10.0.0.2:50052,10.0.0.3:50052"), Some("4096"))
        );
        assert!(a.ends_with(&["--tensor-split".to_string(), "3,1".to_string()]));
    }

    #[tokio::test]
    async fn a_worker_binds_only_its_own_private_link() {
        let mk = |link: Option<&str>| {
            RpcWorkers::new(WorkerConfig { rpc_server: Some("/bin/true".into()), private_link: link.map(Into::into), ..Default::default() })
        };
        let plan = |link: &str| {
            let mut p = GroupPlan::new("split-x", "lead")
                .with_members(vec!["lead".into(), "w".into()])
                .with_model("x")
                .with_link(link)
                .with_worker(WORKER_KIND);
            p.extra.insert("port.w".into(), "50999".into());
            p
        };
        assert!(matches!(mk(None).up("w", &plan("127.0.0.1")).await, Err(RemoteError::Refused { .. })), "no private_link configured");
        assert!(matches!(mk(Some("8.8.8.8")).up("w", &plan("8.8.8.8")).await, Err(RemoteError::Refused { .. })), "public link");
        assert!(
            matches!(mk(Some("127.0.0.1")).up("w", &plan("10.9.9.9")).await, Err(RemoteError::Refused { .. })),
            "the plan asks for another address"
        );
        assert!(matches!(mk(Some("127.0.0.1")).up("stranger", &plan("127.0.0.1")).await, Err(RemoteError::Refused { .. })), "not a member");
        let mut other = plan("127.0.0.1");
        other.worker = Some("ray".into());
        assert!(matches!(mk(Some("127.0.0.1")).up("w", &other).await, Err(RemoteError::Unsupported { .. })), "ray is not built");
    }

    #[test]
    fn the_leader_key_is_kept_private_and_reused() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("split.key");
        let a = load_or_create_key(&p).unwrap();
        assert_eq!(a.len(), 48);
        assert_eq!(load_or_create_key(&p).unwrap(), a);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&p).unwrap().permissions().mode() & 0o777, 0o600);
        }
    }
}
