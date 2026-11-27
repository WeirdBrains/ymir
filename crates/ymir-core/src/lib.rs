//! Ymir: the inference-node coordinator, role wiring, config and setup.
//!
//! One `ymird` runs any mix of the three roles. In M1 they all run in one process and talk through
//! the in-process fast path: the app role (the loopback bridge) calls the inference role through
//! `weirdnet-agent`'s pool over memory channels ([`host::local_pool`]), and the inference role
//! calls the model role through [`ModelNode`]. Since M3 the same calls also go over Weirdnet
//! ([`remote`]).

pub mod adapters;
pub mod autoupdate;
pub mod balance;
pub mod batch;
pub mod blobs_net;
pub mod client;
pub mod config;
pub mod config_edit;
pub mod context;
pub mod control;
pub mod convexport;
pub mod coordinator;
pub mod ctxplan;
pub mod daemon;
pub mod dataset;
pub mod direct;
pub mod engine;
pub mod fair;
pub mod group;
pub mod guests;
pub mod host;
pub mod kvfit;
pub mod mcp_servers;
pub mod membership;
pub mod mlx;
pub mod model_node;
pub mod models;
pub mod paths;
pub mod placement;
pub mod ray;
pub mod remote;
pub mod rollout;
pub mod rollout_edit;
pub mod rollouts_net;
pub mod seed;
pub mod semantic;
pub mod service;
pub mod setup;
pub mod tasks;
pub mod token;
pub mod train;
pub mod train_board;
pub mod train_job;
pub mod train_jobs_ctl;
pub mod train_net;
pub mod update;
pub mod update_publish;
pub mod upload;
pub mod upload_net;
pub mod vision;
pub mod vllm;
pub mod vllm_install;
pub mod wake;

pub use config::NodeConfig;
pub use coordinator::Coordinator;
pub use daemon::Daemon;
pub use model_node::{LocalModelNode, ModelNode};
pub use paths::Paths;
