//! Ymir: the inference-node coordinator, role wiring, config and setup.
//!
//! One `ymird` runs any mix of the three roles. In M1 they all run in one process and talk through
//! the in-process fast path: the app role (the loopback bridge) calls the inference role through
//! `weirdnet-agent`'s pool over memory channels ([`host::local_pool`]), and the inference role
//! calls the model role through [`ModelNode`]. Since M3 the same calls also go over Weirdnet
//! ([`remote`]).

pub mod balance;
pub mod batch;
pub mod client;
pub mod config;
pub mod context;
pub mod control;
pub mod coordinator;
pub mod daemon;
pub mod dataset;
pub mod direct;
pub mod engine;
pub mod fair;
pub mod group;
pub mod host;
pub mod mcp_servers;
pub mod membership;
pub mod mlx;
pub mod model_node;
pub mod models;
pub mod paths;
pub mod remote;
pub mod service;
pub mod setup;
pub mod tasks;
pub mod token;
pub mod update;
pub mod vision;

pub use config::NodeConfig;
pub use coordinator::Coordinator;
pub use daemon::Daemon;
pub use model_node::{LocalModelNode, ModelNode};
pub use paths::Paths;
