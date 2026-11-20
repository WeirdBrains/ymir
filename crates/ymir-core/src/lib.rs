//! Ymir: the inference-node coordinator, role wiring, config and setup.
//!
//! One `ymird` runs any mix of the three roles. In M1 they all run in one process and talk through
//! the in-process fast path: the app role (the loopback bridge) calls the inference role through
//! `weirdnet-agent`'s pool over memory channels ([`host::local_pool`]), and the inference role
//! calls the model role through [`ModelNode`]. Since M3 the same calls also go over Weirdnet
//! ([`remote`]).

pub mod client;
pub mod config;
pub mod control;
pub mod coordinator;
pub mod daemon;
pub mod host;
pub mod membership;
pub mod model_node;
pub mod paths;
pub mod remote;
pub mod service;
pub mod setup;
pub mod token;

pub use config::NodeConfig;
pub use coordinator::Coordinator;
pub use daemon::Daemon;
pub use model_node::{LocalModelNode, ModelNode};
pub use paths::Paths;
