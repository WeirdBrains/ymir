//! Ymir: the inference-node coordinator, role wiring, config and setup.
//!
//! One `ymird` runs any mix of the three roles. In M1 they all run in one process and talk through
//! the in-process fast path: the app role (the loopback bridge) calls the inference role through
//! [`ymir_infer::Pool`], and the inference role calls the model role through [`ModelNode`]. In M3
//! the same calls go over Weirdnet.

pub mod client;
pub mod config;
pub mod control;
pub mod coordinator;
pub mod daemon;
pub mod health;
pub mod membership;
pub mod model_node;
pub mod paths;
pub mod remote;
pub mod service;
pub mod setup;
pub mod token;
pub mod wire;

pub use config::NodeConfig;
pub use coordinator::Coordinator;
pub use daemon::Daemon;
pub use model_node::{LocalModelNode, ModelNode};
pub use paths::Paths;
