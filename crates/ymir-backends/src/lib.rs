//! Engine adapters and the local supervisor for a model node.
//!
//! To be extracted into `weirdnet-backends`. Generic: no Ymir coordinator types here.

mod engine;
mod sse;
mod supervisor;

pub use engine::{EngineConfig, EngineKind, ModelLaunch, Wrap, find_in_path};
pub use sse::{ChunkDecoder, SseParser};
pub use supervisor::{ModelStatus, Supervisor, SupervisorEvent, SupervisorOptions};
