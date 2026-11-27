//! Audio helpers for speech in and out (see `docs/design/audio.md`, "Audio processing").
//!
//! Interim crate, to be extracted into `weirdnet-media`: no Ymir types, no network, no logging of
//! audio or text. Everything here is synchronous and bounded; callers on an async runtime run the
//! ffmpeg parts on a blocking thread.

pub mod ffmpeg;
pub mod pcm;
pub mod subtitles;
pub mod text;
pub mod transcript;
pub mod vad;
pub mod wav;

pub use pcm::Pcm;

/// What can go wrong. Messages name the problem, never the audio or the text.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum MediaError {
    /// Not audio this crate can read, or a header that doesn't add up.
    #[error("{0}")]
    Invalid(String),
    /// Readable, but not something this operation does (a format, a rate).
    #[error("{0}")]
    Unsupported(String),
    /// Over a limit: time, size or memory.
    #[error("{0}")]
    Limit(String),
    /// ffmpeg is missing or failed.
    #[error("{0}")]
    Tool(String),
    #[error("{0}")]
    Io(String),
}

impl From<std::io::Error> for MediaError {
    fn from(e: std::io::Error) -> Self {
        MediaError::Io(e.to_string())
    }
}

pub type Result<T, E = MediaError> = std::result::Result<T, E>;
