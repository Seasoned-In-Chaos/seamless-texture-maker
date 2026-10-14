//! Errors surfaced to the front end.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("open a texture first")]
    NoDocument,

    #[error("the {0} map has not been generated yet")]
    MapNotReady(&'static str),

    #[error("the texture is still being processed; try again in a moment")]
    RenderPending,

    #[error("could not start a background worker: {0}")]
    WorkerStart(#[source] std::io::Error),

    #[error(transparent)]
    Io(#[from] seams_io::IoError),

    #[error(transparent)]
    Image(#[from] seams_core::CoreError),
}

pub type Result<T> = std::result::Result<T, EngineError>;
