//! Errors raised while reading or writing image files.

use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum IoError {
    #[error("file not found: {}", .0.display())]
    NotFound(PathBuf),

    #[error("not a file: {}", .0.display())]
    NotAFile(PathBuf),

    #[error("could not read {}: {source}", path.display())]
    Read { path: PathBuf, source: std::io::Error },

    #[error("could not write {}: {source}", path.display())]
    Write { path: PathBuf, source: std::io::Error },

    #[error("could not decode {}: {reason}", path.display())]
    Decode { path: PathBuf, reason: String },

    #[error("could not encode as {format}: {reason}")]
    Encode { format: &'static str, reason: String },

    #[error("unsupported file type `{0}` (use PNG, JPG, TIFF, TGA, EXR or PSD)")]
    UnsupportedFormat(String),

    #[error(transparent)]
    Image(#[from] seams_core::CoreError),
}

pub type Result<T> = std::result::Result<T, IoError>;
