//! The error type shared by every processing function.

use thiserror::Error;

/// Why a processing call was rejected.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum CoreError {
    #[error("the image has no pixels")]
    EmptyImage,

    #[error("image is too small: {width}x{height} (minimum is {min}x{min})")]
    TooSmall { width: usize, height: usize, min: usize },

    #[error("image is too large: {width}x{height} (maximum is {max}x{max})")]
    TooLarge { width: usize, height: usize, max: usize },

    #[error("unsupported channel count: {0} (expected 1, 3 or 4)")]
    UnsupportedChannels(usize),

    #[error("pixel buffer holds {actual} values but {width}x{height}x{channels} needs {expected}")]
    BufferSizeMismatch {
        width: usize,
        height: usize,
        channels: usize,
        expected: usize,
        actual: usize,
    },

    #[error("invalid value for `{name}`: {reason}")]
    InvalidParameter { name: &'static str, reason: String },

    #[error("the result would need {requested} pixels, more than the {limit} pixel limit")]
    OutputTooLarge { requested: usize, limit: usize },
}

pub type Result<T> = std::result::Result<T, CoreError>;
