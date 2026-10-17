//! Reading and writing image files for SEAMS.
//!
//! This is the only crate that touches the file system for pixels. It converts
//! between file formats and the `f32` [`seams_core::Image`], validates what
//! comes in, and writes atomically so an interrupted save never leaves a
//! corrupt file behind.

mod error;
mod format;
mod load;
mod path;
mod psd;
mod save;

pub use error::{IoError, Result};
pub use format::{BitDepth, FileFormat, ImageMeta};
pub use load::{Loaded, load};
pub use path::{format_file_size, output_path, sanitize_file_stem};
pub use save::{SaveOptions, save};
