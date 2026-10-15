//! SEAMS processing library.
//!
//! Everything here is pure computation on [`Image`] buffers: no file access, no
//! UI, no global state. Pixels are `f32` in `[0, 1]`, interleaved, in RGB(A)
//! order; conversion to and from file formats lives in `seams-io`.
//!
//! - [`seamless`]: the four tiling methods (offset cross-fade, mirror, overlap, splat)
//! - [`delight`]: removal of baked lighting from a photographed base colour
//! - [`crop`]: cropping to a frame that may be turned
//! - [`perspective`]: straightening a four-cornered region into a rectangle
//! - [`pbr`]: normal, roughness, ambient occlusion, displacement and opacity maps
//! - [`cache`]: a bounded LRU used by callers to reuse results

pub mod cache;
pub mod color;
pub mod crop;
pub mod delight;
pub mod error;
pub mod filter;
pub mod image;
pub mod key;
pub mod pbr;
pub mod perspective;
pub mod retouch;
pub mod rng;
pub mod seamless;
pub mod tiling;

pub use error::{CoreError, Result};
pub use image::Image;
