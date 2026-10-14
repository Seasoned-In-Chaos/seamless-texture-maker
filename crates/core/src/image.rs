//! The pixel buffer every processing function works on.

use crate::error::{CoreError, Result};
use crate::key::KeyHasher;

/// Smallest side length the seamless methods accept.
pub const MIN_DIMENSION: usize = 64;
/// Largest side length the app imports.
pub const MAX_DIMENSION: usize = 8192;

const HASH_GRID_STEPS: usize = 16;
const HASH_RANDOM_SAMPLES: usize = 1024;
const HASH_SEED: u64 = 0x005E_A415;

/// An interleaved `f32` image with 1 (gray), 3 (RGB) or 4 (RGBA) channels.
///
/// Values are nominally in `[0, 1]`. The buffer length always equals
/// `width * height * channels`; the constructors enforce it.
#[derive(Debug, Clone, PartialEq)]
pub struct Image {
    width: usize,
    height: usize,
    channels: usize,
    data: Vec<f32>,
}

impl Image {
    /// Wraps an existing buffer, checking its shape.
    pub fn new(width: usize, height: usize, channels: usize, data: Vec<f32>) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(CoreError::EmptyImage);
        }
        if !matches!(channels, 1 | 3 | 4) {
            return Err(CoreError::UnsupportedChannels(channels));
        }
        let expected = expected_len(width, height, channels)?;
        if data.len() != expected {
            return Err(CoreError::BufferSizeMismatch {
                width,
                height,
                channels,
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { width, height, channels, data })
    }

    /// An image where every value is `value`.
    pub fn filled(width: usize, height: usize, channels: usize, value: f32) -> Result<Self> {
        let len = expected_len(width, height, channels)?;
        Self::new(width, height, channels, vec![value; len])
    }

    /// Builds an image by evaluating `pixel(x, y, channel)` for every value.
    pub fn from_fn(
        width: usize,
        height: usize,
        channels: usize,
        pixel: impl Fn(usize, usize, usize) -> f32,
    ) -> Result<Self> {
        let mut image = Self::filled(width, height, channels, 0.0)?;
        for (i, value) in image.data.iter_mut().enumerate() {
            let c = i % channels;
            let x = (i / channels) % width;
            let y = i / (channels * width);
            *value = pixel(x, y, c);
        }
        Ok(image)
    }

    /// A single-channel image from one plane of values.
    pub fn from_plane(width: usize, height: usize, plane: Vec<f32>) -> Result<Self> {
        Self::new(width, height, 1, plane)
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Number of pixels (`width * height`).
    pub fn pixel_count(&self) -> usize {
        self.width * self.height
    }

    /// Values per row (`width * channels`).
    pub fn row_len(&self) -> usize {
        self.width * self.channels
    }

    /// Size of the pixel buffer in bytes.
    pub fn byte_len(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }

    pub fn has_alpha(&self) -> bool {
        self.channels == 4
    }

    /// Number of colour channels, excluding alpha: 1 or 3.
    pub fn color_channels(&self) -> usize {
        if self.channels == 1 { 1 } else { 3 }
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn into_data(self) -> Vec<f32> {
        self.data
    }

    /// The values of one row.
    pub fn row(&self, y: usize) -> &[f32] {
        let row_len = self.row_len();
        self.data.get(y * row_len..(y + 1) * row_len).unwrap_or(&[])
    }

    /// The values of one pixel, or an empty slice when out of bounds.
    pub fn pixel(&self, x: usize, y: usize) -> &[f32] {
        if x >= self.width || y >= self.height {
            return &[];
        }
        let start = (y * self.width + x) * self.channels;
        self.data.get(start..start + self.channels).unwrap_or(&[])
    }

    /// Copies one channel out as a plane.
    pub fn plane(&self, channel: usize) -> Vec<f32> {
        if channel >= self.channels {
            return vec![0.0; self.pixel_count()];
        }
        self.data.iter().skip(channel).step_by(self.channels).copied().collect()
    }

    /// The alpha plane, when the image has one.
    pub fn alpha_plane(&self) -> Option<Vec<f32>> {
        self.has_alpha().then(|| self.plane(3))
    }

    /// Replaces NaN with 0, infinities with the range ends, and clamps to `[0, 1]`.
    pub fn sanitize(&mut self) {
        for value in &mut self.data {
            *value = sanitize_value(*value);
        }
    }

    /// Clamps every value to `[0, 1]`.
    pub fn clamp_unit(&mut self) {
        for value in &mut self.data {
            *value = value.clamp(0.0, 1.0);
        }
    }

    /// Checks the size limits the seamless methods rely on.
    pub fn validate_dimensions(&self) -> Result<()> {
        let (width, height) = (self.width, self.height);
        if width < MIN_DIMENSION || height < MIN_DIMENSION {
            return Err(CoreError::TooSmall { width, height, min: MIN_DIMENSION });
        }
        if width > MAX_DIMENSION || height > MAX_DIMENSION {
            return Err(CoreError::TooLarge { width, height, max: MAX_DIMENSION });
        }
        Ok(())
    }

    /// A fast content hash for cache keys.
    ///
    /// Samples a regular grid plus a fixed pseudo-random set of pixels: the
    /// grid alone can alias with a regularly patterned texture and collide.
    pub fn fingerprint(&self) -> u64 {
        let mut hasher = KeyHasher::new();
        hasher.usize(self.width).usize(self.height).usize(self.channels);
        let step = (self.height / HASH_GRID_STEPS).max(self.width / HASH_GRID_STEPS).max(1);
        for y in (0..self.height).step_by(step) {
            for x in (0..self.width).step_by(step) {
                hasher.slice(self.pixel(x, y));
            }
        }
        let mut rng = crate::rng::Rng::new(HASH_SEED);
        for _ in 0..HASH_RANDOM_SAMPLES.min(self.pixel_count()) {
            let x = rng.below(self.width);
            let y = rng.below(self.height);
            hasher.slice(self.pixel(x, y));
        }
        hasher.finish()
    }
}

fn expected_len(width: usize, height: usize, channels: usize) -> Result<usize> {
    width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or(CoreError::OutputTooLarge { requested: usize::MAX, limit: usize::MAX / 2 })
}

fn sanitize_value(value: f32) -> f32 {
    if value.is_nan() { 0.0 } else { value.clamp(0.0, 1.0) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_mismatched_buffer() {
        let result = Image::new(4, 4, 3, vec![0.0; 10]);

        assert!(matches!(
            result,
            Err(CoreError::BufferSizeMismatch { expected: 48, actual: 10, .. })
        ));
    }

    #[test]
    fn new_rejects_unsupported_channel_count() {
        assert_eq!(Image::new(2, 2, 2, vec![0.0; 8]), Err(CoreError::UnsupportedChannels(2)));
    }

    #[test]
    fn new_rejects_empty_image() {
        assert_eq!(Image::new(0, 4, 3, Vec::new()), Err(CoreError::EmptyImage));
    }

    #[test]
    fn from_fn_addresses_each_value_once() {
        let image = Image::from_fn(3, 2, 3, |x, y, c| (x + 10 * y + 100 * c) as f32).unwrap();

        assert_eq!(image.pixel(2, 1), &[12.0, 112.0, 212.0]);
        assert_eq!(image.plane(1), vec![100.0, 101.0, 102.0, 110.0, 111.0, 112.0]);
    }

    #[test]
    fn pixel_out_of_bounds_is_empty_not_a_panic() {
        let image = Image::filled(2, 2, 3, 0.5).unwrap();

        assert!(image.pixel(2, 0).is_empty());
        assert!(image.row(5).is_empty());
    }

    #[test]
    fn sanitize_replaces_non_finite_values() {
        let mut image =
            Image::new(4, 1, 1, vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.25]).unwrap();

        image.sanitize();

        assert_eq!(image.data(), &[0.0, 1.0, 0.0, 0.25]);
    }

    #[test]
    fn validate_dimensions_enforces_both_limits() {
        let small = Image::filled(32, 128, 1, 0.0).unwrap();
        let fine = Image::filled(64, 64, 1, 0.0).unwrap();

        assert!(matches!(small.validate_dimensions(), Err(CoreError::TooSmall { .. })));
        assert_eq!(fine.validate_dimensions(), Ok(()));
    }

    #[test]
    fn fingerprint_is_stable_and_content_sensitive() {
        let a = Image::from_fn(128, 128, 3, |x, y, _| ((x * y) % 17) as f32 / 17.0).unwrap();
        let mut b = a.clone();
        b.data_mut()[0] += 0.5;

        assert_eq!(a.fingerprint(), a.clone().fingerprint());
        assert_ne!(a.fingerprint(), b.fingerprint());
    }
}
