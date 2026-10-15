//! The four methods that turn a texture into one that tiles without seams.
//!
//! Every method works on the whole image in one pass. Each depends on a global
//! relationship (opposite edges, one centre seam, one periodic patch layout),
//! so processing independent crops would leave visible rectangular seams.

mod mirror;
mod offset;
mod overlap;
mod splat;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::image::Image;
use crate::key::KeyHasher;

pub use mirror::mirror_tile;
pub use offset::{feather_radius, offset_crossfade, roll};
pub use overlap::{fade_half_width, overlap_blend, overlap_extent, seam_fade_ramp};
pub use splat::{splat, splat_mask};

/// Which tiling method to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Method {
    /// Shift by half the image and cross-fade the seams now at the centre.
    #[default]
    OffsetCrossfade,
    /// Build a 2x2 tile from reflections; the result is twice as large.
    Mirror,
    /// Overlap opposite edges, blend the join and resize back.
    Overlap,
    /// Scatter rotated patches over a periodic layout (texture bombing).
    Splat,
    /// Leave the image as it is: it already tiles, for example a result
    /// made the base color to retouch its seams.
    None,
}

impl Method {
    pub const ALL: [Method; 5] =
        [Method::OffsetCrossfade, Method::Mirror, Method::Overlap, Method::Splat, Method::None];

    pub fn label(self) -> &'static str {
        match self {
            Method::OffsetCrossfade => "Offset cross-fade",
            Method::Mirror => "Mirror tiling (2x2)",
            Method::Overlap => "Overlap blend",
            Method::Splat => "Splat synthesis",
            Method::None => "None (already tiles)",
        }
    }

    /// One sentence on what the method does, for the inspector.
    pub fn description(self) -> &'static str {
        match self {
            Method::OffsetCrossfade => {
                "Shifts the image by half and cross-fades the seams at the centre."
            }
            Method::Mirror => "Builds a 2x2 reflected tile. The result is twice the source size.",
            Method::Overlap => {
                "Overlaps opposite edges and blends the join. Suits structured materials."
            }
            Method::Splat => "Scatters rotated patches. Suits irregular, organic materials.",
            Method::None => {
                "Leaves the image as it is. For a texture that already tiles, such as a \
                 result made the base color to retouch."
            }
        }
    }
}

/// Settings for [`Method::Overlap`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OverlapParams {
    /// Fraction of the width overlapped, `0..=0.5`.
    pub x: f32,
    /// Fraction of the height overlapped, `0..=0.5`.
    pub y: f32,
    /// Softness of the join, `0..=1`; 0 is a hard cut.
    pub falloff: f32,
}

impl Default for OverlapParams {
    fn default() -> Self {
        Self { x: 0.20, y: 0.20, falloff: 0.10 }
    }
}

/// Settings for [`Method::Splat`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SplatParams {
    /// Patch size as a fraction of the source, `0.1..=1.5`.
    pub scale: f32,
    /// Rotation applied to every patch, in degrees.
    pub rotation_deg: f32,
    /// How far each patch may deviate from `rotation_deg`, `0..=1`.
    pub random_rotation: f32,
    /// How strongly patch outlines are deformed, `0..=1`.
    pub wobble: f32,
    /// How far patch edges feather inward, `0..=1`.
    pub falloff: f32,
    /// Varies the placement and the patch shapes.
    pub seed: u32,
}

impl Default for SplatParams {
    fn default() -> Self {
        Self {
            scale: 0.45,
            rotation_deg: 0.0,
            random_rotation: 1.0,
            wobble: 0.45,
            falloff: 0.55,
            seed: 0,
        }
    }
}

/// Everything the seamless step needs.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SeamlessParams {
    pub method: Method,
    /// Width of the cross-fade for [`Method::OffsetCrossfade`], `0..=1`.
    pub fade_amount: FadeAmount,
    pub overlap: OverlapParams,
    pub splat: SplatParams,
}

/// The cross-fade width; a newtype so the default can differ from `f32`'s.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FadeAmount(pub f32);

impl Default for FadeAmount {
    fn default() -> Self {
        Self(0.5)
    }
}

impl SeamlessParams {
    /// A copy with every value forced into its valid range.
    pub fn sanitized(&self) -> Self {
        let unit = |v: f32| finite_or(v, 0.0).clamp(0.0, 1.0);
        Self {
            method: self.method,
            fade_amount: FadeAmount(unit(self.fade_amount.0)),
            overlap: OverlapParams {
                x: finite_or(self.overlap.x, 0.0).clamp(0.0, 0.5),
                y: finite_or(self.overlap.y, 0.0).clamp(0.0, 0.5),
                falloff: unit(self.overlap.falloff),
            },
            splat: SplatParams {
                scale: finite_or(self.splat.scale, 0.45).clamp(0.1, 1.5),
                rotation_deg: finite_or(self.splat.rotation_deg, 0.0).rem_euclid(360.0),
                random_rotation: unit(self.splat.random_rotation),
                wobble: unit(self.splat.wobble),
                falloff: unit(self.splat.falloff),
                seed: self.splat.seed,
            },
        }
    }

    /// A key covering only the settings the selected method reads, so moving a
    /// slider of another method does not invalidate a cached result.
    pub fn cache_key(&self) -> u64 {
        let p = self.sanitized();
        let mut hasher = KeyHasher::new();
        hasher.u64(p.method as u64);
        match p.method {
            Method::OffsetCrossfade => hasher.f32(p.fade_amount.0),
            Method::Mirror | Method::None => &mut hasher,
            Method::Overlap => hasher.f32(p.overlap.x).f32(p.overlap.y).f32(p.overlap.falloff),
            Method::Splat => hasher
                .f32(p.splat.scale)
                .f32(p.splat.rotation_deg)
                .f32(p.splat.random_rotation)
                .f32(p.splat.wobble)
                .f32(p.splat.falloff)
                .u64(u64::from(p.splat.seed)),
        };
        hasher.finish()
    }
}

fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() { value } else { fallback }
}

/// Runs the selected method and returns the tileable result.
///
/// The image must be within the size limits of [`Image::validate_dimensions`].
/// Out-of-range settings are clamped, never rejected.
pub fn make_seamless(image: &Image, params: &SeamlessParams) -> Result<Image> {
    image.validate_dimensions()?;
    let params = params.sanitized();
    match params.method {
        Method::OffsetCrossfade => offset_crossfade(image, params.fade_amount.0),
        Method::Mirror => mirror_tile(image),
        Method::Overlap => overlap_blend(image, &params.overlap),
        Method::Splat => splat(image, &params.splat),
        Method::None => Ok(image.clone()),
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use crate::image::Image;

    /// A smooth, clearly non-tiling test texture.
    pub fn sample(width: usize, height: usize, channels: usize) -> Image {
        Image::from_fn(width, height, channels, |x, y, c| {
            let (fx, fy) = (x as f32 / width as f32, y as f32 / height as f32);
            let wave = (fx * 7.3 + c as f32).sin() * (fy * 5.1).cos() * 0.2;
            (0.25 + 0.45 * fx + 0.15 * fy + wave).clamp(0.0, 1.0)
        })
        .unwrap()
    }

    /// Mean absolute difference between two columns.
    pub fn column_difference(image: &Image, a: usize, b: usize) -> f32 {
        let total: f32 = (0..image.height())
            .map(|y| pixel_difference(image.pixel(a, y), image.pixel(b, y)))
            .sum();
        total / image.height() as f32
    }

    /// Mean absolute difference between two rows.
    pub fn row_difference(image: &Image, a: usize, b: usize) -> f32 {
        let total: f32 = (0..image.width())
            .map(|x| pixel_difference(image.pixel(x, a), image.pixel(x, b)))
            .sum();
        total / image.width() as f32
    }

    fn pixel_difference(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(p, q)| (p - q).abs()).sum::<f32>() / a.len().max(1) as f32
    }

    /// How much larger the jump across each wrap is than a typical step
    /// between neighbouring pixels. Near 1 means the image tiles cleanly.
    pub fn wrap_discontinuity(image: &Image) -> f32 {
        let (w, h) = (image.width(), image.height());
        let interior = (column_difference(image, w / 3, w / 3 + 1)
            + row_difference(image, h / 3, h / 3 + 1))
            / 2.0;
        let wrap = (column_difference(image, 0, w - 1) + row_difference(image, 0, h - 1)) / 2.0;
        wrap / interior.max(1e-4)
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{sample, wrap_discontinuity};
    use super::*;
    use crate::error::CoreError;

    #[test]
    fn source_texture_does_not_tile() {
        assert!(wrap_discontinuity(&sample(128, 128, 3)) > 20.0);
    }

    #[test]
    fn every_method_produces_a_tileable_result() {
        let source = sample(128, 96, 3);

        // None trusts the image to tile already; it is checked below.
        for method in Method::ALL.into_iter().filter(|m| *m != Method::None) {
            let params = SeamlessParams { method, ..SeamlessParams::default() };
            let result = make_seamless(&source, &params).unwrap();

            assert!(wrap_discontinuity(&result) < 6.0, "{method:?} left a visible seam");
            assert!(result.data().iter().all(|v| (0.0..=1.0).contains(v)), "{method:?} range");
        }
    }

    #[test]
    fn none_leaves_the_image_as_it_is() {
        let source = sample(128, 96, 3);
        let params = SeamlessParams { method: Method::None, ..SeamlessParams::default() };

        assert_eq!(make_seamless(&source, &params).unwrap(), source);
    }

    #[test]
    fn only_mirror_changes_the_size() {
        let source = sample(100, 80, 3);

        for method in Method::ALL {
            let params = SeamlessParams { method, ..SeamlessParams::default() };
            let result = make_seamless(&source, &params).unwrap();
            let expected = if method == Method::Mirror { (200, 160) } else { (100, 80) };

            assert_eq!((result.width(), result.height()), expected, "{method:?}");
        }
    }

    #[test]
    fn grayscale_and_alpha_images_keep_their_channels() {
        for channels in [1, 4] {
            let source = sample(96, 96, channels);
            for method in Method::ALL {
                let params = SeamlessParams { method, ..SeamlessParams::default() };

                let result = make_seamless(&source, &params).unwrap();

                assert_eq!(result.channels(), channels, "{method:?}");
            }
        }
    }

    #[test]
    fn undersized_image_is_rejected_not_processed() {
        let tiny = sample(32, 32, 3);

        let result = make_seamless(&tiny, &SeamlessParams::default());

        assert!(matches!(result, Err(CoreError::TooSmall { .. })));
    }

    #[test]
    fn sanitized_clamps_hostile_values() {
        let mut params = SeamlessParams::default();
        params.overlap.x = 9.0;
        params.splat.scale = f32::NAN;
        params.splat.wobble = -4.0;
        params.fade_amount = FadeAmount(f32::INFINITY);

        let clean = params.sanitized();

        assert_eq!(clean.overlap.x, 0.5);
        assert_eq!(clean.splat.scale, 0.45);
        assert_eq!(clean.splat.wobble, 0.0);
        assert_eq!(clean.fade_amount.0, 0.0);
    }

    #[test]
    fn cache_key_ignores_settings_of_other_methods() {
        let base = SeamlessParams { method: Method::Overlap, ..SeamlessParams::default() };
        let mut other_method_changed = base;
        other_method_changed.splat.wobble = 0.9;
        let mut own_setting_changed = base;
        own_setting_changed.overlap.falloff = 0.9;

        assert_eq!(base.cache_key(), other_method_changed.cache_key());
        assert_ne!(base.cache_key(), own_setting_changed.cache_key());
    }
}
