//! Overlap blend: lay the far edge over the near edge and join them.

use rayon::prelude::*;

use super::OverlapParams;
use crate::error::Result;
use crate::filter::{Edge, resize_bilinear};
use crate::image::Image;

/// Fraction of the overlap strip the transition may span at full falloff.
/// Letting it span the whole strip leaves a wide band near a 50/50 mix, which
/// visibly double-exposes strong non-repeating features.
const MAX_FADE_FRACTION: f32 = 0.5;

fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// Half the width of the cross-fade, as a fraction of the overlap strip.
pub fn fade_half_width(falloff: f32) -> f32 {
    0.5 * falloff.clamp(0.0, 1.0) * MAX_FADE_FRACTION
}

/// How many pixels of a `side`-pixel axis are overlapped for `fraction`.
pub fn overlap_extent(side: usize, fraction: f32) -> usize {
    (side as f32 * fraction.clamp(0.0, 0.5)) as usize
}

/// Cross-fade weights across an overlap strip of `len` pixels.
///
/// `1.0` takes the opposite edge and `0.0` keeps the original content. The
/// strip starts fully at `1.0` (that run of copied pixels is what makes the
/// tile wrap) and switches to `0.0` around its middle. `falloff` widens the
/// switch: 0 is a hard step, 1 eases across the widest permitted span.
pub fn seam_fade_ramp(len: usize, falloff: f32) -> Vec<f32> {
    let half_width = fade_half_width(falloff);
    (0..len)
        .map(|i| {
            let position = (i as f32 + 0.5) / len as f32;
            if half_width <= 1e-6 {
                return if position < 0.5 { 1.0 } else { 0.0 };
            }
            let linear = ((0.5 + half_width - position) / (2.0 * half_width)).clamp(0.0, 1.0);
            // Smoothstep twice: steeper through the middle, so the near-50/50
            // zone stays narrow while both ends still ease without a hard line.
            smoothstep(smoothstep(linear))
        })
        .collect()
}

/// Makes the image tile by overlapping opposite edges.
///
/// The far edge is blended over the near edge, the consumed strip is cropped
/// away and the tile is resized back to the source size. The overlap amounts
/// always change the image; `falloff` only sets how soft the join is.
pub fn overlap_blend(image: &Image, params: &OverlapParams) -> Result<Image> {
    let (width, height) = (image.width(), image.height());
    let blend_width = overlap_extent(width, params.x);
    let narrowed = blend_columns(image, blend_width, params.falloff)?;
    let blend_height = overlap_extent(height, params.y);
    let tile = blend_rows(&narrowed, blend_height, params.falloff)?;

    // The cropped tile is exactly seamless, so resample across the wrap:
    // clamped sampling would leave a faint line along the tile boundary.
    let mut result = resize_bilinear(&tile, width, height, Edge::Wrap)?;
    result.clamp_unit();
    Ok(result)
}

/// Blends the right `blend` columns over the left ones and drops them.
fn blend_columns(image: &Image, blend: usize, falloff: f32) -> Result<Image> {
    if blend == 0 {
        return Ok(image.clone());
    }
    let (width, channels) = (image.width(), image.channels());
    let kept = width - blend;
    let ramp = seam_fade_ramp(blend, falloff);
    let mut data = vec![0.0f32; kept * channels * image.height()];
    data.par_chunks_mut(kept * channels).zip(image.data().par_chunks(image.row_len())).for_each(
        |(dst, src)| {
            dst.copy_from_slice(src.get(..kept * channels).unwrap_or(&[]));
            let far = src.get(kept * channels..).unwrap_or(&[]);
            for ((near, far), weight) in
                dst.chunks_mut(channels).zip(far.chunks(channels)).zip(&ramp)
            {
                mix_toward(near, far, *weight);
            }
        },
    );
    Image::new(kept, image.height(), channels, data)
}

/// Blends the bottom `blend` rows over the top ones and drops them.
fn blend_rows(image: &Image, blend: usize, falloff: f32) -> Result<Image> {
    if blend == 0 {
        return Ok(image.clone());
    }
    let (height, row_len) = (image.height(), image.row_len());
    let kept = height - blend;
    let ramp = seam_fade_ramp(blend, falloff);
    let mut data = image.data().get(..kept * row_len).unwrap_or(&[]).to_vec();
    data.par_chunks_mut(row_len)
        .zip(ramp.par_iter())
        .enumerate()
        .for_each(|(y, (near, weight))| mix_toward(near, image.row(kept + y), *weight));
    Image::new(image.width(), kept, image.channels(), data)
}

fn mix_toward(near: &mut [f32], far: &[f32], weight: f32) {
    for (n, f) in near.iter_mut().zip(far) {
        *n = *n * (1.0 - weight) + f * weight;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seamless::test_support::{sample, wrap_discontinuity};

    fn params(x: f32, y: f32, falloff: f32) -> OverlapParams {
        OverlapParams { x, y, falloff }
    }

    #[test]
    fn ramp_starts_as_a_full_copy_and_ends_untouched() {
        let ramp = seam_fade_ramp(64, 0.5);

        assert_eq!(ramp[0], 1.0);
        assert_eq!(ramp[63], 0.0);
        assert!(ramp.windows(2).all(|w| w[0] >= w[1]), "ramp must not increase");
    }

    #[test]
    fn zero_falloff_is_a_hard_step_at_the_midpoint() {
        let ramp = seam_fade_ramp(10, 0.0);

        assert_eq!(ramp, vec![1., 1., 1., 1., 1., 0., 0., 0., 0., 0.]);
    }

    #[test]
    fn higher_falloff_widens_the_transition() {
        let partial = |falloff: f32| {
            seam_fade_ramp(200, falloff).iter().filter(|w| **w > 0.0 && **w < 1.0).count()
        };

        assert!(partial(1.0) > partial(0.3));
        assert!(partial(0.3) > partial(0.0));
    }

    #[test]
    fn near_even_mix_zone_stays_narrow_at_full_falloff() {
        let ramp = seam_fade_ramp(200, 1.0);

        let mixed = ramp.iter().filter(|w| (0.35..=0.65).contains(*w)).count();

        assert!(mixed < 30, "{mixed} of 200 samples are near a 50/50 mix");
    }

    #[test]
    fn zero_overlap_leaves_the_image_unchanged() {
        let source = sample(96, 96, 3);

        let result = overlap_blend(&source, &params(0.0, 0.0, 0.5)).unwrap();

        assert_eq!(result, source);
    }

    #[test]
    fn overlap_changes_the_image_even_with_a_hard_join() {
        let source = sample(96, 96, 3);

        let result = overlap_blend(&source, &params(0.2, 0.2, 0.0)).unwrap();

        assert_ne!(result, source);
        assert_eq!((result.width(), result.height()), (96, 96));
    }

    #[test]
    fn result_tiles_far_better_than_the_source() {
        let source = sample(128, 128, 3);

        let result = overlap_blend(&source, &params(0.2, 0.2, 0.4)).unwrap();

        assert!(wrap_discontinuity(&result) < wrap_discontinuity(&source) / 5.0);
    }

    #[test]
    fn overlap_on_one_axis_only_is_supported() {
        let source = sample(96, 80, 1);

        let result = overlap_blend(&source, &params(0.3, 0.0, 0.2)).unwrap();

        assert_eq!((result.width(), result.height(), result.channels()), (96, 80, 1));
    }
}
