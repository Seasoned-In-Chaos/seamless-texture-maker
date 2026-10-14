//! Splat synthesis (texture bombing).
//!
//! Patches cut from the source are scattered over a jittered grid, each
//! rotated and shaped independently, and combined as a weighted average.
//! Every write wraps at the canvas edges, so the result is exactly periodic.

use std::f32::consts::TAU;

use rayon::prelude::*;

use super::SplatParams;
use crate::error::Result;
use crate::filter::{resize_area, rotate_reflect};
use crate::image::Image;
use crate::rng::Rng;

/// Grid spacing as a fraction of patch size. Below 0.5 every canvas point
/// falls inside several patches, which lets the weighted average blend them
/// and guarantees there are no uncovered pixels.
const SPACING: f32 = 0.40;
/// Jitter as a fraction of a cell. Full-cell jitter hides the grid best but
/// lets neighbours drift far enough apart to leave holes.
const JITTER: f32 = 0.4;
const MIN_PATCH_SIDE: usize = 8;
/// Canvases up to this size are treated as live previews and use fewer
/// rotated variations.
const PREVIEW_MAX_SIDE: usize = 384;
const PREVIEW_VARIATIONS: usize = 12;
const FULL_VARIATIONS: usize = 24;
const MIN_RANDOM_ROTATION: f32 = 0.01;
const MIN_ROTATION_DEG: f32 = 0.05;
/// Offsets the placement and per-variation seeds from the user's seed.
const SEED_STRIDE: u64 = 7919;
/// Angular harmonics (frequency, amplitude) that deform a patch outline.
const WOBBLE_HARMONICS: [(f32, f32); 4] = [(2.0, 1.0), (3.0, 0.6), (5.0, 0.35), (7.0, 0.2)];
const MIN_ALPHA: f32 = 1e-7;
const MIN_WEIGHT: f32 = 1e-9;

fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// Alpha mask for one splat: a soft-edged blob inscribed in the patch.
///
/// `falloff` sets how far the edge feathers inward (0 is a crisp disc).
/// `wobble` deforms the outline with a few angular harmonics at random phases
/// drawn from `rng`, so patches read as irregular shapes, not repeated discs.
pub fn splat_mask(
    width: usize,
    height: usize,
    falloff: f32,
    wobble: f32,
    rng: &mut Rng,
) -> Vec<f32> {
    let phases = WOBBLE_HARMONICS.map(|_| rng.uniform(0.0, TAU));
    let amplitude_sum: f32 = WOBBLE_HARMONICS.iter().map(|(_, amplitude)| amplitude).sum();
    let feather = falloff.max(0.02);
    // Patches are averaged, so a flat-topped mask would weigh every overlapping
    // patch equally and average the detail away. Biasing weight toward the
    // centre lets the nearest patch dominate; falloff sets how sharply.
    let dominance = 8.0 * (1.0 - falloff) + 0.5 * falloff;
    let (half_w, half_h) = ((width as f32 / 2.0).max(1e-6), (height as f32 / 2.0).max(1e-6));

    let mut mask = vec![0.0f32; width * height];
    mask.par_chunks_mut(width.max(1)).enumerate().for_each(|(y, row)| {
        let ny = (y as f32 - (height as f32 - 1.0) / 2.0) / half_h;
        for (x, alpha) in row.iter_mut().enumerate() {
            let nx = (x as f32 - (width as f32 - 1.0) / 2.0) / half_w;
            let mut radius = (nx * nx + ny * ny).sqrt();
            if wobble > 0.001 {
                radius *= 1.0 + outline_deformation(ny.atan2(nx), wobble, &phases, amplitude_sum);
            }
            let depth = (1.0 - radius).clamp(0.0, 1.0);
            *alpha = smoothstep((depth / feather).clamp(0.0, 1.0)) * depth.powf(dominance);
        }
    });
    mask
}

/// Relative change of the outline radius at angle `theta`.
///
/// Deliberately lopsided: the outline may bulge outward freely but only dent
/// inward a little, so a strong wobble cannot shrink a blob far enough to
/// leave uncovered gaps between neighbours.
fn outline_deformation(theta: f32, wobble: f32, phases: &[f32; 4], amplitude_sum: f32) -> f32 {
    let deform: f32 = WOBBLE_HARMONICS
        .iter()
        .zip(phases)
        .map(|((frequency, amplitude), phase)| amplitude * (frequency * theta + phase).sin())
        .sum();
    (wobble * 0.6 * deform / amplitude_sum).clamp(-0.15, 0.5)
}

/// One rotated copy of the source patch with its alpha mask.
struct Patch {
    pixels: Image,
    mask: Vec<f32>,
}

/// Where one patch lands and which variation it uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Placement {
    top: isize,
    left: isize,
    variation: usize,
}

fn patch_side(source_side: usize, scale: f32) -> usize {
    ((source_side as f32 * scale).round() as usize).max(MIN_PATCH_SIDE)
}

fn variation_count(width: usize, height: usize, random_rotation: f32) -> usize {
    if random_rotation < MIN_RANDOM_ROTATION {
        return 1;
    }
    let is_preview = width <= PREVIEW_MAX_SIDE && height <= PREVIEW_MAX_SIDE;
    if is_preview { PREVIEW_VARIATIONS } else { FULL_VARIATIONS }
}

/// Renders the rotated patch and mask for one variation.
fn build_patch(base: &Image, params: &SplatParams, variation: usize) -> Result<Patch> {
    let seed = u64::from(params.seed).wrapping_add(variation as u64 * SEED_STRIDE);
    let mut rng = Rng::new(seed);
    let mut angle = params.rotation_deg;
    if params.random_rotation >= MIN_RANDOM_ROTATION {
        angle += rng.uniform(-180.0, 180.0) * params.random_rotation;
    }
    let pixels =
        if angle.abs() > MIN_ROTATION_DEG { rotate_reflect(base, angle)? } else { base.clone() };
    // The mask is generated unrotated: its own random harmonics supply the
    // shape variation, and not warping it avoids resampling the alpha edge.
    let mask = splat_mask(base.width(), base.height(), params.falloff, params.wobble, &mut rng);
    Ok(Patch { pixels, mask })
}

/// Jittered-grid placements covering a `width x height` canvas.
///
/// Cells partition the canvas exactly and every coordinate is later taken
/// modulo the canvas size, so the layout itself is periodic.
fn placements(
    width: usize,
    height: usize,
    patch_w: usize,
    patch_h: usize,
    variations: usize,
    seed: u32,
) -> Vec<Placement> {
    let cells_x = ((width as f32 / (patch_w as f32 * SPACING).max(1.0)).ceil() as usize).max(1);
    let cells_y = ((height as f32 / (patch_h as f32 * SPACING).max(1.0)).ceil() as usize).max(1);
    let (cell_w, cell_h) = (width as f32 / cells_x as f32, height as f32 / cells_y as f32);
    let mut rng = Rng::new(u64::from(seed).wrapping_add(SEED_STRIDE));

    let mut result = Vec::with_capacity(cells_x * cells_y);
    for gy in 0..cells_y {
        for gx in 0..cells_x {
            let centre_x = (gx as f32 + 0.5) * cell_w + (rng.next_f32() - 0.5) * cell_w * JITTER;
            let centre_y = (gy as f32 + 0.5) * cell_h + (rng.next_f32() - 0.5) * cell_h * JITTER;
            result.push(Placement {
                top: (centre_y - patch_h as f32 / 2.0).round() as isize,
                left: (centre_x - patch_w as f32 / 2.0).round() as isize,
                variation: rng.below(variations),
            });
        }
    }
    result
}

/// The colour and alpha sums that the splats accumulate into.
struct Canvas {
    width: usize,
    height: usize,
    channels: usize,
    color: Vec<f32>,
    weight: Vec<f32>,
}

impl Canvas {
    fn new(width: usize, height: usize, channels: usize) -> Self {
        let pixels = width * height;
        Self {
            width,
            height,
            channels,
            color: vec![0.0; pixels * channels],
            weight: vec![0.0; pixels],
        }
    }

    /// Adds `patch * alpha` for every placement, in parallel and race-free.
    ///
    /// Overlapping patches write the same pixels, so splats cannot be spread
    /// across threads. The canvas is split into disjoint row bands instead:
    /// each band scans every placement but only writes rows it owns.
    fn accumulate(&mut self, patch: &Patch, spots: &[Placement]) {
        let band_rows = self.height.div_ceil(rayon::current_num_threads().max(1) * 4).max(1);
        let (width, height, channels) = (self.width, self.height, self.channels);
        self.color
            .par_chunks_mut(band_rows * width * channels)
            .zip(self.weight.par_chunks_mut(band_rows * width))
            .enumerate()
            .for_each(|(band, (color, weight))| {
                let first_row = band * band_rows;
                let band = Band { first_row, rows: weight.len() / width, width, height, channels };
                for spot in spots {
                    band.add_patch(color, weight, patch, *spot);
                }
            });
    }

    /// Divides colour by alpha. Pixels no patch reached fall back to the
    /// source so the result never has holes.
    fn resolve(mut self, fallback: &Image) -> Result<Image> {
        let channels = self.channels;
        self.color
            .par_chunks_mut(channels)
            .zip(self.weight.par_iter())
            .zip(fallback.data().par_chunks(channels))
            .for_each(|((pixel, weight), source)| {
                if *weight > MIN_WEIGHT {
                    pixel.iter_mut().for_each(|v| *v = (*v / weight).clamp(0.0, 1.0));
                } else {
                    pixel.copy_from_slice(source);
                }
            });
        Image::new(self.width, self.height, channels, self.color)
    }
}

/// A horizontal slice of the canvas owned by one thread.
struct Band {
    first_row: usize,
    rows: usize,
    width: usize,
    height: usize,
    channels: usize,
}

impl Band {
    fn add_patch(&self, color: &mut [f32], weight: &mut [f32], patch: &Patch, spot: Placement) {
        let patch_w = patch.pixels.width();
        for (patch_y, mask_row) in patch.mask.chunks(patch_w).enumerate() {
            let y = (spot.top + patch_y as isize).rem_euclid(self.height as isize) as usize;
            if y < self.first_row || y >= self.first_row + self.rows {
                continue;
            }
            let local = y - self.first_row;
            let row_len = self.width * self.channels;
            let (Some(color_row), Some(weight_row)) = (
                color.get_mut(local * row_len..(local + 1) * row_len),
                weight.get_mut(local * self.width..(local + 1) * self.width),
            ) else {
                continue;
            };
            self.add_row(color_row, weight_row, patch.pixels.row(patch_y), mask_row, spot.left);
        }
    }

    fn add_row(
        &self,
        color: &mut [f32],
        weight: &mut [f32],
        pixels: &[f32],
        mask: &[f32],
        left: isize,
    ) {
        let channels = self.channels;
        let mut x = left.rem_euclid(self.width as isize) as usize;
        for (alpha, source) in mask.iter().zip(pixels.chunks(channels)) {
            // Centre-biased masks make legitimate weights very small away from
            // the patch centre, so only exact zeros are skipped: a coarser
            // cutoff would punch holes in the coverage.
            if *alpha > MIN_ALPHA
                && let (Some(w), Some(target)) =
                    (weight.get_mut(x), color.get_mut(x * channels..(x + 1) * channels))
            {
                *w += alpha;
                target.iter_mut().zip(source).for_each(|(t, s)| *t += s * alpha);
            }
            x = if x + 1 == self.width { 0 } else { x + 1 };
        }
    }
}

/// Makes the image tile by scattering rotated patches of itself.
///
/// Variations are rendered and accumulated one at a time, so memory use does
/// not grow with the number of rotations even for very large patches.
pub fn splat(image: &Image, params: &SplatParams) -> Result<Image> {
    let (width, height) = (image.width(), image.height());
    let patch_w = patch_side(width, params.scale);
    let patch_h = patch_side(height, params.scale);
    let base = resize_area(image, patch_w, patch_h)?;
    let variations = variation_count(width, height, params.random_rotation);
    let spots = placements(width, height, patch_w, patch_h, variations, params.seed);

    let mut canvas = Canvas::new(width, height, image.channels());
    for variation in 0..variations {
        let selected: Vec<Placement> =
            spots.iter().copied().filter(|spot| spot.variation == variation).collect();
        if selected.is_empty() {
            continue;
        }
        let patch = build_patch(&base, params, variation)?;
        canvas.accumulate(&patch, &selected);
    }
    canvas.resolve(image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seamless::test_support::{sample, wrap_discontinuity};

    fn contrast(image: &Image) -> f32 {
        let mean = image.data().iter().sum::<f32>() / image.data().len() as f32;
        (image.data().iter().map(|v| (v - mean).powi(2)).sum::<f32>() / image.data().len() as f32)
            .sqrt()
    }

    fn checker(size: usize) -> Image {
        Image::from_fn(size, size, 3, |x, y, _| if (x / 8 + y / 8) % 2 == 0 { 0.9 } else { 0.1 })
            .unwrap()
    }

    #[test]
    fn mask_is_bounded_peaks_at_centre_and_vanishes_at_corners() {
        let mask = splat_mask(65, 65, 0.3, 0.0, &mut Rng::new(1));

        assert!(mask.iter().all(|a| (0.0..=1.0).contains(a)));
        assert!(mask[32 * 65 + 32] > 0.9);
        assert_eq!(mask[0], 0.0);
        assert_eq!(mask[64 * 65 + 64], 0.0);
    }

    #[test]
    fn wobble_deforms_the_outline() {
        let round = splat_mask(64, 64, 0.3, 0.0, &mut Rng::new(5));
        let wobbly = splat_mask(64, 64, 0.3, 0.9, &mut Rng::new(5));

        assert_ne!(round, wobbly);
    }

    #[test]
    fn low_falloff_concentrates_weight_at_the_centre() {
        let total = |falloff: f32| -> f32 {
            splat_mask(64, 64, falloff, 0.0, &mut Rng::new(1)).iter().sum()
        };

        assert!(total(0.05) < total(0.9));
    }

    #[test]
    fn placements_use_valid_variations_and_depend_on_the_seed() {
        let a = placements(256, 256, 64, 64, 12, 1);
        let b = placements(256, 256, 64, 64, 12, 2);

        assert!(a.iter().all(|spot| spot.variation < 12));
        assert_eq!(a.len(), b.len());
        assert_ne!(a, b);
    }

    #[test]
    fn output_is_deterministic_for_a_given_seed() {
        let source = sample(96, 96, 3);
        let params = SplatParams::default();

        assert_eq!(splat(&source, &params).unwrap(), splat(&source, &params).unwrap());
    }

    #[test]
    fn output_tiles_seamlessly() {
        let source = sample(128, 128, 3);

        let result = splat(&source, &SplatParams::default()).unwrap();

        assert!(wrap_discontinuity(&result) < 4.0);
    }

    #[test]
    fn every_control_changes_the_output() {
        let source = sample(96, 96, 3);
        let base = SplatParams::default();
        let reference = splat(&source, &base).unwrap();
        let variants = [
            SplatParams { scale: 0.8, ..base },
            SplatParams { rotation_deg: 90.0, random_rotation: 0.0, ..base },
            SplatParams { random_rotation: 0.3, ..base },
            SplatParams { wobble: 0.95, ..base },
            SplatParams { falloff: 0.1, ..base },
            SplatParams { seed: 99, ..base },
        ];

        for variant in variants {
            assert_ne!(splat(&source, &variant).unwrap(), reference, "{variant:?}");
        }
    }

    #[test]
    fn extreme_settings_leave_no_holes() {
        let source = Image::filled(96, 96, 3, 0.5).unwrap();
        let settings = [(0.1, 1.0, 0.02), (1.5, 1.0, 1.0), (0.25, 0.0, 0.02), (0.6, 1.0, 0.5)];

        for (scale, wobble, falloff) in settings {
            let params = SplatParams { scale, wobble, falloff, ..SplatParams::default() };
            let result = splat(&source, &params).unwrap();

            assert!(
                result.data().iter().all(|v| (v - 0.5).abs() < 1e-3),
                "scale {scale} wobble {wobble} falloff {falloff} left a hole"
            );
        }
    }

    #[test]
    fn contrast_survives_the_weighted_average() {
        let source = checker(128);
        let params = SplatParams { random_rotation: 0.0, wobble: 0.0, ..SplatParams::default() };

        let result = splat(&source, &params).unwrap();

        assert!(contrast(&result) > contrast(&source) * 0.4);
    }

    #[test]
    fn lower_falloff_keeps_more_detail() {
        let source = checker(128);
        let detail = |falloff: f32| {
            let params = SplatParams {
                falloff,
                random_rotation: 0.0,
                wobble: 0.0,
                ..SplatParams::default()
            };
            contrast(&splat(&source, &params).unwrap())
        };

        assert!(detail(0.1) > detail(0.9));
    }

    #[test]
    fn accumulation_wraps_patches_across_the_canvas_edge() {
        let mut canvas = Canvas::new(8, 8, 1);
        let patch = Patch { pixels: Image::filled(4, 4, 1, 1.0).unwrap(), mask: vec![1.0; 16] };

        canvas.accumulate(&patch, &[Placement { top: 6, left: 6, variation: 0 }]);

        assert_eq!(canvas.weight.iter().sum::<f32>(), 16.0);
        assert_eq!(canvas.weight[0], 1.0, "wrapped corner");
        assert_eq!(canvas.weight[7 * 8 + 7], 1.0);
        assert_eq!(canvas.weight[3 * 8 + 3], 0.0);
    }
}
