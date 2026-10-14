//! Delighting: remove baked lighting from a photographed base colour.
//!
//! The work happens on the lightness plane of Lab, on the `[0, 255]` scale the
//! tuning constants below were chosen for. Chroma is only touched by the
//! flatness and colour preservation controls.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::color::{Lab, lab_to_rgb, rgb_to_lab};
use crate::error::Result;
use crate::filter::{Edge, gaussian_blur_plane, resize_area, resize_bilinear};
use crate::image::Image;
use crate::key::KeyHasher;

/// A control below this is treated as off.
const ACTIVE_THRESHOLD: f32 = 0.01;
const L_MAX: f32 = 255.0;
/// Above this sigma a blur is computed on a downsampled copy.
const DIRECT_BLUR_MAX_SIGMA: f32 = 24.0;
const LOWFREQ_SIGMA_PER_STEP: f32 = 12.0;
const LOWFREQ_MIN_SIDE: usize = 8;
/// Neutral `a`/`b` in scaled Lab.
const CHROMA_NEUTRAL: f32 = 128.0;
/// Bounds on how far colour preservation rescales chroma.
const MIN_CHROMA_SCALE: f32 = 0.5;
const MAX_CHROMA_SCALE: f32 = 2.0;

/// Delighting controls, each in `0..=1` where 0 is off.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct DelightParams {
    /// Lift dark shadow regions.
    pub shadow_removal: f32,
    /// Compress bright highlights toward the mean.
    pub highlight_reduction: f32,
    /// Restore contrast lost to flattening.
    pub contrast_recovery: f32,
    /// Boost fine surface detail.
    pub detail_preservation: f32,
    /// Keep saturation where the other controls changed lightness, so lifted
    /// shadows do not turn grey.
    pub color_preservation: f32,
    /// Pull chroma toward the image's average colour.
    pub flatness: f32,
    /// Lift local darkening in crevices.
    pub ao_removal: f32,
    /// Soften harsh tonal transitions left by processing.
    pub edge_consistency: f32,
}

impl DelightParams {
    /// A gentle touch-up.
    pub const SOFT: Self = Self {
        shadow_removal: 0.20,
        highlight_reduction: 0.15,
        contrast_recovery: 0.10,
        detail_preservation: 0.25,
        color_preservation: 0.70,
        flatness: 0.10,
        ao_removal: 0.15,
        edge_consistency: 0.20,
    };
    /// A middle-of-the-road correction.
    pub const BALANCED: Self = Self {
        shadow_removal: 0.45,
        highlight_reduction: 0.35,
        contrast_recovery: 0.20,
        detail_preservation: 0.35,
        color_preservation: 0.40,
        flatness: 0.25,
        ao_removal: 0.35,
        edge_consistency: 0.25,
    };
    /// A heavy correction for strongly lit photos.
    pub const AGGRESSIVE: Self = Self {
        shadow_removal: 0.75,
        highlight_reduction: 0.60,
        contrast_recovery: 0.35,
        detail_preservation: 0.45,
        color_preservation: 0.15,
        flatness: 0.50,
        ao_removal: 0.65,
        edge_consistency: 0.35,
    };

    fn values(&self) -> [f32; 8] {
        [
            self.shadow_removal,
            self.highlight_reduction,
            self.contrast_recovery,
            self.detail_preservation,
            self.color_preservation,
            self.flatness,
            self.ao_removal,
            self.edge_consistency,
        ]
    }

    /// True when no control is high enough to change the image. Colour
    /// preservation only tempers the other controls, so alone it changes
    /// nothing.
    pub fn is_identity(&self) -> bool {
        [
            self.shadow_removal,
            self.highlight_reduction,
            self.contrast_recovery,
            self.detail_preservation,
            self.flatness,
            self.ao_removal,
            self.edge_consistency,
        ]
        .iter()
        .all(|v| *v <= ACTIVE_THRESHOLD)
    }

    /// A copy with every control forced into `0..=1`.
    pub fn sanitized(&self) -> Self {
        let unit = |v: f32| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 };
        Self {
            shadow_removal: unit(self.shadow_removal),
            highlight_reduction: unit(self.highlight_reduction),
            contrast_recovery: unit(self.contrast_recovery),
            detail_preservation: unit(self.detail_preservation),
            color_preservation: unit(self.color_preservation),
            flatness: unit(self.flatness),
            ao_removal: unit(self.ao_removal),
            edge_consistency: unit(self.edge_consistency),
        }
    }

    pub fn cache_key(&self) -> u64 {
        KeyHasher::new().slice(&self.sanitized().values()).finish()
    }
}

fn is_active(value: f32) -> bool {
    value > ACTIVE_THRESHOLD
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    (values.par_iter().map(|v| f64::from(*v)).sum::<f64>() / values.len() as f64) as f32
}

fn clamp_lightness(plane: &mut [f32]) {
    plane.par_iter_mut().for_each(|v| *v = v.clamp(0.0, L_MAX));
}

/// The three Lab planes of an image, plus its size.
struct LabPlanes {
    width: usize,
    height: usize,
    l: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
}

impl LabPlanes {
    fn from_image(image: &Image) -> Self {
        let channels = image.channels();
        let labs: Vec<Lab> =
            image.data().par_chunks(channels).map(|pixel| rgb_to_lab(pixel_rgb(pixel))).collect();
        Self {
            width: image.width(),
            height: image.height(),
            l: labs.iter().map(|lab| lab.l).collect(),
            a: labs.iter().map(|lab| lab.a).collect(),
            b: labs.iter().map(|lab| lab.b).collect(),
        }
    }

    /// Writes the planes back over `image`, leaving any alpha untouched.
    fn write_to(&self, image: &mut Image) {
        let channels = image.channels();
        image.data_mut().par_chunks_mut(channels).enumerate().for_each(|(i, pixel)| {
            let lab = Lab {
                l: self.l.get(i).copied().unwrap_or(0.0),
                a: self.a.get(i).copied().unwrap_or(128.0),
                b: self.b.get(i).copied().unwrap_or(128.0),
            };
            let rgb = lab_to_rgb(lab);
            let colors = pixel.len().min(3);
            pixel.iter_mut().take(colors).zip(rgb).for_each(|(p, v)| *p = v);
        });
    }

    fn blur(&self, plane: &[f32], sigma: f32) -> Vec<f32> {
        blur_lowfreq(plane, self.width, self.height, sigma)
    }

    fn longest_side(&self) -> f32 {
        self.width.max(self.height) as f32
    }
}

fn pixel_rgb(pixel: &[f32]) -> [f32; 3] {
    match pixel {
        [r, g, b, ..] => [*r, *g, *b],
        [gray, ..] => [*gray; 3],
        [] => [0.0; 3],
    }
}

/// Gaussian blur that stays fast for very large sigma.
///
/// Every large-sigma blur here extracts an inherently low-frequency field
/// (broad lighting, soft shadows), so blurring a downsampled copy and
/// upsampling the result is visually indistinguishable and far cheaper than a
/// full-resolution blur with a kernel thousands of pixels wide.
pub fn blur_lowfreq(plane: &[f32], width: usize, height: usize, sigma: f32) -> Vec<f32> {
    let truncated_radius = |s: f32| (((s * 3.0) as usize).max(3) | 1) / 2;
    if sigma <= DIRECT_BLUR_MAX_SIGMA {
        return gaussian_blur_plane(plane, width, height, sigma, truncated_radius(sigma));
    }
    let downscale = ((sigma / LOWFREQ_SIGMA_PER_STEP).round() as usize).max(1);
    let small_w = (width / downscale).max(LOWFREQ_MIN_SIDE).min(width);
    let small_h = (height / downscale).max(LOWFREQ_MIN_SIDE).min(height);
    let small_sigma = sigma / downscale as f32;
    let upsampled = Image::from_plane(width, height, plane.to_vec())
        .and_then(|full| resize_area(&full, small_w, small_h))
        .and_then(|small| {
            let radius = truncated_radius(small_sigma);
            let blurred = gaussian_blur_plane(small.data(), small_w, small_h, small_sigma, radius);
            Image::from_plane(small_w, small_h, blurred)
        })
        .and_then(|small| resize_bilinear(&small, width, height, Edge::Clamp));
    upsampled.map(Image::into_data).unwrap_or_else(|_| plane.to_vec())
}

/// Divides out broad lighting at two scales and re-centres on the mean.
/// Under-lit areas are lifted by `shadow` and over-lit ones lowered by
/// `highlight`, so each control moves only its own side of the light.
///
/// Every blur in this module is sized from the image, not in pixels, so the
/// small live preview and the full-resolution result look the same.
fn remove_lighting(planes: &mut LabPlanes, shadow: f32, highlight: f32) {
    let side = planes.longest_side();
    let large = planes.blur(&planes.l, (side * 0.15).max(15.0));
    let medium = planes.blur(&planes.l, (side * 0.05).max(5.0));
    let target = mean(&planes.l).max(1.0);
    planes.l.par_iter_mut().zip(large.par_iter().zip(medium.par_iter())).for_each(
        |(l, (large, medium))| {
            let lighting = (large * 0.6 + medium * 0.4).clamp(1.0, L_MAX);
            let flattened = *l / lighting * target;
            // Continuous at lighting == target, where flattening changes nothing.
            let strength = if lighting < target { shadow } else { highlight };
            *l = (*l * (1.0 - strength) + flattened * strength).clamp(0.0, L_MAX);
        },
    );
}

/// Adds back a scaled copy of the fine detail layer.
fn boost_detail(planes: &mut LabPlanes, amount: f32) {
    let smoothed = planes.blur(&planes.l, (planes.longest_side() * 0.02).max(2.0));
    planes.l.par_iter_mut().zip(smoothed.par_iter()).for_each(|(l, smooth)| {
        *l = (*l + (*l - smooth) * amount * 0.6).clamp(0.0, L_MAX);
    });
}

/// Lifts the dark end of the range by up to 40 levels.
fn lift_shadows(planes: &mut LabPlanes, amount: f32) {
    let lift = amount * 40.0;
    planes.l.par_iter_mut().for_each(|l| {
        let darkness = (1.0 - *l / 80.0).clamp(0.0, 1.0);
        *l = (*l + darkness * lift).clamp(0.0, L_MAX);
    });
}

/// Pulls values above 180 back toward the mean.
fn reduce_highlights(planes: &mut LabPlanes, amount: f32) {
    let target = mean(&planes.l);
    let pull = amount * 0.6;
    planes.l.par_iter_mut().for_each(|l| {
        let brightness = ((*l - 180.0) / 75.0).clamp(0.0, 1.0);
        *l = (*l - brightness * (*l - target) * pull).clamp(0.0, L_MAX);
    });
}

/// Lifts pixels that are darker than their neighbourhood.
fn lift_occlusion(planes: &mut LabPlanes, amount: f32) {
    let local_mean = planes.blur(&planes.l, (planes.longest_side() * 0.03).max(3.0));
    let lift = amount * 35.0;
    planes.l.par_iter_mut().zip(local_mean.par_iter()).for_each(|(l, local)| {
        let occlusion = ((local - *l) / 60.0).clamp(0.0, 1.0);
        *l = (*l + occlusion * lift).clamp(0.0, L_MAX);
    });
}

/// Stretches each value away from its broad surroundings: texture contrast
/// comes back without the lighting that was just removed.
fn recover_contrast(planes: &mut LabPlanes, amount: f32) {
    let surroundings = planes.blur(&planes.l, (planes.longest_side() * 0.05).max(5.0));
    let factor = 1.0 + amount * 0.8;
    planes.l.par_iter_mut().zip(surroundings.par_iter()).for_each(|(l, around)| {
        *l = (around + (*l - around) * factor).clamp(0.0, L_MAX);
    });
}

/// Blends in a blurred copy to soften harsh transitions.
fn smooth_edges(planes: &mut LabPlanes, amount: f32) {
    let sigma = (planes.longest_side() * (0.001 + 0.004 * amount)).max(0.5);
    let smooth = planes.blur(&planes.l, sigma);
    let blend = amount * 0.6;
    planes.l.par_iter_mut().zip(smooth.par_iter()).for_each(|(l, s)| {
        *l = *l * (1.0 - blend) + s * blend;
    });
}

/// Pulls one chroma plane toward the image's average colour.
fn flatten_chroma(plane: &mut [f32], flatness: f32) {
    let average = mean(plane);
    plane.par_iter_mut().for_each(|v| *v = *v * (1.0 - flatness) + average * flatness);
}

/// Scales chroma with the change in lightness. Saturation is chroma over
/// lightness, so without this a lifted shadow keeps its chroma and turns
/// grey, and a lowered highlight oversaturates.
fn preserve_saturation(planes: &mut LabPlanes, original_l: &[f32], amount: f32) {
    let LabPlanes { l, a, b, .. } = planes;
    let lightness = l.par_iter().zip(original_l.par_iter());
    a.par_iter_mut().zip(b.par_iter_mut()).zip(lightness).for_each(|((a, b), (new, old))| {
        let ratio = ((new + 1.0) / (old + 1.0)).clamp(MIN_CHROMA_SCALE, MAX_CHROMA_SCALE);
        let scale = 1.0 + (ratio - 1.0) * amount;
        *a = CHROMA_NEUTRAL + (*a - CHROMA_NEUTRAL) * scale;
        *b = CHROMA_NEUTRAL + (*b - CHROMA_NEUTRAL) * scale;
    });
}

fn adjust_lightness(planes: &mut LabPlanes, params: &DelightParams) {
    if is_active(params.shadow_removal) || is_active(params.highlight_reduction) {
        remove_lighting(planes, params.shadow_removal, params.highlight_reduction);
    }
    if is_active(params.detail_preservation) {
        boost_detail(planes, params.detail_preservation);
    }
    if is_active(params.shadow_removal) {
        lift_shadows(planes, params.shadow_removal);
    }
    if is_active(params.highlight_reduction) {
        reduce_highlights(planes, params.highlight_reduction);
    }
    if is_active(params.ao_removal) {
        lift_occlusion(planes, params.ao_removal);
    }
    if is_active(params.contrast_recovery) {
        recover_contrast(planes, params.contrast_recovery);
    }
    if is_active(params.edge_consistency) {
        smooth_edges(planes, params.edge_consistency);
    }
    clamp_lightness(&mut planes.l);
}

/// Removes lighting gradients and shadows from the image.
///
/// Returns an unchanged copy when every control is off. Alpha is preserved.
pub fn delight(image: &Image, params: &DelightParams) -> Result<Image> {
    let params = params.sanitized();
    let mut result = image.clone();
    if params.is_identity() {
        return Ok(result);
    }
    let mut planes = LabPlanes::from_image(image);
    let original_l = is_active(params.color_preservation).then(|| planes.l.clone());
    adjust_lightness(&mut planes, &params);
    if is_active(params.flatness) {
        flatten_chroma(&mut planes.a, params.flatness);
        flatten_chroma(&mut planes.b, params.flatness);
    }
    if let Some(original_l) = original_l {
        preserve_saturation(&mut planes, &original_l, params.color_preservation);
    }
    planes.write_to(&mut result);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mid-grey surface with fine texture under a strong left-to-right light.
    fn lit_surface(size: usize) -> Image {
        Image::from_fn(size, size, 3, |x, y, c| {
            let light = 0.25 + 0.6 * x as f32 / size as f32;
            let grain = if (x / 4 + y / 4) % 2 == 0 { 0.04 } else { -0.04 };
            ((0.5 + grain) * light * 1.6 + c as f32 * 0.02).clamp(0.0, 1.0)
        })
        .unwrap()
    }

    fn column_mean(image: &Image, x_start: usize, x_end: usize) -> f32 {
        let mut total = 0.0;
        let mut count = 0.0;
        for y in 0..image.height() {
            for x in x_start..x_end {
                total += image.pixel(x, y)[0];
                count += 1.0;
            }
        }
        total / count
    }

    fn only(set: impl Fn(&mut DelightParams)) -> DelightParams {
        let mut params = DelightParams::default();
        set(&mut params);
        params
    }

    #[test]
    fn all_controls_off_returns_the_image_unchanged() {
        let image = lit_surface(96);

        assert_eq!(delight(&image, &DelightParams::default()).unwrap(), image);
    }

    #[test]
    fn output_stays_in_range_for_every_preset() {
        let image = lit_surface(128);

        for preset in [DelightParams::SOFT, DelightParams::BALANCED, DelightParams::AGGRESSIVE] {
            let result = delight(&image, &preset).unwrap();

            assert!(result.data().iter().all(|v| (0.0..=1.0).contains(v)));
            assert_eq!((result.width(), result.height(), result.channels()), (128, 128, 3));
        }
    }

    #[test]
    fn lighting_gradient_is_reduced() {
        let image = lit_surface(256);
        let gradient = |img: &Image| (column_mean(img, 200, 240) - column_mean(img, 16, 56)).abs();

        let result = delight(&image, &only(|p| p.shadow_removal = 1.0)).unwrap();

        assert!(gradient(&result) < gradient(&image) * 0.6);
    }

    #[test]
    fn every_control_has_an_effect_on_its_own() {
        let image = lit_surface(128);
        type Setter = fn(&mut DelightParams);
        let setters: [(&str, Setter); 7] = [
            ("shadow_removal", |p| p.shadow_removal = 0.8),
            ("highlight_reduction", |p| p.highlight_reduction = 0.8),
            ("contrast_recovery", |p| p.contrast_recovery = 0.8),
            ("detail_preservation", |p| p.detail_preservation = 0.8),
            ("flatness", |p| p.flatness = 0.8),
            ("ao_removal", |p| p.ao_removal = 0.8),
            ("edge_consistency", |p| p.edge_consistency = 0.8),
        ];

        for (name, set) in setters {
            let result = delight(&image, &only(set)).unwrap();

            assert_ne!(result, image, "{name} did nothing on its own");
        }
    }

    fn gradient(image: &Image) -> f32 {
        (column_mean(image, 200, 240) - column_mean(image, 16, 56)).abs()
    }

    #[test]
    fn shadow_removal_lifts_the_dark_side_and_leaves_the_bright_side() {
        let image = lit_surface(256);

        let result = delight(&image, &only(|p| p.shadow_removal = 1.0)).unwrap();

        assert!(column_mean(&result, 16, 56) > column_mean(&image, 16, 56) + 0.05);
        assert!(column_mean(&result, 200, 240) >= column_mean(&image, 200, 240) - 0.01);
    }

    #[test]
    fn highlight_reduction_lowers_the_bright_side_and_leaves_the_dark_side() {
        let image = lit_surface(256);

        let result = delight(&image, &only(|p| p.highlight_reduction = 1.0)).unwrap();

        assert!(column_mean(&result, 200, 240) < column_mean(&image, 200, 240) - 0.05);
        assert!(column_mean(&result, 16, 56) <= column_mean(&image, 16, 56) + 0.01);
    }

    #[test]
    fn ao_removal_keeps_broad_lighting_and_only_lifts_local_darkening() {
        let image = lit_surface(256);

        let result = delight(&image, &only(|p| p.ao_removal = 1.0)).unwrap();

        assert!(gradient(&result) > gradient(&image) * 0.9);
        assert_ne!(result, image);
    }

    #[test]
    fn contrast_recovery_does_not_bring_back_removed_lighting() {
        let image = lit_surface(256);
        let flattened = only(|p| p.shadow_removal = 1.0);
        let recovered = DelightParams { contrast_recovery: 1.0, ..flattened };

        let before = gradient(&delight(&image, &flattened).unwrap());
        let after = gradient(&delight(&image, &recovered).unwrap());

        assert!(after < before * 1.2, "gradient went from {before} to {after}");
    }

    #[test]
    fn colour_preservation_keeps_lifted_shadows_saturated() {
        // Saturated orange under a strong left-to-right light.
        let image = Image::from_fn(256, 256, 3, |x, _, c| {
            let light = 0.2 + 0.8 * x as f32 / 256.0;
            [0.9, 0.45, 0.1][c] * light
        })
        .unwrap();
        let saturation = |img: &Image| {
            let pixel = img.pixel(20, 128);
            let (max, min) =
                (pixel[0].max(pixel[1]).max(pixel[2]), pixel[0].min(pixel[1]).min(pixel[2]));
            (max - min) / max.max(1e-6)
        };
        let lifted = only(|p| p.shadow_removal = 1.0);
        let preserved = DelightParams { color_preservation: 1.0, ..lifted };

        let grey = saturation(&delight(&image, &lifted).unwrap());
        let kept = saturation(&delight(&image, &preserved).unwrap());

        assert!(kept > grey + 0.05, "saturation {kept} with preservation, {grey} without");
    }

    #[test]
    fn colour_preservation_alone_is_identity() {
        let params = only(|p| p.color_preservation = 1.0);

        assert!(params.is_identity());
    }

    #[test]
    fn blur_based_controls_look_the_same_at_preview_and_full_size() {
        // The same picture at two sizes (grain scaled with the image): a
        // control must change the small and the large one by about the same
        // amount.
        let surface = |size: usize| {
            Image::from_fn(size, size, 3, |x, y, _| {
                let cell = size / 32;
                let grain = if (x / cell + y / cell).is_multiple_of(2) { 0.08 } else { -0.08 };
                0.5 + grain
            })
            .unwrap()
        };
        let (small, large) = (surface(128), surface(512));
        let change = |img: &Image, params: &DelightParams| {
            let out = delight(img, params).unwrap();
            out.data().iter().zip(img.data()).map(|(a, b)| (a - b).abs()).sum::<f32>()
                / img.data().len() as f32
        };

        for params in [only(|p| p.edge_consistency = 1.0), only(|p| p.detail_preservation = 1.0)] {
            let (at_small, at_large) = (change(&small, &params), change(&large, &params));
            let ratio = at_small / at_large.max(1e-6);
            assert!((0.5..2.0).contains(&ratio), "{params:?}: {at_small} vs {at_large}");
        }
    }

    #[test]
    fn flatness_reduces_colour_variation() {
        let image = Image::from_fn(96, 96, 3, |x, _, c| match (x < 48, c) {
            (true, 0) | (false, 2) => 0.8,
            _ => 0.3,
        })
        .unwrap();
        let red_spread = |img: &Image| (img.pixel(10, 10)[0] - img.pixel(80, 10)[0]).abs();

        let result = delight(&image, &only(|p| p.flatness = 1.0)).unwrap();

        assert!(red_spread(&result) < red_spread(&image) * 0.5);
    }

    #[test]
    fn alpha_channel_is_preserved() {
        let image = Image::from_fn(96, 96, 4, |x, y, c| {
            if c == 3 { (x + y) as f32 / 192.0 } else { 0.2 + 0.5 * x as f32 / 96.0 }
        })
        .unwrap();

        let result = delight(&image, &DelightParams::BALANCED).unwrap();

        assert_eq!(result.plane(3), image.plane(3));
        assert_ne!(result.plane(0), image.plane(0));
    }

    #[test]
    fn grayscale_input_stays_grayscale() {
        let image = Image::from_fn(96, 96, 1, |x, _, _| 0.2 + 0.6 * x as f32 / 96.0).unwrap();

        let result = delight(&image, &DelightParams::BALANCED).unwrap();

        assert_eq!(result.channels(), 1);
        assert_ne!(result, image);
    }

    #[test]
    fn lowfreq_blur_matches_a_direct_blur_closely() {
        let (w, h) = (256, 256);
        let plane: Vec<f32> =
            (0..w * h).map(|i| 100.0 + 80.0 * ((i % w) as f32 / w as f32)).collect();
        let sigma = 40.0;

        let fast = blur_lowfreq(&plane, w, h, sigma);
        let direct = gaussian_blur_plane(&plane, w, h, sigma, ((sigma * 3.0) as usize | 1) / 2);
        let worst = fast.iter().zip(&direct).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);

        assert_eq!(fast.len(), plane.len());
        assert!(worst < 4.0, "worst difference {worst} of 255");
    }

    #[test]
    fn cache_key_follows_the_values() {
        let a = DelightParams::BALANCED;
        let mut b = a;
        b.flatness += 0.1;

        assert_eq!(a.cache_key(), DelightParams::BALANCED.cache_key());
        assert_ne!(a.cache_key(), b.cache_key());
    }
}
