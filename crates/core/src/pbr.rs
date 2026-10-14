//! PBR material maps derived from one height field.
//!
//! Every map reads the same grayscale field, chosen by [`HeightSource`], and
//! nothing else, so each can be generated and cached independently.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::filter::{Edge, auto_blur_radius, correlate_plane, gaussian_blur_plane};
use crate::image::Image;
use crate::key::KeyHasher;

/// Z of the unscaled normal. A quarter keeps real texture detail visible
/// without saturating X and Y; a flat field is still exactly `(0, 0, 1)`.
pub const NORMAL_Z_BASIS: f32 = 0.25;
/// Brings kernel responses, normalised to unit absolute sum, to slope range.
const KERNEL_SLOPE_GAIN: f32 = 8.0;
const BLUR_SHARP_RANGE: f32 = 32.0;

/// Which channel of the source drives every map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HeightSource {
    #[default]
    AverageRgb,
    Luminance,
    MaxRgb,
    Red,
    Green,
    Blue,
    Alpha,
}

impl HeightSource {
    pub const ALL: [HeightSource; 7] = [
        HeightSource::AverageRgb,
        HeightSource::Luminance,
        HeightSource::MaxRgb,
        HeightSource::Red,
        HeightSource::Green,
        HeightSource::Blue,
        HeightSource::Alpha,
    ];

    pub fn label(self) -> &'static str {
        match self {
            HeightSource::AverageRgb => "Average RGB",
            HeightSource::Luminance => "Luminance",
            HeightSource::MaxRgb => "Max RGB",
            HeightSource::Red => "Red",
            HeightSource::Green => "Green",
            HeightSource::Blue => "Blue",
            HeightSource::Alpha => "Alpha channel",
        }
    }

    fn sample(self, pixel: &[f32]) -> f32 {
        let [r, g, b] = match pixel {
            [r, g, b, ..] => [*r, *g, *b],
            [gray, ..] => return *gray,
            [] => return 0.0,
        };
        match self {
            HeightSource::Luminance => 0.2126 * r + 0.7152 * g + 0.0722 * b,
            HeightSource::MaxRgb => r.max(g).max(b),
            HeightSource::Red => r,
            HeightSource::Green => g,
            HeightSource::Blue => b,
            HeightSource::Alpha => pixel.get(3).copied().unwrap_or((r + g + b) / 3.0),
            HeightSource::AverageRgb => (r + g + b) / 3.0,
        }
    }
}

/// The derivative filter used for the normal map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormalFilter {
    #[default]
    FourSample,
    Sobel3,
    Sobel5,
    Sobel7,
    Sobel9,
    DuDv,
}

impl NormalFilter {
    pub const ALL: [NormalFilter; 6] = [
        NormalFilter::FourSample,
        NormalFilter::Sobel3,
        NormalFilter::Sobel5,
        NormalFilter::Sobel7,
        NormalFilter::Sobel9,
        NormalFilter::DuDv,
    ];

    pub fn label(self) -> &'static str {
        match self {
            NormalFilter::FourSample => "4 sample",
            NormalFilter::Sobel3 => "3x3",
            NormalFilter::Sobel5 => "5x5",
            NormalFilter::Sobel7 => "7x7",
            NormalFilter::Sobel9 => "9x9",
            NormalFilter::DuDv => "dUdV",
        }
    }

    /// Kernel side length, or `None` for the centred four-sample difference.
    fn kernel_size(self) -> Option<usize> {
        match self {
            NormalFilter::FourSample | NormalFilter::DuDv => None,
            NormalFilter::Sobel3 => Some(3),
            NormalFilter::Sobel5 => Some(5),
            NormalFilter::Sobel7 => Some(7),
            NormalFilter::Sobel9 => Some(9),
        }
    }
}

/// Which way the green channel of the normal map points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormalFormat {
    #[default]
    OpenGl,
    DirectX,
}

impl NormalFormat {
    pub const ALL: [NormalFormat; 2] = [NormalFormat::OpenGl, NormalFormat::DirectX];

    pub fn label(self) -> &'static str {
        match self {
            NormalFormat::OpenGl => "OpenGL (Y+)",
            NormalFormat::DirectX => "DirectX (Y-)",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NormalParams {
    pub filter: NormalFilter,
    /// Sample across the borders, as a tiling texture needs.
    pub wrap: bool,
    pub invert_x: bool,
    pub invert_y: bool,
    pub invert_height: bool,
    pub format: NormalFormat,
    /// Smallest Z allowed, `0..=1`; limits how far normals may tilt.
    pub min_z: f32,
    /// Slope multiplier, `0.1..=8`.
    pub scale: f32,
}

impl Default for NormalParams {
    fn default() -> Self {
        Self {
            filter: NormalFilter::default(),
            wrap: true,
            invert_x: false,
            invert_y: false,
            invert_height: false,
            format: NormalFormat::default(),
            min_z: 0.0,
            scale: 4.266,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RoughnessParams {
    /// Overall level, `0..=1`; 0.5 leaves the field unscaled.
    pub intensity: f32,
    /// Contrast boost, `0..=1`.
    pub contrast: f32,
    /// Treat the field as glossiness.
    pub invert: bool,
}

impl Default for RoughnessParams {
    fn default() -> Self {
        Self { intensity: 0.5, contrast: 0.0, invert: false }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AoParams {
    /// How dark occluded areas become, `0..=1`.
    pub strength: f32,
    /// Height that counts as fully lit, `0..=1`.
    pub mean: f32,
    /// Distance from `mean` over which light falls off, `0..=1`.
    pub range: f32,
    /// Negative blurs, positive sharpens, `-32..=32`.
    pub blur_sharp: f32,
    pub invert: bool,
}

impl Default for AoParams {
    fn default() -> Self {
        Self { strength: 0.5, mean: 1.0, range: 1.0, blur_sharp: 0.0, invert: false }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DisplacementParams {
    /// Linear contrast about mid-grey, `-1..=1`.
    pub contrast: f32,
    /// Negative blurs, positive sharpens, `-32..=32`.
    pub blur_sharp: f32,
    pub invert: bool,
}

impl Default for DisplacementParams {
    fn default() -> Self {
        Self { contrast: -0.5, blur_sharp: 0.0, invert: false }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OpacityParams {
    /// How much of the height range is opaque, `0..=1`; 1 is fully opaque.
    pub threshold: f32,
    /// Width of the soft edge, `0..=1`; 0 is a hard cut.
    pub softness: f32,
}

impl Default for OpacityParams {
    fn default() -> Self {
        Self { threshold: 1.0, softness: 0.0 }
    }
}

/// Settings for every generated map.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct MaterialParams {
    pub height_source: HeightSource,
    pub normal: NormalParams,
    pub roughness: RoughnessParams,
    pub ao: AoParams,
    pub displacement: DisplacementParams,
    pub opacity: OpacityParams,
}

/// A generated material map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MapKind {
    Normal,
    Roughness,
    AmbientOcclusion,
    Displacement,
    Opacity,
}

impl MapKind {
    pub const ALL: [MapKind; 5] = [
        MapKind::Normal,
        MapKind::Roughness,
        MapKind::AmbientOcclusion,
        MapKind::Displacement,
        MapKind::Opacity,
    ];

    pub fn label(self) -> &'static str {
        match self {
            MapKind::Normal => "Normal",
            MapKind::Roughness => "Roughness",
            MapKind::AmbientOcclusion => "AO",
            MapKind::Displacement => "Displacement",
            MapKind::Opacity => "Opacity",
        }
    }

    /// Suffix appended to the source file name on export.
    pub fn file_suffix(self) -> &'static str {
        match self {
            MapKind::Normal => "_normal",
            MapKind::Roughness => "_roughness",
            MapKind::AmbientOcclusion => "_ao",
            MapKind::Displacement => "_displacement",
            MapKind::Opacity => "_opacity",
        }
    }
}

impl MaterialParams {
    /// A key covering the height source plus only the settings `kind` reads,
    /// so changing one map's slider does not invalidate the other maps.
    pub fn cache_key(&self, kind: MapKind) -> u64 {
        let mut hasher = KeyHasher::new();
        hasher.u64(kind as u64).u64(self.height_source as u64);
        match kind {
            MapKind::Normal => {
                let n = &self.normal;
                hasher.u64(n.filter as u64).u64(n.format as u64).f32(n.min_z).f32(n.scale);
                hasher.bool(n.wrap).bool(n.invert_x).bool(n.invert_y).bool(n.invert_height)
            }
            MapKind::Roughness => {
                let r = &self.roughness;
                hasher.f32(r.intensity).f32(r.contrast).bool(r.invert)
            }
            MapKind::AmbientOcclusion => {
                let a = &self.ao;
                hasher.f32(a.strength).f32(a.mean).f32(a.range).f32(a.blur_sharp).bool(a.invert)
            }
            MapKind::Displacement => {
                let d = &self.displacement;
                hasher.f32(d.contrast).f32(d.blur_sharp).bool(d.invert)
            }
            MapKind::Opacity => hasher.f32(self.opacity.threshold).f32(self.opacity.softness),
        };
        hasher.finish()
    }
}

/// The grayscale field every map is derived from.
#[derive(Debug, Clone, PartialEq)]
pub struct HeightField {
    pub width: usize,
    pub height: usize,
    /// Height per pixel in `[0, 1]`.
    pub values: Vec<f32>,
    /// The source alpha, carried into every generated map.
    pub alpha: Option<Vec<f32>>,
}

impl HeightField {
    /// Extracts the field from `image` using `source`.
    pub fn from_image(image: &Image, source: HeightSource) -> Self {
        let values = image
            .data()
            .par_chunks(image.channels())
            .map(|pixel| finite_unit(source.sample(pixel)))
            .collect();
        Self { width: image.width(), height: image.height(), values, alpha: image.alpha_plane() }
    }

    fn blur(&self, plane: &[f32], sigma: f32) -> Vec<f32> {
        gaussian_blur_plane(plane, self.width, self.height, sigma, auto_blur_radius(sigma))
    }

    /// Wraps a scalar plane as an image: gray, or gray plus the source alpha.
    fn scalar_image(&self, plane: Vec<f32>) -> Result<Image> {
        let Some(alpha) = &self.alpha else {
            return Image::from_plane(self.width, self.height, plane);
        };
        let data = plane.iter().zip(alpha).flat_map(|(v, a)| [*v, *v, *v, *a]).collect();
        Image::new(self.width, self.height, 4, data)
    }
}

fn finite_unit(value: f32) -> f32 {
    if value.is_finite() { value.clamp(0.0, 1.0) } else { 0.0 }
}

fn clamp_unit(plane: &mut [f32]) {
    plane.par_iter_mut().for_each(|v| *v = v.clamp(0.0, 1.0));
}

/// What a blur/sharpen setting does to a field: a Gaussian blur, optionally
/// used as the mask of an unsharp sharpen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlurSharp {
    /// Sigma of the Gaussian, in pixels.
    pub sigma: f32,
    /// `None` keeps the blur itself. `Some(gain)` sharpens: each value moves
    /// away from its blurred copy by `gain` times their difference.
    pub sharpen: Option<f32>,
}

impl BlurSharp {
    /// Reads a blur/sharpen setting: negative blurs, positive sharpens, and
    /// zero (or a non-finite value) leaves the field alone.
    pub fn from_setting(blur_sharp: f32) -> Option<Self> {
        if blur_sharp == 0.0 || !blur_sharp.is_finite() {
            return None;
        }
        let amount = blur_sharp.abs().min(BLUR_SHARP_RANGE);
        let sharpen = (blur_sharp > 0.0).then_some(amount / BLUR_SHARP_RANGE * 1.5);
        Some(Self { sigma: 0.4 + amount * 0.12, sharpen })
    }
}

/// Blurs (negative) or unsharp-mask sharpens (positive) a `[0, 1]` field.
fn apply_blur_sharp(field: &HeightField, plane: Vec<f32>, blur_sharp: f32) -> Vec<f32> {
    let Some(plan) = BlurSharp::from_setting(blur_sharp) else {
        return plane;
    };
    let blurred = field.blur(&plane, plan.sigma);
    let Some(gain) = plan.sharpen else {
        return blurred;
    };
    plane.par_iter().zip(blurred.par_iter()).map(|(v, b)| v + (v - b) * gain).collect()
}

/// The derivative kernel for horizontal slopes; its transpose gives vertical.
///
/// Row `r`, left half: `-(min(r, n-1-r) + 1 + column)`, mirrored with a
/// positive sign on the right and zero in the centre column. Normalised so the
/// absolute values sum to one.
fn sobel_kernel(size: usize) -> Vec<f32> {
    let centre = size / 2;
    let mut kernel = vec![0.0f32; size * size];
    for (i, value) in kernel.iter_mut().enumerate() {
        let (row, column) = (i / size, i % size);
        let row_weight = row.min(size - 1 - row) + 1;
        *value = match column.cmp(&centre) {
            std::cmp::Ordering::Less => -((row_weight + column) as f32),
            std::cmp::Ordering::Greater => (row_weight + size - 1 - column) as f32,
            std::cmp::Ordering::Equal => 0.0,
        };
    }
    let total: f32 = kernel.iter().map(|v| v.abs()).sum();
    kernel.iter_mut().for_each(|v| *v /= total.max(f32::EPSILON));
    kernel
}

fn transpose(kernel: &[f32], size: usize) -> Vec<f32> {
    (0..size * size)
        .map(|i| kernel.get((i % size) * size + i / size).copied().unwrap_or(0.0))
        .collect()
}

/// The square kernel a normal filter correlates the height field with.
#[derive(Debug, Clone, PartialEq)]
pub struct SlopeKernel {
    /// Row-major weights for horizontal slopes; the transpose gives vertical.
    pub weights: Vec<f32>,
    /// Side length of the kernel.
    pub size: usize,
    /// Multiplier that brings the kernel's response to slope range.
    pub gain: f32,
}

impl SlopeKernel {
    pub fn for_filter(filter: NormalFilter) -> Self {
        match filter.kernel_size() {
            Some(size) => Self { weights: sobel_kernel(size), size, gain: KERNEL_SLOPE_GAIN },
            None => {
                let half_difference = vec![0.0, 0.0, 0.0, -0.5, 0.0, 0.5, 0.0, 0.0, 0.0];
                Self { weights: half_difference, size: 3, gain: 1.0 }
            }
        }
    }
}

/// Horizontal and vertical slopes of the height plane.
fn slopes(
    field: &HeightField,
    plane: &[f32],
    filter: NormalFilter,
    wrap: bool,
) -> (Vec<f32>, Vec<f32>) {
    let edge = if wrap { Edge::Wrap } else { Edge::Clamp };
    let (width, height) = (field.width, field.height);
    let SlopeKernel { weights, size, gain } = SlopeKernel::for_filter(filter);
    let mut du = correlate_plane(plane, width, height, &weights, size, edge);
    let mut dv = correlate_plane(plane, width, height, &transpose(&weights, size), size, edge);
    if gain != 1.0 {
        du.par_iter_mut().chain(dv.par_iter_mut()).for_each(|v| *v *= gain);
    }
    (du, dv)
}

/// A unit normal from two slopes, tilted no further than `min_z` allows.
fn normal_from_slopes(slope_x: f32, slope_y: f32, min_z: f32) -> [f32; 3] {
    let length = (slope_x * slope_x + slope_y * slope_y + NORMAL_Z_BASIS * NORMAL_Z_BASIS).sqrt();
    let (mut nx, mut ny, mut nz) = (-slope_x / length, -slope_y / length, NORMAL_Z_BASIS / length);
    if min_z > 0.0 && nz < min_z {
        let planar = (nx * nx + ny * ny).sqrt().max(1e-8);
        let target = (1.0 - min_z * min_z).max(0.0).sqrt();
        nx *= target / planar;
        ny *= target / planar;
        nz = min_z;
    }
    [nx, ny, nz]
}

/// A tangent-space normal map, RGB = XYZ mapped to `[0, 1]`.
fn normal_map(field: &HeightField, params: &NormalParams) -> Result<Image> {
    let heights: Vec<f32> = if params.invert_height {
        field.values.par_iter().map(|v| 1.0 - v).collect()
    } else {
        field.values.clone()
    };
    let (du, dv) = slopes(field, &heights, params.filter, params.wrap);
    let scale = params.scale.clamp(0.1, 8.0);
    let sign_x = if params.invert_x { -scale } else { scale };
    let flip_y = params.invert_y != (params.format == NormalFormat::DirectX);
    let sign_y = if flip_y { -scale } else { scale };
    let min_z = finite_unit(params.min_z);

    let channels = if field.alpha.is_some() { 4 } else { 3 };
    let mut data = vec![1.0f32; field.values.len() * channels];
    data.par_chunks_mut(channels).enumerate().for_each(|(i, pixel)| {
        let slope_x = du.get(i).copied().unwrap_or(0.0) * sign_x;
        let slope_y = dv.get(i).copied().unwrap_or(0.0) * sign_y;
        let normal = normal_from_slopes(slope_x, slope_y, min_z);
        pixel.iter_mut().zip(normal).for_each(|(p, n)| *p = (n * 0.5 + 0.5).clamp(0.0, 1.0));
        if let (Some(target), Some(alpha)) = (pixel.get_mut(3), &field.alpha) {
            *target = alpha.get(i).copied().unwrap_or(1.0);
        }
    });
    Image::new(field.width, field.height, channels, data)
}

fn roughness_map(field: &HeightField, params: &RoughnessParams) -> Vec<f32> {
    let gain = 1.0 + params.contrast * 2.0;
    let level = params.intensity * 2.0;
    field
        .values
        .par_iter()
        .map(|v| {
            let value = if params.invert { 1.0 - v } else { *v };
            let contrasted = (0.5 + (value - 0.5) * gain).clamp(0.0, 1.0);
            (contrasted * level).clamp(0.0, 1.0)
        })
        .collect()
}

/// Heights within `range` of `mean` stay lit; further away darkens toward
/// `strength`.
fn ao_map(field: &HeightField, params: &AoParams) -> Vec<f32> {
    let range = params.range.max(1e-4);
    let lit: Vec<f32> = field
        .values
        .par_iter()
        .map(|v| {
            let closeness = ((range - (v - params.mean).abs()) / range).max(0.0).sqrt();
            closeness + (1.0 - closeness) * (1.0 - params.strength)
        })
        .collect();
    let mut result = apply_blur_sharp(field, lit, params.blur_sharp);
    if params.invert {
        result.par_iter_mut().for_each(|v| *v = 1.0 - *v);
    }
    clamp_unit(&mut result);
    result
}

fn displacement_map(field: &HeightField, params: &DisplacementParams) -> Vec<f32> {
    let contrast = params.contrast.clamp(-1.0, 0.999);
    let factor = (contrast + 1.0) / (1.0 - contrast);
    let shaped: Vec<f32> = field
        .values
        .par_iter()
        .map(|v| {
            let value = factor * (v - 0.5) + 0.5;
            (if params.invert { 1.0 - value } else { value }).clamp(0.0, 1.0)
        })
        .collect();
    let mut result = apply_blur_sharp(field, shaped, params.blur_sharp);
    clamp_unit(&mut result);
    result
}

fn opacity_map(field: &HeightField, params: &OpacityParams) -> Vec<f32> {
    let cutoff = 1.0 - params.threshold;
    let width = (params.softness * 0.45).max(0.01);
    field
        .values
        .par_iter()
        .map(|v| {
            if params.softness > 0.0 {
                ((v - cutoff + width * 0.5) / width).clamp(0.0, 1.0)
            } else if *v > cutoff {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

/// Generates one map from the height field.
pub fn generate_map(kind: MapKind, field: &HeightField, params: &MaterialParams) -> Result<Image> {
    match kind {
        MapKind::Normal => normal_map(field, &params.normal),
        MapKind::Roughness => field.scalar_image(roughness_map(field, &params.roughness)),
        MapKind::AmbientOcclusion => field.scalar_image(ao_map(field, &params.ao)),
        MapKind::Displacement => field.scalar_image(displacement_map(field, &params.displacement)),
        MapKind::Opacity => field.scalar_image(opacity_map(field, &params.opacity)),
    }
}

/// Generates every map for `image`.
pub fn generate_maps(image: &Image, params: &MaterialParams) -> Result<Vec<(MapKind, Image)>> {
    let field = HeightField::from_image(image, params.height_source);
    MapKind::ALL
        .par_iter()
        .map(|kind| generate_map(*kind, &field, params).map(|map| (*kind, map)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bumps(size: usize, channels: usize) -> Image {
        Image::from_fn(size, size, channels, |x, y, c| {
            if c == 3 {
                return 0.75;
            }
            let (fx, fy) = (x as f32 / size as f32, y as f32 / size as f32);
            0.5 + 0.4 * (fx * 12.0).sin() * (fy * 9.0).cos()
        })
        .unwrap()
    }

    fn map(kind: MapKind, image: &Image, params: &MaterialParams) -> Image {
        let field = HeightField::from_image(image, params.height_source);
        generate_map(kind, &field, params).unwrap()
    }

    #[test]
    fn all_maps_are_generated_at_source_size() {
        let maps = generate_maps(&bumps(64, 3), &MaterialParams::default()).unwrap();

        assert_eq!(maps.iter().map(|(kind, _)| *kind).collect::<Vec<_>>(), MapKind::ALL);
        assert!(maps.iter().all(|(_, m)| (m.width(), m.height()) == (64, 64)));
    }

    #[test]
    fn flat_height_gives_a_straight_up_normal() {
        let flat = Image::filled(32, 32, 3, 0.5).unwrap();

        let normal = map(MapKind::Normal, &flat, &MaterialParams::default());

        assert_eq!(normal.channels(), 3);
        for pixel in normal.data().chunks(3) {
            assert!((pixel[0] - 0.5).abs() < 1e-6 && (pixel[1] - 0.5).abs() < 1e-6);
            assert!((pixel[2] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn normals_are_unit_length() {
        let normal = map(MapKind::Normal, &bumps(64, 3), &MaterialParams::default());

        for pixel in normal.data().chunks(3) {
            let v: Vec<f32> = pixel.iter().map(|c| c * 2.0 - 1.0).collect();
            let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!((length - 1.0).abs() < 1e-3, "length {length}");
        }
    }

    #[test]
    fn min_z_is_enforced() {
        let mut params = MaterialParams::default();
        params.normal.scale = 8.0;
        params.normal.min_z = 0.8;

        let normal = map(MapKind::Normal, &bumps(64, 3), &params);

        assert!(normal.data().chunks(3).all(|p| p[2] * 2.0 - 1.0 >= 0.8 - 1e-3));
    }

    #[test]
    fn every_filter_produces_a_non_flat_normal() {
        for filter in NormalFilter::ALL {
            let mut params = MaterialParams::default();
            params.normal.filter = filter;

            let normal = map(MapKind::Normal, &bumps(64, 3), &params);

            assert!(normal.data().chunks(3).any(|p| (p[0] - 0.5).abs() > 0.02), "{filter:?}");
        }
    }

    #[test]
    fn wrap_changes_only_the_border_derivatives() {
        let ramp = Image::from_fn(32, 32, 1, |x, _, _| x as f32 / 31.0).unwrap();
        let mut clamped = MaterialParams::default();
        clamped.normal.wrap = false;

        let wrapped_map = map(MapKind::Normal, &ramp, &MaterialParams::default());
        let clamped_map = map(MapKind::Normal, &ramp, &clamped);

        assert_ne!(wrapped_map.pixel(0, 16), clamped_map.pixel(0, 16));
        assert_eq!(wrapped_map.pixel(16, 16), clamped_map.pixel(16, 16));
    }

    #[test]
    fn directx_format_flips_green_only() {
        let source = bumps(64, 3);
        let mut directx = MaterialParams::default();
        directx.normal.format = NormalFormat::DirectX;

        let gl = map(MapKind::Normal, &source, &MaterialParams::default());
        let dx = map(MapKind::Normal, &source, &directx);

        for (a, b) in gl.data().chunks(3).zip(dx.data().chunks(3)) {
            assert!((a[0] - b[0]).abs() < 1e-6 && (a[2] - b[2]).abs() < 1e-6);
            assert!((a[1] + b[1] - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn sobel_kernels_are_antisymmetric_and_normalised() {
        for size in [3, 5, 7, 9] {
            let kernel = sobel_kernel(size);

            assert!((kernel.iter().map(|v| v.abs()).sum::<f32>() - 1.0).abs() < 1e-5);
            assert!(kernel.iter().sum::<f32>().abs() < 1e-6);
            assert!(kernel[0] < 0.0 && kernel[size - 1] > 0.0);
        }
        assert_eq!(
            sobel_kernel(3).iter().map(|v| v * 8.0).collect::<Vec<_>>(),
            vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0]
        );
    }

    #[test]
    fn scalar_maps_are_single_channel_and_in_range() {
        let source = bumps(64, 3);
        let kinds = [
            MapKind::Roughness,
            MapKind::AmbientOcclusion,
            MapKind::Displacement,
            MapKind::Opacity,
        ];

        for kind in kinds {
            let result = map(kind, &source, &MaterialParams::default());

            assert_eq!(result.channels(), 1, "{kind:?}");
            assert!(result.data().iter().all(|v| (0.0..=1.0).contains(v)), "{kind:?}");
        }
    }

    #[test]
    fn source_alpha_is_carried_into_every_map() {
        let maps = generate_maps(&bumps(32, 4), &MaterialParams::default()).unwrap();

        for (kind, image) in maps {
            assert_eq!(image.channels(), 4, "{kind:?}");
            assert!(image.plane(3).iter().all(|a| *a == 0.75), "{kind:?}");
        }
    }

    #[test]
    fn roughness_invert_mirrors_the_field() {
        let source = bumps(32, 3);
        let mut inverted = MaterialParams::default();
        inverted.roughness.invert = true;

        let plain = map(MapKind::Roughness, &source, &MaterialParams::default());
        let flipped = map(MapKind::Roughness, &source, &inverted);

        for (a, b) in plain.data().iter().zip(flipped.data()) {
            assert!((a + b - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn default_opacity_is_fully_opaque_and_threshold_cuts() {
        let source = bumps(32, 3);
        let mut cut = MaterialParams::default();
        cut.opacity.threshold = 0.5;

        let opaque = map(MapKind::Opacity, &source, &MaterialParams::default());
        let partial = map(MapKind::Opacity, &source, &cut);

        assert!(opaque.data().iter().all(|v| *v == 1.0));
        assert!(partial.data().contains(&0.0));
        assert!(partial.data().iter().all(|v| *v == 0.0 || *v == 1.0));
    }

    #[test]
    fn blur_sharp_moves_displacement_in_both_directions() {
        let source = bumps(64, 3);
        let with = |blur_sharp: f32| {
            let mut params = MaterialParams::default();
            params.displacement.blur_sharp = blur_sharp;
            map(MapKind::Displacement, &source, &params)
        };

        assert_ne!(with(-16.0), with(0.0));
        assert_ne!(with(16.0), with(0.0));
    }

    #[test]
    fn height_source_selects_the_channel() {
        let source = Image::from_fn(8, 8, 3, |_, _, c| [0.9, 0.5, 0.1][c]).unwrap();

        let value = |s| HeightField::from_image(&source, s).values[0];

        assert_eq!(value(HeightSource::Red), 0.9);
        assert_eq!(value(HeightSource::Blue), 0.1);
        assert_eq!(value(HeightSource::MaxRgb), 0.9);
        assert!((value(HeightSource::AverageRgb) - 0.5).abs() < 1e-6);
        assert!((value(HeightSource::Alpha) - 0.5).abs() < 1e-6, "falls back without alpha");
    }

    #[test]
    fn cache_key_changes_only_for_the_affected_map() {
        let base = MaterialParams::default();
        let mut changed = base;
        changed.roughness.contrast = 0.7;

        assert_ne!(base.cache_key(MapKind::Roughness), changed.cache_key(MapKind::Roughness));
        assert_eq!(base.cache_key(MapKind::Normal), changed.cache_key(MapKind::Normal));

        changed.height_source = HeightSource::Red;
        assert_ne!(base.cache_key(MapKind::Normal), changed.cache_key(MapKind::Normal));
    }
}
