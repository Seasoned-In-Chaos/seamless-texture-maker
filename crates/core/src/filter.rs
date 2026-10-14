//! Shared image filters: blur, resampling, correlation and rotation.
//!
//! Plane functions take a row-major `&[f32]` of `width * height` values.
//! Rows are processed in parallel.

use rayon::prelude::*;

use crate::error::{CoreError, Result};
use crate::image::Image;

/// How samples outside the image are resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edge {
    /// Repeat the nearest edge value.
    Clamp,
    /// Continue on the opposite side, as a tiling texture does.
    Wrap,
    /// Mirror about the edge pixel without repeating it (`dcb|abcd|cba`).
    Reflect,
}

impl Edge {
    /// Maps any signed index into `[0, len)`.
    pub fn resolve(self, index: isize, len: usize) -> usize {
        if len <= 1 {
            return 0;
        }
        let n = len as isize;
        let resolved = match self {
            Edge::Clamp => index.clamp(0, n - 1),
            Edge::Wrap => index.rem_euclid(n),
            Edge::Reflect => {
                let period = 2 * (n - 1);
                let m = index.rem_euclid(period);
                if m < n { m } else { period - m }
            }
        };
        resolved as usize
    }

    /// Resolved source index for each position of an axis padded by `radius`.
    fn padded_indices(self, len: usize, radius: usize) -> Vec<usize> {
        (0..len + 2 * radius).map(|i| self.resolve(i as isize - radius as isize, len)).collect()
    }
}

/// Kernel radius OpenCV picks for a float image when only sigma is given.
pub fn auto_blur_radius(sigma: f32) -> usize {
    (((sigma * 8.0 + 1.0).round().max(1.0) as usize) | 1) / 2
}

/// A normalised Gaussian kernel of `2 * radius + 1` taps.
pub fn gaussian_kernel(sigma: f32, radius: usize) -> Vec<f32> {
    let sigma = sigma.max(1e-3);
    let denom = 2.0 * sigma * sigma;
    let mut kernel: Vec<f32> = (0..=2 * radius)
        .map(|i| {
            let d = i as f32 - radius as f32;
            (-d * d / denom).exp()
        })
        .collect();
    let sum: f32 = kernel.iter().sum();
    if sum > 0.0 {
        kernel.iter_mut().for_each(|k| *k /= sum);
    }
    kernel
}

/// Separable Gaussian blur of one plane with reflected borders.
pub fn gaussian_blur_plane(
    src: &[f32],
    width: usize,
    height: usize,
    sigma: f32,
    radius: usize,
) -> Vec<f32> {
    if radius == 0 || src.len() != width * height || src.is_empty() {
        return src.to_vec();
    }
    let kernel = gaussian_kernel(sigma, radius);
    let horizontal = blur_rows(src, width, &kernel, radius);
    blur_columns(&horizontal, width, height, &kernel, radius)
}

fn blur_rows(src: &[f32], width: usize, kernel: &[f32], radius: usize) -> Vec<f32> {
    let xs = Edge::Reflect.padded_indices(width, radius);
    let mut out = vec![0.0f32; src.len()];
    out.par_chunks_mut(width).zip(src.par_chunks(width)).for_each(|(dst, row)| {
        for (x, value) in dst.iter_mut().enumerate() {
            let taps = xs.iter().skip(x).zip(kernel);
            *value = taps.map(|(sx, k)| row.get(*sx).copied().unwrap_or(0.0) * k).sum();
        }
    });
    out
}

fn blur_columns(
    src: &[f32],
    width: usize,
    height: usize,
    kernel: &[f32],
    radius: usize,
) -> Vec<f32> {
    let ys = Edge::Reflect.padded_indices(height, radius);
    let mut out = vec![0.0f32; src.len()];
    out.par_chunks_mut(width).enumerate().for_each(|(y, dst)| {
        for (sy, k) in ys.iter().skip(y).zip(kernel) {
            let row = src.get(sy * width..(sy + 1) * width).unwrap_or(&[]);
            dst.iter_mut().zip(row).for_each(|(d, s)| *d += s * k);
        }
    });
    out
}

/// Weighted source indices for every destination index along one axis.
struct AxisTaps {
    /// `starts[d]..starts[d + 1]` is the tap range for destination index `d`.
    starts: Vec<usize>,
    indices: Vec<usize>,
    weights: Vec<f32>,
}

impl AxisTaps {
    fn with_capacity(dst: usize) -> Self {
        Self { starts: vec![0], indices: Vec::with_capacity(dst * 2), weights: Vec::new() }
    }

    fn push(&mut self, index: usize, weight: f32) {
        self.indices.push(index);
        self.weights.push(weight);
    }

    fn end_destination(&mut self) {
        self.starts.push(self.indices.len());
    }

    fn len(&self) -> usize {
        self.starts.len().saturating_sub(1)
    }

    fn taps(&self, dst: usize) -> impl Iterator<Item = (usize, f32)> + '_ {
        let start = self.starts.get(dst).copied().unwrap_or(0);
        let end = self.starts.get(dst + 1).copied().unwrap_or(start);
        let indices = self.indices.get(start..end).unwrap_or(&[]);
        let weights = self.weights.get(start..end).unwrap_or(&[]);
        indices.iter().copied().zip(weights.iter().copied())
    }

    /// Two-tap linear interpolation with half-pixel centres.
    fn bilinear(src: usize, dst: usize, edge: Edge) -> Self {
        let mut taps = Self::with_capacity(dst);
        let scale = src as f32 / dst as f32;
        for d in 0..dst {
            let position = (d as f32 + 0.5) * scale - 0.5;
            let base = position.floor();
            let t = position - base;
            taps.push(edge.resolve(base as isize, src), 1.0 - t);
            taps.push(edge.resolve(base as isize + 1, src), t);
            taps.end_destination();
        }
        taps
    }

    /// Exact box averaging for shrinking; falls back to linear when enlarging.
    fn area(src: usize, dst: usize) -> Self {
        if dst >= src {
            return Self::bilinear(src, dst, Edge::Clamp);
        }
        let mut taps = Self::with_capacity(dst);
        let scale = src as f32 / dst as f32;
        for d in 0..dst {
            let (start, end) = (d as f32 * scale, (d as f32 + 1.0) * scale);
            for i in start.floor() as usize..(end.ceil() as usize).min(src) {
                let overlap = end.min(i as f32 + 1.0) - start.max(i as f32);
                if overlap > 0.0 {
                    taps.push(i, overlap / scale);
                }
            }
            taps.end_destination();
        }
        taps
    }
}

fn resample(image: &Image, taps_x: &AxisTaps, taps_y: &AxisTaps) -> Result<Image> {
    let channels = image.channels();
    let (dst_w, dst_h) = (taps_x.len(), taps_y.len());
    let dst_row = dst_w * channels;

    let mut horizontal = vec![0.0f32; dst_row * image.height()];
    horizontal.par_chunks_mut(dst_row).zip(image.data().par_chunks(image.row_len())).for_each(
        |(dst, src)| {
            for (dx, pixel) in dst.chunks_mut(channels).enumerate() {
                for (sx, weight) in taps_x.taps(dx) {
                    let source = src.get(sx * channels..(sx + 1) * channels).unwrap_or(&[]);
                    pixel.iter_mut().zip(source).for_each(|(d, s)| *d += s * weight);
                }
            }
        },
    );

    let mut out = vec![0.0f32; dst_row * dst_h];
    out.par_chunks_mut(dst_row).enumerate().for_each(|(dy, dst)| {
        for (sy, weight) in taps_y.taps(dy) {
            let source = horizontal.get(sy * dst_row..(sy + 1) * dst_row).unwrap_or(&[]);
            dst.iter_mut().zip(source).for_each(|(d, s)| *d += s * weight);
        }
    });
    Image::new(dst_w, dst_h, channels, out)
}

fn check_target(width: usize, height: usize) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(CoreError::InvalidParameter {
            name: "size",
            reason: format!("target size {width}x{height} has no pixels"),
        });
    }
    Ok(())
}

/// Resizes by box-averaging source pixels, the right filter for shrinking.
pub fn resize_area(image: &Image, width: usize, height: usize) -> Result<Image> {
    check_target(width, height)?;
    if (image.width(), image.height()) == (width, height) {
        return Ok(image.clone());
    }
    let taps_x = AxisTaps::area(image.width(), width);
    let taps_y = AxisTaps::area(image.height(), height);
    resample(image, &taps_x, &taps_y)
}

/// Resizes with bilinear interpolation.
///
/// With [`Edge::Wrap`] the borders interpolate against the opposite side, which
/// keeps a tiling image continuous across its edges after resizing.
pub fn resize_bilinear(image: &Image, width: usize, height: usize, edge: Edge) -> Result<Image> {
    check_target(width, height)?;
    if (image.width(), image.height()) == (width, height) {
        return Ok(image.clone());
    }
    let taps_x = AxisTaps::bilinear(image.width(), width, edge);
    let taps_y = AxisTaps::bilinear(image.height(), height, edge);
    resample(image, &taps_x, &taps_y)
}

/// Shrinks so the longer side is at most `max_side`; never enlarges.
pub fn fit_within(image: &Image, max_side: usize) -> Result<Image> {
    let longest = image.width().max(image.height());
    if longest <= max_side || max_side == 0 {
        return Ok(image.clone());
    }
    let scale = max_side as f32 / longest as f32;
    let width = ((image.width() as f32 * scale) as usize).max(1);
    let height = ((image.height() as f32 * scale) as usize).max(1);
    resize_area(image, width, height)
}

/// Correlates a plane with a square `size x size` kernel.
pub fn correlate_plane(
    src: &[f32],
    width: usize,
    height: usize,
    kernel: &[f32],
    size: usize,
    edge: Edge,
) -> Vec<f32> {
    if size == 0 || kernel.len() != size * size || src.len() != width * height {
        return src.to_vec();
    }
    let radius = size / 2;
    let xs = edge.padded_indices(width, radius);
    let ys = edge.padded_indices(height, radius);
    let mut out = vec![0.0f32; src.len()];
    out.par_chunks_mut(width).enumerate().for_each(|(y, dst)| {
        for (ky, kernel_row) in kernel.chunks(size).enumerate() {
            let sy = ys.get(y + ky).copied().unwrap_or(0);
            let row = src.get(sy * width..(sy + 1) * width).unwrap_or(&[]);
            correlate_row(dst, row, kernel_row, &xs);
        }
    });
    out
}

fn correlate_row(dst: &mut [f32], row: &[f32], kernel_row: &[f32], xs: &[usize]) {
    for (kx, weight) in kernel_row.iter().enumerate() {
        if *weight == 0.0 {
            continue;
        }
        for (d, sx) in dst.iter_mut().zip(xs.iter().skip(kx)) {
            *d += row.get(*sx).copied().unwrap_or(0.0) * weight;
        }
    }
}

/// Rotates about the image centre by `degrees` (counter-clockwise), sampling
/// bilinearly with reflected borders so no empty corners appear.
pub fn rotate_reflect(image: &Image, degrees: f32) -> Result<Image> {
    let (width, height, channels) = (image.width(), image.height(), image.channels());
    let (sin, cos) = degrees.to_radians().sin_cos();
    let (cx, cy) = (width as f32 / 2.0, height as f32 / 2.0);
    let mut out = vec![0.0f32; image.data().len()];
    out.par_chunks_mut(image.row_len()).enumerate().for_each(|(y, row)| {
        let dy = y as f32 - cy;
        for (x, pixel) in row.chunks_mut(channels).enumerate() {
            let dx = x as f32 - cx;
            let source_x = cos * dx - sin * dy + cx;
            let source_y = sin * dx + cos * dy + cy;
            sample_bilinear(image, source_x, source_y, Edge::Reflect, pixel);
        }
    });
    Image::new(width, height, channels, out)
}

/// Adds the bilinear sample of `image` at pixel coordinates (`x`, `y`) to
/// `out`, which holds one zeroed pixel. Pixel centres sit on whole numbers;
/// `edge` decides what lies outside the image.
pub(crate) fn sample_bilinear(image: &Image, x: f32, y: f32, edge: Edge, out: &mut [f32]) {
    let (x0, y0) = (x.floor(), y.floor());
    let (tx, ty) = (x - x0, y - y0);
    let xa = edge.resolve(x0 as isize, image.width());
    let xb = edge.resolve(x0 as isize + 1, image.width());
    let ya = edge.resolve(y0 as isize, image.height());
    let yb = edge.resolve(y0 as isize + 1, image.height());
    let corners = [
        (image.pixel(xa, ya), (1.0 - tx) * (1.0 - ty)),
        (image.pixel(xb, ya), tx * (1.0 - ty)),
        (image.pixel(xa, yb), (1.0 - tx) * ty),
        (image.pixel(xb, yb), tx * ty),
    ];
    for (source, weight) in corners {
        out.iter_mut().zip(source).for_each(|(o, s)| *o += s * weight);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() <= tolerance, "index {i}: {a} vs {e}");
        }
    }

    #[test]
    fn edge_modes_resolve_out_of_range_indices() {
        assert_eq!(Edge::Clamp.resolve(-3, 5), 0);
        assert_eq!(Edge::Clamp.resolve(9, 5), 4);
        assert_eq!(Edge::Wrap.resolve(-1, 5), 4);
        assert_eq!(Edge::Wrap.resolve(6, 5), 1);
        assert_eq!(Edge::Reflect.resolve(-1, 5), 1);
        assert_eq!(Edge::Reflect.resolve(5, 5), 3);
        assert_eq!(Edge::Reflect.resolve(-7, 1), 0);
    }

    #[test]
    fn gaussian_kernel_is_normalised_and_symmetric() {
        let kernel = gaussian_kernel(2.0, 6);

        assert!((kernel.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!((kernel[0] - kernel[12]).abs() < 1e-7);
        assert!(kernel[6] > kernel[5]);
    }

    #[test]
    fn blur_leaves_a_constant_plane_unchanged() {
        let plane = vec![0.4f32; 32 * 24];

        let blurred = gaussian_blur_plane(&plane, 32, 24, 3.0, 9);

        assert_close(&blurred, &plane, 1e-5);
    }

    #[test]
    fn blur_preserves_total_energy_of_an_impulse() {
        let mut plane = vec![0.0f32; 41 * 41];
        plane[20 * 41 + 20] = 1.0;

        let blurred = gaussian_blur_plane(&plane, 41, 41, 2.0, 8);

        assert!((blurred.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        assert!(blurred[20 * 41 + 20] < 1.0);
    }

    #[test]
    fn resize_to_same_size_is_identity() {
        let image = Image::from_fn(8, 6, 3, |x, y, c| (x + y + c) as f32 * 0.01).unwrap();

        assert_eq!(resize_area(&image, 8, 6).unwrap(), image);
        assert_eq!(resize_bilinear(&image, 8, 6, Edge::Wrap).unwrap(), image);
    }

    #[test]
    fn area_resize_averages_each_block() {
        let image = Image::from_plane(4, 2, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();

        let half = resize_area(&image, 2, 1).unwrap();

        assert_close(half.data(), &[2.5, 4.5], 1e-5);
    }

    #[test]
    fn area_resize_handles_non_integer_ratios() {
        let image = Image::filled(10, 10, 3, 0.3).unwrap();

        let resized = resize_area(&image, 7, 3).unwrap();

        assert_eq!((resized.width(), resized.height()), (7, 3));
        assert_close(resized.data(), &vec![0.3; 7 * 3 * 3], 1e-5);
    }

    #[test]
    fn wrapped_bilinear_interpolates_across_the_border() {
        let image = Image::from_plane(4, 1, vec![0.0, 0.0, 0.0, 1.0]).unwrap();

        let wrapped = resize_bilinear(&image, 8, 1, Edge::Wrap).unwrap();
        let clamped = resize_bilinear(&image, 8, 1, Edge::Clamp).unwrap();

        assert!(wrapped.data()[0] > 0.2, "first sample should see the last column");
        assert_eq!(clamped.data()[0], 0.0);
    }

    #[test]
    fn resize_rejects_empty_target() {
        let image = Image::filled(4, 4, 1, 0.0).unwrap();

        assert!(matches!(resize_area(&image, 0, 4), Err(CoreError::InvalidParameter { .. })));
    }

    #[test]
    fn fit_within_shrinks_only_when_needed() {
        let image = Image::filled(200, 100, 3, 0.5).unwrap();

        let small = fit_within(&image, 50).unwrap();
        let same = fit_within(&image, 400).unwrap();

        assert_eq!((small.width(), small.height()), (50, 25));
        assert_eq!((same.width(), same.height()), (200, 100));
    }

    #[test]
    fn correlation_with_centre_tap_is_identity() {
        let plane: Vec<f32> = (0..30).map(|i| i as f32).collect();
        let mut kernel = [0.0f32; 9];
        kernel[4] = 1.0;

        let out = correlate_plane(&plane, 6, 5, &kernel, 3, Edge::Wrap);

        assert_close(&out, &plane, 1e-6);
    }

    #[test]
    fn correlation_is_not_flipped() {
        let plane = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let kernel = [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0];

        let out = correlate_plane(&plane, 5, 1, &kernel, 3, Edge::Clamp);

        assert_close(&out, &[1.0, 2.0, 2.0, 2.0, 1.0], 1e-6);
    }

    #[test]
    fn rotating_by_zero_degrees_keeps_the_image() {
        let image = Image::from_fn(9, 7, 3, |x, y, c| (x * 3 + y * 5 + c) as f32 * 0.01).unwrap();

        let rotated = rotate_reflect(&image, 0.0).unwrap();

        assert_close(rotated.data(), image.data(), 1e-5);
    }

    #[test]
    fn rotating_by_a_quarter_turn_moves_content() {
        let image = Image::from_fn(16, 16, 1, |x, _, _| x as f32 / 15.0).unwrap();

        let rotated = rotate_reflect(&image, 90.0).unwrap();
        let top = rotated.pixel(8, 2)[0];
        let bottom = rotated.pixel(8, 13)[0];

        assert!((top - bottom).abs() > 0.5, "a horizontal ramp should become vertical");
    }
}
