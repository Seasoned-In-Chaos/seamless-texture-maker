//! Checking how well a texture tiles: visible seam lines and repetition.
//!
//! Both checks run on a small grey copy, so they take milliseconds at any
//! image size.
//!
//! - A seam is a straight line where the texture changes abruptly: the
//!   difference between neighbouring columns (or rows) jumps far above the
//!   differences around it. The edge where the texture wraps onto itself is
//!   checked like any other line, since that is where tiling shows.
//! - Repetition is the texture matching a shifted copy of itself, which
//!   patch-based methods can produce. Coarse shading is filtered out first,
//!   so a smooth gradient does not read as a repeat.

use crate::filter::{auto_blur_radius, gaussian_blur_plane, resize_area};
use crate::image::Image;

/// Longest side of the copy seams are checked on.
const SEAM_SIDE: usize = 256;
/// Side of the copy repetition is checked on.
const REPEAT_SIDE: usize = 64;
/// Shifts shorter than this fraction of the image are the texture's own
/// grain, not repetition.
const MIN_REPEAT_SHIFT: f32 = 0.1;
/// Correlation above which a shift counts as a repeat.
const REPEAT_THRESHOLD: f32 = 0.5;
/// Fine detail (grey levels, `0..=1`) below which repetition is not judged.
const MIN_DETAIL: f32 = 0.01;
/// Neighbouring lines a seam is compared with, on each side.
const SEAM_NEIGHBOURS: usize = 4;

/// The strongest repeat found.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Repeat {
    /// The shift, as fractions of the width and height in `-0.5..=0.5`.
    pub dx: f32,
    pub dy: f32,
    /// How closely the texture matches itself shifted, `0..=1`.
    pub similarity: f32,
}

/// What [`analyse`] found.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TilingReport {
    /// Size of the grid `columns`, `rows` and `heat` describe.
    pub width: usize,
    pub height: usize,
    /// Seam strength, `0..=1`, on the line before each column; column 0 is
    /// the edge where the texture wraps.
    pub columns: Vec<f32>,
    /// The same for rows; row 0 is the wrapping edge.
    pub rows: Vec<f32>,
    /// The strongest seam anywhere, `0..=1`.
    pub seam: f32,
    pub repeat: Option<Repeat>,
    /// Where repeated content sits, `0..=1` per grid cell, row by row.
    /// Empty when there is no repeat.
    pub heat: Vec<f32>,
}

impl TilingReport {
    /// The strongest seam on the wrapping edges, `0..=1`.
    pub fn edge_seam(&self) -> f32 {
        let first = |lines: &[f32]| lines.first().copied().unwrap_or(0.0);
        first(&self.columns).max(first(&self.rows))
    }
}

/// Checks `image` for seams and repetition.
pub fn analyse(image: &Image) -> TilingReport {
    let grey = grey_plane(image);
    let Some((width, height, grey)) = shrink(image.width(), image.height(), grey, SEAM_SIDE) else {
        return TilingReport::default();
    };
    let columns = line_seams(&grey, width, height, true);
    let rows = line_seams(&grey, width, height, false);
    let seam = columns.iter().chain(&rows).copied().fold(0.0, f32::max);
    let repeat = find_repeat(image);
    let heat = repeat.map(|r| repeat_heat(&grey, width, height, r)).unwrap_or_default();
    TilingReport { width, height, columns, rows, seam, repeat, heat }
}

fn grey_plane(image: &Image) -> Vec<f32> {
    image
        .data()
        .chunks(image.channels())
        .map(|p| match p {
            [r, g, b, ..] => 0.2126 * r + 0.7152 * g + 0.0722 * b,
            [v, ..] => *v,
            [] => 0.0,
        })
        .collect()
}

/// `plane` shrunk so its longest side is at most `side`; `None` when too
/// small to judge.
fn shrink(
    width: usize,
    height: usize,
    plane: Vec<f32>,
    side: usize,
) -> Option<(usize, usize, Vec<f32>)> {
    if width < 8 || height < 8 {
        return None;
    }
    let scale = (side as f32 / width.max(height) as f32).min(1.0);
    let (w, h) =
        (((width as f32 * scale) as usize).max(8), ((height as f32 * scale) as usize).max(8));
    let full = Image::from_plane(width, height, plane).ok()?;
    let small = resize_area(&full, w, h).ok()?;
    Some((w, h, small.into_data()))
}

/// Seam strength on each line between columns (`vertical`) or rows.
fn line_seams(grey: &[f32], width: usize, height: usize, vertical: bool) -> Vec<f32> {
    let (lines, along) = if vertical { (width, height) } else { (height, width) };
    let at = |line: usize, i: usize| {
        let index = if vertical { i * width + line } else { line * width + i };
        grey.get(index).copied().unwrap_or(0.0)
    };
    // Mean absolute change across each line, wrapping at the edge.
    let change: Vec<f32> = (0..lines)
        .map(|line| {
            let before = (line + lines - 1) % lines;
            (0..along).map(|i| (at(line, i) - at(before, i)).abs()).sum::<f32>() / along as f32
        })
        .collect();
    (0..lines)
        .map(|line| {
            let mut around: Vec<f32> = (1..=SEAM_NEIGHBOURS)
                .flat_map(|k| [change[(line + k) % lines], change[(line + lines - k) % lines]])
                .collect();
            around.sort_by(f32::total_cmp);
            let typical = around.get(around.len() / 2).copied().unwrap_or(0.0);
            let jump = change[line] - typical;
            // At least a 2% step, and several times the usual change.
            if jump < 0.02 {
                0.0
            } else {
                ((change[line] / typical.max(1e-4) - 2.0) / 3.0).clamp(0.0, 1.0)
            }
        })
        .collect()
}

/// The strongest self-match of the texture under a shift, if any.
fn find_repeat(image: &Image) -> Option<Repeat> {
    let (n, _, plane) = shrink_square(image)?;
    let fine = high_pass(&plane, n);
    // Too little fine detail to compare: standardising would blow up what
    // is left of a smooth gradient into a false pattern.
    if deviation(&fine) < MIN_DETAIL {
        return None;
    }
    let detail = standardised(&fine);
    let total = (n * n) as f32;
    let min_shift = MIN_REPEAT_SHIFT * n as f32;
    let mut best: Option<(usize, usize, f32)> = None;
    for sy in 0..n {
        for sx in 0..n {
            let (wx, wy) = (sx.min(n - sx) as f32, sy.min(n - sy) as f32);
            if (wx * wx + wy * wy).sqrt() < min_shift {
                continue;
            }
            let mut sum = 0.0;
            for y in 0..n {
                let (row, shifted) = (y * n, ((y + sy) % n) * n);
                for x in 0..n {
                    sum += detail[row + x] * detail[shifted + (x + sx) % n];
                }
            }
            let correlation = sum / total;
            if best.is_none_or(|(_, _, c)| correlation > c) {
                best = Some((sx, sy, correlation));
            }
        }
    }
    let (sx, sy, similarity) = best?;
    let signed = |s: usize| if s > n / 2 { s as f32 - n as f32 } else { s as f32 } / n as f32;
    (similarity > REPEAT_THRESHOLD).then_some(Repeat {
        dx: signed(sx),
        dy: signed(sy),
        similarity: similarity.min(1.0),
    })
}

/// A square grey copy for the repetition check.
fn shrink_square(image: &Image) -> Option<(usize, usize, Vec<f32>)> {
    if image.width() < 8 || image.height() < 8 {
        return None;
    }
    let full = Image::from_plane(image.width(), image.height(), grey_plane(image)).ok()?;
    let small = resize_area(&full, REPEAT_SIDE, REPEAT_SIDE).ok()?;
    Some((REPEAT_SIDE, REPEAT_SIDE, small.into_data()))
}

/// `plane` minus its coarse shading.
fn high_pass(plane: &[f32], side: usize) -> Vec<f32> {
    let sigma = 2.0;
    let blurred = gaussian_blur_plane(plane, side, side, sigma, auto_blur_radius(sigma));
    plane.iter().zip(&blurred).map(|(v, b)| v - b).collect()
}

fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len().max(1) as f32
}

fn deviation(values: &[f32]) -> f32 {
    let m = mean(values);
    (values.iter().map(|v| (v - m).powi(2)).sum::<f32>() / values.len().max(1) as f32).sqrt()
}

/// Zero mean and unit variance; flat input stays all zero.
fn standardised(values: &[f32]) -> Vec<f32> {
    let (m, d) = (mean(values), deviation(values));
    if d < 1e-6 {
        return vec![0.0; values.len()];
    }
    values.iter().map(|v| (v - m) / d).collect()
}

/// Where the content matches itself under `repeat`, on the seam grid.
fn repeat_heat(grey: &[f32], width: usize, height: usize, repeat: Repeat) -> Vec<f32> {
    let detail = standardised(grey);
    let sx = (repeat.dx * width as f32).round() as isize;
    let sy = (repeat.dy * height as f32).round() as isize;
    let at = |x: isize, y: isize| {
        let (x, y) =
            (x.rem_euclid(width as isize) as usize, y.rem_euclid(height as isize) as usize);
        detail.get(y * width + x).copied().unwrap_or(0.0)
    };
    let raw: Vec<f32> = (0..height as isize)
        .flat_map(|y| (0..width as isize).map(move |x| (x, y)))
        .map(|(x, y)| {
            let here = at(x, y);
            let forward = (here - at(x + sx, y + sy)).abs();
            let backward = (here - at(x - sx, y - sy)).abs();
            (1.0 - forward.min(backward) / 0.5).clamp(0.0, 1.0)
        })
        .collect();
    let smooth = gaussian_blur_plane(&raw, width, height, 2.0, auto_blur_radius(2.0));
    smooth.iter().map(|v| (v * repeat.similarity).clamp(0.0, 1.0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn noise(width: usize, height: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        (0..width * height).map(|_| rng.next_f32()).collect()
    }

    fn from_plane(width: usize, height: usize, plane: Vec<f32>) -> Image {
        Image::from_plane(width, height, plane).unwrap()
    }

    #[test]
    fn a_hard_step_shows_as_a_seam_on_its_line_and_the_wrapping_edge() {
        let image = Image::from_fn(256, 256, 3, |x, _, _| if x < 128 { 0.2 } else { 0.8 }).unwrap();

        let report = analyse(&image);

        assert!(report.columns[128] > 0.9, "the step: {}", report.columns[128]);
        assert!(report.columns[0] > 0.9, "the wrapping edge: {}", report.columns[0]);
        assert!(report.rows.iter().all(|s| *s == 0.0), "no horizontal seams");
        assert!(report.edge_seam() > 0.9);
    }

    #[test]
    fn a_smooth_tileable_gradient_has_no_seams() {
        // One period across, one down: it tiles, and no shift short of a
        // whole period matches it.
        let image = Image::from_fn(256, 256, 1, |x, y, _| {
            let (u, v) = (x as f32 / 256.0, y as f32 / 256.0);
            let tau = std::f32::consts::TAU;
            0.5 + 0.2 * (u * tau).sin() + 0.15 * (v * tau).cos()
        })
        .unwrap();

        let report = analyse(&image);

        assert!(report.seam < 0.1, "seam {}", report.seam);
        assert!(report.repeat.is_none());
    }

    #[test]
    fn a_texture_made_of_one_patch_twice_is_found_repeating() {
        // The left half's noise, copied into the right half.
        let half = noise(128, 256, 7);
        let image = from_plane(
            256,
            256,
            (0..256 * 256).map(|i| half[(i / 256) * 128 + (i % 256) % 128]).collect(),
        );

        let report = analyse(&image);

        let repeat = report.repeat.expect("a repeat");
        assert!((repeat.dx.abs() - 0.5).abs() < 0.02 && repeat.dy.abs() < 0.02, "{repeat:?}");
        assert!(repeat.similarity > 0.8);
        assert_eq!(report.heat.len(), report.width * report.height);
        assert!(report.heat.iter().sum::<f32>() / report.heat.len() as f32 > 0.5);
    }

    #[test]
    fn plain_noise_does_not_repeat() {
        let image = from_plane(256, 256, noise(256, 256, 3));

        assert!(analyse(&image).repeat.is_none());
    }

    #[test]
    fn flat_and_tiny_images_report_nothing() {
        let flat = analyse(&Image::filled(128, 128, 3, 0.5).unwrap());
        assert_eq!((flat.seam, flat.repeat), (0.0, None));

        let tiny = analyse(&Image::filled(4, 4, 3, 0.5).unwrap());
        assert_eq!(tiny, TilingReport::default());
    }
}
