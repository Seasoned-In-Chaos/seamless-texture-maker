//! Clone and heal brushes: painting pixels from one place onto another, as
//! Photoshop's Clone Stamp and Healing Brush do.
//!
//! A stroke is a run of dabs. Every dab reads from `source`, the image as it
//! was when the stroke began, so a stroke never copies its own fresh paint.
//! With `wrap`, pixels and samples past an edge continue on the opposite
//! side: a tileable texture stays tileable however a stroke crosses an edge.

use serde::{Deserialize, Serialize};

use crate::filter::{Edge, sample_bilinear};
use crate::image::Image;

/// How a dab turns source pixels into new ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum RetouchMode {
    /// Copy the source exactly.
    #[default]
    Clone,
    /// Copy the source's detail, matched to the colour and brightness of
    /// the place it lands, so the patch blends in.
    Heal,
}

/// The shape of a dab.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Brush {
    /// Radius in image pixels.
    pub radius: f32,
    /// Fraction of the radius painted at full strength, `0..=1`; the rest
    /// fades out. 1 is a hard edge.
    pub hardness: f32,
    /// Strength at the centre, `0..=1`.
    pub opacity: f32,
}

impl Default for Brush {
    fn default() -> Self {
        Self { radius: 40.0, hardness: 0.5, opacity: 1.0 }
    }
}

/// A pixel rectangle `[left, top, right, bottom)` inside the image.
pub type Region = [usize; 4];

/// Strength of a dab at `t` (distance over radius): full inside the hard
/// core, then a smooth fade to nothing at the rim.
fn falloff(t: f32, hardness: f32) -> f32 {
    if t >= 1.0 {
        return 0.0;
    }
    let core = hardness.clamp(0.0, 1.0);
    if t <= core || core >= 1.0 {
        return 1.0;
    }
    let u = (t - core) / (1.0 - core);
    1.0 - u * u * (3.0 - 2.0 * u)
}

/// Weighted mean colour of `image` over a dab at (`cx`, `cy`).
fn dab_mean(image: &Image, cx: f32, cy: f32, brush: &Brush, edge: Edge) -> Vec<f32> {
    let channels = image.channels();
    let (mut sum, mut total) = (vec![0.0f32; channels], 0.0f32);
    let mut pixel = vec![0.0f32; channels];
    let r = brush.radius;
    let step = (r / 8.0).max(1.0);
    let mut y = -r;
    while y <= r {
        let mut x = -r;
        while x <= r {
            let w = falloff((x * x + y * y).sqrt() / r, brush.hardness);
            if w > 0.0 {
                pixel.iter_mut().for_each(|p| *p = 0.0);
                sample_bilinear(image, cx + x - 0.5, cy + y - 0.5, edge, &mut pixel);
                sum.iter_mut().zip(&pixel).for_each(|(s, p)| *s += p * w);
                total += w;
            }
            x += step;
        }
        y += step;
    }
    sum.iter().map(|s| s / total.max(1e-6)).collect()
}

/// Paints one dab centred on (`cx`, `cy`) in image pixels, reading from
/// `source` at the same place moved by (`dx`, `dy`). `target` and `source`
/// must be the same size. Returns the pixel regions it changed (more than
/// one when it wrapped across an edge).
#[allow(clippy::too_many_arguments)]
pub fn dab(
    target: &mut Image,
    source: &Image,
    (cx, cy): (f32, f32),
    (dx, dy): (f32, f32),
    brush: &Brush,
    mode: RetouchMode,
    wrap: bool,
) -> Vec<Region> {
    let (width, height, channels) = (target.width(), target.height(), target.channels());
    let same_shape =
        (source.width(), source.height(), source.channels()) == (width, height, channels);
    let usable =
        [cx, cy, dx, dy, brush.radius, brush.opacity, brush.hardness].iter().all(|v| v.is_finite());
    let opacity = brush.opacity.clamp(0.0, 1.0);
    if !same_shape || !usable || brush.radius < 0.5 || opacity <= 0.0 {
        return Vec::new();
    }
    let edge = if wrap { Edge::Wrap } else { Edge::Clamp };
    // Healing moves the source's colour onto the destination's, keeping its detail.
    let shift: Vec<f32> = match mode {
        RetouchMode::Clone => vec![0.0; channels],
        RetouchMode::Heal => {
            let here = dab_mean(source, cx, cy, brush, edge);
            let there = dab_mean(source, cx + dx, cy + dy, brush, edge);
            let colors = target.color_channels();
            here.iter()
                .zip(&there)
                .enumerate()
                .map(|(c, (h, t))| if c < colors { h - t } else { 0.0 })
                .collect()
        }
    };
    let r = brush.radius;
    let (x0, x1) = window(cx, r, width, wrap);
    let (y0, y1) = window(cy, r, height, wrap);
    let mut sample = vec![0.0f32; channels];
    let mut painted_any = false;
    for py in y0..y1 {
        for px in x0..x1 {
            let inside = (0..width as isize).contains(&px) && (0..height as isize).contains(&py);
            if !wrap && !inside {
                continue;
            }
            let (fx, fy) = (px as f32 + 0.5 - cx, py as f32 + 0.5 - cy);
            let weight = falloff((fx * fx + fy * fy).sqrt() / r, brush.hardness) * opacity;
            if weight <= 0.0 {
                continue;
            }
            let (tx, ty) = (Edge::Wrap.resolve(px, width), Edge::Wrap.resolve(py, height));
            sample.iter_mut().for_each(|s| *s = 0.0);
            sample_bilinear(source, px as f32 + dx, py as f32 + dy, edge, &mut sample);
            let start = (ty * width + tx) * channels;
            let Some(pixel) = target.data_mut().get_mut(start..start + channels) else { continue };
            for ((value, from), lift) in pixel.iter_mut().zip(&sample).zip(&shift) {
                let painted = (from + lift).clamp(0.0, 1.0);
                *value = *value * (1.0 - weight) + painted * weight;
            }
            painted_any = true;
        }
    }
    if !painted_any {
        return Vec::new();
    }
    let (columns, rows) = (spans(x0, x1, width, wrap), spans(y0, y1, height, wrap));
    rows.iter()
        .flat_map(|&(top, bottom)| {
            columns.iter().map(move |&(left, right)| [left, top, right, bottom])
        })
        .collect()
}

/// The pixels a dab at `centre` with radius `r` visits on an axis of `len`.
/// Without wrap, the part inside the image. With wrap, at most `len`
/// pixels centred on the dab, so a brush wider than the image paints each
/// pixel once, nearest its centre.
fn window(centre: f32, r: f32, len: usize, wrap: bool) -> (isize, isize) {
    let (lo, hi) = ((centre - r).floor() as isize, (centre + r).ceil() as isize);
    let n = len as isize;
    if !wrap {
        return (lo.max(0), hi.min(n));
    }
    if hi - lo <= n {
        return (lo, hi);
    }
    let start = centre.floor() as isize - n / 2;
    (start, start + n)
}

/// The part of `[lo, hi)` on an axis of `len` pixels, as at most two
/// in-image spans: wrapped pieces with `wrap`, else clipped to the image.
fn spans(lo: isize, hi: isize, len: usize, wrap: bool) -> Vec<(usize, usize)> {
    let n = len as isize;
    if !wrap {
        let (lo, hi) = (lo.clamp(0, n), hi.clamp(0, n));
        return if lo < hi { vec![(lo as usize, hi as usize)] } else { Vec::new() };
    }
    if hi - lo >= n {
        return vec![(0, len)];
    }
    let (start, end) = (lo.rem_euclid(n), lo.rem_euclid(n) + (hi - lo));
    if end <= n {
        vec![(start as usize, end as usize)]
    } else {
        vec![(start as usize, len), (0, (end - n) as usize)]
    }
}

/// Dab centres from `from` to `to`, spaced a quarter of the radius apart,
/// as a stroke lays them down. `from` itself is not included.
pub fn stroke_dabs(from: (f32, f32), to: (f32, f32), radius: f32) -> Vec<(f32, f32)> {
    let (dx, dy) = (to.0 - from.0, to.1 - from.1);
    let length = (dx * dx + dy * dy).sqrt();
    if !length.is_finite() || length <= 0.0 {
        return Vec::new();
    }
    let spacing = (radius * 0.25).max(1.0);
    // Bounded, so a wild pointer jump cannot queue millions of dabs.
    let count = ((length / spacing).ceil() as usize).clamp(1, 4096);
    (1..=count)
        .map(|i| {
            let t = i as f32 / count as f32;
            (from.0 + dx * t, from.1 + dy * t)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A left half of 0.2 and a right half of 0.8.
    fn halves(size: usize) -> Image {
        Image::from_fn(size, size, 3, |x, _, _| if x < size / 2 { 0.2 } else { 0.8 }).unwrap()
    }

    fn hard(radius: f32) -> Brush {
        Brush { radius, hardness: 1.0, opacity: 1.0 }
    }

    #[test]
    fn clone_copies_the_source_inside_a_hard_dab_and_nothing_outside() {
        let source = halves(64);
        let mut target = source.clone();

        dab(&mut target, &source, (16.0, 32.0), (32.0, 0.0), &hard(6.0), RetouchMode::Clone, false);

        assert!((target.pixel(16, 32)[0] - 0.8).abs() < 1e-5, "centre now copies the right half");
        assert_eq!(target.pixel(16, 20)[0], 0.2, "outside the radius is untouched");
    }

    #[test]
    fn opacity_blends_with_what_was_there() {
        let source = halves(64);
        let mut target = source.clone();
        let half = Brush { opacity: 0.5, ..hard(6.0) };

        dab(&mut target, &source, (16.0, 32.0), (32.0, 0.0), &half, RetouchMode::Clone, false);

        assert!((target.pixel(16, 32)[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn a_soft_dab_fades_towards_its_rim() {
        let source = halves(64);
        let mut target = source.clone();
        let soft = Brush { hardness: 0.0, ..hard(10.0) };

        dab(&mut target, &source, (16.0, 32.0), (32.0, 0.0), &soft, RetouchMode::Clone, false);

        let (centre, rim) = (target.pixel(16, 32)[0], target.pixel(24, 32)[0]);
        assert!(centre > rim && rim > 0.2, "centre {centre}, rim {rim}");
    }

    #[test]
    fn heal_keeps_the_source_detail_at_the_destination_brightness() {
        // Source: a bright checker. Destination area: flat dark.
        let source = Image::from_fn(128, 64, 3, |x, y, _| {
            if x < 64 {
                0.2
            } else if (x / 2 + y / 2) % 2 == 0 {
                0.9
            } else {
                0.7
            }
        })
        .unwrap();
        let mut target = source.clone();

        dab(&mut target, &source, (32.0, 32.0), (64.0, 0.0), &hard(12.0), RetouchMode::Heal, false);

        let (a, b) = (target.pixel(32, 32)[0], target.pixel(34, 32)[0]);
        assert!((a - b).abs() > 0.1, "the checker detail came across");
        assert!(((a + b) / 2.0 - 0.2).abs() < 0.08, "at the dark area's brightness: {a}, {b}");
    }

    #[test]
    fn with_wrap_a_dab_continues_across_the_edge() {
        let source = Image::filled(64, 64, 3, 0.2).unwrap();
        let mut target = source.clone();
        let other = Image::filled(64, 64, 3, 0.9).unwrap();

        let regions =
            dab(&mut target, &other, (1.0, 32.0), (0.0, 0.0), &hard(5.0), RetouchMode::Clone, true);

        assert!((target.pixel(62, 32)[0] - 0.9).abs() < 1e-5, "painted on the far side");
        assert_eq!(regions.len(), 2, "one region on each side: {regions:?}");
    }

    #[test]
    fn without_wrap_nothing_past_the_edge_changes() {
        let source = Image::filled(64, 64, 3, 0.2).unwrap();
        let other = Image::filled(64, 64, 3, 0.9).unwrap();
        let mut target = source.clone();

        dab(&mut target, &other, (1.0, 32.0), (0.0, 0.0), &hard(5.0), RetouchMode::Clone, false);

        assert_eq!(target.pixel(62, 32)[0], 0.2);
    }

    #[test]
    fn hostile_input_changes_nothing() {
        let source = halves(64);
        let mut target = source.clone();
        let brushes = [
            Brush { radius: f32::NAN, ..hard(5.0) },
            Brush { radius: 0.0, ..hard(5.0) },
            Brush { opacity: 0.0, ..hard(5.0) },
        ];

        for brush in brushes {
            assert!(
                dab(
                    &mut target,
                    &source,
                    (10.0, 10.0),
                    (5.0, 0.0),
                    &brush,
                    RetouchMode::Clone,
                    true
                )
                .is_empty()
            );
        }
        assert!(
            dab(
                &mut target,
                &source,
                (f32::INFINITY, 1.0),
                (0.0, 0.0),
                &hard(5.0),
                RetouchMode::Heal,
                true
            )
            .is_empty()
        );
        let smaller = Image::filled(32, 32, 3, 0.0).unwrap();
        assert!(
            dab(
                &mut target,
                &smaller,
                (10.0, 10.0),
                (0.0, 0.0),
                &hard(5.0),
                RetouchMode::Clone,
                true
            )
            .is_empty()
        );
        assert_eq!(target, source);
    }

    #[test]
    fn a_huge_brush_covers_each_pixel_once() {
        let source = Image::filled(64, 64, 3, 0.5).unwrap();
        let mut target = source.clone();

        let regions = dab(
            &mut target,
            &source,
            (32.0, 32.0),
            (0.0, 0.0),
            &hard(500.0),
            RetouchMode::Clone,
            true,
        );

        assert_eq!(regions, vec![[0, 0, 64, 64]]);
        let mut clipped = source.clone();
        let other = Image::filled(64, 64, 3, 0.9).unwrap();
        dab(
            &mut clipped,
            &other,
            (32.0, 32.0),
            (0.0, 0.0),
            &hard(500.0),
            RetouchMode::Clone,
            false,
        );
        assert!(clipped.data().iter().all(|v| (v - 0.9).abs() < 1e-5), "without wrap too");
    }

    #[test]
    fn stroke_dabs_are_spaced_and_bounded() {
        let dabs = stroke_dabs((0.0, 0.0), (100.0, 0.0), 40.0);
        assert_eq!(dabs.len(), 10);
        assert_eq!(dabs.last(), Some(&(100.0, 0.0)));

        assert!(stroke_dabs((0.0, 0.0), (0.0, 0.0), 40.0).is_empty());
        assert!(stroke_dabs((0.0, 0.0), (f32::NAN, 0.0), 40.0).is_empty());
        assert_eq!(stroke_dabs((0.0, 0.0), (1e9, 0.0), 1.0).len(), 4096);
    }
}
