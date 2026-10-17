//! Clone Stamp, Healing Brush and Spot Healing Brush: painting pixels from
//! one place onto another, as Photoshop's do.
//!
//! A stroke is a run of dabs laid down every `spacing` along the pointer's
//! path. Dabs build up the stroke's coverage the way a brush's flow does:
//! each adds `tip × flow × (1 − coverage)`, so slow strokes and overlaps
//! deepen it towards full and never past it. The stroke shows at
//! `coverage × opacity`: Opacity caps the stroke, Flow sets how fast it gets
//! there. A pixel always ends up `before × (1 − k) + paint × k`, where
//! `before` is the image as the stroke found it, so a stroke never copies
//! its own fresh paint.
//!
//! - Clone paints the source exactly, dab by dab.
//! - Heal shows the cloned texture while drawing; on release the whole
//!   stroke is solved at once as a Poisson (gradient-domain) problem, so the
//!   source's detail takes the colour and brightness of where it lands with
//!   no seams between dabs, as Photoshop's Healing Brush does.
//! - Spot heal needs no source: on release the stroke is filled from its
//!   surroundings, by content-aware completion or the best-matching nearby
//!   patch, then blended in the same way.
//!
//! With `wrap`, pixels and samples past an edge continue on the opposite
//! side: a tileable texture stays tileable however a stroke crosses an edge.
//!
//! Ported in approach from PhotoCraft's retouching tools (MIT OR Apache-2.0).

use serde::{Deserialize, Serialize};

use crate::filter::{Edge, sample_bilinear};
use crate::image::Image;
use crate::inpaint::{CompleteParams, best_offset, complete};
use crate::poisson::{membrane_fill, seamless_clone};

/// What a stroke paints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum RetouchMode {
    /// Copy the source exactly.
    Clone,
    /// Copy the source's detail, matched to the colour and brightness of
    /// the place it lands, so the patch blends in.
    #[default]
    Heal,
    /// Fill the stroke from its surroundings: no source needed.
    SpotHeal,
}

/// How Spot heal finds what to paint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SpotMethod {
    /// Rebuild the area patch by patch from everything around it.
    #[default]
    ContentAware,
    /// Copy the one nearby area whose surroundings match best.
    ProximityMatch,
}

/// The brush, as Photoshop's options bar sets it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Brush {
    /// Radius in image pixels.
    pub radius: f32,
    /// Fraction of the radius painted at full strength, `0..=1`; the rest
    /// fades out. 1 is a hard edge, 0 the softest.
    pub hardness: f32,
    /// The most a stroke can cover a pixel, `0..=1`.
    pub opacity: f32,
    /// How much each dab adds, `0..=1`: low flow builds up gradually.
    pub flow: f32,
    /// Distance between dabs as a fraction of the diameter, `0.01..=2`.
    pub spacing: f32,
}

impl Default for Brush {
    fn default() -> Self {
        Self { radius: 40.0, hardness: 0.5, opacity: 1.0, flow: 1.0, spacing: 0.25 }
    }
}

impl Brush {
    /// The distance between dabs in image pixels.
    pub fn spacing_px(&self) -> f32 {
        (self.radius * 2.0 * self.spacing.clamp(0.01, 2.0)).max(0.5)
    }

    fn usable(&self) -> bool {
        [self.radius, self.hardness, self.opacity, self.flow, self.spacing]
            .iter()
            .all(|v| v.is_finite())
            && self.radius >= 0.5
            && self.flow > 0.0
    }
}

/// A pixel rectangle `[left, top, right, bottom)` inside the image.
pub type Region = [usize; 4];

/// Coverage is kept in 16 bits per pixel: exact enough, and half the size of `f32`.
const FULL: f32 = 65535.0;
/// Unmasked pixels kept around a healed stroke, so the solve has a boundary.
const HEAL_MARGIN: isize = 2;

/// Strength of the tip at distance `d` from its centre: full inside the
/// hard core, a smooth fade to the rim, and half a pixel of anti-aliasing.
pub fn tip(d: f32, radius: f32, hardness: f32) -> f32 {
    if d >= radius + 0.5 {
        return 0.0;
    }
    let edge = (radius + 0.5 - d).clamp(0.0, 1.0);
    let inner = radius * hardness.clamp(0.0, 1.0);
    let falloff = if d <= inner || radius - inner < 1e-3 {
        1.0
    } else {
        let t = ((d - inner) / (radius - inner)).clamp(0.0, 1.0);
        1.0 - t * t * (3.0 - 2.0 * t)
    };
    edge * falloff
}

/// One stroke's coverage. Make one per stroke, at the image's size.
pub struct Stroke {
    width: usize,
    height: usize,
    wrap: bool,
    coverage: Vec<u16>,
    /// The dabs' extent `[x0, y0, x1, y1)` in unwrapped image pixels.
    bounds: Option<[isize; 4]>,
}

impl Stroke {
    pub fn new(width: usize, height: usize, wrap: bool) -> Self {
        // Zeroed memory is mapped lazily, so only the pixels a stroke touches cost anything.
        Self { width, height, wrap, coverage: vec![0; width * height], bounds: None }
    }

    /// Whether any dab has landed.
    pub fn is_empty(&self) -> bool {
        self.bounds.is_none()
    }

    fn resolve(&self, x: isize, y: isize) -> Option<usize> {
        if self.wrap {
            let (x, y) = (Edge::Wrap.resolve(x, self.width), Edge::Wrap.resolve(y, self.height));
            return Some(y * self.width + x);
        }
        let inside =
            (0..self.width as isize).contains(&x) && (0..self.height as isize).contains(&y);
        inside.then(|| y as usize * self.width + x as usize)
    }

    fn coverage(&self, slot: usize) -> f32 {
        f32::from(self.coverage[slot]) / FULL
    }

    /// Adds one dab centred on (`cx`, `cy`) to the coverage. Returns the
    /// pixel regions it touched (more than one when it wrapped).
    pub fn dab(&mut self, (cx, cy): (f32, f32), brush: &Brush) -> Vec<Region> {
        if !brush.usable() || !cx.is_finite() || !cy.is_finite() || self.width == 0 {
            return Vec::new();
        }
        let r = brush.radius + 0.5;
        let (x0, x1) = window(cx, r, self.width, self.wrap);
        let (y0, y1) = window(cy, r, self.height, self.wrap);
        if x0 >= x1 || y0 >= y1 {
            return Vec::new();
        }
        let flow = brush.flow.clamp(0.0, 1.0);
        let mut touched = false;
        for py in y0..y1 {
            for px in x0..x1 {
                let (fx, fy) = (px as f32 + 0.5 - cx, py as f32 + 0.5 - cy);
                let v = tip((fx * fx + fy * fy).sqrt(), brush.radius, brush.hardness) * flow;
                let Some(slot) = (v > 0.0).then(|| self.resolve(px, py)).flatten() else {
                    continue;
                };
                let before = self.coverage(slot);
                let after = before + v * (1.0 - before);
                self.coverage[slot] = (after * FULL).round().clamp(1.0, FULL) as u16;
                touched = true;
            }
        }
        if !touched {
            return Vec::new();
        }
        let grown = match self.bounds {
            Some([a, b, c, d]) => [a.min(x0), b.min(y0), c.max(x1), d.max(y1)],
            None => [x0, y0, x1, y1],
        };
        self.bounds = Some(grown);
        regions(self.width, self.height, self.wrap, [x0, y0, x1, y1])
    }

    /// Every region the stroke has touched.
    pub fn regions(&self) -> Vec<Region> {
        self.bounds.map_or_else(Vec::new, |b| regions(self.width, self.height, self.wrap, b))
    }

    /// Redraws `regions` of `target` as `before` with `paint` laid over it
    /// through the coverage at `opacity`. `paint` fills the pixel at the
    /// given image coordinates.
    pub fn composite(
        &self,
        target: &mut Image,
        before: &Image,
        regions: &[Region],
        opacity: f32,
        paint: impl Fn(usize, usize, &mut [f32]),
    ) {
        let channels = target.channels();
        if !self.fits(target) || !self.fits(before) {
            return;
        }
        let colors = target.color_channels();
        let opacity = opacity.clamp(0.0, 1.0);
        let mut sample = vec![0.0f32; channels];
        for &[left, top, right, bottom] in regions {
            for y in top..bottom.min(self.height) {
                for x in left..right.min(self.width) {
                    let slot = y * self.width + x;
                    let k = self.coverage(slot) * opacity;
                    let start = slot * channels;
                    let original = &before.data()[start..start + channels];
                    let pixel = &mut target.data_mut()[start..start + channels];
                    if k <= 0.0 {
                        pixel.copy_from_slice(original);
                        continue;
                    }
                    sample.fill(0.0);
                    paint(x, y, &mut sample);
                    for c in 0..channels {
                        let painted =
                            if c < colors { sample[c].clamp(0.0, 1.0) } else { sample[c] };
                        pixel[c] = original[c] + (painted - original[c]) * k;
                    }
                }
            }
        }
    }

    /// Redraws `regions` with the source at the offset (`dx`, `dy`): the
    /// Clone Stamp, and the Healing Brush while it is being drawn.
    pub fn composite_clone(
        &self,
        target: &mut Image,
        before: &Image,
        regions: &[Region],
        (dx, dy): (f32, f32),
        opacity: f32,
    ) {
        let edge = if self.wrap { Edge::Wrap } else { Edge::Clamp };
        self.composite(target, before, regions, opacity, |x, y, out| {
            sample_bilinear(before, x as f32 + dx, y as f32 + dy, edge, out);
        });
    }

    fn fits(&self, image: &Image) -> bool {
        (image.width(), image.height()) == (self.width, self.height)
    }

    /// The window the stroke's solve runs on: its bounds grown by `margin`,
    /// clipped to the image (without wrap) or to one image's width (with).
    fn solve_window(&self, margin: isize) -> Option<[isize; 4]> {
        let [x0, y0, x1, y1] = self.bounds?;
        let axis = |lo: isize, hi: isize, len: usize| {
            let (lo, hi, n) = (lo - margin, hi + margin, len as isize);
            if !self.wrap {
                (lo.max(0), hi.min(n))
            } else if hi - lo > n {
                (0, n)
            } else {
                (lo, hi)
            }
        };
        let (x0, x1) = axis(x0, x1, self.width);
        let (y0, y1) = axis(y0, y1, self.height);
        (x0 < x1 && y0 < y1).then_some([x0, y0, x1, y1])
    }

    /// Gathers the window's pixels of `image` (wrapped as the stroke is) and the stroke's
    /// coverage there.
    fn gather(&self, image: &Image, [x0, y0, x1, y1]: [isize; 4]) -> Gathered {
        let (w, h, channels) = ((x1 - x0) as usize, (y1 - y0) as usize, image.channels());
        let mut pixels = vec![0.0f32; w * h * channels];
        let mut coverage = vec![0.0f32; w * h];
        let mut slots = vec![None; w * h];
        for wy in 0..h {
            for wx in 0..w {
                let i = wy * w + wx;
                let Some(slot) = self.resolve(x0 + wx as isize, y0 + wy as isize) else {
                    continue;
                };
                pixels[i * channels..(i + 1) * channels]
                    .copy_from_slice(&image.data()[slot * channels..(slot + 1) * channels]);
                coverage[i] = self.coverage(slot);
                slots[i] = Some(slot);
            }
        }
        Gathered { origin: (x0, y0), w, h, pixels, coverage, slots }
    }

    /// Writes `paint` (window pixels) over `target` through the coverage at `opacity`.
    fn lay_down(
        &self,
        target: &mut Image,
        before: &Image,
        window: &Gathered,
        paint: &[f32],
        opacity: f32,
    ) {
        let channels = target.channels();
        let colors = target.color_channels();
        let opacity = opacity.clamp(0.0, 1.0);
        for (i, slot) in window.slots.iter().enumerate() {
            let Some(slot) = *slot else { continue };
            let k = window.coverage[i] * opacity;
            if k <= 0.0 {
                continue;
            }
            for c in 0..channels {
                let original = before.data()[slot * channels + c];
                let painted = paint[i * channels + c];
                let painted = if c < colors { painted.clamp(0.0, 1.0) } else { painted };
                target.data_mut()[slot * channels + c] = original + (painted - original) * k;
            }
        }
    }

    /// Finishes a Healing Brush stroke: the source at the offset, fitted to
    /// where it lands by one Poisson solve over the whole stroke, laid over
    /// `before` into `target`. Returns the regions it changed.
    pub fn heal(
        &self,
        target: &mut Image,
        before: &Image,
        (dx, dy): (f32, f32),
        opacity: f32,
    ) -> Vec<Region> {
        if !self.fits(target) || !self.fits(before) || !dx.is_finite() || !dy.is_finite() {
            return Vec::new();
        }
        let Some(window) = self.solve_window(HEAL_MARGIN) else { return Vec::new() };
        let here = self.gather(before, window);
        let edge = if self.wrap { Edge::Wrap } else { Edge::Clamp };
        let channels = before.channels();
        let mut from = vec![0.0f32; here.pixels.len()];
        for wy in 0..here.h {
            for wx in 0..here.w {
                let at = (wy * here.w + wx) * channels;
                let (px, py) = (here.origin.0 + wx as isize, here.origin.1 + wy as isize);
                sample_bilinear(
                    before,
                    px as f32 + dx,
                    py as f32 + dy,
                    edge,
                    &mut from[at..at + channels],
                );
            }
        }
        let mask: Vec<bool> = here.coverage.iter().map(|c| *c > 0.0).collect();
        let shape = (here.w, here.h, channels, before.color_channels());
        let healed = seamless_clone(shape, &from, &here.pixels, &mask);
        self.lay_down(target, before, &here, &healed, opacity);
        self.regions()
    }

    /// Finishes a Spot Healing stroke: the stroke filled from what surrounds
    /// it by `method`, blended in by a Poisson solve, laid over `before`
    /// into `target`. `size` (the brush diameter) sets how far it looks.
    pub fn spot_heal(
        &self,
        target: &mut Image,
        before: &Image,
        method: SpotMethod,
        size: f32,
        opacity: f32,
    ) -> Vec<Region> {
        if !self.fits(target) || !self.fits(before) || !size.is_finite() {
            return Vec::new();
        }
        let margin = size.max(8.0).ceil() as isize + 8;
        let Some(window) = self.solve_window(margin) else { return Vec::new() };
        let around = self.gather(before, window);
        let (w, h, channels) = (around.w, around.h, before.channels());
        let hole: Vec<bool> = around.coverage.iter().map(|c| *c > 0.0).collect();
        if !hole.iter().any(|m| *m) {
            return Vec::new();
        }
        // Fill a slightly larger area than is blended, so the solve's boundary carries filled
        // values that differ from the untouched surroundings: that mismatch is what the
        // membrane corrects.
        let domain = dilate(w, h, &hole, 2);
        let blend = dilate(w, h, &hole, 1);
        let content_aware = |img: &[f32]| {
            complete(w, h, channels, img, &domain, &CompleteParams::default())
                .unwrap_or_else(|| membrane_fill(w, h, channels, img, &domain))
        };
        let filled = match method {
            SpotMethod::ContentAware => content_aware(&around.pixels),
            SpotMethod::ProximityMatch => {
                let ring = (size / 8.0).clamp(3.0, 16.0) as usize;
                match best_offset(w, h, channels, &around.pixels, &domain, ring, margin as i32) {
                    Some(offset) => shifted(w, h, channels, &around.pixels, &domain, offset),
                    None => content_aware(&around.pixels),
                }
            }
        };
        let shape = (w, h, channels, before.color_channels());
        let healed = seamless_clone(shape, &filled, &around.pixels, &blend);
        self.lay_down(target, before, &around, &healed, opacity);
        self.regions()
    }
}

/// Pixels of a solve window, with where each one lives in the image.
struct Gathered {
    origin: (isize, isize),
    w: usize,
    h: usize,
    pixels: Vec<f32>,
    coverage: Vec<f32>,
    /// The image pixel each window pixel is, if it is one.
    slots: Vec<Option<usize>>,
}

/// `img` with the `domain` pixels copied from `offset` away.
fn shifted(
    w: usize,
    h: usize,
    ch: usize,
    img: &[f32],
    domain: &[bool],
    (dx, dy): (i32, i32),
) -> Vec<f32> {
    let mut out = img.to_vec();
    for y in 0..h {
        for x in 0..w {
            let (sx, sy) = (x as i32 + dx, y as i32 + dy);
            if domain[y * w + x] && (0..w as i32).contains(&sx) && (0..h as i32).contains(&sy) {
                let (i, j) = ((y * w + x) * ch, (sy as usize * w + sx as usize) * ch);
                out.copy_within(j..j + ch, i);
            }
        }
    }
    out
}

/// `mask` grown by `r` pixels (a square neighbourhood).
fn dilate(w: usize, h: usize, mask: &[bool], r: usize) -> Vec<bool> {
    let mut out = mask.to_vec();
    for y in 0..h {
        for x in 0..w {
            if !mask[y * w + x] {
                continue;
            }
            for ny in y.saturating_sub(r)..(y + r + 1).min(h) {
                for nx in x.saturating_sub(r)..(x + r + 1).min(w) {
                    out[ny * w + nx] = true;
                }
            }
        }
    }
    out
}

/// The pixels a dab at `centre` reaching `r` visits on an axis of `len`.
/// Without wrap, the part inside the image. With wrap, at most `len` pixels
/// centred on the dab, so a brush wider than the image paints each pixel once.
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

/// `[x0, y0, x1, y1)` in unwrapped pixels as in-image regions: wrapped
/// pieces with `wrap`, else clipped.
fn regions(width: usize, height: usize, wrap: bool, [x0, y0, x1, y1]: [isize; 4]) -> Vec<Region> {
    let (columns, rows) = (spans(x0, x1, width, wrap), spans(y0, y1, height, wrap));
    rows.iter()
        .flat_map(|&(top, bottom)| {
            columns.iter().map(move |&(left, right)| [left, top, right, bottom])
        })
        .collect()
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

/// Dab centres from the last dab at `from` towards `to`, exactly `spacing`
/// apart; the leftover distance carries into the next call because the
/// caller continues from the last dab returned. Bounded, so a wild pointer
/// jump cannot queue millions of dabs.
pub fn dabs_towards(from: (f32, f32), to: (f32, f32), spacing: f32) -> Vec<(f32, f32)> {
    let (dx, dy) = (to.0 - from.0, to.1 - from.1);
    let length = (dx * dx + dy * dy).sqrt();
    if !length.is_finite() || !spacing.is_finite() || spacing <= 0.0 || length < spacing {
        return Vec::new();
    }
    let count = ((length / spacing).floor() as usize).min(4096);
    let step = if count == 4096 { length / 4096.0 } else { spacing };
    (1..=count)
        .map(|i| {
            let t = i as f32 * step / length;
            (from.0 + dx * t, from.1 + dy * t)
        })
        .collect()
}

/// Photoshop's stroke Smoothing: the brush trails the pointer, moving
/// `1 − amount` of the way each step, so a shaky hand draws a calm line.
/// With catch-up, the brush keeps closing in while the pointer rests, and
/// a stroke ends where the pointer did.
#[derive(Debug, Clone)]
pub struct Smoother {
    amount: f32,
    pos: Option<(f32, f32)>,
    last: Option<((f32, f32), f64)>,
}

/// One smoothing step per this many seconds of pointer time.
const SMOOTHING_STEP: f64 = 1.0 / 60.0;

impl Smoother {
    /// `amount` in `0..=1`; 0 follows the pointer exactly.
    pub fn new(amount: f32) -> Self {
        let amount = if amount.is_finite() { amount.clamp(0.0, 0.95) } else { 0.0 };
        Self { amount, pos: None, last: None }
    }

    /// Feeds the pointer at `time` seconds; returns brush positions to paint through.
    pub fn push(&mut self, point: (f32, f32), time: f64) -> Vec<(f32, f32)> {
        let previous = self.last.replace((point, time));
        let Some(pos) = self.pos.filter(|_| self.amount > 0.0) else {
            self.pos = Some(point);
            return vec![point];
        };
        let a = 1.0 - self.amount;
        let steps = match previous {
            Some((_, t)) if time > t => {
                ((time - t) / SMOOTHING_STEP).round().clamp(1.0, 64.0) as usize
            }
            _ => 1,
        };
        let mut cur = pos;
        let mut out = Vec::with_capacity(steps);
        for _ in 0..steps {
            cur = (cur.0 + (point.0 - cur.0) * a, cur.1 + (point.1 - cur.1) * a);
            out.push(cur);
        }
        self.pos = Some(cur);
        out
    }

    /// End of the stroke: the brush finishes where the pointer is.
    pub fn finish(&mut self) -> Option<(f32, f32)> {
        let (point, _) = self.last?;
        let pos = self.pos?;
        (self.amount > 0.0 && pos != point).then(|| {
            self.pos = Some(point);
            point
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A left half of 0.2 and a right half of 0.8.
    fn halves(size: usize) -> Image {
        Image::from_fn(size, size, 3, |x, _, _| if x < size / 2 { 0.2 } else { 0.8 }).unwrap()
    }

    fn hard(radius: f32) -> Brush {
        Brush { radius, hardness: 1.0, ..Brush::default() }
    }

    /// A clone stroke of dabs at `centres` from `source` shifted by `by`.
    fn clone(
        source: &Image,
        centres: &[(f32, f32)],
        by: (f32, f32),
        brush: &Brush,
        wrap: bool,
    ) -> Image {
        let mut stroke = Stroke::new(source.width(), source.height(), wrap);
        let mut target = source.clone();
        for centre in centres {
            let regions = stroke.dab(*centre, brush);
            stroke.composite_clone(&mut target, source, &regions, by, brush.opacity);
        }
        target
    }

    #[test]
    fn clone_copies_the_source_inside_a_hard_dab_and_nothing_outside() {
        let source = halves(64);

        let target = clone(&source, &[(16.0, 32.0)], (32.0, 0.0), &hard(6.0), false);

        assert!((target.pixel(16, 32)[0] - 0.8).abs() < 1e-5, "centre copies the right half");
        assert_eq!(target.pixel(16, 20)[0], 0.2, "outside the radius is untouched");
    }

    #[test]
    fn opacity_caps_a_stroke_however_many_dabs_overlap() {
        let source = halves(64);
        let half = Brush { opacity: 0.5, ..hard(6.0) };

        let target = clone(&source, &[(16.0, 32.0); 40], (32.0, 0.0), &half, false);

        assert!((target.pixel(16, 32)[0] - 0.5).abs() < 0.01, "{}", target.pixel(16, 32)[0]);
    }

    #[test]
    fn low_flow_builds_up_with_overlapping_dabs() {
        let source = halves(64);
        let light = Brush { flow: 0.2, ..hard(6.0) };

        let once = clone(&source, &[(16.0, 32.0)], (32.0, 0.0), &light, false);
        let many = clone(&source, &[(16.0, 32.0); 8], (32.0, 0.0), &light, false);

        let (a, b) = (once.pixel(16, 32)[0], many.pixel(16, 32)[0]);
        assert!((a - (0.2 + 0.6 * 0.2)).abs() < 0.01, "one dab adds the flow: {a}");
        assert!(b > a + 0.2 && b < 0.8, "more dabs build it up: {b}");
    }

    #[test]
    fn a_soft_tip_fades_towards_its_rim() {
        assert_eq!(tip(0.0, 10.0, 0.0), 1.0);
        assert!(tip(5.0, 10.0, 0.0) > tip(8.0, 10.0, 0.0));
        assert_eq!(tip(11.0, 10.0, 0.0), 0.0);
        assert_eq!(tip(7.0, 10.0, 1.0), 1.0, "hard inside the radius");
    }

    #[test]
    fn heal_keeps_the_source_detail_at_the_destination_brightness() {
        // Left: flat dark. Right: a bright checker, healed over the dark.
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
        let mut stroke = Stroke::new(128, 64, false);
        for x in [26.0, 32.0, 38.0] {
            stroke.dab((x, 32.0), &hard(10.0));
        }
        let mut target = source.clone();

        stroke.heal(&mut target, &source, (64.0, 0.0), 1.0);

        let (a, b) = (target.pixel(32, 32)[0], target.pixel(34, 32)[0]);
        assert!((a - b).abs() > 0.1, "the checker detail came across: {a} vs {b}");
        let mean: f32 = (26..38).map(|x| target.pixel(x, 32)[0]).sum::<f32>() / 12.0;
        assert!((mean - 0.2).abs() < 0.12, "at the dark area's brightness: {mean}");
        assert!((target.pixel(32, 15)[0] - 0.2).abs() < 1e-6, "nothing past the stroke");
    }

    #[test]
    fn a_healed_stroke_has_no_seams_between_its_dabs() {
        // A brightness ramp; heal a long stroke from the bright end onto the dark end.
        let source = Image::from_fn(160, 64, 3, |x, _, _| x as f32 / 160.0).unwrap();
        let mut stroke = Stroke::new(160, 64, false);
        let brush = hard(8.0);
        let start = (10.0, 32.0);
        stroke.dab(start, &brush);
        for centre in dabs_towards(start, (60.0, 32.0), brush.spacing_px()) {
            stroke.dab(centre, &brush);
        }
        let (mut cloned, mut healed) = (source.clone(), source.clone());
        stroke.composite_clone(&mut cloned, &source, &stroke.regions(), (90.0, 0.0), 1.0);
        stroke.heal(&mut healed, &source, (90.0, 0.0), 1.0);

        // Along the stroke the healed result follows the ramp; the clone jumps at its rim.
        let step = |image: &Image| (image.pixel(4, 32)[0] - image.pixel(1, 32)[0]).abs();
        assert!(step(&cloned) > 0.4, "{}", step(&cloned));
        assert!(step(&healed) < 0.06, "{}", step(&healed));
        let row: Vec<f32> = (10..60).map(|x| healed.pixel(x, 32)[0]).collect();
        let worst = row.windows(2).map(|p| (p[1] - p[0]).abs()).fold(0.0, f32::max);
        assert!(worst < 0.03, "smooth along the stroke: {worst}");
    }

    #[test]
    fn spot_heal_removes_a_blemish_without_a_source() {
        let clean = |x: usize, y: usize| 0.4 + 0.2 * (((x / 4) + (y / 4)) % 2) as f32;
        let mut source = Image::from_fn(96, 96, 3, |x, y, _| clean(x, y)).unwrap();
        for y in 44..52 {
            for x in 44..52 {
                source.data_mut()[(y * 96 + x) * 3..(y * 96 + x) * 3 + 3].fill(0.0);
            }
        }
        for method in [SpotMethod::ContentAware, SpotMethod::ProximityMatch] {
            let mut stroke = Stroke::new(96, 96, false);
            stroke.dab((48.0, 48.0), &hard(9.0));
            let mut target = source.clone();

            stroke.spot_heal(&mut target, &source, method, 18.0, 1.0);

            let err: f32 = (44..52)
                .flat_map(|y| (44..52).map(move |x| (x, y)))
                .map(|(x, y)| (target.pixel(x, y)[0] - clean(x, y)).abs())
                .sum::<f32>()
                / 64.0;
            assert!(err < 0.12, "{method:?}: mean error {err}");
        }
    }

    #[test]
    fn with_wrap_a_dab_continues_across_the_edge() {
        let source = Image::filled(64, 64, 3, 0.2).unwrap();
        let mut stroke = Stroke::new(64, 64, true);

        let regions = stroke.dab((1.0, 32.0), &hard(5.0));
        let mut target = source.clone();
        let other = Image::filled(64, 64, 3, 0.9).unwrap();
        stroke.composite(&mut target, &source, &regions, 1.0, |x, y, out| {
            out.copy_from_slice(other.pixel(x, y))
        });

        assert!((target.pixel(62, 32)[0] - 0.9).abs() < 1e-5, "painted on the far side");
        assert_eq!(regions.len(), 2, "one region on each side: {regions:?}");
    }

    #[test]
    fn heal_across_the_edge_keeps_the_texture_tiling() {
        let source =
            Image::from_fn(64, 64, 3, |x, _, _| 0.3 + 0.4 * ((x as f32) * 0.3).sin().abs())
                .unwrap();
        let mut stroke = Stroke::new(64, 64, true);
        for y in [20.0, 26.0, 32.0] {
            stroke.dab((0.0, y), &hard(6.0));
        }
        let mut target = source.clone();

        let regions = stroke.heal(&mut target, &source, (20.0, 0.0), 1.0);

        assert!(regions.len() >= 2, "{regions:?}");
        assert!(target != source, "something healed");
    }

    #[test]
    fn hostile_input_changes_nothing() {
        let source = halves(64);
        for brush in [
            Brush { radius: f32::NAN, ..hard(5.0) },
            Brush { radius: 0.0, ..hard(5.0) },
            Brush { flow: 0.0, ..hard(5.0) },
        ] {
            let mut stroke = Stroke::new(64, 64, true);
            assert!(stroke.dab((10.0, 10.0), &brush).is_empty());
            assert!(stroke.is_empty());
        }
        let mut stroke = Stroke::new(64, 64, true);
        assert!(stroke.dab((f32::INFINITY, 1.0), &hard(5.0)).is_empty());
        stroke.dab((10.0, 10.0), &hard(5.0));
        let mut target = source.clone();
        let smaller = Image::filled(32, 32, 3, 0.0).unwrap();
        assert!(stroke.heal(&mut target, &smaller, (1.0, 0.0), 1.0).is_empty());
        assert!(stroke.heal(&mut target, &source, (f32::NAN, 0.0), 1.0).is_empty());
        assert_eq!(target, source);
    }

    #[test]
    fn healing_keeps_alpha_from_the_source() {
        let source =
            Image::from_fn(64, 64, 4, |x, _, c| if c == 3 { x as f32 / 64.0 } else { 0.5 })
                .unwrap();
        let mut stroke = Stroke::new(64, 64, false);
        stroke.dab((20.0, 32.0), &hard(8.0));
        let mut target = source.clone();

        stroke.heal(&mut target, &source, (20.0, 0.0), 1.0);

        assert!(
            (target.pixel(20, 32)[3] - 40.0 / 64.0).abs() < 0.02,
            "{}",
            target.pixel(20, 32)[3]
        );
    }

    #[test]
    fn dabs_are_spaced_exactly_and_carry_the_remainder() {
        let dabs = dabs_towards((0.0, 0.0), (25.0, 0.0), 10.0);
        assert_eq!(dabs, vec![(10.0, 0.0), (20.0, 0.0)]);
        assert!(dabs_towards((20.0, 0.0), (25.0, 0.0), 10.0).is_empty(), "short moves wait");
        assert!(dabs_towards((0.0, 0.0), (f32::NAN, 0.0), 10.0).is_empty());
        assert_eq!(dabs_towards((0.0, 0.0), (1e9, 0.0), 1.0).len(), 4096);
    }

    #[test]
    fn smoothing_trails_the_pointer_and_catches_up_at_the_end() {
        let mut off = Smoother::new(0.0);
        assert_eq!(off.push((5.0, 5.0), 0.0), vec![(5.0, 5.0)]);
        assert_eq!(off.push((9.0, 5.0), 0.016), vec![(9.0, 5.0)]);

        let mut smooth = Smoother::new(0.5);
        smooth.push((0.0, 0.0), 0.0);
        let step = smooth.push((10.0, 0.0), 1.0 / 60.0);
        assert_eq!(step, vec![(5.0, 0.0)], "half way");
        assert_eq!(smooth.finish(), Some((10.0, 0.0)));
        assert_eq!(smooth.finish(), None);
    }
}
