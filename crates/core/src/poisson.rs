//! Gradient-domain ("Poisson") blending, the core of the Heal brush.
//!
//! Method: P. Perez, M. Gangnet, A. Blake, *Poisson Image Editing*, ACM SIGGRAPH 2003. Seamless
//! cloning finds `f` over a region that keeps the source's gradients and meets the destination on
//! the boundary of the region. That splits into the source plus a *membrane* `h`: harmonic inside
//! the region (`Δh = 0`), equal to `destination - source` on its boundary. The healed pixels keep
//! the source's texture while their colour and lighting follow the destination around the stroke.
//!
//! The membrane is solved by successive over-relaxation, started from the bilinearly enlarged
//! solution of a half-resolution version of the same problem (cascadic multigrid), so a few sweeps
//! per level are enough. Ported, with changes, from the PhotoCraft project's `poisson.rs`, by the
//! same author and under the same MIT / Apache-2.0 licence.

use rayon::prelude::*;

/// Convergence threshold (largest update in one sweep).
const TOLERANCE: f32 = 2e-5;
/// Sweeps and relaxation factor on every level above the coarsest.
const FINE_SWEEPS: usize = 30;
const FINE_OMEGA: f32 = 1.7;
/// The coarsest level is solved to convergence with these.
const COARSE_SWEEPS: usize = 2000;
const COARSE_OMEGA: f32 = 1.85;
/// Problems smaller than this are solved directly.
const MIN_COARSEN_SIDE: usize = 8;
const MIN_COARSEN_UNKNOWNS: usize = 64;
const MAX_DEPTH: usize = 16;

/// Solves `Δv = 0` on the `unknown` pixels, with the values of the known pixels of `v` as the
/// boundary. Neighbours outside the grid are ignored. `v` holds the known values on entry and the
/// solution on exit. Both slices must hold `w * h` values; anything else is left untouched.
pub fn solve_membrane(w: usize, h: usize, unknown: &[bool], v: &mut [f32]) {
    if unknown.len() != w * h || v.len() != w * h {
        return;
    }
    solve(w, h, unknown, v, 0);
}

fn solve(w: usize, h: usize, unknown: &[bool], v: &mut [f32], depth: usize) {
    let unknowns = unknown.iter().filter(|u| **u).count();
    let known: Vec<f32> = v.iter().zip(unknown).filter(|(_, u)| !**u).map(|(x, _)| *x).collect();
    if unknowns == 0 || known.is_empty() {
        return;
    }
    let coarsen = w >= MIN_COARSEN_SIDE
        && h >= MIN_COARSEN_SIDE
        && unknowns > MIN_COARSEN_UNKNOWNS
        && depth < MAX_DEPTH;
    if coarsen {
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let (mut coarse_unknown, mut coarse) = (vec![true; cw * ch], vec![0.0f32; cw * ch]);
        // A coarse pixel is known when any child is; its value is the mean of the known children.
        for cy in 0..ch {
            for cx in 0..cw {
                let (mut sum, mut count) = (0.0f32, 0u32);
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let (x, y) = (cx * 2 + dx, cy * 2 + dy);
                    if x < w && y < h && !unknown[y * w + x] {
                        sum += v[y * w + x];
                        count += 1;
                    }
                }
                if count > 0 {
                    coarse_unknown[cy * cw + cx] = false;
                    coarse[cy * cw + cx] = sum / count as f32;
                }
            }
        }
        solve(cw, ch, &coarse_unknown, &mut coarse, depth + 1);
        enlarge(w, h, unknown, v, (cw, ch), &coarse);
        sweep(w, h, unknown, v, FINE_SWEEPS, FINE_OMEGA);
    } else {
        let mean = known.iter().sum::<f32>() / known.len() as f32;
        for (x, u) in v.iter_mut().zip(unknown) {
            if *u {
                *x = mean;
            }
        }
        sweep(w, h, unknown, v, COARSE_SWEEPS, COARSE_OMEGA);
    }
}

/// Fills the unknown pixels of `v` from the coarse solution, bilinearly.
fn enlarge(
    w: usize,
    h: usize,
    unknown: &[bool],
    v: &mut [f32],
    (cw, ch): (usize, usize),
    coarse: &[f32],
) {
    for y in 0..h {
        let fy = ((y as f32 + 0.5) / 2.0 - 0.5).clamp(0.0, (ch - 1) as f32);
        let (y0, ty) = (fy.floor() as usize, fy - fy.floor());
        let y1 = (y0 + 1).min(ch - 1);
        for x in 0..w {
            if !unknown[y * w + x] {
                continue;
            }
            let fx = ((x as f32 + 0.5) / 2.0 - 0.5).clamp(0.0, (cw - 1) as f32);
            let (x0, tx) = (fx.floor() as usize, fx - fx.floor());
            let x1 = (x0 + 1).min(cw - 1);
            let top = coarse[y0 * cw + x0] + (coarse[y0 * cw + x1] - coarse[y0 * cw + x0]) * tx;
            let bottom = coarse[y1 * cw + x0] + (coarse[y1 * cw + x1] - coarse[y1 * cw + x0]) * tx;
            v[y * w + x] = top + (bottom - top) * ty;
        }
    }
}

/// Over-relaxation sweeps over the unknown pixels until the largest update is below [`TOLERANCE`].
fn sweep(w: usize, h: usize, unknown: &[bool], v: &mut [f32], max_sweeps: usize, omega: f32) {
    let cells: Vec<usize> = (0..w * h).filter(|&i| unknown[i]).collect();
    for _ in 0..max_sweeps {
        let mut largest = 0.0f32;
        for &i in &cells {
            let (x, y) = (i % w, i / w);
            let (mut sum, mut count) = (0.0f32, 0.0f32);
            for (inside, at) in [
                (x > 0, i.wrapping_sub(1)),
                (x + 1 < w, i + 1),
                (y > 0, i.wrapping_sub(w)),
                (y + 1 < h, i + w),
            ] {
                if inside {
                    sum += v.get(at).copied().unwrap_or(0.0);
                    count += 1.0;
                }
            }
            if count == 0.0 {
                continue;
            }
            let change = omega * (sum / count - v[i]);
            v[i] += change;
            largest = largest.max(change.abs());
        }
        if largest < TOLERANCE {
            break;
        }
    }
}

/// Seamless cloning: `src` made to meet `dst` around the `mask`. Returns `dst` outside the mask
/// and, inside, `src` plus the membrane that removes the mismatch at the mask's border. `src` and
/// `dst` hold `w * h * channels` values; `mask` holds `w * h` (`true` = healed). Pixels of the mask
/// on the grid edge see a free boundary there, so leave an unmasked margin where the border
/// matters. Channels from `colors` on (alpha) are copied from `src` unchanged.
pub fn seamless_clone(
    (w, h, channels, colors): (usize, usize, usize, usize),
    src: &[f32],
    dst: &[f32],
    mask: &[bool],
) -> Vec<f32> {
    if src.len() != w * h * channels || dst.len() != src.len() || mask.len() != w * h {
        return src.to_vec();
    }
    let solved: Vec<Vec<f32>> = (0..colors.min(channels))
        .into_par_iter()
        .map(|c| {
            let mut field: Vec<f32> = (0..w * h)
                .map(|i| if mask[i] { 0.0 } else { dst[i * channels + c] - src[i * channels + c] })
                .collect();
            solve_membrane(w, h, mask, &mut field);
            field
        })
        .collect();
    let mut out = dst.to_vec();
    for i in (0..w * h).filter(|&i| mask[i]) {
        for c in 0..channels {
            let lift = solved.get(c).map_or(0.0, |field| field[i]);
            out[i * channels + c] = src[i * channels + c] + lift;
        }
    }
    out
}

/// Fills the `hole` pixels of every channel of `img` (`w * h * channels` values) with the
/// smoothest surface that meets the pixels around it. Returns `img` unchanged when the buffers
/// do not match.
pub fn membrane_fill(w: usize, h: usize, channels: usize, img: &[f32], hole: &[bool]) -> Vec<f32> {
    if img.len() != w * h * channels || hole.len() != w * h {
        return img.to_vec();
    }
    let solved: Vec<Vec<f32>> = (0..channels)
        .into_par_iter()
        .map(|c| {
            let mut v: Vec<f32> =
                (0..w * h).map(|i| if hole[i] { 0.0 } else { img[i * channels + c] }).collect();
            solve_membrane(w, h, hole, &mut v);
            v
        })
        .collect();
    let mut out = img.to_vec();
    for i in (0..w * h).filter(|&i| hole[i]) {
        for (c, plane) in solved.iter().enumerate() {
            out[i * channels + c] = plane[i];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn membrane_fill_meets_both_sides_of_a_hole() {
        let (w, h) = (10, 3);
        let img: Vec<f32> = (0..w * h).map(|i| if i % w < 5 { 0.0 } else { 1.0 }).collect();
        let hole: Vec<bool> = (0..w * h).map(|i| (3..7).contains(&(i % w))).collect();

        let out = membrane_fill(w, h, 1, &img, &hole);

        assert!(out[w + 3] > 0.0 && out[w + 3] < out[w + 6] && out[w + 6] < 1.0);
        assert_eq!(out[w], 0.0);
    }

    #[test]
    fn a_membrane_between_two_levels_is_a_smooth_ramp() {
        // Left column 0, right column 1, ten pixels between them.
        let (w, h) = (12, 5);
        let mut v = vec![0.0f32; w * h];
        let unknown: Vec<bool> = (0..w * h).map(|i| (1..11).contains(&(i % w))).collect();
        for y in 0..h {
            v[y * w + 11] = 1.0;
        }

        solve_membrane(w, h, &unknown, &mut v);

        for x in 1..11 {
            assert!((v[2 * w + x] - x as f32 / 11.0).abs() < 0.02, "column {x}: {}", v[2 * w + x]);
        }
    }

    #[test]
    fn seamless_clone_keeps_the_source_detail_at_the_destination_level() {
        let (w, h) = (48, 48);
        // A bright checker as the source over a flat dark destination.
        let src: Vec<f32> = (0..w * h)
            .map(|i| if ((i % w) / 3 + (i / w) / 3) % 2 == 0 { 0.9 } else { 0.7 })
            .collect();
        let dst = vec![0.2f32; w * h];
        let mask: Vec<bool> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32 - 24.0, (i / w) as f32 - 24.0);
                x * x + y * y < 12.0 * 12.0
            })
            .collect();

        let out = seamless_clone((w, h, 1, 1), &src, &dst, &mask);

        assert_eq!(out[0], 0.2, "unchanged outside the mask");
        let (a, b) = (out[24 * w + 24], out[24 * w + 27]);
        assert!((a - b).abs() > 0.15, "the checker survived: {a} vs {b}");
        let inside = mask.iter().filter(|m| **m).count() as f32;
        let mean =
            mask.iter().zip(&out).filter(|(m, _)| **m).map(|(_, v)| *v).sum::<f32>() / inside;
        assert!((mean - 0.2).abs() < 0.12, "pulled to the destination level: {mean}");
    }

    #[test]
    fn mismatched_buffers_are_left_alone_not_panicked_on() {
        let mut v = vec![1.0f32; 4];
        solve_membrane(3, 3, &[true; 9], &mut v);
        assert_eq!(v, vec![1.0; 4]);
        assert_eq!(seamless_clone((2, 2, 1, 1), &[0.5; 4], &[0.1; 3], &[true; 4]), vec![0.5; 4]);
    }
}
