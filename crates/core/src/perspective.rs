//! Perspective crop: straightening a four-cornered region into a rectangle,
//! as Photoshop's Perspective Crop tool does. Photograph a wall at an angle,
//! put the corners on its corners, and the result is the wall seen square on.
//!
//! The region's corners map to the output's corners through a projective
//! transform (a homography), so straight lines stay straight and parallel
//! lines in the scene come out parallel. The output is sampled bilinearly;
//! past the image, edge pixels repeat.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::crop::{CropFrame, validate_output};
use crate::error::{CoreError, Result};
use crate::filter::{Edge, sample_bilinear};
use crate::image::Image;

/// Smallest turn (as twice a triangle's area, in square pixels) a corner of
/// the region must make: below it the region is treated as flat.
const MIN_CORNER_AREA: f32 = 1e-3;

/// A four-cornered region in image pixels: top-left, top-right,
/// bottom-right and bottom-left, as they should appear in the result.
/// Coordinates are positions, so `[0, 0]` is the image's top-left edge.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Quad {
    pub corners: [[f32; 2]; 4],
}

impl Quad {
    /// The corners of a crop frame, turned as it is.
    pub fn from_frame(frame: &CropFrame) -> Self {
        let (sin, cos) = frame.degrees.to_radians().sin_cos();
        let (w, h) = (frame.width / 2.0, frame.height / 2.0);
        let corner = |x: f32, y: f32| {
            [frame.centre_x + cos * x - sin * y, frame.centre_y + sin * x + cos * y]
        };
        Self { corners: [corner(-w, -h), corner(w, -h), corner(w, h), corner(-w, h)] }
    }

    /// True when the corners are finite and run clockwise on screen around a
    /// convex region: no corner folds inward and no edge crosses another.
    pub fn is_convex(&self) -> bool {
        if self.corners.iter().flatten().any(|v| !v.is_finite()) {
            return false;
        }
        (0..4).all(|i| {
            let [a, b, c] = [i, (i + 1) % 4, (i + 2) % 4].map(|k| self.corners[k]);
            let (e1, e2) = ([b[0] - a[0], b[1] - a[1]], [c[0] - b[0], c[1] - b[1]]);
            e1[0] * e2[1] - e1[1] * e2[0] > MIN_CORNER_AREA
        })
    }

    /// The size the straightened region should have: the mean length of
    /// each pair of opposite edges.
    pub fn measured_size(&self) -> (f32, f32) {
        let length = |a: usize, b: usize| {
            let (p, q) = (self.corners[a], self.corners[b]);
            (q[0] - p[0]).hypot(q[1] - p[1])
        };
        ((length(0, 1) + length(3, 2)) / 2.0, (length(0, 3) + length(1, 2)) / 2.0)
    }

    /// The transform from the unit square to this region, when the region
    /// is convex.
    pub fn homography(&self) -> Option<Homography> {
        self.is_convex().then(|| Homography::square_to(&self.corners)).flatten()
    }
}

/// A projective map from the unit square to a quad: (0, 0) to its first
/// corner, (1, 0) to its second, (1, 1) to its third and (0, 1) to its
/// fourth.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Homography {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    e: f32,
    f: f32,
    g: f32,
    h: f32,
}

impl Homography {
    /// Heckbert's closed form for the square-to-quadrilateral case.
    fn square_to(corners: &[[f32; 2]; 4]) -> Option<Self> {
        let [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] = *corners;
        let (dx1, dx2, dx3) = (x1 - x2, x3 - x2, x0 - x1 + x2 - x3);
        let (dy1, dy2, dy3) = (y1 - y2, y3 - y2, y0 - y1 + y2 - y3);
        let (g, h) = if dx3.abs() < f32::EPSILON && dy3.abs() < f32::EPSILON {
            (0.0, 0.0)
        } else {
            let det = dx1 * dy2 - dx2 * dy1;
            if det.abs() < f32::EPSILON {
                return None;
            }
            ((dx3 * dy2 - dx2 * dy3) / det, (dx1 * dy3 - dx3 * dy1) / det)
        };
        let map = Self {
            a: x1 - x0 + g * x1,
            b: x3 - x0 + h * x3,
            c: x0,
            d: y1 - y0 + g * y1,
            e: y3 - y0 + h * y3,
            f: y0,
            g,
            h,
        };
        [map.a, map.b, map.d, map.e, map.g, map.h].iter().all(|v| v.is_finite()).then_some(map)
    }

    /// Where (`u`, `v`) of the unit square lands.
    pub fn map(&self, u: f32, v: f32) -> [f32; 2] {
        let w = self.g * u + self.h * v + 1.0;
        let w = if w.abs() < f32::EPSILON { f32::EPSILON.copysign(w) } else { w };
        [(self.a * u + self.b * v + self.c) / w, (self.d * u + self.e * v + self.f) / w]
    }
}

/// The region `quad` straightened into a `width` × `height` image.
pub fn perspective_crop(image: &Image, quad: &Quad, width: usize, height: usize) -> Result<Image> {
    validate_output(width, height)?;
    let map = quad.homography().ok_or_else(|| CoreError::InvalidParameter {
        name: "perspective crop",
        reason: "the four corners must make a convex shape, in order around it".to_owned(),
    })?;
    let channels = image.channels();
    let mut data = vec![0.0f32; width * height * channels];
    let (du, dv) = (1.0 / width as f32, 1.0 / height as f32);
    data.par_chunks_mut(width * channels).enumerate().for_each(|(row, line)| {
        let v = (row as f32 + 0.5) * dv;
        for (column, pixel) in line.chunks_mut(channels).enumerate() {
            let [x, y] = map.map((column as f32 + 0.5) * du, v);
            sample_bilinear(image, x - 0.5, y - 0.5, Edge::Clamp, pixel);
        }
    });
    Image::new(width, height, channels, data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crop::crop;

    fn coordinates(width: usize, height: usize) -> Image {
        Image::from_fn(width, height, 3, |x, y, c| match c {
            0 => x as f32 / 1000.0,
            1 => y as f32 / 1000.0,
            _ => 0.5,
        })
        .unwrap()
    }

    /// A trapezoid: the top edge narrower than the bottom, as a wall seen
    /// from below.
    fn keystone() -> Quad {
        Quad { corners: [[60.0, 20.0], [196.0, 20.0], [246.0, 230.0], [10.0, 230.0]] }
    }

    #[test]
    fn the_map_sends_the_square_corners_to_the_quad_corners() {
        let quad = keystone();
        let map = quad.homography().unwrap();

        for ([u, v], expected) in
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]].iter().zip(quad.corners)
        {
            let [x, y] = map.map(*u, *v);
            assert!((x - expected[0]).abs() < 1e-3 && (y - expected[1]).abs() < 1e-3, "{u},{v}");
        }
    }

    #[test]
    fn a_rectangle_straightens_to_the_same_pixels_as_a_plain_crop() {
        let image = coordinates(256, 200);
        let frame =
            CropFrame { centre_x: 60.0, centre_y: 60.0, width: 100.0, height: 80.0, degrees: 0.0 };

        let plain = crop(&image, &frame).unwrap();
        let straightened = perspective_crop(&image, &Quad::from_frame(&frame), 100, 80).unwrap();

        let worst = plain
            .data()
            .iter()
            .zip(straightened.data())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(worst < 1e-4, "worst difference {worst}");
    }

    #[test]
    fn a_keystone_is_straightened_so_its_sides_become_vertical() {
        // The left side of the trapezoid runs from x 60 at the top to x 10
        // at the bottom. Straightened, the output's first column follows it.
        let image = coordinates(256, 256);

        let result = perspective_crop(&image, &keystone(), 128, 128).unwrap();

        let top = result.pixel(0, 0)[0] * 1000.0;
        let bottom = result.pixel(0, 127)[0] * 1000.0;
        assert!((top - 60.0).abs() < 2.0, "top-left reads x {top}");
        assert!((bottom - 10.0).abs() < 2.0, "bottom-left reads x {bottom}");
    }

    #[test]
    fn the_measured_size_averages_opposite_edges() {
        assert_eq!(keystone().measured_size().0, (136.0 + 236.0) / 2.0);
    }

    #[test]
    fn a_folded_or_flat_region_is_refused() {
        let image = coordinates(128, 128);
        let bow_tie = Quad { corners: [[0.0, 0.0], [100.0, 100.0], [100.0, 0.0], [0.0, 100.0]] };
        let flat = Quad { corners: [[0.0, 0.0], [50.0, 0.0], [100.0, 0.0], [10.0, 0.0]] };
        let backwards = Quad { corners: [[0.0, 0.0], [0.0, 100.0], [100.0, 100.0], [100.0, 0.0]] };
        let broken =
            Quad { corners: [[f32::NAN, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]] };

        for quad in [bow_tie, flat, backwards, broken] {
            assert!(!quad.is_convex(), "{quad:?}");
            assert!(perspective_crop(&image, &quad, 64, 64).is_err(), "{quad:?}");
        }
    }

    #[test]
    fn the_output_size_is_checked() {
        let image = coordinates(128, 128);

        assert!(perspective_crop(&image, &keystone(), 32, 128).is_err());
        assert!(perspective_crop(&image, &keystone(), 9000, 128).is_err());
    }
}
