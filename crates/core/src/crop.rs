//! Cropping to a frame that may be turned: what the Crop tool commits.
//!
//! The frame is a rectangle in image pixels, turned about its own centre.
//! The result is the frame's contents made upright, one output pixel per
//! frame pixel. An unturned frame on whole pixels is copied exactly; any
//! other frame is sampled bilinearly. Where a turned frame reaches past the
//! image, the nearest edge pixels are repeated, so a texture never gains
//! empty corners.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::{CoreError, Result};
use crate::filter::{Edge, sample_bilinear};
use crate::image::{Image, MAX_DIMENSION, MIN_DIMENSION};

/// A turn this small (degrees) is treated as none.
const UPRIGHT_TOLERANCE: f32 = 1e-3;

/// A crop frame in image pixels.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CropFrame {
    /// Centre of the frame.
    pub centre_x: f32,
    pub centre_y: f32,
    /// Size of the frame before it is turned.
    pub width: f32,
    pub height: f32,
    /// Turn about the centre in degrees, clockwise as seen on screen.
    pub degrees: f32,
}

impl CropFrame {
    /// The whole of an image, upright.
    pub fn whole(width: usize, height: usize) -> Self {
        let (width, height) = (width as f32, height as f32);
        Self { centre_x: width / 2.0, centre_y: height / 2.0, width, height, degrees: 0.0 }
    }

    pub fn is_upright(&self) -> bool {
        self.degrees.abs() < UPRIGHT_TOLERANCE
    }

    /// The output size: the frame's size in whole pixels.
    pub fn output_size(&self) -> (usize, usize) {
        let side = |v: f32| if v.is_finite() && v > 0.0 { v.round() as usize } else { 0 };
        (side(self.width), side(self.height))
    }

    /// Where the output pixel (`u`, `v`) comes from in the image, in pixel
    /// coordinates (centres on whole numbers).
    fn source(&self, u: usize, v: usize, sin: f32, cos: f32) -> (f32, f32) {
        let (width, height) = self.output_size();
        let dx = u as f32 + 0.5 - width as f32 / 2.0;
        let dy = v as f32 + 0.5 - height as f32 / 2.0;
        // Clockwise on screen, where y points down.
        let x = self.centre_x + cos * dx - sin * dy;
        let y = self.centre_y + sin * dx + cos * dy;
        (x - 0.5, y - 0.5)
    }

    /// The top-left pixel of an upright frame, when it sits on whole pixels
    /// inside `image`.
    fn exact_origin(&self, image: &Image) -> Option<(usize, usize)> {
        if !self.is_upright() {
            return None;
        }
        let (width, height) = self.output_size();
        let left = self.centre_x - width as f32 / 2.0;
        let top = self.centre_y - height as f32 / 2.0;
        let whole = |v: f32| (v - v.round()).abs() < 1e-3 && v.round() >= 0.0;
        if !(whole(left) && whole(top)) {
            return None;
        }
        let (x, y) = (left.round() as usize, top.round() as usize);
        (x + width <= image.width() && y + height <= image.height()).then_some((x, y))
    }

    fn validate(&self) -> Result<(usize, usize)> {
        let values = [self.centre_x, self.centre_y, self.width, self.height, self.degrees];
        if values.iter().any(|v| !v.is_finite()) {
            let reason = "every value must be a finite number".to_owned();
            return Err(CoreError::InvalidParameter { name: "crop frame", reason });
        }
        let (width, height) = self.output_size();
        validate_output(width, height)?;
        Ok((width, height))
    }
}

/// Refuses an output outside the sizes SEAMS takes.
pub(crate) fn validate_output(width: usize, height: usize) -> Result<()> {
    if width < MIN_DIMENSION || height < MIN_DIMENSION {
        return Err(CoreError::TooSmall { width, height, min: MIN_DIMENSION });
    }
    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(CoreError::TooLarge { width, height, max: MAX_DIMENSION });
    }
    Ok(())
}

/// The contents of `frame`, made upright. Alpha and channel count are kept.
pub fn crop(image: &Image, frame: &CropFrame) -> Result<Image> {
    let (width, height) = frame.validate()?;
    let channels = image.channels();
    if let Some((left, top)) = frame.exact_origin(image) {
        let mut data = Vec::with_capacity(width * height * channels);
        for y in top..top + height {
            let row = image.row(y);
            data.extend_from_slice(
                row.get(left * channels..(left + width) * channels).unwrap_or(&[]),
            );
        }
        return Image::new(width, height, channels, data);
    }
    let (sin, cos) = frame.degrees.to_radians().sin_cos();
    let mut data = vec![0.0f32; width * height * channels];
    data.par_chunks_mut(width * channels).enumerate().for_each(|(v, row)| {
        for (u, pixel) in row.chunks_mut(channels).enumerate() {
            let (x, y) = frame.source(u, v, sin, cos);
            sample_bilinear(image, x, y, Edge::Clamp, pixel);
        }
    });
    Image::new(width, height, channels, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every pixel holds its own position, so a crop shows where it read from.
    fn coordinates(width: usize, height: usize) -> Image {
        Image::from_fn(width, height, 3, |x, y, c| match c {
            0 => x as f32 / 1000.0,
            1 => y as f32 / 1000.0,
            _ => 0.5,
        })
        .unwrap()
    }

    fn frame(left: f32, top: f32, width: f32, height: f32, degrees: f32) -> CropFrame {
        CropFrame {
            centre_x: left + width / 2.0,
            centre_y: top + height / 2.0,
            width,
            height,
            degrees,
        }
    }

    #[test]
    fn an_upright_frame_copies_its_pixels_exactly() {
        let image = coordinates(256, 200);

        let result = crop(&image, &frame(10.0, 20.0, 100.0, 80.0, 0.0)).unwrap();

        assert_eq!((result.width(), result.height()), (100, 80));
        assert_eq!(result.pixel(0, 0), image.pixel(10, 20));
        assert_eq!(result.pixel(99, 79), image.pixel(109, 99));
    }

    #[test]
    fn the_whole_frame_returns_the_image() {
        let image = coordinates(128, 96);

        assert_eq!(crop(&image, &CropFrame::whole(128, 96)).unwrap(), image);
    }

    #[test]
    fn a_quarter_turn_reads_the_image_sideways() {
        // Turned 90° clockwise, the frame's top edge runs down the image's
        // right side, so the output's top-left comes from the top-right.
        let image = coordinates(128, 128);

        let result = crop(&image, &frame(0.0, 0.0, 128.0, 128.0, 90.0)).unwrap();

        let top_left = result.pixel(0, 0);
        assert!((top_left[0] - 0.127).abs() < 1e-4, "x {}", top_left[0]);
        assert!(top_left[1].abs() < 1e-4, "y {}", top_left[1]);
        let bottom_left = result.pixel(0, 127);
        assert!(bottom_left[0].abs() < 1e-4 && bottom_left[1].abs() < 1e-4);
    }

    #[test]
    fn a_turned_frame_past_the_edge_repeats_edge_pixels() {
        let image = Image::filled(128, 128, 4, 0.25).unwrap();

        let result = crop(&image, &frame(0.0, 0.0, 128.0, 128.0, 30.0)).unwrap();

        assert_eq!(result.channels(), 4);
        assert!(result.data().iter().all(|v| (v - 0.25).abs() < 1e-5));
    }

    #[test]
    fn frames_below_the_minimum_size_are_refused() {
        let image = coordinates(256, 256);

        let small = crop(&image, &frame(0.0, 0.0, 63.0, 200.0, 0.0));

        assert!(matches!(small, Err(CoreError::TooSmall { width: 63, .. })));
    }

    #[test]
    fn hostile_frames_are_refused_not_panicked_on() {
        let image = coordinates(128, 128);
        let bad = [
            CropFrame { width: f32::NAN, ..CropFrame::whole(128, 128) },
            CropFrame { degrees: f32::INFINITY, ..CropFrame::whole(128, 128) },
            CropFrame { width: 1e9, height: 1e9, ..CropFrame::whole(128, 128) },
            CropFrame { width: -100.0, ..CropFrame::whole(128, 128) },
        ];

        for frame in bad {
            assert!(crop(&image, &frame).is_err(), "{frame:?}");
        }
    }

    #[test]
    fn an_upright_frame_off_the_pixel_grid_is_still_cropped() {
        let image = coordinates(256, 256);

        let result = crop(&image, &frame(10.5, 10.5, 100.0, 100.0, 0.0)).unwrap();

        // Halfway between pixels 10 and 11.
        assert!((result.pixel(0, 0)[0] - 0.0105).abs() < 1e-4);
    }
}
