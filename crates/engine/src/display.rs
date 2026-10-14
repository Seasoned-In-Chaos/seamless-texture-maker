//! Screen-ready copies of images.

use seams_core::Image;
use seams_core::filter::fit_within;

use crate::error::Result;

/// An 8-bit RGBA copy of an image, small enough to upload as one texture.
///
/// Alpha is straight (not premultiplied). Built on worker threads so the UI
/// thread never converts or resizes full-resolution pixels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisplayImage {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
    /// Size of the full-resolution image this was made from.
    pub full_width: usize,
    pub full_height: usize,
}

impl DisplayImage {
    /// Converts `image`, shrinking it so neither side exceeds `max_side`.
    pub fn from_image(image: &Image, max_side: usize) -> Result<Self> {
        let fitted = fit_within(image, max_side)?;
        let rgba = fitted.data().chunks(fitted.channels()).flat_map(pixel_rgba).collect();
        Ok(Self {
            width: fitted.width(),
            height: fitted.height(),
            rgba,
            full_width: image.width(),
            full_height: image.height(),
        })
    }
}

fn to_byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

fn pixel_rgba(pixel: &[f32]) -> [u8; 4] {
    match pixel {
        [r, g, b, a] => [to_byte(*r), to_byte(*g), to_byte(*b), to_byte(*a)],
        [r, g, b] => [to_byte(*r), to_byte(*g), to_byte(*b), 255],
        [gray] => [to_byte(*gray), to_byte(*gray), to_byte(*gray), 255],
        _ => [0, 0, 0, 255],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gray_rgb_and_rgba_all_become_rgba_bytes() {
        let gray = Image::filled(2, 1, 1, 1.0).unwrap();
        let rgb = Image::new(1, 1, 3, vec![1.0, 0.5, 0.0]).unwrap();
        let rgba = Image::new(1, 1, 4, vec![0.0, 0.0, 1.0, 0.25]).unwrap();

        assert_eq!(DisplayImage::from_image(&gray, 64).unwrap().rgba, vec![255; 8]);
        assert_eq!(DisplayImage::from_image(&rgb, 64).unwrap().rgba, vec![255, 128, 0, 255]);
        assert_eq!(DisplayImage::from_image(&rgba, 64).unwrap().rgba, vec![0, 0, 255, 64]);
    }

    #[test]
    fn large_images_are_shrunk_to_the_limit() {
        let image = Image::filled(400, 100, 3, 0.5).unwrap();

        let display = DisplayImage::from_image(&image, 100).unwrap();

        assert_eq!((display.width, display.height), (100, 25));
        assert_eq!((display.full_width, display.full_height), (400, 100));
        assert_eq!(display.rgba.len(), 100 * 25 * 4);
    }

    #[test]
    fn out_of_range_values_are_clamped() {
        let image = Image::new(1, 1, 3, vec![-2.0, 9.0, f32::NAN]).unwrap();

        assert_eq!(DisplayImage::from_image(&image, 8).unwrap().rgba, vec![0, 255, 0, 255]);
    }
}
