//! Mirror tiling: an exact 2x2 reflected tile.

use rayon::prelude::*;

use crate::error::{CoreError, Result};
use crate::image::{Image, MAX_DIMENSION};

/// The largest result allowed: a maximum-size import, doubled on both axes.
const MAX_OUTPUT_PIXELS: usize = (MAX_DIMENSION * 2) * (MAX_DIMENSION * 2);

/// Builds a tile twice as wide and tall from the image and its reflections.
///
/// Opposite borders of the result are identical by construction, so it tiles
/// exactly, at the cost of visible symmetry.
pub fn mirror_tile(image: &Image) -> Result<Image> {
    let (width, height, channels) = (image.width(), image.height(), image.channels());
    let (out_width, out_height) = (width * 2, height * 2);
    let requested = out_width.saturating_mul(out_height);
    if requested > MAX_OUTPUT_PIXELS {
        return Err(CoreError::OutputTooLarge { requested, limit: MAX_OUTPUT_PIXELS });
    }

    let row_len = image.row_len();
    let mut data = vec![0.0f32; requested * channels];
    data.par_chunks_mut(out_width * channels).enumerate().for_each(|(y, dst)| {
        let source_y = if y < height { y } else { out_height - 1 - y };
        let src = image.row(source_y);
        let (left, right) = dst.split_at_mut(row_len);
        left.copy_from_slice(src);
        for (target, pixel) in right.chunks_mut(channels).zip(src.chunks(channels).rev()) {
            target.copy_from_slice(pixel);
        }
    });
    Image::new(out_width, out_height, channels, data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seamless::test_support::sample;

    #[test]
    fn result_is_an_exact_two_by_two_reflection() {
        let source = Image::from_plane(2, 2, vec![1., 2., 3., 4.]).unwrap();

        let tile = mirror_tile(&source).unwrap();

        #[rustfmt::skip]
        let expected = [
            1., 2., 2., 1.,
            3., 4., 4., 3.,
            3., 4., 4., 3.,
            1., 2., 2., 1.,
        ];
        assert_eq!(tile.data(), &expected);
    }

    #[test]
    fn opposite_borders_match_exactly() {
        let tile = mirror_tile(&sample(64, 48, 3)).unwrap();
        let (w, h) = (tile.width(), tile.height());

        for y in 0..h {
            assert_eq!(tile.pixel(0, y), tile.pixel(w - 1, y));
        }
        for x in 0..w {
            assert_eq!(tile.pixel(x, 0), tile.pixel(x, h - 1));
        }
    }

    #[test]
    fn top_left_quadrant_is_the_source() {
        let source = sample(64, 48, 4);

        let tile = mirror_tile(&source).unwrap();

        for (x, y) in [(0, 0), (63, 47), (10, 30)] {
            assert_eq!(tile.pixel(x, y), source.pixel(x, y));
        }
    }
}
