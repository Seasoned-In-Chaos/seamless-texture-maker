//! Offset cross-fade: move the borders to the centre and feather them there.

use rayon::prelude::*;

use crate::error::Result;
use crate::image::Image;

/// Feather radius at full strength, as a fraction of the shorter side. Wide
/// enough to hide a hard border, bounded so the texture is not washed out.
const MAX_FEATHER_FRACTION: f32 = 0.16;
const MIN_FADE_AMOUNT: f32 = 0.1;

/// Circularly shifts the image right by `shift_x` and down by `shift_y`.
pub fn roll(image: &Image, shift_x: usize, shift_y: usize) -> Image {
    let (width, height, channels) = (image.width(), image.height(), image.channels());
    let row_len = image.row_len();
    let split = (shift_x % width) * channels;
    let mut data = vec![0.0f32; image.data().len()];
    data.par_chunks_mut(row_len).enumerate().for_each(|(y, dst)| {
        let source_y = (y + height - shift_y % height) % height;
        let src = image.row(source_y);
        let (dst_head, dst_tail) = dst.split_at_mut(split);
        let (src_head, src_tail) = src.split_at(row_len - split);
        dst_tail.copy_from_slice(src_head);
        dst_head.copy_from_slice(src_tail);
    });
    Image::new(width, height, channels, data).unwrap_or_else(|_| image.clone())
}

/// Makes the image tile by cross-fading its borders.
///
/// A half-image offset moves the four borders onto the centre axes. Pixels on
/// either side of each centre seam are blended into both sides with a linear
/// feather, then the offset is reversed, which puts the now-equal seam pixels
/// on the outside borders.
pub fn offset_crossfade(image: &Image, fade_amount: f32) -> Result<Image> {
    let (width, height) = (image.width(), image.height());
    let (shift_x, shift_y) = (width / 2, height / 2);
    let radius = feather_radius(width, height, fade_amount);

    let mut shifted = roll(image, shift_x, shift_y);
    crossfade_vertical_seam(&mut shifted, radius);
    crossfade_horizontal_seam(&mut shifted, radius);
    shifted.clamp_unit();
    Ok(roll(&shifted, width - shift_x, height - shift_y))
}

/// How many pixels each side of a seam the cross-fade reaches, for an image
/// of this size and a fade amount in `0..=1`.
pub fn feather_radius(width: usize, height: usize, fade_amount: f32) -> usize {
    let strength = fade_amount.clamp(MIN_FADE_AMOUNT, 1.0);
    let shorter = width.min(height) as f32;
    ((shorter * MAX_FEATHER_FRACTION * strength).round() as usize).max(1)
}

/// Blend weight at `distance` pixels from the seam: an equal mix at the seam,
/// falling linearly to no influence at `radius`.
fn feather_weight(distance: usize, radius: usize) -> f32 {
    0.5 * (1.0 - distance as f32 / radius as f32)
}

fn mix_pair(a: &mut [f32], b: &mut [f32], weight: f32) {
    for (p, q) in a.iter_mut().zip(b.iter_mut()) {
        let (left, right) = (*p, *q);
        *p = left * (1.0 - weight) + right * weight;
        *q = right * (1.0 - weight) + left * weight;
    }
}

/// Feathers the seam running down the centre column.
fn crossfade_vertical_seam(image: &mut Image, radius: usize) {
    let (width, channels, row_len) = (image.width(), image.channels(), image.row_len());
    let centre = width / 2;
    let reach = radius.min(centre).min(width - centre);
    image.data_mut().par_chunks_mut(row_len).for_each(|row| {
        let (left, right) = row.split_at_mut(centre * channels);
        for distance in 0..reach {
            let near = (centre - 1 - distance) * channels;
            let far = distance * channels;
            let (Some(a), Some(b)) =
                (left.get_mut(near..near + channels), right.get_mut(far..far + channels))
            else {
                continue;
            };
            mix_pair(a, b, feather_weight(distance, radius));
        }
    });
}

/// Feathers the seam running along the centre row.
fn crossfade_horizontal_seam(image: &mut Image, radius: usize) {
    let (height, row_len) = (image.height(), image.row_len());
    let centre = height / 2;
    let reach = radius.min(centre).min(height - centre);
    let (top, bottom) = image.data_mut().split_at_mut(centre * row_len);
    for distance in 0..reach {
        let near = (centre - 1 - distance) * row_len;
        let far = distance * row_len;
        let (Some(a), Some(b)) =
            (top.get_mut(near..near + row_len), bottom.get_mut(far..far + row_len))
        else {
            continue;
        };
        mix_pair(a, b, feather_weight(distance, radius));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seamless::test_support::{column_difference, row_difference, sample};

    #[test]
    fn roll_is_a_circular_shift() {
        let image = Image::from_plane(4, 2, vec![0., 1., 2., 3., 4., 5., 6., 7.]).unwrap();

        let rolled = roll(&image, 1, 1);

        assert_eq!(rolled.data(), &[7., 4., 5., 6., 3., 0., 1., 2.]);
    }

    #[test]
    fn roll_by_zero_or_full_size_is_identity() {
        let image = sample(10, 6, 3);

        assert_eq!(roll(&image, 0, 0), image);
        assert_eq!(roll(&image, 10, 6), image);
    }

    #[test]
    fn opposite_rolls_cancel() {
        let image = sample(11, 7, 3);

        let back = roll(&roll(&image, 5, 3), 11 - 5, 7 - 3);

        assert_eq!(back, image);
    }

    #[test]
    fn crossfade_makes_outer_borders_equal() {
        let source = sample(128, 128, 3);

        let result = offset_crossfade(&source, 0.5).unwrap();

        assert!(column_difference(&result, 0, 127) < 1e-5);
        assert!(row_difference(&result, 0, 127) < 1e-5);
    }

    #[test]
    fn crossfade_leaves_pixels_far_from_the_border_untouched() {
        let source = sample(128, 128, 3);

        let result = offset_crossfade(&source, 0.5).unwrap();

        assert_eq!(result.pixel(64, 64), source.pixel(64, 64));
    }

    #[test]
    fn larger_fade_changes_more_of_the_image() {
        let source = sample(128, 128, 3);
        let changed = |amount: f32| {
            let result = offset_crossfade(&source, amount).unwrap();
            result.data().iter().zip(source.data()).filter(|(a, b)| a != b).count()
        };

        assert!(changed(1.0) > changed(0.2));
    }

    #[test]
    fn radius_larger_than_half_the_image_does_not_panic() {
        let mut image = sample(70, 64, 1);

        crossfade_vertical_seam(&mut image, 500);
        crossfade_horizontal_seam(&mut image, 500);

        assert!(image.data().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn odd_dimensions_are_supported() {
        let source = sample(101, 77, 3);

        let result = offset_crossfade(&source, 0.5).unwrap();

        assert_eq!((result.width(), result.height()), (101, 77));
    }
}
