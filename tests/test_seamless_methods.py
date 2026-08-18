import numpy as np

from app.core.materialize_methods import synthesis_overlap, synthesis_splat
from app.core.seamless import SeamlessProcessor


def _source(height=64, width=80, dtype=np.float32):
    values = np.arange(height * width * 3, dtype=np.float32).reshape(height, width, 3)
    values = (values * 0.73) % 255.0
    return values.astype(dtype)


def test_offset_crossfade_is_same_size_and_tileable():
    image = _source()
    processor = SeamlessProcessor()
    processor.load_image(image)

    result = processor.process(
        params={"method": "offset_crossfade"},
        use_cache=False,
        chunked=False,
    )

    assert result.shape == image.shape
    assert result.dtype == image.dtype
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result[:, 0], result[:, -1], atol=1e-5)
    np.testing.assert_allclose(result[0], result[-1], atol=1e-5)


def test_offset_crossfade_preserves_16_bit_range():
    image = (_source(dtype=np.float32) / 255.0 * 65535.0).astype(np.uint16)
    processor = SeamlessProcessor()
    processor.load_image(image)

    result = processor.process(
        params={"method": "standard"},
        use_cache=False,
        chunked=False,
    )

    assert result.dtype == np.uint16
    assert result.max() > 255
    np.testing.assert_array_equal(result[:, 0], result[:, -1])
    np.testing.assert_array_equal(result[0], result[-1])


def test_mirror_tiling_builds_exact_2_by_2_reflection():
    image = _source(dtype=np.uint8)
    processor = SeamlessProcessor()
    processor.load_image(image)

    result = processor.process(
        params={"method": "mirror_tiling"},
        use_cache=False,
        chunked=False,
    )
    height, width = image.shape[:2]

    assert result.shape == (height * 2, width * 2, 3)
    np.testing.assert_array_equal(result[:height, :width], image)
    np.testing.assert_array_equal(result[:height, width:], image[:, ::-1])
    np.testing.assert_array_equal(result[height:, :width], image[::-1, :])
    np.testing.assert_array_equal(result[height:, width:], image[::-1, ::-1])
    np.testing.assert_array_equal(result[:, 0], result[:, -1])
    np.testing.assert_array_equal(result[0, :], result[-1, :])


def _seam_to_interior_ratio(result):
    """How different the wrap-adjacent border is vs. typical neighbouring
    pixels. ~1.0 means the seam reads the same as ordinary local detail;
    much higher means a visible discontinuity.

    Overlap Blend and Splat Synthesis don't force literal pixel equality at
    the border -- for Splat that would mean discarding the weighted-average
    patch blending that keeps overlapping patches from ghosting (see
    test_splat.py). Both are tuned to be seamless in the sense that matters
    visually, verified here statistically rather than bit-for-bit.
    """
    o = result.astype(np.float64)
    v = np.mean(np.abs(o[0] - o[-1])) / np.mean(np.abs(o[1:] - o[:-1]))
    h = np.mean(np.abs(o[:, 0] - o[:, -1])) / np.mean(np.abs(o[:, 1:] - o[:, :-1]))
    return v, h


def test_overlap_and_splat_have_matching_repeat_borders():
    image = _source(128, 144)
    overlap = synthesis_overlap(image, overlap_x=0.2, overlap_y=0.2, falloff=0.2)
    splat, _ = synthesis_splat(
        image,
        new_size=(128, 144),
        scale=1.0,
        rand_rot=0.2,
        wobble=0.2,
        falloff=0.2,
    )

    for result in (overlap, splat):
        assert result.shape == image.shape
        assert result.dtype == np.float32
        v, h = _seam_to_interior_ratio(result)
        assert v < 1.5 and h < 1.5
