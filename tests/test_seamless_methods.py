import numpy as np

import app.core.seamless as sm
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


def test_overlap_accepts_uint8_material_channels():
    image = _source(dtype=np.uint8)
    processor = SeamlessProcessor()
    processor.load_image(image)

    result = processor.process(
        params={"method": "overlap"},
        use_cache=False,
        chunked=False,
    )

    assert result.dtype == np.uint8
    assert result.shape == image.shape
    assert int(result.min()) >= 0 and int(result.max()) <= 255


def test_splat_accepts_uint8_material_channels():
    image = _source(dtype=np.uint8)
    processor = SeamlessProcessor()
    processor.load_image(image)

    result = processor.process(
        params={"method": "splat"},
        use_cache=False,
        chunked=False,
    )

    assert result.dtype == np.uint8
    assert result.shape == image.shape
    assert int(result.min()) >= 0 and int(result.max()) <= 255


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


def test_crossfade_seam_blend_boundary_property():
    """The two columns immediately adjacent to the center seam are exactly
    averaged (weight=0.5 at distance=0). Checked on a row outside the
    vertical pass's own blend radius, since that pass runs second and
    also touches these columns near the horizontal center row."""
    image = _source(64, 80)
    radius = 10
    result = SeamlessProcessor._linear_crossfade_center_seams(image, radius)
    cx = 80 // 2
    row = 0  # outside the radius=10 vertical blend zone around cy=32
    expected = (image[row, cx - 1] + image[row, cx]) * 0.5
    np.testing.assert_allclose(result[row, cx - 1], expected, atol=1e-3)
    np.testing.assert_allclose(result[row, cx], expected, atol=1e-3)


def test_crossfade_seam_blend_handles_radius_larger_than_half_dimension():
    """A feather radius clamped against the image bounds must not index
    out of range or produce non-finite output."""
    image = _source(10, 12)
    result = SeamlessProcessor._linear_crossfade_center_seams(image, radius=50)
    assert result.shape == image.shape
    assert np.isfinite(result).all()


def test_crossfade_seam_blend_matches_reference_loop():
    """Regression guard: the vectorized blend must match the original
    per-offset Python loop it replaced, pixel for pixel (up to float
    rounding from reordered arithmetic)."""

    def reference_loop(image, radius):
        result = image.astype(np.float32, copy=True)
        h, w = image.shape[:2]
        cx, cy = w // 2, h // 2
        radius = max(1, int(radius))
        source = result.copy()
        for distance in range(radius):
            weight = 0.5 * (1.0 - (distance / radius))
            left, right = cx - 1 - distance, cx + distance
            if left >= 0 and right < w:
                lv, rv = source[:, left].copy(), source[:, right].copy()
                result[:, left] = lv * (1.0 - weight) + rv * weight
                result[:, right] = rv * (1.0 - weight) + lv * weight
        source = result.copy()
        for distance in range(radius):
            weight = 0.5 * (1.0 - (distance / radius))
            top, bottom = cy - 1 - distance, cy + distance
            if top >= 0 and bottom < h:
                tv, bv = source[top, :].copy(), source[bottom, :].copy()
                result[top, :] = tv * (1.0 - weight) + bv * weight
                result[bottom, :] = bv * (1.0 - weight) + tv * weight
        return np.clip(result, 0, 255.0).astype(image.dtype, copy=False)

    image = _source(96, 128)
    for radius in (1, 5, 30, 63):
        expected = reference_loop(image, radius)
        actual = SeamlessProcessor._linear_crossfade_center_seams(image, radius)
        np.testing.assert_allclose(
            actual.astype(np.float64), expected.astype(np.float64), atol=0.02
        )


class TestChunkedProcessingParallel:
    """run_pipeline_chunked processes tiles concurrently (a ThreadPoolExecutor
    over independent SeamlessProcessor instances) but reassembles them in a
    strictly sequential pass, since each tile's overlap blend reads
    already-written neighbour pixels. These pin down that the reassembly
    order survives parallel tile completion and that results are
    deterministic, not order-dependent on thread scheduling."""

    def _small_image(self, height=192, width=256):
        rng = np.random.RandomState(11)
        return rng.uniform(0, 255, (height, width, 3)).astype(np.float32)

    def test_chunked_output_matches_unchunked_on_uniform_content(self, monkeypatch):
        # A flat/uniform image makes every tile process identically, so
        # chunking (with its overlap blend) should reproduce the unchunked
        # result exactly -- a cheap way to check tile placement and the
        # blend math without depending on exact-equality across genuinely
        # different tile content.
        monkeypatch.setattr(sm, "_CHUNK_THRESHOLD_PX", 100)
        image = np.full((192, 256, 3), 128.0, dtype=np.float32)

        proc = SeamlessProcessor()
        proc.set_parameters(method="overlap")
        proc.load_image(image)
        chunked = proc.run_pipeline_chunked(image, chunk_size=96, overlap=16)

        proc2 = SeamlessProcessor()
        proc2.set_parameters(method="overlap")
        proc2.load_image(image)
        unchunked = proc2.process(image=image, preview=False, chunked=False)

        np.testing.assert_allclose(chunked, unchunked, atol=1e-3)

    def test_chunked_processing_is_deterministic_across_runs(self, monkeypatch):
        monkeypatch.setattr(sm, "_CHUNK_THRESHOLD_PX", 100)
        image = self._small_image()

        results = []
        for _ in range(3):
            proc = SeamlessProcessor()
            proc.set_parameters(method="overlap")
            proc.load_image(image)
            results.append(proc.run_pipeline_chunked(image, chunk_size=96, overlap=16))

        for r in results[1:]:
            np.testing.assert_array_equal(results[0], r)

    def test_chunked_output_shape_and_finite(self, monkeypatch):
        monkeypatch.setattr(sm, "_CHUNK_THRESHOLD_PX", 100)
        image = self._small_image(192, 288)

        proc = SeamlessProcessor()
        proc.set_parameters(method="splat")
        proc.load_image(image)
        result = proc.run_pipeline_chunked(image, chunk_size=96, overlap=16)

        assert result.shape == image.shape
        assert result.dtype == image.dtype
        assert np.isfinite(result).all()

    def test_trailing_remainder_tile_below_minimum_does_not_crash(self, monkeypatch):
        """Regression guard: dimensions that leave a trailing tile under
        SeamlessProcessor's 64px minimum used to raise ImageLoadError
        instead of processing. Unreachable through the app's own hardcoded
        call site (chunk_size=2048, overlap=64 always satisfies the
        minimum), but both are caller-supplied parameters with no
        guarantee a future caller keeps them safe -- this is the exact
        combination that first surfaced the crash."""
        monkeypatch.setattr(sm, "_CHUNK_THRESHOLD_PX", 100)
        image = self._small_image(200, 260)  # 200 % 96 = 8 -> an 8+16=24px
                                              # trailing tile without the fix

        proc = SeamlessProcessor()
        proc.set_parameters(method="overlap")
        proc.load_image(image)
        result = proc.run_pipeline_chunked(image, chunk_size=96, overlap=16)

        assert result.shape == image.shape
        assert np.isfinite(result).all()

    def test_leading_tile_below_minimum_does_not_crash(self, monkeypatch):
        """Regression guard for the other half of the same bug: pulling a
        tile's start backward (the trailing-tile fix above) is a no-op for
        the very first tile in a row/column, since its start is already
        clamped to 0 -- there's nothing to pull back into. If chunk_size +
        overlap alone is under the 64px minimum, that first tile stayed
        too small even after the trailing-tile fix. The fix now pushes the
        end forward instead when pulling the start back doesn't help."""
        monkeypatch.setattr(sm, "_CHUNK_THRESHOLD_PX", 100)
        image = self._small_image(150, 150)

        proc = SeamlessProcessor()
        proc.set_parameters(method="overlap")
        proc.load_image(image)
        # chunk_size=20, overlap=15 -> first tile's y1=min(0+20+15,150)=35,
        # y_start=max(0,0-15)=0, height=35 < 64, and pulling y_start back
        # further does nothing since it's already 0.
        result = proc.run_pipeline_chunked(image, chunk_size=20, overlap=15)

        assert result.shape == image.shape
        assert np.isfinite(result).all()


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
