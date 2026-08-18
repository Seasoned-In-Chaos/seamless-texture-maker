"""Tests for overlap blend synthesis."""
import numpy as np
import pytest

from app.core.materialize_methods import (
    synthesis_overlap,
    _seam_fade_ramp,
    _resize_wrapped,
)


def _make_source(size: int = 128, seed: int = 0) -> np.ndarray:
    """Create a synthetic float32 source texture."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 255, (size, size, 3)).astype(np.float32)


def _seam_error(img: np.ndarray) -> float:
    """Mean abs difference between the wrap-adjacent edges of an image."""
    h_err = np.mean(np.abs(img[:, 0].astype(np.float64) - img[:, -1].astype(np.float64)))
    v_err = np.mean(np.abs(img[0, :].astype(np.float64) - img[-1, :].astype(np.float64)))
    return max(h_err, v_err)


class TestOverlapBlend:
    def test_zero_overlap_leaves_image_unchanged(self):
        img = _make_source(64)
        result = synthesis_overlap(img, overlap_x=0.0, overlap_y=0.0, falloff=0.5)
        np.testing.assert_array_equal(result, np.clip(img, 0, 255))

    def test_overlap_changes_image_even_at_zero_falloff(self):
        # Overlap X/Y replace image content, so they must always affect the
        # output -- otherwise the live preview looks frozen until the user
        # happens to raise Edge Falloff.
        img = _make_source(64)
        untouched = synthesis_overlap(img, overlap_x=0.0, overlap_y=0.0, falloff=0.0)
        overlapped = synthesis_overlap(img, overlap_x=0.25, overlap_y=0.15, falloff=0.0)
        assert not np.allclose(untouched, overlapped)

    def test_output_shape_and_dtype_preserved(self):
        img = _make_source(64)
        result = synthesis_overlap(img, overlap_x=0.2, overlap_y=0.2, falloff=0.5)
        assert result.shape == img.shape
        assert result.dtype == np.float32

    def test_blend_reduces_seam_error(self):
        img = _make_source(64)
        assert _seam_error(img) > 10  # random source has no inherent seam match
        result = synthesis_overlap(img, overlap_x=0.3, overlap_y=0.3, falloff=0.4)
        assert _seam_error(result) < _seam_error(img)

    def test_deterministic(self):
        img = _make_source(64)
        r1 = synthesis_overlap(img, overlap_x=0.2, overlap_y=0.2, falloff=0.5)
        r2 = synthesis_overlap(img, overlap_x=0.2, overlap_y=0.2, falloff=0.5)
        np.testing.assert_array_equal(r1, r2)

    def test_grayscale_input(self):
        img = _make_source(64)[:, :, 0]
        result = synthesis_overlap(img, overlap_x=0.2, overlap_y=0.2, falloff=0.5)
        assert result.shape == img.shape


class TestSeamFadeRamp:
    def test_starts_at_full_copy(self):
        # Index 0 must be exactly 1.0 or the tile stops wrapping.
        for falloff in (0.0, 0.25, 0.5, 1.0):
            ramp = _seam_fade_ramp(100, falloff)
            assert ramp[0] == pytest.approx(1.0)

    def test_leading_run_is_a_solid_copy(self):
        # Regression test: the strip must begin with a substantial run of
        # fully-copied pixels -- that run *is* the overlap. A ramp that
        # starts decaying immediately replaces almost nothing, so the tile
        # reads as a hard-cut offset rather than an overlap.
        blend_n = 200
        for falloff in (0.0, 0.5, 1.0):
            ramp = _seam_fade_ramp(blend_n, falloff)
            solid = np.sum(ramp >= 0.999)
            assert solid > blend_n * 0.2

    def test_zero_falloff_is_a_hard_step_at_the_midpoint(self):
        blend_n = 200
        ramp = _seam_fade_ramp(blend_n, 0.0)
        assert set(np.unique(ramp)) <= {0.0, 1.0}
        assert np.count_nonzero(ramp) == pytest.approx(blend_n // 2, abs=1)

    def test_higher_falloff_widens_the_transition(self):
        def transition_width(falloff):
            ramp = _seam_fade_ramp(200, falloff)
            return np.sum((ramp > 0.01) & (ramp < 0.99))

        assert transition_width(0.0) < transition_width(0.25) < transition_width(1.0)

    def test_near_fifty_fifty_zone_stays_narrow(self):
        blend_n = 200
        ramp = _seam_fade_ramp(blend_n, 1.0)
        mixed = np.sum((ramp > 0.2) & (ramp < 0.8))
        assert mixed < blend_n * 0.2


class TestResizeWrapped:
    def test_identity_when_size_matches(self):
        img = _make_source(32)
        assert _resize_wrapped(img, 32, 32) is img

    def test_preserves_wrap_continuity_better_than_plain_resize(self):
        # Regression test: a plain cv2.resize interpolates edge pixels
        # against a clamped repeat of themselves rather than the content
        # they wrap into, leaving a faint 1px line at the tile boundary.
        # A smooth periodic signal is continuous across the wrap, so any
        # jump at the seam is resampling error rather than real content.
        import cv2

        size = 64
        x = np.arange(size, dtype=np.float32)
        wave = 128 + 100 * np.sin(2 * np.pi * x / size)
        img = np.repeat(wave[np.newaxis, :], size, axis=0)
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2).astype(np.float32)

        def seam_error(out):
            out = out.astype(np.float64)
            # The wrap-adjacent pair should differ by about the same amount
            # as any other adjacent pair near the seam.
            jump = np.mean(np.abs(out[:, 0] - out[:, -1]))
            local = np.mean(np.abs(out[:, 1] - out[:, 0]))
            return abs(jump - local)

        wrapped = _resize_wrapped(img, size * 2, size * 2)
        clamped = cv2.resize(img, (size * 2, size * 2), interpolation=cv2.INTER_LINEAR)

        assert seam_error(wrapped) < seam_error(clamped) * 0.5

    def test_handles_grayscale(self):
        img = _make_source(32)[:, :, 0]
        out = _resize_wrapped(img, 48, 48)
        assert out.shape == (48, 48)
