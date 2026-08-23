"""Tests for delighting.py, particularly the downsample/blur/upsample
approximation _blur_lowfreq uses for sigma > 24.

Verified manually earlier (old direct-blur vs. this approximation, on a
real test photo, across 7 parameter combinations: mean diff 0.15-0.37/255,
worst-case max diff 6/255 under extreme all-sliders-maxed settings) -- this
file turns that one-off verification into a permanent, portable regression
guard so a future change to the approximation can't silently drift past
that tolerance without a test failing. A real photo isn't available on
every machine that runs this suite, so the image here is synthetic but
deliberately not just noise: a smooth large-scale gradient (the "broad
lighting" _blur_lowfreq exists to extract) plus sharp edges placed at and
near the border (the specific region where downsample/upsample resampling
behaves differently from a direct blur) plus higher-frequency detail.
"""
import cv2
import numpy as np
import pytest

from app.core.delighting import _blur_lowfreq, delight_image


def _make_field(size: int = 512) -> np.ndarray:
    """A single-channel float32 field with broad gradient + border edges +
    fine detail -- not pure noise, so it actually exercises low-frequency
    extraction the way a real photo's lighting field would."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    gradient = 80.0 + 120.0 * (x / size) + 40.0 * np.sin(y / size * np.pi)

    field = gradient.copy()
    # Sharp rectangles touching or near the border, where downsample/
    # upsample resampling most plausibly diverges from a direct blur.
    field[0:20, :] += 60.0
    field[:, -15:] -= 50.0
    field[size // 2 - 10:size // 2 + 10, size // 2 - 10:size // 2 + 10] += 90.0

    rng = np.random.RandomState(42)
    field += rng.uniform(-8.0, 8.0, size=(size, size)).astype(np.float32)

    return np.clip(field, 0, 255).astype(np.float32)


class TestBlurLowfreqApproximation:
    @pytest.mark.parametrize("sigma", [30.0, 90.0, 200.0, 400.0, 600.0])
    def test_matches_direct_blur_within_tolerance(self, sigma):
        """The downsample path (sigma > 24) should stay close to a direct
        full-resolution Gaussian blur at the same sigma -- the ground
        truth it's approximating."""
        field = _make_field()
        k = max(3, int(sigma * 3) | 1)
        direct = cv2.GaussianBlur(field, (k, k), sigma)
        approx = _blur_lowfreq(field, sigma)

        assert approx.shape == field.shape
        diff = np.abs(approx.astype(np.float64) - direct.astype(np.float64))
        # Bounds calibrated against this image's measured diff curve (mean
        # 0.06 -> 2.36/255, max 0.61 -> 5.11/255 as sigma runs 30 -> 600),
        # which itself lines up with the earlier real-photo check (max
        # diff 6/255 worst case). Kept tight enough that a genuine
        # regression -- e.g. a downscale-factor change that under-resolves
        # the field -- would still clear these bounds by a wide margin.
        assert diff.mean() < 3.5, f"mean diff {diff.mean():.2f}/255 at sigma={sigma}"
        assert diff.max() < 8.0, f"max diff {diff.max():.2f}/255 at sigma={sigma}"

    @pytest.mark.parametrize("sigma", [30.0, 200.0, 600.0])
    def test_border_region_matches_within_tolerance(self, sigma):
        """Border rows/columns specifically -- GaussianBlur's reflect
        padding is applied at full resolution on the direct path but at
        reduced resolution on the approximation path, which is exactly
        where the two could diverge more than the interior does."""
        field = _make_field()
        k = max(3, int(sigma * 3) | 1)
        direct = cv2.GaussianBlur(field, (k, k), sigma)
        approx = _blur_lowfreq(field, sigma)

        border = 10
        direct_border = np.concatenate([
            direct[:border, :].ravel(), direct[-border:, :].ravel(),
            direct[:, :border].ravel(), direct[:, -border:].ravel(),
        ])
        approx_border = np.concatenate([
            approx[:border, :].ravel(), approx[-border:, :].ravel(),
            approx[:, :border].ravel(), approx[:, -border:].ravel(),
        ])
        diff = np.abs(approx_border.astype(np.float64) - direct_border.astype(np.float64))
        # Measured border mean 0.26 -> 2.34/255, max 0.61 -> 4.93/255 across
        # the same sigma range -- tracking the interior bounds above, since
        # border pixels turned out not to diverge more than interior ones.
        assert diff.mean() < 3.5, f"border mean diff {diff.mean():.2f}/255 at sigma={sigma}"
        assert diff.max() < 8.0, f"border max diff {diff.max():.2f}/255 at sigma={sigma}"

    def test_sigma_at_or_below_24_is_exact_direct_blur(self):
        """The documented boundary: sigma <= 24 must take the plain
        cv2.GaussianBlur path, not the downsample approximation."""
        field = _make_field(128)
        for sigma in (5.0, 15.0, 24.0):
            k = max(3, int(sigma * 3) | 1)
            expected = cv2.GaussianBlur(field, (k, k), sigma)
            np.testing.assert_array_equal(_blur_lowfreq(field, sigma), expected)

    def test_shape_preserved_across_sigma_range(self):
        # Non-square field, since downscale/upscale could mishandle it.
        field = _make_field(256)[:, :180]
        for sigma in (10.0, 24.0, 25.0, 100.0, 600.0):
            result = _blur_lowfreq(field, sigma)
            assert result.shape == field.shape
            assert result.dtype == np.float32
            assert np.isfinite(result).all()


class TestDelightImageEndToEnd:
    """Basic sanity across delight_image's full parameter surface, using
    the same non-trivial synthetic image -- catches a crash or an
    out-of-range result from any of the _blur_lowfreq call sites."""

    def _bgr_image(self):
        field = _make_field(256)
        return np.stack([field, field * 0.9, field * 1.05], axis=-1).astype(np.float32)

    @pytest.mark.parametrize("params", [
        {"strength": 0.5},
        {"strength": 1.0},
        {"strength": 0.5, "detail_preservation": 1.0},
        {"strength": 0.5, "ao_removal": 1.0},
        {"strength": 1.0, "flatness": 1.0, "shadow_removal": 1.0,
         "highlight_reduction": 1.0, "contrast_recovery": 1.0,
         "detail_preservation": 1.0, "color_preservation": 1.0,
         "ao_removal": 1.0, "edge_consistency": 1.0},
    ])
    def test_output_in_valid_range(self, params):
        image = self._bgr_image()
        result = delight_image(image, **params)
        assert result.shape == image.shape
        assert result.dtype == np.float32
        assert np.isfinite(result).all()
        assert result.min() >= 0.0 and result.max() <= 255.0
