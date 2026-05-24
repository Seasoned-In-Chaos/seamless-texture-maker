"""Tests for edge blending algorithms."""
import numpy as np
import pytest

from app.core.edge_blending import blend_seams, create_blend_mask
from app.core.edge_blending_jit import blend_seams_fast


def _make_offset_image(size: int = 128) -> np.ndarray:
    """Create a synthetic float32 image with a visible seam at center."""
    img = np.random.uniform(0, 255, (size, size, 3)).astype(np.float32)
    # Create seam: offset by half
    return np.roll(np.roll(img, size // 2, axis=0), size // 2, axis=1)


class TestEdgeBlending:
    def test_blend_output_shape(self):
        img = _make_offset_image(128)
        result = blend_seams(img, blend_strength=0.5, smoothness=0.5, symmetric=True)
        assert result.shape == img.shape

    def test_blend_dtype(self):
        img = _make_offset_image(128)
        result = blend_seams(img, blend_strength=0.5, smoothness=0.5, symmetric=True)
        assert result.dtype == np.float32

    def test_blend_seam_continuity(self):
        img = _make_offset_image(128)
        result = blend_seams(img, blend_strength=0.5, smoothness=0.5, symmetric=True)
        # Center cross-section should have lower variance than raw seam
        center_col = result[:, 64, :]
        assert np.var(center_col) < np.var(img[:, 64, :])

    def test_blend_symmetric(self):
        img = _make_offset_image(128)
        result = blend_seams(img, blend_strength=0.5, smoothness=0.5, symmetric=True)
        # Left and right of center should be symmetric
        cx = 64
        left = result[32, cx - 8, 0]
        right = result[32, cx + 8, 0]
        assert abs(left - right) < 50.0  # Rough parity check

    def test_blend_small_width(self):
        img = _make_offset_image(8)
        result = blend_seams(img, blend_strength=0.01, smoothness=0.5, symmetric=True)
        assert result.shape == img.shape


class TestEdgeBlendingFast:
    def test_fast_output_shape(self):
        img = _make_offset_image(128)
        result = blend_seams_fast(img, blend_strength=0.5, smoothness=0.5)
        assert result.shape == img.shape

    def test_fast_dtype(self):
        img = _make_offset_image(128)
        result = blend_seams_fast(img, blend_strength=0.5, smoothness=0.5)
        assert result.dtype == np.float32

    def test_rust_python_parity(self):
        try:
            from seams_core import edge_blend_symmetric
        except ImportError:
            pytest.skip("Rust extension not available")

        img = _make_offset_image(128)
        py_result = blend_seams_fast(img, blend_strength=0.5, smoothness=0.5)
        rs_result = edge_blend_symmetric(img, 16, True)
        assert py_result.shape == rs_result.shape
        np.testing.assert_allclose(py_result, rs_result, atol=1e-2)
