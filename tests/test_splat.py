"""Tests for splat synthesis."""
import numpy as np
import pytest

from app.core.materialize_methods import synthesis_splat


def _make_source(size: int = 128) -> np.ndarray:
    """Create a synthetic float32 source texture."""
    return np.random.uniform(0, 255, (size, size, 3)).astype(np.float32)


class TestSplat:
    def test_splat_output_shape(self):
        img = _make_source(128)
        result, _ = synthesis_splat(img, new_size=(128, 128), scale=1.0, falloff=0.2)
        assert result.shape == (128, 128, 3)

    def test_splat_dtype(self):
        img = _make_source(128)
        result, _ = synthesis_splat(img, new_size=(128, 128), scale=1.0, falloff=0.2)
        assert result.dtype == np.float32

    def test_splat_deterministic(self):
        img = _make_source(128)
        r1, _ = synthesis_splat(img, new_size=(64, 64), scale=1.0, falloff=0.2)
        r2, _ = synthesis_splat(img, new_size=(64, 64), scale=1.0, falloff=0.2)
        np.testing.assert_array_equal(r1, r2)

    def test_splat_different_seeds(self):
        img = _make_source(128)
        r1, _ = synthesis_splat(img, new_size=(64, 64), scale=1.0, falloff=0.2)
        # Different wobble should produce different result
        r2, _ = synthesis_splat(img, new_size=(64, 64), scale=1.0, wobble=0.8, falloff=0.2)
        assert not np.array_equal(r1, r2)
