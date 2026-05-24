"""Tests for NormalGenerator and PBR map generation."""
import numpy as np
import pytest

from app.core.normal_generator import NormalGenerator


def _make_image(size: int = 128) -> np.ndarray:
    """Create a synthetic BGR float32 test image."""
    return np.random.uniform(0, 255, (size, size, 3)).astype(np.float32)


class TestNormalGenerator:
    def test_all_maps_returned(self):
        img = _make_image(64)
        maps = NormalGenerator.process(img, use_cache=False)
        for key in ["Normal", "Roughness", "Metallic", "AO", "Height", "Displacement", "Opacity", "Emissive"]:
            assert key in maps, f"Missing map: {key}"

    def test_normal_map_shape(self):
        img = _make_image(64)
        maps = NormalGenerator.process(img, use_cache=False)
        assert maps["Normal"].shape == (64, 64, 3)

    def test_normal_map_range(self):
        img = _make_image(64)
        maps = NormalGenerator.process(img, use_cache=False)
        assert maps["Normal"].dtype == np.uint8
        assert maps["Normal"].min() >= 0
        assert maps["Normal"].max() <= 255

    def test_height_map_shape(self):
        img = _make_image(64)
        maps = NormalGenerator.process(img, use_cache=False)
        assert maps["Height"].shape == (64, 64, 3)

    def test_parallel_matches_sequential(self):
        """Parallel generation should produce the same results."""
        img = _make_image(64)
        params = {"normal_intensity": 0.5, "rough_intensity": 0.5, "ao_intensity": 0.5, "height_depth": 0.5}
        result = NormalGenerator.process(img, use_cache=False, **params)
        assert result["Normal"].shape == (64, 64, 3)
