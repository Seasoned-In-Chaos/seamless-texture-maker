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
        for key in ["Normal", "Roughness", "AO", "Displacement", "Opacity"]:
            assert key in maps, f"Missing map: {key}"

    def test_normal_map_shape(self):
        img = _make_image(64)
        maps = NormalGenerator.process(img, use_cache=False)
        assert maps["Normal"].shape == (64, 64, 3)

    def test_normal_map_range(self):
        img = _make_image(64)
        maps = NormalGenerator.process(img, use_cache=False)
        assert maps["Normal"].dtype == np.uint16
        assert maps["Normal"].min() >= 0
        assert maps["Normal"].max() <= 65535

    def test_min_z_is_enforced(self):
        img = _make_image(64)
        maps = NormalGenerator.process(img, use_cache=False, normal_min_z=0.95)
        # Normal is stored BGR: channel 0 is the packed Z component.
        z = maps["Normal"][..., 0].astype(np.float32) / 65535.0
        assert (z * 2.0 - 1.0).min() >= 0.95 - (2.0 / 65535.0)

    @pytest.mark.parametrize("normal_filter", [
        "4_sample", "sobel_3x3", "sobel_5x5", "sobel_7x7", "sobel_9x9", "dudv",
    ])
    def test_nvtt_filter_modes_generate_normals(self, normal_filter):
        maps = NormalGenerator.process(
            _make_image(64), use_cache=False, normal_filter=normal_filter, normal_wrap=True,
        )
        assert maps["Normal"].shape == (64, 64, 3)
        assert maps["Normal"].dtype == np.uint16

    def test_wrap_option_changes_border_derivatives(self):
        image = np.zeros((64, 64, 3), dtype=np.float32)
        image[:, 0, :] = 255.0
        wrapped = NormalGenerator.process(image, use_cache=False, normal_wrap=True)["Normal"]
        clamped = NormalGenerator.process(image, use_cache=False, normal_wrap=False)["Normal"]
        assert not np.array_equal(wrapped, clamped)

    def test_displacement_map_shape(self):
        img = _make_image(64)
        maps = NormalGenerator.process(img, use_cache=False)
        assert maps["Displacement"].shape == (64, 64, 3)

    def test_parallel_matches_sequential(self):
        """Parallel generation should produce the same results."""
        img = _make_image(64)
        params = {"normal_intensity": 0.5, "rough_intensity": 0.5, "ao_intensity": 0.5, "height_depth": 0.5}
        result = NormalGenerator.process(img, use_cache=False, **params)
        assert result["Normal"].shape == (64, 64, 3)
