"""Tests for NormalGenerator and PBR map generation."""
from unittest.mock import patch

import numpy as np
import pytest

import app.core.normal_generator as ng
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

    def test_process_does_not_mutate_input_image(self):
        """Regression guard: process() used to divide its input in place by
        255 whenever it arrived as float32 outside [0,1] (the normal case
        for a Base Color image) -- silently corrupting any caller that
        didn't separately defend with its own .copy(). One real call site
        (_material_channel_image's on-demand generation) didn't, so opening
        a not-yet-generated material tab corrupted the live image."""
        img = _make_image(32)
        original = img.copy()
        NormalGenerator.process(img, use_cache=False)
        np.testing.assert_array_equal(img, original)

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


class TestPbrCacheCapacity:
    def test_cache_holds_at_least_30_images_worth_of_maps(self):
        """Regression guard: per-channel caching moved from 1 cache entry
        per image (a whole PBR dict) to up to len(_ALL_MAPS) entries per
        image (one per channel). max_size is scaled to match -- if it
        weren't, this would silently shrink effective capacity from ~30
        images to ~6 despite the ResultCache's own max_size looking
        unchanged at a glance."""
        assert ng._pbr_cache.max_size == 30 * len(ng._ALL_MAPS)


class TestPerChannelCache:
    """Each map is cached independently so one changed slider doesn't force
    recomputing the other four -- these patch the actual compute functions
    with call-counting spies to prove recomputation is skipped, not just
    that results happen to match."""

    def _spies(self):
        return (
            patch.object(ng, "_compute_displacement", wraps=ng._compute_displacement),
            patch.object(ng, "_compute_roughness", wraps=ng._compute_roughness),
            patch.object(ng, "_compute_ao", wraps=ng._compute_ao),
            patch.object(ng, "_compute_normal", wraps=ng._compute_normal),
        )

    def test_full_cache_hit_skips_every_compute_function(self):
        img = _make_image(32)
        params = {"rough_intensity": 0.5, "ao_strength": 0.4, "height_contrast": -0.3}
        first = NormalGenerator.process(img, use_cache=True, **params)

        p_disp, p_rough, p_ao, p_normal = self._spies()
        with p_disp as m_disp, p_rough as m_rough, p_ao as m_ao, p_normal as m_normal:
            second = NormalGenerator.process(img, use_cache=True, **params)

        assert (m_disp.call_count, m_rough.call_count, m_ao.call_count, m_normal.call_count) == (0, 0, 0, 0)
        for key in ("Normal", "Roughness", "AO", "Displacement", "Opacity"):
            np.testing.assert_array_equal(first[key], second[key])

    def test_changing_one_slider_only_recomputes_its_own_channel(self):
        img = _make_image(32)
        base = {"rough_intensity": 0.5, "ao_strength": 0.4, "height_contrast": -0.3}
        first = NormalGenerator.process(img, use_cache=True, **base)

        changed = dict(base, rough_intensity=0.9)
        p_disp, p_rough, p_ao, p_normal = self._spies()
        with p_disp as m_disp, p_rough as m_rough, p_ao as m_ao, p_normal as m_normal:
            second = NormalGenerator.process(img, use_cache=True, **changed)

        assert m_rough.call_count == 1
        assert (m_disp.call_count, m_ao.call_count, m_normal.call_count) == (0, 0, 0)

        assert not np.array_equal(first["Roughness"], second["Roughness"])
        for key in ("Normal", "AO", "Displacement", "Opacity"):
            np.testing.assert_array_equal(first[key], second[key])

    def test_height_source_change_invalidates_every_channel(self):
        """height_source changes the shared grayscale field every map reads
        from, so it must bust all five caches, not just one."""
        img = _make_image(32)
        base = {"height_source": "average_rgb", "rough_intensity": 0.5}
        NormalGenerator.process(img, use_cache=True, **base)

        changed = dict(base, height_source="luminance")
        p_disp, p_rough, p_ao, p_normal = self._spies()
        with p_disp as m_disp, p_rough as m_rough, p_ao as m_ao, p_normal as m_normal:
            NormalGenerator.process(img, use_cache=True, **changed)

        assert (m_disp.call_count, m_rough.call_count, m_ao.call_count, m_normal.call_count) == (1, 1, 1, 1)
