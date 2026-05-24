"""Tests for ResultCache and key generation."""
import numpy as np
import pytest

from app.core.cache import ResultCache, hash_image, make_pipeline_key, make_pbr_key


class TestResultCache:
    def test_cache_hit(self):
        cache = ResultCache(max_size=10)
        arr = np.zeros((64, 64, 3), dtype=np.float32)
        cache.set({"a": 1}, arr, "img1")
        result = cache.get({"a": 1}, "img1")
        assert result is not None
        np.testing.assert_array_equal(result, arr)

    def test_cache_miss(self):
        cache = ResultCache(max_size=10)
        result = cache.get({"a": 1}, "nonexistent")
        assert result is None

    def test_cache_eviction(self):
        cache = ResultCache(max_size=2)
        arr = np.zeros((8, 8), dtype=np.float32)
        cache.set({"k": 1}, arr, "h1")
        cache.set({"k": 2}, arr, "h1")
        cache.set({"k": 3}, arr, "h1")
        # First should be evicted
        result = cache.get({"k": 1}, "h1")
        assert result is None
        # Last two should remain
        assert cache.get({"k": 2}, "h1") is not None
        assert cache.get({"k": 3}, "h1") is not None

    def test_make_pipeline_key_stable(self):
        arr = np.zeros((64, 64, 3), dtype=np.float32)
        key1 = make_pipeline_key(arr, {"method": "overlap", "falloff": 0.5})
        key2 = make_pipeline_key(arr, {"method": "overlap", "falloff": 0.5})
        assert key1 == key2

    def test_make_pipeline_key_differs(self):
        arr = np.zeros((64, 64, 3), dtype=np.float32)
        key1 = make_pipeline_key(arr, {"method": "overlap", "falloff": 0.5})
        key2 = make_pipeline_key(arr, {"method": "splat", "falloff": 0.5})
        assert key1 != key2

    def test_make_pbr_key_stable(self):
        arr = np.zeros((64, 64, 3), dtype=np.float32)
        key1 = make_pbr_key(arr, {"normal_intensity": 0.5})
        key2 = make_pbr_key(arr, {"normal_intensity": 0.5})
        assert key1 == key2

    def test_pbr_cache_roundtrip(self):
        cache = ResultCache(max_size=10)
        maps = {"Normal": np.zeros((64, 64, 3), dtype=np.uint8)}
        cache.set_pbr("test_key", maps)
        result = cache.get_pbr("test_key")
        assert result is not None
        assert "Normal" in result
