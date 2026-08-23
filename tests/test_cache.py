"""Tests for ResultCache and key generation."""
import threading

import numpy as np
import pytest

from app.core.cache import ResultCache, hash_image, make_pipeline_key


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

    def test_pipeline_cache_roundtrip(self):
        cache = ResultCache(max_size=10)
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        cache.set_pipeline("test_key", arr)
        result = cache.get_pipeline("test_key")
        assert result is not None
        np.testing.assert_array_equal(result, arr)


class TestResultCacheThreadSafety:
    """ResultCache is hit concurrently in production: a background
    MaterialMapThread recomputing a map races the GUI thread exporting a
    not-yet-generated one, both against the same module-level PBR cache.
    _touch()'s "if key in access_order: remove(key)" and _evict_to_fit()'s
    eviction loop are check-then-act sequences that aren't atomic across
    threads under the GIL's time-sliced switching without a lock."""

    def test_concurrent_get_set_does_not_raise(self):
        cache = ResultCache(max_size=8, max_bytes=10 * 1024 * 1024)
        arr = np.zeros((32, 32, 3), dtype=np.float32)
        errors = []

        def worker(n):
            try:
                for i in range(200):
                    key = f"k{(n * 200 + i) % 20}"
                    cache.set_pipeline(key, arr)
                    cache.get_pipeline(key)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"concurrent cache access raised: {errors}"
        # Internal bookkeeping should still be self-consistent afterward.
        assert len(cache.cache) == len(cache.access_order)
        assert set(cache.cache.keys()) == set(cache.access_order)
