"""
Result caching system for instant preview updates.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Dict, Optional

import numpy as np

__all__ = ["ResultCache", "hash_image", "make_pipeline_key", "make_pbr_key"]

logger = logging.getLogger("seams.cache")


class ResultCache:
    """LRU cache for processed texture results."""

    def __init__(self, max_size: int = 50) -> None:
        self.cache: Dict[str, np.ndarray] = {}
        self.max_size = max_size
        self.access_order: list[str] = []

    def _hash_params(self, params: dict) -> str:
        """Create hash key from parameters."""
        param_str = str(sorted(params.items()))
        return hashlib.md5(param_str.encode()).hexdigest()

    def _evict_if_full(self, key: str) -> None:
        if len(self.cache) >= self.max_size and key not in self.cache:
            if self.access_order:
                oldest = self.access_order.pop(0)
                del self.cache[oldest]

    def _touch(self, key: str) -> None:
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

    def get(self, params: dict, image_hash: Optional[str] = None) -> Optional[np.ndarray]:
        """Get cached result if available.

        Returns a *copy* so that callers cannot accidentally mutate
        the cached array.
        """
        key = self._hash_params(params)
        if image_hash:
            key = f"{image_hash}_{key}"

        if key in self.cache:
            self._touch(key)
            logger.debug("cache HIT  key=%s", key[:16])
            return self.cache[key].copy()

        logger.debug("cache MISS key=%s", key[:16])
        return None

    def set(self, params: dict, result: np.ndarray, image_hash: Optional[str] = None) -> None:
        """Store result in cache (makes an internal copy)."""
        key = self._hash_params(params)
        if image_hash:
            key = f"{image_hash}_{key}"

        self._evict_if_full(key)
        self.cache[key] = result.copy()
        self._touch(key)

    def get_pipeline(self, key: str) -> Optional[np.ndarray]:
        """Retrieve a pipeline result by its pre-computed key.

        Returns a *copy* so that callers cannot accidentally mutate
        the cached array.
        """
        if key in self.cache:
            self._touch(key)
            logger.debug("cache HIT (pipe) key=%s", key[:16])
            return self.cache[key].copy()
        logger.debug("cache MISS (pipe) key=%s", key[:16])
        return None

    def set_pipeline(self, key: str, result: np.ndarray) -> None:
        """Store a pipeline result by its pre-computed key (makes an internal copy)."""
        self._evict_if_full(key)
        self.cache[key] = result.copy()
        self._touch(key)

    def get_pbr(self, key: str) -> Optional[Dict[str, np.ndarray]]:
        """Retrieve a full PBR map dict by its pre-computed key.

        Returns a deep copy (each array is copied) so the caller
        can freely modify the result.
        """
        if key in self.cache:
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            logger.debug("cache HIT (pbr) key=%s", key[:16])
            stored = self.cache[key]
            if isinstance(stored, dict):
                return {k: v.copy() for k, v in stored.items()}
            return None
        logger.debug("cache MISS (pbr) key=%s", key[:16])
        return None

    def set_pbr(self, key: str, maps: Dict[str, np.ndarray]) -> None:
        """Store a full PBR map dict (makes deep copies)."""
        self._evict_if_full(key)
        self.cache[key] = {k: v.copy() for k, v in maps.items()}
        self._touch(key)

    def clear(self) -> None:
        """Clear all cached results."""
        self.cache.clear()
        self.access_order.clear()

    def get_stats(self) -> Dict[str, object]:
        """Get cache statistics."""
        mem = 0
        for v in self.cache.values():
            if isinstance(v, dict):
                mem += sum(a.nbytes for a in v.values())
            elif isinstance(v, np.ndarray):
                mem += v.nbytes
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "memory_mb": mem / (1024 * 1024),
        }


def hash_image(image: np.ndarray) -> str:
    """Create fast hash of image for cache key."""
    h, w = image.shape[:2]
    step = max(h // 16, w // 16, 1)
    sample = image[::step, ::step]
    hash_data = f"{image.shape}_{sample.tobytes()}"
    return hashlib.md5(hash_data.encode()).hexdigest()[:8]


def make_pipeline_key(image: np.ndarray, params: dict) -> str:
    """Build a stable cache key for the seamless pipeline.

    Combines an image content hash with a JSON-serialised parameter
    dict (keys sorted for determinism).
    """
    img_hash = hash_image(image)
    param_str = json.dumps(params, sort_keys=True)
    param_hash = hashlib.md5(param_str.encode()).hexdigest()
    return f"pipe_{img_hash}_{param_hash}"


def make_pbr_key(image: np.ndarray, pbr_params: dict) -> str:
    """Build a stable cache key for PBR map generation.

    Combines an image content hash with PBR-relevant parameter
    values (keys sorted for determinism).
    """
    img_hash = hash_image(image)
    param_str = json.dumps(pbr_params, sort_keys=True)
    param_hash = hashlib.md5(param_str.encode()).hexdigest()
    return f"pbr_{img_hash}_{param_hash}"
