"""
Result caching system for instant preview updates.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from typing import Dict, Optional

import numpy as np

__all__ = ["ResultCache", "hash_image", "make_pipeline_key"]

logger = logging.getLogger("seams.cache")


class ResultCache:
    """LRU cache for processed texture results.

    Bounded by both entry count (``max_size``) and total memory
    (``max_bytes``) -- whichever limit is hit first triggers eviction.
    Results can be full-resolution float32 arrays up to the app's 8192px
    cap (~800MB each), so a count-only limit cannot prevent unbounded
    memory growth on large textures; a byte budget is required.

    Thread-safe: a single ``ResultCache`` instance (e.g. the module-level
    PBR cache) can be hit concurrently -- a background QThread recomputing
    a material map races the GUI thread exporting a not-yet-generated one.
    ``self.cache``/``self.access_order`` mutation is a check-then-act
    sequence (e.g. ``_touch``'s ``if key in access_order: ... remove(key)``)
    that is not atomic across threads under the GIL's time-sliced
    switching, so it's guarded by a lock rather than relying on individual
    dict/list operations happening to be atomic.
    """

    def __init__(self, max_size: int = 50, max_bytes: int = 2 * 1024 ** 3) -> None:
        self.cache: Dict[str, np.ndarray] = {}
        self.max_size = max_size
        self.max_bytes = max_bytes
        self.access_order: list[str] = []
        self._total_bytes = 0
        self._lock = threading.Lock()

    def _hash_params(self, params: dict) -> str:
        """Create hash key from parameters."""
        param_str = str(sorted(params.items()))
        return hashlib.md5(param_str.encode()).hexdigest()

    @staticmethod
    def _entry_bytes(value: np.ndarray) -> int:
        return value.nbytes

    def _evict_to_fit(self, key: str, incoming_bytes: int) -> None:
        """Evict oldest entries until both the count and byte budgets fit
        the incoming entry (or nothing older is left to evict).

        Caller must hold ``self._lock`` and have already confirmed
        ``key not in self.cache``.
        """
        while self.access_order and (
            len(self.cache) >= self.max_size
            or self._total_bytes + incoming_bytes > self.max_bytes
        ):
            oldest = self.access_order.pop(0)
            self._total_bytes -= self._entry_bytes(self.cache.pop(oldest))

    def _store(self, key: str, value: np.ndarray) -> None:
        """Insert or overwrite `key`. Caller must hold ``self._lock``."""
        incoming = self._entry_bytes(value)
        if key in self.cache:
            self._total_bytes -= self._entry_bytes(self.cache[key])
        else:
            self._evict_to_fit(key, incoming)
        self.cache[key] = value
        self._total_bytes += incoming
        self._touch(key)

    def _touch(self, key: str) -> None:
        """Caller must hold ``self._lock``."""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

    def _get(self, key: str) -> Optional[np.ndarray]:
        """Shared get path: returns a *copy* so callers can't mutate the
        cached array, and keeps the lock held only for the dict/list
        touch, not the copy itself."""
        with self._lock:
            stored = self.cache.get(key)
            if stored is not None:
                self._touch(key)
        if stored is not None:
            return stored.copy()
        return None

    def _set(self, key: str, result: np.ndarray) -> None:
        """Shared set path: the copy happens before the lock is taken,
        since it doesn't touch any shared state."""
        value = result.copy()
        with self._lock:
            self._store(key, value)

    def get(self, params: dict, image_hash: Optional[str] = None) -> Optional[np.ndarray]:
        """Get cached result if available (a copy, safe to mutate)."""
        key = self._hash_params(params)
        if image_hash:
            key = f"{image_hash}_{key}"
        result = self._get(key)
        logger.debug("cache %s key=%s", "HIT " if result is not None else "MISS", key[:16])
        return result

    def set(self, params: dict, result: np.ndarray, image_hash: Optional[str] = None) -> None:
        """Store result in cache (makes an internal copy)."""
        key = self._hash_params(params)
        if image_hash:
            key = f"{image_hash}_{key}"
        self._set(key, result)

    def get_pipeline(self, key: str) -> Optional[np.ndarray]:
        """Retrieve a pipeline result by its pre-computed key (a copy)."""
        result = self._get(key)
        logger.debug("cache %s (pipe) key=%s", "HIT " if result is not None else "MISS", key[:16])
        return result

    def set_pipeline(self, key: str, result: np.ndarray) -> None:
        """Store a pipeline result by its pre-computed key (makes an internal copy)."""
        self._set(key, result)

    def clear(self) -> None:
        """Clear all cached results."""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
            self._total_bytes = 0

    def get_stats(self) -> Dict[str, object]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "memory_mb": self._total_bytes / (1024 * 1024),
                "max_memory_mb": self.max_bytes / (1024 * 1024),
            }


_HASH_RNG_SEED = 0x5EA415  # fixed seed: same image -> same hash every call
_HASH_N_RANDOM = 1024
# Fixed sample positions as fractions of (height, width), computed once at
# import time rather than reseeding a RandomState on every hash_image()
# call -- that reseed alone was ~85% of the function's cost (measured
# ~170us/call vs ~9us before this random sample was added), on a path
# called on every interactive live-preview tick. Scaling these fractions
# to whatever image is passed in is a handful of float multiplies.
_HASH_RANDOM_FRACS = np.random.RandomState(_HASH_RNG_SEED).random_sample(
    (2, _HASH_N_RANDOM)
).astype(np.float64)


def hash_image(image: np.ndarray) -> str:
    """Create fast hash of image for cache key.

    Combines the regular grid sample (cheap, catches most real content
    differences) with a fixed-seed random sample of up to 1024 pixels.
    The grid alone is vulnerable to aliasing: a regularly-patterned image
    (a checkerboard, a repeating grid texture) can differ everywhere
    except at exactly the sampled stride and still collide. A random
    sample has no reason to align with an unrelated image's own
    periodicity, at the same negligible cost.
    """
    h, w = image.shape[:2]
    step = max(h // 16, w // 16, 1)
    grid_sample = image[::step, ::step]

    n_random = min(_HASH_N_RANDOM, h * w)
    ys = np.minimum((_HASH_RANDOM_FRACS[0, :n_random] * h).astype(np.intp), h - 1)
    xs = np.minimum((_HASH_RANDOM_FRACS[1, :n_random] * w).astype(np.intp), w - 1)
    random_sample = image[ys, xs]

    hash_data = (
        f"{image.shape}_{image.dtype}_".encode()
        + grid_sample.tobytes()
        + random_sample.tobytes()
    )
    return hashlib.md5(hash_data).hexdigest()[:8]


def make_pipeline_key(image: np.ndarray, params: dict) -> str:
    """Build a stable cache key for the seamless pipeline.

    Combines an image content hash with a JSON-serialised parameter
    dict (keys sorted for determinism).
    """
    img_hash = hash_image(image)
    param_str = json.dumps(params, sort_keys=True)
    param_hash = hashlib.md5(param_str.encode()).hexdigest()
    return f"pipe_{img_hash}_{param_hash}"


