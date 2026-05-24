"""
JIT-compiled edge blending functions using Numba for maximum performance.

All functions accept and return float32 arrays.  uint8 conversion
happens only at I/O boundaries (image_io.py).

If the Rust extension (seams_core) is available, it is used automatically
with zero warmup cost and Rayon-based parallelism.
"""
from __future__ import annotations

import numpy as np
from numba import jit, prange

from .assertions import assert_float32

__all__ = ["blend_seams_fast", "blend_seam_horizontal_jit", "blend_seam_vertical_jit", "calculate_blend_weights"]

try:
    from seams_core import edge_blend_symmetric as _rs_blend
    HAS_RUST = True
except ImportError:
    HAS_RUST = False


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def blend_seam_horizontal_jit(result, image, cx, half_blend, weights):
    h, w = image.shape[:2]

    for i in prange(len(weights)):
        offset = i + 1
        weight = weights[i]
        inv_weight = 1.0 - weight

        left_col = (cx - offset) % w
        right_col = (cx + offset) % w

        for y in range(h):
            if len(image.shape) == 3:
                for c in range(image.shape[2]):
                    result[y, left_col, c] = inv_weight * image[y, left_col, c] + weight * image[y, right_col, c]
            else:
                result[y, left_col] = inv_weight * image[y, left_col] + weight * image[y, right_col]


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def blend_seam_vertical_jit(result, image, cy, half_blend, weights):
    h, w = image.shape[:2]

    for i in prange(len(weights)):
        offset = i + 1
        weight = weights[i]
        inv_weight = 1.0 - weight

        top_row = (cy - offset) % h
        bottom_row = (cy + offset) % h

        for x in range(w):
            if len(image.shape) == 3:
                for c in range(image.shape[2]):
                    result[top_row, x, c] = inv_weight * image[top_row, x, c] + weight * image[bottom_row, x, c]
            else:
                result[top_row, x] = inv_weight * image[top_row, x] + weight * image[bottom_row, x]


@jit(nopython=True, fastmath=True)
def calculate_blend_weights(half_blend, smoothness):
    weights = np.empty(half_blend, dtype=np.float32)

    for i in range(half_blend):
        offset = i + 1
        t = offset / half_blend

        angle = t * np.pi
        weight = 0.25 * (np.cos(angle) + 1.0)

        weights[i] = weight

    return weights


def blend_seams_fast(image: np.ndarray, blend_strength: float = 0.5,
                     smoothness: float = 0.5) -> np.ndarray:
    """Ultra-fast seam blending (Rust if available, else Numba JIT).

    Accepts and returns float32 arrays.

    Args:
        image: Input image (float32, offset so seams are at center).
        blend_strength: Strength of the blend (determines width).
        smoothness: Edge falloff (0.0=sharp, 1.0=soft).

    Returns:
        Blended image (float32).
    """
    assert_float32(image, "blend_seams_fast image")
    h, w = image.shape[:2]

    max_blend_width = min(h, w) // 4
    blend_width = int(max_blend_width * blend_strength)

    if blend_width < 2:
        return image.copy()

    if HAS_RUST:
        return _rs_blend(image, blend_width, True)

    half_blend = blend_width // 2
    if half_blend < 1:
        return image.copy()

    weights = calculate_blend_weights(half_blend, smoothness)
    result = image.copy()

    cx = w // 2
    blend_seam_horizontal_jit(result, image, cx, half_blend, weights)

    cy = h // 2
    blend_seam_vertical_jit(result, result, cy, half_blend, weights)

    return result
