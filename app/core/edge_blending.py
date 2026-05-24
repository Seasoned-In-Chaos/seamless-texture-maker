"""
Edge blending algorithms for seamless texture generation.

All functions accept and return float32 arrays.  uint8 conversion
happens only at I/O boundaries (image_io.py).
"""
from __future__ import annotations

import numpy as np
import cv2

from .assertions import assert_float32

__all__ = ["create_gradient_mask", "create_blend_mask", "blend_seams"]


def create_gradient_mask(width: int, height: int,
                         direction: str = 'horizontal',
                         symmetric: bool = True) -> np.ndarray:
    """Create a gradient mask for blending.

    Returns:
        float32 mask, values 0.0–1.0.
    """
    if direction == 'horizontal':
        if symmetric:
            half = width // 2
            left = np.linspace(0, 1, half)
            right = np.linspace(1, 0, width - half)
            gradient = np.concatenate([left, right])
        else:
            gradient = np.linspace(0, 1, width)
        mask = np.tile(gradient, (height, 1))
    else:
        if symmetric:
            half = height // 2
            top = np.linspace(0, 1, half)
            bottom = np.linspace(1, 0, height - half)
            gradient = np.concatenate([top, bottom])
        else:
            gradient = np.linspace(0, 1, height)
        mask = np.tile(gradient.reshape(-1, 1), (1, width))

    return mask.astype(np.float32)


def create_blend_mask(height: int, width: int, blend_width: int,
                      symmetric: bool = True,
                      falloff: float = 0.5) -> np.ndarray:
    """Create a blend mask for the center cross seam.

    Returns:
        float32 mask, values 0.0–1.0.
    """
    mask = np.ones((height, width), dtype=np.float32)

    hardness = 1.0 / max(0.001, falloff)

    cx = width // 2
    half_blend = blend_width // 2
    x1 = max(0, cx - half_blend)
    x2 = min(width, cx + half_blend)

    if symmetric and half_blend > 0:
        x_coords = np.arange(x1, x2)
        dist = np.abs(x_coords - cx) / half_blend
        curve = np.clip(dist * hardness, 0, 1)
        mask[:, x1:x2] *= curve[np.newaxis, :]

    cy = height // 2
    y1 = max(0, cy - half_blend)
    y2 = min(height, cy + half_blend)

    if symmetric and half_blend > 0:
        y_coords = np.arange(y1, y2)
        dist = np.abs(y_coords - cy) / half_blend
        curve = np.clip(dist * hardness, 0, 1)
        mask[y1:y2, :] *= curve[:, np.newaxis]

    return mask


def blend_seams(image: np.ndarray, blend_strength: float = 0.5,
                smoothness: float = 0.5, symmetric: bool = True,
                original_image: np.ndarray | None = None,
                fixed_width: int | None = None) -> np.ndarray:
    """Blend the seams at the center of an offset image.

    Fully vectorized — no Python per-pixel loops.
    Accepts and returns float32 arrays.

    Args:
        image: Input image (float32, offset so seams are at center).
        blend_strength: Strength of the blend (determines width).
        smoothness: Edge falloff (0.0=sharp, 1.0=soft).
        symmetric: Use symmetric blending.
        original_image: The original offset image (required for non-symmetric).
        fixed_width: Override blend width in pixels.

    Returns:
        Blended image (float32).
    """
    assert_float32(image, "blend_seams image")
    h, w = image.shape[:2]

    if fixed_width is not None:
        blend_width = int(fixed_width * 1.5)
    else:
        max_blend_width = min(h, w) // 10
        blend_width = int(max_blend_width * blend_strength)

    if blend_width < 2:
        return image.copy()

    if not symmetric:
        if original_image is None:
            return image.copy()

        assert_float32(original_image, "blend_seams original_image")
        result = image.copy()
        orig = original_image
        img_f = image

        cx = w // 2
        cy = h // 2
        half_blend = blend_width // 2

        if half_blend > 0:
            offsets = np.arange(1, half_blend + 1)
            t = offsets / half_blend
            weights = (np.cos(t * np.pi) + 1.0) * 0.5

            left_cols  = (cx - offsets) % w
            right_cols = (cx + offsets) % w

            if image.ndim == 3:
                w_col = weights[np.newaxis, :, np.newaxis]
            else:
                w_col = weights[np.newaxis, :]

            result[:, left_cols]  = img_f[:, left_cols]  * w_col + orig[:, left_cols]  * (1 - w_col)
            result[:, right_cols] = img_f[:, right_cols] * w_col + orig[:, right_cols] * (1 - w_col)

            top_rows    = (cy - offsets) % h
            bottom_rows = (cy + offsets) % h

            if image.ndim == 3:
                w_row = weights[:, np.newaxis, np.newaxis]
            else:
                w_row = weights[:, np.newaxis]

            result[top_rows, :]    = img_f[top_rows, :]    * w_row + orig[top_rows, :]    * (1 - w_row)
            result[bottom_rows, :] = img_f[bottom_rows, :] * w_row + orig[bottom_rows, :] * (1 - w_row)

        return result

    # ── Symmetric (Mirror) Blending ──────────────────────────────
    result = image.copy()
    img_f = image

    cx = w // 2
    cy = h // 2
    half_blend = blend_width // 2

    if half_blend > 0:
        offsets = np.arange(1, half_blend + 1)
        t = offsets / half_blend
        weights = 0.25 * (np.cos(t * np.pi) + 1.0)

        left_cols  = (cx - offsets) % w
        right_cols = (cx + offsets) % w

        if image.ndim == 3:
            w_col = weights[np.newaxis, :, np.newaxis]
        else:
            w_col = weights[np.newaxis, :]

        result[:, left_cols] = (1 - w_col) * img_f[:, left_cols] + w_col * img_f[:, right_cols]

        top_rows    = (cy - offsets) % h
        bottom_rows = (cy + offsets) % h

        if image.ndim == 3:
            w_row = weights[:, np.newaxis, np.newaxis]
        else:
            w_row = weights[:, np.newaxis]

        result[top_rows, :] = (1 - w_row) * img_f[top_rows, :] + w_row * img_f[bottom_rows, :]

    return result
