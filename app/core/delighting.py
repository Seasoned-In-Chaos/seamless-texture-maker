"""
Delighting and Flattening logic for textures.
Removes directional lighting gradients and preserves high-frequency details.

All functions accept and return float32 arrays.
"""
from __future__ import annotations

import cv2
import numpy as np

from .assertions import assert_float32


def delight_image(image: np.ndarray, strength: float = 0.5,
                  flatness: float = 0.0) -> np.ndarray:
    """Remove lighting gradients and shadows from the image.

    Args:
        image: Input float32 array (BGR), values in [0, 255].
        strength: Strength of delighting (0.0 to 1.0).
        flatness: Amount of color flattening (0.0 to 1.0).

    Returns:
        Delighted float32 BGR image.
    """
    assert_float32(image, "delight_image input")

    if strength < 0.01 and flatness < 0.01:
        return image.copy()

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)

    if strength > 0:
        h, w = l.shape
        sigma = max(h, w) * 0.1
        sigma = np.clip(sigma, 10, 500)

        k_size = int(sigma * 3)
        if k_size % 2 == 0:
            k_size += 1
        low_freq = cv2.GaussianBlur(l, (k_size, k_size), sigma)

        mean_l = np.mean(l)

        low_freq = np.clip(low_freq, 1, 255)

        delighted_l = (l / low_freq) * mean_l

        l = l * (1.0 - strength) + delighted_l * strength

        l = np.clip(l, 0, 255)

    if flatness > 0:
        mean_a = np.mean(a)
        mean_b = np.mean(b)

        a = a * (1.0 - flatness) + mean_a * flatness
        b = b * (1.0 - flatness) + mean_b * flatness

    lab_processed = cv2.merge([l.astype(np.float32), a.astype(np.float32), b.astype(np.float32)])
    result = cv2.cvtColor(lab_processed.astype(np.uint8), cv2.COLOR_LAB2BGR)

    return result.astype(np.float32)
