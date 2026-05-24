"""
Smart inpainting for seam removal in textures.
Uses OpenCV's inpainting algorithms to seamlessly fill seam areas.

All functions accept and return float32 arrays.  uint8 conversion
happens only at I/O boundaries (image_io.py).
"""
from __future__ import annotations

import numpy as np
import cv2

from .assertions import assert_float32
from .gpu_utils import GPUAccelerator

_gpu_accel = GPUAccelerator()


def create_seam_detection_mask(image: np.ndarray, threshold: float = 30.0,
                               seam_width: int = 20) -> np.ndarray:
    """Detect seams based on gradient discontinuities.

    Returns:
        Binary uint8 mask (0 or 255).
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    seam_mask = (magnitude > threshold).astype(np.uint8) * 255

    kernel = np.ones((seam_width, seam_width), np.uint8)
    seam_mask = cv2.dilate(seam_mask, kernel, iterations=1)

    return seam_mask


def inpaint_seams(image: np.ndarray, mask: np.ndarray,
                  method: str = 'telea', radius: int = 5) -> np.ndarray:
    """Inpaint seam regions using GPU-accelerated inpainting with CPU fallback.

    Accepts float32 image; converts to uint8 internally for
    cv2.inpaint (which requires uint8 input), then converts back.

    Returns:
        float32 inpainted image.
    """
    return _gpu_accel.inpaint_gpu(image, mask, radius, method)


def smart_seam_inpaint(image: np.ndarray, seam_width: int = 30,
                       detail_preservation: float = 0.5,
                       method: str = 'telea') -> np.ndarray:
    """Smart inpainting that preserves texture details while removing seams.

    Accepts and returns float32 arrays.

    Args:
        image: Input image (float32, offset so seams are at center).
        seam_width: Width of the seam region to inpaint.
        detail_preservation: Unused (kept for API compatibility).
        method: Inpainting method ('ns' or 'telea').

    Returns:
        float32 image with inpainted seams.
    """
    assert_float32(image, "smart_seam_inpaint image")
    h, w = image.shape[:2]

    # Memory limit check
    estimated_bytes = h * w * (image.shape[2] if image.ndim == 3 else 1) * 4 * 3
    if estimated_bytes > 2 * 1024**3:
        import logging
        logging.getLogger("seams.inpaint").warning(
            "Large image: %dx%d, ~%.1fGB needed for inpainting",
            w, h, estimated_bytes / (1024**3),
        )

    mask = np.zeros((h, w), dtype=np.uint8)

    adjusted_width = max(2, seam_width)

    cx = w // 2
    x1 = max(0, cx - adjusted_width // 2)
    x2 = min(w, cx + adjusted_width // 2)
    mask[:, x1:x2] = 255

    cy = h // 2
    y1 = max(0, cy - adjusted_width // 2)
    y2 = min(h, cy + adjusted_width // 2)
    mask[y1:y2, :] = 255

    max_dim = max(h, w)

    if max_dim > 2048:
        scale_factor = 2048.0 / max_dim
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)

        s_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        s_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        s_radius = max(3, int((adjusted_width // 2) * scale_factor))

        s_result = inpaint_seams(s_img, s_mask, 'ns', s_radius)

        result_full = cv2.resize(s_result, (w, h), interpolation=cv2.INTER_LINEAR)

        blend_mask = mask.astype(np.float32) / 255.0
        blend_mask = cv2.GaussianBlur(blend_mask, (5, 5), 0)

        if image.ndim == 3:
            blend_mask = blend_mask[:, :, np.newaxis]

        result = image * (1.0 - blend_mask) + result_full * blend_mask

    else:
        radius = max(3, adjusted_width // 2)
        result = inpaint_seams(image, mask, 'ns', radius)

    return result.astype(np.float32)


def multi_scale_inpaint(image: np.ndarray, mask: np.ndarray,
                        scales: list = None) -> np.ndarray:
    """Multi-scale inpainting for better texture preservation.

    Accepts and returns float32 arrays.
    """
    if scales is None:
        scales = [1.0, 0.5, 0.25]
    assert_float32(image, "multi_scale_inpaint image")
    h, w = image.shape[:2]
    result = image.copy()

    for scale in scales:
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            scaled_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            scaled_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            scaled_img = image
            scaled_mask = mask

        radius = max(3, int(5 * scale))
        inpainted = inpaint_seams(scaled_img, scaled_mask, 'telea', radius)

        if scale < 1.0:
            inpainted = cv2.resize(inpainted, (w, h), interpolation=cv2.INTER_CUBIC)

        weight = scale
        result = cv2.addWeighted(result, 1 - weight * 0.3, inpainted, weight * 0.3, 0)

    return result
