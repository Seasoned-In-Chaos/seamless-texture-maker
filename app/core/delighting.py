"""
Delighting and Flattening logic for textures.
Removes directional lighting gradients and preserves high-frequency details.

All functions accept and return float32 arrays with values in [0, 255].
"""
from __future__ import annotations

import cv2
import numpy as np

from .assertions import assert_float32


def delight_image(image: np.ndarray, strength: float = 0.5,
                  flatness: float = 0.0,
                  shadow_removal: float = 0.0,
                  highlight_reduction: float = 0.0,
                  contrast_recovery: float = 0.0,
                  detail_preservation: float = 0.0,
                  color_preservation: float = 0.0,
                  ao_removal: float = 0.0,
                  edge_consistency: float = 0.0,
                  **_kwargs) -> np.ndarray:
    """Remove lighting gradients and shadows from the image.

    Args:
        image: Input float32 array (BGR), values in [0, 255].
        strength: Overall delighting strength (0.0 to 1.0).
        flatness: Amount of color flattening (0.0 to 1.0).
        shadow_removal: Lift dark shadow regions (0.0 to 1.0).
        highlight_reduction: Compress bright highlights (0.0 to 1.0).
        contrast_recovery: Restore local contrast after delighting (0.0 to 1.0).
        detail_preservation: Preserve high-frequency surface detail (0.0 to 1.0).
        color_preservation: Keep original color saturation (0.0 to 1.0).
        ao_removal: Remove ambient occlusion darkening (0.0 to 1.0).
        edge_consistency: Smooth tonal transitions at edges (0.0 to 1.0).

    Returns:
        Delighted float32 BGR image, values in [0, 255].
    """
    assert_float32(image, "delight_image input")

    # Early out if nothing to do. Every slider must be checked here — a
    # slider whose effect block lives further down (contrast_recovery,
    # detail_preservation, color_preservation, edge_consistency) still needs
    # to reach that code, otherwise it silently does nothing when adjusted
    # on its own.
    has_work = (strength > 0.01 or flatness > 0.01 or shadow_removal > 0.01
                or highlight_reduction > 0.01 or ao_removal > 0.01
                or contrast_recovery > 0.01 or detail_preservation > 0.01
                or color_preservation > 0.01 or edge_consistency > 0.01)
    if not has_work:
        return image.copy()

    h, w = image.shape[:2]

    # ── Work in uint8 LAB for correct OpenCV color space conversion ──
    image_u8 = np.clip(image, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(image_u8, cv2.COLOR_BGR2LAB)
    l_orig = lab[:, :, 0].astype(np.float32)
    a_orig = lab[:, :, 1].astype(np.float32)
    b_orig = lab[:, :, 2].astype(np.float32)

    l = l_orig.copy()

    # ── 1. Multi-scale light removal ─────────────────────────────────
    # Use frequency separation: extract low-frequency lighting component
    # at multiple scales and remove it proportionally.
    if strength > 0.01:
        # Compute the effective delighting strength
        delight_str = strength

        # Large-scale: captures broad directional lighting / gradients
        sigma_large = max(h, w) * 0.15
        sigma_large = np.clip(sigma_large, 15, 600)
        k_large = int(sigma_large * 3) | 1  # ensure odd
        low_freq_large = cv2.GaussianBlur(l, (k_large, k_large), sigma_large)

        # Medium-scale: captures softer shadows and light falloff
        sigma_med = max(h, w) * 0.05
        sigma_med = np.clip(sigma_med, 5, 200)
        k_med = int(sigma_med * 3) | 1
        low_freq_med = cv2.GaussianBlur(l, (k_med, k_med), sigma_med)

        # Blend the two scales
        low_freq = low_freq_large * 0.6 + low_freq_med * 0.4
        low_freq = np.clip(low_freq, 1.0, 255.0)

        # Compute the mean luminance as target
        mean_l = np.mean(l)
        mean_l = max(mean_l, 1.0)

        # Divide out the low-frequency lighting and re-center at mean
        delighted_l = (l / low_freq) * mean_l

        # Blend original ↔ delighted
        l = l * (1.0 - delight_str) + delighted_l * delight_str
        l = np.clip(l, 0, 255)

    # ── 1b. Detail preservation ──────────────────────────────────────
    # Independent of step 1: unlike the multi-scale removal above, this
    # boosts local contrast on whatever "l" currently is, so it has a
    # visible effect on its own instead of silently doing nothing unless
    # Shadow/Highlight/AO Removal also happen to be raised.
    if detail_preservation > 0.01:
        sigma_detail = max(h, w) * 0.02
        sigma_detail = np.clip(sigma_detail, 2, 50)
        k_detail = int(sigma_detail * 3) | 1
        smoothed = cv2.GaussianBlur(l, (k_detail, k_detail), sigma_detail)
        detail_layer = l - smoothed
        l = l + detail_layer * (detail_preservation * 0.6)
        l = np.clip(l, 0, 255)

    # ── 2. Shadow removal ────────────────────────────────────────────
    # Lift shadows by compressing the dark end of the luminance range
    if shadow_removal > 0.01:
        shadow_lift = shadow_removal * 40.0  # max lift ~40 levels
        shadow_mask = np.clip(1.0 - l / 80.0, 0, 1)  # mask dark areas
        l = l + shadow_mask * shadow_lift
        l = np.clip(l, 0, 255)

    # ── 3. Highlight reduction ───────────────────────────────────────
    # Compress highlights by pulling bright values toward mean
    if highlight_reduction > 0.01:
        mean_l = np.mean(l)
        highlight_mask = np.clip((l - 180.0) / 75.0, 0, 1)  # mask bright areas
        highlight_pull = highlight_reduction * 0.6
        l = l - highlight_mask * (l - mean_l) * highlight_pull
        l = np.clip(l, 0, 255)

    # ── 4. AO removal ────────────────────────────────────────────────
    # Remove ambient occlusion by detecting and lifting local darkening
    # in corners/crevices (smaller scale than global lighting)
    if ao_removal > 0.01:
        sigma_ao = max(h, w) * 0.03
        sigma_ao = np.clip(sigma_ao, 3, 100)
        k_ao = int(sigma_ao * 3) | 1
        local_mean = cv2.GaussianBlur(l, (k_ao, k_ao), sigma_ao)

        # AO manifests as local darkening relative to neighborhood
        ao_map = np.clip((local_mean - l) / 60.0, 0, 1)
        ao_lift = ao_removal * 35.0
        l = l + ao_map * ao_lift
        l = np.clip(l, 0, 255)

    # ── 5. Contrast recovery ─────────────────────────────────────────
    # After delighting, the image can look flat. Restore local contrast.
    if contrast_recovery > 0.01:
        mean_l = np.mean(l)
        # Increase deviation from mean
        contrast_factor = 1.0 + contrast_recovery * 0.8
        l = mean_l + (l - mean_l) * contrast_factor
        l = np.clip(l, 0, 255)

    # ── 6. Edge consistency ──────────────────────────────────────────
    # Smooth harsh tonal transitions caused by processing
    if edge_consistency > 0.01:
        sigma_edge = 0.5 + edge_consistency * 2.0
        l_smooth = cv2.GaussianBlur(l, (0, 0), sigma_edge)
        blend = edge_consistency * 0.3  # subtle
        l = l * (1.0 - blend) + l_smooth * blend

    # ── 7. Color channel processing ──────────────────────────────────
    a = a_orig.copy()
    b = b_orig.copy()

    # Flatness: push chrominance toward the image's average color,
    # rather than neutral gray, to flatten color variations without desaturating.
    if flatness > 0.01:
        mean_a = np.mean(a)
        mean_b = np.mean(b)
        a = a * (1.0 - flatness) + mean_a * flatness
        b = b * (1.0 - flatness) + mean_b * flatness

    # Color preservation: reduce chrominance changes from delighting
    # by blending processed a/b channels back toward originals
    if color_preservation > 0.01 and flatness < 0.99:
        color_keep = color_preservation
        a = a * (1.0 - color_keep) + a_orig * color_keep
        b = b * (1.0 - color_keep) + b_orig * color_keep

    # ── 8. Reassemble and convert back ───────────────────────────────
    lab_out = cv2.merge([
        np.clip(l, 0, 255).astype(np.uint8),
        np.clip(a, 0, 255).astype(np.uint8),
        np.clip(b, 0, 255).astype(np.uint8),
    ])
    result = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

    return result.astype(np.float32)
