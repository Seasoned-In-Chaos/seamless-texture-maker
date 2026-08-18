"""
Material map generation controls - PBR Pipeline

All internal computation is float32. Final output maps are converted to
OpenCV-compatible BGR arrays at the return boundary. Normal maps retain
16-bit precision; scalar maps use 8-bit precision.
"""
from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

import cv2
import numpy as np

from .ao_generator import generate_ao_map
from .cache import make_pbr_key, ResultCache
from .assertions import assert_float32

_pbr_cache = ResultCache(max_size=30)
_PBR_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="seams-pbr")

HAS_RUST_GRADIENTS = False  # Deprecated in favor of OpenCV Scharr Pipeline
logger = logging.getLogger("seams.pbr")


def _gray_to_bgr_u8(channel: np.ndarray, alpha: np.ndarray | None = None) -> np.ndarray:
    """Convert a float32 [0,1] single-channel map to uint8 BGR (H,W,3) or BGRA (H,W,4)."""
    bgr = cv2.cvtColor(
        np.clip(channel * 255.0, 0, 255).astype(np.uint8),
        cv2.COLOR_GRAY2BGR,
    )
    if alpha is not None:
        a_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        return np.dstack([bgr, a_u8])
    return bgr


# Removed compute_gradients_jit and gradients_to_normals_jit as they are replaced by cv2.Scharr pipeline

def _apply_contrast(src: np.ndarray, mode: str) -> np.ndarray:
    mode = (mode or "balanced").lower()
    out = src
    if mode == "auto":
        lo, hi = np.percentile(out, (2.0, 98.0))
        if hi > lo:
            out = (out - lo) / (hi - lo)
    elif mode == "soft":
        out = 0.5 + (out - 0.5) * 0.65
    elif mode == "sharp":
        out = 0.5 + (out - 0.5) * 1.65
    return np.clip(out, 0.0, 1.0)


def _get_nvtt_sobel_kernel(filter_type: str = "blended_sobel", weights: tuple[float, float, float, float] = (1.0, 0.5, 0.25, 0.125)) -> tuple[np.ndarray, np.ndarray]:
    """
    Build authentic NVIDIA Texture Tools derivative kernels for normal & displacement mapping.
    Matches the exact kernels in NVTT Filter.cpp / NormalMap.cpp.
    """
    ft = (filter_type or "blended_sobel").lower()
    if ft in ("sobel_3x3", "3x3", "sobel 3x3"):
        k = np.array([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ], dtype=np.float32)
    elif ft in ("sobel_5x5", "5x5", "sobel 5x5"):
        k = np.array([
            [-1.0, -2.0, 0.0, 2.0, 1.0],
            [-2.0, -3.0, 0.0, 3.0, 2.0],
            [-3.0, -4.0, 0.0, 4.0, 3.0],
            [-2.0, -3.0, 0.0, 3.0, 2.0],
            [-1.0, -2.0, 0.0, 2.0, 1.0],
        ], dtype=np.float32)
    elif ft in ("sobel_7x7", "7x7", "sobel 7x7"):
        k = np.array([
            [-1.0, -2.0, -3.0, 0.0, 3.0, 2.0, 1.0],
            [-2.0, -3.0, -4.0, 0.0, 4.0, 3.0, 2.0],
            [-3.0, -4.0, -5.0, 0.0, 5.0, 4.0, 3.0],
            [-4.0, -5.0, -6.0, 0.0, 6.0, 5.0, 4.0],
            [-3.0, -4.0, -5.0, 0.0, 5.0, 4.0, 3.0],
            [-2.0, -3.0, -4.0, 0.0, 4.0, 3.0, 2.0],
            [-1.0, -2.0, -3.0, 0.0, 3.0, 2.0, 1.0],
        ], dtype=np.float32)
    elif ft in ("sobel_9x9", "9x9", "sobel 9x9"):
        k = np.array([
            [-1.0, -2.0, -3.0, -4.0, 0.0, 4.0, 3.0, 2.0, 1.0],
            [-2.0, -3.0, -4.0, -5.0, 0.0, 5.0, 4.0, 3.0, 2.0],
            [-3.0, -4.0, -5.0, -6.0, 0.0, 6.0, 5.0, 4.0, 3.0],
            [-4.0, -5.0, -6.0, -7.0, 0.0, 7.0, 6.0, 5.0, 4.0],
            [-5.0, -6.0, -7.0, -8.0, 0.0, 8.0, 7.0, 6.0, 5.0],
            [-4.0, -5.0, -6.0, -7.0, 0.0, 7.0, 6.0, 5.0, 4.0],
            [-3.0, -4.0, -5.0, -6.0, 0.0, 6.0, 5.0, 4.0, 3.0],
            [-2.0, -3.0, -4.0, -5.0, 0.0, 5.0, 4.0, 3.0, 2.0],
            [-1.0, -2.0, -3.0, -4.0, 0.0, 4.0, 3.0, 2.0, 1.0],
        ], dtype=np.float32)
    elif ft in ("blended_sobel", "blended", "nvtt", "blended sobel"):
        w3, w5, w7, w9 = weights
        k = np.zeros((9, 9), dtype=np.float32)
        k9 = np.array([
            [-1.0, -2.0, -3.0, -4.0, 0.0, 4.0, 3.0, 2.0, 1.0],
            [-2.0, -3.0, -4.0, -5.0, 0.0, 5.0, 4.0, 3.0, 2.0],
            [-3.0, -4.0, -5.0, -6.0, 0.0, 6.0, 5.0, 4.0, 3.0],
            [-4.0, -5.0, -6.0, -7.0, 0.0, 7.0, 6.0, 5.0, 4.0],
            [-5.0, -6.0, -7.0, -8.0, 0.0, 8.0, 7.0, 6.0, 5.0],
            [-4.0, -5.0, -6.0, -7.0, 0.0, 7.0, 6.0, 5.0, 4.0],
            [-3.0, -4.0, -5.0, -6.0, 0.0, 6.0, 5.0, 4.0, 3.0],
            [-2.0, -3.0, -4.0, -5.0, 0.0, 5.0, 4.0, 3.0, 2.0],
            [-1.0, -2.0, -3.0, -4.0, 0.0, 4.0, 3.0, 2.0, 1.0],
        ], dtype=np.float32) * w9
        k7 = np.array([
            [-1.0, -2.0, -3.0, 0.0, 3.0, 2.0, 1.0],
            [-2.0, -3.0, -4.0, 0.0, 4.0, 3.0, 2.0],
            [-3.0, -4.0, -5.0, 0.0, 5.0, 4.0, 3.0],
            [-4.0, -5.0, -6.0, 0.0, 6.0, 5.0, 4.0],
            [-3.0, -4.0, -5.0, 0.0, 5.0, 4.0, 3.0],
            [-2.0, -3.0, -4.0, 0.0, 4.0, 3.0, 2.0],
            [-1.0, -2.0, -3.0, 0.0, 3.0, 2.0, 1.0],
        ], dtype=np.float32) * w7
        k5 = np.array([
            [-1.0, -2.0, 0.0, 2.0, 1.0],
            [-2.0, -3.0, 0.0, 3.0, 2.0],
            [-3.0, -4.0, 0.0, 4.0, 3.0],
            [-2.0, -3.0, 0.0, 3.0, 2.0],
            [-1.0, -2.0, 0.0, 2.0, 1.0],
        ], dtype=np.float32) * w5
        k3 = np.array([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ], dtype=np.float32) * w3
        k += k9
        k[1:8, 1:8] += k7
        k[2:7, 2:7] += k5
        k[3:6, 3:6] += k3
    else:  # 4-sample centered diff fallback
        k = np.array([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ], dtype=np.float32)

    # NVTT Normalization: sum of absolute values == 1.0
    total = float(np.sum(np.abs(k)))
    if total > 0:
        k /= total
    return k, k.T


def _compute_nvtt_slopes(h_map: np.ndarray, filter_type: str = "blended_sobel", wrap: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Compute surface slopes (du, dv) using NVIDIA Texture Tools exact convolution filters with seamless tileable wrap."""
    kdu, kdv = _get_nvtt_sobel_kernel(filter_type)
    pad = kdu.shape[0] // 2

    if wrap:
        padded = cv2.copyMakeBorder(h_map, pad, pad, pad, pad, cv2.BORDER_WRAP)
    else:
        padded = cv2.copyMakeBorder(h_map, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    du = cv2.filter2D(padded, -1, kdu)[pad:-pad, pad:-pad]
    dv = cv2.filter2D(padded, -1, kdv)[pad:-pad, pad:-pad]

    # Scale slopes to match standard unit range
    return du * 8.0, dv * 8.0


def _compute_four_sample_slopes(h_map: np.ndarray, wrap: bool) -> tuple[np.ndarray, np.ndarray]:
    """Compute NVTT-style centered four-sample derivatives."""
    if wrap:
        left = np.roll(h_map, 1, axis=1)
        right = np.roll(h_map, -1, axis=1)
        up = np.roll(h_map, 1, axis=0)
        down = np.roll(h_map, -1, axis=0)
    else:
        padded = cv2.copyMakeBorder(h_map, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        left = padded[1:-1, :-2]
        right = padded[1:-1, 2:]
        up = padded[:-2, 1:-1]
        down = padded[2:, 1:-1]
    return (right - left) * 0.5, (down - up) * 0.5


def _compute_displacement(gray: np.ndarray, params: dict) -> np.ndarray:
    """Compute displacement map (float32 [0,1]) using NVTT multi-scale gradient analysis."""
    hi = params.get("height_depth", 0.5)
    hs = params.get("height_smooth", 0.1)

    # Base source
    height_source = 1.0 - gray if params.get("height_invert") else gray
    height_source = _apply_contrast(height_source, params.get("height_contrast", "balanced"))

    # NVTT multi-scale gradient decomposition:
    # Scale 1: Fine detail (Sobel 3x3)
    du3, dv3 = _compute_nvtt_slopes(height_source, "sobel_3x3", wrap=True)
    mag3 = np.sqrt(du3**2 + dv3**2)

    # Scale 2: Medium detail (Sobel 5x5)
    du5, dv5 = _compute_nvtt_slopes(height_source, "sobel_5x5", wrap=True)
    mag5 = np.sqrt(du5**2 + dv5**2)

    # Scale 3: Large detail (Sobel 7x7)
    du7, dv7 = _compute_nvtt_slopes(height_source, "sobel_7x7", wrap=True)
    mag7 = np.sqrt(du7**2 + dv7**2)

    # Blend gradient magnitudes matching NVTT multi-frequency structure
    detail = mag3 * 0.5 + mag5 * 0.35 + mag7 * 0.15
    detail = np.clip(detail * 1.5, 0.0, 1.0)

    # Combine structural elevation with surface gradient details
    displacement = height_source * 0.7 + detail * 0.3

    if hs > 0:
        displacement = cv2.GaussianBlur(displacement, (0, 0), sigmaX=0.35 + hs * 10.0)

    return np.clip(displacement * (hi * 2.0), 0.0, 1.0)


def _compute_roughness(gray: np.ndarray, params: dict) -> np.ndarray:
    """Compute roughness map (float32 [0,1])."""
    ri = params.get("rough_intensity", 0.5)
    rc = params.get("rough_contrast", 0.0)
    rough = gray.copy()
    if params.get("rough_invert"):
        rough = 1.0 - rough
    rough = np.clip(0.5 + (rough - 0.5) * (1.0 + rc * 2.0), 0, 1)
    return np.clip(rough * (ri * 2.0), 0, 1)


def _compute_normal(h_map: np.ndarray, gray: np.ndarray,
                    params: dict, alpha: np.ndarray | None = None) -> np.ndarray:
    """Compute a tileable, slope-space 16-bit tangent-space normal map.

    Normal generation uses the same important property as production texture
    tools: derivatives are converted to slopes first, then the resulting
    vector is normalized.  The previous implementation multiplied a large
    Sobel response by a second fixed Z scale, which saturated the X/Y channels
    and produced the blurred, neon-looking result seen in the UI.
    """
    invert_x = params.get("normal_invert_x", False)
    invert_y = params.get("normal_invert_y", False)
    min_z = params.get("normal_min_z", 0.0)
    scale = float(np.clip(params.get("normal_scale", 4.266), 0.1, 8.0))
    filter_type = str(params.get("normal_filter", "4_sample")).lower()
    wrap = bool(params.get("normal_wrap", True))

    # Invert height if requested
    normal_height = 1.0 - h_map if params.get("normal_invert_height") else h_map
    h_map_detail = np.clip(normal_height, 0.0, 1.0)

    # Match the filter choices exposed by NVIDIA Texture Tools.  dUdV uses
    # the same four-sample derivative field, retained as its own mode for
    # workflows that use derivative-map terminology.
    if filter_type in {"4_sample", "4 sample", "dudv", "du/dv"}:
        gx_total, gy_total = _compute_four_sample_slopes(h_map_detail, wrap)
    else:
        gx_total, gy_total = _compute_nvtt_slopes(h_map_detail, filter_type, wrap)
    gx_total *= scale
    gy_total *= scale

    if invert_x:
        gx_total = -gx_total
    if invert_y:
        gy_total = -gy_total

    # Format fallback (DirectX inverts Y)
    if params.get("normal_format") == "directx":
        gy_total = -gy_total

    # Tangent-space normal from the slopes.  A quarter-height Z basis keeps
    # real texture detail visible at normal-map resolutions while avoiding
    # the old path's X/Y saturation.  A flat field is still exactly (0, 0, 1).
    nz_basis = 0.25
    mag = np.sqrt(gx_total * gx_total + gy_total * gy_total + nz_basis * nz_basis)
    nx = -gx_total / mag
    ny = -gy_total / mag
    nz = nz_basis / mag

    # Enforce true minimum Z while maintaining unit length
    if min_z > 0.0:
        min_z = float(np.clip(min_z, 0.0, 1.0))
        mask = nz < min_z
        xy_length = np.sqrt(nx * nx + ny * ny)
        target_xy = math.sqrt(max(0.0, 1.0 - min_z * min_z))
        xy_scale = target_xy / np.maximum(xy_length, 1e-8)
        nx = np.where(mask, nx * xy_scale, nx)
        ny = np.where(mask, ny * xy_scale, ny)
        nz = np.where(mask, min_z, nz)

    # Pack to display format (B=Z, G=Y, R=X) mapped to [0..1]
    normal_f = np.stack([
        nz * 0.5 + 0.5,
        ny * 0.5 + 0.5,
        nx * 0.5 + 0.5,
    ], axis=-1)
    
    # Return 16-bit array (BGR or BGRA) for high fidelity
    normal_u16 = np.clip(normal_f * 65535.0, 0, 65535).astype(np.uint16)
    if alpha is not None:
        a_u16 = np.clip(alpha * 65535.0, 0, 65535).astype(np.uint16)
        return np.dstack([normal_u16, a_u16])
    return normal_u16


def _compute_ao(h_map: np.ndarray, normal: np.ndarray,
                params: dict) -> np.ndarray:
    """Compute AO map (float32 [0,1])."""
    ai = params.get("ao_intensity", 0.5)
    aspread = params.get("ao_spread", 0.3)
    ainvert = params.get("ao_invert", False)
    # Use the processed height field and multi-scale crevice analysis.
    radius = int(round(4.0 + aspread * 28.0))
    result = generate_ao_map(
        h_map,
        normal_map=normal,
        radius=radius,
        strength=ai,
        contrast=1.0,
        output_float=True,
    )
    if ainvert:
        result = 1.0 - result
    return np.clip(result, 0.0, 1.0).astype(np.float32, copy=False)


class NormalGenerator:
    @staticmethod
    def process(image, use_cache: bool = True, **params) -> Dict[str, np.ndarray]:
        """Process and return PBR maps with parallel generation.

        Independent maps (height and roughness) run concurrently
        in a ThreadPoolExecutor.  Normal and AO depend on the height map
        and run in a second parallel batch.

        Args:
            image: Input image (BGR or BGRA numpy array, float32 or uint8).
            use_cache: If True, check/store results in PBR cache.
            **params: PBR slider values from MaterialControlPanel.

        Returns:
            Dict mapping map name to BGR/BGRA numpy arrays. Normal is uint16;
            scalar maps are uint8.
        """
        t_total = time.perf_counter()

        # Cache lookup
        if use_cache:
            cache_key = make_pbr_key(image, params)
            cached = _pbr_cache.get_pbr(cache_key)
            if cached is not None:
                logger.debug("PBR cache HIT")
                return cached

        # Parse float32 grayscale [0,1] based on Height Source
        if image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0
        else:
            if image.dtype != np.float32:
                image = image.astype(np.float32)
            if image.size and float(np.nanmax(image)) > 1.0:
                image /= 255.0
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        image = np.clip(image, 0.0, 1.0)

        alpha_channel = None
        if image.ndim == 3:
            if image.shape[-1] == 4:
                alpha_channel = image[..., 3].copy()
            height_source = params.get("height_source", "average_rgb")
            if height_source == "red":
                gray = image[..., 2].copy()
            elif height_source == "green":
                gray = image[..., 1].copy()
            elif height_source == "blue":
                gray = image[..., 0].copy()
            elif height_source == "luminance":
                gray = (image[..., 0] * 0.0722 + image[..., 1] * 0.7152 + image[..., 2] * 0.2126)
            elif height_source == "max_rgb":
                gray = np.max(image[..., :3], axis=-1)
            elif height_source == "alpha_channel" and alpha_channel is not None:
                gray = alpha_channel.copy()
            else: # average_rgb
                gray = np.mean(image[..., :3], axis=-1)
        else:
            gray = image.copy()
        gray = gray.astype(np.float32)

        # ── Phase 1: parallel independent maps ──────────────────────
        t1 = time.perf_counter()
        fut_disp = _PBR_EXECUTOR.submit(_compute_displacement, gray, params)
        fut_rough = _PBR_EXECUTOR.submit(_compute_roughness, gray, params)

        disp_f = fut_disp.result()
        rough_f = fut_rough.result()

        phase1_ms = (time.perf_counter() - t1) * 1000.0
        logger.debug("PBR phase 1 (parallel): %.1f ms", phase1_ms)

        # ── Phase 2: normal + AO (depend on height) ─────────────────
        t2 = time.perf_counter()
        # Normals should describe the source height field directly.  Using
        # the separately stylized displacement map here compounds blur and
        # gradient detail, which is the main cause of soft, oversaturated
        # normals.  Displacement remains available as its own output map.
        normal_img = _compute_normal(gray, gray, params, alpha=alpha_channel)
        ao_f = _compute_ao(disp_f, normal_img, params)

        phase2_ms = (time.perf_counter() - t2) * 1000.0
        logger.debug("PBR phase 2 (normal+AO): %.1f ms", phase2_ms)

        # ── Remaining maps (sequential — cheap) ─────────────────────
        # Opacity
        ot = params.get("alpha_threshold", 1.0)
        aso = params.get("alpha_softness", 0.0)
        threshold = 1.0 - ot
        if aso > 0:
            width = max(0.01, aso * 0.45)
            opacity = np.clip((gray - threshold + width * 0.5) / width, 0.0, 1.0)
        else:
            opacity = np.where(gray > threshold, 1.0, 0.0)

        # ── Convert to uint8 BGR(A) at boundary ────────────────────────
        result: Dict[str, np.ndarray] = {
            "Normal": normal_img,
            "Roughness": _gray_to_bgr_u8(rough_f, alpha=alpha_channel),
            "AO": _gray_to_bgr_u8(ao_f, alpha=alpha_channel),
            "Displacement": _gray_to_bgr_u8(disp_f, alpha=alpha_channel),
            "Opacity": _gray_to_bgr_u8(opacity, alpha=alpha_channel),
        }

        total_ms = (time.perf_counter() - t_total) * 1000.0
        logger.info("PBR total: %.1f ms (phase1=%.1f, phase2=%.1f)", total_ms, phase1_ms, phase2_ms)

        if use_cache:
            _pbr_cache.set_pbr(cache_key, result)

        return result
