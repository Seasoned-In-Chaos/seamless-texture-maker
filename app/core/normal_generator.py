"""
Material map generation controls - PBR Pipeline

All internal computation is float32.  Final output maps are
converted to uint8 BGR only at the return boundary for
QImage compatibility.  Independent maps (height, roughness,
metallic) are generated in parallel via ThreadPoolExecutor.
"""
from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

import cv2
import numpy as np
from numba import jit

from .ao_generator import generate_ao_map
from .cache import make_pbr_key, ResultCache
from .assertions import assert_float32

_pbr_cache = ResultCache(max_size=30)

HAS_RUST_GRADIENTS = False  # Deprecated in favor of OpenCV Scharr Pipeline
logger = logging.getLogger("seams.pbr")


def _gray_to_bgr_u8(channel: np.ndarray) -> np.ndarray:
    """Convert a float32 [0,1] single-channel map to uint8 BGR (H,W,3)."""
    return cv2.cvtColor(
        np.clip(channel * 255.0, 0, 255).astype(np.uint8),
        cv2.COLOR_GRAY2BGR,
    )


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


def _compute_height(gray: np.ndarray, params: dict) -> np.ndarray:
    """Compute height map (float32 [0,1])."""
    hi = params.get("height_depth", 0.5)
    hs = params.get("height_smooth", 0.1)
    height_source = 1.0 - gray if params.get("height_invert") else gray
    height_source = _apply_contrast(height_source, params.get("height_contrast", "balanced"))
    if hs > 0:
        height_source = cv2.GaussianBlur(height_source, (0, 0), sigmaX=0.35 + hs * 10.0)
    return np.clip(height_source * (hi * 2.0), 0, 1)


def _compute_roughness(gray: np.ndarray, params: dict) -> np.ndarray:
    """Compute roughness map (float32 [0,1])."""
    ri = params.get("rough_intensity", 0.5)
    rc = params.get("rough_contrast", 0.0)
    rough = gray.copy()
    if params.get("rough_invert"):
        rough = 1.0 - rough
    rough = np.clip(0.5 + (rough - 0.5) * (1.0 + rc * 2.0), 0, 1)
    return np.clip(rough * (ri * 2.0), 0, 1)


def _compute_metallic(gray: np.ndarray, params: dict) -> np.ndarray:
    """Compute metallic map (float32 [0,1])."""
    mi = params.get("metal_intensity", 0.0)
    me = params.get("metal_edge", 0.2)
    metal = np.zeros_like(gray)
    if mi > 0:
        metal = np.clip(gray * (mi * 2.0), 0, 1)
        if me > 0:
            sigma = 0.35 + me * 8.0
            metal = cv2.GaussianBlur(metal, (0, 0), sigmaX=sigma)
            metal = np.clip(metal, 0, 1)
    return metal


def _compute_slopes(h_map: np.ndarray, filter_type: str, wrap: bool) -> tuple[np.ndarray, np.ndarray]:
    border_type = cv2.BORDER_WRAP if wrap else cv2.BORDER_REPLICATE
    padded = cv2.copyMakeBorder(h_map, 1, 1, 1, 1, border_type)
    if filter_type == "4 sample":
        # 4-sample uses a simple cross kernel [-1, 0, 1] instead of full 3x3 grids
        kx = np.array([[-1, 0, 1]], dtype=np.float32) * 0.5
        ky = np.array([[-1], [0], [1]], dtype=np.float32) * 0.5
        gx = cv2.filter2D(padded, cv2.CV_32F, kx)
        gy = cv2.filter2D(padded, cv2.CV_32F, ky)
    elif filter_type == "prewitt":
        kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32) / 6.0
        ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32) / 6.0
        gx = cv2.filter2D(padded, cv2.CV_32F, kx)
        gy = cv2.filter2D(padded, cv2.CV_32F, ky)
    elif filter_type == "central_difference":
        kx = np.array([[-1, 0, 1]], dtype=np.float32) * 0.5
        ky = np.array([[-1], [0], [1]], dtype=np.float32) * 0.5
        gx = cv2.filter2D(padded, cv2.CV_32F, kx)
        gy = cv2.filter2D(padded, cv2.CV_32F, ky)
    elif filter_type == "forward_difference":
        kx = np.array([[0, -1, 1]], dtype=np.float32)
        ky = np.array([[0], [-1], [1]], dtype=np.float32)
        gx = cv2.filter2D(padded, cv2.CV_32F, kx)
        gy = cv2.filter2D(padded, cv2.CV_32F, ky)
    elif filter_type == "backward_difference":
        kx = np.array([[-1, 1, 0]], dtype=np.float32)
        ky = np.array([[-1], [1], [0]], dtype=np.float32)
        gx = cv2.filter2D(padded, cv2.CV_32F, kx)
        gy = cv2.filter2D(padded, cv2.CV_32F, ky)
    elif filter_type == "sobel":
        # Sobel operator is [1, 2, 1], scale=1/8 gives true mathematical derivative
        scale = 1.0 / 8.0
        gx = cv2.Sobel(padded, cv2.CV_32F, 1, 0, scale=scale)
        gy = cv2.Sobel(padded, cv2.CV_32F, 0, 1, scale=scale)
    else:
        # The Scharr operator is [3, 10, 3], scale=1/32 gives true mathematical derivative
        scale = 1.0 / 32.0
        gx = cv2.Scharr(padded, cv2.CV_32F, 1, 0, scale=scale)
        gy = cv2.Scharr(padded, cv2.CV_32F, 0, 1, scale=scale)
    return gx[1:-1, 1:-1], gy[1:-1, 1:-1]


def _compute_normal(h_map: np.ndarray, gray: np.ndarray,
                    params: dict) -> np.ndarray:
    """Compute 16-bit normal map (uint16 BGR)."""
    normal_filter = params.get("normal_filter", "scharr").lower()
    filter_scale = params.get("filter_scale", "multi-scale (recommended)")
    wrap = params.get("normal_wrap", True)
    invert_x = params.get("normal_invert_x", False)
    invert_y = params.get("normal_invert_y", False)
    normalize = params.get("normal_normalize", True)
    if filter_scale == "dudv":
        normalize = False
    min_z = params.get("normal_min_z", 0.0)
    scale = params.get("normal_scale", 0.64) * 8.0  # Multiplier

    # If the user toggled Invert Height from the old UI, we still support it
    normal_height = 1.0 - gray if params.get("normal_invert_height") else gray

    h_map_detail = np.clip(normal_height, 0, 1)

    if params.get("normal_map_type") == "bump":
        return cv2.cvtColor(np.clip(h_map_detail * 65535.0, 0, 65535).astype(np.uint16), cv2.COLOR_GRAY2BGR)

    border_type = cv2.BORDER_WRAP if wrap else cv2.BORDER_REPLICATE

    gx_total = None
    gy_total = None

    if "multi-scale" in filter_scale:
        gxs = []
        gys = []
        for sigma in [0.0, 1.0, 2.0, 4.0]:
            if sigma > 0:
                pad = int(math.ceil(sigma * 3.0)) + 1
                padded = cv2.copyMakeBorder(h_map_detail, pad, pad, pad, pad, border_type)
                blurred = cv2.GaussianBlur(padded, (0, 0), sigmaX=sigma)
                blurred = blurred[pad:-pad, pad:-pad]
            else:
                blurred = h_map_detail
            gx, gy = _compute_slopes(blurred, normal_filter, wrap)
            gxs.append(gx)
            gys.append(gy)
            
        w3, w5, w7, w9 = 0.5, 0.25, 0.15, 0.1
        gx_total = (gxs[0]*w3 + gxs[1]*w5 + gxs[2]*w7 + gxs[3]*w9)
        gy_total = (gys[0]*w3 + gys[1]*w5 + gys[2]*w7 + gys[3]*w9)
    else:
        # Discrete filter scales
        sigma = 0.0
        if filter_scale == "5x5": sigma = 0.5
        elif filter_scale == "7x7": sigma = 1.0
        elif filter_scale == "9x9": sigma = 2.0
        
        if sigma > 0:
            pad = int(math.ceil(sigma * 3.0)) + 1
            padded = cv2.copyMakeBorder(h_map_detail, pad, pad, pad, pad, border_type)
            blurred = cv2.GaussianBlur(padded, (0, 0), sigmaX=sigma)
            blurred = blurred[pad:-pad, pad:-pad]
        else:
            blurred = h_map_detail
            
        if filter_scale == "4 sample":
            gx_total, gy_total = _compute_slopes(blurred, "4 sample", wrap)
        else:
            gx_total, gy_total = _compute_slopes(blurred, normal_filter, wrap)

    # Apply Scale
    gx_total *= scale
    gy_total *= scale

    if invert_x:
        gx_total = -gx_total
    if invert_y:
        gy_total = -gy_total

    # Format fallback
    if params.get("normal_format") == "directx":
        gy_total = -gy_total

    # Convert blended slope to normal vector
    nz = np.ones_like(gx_total)

    if normalize:
        mag = np.sqrt(gx_total**2 + gy_total**2 + 1.0)
        nx = -gx_total / mag
        ny = -gy_total / mag
        nz = 1.0 / mag
    else:
        nx = -gx_total
        ny = -gy_total
        nz = np.ones_like(gx_total)

    # NVIDIA Texture Tools Exact Implementation for Min Z
    if min_z > 0.0:
        # NVTT simply clamps the Z component directly...
        mask = nz < min_z
        nz = np.where(mask, min_z, nz)
        
        if normalize:
            # ...and then blindly re-normalizes the vector, which mathematically means Z 
            # might drop slightly below min_z again, but perfectly replicates the NVTT output.
            mag = np.sqrt(nx**2 + ny**2 + nz**2)
            nx = np.where(mask, nx / mag, nx)
            ny = np.where(mask, ny / mag, ny)
            nz = np.where(mask, nz / mag, nz)

    # Pack to display format (B=Z, G=Y, R=X) mapped to [0..1]
    normal_f = np.stack([
        nz * 0.5 + 0.5,
        ny * 0.5 + 0.5,
        nx * 0.5 + 0.5,
    ], axis=-1)
    
    # Return 16-bit array for high fidelity
    return np.clip(normal_f * 65535.0, 0, 65535).astype(np.uint16)


def _compute_ao(gray: np.ndarray, h_map: np.ndarray,
                params: dict) -> np.ndarray:
    """Compute AO map (float32 [0,1])."""
    ai = params.get("ao_intensity", 0.5)
    aspread = params.get("ao_spread", 0.3)
    ainvert = params.get("ao_invert", False)
    ao_source = gray
    if aspread > 0:
        ao_source = cv2.GaussianBlur(gray, (0, 0), sigmaX=0.75 + aspread * 14.0)
    result = 1.0 - (ao_source * ai)
    if ainvert:
        result = 1.0 - result
    return result


class NormalGenerator:
    @staticmethod
    def process(image, use_cache: bool = True, **params) -> Dict[str, np.ndarray]:
        """Process and return PBR maps with parallel generation.

        Independent maps (height, roughness, metallic) run concurrently
        in a ThreadPoolExecutor.  Normal and AO depend on the height map
        and run in a second parallel batch.

        Args:
            image: Input image (BGR numpy array, float32 or uint8).
            use_cache: If True, check/store results in PBR cache.
            **params: PBR slider values from MaterialControlPanel.

        Returns:
            Dict mapping map name to uint8 BGR numpy array.
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
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if image.max() > 1.0:
            image /= 255.0

        if image.ndim == 3:
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
            elif height_source == "alpha_channel" and image.shape[-1] == 4:
                gray = image[..., 3].copy()
            else: # average_rgb
                gray = np.mean(image[..., :3], axis=-1)
        else:
            if image.max() > 1.0:
                gray = image / 255.0
            else:
                gray = image.copy()
        gray = gray.astype(np.float32)

        # ── Phase 1: parallel independent maps ──────────────────────
        t1 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as executor:
            fut_height = executor.submit(_compute_height, gray, params)
            fut_rough = executor.submit(_compute_roughness, gray, params)
            fut_metal = executor.submit(_compute_metallic, gray, params)

            height_f = fut_height.result()
            rough_f = fut_rough.result()
            metal_f = fut_metal.result()

        phase1_ms = (time.perf_counter() - t1) * 1000.0
        logger.debug("PBR phase 1 (parallel): %.1f ms", phase1_ms)

        # ── Phase 2: normal + AO (depend on height) ─────────────────
        t2 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_normal = executor.submit(_compute_normal, height_f, gray, params)
            fut_ao = executor.submit(_compute_ao, gray, height_f, params)

            normal_img = fut_normal.result()
            ao_f = fut_ao.result()

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

        # Displacement
        displacement_strength = params.get("displacement_strength", 0.2)
        displacement = np.clip(height_f * (0.25 + displacement_strength * 1.75), 0, 1)

        # Emissive
        ei = params.get("glow_intensity", 0.0)
        tint_name = params.get("glow_tint", "white")
        tint_bgr = {
            "white": np.array([1.0, 1.0, 1.0], dtype=np.float32),
            "warm": np.array([0.55, 0.82, 1.0], dtype=np.float32),
            "cool": np.array([1.0, 0.78, 0.48], dtype=np.float32),
            "custom": np.array([0.95, 0.55, 1.0], dtype=np.float32),
        }.get(tint_name, np.array([1.0, 1.0, 1.0], dtype=np.float32))
        emissive = np.clip(gray * ei, 0, 1)
        emissive_img = np.clip(emissive[..., None] * tint_bgr * 255.0, 0, 255).astype(np.uint8)

        # ── Convert to uint8 BGR at boundary ────────────────────────
        result: Dict[str, np.ndarray] = {
            "Normal": normal_img,
            "Roughness": _gray_to_bgr_u8(rough_f),
            "Metallic": _gray_to_bgr_u8(metal_f),
            "AO": _gray_to_bgr_u8(ao_f),
            "Height": _gray_to_bgr_u8(height_f),
            "Displacement": _gray_to_bgr_u8(displacement),
            "Opacity": _gray_to_bgr_u8(opacity),
            "Emissive": emissive_img,
        }

        total_ms = (time.perf_counter() - t_total) * 1000.0
        logger.info("PBR total: %.1f ms (phase1=%.1f, phase2=%.1f)", total_ms, phase1_ms, phase2_ms)

        if use_cache:
            _pbr_cache.set_pbr(cache_key, result)

        return result
