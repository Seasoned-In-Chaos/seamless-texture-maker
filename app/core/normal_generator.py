"""
Material map generation controls - PBR Pipeline

All internal computation is float32.  Final output maps are
converted to uint8 BGR only at the return boundary for
QImage compatibility.  Independent maps (height, roughness,
metallic) are generated in parallel via ThreadPoolExecutor.
"""
from __future__ import annotations

import logging
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

try:
    from seams_core import compute_gradients as _rs_compute_gradients
    HAS_RUST_GRADIENTS = True
except ImportError:
    HAS_RUST_GRADIENTS = False
logger = logging.getLogger("seams.pbr")


def _gray_to_bgr_u8(channel: np.ndarray) -> np.ndarray:
    """Convert a float32 [0,1] single-channel map to uint8 BGR (H,W,3)."""
    return cv2.cvtColor(
        np.clip(channel * 255.0, 0, 255).astype(np.uint8),
        cv2.COLOR_GRAY2BGR,
    )


@jit(nopython=True, fastmath=True)
def compute_gradients_jit(height_map, strength=1.0):
    h, w = height_map.shape
    gradients = np.zeros((h, w, 2), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            prev_x = height_map[y, (x - 1 + w) % w]
            next_x = height_map[y, (x + 1) % w]
            prev_y = height_map[(y - 1 + h) % h, x]
            next_y = height_map[(y + 1) % h, x]
            dx = (next_x - prev_x) * strength
            dy = (next_y - prev_y) * strength
            gradients[y, x, 0] = dx
            gradients[y, x, 1] = dy
    return gradients


def compute_gradients_dispatch(height_map: np.ndarray, strength: float):
    """Dispatch gradient computation to Rust or Numba JIT."""
    if HAS_RUST_GRADIENTS:
        return _rs_compute_gradients(height_map, strength)
    return compute_gradients_jit(height_map, strength)


@jit(nopython=True, fastmath=True)
def gradients_to_normals_jit(gradients, invert_y=False):
    h, w = gradients.shape[:2]
    normals = np.zeros((h, w, 3), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            dx = gradients[y, x, 0]
            dy = gradients[y, x, 1]
            if invert_y:
                dy = -dy
            nx = -dx
            ny = -dy
            nz = 1.0
            mag = np.sqrt(nx * nx + ny * ny + nz * nz)
            normals[y, x, 0] = nx / mag
            normals[y, x, 1] = ny / mag
            normals[y, x, 2] = nz / mag
    return normals


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


def _compute_normal(h_map: np.ndarray, gray: np.ndarray,
                    params: dict) -> np.ndarray:
    """Compute normal map (uint8 BGR)."""
    ni = params.get("normal_intensity", 0.5) * 5.0
    ns = params.get("normal_smooth", 0.3)
    nd = params.get("normal_detail", 0.4)

    normal_height = 1.0 - gray if params.get("normal_invert_height") else gray
    normal_height = _apply_contrast(normal_height, params.get("normal_contrast", "balanced"))

    blurred = cv2.GaussianBlur(normal_height, (0, 0), sigmaX=0.15 + ns * 10.0)
    h_map_detail = gray + (gray - blurred) * nd
    h_map_detail = np.clip(h_map_detail, 0, 1)

    grads = compute_gradients_dispatch(h_map_detail, ni)
    normals_raw = gradients_to_normals_jit(grads, params.get("normal_format") == "directx")

    if params.get("normal_map_type") == "bump":
        return _gray_to_bgr_u8(h_map_detail)

    normal_f = np.stack([
        normals_raw[..., 2] * 0.5 + 0.5,
        normals_raw[..., 1] * 0.5 + 0.5,
        normals_raw[..., 0] * 0.5 + 0.5,
    ], axis=-1)
    return np.clip(normal_f * 255.0, 0, 255).astype(np.uint8)


def _compute_ao(gray: np.ndarray, h_map: np.ndarray,
                params: dict) -> np.ndarray:
    """Compute AO map (float32 [0,1])."""
    ai = params.get("ao_intensity", 0.5)
    aspread = params.get("ao_spread", 0.3)
    ao_source = gray
    if aspread > 0:
        ao_source = cv2.GaussianBlur(gray, (0, 0), sigmaX=0.75 + aspread * 14.0)
    return 1.0 - (ao_source * ai)


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

        # Convert to float32 grayscale [0,1]
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0
        else:
            gray = image / 255.0
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
