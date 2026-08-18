"""
Pre-compile all Numba JIT functions during splash screen display.

Eliminates the 500ms-2s cold-start stall that would otherwise hit
on the first user action by triggering compilation in a background
thread while the splash animation plays.
"""
from __future__ import annotations

import logging
import time
from typing import Dict

import numpy as np

__all__ = ["warmup_all_jit_functions"]

logger = logging.getLogger("seams.warmup")


def warmup_all_jit_functions() -> Dict[str, float]:
    """Call each JIT-compiled function once with a tiny array to trigger
    Numba compilation.  Returns a dict mapping function name to elapsed ms.

    The 64x64 zero arrays are cheap to allocate and guarantee that every
    code-path (color / grayscale, horizontal / vertical) is exercised.
    """
    timings: Dict[str, float] = {}

    tiny_3ch = np.zeros((64, 64, 3), dtype=np.float32)
    tiny_2ch = np.zeros((64, 64), dtype=np.float32)
    tiny_weights = np.zeros(8, dtype=np.float32)

    # --- edge_blending_jit ---
    try:
        from .edge_blending_jit import (
            blend_seam_horizontal_jit,
            blend_seam_vertical_jit,
            calculate_blend_weights,
        )

        t0 = time.perf_counter()
        calculate_blend_weights(8, 0.5)
        result_h = tiny_3ch.copy()
        blend_seam_horizontal_jit(result_h, tiny_3ch, 32, 4, tiny_weights)
        result_v = tiny_3ch.copy()
        blend_seam_vertical_jit(result_v, tiny_3ch, 32, 4, tiny_weights)
        elapsed = (time.perf_counter() - t0) * 1000.0
        timings["edge_blending_jit"] = elapsed
        logger.info("warmup edge_blending_jit: %.1f ms", elapsed)
    except Exception as exc:
        logger.warning("warmup edge_blending_jit failed: %s", exc)

    # --- normal_generator gradients (DEPRECATED: Now uses pure OpenCV Scharr) ---

    # --- materialize_methods_jit (splat) ---
    try:
        from .materialize_methods_jit import (
            splat_accumulate_jit, splat_resolve_jit,
        )

        t0 = time.perf_counter()
        patches = np.stack([tiny_3ch.copy()])
        masks = np.ones((1, 64, 64), dtype=np.float32)
        coords = np.array([[0, 0]], dtype=np.int32)
        indices = np.array([0], dtype=np.int32)

        accum = np.zeros((64, 64, 3), dtype=np.float32)
        weight = np.zeros((64, 64), dtype=np.float32)
        splat_accumulate_jit(accum, weight, patches, masks, coords, indices)
        splat_resolve_jit(accum, weight, tiny_3ch, np.empty_like(accum))
        elapsed = (time.perf_counter() - t0) * 1000.0
        timings["splat_jit"] = elapsed
        logger.info("warmup splat_jit: %.1f ms", elapsed)
    except Exception as exc:
        logger.warning("warmup splat_jit failed: %s", exc)

    # --- Mirror Tiling (2×2) and Offset + Cross-Fade warmup ---
    # These methods are pure-NumPy: no JIT compilation required.  Exercising
    # them here ensures their import paths are resolved and NumPy internal
    # buffers are warm before the first user action.
    try:
        from .seamless import SeamlessProcessor

        tiny_img = np.zeros((64, 64, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        proc = SeamlessProcessor()
        proc.load_image(tiny_img)
        proc.set_parameters(method="mirror")
        proc.process(preview=True, use_cache=False)
        elapsed = (time.perf_counter() - t0) * 1000.0
        timings["mirror_tiling"] = elapsed
        logger.info("warmup mirror_tiling: %.1f ms", elapsed)

        t0 = time.perf_counter()
        proc.set_parameters(method="offset_crossfade")
        proc.process(preview=True, use_cache=False)
        elapsed = (time.perf_counter() - t0) * 1000.0
        timings["offset_crossfade"] = elapsed
        logger.info("warmup offset_crossfade: %.1f ms", elapsed)

    except Exception as exc:
        logger.warning("warmup mirror/offset_crossfade failed: %s", exc)

    total_ms = sum(timings.values())
    logger.info("warmup total: %.1f ms (%d functions)", total_ms, len(timings))
    return timings
