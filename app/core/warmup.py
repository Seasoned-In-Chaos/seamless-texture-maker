"""
Pre-compile Numba JIT functions and warm NumPy paths during splash.

Runs on a background QThread (WarmupThread in splash_screen.py) while
the splash animation plays, so the first user action is stall-free.
"""
from __future__ import annotations

import logging
import time
from typing import Dict

import numpy as np

__all__ = ["warmup_all_jit_functions"]

logger = logging.getLogger("seams.warmup")


def warmup_all_jit_functions() -> Dict[str, float]:
    """Trigger Numba compilation and warm all seamless method paths.

    Each function is called once with a minimal 64×64 array.  Returns a
    dict mapping each component name to its elapsed compile time in ms.
    """
    timings: Dict[str, float] = {}

    tiny_3ch = np.zeros((64, 64, 3), dtype=np.float32)
    tiny_weights = np.zeros(8, dtype=np.float32)

    # ── Overlap Blend JIT (edge blending) ────────────────────────────────
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
        timings["edge_blending_jit"] = (time.perf_counter() - t0) * 1000.0
        logger.info("warmup edge_blending_jit: %.1f ms", timings["edge_blending_jit"])
    except Exception as exc:
        logger.warning("warmup edge_blending_jit failed: %s", exc)

    # ── Splat Synthesis JIT ───────────────────────────────────────────────
    try:
        from .materialize_methods_jit import splat_accumulate_jit, splat_resolve_jit

        patches = np.stack([tiny_3ch.copy()])
        masks = np.ones((1, 64, 64), dtype=np.float32)
        coords = np.array([[0, 0]], dtype=np.int32)
        indices = np.array([0], dtype=np.int32)
        accum = np.zeros((64, 64, 3), dtype=np.float32)
        weight = np.zeros((64, 64), dtype=np.float32)

        t0 = time.perf_counter()
        splat_accumulate_jit(accum, weight, patches, masks, coords, indices)
        splat_resolve_jit(accum, weight, tiny_3ch, np.empty_like(accum))
        timings["splat_jit"] = (time.perf_counter() - t0) * 1000.0
        logger.info("warmup splat_jit: %.1f ms", timings["splat_jit"])
    except Exception as exc:
        logger.warning("warmup splat_jit failed: %s", exc)

    # ── Mirror Tiling (2×2) and Offset + Cross-Fade ───────────────────────
    try:
        from .seamless import SeamlessProcessor

        tiny_img = np.zeros((64, 64, 3), dtype=np.uint8)
        proc = SeamlessProcessor()
        proc.load_image(tiny_img)

        for method in ("mirror", "offset_crossfade"):
            t0 = time.perf_counter()
            proc.set_parameters(method=method)
            proc.process(preview=True, use_cache=False)
            timings[method] = (time.perf_counter() - t0) * 1000.0
            logger.info("warmup %s: %.1f ms", method, timings[method])

    except Exception as exc:
        logger.warning("warmup seamless methods failed: %s", exc)

    total_ms = sum(timings.values())
    logger.info("warmup complete: %.1f ms total (%d components)", total_ms, len(timings))
    return timings
