"""
JIT-compiled Materialize-inspired synthesis methods (Splat, Overlap).
Optimized for real-time performance with proper wrapping support.

If the Rust extension (seams_core) is available, it is used automatically.
"""
from __future__ import annotations

import numpy as np
from numba import jit, prange
import math

try:
    from seams_core import splat_synthesize as _rs_splat
    HAS_RUST_SPLAT = True
except ImportError:
    HAS_RUST_SPLAT = False


@jit(nopython=True, fastmath=True, cache=True)
def blend_patch_jit(canvas, patch, mask, top, left, target_h, target_w):
    """
    Blends a patch onto the canvas using alpha blending (in-place).
    Handles clipping. Wrapping is handled by the caller.
    """
    ph = patch.shape[0]
    pw = patch.shape[1]

    # Compute canvas region that overlaps with patch
    y1 = top if top > 0 else 0
    y2 = top + ph if top + ph < target_h else target_h
    x1 = left if left > 0 else 0
    x2 = left + pw if left + pw < target_w else target_w

    if y1 >= y2 or x1 >= x2:
        return

    # Blend loop — Numba handles this efficiently
    is_color = (canvas.ndim == 3)
    if is_color:
        n_channels = canvas.shape[2]
    else:
        n_channels = 1

    for y in range(y1, y2):
        py = y - top
        for x in range(x1, x2):
            px = x - left

            # Get alpha from first channel of mask
            if mask.ndim == 3:
                alpha = mask[py, px, 0]
            else:
                alpha = mask[py, px]

            if alpha <= 0.001:
                continue

            if is_color:
                for c in range(n_channels):
                    canvas[y, x, c] += (patch[py, px, c] - canvas[y, x, c]) * alpha
            else:
                canvas[y, x] += (patch[py, px] - canvas[y, x]) * alpha


@jit(nopython=True, fastmath=True, cache=True)
def accumulate_patch_jit(accum, weights, patch, mask, top, left, target_h, target_w):
    """Add a patch's premultiplied colour and alpha to wrapped output."""
    ph = patch.shape[0]
    pw = patch.shape[1]
    y1 = top if top > 0 else 0
    y2 = top + ph if top + ph < target_h else target_h
    x1 = left if left > 0 else 0
    x2 = left + pw if left + pw < target_w else target_w

    if y1 >= y2 or x1 >= x2:
        return

    is_color = accum.ndim == 3
    channels = accum.shape[2] if is_color else 1
    for y in range(y1, y2):
        py = y - top
        for x in range(x1, x2):
            px = x - left
            alpha = mask[py, px, 0] if mask.ndim == 3 else mask[py, px]
            if alpha <= 0.001:
                continue
            weights[y, x] += alpha
            if is_color:
                for c in range(channels):
                    accum[y, x, c] += patch[py, px, c] * alpha
            else:
                accum[y, x] += patch[py, px, 0] * alpha


@jit(nopython=True, fastmath=True, cache=True)
def synthesis_splat_jit(canvas, patches, masks, coords, indices, target_h, target_w):
    """
    Execute the splatting loop using Numba JIT.
    Handles full wrapping including patches larger than canvas.

    Args:
        canvas: Initialized canvas (H, W, C) float32
        patches: Array of patch images (N, H, W, C) float32
        masks: Array of masks (N, H, W, 1) float32
        coords: (num_splats, 2) array of (top, left) int32 coordinates
        indices: (num_splats,) array of patch indices int32
        target_h, target_w: Canvas dimensions
    """
    num_splats = coords.shape[0]
    ph = patches.shape[1]
    pw = patches.shape[2]

    # Normalized weighted compositing preserves each patch's full falloff.
    # Sequential alpha compositing allowed the last patch drawn to overwrite
    # the feather of earlier patches, which produced hard bands in places.
    base_weight = np.float32(0.001)
    accum = canvas.astype(np.float32) * base_weight
    weights = np.full((target_h, target_w), base_weight, dtype=np.float32)

    for i in range(num_splats):
        top = coords[i, 0]
        left = coords[i, 1]
        pidx = indices[i]

        patch = patches[pidx]
        mask = masks[pidx]

        # Calculate how many canvas tiles this patch could span
        # This handles patches larger than the canvas correctly
        # We tile the draw offsets to cover all intersections
        
        # Range of tile offsets needed in X (patches can overlap multiple tiles when pw > target_w)
        tiles_x_min = (left) // target_w - 1
        tiles_x_max = (left + pw) // target_w + 1
        tiles_y_min = (top) // target_h - 1
        tiles_y_max = (top + ph) // target_h + 1

        for ty in range(tiles_y_min, tiles_y_max + 1):
            draw_top = top - ty * target_h
            # Quick reject if this tile offset is completely outside
            if draw_top >= target_h or draw_top + ph <= 0:
                continue
            for tx in range(tiles_x_min, tiles_x_max + 1):
                draw_left = left - tx * target_w
                if draw_left >= target_w or draw_left + pw <= 0:
                    continue
                accumulate_patch_jit(accum, weights, patch, mask, draw_top,
                                     draw_left, target_h, target_w)

    if canvas.ndim == 3:
        for y in range(target_h):
            for x in range(target_w):
                for c in range(canvas.shape[2]):
                    canvas[y, x, c] = accum[y, x, c] / weights[y, x]
    else:
        for y in range(target_h):
            for x in range(target_w):
                canvas[y, x] = accum[y, x] / weights[y, x]

    return canvas
