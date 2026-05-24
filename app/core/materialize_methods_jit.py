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
                blend_patch_jit(canvas, patch, mask, draw_top, draw_left,
                                 target_h, target_w)

    return canvas
