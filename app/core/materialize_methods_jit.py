"""
JIT-compiled Materialize-inspired synthesis methods (Splat, Overlap).
"""
import numpy as np
from numba import jit, prange
import math

@jit(nopython=True, fastmath=True)
def smoothstep_jit(t):
    return t * t * (3.0 - 2.0 * t)

@jit(nopython=True, fastmath=True)
def blend_patch_jit(canvas, patch, mask, top, left, target_h, target_w):
    """
    Blends a patch onto the canvas using alpha blending (in-place).
    Handles clipping and wrapping manually.
    """
    ph, pw = patch.shape[:2]
    # Dimensions assumed 3D (H,W,C) or 2D (H,W)
    
    # 1. Calculate clipping
    y1 = max(0, top)
    y2 = min(target_h, top + ph)
    x1 = max(0, left)
    x2 = min(target_w, left + pw)
    
    if y1 >= y2 or x1 >= x2:
        return

    # Offsets in patch
    py1 = y1 - top
    py2 = y2 - top
    px1 = x1 - left
    px2 = x2 - left
    
    # Blend loop
    # Numba handles loops efficiently, no need for vectorization tricks
    for y in range(y1, y2):
        py = y - top
        for x in range(x1, x2):
            px = x - left
            
            # Get alpha
            if mask.ndim == 3:
                alpha = mask[py, px, 0] # Assume single channel alpha for now or same across channels
            else:
                alpha = mask[py, px]
                
            if alpha <= 0.001:
                continue
                
            # Blend
            # Canvas = Canvas * (1-alpha) + Patch * alpha
            # Canvas += (Patch - Canvas) * alpha
            
            if canvas.ndim == 3:
                for c in range(canvas.shape[2]):
                    val_c = canvas[y, x, c]
                    val_p = patch[py, px, c]
                    canvas[y, x, c] = val_c + (val_p - val_c) * alpha
            else:
                 val_c = canvas[y, x]
                 val_p = patch[py, px]
                 canvas[y, x] = val_c + (val_p - val_c) * alpha

@jit(nopython=True, parallel=True, fastmath=True)
def synthesis_splat_jit(canvas, patches, masks, coords, indices, target_h, target_w):
    """
    Execute the splatting loop using Numba.
    
    Args:
        canvas: Initialized canvas (H, W, C) float32
        patches: List/Array of patch images (N, H, W, C)
        masks: List/Array of masks (N, H, W, 1)
        coords: (num_splats, 2) array of (top, left) coordinates
        indices: (num_splats,) array of patch indices to use
        target_h, target_w: Dimensions
    """
    num_splats = coords.shape[0]
    
    # Serial loop because we are modifying the same canvas (race conditions if parallel)
    # So 'parallel=True' applies to outer functions if needed, but this loop must be serial?
    # Actually, standard splatting is order-dependent.
    # We can't easily parallelize the writing to canvas without atomic adds or separate buffers.
    # Stick to serial optimized loop.
    
    for i in range(num_splats):
        top = coords[i, 0]
        left = coords[i, 1]
        pidx = indices[i]
        
        patch = patches[pidx]
        mask = masks[pidx]
        
        ph = patch.shape[0]
        pw = patch.shape[1]
        
        # Main Draw
        blend_patch_jit(canvas, patch, mask, top, left, target_h, target_w)
        
        # Wrapping logic check (do we need to draw wrapped copies?)
        # Yes, standard wrapping
        
        wrap_x = (left + pw > target_w)
        wrap_y = (top + ph > target_h)
        wrap_neg_x = (left < 0)
        wrap_neg_y = (top < 0)
        
        if wrap_x: blend_patch_jit(canvas, patch, mask, top, left - target_w, target_h, target_w)
        if wrap_neg_x: blend_patch_jit(canvas, patch, mask, top, left + target_w, target_h, target_w)
        if wrap_y: blend_patch_jit(canvas, patch, mask, top - target_h, left, target_h, target_w)
        if wrap_neg_y: blend_patch_jit(canvas, patch, mask, top + target_h, left, target_h, target_w)
        
        if wrap_x and wrap_y: blend_patch_jit(canvas, patch, mask, top - target_h, left - target_w, target_h, target_w)
        if wrap_x and wrap_neg_y: blend_patch_jit(canvas, patch, mask, top + target_h, left - target_w, target_h, target_w)
        if wrap_neg_x and wrap_y: blend_patch_jit(canvas, patch, mask, top - target_h, left + target_w, target_h, target_w)
        if wrap_neg_x and wrap_neg_y: blend_patch_jit(canvas, patch, mask, top + target_h, left + target_w, target_h, target_w)

    return canvas
