"""
JIT-compiled edge blending functions using Numba for maximum performance.
"""
import numpy as np
from numba import jit, prange

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def blend_seam_horizontal_jit(result, image, cx, half_blend, weights):
    """
    JIT-compiled horizontal seam blending.
    
    Args:
        result: Output array (modified in-place)
        image: Source image
        cx: Center x coordinate
        half_blend: Half of blend width
        weights: Pre-calculated blend weights
    """
    h, w = image.shape[:2]
    
    for i in prange(len(weights)):
        offset = i + 1
        weight = weights[i]
        inv_weight = 1.0 - weight
        
        left_col = (cx - offset) % w
        right_col = (cx + offset) % w
        
        # Blend all rows for this column
        for y in range(h):
            if len(image.shape) == 3:
                # Color image
                for c in range(image.shape[2]):
                    result[y, left_col, c] = inv_weight * image[y, left_col, c] + weight * image[y, right_col, c]
            else:
                # Grayscale
                result[y, left_col] = inv_weight * image[y, left_col] + weight * image[y, right_col]


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def blend_seam_vertical_jit(result, image, cy, half_blend, weights):
    """
    JIT-compiled vertical seam blending.
    
    Args:
        result: Output array (modified in-place)
        image: Source image
        cy: Center y coordinate
        half_blend: Half of blend width
        weights: Pre-calculated blend weights
    """
    h, w = image.shape[:2]
    
    for i in prange(len(weights)):
        offset = i + 1
        weight = weights[i]
        inv_weight = 1.0 - weight
        
        top_row = (cy - offset) % h
        bottom_row = (cy + offset) % h
        
        # Blend all columns for this row
        for x in range(w):
            if len(image.shape) == 3:
                # Color image
                for c in range(image.shape[2]):
                    result[top_row, x, c] = inv_weight * image[top_row, x, c] + weight * image[bottom_row, x, c]
            else:
                # Grayscale
                result[top_row, x] = inv_weight * image[top_row, x] + weight * image[bottom_row, x]


@jit(nopython=True, fastmath=True)
def calculate_blend_weights(half_blend, smoothness):
    """
    Pre-calculate all blend weights with smoothstep.
    
    Args:
        half_blend: Half of blend width
        smoothness: Smoothness parameter
    
    Returns:
        Array of blend weights
    """
    weights = np.empty(half_blend, dtype=np.float32)
    
    for i in range(half_blend):
        offset = i + 1
        t = offset / half_blend
        
        # Apply smoothstep
        if smoothness > 0.01:
            t = t * t * (3.0 - 2.0 * t)
            gamma = 1.0 / (smoothness * 2.0)
            t = t ** gamma
        
        weights[i] = t
    
    return weights


def blend_seams_fast(image, blend_strength=0.5, smoothness=0.5):
    """
    Ultra-fast JIT-compiled seam blending.
    
    Args:
        image: Input image (offset so seams are at center)
        blend_strength: Strength of the blend (determines width)
        smoothness: Edge Falloff (0.0=Sharp, 1.0=Soft)
    
    Returns:
        Image with blended seams
    """
    h, w = image.shape[:2]
    
    # Calculate blend width
    max_blend_width = min(h, w) // 4
    blend_width = int(max_blend_width * blend_strength)
    
    if blend_width < 2:
        return image.copy()
    
    half_blend = blend_width // 2
    
    if half_blend < 1:
        return image.copy()
    
    # Pre-calculate weights once
    weights = calculate_blend_weights(half_blend, smoothness)
    
    # Work on float32 for precision
    result = image.astype(np.float32)
    
    # Horizontal blending (JIT-compiled)
    cx = w // 2
    blend_seam_horizontal_jit(result, image.astype(np.float32), cx, half_blend, weights)
    
    # Vertical blending (JIT-compiled)
    cy = h // 2
    blend_seam_vertical_jit(result, result, cy, half_blend, weights)
    
    return result.astype(image.dtype)
