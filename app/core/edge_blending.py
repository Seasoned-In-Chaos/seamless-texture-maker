"""
Edge blending algorithms for seamless texture generation.
"""
import numpy as np
import cv2


def create_gradient_mask(width, height, direction='horizontal', symmetric=True):
    """
    Create a gradient mask for blending.
    
    Args:
        width: Mask width
        height: Mask height
        direction: 'horizontal' or 'vertical'
        symmetric: If True, creates a symmetric gradient (0->1->0)
    
    Returns:
        Gradient mask (float32, 0.0-1.0)
    """
    if direction == 'horizontal':
        if symmetric:
            # Gradient from edges to center
            half = width // 2
            left = np.linspace(0, 1, half)
            right = np.linspace(1, 0, width - half)
            gradient = np.concatenate([left, right])
        else:
            gradient = np.linspace(0, 1, width)
        mask = np.tile(gradient, (height, 1))
    else:  # vertical
        if symmetric:
            half = height // 2
            top = np.linspace(0, 1, half)
            bottom = np.linspace(1, 0, height - half)
            gradient = np.concatenate([top, bottom])
        else:
            gradient = np.linspace(0, 1, height)
        mask = np.tile(gradient.reshape(-1, 1), (1, width))
    
    return mask.astype(np.float32)


def create_blend_mask(height, width, blend_width, symmetric=True, falloff=0.5):
    """
    Create a blend mask for the center cross seam using overlap-style hardness falloff.
    
    Args:
        height: Image height
        width: Image width
        blend_width: Width of the blend zone
        symmetric: Use symmetric blending
        falloff: Gradient falloff (0.0-1.0). 0.5 is linear.
    
    Returns:
        Blend mask (float32, 0.0-1.0)
    """
    mask = np.ones((height, width), dtype=np.float32)
    
    # Calculate hardness
    # falloff 1.0 -> hardness 1.0 (linear)
    # falloff 0.0 -> hardness 1000 (hard step)
    hardness = 1.0 / max(0.001, falloff)
    
    # Horizontal blend zone (vertical seam at center)
    cx = width // 2
    half_blend = blend_width // 2
    x1 = max(0, cx - half_blend)
    x2 = min(width, cx + half_blend)
    
    if symmetric and half_blend > 0:
        # Vectorized gradient calculation
        x_coords = np.arange(x1, x2)
        # Normalized distance 0..1 (0 at center, 1 at edge)
        dist = np.abs(x_coords - cx) / half_blend
        # Apply hardness curve
        curve = np.clip(dist * hardness, 0, 1)
        mask[:, x1:x2] *= curve[np.newaxis, :]
    
    # Vertical blend zone (horizontal seam at center)
    cy = height // 2
    y1 = max(0, cy - half_blend)
    y2 = min(height, cy + half_blend)
    
    if symmetric and half_blend > 0:
        y_coords = np.arange(y1, y2)
        dist = np.abs(y_coords - cy) / half_blend
        curve = np.clip(dist * hardness, 0, 1)
        mask[y1:y2, :] *= curve[:, np.newaxis]
    
    return mask


def blend_seams(image, blend_strength=0.5, smoothness=0.5, symmetric=True, original_image=None, fixed_width=None):
    """
    Blend the seams at the center of an offset image.
    Fully vectorized — no Python per-pixel loops.
    
    Args:
        image: Input image (offset so seams are at center)
        blend_strength: Strength of the blend (determines width)
        smoothness: Edge Falloff (0.0=Sharp, 1.0=Soft) (Used as falloff param)
        symmetric: Use symmetric blending
        original_image: The original offset image (required for non-symmetric falloff)
        fixed_width: Override blend width in pixels (useful to match inpaint mask)
    
    Returns:
        Image with blended seams
    """
    h, w = image.shape[:2]
    
    # Calculate blend width
    if fixed_width is not None:
        blend_width = int(fixed_width * 1.5)
    else:
        max_blend_width = min(h, w) // 10
        blend_width = int(max_blend_width * blend_strength)
    
    if blend_width < 2:
        return image.copy()
    
    if not symmetric:
        # ── Non-Symmetric Blend (vectorized) ─────────────────────
        if original_image is None:
            return image.copy()

        result = image.copy().astype(np.float32)
        orig = original_image.astype(np.float32)
        img_f = image.astype(np.float32)
        
        cx = w // 2
        cy = h // 2
        half_blend = blend_width // 2
        
        if half_blend > 0:
            offsets = np.arange(1, half_blend + 1)
            t = offsets / half_blend
            # Cosine S-curve: 1 → 0
            weights = (np.cos(t * np.pi) + 1.0) * 0.5  # shape: (half_blend,)
            
            # ── Horizontal seam (column indices) ──
            left_cols  = (cx - offsets) % w
            right_cols = (cx + offsets) % w
            
            # weights shape for broadcasting: (1, half_blend) or (1, half_blend, 1)
            if len(image.shape) == 3:
                w_col = weights[np.newaxis, :, np.newaxis]  # (1, N, C)
            else:
                w_col = weights[np.newaxis, :]              # (1, N)
            
            result[:, left_cols]  = img_f[:, left_cols]  * w_col + orig[:, left_cols]  * (1 - w_col)
            result[:, right_cols] = img_f[:, right_cols] * w_col + orig[:, right_cols] * (1 - w_col)
            
            # ── Vertical seam (row indices) ──
            top_rows    = (cy - offsets) % h
            bottom_rows = (cy + offsets) % h
            
            if len(image.shape) == 3:
                w_row = weights[:, np.newaxis, np.newaxis]  # (N, 1, C)
            else:
                w_row = weights[:, np.newaxis]               # (N, 1)
            
            result[top_rows, :]    = img_f[top_rows, :]    * w_row + orig[top_rows, :]    * (1 - w_row)
            result[bottom_rows, :] = img_f[bottom_rows, :] * w_row + orig[bottom_rows, :] * (1 - w_row)
                     
        return result.astype(image.dtype)
        
    # ── Symmetric (Mirror) Blending (vectorized) ──────────────
    result = image.copy().astype(np.float32)
    img_f = image.astype(np.float32)
    
    cx = w // 2
    cy = h // 2
    half_blend = blend_width // 2
    
    if half_blend > 0:
        offsets = np.arange(1, half_blend + 1)
        t = offsets / half_blend
        weights = 0.25 * (np.cos(t * np.pi) + 1.0)  # shape: (half_blend,)
        
        # ── Horizontal seam ──
        left_cols  = (cx - offsets) % w
        right_cols = (cx + offsets) % w
        
        if len(image.shape) == 3:
            w_col = weights[np.newaxis, :, np.newaxis]
        else:
            w_col = weights[np.newaxis, :]
        
        result[:, left_cols] = (1 - w_col) * img_f[:, left_cols] + w_col * img_f[:, right_cols]
        
        # ── Vertical seam ──
        top_rows    = (cy - offsets) % h
        bottom_rows = (cy + offsets) % h
        
        if len(image.shape) == 3:
            w_row = weights[:, np.newaxis, np.newaxis]
        else:
            w_row = weights[:, np.newaxis]
        
        result[top_rows, :] = (1 - w_row) * img_f[top_rows, :] + w_row * img_f[bottom_rows, :]
    
    return result.astype(image.dtype)
