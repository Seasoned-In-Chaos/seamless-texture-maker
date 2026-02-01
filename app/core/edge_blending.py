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
    Create a blend mask for the center cross seam using power curve falloff.
    
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
    
    # Calculate gamma for power curve
    # falloff 0.5 -> gamma 1.0 (linear)
    # falloff 0.2 -> gamma 2.5 (sharper transition)
    # falloff 0.8 -> gamma 0.6 (softer transition)
    gamma = 1.0 / (max(0.01, falloff) * 2.0)
    
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
        # Apply power curve
        curve = np.power(dist, gamma)
        mask[:, x1:x2] *= curve[np.newaxis, :]
    
    # Vertical blend zone (horizontal seam at center)
    cy = height // 2
    y1 = max(0, cy - half_blend)
    y2 = min(height, cy + half_blend)
    
    if symmetric and half_blend > 0:
        y_coords = np.arange(y1, y2)
        dist = np.abs(y_coords - cy) / half_blend
        curve = np.power(dist, gamma)
        mask[y1:y2, :] *= curve[:, np.newaxis]
    
    return mask


def blend_seams(image, blend_strength=0.5, smoothness=0.5, symmetric=True):
    """
    Blend the seams at the center of an offset image.
    
    Args:
        image: Input image (offset so seams are at center)
        blend_strength: Strength of the blend (determines width)
        smoothness: Edge Falloff (0.0=Sharp, 1.0=Soft) (Used as falloff param)
        symmetric: Use symmetric blending
    
    Returns:
        Image with blended seams
    """
    h, w = image.shape[:2]
    
    # Calculate blend width based on strength (seam width)
    max_blend_width = min(h, w) // 4
    blend_width = int(max_blend_width * blend_strength)
    
    if blend_width < 2:
        return image.copy()
    
    
    if not symmetric:
        # Soft Falloff (Gaussian Blur Method)
        # This blurs the seam area to create a smooth transition without mirroring
        
        # Create mask for the blend region
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        
        # Horizontal seam (Vertical strip at center)
        cx = w // 2
        half_blend = blend_width // 2
        x1 = max(0, cx - half_blend)
        x2 = min(w, cx + half_blend)
        
        # Create gradient for mask
        if half_blend > 0:
             x_coords = np.arange(x1, x2)
             dist = np.abs(x_coords - cx) / half_blend # 0 at center, 1 at edge
             # Invert: 1 at center, 0 at edge
             mask_vals = 1.0 - dist
             # Smooth it
             mask_vals = mask_vals * mask_vals * (3.0 - 2.0 * mask_vals)
             mask[:, x1:x2] = np.maximum(mask[:, x1:x2], mask_vals[np.newaxis, :])
        
        # Vertical seam (Horizontal strip at center)
        cy = h // 2
        y1 = max(0, cy - half_blend)
        y2 = min(h, cy + half_blend)
        
        if half_blend > 0:
             y_coords = np.arange(y1, y2)
             dist = np.abs(y_coords - cy) / half_blend
             mask_vals = 1.0 - dist
             mask_vals = mask_vals * mask_vals * (3.0 - 2.0 * mask_vals)
             mask[y1:y2, :] = np.maximum(mask[y1:y2, :], mask_vals[:, np.newaxis])
             
        # Apply blur
        # Blur radius controlled by smoothness param
        # smoothness 0.0 -> small radius (3)
        # smoothness 1.0 -> large radius (proportional to blend width)
        
        base_radius = max(3, blend_width // 2)
        scaled_radius = int(3 + (base_radius - 3) * smoothness)
        blur_radius = scaled_radius | 1 # Must be odd
        
        if blur_radius < 3: blur_radius = 3
            
        blurred = cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)
        
        # Blend original and blurred using mask
        if len(image.shape) == 3:
            mask = mask[:, :, np.newaxis]
            
        result = image.astype(np.float32) * (1 - mask) + blurred.astype(np.float32) * mask
        return result.astype(image.dtype)
        
    # Standard Symmetric (Mirror) Blending
    # VECTORIZED IMPROVED: Use distance-based gradient blending without Python loops
    # This preserves edge details while creating smooth transitions
    
    result = image.copy().astype(np.float32)
    
    # Horizontal seam blending (at center vertical line)
    cx = w // 2
    half_blend = blend_width // 2
    
    if half_blend > 0:
        # Vectorized blending - create all weights at once
        offsets = np.arange(1, half_blend + 1)
        t_values = offsets / half_blend
        
        # Apply smoothstep interpolation vectorized
        if smoothness > 0.01:
            t_values = t_values * t_values * (3.0 - 2.0 * t_values)
            gamma = 1.0 / (smoothness * 2.0)
            t_values = np.power(t_values, gamma)
        
        # Process all columns at once using broadcasting
        for i, (offset, weight) in enumerate(zip(offsets, t_values)):
            left_col = (cx - offset) % w
            right_col = (cx + offset) % w
            result[:, left_col] = (1 - weight) * image[:, left_col] + weight * image[:, right_col]
    
    # Vertical seam blending (at center horizontal line)
    cy = h // 2
    
    if half_blend > 0:
        offsets = np.arange(1, half_blend + 1)
        t_values = offsets / half_blend
        
        # Apply smoothstep interpolation vectorized
        if smoothness > 0.01:
            t_values = t_values * t_values * (3.0 - 2.0 * t_values)
            gamma = 1.0 / (smoothness * 2.0)
            t_values = np.power(t_values, gamma)
        
        # Process all rows at once
        for i, (offset, weight) in enumerate(zip(offsets, t_values)):
            top_row = (cy - offset) % h
            bottom_row = (cy + offset) % h
            result[top_row, :] = (1 - weight) * image[top_row, :] + weight * image[bottom_row, :]
    
    return result.astype(image.dtype)


def apply_edge_blend(image, left_edge, right_edge, blend_width):
    """
    Blend two edges together smoothly.
    
    Args:
        image: Full image
        left_edge: Left edge strip
        right_edge: Right edge strip
        blend_width: Width of blend zone
    
    Returns:
        Blended image
    """
    result = image.copy()
    
    # Create linear gradient for blending
    gradient = np.linspace(0, 1, blend_width).astype(np.float32)
    
    if len(image.shape) == 3:
        gradient = gradient[np.newaxis, :, np.newaxis]
        gradient = np.broadcast_to(gradient, (image.shape[0], blend_width, image.shape[2]))
    else:
        gradient = np.broadcast_to(gradient, (image.shape[0], blend_width))
    
    # Blend the edges
    blended = (left_edge * (1 - gradient) + right_edge * gradient).astype(image.dtype)
    
    return blended
