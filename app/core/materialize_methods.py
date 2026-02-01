"""
Seamless texture generation using Materialize-inspired techniques.
Includes Overlap and Splat methods.
"""
import numpy as np
import cv2
from .gpu_utils import GPUAccelerator, is_cuda_available


def create_falloff_mask(shape, falloff=0.2, circular=False):
    """
    Create a falloff mask (alpha) for blending.
    Smoothly transitions from Hard Square -> Soft Square -> Soft Circle.
    """
    h, w = shape
    
    # 0% Falloff is always solid
    if falloff < 0.001:
        return np.ones((h, w), dtype=np.float32)
        
    # Get normalized coordinates (-1 to 1)
    y, x = np.ogrid[:h, :w]
    ny = (y - h/2.0 + 0.5) / (h/2.0)
    nx = (x - w/2.0 + 0.5) / (w/2.0)
    
    # Distance measures
    dist_box = np.maximum(np.abs(nx), np.abs(ny))
    dist_circ = np.sqrt(nx*nx + ny*ny)
    
    if circular:
        # SPLAT: Transition from Square to Circle as falloff increases
        # 0.0-0.2: Square with increasing soft edge
        # 0.2-1.0: Square becomes Circle
        shape_t = np.clip((falloff - 0.1) * 2.0, 0, 1)
        dist = dist_box * (1.0 - shape_t) + dist_circ * shape_t
    else:
        # OVERLAP: Always stay Square
        dist = dist_box
        
    # FALLOFF LOGIC:
    # Instead of global power, we use "Edge Width".
    # Mask = 1.0 until we reach the edge zone.
    # Edge zone width = falloff.
    
    # Simple linear slope: 1.0 at (1-falloff), 0.0 at 1.0
    edge_width = max(0.005, falloff)
    mask = (1.0 - dist) / edge_width
    mask = np.clip(mask, 0, 1)
    
    # Smoothstep the edge for premium look
    mask = mask * mask * (3.0 - 2.0 * mask)
    
    # Apply light power at very high falloff for extra softness
    if falloff > 0.5:
        p = 1.0 + (falloff - 0.5) * 4.0
        mask = np.power(mask, p)
        
    return mask.astype(np.float32)


def synthesis_overlap(image, overlap_x=0.2, overlap_y=0.2, falloff=0.5):
    """
    Create seamless texture using tile overlap method with resizing.
    1. Overlaps the right edge onto the left edge (Left-to-Right).
    2. Overlaps bottom edge onto top edge.
    3. Crops the seamless result and resizes back to original.
    """
    h, w = image.shape[:2]
    
    # Needs float for blending
    img_f = image.astype(np.float32)
    
    # --- X Pass ---
    if overlap_x > 0:
        blend_w = int(w * overlap_x)
        if blend_w > 0:
            # We blend the Right Edge (src) onto the Left Edge (dst).
            # The region [0 : blend_w] becomes a mix of Image[0:blend_w] and Image[W-blend_w : W].
            
            # Create gradient 0 -> 1
            # 0 means pure Left (Original), 1 means pure Right (Wrapped)
            # Actually, to make it seamless at the new cut point (W-blend_w), 
            # the Left Edge (0) must match the New Right Edge.
            # The New Right Edge corresponds to Image[W-blend_w].
            # So at 0, we need 100% of Image[W-blend_w].
            # At blend_w, we need 100% of Image[blend_w].
            
            # Wait, "Left to Right 0 to 1". 
            # If 0 is Left edge of blended region.
            # We want Left Edge of Result to match Right Edge of Result.
            # Right Edge of Result is Image[W-blend_w].
            # So Left Edge of Result (at 0) must be equal to Image[W-blend_w].
            
            # So at x=0, we want 100% of (Right Side of Image).
            # At x=blend_w, we want 100% of (Left Side of Image).
            
            # Gradient: 1 -> 0
            t = np.linspace(1, 0, blend_w)
            
            # Apply Falloff (Gamma)
            # falloff 0.5 is linear.
            # < 0.5 is sharper transition?
            # Let's map 0..1 to power.
            # Power = 1 / (falloff * 2 + 0.001) ?
            # Let's try simple power law: t^Power.
            
            # If falloff is small (0.1), we want hard edge.
            # If falloff is large (0.9), we want soft.
            # Actually linearity is usually preferred.
            # Let's treat falloff as controlling the shape.
            
            # Safe Falloff Mapping:
            # 0.1 -> Hard (Power ~ 10)
            # 0.5 -> Linear (Power 1)
            # 0.9 -> Smoother (Power ~ 0.5)
            
            if falloff > 0.01:
                gamma = 1.0 / (falloff * 2.0) # 0.5 -> 1.0. 0.1 -> 5.0. 1.0 -> 0.5.
                t = np.power(t, gamma)
            
            # Reshape for broadcasting
            t = t[np.newaxis, :] # (1, blend_w)
            if len(image.shape) == 3:
                t = t[:, :, np.newaxis]
            
            # Left Region (Original)
            left_strip = img_f[:, 0:blend_w]
            # Right Region (Wrapped)
            right_strip = img_f[:, w-blend_w:w]
            
            # Blend: Result = Left * (1-t) + Right * t
            # At 0: t=1 -> Right. Matches Right Edge. Correct.
            # At blend_w: t=0 -> Left. Matches continuing image. Correct.
            blended_strip = left_strip * (1.0 - t) + right_strip * t
            
            # Replace Left Strip
            img_f[:, 0:blend_w] = blended_strip
            
            # Crop the Right Strip (it's now redundant/wrapped)
            new_w = w - blend_w
            img_f = img_f[:, 0:new_w]
            
            # Resize back to W
            # Note: We resize BEFORE Y-pass to keep Y logic simple, or AFTER?
            # Materialize might process both then resize.
            # If we crop X, the aspect ratio changes. X coordinates for Y-pass are different.
            # But Y-pass operates on rows. It doesn't care about X width.
            # So we can proceed.

    # --- Y Pass ---
    # Update current shape
    h_curr, w_curr = img_f.shape[:2]
    
    if overlap_y > 0:
        blend_h = int(h * overlap_y) # Blend amount based on ORIGINAL height usually
        # But we are working on cropped image?
        # Let's use fraction of CURRENT height or ORIGINAL?
        # "Overlap Y" slider usually means fraction of final?
        # Let's use fraction of CURRENT height to be safe.
        blend_h = int(h_curr * overlap_y)
        
        if blend_h > 0:
            # Gradient 1 -> 0 (Top needs to match Bottom)
            t = np.linspace(1, 0, blend_h)
            
            if falloff > 0.01:
                gamma = 1.0 / (falloff * 2.0)
                t = np.power(t, gamma)
            
            t = t[:, np.newaxis] # (blend_h, 1)
            if len(image.shape) == 3:
                t = t[:, :, np.newaxis]
            
            top_strip = img_f[0:blend_h, :]
            bottom_strip = img_f[h_curr-blend_h:h_curr, :]
            
            blended_strip = top_strip * (1.0 - t) + bottom_strip * t
            
            img_f[0:blend_h, :] = blended_strip
            
            # Crop Bottom
            new_h = h_curr - blend_h
            img_f = img_f[0:new_h, :]

    # --- Final Resize ---
    # Resize back to original W, H
    result = cv2.resize(img_f, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return np.clip(result, 0, 255).astype(image.dtype)


from .materialize_methods_jit import synthesis_splat_jit

def synthesis_splat(image, new_size=(1024, 1024), 
                   grid_size=8, scale=1.0, 
                   rotation=0, rand_rot=0, 
                   wobble=0.2, falloff=0.2):
    """
    Create seamless texture using splatting (Texture Bombing).
    Optimized with Numba JIT.
    """
    target_h, target_w = new_size
    h, w = image.shape[:2]
    
    # 1. Initialize canvas
    # VISUAL FIX: User complained about "Improper" look (Grid pattern).
    # The cause was the underlying Tiled Background showing through.
    # Solution: Initialize canvas with the AVERAGE COLOR of the image.
    # This removes the repetitive grid entirely.
    
    mean_color = cv2.mean(image)[:3] if len(image.shape) == 3 else cv2.mean(image)[0]
    
    if len(image.shape) == 3:
        canvas = np.full((target_h, target_w, 3), mean_color, dtype=np.uint8)
    else:
        canvas = np.full((target_h, target_w), mean_color, dtype=np.uint8)
        
    # No need to blur anymore since it's a solid color
    # canvas = cv2.GaussianBlur(canvas, (21, 21), 0)
        
    canvas = canvas.astype(np.float32)
    
    # 2. Pre-calculate rotated patches
    is_preview = (target_h <= 384 and target_w <= 384)
    
    # Optimization: Cap variations for performance
    max_variations = 4 if is_preview else 16 
    num_variations = 1 if rand_rot < 0.01 else max_variations 
    
    # Optimization: Reduce grid density for preview if needed
    effective_grid_size = int(grid_size)
    if is_preview:
        if effective_grid_size > 12: effective_grid_size = 12
    
    # Calculate Cell Size
    cells_x = effective_grid_size
    cells_y = effective_grid_size
    cell_w = target_w / cells_x
    cell_h = target_h / cells_y
    
    # MAJOR LOGIC FIX: Splat Scale 1x = Original Image Size
    # We ignore cell_w for patch sizing to ensure 1:1 resolution matches user expectation.
    # Scale adjusts the patch size relative to the original image.
    
    target_patch_w = int(w * scale)
    target_patch_h = int(h * scale)
    
    # Ensure minimum size
    target_patch_w = max(4, target_patch_w)
    target_patch_h = max(4, target_patch_h)
    
    # Resize base patch
    base_patch = cv2.resize(image, (target_patch_w, target_patch_h), interpolation=cv2.INTER_AREA)
    h_small, w_small = base_patch.shape[:2] # Update dimensions
    
    patches = []
    masks = []
    
    # Adjust mask falloff 
    base_mask = create_falloff_mask((h_small, w_small), falloff=falloff, circular=True)
    
    # Ensure 3D dimensions
    if len(image.shape) == 3 and len(base_mask.shape) == 2:
        base_mask = base_mask[:, :, np.newaxis]
    elif len(image.shape) == 2 and len(base_mask.shape) == 2:
        if len(base_patch.shape) == 2:
             base_patch = base_patch[:, :, np.newaxis]
        base_mask = base_mask[:, :, np.newaxis]

    # Pre-generate variations
    for i in range(num_variations):
        if num_variations == 1:
            angle = rotation
        else:
            step = (i / (num_variations - 1)) - 0.5
            angle = rotation + step * rand_rot * 360
        
        if abs(angle) > 0.1:
            M = cv2.getRotationMatrix2D((w_small/2, h_small/2), angle, 1.0)
            p = cv2.warpAffine(base_patch, M, (w_small, h_small))
            m = cv2.warpAffine(base_mask, M, (w_small, h_small))
            
            if len(p.shape) == 2: p = p[:, :, np.newaxis]
            if len(m.shape) == 2: m = m[:, :, np.newaxis]
        else:
            p = base_patch.copy()
            m = base_mask.copy()
            
        patches.append(p.astype(np.float32))
        masks.append(m)
        
    # Convert list to array for Numba
    # Numba needs uniform arrays. All patches must be same size.
    # They are (w, h) so it's fine.
    patches_arr = np.array(patches) # (N, H, W, C)
    masks_arr = np.array(masks)     # (N, H, W, 1)

    # 3. Coordinate Generation
    cells_x = effective_grid_size
    cells_y = effective_grid_size
    cell_w = target_w / cells_x
    cell_h = target_h / cells_y
    
    grid_y, grid_x = np.mgrid[0:cells_y, 0:cells_x]
    coords = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    
    # Randomize
    rng = np.random.RandomState(42)  # Fixed seed for stability during drag usually, but user wants randomize?
    # Actually param 'splat_randomize' passes seed?
    # We don't have seed param here? 
    # Ah, the caller sets self.splat_randomize. 
    # But wait, synthesis_splat signature doesn't take seed!
    # I should add seed param or just use fixed.
    # User complained "crashes when I edit much", consistent shuffle is better.
    rng.shuffle(coords) 
    
    num_splats = coords.shape[0]
    final_coords = np.zeros((num_splats, 2), dtype=np.int32)
    indices = np.zeros(num_splats, dtype=np.int32)
    
    for i in range(num_splats):
        gx, gy = coords[i]
        
        # Base pos
        cx = (gx + 0.5) * cell_w
        cy = (gy + 0.5) * cell_h
        
        # Wobble
        cx += (rng.rand() - 0.5) * cell_w * wobble * 2
        cy += (rng.rand() - 0.5) * cell_h * wobble * 2
        
        # Wrap
        cx = cx % target_w
        cy = cy % target_h
        
        # MEANINGFUL CHANGE: Use the *small* patch dimensions.
        # Previously used h, w (original image size).
        ph, pw = h_small, w_small 
        top = int(cy - ph/2)
        left = int(cx - pw/2)
        
        final_coords[i, 0] = top
        final_coords[i, 1] = left
        indices[i] = rng.randint(0, num_variations)
    
    # 4. Execute JIT Splatting
    # Ensure canvas is contiguous
    canvas = np.ascontiguousarray(canvas)
    
    result = synthesis_splat_jit(
        canvas, 
        patches_arr, 
        masks_arr, 
        final_coords, 
        indices, 
        target_h, 
        target_w
    )

    return np.clip(result, 0, 255).astype(np.uint8)
