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
    
    Args:
        shape: (h, w) tuple
        falloff: Falloff strength (0.0-1.0)
        circular: If True, creates circular falloff (for splats)
    
    Returns:
        Float32 mask 0.0-1.0
    """
    h, w = shape
    
    if circular:
        # Create circular mask with smooth falloff
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        
        # Calculate normalized distance from center
        ny = (y - center_y) / (h / 2)
        nx = (x - center_x) / (w / 2)
        dist = np.sqrt(nx*nx + ny*ny)
        
        # Invert distance so center is 1, edges are 0
        mask = 1.0 - np.clip(dist, 0, 1)
        
        # Apply smoothstep for smoother falloff
        # smoothstep(t) = 3t^2 - 2t^3
        mask = mask * mask * (3.0 - 2.0 * mask)
        
        # Apply falloff parameter
        # Higher falloff = softer edge
        if falloff > 0.01:
            # Remap to create softer or harder edges
            falloff_factor = max(0.1, falloff)
            mask = np.power(mask, 1.0 / falloff_factor)
        
        mask = np.clip(mask, 0, 1)
        
    else:
        # Linear box falloff (for overlap) with smoothstep
        # Create gradients for all 4 edges
        x_grad = np.linspace(0, 1, w)
        x_mask = np.minimum(x_grad, x_grad[::-1]) * 2.0  # Scale to 0-1
        x_mask = np.clip(x_mask, 0, 1)
        
        # Apply smoothstep
        x_mask = x_mask * x_mask * (3.0 - 2.0 * x_mask)
        
        # y gradient
        y_grad = np.linspace(0, 1, h)
        y_mask = np.minimum(y_grad, y_grad[::-1]) * 2.0
        y_mask = np.clip(y_mask, 0, 1)
        
        # Apply smoothstep
        y_mask = y_mask * y_mask * (3.0 - 2.0 * y_mask)
        
        # Combine
        mask = x_mask[np.newaxis, :] * y_mask[:, np.newaxis]
        
        # Apply falloff
        if falloff > 0.01:
            falloff_factor = max(0.1, falloff * 2.0)
            mask = np.power(mask, 1.0 / falloff_factor)
    
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


def synthesis_splat(image, new_size=(1024, 1024), 
                   grid_size=8, scale=1.0, 
                   rotation=0, rand_rot=0, 
                   wobble=0.2, falloff=0.2):
    """
    Create seamless texture using splatting (Texture Bombing).
    Optimized: Pre-calculates rotations and uses alpha blending (Painter's Algo) to preserve detail.
    """
    target_h, target_w = new_size
    h, w = image.shape[:2]
    
    # 1. Initialize canvas
    # Use tiled original image as background to avoid black gaps
    # Or mean color? Tiling is safer for "seamlessness" base.
    if target_h > h or target_w > w:
        # Simple tile logic
        canvas = np.tile(image, (int(np.ceil(target_h/h)), int(np.ceil(target_w/w)), 1)[:len(image.shape)])
        canvas = canvas[:target_h, :target_w]
    else:
        canvas = cv2.resize(image, (target_w, target_h))
        
    canvas = canvas.astype(np.float32)
    
    # 2. Pre-calculate rotated patches to avoid expensive warpAffine in loop
    # We'll create N variations based on rotation randomness
    # plus the base rotation.
    
    # If random rotation is 0, we only need 1 patch.
    # If random rotation > 0, we create a bank of patches.
    # Optimization: Reduce variations for small preview sizes
    is_preview = (target_h <= 384 and target_w <= 384)
    max_variations = 2 if is_preview else 8  # Reduced from 4/16 for better performance
    
    num_variations = 1 if rand_rot < 0.01 else max_variations 
    
    patches = []
    masks = []
    
    # Base patch and mask
    base_patch = image.copy()
    base_mask = create_falloff_mask((h, w), falloff=falloff, circular=True)
    
    # Ensure 3D for broadcasting
    if len(image.shape) == 3 and len(base_mask.shape) == 2:
        base_mask = base_mask[:, :, np.newaxis]
    elif len(image.shape) == 2 and len(base_mask.shape) == 2:
         # Grayscale image, mask is 2D. 
         # We need to be careful. If image is (H,W), mask (H,W) works.
         # But usually we work in (H,W,C) locally for consistency.
         if len(base_patch.shape) == 2:
             base_patch = base_patch[:, :, np.newaxis]
         base_mask = base_mask[:, :, np.newaxis]

    rng = np.random.RandomState(42)
    
    for i in range(num_variations):
        # Calculate angle
        # If variants=1, use just 'rotation'.
        # If variants>1, sample linearly or randomly? 
        # Better to sample linearly to cover the range, then pick randomly.
        if num_variations == 1:
            angle = rotation
        else:
            # Distribute angles evenly across the random range centered on 'rotation'
            # Or just random? Random is fine for pre-calc cache.
            # Let's cover the distribution: -0.5 to 0.5 range
            step = (i / (num_variations - 1)) - 0.5
            angle = rotation + step * rand_rot * 360
        
        if abs(angle) > 0.1:
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            p = cv2.warpAffine(base_patch, M, (w, h))
            m = cv2.warpAffine(base_mask, M, (w, h))
            
            # Restore dims if lost
            if len(p.shape) == 2: p = p[:, :, np.newaxis]
            if len(m.shape) == 2: m = m[:, :, np.newaxis]
        else:
            p = base_patch.copy()
            m = base_mask.copy()
            
        patches.append(p.astype(np.float32))
        masks.append(m)

    # 3. Splatting Loop
    cells_x = grid_size
    cells_y = grid_size
    cell_w = target_w / cells_x
    cell_h = target_h / cells_y
    
    # Coordinate generation
    # We can pre-generate all coordinates
    grid_y, grid_x = np.mgrid[0:cells_y, 0:cells_x]
    
    # Randomize order to prevent rigid stacking?
    # Actually iterating linearly is fine if we randomize the patch selection.
    
    coords = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    rng.shuffle(coords) # Shuffle draw order for better blending
    
    for gx, gy in coords:
        # Base pos
        cx = (gx + 0.5) * cell_w
        cy = (gy + 0.5) * cell_h
        
        # Wobble
        cx += (rng.rand() - 0.5) * cell_w * wobble * 2
        cy += (rng.rand() - 0.5) * cell_h * wobble * 2
        
        # Wrap
        cx = cx % target_w
        cy = cy % target_h
        
        # Select patch
        pidx = rng.randint(0, num_variations)
        cur_patch = patches[pidx]
        cur_mask = masks[pidx] # Alpha 0-1
        
        ph, pw = cur_patch.shape[:2]
        top = int(cy - ph/2)
        left = int(cx - pw/2)
        
        # Painter's Algorithm: Canvas = Canvas * (1 - alpha) + Patch * alpha
        # Note: We are blending ON TOP.
        # Optimized Paste Function
        def blend_clipped(y, x, p_img, p_msk):
             # Clip bounds
             y1, y2 = max(0, y), min(target_h, y + ph)
             x1, x2 = max(0, x), min(target_w, x + pw)
             
             if y1 >= y2 or x1 >= x2: return
             
             sy1, sy2 = y1 - y, y2 - y
             sx1, sx2 = x1 - x, x2 - x
             
             alpha = p_msk[sy1:sy2, sx1:sx2]
             patch_slice = p_img[sy1:sy2, sx1:sx2]
             
             # In-place blending: Canvas = Canvas + (Patch - Canvas) * Alpha
             # Avoids allocating '1-alpha' and 'bg * (1-alpha)'
             
             # ROI reference
             roi = canvas[y1:y2, x1:x2]
             
             # We need to perform: roi[:] = roi + (patch - roi) * alpha
             # Use in-place operators where possible
             
             diff = patch_slice - roi
             diff *= alpha
             roi += diff 
             # roi is a view, so canvas is updated? 
             # Yes, basic slicing returns a view in numpy. But advanced slicing doesn't.
             # Basic slicing: canvas[y1:y2, x1:x2] IS a view.
             # So 'roi += diff' modifies canvas in-place.
             pass

        # Main Draw
        blend_clipped(top, left, cur_patch, cur_mask)
        
        # Wrapping
        wrap_x = left + pw > target_w
        wrap_y = top + ph > target_h
        wrap_neg_x = left < 0
        wrap_neg_y = top < 0
        
        if wrap_x: blend_clipped(top, left - target_w, cur_patch, cur_mask)
        if wrap_neg_x: blend_clipped(top, left + target_w, cur_patch, cur_mask)
        if wrap_y: blend_clipped(top - target_h, left, cur_patch, cur_mask)
        if wrap_neg_y: blend_clipped(top + target_h, left, cur_patch, cur_mask)
        
        if wrap_x and wrap_y: blend_clipped(top - target_h, left - target_w, cur_patch, cur_mask)
        if wrap_x and wrap_neg_y: blend_clipped(top + target_h, left - target_w, cur_patch, cur_mask)
        if wrap_neg_x and wrap_y: blend_clipped(top - target_h, left + target_w, cur_patch, cur_mask)
        if wrap_neg_x and wrap_neg_y: blend_clipped(top + target_h, left + target_w, cur_patch, cur_mask)

    return np.clip(canvas, 0, 255).astype(np.uint8)
