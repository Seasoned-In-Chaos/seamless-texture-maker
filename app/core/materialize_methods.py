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
    Always has a guaranteed 1px taper at edges for seamless tiling.
    """
    h, w = shape

    # 0% Falloff is always solid (but still taper 1px at boundary)
    if falloff < 0.001:
        mask = np.ones((h, w), dtype=np.float32)
        # Hard 1px border fade to guarantee no visible seam at tile boundary
        mask[0, :] = 0; mask[-1, :] = 0
        mask[:, 0] = 0; mask[:, -1] = 0
        return mask

    # Get normalized coordinates (-1 to 1)
    y, x = np.ogrid[:h, :w]
    ny = (y - h / 2.0 + 0.5) / (h / 2.0)
    nx = (x - w / 2.0 + 0.5) / (w / 2.0)

    # Distance measures
    dist_box = np.maximum(np.abs(nx), np.abs(ny))
    dist_circ = np.sqrt(nx * nx + ny * ny)

    if circular:
        # SPLAT: Transition from Square to Circle as falloff increases
        shape_t = np.clip((falloff - 0.1) * 2.0, 0, 1)
        dist = dist_box * (1.0 - shape_t) + dist_circ * shape_t
    else:
        # OVERLAP: Always stay Square
        dist = dist_box

    # FALLOFF LOGIC: Edge zone width = falloff fraction.
    # mask = 1.0 inside, falls to 0 over the edge zone.
    edge_width = max(0.005, falloff)
    mask = (1.0 - dist) / edge_width
    mask = np.clip(mask, 0, 1)

    # Smoothstep for premium soft edge
    mask = mask * mask * (3.0 - 2.0 * mask)

    # Extra softness at very high falloff
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
            # Create gradient 1 -> 0
            t = np.linspace(1, 0, blend_w)

            hardness = 1.0 / (max(0.001, falloff))
            t = (t - 0.5) * hardness + 0.5
            t = np.clip(t, 0, 1)

            t = t[np.newaxis, :]  # (1, blend_w)
            if len(image.shape) == 3:
                t = t[:, :, np.newaxis]

            left_strip = img_f[:, 0:blend_w]
            right_strip = img_f[:, w - blend_w:w]

            blended_strip = left_strip * (1.0 - t) + right_strip * t
            img_f[:, 0:blend_w] = blended_strip

            new_w = w - blend_w
            img_f = img_f[:, 0:new_w]

    # --- Y Pass ---
    h_curr, w_curr = img_f.shape[:2]

    if overlap_y > 0:
        blend_h = int(h_curr * overlap_y)

        if blend_h > 0:
            t = np.linspace(1, 0, blend_h)

            hardness = 1.0 / (max(0.001, falloff))
            t = (t - 0.5) * hardness + 0.5
            t = np.clip(t, 0, 1)

            t = t[:, np.newaxis]  # (blend_h, 1)
            if len(image.shape) == 3:
                t = t[:, :, np.newaxis]

            top_strip = img_f[0:blend_h, :]
            bottom_strip = img_f[h_curr - blend_h:h_curr, :]

            blended_strip = top_strip * (1.0 - t) + bottom_strip * t
            img_f[0:blend_h, :] = blended_strip

            new_h = h_curr - blend_h
            img_f = img_f[0:new_h, :]

    # --- Final Resize ---
    result = cv2.resize(img_f, (w, h), interpolation=cv2.INTER_LINEAR)

    return np.clip(result, 0, 255).astype(image.dtype)


from .materialize_methods_jit import synthesis_splat_jit


def synthesis_splat(image, new_size=(1024, 1024),
                    grid_size=8, scale=1.0,
                    rotation=0, rand_rot=0,
                    wobble=0.2, falloff=0.2,
                    cached_batches=None):
    """
    Create seamless texture using splatting (Texture Bombing).
    Optimized with Numba JIT.

    Key fixes:
    - Grid density automatically scales with patch size to avoid coverage gaps.
    - Wrapping handles patches larger than the canvas (multi-tile wrapping).
    - Falloff mask ensures clean blending with no harsh seam lines.

    Args:
        cached_batches: Tuple (patches_arr, masks_arr, patch_hw) or None
    Returns:
        (result_image, (patches_arr, masks_arr))
    """
    target_h, target_w = new_size
    h, w = image.shape[:2]

    # 1. Initialize canvas with mean color
    mean_color = cv2.mean(image)[:3] if len(image.shape) == 3 else cv2.mean(image)[0]

    if len(image.shape) == 3:
        canvas = np.full((target_h, target_w, 3), mean_color, dtype=np.float32)
    else:
        canvas = np.full((target_h, target_w), mean_color, dtype=np.float32)

    # 2. Prepare Patches (Use Cache if available)
    if cached_batches is not None:
        patches_arr, masks_arr = cached_batches
        if patches_arr.ndim == 4:
            h_small, w_small = patches_arr.shape[1:3]
        else:
            h_small, w_small = patches_arr.shape[1:3]
        num_variations = patches_arr.shape[0]

    else:
        is_preview = (target_h <= 384 and target_w <= 384)

        # Cap variations for performance
        max_variations = 4 if is_preview else 16
        num_variations = 1 if rand_rot < 0.01 else max_variations

        # Calculate patch size from scale
        target_patch_w = int(w * scale)
        target_patch_h = int(h * scale)
        target_patch_w = max(4, target_patch_w)
        target_patch_h = max(4, target_patch_h)

        # Resize base patch
        base_patch = cv2.resize(image, (target_patch_w, target_patch_h),
                                interpolation=cv2.INTER_AREA)
        h_small, w_small = base_patch.shape[:2]

        patches = []
        masks = []

        # Create improved falloff mask — always use circular=True for splat
        base_mask = create_falloff_mask((h_small, w_small), falloff=falloff, circular=True)

        # Ensure 3D masks
        if len(image.shape) == 3 and len(base_mask.shape) == 2:
            base_mask = base_mask[:, :, np.newaxis]
        elif len(image.shape) == 2 and len(base_mask.shape) == 2:
            if len(base_patch.shape) == 2:
                base_patch = base_patch[:, :, np.newaxis]
            base_mask = base_mask[:, :, np.newaxis]

        # Generate variations
        for i in range(num_variations):
            if num_variations == 1:
                angle = rotation
            else:
                step = (i / (num_variations - 1)) - 0.5
                angle = rotation + step * rand_rot * 360

            if abs(angle) > 0.1:
                M = cv2.getRotationMatrix2D((w_small / 2, h_small / 2), angle, 1.0)
                p = cv2.warpAffine(base_patch, M, (w_small, h_small),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT_101)
                m = cv2.warpAffine(base_mask, M, (w_small, h_small),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                if len(p.shape) == 2:
                    p = p[:, :, np.newaxis]
                if len(m.shape) == 2:
                    m = m[:, :, np.newaxis]
            else:
                p = base_patch.copy()
                m = base_mask.copy()

            patches.append(p.astype(np.float32))
            masks.append(m.astype(np.float32))

        patches_arr = np.array(patches)  # (N, H, W, C)
        masks_arr = np.array(masks)       # (N, H, W, 1)

    # 3. Coordinate Generation
    # KEY FIX: scale the grid density so patches always cover the canvas.
    # When scale > 1, patches are large and we need FEWER cells but ALWAYS enough
    # to tile the canvas with guaranteed overlap.
    # cell_size = patch_size / 2 ensures 50% minimum overlap between adjacent patches.
    cell_w = max(1.0, w_small / 2.0)
    cell_h = max(1.0, h_small / 2.0)
    cells_x = max(2, int(np.ceil(target_w / cell_w)) + 2)  # +2 for edge coverage
    cells_y = max(2, int(np.ceil(target_h / cell_h)) + 2)

    # For preview mode, cap total cells to keep it fast
    is_preview = (target_h <= 384 and target_w <= 384)
    if is_preview:
        cells_x = min(cells_x, 16)
        cells_y = min(cells_y, 16)

    # Generate grid positions, including a border ring of extra cells to cover edges
    # We distribute splats covering a slightly larger area than the canvas and wrap
    gx_arr, gy_arr = np.meshgrid(np.arange(cells_x), np.arange(cells_y))
    gx_flat = gx_arr.ravel().astype(np.float32)
    gy_flat = gy_arr.ravel().astype(np.float32)

    # Cell center positions (in canvas space)
    cx_base = (gx_flat + 0.5) * cell_w - cell_w  # offset by one cell to start from negative edge
    cy_base = (gy_flat + 0.5) * cell_h - cell_h

    n_splats = len(cx_base)

    # Fixed RNG for stability
    rng = np.random.RandomState(42)

    # Vectorized wobble
    wobble_x = (rng.rand(n_splats) - 0.5) * cell_w * wobble * 2.0
    wobble_y = (rng.rand(n_splats) - 0.5) * cell_h * wobble * 2.0

    cx_final = (cx_base + wobble_x) % target_w
    cy_final = (cy_base + wobble_y) % target_h

    # Convert to top-left integer coordinates
    top_arr = (cy_final - h_small / 2.0).astype(np.int32)
    left_arr = (cx_final - w_small / 2.0).astype(np.int32)

    # Random patch variation indices
    idx_arr = rng.randint(0, num_variations, size=n_splats, dtype=np.int32)

    final_coords = np.column_stack((top_arr, left_arr)).astype(np.int32)

    # 4. Execute JIT Splatting
    canvas = np.ascontiguousarray(canvas)
    patches_arr = np.ascontiguousarray(patches_arr)
    masks_arr = np.ascontiguousarray(masks_arr)

    result = synthesis_splat_jit(
        canvas,
        patches_arr,
        masks_arr,
        final_coords,
        idx_arr,
        target_h,
        target_w
    )

    return np.clip(result, 0, 255).astype(np.uint8), (patches_arr, masks_arr)
