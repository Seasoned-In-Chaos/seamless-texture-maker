"""
Seamless texture generation using Materialize-inspired techniques.
Includes Overlap and Splat methods.

All functions accept and return float32 arrays.
"""
from __future__ import annotations

import numpy as np
import cv2
from .gpu_utils import GPUAccelerator, is_cuda_available
from .assertions import assert_float32


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

    # Get normalized pixel-center coordinates (-1 to 1).  Using the actual
    # image bounds is important here: the first and last rows/columns must
    # be part of the falloff, otherwise one-pixel seams can remain on one or
    # more sides of a splat.
    y, x = np.ogrid[:h, :w]
    ny = (2.0 * y / max(1, h - 1)) - 1.0
    nx = (2.0 * x / max(1, w - 1)) - 1.0

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

    # Keep all four edges unambiguously transparent.  This also protects the
    # guarantee for tiny patches where floating-point rounding could leave a
    # non-zero alpha at a corner.
    if h > 1 and w > 1:
        mask[0, :] = 0.0
        mask[-1, :] = 0.0
        mask[:, 0] = 0.0
        mask[:, -1] = 0.0

    return mask.astype(np.float32)


def _blend_periodic_edges(image: np.ndarray, width_x: int = 0,
                          width_y: int = 0, falloff: float = 0.2) -> np.ndarray:
    """Make opposite output borders agree using a local linear cross-fade.

    Synthesis is performed on a wrapped canvas, but discrete patch placement
    can still leave the first and last sampled pixels slightly different. This
    short final pass makes the repeat boundary deterministic without blurring
    the interior of the texture.
    """
    result = image.astype(np.float32, copy=True)
    height, width = result.shape[:2]
    hardness = 1.0 / max(0.05, float(falloff))

    def blend_axis(axis_size, blend_width, axis):
        nonlocal result
        radius = min(max(0, int(blend_width)), axis_size // 2)
        if radius <= 0:
            return
        source = result.copy()
        for distance in range(radius):
            # Full averaging at the boundary, fading linearly toward the
            # untouched interior. Falloff controls how quickly that fade is
            # concentrated near the border.
            linear = 1.0 - (distance / max(1, radius - 1))
            influence = linear ** hardness
            weight = 0.5 * influence
            low = distance
            high = axis_size - 1 - distance
            if axis == 1:
                low_value = source[:, low].copy()
                high_value = source[:, high].copy()
                result[:, low] = low_value * (1.0 - weight) + high_value * weight
                result[:, high] = high_value * (1.0 - weight) + low_value * weight
            else:
                low_value = source[low, :].copy()
                high_value = source[high, :].copy()
                result[low, :] = low_value * (1.0 - weight) + high_value * weight
                result[high, :] = high_value * (1.0 - weight) + low_value * weight

    blend_axis(width, width_x, 1)
    blend_axis(height, width_y, 0)
    return np.clip(result, 0, 255).astype(image.dtype, copy=False)


def synthesis_overlap(image: np.ndarray, overlap_x: float = 0.2,
                      overlap_y: float = 0.2,
                      falloff: float = 0.5) -> np.ndarray:
    """Create seamless texture using tile overlap method with resizing.

    Accepts and returns float32 arrays.
    """
    assert_float32(image, "synthesis_overlap image")
    h, w = image.shape[:2]

    blend_w = int(w * np.clip(overlap_x, 0.0, 0.5))
    blend_h = int(h * np.clip(overlap_y, 0.0, 0.5))
    return _blend_periodic_edges(image, blend_w, blend_h, falloff)


from .materialize_methods_jit import synthesis_splat_jit


def _minimum_error_seam(error: np.ndarray, vertical: bool) -> np.ndarray:
    """Return a minimum-cost seam through a patch-overlap error image."""
    work = error if vertical else error.T
    height, width = work.shape
    cost = work.astype(np.float32, copy=True)
    parent = np.zeros((height, width), dtype=np.int16)
    for row in range(1, height):
        previous = cost[row - 1]
        for col in range(width):
            start = max(0, col - 1)
            end = min(width, col + 2)
            offset = int(np.argmin(previous[start:end]))
            parent[row, col] = start + offset
            cost[row, col] += previous[start + offset]

    seam = np.empty(height, dtype=np.int32)
    seam[-1] = int(np.argmin(cost[-1]))
    for row in range(height - 1, 0, -1):
        seam[row - 1] = parent[row, seam[row]]
    return seam


def _quilt_splats(patches: np.ndarray, target_h: int, target_w: int,
                  falloff: float, seed: int = 42) -> np.ndarray:
    """Synthesize a tile from fixed-scale source patches using cut seams.

    This is image quilting rather than repeated alpha compositing.  It avoids
    averaging independent source details, which is especially important for
    high-frequency materials such as pebbles, bark, and gravel.
    """
    patch_h, patch_w, channels = patches.shape[1:]
    overlap_x = min(patch_w - 1, max(2, int(round(patch_w * max(0.08, falloff)))))
    overlap_y = min(patch_h - 1, max(2, int(round(patch_h * max(0.08, falloff)))))
    stride_x = max(1, patch_w - overlap_x)
    stride_y = max(1, patch_h - overlap_y)

    def positions(length: int, patch_size: int, stride: int) -> list[int]:
        if patch_size >= length:
            return [0]
        values = list(range(0, length - patch_size + 1, stride))
        last = length - patch_size
        if values[-1] != last:
            values.append(last)
        return values

    xs = positions(target_w, patch_w, stride_x)
    ys = positions(target_h, patch_h, stride_y)
    canvas = np.zeros((target_h, target_w, channels), dtype=np.float32)
    filled = np.zeros((target_h, target_w), dtype=bool)
    rng = np.random.RandomState(seed)

    for y in ys:
        for x in xs:
            y2 = min(target_h, y + patch_h)
            x2 = min(target_w, x + patch_w)
            region_h, region_w = y2 - y, x2 - x
            existing = canvas[y:y2, x:x2]
            occupied = filled[y:y2, x:x2]

            # Choose the crop that best agrees with the already placed
            # neighbours.  A small candidate set gives substantial quality
            # improvement without making live previews sluggish.
            candidate_ids = rng.randint(0, patches.shape[0], size=min(6, patches.shape[0]))
            best_index = int(candidate_ids[0])
            best_error = np.inf
            for candidate in candidate_ids:
                candidate_patch = patches[candidate, :region_h, :region_w]
                if not occupied.any():
                    score = 0.0
                else:
                    diff = candidate_patch - existing
                    score = float(np.mean(diff[occupied] * diff[occupied]))
                if score < best_error:
                    best_error = score
                    best_index = int(candidate)

            patch = patches[best_index, :region_h, :region_w]
            use_patch = ~occupied

            if occupied.any():
                # The overlap is rectangular for the row-major grid.  Find
                # an inexpensive path through each shared edge, retaining
                # existing pixels on one side and the new crop on the other.
                left_overlap = 0
                if x > 0:
                    for col in range(region_w):
                        if np.all(occupied[:, col]):
                            left_overlap += 1
                        else:
                            break
                if left_overlap > 0:
                    error = np.mean((patch[:, :left_overlap] - existing[:, :left_overlap]) ** 2, axis=2)
                    seam = _minimum_error_seam(error, vertical=True)
                    for row, col in enumerate(seam):
                        use_patch[row, :col] = False

                top_overlap = 0
                if y > 0:
                    for row in range(region_h):
                        if np.all(occupied[row, :]):
                            top_overlap += 1
                        else:
                            break
                if top_overlap > 0:
                    error = np.mean((patch[:top_overlap] - existing[:top_overlap]) ** 2, axis=2)
                    seam = _minimum_error_seam(error, vertical=False)
                    for col, row in enumerate(seam):
                        use_patch[:row, col] = False

                # Feather only around the selected cut.  Falloff widens the
                # search/feather region, never the whole patch interior.
                feather = max(1, int(round(max(left_overlap, top_overlap) * falloff)))
                alpha = cv2.GaussianBlur(
                    use_patch.astype(np.float32), (0, 0), sigmaX=feather,
                    sigmaY=feather, borderType=cv2.BORDER_REPLICATE,
                )
                alpha[~occupied] = 1.0
                canvas[y:y2, x:x2] = existing * (1.0 - alpha[:, :, np.newaxis]) + patch * alpha[:, :, np.newaxis]
            else:
                canvas[y:y2, x:x2] = patch
            filled[y:y2, x:x2] = True

    return canvas


def _materialize_splat_kernel(width: int, height: int) -> list[tuple[float, float, float]]:
    """Use the same aspect-ratio-dependent kernel layouts as Materialize."""
    if width == height:
        return [(0.0, 0.25, 0.8), (0.5, 0.25, 0.8), (0.25, 0.75, 0.8), (0.75, 0.75, 0.8)]
    if width > height:
        return [
            (0.0, 0.375, 0.25), (0.125, 0.625, 0.25),
            (0.25, 0.375, 0.25), (0.375, 0.625, 0.25),
            (0.5, 0.375, 0.25), (0.625, 0.625, 0.25),
            (0.75, 0.375, 0.25), (0.875, 0.625, 0.25),
        ]
    return [
        (0.375, 0.0, 0.25), (0.625, 0.125, 0.25),
        (0.375, 0.25, 0.25), (0.625, 0.375, 0.25),
        (0.375, 0.5, 0.25), (0.625, 0.625, 0.25),
        (0.375, 0.75, 0.25), (0.625, 0.875, 0.25),
    ]


def _smoothstep(low: float, high: float, value: np.ndarray) -> np.ndarray:
    t = np.clip((value - low) / max(1e-6, high - low), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _materialize_splat(image: np.ndarray, target_h: int, target_w: int,
                       scale: float, rotation: float, rand_rot: float,
                       wobble: float, falloff: float) -> np.ndarray:
    """CPU port of Materialize's height-priority splat shader.

    Its key property is that it selects a dominant textured sample at each
    point, instead of averaging multiple full texture copies.  That preserves
    sharp high-frequency detail (such as individual pebbles) while its
    periodic coordinates make the output repeat at every edge.
    """
    source = image.astype(np.float32, copy=False)
    source_3d = source[:, :, np.newaxis] if source.ndim == 2 else source
    src_h, src_w, channels = source_3d.shape
    yy, xx = np.mgrid[0:target_h, 0:target_w].astype(np.float32)
    uv_x = (xx + 0.5) / target_w
    uv_y = (yy + 0.5) / target_h
    height_map = cv2.cvtColor(source_3d[:, :, :3], cv2.COLOR_BGR2GRAY) if channels >= 3 else source_3d[:, :, 0]
    if height_map.max() > 1.0:
        height_map = height_map / 255.0

    output = np.zeros((target_h, target_w, channels), dtype=np.float32)
    depth = np.zeros((target_h, target_w), dtype=np.float32)
    offsets = ((1, 1), (0, 1), (-1, 1), (1, 0), (0, 0), (-1, 0), (1, -1), (0, -1), (-1, -1))
    aspect = min(target_w / target_h, target_h / target_w)
    base_rotation = np.deg2rad(rotation)
    low = -0.01 - 0.5 * falloff
    high = 0.01 + 0.5 * falloff

    for index, (center_x, center_y, kernel_size) in enumerate(_materialize_splat_kernel(target_w, target_h)):
        wobble_x = np.sin((index + 1.0) * 128.352) * wobble
        wobble_y = np.cos((index + 1.0) * 243.767) * wobble
        random_rotation = np.sin((index + 1.0) * 472.361) * rand_rot * (2.0 * np.pi)
        angle = base_rotation + random_rotation
        cos_angle, sin_angle = np.cos(angle), np.sin(angle)

        for offset_x, offset_y in offsets:
            local_x = (uv_x - center_x + offset_x) * (aspect / max(1e-6, scale * kernel_size))
            local_y = (uv_y - center_y + offset_y) * (aspect / max(1e-6, scale * kernel_size))
            rotated_x = cos_angle * local_x - sin_angle * local_y
            rotated_y = sin_angle * local_x + cos_angle * local_y
            mask_x = rotated_x * 2.0
            mask_y = rotated_y * 2.0
            edge_x = 1.0 - np.clip(np.abs(mask_x), 0.0, 1.0)
            edge_y = 1.0 - np.clip(np.abs(mask_y), 0.0, 1.0)
            center_mask = np.power(np.clip((edge_x * edge_y - 0.1) * 2.0, 0.0, 1.0), 0.3)
            uv_mask = np.clip(edge_x * edge_y * 10.0, 0.0, 1.0)
            splat_height_mask = center_mask * uv_mask
            if not np.any(splat_height_mask > 0.0):
                continue

            sample_u = (rotated_x / (1.0 + wobble)) + wobble_x + 0.5
            sample_v = (rotated_y / (1.0 + wobble)) + wobble_y + 0.5
            map_x = (np.mod(sample_u, 1.0) * (src_w - 1)).astype(np.float32)
            map_y = (np.mod(sample_v, 1.0) * (src_h - 1)).astype(np.float32)
            sampled = cv2.remap(source_3d, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            sampled_height = cv2.remap(height_map, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            this_depth = (sampled_height + 0.2) * splat_height_mask
            keep_existing = _smoothstep(low, high, depth - this_depth)
            output = sampled * (1.0 - keep_existing[:, :, np.newaxis]) + output * keep_existing[:, :, np.newaxis]
            depth = np.maximum(depth, this_depth)

    # Materialize's kernel covers the frame, but retain a periodic source
    # fallback for degenerate dimensions or extremely small scale values.
    fallback_x = (uv_x * (src_w - 1)).astype(np.float32)
    fallback_y = (uv_y * (src_h - 1)).astype(np.float32)
    fallback = cv2.remap(source_3d, fallback_x, fallback_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    output[depth <= 1e-6] = fallback[depth <= 1e-6]
    return output[:, :, 0] if image.ndim == 2 else output


def synthesis_splat(image: np.ndarray, new_size: tuple = (1024, 1024),
                    grid_size: int = 8, scale: float = 1.0,
                    rotation: float = 0, rand_rot: float = 0,
                    wobble: float = 0.2, falloff: float = 0.2,
                    cached_batches=None):
    """Create seamless texture using splatting (Texture Bombing).

    Accepts float32 input and returns float32 output.

    Args:
        cached_batches: Tuple (patches_arr, masks_arr) or None
    Returns:
        (result_image, (patches_arr, masks_arr))
    """
    assert_float32(image, "synthesis_splat image")
    target_h, target_w = new_size
    h, w = image.shape[:2]
    falloff = float(np.clip(falloff, 0.0, 1.0))

    # A feathered, axis-aligned splat must preserve the source patch's scale
    # and orientation.  Rotating a rectangular patch inside its fixed bounds
    # clips its corners, which makes identical patches appear to have random
    # sizes and leaves hard-looking transitions.  Rotation remains available
    # for hard-edged splats only.
    if falloff > 0.001:
        rotation = 0.0
        rand_rot = 0.0

    img_f = image.copy()
    if img_f.ndim == 2:
        img_f = img_f[:, :, np.newaxis]
    h_src, w_src = img_f.shape[:2]
    # Create a 2x2 tile of the source, then crop to target size with a quarter offset
    tiled_2x = np.tile(img_f, (2, 2, 1))
    # Quarter-offset so the "seam" of the tile is in the middle, not at the corners
    qx = w_src // 4
    qy = h_src // 4
    # Crop target_h x target_w starting at the quarter offset (wrapping via tile)
    crop_y = qy % h_src
    crop_x = qx % w_src
    # The 2x tile is always large enough for a quarter-offset crop
    canvas = tiled_2x[crop_y:crop_y + target_h, crop_x:crop_x + target_w].copy()
    if canvas.shape[0] < target_h or canvas.shape[1] < target_w:
        # Fallback: repeat-pad if crop exceeded the 2x tile
        canvas = np.tile(img_f, (
            (target_h // h_src) + 2,
            (target_w // w_src) + 2,
            1
        ))[:target_h, :target_w].copy()
    # Restore original channel count for grayscale
    if len(image.shape) == 2:
        canvas = canvas[:, :, 0]

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

        # Cap variations for performance.  Variations are source crops, not
        # random scales: every splat keeps the exact same output dimensions.
        max_variations = 4 if is_preview else 16
        num_variations = max_variations

        # A splat is a sampled portion of the source, not the complete image
        # repeatedly resized and layered over itself.  Repeated full-image
        # splats average unrelated stones at every pixel and destroy detail.
        # Scale changes every patch by the same factor while source sampling
        # keeps the material's local structure intact.
        source_patch_w = max(16, int(round(w * 0.20)))
        source_patch_h = max(16, int(round(h * 0.20)))
        target_patch_w = max(16, int(round(source_patch_w * scale)))
        target_patch_h = max(16, int(round(source_patch_h * scale)))

        source_tiled = np.tile(image, (2, 2, 1)) if image.ndim == 3 else np.tile(image, (2, 2))
        source_rng = np.random.RandomState(42)
        target_patch_w = max(4, target_patch_w)
        target_patch_h = max(4, target_patch_h)

        h_small, w_small = target_patch_h, target_patch_w

        patches = []
        masks = []

        # Feather every side of the rectangular patch.  A square distance
        # field avoids uncovered corner regions and keeps the same scale
        # across every splat.
        base_mask = create_falloff_mask((h_small, w_small), falloff=falloff, circular=False)

        # Masks are shared by all same-scale source crops.
        base_mask = base_mask[:, :, np.newaxis]

        # Generate same-scale source-crop variations.  Edge falloff is
        # strictly an alpha-mask operation and must not introduce any random
        # transform.
        for i in range(num_variations):
            crop_x = source_rng.randint(0, w)
            crop_y = source_rng.randint(0, h)
            crop = source_tiled[
                crop_y:crop_y + source_patch_h,
                crop_x:crop_x + source_patch_w,
            ]
            base_patch = cv2.resize(
                crop, (target_patch_w, target_patch_h),
                interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA,
            )
            if base_patch.ndim == 2:
                base_patch = base_patch[:, :, np.newaxis]
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

            patches.append(p)
            masks.append(m)

        patches_arr = np.array(patches)
        masks_arr = np.array(masks)

    # 3. Quilt fixed-scale source crops with minimum-error boundaries.  Wobble
    # changes the deterministic crop-selection sequence without breaking the
    # carefully covered patch grid.
    patches_arr = np.ascontiguousarray(patches_arr, dtype=np.float32)
    masks_arr = np.ascontiguousarray(masks_arr, dtype=np.float32)
    result = _quilt_splats(
        patches_arr,
        target_h,
        target_w,
        falloff=falloff,
        seed=42 + int(round(np.clip(wobble, 0.0, 1.0) * 10_000)),
    )
    if image.ndim == 2:
        result = result[:, :, 0]

    result = _blend_periodic_edges(
        result,
        width_x=max(1, int(target_w * 0.02)),
        width_y=max(1, int(target_h * 0.02)),
        falloff=falloff,
    )
    return result, (patches_arr, masks_arr)
