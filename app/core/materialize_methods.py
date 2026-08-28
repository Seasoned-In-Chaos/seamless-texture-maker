"""
Seamless texture generation using Materialize-inspired techniques.
Includes Overlap and Splat methods.

All functions accept and return float32 arrays.
"""
from __future__ import annotations

import numba
import numpy as np
import cv2
from .assertions import assert_float32
from ..utils.app_logging import get_logger

logger = get_logger(__name__)


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


# Fraction of the overlap strip the transition may span at falloff == 1.0.
# Letting the transition cover the *whole* strip (what a plain linear ramp
# does) leaves a wide band sitting near a 50/50 mix, which visibly
# double-exposes strong non-repeating features -- a face, a logo. Capping it
# keeps the join soft without the two sides ghosting over each other.
_MAX_FADE_FRACTION = 0.5


def _seam_fade_ramp(blend_n: int, falloff: float) -> np.ndarray:
    """Cross-fade weights across an overlap strip of `blend_n` pixels.

    Weight 1.0 means "take the opposite edge", 0.0 means "keep the original
    content here". The strip starts fully at 1.0 -- that leading run of
    copied pixels is the actual overlap, and it is what makes the tile wrap
    -- then switches to 0.0 around the middle of the strip.

    `falloff` widens the transition around that midpoint: 0 gives a hard
    step (half the strip is a solid copy of the opposite edge, exactly like
    a plain cut), and 1.0 eases the switch across the widest permitted span.
    """
    p = (np.arange(blend_n, dtype=np.float32) + 0.5) / blend_n
    half_width = 0.5 * falloff * _MAX_FADE_FRACTION

    if half_width <= 1e-6:
        return (p < 0.5).astype(np.float32)

    ramp = np.clip((0.5 + half_width - p) / (2.0 * half_width), 0.0, 1.0)
    # Smoothstep twice: steeper through the middle, so the near-50/50 zone
    # stays narrow while the ends still ease in and out without a hard line.
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)

    return ramp.astype(np.float32)


def _resize_wrapped(image: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize a tile to (w, h) while preserving its wrap continuity.

    The cropped tile is exactly seamless -- its last column continues into
    its first. A plain ``cv2.resize`` resamples with clamped borders, so
    edge pixels are interpolated against a repeat of themselves instead of
    against the content they wrap into. That leaves a faint 1px line along
    the tile boundary once tiled. Sampling with BORDER_WRAP instead reads
    across the wrap, keeping the seam as continuous as the source content.
    """
    src_h, src_w = image.shape[:2]
    if (src_w, src_h) == (w, h):
        return image

    map_x = ((np.arange(w, dtype=np.float32) + 0.5) * (src_w / float(w)) - 0.5)
    map_y = ((np.arange(h, dtype=np.float32) + 0.5) * (src_h / float(h)) - 0.5)
    map_x, map_y = np.meshgrid(map_x, map_y)

    return cv2.remap(image, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_WRAP)


def synthesis_overlap(image: np.ndarray, overlap_x: float = 0.2,
                      overlap_y: float = 0.2,
                      falloff: float = 0.5) -> np.ndarray:
    """Create seamless texture using tile overlap method with resizing.

    Overlaps the far edge onto the near edge, joins the two, crops the
    consumed strip and resizes back to the original dimensions.
    `overlap_x`/`overlap_y` set how much is overlapped -- they always change
    the image, so the preview responds to them at any falloff.
    `falloff` only controls the softness of the join: 0 leaves a hard seam,
    1.0 gives the widest permitted gradient.

    Accepts and returns float32 arrays.
    """
    assert_float32(image, "synthesis_overlap image")
    h, w = image.shape[:2]

    img_f = image.copy()

    if overlap_x > 0:
        blend_w = int(w * overlap_x)
        if blend_w > 0:
            t = _seam_fade_ramp(blend_w, falloff)[np.newaxis, :]
            if image.ndim == 3:
                t = t[:, :, np.newaxis]

            left_strip = img_f[:, 0:blend_w]
            right_strip = img_f[:, w - blend_w:w]

            img_f[:, 0:blend_w] = left_strip * (1.0 - t) + right_strip * t
            img_f = img_f[:, 0:w - blend_w]

    h_curr = img_f.shape[0]

    if overlap_y > 0:
        blend_h = int(h_curr * overlap_y)
        if blend_h > 0:
            t = _seam_fade_ramp(blend_h, falloff)[:, np.newaxis]
            if image.ndim == 3:
                t = t[:, :, np.newaxis]

            top_strip = img_f[0:blend_h, :]
            bottom_strip = img_f[h_curr - blend_h:h_curr, :]

            img_f[0:blend_h, :] = top_strip * (1.0 - t) + bottom_strip * t
            img_f = img_f[0:h_curr - blend_h, :]

    result = _resize_wrapped(img_f, w, h)

    return np.clip(result, 0, 255)


from . import materialize_methods_cuda
from .materialize_methods_jit import (
    splat_accumulate_jit,
    splat_accumulate_parallel_jit,
    splat_resolve_jit,
)


# Patch bank memory ceiling. Rotated copies are pre-rendered once and reused
# across every splat, so the bank is the only sizeable allocation here; this
# keeps a large scale + high random rotation from exhausting memory.
_PATCH_BANK_BUDGET_BYTES = 64 * 1024 * 1024

# Grid spacing as a fraction of patch size. Below 0.5 every point of the
# canvas falls inside several patches, which is what lets the weighted
# average blend them and guarantees no uncovered pixels.
_SPLAT_SPACING = 0.40

# Jitter as a fraction of a cell. Full-cell jitter hides the grid best but
# lets neighbours drift apart far enough to leave holes, so it stops short.
_SPLAT_JITTER = 0.4


def create_splat_mask(shape, falloff: float = 0.3, wobble: float = 0.0,
                      rng=None) -> np.ndarray:
    """Alpha mask for one splat: a soft-edged blob inscribed in the patch.

    `falloff` sets how far the edge feathers inward -- 0 is a crisp disc,
    1.0 fades all the way from the centre. `wobble` deforms the outline with
    a few angular harmonics at random phases, so patches read as irregular
    natural shapes instead of repeated circles. Pass a seeded `rng` to vary
    the deformation per patch variation.
    """
    h, w = shape
    y, x = np.ogrid[:h, :w]
    ny = (y - (h - 1) / 2.0) / max(1e-6, h / 2.0)
    nx = (x - (w - 1) / 2.0) / max(1e-6, w / 2.0)

    radius = np.sqrt(nx * nx + ny * ny)

    if wobble > 0.001 and rng is not None:
        theta = np.arctan2(ny, nx)
        harmonics = ((2, 1.0), (3, 0.6), (5, 0.35), (7, 0.2))
        deform = np.zeros_like(radius)
        for k, amp in harmonics:
            deform += amp * np.sin(k * theta + rng.uniform(0.0, 2.0 * np.pi))
        deform /= sum(amp for _, amp in harmonics)
        # Deliberately lopsided: the outline may bulge outward freely but
        # only dent inward a little, so a strong wobble cannot shrink the
        # blob far enough to leave uncovered gaps between neighbours.
        radius = radius * (1.0 + np.clip(wobble * 0.6 * deform, -0.15, 0.5))

    # Distance inward from the blob outline: 1 at the centre, 0 at the edge.
    depth = np.clip(1.0 - radius, 0.0, 1.0)

    feather = max(0.02, float(falloff))
    coverage = np.clip(depth / feather, 0.0, 1.0)
    coverage = coverage * coverage * (3.0 - 2.0 * coverage)  # smoothstep

    # Patches are combined as a weighted average, so a flat-topped mask makes
    # every overlapping patch count equally and averages the detail away.
    # Biasing the weight toward the patch centre lets the nearest patch
    # dominate instead, which preserves contrast; `falloff` sets how sharply.
    dominance = 8.0 * (1.0 - float(falloff)) + 0.5 * float(falloff)
    alpha = coverage * np.power(depth, dominance)

    return alpha.astype(np.float32)


def _splat_variation_counts(image: np.ndarray, scale: float,
                            rand_rot: float, preview: bool):
    """Return (stored, desired) rotated-patch variation counts.

    ``stored`` respects the patch-bank memory ceiling.  ``desired`` is the
    number of orientations the synthesis should use; callers can stream those
    one at a time when retaining them all would exceed the ceiling.
    """
    if rand_rot < 0.01:
        return 1, 1

    h, w = image.shape[:2]
    patch_w = max(8, int(round(w * scale)))
    patch_h = max(8, int(round(h * scale)))
    channels = 1 if image.ndim == 2 else image.shape[2]
    patch_bytes = patch_h * patch_w * channels * 4
    affordable = int(_PATCH_BANK_BUDGET_BYTES // max(1, patch_bytes))
    desired = 12 if preview else 24
    return max(1, min(desired, affordable)), desired


def _build_patch_bank(image: np.ndarray, scale: float, rotation: float,
                      rand_rot: float, wobble: float, falloff: float,
                      seed: int, preview: bool, num_variations: int | None = None):
    """Pre-render the rotated patch copies and their masks.

    Returns (patches, masks) with shapes (N, ph, pw, C) and (N, ph, pw).
    """
    h, w = image.shape[:2]

    patch_w = max(8, int(round(w * scale)))
    patch_h = max(8, int(round(h * scale)))

    base_patch = cv2.resize(image, (patch_w, patch_h),
                            interpolation=cv2.INTER_AREA)
    if base_patch.ndim == 2:
        base_patch = base_patch[:, :, np.newaxis]
    patch_h, patch_w = base_patch.shape[:2]
    channels = base_patch.shape[2]

    if num_variations is None:
        # Wobble affects the alpha mask, not the pixels in a patch. Without
        # random rotation every pixel patch is identical, so keeping several
        # full-resolution copies wastes a prohibitive amount of memory.
        num_variations, _ = _splat_variation_counts(
            image, scale, rand_rot, preview)

    rng = np.random.RandomState(np.uint32(seed) & 0x7FFFFFFF)

    patches = np.empty((num_variations, patch_h, patch_w, channels),
                       dtype=np.float32)
    masks = np.empty((num_variations, patch_h, patch_w), dtype=np.float32)

    for i in range(num_variations):
        angle = rotation
        if rand_rot >= 0.01:
            # Apply a sampled deviation even when the memory budget permits
            # only one patch.  Previously that fallback skipped this branch,
            # so large images could clamp the bank to one variation and make
            # the Random Rotation control look completely inert.  The seed
            # keeps the single-patch fallback deterministic while the slider
            # still changes its appearance.
            angle += rng.uniform(-180.0, 180.0) * rand_rot

        if abs(angle) > 0.05:
            matrix = cv2.getRotationMatrix2D(
                (patch_w / 2.0, patch_h / 2.0), angle, 1.0)
            rotated = cv2.warpAffine(base_patch, matrix, (patch_w, patch_h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT_101)
            if rotated.ndim == 2:
                rotated = rotated[:, :, np.newaxis]
        else:
            rotated = base_patch

        patches[i] = rotated
        # The mask is generated unrotated -- its own random harmonics supply
        # the shape variation, and skipping a warp avoids resampling the
        # alpha edge (which would leave faint hard rims on every patch).
        masks[i] = create_splat_mask((patch_h, patch_w), falloff=falloff,
                                     wobble=wobble, rng=rng)

    return patches, masks


def _splat_placements(target_h: int, target_w: int, patch_h: int, patch_w: int,
                      num_variations: int, seed: int):
    """Jittered-grid placements covering the canvas, plus patch indices.

    Cells partition the canvas exactly and every coordinate is later taken
    modulo the canvas size, so the layout itself is periodic and cannot
    introduce an edge seam.
    """
    cells_x = max(1, int(np.ceil(target_w / max(1.0, patch_w * _SPLAT_SPACING))))
    cells_y = max(1, int(np.ceil(target_h / max(1.0, patch_h * _SPLAT_SPACING))))

    cell_w = target_w / cells_x
    cell_h = target_h / cells_y

    gx, gy = np.meshgrid(np.arange(cells_x), np.arange(cells_y))
    gx = gx.ravel().astype(np.float32)
    gy = gy.ravel().astype(np.float32)
    count = gx.size

    rng = np.random.RandomState((np.uint32(seed) + 7919) & 0x7FFFFFFF)

    jitter_x = (rng.rand(count).astype(np.float32) - 0.5) * cell_w * _SPLAT_JITTER
    jitter_y = (rng.rand(count).astype(np.float32) - 0.5) * cell_h * _SPLAT_JITTER

    centre_x = (gx + 0.5) * cell_w + jitter_x
    centre_y = (gy + 0.5) * cell_h + jitter_y

    left = np.round(centre_x - patch_w / 2.0).astype(np.int32)
    top = np.round(centre_y - patch_h / 2.0).astype(np.int32)

    coords = np.column_stack((top, left)).astype(np.int32)
    indices = rng.randint(0, num_variations, size=count).astype(np.int32)

    # Shuffling decouples patch choice from grid position, so no diagonal
    # banding shows up when the variation count divides the row length.
    order = rng.permutation(count)
    return np.ascontiguousarray(coords[order]), np.ascontiguousarray(indices[order])


def _stream_patch_batches(source, variation_count, coords, indices, scale, rotation,
                          rand_rot, wobble, falloff, seed, preview):
    """Yield one (selected_coords, local_indices, patch_batch, mask_batch)
    tuple per non-empty patch variation, shared by the GPU and CPU
    streamed-accumulation branches of synthesis_splat so they build
    identical per-variation batches from a single code path."""
    for variation in range(variation_count):
        selected = np.ascontiguousarray(coords[indices == variation])
        if not len(selected):
            continue
        variation_seed = int(seed) + variation * 7919
        patch_batch, mask_batch = _build_patch_bank(
            source, scale=scale, rotation=rotation, rand_rot=rand_rot,
            wobble=wobble, falloff=falloff, seed=variation_seed,
            preview=preview, num_variations=1)
        local_indices = np.zeros(len(selected), dtype=np.int32)
        yield selected, local_indices, patch_batch, mask_batch


def synthesis_splat(image: np.ndarray, new_size: tuple = (1024, 1024),
                    grid_size: int = 8, scale: float = 1.0,
                    rotation: float = 0, rand_rot: float = 0,
                    wobble: float = 0.2, falloff: float = 0.2,
                    cached_batches=None, seed: int = 0):
    """Create a seamless texture by splatting patches (texture bombing).

    Patches cut from the source are scattered over a jittered grid, each
    rotated and shaped independently, and combined as a weighted average.
    Placements wrap at the canvas edges, so the result tiles seamlessly.

    Args:
        new_size: (height, width) of the output canvas.
        scale: Patch size as a fraction of the source image.
        rotation: Base rotation applied to every patch, in degrees.
        rand_rot: 0-1, how much each patch deviates from `rotation`.
        wobble: 0-1, how strongly patch outlines are deformed.
        falloff: 0-1, how far the patch edges feather inward.
        cached_batches: (patches, masks) from a previous call, or None.
        seed: Varies the random placement and patch shapes.

    Returns:
        (result_image, (patches, masks))
    """
    assert_float32(image, "synthesis_splat image")
    target_h, target_w = new_size

    grayscale = image.ndim == 2
    source = image[:, :, np.newaxis] if grayscale else image

    # Base canvas, only ever seen where no patch lands (see splat_resolve_jit).
    src_h, src_w = source.shape[:2]
    if (src_h, src_w) == (target_h, target_w):
        # This is the normal SeamlessProcessor path. Reusing the source
        # avoids a second 768MB float32 canvas at 8192x8192.
        fallback = source
    else:
        reps_y = target_h // src_h + 2
        reps_x = target_w // src_w + 2
        fallback = np.ascontiguousarray(
            np.tile(source, (reps_y, reps_x, 1))[:target_h, :target_w])

    preview = target_h <= 384 and target_w <= 384

    stored_variations, desired_variations = _splat_variation_counts(
        source, scale, rand_rot, preview)
    stream_variations = cached_batches is None and desired_variations > stored_variations

    if cached_batches is not None:
        patches, masks = cached_batches
        variation_count = patches.shape[0]
    elif stream_variations:
        patches = masks = None
        variation_count = desired_variations
    else:
        patches, masks = _build_patch_bank(
            source, scale=scale, rotation=rotation, rand_rot=rand_rot,
            wobble=wobble, falloff=falloff, seed=seed, preview=preview)
        variation_count = patches.shape[0]

    if patches is None:
        patch_w = max(8, int(round(source.shape[1] * scale)))
        patch_h = max(8, int(round(source.shape[0] * scale)))
    else:
        patch_h, patch_w = patches.shape[1:3]
    coords, indices = _splat_placements(
        target_h, target_w, patch_h, patch_w, variation_count, seed)

    channels = source.shape[2]
    num_splats = coords.shape[0]

    # GPU (NVIDIA, via Numba CUDA) is only worth dispatching to above a
    # work-volume threshold and never for live-preview canvases -- see
    # materialize_methods_cuda.gpu_eligible. Any failure past this point
    # (including mid-render) falls back to the CPU path below and marks the
    # GPU broken for the rest of the process, same defensive style as
    # gpu_utils.GPUAccelerator.
    use_gpu = materialize_methods_cuda.gpu_eligible(
        num_splats=num_splats, patch_h=patch_h, patch_w=patch_w, preview=preview)

    if use_gpu:
        try:
            session = materialize_methods_cuda.SplatCudaSession(target_h, target_w, channels)
            if stream_variations:
                for selected, local_indices, patch_batch, mask_batch in _stream_patch_batches(
                        source, variation_count, coords, indices, scale, rotation,
                        rand_rot, wobble, falloff, seed, preview):
                    session.accumulate(patch_batch, mask_batch, selected, local_indices)
            else:
                session.accumulate(np.ascontiguousarray(patches),
                                   np.ascontiguousarray(masks), coords, indices)
            result = session.resolve(np.ascontiguousarray(fallback), np.empty_like(fallback))
        except Exception as exc:
            logger.debug("Splat GPU accumulate failed, falling back to CPU: %s", exc)
            materialize_methods_cuda.mark_broken()
            use_gpu = False

    if not use_gpu:
        accum = np.zeros((target_h, target_w, channels), dtype=np.float32)
        weight = np.zeros((target_h, target_w), dtype=np.float32)
        num_bands = max(1, min(target_h, numba.get_num_threads()))

        if stream_variations:
            # Full-resolution photographs can have 40+ MB patches, so storing
            # all 24 random rotations would be needlessly expensive. Generate
            # and accumulate one variation at a time instead. This makes the
            # settled full-quality render match the randomized live preview
            # without ever holding a large patch bank in memory.
            for selected, local_indices, patch_batch, mask_batch in _stream_patch_batches(
                    source, variation_count, coords, indices, scale, rotation,
                    rand_rot, wobble, falloff, seed, preview):
                splat_accumulate_parallel_jit(accum, weight, patch_batch, mask_batch,
                                              selected, local_indices, num_bands)
        else:
            splat_accumulate_parallel_jit(accum, weight,
                                          np.ascontiguousarray(patches),
                                          np.ascontiguousarray(masks),
                                          coords, indices, num_bands)

        # Resolving pixel-by-pixel is safe in place: each output location only
        # reads its own accumulator value. This removes another full-size float
        # canvas (768MB for an 8K RGB texture).
        splat_resolve_jit(accum, weight, fallback, accum)
        result = accum

    if grayscale:
        result = result[:, :, 0]

    np.clip(result, 0, 255, out=result)
    return result, None if stream_variations else (patches, masks)
