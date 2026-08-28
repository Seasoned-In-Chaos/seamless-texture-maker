"""
JIT-compiled Materialize-inspired synthesis methods (Splat, Overlap).
Optimized for real-time performance with proper wrapping support.
"""
from __future__ import annotations

import numpy as np
from numba import jit, prange


@jit(nopython=True, fastmath=True, cache=True)
def splat_accumulate_jit(accum, weight, patches, masks, coords, indices):
    """Accumulate weighted patches onto a canvas, wrapping at the edges.

    Rather than compositing patches one over another, this sums
    ``patch * alpha`` into `accum` and ``alpha`` into `weight`. Dividing the
    two afterwards gives a weighted average of every patch covering a pixel.
    That keeps the result independent of splat ordering and, because every
    write wraps modulo the canvas size, makes the output exactly periodic --
    i.e. seamlessly tileable by construction.

    Args:
        accum: (H, W, C) float32 accumulator, zeroed by the caller.
        weight: (H, W) float32 alpha accumulator, zeroed by the caller.
        patches: (N, ph, pw, C) float32 patch bank.
        masks: (N, ph, pw) float32 alpha masks.
        coords: (num_splats, 2) int32 (top, left) placements.
        indices: (num_splats,) int32 index into the patch bank.
    """
    num_splats = coords.shape[0]
    ph = patches.shape[1]
    pw = patches.shape[2]
    channels = accum.shape[2]
    height = accum.shape[0]
    width = accum.shape[1]

    for i in range(num_splats):
        top = coords[i, 0]
        left = coords[i, 1]
        pidx = indices[i]

        for py in range(ph):
            y = (top + py) % height
            for px in range(pw):
                alpha = masks[pidx, py, px]
                # Centre-biased masks make legitimate weights very small away
                # from the patch centre, so this only skips exact zeros --
                # a coarser cutoff would punch holes in the coverage.
                if alpha <= 1e-7:
                    continue
                x = (left + px) % width
                weight[y, x] += alpha
                for c in range(channels):
                    accum[y, x, c] += patches[pidx, py, px, c] * alpha

    return accum, weight


@jit(nopython=True, fastmath=True, cache=True, parallel=True)
def splat_accumulate_parallel_jit(accum, weight, patches, masks, coords, indices, num_bands):
    """Race-free parallel version of ``splat_accumulate_jit``.

    Patches placed by different splats routinely overlap in output space (the
    weighted-average blend depends on it), so a naive `prange` over splats
    would race on `accum`/`weight`. Instead this partitions the *canvas* into
    `num_bands` disjoint row bands: every band scans all splats but only
    writes a pixel whose wrapped `y = (top+py) % height` falls inside its own
    band. Two bands can never touch the same `(y, x)`, so this is race-free
    regardless of wraparound (a splat near the bottom/top edge simply gets
    part of its write accepted by each of the bands its wrapped rows land
    in). Per-thread private accumulator copies were considered and rejected:
    at the app's 8192px cap, one extra `accum` copy is already ~768MB, so N
    thread-local copies plus a reduction pass would cost gigabytes instead of
    the zero extra memory this needs.
    """
    num_splats = coords.shape[0]
    ph = patches.shape[1]
    pw = patches.shape[2]
    channels = accum.shape[2]
    height = accum.shape[0]
    width = accum.shape[1]

    band_height = (height + num_bands - 1) // num_bands

    for band in prange(num_bands):
        row_start = band * band_height
        row_end = min(row_start + band_height, height)
        if row_start >= row_end:
            continue

        for i in range(num_splats):
            top = coords[i, 0]
            left = coords[i, 1]
            pidx = indices[i]

            for py in range(ph):
                y = (top + py) % height
                if y < row_start or y >= row_end:
                    continue
                for px in range(pw):
                    alpha = masks[pidx, py, px]
                    if alpha <= 1e-7:
                        continue
                    x = (left + px) % width
                    weight[y, x] += alpha
                    for c in range(channels):
                        accum[y, x, c] += patches[pidx, py, px, c] * alpha

    return accum, weight


@jit(nopython=True, fastmath=True, cache=True, parallel=True)
def splat_resolve_jit(accum, weight, fallback, out):
    """Divide accumulated colour by accumulated alpha.

    Pixels no patch reached (weight ~ 0) fall back to the base canvas so the
    result never contains holes, even with extreme wobble or falloff. Each
    output pixel only ever reads/writes its own `(y, x)`, so this is safe to
    parallelize over rows -- including in-place, where `out is accum`.
    """
    height = accum.shape[0]
    width = accum.shape[1]
    channels = accum.shape[2]

    for y in prange(height):
        for x in range(width):
            w = weight[y, x]
            if w > 1e-9:
                inv = 1.0 / w
                for c in range(channels):
                    out[y, x, c] = accum[y, x, c] * inv
            else:
                for c in range(channels):
                    out[y, x, c] = fallback[y, x, c]

    return out
