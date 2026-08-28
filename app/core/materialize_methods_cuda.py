"""
CUDA-accelerated Splat synthesis (NVIDIA GPU path).

Numba's @cuda.jit target needs NVVM (the CUDA compiler), which is only
present when the `nvidia-cuda-nvcc-cu12` package (or a system CUDA Toolkit)
is installed -- most machines, including every AMD/Intel GPU or no-GPU one,
will never have it. Every entry point here is safe to call unconditionally:
`gpu_eligible()` returns False and nothing else in this module runs unless a
real, working CUDA+NVVM stack is detected, and any failure past that point
is caught and marks the GPU path broken for the rest of the process.
`synthesis_splat` in materialize_methods.py always has the prange-parallel
CPU path ready and never depends on this module succeeding.
"""
from __future__ import annotations

import numpy as np

from ..utils.app_logging import get_logger
from .gpu_utils import is_numba_cuda_available

logger = get_logger(__name__)

try:
    from numba import cuda
    _CUDA_IMPORT_OK = True
except Exception as exc:  # pragma: no cover - numba always ships the cuda module
    cuda = None
    _CUDA_IMPORT_OK = False

_ACC_TPB = 256             # threads per block, accumulate kernel
_RESOLVE_BLOCK = (16, 16)  # 2D block, resolve/zero kernels

# num_splats * patch_h * patch_w, a proxy for total scatter-write volume.
# Below this, GPU kernel-launch and host<->device transfer overhead
# outweighs the win over the prange CPU path. Measured directly on an RTX
# 3060 across canvas/patch combinations from 768px to 2048px: CPU wins up
# to ~17M work units, GPU wins consistently from ~23M on, with a noisy
# transition band in between (kernel-launch overhead is close to fixed, so
# the exact crossover shifts a few million either way with patch/grid
# shape). 20M sits just under the point GPU starts winning reliably.
_MIN_GPU_WORK_UNITS = 20_000_000

_gpu_broken = False  # sticky once True: never retried again this process
_gpu_warmed = False  # whether warmup_cuda_kernels() has run at all


if _CUDA_IMPORT_OK:

    @cuda.jit(cache=True)
    def _splat_accumulate_kernel(accum, weight, patches, masks, coords, indices):
        """One thread per (splat, patch-pixel). Scatters into accum/weight
        with atomic adds, since different splats' patches routinely overlap
        by design (that's what makes the weighted-average blend work).
        Launch grid: (num_splats, ceil(ph*pw / _ACC_TPB)), block: _ACC_TPB.
        """
        splat_i = cuda.blockIdx.x
        pixel_i = cuda.blockIdx.y * cuda.blockDim.x + cuda.threadIdx.x

        num_splats = coords.shape[0]
        ph = patches.shape[1]
        pw = patches.shape[2]
        if splat_i >= num_splats or pixel_i >= ph * pw:
            return

        py = pixel_i // pw
        px = pixel_i % pw

        top = coords[splat_i, 0]
        left = coords[splat_i, 1]
        pidx = indices[splat_i]

        alpha = masks[pidx, py, px]
        if alpha <= 1e-7:
            return

        height = accum.shape[0]
        width = accum.shape[1]
        channels = accum.shape[2]
        y = (top + py) % height
        x = (left + px) % width

        cuda.atomic.add(weight, (y, x), alpha)
        for c in range(channels):
            cuda.atomic.add(accum, (y, x, c), patches[pidx, py, px, c] * alpha)

    @cuda.jit(cache=True)
    def _splat_resolve_kernel(accum, weight, fallback, out):
        """One thread per output pixel -- no atomics needed, every pixel is
        independent. Launch grid: ceil(width/16) x ceil(height/16), block
        _RESOLVE_BLOCK.
        """
        x, y = cuda.grid(2)
        height = out.shape[0]
        width = out.shape[1]
        if y >= height or x >= width:
            return
        w = weight[y, x]
        channels = out.shape[2]
        if w > 1e-9:
            inv = 1.0 / w
            for c in range(channels):
                out[y, x, c] = accum[y, x, c] * inv
        else:
            for c in range(channels):
                out[y, x, c] = fallback[y, x, c]

    @cuda.jit(cache=True)
    def _zero_hwc_kernel(arr):
        x, y = cuda.grid(2)
        if y < arr.shape[0] and x < arr.shape[1]:
            for c in range(arr.shape[2]):
                arr[y, x, c] = 0.0

    @cuda.jit(cache=True)
    def _zero_hw_kernel(arr):
        x, y = cuda.grid(2)
        if y < arr.shape[0] and x < arr.shape[1]:
            arr[y, x] = 0.0


def _resolve_grid(height: int, width: int) -> tuple[int, int]:
    bx, by = _RESOLVE_BLOCK
    return ((width + bx - 1) // bx, (height + by - 1) // by)


class SplatCudaSession:
    """Device-resident accum/weight for one canvas.

    Reused across the repeated small `.accumulate()` calls synthesis_splat's
    `stream_variations` path makes, so only the per-variation patch/mask
    batch re-uploads each call -- never the full-canvas accumulators. Call
    `.resolve()` exactly once at the end to download the final result.
    """

    def __init__(self, height: int, width: int, channels: int):
        self.height = height
        self.width = width
        self.accum_d = cuda.device_array((height, width, channels), dtype=np.float32)
        self.weight_d = cuda.device_array((height, width), dtype=np.float32)
        grid = _resolve_grid(height, width)
        _zero_hwc_kernel[grid, _RESOLVE_BLOCK](self.accum_d)
        _zero_hw_kernel[grid, _RESOLVE_BLOCK](self.weight_d)

    def accumulate(self, patches: np.ndarray, masks: np.ndarray,
                   coords: np.ndarray, indices: np.ndarray) -> None:
        num_splats = coords.shape[0]
        if num_splats == 0:
            return
        patches_d = cuda.to_device(np.ascontiguousarray(patches))
        masks_d = cuda.to_device(np.ascontiguousarray(masks))
        coords_d = cuda.to_device(coords)
        indices_d = cuda.to_device(indices)
        ph, pw = patches.shape[1:3]
        blocks_y = (ph * pw + _ACC_TPB - 1) // _ACC_TPB
        _splat_accumulate_kernel[(num_splats, blocks_y), _ACC_TPB](
            self.accum_d, self.weight_d, patches_d, masks_d, coords_d, indices_d)

    def resolve(self, fallback: np.ndarray, out: np.ndarray) -> np.ndarray:
        fallback_d = cuda.to_device(fallback)
        out_d = cuda.device_array_like(out)
        grid = _resolve_grid(self.height, self.width)
        _splat_resolve_kernel[grid, _RESOLVE_BLOCK](
            self.accum_d, self.weight_d, fallback_d, out_d)
        out_d.copy_to_host(out)
        cuda.synchronize()
        return out


def warmup_cuda_kernels() -> bool:
    """Compile and smoke-run the CUDA kernels once.

    Safe to call from anywhere, any number of times, on any machine: returns
    False immediately when Numba CUDA isn't usable, and never raises.

    Called from the background precompile thread at startup (see
    `precompile_jit_functions` in seamless.py) so the expensive NVVM compile
    happens during the splash screen, not on a user's first Splat render --
    and again defensively from `gpu_eligible()` in case that startup call
    never ran.
    """
    global _gpu_broken, _gpu_warmed
    if _gpu_warmed:
        return not _gpu_broken
    _gpu_warmed = True

    if not _CUDA_IMPORT_OK or not is_numba_cuda_available():
        _gpu_broken = True
        return False

    try:
        import warnings
        from numba.core.errors import NumbaPerformanceWarning

        patches = np.zeros((1, 64, 64, 3), dtype=np.float32)
        masks = np.ones((1, 64, 64), dtype=np.float32)
        coords = np.array([[0, 0]], dtype=np.int32)
        indices = np.array([0], dtype=np.int32)
        fallback = np.zeros((64, 64, 3), dtype=np.float32)
        out = np.empty((64, 64, 3), dtype=np.float32)

        # This tiny single-splat smoke call is deliberately far below real
        # occupancy (real calls launch across hundreds/thousands of splats
        # and full patch sizes) -- suppress the resulting low-occupancy
        # warning here only, so it doesn't spam the log on every launch.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            session = SplatCudaSession(64, 64, 3)
            session.accumulate(patches, masks, coords, indices)
            session.resolve(fallback, out)
    except Exception as exc:
        logger.debug("CUDA splat warm-up failed, disabling GPU path: %s", exc)
        _gpu_broken = True
        return False

    logger.info("CUDA splat path ready")
    return True


def mark_broken() -> None:
    """Disable the GPU path for the rest of this process.

    Called after a real (post-warmup) GPU call fails, so a reproducible
    failure isn't retried on every subsequent Splat render.
    """
    global _gpu_broken
    _gpu_broken = True


def gpu_eligible(num_splats: int, patch_h: int, patch_w: int, preview: bool) -> bool:
    """Whether this Splat call should run on the GPU.

    Hard-rejects preview renders regardless of size -- those are capped at
    384px and must fit a 50ms live-preview budget, where kernel-launch and
    transfer overhead is not worth paying. Otherwise gates on total
    scatter-write volume, a proxy for how much parallel work there is to
    amortize that overhead against.
    """
    if _gpu_broken or preview:
        return False
    if not warmup_cuda_kernels():
        return False
    return num_splats * patch_h * patch_w >= _MIN_GPU_WORK_UNITS
