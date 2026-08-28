"""
GPU utility functions for detecting and utilizing GPU acceleration.
"""
from __future__ import annotations

import os
import sys
import time
import cv2
import numpy as np

from ..utils.app_logging import get_logger


logger = get_logger(__name__)


def is_cuda_available():
    """Check if CUDA is available through OpenCV."""
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        return count > 0
    except Exception as exc:
        logger.debug("CUDA availability check failed: %s", exc)
        return False


_numba_cuda_state = None  # None = unchecked; True/False once determined


def _cuda_redist_roots() -> tuple[str | None, str | None]:
    """Locate the NVVM (nvcc) and CUDA Runtime redistributable roots.

    Returns (nvcc_root, runtime_root) -- either may be None if that piece
    isn't available. Two independent sources, depending on how the app is
    running:

    - **Frozen (PyInstaller) build**: build.spec bundles just the DLLs/data
      Numba actually loads (not the ~37MB of headers and ptxas.exe these
      pip packages also carry) under a fixed `cuda_redist/` layout next to
      the executable -- `cuda_redist/nvcc/nvvm/...` and
      `cuda_redist/runtime/bin/...`, deliberately mirroring each source pip
      package's own internal structure so the same downstream path-joining
      logic in _ensure_cuda_home works for both cases. Namespace packages
      like `nvidia.cuda_nvcc` (no __init__.py, no .py files at all) aren't
      something PyInstaller's import scanner reliably discovers on its own,
      so the frozen build intentionally does not rely on `import nvidia...`
      at runtime -- this looks the files up by a fixed relative path next
      to the frozen executable instead.
    - **Running from source**: resolved via the installed
      `nvidia-cuda-nvcc-cu12` / `nvidia-cuda-runtime-cu12` pip packages'
      own namespace-package `__path__`, same as any other import.
    """
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", None) or os.path.dirname(sys.executable)
        redist = os.path.join(base, "cuda_redist")
        nvcc_root = os.path.join(redist, "nvcc")
        runtime_root = os.path.join(redist, "runtime")
        return (nvcc_root if os.path.isdir(nvcc_root) else None,
                runtime_root if os.path.isdir(runtime_root) else None)

    nvcc_root = None
    try:
        import nvidia.cuda_nvcc as _cuda_nvcc
        nvcc_root = next(iter(_cuda_nvcc.__path__), None)
    except Exception as exc:
        logger.debug("nvidia-cuda-nvcc-cu12 not available: %s", exc)

    runtime_root = None
    try:
        import nvidia.cuda_runtime as _cuda_runtime
        runtime_root = next(iter(_cuda_runtime.__path__), None)
    except Exception as exc:
        logger.debug("nvidia-cuda-runtime-cu12 not available: %s", exc)

    return nvcc_root, runtime_root


def _ensure_cuda_home() -> None:
    """Point Numba at the CUDA redistributables, if present.

    Numba's CUDA target needs two pieces neither end users nor this app's
    build ships a system CUDA Toolkit for: NVVM (its CUDA compiler) and the
    CUDA Runtime (`cudart`, needed to query which GPU architectures NVVM can
    target -- not just for caching). See `_cuda_redist_roots` for where
    each comes from.

    This Numba version only looks for NVVM inside a conda environment or a
    $CUDA_HOME it's told about, not a plain pip install's site-packages or
    an arbitrary bundled directory, so CUDA_HOME is pointed at the nvcc
    root -- its internal layout (nvvm/bin/*.dll, nvvm/libdevice/*.bc)
    already matches what Numba expects under CUDA_HOME, confirmed directly
    against this Numba version. Uses setdefault so a real system CUDA
    Toolkit the user already configured via CUDA_HOME is never overridden.

    cudart has no equivalent env var: Numba's only non-conda lookup for it
    is CUDA_HOME/bin, but cudart64_*.dll lives under the separate runtime
    root, which CUDA_HOME can't simultaneously point at alongside nvvm/.
    Numba caches its resolved path dict in a plain function attribute
    (numba.cuda.cuda_paths.get_cuda_paths._cached_result) documented as
    {"nvvm": path, "libdevice": [...], "cudalib_dir": path, ...}, so once
    that cache is populated (via the CUDA_HOME-based lookup above, which
    gets nvvm/libdevice right already), 'cudalib_dir' is overwritten
    in-place to point at the runtime root's bin/ instead. If a future Numba
    version changes this caching detail, this patch simply becomes a no-op
    (the attribute won't be there to overwrite) and the GPU path falls back
    to unavailable, same as any other detection failure -- not a crash.

    Safe to call unconditionally: silently does nothing wherever a piece
    isn't available, e.g. a non-Windows dev checkout or before `pip
    install`.
    """
    nvcc_root, runtime_root = _cuda_redist_roots()

    if nvcc_root:
        os.environ.setdefault("CUDA_HOME", nvcc_root)

    if not runtime_root:
        return
    runtime_bin = os.path.join(runtime_root, "bin")
    if not os.path.isdir(runtime_bin):
        return

    try:
        from numba.cuda.cuda_paths import get_cuda_paths, _env_path_tuple
        from numba.misc.findlib import find_lib

        paths = get_cuda_paths()  # populates the cache via CUDA_HOME above
        # The resolved directory can exist yet still be the wrong one (e.g.
        # nvidia-cuda-nvcc-cu12's own bin/, which only holds ptxas.exe) --
        # check for an actual cudart library, the same way Numba itself
        # looks for one, rather than just checking the directory exists.
        current_dir = paths.get("cudalib_dir", _env_path_tuple(None, None)).info
        if not find_lib("cudart", libdir=current_dir):
            paths["cudalib_dir"] = _env_path_tuple("bundled CUDA runtime", runtime_bin)
    except Exception as exc:
        logger.debug("Could not point Numba at the CUDA runtime: %s", exc)


def is_numba_cuda_available():
    """Check if Numba's @cuda.jit target (driver + NVVM) can compile and run.

    Deliberately separate from ``is_cuda_available()`` above, which checks
    OpenCV's CUDA module -- unused and always False in shipped installs,
    since the pinned opencv-python-headless build has no CUDA compiled in
    (see GPUAccelerator). This checks the unrelated Numba/NVVM stack used by
    the Splat GPU path in materialize_methods_cuda.py. Cached for the
    process lifetime: the underlying check touches the CUDA driver, and a
    machine's CUDA capability cannot change mid-run.
    """
    global _numba_cuda_state
    if _numba_cuda_state is None:
        try:
            _ensure_cuda_home()
            from numba import cuda
            _numba_cuda_state = bool(cuda.is_available())
        except Exception as exc:
            logger.debug("Numba CUDA availability check failed: %s", exc)
            _numba_cuda_state = False
    return _numba_cuda_state


def get_gpu_info():
    """Get information about available GPU."""
    if not is_cuda_available():
        return None
    
    try:
        device = cv2.cuda.getDevice()
        return {
            'device_id': device,
            'name': f'CUDA Device {device}',
        }
    except Exception as exc:
        logger.debug("CUDA device info unavailable: %s", exc)
        return None


class GPUAccelerator:
    """Context manager for GPU-accelerated operations with CPU fallback."""
    
    def __init__(self):
        self.use_gpu = is_cuda_available()
        self._stream = None
    
    def __enter__(self):
        if self.use_gpu:
            try:
                self._stream = cv2.cuda.Stream()
            except Exception as exc:
                logger.debug("CUDA stream unavailable, using CPU: %s", exc)
                self.use_gpu = False
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._stream is not None:
            self._stream.waitForCompletion()
        return False
    
    def upload(self, img):
        """Upload image to GPU memory."""
        if self.use_gpu:
            try:
                gpu_mat = cv2.cuda_GpuMat()
                gpu_mat.upload(img)
                return gpu_mat
            except Exception as exc:
                logger.debug("CUDA upload failed, using CPU: %s", exc)
                self.use_gpu = False
        return img
    
    def download(self, gpu_mat):
        """Download image from GPU memory."""
        if self.use_gpu and hasattr(gpu_mat, 'download'):
            return gpu_mat.download()
        return gpu_mat
    
    def gaussian_blur(self, img, ksize, sigma):
        """Apply Gaussian blur with GPU acceleration."""
        if self.use_gpu:
            try:
                gpu_img = self.upload(img)
                gpu_filter = cv2.cuda.createGaussianFilter(
                    gpu_img.type(), -1, ksize, sigma
                )
                gpu_result = gpu_filter.apply(gpu_img)
                return self.download(gpu_result)
            except Exception as exc:
                logger.debug("CUDA Gaussian blur failed, using CPU: %s", exc)
                self.use_gpu = False
        
        return cv2.GaussianBlur(img, ksize, sigma)
    
    def resize(self, img, size, interpolation=cv2.INTER_LINEAR):
        """Resize image with GPU acceleration."""
        if self.use_gpu:
            try:
                gpu_img = self.upload(img)
                gpu_result = cv2.cuda.resize(gpu_img, size, interpolation=interpolation)
                return self.download(gpu_result)
            except Exception as exc:
                logger.debug("CUDA resize failed, using CPU: %s", exc)
                self.use_gpu = False
        
        return cv2.resize(img, size, interpolation=interpolation)
    
    def alpha_blend(self, img1, img2, alpha):
        """
        Alpha blend two images with GPU acceleration.
        result = img1 * (1 - alpha) + img2 * alpha
        
        Args:
            img1: Background image
            img2: Foreground image
            alpha: Alpha mask (0-1 float or 0-255 uint8)
        """
        if self.use_gpu:
            try:
                gpu_img1 = self.upload(img1.astype(np.float32))
                gpu_img2 = self.upload(img2.astype(np.float32))
                
                if alpha.dtype == np.uint8:
                    alpha = alpha.astype(np.float32) / 255.0
                
                gpu_alpha = self.upload(alpha.astype(np.float32))
                
                # result = img1 + (img2 - img1) * alpha
                gpu_diff = cv2.cuda.subtract(gpu_img2, gpu_img1)
                gpu_scaled = cv2.cuda.multiply(gpu_diff, gpu_alpha)
                gpu_result = cv2.cuda.add(gpu_img1, gpu_scaled)
                
                return self.download(gpu_result)
            except Exception as exc:
                logger.debug("CUDA alpha blend failed, using CPU: %s", exc)
                self.use_gpu = False
        
        # CPU fallback
        if img1.dtype != np.float32:
            img1 = img1.astype(np.float32)
        if img2.dtype != np.float32:
            img2 = img2.astype(np.float32)
        if alpha.dtype == np.uint8:
            alpha = alpha.astype(np.float32) / 255.0
        
        if len(alpha.shape) == 2 and len(img1.shape) == 3:
            alpha = alpha[:, :, np.newaxis]
        
        result = img1 * (1.0 - alpha) + img2 * alpha
        return result

    def inpaint_gpu(self, image: np.ndarray, mask: np.ndarray,
                    radius: int = 3,
                    method: str = 'telea') -> np.ndarray:
        """Inpaint with GPU acceleration, falling back to CPU.

        OpenCV does not ship a native ``cv2.cuda.inpaint`` in all builds,
        so this method attempts GPU upload/download for the surrounding
        blur/preprocessing and falls back to ``cv2.inpaint`` on CPU.

        Args:
            image: Input image (float32 or uint8, BGR).
            mask: Binary uint8 mask (255 = inpaint region).
            radius: Inpainting radius.
            method: 'telea' or 'ns'.

        Returns:
            float32 inpainted image.
        """
        t0 = time.perf_counter()
        flags = cv2.INPAINT_TELEA if method == 'telea' else cv2.INPAINT_NS

        mask_u8 = mask.astype(np.uint8)
        img_u8 = np.clip(image, 0, 255).astype(np.uint8) if image.dtype != np.uint8 else image

        if self.use_gpu:
            try:
                gpu_img = self.upload(img_u8)
                gpu_mask = self.upload(mask_u8)
                # OpenCV CUDA does not expose inpaint directly in most builds;
                # download and run CPU inpaint, but log the attempt.
                cpu_img = self.download(gpu_img)
                cpu_mask = self.download(gpu_mask)
                result_u8 = cv2.inpaint(cpu_img, cpu_mask, radius, flags)
                elapsed = (time.perf_counter() - t0) * 1000.0
                logger.info("GPU inpaint (CPU fallback): %.1f ms", elapsed)
                return result_u8.astype(np.float32)
            except Exception as exc:
                logger.debug("CUDA inpaint failed, falling back: %s", exc)
                self.use_gpu = False

        # CPU path
        result_u8 = cv2.inpaint(img_u8, mask_u8, radius, flags)
        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info("CPU inpaint fallback: %.1f ms", elapsed)
        return result_u8.astype(np.float32)

