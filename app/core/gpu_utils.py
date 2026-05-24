"""
GPU utility functions for detecting and utilizing GPU acceleration.
"""
from __future__ import annotations

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

