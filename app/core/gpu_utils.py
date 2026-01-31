"""
GPU utility functions for detecting and utilizing GPU acceleration.
"""
import cv2
import numpy as np


def is_cuda_available():
    """Check if CUDA is available through OpenCV."""
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        return count > 0
    except Exception:
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
    except Exception:
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
            except Exception:
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
            except Exception:
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
            except Exception:
                self.use_gpu = False
        
        return cv2.GaussianBlur(img, ksize, sigma)
    
    def resize(self, img, size, interpolation=cv2.INTER_LINEAR):
        """Resize image with GPU acceleration."""
        if self.use_gpu:
            try:
                gpu_img = self.upload(img)
                gpu_result = cv2.cuda.resize(gpu_img, size, interpolation=interpolation)
                return self.download(gpu_result)
            except Exception:
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
            except Exception:
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

