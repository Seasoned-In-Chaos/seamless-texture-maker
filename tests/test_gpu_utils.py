import cv2
import numpy as np

from app.core.gpu_utils import is_cuda_available, get_gpu_info, GPUAccelerator


class TestCudaDetection:
    def test_is_cuda_available_returns_bool(self):
        assert isinstance(is_cuda_available(), bool)

    def test_get_gpu_info_is_none_when_cuda_unavailable(self):
        if not is_cuda_available():
            assert get_gpu_info() is None


class TestGPUAcceleratorFallback:
    """Exercises the CPU fallback paths. In the shipped app these are the
    ONLY paths that ever run: is_cuda_available() is False for every
    install, because the pinned opencv-python-headless build has no CUDA
    support compiled in. use_gpu is forced False explicitly so these stay
    deterministic even if CUDA happens to be present wherever this runs.
    """

    def _cpu_only(self):
        gpu = GPUAccelerator()
        gpu.use_gpu = False
        return gpu

    def test_context_manager(self):
        with GPUAccelerator() as gpu:
            assert gpu is not None

    def test_upload_returns_input_unchanged_without_gpu(self):
        gpu = self._cpu_only()
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        assert gpu.upload(img) is img

    def test_download_returns_input_unchanged_without_gpu(self):
        gpu = self._cpu_only()
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        assert gpu.download(img) is img

    def test_gaussian_blur_matches_cv2_cpu(self):
        gpu = self._cpu_only()
        img = np.random.RandomState(0).uniform(0, 255, (32, 32, 3)).astype(np.float32)
        result = gpu.gaussian_blur(img, (5, 5), 1.5)
        expected = cv2.GaussianBlur(img, (5, 5), 1.5)
        np.testing.assert_array_equal(result, expected)

    def test_resize_matches_cv2_cpu(self):
        gpu = self._cpu_only()
        img = np.random.RandomState(0).uniform(0, 255, (32, 32, 3)).astype(np.float32)
        result = gpu.resize(img, (16, 16))
        expected = cv2.resize(img, (16, 16), interpolation=cv2.INTER_LINEAR)
        np.testing.assert_array_equal(result, expected)

    def test_alpha_blend_uint8_alpha_midpoint(self):
        gpu = self._cpu_only()
        img1 = np.zeros((4, 4, 3), dtype=np.float32)
        img2 = np.full((4, 4, 3), 255.0, dtype=np.float32)
        alpha = np.full((4, 4), 128, dtype=np.uint8)  # ~0.502
        result = gpu.alpha_blend(img1, img2, alpha)
        np.testing.assert_allclose(result, 128.0, atol=1.0)

    def test_alpha_blend_float_alpha_endpoints(self):
        gpu = self._cpu_only()
        img1 = np.full((4, 4, 3), 10.0, dtype=np.float32)
        img2 = np.full((4, 4, 3), 90.0, dtype=np.float32)
        zero = np.zeros((4, 4), dtype=np.float32)
        one = np.ones((4, 4), dtype=np.float32)
        np.testing.assert_allclose(gpu.alpha_blend(img1, img2, zero), img1)
        np.testing.assert_allclose(gpu.alpha_blend(img1, img2, one), img2)

    def test_inpaint_gpu_fills_masked_region(self):
        gpu = self._cpu_only()
        img = np.full((32, 32, 3), 100.0, dtype=np.float32)
        img[10:14, 10:14] = 0.0  # a hole surrounded by uniform content
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10:14, 10:14] = 255
        result = gpu.inpaint_gpu(img, mask, radius=3, method="telea")
        assert result.dtype == np.float32
        assert result.shape == img.shape
        assert result[12, 12].mean() > 50.0  # hole filled, not left at 0
