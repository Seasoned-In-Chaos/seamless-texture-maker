"""Tests for image I/O utilities."""
import os
import tempfile
import numpy as np
import pytest

from app.utils.image_io import load_as_float32, save_from_float32, load_image, save_image


class TestImageIO:
    def test_load_as_float32(self, tmp_path):
        # Create a test PNG
        img_u8 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        path = str(tmp_path / "test.png")
        import cv2
        cv2.imwrite(path, img_u8)

        result = load_as_float32(path)
        assert result.dtype == np.float32
        assert result.shape == (64, 64, 3)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_save_roundtrip(self, tmp_path):
        img_u8 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        path = str(tmp_path / "test.png")
        import cv2
        cv2.imwrite(path, img_u8)

        loaded = load_as_float32(path)
        out_path = str(tmp_path / "out.png")
        save_from_float32(loaded, out_path)

        reloaded, _meta = load_image(out_path)
        diff = np.abs(img_u8.astype(np.float32) - reloaded.astype(np.float32))
        assert diff.max() <= 2.0  # Within 1/255 rounding

    def test_atomic_write(self, tmp_path):
        img_u8 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        out_path = str(tmp_path / "atomic.png")
        save_image(img_u8, out_path)
        assert os.path.exists(out_path)

    def test_unicode_path(self, tmp_path):
        unicode_dir = tmp_path / "tëst_öd"
        unicode_dir.mkdir(exist_ok=True)
        img_u8 = np.zeros((8, 8, 3), dtype=np.uint8)
        path = str(unicode_dir / "test.png")
        save_image(img_u8, path)
        assert os.path.exists(path)
