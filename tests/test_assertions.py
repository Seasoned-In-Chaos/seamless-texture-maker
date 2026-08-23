import numpy as np
import pytest

from app.core.assertions import assert_float32, assert_3d, assert_2d


class TestAssertFloat32:
    def test_passes_for_float32_array(self):
        assert_float32(np.zeros((4, 4), dtype=np.float32))

    def test_raises_for_wrong_dtype(self):
        with pytest.raises(TypeError):
            assert_float32(np.zeros((4, 4), dtype=np.uint8))

    def test_raises_for_non_ndarray(self):
        with pytest.raises(TypeError):
            assert_float32([1, 2, 3])

    def test_error_message_includes_name(self):
        with pytest.raises(TypeError, match="my_array"):
            assert_float32(np.zeros((2, 2), dtype=np.uint8), "my_array")


class TestAssert3D:
    def test_passes_for_3d_array(self):
        assert_3d(np.zeros((4, 4, 3)))

    def test_raises_for_2d_array(self):
        with pytest.raises(ValueError):
            assert_3d(np.zeros((4, 4)))


class TestAssert2D:
    def test_passes_for_2d_array(self):
        assert_2d(np.zeros((4, 4)))

    def test_raises_for_3d_array(self):
        with pytest.raises(ValueError):
            assert_2d(np.zeros((4, 4, 3)))
