import numpy as np

from app.core.offset_mapping import (
    offset_image,
    reverse_offset,
    get_seam_mask,
    create_cross_mask,
)


def _source(height=40, width=60):
    values = np.arange(height * width * 3, dtype=np.float32).reshape(height, width, 3)
    return (values * 0.37) % 255.0


class TestOffsetReverse:
    def test_reverse_offset_undoes_offset(self):
        image = _source()
        offset = offset_image(image, 0.5, 0.5)
        restored = reverse_offset(offset, 0.5, 0.5)
        np.testing.assert_array_equal(restored, image)

    def test_offset_relocates_content(self):
        image = _source()
        offset = offset_image(image, 0.5, 0.5)
        assert offset.shape == image.shape
        assert not np.array_equal(offset, image)

    def test_offset_is_a_circular_shift(self):
        # np.roll relocates pixels, never drops or invents them -- the
        # multiset of values must be identical before and after.
        image = _source()
        offset = offset_image(image, 0.25, 0.75)
        assert sorted(offset.ravel().tolist()) == sorted(image.ravel().tolist())

    def test_zero_offset_is_identity(self):
        image = _source()
        np.testing.assert_array_equal(offset_image(image, 0.0, 0.0), image)


class TestSeamMask:
    def test_shape_and_dtype(self):
        mask = get_seam_mask(_source(50, 70), seam_width=10)
        assert mask.shape == (50, 70)
        assert mask.dtype == np.uint8

    def test_marks_center_cross(self):
        mask = get_seam_mask(_source(50, 70), seam_width=10)
        assert mask[25, 35] == 255  # centre falls inside the cross
        assert mask[0, 0] == 0      # corner is untouched


class TestCrossMask:
    def test_shape_and_dtype(self):
        mask = create_cross_mask(50, 70, thickness=10)
        assert mask.shape == (50, 70)
        assert mask.dtype == np.uint8

    def test_marks_center_cross(self):
        mask = create_cross_mask(50, 70, thickness=10)
        assert mask[25, 35] == 255
        assert mask[0, 0] == 0
