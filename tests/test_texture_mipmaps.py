"""Tests for map-aware viewport mipmap filtering."""

import numpy as np

from app.core.texture_mipmaps import generate_mipmaps


def test_mipmap_chain_reaches_one_pixel():
    image = np.zeros((8, 4, 3), dtype=np.uint8)
    levels = generate_mipmaps(image, "Base Color")
    assert [level.shape[:2] for level in levels] == [(8, 4), (4, 2), (2, 1), (1, 1)]


def test_color_mipmaps_keep_transparent_rgb_premultiplied():
    # A transparent red texel must not bleed red into an opaque neighboring texel.
    image = np.zeros((2, 2, 4), dtype=np.uint8)
    image[:, 0] = [0, 0, 255, 0]       # transparent red in BGRA
    image[:, 1] = [255, 0, 0, 255]     # opaque blue in BGRA
    mip = generate_mipmaps(image, "Base Color")[1]
    assert mip.shape == (1, 1, 4)
    assert mip[0, 0, 3] == 128
    assert mip[0, 0, 0] < 10
    assert mip[0, 0, 2] > 240


def test_normal_mipmaps_average_slopes_and_renormalize():
    # Flat +Z normal encoded in BGR is (128, 128, 255).
    image = np.empty((4, 4, 3), dtype=np.uint8)
    image[:] = [255, 128, 128]
    mip = generate_mipmaps(image, "Normal")[1]
    assert np.allclose(mip[0, 0, :3], [128, 128, 255], atol=1)
    assert mip[0, 0, 3] == 255


def test_normal_mipmaps_cancel_opposing_slopes():
    # Opposing tangent slopes should average to a flat normal, rather than
    # weakening the Z component as encoded-RGB averaging would do.
    image = np.array(
        [
            [[230, 128, 204], [230, 128, 51]],
            [[230, 128, 204], [230, 128, 51]],
        ],
        dtype=np.uint8,
    )
    mip = generate_mipmaps(image, "Normal")[1]
    assert np.allclose(mip[0, 0, :3], [128, 128, 255], atol=1)
