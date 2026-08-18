"""Tests for splat synthesis."""
import numpy as np
import pytest

from app.core.materialize_methods import create_falloff_mask, synthesis_splat


def _make_source(size: int = 128) -> np.ndarray:
    """Create a synthetic float32 source texture."""
    return np.random.uniform(0, 255, (size, size, 3)).astype(np.float32)


class TestSplat:
    def test_splat_output_shape(self):
        img = _make_source(128)
        result, _ = synthesis_splat(img, new_size=(128, 128), scale=1.0, falloff=0.2)
        assert result.shape == (128, 128, 3)

    def test_splat_dtype(self):
        img = _make_source(128)
        result, _ = synthesis_splat(img, new_size=(128, 128), scale=1.0, falloff=0.2)
        assert result.dtype == np.float32

    def test_splat_deterministic(self):
        img = _make_source(128)
        r1, _ = synthesis_splat(img, new_size=(64, 64), scale=1.0, falloff=0.2)
        r2, _ = synthesis_splat(img, new_size=(64, 64), scale=1.0, falloff=0.2)
        np.testing.assert_array_equal(r1, r2)

    def test_splat_different_seeds(self):
        img = _make_source(128)
        r1, _ = synthesis_splat(img, new_size=(64, 64), scale=1.0, falloff=0.2)
        # Different wobble should produce different result
        r2, _ = synthesis_splat(img, new_size=(64, 64), scale=1.0, wobble=0.8, falloff=0.2)
        assert not np.array_equal(r1, r2)

    def test_falloff_reaches_all_four_borders(self):
        mask = create_falloff_mask((32, 48), falloff=0.25, circular=True)
        assert np.all(mask[0, :] == 0)
        assert np.all(mask[-1, :] == 0)
        assert np.all(mask[:, 0] == 0)
        assert np.all(mask[:, -1] == 0)

    def test_splat_variations_keep_one_scale(self):
        img = _make_source(128)
        _, batches = synthesis_splat(
            img, new_size=(128, 128), scale=0.75, rand_rot=0.5, falloff=0.2
        )
        patches, masks = batches
        assert patches.ndim == 4
        assert masks.ndim == 4
        assert len({tuple(p.shape) for p in patches}) == 1
        assert len({tuple(m.shape) for m in masks}) == 1

    def test_edge_falloff_disables_patch_rotation(self):
        img = _make_source(96)
        stable, _ = synthesis_splat(
            img, new_size=(96, 96), scale=0.5, rotation=0, rand_rot=0,
            wobble=0.2, falloff=0.25,
        )
        rotated, _ = synthesis_splat(
            img, new_size=(96, 96), scale=0.5, rotation=65, rand_rot=0.77,
            wobble=0.2, falloff=0.25,
        )
        np.testing.assert_array_equal(stable, rotated)

    def test_splat_preserves_source_detail(self):
        """Splatting source crops must not average the texture into a blur."""
        checker = np.indices((128, 128)).sum(axis=0) % 2
        image = np.repeat((checker * 255).astype(np.float32)[..., np.newaxis], 3, axis=2)
        result, _ = synthesis_splat(
            image, new_size=(128, 128), scale=1.0, wobble=0.25, falloff=0.06,
        )
        assert result.std() > image.std() * 0.45
