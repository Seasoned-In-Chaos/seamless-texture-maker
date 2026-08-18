"""Tests for splat synthesis."""
import numpy as np
import pytest

from app.core.materialize_methods import (
    synthesis_splat,
    create_splat_mask,
    _build_patch_bank,
    _splat_placements,
)
from app.core.materialize_methods_jit import splat_accumulate_jit


def _make_source(size: int = 128) -> np.ndarray:
    """Create a synthetic float32 source texture."""
    rng = np.random.default_rng(0)
    return rng.uniform(0, 255, (size, size, 3)).astype(np.float32)


def _structured_source(size: int = 128) -> np.ndarray:
    """A source with spatial structure, like a real texture.

    White noise is uncorrelated pixel to pixel, so overlapping patches
    average it toward flat grey regardless of how the weights are shaped.
    Contrast behaviour is only meaningful on structured content.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    pattern = 128.0 + 100.0 * np.sin(xx / 9.0) * np.cos(yy / 7.0)
    return np.repeat(pattern[:, :, np.newaxis], 3, axis=2).astype(np.float32)


def _seam_ratio(img: np.ndarray) -> tuple:
    """Wrap-boundary difference relative to typical interior difference.

    ~1.0 means the seam is indistinguishable from ordinary detail.
    """
    o = img.astype(np.float64)
    v = np.mean(np.abs(o[0] - o[-1])) / np.mean(np.abs(o[1:] - o[:-1]))
    h = np.mean(np.abs(o[:, 0] - o[:, -1])) / np.mean(np.abs(o[:, 1:] - o[:, :-1]))
    return v, h


_BASE = dict(new_size=(128, 128), scale=0.45, rotation=0.0,
             rand_rot=0.5, wobble=0.4, falloff=0.4, seed=0)


def _run(**overrides):
    params = dict(_BASE)
    params.update(overrides)
    return synthesis_splat(_make_source(128), **params)[0]


class TestSplat:
    def test_splat_output_shape(self):
        result, _ = synthesis_splat(_make_source(128), new_size=(128, 128),
                                    scale=1.0, falloff=0.2)
        assert result.shape == (128, 128, 3)

    def test_splat_dtype(self):
        result, _ = synthesis_splat(_make_source(128), new_size=(128, 128),
                                    scale=1.0, falloff=0.2)
        assert result.dtype == np.float32

    def test_splat_deterministic(self):
        np.testing.assert_array_equal(_run(), _run())

    def test_grayscale_round_trips(self):
        img = _make_source(96)[:, :, 0]
        result, _ = synthesis_splat(img, new_size=(96, 96), scale=0.4)
        assert result.shape == (96, 96)
        assert result.dtype == np.float32

    def test_output_is_seamlessly_tileable(self):
        # Placements wrap modulo the canvas, so the result must be periodic.
        for falloff in (0.05, 0.5, 1.0):
            v, h = _seam_ratio(_run(falloff=falloff))
            assert v < 1.35 and h < 1.35

    @pytest.mark.parametrize("param,value", [
        ("scale", 0.8),
        ("rotation", 45.0),
        ("rand_rot", 1.0),
        ("wobble", 0.95),
        ("falloff", 0.95),
        ("seed", 7),
    ])
    def test_every_control_changes_the_output(self, param, value):
        # Each slider must visibly do something -- several of these were
        # silently inert before (seed never reached the synthesis at all).
        assert not np.allclose(_run(), _run(**{param: value}))

    def test_preserves_contrast(self):
        # A flat-topped mask averages every overlapping patch equally and
        # washes the texture out; the centre-biased weighting must keep a
        # good share of the source's contrast. Measured on a structured
        # source -- averaging uncorrelated white noise loses contrast no
        # matter how the weights are shaped, so it proves nothing here.
        source = _structured_source(128)
        result = synthesis_splat(source, **dict(_BASE, falloff=0.1))[0]
        assert result.std() > source.std() * 0.7

    def test_lower_falloff_keeps_more_detail(self):
        source = _structured_source(128)
        sharp = synthesis_splat(source, **dict(_BASE, falloff=0.05))[0]
        soft = synthesis_splat(source, **dict(_BASE, falloff=1.0))[0]
        assert sharp.std() > soft.std()


class TestSplatCoverage:
    # Any pixel no patch reaches falls back to the (non-seamless) source,
    # which would both show as an artefact and break tiling.
    @pytest.mark.parametrize("scale", [0.15, 0.45, 0.9])
    @pytest.mark.parametrize("wobble", [0.0, 1.0])
    @pytest.mark.parametrize("falloff", [0.0, 1.0])
    def test_full_coverage(self, scale, wobble, falloff):
        img = _make_source(96)
        patches, masks = _build_patch_bank(img, scale=scale, rotation=0.0,
                                           rand_rot=1.0, wobble=wobble,
                                           falloff=falloff, seed=0,
                                           preview=True)
        coords, indices = _splat_placements(96, 96, patches.shape[1],
                                            patches.shape[2],
                                            patches.shape[0], 0)
        accum = np.zeros((96, 96, 3), dtype=np.float32)
        weight = np.zeros((96, 96), dtype=np.float32)
        splat_accumulate_jit(accum, weight, patches, masks, coords, indices)
        assert np.count_nonzero(weight < 1e-9) == 0


class TestSplatMask:
    def test_alpha_is_bounded(self):
        mask = create_splat_mask((64, 64), falloff=0.5, wobble=0.5,
                                 rng=np.random.RandomState(0))
        assert mask.min() >= 0.0 and mask.max() <= 1.0

    def test_peaks_at_centre_and_vanishes_at_corners(self):
        mask = create_splat_mask((64, 64), falloff=0.3)
        # The true centre of an even-sized patch falls between pixels, so
        # the peak sits on one of the four pixels straddling it.
        peak_y, peak_x = np.unravel_index(mask.argmax(), mask.shape)
        assert peak_y in (31, 32) and peak_x in (31, 32)
        assert mask.max() > 0.8
        assert mask[0, 0] == 0.0
        assert mask[-1, -1] == 0.0

    def test_wobble_deforms_the_outline(self):
        plain = create_splat_mask((64, 64), falloff=0.3, wobble=0.0)
        wobbled = create_splat_mask((64, 64), falloff=0.3, wobble=0.9,
                                    rng=np.random.RandomState(0))
        assert not np.allclose(plain, wobbled)

    def test_low_falloff_concentrates_weight(self):
        # Low falloff must let the nearest patch dominate rather than
        # spreading weight evenly, otherwise overlaps average detail away.
        sharp = create_splat_mask((64, 64), falloff=0.05)
        soft = create_splat_mask((64, 64), falloff=1.0)
        assert sharp.mean() < soft.mean()


class TestSplatPlacements:
    def test_indices_stay_in_range(self):
        coords, indices = _splat_placements(128, 96, 40, 40, 5, seed=0)
        assert indices.min() >= 0 and indices.max() < 5
        assert coords.shape[0] == indices.shape[0]

    def test_seed_changes_layout(self):
        a, _ = _splat_placements(128, 128, 40, 40, 4, seed=0)
        b, _ = _splat_placements(128, 128, 40, 40, 4, seed=1)
        assert not np.array_equal(a, b)
