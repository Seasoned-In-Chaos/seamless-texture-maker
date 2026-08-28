"""Tests for splat synthesis."""
import numba
import numpy as np
import pytest

from app.core.materialize_methods import (
    synthesis_splat,
    create_splat_mask,
    _build_patch_bank,
    _splat_placements,
)
import app.core.materialize_methods as materialize_methods
import app.core.materialize_methods_cuda as materialize_methods_cuda
from app.core.gpu_utils import is_numba_cuda_available
from app.core.materialize_methods_jit import (
    splat_accumulate_jit,
    splat_accumulate_parallel_jit,
    splat_resolve_jit,
)


@pytest.fixture
def force_cpu_splat(monkeypatch):
    """Force synthesis_splat onto the CPU path regardless of GPU threshold
    tuning or whether this happens to run on a CUDA-enabled machine, so
    exact-equality tests stay meaningful either way."""
    monkeypatch.setattr(materialize_methods_cuda, "gpu_eligible", lambda **kw: False)


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

    def test_patch_bank_never_exceeds_budget_with_forced_variations(self, monkeypatch):
        """An 8K-scale patch may be larger than the whole bank budget.

        It must still synthesize with one variation rather than forcing four
        large copies and exhausting memory before the algorithm starts.
        """
        monkeypatch.setattr(materialize_methods, "_PATCH_BANK_BUDGET_BYTES", 1)
        patches, masks = _build_patch_bank(
            _make_source(64), scale=1.0, rotation=0.0, rand_rot=1.0,
            wobble=0.2, falloff=0.2, seed=0, preview=False,
        )
        assert patches.shape[0] == masks.shape[0] == 1

    def test_random_rotation_affects_single_patch_fallback(self, monkeypatch):
        """Large splats may fit only one patch, but rotation must still work."""
        monkeypatch.setattr(materialize_methods, "_PATCH_BANK_BUDGET_BYTES", 1)
        source = _make_source(64)
        low, _ = synthesis_splat(
            source, new_size=(64, 64), scale=1.0, rotation=0.0,
            rand_rot=0.1, wobble=0.2, falloff=0.2, seed=0,
        )
        high, _ = synthesis_splat(
            source, new_size=(64, 64), scale=1.0, rotation=0.0,
            rand_rot=1.0, wobble=0.2, falloff=0.2, seed=0,
        )
        assert not np.allclose(low, high)

    def test_random_rotation_streams_when_full_bank_would_be_too_large(self, monkeypatch):
        """Memory-limited final renders keep multiple random orientations."""
        monkeypatch.setattr(materialize_methods, "_PATCH_BANK_BUDGET_BYTES", 1)
        result, batches = synthesis_splat(
            _make_source(64), new_size=(512, 512), scale=1.0,
            rotation=0.0, rand_rot=1.0, wobble=0.2, falloff=0.2, seed=0,
        )
        assert batches is None
        assert result.shape == (512, 512, 3)

    def test_splat_deterministic(self, force_cpu_splat):
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
    # which would both show as an artefact and break tiling. Parametrized
    # over both the serial and prange-parallel accumulate kernels -- full
    # coverage is a correctness property that must hold for either.
    @pytest.mark.parametrize("scale", [0.15, 0.45, 0.9])
    @pytest.mark.parametrize("wobble", [0.0, 1.0])
    @pytest.mark.parametrize("falloff", [0.0, 1.0])
    @pytest.mark.parametrize("accumulate_fn,extra_args", [
        (splat_accumulate_jit, ()),
        (splat_accumulate_parallel_jit, (numba.get_num_threads(),)),
    ], ids=["serial", "parallel"])
    def test_full_coverage(self, scale, wobble, falloff, accumulate_fn, extra_args):
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
        accumulate_fn(accum, weight, patches, masks, coords, indices, *extra_args)
        assert np.count_nonzero(weight < 1e-9) == 0


class TestSplatAccumulateParallel:
    """splat_accumulate_parallel_jit must match splat_accumulate_jit (kept
    unchanged as the serial reference oracle) for any band count."""

    def _reference_and_parallel(self, canvas, patches, masks, coords, indices, num_bands):
        h, w = canvas
        c = patches.shape[-1]
        accum_s = np.zeros((h, w, c), dtype=np.float32)
        weight_s = np.zeros((h, w), dtype=np.float32)
        splat_accumulate_jit(accum_s, weight_s, patches, masks, coords, indices)

        accum_p = np.zeros((h, w, c), dtype=np.float32)
        weight_p = np.zeros((h, w), dtype=np.float32)
        splat_accumulate_parallel_jit(accum_p, weight_p, patches, masks,
                                      coords, indices, num_bands)
        return (accum_s, weight_s), (accum_p, weight_p)

    @pytest.mark.parametrize("num_bands", [1, 2, 7, numba.get_num_threads()])
    def test_matches_serial_baseline(self, num_bands):
        source = _make_source(96)
        patches, masks = _build_patch_bank(source, scale=0.4, rotation=0.0,
                                           rand_rot=0.6, wobble=0.5, falloff=0.4,
                                           seed=0, preview=True)
        coords, indices = _splat_placements(96, 96, patches.shape[1], patches.shape[2],
                                            patches.shape[0], seed=0)
        (accum_s, weight_s), (accum_p, weight_p) = self._reference_and_parallel(
            (96, 96), patches, masks, coords, indices, num_bands)
        np.testing.assert_allclose(accum_p, accum_s, rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(weight_p, weight_s, rtol=1e-4, atol=1e-3)

    @pytest.mark.parametrize("num_bands", [1, 3, 8])
    def test_matches_serial_with_wraparound(self, num_bands):
        """Hand-crafted coords with top < 0 and top + ph > height, to
        specifically exercise the row-band-vs-wraparound interaction."""
        h, w, ph, pw = 32, 32, 20, 20
        patches = np.random.default_rng(1).uniform(0, 255, (1, ph, pw, 3)).astype(np.float32)
        masks = np.ones((1, ph, pw), dtype=np.float32)
        coords = np.array([[-5, -5], [h - 3, w - 3], [0, 0], [15, 15]], dtype=np.int32)
        indices = np.zeros(4, dtype=np.int32)
        (accum_s, weight_s), (accum_p, weight_p) = self._reference_and_parallel(
            (h, w), patches, masks, coords, indices, num_bands)
        np.testing.assert_allclose(accum_p, accum_s, rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(weight_p, weight_s, rtol=1e-5, atol=1e-4)

    def test_resolve_matches_between_serial_and_parallel_accumulate(self):
        """End-to-end: resolving each accumulate path's output must still
        agree, not just the raw accum/weight buffers."""
        source = _make_source(96)
        patches, masks = _build_patch_bank(source, scale=0.4, rotation=0.0,
                                           rand_rot=0.6, wobble=0.5, falloff=0.4,
                                           seed=0, preview=True)
        coords, indices = _splat_placements(96, 96, patches.shape[1], patches.shape[2],
                                            patches.shape[0], seed=0)
        (accum_s, weight_s), (accum_p, weight_p) = self._reference_and_parallel(
            (96, 96), patches, masks, coords, indices, numba.get_num_threads())

        fallback = source
        out_s = np.empty_like(accum_s)
        out_p = np.empty_like(accum_p)
        splat_resolve_jit(accum_s, weight_s, fallback, out_s)
        splat_resolve_jit(accum_p, weight_p, fallback, out_p)
        np.testing.assert_allclose(out_p, out_s, rtol=1e-4, atol=1e-3)


requires_cuda = pytest.mark.skipif(
    not is_numba_cuda_available(),
    reason="Numba CUDA (driver + NVVM + cudart) not available in this environment",
)


class TestSplatAccumulateGPU:
    """SplatCudaSession must match the serial CPU oracle. Skips cleanly on
    any machine without a working CUDA setup -- most CI/dev machines."""

    @requires_cuda
    def test_gpu_matches_cpu_serial_baseline(self):
        from app.core.materialize_methods_cuda import SplatCudaSession, warmup_cuda_kernels
        assert warmup_cuda_kernels()

        source = _make_source(96)
        patches, masks = _build_patch_bank(source, scale=0.4, rotation=0.0,
                                           rand_rot=0.6, wobble=0.5, falloff=0.4,
                                           seed=0, preview=True)
        coords, indices = _splat_placements(96, 96, patches.shape[1], patches.shape[2],
                                            patches.shape[0], seed=0)
        accum_s = np.zeros((96, 96, 3), dtype=np.float32)
        weight_s = np.zeros((96, 96), dtype=np.float32)
        splat_accumulate_jit(accum_s, weight_s, patches, masks, coords, indices)

        session = SplatCudaSession(96, 96, 3)
        session.accumulate(patches, masks, coords, indices)
        gpu_accum = session.accum_d.copy_to_host()
        gpu_weight = session.weight_d.copy_to_host()

        # Looser tolerance than CPU-vs-CPU: GPU atomics resolve contributions
        # in a hardware-scheduled, non-deterministic order, so this is a
        # numerically-equivalent check, not an order-preserving one.
        np.testing.assert_allclose(gpu_accum, accum_s, rtol=1e-3, atol=1e-2)
        np.testing.assert_allclose(gpu_weight, weight_s, rtol=1e-3, atol=1e-2)

    @requires_cuda
    def test_gpu_resolve_matches_cpu(self):
        from app.core.materialize_methods_cuda import SplatCudaSession, warmup_cuda_kernels
        assert warmup_cuda_kernels()

        source = _make_source(96)
        patches, masks = _build_patch_bank(source, scale=0.4, rotation=0.0,
                                           rand_rot=0.6, wobble=0.5, falloff=0.4,
                                           seed=0, preview=True)
        coords, indices = _splat_placements(96, 96, patches.shape[1], patches.shape[2],
                                            patches.shape[0], seed=0)
        accum_s = np.zeros((96, 96, 3), dtype=np.float32)
        weight_s = np.zeros((96, 96), dtype=np.float32)
        splat_accumulate_jit(accum_s, weight_s, patches, masks, coords, indices)
        out_s = np.empty_like(accum_s)
        splat_resolve_jit(accum_s, weight_s, source, out_s)

        session = SplatCudaSession(96, 96, 3)
        session.accumulate(patches, masks, coords, indices)
        out_g = session.resolve(np.ascontiguousarray(source),
                                np.empty((96, 96, 3), dtype=np.float32))

        np.testing.assert_allclose(out_g, out_s, rtol=1e-3, atol=1e-2)


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
