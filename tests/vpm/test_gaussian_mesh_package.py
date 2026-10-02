"""Current Gaussian field lifecycle and independent direct qualification."""

import numpy as np
import pytest

from source.solvers.vpm.physics.induction.gaussian_mesh.coordinates import (
    finite_images,
)
from source.solvers.vpm.physics.induction.gaussian_mesh.fields import GaussianImageFields


def _case():
    x = np.array(
        [
            [-0.082, 0.012, 0.023],
            [-0.042, 0.012, 0.023],
            [-0.002, 0.012, 0.023],
            [0.038, 0.012, 0.023],
            [0.078, 0.012, 0.023],
        ]
    )
    gamma = np.tile([0.3, -0.2, 0.7], (5, 1)) * np.array([1, -4, 6, -4, 1])[:, None]
    sigma = np.array([0.035, 0.04, 0.05, 0.045, 0.04])
    q = np.array([[0.012, 0.014, 0.024], [-0.068, 0.053, 0.061], [0.097, -0.036, 0.04]])
    images = [(0, True), (-1, False), (1, False), (1, True)]
    return x, gamma, sigma, q, images


def test_primary_permission_and_bounded_descriptor_consumption():
    with pytest.raises(ValueError, match="primary"):
        finite_images([(0, False)], 0.0, 0.193, 6, 4)
    saved, world, integer = finite_images([(0, False)], 0.0, 0.193, 6, 4, include_primary=True)
    assert saved == ((0, False),) and world == ((0.0, False),) and integer == ((0, False),)
    for descriptors in ([(0, True)], [(0, False), (0, False)]):
        with pytest.raises(ValueError, match="exactly once"):
            finite_images(descriptors, 0.0, 0.193, 6, 4, include_primary=True)
    visits = []

    def endless():
        while True:
            visits.append(1)
            if len(visits) > 5:
                raise AssertionError("unbounded descriptor consumption")
            yield 0, True

    with pytest.raises(ValueError, match="cap"):
        finite_images(endless(), 0.0, 0.193, 6, 4)
    assert len(visits) == 5


def test_host_admission_precedes_optional_cuda_dependency():
    x, gamma, sigma, q, _ = _case()
    with pytest.raises(MemoryError, match="combined"):
        GaussianImageFields(
            x,
            gamma,
            sigma,
            q,
            zmin=0.0,
            zmax=0.193,
            tau=0.12,
            spacing=0.035,
            cutoff=0.6,
            max_total_bytes=1,
        )
    with pytest.raises(ValueError, match="core"):
        GaussianImageFields(
            x,
            gamma,
            np.zeros_like(sigma),
            q,
            zmin=0.0,
            zmax=0.193,
            tau=0.12,
            spacing=0.035,
            cutoff=0.6,
        )


def test_host_snapshots_cannot_be_made_writable():
    original = np.arange(15.0, dtype=np.float64).reshape(5, 3)
    snapshot = GaussianImageFields._host_array(original, "source")
    saved = snapshot.copy()
    original[:] = -1
    np.testing.assert_array_equal(snapshot, saved)
    for value in (snapshot, snapshot.reshape(-1), snapshot.base):
        assert not value.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE"):
            value.setflags(write=True)
    with pytest.raises(ValueError, match="read-only"):
        snapshot[0, 0] = 10.0


def test_failed_gpu_drain_cannot_be_reclassified_as_completed_cleanup():
    from types import SimpleNamespace

    failure = RuntimeError("failed owned stream drain")

    def failed_drain():
        raise failure

    owner = GaussianImageFields.__new__(GaussianImageFields)
    owner.closed, owner._cleanup_failure = False, None
    owner._owner = SimpleNamespace(admit=lambda: None)
    owner.stream = SimpleNamespace(synchronize=failed_drain)
    with pytest.raises(RuntimeError, match="failed owned stream drain"):
        owner.close()
    assert owner.closed and owner._cleanup_failure is failure
    with pytest.raises(RuntimeError, match="cleanup remains uncertain") as retry:
        owner.close()
    assert retry.value.__cause__ is failure


@pytest.mark.parametrize("dtype,bits", [("float32", 24), ("float64", 53)])
def test_kernel_coordinate_guard_includes_all_signed_lags(dtype, bits):
    owner = GaussianImageFields.__new__(GaussianImageFields)
    owner.dtype, owner.shape = np.dtype(dtype), (17, 18, 19)
    limit = 2**bits - owner.shape[2] + 1
    for sign in (-1, 1):
        owner._admit_integer_images([(sign * limit, False)])
        with pytest.raises(ValueError, match="image lags"):
            owner._admit_integer_images([(sign * (limit + 1), False)])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cuda_fields_match_independent_direct_gaussian_sums(dtype):
    cp = pytest.importorskip("cupy")
    from tests.vpm._direct_gaussian_reference import direct_finite_images

    x, gamma, sigma, q, images = _case()
    with GaussianImageFields(
        x,
        gamma,
        sigma,
        q,
        zmin=0.0,
        zmax=0.193,
        tau=0.12,
        spacing=0.035,
        cutoff=0.6,
        dtype=dtype,
        correction_dtype=dtype,
    ) as new:
        for snapshot in (new.host_x, new.host_gamma, new.host_targets, new._host_sigma):
            with pytest.raises(ValueError, match="WRITEABLE"):
                snapshot.setflags(write=True)
        u, j, report = new.evaluate(images)
        current = cp.asnumpy(u), cp.asnumpy(j)
        assert report["inverse_transforms"] == 0 and not report["tail_certified"]
        assert report["core_correction_included"]
        assert new._plans.work_bytes <= new.max_plan_bytes
        _, world, _ = finite_images(images, 0.0, 0.193, new.cells, new.max_images)
        truth = direct_finite_images(x, gamma, sigma, q, world)[:2]
        for got, direct in zip(current, truth, strict=True):
            assert np.linalg.norm(got - direct) / np.linalg.norm(direct) < 1e-4
        new.prepare([(1, True)])
        later_u, later_j, _ = new.evaluate_prepared(q)
        np.testing.assert_array_equal(cp.asnumpy(u), current[0])
        np.testing.assert_array_equal(cp.asnumpy(j), current[1])
        assert not cp.shares_memory(u, later_u)
        with pytest.raises(ValueError, match="stencil"):
            new.evaluate_prepared(np.array([[100.0, 0.0, 0.02]]))
        del u, j, later_u, later_j


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_native_channel_subsets_preserve_scalar_radial_fields(dtype):
    cp = pytest.importorskip("cupy")
    from source.solvers.vpm.physics.induction.gaussian_mesh.fields import _CHANNEL_COMPONENTS

    x, gamma, sigma, q, _ = _case()
    with (
        GaussianImageFields(
            x,
            gamma,
            sigma,
            q,
            zmin=0.0,
            zmax=0.193,
            tau=0.12,
            spacing=0.035,
            cutoff=0.6,
            dtype=dtype,
        ) as owner,
        owner._owner.allocation_scope(),
    ):
        # Include coincidence, the series branch, distant signed lags,
        # and padding. Test a reordered subset as well as all channels.
        shifts = cp.asarray([0, -1, 7, -1200, 1200], dtype=cp.int64)
        full = cp.empty((9, *owner.fft_shape), dtype=owner.dtype)
        arguments = (
            np.int64(owner.volume),
            *(np.int32(n) for n in owner.shape),
            *(np.int32(n) for n in owner.fft_shape),
            *(owner.real_type(h) for h in owner.steps),
            shifts,
            np.int32(len(shifts)),
            owner.real_type(owner.tau),
        )
        owner._launch("gaussian_kernel_fused", owner.volume, (*arguments, full))
        for selected in (tuple(range(9)), (8, 0, 4)):
            channels = cp.asarray(selected, dtype=cp.int32)
            subset = cp.empty((len(selected), *owner.fft_shape), dtype=owner.dtype)
            owner._launch(
                "gaussian_kernel_batch",
                owner.volume,
                (*arguments, channels, np.int32(len(selected)), subset),
            )
            np.testing.assert_array_equal(cp.asnumpy(subset), cp.asnumpy(full[list(selected)]))
            del subset, channels
        allowance = 12 * np.finfo(dtype).eps
        for channel, components in enumerate(_CHANNEL_COMPONENTS):
            for axis, derivative in components:
                scalar = cp.empty(owner.fft_shape, dtype=owner.dtype)
                owner._launch(
                    "gaussian_kernel",
                    owner.volume,
                    (*arguments, np.int32(axis), np.int32(derivative), scalar),
                )
                original = cp.asnumpy(scalar)
                np.testing.assert_allclose(
                    cp.asnumpy(full[channel]),
                    original,
                    rtol=allowance,
                    atol=allowance * np.max(np.abs(original)),
                )
                del scalar
        del full, shifts


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_explicit_fft_plans_preserve_foreign_cache_and_live_buffers(dtype):
    cp = pytest.importorskip("cupy")
    from cupyx.scipy.fft import get_fft_plan

    x, gamma, sigma, q, images = _case()
    cache = cp.fft.config.get_plan_cache()
    with GaussianImageFields(
        x,
        gamma,
        sigma,
        q,
        zmin=0.0,
        zmax=0.193,
        tau=0.12,
        spacing=0.035,
        cutoff=0.6,
        dtype=dtype,
        correction_dtype=dtype,
    ) as owner:
        # Populate the FOREIGN global cache with this exact transform shape;
        # get_fft_plan would borrow this plan, while our adapter must not.
        foreign = cp.ones(owner.fft_shape, dtype=dtype)
        sentinel = cp.fft.rfftn(foreign)
        foreign_plan = get_fft_plan(foreign, axes=(0, 1, 2), value_type="R2C")
        before = (
            cache.get_curr_size(),
            cache.get_curr_memsize(),
            cache.get_size(),
            cache.get_memsize(),
        )
        allocator = cp.cuda.get_allocator()
        owner.prepare(images)
        assert owner._plans.forward is not foreign_plan
        assert owner._plans.forward.handle != foreign_plan.handle
        assert owner._plans.forward.work_area.mem is not foreign_plan.work_area.mem
        u, j, _ = owner.evaluate_prepared(q)
        owner.close()
        assert (
            cache.get_curr_size(),
            cache.get_curr_memsize(),
            cache.get_size(),
            cache.get_memsize(),
        ) == before
        assert cp.cuda.get_allocator() is allocator
        np.testing.assert_array_equal(cp.asnumpy(foreign), 1.0)
        np.testing.assert_array_equal(cp.asnumpy(cp.fft.rfftn(foreign)), cp.asnumpy(sentinel))
        assert bool(cp.isfinite(u).all() & cp.isfinite(j).all())
        del u, j, foreign, sentinel, foreign_plan
    # This test owns these global cached plans; package teardown above did
    # not touch them. Remove test-created cache state before later CUDA tests.
    cache.clear()


def test_plan_allocation_and_mid_capture_failures_revoke_fields(monkeypatch):
    cp = pytest.importorskip("cupy")
    x, gamma, sigma, q, images = _case()
    with GaussianImageFields(
        x,
        gamma,
        sigma,
        q,
        zmin=0.0,
        zmax=0.193,
        tau=0.12,
        spacing=0.035,
        cutoff=0.6,
        max_plan_bytes=1,
    ) as owner:
        with pytest.raises(MemoryError, match="plan cap"):
            owner.prepare(images)
        assert owner._plans is None
        with pytest.raises(RuntimeError, match="prepared"):
            owner.evaluate_prepared(q)
    with GaussianImageFields(
        x, gamma, sigma, q, zmin=0.0, zmax=0.193, tau=0.12, spacing=0.035, cutoff=0.6
    ) as owner:
        owner.prepare(images)
        u, j, _ = owner.evaluate_prepared(q)
        saved = cp.asnumpy(u), cp.asnumpy(j)
        original = owner._plans.irfft
        calls = []

        def failed_inverse(array, output):
            calls.append(1)
            if len(calls) == 4:
                raise RuntimeError("injected inverse failure")
            return original(array, output)

        monkeypatch.setattr(owner._plans, "irfft", failed_inverse)
        with pytest.raises(RuntimeError, match="injected"):
            owner.prepare([(1, True)])
        with pytest.raises(RuntimeError, match="prepared"):
            owner.evaluate_prepared(q)
        np.testing.assert_array_equal(cp.asnumpy(u), saved[0])
        np.testing.assert_array_equal(cp.asnumpy(j), saved[1])
        del u, j


def test_explicit_source_only_primary_is_not_a_particle_core_shortcut():
    cp = pytest.importorskip("cupy")
    from tests.vpm._direct_gaussian_reference import direct_finite_images

    x, gamma, sigma, q, images = _case()
    with (
        GaussianImageFields(
            x, gamma, sigma, q, zmin=0.0, zmax=0.193, tau=0.12, spacing=0.035, cutoff=0.6
        ) as owner,
        pytest.raises(ValueError, match="primary"),
    ):
        owner.prepare([(0, False), *images])
    with GaussianImageFields(
        x,
        gamma,
        sigma,
        q,
        zmin=0.0,
        zmax=0.193,
        tau=0.12,
        spacing=0.035,
        cutoff=0.6,
        source_only_primary=True,
    ) as owner:
        u, j, report = owner.evaluate([(0, False), *images])
        _, world, _ = finite_images(
            [(0, False), *images], 0.0, 0.193, owner.cells, owner.max_images, include_primary=True
        )
        truth = direct_finite_images(x, gamma, sigma, q, world)[:2]
        for actual, exact in zip((cp.asnumpy(u), cp.asnumpy(j)), truth, strict=True):
            assert np.linalg.norm(actual - exact) / np.linalg.norm(exact) < 1e-4
        assert not report["tail_certified"]
        del u, j
