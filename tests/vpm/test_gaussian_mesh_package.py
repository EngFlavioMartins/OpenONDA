"""Optional package extraction/lifecycle qualification; no solver admission."""

import numpy as np
import pytest

from source.solvers.vpm.physics.induction.gaussian_mesh.coordinates import (
    finite_images,
    slab_coordinates,
)
from source.solvers.vpm.physics.induction.gaussian_mesh.fields import GaussianImageFields


def _case():
    x = np.array([[-.082, .012, .023], [-.042, .012, .023], [-.002, .012, .023],
                  [.038, .012, .023], [.078, .012, .023]])
    gamma = np.tile([.3, -.2, .7], (5, 1))*np.array([1, -4, 6, -4, 1])[:, None]
    sigma = np.array([.035, .04, .05, .045, .04])
    q = np.array([[.012, .014, .024], [-.068, .053, .061], [.097, -.036, .04]])
    images = [(0, True), (-1, False), (1, False), (1, True)]
    return x, gamma, sigma, q, images


def test_extracted_cuda_formulas_and_coordinate_maps_are_unchanged():
    from source.solvers.vpm.physics.induction.gaussian_mesh import correction, fields, stencil
    from tests.vpm import _cupy_cardinal_stencil as old_stencil
    from tests.vpm import _cupy_slab_field_mesh as old_mesh
    from tests.vpm import _cupy_slab_field_mesh_fused as old_fused
    from tests.vpm import _gaussian_core_correction_gpu as old_correction
    from tests.vpm._finite_slab_field_mesh_reference import slab_coordinates as old_coordinates

    assert fields._CUDA == old_mesh._CUDA
    assert fields._FUSED_CUDA == old_fused._FUSED_CUDA
    for dtype in ("float32", "float64"):
        assert correction._kernel_source(dtype) == old_correction._kernel_source(dtype)
        for order in (4, 6, 8, 10):
            assert stencil._source(dtype, order) == old_stencil._source(dtype, order)
    x, _, _, _, _ = _case()
    for translation in (0., 1024.3):
        points = x.copy()
        points[:, 2] += translation
        new = slab_coordinates(points, translation, translation+.193, .035)
        old = old_coordinates(points, translation, translation+.193, .035)
        for a, b in zip(new, old, strict=True):
            np.testing.assert_array_equal(a, b)


def test_primary_permission_and_bounded_descriptor_consumption():
    with pytest.raises(ValueError, match="primary"):
        finite_images([(0, False)], 0., .193, 6, 4)
    saved, world, integer = finite_images([(0, False)], 0., .193, 6, 4, include_primary=True)
    assert saved == ((0, False),) and world == ((0., False),) and integer == ((0, False),)
    for descriptors in ([(0, True)], [(0, False), (0, False)]):
        with pytest.raises(ValueError, match="exactly once"):
            finite_images(descriptors, 0., .193, 6, 4, include_primary=True)
    visits = []

    def endless():
        while True:
            visits.append(1)
            if len(visits) > 5:
                raise AssertionError("unbounded descriptor consumption")
            yield 0, True

    with pytest.raises(ValueError, match="cap"):
        finite_images(endless(), 0., .193, 6, 4)
    assert len(visits) == 5


def test_host_admission_precedes_optional_cuda_dependency():
    x, gamma, sigma, q, _ = _case()
    with pytest.raises(MemoryError, match="combined"):
        GaussianImageFields(x, gamma, sigma, q, zmin=0., zmax=.193,
                            tau=.12, spacing=.035, cutoff=.6, max_total_bytes=1)
    with pytest.raises(ValueError, match="core"):
        GaussianImageFields(x, gamma, np.zeros_like(sigma), q, zmin=0., zmax=.193,
                            tau=.12, spacing=.035, cutoff=.6)


def test_host_snapshots_cannot_be_made_writable():
    original = np.arange(15., dtype=np.float64).reshape(5, 3)
    snapshot = GaussianImageFields._host_array(original, "source")
    saved = snapshot.copy()
    original[:] = -1
    np.testing.assert_array_equal(snapshot, saved)
    for value in (snapshot, snapshot.reshape(-1), snapshot.base):
        assert not value.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE"):
            value.setflags(write=True)
    with pytest.raises(ValueError, match="read-only"):
        snapshot[0, 0] = 10.


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
    limit = 2**bits-owner.shape[2]+1
    for sign in (-1, 1):
        owner._admit_integer_images([(sign*limit, False)])
        with pytest.raises(ValueError, match="image lags"):
            owner._admit_integer_images([(sign*(limit+1), False)])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_extracted_field_matches_frozen_compact_and_direct(dtype):
    cp = pytest.importorskip("cupy")
    from tests.vpm._cupy_slab_compact_fields import CompactSlabFieldMeshGPU
    from tests.vpm._finite_image_mesh_reference import direct_finite_images

    x, gamma, sigma, q, images = _case()
    with CompactSlabFieldMeshGPU(x, gamma, sigma, q, zmin=0., zmax=.193,
                                 dtype=dtype, correction_dtype=dtype, stencil_backend="gpu") as old:
        u, j, _ = old.evaluate(images)
        expected = cp.asnumpy(u), cp.asnumpy(j)
        del u, j
    with GaussianImageFields(x, gamma, sigma, q, zmin=0., zmax=.193,
                             tau=.12, spacing=.035, cutoff=.6, dtype=dtype,
                             correction_dtype=dtype) as new:
        for snapshot in (new.host_x, new.host_gamma, new.host_targets, new._host_sigma):
            with pytest.raises(ValueError, match="WRITEABLE"):
                snapshot.setflags(write=True)
        u, j, report = new.evaluate(images)
        current = cp.asnumpy(u), cp.asnumpy(j)
        assert report["inverse_transforms"] == 0 and not report["tail_certified"]
        assert report["core_correction_included"]
        assert new._plans.work_bytes <= new.max_plan_bytes
        _, world, _ = finite_images(images, 0., .193, new.cells, new.max_images)
        truth = direct_finite_images(x, gamma, sigma, q, world)[:2]
        for got, prior, direct in zip(current, expected, truth, strict=True):
            assert np.linalg.norm(got-direct)/np.linalg.norm(direct) < 1e-4
            np.testing.assert_allclose(got, prior, rtol=16*np.finfo(dtype).eps,
                                       atol=16*np.finfo(dtype).eps*np.max(np.abs(direct)))
        new.prepare([(1, True)])
        later_u, later_j, _ = new.evaluate_prepared(q)
        np.testing.assert_array_equal(cp.asnumpy(u), current[0])
        np.testing.assert_array_equal(cp.asnumpy(j), current[1])
        assert not cp.shares_memory(u, later_u)
        with pytest.raises(ValueError, match="stencil"):
            new.evaluate_prepared(np.array([[100., 0., .02]]))
        del u, j, later_u, later_j


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_explicit_fft_plans_preserve_foreign_cache_and_live_buffers(dtype):
    cp = pytest.importorskip("cupy")
    from cupyx.scipy.fft import get_fft_plan

    x, gamma, sigma, q, images = _case()
    cache = cp.fft.config.get_plan_cache()
    with GaussianImageFields(x, gamma, sigma, q, zmin=0., zmax=.193,
                             tau=.12, spacing=.035, cutoff=.6, dtype=dtype,
                             correction_dtype=dtype) as owner:
        # Populate the FOREIGN global cache with this exact transform shape;
        # get_fft_plan would borrow this plan, while our adapter must not.
        foreign = cp.ones(owner.fft_shape, dtype=dtype)
        sentinel = cp.fft.rfftn(foreign)
        foreign_plan = get_fft_plan(foreign, axes=(0, 1, 2), value_type="R2C")
        before = (cache.get_curr_size(), cache.get_curr_memsize(), cache.get_size(), cache.get_memsize())
        allocator = cp.cuda.get_allocator()
        owner.prepare(images)
        assert owner._plans.forward is not foreign_plan
        assert owner._plans.forward.handle != foreign_plan.handle
        assert owner._plans.forward.work_area.mem is not foreign_plan.work_area.mem
        u, j, _ = owner.evaluate_prepared(q)
        owner.close()
        assert (cache.get_curr_size(), cache.get_curr_memsize(), cache.get_size(), cache.get_memsize()) == before
        assert cp.cuda.get_allocator() is allocator
        np.testing.assert_array_equal(cp.asnumpy(foreign), 1.)
        np.testing.assert_array_equal(cp.asnumpy(cp.fft.rfftn(foreign)), cp.asnumpy(sentinel))
        assert bool(cp.isfinite(u).all() & cp.isfinite(j).all())
        del u, j, foreign, sentinel, foreign_plan
    # This test owns these global cached plans; package teardown above did
    # not touch them. Remove test-created cache state before prototype tests.
    cache.clear()


def test_plan_allocation_and_mid_capture_failures_revoke_fields(monkeypatch):
    cp = pytest.importorskip("cupy")
    x, gamma, sigma, q, images = _case()
    with GaussianImageFields(x, gamma, sigma, q, zmin=0., zmax=.193,
                             tau=.12, spacing=.035, cutoff=.6, max_plan_bytes=1) as owner:
        with pytest.raises(MemoryError, match="plan cap"):
            owner.prepare(images)
        assert owner._plans is None
        with pytest.raises(RuntimeError, match="prepared"):
            owner.evaluate_prepared(q)
    with GaussianImageFields(x, gamma, sigma, q, zmin=0., zmax=.193,
                             tau=.12, spacing=.035, cutoff=.6) as owner:
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
    from tests.vpm._finite_image_mesh_reference import direct_finite_images

    x, gamma, sigma, q, images = _case()
    with GaussianImageFields(x, gamma, sigma, q, zmin=0., zmax=.193,
                             tau=.12, spacing=.035, cutoff=.6) as owner, pytest.raises(ValueError, match="primary"):
        owner.prepare([(0, False), *images])
    with GaussianImageFields(x, gamma, sigma, q, zmin=0., zmax=.193,
                             tau=.12, spacing=.035, cutoff=.6, source_only_primary=True) as owner:
        u, j, report = owner.evaluate([(0, False), *images])
        _, world, _ = finite_images([(0, False), *images], 0., .193, owner.cells,
                                    owner.max_images, include_primary=True)
        truth = direct_finite_images(x, gamma, sigma, q, world)[:2]
        for actual, exact in zip((cp.asnumpy(u), cp.asnumpy(j)), truth, strict=True):
            assert np.linalg.norm(actual-exact)/np.linalg.norm(exact) < 1e-4
        assert not report["tail_certified"]
        del u, j
