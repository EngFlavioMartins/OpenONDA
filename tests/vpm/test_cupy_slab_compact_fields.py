"""Bounded compact-grid reuse qualification; no runtime or tail admission."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from tests.vpm._cupy_slab_compact_fields import CompactSlabFieldMeshGPU


def _case():
    x = np.array([[-.082, .012, .023], [-.042, .012, .023], [-.002, .012, .023],
                  [.038, .012, .023], [.078, .012, .023]])
    gamma = np.tile([.3, -.2, .7], (5, 1))*np.array([1, -4, 6, -4, 1])[:, None]
    sigma = np.array([.035, .04, .05, .045, .04])
    q = np.array([[-.15, -.1, .015], [.15, .1, .16], [.011, .019, .018]])
    new = np.array([[.012, .014, .024], [-.068, .053, .061], [.097, -.036, .04]])
    images = [(0, True), (-1, False), (1, False), (1, True)]
    return x, gamma, sigma, q, new, images


def test_combined_owner_cap_rejected_before_runtime_import():
    x, gamma, sigma, q, _, _ = _case()
    with pytest.raises(MemoryError, match="combined"):
        CompactSlabFieldMeshGPU(x, gamma, sigma, q, zmin=0., zmax=.193,
                               max_total_bytes=128*1024**2)


def test_descriptor_snapshot_stops_at_first_over_cap_without_runtime():
    owner = object.__new__(CompactSlabFieldMeshGPU)
    owner._admit = lambda: None
    owner.max_images = 3
    visited = []

    def descriptors():
        for k in range(4):
            visited.append(k)
            yield k, True
        raise AssertionError("must not consume beyond max_images+1")

    with pytest.raises(ValueError, match="image count"):
        owner.prepare(descriptors())
    assert visited == [0, 1, 2, 3]
    assert owner._prepared_images is None


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_new_queries_match_fresh_owner_and_direct_without_source_rebuild(dtype):
    cp = pytest.importorskip("cupy")
    from tests.vpm._cupy_slab_field_mesh_fused import SlabFieldMeshFusedGPU
    from tests.vpm._finite_image_mesh_reference import direct_finite_images
    from tests.vpm._finite_slab_field_mesh_reference import slab_world_images
    from tests.vpm._gaussian_core_correction_gpu import GaussianCoreCorrectionGPU

    x, gamma, sigma, q, new, images = _case()
    originals = tuple(a.copy() for a in (x, gamma, sigma, q, new))
    with CompactSlabFieldMeshGPU(x, gamma, sigma, q, zmin=0., zmax=.193,
                                 dtype=dtype, correction_dtype=dtype,
                                 stencil_backend="gpu") as engine:
        preparation = engine.prepare(images)
        compact_pointer = engine._compact_fields.data.ptr
        source_pointers = {odd: tuple(array.data.ptr for array in spectra)
                           for odd, spectra in engine.spectra.items()}
        for targets in (q, new, new[::-1], np.empty((0, 3))):
            u, j, diagnostics = engine.evaluate_prepared(targets)
            assert diagnostics["inverse_transforms"] == diagnostics["source_scatters"] == 0
            assert not diagnostics["runtime_admissible"] and not diagnostics["tail_certified"]
            assert not preparation["tail_certified"]
            assert engine._compact_fields.data.ptr == compact_pointer
            assert {odd: tuple(a.data.ptr for a in arrays) for odd, arrays in engine.spectra.items()} == source_pointers
            assert not cp.shares_memory(u, engine._compact_fields)
            assert not cp.shares_memory(j, engine._compact_fields)
            if targets is new:
                retained_u, retained_j = u, j
                saved = cp.asnumpy(u), cp.asnumpy(j)
        u2, j2, _ = engine.evaluate_prepared(new)
        np.testing.assert_array_equal(cp.asnumpy(retained_u), saved[0])
        np.testing.assert_array_equal(cp.asnumpy(retained_j), saved[1])
        np.testing.assert_array_equal(cp.asnumpy(u2), saved[0])
        np.testing.assert_array_equal(cp.asnumpy(j2), saved[1])
        assert not cp.shares_memory(u2, retained_u)
        del u, j, u2, j2, retained_u, retained_j
    world, _ = slab_world_images(images, 0., .193, 6)
    with SlabFieldMeshFusedGPU(x, gamma, new, zmin=0., zmax=.193, dtype=dtype,
                              stencil_backend="gpu") as fresh:
        smooth_u, smooth_j, _ = fresh.evaluate(images)
        with GaussianCoreCorrectionGPU(x, gamma, sigma, tau=.12, cutoff=.6,
                                        accumulation_dtype=dtype) as correction:
            cu, cj, _ = correction.evaluate(new, world)
            expected = cp.asnumpy(smooth_u+cu), cp.asnumpy(smooth_j+cj)
            del cu, cj
        del smooth_u, smooth_j
    truth = direct_finite_images(x, gamma, sigma, new, world)[:2]
    for cached, other, direct in zip(saved, expected, truth, strict=True):
        assert np.linalg.norm(cached-direct)/np.linalg.norm(direct) < 1e-4
        # Fresh FFT shapes can differ; no claim of bitwise equality across
        # different cuFFT plans or separately scattered density fields.
        allowance = 16*np.finfo(dtype).eps*np.linalg.norm(direct)
        assert np.linalg.norm(cached-other) <= allowance
    for array, old in zip((x, gamma, sigma, q, new), originals, strict=True):
        np.testing.assert_array_equal(array, old)


def test_bounds_snapshot_alias_and_owner_contracts():
    cp = pytest.importorskip("cupy")
    x, gamma, sigma, q, new, images = _case()
    with CompactSlabFieldMeshGPU(x, gamma, sigma, q, zmin=0., zmax=.193,
                                 stencil_backend="gpu") as engine:
        with pytest.raises(RuntimeError, match="prepared"):
            engine.evaluate_prepared(new)
        engine.prepare(images)
        first_u, first_j, _ = engine.evaluate_prepared(new)
        before = cp.asnumpy(first_u), cp.asnumpy(first_j)
        # All constructor arrays and descriptors were copied. Caller mutation
        # cannot change the prepared source field or owned correction index.
        x[:] = 9
        gamma[:] = 12
        sigma[:] = .09
        q[:] = 15
        images.clear()
        again_u, again_j, _ = engine.evaluate_prepared(new)
        np.testing.assert_array_equal(cp.asnumpy(again_u), before[0])
        np.testing.assert_array_equal(cp.asnumpy(again_j), before[1])
        with pytest.raises(ValueError, match="stencil"):
            engine.evaluate_prepared(np.array([[999., 0., .02]]))
        with pytest.raises(TypeError, match="host"):
            engine.evaluate_prepared(first_u)
        with cp.cuda.Stream(), pytest.raises(RuntimeError, match="stream"):
            engine.evaluate_prepared(new)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(engine.evaluate_prepared, new)
            with pytest.raises(RuntimeError, match="thread"):
                future.result()
        engine.prepare([(1, True)])
        later_u, later_j, _ = engine.evaluate_prepared(new)
        np.testing.assert_array_equal(cp.asnumpy(first_u), before[0])
        np.testing.assert_array_equal(cp.asnumpy(first_j), before[1])
        assert not cp.shares_memory(later_u, first_u)
        del first_u, first_j, again_u, again_j, later_u, later_j
        engine.close()
        engine.close()
        with pytest.raises(RuntimeError, match="closed"):
            engine.evaluate_prepared(new)
    assert engine._compact_fields is None and engine._correction is None
    assert cp.fft.config.get_plan_cache().get_curr_size() == 0


def test_capture_failure_revokes_cache_without_overwriting_old_outputs(monkeypatch):
    cp = pytest.importorskip("cupy")
    x, gamma, sigma, q, new, images = _case()
    with CompactSlabFieldMeshGPU(x, gamma, sigma, q, zmin=0., zmax=.193) as engine:
        engine.prepare(images)
        u, j, _ = engine.evaluate_prepared(new)
        before = cp.asnumpy(u), cp.asnumpy(j)
        old_launch = engine._launch

        def fail_column(name, count, arguments):
            if name == "gather" and int(arguments[7]) == 3:
                raise MemoryError("injected incomplete compact capture")
            return old_launch(name, count, arguments)

        monkeypatch.setattr(engine, "_launch", fail_column)
        with pytest.raises(MemoryError, match="injected"):
            engine.prepare([(1, True)])
        assert engine._capture_columns is None
        with pytest.raises(RuntimeError, match="prepared"):
            engine.evaluate_prepared(new)
        np.testing.assert_array_equal(cp.asnumpy(u), before[0])
        np.testing.assert_array_equal(cp.asnumpy(j), before[1])
        monkeypatch.setattr(engine, "_launch", old_launch)
        engine.prepare(images)
        del u, j


def test_constructor_correction_failure_cleans_compact_and_fft_owners():
    cp = pytest.importorskip("cupy")
    x, gamma, sigma, q, _, _ = _case()
    sigma[0] = .2
    with pytest.raises(ValueError, match="source core"):
        CompactSlabFieldMeshGPU(x, gamma, sigma, q, zmin=0., zmax=.193)
    assert cp.fft.config.get_plan_cache().get_curr_size() == 0
