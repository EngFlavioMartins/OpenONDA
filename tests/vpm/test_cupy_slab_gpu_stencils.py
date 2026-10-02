"""Complete-field parity and constructor rollback for UNWIRED GPU stencils."""

import numpy as np
import pytest

from tests.vpm._cupy_slab_field_mesh import SlabFieldMeshGPU, _weights
from tests.vpm._finite_slab_field_mesh_reference import finite_slab_field_mesh, slab_coordinates

cp = pytest.importorskip("cupy")


def _case():
    zmin, zmax = 1024.3, 1024.493
    x = np.array([[-.078, .012, zmin+.023], [-.038, .012, zmin+.023], [.002, .012, zmin+.023],
                  [.042, .012, zmin+.023], [.082, .012, zmin+.023], [.013, -.017, zmin], [.021, .031, zmax]])
    gamma = np.array([1, -4, 6, -4, 1, .3, -.2])[:, None]*np.array([.3, -.2, .7])
    q = np.array([[.011, .019, zmin+.018], [-.073, .058, zmin+.061], [.112, -.036, zmin+.04], x[5], x[6]])
    return x, gamma, q, zmin, zmax


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_complete_gpu_stencil_fields_preserve_cpu_reference_and_reflections(dtype):
    x, gamma, q, zmin, zmax = _case()
    before = [value.copy() for value in (x, gamma, q)]
    images = [(0, True), (1, True), (-1, False), (1, False)]
    reference = finite_slab_field_mesh(x, gamma, np.full(len(x), .12), q, images,
                                       zmin=zmin, zmax=zmax, tau=.12, spacing=.035, order=10)
    fields = {}
    for backend in ("cpu", "gpu"):
        with SlabFieldMeshGPU(x, gamma, q, zmin=zmin, zmax=zmax, dtype=dtype, stencil_backend=backend) as engine:
            source, _, _ = slab_coordinates(x, zmin, zmax, .035)
            target, _, _ = slab_coordinates(q, zmin, zmax, .035)
            reflected = source.copy()
            reflected[:, 2] *= -1
            expected_first, expected_weights = _weights(target, engine.origin, 10, engine.shape)
            np.testing.assert_array_equal(cp.asnumpy(engine._first), expected_first)
            np.testing.assert_allclose(cp.asnumpy(engine._weight), expected_weights.astype(dtype),
                                       rtol=8*np.finfo(dtype).eps, atol=2*np.finfo(dtype).eps)
            if backend == "gpu":
                assert not engine._host_stencils and engine._target_stencil is None
                for odd, points in ((False, source), (True, reflected)):
                    first, weights = engine._device_stencils[odd]
                    expected_first, expected_weights = _weights(points, engine.origin, 10, engine.shape)
                    np.testing.assert_array_equal(cp.asnumpy(first), expected_first)
                    np.testing.assert_allclose(cp.asnumpy(weights), expected_weights.astype(dtype),
                                               rtol=8*np.finfo(dtype).eps, atol=2*np.finfo(dtype).eps)
                del first, weights
            u, j, report = engine.evaluate(images)
            fields[backend] = cp.asnumpy(u), cp.asnumpy(j)
            assert report["stencil_backend"] == backend
            assert not engine._device_stencils
            for actual, truth in zip(fields[backend], (reference.smooth_velocity, reference.smooth_gradient), strict=True):
                assert np.linalg.norm(actual-truth)/np.linalg.norm(truth) < (1e-5 if dtype == "float32" else 1e-9)
            del u, j
    for candidate, control in zip(fields["gpu"], fields["cpu"], strict=True):
        assert np.linalg.norm(candidate-control)/np.linalg.norm(control) < (2e-6 if dtype == "float32" else 2e-13)
    for value, original in zip((x, gamma, q), before, strict=True):
        np.testing.assert_array_equal(value, original)


def test_gpu_stencil_constructor_failure_drains_private_fields_and_restores_fft_cache(monkeypatch):
    x, gamma, q, zmin, zmax = _case()
    cache = cp.fft.config.get_plan_cache()
    assert cache.get_curr_size() == 0
    original_limits = cache.get_size(), cache.get_memsize()
    entered_allocator = cp.cuda.get_allocator()
    original = SlabFieldMeshGPU._prepare_stencils

    def fail_after_build(self, *arguments):
        original(self, *arguments)
        assert len(self._device_stencils) == 2 and self._weight is not None
        raise MemoryError("injected after complete private GPU stencil preparation")

    monkeypatch.setattr(SlabFieldMeshGPU, "_prepare_stencils", fail_after_build)
    owner = object.__new__(SlabFieldMeshGPU)
    with pytest.raises(MemoryError, match="injected"):
        owner.__init__(x, gamma, q, zmin=zmin, zmax=zmax, stencil_backend="gpu")
    assert owner.closed and not owner._device_stencils and not owner.spectra
    assert owner._first is None and owner._weight is None
    assert owner.pool.used_bytes() == 0
    assert (cache.get_size(), cache.get_memsize()) == original_limits
    assert cache.get_curr_size() == 0
    assert cp.cuda.get_allocator() == entered_allocator
    owner.close()


def test_gpu_stencil_constructor_failed_device_bounds_do_not_publish_owner(monkeypatch):
    from tests.vpm import _cupy_slab_field_mesh as module

    x, gamma, q, zmin, zmax = _case()
    original = module.slab_coordinates
    calls = 0

    def inconsistent_coordinate_shape(*arguments, **options):
        nonlocal calls
        values, steps, count = original(*arguments, **options)
        calls += 1
        if calls == 2:
            values[0, 0] = np.nan
        return values, steps, count

    # Geometry admission must reject before creating any pooled owner.
    monkeypatch.setattr(module, "slab_coordinates", inconsistent_coordinate_shape)
    with pytest.raises(ValueError, match="shape"):
        SlabFieldMeshGPU(x, gamma, q, zmin=zmin, zmax=zmax, stencil_backend="gpu")
