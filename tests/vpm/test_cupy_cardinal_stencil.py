"""New helper only; launch CUDA tests exclusively in the assigned GPU slot."""

import numpy as np
import pytest

from tests.vpm._cupy_cardinal_stencil import _parameters, _source, cardinal_stencil_gpu
from tests.vpm._cupy_slab_field_mesh import _weights
from tests.vpm._finite_slab_field_mesh_reference import slab_coordinates


def test_parameter_admission_and_generated_code_are_bounded():
    origin = np.array([-11., -12., -13.])
    result = _parameters(origin, 10, (40, 41, 42), "float64", 100)
    assert result[1:] == (10, (40, 41, 42), np.dtype("float64"), 100)
    assert result[0] is not origin
    for order in (True, 3, 12, 4.5):
        with pytest.raises(ValueError):
            _parameters(origin, order, (40, 41, 42), "float64", 100)
    for shape in ((40, 41), (40, 41, 4), (40, True, 42), (40, 41, 2**31)):
        with pytest.raises(ValueError):
            _parameters(origin, 10, shape, "float64", 100)
    for order in (4, 6, 8, 10):
        for dtype in ("float32", "float64"):
            source = _source(dtype, order)
            assert "ORDER" not in source and "WEIGHT" not in source
            assert "atomicOr(error,2)" in source


@pytest.fixture(scope="module")
def cupy_runtime():
    cp = pytest.importorskip("cupy")
    assert cp.cuda.runtime.getDeviceCount() > 0
    return cp


@pytest.mark.parametrize("order", [4, 6, 8, 10])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_gpu_weights_match_ordered_host_product_and_cardinal_boundaries(cupy_runtime, order, dtype):
    cp = cupy_runtime
    rng = np.random.default_rng(2400+order)
    origin = np.array([-32., -64., -96.])
    points = origin+rng.uniform(order+1, 35, (41, 3))
    # Cardinal, negative signed zero, and immediately adjacent floating values.
    points[:4] = origin+np.array([[20., 21., 22.],
                                 [np.nextafter(20., 0), np.nextafter(21., 0), np.nextafter(22., 0)],
                                 [np.nextafter(20., np.inf), np.nextafter(21., np.inf), np.nextafter(22., np.inf)],
                                 [10., 10.5, 11.25]])
    original = points.copy()
    first, expected = _weights(points, origin, order, (60, 61, 62))
    pool = cp.cuda.MemoryPool()
    pool.set_limit(size=32*1024**2)
    entered_allocator = cp.cuda.get_allocator()
    first_d, weights_d, report = cardinal_stencil_gpu(points, origin, order, (60, 61, 62), dtype=dtype, pool=pool)
    np.testing.assert_array_equal(cp.asnumpy(first_d), first)
    actual = cp.asnumpy(weights_d)
    epsilon = np.finfo(dtype).eps
    np.testing.assert_allclose(actual, expected.astype(dtype), rtol=8*epsilon, atol=2*epsilon)
    np.testing.assert_allclose(actual.sum(axis=2), 1, rtol=0, atol=8*epsilon)
    np.testing.assert_array_equal(points, original)
    assert cp.cuda.get_allocator() == entered_allocator
    assert report["pool_reserved_after"] <= 32*1024**2
    # Same-device immutable normalized input bypasses a redundant coordinate copy.
    device_points = cp.asarray(points)
    before = device_points.copy()
    first_again, weights_again, again = cardinal_stencil_gpu(device_points, origin, order, (60, 61, 62), dtype=dtype, pool=pool)
    np.testing.assert_array_equal(cp.asnumpy(first_again), first)
    np.testing.assert_array_equal(cp.asnumpy(weights_again), actual)
    np.testing.assert_array_equal(cp.asnumpy(device_points), cp.asnumpy(before))
    assert again["borrowed_f64_device_coordinates"]


@pytest.mark.parametrize("offset", [0., 1024., -65536.])
def test_gpu_preserves_normalized_translated_slab_planes_and_reflections(cupy_runtime, offset):
    cp = cupy_runtime
    zmin, zmax, spacing = offset+.081, offset+.794, .035
    physical = np.array([[.017, -.023, zmin], [-.073, .027, zmax], [.111, .089, (zmin+zmax)/2]])
    before = physical.copy()
    lattice, _, count = slab_coordinates(physical, zmin, zmax, spacing)
    reflected = lattice.copy()
    reflected[:, 2] *= -1
    points = np.concatenate((lattice, reflected))
    origin = np.floor(points.min(0))-10
    shape = tuple((np.ceil(points.max(0)-origin)+11).astype(int))
    first, expected = _weights(points, origin, 10, shape)
    pool = cp.cuda.MemoryPool()
    pool.set_limit(size=8*1024**2)
    first_d, weights_d, _ = cardinal_stencil_gpu(points, origin, 10, shape, dtype="float64", pool=pool)
    np.testing.assert_array_equal(cp.asnumpy(first_d), first)
    np.testing.assert_allclose(cp.asnumpy(weights_d), expected, rtol=8*np.finfo(float).eps, atol=2e-16)
    np.testing.assert_array_equal(lattice[:2, 2], [-count/2, count/2])
    np.testing.assert_array_equal(physical, before)


def test_gpu_rejects_outside_huge_nonfinite_or_unbounded_storage_without_mutation(cupy_runtime):
    cp = cupy_runtime
    pool = cp.cuda.MemoryPool()
    pool.set_limit(size=8*1024**2)
    origin = np.zeros(3)
    for invalid in ([[0., 20., 20.]], [[49., 20., 20.]], [[np.nan, 20., 20.]], [[2**31, 20., 20.]]):
        points = np.array(invalid)
        original = points.copy()
        with pytest.raises(ValueError):
            cardinal_stencil_gpu(points, origin, 10, (50, 50, 50), pool=pool)
        np.testing.assert_array_equal(points, original)
    points = np.ones((3, 3))*20
    with pytest.raises(MemoryError):
        cardinal_stencil_gpu(points, origin, 10, (50, 50, 50), pool=pool, max_new_bytes=16)
    with pytest.raises(ValueError):
        cardinal_stencil_gpu(points, origin, 10, (50, 50, 50), pool=cp.cuda.MemoryPool())
    with pytest.raises(ValueError):
        cardinal_stencil_gpu(points.astype(complex), origin, 10, (50, 50, 50), pool=pool)
    first, weights, report = cardinal_stencil_gpu(np.empty((0, 3)), origin, 10, (50, 50, 50), pool=pool)
    assert first.shape == (0, 3) and weights.shape == (0, 3, 10) and report["point_count"] == 0
