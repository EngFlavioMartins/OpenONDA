"""Real CUDA/host finite-operator parity and allocation-only host recovery."""

import numpy as np
import pytest

from source.solvers.vpm.physics.induction.gaussian_mesh.execution import PortableGaussianImageFields

pytestmark = pytest.mark.gpu


def _case(source_only):
    x = np.array([[-0.075, 0.012, 0.023], [0.018, -0.023, 0.079], [0.086, 0.031, 0.15]])
    gamma = np.array([[0.3, -0.2, 0.7], [-0.25, 0.5, -0.4], [0.1, 0.2, 0.3]])
    sigma = np.array([0.035, 0.04, 0.05])
    q = np.array([[0.012, 0.014, 0.024], [-0.068, 0.053, 0.061], [0.097, -0.036, 0.04]])
    images = [(0, True), (-1, False), (1, False), (1, True)]
    if source_only:
        images.insert(0, (0, False))
    options = {
        "zmin": 0.0,
        "zmax": 0.193,
        "tau": 0.12,
        "spacing": 0.035,
        "cutoff": 0.6,
        "order": 10,
        "dtype": "float32",
        "correction_dtype": "float32",
        "source_only_primary": source_only,
        "max_scratch_bytes": 64 * 1024**2,
        "max_plan_bytes": 8 * 1024**2,
        "max_correction_bytes": 16 * 1024**2,
    }
    return x, gamma, sigma, q, images, options


def _assert_same_finite_fields(actual, expected):
    for got, truth in zip(actual, expected, strict=True):
        # Finite FFT/interpolation and float32 arithmetic have an independent
        # envelope; this comparison does not alter mathematical tail budgets.
        scale = max(float(np.max(np.abs(truth))), np.finfo("float32").tiny)
        np.testing.assert_allclose(got, truth, rtol=2e-5, atol=2e-5 * scale)


@pytest.mark.parametrize("source_only", [False, True])
def test_cuda_and_host_match_each_other_and_independent_direct_cloud(source_only):
    cp = pytest.importorskip("cupy")
    from source.solvers.vpm.physics.induction.gaussian_mesh.fields import GaussianImageFields
    from source.solvers.vpm.physics.induction.gaussian_mesh.host_fields import (
        GaussianHostImageFields,
    )
    from tests.vpm._direct_gaussian_reference import direct_finite_images

    x, gamma, sigma, q, images, options = _case(source_only)
    with GaussianImageFields(x, gamma, sigma, q, **options) as cuda:
        u, j, cuda_report = cuda.evaluate(images)
        cuda_fields = cp.asnumpy(u), cp.asnumpy(j)
        world_images = cuda._prepared_world_images
        del u, j
    with GaussianHostImageFields(x, gamma, sigma, q, **options) as host:
        u, j, host_report = host.evaluate(images)
        host_fields = u, j
        assert host._prepared_world_images == world_images
    direct = direct_finite_images(x, gamma, sigma, q, world_images)[:2]
    for actual in (cuda_fields, host_fields):
        for got, truth in zip(actual, direct, strict=True):
            assert np.linalg.norm(got - truth) / np.linalg.norm(truth) < 2e-5
    _assert_same_finite_fields(host_fields, cuda_fields)
    assert cuda_report["core_correction_included"] and host_report["core_correction_included"]


@pytest.mark.parametrize("source_only", [False, True])
def test_real_cuda_memory_validation_recovers_same_host_operator_without_touching_default_pool(
    monkeypatch,
    source_only,
):
    cp = pytest.importorskip("cupy")
    from source.solvers.vpm.physics.induction.gaussian_mesh.fields import GaussianImageFields
    from source.solvers.vpm.physics.induction.gaussian_mesh.host_fields import (
        GaussianHostImageFields,
    )

    x, gamma, sigma, q, images, options = _case(source_only)
    with GaussianImageFields(x, gamma, sigma, q, **options) as cuda:
        u, j, _ = cuda.evaluate(images)
        expected = cp.asnumpy(u), cp.asnumpy(j)
        del u, j
    normal_pool = cp.get_default_memory_pool()
    with cp.cuda.using_allocator(normal_pool.malloc):
        sentinel = cp.arange(17, dtype=cp.float32)
    cp.cuda.get_current_stream().synchronize()
    normal_before = normal_pool.used_bytes(), normal_pool.total_bytes(), normal_pool.get_limit()
    allocator_before = cp.cuda.get_allocator()
    _, total = cp.cuda.runtime.memGetInfo()
    # Leave only minimum correction-query scratch and no field data.
    # Only the validation result is constrained; real CUDA produced the
    # baseline and remains available while the same request executes on CPU.
    from source.solvers.vpm.physics.induction.gaussian_mesh.planning import correction_query_reserve

    constrained_free = correction_query_reserve(1, np.dtype(options["correction_dtype"]).itemsize)
    monkeypatch.setattr(cp.cuda.runtime, "memGetInfo", lambda: (constrained_free, total))
    field = PortableGaussianImageFields(
        x, gamma, sigma, q, execution_backend="cupy_cuda", **options
    )
    try:
        field.prepare(images)
        u, j, report = field.evaluate_prepared(q)
        assert field.execution_backend == "cpu"
        assert isinstance(field._implementation, GaussianHostImageFields)
        assert field._failed_implementation is None
        assert "unchanged cardinal source/query block" in field.fallback_reason
        assert isinstance(u, np.ndarray) and isinstance(j, np.ndarray)
        assert report["execution_backend"] == "cpu"
        assert report["memory_fallback"] == field.fallback_reason
        _assert_same_finite_fields((u, j), expected)
        assert cp.cuda.get_allocator() is allocator_before
        assert (
            normal_pool.used_bytes(),
            normal_pool.total_bytes(),
            normal_pool.get_limit(),
        ) == normal_before
        np.testing.assert_array_equal(cp.asnumpy(sentinel), np.arange(17, dtype=np.float32))
    finally:
        field.close()
        del sentinel
