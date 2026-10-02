"""Independent small-cloud qualification; CUDA execution requires root slot."""

import math

import numpy as np
import pytest

from tests.vpm._gaussian_broadening_reference import correction_factors, correction_fields
from tests.vpm._gaussian_core_correction_gpu import (
    GaussianCoreCorrectionGPU,
    _image_list,
    _kernel_source,
    _possibly_near,
    correction_factors_host,
)


@pytest.mark.parametrize("ratio", [1., 1.0001, 1.009, 1.02, 3., 20.])
@pytest.mark.parametrize("rho", [0., 1e-9, .499999, .5, .500001, 1., 3., 5.])
def test_device_factor_design_against_independent_positive_integral(ratio, rho):
    sigma, tau = .04, .04*ratio
    actual = correction_factors_host(rho*sigma, sigma, tau)
    expected = correction_factors(rho*sigma, sigma, tau)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=0)


def test_coincident_almost_equal_cores_and_guard_branches():
    sigma, tau = .04, np.nextafter(.04, np.inf)
    actual = correction_factors_host(0., sigma, tau)
    log_ratio = math.log1p((tau-sigma)/sigma)
    expected = (math.pi**-1.5/sigma**3*(-math.expm1(-3*log_ratio))/3,
                2*math.pi**-1.5/sigma**5*(-math.expm1(-5*log_ratio))/5)
    np.testing.assert_allclose(actual, expected, rtol=3e-16, atol=0)
    for args in ((0, 0, 1), (-1, 1, 2), (1, 2, 1), (np.inf, 1, 2)):
        with pytest.raises(ValueError):
            correction_factors_host(*args)


def test_aabb_filter_is_conservative_and_descriptors_are_bounded():
    rng = np.random.default_rng(48002)
    x, q = rng.normal(size=(40, 3)), rng.normal(size=(35, 3))
    for shift in (-100., -3., 0., 1., 100.):
        for odd in (False, True):
            transformed = x.copy()
            transformed[:, 2] *= -1 if odd else 1
            transformed[:, 2] += shift
            near = np.any(np.linalg.norm(q[:, None]-transformed[None], axis=2) < .6)
            if near:
                assert _possibly_near(x.min(0), x.max(0), q.min(0), q.max(0), shift, odd, .6)
    assert not _possibly_near(x.min(0), x.max(0), q.min(0), q.max(0), 100., False, .6)
    with pytest.raises(ValueError):
        _image_list([(0, 1)], 2)
    with pytest.raises(ValueError):
        _image_list([(0, True)]*3, 2)
    for dtype in ("float32", "float64"):
        code = _kernel_source(dtype)
        assert "SUFFIX" not in code and "QUAD_" not in code and "REAL" not in code
        assert "atomicAdd(work" in code


@pytest.fixture(scope="module")
def cupy_runtime():
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA device unavailable")
    return cp


def _direct(x, gamma, sigma, query, images, tau, cutoff):
    u, j = np.zeros((len(query), 3)), np.zeros((len(query), 3, 3))
    cond_u, cond_j = np.zeros(len(query)), np.zeros(len(query))
    count = 0
    for shift, odd in images:
        source, strength = x.copy(), gamma.copy()
        if odd:
            source[:, 2] *= -1
            strength[:, :2] *= -1
        source[:, 2] += shift
        for t, point in enumerate(query):
            for s, location in enumerate(source):
                displacement = point-location
                if float(displacement@displacement) >= cutoff*cutoff:
                    continue
                du, dj = correction_fields(displacement, strength[s], sigma[s], tau)
                u[t] += du
                j[t] += dj
                cond_u[t] += np.linalg.norm(du)
                cond_j[t] += np.linalg.norm(dj)
                count += 1
    return u, j, cond_u, cond_j, count


def _assert_fields(actual_u, actual_j, reference, dtype):
    u, j, cu, cj, _ = reference
    # Absolute conditioning guard, not relative-to-cancelled-net circulation.
    eps = np.finfo(dtype).eps
    factor = 128 if dtype == "float32" else 4096
    assert np.all(np.linalg.norm(actual_u-u, axis=1) <= factor*eps*cu+1e-30)
    assert np.all(np.linalg.norm(actual_j-j, axis=(1, 2)) <= factor*eps*cj+1e-28)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_gpu_mixed_cores_reflection_cancellation_and_no_mutation(cupy_runtime, dtype):
    cp = cupy_runtime
    rng = np.random.default_rng(71892)
    x = rng.uniform([-.3, -.25, -.48], [.3, .25, .48], (67, 3))
    gamma = rng.normal(size=x.shape)*1e-3
    sigma = rng.uniform(.028, .119, len(x))
    sigma[:4] = [.04, .12, np.nextafter(.12, 0), .12*(1-1e-4)]
    x[:5, 0] = np.arange(-2, 3)*.013
    x[:5, 1:] = [0., -.48]
    gamma[:5] = np.array([1., -4., 6., -4., 1.])[:, None]*[1e-3, -2e-3, 3e-3]
    query = np.vstack((x[:9], [[0, 0, -.48], [0, 0, .48]], rng.uniform(-.6, .6, (28, 3))))
    images = [(-.96, True), (.96, True), (1.92, False), (-1.92, False), (200., True)]
    originals = [a.copy() for a in (x, gamma, sigma, query)]
    source_arrays = [cp.asarray(a) for a in (x, gamma, sigma)]
    reference = _direct(x, gamma, sigma, query, images, .12, .6)
    with GaussianCoreCorrectionGPU(*source_arrays, tau=.12, cutoff=.6,
                                   accumulation_dtype=dtype, max_scratch_bytes=32*1024**2) as owner:
        u, j, report = owner.evaluate(cp.asarray(query), images)
        _assert_fields(cp.asnumpy(u), cp.asnumpy(j), reference, dtype)
        assert report["accepted_pairs"] == reference[-1]
        assert report["candidate_pairs"] >= report["accepted_pairs"]
        assert report["pool_high_water_bytes"] <= 32*1024**2
        assert (200., True) in report["aabb_skipped_images"]
        repeated_u, repeated_j, _ = owner.evaluate(query, images)
        np.testing.assert_array_equal(cp.asnumpy(repeated_u), cp.asnumpy(u))
        np.testing.assert_array_equal(cp.asnumpy(repeated_j), cp.asnumpy(j))
    owner.close()
    with pytest.raises(RuntimeError, match="closed"):
        owner.evaluate(query, images)
    for actual, original in zip((x, gamma, sigma, query), originals, strict=True):
        np.testing.assert_array_equal(actual, original)
    for actual, original in zip(source_arrays, originals[:3], strict=True):
        np.testing.assert_array_equal(cp.asnumpy(actual), original)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_gpu_exact_cutoff_cell_boundaries_coincident_j_and_remainders(cupy_runtime, dtype):
    cp = cupy_runtime
    cutoff = .6
    x = np.array([[0., 0., 0.], [.6, .6, .6], [-.6, -.6, -.6], [1.2, 0., 0.]])
    gamma = np.array([[1e-3, 2e-3, 3e-3], [-2e-3, 1e-3, 0], [0, -3e-3, 1e-3], [1e-3, 1e-3, 1e-3]])
    sigma = np.array([.04, .04, .12*(1-1e-4), .12])
    query = np.array([[0, 0, 0], [np.nextafter(.6, 0), 0, 0], [.6, 0, 0],
                      [np.nextafter(.6, 1), 0, 0], [.6, .6, .6], [1e12, 0, 0],
                      [0, 0, 1e-14], [0, 0, .019999999], [0, 0, .020000001]])
    images = [(0., False)]
    reference = _direct(x, gamma, sigma, query, images, .12, cutoff)
    with GaussianCoreCorrectionGPU(x, gamma, sigma, tau=.12, cutoff=cutoff, accumulation_dtype=dtype) as owner:
        u, j, report = owner.evaluate(query, images)
        _assert_fields(cp.asnumpy(u), cp.asnumpy(j), reference, dtype)
        assert report["accepted_pairs"] == reference[-1]
        assert np.linalg.norm(cp.asnumpy(j)[0]) > 0
        empty_u, empty_j, empty_report = owner.evaluate(np.empty((0, 3)), images)
        assert empty_u.shape == (0, 3) and empty_j.shape == (0, 3, 3)
        assert empty_report["accepted_pairs"] == 0
        zero_u, zero_j, _ = owner.evaluate(query, [])
        assert not bool(cp.any(zero_u)) and not bool(cp.any(zero_j))


def test_gpu_allocation_admission_invalid_inputs_and_output_survives_owner(cupy_runtime):
    cp = cupy_runtime
    x = np.zeros((1, 3))
    gamma, sigma = np.array([[.001, -.002, .003]]), np.array([.04])
    with pytest.raises((MemoryError, cp.cuda.memory.OutOfMemoryError)):
        GaussianCoreCorrectionGPU(x, gamma, sigma, tau=.12, cutoff=.6, max_scratch_bytes=16)
    with pytest.raises(ValueError):
        GaussianCoreCorrectionGPU(x, gamma, sigma, tau=.02, cutoff=.6)
    with GaussianCoreCorrectionGPU(x, gamma, sigma, tau=.12, cutoff=.6) as owner:
        with pytest.raises(ValueError):
            owner.evaluate([[np.nan, 0, 0]], [(0, False)])
        with cp.cuda.Stream(non_blocking=True), pytest.raises(RuntimeError, match="original device and CUDA stream"):
            owner.evaluate(x, [(0, False)])
        u, j, _ = owner.evaluate(x, [(0, False)])
    np.testing.assert_array_equal(cp.asnumpy(u), np.zeros((1, 3)))
    expected = correction_fields(np.zeros(3), gamma[0], .04, .12)[1]
    np.testing.assert_allclose(cp.asnumpy(j)[0], expected, rtol=1e-14, atol=0)
