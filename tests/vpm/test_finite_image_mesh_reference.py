"""Small finite-block particle-mesh feasibility tests; no production imports."""

import itertools
import math

import numpy as np
import pytest
from scipy.fft import irfftn, next_fast_len, rfftn

from tests.vpm import _finite_image_mesh_reference as mesh_module
from tests.vpm._finite_image_mesh_reference import (
    _assign,
    _finite_kernel,
    _stencil,
    direct_finite_images,
    finite_image_mesh,
    observed_error,
)


def test_signed_lag_fft_embedding_is_exact_finite_linear_convolution():
    shape = (4, 3, 2)
    padded = tuple(next_fast_len(2*n-1) for n in shape)
    density = np.arange(np.prod(shape), dtype=float).reshape(shape) - 4
    # Include a shift larger than the FFT box: it must not fold modulo it.
    kernel = _finite_kernel(shape, padded, 0.1, [0.0, 8.3, -7.9], 0.2)
    result = irfftn(rfftn(density, s=padded)*rfftn(kernel), s=padded)
    for target in np.ndindex(shape):
        direct = 0.0
        for source in np.ndindex(shape):
            # Independent analytic lag oracle, not a lookup in the same
            # embedded kernel: wrong lag signs or period-folded image shifts
            # must fail this comparison.
            lag = 0.1 * (np.array(target)-source)
            value = 0.0
            for shift in (0.0, 8.3, -7.9):
                r = math.sqrt(lag[0]**2 + lag[1]**2 + (lag[2]-shift)**2)
                potential = math.erf(r/0.2)/(4*math.pi*r) if r else 1/(2*math.pi**1.5*0.2)
                gauge = math.erf(abs(shift)/0.2)/(4*math.pi*abs(shift)) if shift else 1/(2*math.pi**1.5*0.2)
                value += potential-gauge
            direct += density[source]*value
        np.testing.assert_allclose(result[target], direct, atol=2e-13, rtol=2e-14)


def test_off_grid_assignment_preserves_total_strength_without_volume_factor():
    positions = np.array([[0.013, 0.039, 0.017], [0.072, 0.063, 0.097]])
    strengths = np.array([[1, -2, 3], [-0.3, 0.5, -1]])
    first, weights = _stencil(positions, np.full(3, -0.2), 0.04, 6)
    grid = _assign((16, 16, 16), first, weights, strengths)
    np.testing.assert_allclose(grid.sum(axis=(0, 1, 2)), strengths.sum(axis=0), atol=2e-12)


def _cloud():
    position = np.array([[-0.047, 0.018, 0.073], [0.056, -0.036, 0.041], [0.011, 0.029, 0.096]])
    strength = np.array([[0.3, -0.2, 0.5], [-0.7, 0.4, 0.2], [0.4, -0.2, -0.7]])
    core = np.array([0.04, 0.07, 0.09])
    query = np.array([[0.127, -0.031, 0.082], [-0.083, 0.074, 0.106], [0.021, 0.052, 0.023]])
    return position, strength, core, query


def test_off_grid_mixed_core_finite_block_refines_toward_direct_truth():
    x, gamma, sigma, query = _cloud()
    images = [(0.0, True), (0.64, False), (-0.64, False), (0.64, True)]
    exact_u, exact_j, conditioning = direct_finite_images(x, gamma, sigma, query, images)
    runs = [finite_image_mesh(x, gamma, sigma, query, images, tau=0.25, spacing=h, order=8)
            for h in (0.05, 0.025)]
    for field in ("velocity", "gradient"):
        exact = exact_u if field == "velocity" else exact_j
        errors = [np.linalg.norm(getattr(run, field)-exact) for run in runs]
        assert errors[1] < errors[0]/4, (field, errors)
        assert errors[1]/np.linalg.norm(exact) < 1e-4, (field, errors)
    fine = runs[-1]
    np.testing.assert_array_equal(fine.correction_velocity_tail, 0)
    np.testing.assert_array_equal(fine.correction_gradient_tail, 0)
    assert not fine.diagnostics["fields_runtime_qualified"]
    assert not fine.diagnostics["infinite_periodic_operator"]
    assert fine.diagnostics["finite_image_count"] == len(images)
    np.testing.assert_allclose(np.trace(fine.gradient, axis1=1, axis2=2), 0, atol=1e-11)
    assert observed_error(fine.velocity, exact_u, conditioning[:, 0])["relative_l2"] < 1e-4


@pytest.mark.parametrize("odd", [False, True])
def test_coincident_pair_refines_toward_finite_self_j_without_pointwise_override(odd):
    source = np.array([[0.037, -0.011, 0.08]])
    strength = np.array([[0.2, 0.7, -0.4]])
    core = np.array([0.08])
    shift = 0.16 if odd else 0.0
    images = [(shift, odd)]
    exact_u, exact_j, _ = direct_finite_images(source, strength, core, source, images)
    runs = [finite_image_mesh(source, strength, core, source, images, tau=0.2, spacing=h, order=8)
            for h in (0.05, 0.025)]
    errors = [(np.linalg.norm(run.velocity-exact_u), np.linalg.norm(run.gradient-exact_j)) for run in runs]
    assert errors[1][0] < errors[0][0]/4
    assert errors[1][1] < errors[0][1]/4
    assert errors[1][0] < 1e-5  # Absolute cap: true coincident velocity is zero.
    assert errors[1][1]/np.linalg.norm(exact_j) < 1e-4
    assert np.linalg.norm(exact_j) > 1
    assert not runs[-1].diagnostics["exact_coincidence_override"]


def test_remote_cancelled_finite_block_is_not_an_infinite_sum_or_zero_mode_drop():
    x, gamma, sigma, query = _cloud()
    images = [(8.0, False), (-8.0, False), (8.0, True), (-8.0, True)]
    exact_u, exact_j, conditioning = direct_finite_images(x, gamma, sigma, query, images)
    result = finite_image_mesh(x, gamma, sigma, query, images, tau=0.25, spacing=0.04, order=6)
    for candidate, exact in ((result.velocity, exact_u), (result.gradient, exact_j)):
        np.testing.assert_allclose(candidate, exact, atol=1e-10, rtol=1e-4)
    assert np.linalg.norm(exact_u) > 0
    assert np.max(conditioning[:, 0]) > np.max(np.linalg.norm(exact_u, axis=1))


@pytest.mark.parametrize("cancelled", [False, True])
def test_image_only_near_plane_physical_core_and_nonzero_z_or_moment_cancellation(cancelled):
    x = np.array([[-0.082, 0.012, 0.023], [-0.042, 0.012, 0.023],
                  [-0.002, 0.012, 0.023], [0.038, 0.012, 0.023], [0.078, 0.012, 0.023]])
    gamma = np.tile([0.3, -0.2, 0.7], (5, 1))
    if cancelled:
        gamma *= np.array([1, -4, 6, -4, 1])[:, None]
    else:
        assert gamma[:, 2].sum() != 0
    sigma = np.full(5, 0.04)
    query = np.array([[0.011, 0.019, 0.018], [-0.073, 0.058, 0.061], [0.112, -0.036, 0.04]])
    images = [(0.0, True), (0.64, False), (-0.64, False)]
    exact_u, exact_j, _ = direct_finite_images(x, gamma, sigma, query, images)
    result = finite_image_mesh(x, gamma, sigma, query, images, tau=0.2, spacing=0.025, order=8)
    for candidate, exact in ((result.velocity, exact_u), (result.gradient, exact_j)):
        assert np.linalg.norm(candidate-exact)/np.linalg.norm(exact) < 1e-4
        np.testing.assert_allclose(candidate, exact, atol=1e-4, rtol=1e-4)


def test_translation_recenters_odd_family_without_exploding_grid():
    x, gamma, sigma, query = _cloud()
    images = [(0.0, True), (0.64, False)]
    original = finite_image_mesh(x, gamma, sigma, query, images, tau=0.25, spacing=0.04, order=6)
    delta = 1024.0
    translated_x, translated_q = x.copy(), query.copy()
    translated_x[:, 2] += delta
    translated_q[:, 2] += delta
    moved = finite_image_mesh(translated_x, gamma, sigma, translated_q,
                             [(s+2*delta if odd else s, odd) for s, odd in images],
                             tau=0.25, spacing=0.04, order=6)
    assert moved.diagnostics["fft_shape"] == original.diagnostics["fft_shape"]
    np.testing.assert_allclose(moved.velocity, original.velocity, rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(moved.gradient, original.gradient, rtol=2e-9, atol=2e-8)


def test_omitted_core_correction_is_separately_charged_and_work_is_bounded():
    x, gamma, sigma, query = _cloud()
    images = [(0.0, True)]
    full = finite_image_mesh(x, gamma, sigma, query, images, tau=0.25, spacing=0.04, order=6)
    limited = finite_image_mesh(x, gamma, sigma, query, images, tau=0.25, spacing=0.04,
                                order=6, correction_cutoff=0.05)
    assert np.all(np.linalg.norm(full.velocity-limited.velocity, axis=1) <= limited.correction_velocity_tail)
    assert np.all(np.linalg.norm(full.gradient-limited.gradient, axis=(1, 2)) <= limited.correction_gradient_tail)
    with pytest.raises(ValueError, match="grid exceeded"):
        finite_image_mesh(x, gamma, sigma, query, images, tau=0.25, spacing=0.04, max_grid_nodes=10)
    with pytest.raises(ValueError, match="pair budget"):
        finite_image_mesh(x, gamma, sigma, query, images, tau=0.25, spacing=0.04, max_pairs=1)
    assert observed_error(np.ones((1, 3)), np.zeros((1, 3)), [0])["relative_l2"] is None


@pytest.mark.parametrize("cap", [0, -1, True, 1.5, np.nan, np.inf])
@pytest.mark.parametrize("name", ["max_pairs", "max_grid_nodes", "max_images"])
def test_caps_are_strict_positive_integer_limits(cap, name):
    with pytest.raises(ValueError, match="positive integer"):
        finite_image_mesh(*_cloud(), [(0.0, True)], tau=0.25, spacing=0.04, **{name: cap})


def test_empty_inputs_and_infinite_image_iterable_are_bounded():
    x, gamma, sigma, query = _cloud()
    cases = [(x[:0], gamma[:0], sigma[:0], query, [(0.0, True)]),
             (x, gamma, sigma, query[:0], [(0.0, True)]),
             (x, gamma, sigma, query, [])]
    for args in cases:
        result = finite_image_mesh(*args, tau=0.25, spacing=0.04)
        np.testing.assert_array_equal(result.velocity, np.zeros((len(args[3]), 3)))
        np.testing.assert_array_equal(result.gradient, np.zeros((len(args[3]), 3, 3)))
    with pytest.raises(ValueError, match="image count"):
        finite_image_mesh(x[:0], gamma[:0], sigma[:0], query,
                          itertools.repeat((0.0, True)), tau=0.25, spacing=0.04, max_images=2)


def test_overflow_and_fft_failure_do_not_publish_or_modify_inputs(monkeypatch):
    x, gamma, sigma, query = _cloud()
    copies = [value.copy() for value in (x, gamma, sigma, query)]
    with pytest.raises((FloatingPointError, ValueError), match="overflow|grid|nonfinite"):
        finite_image_mesh(x, gamma, sigma, query, [(0.0, True)], tau=0.25, spacing=1e-300)

    def fail_fft(*args, **kwargs):
        raise RuntimeError("injected FFT failure")

    monkeypatch.setattr(mesh_module, "rfftn", fail_fft)
    with pytest.raises(RuntimeError, match="injected FFT"):
        finite_image_mesh(x, gamma, sigma, query, [(0.0, True)], tau=0.25, spacing=0.04)
    for original, copy in zip((x, gamma, sigma, query), copies, strict=True):
        np.testing.assert_array_equal(original, copy)


def test_observed_error_rejects_broadcasting_nonfinite_and_invalid_conditioning():
    for candidate, exact, conditioning in (
        (np.ones((1, 3)), np.ones((2, 3)), [1, 1]),
        (np.full((1, 3), np.nan), np.zeros((1, 3)), [1]),
        (np.ones((1, 3)), np.zeros((1, 3)), [-1]),
        (np.ones((1, 3)), np.zeros((1, 3)), [np.inf]),
    ):
        with pytest.raises(ValueError, match="identical finite"):
            observed_error(candidate, exact, conditioning)
