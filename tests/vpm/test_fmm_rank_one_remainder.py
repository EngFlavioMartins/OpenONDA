"""Adversarial analytic reference tests; no Taichi initialization or GPU work."""

import math

import numpy as np
import pytest

from tests.vpm._fmm_rank_one_remainder import core_tail_bound, rank_one_remainder
from tests.vpm._fmm_source_remainder_prototype import singular_source_taylor
from tests.vpm._gaussian_broadening_reference import gaussian_fields, singular_fields
from tests.vpm.test_fmm_source_remainder import _exact


@pytest.mark.parametrize("order", [0, 1, 3, 5, 7, 9])
@pytest.mark.parametrize("ratio,aspect", [(0.05, (1, 1, 1)), (0.25, (1, 0.03, 0.001)), (0.7, (1, 1, 0.01))])
def test_rank_one_remainder_encloses_anisotropic_source_error(order, ratio, aspect):
    rng = np.random.default_rng(98831)
    offsets = rng.uniform(-1, 1, (9, 3)) * aspect
    offsets *= ratio / np.linalg.norm(offsets, axis=1).max()
    strength = rng.normal(0, 1, offsets.shape)
    strength -= strength.mean(axis=0)
    centre = np.array([1e3, -2e3, 3e3])
    position = centre + offsets
    target = centre + np.array([0.8, -0.36, 0.48])
    # Use the actual represented differences, not pre-translation intentions.
    lengths = np.linalg.norm(position - centre, axis=1)
    radius = lengths.max()
    distance = np.linalg.norm(target - centre)
    moment = np.dot(np.linalg.norm(strength, axis=1), lengths**(order + 1))
    exact = _exact(position, strength, target)
    approximation = singular_source_taylor(position, strength, target, centre, order)
    for derivative, actual, candidate in zip((1, 2), exact, approximation, strict=True):
        bound = rank_one_remainder(order, derivative, distance, radius, moment)
        # This only covers f64 reference evaluation at vanishing Taylor errors;
        # it is not the production f32 arithmetic reserve/certificate.
        reference_rounding = 64 * np.finfo(float).eps * np.linalg.norm(strength, axis=1).sum() / (distance - radius)**(derivative + 1)
        assert np.linalg.norm(actual - candidate) <= bound + reference_rounding


@pytest.mark.parametrize("order", [0, 1, 3, 5, 7, 9])
def test_signed_moments_zero_through_order_do_not_erase_absolute_remainder(order):
    n = order + 1
    position = np.zeros((n + 1, 3))
    position[:, 0] = np.arange(n + 1) - n / 2
    strength = np.zeros_like(position)
    strength[:, 1] = [(-1)**k * math.comb(n, k) for k in range(n + 1)]
    target = np.array([max(n, 1) * 2.0, 0.0, 0.0])
    for power in range(n):
        assert np.sum(strength[:, 1] * position[:, 0]**power) == 0
    moment = np.dot(np.linalg.norm(strength, axis=1), np.linalg.norm(position, axis=1)**n)
    exact = _exact(position, strength, target)
    candidate = singular_source_taylor(position, strength, target, np.zeros(3), order)
    assert moment > 0
    for derivative, value, truncated in zip((1, 2), exact, candidate, strict=True):
        assert np.linalg.norm(value) > 0
        assert np.linalg.norm(truncated) < 1e-14
        bound = rank_one_remainder(order, derivative, np.linalg.norm(target), np.linalg.norm(position, axis=1).max(), moment)
        assert 0 < np.linalg.norm(value - truncated) <= bound
    if order == 3:
        np.testing.assert_array_equal(strength[:, 1], [1, -4, 6, -4, 1])
        assert moment == 40


def _winckelmans_fields(displacement, strength, sigma):
    radius = np.linalg.norm(displacement)
    base = radius**2 + sigma**2
    a = (radius**2 + 2.5 * sigma**2) / (4 * math.pi * base**2.5)
    b = (3 * radius**2 + 10.5 * sigma**2) / (4 * math.pi * base**3.5)
    x, y, z = strength
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return a * (cross @ displacement), cross @ (a * np.eye(3) - b * np.outer(displacement, displacement))


@pytest.mark.parametrize("kernel", ["GAUSSIAN", "WINCKELMANS"])
@pytest.mark.parametrize("ratio", [0.1, 1.0, 3.0, 7.0, 15.0, 50.0])
def test_core_envelope_covers_actual_variable_core_defects(kernel, ratio):
    maximum_core, cutoff = 0.04, 0.04 * ratio
    strength = np.array([0.7, -0.2, 0.4])
    bound = core_tail_bound(kernel, cutoff, maximum_core, np.linalg.norm(strength))
    for scale in (1.0, 1.1, 2.0):
        displacement = cutoff * scale * np.array([0.8, 0.0, 0.6])
        singular = singular_fields(displacement, strength)
        for sigma in maximum_core * np.array([0.03, 0.2, 0.7, 1.0]):
            regular = (gaussian_fields if kernel == "GAUSSIAN" else _winckelmans_fields)(displacement, strength, sigma)
            for exact, blob, allowance in zip(singular, regular, (bound.velocity, bound.gradient), strict=True):
                rounding = 16 * np.finfo(float).eps * max(np.linalg.norm(exact), np.linalg.norm(blob))
                assert np.linalg.norm(exact - blob) <= allowance + rounding


@pytest.mark.parametrize("kernel", ["GAUSSIAN", "WINCKELMANS"])
@pytest.mark.parametrize("order", [3, 7])
def test_combined_source_and_core_bound_encloses_regularized_cloud(kernel, order):
    rng = np.random.default_rng(5031)
    centre = np.array([8, -11, 3.0])
    position = centre + rng.uniform(-0.06, 0.06, (7, 3)) * [1, 0.03, 0.002]
    strength = rng.normal(size=position.shape)
    strength[1::2] *= -1
    core = rng.uniform(0.002, 0.016, len(position))
    target = centre + np.array([0.5, 0.1, -0.03])
    lengths = np.linalg.norm(position - centre, axis=1)
    absolute = np.linalg.norm(strength, axis=1)
    d, a = np.linalg.norm(target - centre), lengths.max()
    moment = absolute @ lengths**(order + 1)
    singular_expansion = singular_source_taylor(position, strength, target, centre, order)
    core_error = core_tail_bound(kernel, d - a, core.max(), absolute.sum())
    regular = [np.zeros(3), np.zeros((3, 3))]
    for x, gamma, sigma in zip(position, strength, core, strict=True):
        pair = (gaussian_fields if kernel == "GAUSSIAN" else _winckelmans_fields)(target - x, gamma, sigma)
        for i in (0, 1):
            regular[i] += pair[i]
    for i, core_bound in enumerate((core_error.velocity, core_error.gradient)):
        total = rank_one_remainder(order, i + 1, d, a, moment) + core_bound
        assert np.linalg.norm(regular[i] - singular_expansion[i]) <= total


def test_invalid_or_unrepresentable_geometry_never_yields_nan_certificate():
    for arguments in ((1, 1, 1), (math.nan, 0.1, 1), (1, 0.1, math.inf)):
        assert math.isinf(rank_one_remainder(3, 1, *arguments))
    for kernel in ("GAUSSIAN", "WINCKELMANS"):
        for distance, core in ((1e200, 1e-200), (1e-200, 1e200), (0, 1)):
            result = core_tail_bound(kernel, distance, core, 1)
            assert not math.isnan(result.velocity) and not math.isnan(result.gradient)
