"""Pure-math feasibility checks; these import no production Taichi kernels."""

import math

import numpy as np
import pytest

from tests.vpm._fmm_source_remainder_prototype import (
    FieldBound,
    certify_extra_source_admission,
    legacy_envelope_budget,
    regularization_error_bound,
    singular_source_taylor,
    source_derivative_remainder,
)


def _exact(position, strength, target):
    displacement = target - position
    radius = np.linalg.norm(displacement, axis=1)
    cross = np.cross(displacement, strength)
    velocity = (-cross / radius[:, None] ** 3).sum(axis=0) / (4 * math.pi)
    skew = np.zeros((len(strength), 3, 3))
    skew[:, 0, 1], skew[:, 0, 2] = -strength[:, 2], strength[:, 1]
    skew[:, 1, 0], skew[:, 1, 2] = strength[:, 2], -strength[:, 0]
    skew[:, 2, 0], skew[:, 2, 1] = -strength[:, 1], strength[:, 0]
    gradient = (
        skew / radius[:, None, None] ** 3
        + 3 * cross[:, :, None] * displacement[:, None, :] / radius[:, None, None] ** 5
    ).sum(axis=0) / (4 * math.pi)
    return velocity, gradient


@pytest.mark.parametrize("ratio", [0.015, 0.05, 0.1, 0.25, 0.4])
@pytest.mark.parametrize("cancel", [False, True])
def test_absolute_moment_bound_encloses_actual_cubic_error(ratio, cancel):
    rng = np.random.default_rng(4808)
    position = rng.normal(size=(24, 3))
    position *= ratio / np.linalg.norm(position, axis=1).max()
    strength = rng.normal(size=(24, 3))
    if cancel:
        strength -= strength.mean(axis=0)
    target = np.array([0.7, -0.5, math.sqrt(0.26)])
    radii = np.linalg.norm(position, axis=1)
    absolute_moment = np.dot(np.linalg.norm(strength, axis=1), radii**4)
    exact = _exact(position, strength, target)
    expansion = singular_source_taylor(position, strength, target, np.zeros(3), 3)
    for derivative, actual, approximate in zip((1, 2), exact, expansion, strict=True):
        bound = source_derivative_remainder(3, derivative, 1, radii.max(), absolute_moment)
        assert np.linalg.norm(actual - approximate) <= bound


def test_cancelled_zero_cubic_moments_are_not_a_zero_error_certificate():
    position = np.zeros((16, 3))
    position[:, 2] = np.arange(-15, 16, 2) / 16
    strength = np.zeros_like(position)
    strength[:, 1] = [(-1) ** i.bit_count() for i in range(16)]
    target = np.array([0.0, 0.0, 5.7])
    moment = np.sum(np.linalg.norm(strength, axis=1) * np.linalg.norm(position, axis=1) ** 4)
    exact = _exact(position, strength, target)
    approximate = singular_source_taylor(position, strength, target, np.zeros(3), 3)
    assert np.linalg.norm(approximate[0]) < 1e-16
    assert np.linalg.norm(exact[0]) > 1e-6
    decision, bounds = certify_extra_source_admission(
        legacy_all=False, order=3, distance=5.7, radius=0.9375, absolute_moment=moment,
        budget=FieldBound(1e-9, 1e-9), core_error=FieldBound(0, 0),
        local_error=FieldBound(0, 0), rounding_error=FieldBound(0, 0),
    )
    assert decision == "refine"
    assert bounds.velocity >= np.linalg.norm(exact[0])
    assert bounds.gradient >= np.linalg.norm(exact[1])


def test_extra_admission_requires_explicit_complete_budget_and_preserves_legacy():
    args = {
        "legacy_all": False, "order": 3, "distance": 1, "radius": 0.01,
        "absolute_moment": 1e-8, "budget": FieldBound(1, 1),
        "core_error": FieldBound(0, 0), "local_error": FieldBound(0, 0),
        "rounding_error": FieldBound(0, 0),
    }
    assert certify_extra_source_admission(**args)[0] == "higher_order"
    for name in ("budget", "core_error", "local_error", "rounding_error"):
        assert certify_extra_source_admission(**(args | {name: None}))[0] == "refine"
    assert certify_extra_source_admission(**(args | {"legacy_all": True}))[0] == "legacy"
    assert certify_extra_source_admission(**(args | {"core_error": FieldBound(2, 0)}))[0] == "refine"
    assert math.isinf(source_derivative_remainder(3, 2, 1, 1, 1))


def test_cubic_near_admission_cannot_claim_roundoff_accuracy_from_angle_alone():
    # Source radius/distance=.05 is the old diameter/distance=.1 boundary.
    # Worst-case radial fourth moment is radius**4 for unit absolute strength.
    cubic = source_derivative_remainder(3, 2, 1, 0.05, 0.05**4) * (4 * math.pi)
    assert cubic > 7e-4
    assert cubic > 16 * np.finfo(np.float32).eps
    # Higher order changes arithmetic count, not the error budget. Its bound
    # can be competitive at larger radius ratios, unlike a relaxed p3 MAC.
    ninth = source_derivative_remainder(9, 2, 1, 0.1, 0.1**10) * (4 * math.pi)
    assert ninth < 16 * np.finfo(np.float32).eps


def test_coarse_budget_does_not_expand_under_any_descendant_partition():
    rng = np.random.default_rng(1670)
    positions = rng.uniform(-0.1, 0.1, (96, 3))
    weights = rng.uniform(0.1, 2.0, 96)
    radius = np.linalg.norm(positions, axis=1).max()
    target = np.array([0.0, 0.0, 1.0])
    parent = legacy_envelope_budget(0.1, weights.sum(), 1 + radius)
    for child_count in (1, 2, 3, 8, 12, 24, 96):
        children = FieldBound(0, 0)
        for indices in np.array_split(rng.permutation(96), child_count):
            weight = weights[indices]
            centre = (weight[:, None] * positions[indices]).sum(axis=0) / weight.sum()
            children += legacy_envelope_budget(0.1, weight.sum(), np.linalg.norm(target - centre))
        assert parent.velocity <= children.velocity
        assert parent.gradient <= children.gradient


def test_matched_monopole_envelope_allows_materially_larger_cubic_cells():
    # Uses worst possible absolute fourth moment; no native case data enter.
    # Child COM-distance normalization is intentionally conservative.
    radius = 0.19
    budget = legacy_envelope_budget(0.1, 1, 1 + radius)
    decision, error = certify_extra_source_admission(
        legacy_all=False, order=3, distance=1, radius=radius, absolute_moment=radius**4,
        budget=budget, core_error=FieldBound(0, 0),
        local_error=FieldBound(0, 0), rounding_error=FieldBound(0, 0),
    )
    assert radius > 3 * (0.1 / 2)
    assert decision == "higher_order"
    assert error.gradient < budget.gradient


def test_regularization_coefficient_bounds_charge_velocity_and_full_jacobian():
    strength, distance = 2.0, 4.0
    result = regularization_error_bound(strength, distance, 1e-7, 3e-7)
    assert result.velocity == strength * 1e-7 / distance**2
    assert result.gradient == strength * (math.sqrt(2) * 1e-7 + 3e-7) / distance**3
    invalid = regularization_error_bound(1, 0, 0, 0)
    assert math.isinf(invalid.velocity) and math.isinf(invalid.gradient)


def test_matched_worst_case_cubic_envelope_is_not_pointwise_no_loss():
    # Both old monopole nodes fail theta=.1 and their singleton descendants
    # are exact. A same-worst-case-budget cubic replacement can nevertheless
    # introduce visible error. This is an explicit rejection of that rollout,
    # not a loosened accuracy test for a candidate production operator.
    radius = 0.19
    position = np.array([[-radius, 0, 0], [radius, 0, 0]])
    strength = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    target = np.array([1.0, 0, 0])
    exact = _exact(position, strength, target)
    cubic = singular_source_taylor(position, strength, target, np.zeros(3), 3)
    decision, _ = certify_extra_source_admission(
        legacy_all=False, order=3, distance=1, radius=radius,
        absolute_moment=2 * radius**4,
        budget=legacy_envelope_budget(0.1, 2, 1 + radius),
        core_error=FieldBound(0, 0), local_error=FieldBound(0, 0),
        rounding_error=FieldBound(0, 0),
    )
    assert decision == "higher_order"
    cubic_errors = [np.linalg.norm(a - b) / np.linalg.norm(a) for a, b in zip(exact, cubic, strict=True)]
    assert cubic_errors[0] > 0.005
    assert cubic_errors[1] > 0.01
    ninth = singular_source_taylor(position, strength, target, np.zeros(3), 9)
    for actual, approximate in zip(exact, ninth, strict=True):
        assert np.linalg.norm(actual - approximate) / np.linalg.norm(actual) < 1e-5
