"""Independent checks for the offline 3D joint-reconstruction experiment.

These qualify its measurement and fit components, not the live hybrid method.
"""

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.sparse.linalg import aslinearoperator

from source.coupler.renewal_projection import (
    gaussian_velocity_operator,
    sparse_gaussian_vorticity_basis,
)
from studies.coupler_accuracy.cube_snapshot_induction import volume_velocity
from studies.coupler_accuracy.joint_reconstruction_3d import (
    CubePanelResponse,
    constrained_least_squares,
    gaussian_velocity_curl_operator,
    project_strength_budget,
    regularized_projection,
)


def differentiated(field, points, axis, step):
    offset = np.eye(3)[axis] * step
    return (field(points - 2 * offset) - 8 * field(points - offset)
            + 8 * field(points + offset) - field(points + 2 * offset)) / (12 * step)


def test_gaussian_curl_matches_independently_differentiated_3d_velocity():
    rng = np.random.default_rng(739)
    position = rng.uniform(-1, 1, (7, 3))
    strength = rng.normal(0, 0.03, position.shape)
    radius = rng.uniform(0.2, 0.6, len(position))
    points = np.vstack((rng.uniform(-1.3, 1.3, (19, 3)), position,
                        position + radius[:, None] * 1e-7))

    def velocity(target):
        return volume_velocity(target, position, strength, radius)[0]

    derivatives = [differentiated(velocity, points, i, 4e-5) for i in range(3)]
    expected = np.column_stack((derivatives[1][:, 2] - derivatives[2][:, 1],
                                derivatives[2][:, 0] - derivatives[0][:, 2],
                                derivatives[0][:, 1] - derivatives[1][:, 0]))
    actual = (gaussian_velocity_curl_operator(points, position, radius) @ strength.ravel()).reshape(-1, 3)
    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-10)


def test_self_curl_is_finite_and_distinct_from_raw_gaussian_vorticity():
    position = np.array([[0.4, -0.2, 0.6]])
    strength = np.array([[0.2, -0.4, 0.1]])
    radius = 0.17
    raw = strength / (np.pi**1.5 * radius**3)
    actual = gaussian_velocity_curl_operator(position, position, radius) @ strength.ravel()
    np.testing.assert_allclose(actual, (2 * raw / 3).ravel(), rtol=3e-16)
    assert np.linalg.norm(actual - raw.ravel()) > 1


def test_velocity_curl_is_solenoidal_for_arbitrary_3d_particle_strengths():
    rng = np.random.default_rng(114)
    position = rng.normal(size=(5, 3))
    strength = rng.normal(size=(5, 3))
    points = rng.normal(size=(11, 3))

    def curl(target):
        return (gaussian_velocity_curl_operator(target, position, 0.7) @ strength.ravel()).reshape(-1, 3)

    divergence = sum(differentiated(curl, points, i, 1e-4)[:, i] for i in range(3))
    np.testing.assert_allclose(divergence, 0, rtol=0, atol=2e-10)


def test_complete_velocity_map_matches_actual_constrained_cube_panel_solve():
    body = CubePanelResponse()
    position = np.array([[0.9, 0.4, 0.2], [-0.8, -0.2, 0.7], [0.3, 0.8, -0.7]])
    strength = np.array([[0.02, -0.04, 0.03], [-0.01, 0.025, -0.015], [0.03, 0.015, 0.04]])
    radius = np.array([0.15, 0.23, 0.31])
    points = np.array([[0.8, 0.2, -0.3], [-1.1, 0.7, 0.6], [0.2, 0.7, -0.8], [2, -1, 0.9]])
    incident = volume_velocity(body.centres, position, strength, radius)[0]
    body.panel.solve(np.array([1., 0., 0.]), incident, time=0.5)
    expected = (volume_velocity(points, position, strength, radius)[0]
                + body.panel.compute_induced_velocity(points) + [1, 0, 0])
    operator, background = body.velocity_operator(points, position, radius)
    actual = (operator @ strength.ravel() + background).reshape(-1, 3)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-13)
    sigma = body.panel.lattice.source_strength.to_numpy()[:body.count]
    np.testing.assert_allclose(body.area @ sigma, 0, rtol=0, atol=3e-14)


@pytest.mark.parametrize("coupled_components", [False, True])
def test_regularized_fit_matches_independent_dense_least_squares(coupled_components):
    rng = np.random.default_rng(925)
    prior = rng.normal(size=(5, 3))
    omega = rng.normal(size=(9 * 3, 5 * 3)) if coupled_components else rng.normal(size=(9, 5))
    dense_omega = omega if coupled_components else np.kron(omega, np.eye(3))
    velocity = rng.normal(size=(7 * 3, 5 * 3))
    target_omega, target_velocity = rng.normal(size=(9, 3)), rng.normal(size=(7, 3))
    ow, vw = rng.uniform(0.1, 1, 9), rng.uniform(0.1, 1, 7)
    matrix = np.vstack((np.repeat(ow, 3)[:, None] * dense_omega,
                        2 * np.repeat(vw, 3)[:, None] * velocity,
                        0.1 / np.linalg.norm(prior) * np.eye(prior.size)))
    rhs = np.r_[np.repeat(ow, 3) * target_omega.ravel(),
                2 * np.repeat(vw, 3) * target_velocity.ravel(),
                0.1 / np.linalg.norm(prior) * prior.ravel()]
    expected = np.linalg.lstsq(matrix, rhs, rcond=None)[0].reshape(-1, 3)
    actual, diagnostics = regularized_projection(
        omega, velocity, target_omega, target_velocity, prior, ow, vw,
        velocity_weight=2, prior_weight=0.1, max_strength_l1_ratio=10)
    assert diagnostics["segment_correction_fraction"] == 1
    assert diagnostics["lsmr_stop"] in (1, 2)
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-8)


def test_absolute_anchor_is_repeatable_and_enforces_finite_strength_budget():
    prior = np.ones((4, 3))
    arguments = (np.eye(4), np.eye(12), 100 * prior, 100 * prior,
                 prior, np.ones(4), np.ones(4))
    first, diagnostics = regularized_projection(*arguments, velocity_weight=1, max_strength_l1_ratio=1.2)
    second, _ = regularized_projection(*arguments, velocity_weight=1, max_strength_l1_ratio=1.2)
    np.testing.assert_array_equal(first, second)
    assert 0 < diagnostics["segment_correction_fraction"] < 1
    actual_l1 = np.linalg.norm(first, axis=1).sum()
    budget = 1.2 * np.linalg.norm(prior, axis=1).sum()
    assert actual_l1 <= budget
    # Bisection bounds the correction fraction to 2^-55; here its slope is ~700.
    np.testing.assert_allclose(actual_l1, budget, rtol=0, atol=3e-14)


def test_joint_fit_recovers_attainable_3d_field_at_independent_velocity_points():
    rng = np.random.default_rng(123)
    position = rng.uniform(-0.5, 0.5, (6, 3))
    strength = rng.normal(0, 0.03, position.shape)
    fit, held = rng.uniform(-1, 1, (60, 3)), rng.uniform(-1.1, 1.1, (37, 3))
    g = sparse_gaussian_vorticity_basis(fit, position, 0.4)
    velocity = gaussian_velocity_operator(fit, position, 0.4)
    candidate, diagnostics = regularized_projection(
        g, velocity, g @ strength, volume_velocity(fit, position, strength, 0.4)[0],
        0.7 * strength, np.ones(len(fit)), np.ones(len(fit)),
        velocity_weight=1, prior_weight=1e-8, max_strength_l1_ratio=2)
    assert diagnostics["segment_correction_fraction"] == 1
    actual = volume_velocity(held, position, candidate, 0.4)[0]
    expected = volume_velocity(held, position, strength, 0.4)[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)


def test_constrained_fit_matches_independent_slsqp_with_active_group_budget():
    rng = np.random.default_rng(79)
    prior = rng.normal(size=(4, 3))
    matrix = rng.normal(size=(31, 12))
    rhs = rng.normal(0, 10, 31)
    budget = 1.2 * np.linalg.norm(prior, axis=1).sum()

    def objective(x):
        residual = matrix @ (x - prior.ravel()) - rhs
        return 0.5 * residual @ residual

    def gradient(x):
        return matrix.T @ (matrix @ (x - prior.ravel()) - rhs)

    def constraint(x):
        return budget - np.linalg.norm(x.reshape(-1, 3), axis=1).sum()

    def constraint_gradient(x):
        rows = x.reshape(-1, 3)
        return (-rows / np.maximum(np.linalg.norm(rows, axis=1)[:, None], 1e-30)).ravel()

    reference = minimize(objective, prior.ravel(), jac=gradient, method="SLSQP",
                         constraints={"type": "ineq", "fun": constraint, "jac": constraint_gradient},
                         options={"ftol": 1e-11, "maxiter": 2000})
    assert reference.success, reference.message
    actual, diagnostics = constrained_least_squares(aslinearoperator(matrix), rhs, prior, budget, prior)
    assert diagnostics["constrained_converged"]
    assert constraint(actual.ravel()) > -2e-13
    np.testing.assert_allclose(actual.ravel(), reference.x, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(objective(actual.ravel()), reference.fun, rtol=2e-12)


def test_group_budget_projection_preserves_rotation_and_inactive_fields():
    rng = np.random.default_rng(331)
    values = rng.normal(size=(13, 3))
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    projected = project_strength_budget(values, 5)
    np.testing.assert_allclose(project_strength_budget(values @ rotation, 5), projected @ rotation,
                               rtol=0, atol=1e-15)
    np.testing.assert_array_equal(project_strength_budget(values, 100), values)
    np.testing.assert_allclose(np.linalg.norm(projected, axis=1).sum(), 5, rtol=1e-15)
