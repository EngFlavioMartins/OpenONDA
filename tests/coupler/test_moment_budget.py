"""Independent constrained optimization and physical invariants for the study."""

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

from studies.coupler_accuracy.moment_budget import MomentBudgetProjector, particle_moments
from studies.coupler_accuracy.native_reconstruction_3d import fit_native


def test_projection_matches_independent_slsqp_with_active_budget_and_all_moments():
    rng = np.random.default_rng(839)
    position, prior = rng.normal(size=(7, 3)), rng.normal(size=(7, 3))
    candidate = rng.normal(0, 5, size=prior.shape)
    budget = 1.4 * np.linalg.norm(prior, axis=1).sum()
    project = MomentBudgetProjector(position, prior, budget)
    actual = project(candidate)

    def physical_equalities(x):
        old = particle_moments(position, prior)
        new = particle_moments(position, x.reshape(-1, 3))
        return np.r_[new[0] - old[0], new[1] - old[1]]

    reference = minimize(lambda x: 0.5 * np.sum((x - candidate.ravel())**2), prior.ravel(),
                         jac=lambda x: x - candidate.ravel(), method="SLSQP",
                         constraints=[{"type": "eq", "fun": physical_equalities},
                                      {"type": "ineq", "fun": lambda x: budget - np.linalg.norm(x.reshape(-1, 3), axis=1).sum()}],
                         options={"ftol": 1e-11, "maxiter": 1000})
    assert reference.success, reference.message
    np.testing.assert_allclose(actual.ravel(), reference.x, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(physical_equalities(actual.ravel()), 0, rtol=0, atol=5e-12)
    np.testing.assert_allclose(np.linalg.norm(actual, axis=1).sum(), budget, rtol=2e-15)
    np.testing.assert_allclose(project(actual), actual, rtol=0, atol=1e-12)


def test_physical_moment_projection_is_translation_and_rotation_covariant():
    rng = np.random.default_rng(912)
    position, prior = rng.normal(size=(11, 3)), rng.normal(size=(11, 3))
    candidate = rng.normal(0, 3, size=prior.shape)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotation[:, 0] *= np.linalg.det(rotation)
    budget = 1.3 * np.linalg.norm(prior, axis=1).sum()
    expected = MomentBudgetProjector(position, prior, budget)(candidate) @ rotation
    actual = MomentBudgetProjector(position @ rotation + [23, -45, 78], prior @ rotation, budget)(candidate @ rotation)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-12)


def test_projection_refuses_an_infeasible_anchor_and_preserves_feasible_donor():
    position = np.eye(3)
    prior = np.array([[0.4, 0.1, 0.2], [-0.3, 0.2, 0.1], [0.2, 0.3, -0.4]])
    project = MomentBudgetProjector(position, prior, 2 * np.linalg.norm(prior, axis=1).sum())
    np.testing.assert_allclose(project(prior), prior, rtol=0, atol=2e-16)
    with pytest.raises(ValueError, match="feasible"):
        MomentBudgetProjector(position, prior, 0.5)


def test_complete_native_fit_matches_independent_constrained_objective():
    rng = np.random.default_rng(739)
    position = rng.normal(size=(6, 3))
    prior = rng.normal(0, 0.05, size=position.shape)
    velocity_map = rng.normal(size=(45, prior.size))
    observation = csr_matrix(rng.normal(size=(30, 45)))
    background, target = rng.normal(size=45), rng.normal(0, 3, size=30)
    a, b = observation @ velocity_map, observation @ background - target
    penalty = 0.05 / np.linalg.norm(prior)
    budget = 2 * np.linalg.norm(prior, axis=1).sum()
    original_moments = particle_moments(position, prior)

    def objective(x):
        return 0.5 * np.linalg.norm(a @ x + b)**2 + 0.5 * penalty**2 * np.linalg.norm(x - prior.ravel())**2

    def equalities(x):
        current = particle_moments(position, x.reshape(-1, 3))
        return np.r_[current[0] - original_moments[0], current[1] - original_moments[1]]

    reference = minimize(objective, prior.ravel(), method="SLSQP",
                         jac=lambda x: a.T @ (a @ x + b) + penalty**2 * (x - prior.ravel()),
                         constraints=[{"type": "eq", "fun": equalities},
                                      {"type": "ineq", "fun": lambda x: budget - np.linalg.norm(x.reshape(-1, 3), axis=1).sum()}],
                         options={"ftol": 1e-10, "maxiter": 2000})
    assert reference.success, reference.message
    actual, diagnostics = fit_native(velocity_map, background, observation, target, prior, position,
                                      conserve_moments=True, max_iterations=6000)
    assert diagnostics["constrained_converged"]
    np.testing.assert_allclose(equalities(actual.ravel()), 0, rtol=0, atol=5e-13)
    np.testing.assert_allclose(objective(actual.ravel()), reference.fun, rtol=2e-10)
    np.testing.assert_allclose(actual.ravel(), reference.x, rtol=2e-5, atol=2e-6)
