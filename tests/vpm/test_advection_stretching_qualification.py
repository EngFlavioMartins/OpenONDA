"""Compact permanent regressions selected by the offline qualification study."""

import numpy as np

from studies.vpm.advection_stretching import setup
from studies.vpm.advection_stretching.assets.core import (
    AnalyticEvaluator,
    State,
    contract,
    errors,
    flows,
    integrate,
    pair_rate,
)
from studies.vpm.advection_stretching.assets.run_manufactured import random_flow


def test_strength_tensor_orientation():
    j = np.broadcast_to(np.array(((0.0, 2.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))), (3, 3, 3))
    basis = np.eye(3)
    np.testing.assert_array_equal(contract(j, basis, "DIRECT")[1], (2.0, 0.0, 0.0))
    np.testing.assert_array_equal(contract(j, basis, "TRANSPOSED")[1], (0.0, 0.0, 0.0))
    np.testing.assert_array_equal(contract(j, basis, "MIXED")[1], (1.0, 0.0, 0.0))


def test_nonlinear_deformation_has_third_order_coupled_convergence():
    initial = State(*setup.cloud())
    flow = flows()[-1]
    reference = flow.exact(initial, 1.0, "DIRECT")
    e16 = errors(
        integrate("coupled_rk3", AnalyticEvaluator(flow), initial, 1.0, 16, "DIRECT"), reference
    )["strength_error"]
    e32 = errors(
        integrate("coupled_rk3", AnalyticEvaluator(flow), initial, 1.0, 32, "DIRECT"), reference
    )["strength_error"]
    assert e16 / e32 > 2**2.5


def test_joint_position_strength_closed_cycle_regression():
    initial = State(*setup.cloud())
    flow = flows()[-1]
    result = integrate("coupled_rk3", AnalyticEvaluator(flow), initial, 1.0, 8, "DIRECT")
    np.testing.assert_allclose(
        result.x[0], (-0.45487914, -0.44906286, -0.45173778), rtol=2e-7, atol=2e-8
    )
    np.testing.assert_allclose(
        result.gamma[0], (1.00725443, -0.000513091849, -0.00166756822), rtol=2e-6, atol=2e-8
    )


def test_pairwise_transposed_rate_conserves_total_strength():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(16, 3))
    gamma = rng.normal(size=(16, 3))
    rate = pair_rate(x, gamma, np.full(16, 0.3), "TRANSPOSED")
    np.testing.assert_allclose(rate.sum(axis=0), 0.0, atol=8e-15)


def test_retained_random_challenge_seed_is_finite_and_bounded():
    initial = State(*setup.cloud())
    flow = random_flow(20261028)
    reference = flow.exact(initial, 1.0, "DIRECT")
    result = integrate("coupled_rk3", AnalyticEvaluator(flow), initial, 1.0, 16, "DIRECT")
    assert np.isfinite(result.x).all() and np.isfinite(result.gamma).all()
    assert errors(result, reference)["strength_error"] < 0.005
