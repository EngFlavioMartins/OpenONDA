"""Compact permanent regressions selected by the offline qualification study."""
from pathlib import Path
import sys
import numpy as np

STUDY = Path(__file__).resolve().parents[2] / "studies/vpm/advection_stretching"
sys.path.insert(0, str(STUDY))

import setup  # noqa: E402
from assets.core import (AnalyticEvaluator, State, contract, errors, flows, gradient,
                         integrate, pair_rate)  # noqa: E402
from assets.run_manufactured import random_flow  # noqa: E402


def test_strength_tensor_orientation():
    j = np.broadcast_to(np.array(((0., 2., 0.), (0., 0., 0.), (0., 0., 0.))), (3, 3, 3))
    basis = np.eye(3)
    np.testing.assert_array_equal(contract(j, basis, "DIRECT")[1], (2., 0., 0.))
    np.testing.assert_array_equal(contract(j, basis, "TRANSPOSED")[1], (0., 0., 0.))
    np.testing.assert_array_equal(contract(j, basis, "MIXED")[1], (1., 0., 0.))


def test_nonlinear_deformation_has_third_order_coupled_convergence():
    initial = State(*setup.cloud()); flow = flows()[-1]; reference = flow.exact(initial, 1., "DIRECT")
    e16 = errors(integrate("coupled_rk3", AnalyticEvaluator(flow), initial, 1., 16, "DIRECT"), reference)["strength_error"]
    e32 = errors(integrate("coupled_rk3", AnalyticEvaluator(flow), initial, 1., 32, "DIRECT"), reference)["strength_error"]
    assert e16 / e32 > 2**2.5


def test_joint_position_strength_closed_cycle_regression():
    initial = State(*setup.cloud()); flow = flows()[-1]
    result = integrate("coupled_rk3", AnalyticEvaluator(flow), initial, 1., 8, "DIRECT")
    np.testing.assert_allclose(result.x[0], (-.45487914, -.44906286, -.45173778), rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(result.gamma[0], (1.00725443, -.000513091849, -.00166756822), rtol=2e-6, atol=2e-8)


def test_pairwise_transposed_rate_conserves_total_strength():
    rng = np.random.default_rng(4); x = rng.normal(size=(16, 3)); gamma = rng.normal(size=(16, 3))
    rate = pair_rate(x, gamma, np.full(16, .3), "TRANSPOSED")
    np.testing.assert_allclose(rate.sum(axis=0), 0., atol=8e-15)


def test_retained_random_challenge_seed_is_finite_and_bounded():
    initial = State(*setup.cloud()); flow = random_flow(20261028); reference = flow.exact(initial, 1., "DIRECT")
    result = integrate("coupled_rk3", AnalyticEvaluator(flow), initial, 1., 16, "DIRECT")
    assert np.isfinite(result.x).all() and np.isfinite(result.gamma).all()
    assert errors(result, reference)["strength_error"] < .005
