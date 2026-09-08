"""Finite Gaussian field limits at sampler/source coincidences."""

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.physics.induction.treecode.lbvh import TaichiTreecode


@pytest.mark.parametrize("order", [1, 3])
def test_gaussian_target_velocity_and_jacobian_have_the_correct_origin_limit(order):
    if ti.lang.impl.get_runtime().prog is None:
        ti.init(arch=ti.cpu)
    sigma = 0.04
    tree = TaichiTreecode(
        max_n_particles=8, max_nodes=16, theta=0.5, kernel_type="GAUSSIAN", multipole_order=order
    )
    tree.build(
        np.array([[0, 0, 0]], dtype=np.float32),
        np.array([[0, 0, 1]], dtype=np.float32),
        np.array([sigma], dtype=np.float32),
    )
    radius = np.array([0, 1e-7, 1e-6, 3e-6, 1e-5])
    targets = np.c_[radius, np.zeros((len(radius), 2))]
    velocity, gradient = tree.compute_target_velocity_and_gradients(targets)
    # Analytic Gaussian near-origin series, independently differentiated.
    q2 = (radius / sigma) ** 2
    constant = 1 / (3 * np.pi**1.5 * sigma**3)
    f = constant * (1 - 0.6 * q2 + 3 / 14 * q2**2)
    expected_v = np.zeros_like(velocity)
    expected_v[:, 1] = f * radius
    expected_g = np.zeros_like(gradient)
    expected_g[:, 0, 1] = -f
    expected_g[:, 1, 0] = constant * (1 - 1.8 * q2 + 15 / 14 * q2**2)
    assert np.isfinite(velocity).all() and np.isfinite(gradient).all()
    np.testing.assert_allclose(velocity, expected_v, rtol=2e-5, atol=1e-8)
    np.testing.assert_allclose(gradient, expected_g, rtol=2e-5, atol=1e-5)
