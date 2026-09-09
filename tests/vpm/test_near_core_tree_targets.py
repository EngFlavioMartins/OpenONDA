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


@pytest.mark.parametrize("order", [1, 2, 3])
def test_public_tree_approximation_survives_runtime_construction(order):
    import openonda.vpm as vpm

    configured = vpm.TreecodeInduction(theta=0.5, multipole_order=order, stretching_scheme="mixed")
    runtime = configured.build()
    assert runtime.theta == 0.5
    assert runtime.multipole_order == order
    assert runtime.stretching_scheme == "MIXED"


@pytest.mark.parametrize(
    "settings",
    [{"theta": 0}, {"theta": float("nan")}, {"multipole_order": 1.5}, {"multipole_order": 4}],
)
def test_public_tree_approximation_validates_in_the_solver_api(settings):
    import openonda.vpm as vpm

    with pytest.raises(ValueError, match="treecode"):
        vpm.TreecodeInduction(**settings)


@pytest.mark.parametrize("kernel_name", ["GAUSSIAN", "WINCKELMANS"])
def test_mixed_core_wake_matches_direct_near_and_far_fields(kernel_name):
    from source.solvers.vpm.kernels.base import make_vortex_kernel

    if ti.lang.impl.get_runtime().prog is None:
        ti.init(arch=ti.cpu, cpu_max_num_threads=2)
    rng = np.random.default_rng(583)
    position = rng.uniform([-2, -0.05, -0.05], [2, 0.05, 0.05], (192, 3)).astype(np.float32)
    strength = (rng.normal(size=(192, 3)) + [0, 0, 2]).astype(np.float32)
    radii = rng.uniform(0.02, 0.08, len(position)).astype(np.float32)
    near = position[::16].copy()
    far = rng.uniform([-3, 1, 1], [3, 2, 2], (12, 3)).astype(np.float32)
    remote = 10 * far
    targets = np.concatenate((near, far, remote))
    tree = TaichiTreecode(
        max_n_particles=256,
        max_nodes=512,
        theta=0.3,
        kernel_type=kernel_name,
        multipole_order=3,
    )
    tree.build(position, strength, radii)
    velocity, gradient = tree.compute_target_velocity_and_gradients(targets)
    kernel = make_vortex_kernel(kernel_name)
    displacement = targets.astype(float)[:, None, :] - position.astype(float)
    direct_velocity = kernel.velocity_pair(displacement, strength, radii, radii).sum(axis=1)
    direct_gradient = kernel.gradient_pair(displacement, strength, radii, radii).sum(axis=1)
    for actual, expected in ((velocity, direct_velocity), (gradient, direct_gradient)):
        for first in (0, len(near), len(near) + len(far)):
            difference = actual[first : first + 12] - expected[first : first + 12]
            relative_l2 = np.linalg.norm(difference) / np.linalg.norm(expected[first : first + 12])
            assert relative_l2 < 2e-3
