"""Direct blob Jacobians retain f64 precision for sub-micron separations."""

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.kernels.base import make_vortex_kernel
from source.solvers.vpm.physics.base import PhysicsBase
from source.solvers.vpm.physics.induction.direct import DirectInduction


@pytest.fixture(scope="module", autouse=True)
def f64_cpu_runtime():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False, cpu_max_num_threads=2)
    yield
    ti.reset()


@pytest.mark.parametrize("kernel_name", ["GAUSSIAN", "WINCKELMANS"])
def test_direct_f64_close_pair_gradient_matches_host_kernel(kernel_name):
    """A distinct sub-micron pair must retain the analytic f64 blob Jacobian."""
    physics = PhysicsBase(
        particle_kernel=kernel_name,
        max_n_particles=2,
        max_evaluation_points=2,
        accumulator_dtype=ti.f64,
    )
    induction = DirectInduction(stretching_scheme="TRANSPOSED").bind(
        physics, kernel=make_vortex_kernel(kernel_name)
    )
    position = ti.Vector.field(3, ti.f64, shape=2)
    strength = ti.Vector.field(3, ti.f64, shape=2)
    radius = ti.field(ti.f64, shape=2)
    velocity = ti.Vector.field(3, ti.f64, shape=2)
    rate = ti.Vector.field(3, ti.f64, shape=2)
    gradient = ti.Matrix.field(3, 3, ti.f64, shape=2)
    centres = np.array([[0.0, 0.0, 0.0], [1e-12, 0.0, 0.0]])
    strengths = np.array([[0.2, 0.4, -0.1], [-0.3, 0.1, 0.5]])
    radii = np.array([0.2, 0.3])
    position.from_numpy(centres)
    strength.from_numpy(strengths)
    radius.from_numpy(radii)
    induction.evaluate_stage(
        position=position,
        vortex_strength=strength,
        core_radius=radius,
        count=2,
        velocity_out=velocity,
        vortex_strength_rate_out=rate,
        velocity_gradient_out=gradient,
    )
    kernel = make_vortex_kernel(kernel_name)
    expected = np.array(
        [
            sum(
                kernel.gradient_pair(centres[i] - centres[j], strengths[j], radii[i], radii[j])
                for j in range(2)
            )
            for i in range(2)
        ]
    )
    assert np.isfinite(velocity.to_numpy()).all()
    assert np.isfinite(rate.to_numpy()).all()
    np.testing.assert_allclose(gradient.to_numpy(), expected, rtol=2e-13, atol=2e-13)
