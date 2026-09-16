"""The regularized velocity Jacobian stays finite at particle centres."""

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.kernels.base import make_vortex_kernel
from source.solvers.vpm.physics.base import PhysicsBase
from source.solvers.vpm.physics.induction.direct import DirectInduction
from source.solvers.vpm.physics.induction.fmm import FMMInduction
from source.solvers.vpm.physics.induction.treecode import TreecodeInduction


@pytest.fixture(scope="module", autouse=True)
def cpu_runtime():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f32, offline_cache=False, cpu_max_num_threads=2)
    yield
    ti.reset()


def fields(backend, kernel_name, count, scheme="transposed"):
    physics = PhysicsBase(
        particle_kernel=kernel_name,
        max_n_particles=count,
        max_evaluation_points=count,
        accumulator_dtype=ti.f32,
    )
    induction = backend(stretching_scheme=scheme).bind(
        physics, kernel=make_vortex_kernel(kernel_name)
    )
    position = ti.Vector.field(3, ti.f32, shape=count)
    strength = ti.Vector.field(3, ti.f32, shape=count)
    radius = ti.field(ti.f32, shape=count)
    velocity = ti.Vector.field(3, ti.f32, shape=count)
    rate = ti.Vector.field(3, ti.f32, shape=count)
    gradient = ti.Matrix.field(3, 3, ti.f32, shape=count)
    position.fill(0)
    return physics, induction, position, strength, radius, velocity, rate, gradient


@pytest.mark.parametrize("backend", [DirectInduction, TreecodeInduction, FMMInduction])
@pytest.mark.parametrize("kernel_name", ["GAUSSIAN", "WINCKELMANS"])
@pytest.mark.parametrize("count", [1, 2])
def test_particle_centre_gradient_matches_velocity_derivative(backend, kernel_name, count):
    _, induction, x, g, r, u, rate, gradient = fields(backend, kernel_name, count)
    strengths = np.array([[0.2, 0.4, -0.1], [-0.3, 0.1, 0.5]], dtype=np.float32)[:count]
    radii = np.array([0.2, 0.3], dtype=np.float32)[:count]
    g.from_numpy(strengths)
    r.from_numpy(radii)
    induction.evaluate_stage(
        position=x,
        vortex_strength=g,
        core_radius=r,
        count=count,
        velocity_out=u,
        vortex_strength_rate_out=rate,
        velocity_gradient_out=gradient,
    )
    kernel = make_vortex_kernel(kernel_name)
    epsilon = 2e-5
    offsets = epsilon * np.eye(3)
    # Independent central differences of the regularized velocity. A source at
    # the evaluation point has zero velocity, but a nonzero derivative.
    expected = np.array(
        [
            sum(
                (
                    kernel.velocity_pair(offsets, alpha, target_radius, source_radius)
                    - kernel.velocity_pair(-offsets, alpha, target_radius, source_radius)
                ).T
                / (2 * epsilon)
                for alpha, source_radius in zip(strengths, radii, strict=True)
            )
            for target_radius in radii
        ]
    )
    np.testing.assert_allclose(gradient.to_numpy(), expected, rtol=4e-5, atol=3e-7)
    np.testing.assert_allclose(u.to_numpy(), 0, atol=1e-12)
    np.testing.assert_allclose(
        rate.to_numpy(), np.einsum("nji,nj->ni", expected, strengths), rtol=4e-5, atol=3e-7
    )
    # The centre term is skew: it supplies no strain or self stretching.
    np.testing.assert_allclose(expected + expected.swapaxes(1, 2), 0, atol=1e-10)


@pytest.mark.parametrize("kernel_name", ["GAUSSIAN", "WINCKELMANS"])
@pytest.mark.parametrize("separation", [1e-8, 0.06])
@pytest.mark.parametrize("backend", [DirectInduction, TreecodeInduction, FMMInduction])
def test_nearly_coincident_sources_have_finite_jacobians(backend, kernel_name, separation):
    physics, induction, x, g, r, u, rate, gradient = fields(backend, kernel_name, 3)
    position = np.array(
        [[0.0, 0.0, 0.0], [separation, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
    )
    strength = np.array([[0.2, 0.4, -0.1], [-0.3, 0.1, 0.5], [50.0, 50.0, 50.0]], dtype=np.float32)
    radius = np.array([0.2, 0.3, 0.01], dtype=np.float32)
    x.from_numpy(position)
    g.from_numpy(strength)
    r.from_numpy(radius)
    kernel = make_vortex_kernel(kernel_name)

    # The tiny pair uses the finite centre limit; the resolved core pair uses
    # the independent host Jacobian. The inactive third source must not enter.
    displacement = position[:2, None, :].astype(np.float64) - position[None, :2, :]
    if separation < 1e-7:
        displacement.fill(0.0)

    expected_particle_gradient = np.array(
        [
            sum(
                kernel.gradient_pair(displacement[i, j], strength[j], radius[i], radius[j])
                for j in range(2)
            )
            for i in range(2)
        ]
    )
    induction.evaluate_stage(
        position=x,
        vortex_strength=g,
        core_radius=r,
        count=2,
        velocity_out=u,
        vortex_strength_rate_out=rate,
        velocity_gradient_out=gradient,
    )
    assert np.isfinite(u.to_numpy()[:2]).all()
    assert np.isfinite(gradient.to_numpy()[:2]).all()
    assert np.isfinite(rate.to_numpy()[:2]).all()
    np.testing.assert_allclose(
        gradient.to_numpy()[:2], expected_particle_gradient, rtol=5e-5, atol=3e-7
    )
    np.testing.assert_allclose(
        rate.to_numpy()[:2],
        np.einsum("nji,nj->ni", expected_particle_gradient, strength[:2]),
        rtol=5e-5,
        atol=3e-7,
    )

    induction.evaluate_targets(
        target_position=x,
        source_position=x,
        source_vortex_strength=g,
        source_core_radius=r,
        target_velocity=u,
        target_velocity_gradient=gradient,
        target_count=2,
        source_count=2,
        include_freestream=False,
        background_velocity=physics._zero_velocity,
    )
    expected_target_gradient = np.array(
        [
            sum(
                kernel.gradient_pair(displacement[i, j], strength[j], radius[j], radius[j])
                for j in range(2)
            )
            for i in range(2)
        ]
    )
    assert np.isfinite(gradient.to_numpy()[:2]).all()
    np.testing.assert_allclose(
        gradient.to_numpy()[:2],
        expected_target_gradient,
        rtol=5e-5,
        atol=3e-7,
    )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_direct_fused_and_target_gradients_include_finite_core(kernel_name):
    physics, induction, x, g, r, u, rate, gradient = fields(DirectInduction, kernel_name, 2)
    strengths = np.array([[0.2, 0.4, -0.1], [-0.3, 0.1, 0.5]], dtype=np.float32)
    radii = np.array([0.2, 0.3], dtype=np.float32)
    g.from_numpy(strengths)
    r.from_numpy(radii)
    kernel = make_vortex_kernel(kernel_name)
    expected = kernel.gradient_pair(
        np.zeros((2, 2, 3)), strengths[None, :, :], radii[:, None], radii[None, :]
    ).sum(axis=1)
    strain = ti.Matrix.field(3, 3, ti.f32, shape=2)
    physics.compute_velocity_and_gradient_kernel(
        x, g, r, u, gradient, strain, physics._zero_velocity, 2
    )
    np.testing.assert_allclose(gradient.to_numpy(), expected, rtol=4e-5, atol=3e-7)
    np.testing.assert_allclose(strain.to_numpy(), 0, atol=1e-12)
    for mode in range(3):
        expected_rate = np.einsum(
            "nij,nj->ni",
            expected
            if mode == 0
            else expected.swapaxes(1, 2)
            if mode == 1
            else 0.5 * (expected + expected.swapaxes(1, 2)),
            strengths,
        )
        physics.compute_velocity_and_stretching_rate_kernel(
            x, g, r, u, rate, physics._zero_velocity, mode, 2
        )
        np.testing.assert_allclose(rate.to_numpy(), expected_rate, rtol=4e-5, atol=3e-7)
        physics.compute_stretching_rate_kernel(x, g, r, rate, mode, 2)
        np.testing.assert_allclose(rate.to_numpy(), expected_rate, rtol=4e-5, atol=3e-7)
    induction.evaluate_targets(
        target_position=x,
        source_position=x,
        source_vortex_strength=g,
        source_core_radius=r,
        target_velocity=u,
        target_velocity_gradient=gradient,
        target_count=2,
        source_count=2,
        include_freestream=False,
        background_velocity=physics._zero_velocity,
    )
    expected_target = kernel.gradient_pair(np.zeros((2, 3)), strengths, radii, radii).sum(axis=0)
    np.testing.assert_allclose(
        gradient.to_numpy(), np.broadcast_to(expected_target, (2, 3, 3)), rtol=4e-5, atol=3e-7
    )
