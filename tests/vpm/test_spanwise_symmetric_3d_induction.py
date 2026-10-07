"""Planar symmetry must emerge from the common three-component kernel."""

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.physics.base import PhysicsBase
from source.solvers.vpm.physics.induction.direct import DirectInduction
from source.solvers.vpm.physics.induction.slip_slab import SlipSlabInduction


@pytest.fixture(scope="module", autouse=True)
def cpu_runtime():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=2, offline_cache=False)
    yield
    ti.reset()


def test_common_3d_stretching_preserves_coplanar_symmetry_and_responds_to_tilt():
    physics = PhysicsBase("GAUSSIAN", 3, ti.f64)
    model = DirectInduction(stretching_scheme="DIRECT").bind(physics)
    positions = ti.Vector.field(3, ti.f64, shape=3)
    strengths = ti.Vector.field(3, ti.f64, shape=3)
    radii = ti.field(ti.f64, shape=3)
    velocities = ti.Vector.field(3, ti.f64, shape=3)
    rates = ti.Vector.field(3, ti.f64, shape=3)
    gradients = ti.Matrix.field(3, 3, ti.f64, shape=3)
    positions.from_numpy(np.array([[0, 0, 0], [0.3, 0.1, 0], [-0.2, 0.4, 0]], float))
    gamma = np.array([[0, 0, 0.1], [0, 0, -0.2], [0, 0, 0.15]], float)
    radii.fill(0.1)
    for tilted in (False, True):
        if tilted:
            gamma[0, 0] = 0.07
        strengths.from_numpy(gamma)
        model.evaluate_stage(
            position=positions,
            vortex_strength=strengths,
            core_radius=radii,
            count=3,
            velocity_out=velocities,
            vortex_strength_rate_out=rates,
            velocity_gradient_out=gradients,
        )
        if tilted:
            assert np.max(np.abs(velocities.to_numpy()[:, 2])) > 0.01
            assert np.max(np.abs(rates.to_numpy()[:, :2])) > 0.001
        else:
            np.testing.assert_array_equal(velocities.to_numpy()[:, 2], 0)
            np.testing.assert_allclose(rates.to_numpy(), 0, atol=1e-15)


def test_full_3d_slab_column_converges_to_spanwise_uniform_gaussian_vortex():
    # The analytic filament is an independent test reference. The native
    # evaluator retains 3D particles, vector stretching and reflected sources.
    targets = np.array([[0.15, 0.1, 0], [0.4, -0.2, 0.13], [0.7, 0.3, -0.27]], float)
    sigma = 0.12
    r2 = np.sum(targets[:, :2] ** 2, axis=1)
    coefficient = -np.expm1(-r2 / sigma**2) / (2 * np.pi * r2)
    expected = np.column_stack(
        (-targets[:, 1] * coefficient, targets[:, 0] * coefficient, np.zeros(3))
    )
    errors = []
    for layers in (4, 8, 16):
        spacing = 1 / layers
        physics = PhysicsBase("GAUSSIAN", layers, ti.f64, max_evaluation_points=3)
        model = SlipSlabInduction(
            DirectInduction(),
            z_min=-0.5,
            z_max=0.5,
            tail_tolerance=1e-5,
            max_shells=257,
        ).bind(physics)
        positions = ti.Vector.field(3, ti.f64, shape=layers)
        strengths = ti.Vector.field(3, ti.f64, shape=layers)
        radii = ti.field(ti.f64, shape=layers)
        query = ti.Vector.field(3, ti.f64, shape=3)
        velocities = ti.Vector.field(3, ti.f64, shape=3)
        gradients = ti.Matrix.field(3, 3, ti.f64, shape=3)
        x = np.zeros((layers, 3))
        x[:, 2] = -0.5 + (np.arange(layers) + 0.5) * spacing
        gamma = np.zeros_like(x)
        gamma[:, 2] = spacing
        positions.from_numpy(x)
        strengths.from_numpy(gamma)
        radii.fill(sigma)
        query.from_numpy(targets)
        try:
            model.evaluate_targets(
                target_position=query,
                source_position=positions,
                source_vortex_strength=strengths,
                source_core_radius=radii,
                target_velocity=velocities,
                target_velocity_gradient=gradients,
                target_count=3,
                source_count=layers,
                include_freestream=False,
                background_velocity=physics._zero_velocity,
            )
            actual = velocities.to_numpy()
            errors.append(float(np.max(np.abs(actual - expected))))
            np.testing.assert_allclose(actual[:, 2], 0, atol=1e-12)
        finally:
            model.close_mesh_session()
    assert errors[0] > 10 * errors[-1]
    assert errors[-1] < 5e-5


def test_one_cubic_particle_in_thin_slab_uses_full_3d_image_kernel():
    span, sigma = 0.04, 0.04
    physics = PhysicsBase("GAUSSIAN", 1, ti.f64, max_evaluation_points=3)
    model = SlipSlabInduction(
        DirectInduction(),
        z_min=-span / 2,
        z_max=span / 2,
        tail_tolerance=1e-4,
        max_shells=1024,
    ).bind(physics)
    position = ti.Vector.field(3, ti.f64, shape=1)
    strength = ti.Vector.field(3, ti.f64, shape=1)
    radius = ti.field(ti.f64, shape=1)
    query = ti.Vector.field(3, ti.f64, shape=3)
    velocity = ti.Vector.field(3, ti.f64, shape=3)
    gradient = ti.Matrix.field(3, 3, ti.f64, shape=3)
    targets = np.array([[0.12, 0.05, 0], [0.2, -0.1, 0.015], [0.3, 0.2, -0.013]])
    position[0] = [0, 0, 0]
    strength[0] = [0, 0, span]
    radius.fill(sigma)
    query.from_numpy(targets)
    try:
        model.evaluate_targets(
            target_position=query,
            source_position=position,
            source_vortex_strength=strength,
            source_core_radius=radius,
            target_velocity=velocity,
            target_velocity_gradient=gradient,
            target_count=3,
            source_count=1,
            include_freestream=False,
            background_velocity=physics._zero_velocity,
        )
        r2 = np.sum(targets[:, :2] ** 2, axis=1)
        coefficient = -np.expm1(-r2 / sigma**2) / (2 * np.pi * r2)
        expected = np.column_stack(
            (-targets[:, 1] * coefficient, targets[:, 0] * coefficient, np.zeros(3))
        )
        np.testing.assert_allclose(velocity.to_numpy(), expected, rtol=1e-3, atol=2e-4)
    finally:
        model.close_mesh_session()
