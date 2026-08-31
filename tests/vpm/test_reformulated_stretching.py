"""Reformulated VPM couples filament strength and core size."""

import numpy as np
import pytest

taichi = pytest.importorskip("taichi", reason="VPM requires taichi")

from source.solvers.vpm import AdvectionConfig, StretchingConfig, TurbulenceConfig, VPMSetup
from source.solvers.vpm.acceleration.treecode_gpu import TaichiTreecode
from source.solvers.vpm.particles.container import Particles
from source.solvers.vpm.physics.engine import PhysicsEngine


def _ensure_taichi_cpu() -> None:
    if taichi.lang.impl.get_runtime().prog is None:
        taichi.init(arch=taichi.cpu)


def test_reformulated_stretching_rate_preserves_local_tube_relation():
    _ensure_taichi_cpu()
    engine = PhysicsEngine(max_n_particles=2, max_evaluation_points=1)

    strength = taichi.Vector.field(3, dtype=taichi.f32, shape=2)
    radius = taichi.field(dtype=taichi.f32, shape=2)
    strength_rate = taichi.Vector.field(3, dtype=taichi.f32, shape=2)
    radius_rate = taichi.field(dtype=taichi.f32, shape=2)

    strength.from_numpy(np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32))
    radius.from_numpy(np.array([0.5, 0.5], dtype=np.float32))
    strength_rate.from_numpy(np.array([[6.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32))

    engine.reformulated_stretching_rate_kernel(
        strength, radius, strength_rate, radius_rate, 2
    )

    np.testing.assert_allclose(
        strength_rate.to_numpy(),
        np.array([[2.4, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        rtol=2.0e-6,
    )
    np.testing.assert_allclose(radius_rate.to_numpy(), [-0.3, 0.0], rtol=2.0e-6)


def test_coupled_rk3_advances_core_size_with_strength():
    _ensure_taichi_cpu()
    particles = Particles(max_n_particles=3)
    position = np.array(
        [[-0.4, 0.1, 0.0], [0.3, -0.2, 0.2], [0.1, 0.5, -0.3]],
        dtype=np.float32,
    )
    strength = np.array(
        [[0.2, 0.7, -0.1], [-0.4, 0.3, 0.5], [0.6, -0.2, 0.4]],
        dtype=np.float32,
    )
    radius = np.full(3, 0.15, dtype=np.float32)
    particles.add_vortex_particles(
        position=position,
        velocity=np.zeros_like(position),
        vortex_strength=strength,
        core_radius=radius,
        particle_volume=np.full(3, 0.01, dtype=np.float32),
        kinematic_viscosity=np.zeros(3, dtype=np.float32),
    )
    engine = PhysicsEngine(max_n_particles=3, max_evaluation_points=1)
    engine.configure_velocity("DIRECT")

    tube_measure = np.linalg.norm(strength, axis=1) * radius**2
    engine.update_positions_and_strengths(
        particles,
        1.0e-4,
        scheme="RK3",
        mode="TRANSPOSED",
        reformulated=True,
    )

    new_radius = particles.core_radius.to_numpy()[:3]
    new_strength = particles.vortex_strength.to_numpy()[:3]
    assert np.max(np.abs(new_radius - radius)) > 1.0e-8
    assert np.all(new_radius > 0.0)
    np.testing.assert_allclose(
        np.linalg.norm(new_strength, axis=1) * new_radius**2,
        tube_measure,
        rtol=2.0e-5,
    )


def test_reformulated_configuration_requires_coupled_integration():
    stretching = StretchingConfig.transposed(scheme="RK3", reformulated=True)
    setup = VPMSetup(
        time_integration="COUPLED",
        advection=AdvectionConfig(scheme="RK3"),
        stretching=stretching,
    )

    assert setup.stretching.reformulated
    assert VPMSetup.from_dict(setup.to_dict()).stretching.reformulated

    with pytest.raises(ValueError, match="requires COUPLED"):
        VPMSetup(stretching=stretching)


def test_reformulated_configuration_supports_moment_projection_only():
    stretching = StretchingConfig.transposed(
        reformulated=True,
        conserve_moments=True,
    )
    assert stretching.conserve_moments

    with pytest.raises(ValueError, match="energy-rate projection"):
        StretchingConfig.transposed(
            reformulated=True,
            conserve_moments=True,
            conserve_energy=True,
        )


def test_reformulated_projection_includes_changing_core_radius():
    _ensure_taichi_cpu()
    particles = Particles(max_n_particles=4)
    position = np.array(
        [[-0.4, 0.1, 0.0], [0.3, -0.2, 0.2], [0.1, 0.5, -0.3], [0.2, -0.3, -0.1]],
        dtype=np.float32,
    )
    strength = np.array(
        [[0.2, 0.7, -0.1], [-0.4, 0.3, 0.5], [0.6, -0.2, 0.4], [-0.4, -0.8, -0.8]],
        dtype=np.float32,
    )
    radius = np.array([0.14, 0.15, 0.16, 0.17], dtype=np.float32)
    particles.add_vortex_particles(
        position=position,
        velocity=np.zeros_like(position),
        vortex_strength=strength,
        core_radius=radius,
        particle_volume=np.full(4, 0.01, dtype=np.float32),
        kinematic_viscosity=np.zeros(4, dtype=np.float32),
    )
    engine = PhysicsEngine(max_n_particles=4, max_evaluation_points=1)
    engine.configure_velocity("DIRECT")

    def moments(pos, gamma, sigma):
        total = gamma.sum(axis=0, dtype=np.float64)
        linear = 0.5 * np.cross(pos, gamma).sum(axis=0, dtype=np.float64)
        angular = np.cross(pos, np.cross(pos, gamma)).sum(axis=0, dtype=np.float64) / 3.0
        angular -= (
            engine._angular_core_coefficient
            * (sigma[:, None] ** 2 * gamma).sum(axis=0, dtype=np.float64)
        )
        return total, linear, angular

    before = moments(position, strength, radius)
    engine.update_positions_and_strengths(
        particles,
        1.0e-4,
        scheme="RK3",
        mode="TRANSPOSED",
        conserve_moments=True,
        reformulated=True,
    )
    after = moments(
        particles.position.to_numpy()[:4],
        particles.vortex_strength.to_numpy()[:4],
        particles.core_radius.to_numpy()[:4],
    )

    for observed, expected in zip(after, before, strict=True):
        np.testing.assert_allclose(observed, expected, atol=2.0e-6, rtol=2.0e-5)


def test_vortex_stretching_sfs_clips_backscatter():
    _ensure_taichi_cpu()
    engine = PhysicsEngine(max_n_particles=2, max_evaluation_points=1)

    position = taichi.Vector.field(3, dtype=taichi.f32, shape=2)
    strength = taichi.Vector.field(3, dtype=taichi.f32, shape=2)
    radius = taichi.field(dtype=taichi.f32, shape=2)
    gradient = taichi.Matrix.field(3, 3, dtype=taichi.f32, shape=2)
    strength_rate = taichi.Vector.field(3, dtype=taichi.f32, shape=2)
    sfs_rate = taichi.Vector.field(3, dtype=taichi.f32, shape=2)

    position.from_numpy(np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]], dtype=np.float32))
    strength.from_numpy(np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32))
    radius.from_numpy(np.array([0.5, 0.5], dtype=np.float32))
    gradient.from_numpy(
        np.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=np.float32,
        )
    )
    strength_rate.fill(0.0)

    engine.vortex_stretching_sfs_rate_kernel(
        position,
        strength,
        radius,
        gradient,
        strength_rate,
        sfs_rate,
        1.0,
        4.0,
        1,
        2,
    )

    expected_forward_transfer = np.exp(-0.5**2)
    np.testing.assert_allclose(
        sfs_rate.to_numpy(),
        [[-expected_forward_transfer, 0.0, 0.0], [0.0, 0.0, 0.0]],
        rtol=2.0e-6,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(strength_rate.to_numpy(), sfs_rate.to_numpy())


def test_tree_sfs_matches_direct_compact_sum():
    _ensure_taichi_cpu()
    engine = PhysicsEngine(max_n_particles=2, max_evaluation_points=1)
    tree = TaichiTreecode(
        max_n_particles=2,
        max_nodes=4,
        theta=0.3,
        kernel_type="GAUSSIAN",
        sort_particle_targets=True,
    )
    position = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]], dtype=np.float32)
    strength = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    radius = np.array([0.5, 0.5], dtype=np.float32)
    gradient = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    tree.build(position, strength, radius)
    tree.velocity_gradient.from_numpy(gradient)

    direct_rate = taichi.Vector.field(3, dtype=taichi.f32, shape=2)
    direct_sfs = taichi.Vector.field(3, dtype=taichi.f32, shape=2)
    tree_rate = taichi.Vector.field(3, dtype=taichi.f32, shape=2)
    tree_sfs = taichi.Vector.field(3, dtype=taichi.f32, shape=2)
    direct_rate.fill(0.0)
    tree_rate.fill(0.0)

    engine.vortex_stretching_sfs_rate_kernel(
        tree.position,
        tree.vortex_strength,
        tree.core_radius,
        tree.velocity_gradient,
        direct_rate,
        direct_sfs,
        1.0,
        4.0,
        1,
        2,
    )
    tree.apply_vortex_stretching_sfs(tree_rate, tree_sfs, 1.0, 4.0, 1)

    np.testing.assert_allclose(tree_sfs.to_numpy(), direct_sfs.to_numpy(), rtol=2.0e-6)
    np.testing.assert_allclose(tree_rate.to_numpy(), direct_rate.to_numpy(), rtol=2.0e-6)


def test_vortex_stretching_sfs_requires_reformulated_vpm():
    turbulence = TurbulenceConfig.les_smagorinsky(
        vortex_stretching_sfs_coefficient=1.0
    )
    with pytest.raises(ValueError, match="requires reformulated"):
        VPMSetup(turbulence=turbulence)

    setup = VPMSetup(
        time_integration="COUPLED",
        advection=AdvectionConfig(scheme="RK3"),
        stretching=StretchingConfig.transposed(scheme="RK3", reformulated=True),
        turbulence=turbulence,
    )
    restored = VPMSetup.from_dict(setup.to_dict())
    assert restored.turbulence.vortex_stretching_sfs_coefficient == 1.0


def test_coupled_rk3_applies_vortex_stretching_sfs():
    _ensure_taichi_cpu()
    particles = Particles(max_n_particles=3)
    position = np.array(
        [[-0.3, 0.1, 0.0], [0.2, -0.2, 0.1], [0.1, 0.4, -0.2]],
        dtype=np.float32,
    )
    strength = np.array(
        [[0.5, 0.3, -0.1], [-0.2, 0.4, 0.3], [0.4, -0.1, 0.2]],
        dtype=np.float32,
    )
    particles.add_vortex_particles(
        position=position,
        velocity=np.zeros_like(position),
        vortex_strength=strength,
        core_radius=np.full(3, 0.4, dtype=np.float32),
        particle_volume=np.full(3, 0.01, dtype=np.float32),
        kinematic_viscosity=np.zeros(3, dtype=np.float32),
    )
    engine = PhysicsEngine(max_n_particles=3, max_evaluation_points=1)
    engine.configure_velocity("DIRECT")

    engine.update_positions_and_strengths(
        particles,
        1.0e-4,
        scheme="RK3",
        mode="TRANSPOSED",
        reformulated=True,
        vortex_stretching_sfs_coefficient=1.0,
    )

    correction = engine.sfs_rate_temp.to_numpy()[:3]
    assert np.all(np.isfinite(correction))
