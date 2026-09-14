"""Particle state: numerical and lifecycle contracts."""

from __future__ import annotations

import numpy as np
import pytest
import taichi as ti

from openonda import vpm
from source.solvers.vpm.config import RestartState
from source.solvers.vpm.config.state import cached_particle_property
from source.solvers.vpm.particles.container import Particles
from source.solvers.vpm.stabilization.operators import StabilizationOperators


class _RevisionedParticles:
    def __init__(self) -> None:
        self.state_revision = 0
        self.evaluations = 0

    @cached_particle_property
    def position_cpu(self):
        self.evaluations += 1
        return self.state_revision


def check_bounded_replacement(arch):
    """Exercise shrinking, growing and chunk-boundary replacements on a device."""
    ti.init(arch=arch, offline_cache=False)
    try:
        particles = Particles(max_n_particles=140_000)
        rng = np.random.default_rng(73)
        for count in (17, 65_549, 53, 0, 29):
            state = {
                "position": rng.normal(size=(count, 3)).astype(np.float32),
                "velocity": rng.normal(size=(count, 3)).astype(np.float32),
                "vortex_strength": rng.normal(size=(count, 3)).astype(np.float32),
                "core_radius": rng.uniform(0.1, 0.2, count).astype(np.float32),
                "particle_volume": rng.uniform(0.01, 0.02, count).astype(np.float32),
                "kinematic_viscosity": rng.uniform(0.001, 0.002, count).astype(np.float32),
                "eddy_viscosity": rng.uniform(0.002, 0.004, count).astype(np.float32),
                "group_id": np.arange(count, dtype=np.int32) % 5,
                "zone_id": np.arange(count, dtype=np.int32) % 7,
                "velocity_gradient": rng.normal(size=(count, 3, 3)).astype(np.float32),
                "strain_rate": rng.normal(size=(count, 3, 3)).astype(np.float32),
            }
            particles.replace_from_numpy(**state)
            assert particles.n_particles_total == count
            for name, expected in state.items():
                np.testing.assert_array_equal(getattr(particles, name + "_cpu")(), expected)
        assert not particles._native_matrix_uploads
        assert not particles._native_vector_uploads
    finally:
        ti.reset()


def test_particle_replacement_preserves_every_field_across_chunk_boundaries():
    check_bounded_replacement(ti.cpu)


def test_particle_snapshot_cache_invalidates_on_source_revision_not_step():
    particles = _RevisionedParticles()

    assert particles.position_cpu() == 0
    assert particles.position_cpu() == 0
    assert particles.evaluations == 1

    # The solver step need not change for a coupled/stabilization mutation.
    particles.state_revision += 1
    assert particles.position_cpu() == 1
    assert particles.evaluations == 2


def test_realignment_invalidates_cached_strength_and_preserves_norm():
    ti.init(arch=ti.cpu, offline_cache=False)
    try:
        particles = Particles(max_n_particles=8)
        particles.add_vortex_particles(
            position=np.zeros((1, 3)),
            velocity=np.zeros((1, 3)),
            vortex_strength=np.array([[1.0, 0.0, 0.0]]),
            core_radius=np.ones(1),
            particle_volume=np.ones(1),
            kinematic_viscosity=np.zeros(1),
        )
        gradient = np.zeros((8, 3, 3), dtype=np.float32)
        gradient[0, 0, 2] = 1.0  # curl(u) = (0, 1, 0)
        particles.velocity_gradient.from_numpy(gradient)
        before = particles.vortex_strength_cpu().copy()
        revision = particles.state_revision
        operator = StabilizationOperators(ti.f32, 8)
        operator.apply_pedrizzetti_relaxation(particles, 0.3)
        after = particles.vortex_strength_cpu()
        assert particles.state_revision > revision
        assert not np.allclose(after, before)
        np.testing.assert_allclose(
            after[0], np.array([0.7, 0.3, 0.0]) / np.hypot(0.7, 0.3), rtol=1e-6
        )
        np.testing.assert_allclose(np.linalg.norm(after, axis=1), 1.0, rtol=1e-6)
    finally:
        ti.reset()


def test_pressure_is_invariant_to_redundant_refresh_after_deferred_advance(tmp_path):
    solver = vpm.VPMSolver(
        vpm.VPMCase(
            directory=tmp_path,
            backup=vpm.Backup(interval_steps=0),
            numerics=vpm.Numerics(
                time_step_size=0.01,
                compute_device="CPU",
                precision="f32",
                max_n_particles=16,
                max_evaluation_points=16,
                induction=vpm.DirectInduction(),
                viscous=vpm.ViscousConfig.inviscid(),
                verbose=False,
            ),
        )
    )
    solver.add_vortex_particles(
        position=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        velocity=np.zeros((2, 3)),
        vortex_strength=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        core_radius=np.full(2, 0.2),
        particle_volume=np.full(2, 1.0e-3),
        kinematic_viscosity=np.zeros(2),
    )

    body_gradient = np.diag([1.0, -1.0, 0.0])

    def body_velocity(points, _time):
        return np.asarray(points) * np.array([1.0, -1.0, 0.0])

    def body_velocity_gradient(points, _time):
        return np.broadcast_to(body_gradient, (len(points), 3, 3)).copy()

    solver.set_body_induced_velocity(body_velocity, body_velocity_gradient)
    solver.advance(defer_output=True)
    target = np.array([[0.5, 0.3, 0.1]])

    before = solver.compute_pressure_gradient_at_points(
        target,
        include_viscous=False,
        particle_spacing=0.05,
    )
    solver.stepper._update_velocity_and_gradients()
    after = solver.compute_pressure_gradient_at_points(
        target,
        include_viscous=False,
        particle_spacing=0.05,
    )

    for name in before:
        np.testing.assert_allclose(before[name], after[name], rtol=0.0, atol=1.0e-12, err_msg=name)
    solver.close()


def test_restart_state_rejects_negative_clock_values():
    with pytest.raises(ValueError, match="time"):
        RestartState(time=-0.01)
    with pytest.raises(ValueError, match="step"):
        RestartState(step=-1)
