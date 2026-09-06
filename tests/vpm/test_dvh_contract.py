"""Focused tests for the supported DVH diffusion contract."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from openonda import vpm


def _make_dvh_solver(tmp_path, *, time_step_size: float, viscosity: float):
    return vpm.VPMSolver(
        vpm.VPMCase(
            directory=tmp_path,
            backup=vpm.Backup(interval_steps=0),
            numerics=vpm.Numerics(
                time_step_size=time_step_size,
                compute_device="CPU",
                precision="f32",
                max_n_particles=4096,
                domain_bounds=(-0.6, 0.6, -0.6, 0.6, -0.6, 0.6),
                induction=vpm.DirectInduction(),
                viscous=vpm.ViscousConfig.dvh(
                    particle_spacing=0.1,
                    padding=6.0,
                    threshold=1.0e-12,
                    threshold_mode="absolute",
                    kinematic_viscosity=viscosity,
                    max_nodes=4096,
                    core_radius_ratio=2.0,
                ),
                verbose=False,
            ),
        )
    )


def _add_single_particle(solver, *, viscosity: float = 0.0) -> None:
    solver.add_vortex_particles(
        position=np.array([[0.0, 0.0, 0.0]]),
        velocity=np.zeros((1, 3)),
        vortex_strength=np.array([[0.0, 0.0, 1.0e-3]]),
        core_radius=np.array([0.2]),
        particle_volume=np.array([0.1**3]),
        kinematic_viscosity=np.array([viscosity]),
    )


def test_dvh_rejects_nonuniform_effective_viscosity(tmp_path):
    solver = _make_dvh_solver(tmp_path, time_step_size=0.2, viscosity=0.1)
    physics = solver.physics
    shape = physics._ensure_grid_capacity(15, 15, 15)
    position = np.stack(
        np.meshgrid(
            *[(-1.4 + 0.2 * np.arange(15)) for _ in range(3)],
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    strength = np.zeros_like(position)
    strength[:, 2] = 0.2**3

    with pytest.raises(ValueError, match="spatially uniform effective viscosity"):
        physics._dvh_scatter_vortex_strength(
            position,
            strength,
            np.full(3, -1.4),
            0.2,
            0.1,
            0.2,
            *shape,
            effective_viscosity_np=np.linspace(0.1, 0.2, len(position)),
        )
    small_effective = np.full(len(position), 1.0e-6)
    small_effective[0] = 1.5e-6
    with pytest.raises(ValueError, match="spatially uniform effective viscosity"):
        physics._dvh_scatter_vortex_strength(
            position,
            strength,
            np.full(3, -1.4),
            0.2,
            1.0e-6,
            1.0,
            *shape,
            effective_viscosity_np=small_effective,
        )
    solver.close()


def test_dvh_zero_viscosity_is_an_identity_for_off_lattice_particles(tmp_path):
    solver = _make_dvh_solver(tmp_path, time_step_size=0.1, viscosity=0.0)
    solver.add_vortex_particles(
        position=np.array([[0.033, 0.0, 0.0], [0.177, 0.0, 0.0]]),
        velocity=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        vortex_strength=np.array([[0.0, 0.0, 1.0e-3], [0.0, 2.0e-3, 0.0]]),
        core_radius=np.array([0.2, 0.2]),
        particle_volume=np.array([0.1**3, 0.1**3]),
        kinematic_viscosity=np.zeros(2),
    )
    before = {
        "position": solver.particles.position_cpu(use_cache=False).copy(),
        "velocity": solver.particles.velocity_cpu(use_cache=False).copy(),
        "vortex_strength": solver.particles.vortex_strength_cpu(use_cache=False).copy(),
        "core_radius": solver.particles.core_radius_cpu(use_cache=False).copy(),
        "particle_volume": solver.particles.particle_volume_cpu(use_cache=False).copy(),
        "kinematic_viscosity": solver.particles.kinematic_viscosity_cpu(use_cache=False).copy(),
        "eddy_viscosity": solver.particles.eddy_viscosity_cpu(use_cache=False).copy(),
    }

    new_particles = solver.stepper._apply_grid_diffusion(
        solver.stepper._viscous_config,
        0.1,
    )

    assert new_particles is None
    for name, expected in before.items():
        actual = getattr(solver.particles, f"{name}_cpu")(use_cache=False)
        np.testing.assert_array_equal(actual, expected)
    solver.close()


def test_dvh_accumulates_subresolution_accepted_steps(tmp_path):
    solver = _make_dvh_solver(tmp_path, time_step_size=0.01, viscosity=0.0308)
    _add_single_particle(solver, viscosity=0.0308)
    for _ in range(9):
        solver.stepper._apply_viscous_diffusion(0.01)
    assert solver._n_steps_since_dvh_diffusion == 9

    solver.stepper._apply_viscous_diffusion(0.01)
    assert solver._n_steps_since_dvh_diffusion == 0
    assert solver.particles.n_particles_total > 0
    np.testing.assert_allclose(
        solver.particles.vortex_strength_cpu(use_cache=False).sum(axis=0),
        [0.0, 0.0, 1.0e-3],
        rtol=0.0,
        atol=2.0e-7,
    )
    solver.close()


def test_dvh_fixed_grid_covers_repeated_heat_transfers(tmp_path):
    solver = _make_dvh_solver(tmp_path, time_step_size=0.1, viscosity=0.0308)
    _add_single_particle(solver, viscosity=0.0308)
    # Exercise the preallocated GPU layout on CPU. The active box must grow
    # by the heat support even when the configured padding is smaller.
    solver.physics.configure_max_grid_extent([-1.0, 1.0] * 3, 0.1, padding=2.0)
    allocation = solver.physics._grid_shape
    solver._viscous_config = replace(solver._viscous_config, dvh_domain_padding=2.0)
    try:
        for event in (1, 2):
            solver.stepper._apply_viscous_diffusion(0.1)
            position = solver.particles.position_cpu(use_cache=False)
            strength = solver.particles.vortex_strength_cpu(use_cache=False)[:, 2]
            total = strength.sum()
            np.testing.assert_allclose(total, 1.0e-3, rtol=2.0e-5)
            np.testing.assert_allclose(
                (strength[:, None] * position).sum(axis=0) / total, 0.0, atol=2.0e-6
            )
            center_moment = np.sum(strength * np.sum(position**2, axis=1)) / total
            assert center_moment == pytest.approx(6.0 * 0.0308 * 0.1 * event, rel=0.01)
            assert solver.physics._grid_shape == allocation
    finally:
        solver.close()


def test_dvh_fixed_grid_includes_sources_in_padding_halo(tmp_path):
    solver = _make_dvh_solver(tmp_path, time_step_size=0.1, viscosity=0.0308)
    _add_single_particle(solver, viscosity=0.0308)
    solver.particles.position[0] = [0.3, 0.0, 0.0]
    solver.particles.touch_state()
    # This particle is outside the nominal domain but its complete heat
    # support fits in the allocated halo. It must not disappear from bounds.
    solver.physics.configure_max_grid_extent([-0.2, 0.2] * 3, 0.1, padding=12.0)
    try:
        solver.stepper._apply_viscous_diffusion(0.1)
        position = solver.particles.position_cpu(use_cache=False)
        strength = solver.particles.vortex_strength_cpu(use_cache=False)[:, 2]
        np.testing.assert_allclose(
            (strength[:, None] * position).sum(axis=0) / strength.sum(),
            [0.3, 0.0, 0.0],
            atol=2.0e-6,
        )
    finally:
        solver.close()


def test_dvh_still_rejects_an_allocation_that_truncates_heat_support(tmp_path):
    solver = _make_dvh_solver(tmp_path, time_step_size=0.1, viscosity=0.0308)
    _add_single_particle(solver, viscosity=0.0308)
    solver.physics.configure_max_grid_extent([-0.1, 0.1] * 3, 0.1, padding=1.0)
    before = solver.particles.position_cpu(use_cache=False).copy()
    try:
        with pytest.raises(RuntimeError, match="heat support exceeds"):
            solver.stepper._apply_viscous_diffusion(0.1)
        np.testing.assert_array_equal(solver.particles.position_cpu(use_cache=False), before)
    finally:
        solver.close()
