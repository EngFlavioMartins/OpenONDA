"""Conservation tests for common-stage advection/stretching integration."""

from types import SimpleNamespace

import numpy as np
import pytest

from source.solvers.VPM import VPMSetup, VPMSolver
from source.solvers.VPM.config.types import (
    AdvectionConfig,
    StabilizationConfig,
    StretchingConfig,
    VelocityConfig,
    ViscousConfig,
)
from source.solvers.VPM.core.evolution import EvolutionStepper


def test_coupled_limits_can_be_disabled_for_one_common_stage():
    stepper = EvolutionStepper(
        SimpleNamespace(
            particles_velocity_gradients=np.full((2, 3, 3), 1.0e6),
            particles_velocities=np.full((2, 3), 1.0e6),
            coupled_max_strain_increment=None,
            coupled_max_advection_fraction=None,
            _viscous_config=SimpleNamespace(particle_spacing=0.03125),
        )
    )

    assert stepper._coupled_stable_time_step_size(0.03) == pytest.approx(0.03)


def test_coupled_transposed_step_preserves_total_strength(tmp_path):
    rng = np.random.default_rng(731)
    n_particles = 12
    position = rng.uniform(-0.5, 0.5, (n_particles, 3))
    circulation = 0.05 * rng.normal(size=(n_particles, 3))
    radius = np.full(n_particles, 0.18)
    volume = np.full(n_particles, 0.18**3)

    solver = VPMSolver(
        setup=VPMSetup(
            time_step_size=1.0e-3,
            time_integration="COUPLED",
            compute_device="CPU",
            precision="f64",
            advection=AdvectionConfig(scheme="RK2"),
            stretching=StretchingConfig.transposed(scheme="RK2"),
            viscous=ViscousConfig.inviscid(),
            stabilization=StabilizationConfig.disabled(),
            velocity=VelocityConfig.direct(),
            checkpoint_interval_steps=0,
            logging_interval_steps=0,
            checkpoint_directory=str(tmp_path),
        )
    )
    solver.add_vortex_particles(
        position=position,
        velocity=np.zeros_like(position),
        vortex_strength=circulation,
        core_radius=radius,
        volume=volume,
        kinematic_viscosity=np.zeros(n_particles),
    )

    strength_before = circulation.sum(axis=0)
    solver.advance()
    strength_after = solver.particle_vortex_strength.sum(axis=0)

    np.testing.assert_allclose(strength_after, strength_before, rtol=0.0, atol=2e-13)


def test_coupled_direct_projection_preserves_closed_flow_invariants_and_energy(tmp_path):
    rng = np.random.default_rng(904)
    n_particles = 24
    position = rng.uniform(-0.7, 0.7, (n_particles, 3))
    circulation = 0.08 * rng.normal(size=(n_particles, 3))
    radius = np.full(n_particles, 0.2)
    volume = np.full(n_particles, 0.2**3)

    solver = VPMSolver(
        setup=VPMSetup(
            time_step_size=2.0e-4,
            time_integration="COUPLED",
            compute_device="CPU",
            precision="f64",
            advection=AdvectionConfig(scheme="RK2"),
            stretching=StretchingConfig.direct(
                scheme="RK2", conserve_moments=True, conserve_energy=True
            ),
            viscous=ViscousConfig.inviscid(),
            stabilization=StabilizationConfig.disabled(),
            velocity=VelocityConfig.direct(),
            checkpoint_interval_steps=0,
            logging_interval_steps=0,
            checkpoint_directory=str(tmp_path),
        )
    )
    solver.add_vortex_particles(
        position=position,
        velocity=np.zeros_like(position),
        vortex_strength=circulation,
        core_radius=radius,
        volume=volume,
        kinematic_viscosity=np.zeros(n_particles),
    )

    strength_before = circulation.sum(axis=0)
    impulse_before = 0.5 * np.cross(position, circulation).sum(axis=0)
    angular_before = (
        np.cross(position, np.cross(position, circulation)).sum(axis=0) / 3.0
        - (radius[:, None] ** 2 * circulation).sum(axis=0) / 3.0
    )
    energy_before = solver.field_diagnostics.compute_flow_integrals(
        solver.particles, solver.time, record_history=False
    )["kinetic_energy"]
    for _ in range(20):
        solver.advance()
    evolved_position = solver.particles_positions
    evolved_circulation = solver.particle_vortex_strength
    energy_after = solver.field_diagnostics.compute_flow_integrals(
        solver.particles, solver.time, record_history=False
    )["kinetic_energy"]

    np.testing.assert_allclose(
        evolved_circulation.sum(axis=0), strength_before, rtol=0.0, atol=2e-12
    )
    np.testing.assert_allclose(
        0.5 * np.cross(evolved_position, evolved_circulation).sum(axis=0),
        impulse_before,
        rtol=0.0,
        atol=2e-10,
    )
    assert energy_after == pytest.approx(energy_before, rel=2.0e-10, abs=2.0e-13)
    np.testing.assert_allclose(
        np.cross(evolved_position, np.cross(evolved_position, evolved_circulation)).sum(axis=0)
        / 3.0
        - (radius[:, None] ** 2 * evolved_circulation).sum(axis=0) / 3.0,
        angular_before,
        rtol=0.0,
        atol=2e-10,
    )


def test_axisymmetric_coupled_stages_preserve_complete_particle_orbits(tmp_path):
    azimuth_count = 12
    theta = 2.0 * np.pi * np.arange(azimuth_count) / azimuth_count
    tangent = np.column_stack((np.zeros(azimuth_count), -np.sin(theta), np.cos(theta)))
    ring = np.column_stack((np.zeros(azimuth_count), np.cos(theta), np.sin(theta)))
    position = np.vstack((ring + [-0.3, 0.0, 0.0], ring + [0.3, 0.0, 0.0]))
    circulation = np.vstack((0.02 * tangent, 0.02 * tangent))
    orbit_id = np.repeat(np.arange(2, dtype=np.int32), azimuth_count)
    count = len(position)
    radius = np.full(count, 0.2)
    strength_before = circulation.sum(axis=0)
    impulse_before = 0.5 * np.cross(position, circulation).sum(axis=0)

    solver = VPMSolver(
        setup=VPMSetup(
            time_step_size=1.0e-3,
            time_integration="COUPLED",
            axisymmetric_no_swirl_axis="x",
            compute_device="CPU",
            precision="f64",
            advection=AdvectionConfig(scheme="RK2"),
            stretching=StretchingConfig.mixed(scheme="RK2", conserve_moments=True),
            viscous=ViscousConfig.inviscid(),
            stabilization=StabilizationConfig.disabled(),
            velocity=VelocityConfig.direct(),
            checkpoint_interval_steps=0,
            logging_interval_steps=0,
            checkpoint_directory=str(tmp_path),
        )
    )
    solver.add_vortex_particles(
        position=position,
        velocity=np.zeros_like(position),
        vortex_strength=circulation,
        core_radius=radius,
        volume=np.full(count, 0.2**3),
        kinematic_viscosity=np.zeros(count),
        group_id=orbit_id,
        zone_id=orbit_id,
    )

    for _ in range(5):
        solver.advance()

    evolved_position = solver.particles_positions
    evolved_circulation = solver.particle_vortex_strength
    for orbit in range(2):
        selected = orbit_id == orbit
        rho = np.linalg.norm(evolved_position[selected, 1:], axis=1)
        radial = evolved_position[selected, 1:] / rho[:, None]
        azimuthal = np.column_stack((-radial[:, 1], radial[:, 0]))
        cylindrical_circulation = np.column_stack(
            (
                evolved_circulation[selected, 0],
                np.einsum("ij,ij->i", evolved_circulation[selected, 1:], radial),
                np.einsum("ij,ij->i", evolved_circulation[selected, 1:], azimuthal),
            )
        )
        assert np.ptp(evolved_position[selected, 0]) < 2.0e-12
        assert np.ptp(rho) < 2.0e-12
        assert np.max(np.ptp(cylindrical_circulation, axis=0)) < 2.0e-12
        assert np.max(np.abs(cylindrical_circulation[:, :2])) < 2.0e-12
    np.testing.assert_allclose(
        evolved_circulation.sum(axis=0), strength_before, rtol=0.0, atol=2.0e-12
    )
    np.testing.assert_allclose(
        0.5 * np.cross(evolved_position, evolved_circulation).sum(axis=0),
        impulse_before,
        rtol=0.0,
        atol=2.0e-10,
    )


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {
                "advection": AdvectionConfig(scheme="RK3"),
                "stretching": StretchingConfig.transposed(scheme="RK2"),
            },
            "matching RK2 or RK3",
        ),
        (
            {
                "time_integration": "FRACTIONAL",
                "stretching": StretchingConfig.direct(conserve_moments=True),
            },
            "invariant projection requires COUPLED",
        ),
        (
            {
                "time_integration": "FRACTIONAL",
                "axisymmetric_no_swirl_axis": "x",
            },
            "axisymmetric_no_swirl_axis requires COUPLED",
        ),
    ],
)
def test_coupled_config_rejects_incompatible_physics(overrides, message):
    values = {
        "time_integration": "COUPLED",
        "advection": AdvectionConfig(scheme="RK2"),
        "stretching": StretchingConfig.transposed(scheme="RK2"),
        "viscous": ViscousConfig.inviscid(),
        "stabilization": StabilizationConfig.disabled(),
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        VPMSetup(**values)


def test_moment_projection_survives_config_round_trip():
    original = VPMSetup(
        time_integration="COUPLED",
        advection=AdvectionConfig(scheme="RK2"),
        stretching=StretchingConfig.mixed(
            scheme="RK2",
            use_treecode=True,
            conserve_moments=True,
            conserve_energy=True,
        ),
        viscous=ViscousConfig.cs(),
        axisymmetric_no_swirl_axis="x",
    )

    restored = VPMSetup.from_dict(original.to_dict())

    assert restored.stretching == original.stretching
    assert restored.axisymmetric_no_swirl_axis == "x"


def test_coupled_dvh_runs_after_the_inviscid_update(tmp_path, monkeypatch):
    solver = VPMSolver(
        VPMSetup(
            time_step_size=0.01,
            time_integration="COUPLED",
            advection=AdvectionConfig(scheme="RK2"),
            stretching=StretchingConfig.mixed(scheme="RK2"),
            viscous=ViscousConfig.dvh(particle_spacing=0.05, kinematic_viscosity=1.0e-3),
            compute_device="CPU",
            checkpoint_directory=str(tmp_path),
            checkpoint_interval_steps=0,
            logging_interval_steps=0,
        )
    )
    calls = []
    stepper = solver.stepper
    monkeypatch.setattr(stepper, "_coupled_stable_time_step_size", lambda remaining: remaining)
    monkeypatch.setattr(
        stepper,
        "_apply_coupled_advection_stretching",
        lambda time_step_size, *, precomputed_velocity_k1: calls.append(
            ("inviscid", time_step_size)
        ),
    )
    monkeypatch.setattr(
        stepper,
        "_apply_viscous_diffusion",
        lambda time_step_size: calls.append(("diffusion", time_step_size)),
    )

    stepper._apply_coupled_update_with_subcycling(0.01, precomputed_velocity_k1=False)

    assert calls == [("inviscid", 0.01), ("diffusion", 0.01)]
