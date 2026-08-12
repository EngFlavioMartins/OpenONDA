"""The stabilization master: its global criteria and its dispatch."""

import numpy as np
import pytest

from source.solvers.VPM import Solver, StabilizationConfig, VPMSetup
from source.solvers.VPM.config.types import (
    AdvectionConfig,
    DivergenceRelaxationConfig,
    FilamentRefinementConfig,
    StretchingConfig,
    VelocityConfig,
    ViscousConfig,
)
from source.solvers.VPM.stabilization import StabilizationError


def _solver(tmp_path, stabilization: StabilizationConfig) -> Solver:
    solver = Solver(
        setup=VPMSetup(
            processing_unit="CPU",
            precision="f64",
            max_particles=16,
            advection=AdvectionConfig(scheme="NONE"),
            stretching=StretchingConfig.disabled(),
            viscous=ViscousConfig(scheme="NONE"),
            stabilization=stabilization,
            velocity=VelocityConfig.direct(),
            backup_frequency=0,
            logging_frequency=0,
            backup_directory=str(tmp_path),
        )
    )
    count = 3
    solver.add_vortex_particles(
        position=np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.3, 0.0]]),
        velocity=np.zeros((count, 3)),
        circulation=np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]),
        radius=np.full(count, 0.2),
        volume=np.full(count, 0.5**3),
        viscosity=np.zeros(count),
    )
    return solver


@pytest.mark.unit
def test_health_is_measured_from_the_uploaded_field(tmp_path):
    solver = _solver(tmp_path, StabilizationConfig.disabled())

    health = solver.stabilization.measure()

    assert health.particles == 3
    np.testing.assert_allclose(health.circulation, [1.0, 2.0, 3.0])
    assert health.strength_magnitude == pytest.approx(6.0)
    assert health.peak_strength == pytest.approx(3.0)
    assert health.peak_vorticity == pytest.approx(3.0 / 0.5**3)


@pytest.mark.unit
def test_master_accepts_a_dissipative_event_and_rejects_an_amplifying_one(tmp_path):
    solver = _solver(tmp_path, StabilizationConfig.disabled())
    manager = solver.stabilization
    before = manager.measure()

    # Halving every strength is what a filter does: strength and peak vorticity
    # fall, and the master lets it through even though circulation moved with it.
    solver.set_particles_properties(strengths=0.5 * solver.particles.circulation_cpu())
    manager.accept("test filter", before, conserves_circulation=False)
    assert manager.events == 1
    assert manager.last_strength_growth == pytest.approx(-0.5)
    assert manager.last_vorticity_growth == pytest.approx(-0.5)

    # Doubling it back is amplification, and no mechanism is allowed to do that.
    amplified = manager.measure()
    solver.set_particles_properties(strengths=4.0 * solver.particles.circulation_cpu())
    with pytest.raises(StabilizationError, match="test amplifier produced a .* growth"):
        manager.accept("test amplifier", amplified, conserves_circulation=False)


@pytest.mark.unit
def test_growth_criteria_do_not_apply_to_a_rebuilt_discretization(tmp_path):
    """Total variation and peak vorticity are per-particle sums.

    A worker that redistributes the cloud onto its own grid reports them
    against a different discretization, so the master records the numbers but
    holds such a worker to circulation alone.
    """
    solver = _solver(tmp_path, StabilizationConfig.disabled())
    manager = solver.stabilization
    before = manager.measure()

    # Same field, coarser representation: one particle carrying the total.
    solver.replace_vortex_particles(
        position=np.zeros((1, 3)),
        velocity=np.zeros((1, 3)),
        circulation=before.circulation.reshape(1, 3),
        radius=np.full(1, 0.2),
        volume=np.full(1, 0.5**3),
        viscosity=np.zeros(1),
    )

    manager.accept("test remesh", before, preserves_discretization=False)

    assert manager.events == 1
    assert manager.last_strength_growth < 0.0
    with pytest.raises(StabilizationError, match="strength growth|peak-vorticity growth"):
        manager.accept("test remesh", before)


@pytest.mark.unit
def test_master_holds_conserving_mechanisms_to_their_circulation(tmp_path):
    solver = _solver(tmp_path, StabilizationConfig.disabled())
    manager = solver.stabilization
    before = manager.measure()

    circulation = solver.particles.circulation_cpu()
    circulation[0] += np.array([0.1, 0.0, 0.0])
    solver.set_particles_properties(strengths=circulation)

    with pytest.raises(StabilizationError, match="circulation error"):
        manager.accept("test reassignment", before)


@pytest.mark.unit
def test_active_mechanisms_names_the_configured_policy(tmp_path):
    stabilization = StabilizationConfig(
        stretching_viscosity_coefficient=0.5,
        pedrizzetti_relaxation_factor=0.3,
        filament_refinement=FilamentRefinementConfig.adaptive(frequency=10),
        divergence_relaxation=DivergenceRelaxationConfig.constrained(
            frequency=10, grid_spacing=0.1
        ),
        remove_particles_by_bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
    )
    solver = _solver(tmp_path, stabilization)

    assert solver.stabilization.active_mechanisms() == (
        "residual stretching viscosity",
        "Pedrizzetti relaxation",
        "filament refinement",
        "divergence relaxation",
        "bounded-domain retention",
    )
    assert set(solver.stabilization.diagnostics) == {
        "stabilization_events",
        "stabilization_last_mechanism",
        "stabilization_circulation_error",
        "stabilization_strength_growth",
        "stabilization_vorticity_growth",
        "stabilization_max_vorticity_growth",
    }


@pytest.mark.unit
def test_rejected_regularization_restores_the_original_field(tmp_path):
    """A refused redistribution must leave the cloud bit-for-bit unchanged."""
    spacing = 0.1
    coordinates = np.arange(-3, 4) * spacing
    position = (
        np.array(np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")).reshape(3, -1).T
    )
    radius_squared = (position**2).sum(axis=1)
    vorticity = (
        np.column_stack((-position[:, 1], position[:, 0], np.zeros(len(position))))
        * np.exp(-radius_squared / 0.05)[:, None]
    )
    volume = np.full(len(position), spacing**3)

    solver = Solver(
        setup=VPMSetup(
            processing_unit="CPU",
            precision="f64",
            max_particles=2048,
            advection=AdvectionConfig(scheme="NONE"),
            stretching=StretchingConfig.disabled(),
            viscous=ViscousConfig.cs(viscosity=1e-3, characteristic_distance=spacing),
            velocity=VelocityConfig.direct(),
            stabilization=StabilizationConfig.conservative_filter(
                frequency=1,
                start_step=0,
                grid_spacing=spacing,
                max_particles=1024,
                divergence_trigger=0.0,
                misalignment_trigger=0.0,
            ),
            backup_frequency=0,
            logging_frequency=0,
            backup_directory=str(tmp_path),
        )
    )
    solver.add_vortex_particles(
        position=position,
        velocity=np.zeros_like(position),
        circulation=vorticity * volume[:, None],
        radius=np.full(len(position), 1.5 * spacing),
        volume=volume,
        viscosity=np.full(len(position), 1e-3),
    )
    circulation = solver.particles.circulation_cpu().copy()
    solver.time_step = 1

    # This cloud is far too small for the redistribution grid, so the worker's
    # own dissipation limits refuse the candidate it builds.
    with pytest.raises(RuntimeError, match="regularization"):
        solver.stabilization.apply_regularization()

    np.testing.assert_array_equal(solver.particles.position_cpu(), position)
    np.testing.assert_array_equal(solver.particles.circulation_cpu(), circulation)
    assert solver.stabilization.events == 0
