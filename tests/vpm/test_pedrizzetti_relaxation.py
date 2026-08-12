"""Physics tests for Pedrizzetti relaxation of the particle strengths."""

import numpy as np
import pytest

from source.solvers.VPM import Solver, VPMSetup
from source.solvers.VPM.config.types import (
    AdvectionConfig,
    StabilizationConfig,
    StretchingConfig,
    VelocityConfig,
    ViscousConfig,
)


def _relaxation_solver(tmp_path, stabilization: StabilizationConfig) -> Solver:
    solver = Solver(
        setup=VPMSetup(
            processing_unit="CPU",
            precision="f64",
            max_particles=16,
            advection=AdvectionConfig(scheme="NONE"),
            stretching=StretchingConfig.disabled(),
            viscous=ViscousConfig.cs(viscosity=0.01, characteristic_distance=0.5),
            stabilization=stabilization,
            velocity=VelocityConfig.direct(),
            backup_frequency=0,
            logging_frequency=0,
            backup_directory=str(tmp_path),
        )
    )
    count = 2
    solver.add_vortex_particles(
        position=np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]]),
        velocity=np.zeros((count, 3)),
        circulation=np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        radius=np.full(count, 0.2),
        volume=np.full(count, 0.5**3),
        viscosity=np.full(count, 0.01),
        viscosity_turbulent=np.zeros(count),
    )
    # curl of this gradient is 2 e_z for both particles: the first strength is
    # perpendicular to it, the second is perpendicular and twice as strong.
    gradient = np.zeros((count, 3, 3))
    gradient[:, 1, 0] = 1.0
    gradient[:, 0, 1] = -1.0
    solver.particles.set_field("velocity_gradient", gradient)
    return solver


@pytest.mark.unit
def test_relaxation_rotates_toward_vorticity_and_keeps_each_strength(tmp_path):
    factor = 0.25
    solver = _relaxation_solver(tmp_path, StabilizationConfig.pedrizzetti_relaxation(factor=factor))

    diagnostics = solver.physics.apply_pedrizzetti_relaxation(solver.particles, factor)

    direction = np.array(
        [
            [1.0 - factor, 0.0, factor],
            [0.0, 1.0 - factor, factor],
        ]
    )
    expected = (
        np.array([1.0, 2.0])[:, None] * direction / np.linalg.norm(direction, axis=1)[:, None]
    )
    np.testing.assert_allclose(solver.particles.circulation_cpu(), expected, rtol=1.0e-12)
    assert diagnostics["pedrizzetti_misalignment_deg"] == pytest.approx(90.0)
    assert diagnostics["pedrizzetti_misalignment_max_deg"] == pytest.approx(90.0)
    assert diagnostics["pedrizzetti_strength_change_relative"] == pytest.approx(0.0, abs=1.0e-12)
    assert diagnostics["pedrizzetti_relaxed_fraction"] == pytest.approx(1.0)


@pytest.mark.unit
def test_uncorrected_relaxation_shortens_a_misaligned_strength(tmp_path):
    factor = 0.25
    solver = _relaxation_solver(
        tmp_path,
        StabilizationConfig.pedrizzetti_relaxation(factor=factor, conserve_strength=False),
    )

    diagnostics = solver.physics.apply_pedrizzetti_relaxation(
        solver.particles, factor, conserve_strength=False
    )

    # cos(theta) = 0 here, so every strength contracts by sqrt(1 - 2 f (1 - f)).
    contraction = np.sqrt(1.0 - 2.0 * factor * (1.0 - factor))
    strength = np.linalg.norm(solver.particles.circulation_cpu(), axis=1)
    np.testing.assert_allclose(strength, contraction * np.array([1.0, 2.0]), rtol=1.0e-12)
    assert diagnostics["pedrizzetti_strength_change_relative"] == pytest.approx(contraction - 1.0)


@pytest.mark.unit
def test_aligned_field_is_left_untouched(tmp_path):
    solver = _relaxation_solver(tmp_path, StabilizationConfig.pedrizzetti_relaxation())
    aligned = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
    solver.particles.set_field("circulation", aligned)

    diagnostics = solver.physics.apply_pedrizzetti_relaxation(solver.particles, 0.3)

    np.testing.assert_allclose(solver.particles.circulation_cpu(), aligned, atol=1.0e-12)
    assert diagnostics["pedrizzetti_misalignment_deg"] == pytest.approx(0.0, abs=1.0e-6)
    assert diagnostics["pedrizzetti_strength_change_relative"] == pytest.approx(0.0, abs=1.0e-12)


@pytest.mark.unit
def test_step_applies_the_schedule_and_rotates_without_changing_strength(tmp_path):
    solver = _relaxation_solver(
        tmp_path,
        StabilizationConfig.pedrizzetti_relaxation(factor=0.5, frequency=2, start_step=2),
    )

    solver.update_state()
    assert solver.stabilization.events == 0

    solver.update_state()
    assert solver.stabilization.events == 1
    assert solver.stabilization.last_mechanism == "Pedrizzetti relaxation"
    # A pure rotation neither creates strength nor amplifies peak vorticity.
    assert solver.stabilization.last_strength_growth == pytest.approx(0.0, abs=1e-12)
    assert solver.stabilization.last_vorticity_growth <= 1e-12
    np.testing.assert_allclose(
        np.linalg.norm(solver.particles.circulation_cpu(), axis=1),
        [1.0, 2.0],
        rtol=1.0e-12,
    )

    solver.update_state()
    assert solver.stabilization.events == 1


@pytest.mark.unit
def test_relaxation_configuration_is_validated_and_round_trips():
    stabilization = StabilizationConfig.pedrizzetti_relaxation(
        factor=0.4,
        frequency=5,
        start_step=10,
        conserve_strength=False,
    )
    restored = VPMSetup.from_dict(VPMSetup(stabilization=stabilization).to_dict())

    assert restored.stabilization == stabilization
    assert stabilization.pedrizzetti_relaxation_enabled
    assert not StabilizationConfig.disabled().pedrizzetti_relaxation_enabled
    with pytest.raises(ValueError, match="relaxation factor"):
        StabilizationConfig.pedrizzetti_relaxation(factor=1.5)
    with pytest.raises(ValueError, match="relaxation frequency"):
        StabilizationConfig.pedrizzetti_relaxation(frequency=0)
    with pytest.raises(ValueError, match="relaxation start step"):
        StabilizationConfig.pedrizzetti_relaxation(start_step=-1)
