from types import SimpleNamespace

import numpy as np
import pytest

from source.solvers.vpm.config.filament_refinement import FilamentRefinementConfig
from source.solvers.vpm.config.stabilization import StabilizationConfig
from source.solvers.vpm.stabilization.divergence_relaxation import (
    restore_particle_moments,
)
from source.solvers.vpm.stabilization.filament_refinement import (
    particle_moments,
    split_stretched_filaments,
)
from source.solvers.vpm.stabilization.manager import StabilizationError, StabilizationManager
from source.solvers.vpm.stabilization.regularization import _regularization_triggered


def test_combined_stabilization_schedule_is_representable():
    refinement = FilamentRefinementConfig.adaptive(
        interval_steps=25,
        max_vortex_strength_factor=3.0,
        max_n_particles=60_000,
        max_absolute_vortex_strength=0.5,
        late_interval_steps=5,
        late_start_step=750,
        late_absolute_only=True,
        end_step=800,
    )
    config = StabilizationConfig(
        stretching_viscosity_coefficient=1.6,
        stretching_viscosity_start_step=550,
        stretching_viscosity_feedback_gain=1.0,
        stretching_viscosity_feedback_growth_limit=0.5,
        stretching_viscosity_max_coefficient=8.0,
        pedrizzetti_relaxation_factor=0.005,
        pedrizzetti_relaxation_end_step=650,
        filament_refinement=refinement,
        regularization_interval_steps=25,
        regularization_start_step=475,
        regularization_grid_spacing=0.055,
        regularization_max_particles=30_000,
        regularization_capacity_max_particles=45_000,
        regularization_capacity_energy_rate_trigger=4.0,
        regularization_max_events=2,
    )

    assert config.filament_refinement.end_step == 800
    assert config.filament_refinement.late_interval_steps == 5
    assert config.filament_refinement.late_start_step == 750
    assert config.filament_refinement.late_absolute_only
    assert config.regularization_max_events == 2
    assert config.regularization_capacity_energy_rate_trigger == pytest.approx(4.0)
    assert config.stretching_viscosity_feedback_growth_limit == pytest.approx(0.5)
    assert config.stretching_viscosity_max_coefficient == pytest.approx(8.0)
    assert config.pedrizzetti_relaxation_end_step == 650


def test_pedrizzetti_relaxation_stops_at_end_step():
    state = {"step": 625}
    config = StabilizationConfig(
        pedrizzetti_relaxation_factor=0.005,
        pedrizzetti_relaxation_interval_steps=25,
        pedrizzetti_relaxation_end_step=650,
    )
    calls = []
    manager = object.__new__(StabilizationManager)
    manager.config = config
    manager.ctx = SimpleNamespace(
        step=lambda: state["step"],
        flow_model="VISCOUS",
        particles=object(),
    )
    manager.operators = SimpleNamespace(
        apply_pedrizzetti_relaxation=lambda *args, **kwargs: (
            calls.append(state["step"]) or {"pedrizzetti_misalignment_deg": 10.0}
        )
    )
    manager.measure = lambda: object()
    manager.accept = lambda *args, **kwargs: None

    manager.apply_relaxation()
    state["step"] = 650
    manager.apply_relaxation()

    assert calls == [625]


def test_pedrizzetti_moment_correction_restores_closed_field_invariants():
    rng = np.random.default_rng(12)
    position = rng.normal(size=(16, 3))
    reference = rng.normal(size=(16, 3))
    relaxed = reference + 0.04 * rng.normal(size=(16, 3))
    core_radius = rng.uniform(0.06, 0.09, size=16)
    particle_volume = rng.uniform(0.8, 1.2, size=16)

    corrected, correction_relative = restore_particle_moments(
        position,
        relaxed,
        core_radius,
        particle_volume,
        reference,
        angular_core_coefficient=1.0 / 3.0,
    )
    target = particle_moments(
        position,
        reference,
        core_radius,
        angular_core_coefficient=1.0 / 3.0,
    )
    restored = particle_moments(
        position,
        corrected,
        core_radius,
        angular_core_coefficient=1.0 / 3.0,
    )

    assert correction_relative > 0.0
    for index in (0, 2, 3):
        np.testing.assert_allclose(restored[index], target[index], rtol=1.0e-12, atol=1.0e-12)


def test_regularization_event_limit_stops_the_schedule(monkeypatch):
    config = StabilizationConfig(
        regularization_interval_steps=5,
        regularization_start_step=10,
        regularization_grid_spacing=0.1,
        regularization_max_particles=100,
        regularization_max_events=2,
    )
    manager = object.__new__(StabilizationManager)
    manager.config = config
    manager.ctx = SimpleNamespace(
        step=lambda: 10,
        set_domain_bounds_enforced=lambda value: None,
    )
    manager.regularization_events = 0
    manager.measure = lambda: object()
    manager.accept = lambda *args, **kwargs: None

    calls = []

    def regularize(context, active_config):
        calls.append((context, active_config))
        return SimpleNamespace(detail="accepted")

    monkeypatch.setattr(
        "source.solvers.vpm.stabilization.regularization.regularize",
        regularize,
    )

    manager.apply_regularization()
    manager.apply_regularization()
    manager.apply_regularization()

    assert len(calls) == 2
    assert manager.regularization_events == 2


def test_regularization_can_be_triggered_only_by_core_radius():
    health = {
        "vorticity_divergence_error": 1.0,
        "vortex_strength_misalignment_degrees": 90.0,
    }
    arguments = {
        "divergence_trigger": None,
        "misalignment_trigger": None,
        "core_radius_trigger": 0.2,
        "energy_growth": False,
    }

    assert not _regularization_triggered(health, np.array([0.1, 0.199]), **arguments)
    assert _regularization_triggered(health, np.array([0.1, 0.2]), **arguments)


def test_filament_refinement_prioritizes_strongest_particles_at_capacity():
    strength = np.array([[2.0, 0.0, 0.0], [5.0, 0.0, 0.0], [4.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    result = split_stretched_filaments(
        np.zeros((4, 3)),
        strength,
        np.ones(4),
        np.ones(4),
        reference_vortex_strength=np.ones(4),
        reference_length=np.ones(4),
        max_stretch_factor=1.5,
        max_n_particles=6,
    )

    assert result.refined_parent_index.tolist() == [1, 2]
    assert result.refined_particles == 2
    assert len(result.position) == 6
    assert 0 in result.source_index


def test_filament_refinement_catches_absolute_strength_after_reference_reset():
    result = split_stretched_filaments(
        np.zeros((2, 3)),
        np.array([[0.8, 0.0, 0.0], [0.4, 0.0, 0.0]]),
        np.ones(2),
        np.ones(2),
        reference_vortex_strength=np.array([0.4, 0.4]),
        reference_length=np.ones(2),
        max_stretch_factor=3.0,
        max_absolute_vortex_strength=0.5,
    )

    assert result.refined_parent_index.tolist() == [0]
    assert np.linalg.norm(result.vortex_strength, axis=1).max() == pytest.approx(0.4)


def test_absolute_only_refinement_ignores_lineage_growth():
    result = split_stretched_filaments(
        np.zeros((2, 3)),
        np.array([[0.8, 0.0, 0.0], [0.4, 0.0, 0.0]]),
        np.ones(2),
        np.ones(2),
        reference_vortex_strength=np.array([0.1, 0.1]),
        reference_length=np.ones(2),
        max_stretch_factor=np.inf,
        max_absolute_vortex_strength=0.5,
    )

    assert result.refined_parent_index.tolist() == [0]


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("max_absolute_vortex_strength", 0.0),
        ("late_interval_steps", 0),
        ("late_start_step", -1),
    ],
)
def test_filament_refinement_rejects_invalid_staging(keyword, value):
    arguments = {
        "interval_steps": 10,
        "late_interval_steps": 5,
        "late_start_step": 750,
        keyword: value,
    }
    with pytest.raises(ValueError):
        FilamentRefinementConfig.adaptive(**arguments)


def test_late_absolute_only_requires_an_absolute_threshold():
    with pytest.raises(ValueError):
        FilamentRefinementConfig.adaptive(
            interval_steps=10,
            late_interval_steps=5,
            late_start_step=750,
            late_absolute_only=True,
        )


def test_residual_viscosity_feedback_is_bounded_per_update():
    state = {"step": 550, "energy_rate": 4.0, "viscous_rate": -2.0}
    config = StabilizationConfig(
        stretching_viscosity_coefficient=1.6,
        stretching_viscosity_start_step=550,
        stretching_viscosity_feedback_gain=1.0,
        stretching_viscosity_feedback_interval_steps=5,
        stretching_viscosity_feedback_growth_limit=0.5,
        stretching_viscosity_max_coefficient=8.0,
    )
    applied = []
    manager = object.__new__(StabilizationManager)
    manager.config = config
    manager.ctx = SimpleNamespace(
        step=lambda: state["step"],
        kinetic_energy_rate=lambda: state["energy_rate"],
        viscous_kinetic_energy_rate=lambda: state["viscous_rate"],
        particles=object(),
    )
    manager.operators = SimpleNamespace(
        apply_stretching_viscosity=lambda particles, coefficient: applied.append(coefficient)
    )
    manager.residual_viscosity_coefficient = 1.6
    manager._last_residual_feedback_step = -1

    manager.update_residual_viscosity()
    manager.update_residual_viscosity()
    state.update(step=555, energy_rate=-1.0)
    manager.update_residual_viscosity()

    assert applied == pytest.approx([2.4, 2.4, 1.92])


def test_solution_stability_check_uses_current_lagrangian_cfl_number():
    config = StabilizationConfig(max_lagrangian_cfl=0.8)
    manager = object.__new__(StabilizationManager)
    manager.config = config
    manager.ctx = SimpleNamespace(
        step=lambda: 12,
        time_step_size=lambda: 0.05,
        particles=object(),
    )
    manager.operators = SimpleNamespace(
        inspect_solution=lambda *args, **kwargs: {
            "valid": True,
            "lagrangian_cfl": 0.9,
        }
    )
    manager.lagrangian_cfl = 0.0

    with pytest.raises(StabilizationError, match="Lagrangian CFL number 0.9"):
        manager.check_solution_stability()


def test_solution_stability_check_can_be_disabled():
    manager = object.__new__(StabilizationManager)
    manager.config = StabilizationConfig(max_lagrangian_cfl=None)
    manager.ctx = SimpleNamespace()
    manager.operators = SimpleNamespace(
        inspect_solution=lambda *args, **kwargs: pytest.fail("disabled check ran")
    )

    manager.check_solution_stability()


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("stretching_viscosity_feedback_gain", -1.0),
        ("stretching_viscosity_feedback_growth_limit", 1.1),
        ("stretching_viscosity_max_coefficient", 0.5),
        ("pedrizzetti_relaxation_end_step", -1),
        ("regularization_capacity_max_particles", 0),
        ("regularization_capacity_energy_rate_trigger", -1.0),
        ("regularization_max_events", 0),
        ("max_lagrangian_cfl", 0.0),
    ],
)
def test_stabilization_schedule_rejects_invalid_limits(keyword, value):
    arguments = {keyword: value}
    if keyword == "stretching_viscosity_max_coefficient":
        arguments["stretching_viscosity_coefficient"] = 1.0
    with pytest.raises(ValueError):
        StabilizationConfig(**arguments)
