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
    state = SimpleNamespace(step=625)
    config = StabilizationConfig(
        pedrizzetti_relaxation_factor=0.005,
        pedrizzetti_relaxation_interval_steps=25,
        pedrizzetti_relaxation_end_step=650,
    )
    calls = []
    manager = object.__new__(StabilizationManager)
    manager.config = config
    manager.ctx = SimpleNamespace(
        state=state,
        flow_model="VISCOUS",
        particles=SimpleNamespace(
            n_particles_total=1, vortex_strength_cpu=lambda **kwargs: np.ones((1, 3))
        ),
    )
    manager.operators = SimpleNamespace(
        apply_pedrizzetti_relaxation=lambda *args, **kwargs: (
            calls.append(state.step) or {"pedrizzetti_misalignment_deg": 10.0}
        )
    )
    manager.ctx.particles.position_cpu = lambda **kw: np.zeros((1, 3))
    manager.ctx.particles.core_radius_cpu = lambda **kw: np.ones(1)
    manager.ctx.physics = SimpleNamespace(_angular_core_coefficient=1.0 / 3.0)
    manager.pedrizzetti_moment_transfer = np.zeros((3, 3))
    manager.measure = lambda: object()
    manager.accept = lambda *args, **kwargs: None

    manager.apply_relaxation()
    state.step = 650
    manager.apply_relaxation()

    assert calls == [625]


@pytest.mark.parametrize("preserve_moments", [False, True])
def test_pedrizzetti_accepts_an_empty_wake_before_first_shedding(preserve_moments):
    manager = object.__new__(StabilizationManager)
    manager.config = StabilizationConfig.pedrizzetti_relaxation(preserve_moments=preserve_moments)
    manager.ctx = SimpleNamespace(flow_model="LES", particles=SimpleNamespace(n_particles_total=0))
    # There is no particle state to measure or mutate at this initial clock.
    manager.apply_relaxation()


@pytest.mark.parametrize("operator_failure", [False, True])
def test_rejected_relaxation_restores_strength_and_does_not_record_acceptance(operator_failure):
    original = np.array([[1.0, 2.0, 3.0]])
    particles = SimpleNamespace(n_particles_total=1, strength=original.copy())
    particles.vortex_strength_cpu = lambda **kwargs: particles.strength
    particles.particle_volume_cpu = lambda **kwargs: np.ones(1)
    particles.position_cpu = lambda **kwargs: np.zeros((1, 3))
    particles.core_radius_cpu = lambda **kwargs: np.ones(1)
    manager = object.__new__(StabilizationManager)
    manager.config = StabilizationConfig.pedrizzetti_relaxation()
    manager.events = 7
    manager.last_mechanism = "previous accepted event"
    manager.pedrizzetti_moment_transfer = np.ones((3, 3))
    manager.ctx = SimpleNamespace(
        flow_model="LES",
        particles=particles,
        state=SimpleNamespace(step=0),
        physics=SimpleNamespace(_angular_core_coefficient=1.0 / 3.0),
        mutations=SimpleNamespace(
            set_properties=lambda **kw: setattr(particles, "strength", kw["vortex_strength"].copy())
        ),
    )

    def modify(*args, **kwargs):
        particles.strength *= 2
        if operator_failure:
            raise RuntimeError("interrupted operator")
        return {"pedrizzetti_misalignment_deg": 0.0}

    manager.operators = SimpleNamespace(apply_pedrizzetti_relaxation=modify)
    with pytest.raises(RuntimeError if operator_failure else StabilizationError):
        manager.apply_relaxation()
    np.testing.assert_array_equal(particles.strength, original)
    assert manager.events == 7
    assert manager.last_mechanism == "previous accepted event"
    np.testing.assert_array_equal(manager.pedrizzetti_moment_transfer, np.ones((3, 3)))


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
        state=SimpleNamespace(step=10, domain_bounds_enforced=False),
    )
    manager.regularization_events = 0
    manager.regularization_energy_transfer = 0.0
    manager.regularization_enstrophy_transfer = 0.0
    manager.measure = lambda: object()
    manager.accept = lambda *args, **kwargs: None

    calls = []

    def regularize(context, active_config):
        calls.append((context, active_config))
        return SimpleNamespace(
            detail="accepted", total_kinetic_energy_transfer=-0.2, total_enstrophy_transfer=-0.5
        )

    monkeypatch.setattr(
        "source.solvers.vpm.stabilization.regularization.regularize",
        regularize,
    )

    manager.apply_regularization()
    manager.apply_regularization()
    manager.apply_regularization()

    assert len(calls) == 2
    assert manager.regularization_events == 2
    assert manager.regularization_energy_transfer == pytest.approx(-0.4)
    assert manager.regularization_enstrophy_transfer == pytest.approx(-1.0)


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
    state = SimpleNamespace(step=550)
    metrics = SimpleNamespace(kinetic_energy_rate=4.0, viscous_kinetic_energy_rate=-2.0)
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
        state=state,
        metrics=metrics,
        particles=object(),
    )
    manager.operators = SimpleNamespace(
        apply_stretching_viscosity=lambda particles, coefficient: applied.append(coefficient)
    )
    manager.residual_viscosity_coefficient = 1.6
    manager._last_residual_feedback_step = -1

    manager.update_residual_viscosity()
    manager.update_residual_viscosity()
    state.step = 555
    metrics.kinetic_energy_rate = -1.0
    manager.update_residual_viscosity()

    assert applied == pytest.approx([2.4, 2.4, 1.92])


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
    ],
)
def test_stabilization_schedule_rejects_invalid_limits(keyword, value):
    arguments = {keyword: value}
    if keyword == "stretching_viscosity_max_coefficient":
        arguments["stretching_viscosity_coefficient"] = 1.0
    with pytest.raises(ValueError):
        StabilizationConfig(**arguments)


def test_accepted_relaxation_records_exact_momentum_transfers():
    position = np.array([[1.0, 2.0, 3.0], [-2.0, 1.0, 4.0]])
    initial = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    radius = np.array([0.2, 0.3])
    particles = SimpleNamespace(n_particles_total=2, strength=initial.copy())
    particles.vortex_strength_cpu = lambda **kw: particles.strength
    particles.position_cpu = lambda **kw: position
    particles.core_radius_cpu = lambda **kw: radius
    particles.particle_volume_cpu = lambda **kw: np.ones(2)
    manager = object.__new__(StabilizationManager)
    manager.config = StabilizationConfig.pedrizzetti_relaxation()
    manager.events = 0
    manager.max_vorticity_growth = 0
    manager.pedrizzetti_moment_transfer = np.zeros((3, 3))
    manager.ctx = SimpleNamespace(
        particles=particles,
        flow_model="LES",
        state=SimpleNamespace(step=0),
        physics=SimpleNamespace(_angular_core_coefficient=1 / 3),
        mutations=SimpleNamespace(
            set_properties=lambda **kw: setattr(particles, "strength", kw["vortex_strength"])
        ),
    )

    def rotate(*args, **kw):
        particles.strength = np.column_stack(
            (-particles.strength[:, 1], particles.strength[:, 0], particles.strength[:, 2])
        )
        return {"pedrizzetti_misalignment_deg": 90.0}

    manager.operators = SimpleNamespace(apply_pedrizzetti_relaxation=rotate)
    for _ in range(2):
        manager.apply_relaxation()
    change = particles.strength - initial
    expected = np.array(
        [
            change.sum(axis=0),
            0.5 * np.cross(position, change).sum(axis=0),
            (np.cross(position, np.cross(position, change)) - radius[:, None] ** 2 * change).sum(
                axis=0
            )
            / 3,
        ]
    )
    np.testing.assert_allclose(manager.pedrizzetti_moment_transfer, expected, atol=1e-14)
    assert manager.events == 2
