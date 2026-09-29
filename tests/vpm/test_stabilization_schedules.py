from types import SimpleNamespace

import numpy as np
import pytest

from source.solvers.vpm.config.divergence_relaxation import DivergenceRelaxationConfig
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
        max_absolute_vortex_strength=0.5,
    )
    config = StabilizationConfig(
        selective_eddy_viscosity_coefficient=1.6,
        selective_eddy_viscosity_start_step=550,
        pedrizzetti_relaxation_factor=0.005,
        pedrizzetti_relaxation_end_step=650,
        filament_refinement=refinement,
        regularization_interval_steps=25,
        regularization_start_step=475,
        regularization_grid_spacing=0.055,
        regularization_max_particles=30_000,
    )

    assert config.filament_refinement.interval_steps == 25
    assert config.filament_refinement.max_absolute_vortex_strength == 0.5
    assert config.selective_eddy_viscosity_coefficient == 1.6
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


def test_empty_refinement_records_birth_lineage_before_nonempty_moments():
    manager = object.__new__(StabilizationManager)
    manager.config = StabilizationConfig(
        filament_refinement=FilamentRefinementConfig.adaptive(interval_steps=1),
        divergence_relaxation=DivergenceRelaxationConfig.constrained(
            interval_steps=1, grid_spacing=0.1
        ),
    )
    particles = SimpleNamespace(n_particles_total=0, strength=np.empty((0, 3)))
    particles.vortex_strength_cpu = lambda: particles.strength
    particles.particle_volume_cpu = lambda: np.ones(particles.n_particles_total)
    particles.position_cpu = lambda: np.zeros((particles.n_particles_total, 3))
    particles.core_radius_cpu = lambda: np.ones(particles.n_particles_total)
    manager.ctx = SimpleNamespace(particles=particles, state=SimpleNamespace(step=0))
    manager.reference_vortex_strength = None
    manager.reference_lengths = None
    manager.reference_moments = None

    manager.capture_reference_state()
    manager.apply_filament_refinement()
    manager.apply_divergence_relaxation()
    assert manager.reference_vortex_strength.shape == (0,)
    assert manager.reference_moments is None

    particles.n_particles_total = 1
    particles.strength = np.array([[0.0, 0.0, 2.0]])
    manager.on_add(np.array([2.0]), np.ones(1), start=0)
    particles.strength *= 2
    manager.capture_reference_state()
    np.testing.assert_array_equal(manager.reference_vortex_strength, [2.0])
    np.testing.assert_array_equal(manager.reference_moments[0], [0.0, 0.0, 4.0])


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


def test_regularization_schedule_continues_after_multiple_events(monkeypatch):
    config = StabilizationConfig(
        regularization_interval_steps=5,
        regularization_start_step=10,
        regularization_grid_spacing=0.1,
        regularization_max_particles=100,
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

    assert len(calls) == 3
    assert manager.regularization_events == 3
    assert manager.regularization_energy_transfer == pytest.approx(-0.6)
    assert manager.regularization_enstrophy_transfer == pytest.approx(-1.5)


def test_regularization_can_be_triggered_only_by_core_radius():
    health = {
        "vorticity_divergence_error": 1.0,
        "vortex_strength_misalignment_degrees": 90.0,
    }
    arguments = {
        "divergence_trigger": None,
        "misalignment_trigger": None,
        "core_radius_trigger": 0.2,
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
    assert result.deferred_particles == 1
    assert len(result.position) == 6
    assert 0 in result.source_index


def test_winckelmans_fixed_core_bisection_and_reset_at_exact_threshold():
    from source.solvers.vpm.stabilization.filament_refinement import gaussian_particle_moments

    position = np.array([[0.2, -0.3, 0.4]])
    strength = np.array([[0.0, 0.0, 2.0]])
    core, volume = np.array([0.1]), np.array([0.008])
    result = split_stretched_filaments(
        position,
        strength,
        core,
        volume,
        reference_vortex_strength=np.ones(1),
        reference_length=np.array([0.2]),
        max_stretch_factor=2,
    )
    np.testing.assert_allclose(result.position, position + [[0, 0, 0.1], [0, 0, -0.1]])
    np.testing.assert_array_equal(result.vortex_strength, [[0, 0, 1], [0, 0, 1]])
    np.testing.assert_array_equal(result.core_radius, [0.1, 0.1])
    np.testing.assert_array_equal(result.particle_volume, [0.004, 0.004])
    np.testing.assert_array_equal(result.source_index, [0, 0])
    np.testing.assert_allclose(result.reference_length, [0.2, 0.2])
    repeated = split_stretched_filaments(
        result.position,
        result.vortex_strength,
        result.core_radius,
        result.particle_volume,
        reference_vortex_strength=result.reference_vortex_strength,
        reference_length=result.reference_length,
        max_stretch_factor=2,
    )
    assert repeated.refined_particles == 0
    before = gaussian_particle_moments(position, strength, core)
    after = gaussian_particle_moments(result.position, result.vortex_strength, result.core_radius)
    for index in range(4):
        np.testing.assert_allclose(after[index], before[index], atol=1e-14)
    # The field is changed despite exact moments; do not claim energy conservation.
    assert result.isolated_kinetic_energy_change < 0


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


@pytest.mark.parametrize("threshold", [0.0, -1.0, np.inf])
def test_filament_refinement_rejects_invalid_absolute_threshold(threshold):
    with pytest.raises(ValueError):
        FilamentRefinementConfig.adaptive(interval_steps=10, max_absolute_vortex_strength=threshold)


def test_selective_eddy_viscosity_coefficient_is_fixed_despite_energy_changes():
    state = SimpleNamespace(step=550)
    metrics = SimpleNamespace(kinetic_energy_rate=4.0, viscous_kinetic_energy_rate=-2.0)
    config = StabilizationConfig(
        selective_eddy_viscosity_coefficient=1.6,
        selective_eddy_viscosity_start_step=550,
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
        apply_selective_eddy_viscosity=lambda particles, coefficient: applied.append(coefficient)
    )
    manager.selective_eddy_viscosity_coefficient = 8.0  # Retired checkpoint diagnostic.

    manager.update_selective_eddy_viscosity()
    manager.update_selective_eddy_viscosity()
    state.step = 555
    metrics.kinetic_energy_rate = -1.0
    manager.update_selective_eddy_viscosity()

    assert applied == pytest.approx([1.6, 1.6, 1.6])


def test_stabilization_schedule_rejects_invalid_relaxation_window():
    with pytest.raises(ValueError):
        StabilizationConfig(pedrizzetti_relaxation_end_step=-1)


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


@pytest.mark.parametrize(
    "name",
    ["regularization_capacity_fraction", "regularization_capacity_max_particles"],
)
def test_capacity_specific_remeshing_controls_are_not_public(name):
    with pytest.raises(TypeError, match=name):
        StabilizationConfig(**{name: 1})


def test_filament_refinement_keeps_one_cadence_and_both_strength_criteria(monkeypatch):
    config = StabilizationConfig(
        filament_refinement=FilamentRefinementConfig.adaptive(
            interval_steps=5,
            max_vortex_strength_factor=3.0,
            max_absolute_vortex_strength=0.5,
        )
    )
    state = SimpleNamespace(step=749)
    particles = SimpleNamespace(
        n_particles_total=1,
        _max_particles=64,
        position_cpu=lambda: np.zeros((1, 3)),
        vortex_strength_cpu=lambda: np.ones((1, 3)),
        core_radius_cpu=lambda: np.ones(1),
        particle_volume_cpu=lambda: np.ones(1),
    )
    manager = object.__new__(StabilizationManager)
    manager.config = config
    manager.ctx = SimpleNamespace(state=state, particles=particles)
    manager.reference_vortex_strength = np.ones(1)
    manager.reference_lengths = np.ones(1)
    manager.measure = lambda: None
    calls = []

    def split(*args, **kwargs):
        calls.append(
            (state.step, kwargs["max_stretch_factor"], kwargs["max_absolute_vortex_strength"])
        )
        return SimpleNamespace(refined_particles=0, deferred_particles=0)

    monkeypatch.setattr(
        "source.solvers.vpm.stabilization.filament_refinement.split_stretched_filaments", split
    )
    for step in [749, 750, 755, 800]:
        state.step = step
        manager.apply_filament_refinement()

    assert calls == [(750, 3.0, 0.5), (755, 3.0, 0.5), (800, 3.0, 0.5)]
