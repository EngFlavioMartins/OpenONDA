"""Accepted-step health-limit regression tests."""

from types import SimpleNamespace

import numpy as np
import pytest

from source.solvers.vpm.config.health import (
    DivergenceLimit,
    FiniteStateCheck,
    GrowthLimit,
    HealthError,
    HealthLimits,
    LagrangianCFLLimit,
    MisalignmentLimit,
    ParticleStrengthLimit,
    accepted_step_health,
)


def test_misalignment_uses_current_curl_independently_of_backup_vorticity():
    from source.solvers.vpm.core.solver import VPMSolver

    gradient = np.zeros((8, 3, 3))
    gradient[:, 2, 1] = 2.0  # curl(u) points along x
    state = SimpleNamespace(
        health_limits=HealthLimits(),
        particles=SimpleNamespace(
            n_particles_total=8,
            velocity_gradient_cpu=lambda **_: gradient,
        ),
        particle_position=np.column_stack((np.arange(8), np.zeros((8, 2)))),
        particle_vortex_strength=np.tile([1.0, 0.0, 0.0], (8, 1)),
        particle_core_radius=np.ones(8),
        particle_vorticity=np.tile([0.0, 1.0, 0.0], (8, 1)),
    )
    VPMSolver._update_discretization_health(state)
    assert state._discretization_health["vortex_strength_misalignment_degrees"] == 0.0
    # A stored backup field may change arbitrarily without changing this check.
    state.particle_vorticity[:] = np.nan
    VPMSolver._update_discretization_health(state)
    assert state._discretization_health["vortex_strength_misalignment_degrees"] == 0.0
    gradient[:] = 0.0
    gradient[:, 0, 2] = 2.0  # current curl turns toward y
    VPMSolver._update_discretization_health(state)
    assert state._discretization_health["vortex_strength_misalignment_degrees"] == 90.0


@pytest.mark.parametrize(
    ("metric", "limit", "sampled", "full", "should_stop", "confirm"),
    [
        ("vorticity_divergence_error", 0.12, 0.120173, 0.103383, False, True),
        ("vorticity_divergence_error", 0.12, 0.121, 0.125, True, True),
        ("vorticity_divergence_error", 0.12, np.nan, 0.125, True, True),
        ("vorticity_divergence_error", 0.12, 0.121, np.nan, True, True),
        ("vorticity_divergence_error", 0.12, 0.12, 0.103, False, False),
        ("vorticity_divergence_error", None, 0.121, 0.103, False, False),
        ("vortex_strength_misalignment_degrees", 25.0, 26.0, 23.0, False, True),
        ("vortex_strength_misalignment_degrees", 25.0, 26.0, 27.0, True, True),
    ],
)
def test_sampled_resolution_crossing_requires_full_field_confirmation(
    monkeypatch, metric, limit, sampled, full, should_stop, confirm
):
    from source.solvers.vpm.core import solver as solver_module

    divergence = metric == "vorticity_divergence_error"
    limits = HealthLimits(
        divergence=DivergenceLimit(maximum=limit if divergence else None),
        misalignment=MisalignmentLimit(maximum_degrees=None if divergence else limit),
    )
    position = np.column_stack((np.arange(8), np.zeros((8, 2))))
    strength = np.tile([1.0, 0.0, 0.0], (8, 1))
    core = np.ones(8)
    gradient = np.zeros((8, 3, 3))
    state = SimpleNamespace(
        particles=SimpleNamespace(
            n_particles_total=8,
            velocity_gradient_cpu=lambda **_: gradient,
        ),
        health_limits=limits,
        particle_position=position,
        particle_vortex_strength=strength,
        particle_core_radius=core,
    )
    calls = []

    def measured(actual_position, actual_strength, actual_core, *, vorticity, sample_all=False):
        assert actual_position is position
        assert actual_strength is strength
        assert actual_core is core
        calls.append(sample_all)
        return {metric: full if sample_all else sampled}

    monkeypatch.setattr(solver_module, "discretization_health", measured)
    solver_module.VPMSolver._update_discretization_health(state)
    assert calls == ([False, True] if confirm else [False])
    values = state._discretization_health
    if should_stop:
        with pytest.raises(HealthError):
            _check(limits, resolution=values)
    else:
        _check(limits, resolution=values)


def _state(**overrides):
    state = {
        "position": np.array([[0.0, 0.0, 0.0]]),
        "velocity": np.array([[0.0, 0.0, 0.0]]),
        "velocity_gradient": np.zeros((1, 3, 3)),
        "vortex_strength": np.array([[1.0, 0.0, 0.0]]),
        "core_radius": np.array([0.2]),
        "particle_volume": np.array([0.5]),
        "resolution": {
            "vorticity_divergence_error": 0.01,
            "vortex_strength_misalignment_degrees": 2.0,
        },
    }
    state.update(overrides)
    return state


def _check(limits=HealthLimits(), previous=None, **overrides):
    return accepted_step_health(
        limits=limits,
        step=3,
        time_step_size=0.1,
        previous=previous,
        **_state(**overrides),
    )


def test_health_limits_enforce_finite_state_and_cfl_after_field_refresh():
    gradient = np.zeros((1, 3, 3))
    gradient[0, 0, 0] = 5.0
    with pytest.raises(HealthError, match="Lagrangian CFL number 0.5") as cfl_error:
        _check(
            HealthLimits(lagrangian_cfl=LagrangianCFLLimit(maximum=0.4)),
            velocity_gradient=gradient,
        )
    assert cfl_error.value.restartable

    with pytest.raises(HealthError, match="core_radius") as finite_state_error:
        _check(core_radius=np.array([np.nan]))
    assert not finite_state_error.value.restartable


def test_health_limits_accept_an_empty_particle_state():
    snapshot = _check(
        position=np.empty((0, 3)),
        velocity=np.empty((0, 3)),
        velocity_gradient=np.empty((0, 3, 3)),
        vortex_strength=np.empty((0, 3)),
        core_radius=np.empty(0),
        particle_volume=np.empty(0),
    )

    assert snapshot.strain_increment_infinity == 0.0
    assert snapshot.strain_increment_spectral == 0.0
    assert snapshot.maximum_particle_strength == 0.0
    assert snapshot.maximum_vorticity == 0.0
    assert snapshot.strain_increment_infinity_particle == -1
    assert snapshot.strain_increment_spectral_particle == -1


def test_cfl_failure_identifies_the_particle_and_does_not_round_away_the_crossing():
    gradient = np.zeros((2, 3, 3))
    gradient[0] = np.diag([2.0, -2.0, 0.0])
    gradient[1] = np.diag([10.001, -10.001, 0.0])
    with pytest.raises(HealthError) as error:
        _check(
            position=np.array([[0.0, 0.0, 0.0], [2.95, -0.17, -1.37]]),
            velocity=np.zeros((2, 3)),
            velocity_gradient=gradient,
            vortex_strength=np.ones((2, 3)),
            core_radius=np.full(2, 0.066),
            particle_volume=np.full(2, 0.06**3),
        )
    message = str(error.value)
    assert "Lagrangian CFL number 1.0001 " in message
    assert "maximum=1; particle=1, position=(2.95, -0.17, -1.37) m" in message
    assert "strain_rate=10.001 1/s, time_step_size=0.1 s" in message
    assert "time-step ceiling is 0.099990001 s" in message
    assert "vorticity divergence/alignment" in message
    assert error.value.restartable


def test_health_limits_apply_strength_resolution_and_growth_limits():
    limits = HealthLimits(
        finite_state=FiniteStateCheck(),
        maximum_particle_strength=ParticleStrengthLimit(maximum=1.5),
        divergence=DivergenceLimit(maximum=0.02),
        misalignment=MisalignmentLimit(maximum_degrees=3.0),
        growth=GrowthLimit(
            maximum_particle_strength_growth=0.1,
            maximum_vorticity_growth=0.1,
        ),
    )
    before = _check(limits)

    with pytest.raises(HealthError, match="maximum particle strength"):
        _check(limits, vortex_strength=np.array([[2.0, 0.0, 0.0]]))
    with pytest.raises(HealthError, match="vorticity divergence error"):
        _check(limits, resolution={"vorticity_divergence_error": 0.03})
    with pytest.raises(HealthError, match="misalignment"):
        _check(
            limits,
            resolution={
                "vorticity_divergence_error": 0.01,
                "vortex_strength_misalignment_degrees": 4.0,
            },
        )
    with pytest.raises(HealthError, match="particle-strength growth"):
        _check(limits, previous=before, vortex_strength=np.array([[1.2, 0.0, 0.0]]))


@pytest.mark.parametrize(
    ("factory", "value"),
    [
        (lambda value: LagrangianCFLLimit(maximum=value), 0.0),
        (lambda value: ParticleStrengthLimit(maximum=value), -1.0),
        (lambda value: DivergenceLimit(maximum=value), -1.0),
        (lambda value: MisalignmentLimit(maximum_degrees=value), 181.0),
        (lambda value: GrowthLimit(maximum_vorticity_growth=value), -1.0),
    ],
)
def test_health_limit_values_are_validated(factory, value):
    with pytest.raises(ValueError):
        factory(value)
