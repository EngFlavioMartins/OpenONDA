"""Accepted-step health-limit regression tests."""

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
    with pytest.raises(HealthError, match="Lagrangian CFL number 0.5"):
        _check(
            HealthLimits(lagrangian_cfl=LagrangianCFLLimit(maximum=0.4)),
            velocity_gradient=gradient,
        )

    with pytest.raises(HealthError, match="core_radius"):
        _check(core_radius=np.array([np.nan]))


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
