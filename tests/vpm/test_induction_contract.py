"""Contract tests for interchangeable VPM induction methods."""

import numpy as np

from source.solvers.vpm.physics.induction.base import InductionMethod, StageRates, StageState


class _SpyInduction:
    def __init__(self):
        self.calls = []

    def evaluate_stage(self, **kwargs):
        self.calls.append(kwargs)
        kwargs["velocity_out"][:] = 7.0
        kwargs["vortex_strength_rate_out"][:] = -3.0


def test_stage_state_and_rates_are_explicit_value_objects():
    state = StageState("stage-position", "stage-strength", "stage-radius", 3, 1.25)
    rates = StageRates("velocity", "strength-rate", "gradient")

    assert state.position == "stage-position"
    assert state.vortex_strength == "stage-strength"
    assert state.core_radius == "stage-radius"
    assert state.count == 3
    assert state.time == 1.25
    assert rates.velocity_gradient == "gradient"


def test_spy_receives_the_exact_supplied_stage_fields_and_writes_only_outputs():
    position = np.zeros((3, 3))
    strength = np.ones((3, 3))
    radius = np.full(3, 0.2)
    velocity = np.empty((3, 3))
    rate = np.empty((3, 3))
    spy: InductionMethod = _SpyInduction()

    spy.evaluate_stage(
        position=position,
        vortex_strength=strength,
        core_radius=radius,
        count=3,
        velocity_out=velocity,
        vortex_strength_rate_out=rate,
        stage_time=1.25,
    )

    call = spy.calls[0]
    assert call["position"] is position
    assert call["vortex_strength"] is strength
    assert call["core_radius"] is radius
    assert call["count"] == 3
    assert call["stage_time"] == 1.25
    np.testing.assert_allclose(velocity, 7.0)
    np.testing.assert_allclose(rate, -3.0)
