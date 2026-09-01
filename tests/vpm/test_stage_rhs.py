"""Tests for the central coupled stage right-hand side."""

import numpy as np

from source.solvers.vpm.physics.induction.base import StageRates, StageState
from source.solvers.vpm.physics.stage_rhs import StageRHS


class _Induction:
    def __init__(self):
        self.received = None

    def evaluate_stage(self, **kwargs):
        self.received = kwargs
        kwargs["velocity_out"][:] = 1.0
        kwargs["vortex_strength_rate_out"][:] = 2.0


class _External:
    def __init__(self):
        self.received = None

    def add_stage_rates(self, stage_state, stage_time, stage_rates):
        self.received = (stage_state, stage_time)
        stage_rates.velocity[:] += 3.0
        stage_rates.vortex_strength_rate[:] += 4.0


def test_stage_rhs_passes_stage_time_and_state_to_induction_and_external_provider():
    position = np.zeros((2, 3))
    strength = np.ones((2, 3))
    radius = np.full(2, 0.2)
    velocity = np.empty((2, 3))
    rate = np.empty((2, 3))
    induction = _Induction()
    external = _External()
    rhs = StageRHS(induction, (external,))
    state = StageState(position, strength, radius, 2, time=4.5)
    rates = StageRates(velocity, rate)

    rhs.evaluate(state, 4.5, rates)

    assert induction.received["position"] is position
    assert induction.received["vortex_strength"] is strength
    assert induction.received["core_radius"] is radius
    assert induction.received["stage_time"] == 4.5
    assert external.received[0] is state
    assert external.received[1] == 4.5
    np.testing.assert_allclose(velocity, 4.0)
    np.testing.assert_allclose(rate, 6.0)
