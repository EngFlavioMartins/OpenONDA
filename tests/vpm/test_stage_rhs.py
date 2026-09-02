"""Tests for the central coupled stage right-hand side."""

import numpy as np

from source.solvers.vpm.physics.induction.base import StageRates, StageState
from source.solvers.vpm.physics.stage_rhs import (
    CallableStageContribution,
    StageRHS,
    VLMStageContribution,
)


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


def test_callable_stage_contribution_receives_time_and_all_coupled_stage_arrays():
    position = np.zeros((2, 3))
    strength = np.ones((2, 3))
    radius = np.full(2, 0.2)
    velocity = np.zeros((2, 3))
    rate = np.zeros((2, 3))
    gradient = np.zeros((2, 3, 3))
    received = {}

    def evaluate(stage_time, stage_position, stage_strength, velocity_out, rate_out, gradient_out):
        received.update(
            time=stage_time,
            position=stage_position,
            strength=stage_strength,
            gradient=gradient_out,
        )
        velocity_out[:, 0] = stage_time + stage_position[:, 0]
        rate_out[:, 1] = stage_strength[:, 1]
        gradient_out[:, 2, 0] = 2.0

    state = StageState(position, strength, radius, 2, time=3.25)
    rates = StageRates(velocity, rate, gradient)
    CallableStageContribution(evaluate).add_stage_rates(state, 3.25, rates)

    assert received["time"] == 3.25
    np.testing.assert_allclose(received["position"], position)
    np.testing.assert_allclose(received["strength"], strength)
    np.testing.assert_allclose(velocity[:, 0], 3.25)
    np.testing.assert_allclose(rate[:, 1], 1.0)
    np.testing.assert_allclose(gradient[:, 2, 0], 2.0)


def test_callable_stage_contribution_always_provides_a_writable_gradient_array():
    received = {}

    def evaluate(_time, _position, _strength, _velocity, _rate, gradient_out):
        received["gradient"] = gradient_out
        gradient_out[:, 0, 0] = 4.0

    state = StageState(np.zeros((1, 3)), np.ones((1, 3)), np.full(1, 0.2), 1)
    rates = StageRates(np.zeros((1, 3)), np.zeros((1, 3)))
    CallableStageContribution(evaluate).add_stage_rates(state, 0.0, rates)

    assert received["gradient"].shape == (1, 3, 3)
    np.testing.assert_allclose(rates.vortex_strength_rate, 0.0)


def test_stage_provider_prepend_keeps_projection_last():
    first = object()
    second = object()
    rhs = StageRHS(_Induction(), (first,))
    rhs.add_provider(second, prepend=True)

    assert rhs.providers == (second, first)


def test_vlm_stage_contribution_forwards_temporary_targets_and_stage_time():
    class _VLM:
        def __init__(self):
            self.received = None

        def add_stage_velocity(self, position, velocity, count, stage_time):
            self.received = (position, count, stage_time)
            velocity[:count, 2] += stage_time

    position = np.zeros((2, 3))
    state = StageState(position, np.ones((2, 3)), np.full(2, 0.2), 2, time=1.5)
    velocity = np.zeros((2, 3))
    rates = StageRates(velocity, np.zeros((2, 3)))
    vlm = _VLM()

    VLMStageContribution(vlm).add_stage_rates(state, 1.75, rates)

    assert vlm.received[0] is position
    assert vlm.received[1:] == (2, 1.75)
    np.testing.assert_allclose(velocity[:, 2], 1.75)


def test_callable_stage_contribution_can_contract_external_gradient_into_rate():
    def evaluate(_time, _position, _strength, _velocity, _rate, gradient_out):
        gradient_out[:, 1, 0] = 2.0

    state = StageState(np.zeros((1, 3)), np.array([[0.0, 3.0, 0.0]]), np.array([0.2]), 1)
    rates = StageRates(np.zeros((1, 3)), np.zeros((1, 3)))

    CallableStageContribution(evaluate, include_external_stretching=True).add_stage_rates(
        state, 0.0, rates
    )

    np.testing.assert_allclose(rates.vortex_strength_rate, [[6.0, 0.0, 0.0]])
