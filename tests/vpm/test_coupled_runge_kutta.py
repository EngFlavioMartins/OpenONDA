"""Tests for the generic coupled Runge--Kutta engine."""

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.numerics.rk_tableaux import RK2, RK4, SSPRK3
from source.solvers.vpm.numerics.runge_kutta import RungeKutta
from source.solvers.vpm.physics.induction.base import StageRates, StageState


def _ensure_taichi_cpu() -> None:
    if ti.lang.impl.get_runtime().prog is None:
        ti.init(arch=ti.cpu)


class _SpyRHS:
    def __init__(self):
        self.calls = []

    def evaluate(self, stage_state: StageState, stage_time: float, stage_rates: StageRates):
        self.calls.append(
            (
                stage_time,
                stage_state.position.to_numpy()[: stage_state.count].copy(),
                stage_state.vortex_strength.to_numpy()[: stage_state.count].copy(),
            )
        )
        for i in range(stage_state.count):
            stage_rates.velocity[i] = [1.0, 2.0, 3.0]
            stage_rates.vortex_strength_rate[i] = [2.0, 3.0, 4.0]


def test_tableaux_retain_the_three_coupled_schemes():
    assert (RK2().order, RK2().stages) == (2, 2)
    assert (SSPRK3().order, SSPRK3().stages) == (3, 3)
    assert (RK4().order, RK4().stages) == (4, 4)
    np.testing.assert_allclose(SSPRK3().b, (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0))


def test_every_rk_stage_uses_one_common_position_strength_state():
    _ensure_taichi_cpu()
    position = ti.Vector.field(3, dtype=ti.f32, shape=(1,))
    vortex_strength = ti.Vector.field(3, dtype=ti.f32, shape=(1,))
    core_radius = ti.field(dtype=ti.f32, shape=(1,))
    position[0] = [0.0, 0.0, 0.0]
    vortex_strength[0] = [1.0, 2.0, 3.0]
    core_radius[0] = 0.2
    rhs = _SpyRHS()
    integrator = RungeKutta(RK4(), max_n_particles=1)

    integrator.advance(
        position=position,
        vortex_strength=vortex_strength,
        core_radius=core_radius,
        count=1,
        time=2.0,
        time_step_size=0.1,
        right_hand_side=rhs,
    )

    assert len(rhs.calls) == 4
    np.testing.assert_allclose([call[0] for call in rhs.calls], [2.0, 2.05, 2.05, 2.1])
    np.testing.assert_allclose(
        [call[1][0] for call in rhs.calls],
        [[0.0, 0.0, 0.0], [0.05, 0.1, 0.15], [0.05, 0.1, 0.15], [0.1, 0.2, 0.3]],
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        [call[2][0] for call in rhs.calls],
        [
            [1.0, 2.0, 3.0],
            [1.1, 2.15, 3.2],
            [1.1, 2.15, 3.2],
            [1.2, 2.3, 3.4],
        ],
        rtol=2e-6,
    )
    np.testing.assert_allclose(position.to_numpy(), [[0.1, 0.2, 0.3]], rtol=2e-6)
    np.testing.assert_allclose(vortex_strength.to_numpy(), [[1.2, 2.3, 3.4]], rtol=2e-6)
    for _, stage_position, stage_strength in rhs.calls:
        assert stage_position.shape == stage_strength.shape == (1, 3)


class _LinearCoupledRHS:
    def evaluate(self, stage_state: StageState, stage_time: float, stage_rates: StageRates):
        del stage_time
        position = stage_state.position.to_numpy()[: stage_state.count]
        strength = stage_state.vortex_strength.to_numpy()[: stage_state.count]
        velocity = np.zeros((stage_state.count, 3), dtype=np.float32)
        rate = np.zeros((stage_state.count, 3), dtype=np.float32)
        velocity[:, 0] = position[:, 0]
        rate[:, 0] = 2.0 * strength[:, 0]
        stage_rates.velocity.from_numpy(velocity)
        stage_rates.vortex_strength_rate.from_numpy(rate)


def _integrate_linear_coupled(tableau, time_step_size: float, steps: int):
    position = ti.Vector.field(3, dtype=ti.f32, shape=(1,))
    vortex_strength = ti.Vector.field(3, dtype=ti.f32, shape=(1,))
    core_radius = ti.field(dtype=ti.f32, shape=(1,))
    position[0] = [1.0, 0.0, 0.0]
    vortex_strength[0] = [1.0, 0.0, 0.0]
    core_radius[0] = 0.2
    integrator = RungeKutta(tableau, max_n_particles=1)
    rhs = _LinearCoupledRHS()
    for _ in range(steps):
        integrator.advance(
            position=position,
            vortex_strength=vortex_strength,
            core_radius=core_radius,
            count=1,
            time=0.0,
            time_step_size=time_step_size,
            right_hand_side=rhs,
        )
    return float(position[0][0]), float(vortex_strength[0][0])


@pytest.mark.parametrize("tableau", [RK2(), SSPRK3(), RK4()], ids=["rk2", "ssprk3", "rk4"])
def test_coupled_runge_kutta_observes_declared_order_for_both_state_components(tableau):
    final_time = 0.8
    exact_position = np.exp(final_time)
    exact_strength = np.exp(2.0 * final_time)
    coarse = _integrate_linear_coupled(tableau, 0.2, 4)
    fine = _integrate_linear_coupled(tableau, 0.1, 8)
    errors = np.array(
        [
            abs(coarse[0] - exact_position),
            abs(fine[0] - exact_position),
            abs(coarse[1] - exact_strength),
            abs(fine[1] - exact_strength),
        ]
    )
    observed_position = np.log(errors[0] / errors[1]) / np.log(2.0)
    observed_strength = np.log(errors[2] / errors[3]) / np.log(2.0)
    assert observed_position > tableau.order - 0.25
    assert observed_strength > tableau.order - 0.25
