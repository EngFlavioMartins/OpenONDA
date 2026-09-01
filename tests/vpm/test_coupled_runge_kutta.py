"""Tests for the generic coupled Runge--Kutta engine."""

import numpy as np
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
