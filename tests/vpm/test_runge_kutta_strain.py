"""Qualify Heun stage times and 3D advection/stretching convergence."""

from contextlib import contextmanager

import numpy as np
from scipy.linalg import expm
import taichi as ti

from source.solvers.vpm.numerics.rk_tableaux import RK2
from source.solvers.vpm.numerics.runge_kutta import RungeKutta


class StrainRHS:
    def __init__(self, matrix, acceleration, origin):
        self.matrix, self.acceleration, self.origin = matrix, acceleration, origin
        self.times, self.intervals = [], []

    @contextmanager
    def integration_step(self, tableau, dt):
        self.intervals.append((tableau.name, dt))
        yield

    def evaluate(self, state, time, rates):
        self.times.append(time)
        position, strength = state.position.to_numpy(), state.vortex_strength.to_numpy()
        velocity = position @ self.matrix.T + (time - self.origin) * self.acceleration
        rates.velocity.from_numpy(velocity.astype(position.dtype))
        rates.vortex_strength_rate.from_numpy((strength @ self.matrix.T).astype(strength.dtype))


def integrate(steps, dtype=ti.f64):
    if ti.lang.impl.get_runtime().prog is None:
        ti.init(arch=ti.cpu, cpu_max_num_threads=1)
    position = ti.Vector.field(3, dtype=dtype, shape=2)
    strength = ti.Vector.field(3, dtype=dtype, shape=2)
    radius = ti.field(dtype=dtype, shape=2)
    initial_position = np.array([[0.3, -0.7, 1.1], [-0.8, 0.6, -0.2]])
    initial_strength = np.array([[0.4, -0.2, 0.1], [0.2, 0.3, -0.6]])
    for i in range(2):
        position[i], strength[i], radius[i] = initial_position[i], initial_strength[i], 0.2
    basis, _ = np.linalg.qr(np.array([[1.0, 2.0, 3.0], [3.0, -2.0, 1.0], [2.0, 1.0, -0.5]]))
    matrix = basis @ np.diag([0.3, -0.7, 0.4]) @ basis.T
    acceleration, start, duration = np.array([0.2, -0.1, 0.3]), 1.3, 0.8
    rhs = StrainRHS(matrix, acceleration, start)
    rk = RungeKutta(RK2(), max_n_particles=2, dtype=dtype)
    arguments = {
        "position": position,
        "vortex_strength": strength,
        "core_radius": radius,
        "count": 2,
        "time": start,
        "time_step_size": duration,
        "right_hand_side": rhs,
    }
    radius_before = radius.to_numpy().copy()
    for step in range(steps):
        rk.advance(
            **{
                **arguments,
                "time": start + step * duration / steps,
                "time_step_size": duration / steps,
            }
        )
    np.testing.assert_array_equal(radius.to_numpy(), radius_before)
    assert arguments["time"] == start and arguments["time_step_size"] == duration
    exponential = expm(duration * matrix)
    forcing = np.linalg.solve(
        matrix @ matrix, (exponential - np.eye(3) - duration * matrix) @ acceleration
    )
    exact = (initial_position @ exponential.T + forcing, initial_strength @ exponential.T)
    return (position.to_numpy(), strength.to_numpy()), exact, rhs


def test_time_dependent_3d_strain_converges_for_position_and_vector_strength():
    errors = []
    for count in (4, 8):
        fields, exact, rhs = integrate(count)
        errors.append(
            [np.linalg.norm(value - target) for value, target in zip(fields, exact, strict=True)]
        )
        np.testing.assert_allclose(
            rhs.times,
            [1.3 + (i + c) * (0.8 / count) for i in range(count) for c in (0.0, 1.0)],
            rtol=0,
            atol=5e-16,
        )
        assert rhs.intervals == [("RK2", 0.8 / count)] * count
    assert np.all(np.array(errors[0]) / errors[1] > 3.7)
