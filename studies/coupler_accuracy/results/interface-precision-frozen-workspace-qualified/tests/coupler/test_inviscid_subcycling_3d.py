"""Qualify physical stage times and 3D advection/stretching convergence."""

from contextlib import contextmanager

import numpy as np
from scipy.linalg import expm
import taichi as ti

from source.solvers.vpm.numerics.rk_tableaux import RK2
from source.solvers.vpm.numerics.runge_kutta import RungeKutta
from studies.coupler_accuracy.experimental_inviscid_subcycling_3d import advance_subcycles


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


def integrate(substeps, dtype=ti.f64, *, native=False):
    if ti.lang.impl.get_runtime().prog is None:
        ti.init(arch=ti.cpu, cpu_max_num_threads=1)
    position = ti.Vector.field(3, dtype=dtype, shape=2)
    strength = ti.Vector.field(3, dtype=dtype, shape=2)
    radius = ti.field(dtype=dtype, shape=2)
    initial_position = np.array([[.3, -.7, 1.1], [-.8, .6, -.2]])
    initial_strength = np.array([[.4, -.2, .1], [.2, .3, -.6]])
    for i in range(2):
        position[i], strength[i], radius[i] = initial_position[i], initial_strength[i], .2
    basis, _ = np.linalg.qr(np.array([[1., 2., 3.], [3., -2., 1.], [2., 1., -.5]]))
    matrix = basis @ np.diag([.3, -.7, .4]) @ basis.T
    acceleration, start, duration = np.array([.2, -.1, .3]), 1.3, .8
    rhs = StrainRHS(matrix, acceleration, start)
    rk = RungeKutta(RK2(), max_n_particles=2, dtype=dtype)
    arguments = {"position": position, "vortex_strength": strength, "core_radius": radius,
                 "count": 2, "time": start, "time_step_size": duration, "right_hand_side": rhs}
    radius_before = radius.to_numpy().copy()
    if native:
        rk.advance(**arguments)
    else:
        advance_subcycles(RungeKutta.advance, rk, substeps, arguments)
    np.testing.assert_array_equal(radius.to_numpy(), radius_before)
    assert arguments["time"] == start and arguments["time_step_size"] == duration
    exponential = expm(duration * matrix)
    forcing = np.linalg.solve(matrix @ matrix, (exponential - np.eye(3) - duration * matrix) @ acceleration)
    exact = (initial_position @ exponential.T + forcing, initial_strength @ exponential.T)
    return (position.to_numpy(), strength.to_numpy()), exact, rhs


def test_time_dependent_3d_strain_converges_for_position_and_vector_strength():
    errors = []
    for count in (4, 8):
        fields, exact, rhs = integrate(count)
        errors.append([np.linalg.norm(value - target) for value, target in zip(fields, exact, strict=True)])
        np.testing.assert_allclose(rhs.times, [1.3 + (i + c) * (.8 / count) for i in range(count) for c in (0., .5)], rtol=0, atol=5e-16)
        assert rhs.intervals == [("RK2", .8 / count)] * count
    assert np.all(np.array(errors[0]) / errors[1] > 3.7)


def test_one_substep_is_bitwise_native_in_single_precision():
    wrapped, _, wrapped_rhs = integrate(1, dtype=ti.f32)
    native, _, native_rhs = integrate(1, dtype=ti.f32, native=True)
    for left, right in zip(wrapped, native, strict=True):
        np.testing.assert_array_equal(left, right)
    assert wrapped_rhs.times == native_rhs.times and wrapped_rhs.intervals == native_rhs.intervals
