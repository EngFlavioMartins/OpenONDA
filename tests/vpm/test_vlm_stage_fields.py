"""VLM stretching uses the selected formulation without host field transfers."""

from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.physics.induction.base import StageRates, StageState
from source.solvers.vpm.physics.stage_rhs import VLMStageContribution


@pytest.fixture(scope="module", autouse=True)
def cpu_runtime():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False, cpu_max_num_threads=2)
    yield
    ti.reset()


@ti.kernel
def _linear_field(
    position: ti.template(), velocity: ti.template(), gradient: ti.template(), count: ti.i32
):
    for i in range(count):
        jacobian = ti.Matrix([[0.0, 2.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        velocity[i] += jacobian @ position[i]
        gradient[i] = jacobian


class _VLM:
    def add_stage_velocity_and_gradient(
        self, position, velocity, gradient, count, stage_time, *, target_core_radius
    ):
        _linear_field(position, velocity, gradient, count)


@pytest.mark.parametrize(
    "scheme, expected",
    [
        ("direct", [4.0, -1.0, 0.0]),
        ("transposed", [-2.0, 2.0, 0.0]),
        ("mixed", [1.0, 0.5, 0.0]),
    ],
)
@pytest.mark.parametrize("with_gradient", [False, True])
def test_vlm_stretching_preserves_existing_rates_and_inactive_prefix(
    scheme, expected, with_gradient
):
    # This workspace deliberately has no NumPy download/upload helpers.
    physics = SimpleNamespace(
        accumulator_dtype=ti.f64,
        max_n_particles=2,
        induction=SimpleNamespace(stretching_scheme=scheme),
    )
    provider = VLMStageContribution(_VLM(), physics)
    position = ti.Vector.field(3, ti.f64, shape=2)
    strength = ti.Vector.field(3, ti.f64, shape=2)
    radius = ti.field(ti.f64, shape=2)
    velocity = ti.Vector.field(3, ti.f64, shape=2)
    rate = ti.Vector.field(3, ti.f64, shape=2)
    gradient = ti.Matrix.field(3, 3, ti.f64, shape=2) if with_gradient else None
    position.fill(1.0)
    strength.from_numpy(np.array([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]]))
    radius.fill(0.1)
    velocity.fill(10.0)
    rate.fill(20.0)
    if with_gradient:
        gradient.fill(30.0)

    provider.add_stage_rates(
        StageState(position, strength, radius, 1, time=1.0),
        1.0,
        StageRates(velocity, rate, gradient),
    )

    np.testing.assert_allclose(velocity.to_numpy(), [[12.0, 9.0, 10.0], [10.0, 10.0, 10.0]])
    np.testing.assert_allclose(rate.to_numpy()[0], np.asarray(expected) + 20.0)
    np.testing.assert_allclose(rate.to_numpy()[1], 20.0)
    if with_gradient:
        expected_gradient = np.full((2, 3, 3), 30.0)
        expected_gradient[0, 0, 1] += 2.0
        expected_gradient[0, 1, 0] -= 1.0
        np.testing.assert_allclose(gradient.to_numpy(), expected_gradient)
