"""Independent finite-target particle induction checks."""

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.kernels.base import make_vortex_kernel
from source.solvers.vpm.physics.base import PhysicsBase
from source.solvers.vpm.physics.induction.base import StageState
from source.solvers.vpm.physics.stage_rhs import _StageParticleView


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
@pytest.mark.parametrize("dtype", [ti.f32, ti.f64])
def test_finite_target_taichi_matches_independent_host_sum(kernel_name, dtype):
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=dtype, offline_cache=False, cpu_max_num_threads=2)
    try:
        rng = np.random.default_rng(23)
        count, target_count = 17, 23
        physics = PhysicsBase(
            particle_kernel=kernel_name,
            max_n_particles=count,
            max_evaluation_points=target_count,
            accumulator_dtype=dtype,
        )
        x = ti.Vector.field(3, dtype=dtype, shape=count)
        g = ti.Vector.field(3, dtype=dtype, shape=count)
        core = ti.field(dtype=dtype, shape=count)
        background = ti.Vector.field(3, dtype=dtype, shape=())
        x.from_numpy(rng.normal(size=(count, 3)))
        g.from_numpy(rng.normal(size=(count, 3)))
        core.from_numpy(rng.uniform(0.05, 0.4, count))
        background[None] = [0.1, -0.2, 0.3]

        class Cloud:
            position, vortex_strength, core_radius, velocity_background = x, g, core, background

            def __len__(self):
                return count

        cloud = Cloud()
        targets = rng.normal(size=(target_count, 3)).astype(physics.np_dtype)
        targets[0] = x.to_numpy()[0]  # include a coincident source/target
        radii = rng.uniform(0.1, 0.8, target_count).astype(physics.np_dtype)
        expected = (
            make_vortex_kernel(kernel_name)
            .velocity_pair(
                targets[:, None, :].astype(float) - x.to_numpy()[None, :, :],
                g.to_numpy()[None, :, :],
                radii[:, None],
                core.to_numpy()[None, :],
            )
            .sum(axis=1)
        )
        tolerance = 2e-6 if dtype == ti.f32 else 3e-12
        for include in (True, False):
            actual = physics.compute_transport_target_velocity(
                cloud, targets, radii, include_freestream=include
            )
            reference = expected + (np.asarray(background[None]) if include else 0)
            np.testing.assert_allclose(actual, reference, atol=tolerance, rtol=tolerance)
        state = StageState(x, g, core, count)
        first = _StageParticleView(state, cloud, physics)
        second = _StageParticleView(state, cloud, physics)
        assert first.velocity_background is background
        assert second.velocity_background is background
        assert second.state_revision > first.state_revision
    finally:
        ti.reset()
