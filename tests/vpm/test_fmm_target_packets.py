"""Independent target-packet coverage across multiple coarse target starts."""

import numpy as np
import taichi as ti

from source.solvers.vpm.kernels.base import make_vortex_kernel
from source.solvers.vpm.physics.induction.fmm.targets import FMMTargetEvaluator
from tests.vpm.test_fmm_device import _DeviceFMMHarness


def test_disjoint_coarse_starts_cover_all_targets_and_image_parities_once():
    rng = np.random.default_rng(41957)
    count = 1057  # More than the maximal coarse target-start capacity.
    targets = rng.uniform(-2, 2, (count, 3)).astype(np.float32)
    targets[:, 2] += 10
    position = np.zeros((1, 3), dtype=np.float32)
    strength = np.array([[0.3, 0.5, 0.7]], dtype=np.float32)
    core = np.array([0.01], dtype=np.float32)
    harness = _DeviceFMMHarness(capacity=1)
    harness.evaluate(position, strength, core)
    backend = FMMTargetEvaluator(harness.induction.workspace, count)
    query = ti.Vector.field(3, ti.f32, shape=count)
    velocity = ti.Vector.field(3, ti.f32, shape=count)
    gradient = ti.Matrix.field(3, 3, ti.f32, shape=count)
    query.from_numpy(targets)
    images = [(0.0, False), (3.0, True), (100.0, False)]
    try:
        backend.prepare_targets(query, count)
        assert int(backend.leaf_count[None]) > 1
        backend.evaluate_image_block(
            images, velocity, gradient, harness.induction.physics._zero_velocity
        )
        assert backend.last_diagnostics["m2l_pairs"] > 0
        assert backend.last_diagnostics["monopole_target_pairs"] > 0
        kernel = make_vortex_kernel("GAUSSIAN")
        exact_velocity = np.zeros((count, 3), dtype=np.float64)
        exact_gradient = np.zeros((count, 3, 3), dtype=np.float64)
        velocity_scale = np.zeros(count, dtype=np.float64)
        gradient_scale = np.zeros(count, dtype=np.float64)
        for shift, odd in images:
            image_position = position.astype(np.float64).copy()
            image_strength = strength.astype(np.float64).copy()
            image_position[:, 2] = shift
            if odd:
                image_strength[:, :2] *= -1
            difference = targets.astype(np.float64) - image_position
            v = kernel.velocity_pair(difference, image_strength, core, core)
            j = kernel.gradient_pair(difference, image_strength, core, core)
            exact_velocity += v
            exact_gradient += j
            velocity_scale += np.linalg.norm(v, axis=1)
            gradient_scale += np.linalg.norm(j.reshape(count, -1), axis=1)
        for actual, exact, scale in (
            (velocity.to_numpy(), exact_velocity, velocity_scale),
            (gradient.to_numpy(), exact_gradient, gradient_scale),
        ):
            errors = np.linalg.norm((actual - exact).reshape(count, -1), axis=1)
            assert np.all(errors <= 16 * np.finfo(np.float32).eps * scale)
    finally:
        backend.destroy()
