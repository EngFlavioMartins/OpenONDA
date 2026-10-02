"""Real-operator CPU checks of instrumentation-only AABB classification."""

import numpy as np
import pytest
import taichi as ti

from tests.vpm._fmm_aabb_device_census import AABBCensusEvaluator
from tests.vpm.test_fmm_device import _DeviceFMMHarness


@pytest.mark.parametrize("kernel", ["GAUSSIAN", "WINCKELMANS"])
def test_aabb_census_validates_actual_points_without_changing_outputs(kernel):
    rng = np.random.default_rng(35171)
    position = rng.uniform(-0.5, 0.5, (64, 3)).astype(np.float32)
    strength = rng.normal(0, 0.1, (64, 3)).astype(np.float32)
    core = rng.uniform(0.005, 0.02, 64).astype(np.float32)
    targets = rng.uniform(-1, 1, (37, 3)).astype(np.float32)
    targets[:, 1] *= 0.001
    targets[:, 2] += 2
    harness = _DeviceFMMHarness(capacity=64, kernel_name=kernel)
    harness.evaluate(position, strength, core)
    evaluator = AABBCensusEvaluator(harness.induction.workspace, len(targets), max_pairs=16384)
    query = ti.Vector.field(3, ti.f32, shape=len(targets))
    output = ti.Vector.field(3, ti.f32, shape=len(targets))
    gradient = ti.Matrix.field(3, 3, ti.f32, shape=len(targets))
    query.from_numpy(targets)
    try:
        evaluator.prepare_targets(query, len(targets))
        evaluator.evaluate_image_block([(0, False), (4, True)], output, gradient, harness.induction.physics._zero_velocity)
        result = evaluator.last_diagnostics["aabb_decision_census"]
        assert result["pointwise_conservatism_errors"] == 0
        assert sum(result["cell_jobs"]) > 0
        before = (output.to_numpy(), gradient.to_numpy())
        # Re-recording only instrumentation cannot change output arrays.
        evaluator._record_aabb_decisions(int(evaluator.near_pair_count[None]))
        assert int(evaluator.census_errors[None]) == 0
        np.testing.assert_array_equal(output.to_numpy(), before[0])
        np.testing.assert_array_equal(gradient.to_numpy(), before[1])
    finally:
        evaluator.destroy()
