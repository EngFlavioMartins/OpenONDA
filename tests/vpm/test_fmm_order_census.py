"""Coverage/qualification checks for the unwired device feasibility census."""

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.physics.induction.fmm.targets import FMMTargetEvaluator
from tests.vpm._fmm_order_census_prototype import SourceOrderFeasibilityCensus
from tests.vpm.test_fmm_device import _DeviceFMMHarness


@pytest.mark.parametrize("kernel", ["GAUSSIAN", "WINCKELMANS"])
def test_census_preserves_coverage_and_never_claims_runtime_admission(kernel):
    rng = np.random.default_rng(18183)
    source_count, target_count = 96, 9
    position = rng.uniform(-0.25, 0.25, (source_count, 3)).astype(np.float32)
    strength = rng.normal(0, 0.03, (source_count, 3)).astype(np.float32)
    core = np.full(source_count, 0.001, np.float32)
    targets = rng.uniform(-0.002, 0.002, (target_count, 3)).astype(np.float32)
    targets[:, 2] += 2
    harness = _DeviceFMMHarness(capacity=source_count, kernel_name=kernel)
    harness.evaluate(position, strength, core)
    evaluator = FMMTargetEvaluator(harness.induction.workspace, target_count, max_pairs=4096)
    queries = ti.Vector.field(3, ti.f32, shape=target_count)
    queries.from_numpy(targets)
    census = None
    try:
        evaluator.prepare_targets(queries, target_count)
        shifts = np.zeros(evaluator.max_images, np.float32)
        shifts[1] = 4
        odd = np.zeros(evaluator.max_images, np.int32)
        odd[1] = 1
        evaluator.image_shift.from_numpy(shifts)
        evaluator.image_odd.from_numpy(odd)
        evaluator.image_count[None] = 2
        evaluator.block_mode[None] = 1
        census = SourceOrderFeasibilityCensus(evaluator)
        result = census.run(2)
        assert result["runtime_admissible_interactions"] == 0
        assert sum(result["physical_source_target_coverage"]) == source_count * target_count * 2
        assert result["expected_coverage"] == source_count * target_count * 2
        assert result["source_active_cells"] < 2 * source_count - 1
        assert sum(result["cell_jobs"]) > 0
        np.testing.assert_array_equal(harness.induction.workspace.tree.position.to_numpy()[:source_count], position)
        np.testing.assert_array_equal(harness.induction.workspace.tree.vortex_strength.to_numpy()[:source_count], strength)
    finally:
        if census is not None:
            census.destroy()
        evaluator.destroy()
