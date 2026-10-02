"""Pure bound and small real-tree tests of a non-evaluating cost census."""

# ruff: noqa: I001 -- Frozen archive admission precedes numerical test imports.

import numpy as np
import pytest
import taichi as ti

from tests.vpm._fmm_target_order_census import (
    TargetOrderCensusEvaluator, local_evaluation_costs, taylor_coefficient,
)
from tests.vpm.test_fmm_device import _DeviceFMMHarness


def test_p12_point_one_has_tighter_taylor_majorants_than_p7_point_zero_four():
    ratios = [taylor_coefficient(12, d, 0.1) / taylor_coefficient(7, d, 0.04) for d in (1, 2)]
    assert ratios[0] < 1 / 18
    assert ratios[1] < 1 / 26
    assert taylor_coefficient(12, 1, 0) == 0
    with pytest.raises(ValueError):
        taylor_coefficient(1, 2, 0.1)


def test_cost_ledger_charges_old_only_nodes_for_global_order():
    costs = local_evaluation_costs(old_targets=100, promoted_targets=40, overlap_targets=10)
    assert costs["p12_selective_additional_local_target_coefficient_visits"] == 455 * 40 - 120 * 10
    assert costs["p12_global_additional_local_target_coefficient_visits"] == 335 * 100 + 455 * 30
    with pytest.raises(ValueError):
        local_evaluation_costs(10, 20, 11)


@pytest.mark.parametrize("kernel", ["GAUSSIAN", "WINCKELMANS"])
@pytest.mark.parametrize("core", [0.002, 2.0])
def test_census_promotes_only_all_outside_tail_without_evaluating_outputs(kernel, core):
    count = 137
    harness = _DeviceFMMHarness(capacity=1, kernel_name=kernel)
    harness.evaluate(np.zeros((1, 3), np.float32), np.array([[0, 1, 0]], np.float32), np.array([core], np.float32))
    evaluator = TargetOrderCensusEvaluator(harness.induction.workspace, count, max_pairs=4096)
    query = ti.Vector.field(3, ti.f32, shape=count)
    positions = np.zeros((count, 3), np.float32)
    positions[:, 0] = np.linspace(-0.4, 0.4, count)
    positions[:, 2] = 6.1
    query.from_numpy(positions)
    try:
        evaluator.prepare_targets(query, count)
        evaluator.velocity.fill(17)
        evaluator.gradient.fill(19)
        result = evaluator.census_image_block([(0, False), (12.2, True)])
        assert result["old_monopole_packets"] == 2
        assert result["old_monopole_target_terms"] == 2 * count
        assert result["old_m2l_packets"] == 0
        assert result["legacy_subtree_target_jobs_unchanged"] == 0
        assert result["partition_classification_errors"] == 0
        assert result["promotable_packets"] == (2 if core < 1 else 0)
        assert result["promotable_target_terms"] == (2 * count if core < 1 else 0)
        assert not result["fields_evaluated"] and not result["outputs_published"]
        assert result["p12_added_translation_vector_coefficients"] == (910 if core < 1 else 0)
        np.testing.assert_array_equal(evaluator.velocity.to_numpy(), 17)
        np.testing.assert_array_equal(evaluator.gradient.to_numpy(), 19)
        if core < 1:
            # An old-only local receives no newly promoted source. A global
            # order change must still charge its additional 335 coefficients.
            positions[:, 0] *= 0.25
            query.from_numpy(positions)
            evaluator.prepare_targets(query, count)
            existing = evaluator.census_image_block([(0, False), (12.2, True)])
            assert existing["old_m2l_packets"] == 2
            assert existing["promotable_packets"] == 0
            assert existing["old_local_nodes_per_successful_batch"] == 1
            assert existing["old_local_target_evaluations_per_batch"] == count
            assert existing["p12_selective_additional_local_target_coefficient_visits"] == 0
            assert existing["p12_global_additional_local_target_coefficient_visits"] == 335 * count
            np.testing.assert_array_equal(evaluator.velocity.to_numpy(), 17)
            np.testing.assert_array_equal(evaluator.gradient.to_numpy(), 19)
    finally:
        evaluator.destroy()
        evaluator.destroy()
