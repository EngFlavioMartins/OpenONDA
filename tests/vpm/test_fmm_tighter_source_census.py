"""Small numerical metadata checks against the immutable root-first source."""

# ruff: noqa: I001 -- Archive admission precedes every numerical source import.
from tests.vpm._fmm_tighter_source_census import TighterSourceOrderCensus
from tests.vpm._fmm_flattened_near_prototype import assert_frozen_numerical_imports

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.physics.induction.fmm.targets import FMMTargetEvaluator
from tests.vpm._fmm_order_census_prototype import SourceOrderFeasibilityCensus
from tests.vpm._fmm_rank_one_remainder import core_tail_bound
from tests.vpm.test_fmm_device import _DeviceFMMHarness


@ti.kernel
def _core_probe(census: ti.template(), output: ti.template(), distance: ti.f64, core: ti.f64):
    for i in range(4):
        output[i] = census._actual_core_defect(
            distance * (i + 1), core, ti.cast(2.5, ti.f64)
        )


@ti.kernel
def _root_opportunity(census: ti.template(), output: ti.template()):
    output[None] = census._potential_order(
        census.evaluator.tree._root[None], census.tree._root[None], 0
    )


@pytest.mark.parametrize("kernel", ["GAUSSIAN", "WINCKELMANS"])
def test_tighter_census_shares_metadata_preserves_coverage_and_checks_actual_work(kernel):
    rng = np.random.default_rng(18183)
    position = rng.uniform(-0.25, 0.25, (96, 3)).astype(np.float32)
    strength = rng.normal(0, 0.03, position.shape).astype(np.float32)
    core = np.full(96, 0.001, np.float32)
    targets = rng.uniform(-0.002, 0.002, (9, 3)).astype(np.float32)
    targets[:, 2] += 2
    harness = _DeviceFMMHarness(capacity=96, kernel_name=kernel)
    harness.evaluate(position, strength, core)
    evaluator = FMMTargetEvaluator(harness.induction.workspace, 9, max_pairs=4096)
    query = ti.Vector.field(3, ti.f32, shape=9)
    query.from_numpy(targets)
    census = historical = None
    try:
        evaluator.prepare_targets(query, 9)
        shifts = np.zeros(evaluator.max_images, np.float32)
        shifts[1] = 4
        odds = np.zeros(evaluator.max_images, np.int32)
        odds[1] = 1
        evaluator.image_shift.from_numpy(shifts)
        evaluator.image_odd.from_numpy(odds)
        evaluator.image_count[None] = 2
        evaluator.block_mode[None] = 1
        evaluator.velocity.fill(17)
        evaluator.gradient.fill(19)
        census = TighterSourceOrderCensus(evaluator)
        original_metadata = census.moments.to_numpy()
        reference = census.run(2, policy="historical")
        tighter = census.run(2, policy="tight")
        historical = SourceOrderFeasibilityCensus(evaluator)
        previous = historical.run(2)
        for key in ("cell_jobs", "target_jobs", "physical_source_target_coverage"):
            assert reference[key] == previous[key]
        for result in (reference, tighter):
            assert result["runtime_admissible_interactions"] == 0
            assert result["relative_conditioning_budget"] == 16 * np.finfo(np.float32).eps
            assert result["remaining_arithmetic_budget_fraction"] == 0.5
            assert sum(result["physical_source_target_coverage"]) == 96 * 9 * 2
            work = result["legacy_descendant_work"]
            assert work["structural_or_decision_errors"] == [0, 0, 0]
            for index, order in enumerate((3, 5, 7, 9)):
                assert work["by_order"][str(order)]["covered_target_jobs"] == result["target_jobs"][index + 1]
                assert work["by_order"][str(order)]["packets"] == result["cell_jobs"][index + 1]
        np.testing.assert_array_equal(census.moments.to_numpy(), original_metadata)
        assert np.all(evaluator.velocity.to_numpy() == 17)
        assert np.all(evaluator.gradient.to_numpy() == 19)
        output = ti.Vector.field(2, ti.f64, shape=4)
        _core_probe(census, output, 0.03, 0.012)
        expected = [core_tail_bound(kernel, 0.03 * (i + 1), 0.012, 2.5) for i in range(4)]
        np.testing.assert_allclose(output.to_numpy(), [[x.velocity, x.gradient] for x in expected], rtol=2e-13, atol=0)
        decision = ti.field(ti.i32, shape=())
        original_strength = census.strength.to_numpy()
        for value in (np.inf, np.nan):
            census.strength.fill(value)
            _root_opportunity(census, decision)
            assert decision[None] == -1
        census.strength.from_numpy(original_strength)
        census.moments.fill(np.inf)
        _root_opportunity(census, decision)
        assert decision[None] == -1
        census.moments.from_numpy(original_metadata)
        assert_frozen_numerical_imports()
    finally:
        if historical is not None:
            historical.destroy()
        if census is not None:
            census.destroy()
        evaluator.destroy()
