"""Bounded CPU checks of the unwired class-specialized near-field experiment."""

import numpy as np
import pytest
import taichi as ti

from tests.vpm import test_fmm_target_accuracy as accuracy_tests
from tests.vpm import test_fmm_targets as image_tests
from tests.vpm._fmm_specialized_near_prototype import (
    SpecializedNearEvaluator,
    specialized_near_factory,
)
from tests.vpm.test_fmm_device import _DeviceFMMHarness


def test_qualification_factory_preserves_the_original_reuse_whitelist():
    from source.solvers.vpm.physics.induction import reuse_backends
    from source.solvers.vpm.physics.induction.fmm import targets

    original = targets.FMMTargetEvaluator
    with specialized_near_factory():
        assert targets.FMMTargetEvaluator is SpecializedNearEvaluator
        assert reuse_backends.FMMTargetEvaluator is original
    assert targets.FMMTargetEvaluator is original


def _order_private_near(evaluator, node_count):
    evaluator._copy_counts(node_count)
    inclusive, scratch = evaluator.scan_a, evaluator.scan_b
    stride = 1
    while stride < node_count:
        evaluator._scan_counts(inclusive, scratch, node_count, stride)
        inclusive, scratch = scratch, inclusive
        stride *= 2
    evaluator._initialise_cursor(inclusive, node_count)
    evaluator._order_near(int(evaluator.near_pair_count[None]))
    return inclusive


@pytest.mark.parametrize("kernel", ["GAUSSIAN", "WINCKELMANS"])
def test_split_near_uses_identical_ordered_pairs_and_private_buffers(kernel):
    rng = np.random.default_rng(521018)
    count = 24
    position = rng.uniform(-0.15, 0.15, (8, 3)).astype(np.float32)
    strength = rng.normal(0, 0.1, (8, 3)).astype(np.float32)
    core = np.full(8, 1.5, np.float32)
    targets = rng.uniform(-0.25, 0.25, (count, 3)).astype(np.float32)
    targets[:, 2] += 5
    harness = _DeviceFMMHarness(capacity=8, kernel_name=kernel)
    harness.evaluate(position, strength, core)
    evaluator = SpecializedNearEvaluator(harness.induction.workspace, count, max_pairs=4096)
    query = ti.Vector.field(3, ti.f32, shape=count)
    query.from_numpy(targets)
    try:
        evaluator.prepare_targets(query, count)
        shifts = np.zeros(evaluator.max_images, np.float32)
        shifts[:4] = [0, 3, 5, 10]
        odd = np.zeros(evaluator.max_images, np.int32)
        odd[3] = 1
        evaluator.image_shift.from_numpy(shifts)
        evaluator.image_odd.from_numpy(odd)
        evaluator.image_count[None] = 4
        evaluator.block_mode[None] = 1
        evaluator._initialise(2 * count - 1)
        evaluator._walk_sources(0, int(evaluator.leaf_count[None]), 0, 4, -1)
        assert int(evaluator.error[None]) == 0
        inclusive = _order_private_near(evaluator, 2 * count - 1)
        pairs = int(evaluator.near_pair_count[None])
        encoded = evaluator.ordered_near_source.to_numpy()[:pairs].copy()
        legacy = evaluator.ordered_near_legacy.to_numpy()[:pairs].copy()
        assert np.any(encoded < 0)
        assert np.any(legacy)
        assert np.any((encoded >= 0) & (legacy == 0))
        evaluator._clear_outputs(count)
        evaluator._evaluate_monolithic_near_lanes(inclusive, count)
        evaluator._reduce_near_lanes(count)
        expected = (evaluator.velocity.to_numpy(), evaluator.gradient.to_numpy())
        evaluator._clear_outputs(count)
        evaluator._evaluate_near_lanes(inclusive, count)
        evaluator._reduce_near_lanes(count)
        actual = (evaluator.velocity.to_numpy(), evaluator.gradient.to_numpy())
        np.testing.assert_array_equal(evaluator.ordered_near_source.to_numpy()[:pairs], encoded)
        np.testing.assert_array_equal(evaluator.ordered_near_legacy.to_numpy()[:pairs], legacy)
        for previous, current in zip(expected, actual, strict=True):
            difference = np.linalg.norm((previous - current).reshape(count, -1), axis=1)
            scale = np.linalg.norm(previous.reshape(count, -1), axis=1)
            assert np.all(difference <= 16 * np.finfo(np.float32).eps * scale)
        assert evaluator.last_diagnostics["prototype_near_classes"] == {
            "monopole_passes": 1, "heavy_passes": 1
        }
    finally:
        evaluator.destroy()


@pytest.mark.parametrize("kernel", ["GAUSSIAN", "WINCKELMANS"])
def test_split_near_preserves_adversarial_direct_envelope(monkeypatch, kernel):
    monkeypatch.setattr(accuracy_tests, "FMMTargetEvaluator", SpecializedNearEvaluator)
    accuracy_tests.test_target_local_error_preserves_legacy_envelope_at_cell_extremes(kernel)


@pytest.mark.parametrize("kernel", ["GAUSSIAN", "WINCKELMANS"])
def test_split_near_preserves_full_slab_tail_rates_and_direct_envelope(kernel):
    with specialized_near_factory():
        image_tests.test_full_slab_blocks_preserve_tail_stage_rates_and_target_operator(kernel)
