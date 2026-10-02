"""Coverage and accuracy checks for the unwired root-first ancestry experiment."""

import numpy as np
import pytest
import taichi as ti

from tests.vpm import test_fmm_target_accuracy as accuracy_tests
from tests.vpm import test_fmm_targets as image_tests
from tests.vpm._fmm_topdown_near_prototype import TopdownNearEvaluator, topdown_near_factory
from tests.vpm.test_fmm_device import _DeviceFMMHarness
from tests.vpm.test_fmm_specialized_near import _order_private_near


def test_topdown_factory_keeps_original_cache_whitelist():
    from source.solvers.vpm.physics.induction import reuse_backends
    from source.solvers.vpm.physics.induction.fmm import targets

    original = targets.FMMTargetEvaluator
    with topdown_near_factory():
        assert targets.FMMTargetEvaluator is TopdownNearEvaluator
        assert reuse_backends.FMMTargetEvaluator is original
    assert targets.FMMTargetEvaluator is original


@pytest.mark.parametrize("count", [1, 137])
def test_topdown_paths_reverse_exact_parent_chain_and_refresh(count):
    harness = _DeviceFMMHarness(capacity=1)
    harness.evaluate(np.zeros((1, 3), np.float32), np.ones((1, 3), np.float32), np.ones(1, np.float32))
    evaluator = TopdownNearEvaluator(harness.induction.workspace, count)
    query = ti.Vector.field(3, ti.f32, shape=count)
    rng = np.random.default_rng(16573)
    try:
        for generation in range(2):
            positions = rng.normal(size=(count, 3)).astype(np.float32)
            positions[:count // 2] *= 1e-4 if generation else 0.1
            query.from_numpy(positions)
            evaluator.prepare_targets(query, count)
            parents = evaluator.tree.node_parent.to_numpy()
            path = evaluator.target_path.to_numpy()
            lengths = evaluator.target_path_length.to_numpy()
            for slot in range(count):
                bottom_up = []
                node = slot
                while node >= 0:
                    bottom_up.append(node)
                    node = parents[node]
                assert lengths[slot] == len(bottom_up)
                assert list(path[:lengths[slot], slot]) == bottom_up[::-1]
        evaluator.prepare_targets(query, 0)
        assert evaluator._prepared_count == 0
    finally:
        evaluator.destroy()
        evaluator.destroy()


@pytest.mark.parametrize("kernel", ["GAUSSIAN", "WINCKELMANS"])
def test_topdown_preserves_pair_lists_and_private_outputs(kernel):
    rng = np.random.default_rng(521018)
    count = 24
    position = rng.uniform(-0.15, 0.15, (8, 3)).astype(np.float32)
    strength = rng.normal(0, 0.1, (8, 3)).astype(np.float32)
    core = np.full(8, 1.5, np.float32)
    targets = rng.uniform(-0.25, 0.25, (count, 3)).astype(np.float32)
    targets[:, 2] += 5
    harness = _DeviceFMMHarness(capacity=8, kernel_name=kernel)
    harness.evaluate(position, strength, core)
    evaluator = TopdownNearEvaluator(harness.induction.workspace, count, max_pairs=4096)
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
        assert np.any(encoded < 0) and np.any(legacy)
        assert np.any((encoded >= 0) & (legacy == 0))
        evaluator._clear_outputs(count)
        evaluator._evaluate_bottomup_near_lanes(inclusive, count)
        evaluator._reduce_near_lanes(count)
        expected = evaluator.velocity.to_numpy(), evaluator.gradient.to_numpy()
        evaluator._clear_outputs(count)
        evaluator._evaluate_near_lanes(inclusive, count)
        evaluator._reduce_near_lanes(count)
        actual = evaluator.velocity.to_numpy(), evaluator.gradient.to_numpy()
        np.testing.assert_array_equal(evaluator.ordered_near_source.to_numpy()[:pairs], encoded)
        np.testing.assert_array_equal(evaluator.ordered_near_legacy.to_numpy()[:pairs], legacy)
        for old, new in zip(expected, actual, strict=True):
            difference = np.linalg.norm((old - new).reshape(count, -1), axis=1)
            scale = np.linalg.norm(old.reshape(count, -1), axis=1)
            assert np.all(difference <= 16 * np.finfo(np.float32).eps * scale)
    finally:
        evaluator.destroy()


@pytest.mark.parametrize("kernel", ["GAUSSIAN", "WINCKELMANS"])
def test_topdown_preserves_adversarial_direct_envelope(monkeypatch, kernel):
    monkeypatch.setattr(accuracy_tests, "FMMTargetEvaluator", TopdownNearEvaluator)
    accuracy_tests.test_target_local_error_preserves_legacy_envelope_at_cell_extremes(kernel)


@pytest.mark.parametrize("kernel", ["GAUSSIAN", "WINCKELMANS"])
def test_topdown_preserves_full_slab_tail_rates_and_direct_envelope(kernel):
    with topdown_near_factory():
        image_tests.test_full_slab_blocks_preserve_tail_stage_rates_and_target_operator(kernel)
