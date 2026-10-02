"""Frozen-source coverage, grouping and parity gates for an unwired experiment.

Run this file alone in a fresh process: its prototype admits only the archived
root-first numerical modules, never a mixture with live production imports.
"""

# ruff: noqa: I001 -- Frozen source admission must precede numerical test imports.

import numpy as np
import pytest
import taichi as ti

from tests.vpm._fmm_flattened_near_prototype import (
    FROZEN_SOURCE_ROOT,
    FlattenedNearEvaluator,
    assert_frozen_numerical_imports,
    flattened_near_factory,
)
from tests.vpm import test_fmm_target_accuracy as accuracy_tests
from tests.vpm import test_fmm_targets as image_tests
from tests.vpm.test_fmm_device import _DeviceFMMHarness
from tests.vpm.test_fmm_specialized_near import _order_private_near


def test_flattened_factory_retains_frozen_reuse_whitelist():
    from source.solvers.vpm.physics.induction import reuse_backends
    from source.solvers.vpm.physics.induction.fmm import targets

    assert_frozen_numerical_imports()
    assert str(FROZEN_SOURCE_ROOT) in targets.__file__
    original = targets.FMMTargetEvaluator
    with flattened_near_factory():
        assert targets.FMMTargetEvaluator is FlattenedNearEvaluator
        assert reuse_backends.FMMTargetEvaluator is original
    assert targets.FMMTargetEvaluator is original


@pytest.mark.parametrize("kernel", ["GAUSSIAN", "WINCKELMANS"])
def test_flattened_identical_pairs_mixed_classes_parity_and_empty_private_slots(kernel):
    rng = np.random.default_rng(521018)
    count = 24
    position = rng.uniform(-0.15, 0.15, (8, 3)).astype(np.float32)
    strength = rng.normal(0, 0.1, (8, 3)).astype(np.float32)
    core = np.full(8, 1.5, np.float32)
    targets = rng.uniform(-0.25, 0.25, (count, 3)).astype(np.float32)
    targets[:, 2] += 5
    harness = _DeviceFMMHarness(capacity=8, kernel_name=kernel)
    harness.evaluate(position, strength, core)
    evaluator = FlattenedNearEvaluator(harness.induction.workspace, count, max_pairs=4096)
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
        names = ("ordered_near_source", "ordered_near_image", "ordered_near_legacy")
        saved = {name: getattr(evaluator, name).to_numpy()[:pairs].copy() for name in names}
        assert np.any(saved["ordered_near_source"] < 0)
        assert np.any(saved["ordered_near_legacy"])
        assert np.any((saved["ordered_near_source"] >= 0) & (saved["ordered_near_legacy"] == 0))
        evaluator._clear_outputs(count)
        evaluator._evaluate_grouped_near_lanes(inclusive, count)
        evaluator._reduce_near_lanes(count)
        expected = evaluator.velocity.to_numpy(), evaluator.gradient.to_numpy()
        evaluator._clear_outputs(count)
        evaluator._evaluate_near_lanes(inclusive, count)
        evaluator._reduce_near_lanes(count)
        actual = evaluator.velocity.to_numpy(), evaluator.gradient.to_numpy()
        for name in names:
            np.testing.assert_array_equal(getattr(evaluator, name).to_numpy()[:pairs], saved[name])
        for old, new in zip(expected, actual, strict=True):
            error = np.linalg.norm((old - new).reshape(count, -1), axis=1)
            scale = np.linalg.norm(old.reshape(count, -1), axis=1)
            assert np.all(error <= 16 * np.finfo(np.float32).eps * scale)
        evaluator.near_partial_velocity.fill(17)
        evaluator.near_partial_gradient.fill(19)
        evaluator.near_count.fill(0)
        inclusive.fill(0)
        evaluator._evaluate_near_lanes(inclusive, count)
        np.testing.assert_array_equal(evaluator.near_partial_velocity.to_numpy(), 0)
        np.testing.assert_array_equal(evaluator.near_partial_gradient.to_numpy(), 0)
    finally:
        evaluator.destroy()
        evaluator.destroy()


@pytest.mark.parametrize("kernel", ["GAUSSIAN", "WINCKELMANS"])
def test_flattened_preserves_adversarial_direct_envelope(monkeypatch, kernel):
    monkeypatch.setattr(accuracy_tests, "FMMTargetEvaluator", FlattenedNearEvaluator)
    accuracy_tests.test_target_local_error_preserves_legacy_envelope_at_cell_extremes(kernel)


@pytest.mark.parametrize("kernel", ["GAUSSIAN", "WINCKELMANS"])
def test_flattened_preserves_full_slab_tail_rates_and_direct_envelope(kernel):
    with flattened_near_factory():
        image_tests.test_full_slab_blocks_preserve_tail_stage_rates_and_target_operator(kernel)


def test_flattened_decline_keeps_caller_outputs_unpublished(monkeypatch):
    monkeypatch.setattr(image_tests, "FMMTargetEvaluator", FlattenedNearEvaluator)
    image_tests.test_bounded_block_decline_does_not_publish_any_output()
