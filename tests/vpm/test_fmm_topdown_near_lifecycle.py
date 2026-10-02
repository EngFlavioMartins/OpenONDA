"""Bounded ownership/streaming checks for the unwired ancestry scheduling."""

import numpy as np
import taichi as ti

from source.solvers.vpm.physics.induction.fmm.targets import FMMTargetEvaluator
from tests.vpm import test_fmm_targets as original_tests
from tests.vpm._fmm_topdown_near_prototype import TopdownNearEvaluator
from tests.vpm.test_fmm_device import _DeviceFMMHarness


def test_topdown_keeps_streaming_and_late_decline_atomic(monkeypatch):
    monkeypatch.setattr(original_tests, "FMMTargetEvaluator", TopdownNearEvaluator)
    original_tests.test_streamed_image_subblocks_match_single_batch_without_partial_publication()
    original_tests.test_bounded_block_decline_does_not_publish_any_output()


def test_topdown_prepared_target_offset_reflections_and_geometry(monkeypatch):
    monkeypatch.setattr(original_tests, "FMMTargetEvaluator", TopdownNearEvaluator)
    original_tests.test_prepared_target_reflections_preserve_full_jacobian_and_geometry()


def test_topdown_path_memory_is_bounded_and_reused_for_prefix_changes():
    harness = _DeviceFMMHarness(capacity=1)
    harness.evaluate(np.zeros((1, 3), np.float32), np.ones((1, 3), np.float32), np.ones(1, np.float32))
    plain = FMMTargetEvaluator(harness.induction.workspace, 32)
    topdown = TopdownNearEvaluator(harness.induction.workspace, 32)
    query = ti.Vector.field(3, ti.f32, shape=40)
    rng = np.random.default_rng(789314)
    query.from_numpy(rng.normal(size=(40, 3)).astype(np.float32))
    try:
        # Both production and the instrumented wrapper own exactly one copy.
        assert topdown.estimated_memory_bytes() == plain.estimated_memory_bytes()
        path_handle = topdown.target_path
        owner_handle = topdown._fields
        allocated_bytes = topdown.estimated_memory_bytes()
        for count in (32, 7, 0, 1, 32):
            topdown.prepare_targets(query, count, target_start=8)
            assert topdown.target_path is path_handle
            assert topdown._fields is owner_handle
            assert topdown.estimated_memory_bytes() == allocated_bytes
            if count:
                path = topdown.target_path.to_numpy()
                lengths = topdown.target_path_length.to_numpy()
                for slot in range(count):
                    assert path[lengths[slot] - 1, slot] == slot
                    assert topdown.tree.node_parent[path[0, slot]] == -1
        topdown.tree.node_depth[0] = topdown.path_capacity
        topdown._prepare_target_paths(1)
        assert int(topdown.path_error[None]) == 1
    finally:
        plain.destroy()
        topdown.destroy()
        topdown.destroy()
