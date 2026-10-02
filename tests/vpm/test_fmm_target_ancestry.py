"""Target ancestry coverage, memory and preparation failures."""

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.physics.induction.fmm.targets import FMMTargetEvaluator
from tests.vpm.test_fmm_device import _DeviceFMMHarness


@pytest.mark.parametrize("count", [1, 137])
def test_paths_refresh_with_geometry_offset_and_prefix_and_are_accounted(count):
    harness = _DeviceFMMHarness(capacity=1)
    evaluator = FMMTargetEvaluator(harness.induction.workspace, count)
    query = ti.Vector.field(3, ti.f32, shape=count + 4)
    rng = np.random.default_rng(378174)
    try:
        allocated = evaluator.estimated_memory_bytes()
        fields = {
            name: getattr(evaluator, name)
            for name in ("target_path", "target_path_length", "target_path_error")
        }
        expected = (evaluator.target_path_capacity + 1) * count * 4 + 4
        for name in fields:
            delattr(evaluator, name)
        assert allocated - evaluator.estimated_memory_bytes() == expected
        for name, field in fields.items():
            setattr(evaluator, name, field)
        for generation, current in enumerate((count, min(count, 7), 0, count)):
            positions = rng.normal(size=(count + 4, 3)).astype(np.float32)
            positions[: count // 2] *= 0.001 if generation % 2 else 0.5
            query.from_numpy(positions)
            evaluator.prepare_targets(query, current, target_start=4)
            assert evaluator.estimated_memory_bytes() == allocated
            assert all(getattr(evaluator, name) is field for name, field in fields.items())
            paths = evaluator.target_path.to_numpy()
            lengths = evaluator.target_path_length.to_numpy()
            parents = evaluator.tree.node_parent.to_numpy()
            for slot in range(current):
                reverse = []
                node = slot
                while node >= 0:
                    reverse.append(node)
                    node = parents[node]
                assert list(paths[: lengths[slot], slot]) == reverse[::-1]
    finally:
        evaluator.destroy()
        evaluator.destroy()


def test_failed_path_preparation_revokes_previous_tile_without_publication(monkeypatch):
    harness = _DeviceFMMHarness(capacity=1)
    evaluator = FMMTargetEvaluator(harness.induction.workspace, 7)
    query = ti.Vector.field(3, ti.f32, shape=7)
    query.from_numpy(np.arange(21, dtype=np.float32).reshape(7, 3))
    output = ti.Vector.field(3, ti.f32, shape=7)
    sentinel = np.full((7, 3), 79.0, np.float32)
    output.from_numpy(sentinel)
    try:
        evaluator.prepare_targets(query, 7)
        original_build = evaluator.tree.build

        def invalid_depth(*args, **kwargs):
            original_build(*args, **kwargs)
            evaluator.tree.node_depth[0] = evaluator.target_path_capacity

        monkeypatch.setattr(evaluator.tree, "build", invalid_depth)
        with pytest.raises(RuntimeError, match="bounded ancestry scratch"):
            evaluator.prepare_targets(query, 7)
        assert evaluator._prepared_count == 0
        evaluator.evaluate_prepared(output, None, harness.induction.physics._zero_velocity)
        np.testing.assert_array_equal(output.to_numpy(), sentinel)
        monkeypatch.setattr(evaluator.tree, "build", original_build)
        evaluator.prepare_targets(query, 7)
        assert evaluator._prepared_count == 7
    finally:
        evaluator.destroy()
