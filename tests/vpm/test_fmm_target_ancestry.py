"""Production ancestry coverage, memory and frozen bottom-up comparisons."""

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.physics.induction.fmm.targets import _NEAR_LANES, FMMTargetEvaluator
from tests.vpm.test_fmm_device import _DeviceFMMHarness
from tests.vpm.test_fmm_specialized_near import _order_private_near


@ti.data_oriented
class _BottomUpOracle(FMMTargetEvaluator):
    @ti.kernel
    def bottom_up_near(self, inclusive: ti.template(), count: ti.i32):
        # Frozen pre-ancestry traversal. Source kernels and private ordered
        # lists are shared so the only difference is target ancestor order.
        for work in range(count * _NEAR_LANES):
            lane, slot = work // count, work % count
            target = self.tree.sorted_indices[slot]
            position = self.tree.position[target]
            if self.block_mode[None] == 0:
                position = self._transform(position, 0)
            velocity = ti.Vector.zero(ti.f32, 3)
            gradient = ti.Matrix.zero(ti.f32, 3, 3)
            node = slot
            while node >= 0:
                pairs = self.near_count[node]
                first = inclusive[node] - pairs
                begin = first + (pairs * lane) // _NEAR_LANES
                end = first + (pairs * (lane + 1)) // _NEAR_LANES
                v, j = self._near_pairs_fields(begin, end, position)
                velocity += v
                gradient += j
                node = self.tree.node_parent[node]
            self.near_partial_velocity[target, lane] = velocity
            self.near_partial_gradient[target, lane] = gradient


@pytest.mark.parametrize("count", [1, 137])
def test_paths_refresh_with_geometry_offset_and_prefix_and_are_accounted(count):
    harness = _DeviceFMMHarness(capacity=1)
    evaluator = FMMTargetEvaluator(harness.induction.workspace, count)
    query = ti.Vector.field(3, ti.f32, shape=count + 4)
    rng = np.random.default_rng(378174)
    try:
        allocated = evaluator.estimated_memory_bytes()
        fields = {name: getattr(evaluator, name) for name in (
            "target_path", "target_path_length", "target_path_error"
        )}
        expected = (evaluator.target_path_capacity + 1) * count * 4 + 4
        for name in fields:
            delattr(evaluator, name)
        assert allocated - evaluator.estimated_memory_bytes() == expected
        for name, field in fields.items():
            setattr(evaluator, name, field)
        for generation, current in enumerate((count, min(count, 7), 0, count)):
            positions = rng.normal(size=(count + 4, 3)).astype(np.float32)
            positions[:count // 2] *= 0.001 if generation % 2 else 0.5
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
                assert list(paths[:lengths[slot], slot]) == reverse[::-1]
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


@pytest.mark.parametrize("kernel", ["GAUSSIAN", "WINCKELMANS"])
def test_root_first_matches_frozen_bottom_up_on_identical_mixed_pair_lists(kernel):
    rng = np.random.default_rng(521018)
    count = 24
    position = rng.uniform(-0.15, 0.15, (8, 3)).astype(np.float32)
    strength = rng.normal(0, 0.1, (8, 3)).astype(np.float32)
    core = np.full(8, 1.5, np.float32)
    targets = rng.uniform(-0.25, 0.25, (count, 3)).astype(np.float32)
    targets[:, 2] += 5
    harness = _DeviceFMMHarness(capacity=8, kernel_name=kernel)
    harness.evaluate(position, strength, core)
    evaluator = _BottomUpOracle(harness.induction.workspace, count, max_pairs=4096)
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
        evaluator.bottom_up_near(inclusive, count)
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
