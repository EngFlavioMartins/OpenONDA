"""Exact legacy-decision accounting against a separate host tree walk."""

# ruff: noqa: I001 -- Frozen source admission must precede all source imports.

import numpy as np
import pytest
import taichi as ti

from tests.vpm._fmm_flattened_near_prototype import FMMTargetEvaluator, assert_frozen_numerical_imports
from tests.vpm._fmm_legacy_descendant_work import LegacyDescendantWorkCounter
from tests.vpm.test_fmm_device import _DeviceFMMHarness


@ti.kernel
def _decisions(evaluator: ti.template(), accepted: ti.template(), count: ti.i32, image: ti.i32):
    for target, node in ti.ndrange(count, evaluator.source.max_nodes):
        point = evaluator.tree.position[evaluator.tree.sorted_indices[target]]
        if evaluator.block_mode[None]:
            point = evaluator._transform(point, image)
        accepted[target, node] = evaluator._legacy_accept(
            node, point - evaluator.source.tree.node_com[node]
        )


@ti.kernel
def _record(counter: ti.template(), target: ti.i32, source: ti.i32, image: ti.i32, order: ti.i32):
    counter.record(target, source, image, order)


def _host_walk(tree, accepted, stride, offset):
    left, right = tree.node_left.to_numpy(), tree.node_right.to_numpy()
    leaf, sizes = tree.node_is_leaf.to_numpy(), tree.node_particle_count.to_numpy()
    root = int(tree._root[None])
    result = dict.fromkeys((
        "sampled_target_jobs", "node_visits", "accepted_monopole_terminals",
        "exact_leaf_terminals", "exact_particle_terms", "opened_internal_nodes",
        "root_accepted_target_jobs", "terminal_source_coverage",
    ), 0)
    for target in range(offset, len(accepted), stride):
        result["sampled_target_jobs"] += 1
        pending = [root]
        while pending:
            node = pending.pop()
            result["node_visits"] += 1
            if accepted[target, node] or leaf[node]:
                if accepted[target, node]:
                    result["accepted_monopole_terminals"] += 1
                    result["root_accepted_target_jobs"] += int(node == root)
                else:
                    result["exact_leaf_terminals"] += 1
                    result["exact_particle_terms"] += int(sizes[node])
                result["terminal_source_coverage"] += int(sizes[node])
            else:
                result["opened_internal_nodes"] += 1
                pending.extend((int(right[node]), int(left[node])))
    return result


@pytest.mark.parametrize("kernel", ["GAUSSIAN", "WINCKELMANS"])
def test_actual_descendant_counts_match_host_walk_and_leave_fields_untouched(kernel):
    rng = np.random.default_rng(620341)
    count, targets = 33, 7
    positions = rng.uniform(-0.3, 0.3, (count, 3)).astype(np.float32)
    strengths = rng.normal(0, 0.03, (count, 3)).astype(np.float32)
    # Include both a cancelled pair and a zero-strength singleton: the latter
    # cannot be accepted as a monopole even at very long distances.
    positions[1] = positions[0]
    strengths[1] = -strengths[0]
    strengths[2] = 0
    cores = np.linspace(0.02, 0.2, count, dtype=np.float32)
    harness = _DeviceFMMHarness(capacity=count, kernel_name=kernel)
    harness.evaluate(positions, strengths, cores)
    evaluator = FMMTargetEvaluator(harness.induction.workspace, targets, max_pairs=4096)
    query = ti.Vector.field(3, ti.f32, shape=targets)
    points = rng.uniform(-0.02, 0.02, (targets, 3)).astype(np.float32)
    points[:, 2] += np.linspace(0, 6, targets)
    query.from_numpy(points)
    counters = []
    try:
        evaluator.prepare_targets(query, targets)
        evaluator.block_mode[None] = 1
        evaluator.image_count[None] = 2
        shifts = np.zeros(evaluator.max_images, np.float32)
        shifts[1] = 3
        odds = np.zeros(evaluator.max_images, np.int32)
        odds[1] = 1
        evaluator.image_shift.from_numpy(shifts)
        evaluator.image_odd.from_numpy(odds)
        evaluator.velocity.fill(17)
        evaluator.gradient.fill(19)
        accepted = ti.field(ti.i32, shape=(targets, evaluator.source.max_nodes))
        source = int(evaluator.source.tree._root[None])
        target = int(evaluator.tree._root[None])
        for stride, offset in ((1, 0), (3, 1)):
            counter = LegacyDescendantWorkCounter(evaluator, target_stride=stride, target_offset=offset)
            counters.append(counter)
            for image in range(2):
                _decisions(evaluator, accepted, targets, image)
                oracle = _host_walk(evaluator.source.tree, accepted.to_numpy(), stride, offset)
                _record(counter, target, source, image, image)
                report = counter.report()
                observed = report["by_order"][str((3, 5)[image])]
                assert observed["packets"] == 1
                assert observed["covered_target_jobs"] == targets
                for name, expected in oracle.items():
                    assert observed[name] == expected, (name, observed, oracle)
                assert observed["terminal_source_coverage"] == count * observed["sampled_target_jobs"]
                assert report["structural_or_decision_errors"] == [0, 0, 0]
                assert report["sampling"]["exhaustive"] == (stride == 1)
            counter.reset()
            assert not any(counter.report()["by_order"]["3"].values())
        np.testing.assert_array_equal(evaluator.velocity.to_numpy(), 17)
        np.testing.assert_array_equal(evaluator.gradient.to_numpy(), 19)
        assert_frozen_numerical_imports()
        capped = LegacyDescendantWorkCounter(evaluator, max_visits_per_target=1)
        counters.append(capped)
        _record(capped, target, source, 0, 0)
        with pytest.raises(RuntimeError, match="incomplete or inconsistent"):
            capped.report()
        counter = counters[0]
        evaluator.source.source_multipole_generation += 1
        with pytest.raises(RuntimeError, match="source changed"):
            counter.report()
    finally:
        for counter in counters:
            counter.destroy()
            counter.destroy()
        evaluator.destroy()


def test_constructor_rejects_invalid_sampling_before_allocation():
    for kwargs in ({"target_stride": 0}, {"target_stride": 1.5}, {"target_offset": 1}):
        with pytest.raises(ValueError):
            LegacyDescendantWorkCounter(None, **kwargs)
