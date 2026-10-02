"""Accuracy regressions for eliminating singleton FMM expansion work."""

import math

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.kernels.base import make_vortex_kernel
from source.solvers.vpm.physics.induction.fmm.device import (
    _FMM_LEAF_CAPACITY,
    _MAX_TREE_LEVELS,
    _MULTI_INDICES,
)
from tests.vpm.test_fmm_device import _DeviceFMMHarness


@ti.kernel
def _count_full_tree_levels(workspace: ti.template(), count: ti.i32):
    for level in range(_MAX_TREE_LEVELS):
        workspace._level_node_count[level] = 0
    for node in range(count, 2 * count - 1):
        ti.atomic_add(workspace._level_node_count[workspace.tree.node_depth[node]], 1)
    for slot in range(count):
        workspace._active_leaf_nodes[slot] = slot
        workspace._particle_cell[slot] = slot
    workspace._active_leaf_count[None] = count


@ti.kernel
def _write_full_tree_levels(workspace: ti.template(), count: ti.i32):
    for node in range(count, 2 * count - 1):
        destination = ti.atomic_add(
            workspace._level_node_cursor[workspace.tree.node_depth[node]], 1
        )
        workspace._active_internal_nodes[destination] = node


@ti.kernel
def _restore_near_cell_map(workspace: ti.template(), count: ti.i32):
    for slot in range(count):
        cell = slot
        parent = workspace.tree.node_parent[cell]
        while parent >= 0 and workspace.tree.node_particle_count[parent] <= _FMM_LEAF_CAPACITY:
            cell = parent
            parent = workspace.tree.node_parent[cell]
        workspace._particle_cell[slot] = cell


def _full_singleton_reference(workspace, count):
    """Execute the previous full-tree expansion algebra on the identical tree.

    The near/far traversal and kernels are shared deliberately: this reference
    isolates the change in expansion scheduling, without changing a MAC or
    replacing the operator by a different approximation.
    """
    node_count = 2 * count - 1
    levels = int(workspace.tree._max_depth[None]) + 1
    _count_full_tree_levels(workspace, count)
    workspace._initialize_active_cell_offsets()
    _write_full_tree_levels(workspace, count)
    workspace.multipole.fill(0)
    workspace.local.fill(0)
    workspace.velocity.fill(0)
    workspace.gradient.fill(0)
    workspace.rate.fill(0)
    workspace._p2m_pass(count)
    for level in range(levels - 1, -1, -1):
        workspace._m2m_level(count, level)
    workspace._build_interaction_lists(2 * levels)
    assert int(workspace._list_error[None]) == 0
    pair_count = int(workspace._m2l_count[None])
    for start in range(0, pair_count, workspace.m2l_batch_size):
        length = min(workspace.m2l_batch_size, pair_count - start)
        workspace._m2l_derivative_pass(start, length)
        workspace._m2l_accumulate_pass(start, length)
    for level in range(levels):
        workspace._l2l_level(count, level)
    workspace._l2p_pass(count)
    _restore_near_cell_map(workspace, count)
    workspace._near_field_pass(node_count, count, int(workspace._near_count[None]))
    workspace._reset_rate_diagnostics()
    workspace._rate_pass(count, 1)  # TRANSPOSED
    workspace._finalize_rate_diagnostics()
    return (
        workspace.velocity.to_numpy()[:count].copy(),
        workspace.gradient.to_numpy()[:count].copy(),
        workspace.rate.to_numpy()[:count].copy(),
    )


def _ordered_pairs(workspace, kind):
    count = int(getattr(workspace, f"_{kind}_count")[None])
    target = getattr(workspace, f"{kind}_target").to_numpy()[:count]
    source = getattr(workspace, f"{kind}_source").to_numpy()[:count]
    order = np.lexsort((source, target))
    return np.column_stack((target[order], source[order]))


def test_active_cells_partition_particles_and_preserve_all_source_moments():
    rng = np.random.default_rng(6213)
    count = 257
    harness = _DeviceFMMHarness(capacity=count)
    position = rng.normal(size=(count, 3)).astype(np.float32)
    strength = rng.normal(scale=0.02, size=(count, 3)).astype(np.float32)
    radius = rng.uniform(0.005, 0.04, size=count).astype(np.float32)
    harness.evaluate(position, strength, radius)
    workspace = harness.induction.workspace
    tree = workspace.tree
    internal_count = int(workspace._active_internal_count[None])
    leaf_count = int(workspace._active_leaf_count[None])
    leaves = workspace._active_leaf_nodes.to_numpy()[:leaf_count]
    internals = workspace._active_internal_nodes.to_numpy()[:internal_count]
    sorted_indices = tree.sorted_indices.to_numpy()[:count]
    counts = tree.node_particle_count.to_numpy()
    starts = tree.node_particle_start.to_numpy()
    centres = tree.node_centre.to_numpy()
    moments = workspace.multipole.to_numpy().reshape(workspace.max_nodes, 20, 3)
    particle_cell = workspace._particle_cell.to_numpy()[:count]
    assert len(set(leaves)) == leaf_count
    assert set(leaves).isdisjoint(internals)
    assert np.all(counts[leaves] <= _FMM_LEAF_CAPACITY)
    assert np.all(counts[internals] > _FMM_LEAF_CAPACITY)
    assert internal_count < count // 3
    coverage = np.zeros(count, dtype=int)
    for node in leaves:
        slots = slice(starts[node], starts[node] + counts[node])
        coverage[slots] += 1
        assert np.all(particle_cell[slots] == node)
    np.testing.assert_array_equal(coverage, np.ones(count))
    for node in np.concatenate((leaves, internals)):
        selected = sorted_indices[starts[node] : starts[node] + counts[node]]
        offset = position[selected].astype(np.float64) - centres[node].astype(np.float64)
        for index, alpha in enumerate(_MULTI_INDICES):
            weight = np.prod(offset**alpha, axis=1) / math.prod(
                math.factorial(value) for value in alpha
            )
            expected = np.sum(strength[selected].astype(np.float64) * weight[:, None], axis=0)
            np.testing.assert_allclose(moments[node, index], expected, rtol=8e-5, atol=2e-6)
    assert harness.induction.diagnostics.last_active_cell_count == internal_count + leaf_count


def _polynomial_derivative(coefficients, offset, derivative):
    result = np.zeros(3, dtype=np.float64)
    for index, alpha in enumerate(_MULTI_INDICES):
        if any(alpha[axis] < derivative[axis] for axis in range(3)):
            continue
        factor = 1.0
        for axis in range(3):
            factor *= math.factorial(alpha[axis]) / math.factorial(alpha[axis] - derivative[axis])
            factor *= offset[axis] ** (alpha[axis] - derivative[axis])
        result += coefficients[index] * factor
    return result


@pytest.mark.parametrize("count", (21, 129))
def test_multi_particle_l2p_evaluates_complete_cubic_velocity_and_jacobian(count):
    rng = np.random.default_rng(82323)
    harness = _DeviceFMMHarness(capacity=count)
    position = rng.uniform(-0.5, 0.5, (count, 3)).astype(np.float32)
    strength = rng.normal(size=(count, 3)).astype(np.float32)
    harness.evaluate(position, strength, np.full(count, 0.03, dtype=np.float32))
    workspace = harness.induction.workspace
    assert (int(workspace._active_leaf_count[None]) == 1) == (count <= _FMM_LEAF_CAPACITY)
    root = int(workspace.tree._root[None])
    coefficients = rng.normal(size=(20, 3)).astype(np.float32)
    all_locals = np.zeros((workspace.max_nodes * 20, 3), dtype=np.float32)
    all_locals[root * 20 : (root + 1) * 20] = coefficients
    workspace.local.from_numpy(all_locals)
    workspace._nonzero_l2l_count[None] = 0
    for level in range(int(workspace.tree._max_depth[None]) + 1):
        workspace._l2l_level(count, level)
    if count > _FMM_LEAF_CAPACITY:
        assert int(workspace._nonzero_l2l_count[None]) > 0
    workspace._l2p_pass(count)
    centre = np.asarray(workspace.tree.node_centre[root], dtype=np.float64)
    expected_velocity = []
    expected_gradient = []
    for point in position:
        offset = point.astype(np.float64) - centre
        first = [
            _polynomial_derivative(coefficients, offset, tuple(np.eye(3, dtype=int)[a]))
            for a in range(3)
        ]
        second = np.empty((3, 3, 3))
        for a in range(3):
            for b in range(3):
                derivative = np.eye(3, dtype=int)[a] + np.eye(3, dtype=int)[b]
                second[a, b] = _polynomial_derivative(coefficients, offset, tuple(derivative))
        expected_velocity.append(
            [first[1][2] - first[2][1], first[2][0] - first[0][2], first[0][1] - first[1][0]]
        )
        expected_gradient.append(
            np.array(
                [
                    second[1, :, 2] - second[2, :, 1],
                    second[2, :, 0] - second[0, :, 2],
                    second[0, :, 1] - second[1, :, 0],
                ]
            )
        )
    np.testing.assert_allclose(
        workspace.velocity.to_numpy()[:count], expected_velocity, rtol=2e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        workspace.gradient.to_numpy()[:count], expected_gradient, rtol=2e-6, atol=2e-6
    )


@pytest.mark.parametrize(
    "kernel_name", ("GAUSSIAN", "WINCKELMANS", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN")
)
def test_active_cell_operator_preserves_partition_and_baseline_direct_error(
    kernel_name, monkeypatch
):
    count = 384
    rng = np.random.default_rng(67726)
    position = rng.normal(scale=0.09, size=(count, 3)).astype(np.float32)
    position[:128, 0] -= 3.0
    position[-128:, 0] += 3.0
    strength = rng.normal(scale=0.01, size=(count, 3)).astype(np.float32)
    strength[1::2] = -strength[::2]
    radius = rng.uniform(0.006, 0.035, size=count).astype(np.float32)
    harness = _DeviceFMMHarness(capacity=count, kernel_name=kernel_name)
    harness.induction._ensure_workspace(count)
    workspace = harness.induction.workspace
    recorded_partitions = []
    near_field = workspace._near_field_pass

    def record_before_near_reuses_m2l_scratch(*args):
        recorded_partitions.append(
            (_ordered_pairs(workspace, "m2l"), _ordered_pairs(workspace, "near"))
        )
        near_field(*args)

    monkeypatch.setattr(workspace, "_near_field_pass", record_before_near_reuses_m2l_scratch)
    actual = harness.evaluate(position, strength, radius)
    partition = recorded_partitions[0]
    assert len(partition[0]) > 0
    baseline = _full_singleton_reference(workspace, count)
    assert len(recorded_partitions) == 2
    np.testing.assert_array_equal(partition[0], recorded_partitions[1][0])
    np.testing.assert_array_equal(partition[1], recorded_partitions[1][1])
    kernel = make_vortex_kernel(kernel_name)
    delta = position[:, None, :].astype(np.float64) - position[None, :, :].astype(np.float64)
    velocity = kernel.velocity_pair(
        delta, strength[None, :, :], radius[:, None], radius[None, :]
    ).sum(axis=1)
    gradient = kernel.gradient_pair(
        delta, strength[None, :, :], radius[:, None], radius[None, :]
    ).sum(axis=1)
    direct = (velocity, gradient, np.einsum("nji,nj->ni", gradient, strength))
    for new, old, oracle in zip(actual, baseline, direct, strict=True):
        scale = np.linalg.norm(oracle)
        assert np.linalg.norm(new - old) / scale < 2e-6
        # The allowance is float32 summation roundoff, not the much looser
        # FMM qualification ceiling. Preserve the measured baseline error.
        assert np.linalg.norm(new - oracle) <= np.linalg.norm(old - oracle) + 2e-6 * scale
