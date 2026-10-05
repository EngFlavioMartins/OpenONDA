"""CPU numerical checks for independent FMM near, far and queue scratch."""

import numpy as np
import pytest

from source.solvers.vpm.physics.induction.fmm import device
from tests.vpm.test_fmm_device import _DeviceFMMHarness


def _clustered_sources():
    rng = np.random.default_rng(97163)
    count = 64
    position = rng.normal(scale=0.025, size=(count, 3))
    position[: count // 2, 0] -= 2.0
    position[count // 2 :, 0] += 2.0
    strength = rng.normal(scale=0.015, size=(count, 3))
    radius = rng.uniform(0.09, 0.15, size=count)
    return tuple(np.asarray(value, dtype=np.float32) for value in (position, strength, radius))


def _interaction_partition(workspace, count):
    # The near pass reuses m2l_source for ordered adjacency. Rebuild the lists
    # before reading far pairs, without rerunning any numerical field pass.
    levels = int(workspace.tree._max_depth[None]) + 1
    workspace._build_interaction_lists(2 * levels)
    assert int(workspace._list_error[None]) == 0
    pairs = {}
    for kind in ("m2l", "near"):
        length = int(getattr(workspace, f"_{kind}_count")[None])
        target = getattr(workspace, f"{kind}_target").to_numpy()[:length]
        source = getattr(workspace, f"{kind}_source").to_numpy()[:length]
        pairs[kind] = set(zip(target.tolist(), source.tolist(), strict=True))
        assert len(pairs[kind]) == length, "a cell pair must appear once"

    # Each cell pair represents the Cartesian product of its particle ranges.
    # Both lists together must cover every directed pair, including self.
    tree = workspace.tree
    starts = tree.node_particle_start.to_numpy()
    sizes = tree.node_particle_count.to_numpy()
    order = tree.sorted_indices.to_numpy()[:count]
    coverage = np.zeros((count, count), dtype=np.int32)
    for family in pairs.values():
        for target, source in family:
            target_ids = order[starts[target] : starts[target] + sizes[target]]
            source_ids = order[starts[source] : starts[source] + sizes[source]]
            coverage[np.ix_(target_ids, source_ids)] += 1
    np.testing.assert_array_equal(coverage, np.ones_like(coverage))
    demand = device._ListCapacities(
        len(pairs["m2l"]), len(pairs["near"]), int(workspace._queue_peak_count[None])
    )
    return pairs, demand


@pytest.mark.parametrize("shortage", ("m2l", "near", "queue"))
def test_independent_list_growth_preserves_fields_and_pair_partition(monkeypatch, shortage):
    monkeypatch.setattr(device, "_FMM_LEAF_CAPACITY", 1)
    position, strength, radius = _clustered_sources()
    count = len(position)
    harness = _DeviceFMMHarness(capacity=count)
    backend = harness.induction
    oversized = 4 * count * count
    backend._replace_workspace(count, oversized)
    assert backend.workspace.list_capacities == (oversized,) * 3
    expected_fields = harness.evaluate(position, strength, radius)
    expected_pairs, demand = _interaction_partition(backend.workspace, count)
    assert demand.near > demand.m2l > 1
    assert demand.queue > 1

    initial = [value + count for value in demand]
    index = device._ListCapacities._fields.index(shortage)
    initial[index] = demand[index] - 1
    initial = device._ListCapacities(*initial)
    backend._replace_workspace(count, initial)
    errors = []
    original_grow = backend._grow_interaction_lists

    def capture_growth(error):
        errors.append(error)
        original_grow(error)

    monkeypatch.setattr(backend, "_grow_interaction_lists", capture_growth)
    with pytest.warns(RuntimeWarning, match="FMM interaction-list storage"):
        actual_fields = harness.evaluate(position, strength, radius)
    assert errors, "the undersized device allocation must overflow"
    for error in errors:
        overflowing = [
            name
            for name, observed, capacity in zip(
                device._ListCapacities._fields, error.counts, error.capacities, strict=True
            )
            if observed > capacity
        ]
        assert overflowing == [shortage]

    workspace = backend.workspace
    for component, before, after in zip(
        device._ListCapacities._fields, initial, workspace.list_capacities, strict=True
    ):
        assert after > before if component == shortage else after == before
    assert workspace.m2l_source.shape == (max(workspace.max_m2l_pairs, workspace.max_near_pairs),)
    if shortage == "near":
        assert workspace.max_near_pairs > workspace.max_m2l_pairs
    actual_pairs, actual_demand = _interaction_partition(workspace, count)
    assert actual_pairs == expected_pairs
    assert actual_demand == demand
    for name, expected, actual in zip(
        ("velocity", "gradient", "strength rate"), expected_fields, actual_fields, strict=True
    ):
        if name == "gradient":
            # Repeated parallel reference reductions have cancellation noise
            # in near-zero J components; constrain both whole-field norms.
            reference = expected.astype(np.float64)
            error = actual.astype(np.float64) - reference
            assert np.linalg.norm(error) <= 2e-6 * np.linalg.norm(reference) + 2e-7
            assert np.abs(error).max() <= 2e-6 * np.abs(reference).max() + 2e-7
        else:
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7, err_msg=name)
