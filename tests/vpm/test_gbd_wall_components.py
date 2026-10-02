"""Wall pruning labels preserve lattice connectivity with bounded scratch."""

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from source.grid_connectivity import connected_grid_components


def _graph_labels(magnitude, groups, links):
    """Independent sparse-graph reference for the lattice's open links."""
    active = magnitude > 0
    ids = np.full(magnitude.shape, -1, dtype=np.int64)
    ids[active] = np.arange(active.sum())
    sources, targets = [], []
    for axis in range(3):
        lower, upper = [slice(None)] * 3, [slice(None)] * 3
        lower[axis], upper[axis] = slice(None, -1), slice(1, None)
        lower, upper = tuple(lower), tuple(upper)
        connected = active[lower] & active[upper] & ((links[lower] & (1 << axis)) == 0)
        connected &= groups[lower] == groups[upper]
        sources.append(ids[lower][connected])
        targets.append(ids[upper][connected])
    rows, columns = np.concatenate(sources), np.concatenate(targets)
    graph = coo_matrix(
        (np.ones(len(rows), dtype=bool), (rows, columns)), shape=(int(active.sum()),) * 2
    ).tocsr()
    _, components = connected_components(graph, directed=False)
    labels = np.full(magnitude.shape, -1, dtype=np.int32)
    labels[active] = components
    return labels


@pytest.mark.parametrize("shape", [(1, 3, 4), (7, 8, 9)])
@pytest.mark.parametrize("fraction", [0.0, 0.05, 1.0])
@pytest.mark.parametrize("blocked", [False, True])
def test_wall_components_match_graph_reference(shape, fraction, blocked):
    rng = np.random.default_rng(41)
    magnitude = (rng.random(shape) < fraction).astype(np.float32)
    groups = (
        rng.integers(0, 3, size=shape, dtype=np.int32) if blocked else np.zeros(shape, np.int32)
    )
    links = rng.integers(0, 8, size=shape, dtype=np.int32) if blocked else np.zeros(shape, np.int32)
    actual = connected_grid_components(magnitude, groups, links)
    expected = _graph_labels(magnitude, groups, links)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.int32
