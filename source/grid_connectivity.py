"""Compact connected components for grouped, blocked Cartesian grid links."""

import numpy as np

from source._numba import cacheable_njit as njit


@njit(cache=True)
def connected_grid_components(magnitude, groups, links):
    """Label active fluid support without constructing a sparse edge graph.

    Union-find visits the three preceding lattice neighbours once. Scratch
    storage is one int32 parent per active node in addition to the returned
    labels, independent of the number of open links.
    """
    nx, ny, nz = magnitude.shape
    labels = np.full(magnitude.shape, -1, dtype=np.int32)
    active_count = 0
    for value in magnitude.flat:
        active_count += value > 0
    parents = np.arange(active_count, dtype=np.int32)
    node = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if not magnitude[i, j, k] > 0:
                    continue
                labels[i, j, k] = node
                for axis in range(3):
                    ni, nj, nk = i, j, k
                    if axis == 0:
                        ni -= 1
                    elif axis == 1:
                        nj -= 1
                    else:
                        nk -= 1
                    if ni < 0 or nj < 0 or nk < 0:
                        continue
                    neighbour = labels[ni, nj, nk]
                    if (
                        neighbour < 0
                        or links[ni, nj, nk] & (1 << axis)
                        or groups[ni, nj, nk] != groups[i, j, k]
                    ):
                        continue
                    first, second = node, neighbour
                    while parents[first] != first:
                        parents[first] = parents[parents[first]]
                        first = parents[first]
                    while parents[second] != second:
                        parents[second] = parents[parents[second]]
                        second = parents[second]
                    # Root order makes component labels deterministic in the
                    # same first-node order as the previous graph traversal.
                    if first < second:
                        parents[second] = first
                    else:
                        parents[first] = second
                node += 1

    for node in range(active_count):
        root = node
        while parents[root] != root:
            parents[root] = parents[parents[root]]
            root = parents[root]
        parents[node] = root
    component_count = 0
    for node in range(active_count):
        root = parents[node]
        if root == node:
            parents[node] = -component_count - 1
            component_count += 1
        else:
            parents[node] = parents[root]
    for index in range(labels.size):
        node = labels.flat[index]
        if node >= 0:
            labels.flat[index] = -parents[node] - 1
    return labels
