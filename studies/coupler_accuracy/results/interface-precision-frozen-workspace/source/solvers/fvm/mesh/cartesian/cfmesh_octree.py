# SPDX-License-Identifier: GPL-3.0-or-later
"""Sparse octree refinement primitives following cfMesh's serial modifier.

cfMesh's object-size conversion deliberately uses a strict upper size bound,
unlike its surface-size conversion. Regularity propagates refinement to all
coarser face, edge and corner neighbours before splitting any selected leaf.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import product

import numpy as np

from source._numba import cacheable_njit as njit

from ..progress import mesh_event
from .config import BoxRefinement

Leaf = tuple[int, int, int, int, int, int]
_NEIGHBOURS = tuple(offset for offset in product((-1, 0, 1), repeat=3) if any(offset))


def object_additional_level(max_cell_size: float, requested: float) -> int:
    """Native ``objectRefinement::calculateAdditionalRefLevels`` conversion."""
    level = 0
    size = max_cell_size
    while requested <= size * (1.0 + 1.0e-15):
        level += 1
        size /= 2.0
    return level


class LeafLookup:
    """O(depth) leaf lookup without allocating a finest-level dense cube."""

    def __init__(self, leaves: Sequence[Leaf] | np.ndarray, max_level: int) -> None:
        """Index octree leaves by level for logarithmic spatial lookup.

        Parameters
        ----------
        leaves : sequence or ndarray, shape (n_leaves, 6)
            Rows ``(x, y, z, width, level, kind)`` in finest-grid integer
            coordinates.
        max_level : int
            Finest octree level and coordinate extent exponent.

        Notes
        -----
        The constructor stores integer lookup dictionaries only; it does not
        alter the supplied leaf array. Coordinates outside
        ``[0, 2**max_level)`` are rejected later by :meth:`find` with the
        sentinel ``-1``.
        """
        self.max_level = max_level
        self.limit = 2**max_level
        self.maps: list[dict[tuple[int, int, int], int]] = [{} for _ in range(max_level + 1)]
        for leaf_id, row in enumerate(leaves):
            x, y, z, _width, level, _kind = map(int, row)
            self.maps[level][(x, y, z)] = leaf_id

    def find(self, x: int, y: int, z: int, *, level: int | None = None) -> int:
        """Return a containing leaf; -1 also denotes a refined query cube."""
        if min(x, y, z) < 0 or max(x, y, z) >= self.limit:
            return -1
        top = self.max_level if level is None else level
        for current in range(top, -1, -1):
            width = 2 ** (self.max_level - current)
            key = ((x // width) * width, (y // width) * width, (z // width) * width)
            found = self.maps[current].get(key)
            if found is not None:
                return found
        return -1

    def neighbours(self, leaf: Leaf) -> set[int]:
        """Same-level or coarser neighbours in the 26 regularity positions."""
        x, y, z, width, level, _kind = leaf
        return {
            found
            for dx, dy, dz in _NEIGHBOURS
            if (found := self.find(x + dx * width, y + dy * width, z + dz * width, level=level))
            >= 0
        }

    def in_box(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """Morton-ordered leaves intersecting an inclusive integer query box."""
        result: list[int] = []

        def visit(x: int, y: int, z: int, width: int, level: int) -> None:
            if (
                x > upper[0]
                or y > upper[1]
                or z > upper[2]
                or x + width <= lower[0]
                or y + width <= lower[1]
                or z + width <= lower[2]
            ):
                return
            found = self.maps[level].get((x, y, z))
            if found is not None:
                result.append(found)
                return
            if level == self.max_level:
                return
            child = width // 2
            for dz in (0, child):
                for dy in (0, child):
                    for dx in (0, child):
                        visit(x + dx, y + dy, z + dz, child, level + 1)

        visit(0, 0, 0, self.limit, 0)
        return np.asarray(result, dtype=np.int64)


def refine_selected_leaves(
    leaves: list[Leaf],
    selected: set[int],
    max_level: int,
    classify: Callable[[int, int, int, int, int], Leaf],
) -> list[Leaf]:
    """Ensure 1-irregularity, split selected leaves, retain Morton order."""
    if not selected:
        return leaves
    lookup = LeafLookup(leaves, max_level)
    front = list(selected)
    while front:
        leaf_id = front.pop()
        leaf = leaves[leaf_id]
        for neighbour in lookup.neighbours(leaf):
            if leaves[neighbour][4] < leaf[4] and neighbour not in selected:
                selected.add(neighbour)
                front.append(neighbour)
    refined: list[Leaf] = []
    for leaf_id, leaf in enumerate(leaves):
        if leaf_id not in selected:
            refined.append(leaf)
            continue
        x, y, z, width, level, _kind = leaf
        child = width // 2
        if child < 1:
            raise ValueError("cfMesh refinement exceeded the configured finest lattice")
        for dz in (0, child):
            for dy in (0, child):
                for dx in (0, child):
                    refined.append(classify(x + dx, y + dy, z + dz, child, level + 1))
    return refined


def refine_objects(
    leaves: list[Leaf],
    requests: Sequence[BoxRefinement],
    *,
    root_lower: np.ndarray,
    root_size: float,
    max_cell_size: float,
    global_level: int,
    max_level: int,
    classify: Callable[[int, int, int, int, int], Leaf],
) -> list[Leaf]:
    """Native box intersection and regularity, excluding outside leaves."""
    finest = root_size / (2**max_level)
    tolerance = 1.0e-15 * root_size
    controls = [
        (
            np.asarray(request.bounds[::2]),
            np.asarray(request.bounds[1::2]),
            global_level + object_additional_level(max_cell_size, request.cell_size),
        )
        for request in requests
    ]
    pass_number = 0
    while True:
        pass_number += 1
        selected: set[int] = set()
        for leaf_id, leaf in enumerate(leaves):
            x, y, z, width, level, kind = leaf
            if kind == 0:
                continue
            lower = root_lower + finest * np.asarray((x, y, z)) - tolerance
            upper = root_lower + finest * (np.asarray((x, y, z)) + width) + tolerance
            if any(
                level < target and np.all(lower <= high) and np.all(upper >= low)
                for low, high, target in controls
            ):
                selected.add(leaf_id)
        mesh_event(
            "object refinement pass",
            pass_number=pass_number,
            leaves=len(leaves),
            selected=len(selected),
        )
        if not selected:
            return leaves
        leaves = refine_selected_leaves(leaves, selected, max_level, classify)


@njit(cache=True, fastmath=False)
def _balance_selection_kernel(leaves: np.ndarray, max_level: int) -> np.ndarray:
    """Select coarse neighbours in compiled code without a dense lattice.

    Use separate level/x/y/z keys: packing them into one integer would
    overflow for deep octrees. The caller supplies contiguous int64 records.
    The returned mask is independent of traversal order; splitting and
    geometric classification retain their existing Morton order in Python.
    """
    lookup = {(row[4], row[0], row[1], row[2]): index for index, row in enumerate(leaves)}
    selected = np.zeros(len(leaves), dtype=np.bool_)
    limit = 1 << max_level
    for leaf_id in range(len(leaves)):
        x, y, z, width, level = leaves[leaf_id, :5]
        for dz in range(-1, 2):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    qx, qy, qz = x + dx * width, y + dy * width, z + dz * width
                    if min(qx, qy, qz) < 0 or max(qx, qy, qz) >= limit:
                        continue
                    for current in range(level, -1, -1):
                        neighbour_width = 1 << (max_level - current)
                        key = (
                            current,
                            (qx // neighbour_width) * neighbour_width,
                            (qy // neighbour_width) * neighbour_width,
                            (qz // neighbour_width) * neighbour_width,
                        )
                        if key in lookup:
                            neighbour = lookup[key]
                            if leaves[neighbour, 4] + 1 < level:
                                selected[neighbour] = True
                            break
    return np.flatnonzero(selected)


def balance_leaves(
    leaves: list[Leaf],
    max_level: int,
    classify: Callable[[int, int, int, int, int], Leaf],
) -> list[Leaf]:
    """Close 2:1 gaps with a cached, CPU-compiled neighbour scan."""
    pass_number = 0
    while True:
        pass_number += 1
        records = np.asarray(leaves, dtype=np.int64).reshape(-1, 6)
        selected = set(map(int, _balance_selection_kernel(records, max_level)))
        mesh_event(
            "octree balance pass",
            pass_number=pass_number,
            leaves=len(leaves),
            selected=len(selected),
        )
        if not selected:
            return leaves
        leaves = refine_selected_leaves(leaves, selected, max_level, classify)


def refine_near_data(
    leaves: list[Leaf],
    max_level: int,
    classify: Callable[[int, int, int, int, int], Leaf],
) -> tuple[list[Leaf], int]:
    """One native near-DATA layer, followed by the regularity closure."""
    lookup = LeafLookup(leaves, max_level)
    selected: set[int] = set()
    for leaf in leaves:
        if leaf[5] != 2:
            continue
        for neighbour in lookup.neighbours(leaf):
            candidate = leaves[neighbour]
            if candidate[5] != 0 and candidate[4] != leaf[4]:
                selected.add(neighbour)
    requested_count = len(selected)
    return refine_selected_leaves(leaves, selected, max_level, classify), requested_count
