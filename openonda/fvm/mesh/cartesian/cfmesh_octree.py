# SPDX-License-Identifier: GPL-3.0-or-later
"""Sparse octree refinement primitives following cfMesh's serial modifier.

cfMesh's object-size conversion deliberately uses a strict upper size bound,
unlike its surface-size conversion. Regularity propagates refinement to all
coarser face, edge and corner neighbours before splitting any selected leaf.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from itertools import product

import numpy as np

from openonda.core._numba import cacheable_njit as njit

from ..progress import mesh_event
from .config import BoxRefinement

Leaf = tuple[int, int, int, int, int, int]
# Classifies new leaves: rows (x, y, z, width, level) to rows with their kind
LeafClassifier = Callable[[np.ndarray], np.ndarray]
_NEIGHBOURS = tuple(offset for offset in product((-1, 0, 1), repeat=3) if any(offset))
# Child corner offsets in cfMesh's z, y, x nesting order, x fastest
_CHILD_CORNERS = (np.arange(8)[:, None] >> np.arange(3)) & 1


def leaf_records(leaves: Sequence[Leaf] | np.ndarray) -> np.ndarray:
    """Leaves as contiguous int64 rows ``(x, y, z, width, level, kind)``."""
    return np.ascontiguousarray(np.asarray(leaves, dtype=np.int64).reshape(-1, 6))


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


@njit(cache=True, fastmath=False)
def _find_leaf(lookup: dict, x: int, y: int, z: int, level: int, max_level: int) -> int:
    """Leaf of ``level`` or coarser containing lattice cell ``(x, y, z)``, else -1."""
    if min(x, y, z) < 0 or max(x, y, z) >= 1 << max_level:
        return -1
    for current in range(level, -1, -1):
        width = 1 << (max_level - current)
        key = (current, (x // width) * width, (y // width) * width, (z // width) * width)
        if key in lookup:
            return lookup[key]
    return -1


@njit(cache=True, fastmath=False)
def _regularity_closure_kernel(
    leaves: np.ndarray, selected: np.ndarray, max_level: int
) -> np.ndarray:
    """Add every coarser face, edge and corner neighbour of a selected leaf, recursively.

    The closure is a reachability set, so it does not depend on visit order.
    """
    lookup = {(row[4], row[0], row[1], row[2]): index for index, row in enumerate(leaves)}
    closed = selected.copy()
    front = list(np.flatnonzero(closed))
    while len(front) > 0:
        leaf_id = front.pop()
        x, y, z, width, level = leaves[leaf_id, :5]
        for dz in range(-1, 2):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbour = _find_leaf(
                        lookup, x + dx * width, y + dy * width, z + dz * width, level, max_level
                    )
                    if neighbour >= 0 and leaves[neighbour, 4] < level and not closed[neighbour]:
                        closed[neighbour] = True
                        front.append(neighbour)
    return closed


def refine_selected_leaves(
    leaves: Sequence[Leaf] | np.ndarray,
    selected: Collection[int] | np.ndarray,
    max_level: int,
    classify: LeafClassifier,
) -> np.ndarray:
    """Ensure 1-irregularity, split selected leaves, retain Morton order."""
    records = leaf_records(leaves)
    if not len(selected):
        return records
    mask = np.zeros(len(records), dtype=np.bool_)
    mask[np.fromiter(selected, dtype=np.int64, count=len(selected))] = True
    mask = _regularity_closure_kernel(records, mask, max_level)
    parents = records[mask]
    child = parents[:, 3] // 2
    if np.any(child < 1):
        raise ValueError("cfMesh refinement exceeded the configured finest lattice")
    children = np.empty((len(parents), 8, 5), dtype=np.int64)
    children[:, :, :3] = parents[:, None, :3] + _CHILD_CORNERS[None] * child[:, None, None]
    children[:, :, 3] = child[:, None]
    children[:, :, 4] = parents[:, None, 4] + 1
    # Each leaf keeps its position, a split leaf giving way to its 8 children.
    counts = np.where(mask, 8, 1)
    starts = np.cumsum(counts) - counts
    refined = np.empty((int(counts.sum()), 6), dtype=np.int64)
    refined[starts[~mask]] = records[~mask]
    refined[(starts[mask][:, None] + np.arange(8)).ravel()] = classify(children.reshape(-1, 5))
    return refined


@njit(cache=True, fastmath=False)
def _object_selection_kernel(
    leaves: np.ndarray,
    root_lower: np.ndarray,
    finest: float,
    tolerance: float,
    lows: np.ndarray,
    highs: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Non-outside leaves below the level of a box they touch, closed bounds."""
    selected = np.zeros(len(leaves), dtype=np.bool_)
    for leaf_id in range(len(leaves)):
        x, y, z, width, level, kind = leaves[leaf_id]
        if kind == 0:
            continue
        lower_x = root_lower[0] + finest * x - tolerance
        lower_y = root_lower[1] + finest * y - tolerance
        lower_z = root_lower[2] + finest * z - tolerance
        upper_x = root_lower[0] + finest * (x + width) + tolerance
        upper_y = root_lower[1] + finest * (y + width) + tolerance
        upper_z = root_lower[2] + finest * (z + width) + tolerance
        for control in range(len(targets)):
            if (
                level < targets[control]
                and lower_x <= highs[control, 0]
                and lower_y <= highs[control, 1]
                and lower_z <= highs[control, 2]
                and upper_x >= lows[control, 0]
                and upper_y >= lows[control, 1]
                and upper_z >= lows[control, 2]
            ):
                selected[leaf_id] = True
                break
    return selected


def refine_objects(
    leaves: Sequence[Leaf] | np.ndarray,
    requests: Sequence[BoxRefinement],
    *,
    root_lower: np.ndarray,
    root_size: float,
    max_cell_size: float,
    global_level: int,
    max_level: int,
    exact_cell_sizes: bool,
    classify: LeafClassifier,
) -> np.ndarray:
    """Apply box refinement using strict cfMesh or exact lattice levels."""
    finest = root_size / (2**max_level)
    tolerance = 1.0e-15 * root_size
    level_for_size = _inclusive_additional_level if exact_cell_sizes else object_additional_level
    lows = np.asarray([request.bounds[::2] for request in requests], dtype=np.float64)
    highs = np.asarray([request.bounds[1::2] for request in requests], dtype=np.float64)
    targets = np.asarray(
        [global_level + level_for_size(max_cell_size, request.cell_size) for request in requests],
        dtype=np.int64,
    )
    records = leaf_records(leaves)
    root_lower = np.asarray(root_lower, dtype=np.float64)
    pass_number = 0
    while True:
        pass_number += 1
        selected = np.flatnonzero(
            _object_selection_kernel(
                records,
                root_lower,
                finest,
                tolerance,
                lows.reshape(-1, 3),
                highs.reshape(-1, 3),
                targets,
            )
        )
        mesh_event(
            "object refinement pass",
            pass_number=pass_number,
            leaves=len(records),
            selected=len(selected),
        )
        if not len(selected):
            return records
        records = refine_selected_leaves(records, selected, max_level, classify)


def _inclusive_additional_level(max_cell_size: float, requested: float) -> int:
    """Return the first dyadic level whose size is at most ``requested``."""
    level = 0
    while max_cell_size / (2**level) > requested * (1.0 + 1.0e-15):
        level += 1
    return level


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
    leaves: Sequence[Leaf] | np.ndarray,
    max_level: int,
    classify: LeafClassifier,
) -> np.ndarray:
    """Close 2:1 gaps with a cached, CPU-compiled neighbour scan."""
    records = leaf_records(leaves)
    pass_number = 0
    while True:
        pass_number += 1
        selected = _balance_selection_kernel(records, max_level)
        mesh_event(
            "octree balance pass",
            pass_number=pass_number,
            leaves=len(records),
            selected=len(selected),
        )
        if not len(selected):
            return records
        records = refine_selected_leaves(records, selected, max_level, classify)


@njit(cache=True, fastmath=False)
def _near_data_selection_kernel(leaves: np.ndarray, max_level: int) -> np.ndarray:
    """Non-outside same-level-or-coarser neighbours, of another level, of every DATA leaf."""
    lookup = {(row[4], row[0], row[1], row[2]): index for index, row in enumerate(leaves)}
    selected = np.zeros(len(leaves), dtype=np.bool_)
    for leaf_id in range(len(leaves)):
        x, y, z, width, level, kind = leaves[leaf_id]
        if kind != 2:
            continue
        for dz in range(-1, 2):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbour = _find_leaf(
                        lookup, x + dx * width, y + dy * width, z + dz * width, level, max_level
                    )
                    if (
                        neighbour >= 0
                        and leaves[neighbour, 5] != 0
                        and leaves[neighbour, 4] != level
                    ):
                        selected[neighbour] = True
    return selected


def refine_near_data(
    leaves: Sequence[Leaf] | np.ndarray,
    max_level: int,
    classify: LeafClassifier,
) -> tuple[np.ndarray, int]:
    """One native near-DATA layer, followed by the regularity closure."""
    records = leaf_records(leaves)
    selected = np.flatnonzero(_near_data_selection_kernel(records, max_level))
    return refine_selected_leaves(records, selected, max_level, classify), len(selected)
