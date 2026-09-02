"""Deterministic octree used by the regularized vortex FMM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FMMCell:
    """One axis-aligned FMM cell and its source metadata."""

    centre: np.ndarray
    half_width: float
    indices: np.ndarray
    max_core_radius: float
    level: int


@dataclass(frozen=True, slots=True)
class FMMNode:
    """One node in the complete octree, including internal nodes."""

    centre: np.ndarray
    half_width: float
    indices: np.ndarray
    max_core_radius: float
    level: int
    children: tuple[int, ...] = ()


class FMMTree:
    """Build a deterministic octree hierarchy from one supplied stage state.

    ``cells`` remains the leaf-only view used by target-side callers.  The
    ``nodes`` and ``root`` fields expose the internal hierarchy needed by the
    upward/downward FMM passes.  All indices refer to the original particle
    ordering; no stage state is mutated by tree construction.
    """

    def __init__(self, leaf_capacity: int = 32, max_depth: int = 24) -> None:
        if leaf_capacity < 1 or max_depth < 1:
            raise ValueError("FMM leaf_capacity and max_depth must be positive")
        self.leaf_capacity = int(leaf_capacity)
        self.max_depth = int(max_depth)
        self.cells: tuple[FMMCell, ...] = ()
        self.nodes: tuple[FMMNode, ...] = ()
        self.root: int | None = None
        self.position = None
        self.vortex_strength = None
        self.core_radius = None

    def build(self, position, vortex_strength, core_radius, count: int) -> None:
        """Construct leaves and retain only stage-owned source metadata."""
        self.position = _stage_prefix(position, count, dtype=np.float64)
        self.vortex_strength = _stage_prefix(vortex_strength, count, dtype=np.float64)
        self.core_radius = _stage_prefix(core_radius, count, dtype=np.float64)
        if count == 0:
            self.cells = ()
            self.nodes = ()
            self.root = None
            return
        lo = self.position.min(axis=0)
        hi = self.position.max(axis=0)
        span = float(np.max(hi - lo))
        half_width = max(0.5 * span, np.finfo(float).eps)
        centre = 0.5 * (lo + hi)
        nodes: list[FMMNode] = []
        self.root = self._build_node(nodes, np.arange(count, dtype=np.int64), centre, half_width, 0)
        self.nodes = tuple(nodes)
        self.cells = tuple(
            FMMCell(
                node.centre,
                node.half_width,
                node.indices,
                node.max_core_radius,
                node.level,
            )
            for node in self.nodes
            if not node.children
        )

    def _build_node(self, nodes, indices, centre, half_width, level) -> int:
        node_index = len(nodes)
        # Reserve the slot before recursing so child references are stable.
        nodes.append(None)  # type: ignore[arg-type]
        max_core_radius = float(np.max(self.core_radius[indices]))
        if len(indices) <= self.leaf_capacity or level >= self.max_depth:
            nodes[node_index] = FMMNode(
                np.asarray(centre, dtype=np.float64),
                float(half_width),
                np.asarray(indices, dtype=np.int64),
                max_core_radius,
                level,
            )
            return node_index
        half = 0.5 * half_width
        octants = ((self.position[indices] >= centre).astype(np.int64) * np.array([1, 2, 4])).sum(
            axis=1
        )
        children: list[int] = []
        for octant in range(8):
            selected = indices[octants == octant]
            if len(selected) == 0:
                continue
            offset = np.array(
                [
                    1.0 if octant & 1 else -1.0,
                    1.0 if octant & 2 else -1.0,
                    1.0 if octant & 4 else -1.0,
                ]
            )
            children.append(
                self._build_node(nodes, selected, centre + half * offset, half, level + 1)
            )
        nodes[node_index] = FMMNode(
            np.asarray(centre, dtype=np.float64),
            float(half_width),
            np.asarray(indices, dtype=np.int64),
            max_core_radius,
            level,
            tuple(children),
        )
        return node_index


def _stage_prefix(values, count: int, *, dtype) -> np.ndarray:
    """Copy only the active stage prefix from a field or NumPy array."""
    raw = values.to_numpy() if hasattr(values, "to_numpy") else values
    return np.asarray(raw[:count], dtype=dtype).copy()


__all__ = ["FMMCell", "FMMNode", "FMMTree"]
