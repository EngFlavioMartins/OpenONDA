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


class FMMTree:
    """Build a deterministic leaf hierarchy from one supplied stage state."""

    def __init__(self, leaf_capacity: int = 32, max_depth: int = 24) -> None:
        if leaf_capacity < 1 or max_depth < 1:
            raise ValueError("FMM leaf_capacity and max_depth must be positive")
        self.leaf_capacity = int(leaf_capacity)
        self.max_depth = int(max_depth)
        self.cells: tuple[FMMCell, ...] = ()
        self.position = None
        self.vortex_strength = None
        self.core_radius = None

    def build(self, position, vortex_strength, core_radius, count: int) -> None:
        """Construct leaves and retain only stage-owned source metadata."""
        self.position = np.asarray(position.to_numpy()[:count], dtype=np.float64).copy()
        self.vortex_strength = np.asarray(vortex_strength.to_numpy()[:count], dtype=np.float64).copy()
        self.core_radius = np.asarray(core_radius.to_numpy()[:count], dtype=np.float64).copy()
        if count == 0:
            self.cells = ()
            return
        lo = self.position.min(axis=0)
        hi = self.position.max(axis=0)
        span = float(np.max(hi - lo))
        half_width = max(0.5 * span, np.finfo(float).eps)
        centre = 0.5 * (lo + hi)
        self.cells = tuple(self._split(np.arange(count, dtype=np.int64), centre, half_width, 0))

    def _split(self, indices, centre, half_width, level):
        if len(indices) <= self.leaf_capacity or level >= self.max_depth:
            max_core_radius = (
                float(np.max(self.core_radius[indices])) if len(indices) else 0.0
            )
            yield FMMCell(
                np.asarray(centre, dtype=np.float64),
                float(half_width),
                np.asarray(indices, dtype=np.int64),
                max_core_radius,
                level,
            )
            return
        half = 0.5 * half_width
        octants = ((self.position[indices] >= centre).astype(np.int64) * np.array([1, 2, 4])).sum(axis=1)
        for octant in range(8):
            selected = indices[octants == octant]
            if len(selected) == 0:
                continue
            offset = np.array(
                [1.0 if octant & 1 else -1.0, 1.0 if octant & 2 else -1.0, 1.0 if octant & 4 else -1.0]
            )
            yield from self._split(selected, centre + half * offset, half, level + 1)


__all__ = ["FMMCell", "FMMTree"]
