# SPDX-License-Identifier: GPL-3.0-or-later
"""Typed octree stage result used by the Cartesian orchestration layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class OctreeResult:
    """Immutable leaf coordinates, widths, and requested local sizes."""

    leaves: np.ndarray
    levels: np.ndarray
    cell_sizes: np.ndarray

    def __post_init__(self) -> None:
        for name in ("leaves", "levels", "cell_sizes"):
            values = np.ascontiguousarray(getattr(self, name))
            values.setflags(write=False)
            object.__setattr__(self, name, values)


def result_from_mesh(mesh_data: dict) -> OctreeResult:
    """Adapt the existing Cartesian leaf metadata into a typed result."""
    return OctreeResult(
        np.asarray(mesh_data.get("cell_vertex_indices", ())),
        np.asarray(mesh_data.get("cell_levels", ())),
        np.asarray(mesh_data.get("cell_sizes", ())),
    )


__all__ = ["OctreeResult", "result_from_mesh"]
