"""Focused regressions for Cartesian cut-face recovery."""

from __future__ import annotations

import numpy as np

from source.solvers.fvm.mesh.cartesian.surface_recovery import _face_fluid_polygons


class _CornerSolid:
    """Classify only the four source-face corners as solid test points."""

    @staticmethod
    def is_inside(points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        return np.isclose(np.abs(values[:, 0]), 1.0) & np.isclose(np.abs(values[:, 1]), 1.0)


def test_tangent_cut_line_is_not_sent_to_delaunay() -> None:
    """A rank-one tangent arrangement has no finite-area face polygon."""
    original = np.asarray(((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)))
    points = [
        np.asarray(((-0.75, 0.0, 0.0), (-0.25, 0.0, 0.0), (0.25, 0.0, 0.0))),
        np.asarray(((-0.25, 0.0, 0.0), (0.25, 0.0, 0.0), (0.75, 0.0, 0.0))),
    ]

    polygons = _face_fluid_polygons(
        original,
        points,
        (_CornerSolid(),),  # type: ignore[arg-type]
        1.0e-12,
    )

    assert polygons == []
