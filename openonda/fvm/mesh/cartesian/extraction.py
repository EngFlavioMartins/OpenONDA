# SPDX-License-Identifier: GPL-3.0-or-later
"""Typed extraction diagnostics for native face topology."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExtractionDiagnostics:
    """Counts produced while converting leaves to native faces."""

    cells: int
    faces: int
    interior_faces: int
    boundary_patches: int


def diagnostics_from_mesh(mesh_data: dict) -> ExtractionDiagnostics:
    """Return extraction counts from a native mesh dictionary."""
    return ExtractionDiagnostics(
        int(mesh_data["n_cells"]),
        int(mesh_data["n_faces"]),
        int(mesh_data["n_interior_faces"]),
        len(mesh_data["boundary"]),
    )


__all__ = ["ExtractionDiagnostics", "diagnostics_from_mesh"]
