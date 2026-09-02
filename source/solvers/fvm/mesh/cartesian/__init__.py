# SPDX-License-Identifier: GPL-3.0-or-later
"""Typed configuration and staged implementation of the native Cartesian mesher."""

import numpy as np

from .config import (
    BoundaryLayers,
    BoxDomain,
    BoxPatches,
    BoxRefinement,
    CompositeSizeField,
    ConeRefinement,
    FeatureRefinement,
    LineRefinement,
    SizeField,
    SphereRefinement,
    STLSurface,
)
from .mesher import CartesianMesher
from .report import GenerationReport, SizeReport


def structured_box(
    nx: int,
    ny: int,
    nz: int,
    lx: float = 1.0,
    ly: float = 1.0,
    lz: float = 1.0,
) -> dict:
    """Return the legacy test-only structured box through the native helper."""
    if min(nx, ny, nz) < 1 or min(lx, ly, lz) <= 0.0:
        raise ValueError("Cell counts and box lengths must be positive")
    from ..rectilinear import box_mesh_3d

    mesh = box_mesh_3d(
        np.linspace(0.0, lx, nx + 1),
        np.linspace(0.0, ly, ny + 1),
        np.linspace(0.0, lz, nz + 1),
    )
    names = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
    for patch, name in zip(mesh["boundary"], names, strict=True):
        patch["name"] = name
    return mesh


__all__ = [
    "BoundaryLayers",
    "BoxDomain",
    "BoxPatches",
    "BoxRefinement",
    "CartesianMesher",
    "CompositeSizeField",
    "ConeRefinement",
    "FeatureRefinement",
    "GenerationReport",
    "LineRefinement",
    "SizeField",
    "SizeReport",
    "SphereRefinement",
    "STLSurface",
    "structured_box",
]
