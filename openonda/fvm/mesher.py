# SPDX-License-Identifier: GPL-3.0-or-later
"""Public mesher namespace for the finite-volume solver.

This module is the single import surface for mesh construction. It keeps
typed geometry, sizing, and mesh-import objects together while
leaving :mod:`openonda.fvm` focused on solver configuration and runtime.
"""

from .mesh.cartesian import (
    BoxDomain,
    BoxPatches,
    BoxRefinement,
    CartesianMesher,
    ExtrudedCartesianMesher,
    PatchRefinement,
    STLSurface,
)
from .mesh.rectilinear import (
    box_mesh_3d,
    coupling_box_mesh,
)

__all__ = [
    "BoxDomain",
    "BoxPatches",
    "BoxRefinement",
    "CartesianMesher",
    "ExtrudedCartesianMesher",
    "PatchRefinement",
    "STLSurface",
    "box_mesh_3d",
    "coupling_box_mesh",
]
