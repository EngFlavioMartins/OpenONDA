# SPDX-License-Identifier: GPL-3.0-or-later
"""Public mesher namespace for the finite-volume solver.

This module is the single import surface for mesh construction. It keeps
typed geometry, sizing, layer, report, and mesh-import objects together while
leaving :mod:`openonda.fvm` focused on solver configuration and runtime.
"""

from .mesh import geometry
from .mesh.cartesian import (
    BoundaryLayers,
    BoxDomain,
    BoxPatches,
    BoxRefinement,
    CartesianMesher,
    CompositeSizeField,
    ConeRefinement,
    ExtrudedCartesianMesher,
    FeatureRefinement,
    LineRefinement,
    PatchRefinement,
    SizeField,
    SphereRefinement,
    STLSurface,
)
from .mesh.cartesian.report import GenerationReport, SizeReport
from .mesh.gmsh_importer import GmshImporter
from .mesh.rectilinear import (
    box_mesh_3d,
    coupling_box_mesh,
    stretched,
    wall_refined_axis,
)

__all__ = [
    "BoundaryLayers",
    "BoxDomain",
    "BoxPatches",
    "BoxRefinement",
    "CartesianMesher",
    "ExtrudedCartesianMesher",
    "CompositeSizeField",
    "ConeRefinement",
    "FeatureRefinement",
    "GenerationReport",
    "GmshImporter",
    "LineRefinement",
    "PatchRefinement",
    "SizeField",
    "SizeReport",
    "SphereRefinement",
    "STLSurface",
    "box_mesh_3d",
    "coupling_box_mesh",
    "geometry",
    "stretched",
    "wall_refined_axis",
]
