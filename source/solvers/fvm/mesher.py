# SPDX-License-Identifier: GPL-3.0-or-later
"""Public mesher namespace for the finite-volume solver.

This module is the single import surface for mesh construction. It keeps
typed geometry, sizing, layer, report, and mesh-import objects together while
leaving :mod:`source.solvers.fvm` focused on solver configuration and runtime.
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
    FeatureRefinement,
    LineRefinement,
    SizeField,
    SphereRefinement,
    STLSurface,
    structured_box,
)
from .mesh.cartesian.report import GenerationReport, SizeReport
from .mesh.gmsh_importer import GmshImporter
from .mesh.rectilinear import (
    coupling_box_mesh,
    periodic_square_mesh,
    stretched,
    wall_refined_axis,
)

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
    "GmshImporter",
    "LineRefinement",
    "SizeField",
    "SizeReport",
    "SphereRefinement",
    "STLSurface",
    "coupling_box_mesh",
    "geometry",
    "periodic_square_mesh",
    "stretched",
    "structured_box",
    "wall_refined_axis",
]
