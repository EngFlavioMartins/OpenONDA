"""Public mesh-construction namespace for :mod:`openonda.fvm`.

Use this module for all meshing objects and helpers::

    import openonda.fvm.mesher as msh

    mesh = msh.CartesianMesher(
        domain=msh.BoxDomain(...),
        surfaces=(msh.STLSurface(...),),
    )

The solver facade intentionally does not re-export these names at
``openonda.fvm.BoxRefinement`` or similar flat paths.
"""

from source.solvers.fvm.mesher import (
    BoundaryLayers,
    BoxDomain,
    BoxPatches,
    BoxRefinement,
    CartesianMesher,
    CompositeSizeField,
    ConeRefinement,
    FeatureRefinement,
    GenerationReport,
    GmshImporter,
    LineRefinement,
    SizeField,
    SizeReport,
    SphereRefinement,
    STLSurface,
    coupling_box_mesh,
    geometry,
    periodic_square_mesh,
    stretched,
    structured_box,
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
