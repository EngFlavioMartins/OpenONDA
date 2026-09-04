"""Differential topology checks for OpenONDA and cfMesh polyhedral meshes.

The package is deliberately development-only: production meshing never imports
or invokes cfMesh.  See :mod:`tools.mesh_parity.parity_report` for the command
line entry point.
"""

from .compare_meshes import ComparisonResult, compare_meshes
from .mesh_fingerprint import MeshFingerprint, fingerprint_mesh
from .openfoam_poly_mesh import BoundaryPatch, PolyMesh, read_poly_mesh, write_poly_mesh

__all__ = [
    "BoundaryPatch",
    "ComparisonResult",
    "MeshFingerprint",
    "PolyMesh",
    "compare_meshes",
    "fingerprint_mesh",
    "read_poly_mesh",
    "write_poly_mesh",
]
