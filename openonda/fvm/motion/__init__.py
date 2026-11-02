"""Rigid-body motion of the finite-volume mesh.

The prescribed motion classes are the shared ones of
:mod:`openonda.core.kinematics`; this package holds the mesh placement, the
elastically mounted body driven by the fluid force, and the moving-mesh
runtime.
"""

from .elastic import ElasticMount
from .mesh_motion import MeshMotion

__all__ = ["ElasticMount", "MeshMotion"]
