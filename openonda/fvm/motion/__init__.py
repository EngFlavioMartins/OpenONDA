"""Prescribed rigid-body motion of the finite-volume mesh.

The motion classes are the shared ones of :mod:`openonda.core.kinematics`;
this package holds the mesh placement and the moving-mesh runtime.
"""

from .mesh_motion import MeshMotion

__all__ = ["MeshMotion"]
