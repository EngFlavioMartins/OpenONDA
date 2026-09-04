"""Small numbering-variant polyhedral meshes used by parity-tool tests."""

from __future__ import annotations

import numpy as np

from tools.mesh_parity.openfoam_poly_mesh import BoundaryPatch, PolyMesh


def two_cell_mesh(*, renumber: bool = False) -> PolyMesh:
    """Return two joined unit hexahedra, optionally with point/cell IDs permuted."""
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (2.0, 0.0, 1.0),
            (2.0, 1.0, 1.0),
        )
    )
    faces = (
        (1, 2, 6, 5),  # internal x=1 face
        (0, 4, 7, 3),
        (8, 9, 11, 10),
        (0, 1, 5, 4),
        (1, 8, 10, 5),
        (3, 7, 6, 2),
        (2, 6, 11, 9),
        (0, 3, 2, 1),
        (1, 2, 9, 8),
        (4, 5, 6, 7),
        (5, 10, 11, 6),
    )
    owner = np.asarray((0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1), dtype=np.int64)
    neighbour = np.asarray((1,), dtype=np.int64)
    boundary = (
        BoundaryPatch("inlet", "patch", 1, 1),
        BoundaryPatch("outlet", "patch", 2, 1),
        BoundaryPatch("ymin", "patch", 3, 2),
        BoundaryPatch("ymax", "patch", 5, 2),
        BoundaryPatch("zmin", "patch", 7, 2),
        BoundaryPatch("zmax", "patch", 9, 2),
    )
    if renumber:
        new_to_old = np.asarray((8, 2, 7, 10, 5, 0, 11, 3, 1, 9, 4, 6), dtype=np.int64)
        old_to_new = np.empty(len(new_to_old), dtype=np.int64)
        old_to_new[new_to_old] = np.arange(len(new_to_old), dtype=np.int64)
        points = points[new_to_old]
        faces = tuple(tuple(old_to_new[np.asarray(face, dtype=np.int64)]) for face in faces)
        owner = 1 - owner
        neighbour = 1 - neighbour
    return PolyMesh(
        points=points,
        faces=tuple(np.asarray(face, dtype=np.int64) for face in faces),
        owner=owner,
        neighbour=neighbour,
        boundary=boundary,
        n_cells=2,
    )
