"""A fluid template cell beside finer surface-data leaves gets one face per leaf.

cfMesh refines one layer of leaves around the surface-data leaves of a finer
box. A coarse data leaf split by that layer can leave its finer data children
against a coarse fluid leaf, as next to the trailing-edge box of the
03_naca4412 coupled mesh. cfMesh then creates a boundary face from each finer
data leaf towards the fluid leaf; the native template must cover that side with
the same faces, or the template stops on "cfMesh cell-face ordering references
an absent octree face".
"""

import numpy as np
import pytest

from openonda.fvm.mesh.cartesian.cfmesh_template import _extract_mesh

MAX_LEVEL = 3  # finest lattice 8 x 8 x 8
LEVEL = 2  # leaves of width 2: a 4 x 4 x 4 block of the root cube
SPLIT = (2, 2, 2)  # lattice origin of the leaf split into finer data leaves


def _octree(fluid_children):
    """Morton-ordered leaves ``(x, y, z, width, level, kind)`` of the 8-cube.

    The outer shell of width-2 leaves carries surface data, as the leaves at a
    domain side do; the other width-2 leaves are fluid, except the one at
    ``SPLIT``, whose eight width-1 children are surface data unless their
    offset is in ``fluid_children``.
    """
    leaves = []

    def visit(x, y, z, width, level):
        if level < LEVEL or (level == LEVEL and (x, y, z) == SPLIT):
            child = width // 2
            for dz in (0, child):
                for dy in (0, child):
                    for dx in (0, child):
                        visit(x + dx, y + dy, z + dz, child, level + 1)
            return
        if level > LEVEL:
            offset = (x - SPLIT[0], y - SPLIT[1], z - SPLIT[2])
            kind = 1 if offset in fluid_children else 2
        else:
            shell = min(x, y, z) == 0 or max(x, y, z) == 6
            kind = 2 if shell else 1
        leaves.append((x, y, z, width, level, kind))

    visit(0, 0, 0, 8, 0)
    return np.asarray(leaves, dtype=np.int32)


def _cell_faces(mesh, cell):
    owners = np.asarray(mesh["owners"])
    neighbours = np.asarray(mesh["neighbours"])
    return sorted(set(np.flatnonzero(owners == cell)) | set(np.flatnonzero(neighbours == cell)))


def _area_vector(points):
    centre = points.mean(axis=0)
    return 0.5 * sum(
        np.cross(points[k] - centre, points[(k + 1) % len(points)] - centre)
        for k in range(len(points))
    )


@pytest.mark.parametrize(
    "fluid_children",
    [(), ((1, 0, 0), (1, 0, 1))],
    ids=["data-side", "mixed-side"],
)
def test_fluid_cell_faces_each_finer_data_leaf(fluid_children):
    octree = _octree(fluid_children)
    leaves = octree[octree[:, 5] == 1, :5]
    mesh = _extract_mesh((0.0, 8.0, 0.0, 8.0, 0.0, 8.0), 8.0, leaves, octree, MAX_LEVEL, 2, 2)

    points = np.asarray(mesh["vertex_position"])
    faces = [np.asarray(face) for face in mesh["faces"]]
    owners = np.asarray(mesh["owners"])
    n_interior = int(mesh["n_interior_faces"])
    # Every cell is closed and lists exactly its own faces, in cfMesh's order
    closure = np.zeros((len(leaves), 3))
    for face_id, face in enumerate(faces):
        area = _area_vector(points[face])
        closure[owners[face_id]] += area
        if face_id < n_interior:
            closure[mesh["neighbours"][face_id]] -= area
    assert np.abs(closure).max() < 1.0e-12
    for cell in range(len(leaves)):
        assert sorted(mesh["_cfmesh_cell_face_order"][cell]) == _cell_faces(mesh, cell)

    # The +x neighbour of the split leaf: its lower x side is covered by one face per finer
    # leaf across, a boundary face for a data leaf and an interior face for a fluid one
    cell = int(np.flatnonzero((leaves[:, :3] == (4, 2, 2)).all(axis=1))[0])
    lower_x = [
        face_id
        for face_id in _cell_faces(mesh, cell)
        if np.allclose(points[faces[face_id], 0], 4.0)
    ]
    assert len(lower_x) == 4
    boundary = [face_id for face_id in lower_x if face_id >= n_interior]
    assert len(boundary) == 4 - sum(offset[0] == 1 for offset in fluid_children)
    for face_id in lower_x:
        area = _area_vector(points[faces[face_id]])
        assert np.isclose(np.linalg.norm(area), 1.0)
        assert (area[0] < 0.0) == (owners[face_id] == cell)
