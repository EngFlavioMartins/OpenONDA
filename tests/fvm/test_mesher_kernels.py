"""Geometric boundary cases for the mesher's compiled hot paths."""

import numpy as np

from source.solvers.fvm.mesh.cartesian.cfmesh_template import _expand_lattice_faces
from source.solvers.fvm.mesh.surface_classification import triangle_box_overlap


def test_triangle_box_overlap_keeps_contacts_and_rejects_separating_planes():
    triangles = np.asarray(
        [
            # Inside, crossing with all vertices outside, and face/edge/vertex contact.
            [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]],
            [[-2, 0, 0], [2, 0, 0], [0, 2, 0]],
            [[1, 0, 0], [1, 2, 0], [1, 0, 2]],
            [[1, 1, -2], [1, 1, 2], [2, 2, 0]],
            [[1, 1, 1], [2, 1, 1], [1, 2, 1]],
            # AABB separation and diagonal separation despite overlapping AABBs.
            [[1.000001, 0, 0], [2, 0, 0], [2, 1, 0]],
            [[0.9, 1.2, 0], [1.2, 0.9, 0], [1.2, 1.2, 0]],
            # Degenerate triangles preserve point/line overlap semantics.
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[-2, 0, 0], [2, 0, 0], [2, 0, 0]],
            [[2, 0, 0], [2, 0, 0], [2, 0, 0]],
        ],
        dtype=np.float64,
    )
    expected = [True, True, True, True, True, False, False, True, True, False]
    for shift in (np.zeros(3), np.asarray([3.25, -4.5, 2.0])):
        shifted = triangles + shift
        np.testing.assert_array_equal(
            triangle_box_overlap(shift, np.ones(3), shifted[:, 0], shifted[:, 1], shifted[:, 2]),
            expected,
        )
    empty = np.empty((0, 3))
    assert triangle_box_overlap(np.zeros(3), np.ones(3), empty, empty, empty).shape == (0,)


def test_lattice_face_expansion_keeps_hanging_nodes_in_cyclic_order():
    # A coarse 2x2 square and its two fine neighbours. Only existing lattice
    # points belong in the output: the coarse edge gets its hanging midpoint.
    faces = np.asarray([[0, 2, 12, 10], [2, 3, 8, 7], [7, 8, 13, 12]], dtype=np.int64)
    nodes, offsets, expanded = _expand_lattice_faces(faces, 5, 5)
    assert expanded == 1
    np.testing.assert_array_equal(offsets, [0, 5, 9, 13])
    np.testing.assert_array_equal(nodes[:5], [0, 2, 7, 12, 10])
    np.testing.assert_array_equal(nodes[5:], faces[1:].ravel())
