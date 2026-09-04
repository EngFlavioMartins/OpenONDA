"""Focused contracts for the cfMesh finite-volume optimization port."""

from __future__ import annotations

import numpy as np
import pytest

from source.solvers.fvm.mesh.cartesian.cfmesh_mesh_optimisation import (
    _cfmesh_bad_faces,
    _cfmesh_low_quality_faces,
    _mesh_addressing,
)
from source.solvers.fvm.mesh.cartesian.cfmesh_surface_optimisation import (
    _optimise_point,
)


def _unit_cube() -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    points = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        ]
    )
    faces = [
        np.asarray(face, dtype=np.int32)
        for face in (
            (0, 3, 2, 1),
            (4, 5, 6, 7),
            (0, 4, 7, 3),
            (1, 2, 6, 5),
            (0, 1, 5, 4),
            (3, 7, 6, 2),
        )
    ]
    return points, faces, np.zeros(6, dtype=np.int32), np.empty(0, dtype=np.int32)


def test_quality_scans_accept_an_orthogonal_unit_cube():
    points, faces, owners, neighbours = _unit_cube()

    assert not _cfmesh_bad_faces(points, faces, owners, neighbours, 1)
    assert not _cfmesh_low_quality_faces(points, faces, owners, neighbours, 1)


def test_bad_face_scan_detects_an_inward_boundary_face():
    points, faces, owners, neighbours = _unit_cube()
    faces[1] = faces[1][::-1].copy()

    assert 1 in _cfmesh_bad_faces(points, faces, owners, neighbours, 1)


def test_mesh_addressing_preserves_and_validates_native_cell_face_order():
    points, faces, owners, neighbours = _unit_cube()
    native_order = [[5, 3, 1, 4, 2, 0]]

    cell_faces, point_cells = _mesh_addressing(
        faces,
        owners,
        neighbours,
        1,
        len(points),
        cell_face_order=native_order,
    )

    assert cell_faces == native_order
    assert all(cells == {0} for cells in point_cells)
    with pytest.raises(ValueError, match="inconsistent with mesh topology"):
        _mesh_addressing(
            faces,
            owners,
            neighbours,
            1,
            len(points),
            cell_face_order=[[0, 1, 2, 3, 4]],
        )


def test_surface_optimizer_uses_cfmesh_branch_for_symmetric_simplex():
    points = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (0.175925104244658, 8.18286074045122e-18, 0.0),
            (0.129393069484928, 0.0928852466733864, 0.0),
            (0.224662907280417, 0.269029853901011, 0.0),
            (0.175925104244681, 0.18430756870769, 0.0),
            (0.224662907280378, -0.269029853901026, 0.0),
            (0.129393069484909, -0.0928852466733874, 0.0),
            (0.175925104244635, -0.18430756870769, 0.0),
            (-0.152240294632501, 8.55172432869605e-15, 0.0),
            (-0.906319307392956, 3.91363527458225e-14, 0.0),
        ]
    )
    triangles = np.asarray(
        [
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 3),
            (0, 5, 6),
            (0, 5, 1),
            (0, 6, 1),
            (0, 3, 8),
            (0, 3, 5),
            (0, 8, 5),
        ],
        dtype=np.int32,
    )

    result = _optimise_point(points, triangles)

    assert result == pytest.approx((0.223926768214181, -0.0283742424036304, 0.0), abs=1.0e-12)
