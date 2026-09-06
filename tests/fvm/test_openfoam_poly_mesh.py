"""OpenFOAM interchange canonicalizes addressing without changing geometry."""

import numpy as np
import pytest

from source.solvers.fvm.io.openfoam_poly_mesh import read_poly_mesh, write_poly_mesh


def test_import_counts_cells_that_appear_only_as_neighbours(tmp_path):
    # Minimal closed two-sided tetrahedral topology for the addressing parser:
    # cell 1 is exclusively a neighbour. Geometry is not under test here.
    mesh = {
        "vertex_position": np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]),
        "faces": np.array([(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]),
        "owners": np.zeros(4, dtype=int),
        "neighbours": np.ones(4, dtype=int),
        "n_cells": 2,
        "n_faces": 4,
        "n_interior_faces": 4,
        "n_points": 4,
        "boundary": [],
    }
    write_poly_mesh(mesh, tmp_path / "polyMesh")
    assert read_poly_mesh(tmp_path / "polyMesh")["n_cells"] == 2


@pytest.mark.parametrize("legacy_neighbours", [False, True])
def test_export_sorts_internal_faces_and_reverses_swapped_owners(tmp_path, legacy_neighbours):
    points = np.array(
        [(x, y, z) for x in range(4) for y in range(2) for z in range(2)], dtype=float
    )
    records = {}
    for cell in range(3):
        for face in (
            (0, 1, 3, 2),
            (4, 6, 7, 5),
            (0, 4, 5, 1),
            (2, 3, 7, 6),
            (0, 2, 6, 4),
            (1, 5, 7, 3),
        ):
            ids = np.array(face) + 4 * cell
            key = tuple(sorted(ids))
            if key in records:
                records[key][2] = cell
            else:
                records[key] = [ids, cell, -1]
    internal = [r for r in records.values() if r[2] >= 0]
    boundary = [r for r in records.values() if r[2] < 0]
    # Reverse global face order and swap an owner/neighbour, with its normal.
    internal.reverse()
    internal[0] = [internal[0][0][::-1], internal[0][2], internal[0][1]]
    all_faces = internal + boundary
    mesh = {
        "vertex_position": points,
        "faces": [r[0].copy() for r in all_faces],
        "owners": np.array([r[1] for r in all_faces]),
        "neighbours": np.array([r[2] for r in internal]),
        "n_cells": 3,
        "n_faces": len(all_faces),
        "n_interior_faces": 2,
        "n_points": len(points),
        "boundary": [{"name": "walls", "type": "wall", "start_face": 2, "n_faces": len(boundary)}],
    }
    original_owners = mesh["owners"].copy()
    original_faces = [f.copy() for f in mesh["faces"]]
    destination = tmp_path / "polyMesh"
    write_poly_mesh(mesh, destination)
    if legacy_neighbours:
        from source.solvers.fvm.io.openfoam_poly_mesh import _foam_list

        (destination / "neighbour").write_text(
            _foam_list(["1", "2"] + ["-1"] * len(boundary), "labelList", "neighbour")
        )
    restored = read_poly_mesh(destination)
    np.testing.assert_array_equal(restored["owners"][:2], [0, 1])
    np.testing.assert_array_equal(restored["neighbours"], [1, 2])
    np.testing.assert_array_equal(restored["vertex_position"], points)
    np.testing.assert_array_equal(mesh["owners"], original_owners)
    for before, after in zip(original_faces, mesh["faces"], strict=True):
        np.testing.assert_array_equal(before, after)
    for face in restored["faces"][:2]:
        p = points[face]
        assert np.cross(p[1] - p[0], p[2] - p[0])[0] > 0
    for before, after in zip(original_faces[2:], restored["faces"][2:], strict=True):
        np.testing.assert_array_equal(before, after)
