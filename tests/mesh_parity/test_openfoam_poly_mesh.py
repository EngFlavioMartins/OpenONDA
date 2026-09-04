"""ASCII polyMesh reader/writer contracts."""

from __future__ import annotations

import numpy as np

from tools.mesh_parity.openfoam_poly_mesh import read_poly_mesh, write_poly_mesh

from ._fixtures import two_cell_mesh


def test_ascii_poly_mesh_round_trip_preserves_general_face_topology(tmp_path):
    mesh = two_cell_mesh()
    path = write_poly_mesh(mesh, tmp_path / "constant" / "polyMesh")

    read = read_poly_mesh(path)

    np.testing.assert_array_equal(read.points, mesh.points)
    np.testing.assert_array_equal(read.owner, mesh.owner)
    np.testing.assert_array_equal(read.neighbour, mesh.neighbour)
    assert [face.tolist() for face in read.faces] == [face.tolist() for face in mesh.faces]
    assert read.boundary == mesh.boundary
    assert read.n_cells == mesh.n_cells


def test_reader_normalizes_cfmesh_full_neighbour_list(tmp_path):
    mesh = two_cell_mesh()
    path = write_poly_mesh(mesh, tmp_path / "constant" / "polyMesh")
    values = [*(str(int(value)) for value in mesh.neighbour)]
    values.extend("-1" for _ in range(mesh.n_boundary_faces))
    (path / "neighbour").write_text(
        f"{mesh.n_faces}\n(\n" + "\n".join(values) + "\n)\n",
        encoding="ascii",
    )

    read = read_poly_mesh(path)

    np.testing.assert_array_equal(read.neighbour, mesh.neighbour)
    assert read.n_internal_faces == mesh.n_internal_faces
