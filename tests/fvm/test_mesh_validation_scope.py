"""Geometry that cannot change need not be revalidated during wall movement."""

from __future__ import annotations

import numpy as np
import pytest

from source.solvers.fvm.io.vtk_exporter import VTKExporter
from source.solvers.fvm.mesh.cartesian import structured_box
from source.solvers.fvm.mesh.topology import build_cell_face_csr
from source.solvers.fvm.mesh.validation import (
    cells_incident_to_points,
    extract_cell_subset_mesh,
    validate_vtk_cell_intersections,
)


@pytest.mark.parametrize("explicit_reversal", [False, True])
def test_compiled_hex_recovery_matches_general_vtk_connectivity(explicit_reversal):
    mesh = structured_box(12, 10, 9)
    mesh.pop("cell_vertex_indices")
    if explicit_reversal:
        face_ids, offsets = build_cell_face_csr(
            mesh["owners"], mesh["neighbours"], mesh["n_cells"], mesh["n_faces"]
        )
        cell_ids = np.repeat(np.arange(mesh["n_cells"]), np.diff(offsets))
        reversed_faces = np.zeros(len(face_ids), dtype=np.bool_)
        internal = face_ids < mesh["n_interior_faces"]
        reversed_faces[internal] = mesh["neighbours"][face_ids[internal]] == cell_ids[internal]
        mesh["cell_face_indices"] = face_ids
        mesh["cell_face_offset"] = offsets
        mesh["cell_face_reversed"] = reversed_faces

    compiled = VTKExporter(mesh)._grid
    general_mesh = {**mesh}
    general_mesh.pop("cell_type_code")
    general = VTKExporter(general_mesh)._grid

    np.testing.assert_array_equal(compiled.celltypes, general.celltypes)
    np.testing.assert_array_equal(compiled.cells, general.cells)


def test_point_incident_subset_detects_same_intersections_as_full_mesh():
    mesh = structured_box(3, 3, 3, 3.0, 3.0, 3.0)
    point = int(np.flatnonzero(np.all(mesh["vertex_position"] == (1.0, 1.0, 1.0), axis=1))[0])
    affected = cells_incident_to_points(mesh, [point])
    assert len(affected) == 8
    subset, source_point_ids = extract_cell_subset_mesh(mesh, affected, return_point_ids=True)

    for movement in (0.0, 1.1):
        trial = np.asarray(mesh["vertex_position"]).copy()
        trial[point, 0] = 1.0 + movement
        mesh["vertex_position"] = trial
        subset["vertex_position"] = trial[source_point_ids]
        whole = validate_vtk_cell_intersections(
            VTKExporter(mesh)._grid, maximum_intersections=mesh["n_cells"]
        )
        local = validate_vtk_cell_intersections(
            VTKExporter(subset)._grid, maximum_intersections=subset["n_cells"]
        )
        assert whole["intersecting_cells"] == local["intersecting_cells"]
        assert whole["intersecting_cells"] == (0 if movement == 0.0 else 4)
