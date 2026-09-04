"""Bounded regression for the migrated native cylinder reference mesh."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import openonda.fvm.mesher as msh
from source.solvers.fvm.mesh.validation import (
    MeshValidationError,
    validate_no_fluid_cell_centres_inside_surface,
    validate_vtk_cell_intersections,
    validate_wall_vertex_conformance,
)
from tutorials.coupled_fvm_vpm.cylinder_shedding_flow.reference_flow import setup


def test_cylinder_reference_builds_smooth_conformal_wall_cells(tmp_path):
    dx = 0.15
    surface = msh.STLSurface(Path(setup.CYLINDER_STL), patch="cylinder")
    mesher = msh.CartesianMesher(
        domain=msh.BoxDomain(
            bounds=(-1.5, 1.5, -1.5, 1.5, -0.6, 0.6),
            patches=msh.BoxPatches("inlet", "outlet", "ymin", "ymax", "zmin", "zmax"),
        ),
        surfaces=(surface,),
        max_cell_size=0.3,
        boundary_cell_size=dx,
        min_cell_size=dx,
        boundary_layers=(),
        surface_may_cross_domain_boundary=True,
    )
    mesh = mesher.build()
    layer_index = np.asarray(mesh["boundary_layer_index"])
    assert set(layer_index) == {-1, 0}
    assert np.count_nonzero(layer_index == 0) == 128
    assert "layer_termination" not in {patch["name"] for patch in mesh["boundary"]}
    assert (
        validate_wall_vertex_conformance(mesh, surface.triangles, "cylinder")["max_vertex_distance"]
        < 1.0e-10
    )
    assert mesher.report is not None
    quality = mesher.report.diagnostics["quality"]
    assert quality["max_non_orthogonality_deg"] < 85.0, quality
    assert quality["inverted_owner_face_pyramids"] == 0, quality
    assert quality["inverted_neighbour_face_pyramids"] == 0, quality
    assert quality["min_owner_face_pyramid_cosine"] > 0.0, quality
    assert quality["min_neighbour_face_pyramid_cosine"] > 0.0, quality
    assert quality["max_adjacent_cell_size_ratio"] <= 2.0, quality
    context = quality["max_skewness_context"]
    assert quality["max_skewness"] < 1.0, (
        context["face_points"],
        context["owner_centre"],
        context["neighbour_centre"],
    )

    # The mapped wall and core-interface faces must form cfMesh's smooth,
    # body-normal wrapper; a snapped staircase cannot satisfy these checks.
    from source.solvers.fvm.mesh.geometry import compute_mesh_geometry

    geometry = compute_mesh_geometry(mesh, compute_lsq=False)
    n_internal = mesh["n_interior_faces"]
    owners = np.asarray(mesh["owners"])
    neighbours = np.asarray(mesh["neighbours"])
    interface_faces = np.flatnonzero(
        (layer_index[owners[:n_internal]] < 0) & (layer_index[neighbours] == 0)
    )
    wall_patch = next(patch for patch in mesh["boundary"] if patch["name"] == "cylinder")
    wall_faces = np.arange(
        wall_patch["start_face"],
        wall_patch["start_face"] + wall_patch["n_faces"],
    )
    centres = geometry["face_centre"][wall_faces]
    radial = centres.copy()
    radial[:, 2] = 0.0
    radial /= np.linalg.norm(radial, axis=1)[:, None]
    normals = geometry["face_area_vector"][wall_faces]
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    assert np.min(np.abs(np.einsum("ij,ij->i", normals, radial))) > 0.98
    wall_owners = owners[wall_faces]
    assert np.max(np.asarray(mesh["cell_sizes"])[wall_owners]) <= dx * (1.0 + 1.0e-12)
    interface_by_cell = {int(neighbours[face]): int(face) for face in interface_faces}
    wall_by_cell = {int(owners[face]): int(face) for face in wall_faces}
    assert set(interface_by_cell) == set(wall_by_cell)
    alignment = []
    wall_normal_path = []
    for cell, wall_face in wall_by_cell.items():
        interface_face = interface_by_cell[cell]
        wall_normal = geometry["face_area_vector"][wall_face] / geometry["face_area"][wall_face]
        interface_normal = (
            geometry["face_area_vector"][interface_face] / geometry["face_area"][interface_face]
        )
        alignment.append(abs(float(np.dot(wall_normal, interface_normal))))
        path = geometry["face_centre"][wall_face] - geometry["face_centre"][interface_face]
        wall_normal_path.append(abs(float(np.dot(path, wall_normal))) / np.linalg.norm(path))
    assert min(alignment) > 0.97
    assert min(wall_normal_path) > 0.98

    import pyvista as pv

    from source.solvers.fvm.io.vtk_exporter import VTKExporter

    exporter = VTKExporter(mesh)
    assert validate_vtk_cell_intersections(exporter._grid)["intersecting_cells"] == 0
    output = tmp_path / "mesh.vtu"
    exporter.export(str(output), {})
    written = pv.read(output)
    assert written.n_cells == mesh["n_cells"]
    assert validate_vtk_cell_intersections(written)["intersecting_cells"] == 0


def test_fluid_centres_inside_surface_are_a_hard_failure():
    surface = msh.STLSurface(Path(setup.CYLINDER_STL), patch="cylinder")
    with pytest.raises(MeshValidationError, match="inside the input surface"):
        validate_no_fluid_cell_centres_inside_surface(
            np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))),
            surface.triangles,
        )
