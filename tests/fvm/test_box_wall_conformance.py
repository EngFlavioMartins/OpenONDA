"""Box walls must preserve whole faces, including shared edges and corners."""

from pathlib import Path

import numpy as np
import pytest

import openonda.fvm.mesher as msh
from source.solvers.fvm.mesh.cartesian.mesher import _box_patch_point_constraints
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.rectilinear import box_mesh_3d


def _wall_faces(mesh):
    patch = next(p for p in mesh["boundary"] if p["name"] == "cube")
    return mesh["faces"][patch["start_face"] : patch["start_face"] + patch["n_faces"]]


@pytest.mark.parametrize("extent", [(1.0, 1.0, 1.0), (1.0, 1.5, 0.5)])
def test_incident_planes_repair_faces_whose_vertices_are_already_on_the_box(extent):
    axis = np.linspace(-1, 1, 9)
    bounds = np.array([[-size / 2, size / 2] for size in extent]).ravel()
    mesh = box_mesh_3d(axis, axis, axis, hole_box=bounds, wall_patch_name="cube")
    faces = _wall_faces(mesh)
    wall_ids = np.unique(np.concatenate(faces))
    points = mesh["vertex_position"].copy()
    # Reproduce the defect: move all edge/corner coordinates inward, then
    # put each vertex on just its nearest plane. Vertex conformance is exact;
    # polygons crossing different planes still cut through the solid.
    points[wall_ids] *= 0.98
    lower, upper = bounds[::2], bounds[1::2]
    for point_id in wall_ids:
        distance = np.abs(points[point_id, np.repeat(np.arange(3), 2)] - bounds)
        side = int(np.argmin(distance))
        points[point_id, side // 2] = bounds[side]
    np.testing.assert_allclose(
        np.min(
            np.minimum(np.abs(points[wall_ids] - lower), np.abs(points[wall_ids] - upper)), axis=1
        ),
        0,
        rtol=0,
        atol=0,
    )
    assert any(
        np.all(np.mean(points[face], axis=0) > lower)
        and np.all(np.mean(points[face], axis=0) < upper)
        for face in faces
    )

    constraints = _box_patch_point_constraints(points, faces, bounds)
    for point_id, planes in constraints.items():
        for axis_, bound in planes.items():
            points[point_id, axis_] = bound
    mesh["vertex_position"] = points
    geo = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    patch = next(p for p in mesh["boundary"] if p["name"] == "cube")
    rows = slice(patch["start_face"], patch["start_face"] + patch["n_faces"])
    expected_area = 2 * (extent[0] * extent[1] + extent[1] * extent[2] + extent[2] * extent[0])
    np.testing.assert_allclose(geo["face_area"][rows].sum(), expected_area, rtol=0, atol=2e-14)
    solid_volume = (
        -np.einsum("ij,ij->i", geo["face_centre"][rows], geo["face_area_vector"][rows]).sum() / 3
    )
    np.testing.assert_allclose(solid_volume, np.prod(extent), rtol=0, atol=2e-14)
    for face in faces:
        assert any(np.all(points[face, side // 2] == bound) for side, bound in enumerate(bounds))


@pytest.mark.slow
def test_native_cube_mesher_preserves_face_area_and_enclosed_volume():
    case = Path(__file__).resolve().parents[2] / "tutorials/coupled_fvm_vpm/02_cube_flow"
    mesh = msh.CartesianMesher(
        domain=msh.BoxDomain(
            bounds=(-2, 2) * 3,
            patches=msh.BoxPatches("inlet", "outlet", "ymin", "ymax", "zmin", "zmax"),
        ),
        surfaces=(msh.STLSurface(case / "assets/cube.stl", patch="cube"),),
        max_cell_size=0.5,
        patch_refinements=(msh.PatchRefinement("cube", 0.25),),
    ).build()
    geo = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    patch = next(p for p in mesh["boundary"] if p["name"] == "cube")
    rows = slice(patch["start_face"], patch["start_face"] + patch["n_faces"])
    np.testing.assert_allclose(geo["face_area"][rows].sum(), 6, rtol=0, atol=2e-13)
    volume = (
        -np.einsum("ij,ij->i", geo["face_centre"][rows], geo["face_area_vector"][rows]).sum() / 3
    )
    np.testing.assert_allclose(volume, 1, rtol=0, atol=2e-13)
    for face in _wall_faces(mesh):
        assert any(
            np.all(mesh["vertex_position"][face, side // 2] == bound)
            for side, bound in enumerate((-0.5, 0.5) * 3)
        )
