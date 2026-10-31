"""Coupling boxes: exact Cartesian outer sides from the Cartesian mesher.

The FVM–VPM coupling closes the outer FVM boundary with particle data. Its
boundary closure and the particle-mesh transfer need planar sides and
undistorted, axis-aligned cells at faces, edges and corners. The 03_naca4412
case went unstable in thin, skewed corner cells that the cfMesh-style wrapper
had produced at the box edges.
"""

from pathlib import Path

import numpy as np
import pytest

from openonda.fvm.mesh.geometry import compute_mesh_geometry
import openonda.fvm.mesher as msh

H = 0.1


def _box_stl(path: Path, lower, upper) -> Path:
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    vertices = np.array(
        [
            [x, y, z]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ]
    )
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    lines = ["solid box"]
    for a, b, c, d in quads:
        for triangle in ((a, b, c), (a, c, d)):
            p = vertices[list(triangle)]
            n = np.cross(p[1] - p[0], p[2] - p[0])
            n /= np.linalg.norm(n)
            lines += [f" facet normal {n[0]} {n[1]} {n[2]}", "  outer loop"]
            lines += [f"   vertex {q[0]} {q[1]} {q[2]}" for q in p]
            lines += ["  endloop", " endfacet"]
    path.write_text("\n".join([*lines, "endsolid box"]) + "\n")
    return path


def _outer_patch_geometry(mesh, name):
    geometry = compute_mesh_geometry(mesh, compute_lsq=False)
    patch = next(p for p in mesh["boundary"] if p["name"] == name)
    faces = np.arange(patch["start_face"], patch["start_face"] + patch["n_faces"])
    area_vector = geometry["face_area_vector"][faces]
    normal = area_vector / np.linalg.norm(area_vector, axis=1)[:, None]
    offset = geometry["cell_connection_vector"][faces]
    distance = np.einsum("ij,ij->i", offset, normal)
    tangential = np.linalg.norm(offset - distance[:, None] * normal, axis=1)
    return geometry["face_centre"][faces], normal, distance, tangential


@pytest.fixture(scope="module")
def coupled_cube_mesh(tmp_path_factory):
    stl = _box_stl(tmp_path_factory.mktemp("cube") / "cube.stl", (-0.1,) * 3, (0.1,) * 3)
    mesher = msh.CartesianMesher(
        domain=msh.BoxDomain(
            bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
            patches=msh.BoxPatches.coupled("numericalBoundary"),
        ),
        surfaces=(msh.STLSurface(stl, patch="cube"),),
        max_cell_size=H,
    )
    return mesher, mesher.build()


def test_coupled_box_sides_are_exact_planes_with_box_cells(coupled_cube_mesh):
    mesher, mesh = coupled_cube_mesh
    assert mesher.cell_size_anchor == H  # coupling boxes anchor the lattice
    centre, normal, distance, tangential = _outer_patch_geometry(mesh, "numericalBoundary")
    bounds = np.asarray(mesher.domain.bounds)
    # Every outer face lies on a box plane, with the outward axis normal.
    axis = np.argmax(np.abs(normal), axis=1)
    np.testing.assert_allclose(np.abs(normal[np.arange(len(axis)), axis]), 1.0, atol=1e-14)
    plane = bounds[2 * axis + (normal[np.arange(len(axis)), axis] > 0)]
    np.testing.assert_allclose(centre[np.arange(len(axis)), axis], plane, atol=1e-12)
    # Owner cells are the undistorted lattice cubes, also at edges and corners.
    np.testing.assert_allclose(distance, 0.5 * H, rtol=1e-12)
    assert np.max(tangential / distance) < 1e-10
    report = mesh["mesh_generation"]["coupling_box"]
    assert report["exact_sides"] == [0, 1, 2, 3, 4, 5]
    for side in report["validation"]["sides"].values():
        assert side["non_box_first_layer_cells"] == 0
        assert side["non_box_second_layer_cells"] == 0
        assert side["area_fraction"] == pytest.approx(1.0, abs=1e-12)


def test_coupled_box_keeps_the_requested_domain(coupled_cube_mesh):
    mesher, mesh = coupled_cube_mesh
    points = np.asarray(mesh["vertex_position"])
    np.testing.assert_allclose(points.min(axis=0), mesher.domain.bounds[::2], atol=1e-12)
    np.testing.assert_allclose(points.max(axis=0), mesher.domain.bounds[1::2], atol=1e-12)
    assert {p["name"] for p in mesh["boundary"]} == {"cube", "numericalBoundary"}


def test_coupling_sides_next_to_a_crossed_side_are_exact(tmp_path):
    # A rod through the z faces, as in the extruded cylinder cases: z sides keep
    # their surface-conforming cells, the x and y coupling sides are exact.
    stl = _box_stl(tmp_path / "rod.stl", (-0.1, -0.1, -2.0), (0.1, 0.1, 2.0))
    patches = msh.BoxPatches(
        "numericalBoundary",
        "numericalBoundary",
        "numericalBoundary",
        "numericalBoundary",
        "zmin",
        "zmax",
        coupling=("numericalBoundary",),
    )
    mesher = msh.CartesianMesher(
        domain=msh.BoxDomain(bounds=(-1.0, 1.0, -1.0, 1.0, -0.8, 0.8), patches=patches),
        surfaces=(msh.STLSurface(stl, patch="rod"),),
        max_cell_size=H,
        cell_size_anchor=H,
        surface_may_cross_domain_boundary=True,
    )
    assert mesher.exact_sides == (0, 1, 2, 3)
    assert mesher.crossed_sides == (4, 5)
    mesh = mesher.build()
    centre, normal, distance, tangential = _outer_patch_geometry(mesh, "numericalBoundary")
    junction = np.abs(np.abs(centre[:, 2]) - 0.8) < 4.0 * H
    assert np.max(tangential[~junction] / distance[~junction]) < 1e-10
    np.testing.assert_allclose(distance[~junction], 0.5 * H, rtol=1e-12)


def test_boxes_without_coupling_patches_keep_the_cfmesh_workflow(tmp_path):
    stl = _box_stl(tmp_path / "cube.stl", (-0.1,) * 3, (0.1,) * 3)
    mesher = msh.CartesianMesher(
        domain=msh.BoxDomain(
            bounds=(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5),
            patches=msh.BoxPatches(*(["farfield"] * 6)),
        ),
        surfaces=(msh.STLSurface(stl, patch="cube"),),
        max_cell_size=H,
    )
    assert mesher.exact_sides == () and mesher.cell_size_anchor is None
    mesh = mesher.build()
    assert "coupling_box" not in mesh["mesh_generation"]
    # The cfMesh root cube stays centred on the domain.
    root = np.asarray(mesh["mesh_generation"]["root_box"])
    np.testing.assert_allclose(0.5 * (root[::2] + root[1::2]), 0.0, atol=1e-12)


def test_coupling_patch_names_are_validated():
    names = ("a", "a", "a", "a", "b", "c")
    with pytest.raises(ValueError, match="names no box face"):
        msh.BoxPatches(*names, coupling=("d",))
    with pytest.raises(TypeError, match="tuple"):
        msh.BoxPatches(*names, coupling="a")
    with pytest.raises(ValueError, match="duplicate"):
        msh.BoxPatches(*names, coupling=("a", "a"))
    patches = msh.BoxPatches(*names, coupling=("a", "c"))
    assert patches.coupling_sides() == (0, 1, 2, 3, 5)
    assert msh.BoxPatches.coupled("outer").coupling_sides() == tuple(range(6))


def test_a_surface_may_not_cross_or_crowd_a_coupling_side(tmp_path):
    crossing = _box_stl(tmp_path / "crossing.stl", (-2.0, -0.1, -0.1), (2.0, 0.1, 0.1))
    with pytest.raises(ValueError, match="crossed by a surface"):
        msh.CartesianMesher(
            domain=msh.BoxDomain(
                bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
                patches=msh.BoxPatches.coupled("numericalBoundary"),
            ),
            surfaces=(msh.STLSurface(crossing, patch="rod"),),
            max_cell_size=H,
            surface_may_cross_domain_boundary=True,
        )
    close = _box_stl(tmp_path / "close.stl", (0.5, -0.1, -0.1), (0.7, 0.1, 0.1))
    with pytest.raises(ValueError, match="Enlarge the box"):
        msh.CartesianMesher(
            domain=msh.BoxDomain(
                bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
                patches=msh.BoxPatches.coupled("numericalBoundary"),
            ),
            surfaces=(msh.STLSurface(close, patch="cube"),),
            max_cell_size=H,
        )
