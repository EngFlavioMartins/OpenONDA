"""Mesh contracts: numerical and lifecycle contracts."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from source.solvers.fvm.mesh.cartesian import BoxDomain, BoxPatches, structured_box
from source.solvers.fvm.mesh.cartesian.extrusion import extrude_mesh_section
from source.solvers.fvm.mesh.cartesian.surface_recovery import _face_fluid_polygons
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.rectilinear import box_mesh_3d
from source.solvers.fvm.mesh.validation import (
    MeshValidationError,
    validate_geometry,
    validate_topology,
)
from tests.fvm.cartesian_acceptance_fixtures import make_acceptance_fixtures

VALID_FIXTURE_NAMES = (
    "rotated_box",
    "ellipsoid",
    "torus",
    "finite_naca_wing",
    "two_disjoint_bodies",
)


INVALID_FIXTURE_NAMES = (
    "open_edge",
    "non_manifold_edge",
    "inverted_component",
    "degenerate_triangle",
)


@pytest.mark.parametrize("fixture_name", VALID_FIXTURE_NAMES)
@pytest.mark.slow
def test_geometry_independence_acceptance_matrix(tmp_path, fixture_name):
    """Every fixture uses one path and rejects unsupported recovery explicitly."""
    import numpy as np

    import openonda.fvm.mesher as msh
    from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
    from source.solvers.fvm.mesh.validation import validate_geometry, validate_topology

    fixtures = make_acceptance_fixtures(tmp_path)
    fixture = fixtures[fixture_name]
    patches = ("body_a", "body_b") if fixture_name == "two_disjoint_bodies" else (fixture_name,)
    surfaces = tuple(
        msh.STLSurface(path, patch=patch)
        for path, patch in zip(fixture.paths, patches, strict=True)
    )
    mesher = msh.CartesianMesher(
        domain=msh.BoxDomain(
            bounds=(-1.5, 1.5, -1.5, 1.5, -1.5, 1.5),
            patches=msh.BoxPatches(
                xmin="inlet",
                xmax="outlet",
                ymin="farfield",
                ymax="farfield",
                zmin="front",
                zmax="back",
            ),
        ),
        surfaces=surfaces,
        max_cell_size=0.50,
        boundary_cell_size=0.25,
        min_cell_size=0.125,
        features=msh.FeatureRefinement(angle=35.0, cell_size=0.125),
    )
    mesh = mesher.build()
    validate_topology(mesh)
    geometry = compute_mesh_geometry(mesh, gradient_scheme="lsq")
    validate_geometry(mesh, geometry)
    assert np.all(np.isfinite(geometry["cell_volume"]))
    assert np.all(geometry["cell_volume"] > 0)
    assert {patch["name"] for patch in mesh["boundary"]} >= {"inlet", "outlet", *patches}


@pytest.mark.parametrize("fixture_name", INVALID_FIXTURE_NAMES)
def test_invalid_surface_fixtures_fail_diagnostically(tmp_path, fixture_name):
    """Broken topology must fail explicitly during target surface construction."""
    import openonda.fvm.mesher as msh

    fixture = make_acceptance_fixtures(tmp_path)[fixture_name]
    with pytest.raises((ValueError, RuntimeError)):
        msh.STLSurface(fixture.paths[0], patch="broken")


def test_section_extrusion_conserves_volume_and_shared_faces():
    source = structured_box(3, 2, 4, lx=3, ly=2, lz=4)
    domain = BoxDomain(
        bounds=(0, 3, 0, 2, -0.5, 0.5),
        patches=BoxPatches("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"),
    )
    mesh = extrude_mesh_section(source, coordinate=1.3, levels=(-0.5, 0, 0.5), domain=domain)
    assert mesh["n_cells"] == 12
    assert mesh["n_interior_faces"] == 20
    validate_topology(mesh)
    geometry = compute_mesh_geometry(mesh, gradient_scheme="lsq")
    quality = validate_geometry(mesh, geometry)
    assert geometry["cell_volume"].sum() == pytest.approx(6)
    assert quality["max_skewness"] < 1e-12
    assert quality["max_lsq_condition"] < 4
    assert {p["name"] for p in mesh["boundary"]} == set(domain.patches.as_tuple())
    np.testing.assert_allclose(mesh["vertex_position"][:, 2].min(), -0.5)
    np.testing.assert_allclose(mesh["vertex_position"][:, 2].max(), 0.5)


def test_section_extrusion_recovers_displaced_outer_edges():
    source = structured_box(2, 2, 1, lx=2, ly=2, lz=1)
    points = np.asarray(source["vertex_position"], dtype=float).copy()
    points[np.isclose(points[:, 0], 0.0), 0] = 1.0e-3
    points[np.isclose(points[:, 0], 2.0), 0] = 2.0 - 1.0e-3
    points[np.isclose(points[:, 1], 0.0), 1] = 1.0e-3
    points[np.isclose(points[:, 1], 2.0), 1] = 2.0 - 1.0e-3
    source["vertex_position"] = points
    domain = BoxDomain(
        bounds=(0, 2, 0, 2, -0.5, 0.5),
        patches=BoxPatches("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"),
    )

    mesh = extrude_mesh_section(source, coordinate=0.4, levels=(-0.5, 0.5), domain=domain)

    geometry = compute_mesh_geometry(mesh, gradient_scheme="lsq")
    validate_geometry(mesh, geometry)
    assert geometry["cell_volume"].sum() == pytest.approx(4.0)
    assert {patch["name"] for patch in mesh["boundary"]} == set(domain.patches.as_tuple())


@pytest.mark.parametrize("levels", [(-0.5, -0.5, 0.5), (-0.5, -0.6, 0.5), (-0.5, 0.4)])
def test_invalid_extrusion_levels_are_rejected(levels):
    source = structured_box(1, 1, 1)
    domain = BoxDomain(
        bounds=(0, 1, 0, 1, -0.5, 0.5),
        patches=BoxPatches("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"),
    )
    with pytest.raises(ValueError, match="levels"):
        extrude_mesh_section(source, coordinate=0.4, levels=levels, domain=domain)


def test_topology_rejects_a_coarse_face_missing_a_transition_midpoint():
    mesh = structured_box(1, 1, 1)
    points = np.asarray(mesh["vertex_position"], dtype=np.float64)
    points = np.vstack((points, np.asarray(((0.0, 0.5, 1.0),))))
    faces = [np.asarray(face, dtype=np.int32).copy() for face in mesh["faces"]]
    # Split one perimeter edge on the x-min face without splitting the
    # adjacent z-max face.  This is the minimal form of the adaptive
    # coarse/fine transition defect found by the audit.
    faces[0] = np.asarray((4, 8, 6, 2, 0), dtype=np.int32)
    mesh["vertex_position"] = points
    mesh["faces"] = faces
    mesh["n_points"] = len(points)

    with pytest.raises(MeshValidationError, match="non-closed polygon edges"):
        validate_topology(mesh)


class _CornerSolid:
    """Classify only the four source-face corners as solid test points."""

    @staticmethod
    def is_inside(points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        return np.isclose(np.abs(values[:, 0]), 1.0) & np.isclose(np.abs(values[:, 1]), 1.0)


def test_tangent_cut_line_is_not_sent_to_delaunay() -> None:
    """A rank-one tangent arrangement has no finite-area face polygon."""
    original = np.asarray(((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)))
    points = [
        np.asarray(((-0.75, 0.0, 0.0), (-0.25, 0.0, 0.0), (0.25, 0.0, 0.0))),
        np.asarray(((-0.25, 0.0, 0.0), (0.25, 0.0, 0.0), (0.75, 0.0, 0.0))),
    ]

    polygons = _face_fluid_polygons(
        original,
        points,
        (_CornerSolid(),),  # type: ignore[arg-type]
        1.0e-12,
    )

    assert polygons == []


def test_cell_geometry_preserves_the_global_output_across_chunks():
    cell_count = 50_001
    mesh = box_mesh_3d(
        np.linspace(0.0, 1.0, cell_count + 1),
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
    )

    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)

    expected_x = (np.arange(cell_count) + 0.5) / cell_count
    np.testing.assert_allclose(
        geometry["cell_centre"],
        np.column_stack((expected_x, np.full(cell_count, 0.5), np.full(cell_count, 0.5))),
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        geometry["cell_volume"],
        np.full(cell_count, 1.0 / cell_count),
        rtol=1.0e-11,
        atol=0.0,
    )


def test_partitioned_lsq_quality_excludes_halo_placeholders():
    mesh = structured_box(2, 2, 2)
    geometry = compute_mesh_geometry(mesh, gradient_scheme="lsq", compute_lsq=True)
    geometry["lsq_condition"][-1] = 1.0e90
    geometry["lsq_rank"][-1] = 0
    geometry["lsq_solver_method"][-1] = "svd"
    mesh["_parallel_context"] = SimpleNamespace(is_partitioned=True, n_owned=7)

    report = validate_geometry(mesh, geometry)

    assert report["max_lsq_condition"] < 2.0
    assert report["rank_deficient_lsq_cells"] == 0
    assert report["svd_lsq_cells"] == 0
