# SPDX-License-Identifier: GPL-3.0-or-later
"""Passing Phase 1 tests for typed Cartesian-mesher configuration."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

import openonda.fvm.mesher as msh
from source.solvers.fvm.mesh.cartesian.config import CompositeSizeField
from source.solvers.fvm.mesh.cartesian.features import classify_features
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.surface_classification import SurfaceIndex
from source.solvers.fvm.mesh.validation import validate_geometry, validate_topology
from tests.fvm.cartesian_acceptance_fixtures import make_acceptance_fixtures

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _domain() -> msh.BoxDomain:
    return msh.BoxDomain(
        bounds=(-2.0, 2.0, -2.0, 2.0, -1.0, 1.0),
        patches=msh.BoxPatches(
            xmin="inlet",
            xmax="outlet",
            ymin="farfield",
            ymax="farfield",
            zmin="front",
            zmax="back",
        ),
    )


def test_configuration_objects_are_frozen_and_validate_units():
    patches = msh.BoxPatches("inlet", "outlet", "farfield", "farfield", "front", "back")
    with pytest.raises(FrozenInstanceError):
        patches.xmin = "changed"
    with pytest.raises(ValueError, match="positive"):
        msh.SphereRefinement("bad", (0.0, 0.0, 0.0), 0.0, 0.1)
    with pytest.raises(ValueError, match="between"):
        msh.FeatureRefinement(angle=180.0, cell_size=0.1)
    with pytest.raises(ValueError, match="differ"):
        msh.LineRefinement("bad", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.1)


def test_all_phase_one_volume_controls_combine_by_smallest_requested_size():
    controls = (
        msh.BoxRefinement("box", (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0), 0.25),
        msh.SphereRefinement("sphere", (0.0, 0.0, 0.0), 0.5, 0.125),
        msh.ConeRefinement("cone", (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.5, 1.0, 0.2),
        msh.LineRefinement("line", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.1),
    )
    field = CompositeSizeField(0.5, controls, minimum_size=0.0625)
    values = field.requested_size(np.asarray(((0.0, 0.0, 0.0), (1.5, 1.5, 1.5))))
    np.testing.assert_allclose(values, (0.1, 0.5))
    assert field.requested_sizes == (0.5, 0.25, 0.2, 0.125, 0.1, 0.0625)


def test_sphere_refinement_does_not_refine_disjoint_boxes_inside_its_aabb():
    """The octree adapter must retain primitive containment semantics."""
    sphere = msh.SphereRefinement("sphere", (0.0, 0.0, 0.0), 0.5, 0.1)
    assert sphere.intersects_box(np.asarray((-0.1, -0.1, -0.1)), np.asarray((0.1, 0.1, 0.1)))
    assert not sphere.intersects_box(np.asarray((0.4, 0.4, 0.4)), np.asarray((0.5, 0.5, 0.5)))


def test_cartesian_constructor_does_not_import_tutorial_configuration(tmp_path: Path):
    surface = tmp_path / "surface.stl"
    surface.write_text(
        "solid body\n"
        " facet normal 0 0 -1\n outer loop\n"
        "  vertex -0.25 -0.25 -0.25\n  vertex 0.25 0.25 -0.25\n  vertex 0.25 -0.25 -0.25\n"
        " endloop\n endfacet\n"
        "endsolid body\n",
        encoding="ascii",
    )
    # The intentionally malformed single-triangle file must fail in the
    # typed surface object, before any tutorial or meshing engine is touched.
    with pytest.raises(ValueError):
        msh.STLSurface(surface, patch="body")


def test_cartesian_build_preserves_declared_patch_names_and_reports_effective_sizes():
    surface = msh.STLSurface(
        REPOSITORY_ROOT / "tutorials/coupled_fvm_vpm/cube_flow/assets/cube.stl", patch="body"
    )
    mesher = msh.CartesianMesher(
        domain=_domain(),
        surfaces=(surface,),
        max_cell_size=0.5,
        boundary_cell_size=0.25,
        min_cell_size=0.125,
    )
    mesh = mesher.build()
    validate_topology(mesh)
    validate_geometry(mesh, compute_mesh_geometry(mesh, compute_lsq=False))
    assert {patch["name"] for patch in mesh["boundary"]} == {
        "inlet",
        "outlet",
        "farfield",
        "front",
        "back",
        "body",
    }
    assert mesher.report is not None
    assert mesher.report.sizes[0].effective == 0.5
    assert mesher.report.surface_hashes == (surface.sha256,)


@pytest.mark.parametrize("fixture_name", ("ellipsoid", "rotated_box", "finite_naca_wing"))
def test_curved_boundary_layers_fail_before_staircase_generation(tmp_path: Path, fixture_name: str):
    fixtures = make_acceptance_fixtures(tmp_path)
    surface = msh.STLSurface(fixtures[fixture_name].paths[0], patch="body")
    with pytest.raises(ValueError, match="curved/non-planar"):
        msh.CartesianMesher(
            domain=_domain(),
            surfaces=(surface,),
            max_cell_size=0.5,
            boundary_cell_size=0.25,
            min_cell_size=0.125,
            boundary_layers=(msh.BoundaryLayers(("body",), 2, 0.02, 1.1),),
        )


def test_surface_index_and_features_are_deterministic_for_smooth_and_sharp_inputs(
    tmp_path: Path,
):
    fixtures = make_acceptance_fixtures(tmp_path)
    sharp = msh.STLSurface(fixtures["rotated_box"].paths[0], patch="sharp")
    smooth = msh.STLSurface(fixtures["ellipsoid"].paths[0], patch="smooth")
    sharp_index = SurfaceIndex.build(sharp.triangles)
    smooth_index = SurfaceIndex.build(smooth.triangles)
    assert sharp_index.is_inside(np.asarray(((0.0, 0.0, 0.0),)))[0]
    assert not smooth_index.is_inside(np.asarray(((1.0, 1.0, 1.0),)))[0]
    sharp_features = classify_features(sharp.triangles, 35.0)
    assert len(sharp_features.edges) > 0
    assert len(sharp_features.corners) > 0
    assert len(classify_features(sharp.triangles, 35.0).edges) == len(sharp_features.edges)


def test_repeated_cartesian_builds_are_canonically_equal(tmp_path: Path):
    surface = msh.STLSurface(make_acceptance_fixtures(tmp_path)["ellipsoid"].paths[0], patch="body")
    mesher = msh.CartesianMesher(domain=_domain(), surfaces=(surface,), max_cell_size=0.5)
    first = mesher.build()
    second = mesher.build()
    for key in ("vertex_position", "faces", "owners", "neighbours"):
        np.testing.assert_array_equal(first[key], second[key])
    assert first["boundary"] == second["boundary"]
    assert (
        first["mesh_generation"]["cartesian_report"]
        == second["mesh_generation"]["cartesian_report"]
    )
