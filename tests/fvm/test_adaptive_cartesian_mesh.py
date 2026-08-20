"""Certification tests for the native cfMesh-inspired Cartesian subset."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from source.solvers.FVM.mesh.adaptive_cartesian import (
    AdaptiveCartesianMesher,
    BoxRefinement,
)
from source.solvers.FVM.mesh.geometry import compute_mesh_geometry
from source.solvers.FVM.mesh.topology import build_cell_face_csr
from source.solvers.FVM.mesh.validation import validate_mesh

DOMAIN = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
BODY = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
SURFACE_FILE = (
    Path(__file__).parents[2] / "tutorials/coupled_fvm_vpm/cube_flow/referenceFlow/assets/cube.stl"
)


@pytest.fixture(scope="module")
def adaptive_mesh():
    mesher = AdaptiveCartesianMesher(
        DOMAIN,
        0.5,
        surface_file=SURFACE_FILE,
        wall_patch_name="cube",
        surface_cell_size=0.125,
        refinements=(BoxRefinement((0.25, 0.75, -0.5, 0.5, -0.5, 0.5), 0.25, "wake"),),
        merge_outer_patch="numericalBoundary",
    )
    mesh = mesher.build()
    geometry = compute_mesh_geometry(mesh, compute_lsq=False)
    return mesher, mesh, geometry


def _patch(mesh, name):
    return next(patch for patch in mesh["boundary"] if patch["name"] == name)


def test_dyadic_requested_sizes(adaptive_mesh):
    mesher, mesh, _ = adaptive_mesh
    assert mesher.surface_file == str(SURFACE_FILE.resolve())
    assert mesher.surface_bounds == BODY
    metadata = mesh["mesh_generation"]
    assert metadata["surface_file"] == str(SURFACE_FILE.resolve())
    assert metadata["surface_bounds"] == BODY
    assert metadata["surface_triangle_count"] == 12
    assert len(metadata["surface_sha256"]) == 64
    assert mesher.effective_cell_size(0.125) == pytest.approx(0.125)
    assert mesher.effective_cell_size(0.30) == pytest.approx(0.25)
    assert set(np.unique(mesh["cell_sizes"])) == {0.125, 0.25}


def test_topology_and_geometry_are_solver_valid(adaptive_mesh):
    _, mesh, geometry = adaptive_mesh
    validate_mesh(mesh)
    report = validate_mesh(mesh, geometry)
    assert report["min_volume"] > 0.0
    assert report["out_of_bounds_interpolation_weights"] == 0
    assert report["max_non_orthogonality_deg"] < 30.0


def test_no_fluid_cells_inside_body(adaptive_mesh):
    _, mesh, geometry = adaptive_mesh
    centres = geometry["cell_centroids"][: mesh["n_cells"]]
    inside = np.all(
        (centres > np.asarray(BODY[::2])) & (centres < np.asarray(BODY[1::2])),
        axis=1,
    )
    assert not inside.any()


def test_wall_is_closed_and_points_into_solid(adaptive_mesh):
    _, mesh, geometry = adaptive_mesh
    wall = _patch(mesh, "cube")
    ids = np.arange(wall["start_face"], wall["start_face"] + wall["n_faces"])
    sf = geometry["face_sf"][ids]
    centres = geometry["face_centroids"][ids]
    assert np.allclose(sf.sum(axis=0), 0.0, atol=1.0e-12)
    assert np.linalg.norm(sf, axis=1).sum() == pytest.approx(6.0)
    assert np.all(np.einsum("fi,fi->f", sf, -centres) > 0.0)

    edge_counts: Counter[tuple[int, int]] = Counter()
    for face in np.asarray(mesh["faces"])[ids]:
        for index, start in enumerate(face):
            end = face[(index + 1) % len(face)]
            edge_counts[tuple(sorted((int(start), int(end))))] += 1
    assert set(edge_counts.values()) == {2}


def test_transition_cells_are_polyhedral_and_two_to_one(adaptive_mesh):
    _, mesh, _ = adaptive_mesh
    cell_faces, offsets = build_cell_face_csr(
        mesh["owners"], mesh["neighbours"], mesh["n_cells"], mesh["n_faces"]
    )
    del cell_faces
    counts = np.diff(offsets)
    assert counts.max() == 9
    assert np.any(counts > 6)

    levels = np.asarray(mesh["cell_levels"])
    owners = np.asarray(mesh["owners"][: mesh["n_interior_faces"]])
    neighbours = np.asarray(mesh["neighbours"])
    assert np.max(np.abs(levels[owners] - levels[neighbours])) <= 1


def test_patch_layout_matches_coupler_contract(adaptive_mesh):
    _, mesh, _ = adaptive_mesh
    assert [(patch["name"], patch["type"]) for patch in mesh["boundary"]] == [
        ("numericalBoundary", "patch"),
        ("cube", "wall"),
    ]


def test_separate_outer_patch_layout():
    mesh = AdaptiveCartesianMesher((0, 1, 0, 1, 0, 1), 0.5).build()
    assert [patch["name"] for patch in mesh["boundary"]] == [
        "inlet",
        "outlet",
        "ymin",
        "ymax",
        "zmin",
        "zmax",
    ]
    assert all(patch["n_faces"] == 4 for patch in mesh["boundary"])


def test_fits_requested_background_and_snaps_refinement_bounds():
    mesher = AdaptiveCartesianMesher(
        DOMAIN,
        0.3,
        refinements=(BoxRefinement((-0.65, 0.65, -0.5, 0.5, -0.5, 0.5), 0.1),),
    )
    mesh = mesher.build()

    assert mesher.requested_max_cell_size == pytest.approx(0.3)
    assert mesher.max_cell_size == pytest.approx(2.0 / 7.0)
    assert mesher._base_counts() == (7, 7, 7)
    assert mesh["mesh_generation"]["requested_max_cell_size"] == pytest.approx(0.3)
    assert mesh["n_cells"] > 7**3


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "max_cell_size": 0.5,
                "surface_cell_size": 0.2,
            },
            "requires surface_file",
        ),
        (
            {
                "max_cell_size": 0.5,
                "surface_file": SURFACE_FILE,
            },
            "must be supplied together",
        ),
    ],
)
def test_rejects_nonconforming_geometry(kwargs, message):
    with pytest.raises(ValueError, match=message):
        AdaptiveCartesianMesher(DOMAIN, **kwargs).build()
