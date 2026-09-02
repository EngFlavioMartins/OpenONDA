"""Regression tests for solver-safe curved cut-cell extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from source.solvers.fvm.mesh.adaptive_cartesian import (
    AdaptiveCartesianMesher,
    BoundaryLayerSpec,
)
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.validation import validate_geometry, validate_topology

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = REPOSITORY_ROOT / "tutorials" / "coupled_fvm_vpm" / "cylinder_shedding_flow"
CYLINDER_STL = CASE_DIR / "assets" / "cylinder_long.stl"


def test_curved_cylinder_mapping_has_no_solid_centres_or_degenerate_faces():
    mesher = AdaptiveCartesianMesher(
        domain=(-1.0, 1.0, -1.0, 1.0, -2.5, 2.5),
        max_cell_size=0.5,
        surface_file=CYLINDER_STL,
        wall_patch_name="cylinder",
        surface_cell_size=0.125,
        surface_may_cross_domain_boundary=True,
    )

    mesh = mesher.build()
    validate_topology(mesh)
    geometry = compute_mesh_geometry(mesh, compute_lsq=False)
    validate_geometry(mesh, geometry)

    centres = np.asarray(geometry["cell_centre"])
    inside = (np.linalg.norm(centres[:, :2], axis=1) < 0.5 - 1.0e-10) & (
        np.abs(centres[:, 2]) < 2.5 - 1.0e-10
    )
    assert not np.any(inside)
    assert np.all(np.asarray(geometry["face_area"]) > 0.0)
    assert np.all(np.asarray(geometry["cell_volume"]) > 0.0)

    wall = next(patch for patch in mesh["boundary"] if patch["name"] == "cylinder")
    first = int(wall["start_face"])
    faces = mesh["faces"][first : first + int(wall["n_faces"])]
    assert {len(face) for face in faces}.issubset({3, 4})
    projection = mesh["mesh_generation"]["surface_projection"]
    assert projection["accepted_points"] == projection["attempted_points"]
    assert projection["rejected_nonpositive_volume"] == 0
    assert projection["rejected_volume_ratio"] == 0


def test_cylinder_boundary_layer_is_complete_and_stitched():
    mesher = AdaptiveCartesianMesher(
        domain=(-1.5, 1.5, -1.5, 1.5, -0.5, 0.5),
        max_cell_size=0.5,
        surface_file=CYLINDER_STL,
        wall_patch_name="cylinder",
        surface_cell_size=0.125,
        surface_may_cross_domain_boundary=True,
        boundary_layer=BoundaryLayerSpec(
            first_cell_height=1.0 / 64.0,
            layers=4,
            growth_ratio=1.1,
            transition_layers=4,
            interface_half_width=0.75,
        ),
    )

    # Four transition rings would leave oversized square-to-cylinder cells.
    # The mesher treats that input as a minimum and resolves the bridge.
    assert mesher.requested_boundary_layer is not None
    assert mesher.requested_boundary_layer.transition_layers == 4
    assert mesher.boundary_layer is not None
    assert mesher.boundary_layer.transition_layers == 8

    mesh = mesher.build()
    validate_topology(mesh)
    geometry = compute_mesh_geometry(mesh, compute_lsq=False)
    quality = validate_geometry(mesh, geometry)

    generation = mesh["mesh_generation"]["boundary_layer"]
    layer_index = np.asarray(mesh["boundary_layer_index"])
    expected = generation["theta_cells"] * generation["z_cells"]
    assert generation["wall_layers"] == 4
    assert generation["requested_transition_layers"] == 4
    assert generation["transition_layers"] == 8
    assert generation["transition_layers_auto_expanded"]
    assert generation["transition_to_lattice_ratio_max"] <= 1.0 + 1.0e-10
    assert all(np.count_nonzero(layer_index == layer) == expected for layer in range(4))
    assert "__boundary_layer_interface__" not in {patch["name"] for patch in mesh["boundary"]}
    assert quality["max_non_orthogonality_deg"] < 60.0
    assert quality["max_skewness"] < 0.5
    assert "surface_projection" not in mesh["mesh_generation"]


def test_cylinder_boundary_layer_supports_independent_spanwise_extrusion():
    mesher = AdaptiveCartesianMesher(
        domain=(-1.5, 1.5, -1.5, 1.5, -0.5, 0.5),
        max_cell_size=0.5,
        surface_file=CYLINDER_STL,
        wall_patch_name="cylinder",
        surface_cell_size=0.125,
        surface_may_cross_domain_boundary=True,
        boundary_layer=BoundaryLayerSpec(
            first_cell_height=1.0 / 64.0,
            layers=4,
            growth_ratio=1.1,
            transition_layers=4,
            interface_half_width=0.75,
            spanwise_cell_size=0.25,
        ),
    )

    mesh = mesher.build()
    validate_topology(mesh)
    geometry = compute_mesh_geometry(mesh, compute_lsq=False)
    quality = validate_geometry(mesh, geometry)
    layer = mesh["mesh_generation"]["boundary_layer"]

    assert mesh["mesh_generation"]["spanwise_cells"] == 4
    assert mesh["mesh_generation"]["spanwise_cell_size"] == 0.25
    assert layer["z_cells"] == 4
    assert np.array_equal(
        np.unique(np.asarray(mesh["vertex_position"])[:, 2]),
        np.linspace(-0.5, 0.5, 5),
    )
    assert quality["max_non_orthogonality_deg"] < 60.0
    assert quality["max_skewness"] < 0.5
