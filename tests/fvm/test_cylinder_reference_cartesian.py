"""Bounded regression for the migrated native cylinder reference mesh."""

from __future__ import annotations

import numpy as np

from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.validation import validate_geometry, validate_topology
from tutorials.coupled_fvm_vpm.cylinder_shedding_flow.reference_flow import setup


def test_cylinder_reference_uses_declarative_native_controls():
    mesher = setup.grid_mesh(1.0 / 12.0)

    assert type(mesher).__name__ == "CartesianMesher"
    assert mesher.surface_may_cross_domain_boundary
    assert [refinement.name for refinement in mesher.refinements] == ["near_body", "wake"]
    assert mesher.boundary_layers[0].patches == ("cylinder",)
    assert mesher.boundary_layers[0].layers == 10
    assert mesher.effective_cell_size(2.0 / 12.0) == 0.125
    assert mesher.effective_cell_size(4.0 / 12.0) == 0.25


def test_cylinder_reference_build_is_native_and_layered():
    mesh = setup.grid_mesh(0.25).build()
    validate_topology(mesh)
    geometry = compute_mesh_geometry(mesh, compute_lsq=False)
    quality = validate_geometry(mesh, geometry)

    assert mesh["mesh_generation"]["method"] == "cartesian.adapter"
    assert mesh["mesh_generation"]["boundary_layer"]["method"] == "patch_normal_layers"
    assert {patch["name"] for patch in mesh["boundary"]} == {
        "inlet",
        "outlet",
        "ymin",
        "ymax",
        "zmin",
        "zmax",
        "cylinder",
        "layer_termination",
    }
    assert mesh["mesh_generation"]["boundary_layer"]["layers"] == 10
    assert np.max(np.asarray(mesh["boundary_layer_index"])) == 9
    assert quality["min_volume"] > 0.0
    assert np.all(np.isfinite(np.asarray(mesh["vertex_position"])))
