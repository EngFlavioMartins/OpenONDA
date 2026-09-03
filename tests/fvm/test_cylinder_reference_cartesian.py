"""Bounded regression for the migrated native cylinder reference mesh."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import openonda.fvm.mesher as msh
from source.solvers.fvm.mesh.validation import (
    MeshValidationError,
    validate_no_fluid_cell_centres_inside_surface,
    validate_wall_vertex_conformance,
)
from tutorials.coupled_fvm_vpm.cylinder_shedding_flow.reference_flow import setup


def test_cylinder_reference_builds_ten_conformal_native_layers():
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
        boundary_layers=(
            msh.BoundaryLayers(
                patches=("cylinder",),
                layers=10,
                first_cell_height=dx / 16.0,
                growth_ratio=1.15,
            ),
        ),
        surface_may_cross_domain_boundary=True,
    )
    mesh = mesher.build()
    labels = np.asarray(mesh["boundary_layer_index"])
    layer_counts = np.bincount(labels[labels >= 0])
    assert len(layer_counts) == 10
    assert len(set(layer_counts.tolist())) == 1
    assert layer_counts[0] > 0
    assert "layer_termination" not in {patch["name"] for patch in mesh["boundary"]}
    assert validate_wall_vertex_conformance(mesh, surface.triangles, "cylinder")[
        "max_vertex_distance"
    ] < 1.0e-10
    assert mesher.report is not None
    quality = mesher.report.diagnostics["quality"]
    assert quality["max_non_orthogonality_deg"] < 80.0
    assert quality["max_skewness"] < 2.0


def test_fluid_centres_inside_surface_are_a_hard_failure():
    surface = msh.STLSurface(Path(setup.CYLINDER_STL), patch="cylinder")
    with pytest.raises(MeshValidationError, match="inside the input surface"):
        validate_no_fluid_cell_centres_inside_surface(
            np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))),
            surface.triangles,
        )
