"""Bounded regression for the migrated native cylinder reference mesh."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import openonda.fvm.mesher as msh
from source.solvers.fvm.mesh.adaptive_cartesian import AdaptiveCartesianMesher
from source.solvers.fvm.mesh.validation import (
    MeshValidationError,
    validate_no_fluid_cell_centres_inside_surface,
    validate_wall_vertex_conformance,
)
from tutorials.coupled_fvm_vpm.cylinder_shedding_flow.reference_flow import setup


def test_cylinder_reference_curved_layers_fail_fast_in_native_cartesian_path():
    with pytest.raises(ValueError, match="curved/non-planar"):
        setup.grid_mesh(1.0 / 12.0)


def test_staircase_wall_is_rejected_by_surface_conformance_gate():
    surface = msh.STLSurface(
        Path(setup.CYLINDER_STL),
        patch="cylinder",
    )
    legacy = AdaptiveCartesianMesher(
        domain=(-3.0, 6.0, -3.0, 3.0, -0.6, 0.6),
        max_cell_size=0.5,
        surface_data=surface.surface_data,
        surface_exclusion_distance=0.3,
        skip_surface_recovery=True,
        wall_patch_name="cylinder",
        surface_cell_size=0.25,
        surface_may_cross_domain_boundary=True,
    )
    mesh = legacy.build()
    with pytest.raises(MeshValidationError, match="not conformal"):
        validate_wall_vertex_conformance(mesh, surface.triangles, "cylinder")


def test_fluid_centres_inside_surface_are_a_hard_failure():
    surface = msh.STLSurface(Path(setup.CYLINDER_STL), patch="cylinder")
    with pytest.raises(MeshValidationError, match="inside the input surface"):
        validate_no_fluid_cell_centres_inside_surface(
            np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))),
            surface.triangles,
        )
