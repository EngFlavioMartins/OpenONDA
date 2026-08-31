"""Regression test for production-size chunked mesh geometry."""

from __future__ import annotations

import numpy as np

from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.rectilinear import box_mesh_3d


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
