"""Geometry contract for the tracked spanwise cylinder reference surface."""

from __future__ import annotations

import inspect

import numpy as np

from tutorials.coupled_fvm_vpm.cylinder_shedding_flow.reference_flow import setup


def test_tracked_cylinder_surface_crosses_span_with_caps_outside_domain():
    assert setup.CYLINDER_STL.name == "cylinder_long.stl"
    assert setup.CYLINDER_STL.is_file()

    vertices = np.asarray(
        [
            [float(value) for value in line.split()[1:]]
            for line in setup.CYLINDER_STL.read_text(encoding="ascii").splitlines()
            if line.lstrip().startswith("vertex ")
        ]
    )
    triangles = vertices.reshape(-1, 3, 3)
    radius = np.linalg.norm(vertices[:, :2], axis=1)
    side_vertices = radius > 0.0
    np.testing.assert_allclose(radius[side_vertices], 0.5, atol=1.0e-12)
    np.testing.assert_allclose(np.abs(vertices[~side_vertices, 2]), 6.0, atol=0.0)
    assert vertices[:, 2].min() == -6.0
    assert vertices[:, 2].max() == 6.0
    assert vertices[:, 2].min() < setup.DOMAIN[4]
    assert vertices[:, 2].max() > setup.DOMAIN[5]
    cap_triangles = np.ptp(triangles[:, :, 2], axis=1) == 0.0
    assert np.any(cap_triangles)
    np.testing.assert_allclose(np.abs(triangles[cap_triangles, :, 2]), 6.0, atol=0.0)

    grid_source = inspect.getsource(setup.grid_mesh)
    assert "surface_may_cross_domain_boundary=True" in grid_source
