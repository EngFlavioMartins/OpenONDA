"""Regression contract for CPU-side VLM panel geometry generation."""

from __future__ import annotations

import numpy as np

from source.solvers.vpm.boundary_elements.vlm.solver.mesh import (
    _compute_bilinear_coefficients,
    _fill_panel_geometry,
)


def test_panel_geometry_keeps_output_arrays_distinct_from_panel_values():
    alpha = _compute_bilinear_coefficients(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    )
    panel_corner_position = np.zeros((1, 4, 3))
    collocation_point = np.zeros((1, 3))
    normal = np.zeros((1, 3))
    area = np.zeros(1)
    trailing_direction = np.zeros((1, 2, 3))
    vortex_point_position = np.zeros((1, 4, 3))
    bound_vortex_midpoint = np.zeros((1, 3))

    _fill_panel_geometry(
        alpha,
        0.0,
        1.0,
        0.0,
        1.0,
        1.0,
        0,
        panel_corner_position,
        collocation_point,
        normal,
        area,
        trailing_direction,
        vortex_point_position,
        bound_vortex_midpoint,
        False,
    )

    np.testing.assert_allclose(normal[0], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(area[0], 1.0)
    np.testing.assert_allclose(collocation_point[0], [0.75, 0.5, 0.0])
