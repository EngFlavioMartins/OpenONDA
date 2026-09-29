"""Scalar and batched wall queries enforce the same geometric tolerance."""

import sys

import numpy as np
import pytest

from source.solvers.fvm.mesh.surface_classification import SurfaceIndex
from source.solvers.fvm.mesh.validation import (
    MeshValidationError,
    validate_wall_vertex_conformance,
)


def test_batched_surface_queries_without_vtk(monkeypatch):
    index = SurfaceIndex.build(np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]]))
    monkeypatch.setitem(sys.modules, "pyvista", None)
    nearest, distances, _ids = index.nearest_points([[0.25, 0.25, 1.0], [-1.0, 0.0, 1.0]])
    np.testing.assert_allclose(nearest, [[0.25, 0.25, 0.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(distances, [1.0, np.sqrt(2.0)])


@pytest.mark.parametrize("scale", [0.01, 1.0, 100.0])
def test_batched_surface_queries_match_scalar_queries_on_curved_surface(scale):
    import pyvista as pv

    surface = pv.Sphere(theta_resolution=12, phi_resolution=12).triangulate()
    triangles = scale * surface.points[surface.faces.reshape(-1, 4)[:, 1:]].astype(float)
    index = SurfaceIndex.build(triangles)
    rng = np.random.default_rng(73)
    points = scale * rng.uniform(-1.0, 1.0, (64, 3))
    scalar = [index.nearest_point(point) for point in points]
    nearest, distances, _ids = index.nearest_points(points)
    np.testing.assert_allclose(distances, [row[1] for row in scalar], atol=scale * 1e-13)
    np.testing.assert_allclose(nearest, [row[0] for row in scalar], atol=scale * 1e-13)


@pytest.mark.parametrize("distance_factor,accepted", [(0.5, True), (2.0, False)])
def test_wall_conformance_preserves_tolerance_on_rotated_facet(distance_factor, accepted):
    angle = 0.37
    rotation = np.array([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ])
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
    vertices = vertices @ rotation.T + np.array([2.0, -1.0, 0.3])
    triangles = vertices[[[0, 1, 2], [0, 2, 3]]]
    tolerance = 1e-8
    points = vertices.copy()
    points[2] += distance_factor * tolerance * rotation[:, 2]
    mesh = {
        "vertex_position": points,
        "boundary": [{"name": "wall", "start_face": 0, "n_faces": 1}],
        "faces": [np.arange(4)],
    }
    if accepted:
        result = validate_wall_vertex_conformance(mesh, triangles, "wall", tolerance=tolerance)
        assert result["max_vertex_distance"] == pytest.approx(distance_factor * tolerance, abs=1e-15)
    else:
        with pytest.raises(MeshValidationError, match="worst vertex 2"):
            validate_wall_vertex_conformance(mesh, triangles, "wall", tolerance=tolerance)
