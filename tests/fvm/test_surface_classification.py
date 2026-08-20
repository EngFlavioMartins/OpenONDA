"""Geometric primitives for curved-surface classification, verified against
synthetic bodies with known analytic answers (sphere, capped cylinder)."""

from __future__ import annotations

import numpy as np
import pytest

from source.solvers.FVM.mesh.surface_classification import (
    SurfaceIndex,
    closest_point_on_triangles,
    triangle_box_overlap,
)


def uv_sphere_triangles(radius: float, n_theta: int = 24, n_phi: int = 16) -> np.ndarray:
    """Watertight UV-sphere triangulation centred at the origin."""
    face_flux = np.linspace(0.0, np.pi, n_phi + 1)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    verts = np.empty((n_phi + 1, n_theta, 3))
    for i, p in enumerate(face_flux):
        verts[i, :, 0] = radius * np.sin(p) * np.cos(theta)
        verts[i, :, 1] = radius * np.sin(p) * np.sin(theta)
        verts[i, :, 2] = radius * np.cos(p)
    triangles = []
    for i in range(n_phi):
        for j in range(n_theta):
            j2 = (j + 1) % n_theta
            a, b = verts[i, j], verts[i, j2]
            c, d = verts[i + 1, j], verts[i + 1, j2]
            if i > 0:
                triangles.append((a, b, c))
            if i < n_phi - 1:
                triangles.append((b, d, c))
    return np.asarray(triangles, dtype=np.float64)


def capped_cylinder_triangles(radius: float, z0: float, z1: float, n_theta: int = 32) -> np.ndarray:
    """Watertight capped-cylinder triangulation, axis along z."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    ring0 = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), np.full(n_theta, z0)))
    ring1 = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), np.full(n_theta, z1)))
    centre0 = np.array([0.0, 0.0, z0])
    centre1 = np.array([0.0, 0.0, z1])
    triangles = []
    for j in range(n_theta):
        j2 = (j + 1) % n_theta
        triangles.append((ring0[j], ring1[j], ring1[j2]))
        triangles.append((ring0[j], ring1[j2], ring0[j2]))
        triangles.append((centre0, ring0[j2], ring0[j]))
        triangles.append((centre1, ring1[j], ring1[j2]))
    return np.asarray(triangles, dtype=np.float64)


class TestClosestPointOnTriangles:
    def test_vertex_region(self):
        a = np.array([[0.0, 0.0, 0.0]])
        b = np.array([[1.0, 0.0, 0.0]])
        c = np.array([[0.0, 1.0, 0.0]])
        point = np.array([-1.0, -1.0, 0.0])
        result = closest_point_on_triangles(point, a, b, c)
        np.testing.assert_allclose(result[0], [0.0, 0.0, 0.0], atol=1e-12)

    def test_edge_region(self):
        a = np.array([[0.0, 0.0, 0.0]])
        b = np.array([[2.0, 0.0, 0.0]])
        c = np.array([[0.0, 2.0, 0.0]])
        point = np.array([1.0, -1.0, 0.0])
        result = closest_point_on_triangles(point, a, b, c)
        np.testing.assert_allclose(result[0], [1.0, 0.0, 0.0], atol=1e-12)

    def test_face_region(self):
        a = np.array([[0.0, 0.0, 0.0]])
        b = np.array([[1.0, 0.0, 0.0]])
        c = np.array([[0.0, 1.0, 0.0]])
        point = np.array([0.25, 0.25, 1.0])
        result = closest_point_on_triangles(point, a, b, c)
        np.testing.assert_allclose(result[0], [0.25, 0.25, 0.0], atol=1e-12)


class TestTriangleBoxOverlap:
    def test_triangle_through_box(self):
        v0 = np.array([[-1.0, 0.5, 0.5]])
        v1 = np.array([[1.0, 0.5, 0.5]])
        v2 = np.array([[0.0, -1.0, 0.5]])
        overlap = triangle_box_overlap(
            np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]), v0, v1, v2
        )
        assert bool(overlap[0])

    def test_triangle_far_from_box(self):
        v0 = np.array([[10.0, 10.0, 10.0]])
        v1 = np.array([[11.0, 10.0, 10.0]])
        v2 = np.array([[10.0, 11.0, 10.0]])
        overlap = triangle_box_overlap(
            np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]), v0, v1, v2
        )
        assert not bool(overlap[0])

    def test_triangle_edge_grazes_box_corner(self):
        # A thin triangle whose plane passes near the box but whose extent
        # misses it entirely: the separating-axis edge tests must reject it.
        v0 = np.array([[2.0, 2.0, -5.0]])
        v1 = np.array([[2.0, 2.0, 5.0]])
        v2 = np.array([[3.0, 3.0, 0.0]])
        overlap = triangle_box_overlap(
            np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]), v0, v1, v2
        )
        assert not bool(overlap[0])


class TestSurfaceIndexSphere:
    RADIUS = 1.0

    @pytest.fixture(scope="class")
    @classmethod
    def index(cls):
        return SurfaceIndex.build(uv_sphere_triangles(cls.RADIUS, n_theta=48, n_phi=32))

    def test_centre_is_inside(self, index):
        assert bool(index.is_inside(np.array([[0.0, 0.0, 0.0]]))[0])

    def test_far_point_is_outside(self, index):
        assert not bool(index.is_inside(np.array([[5.0, 0.0, 0.0]]))[0])

    def test_points_along_multiple_axes(self, index):
        inside_points = np.array(
            [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5], [-0.3, -0.3, 0.3]]
        )
        outside_points = np.array(
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0], [-1.5, -1.5, 1.5]]
        )
        assert np.all(index.is_inside(inside_points))
        assert not np.any(index.is_inside(outside_points))

    def test_nearest_point_lies_on_sphere_within_facet_tolerance(self, index):
        query = np.array([2.0, 0.0, 0.0])
        point, distance = index.nearest_point(query)
        radius = np.linalg.norm(point)
        # A 48x32 UV sphere facets the true sphere to within a few percent.
        assert radius == pytest.approx(self.RADIUS, rel=0.05)
        assert distance == pytest.approx(2.0 - radius, rel=0.05)

    def test_box_straddling_surface_intersects(self, index):
        assert index.box_intersects_surface(np.array([0.9, -0.1, -0.1]), np.array([1.1, 0.1, 0.1]))

    def test_box_fully_inside_does_not_intersect_surface(self, index):
        assert not index.box_intersects_surface(
            np.array([-0.1, -0.1, -0.1]), np.array([0.1, 0.1, 0.1])
        )

    def test_box_fully_outside_does_not_intersect_surface(self, index):
        assert not index.box_intersects_surface(
            np.array([5.0, 5.0, 5.0]), np.array([5.2, 5.2, 5.2])
        )


class TestSurfaceIndexCylinder:
    RADIUS = 0.5
    Z0, Z1 = -2.0, 2.0

    @pytest.fixture(scope="class")
    @classmethod
    def index(cls):
        return SurfaceIndex.build(capped_cylinder_triangles(cls.RADIUS, cls.Z0, cls.Z1, n_theta=48))

    def test_axis_points_inside_span(self, index):
        points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.9], [0.0, 0.0, -1.9]])
        assert np.all(index.is_inside(points))

    def test_points_beyond_caps_are_outside(self, index):
        points = np.array([[0.0, 0.0, 2.1], [0.0, 0.0, -2.1]])
        assert not np.any(index.is_inside(points))

    def test_points_outside_radius_are_outside(self, index):
        points = np.array([[0.6, 0.0, 0.0], [0.0, 0.6, 0.0]])
        assert not np.any(index.is_inside(points))

    def test_points_inside_radius_are_inside(self, index):
        points = np.array([[0.4, 0.0, 0.0], [0.0, -0.4, 0.5]])
        assert np.all(index.is_inside(points))
