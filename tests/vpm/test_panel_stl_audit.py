"""Fail-fast STL mesh-audit and topological-orientation contracts."""

from __future__ import annotations

import numpy as np
import pytest

from source.solvers.vpm.boundary_elements.panels.geometry.stl_audit import (
    StlAuditError,
    audit_stl_mesh,
    orient_components_by_signed_volume,
    signed_volume,
)
from source.solvers.vpm.boundary_elements.panels.geometry.stl_io import (
    _compute_unit_normals,
    load_stl,
    save_stl,
)
from source.solvers.vpm.boundary_elements.panels.solver.panel_solver import PanelSolver


def _cube_triangles(size: float = 1.0, centre: tuple = (0.0, 0.0, 0.0)) -> np.ndarray:
    """A single closed, consistently-wound, outward-oriented unit cube."""
    s = size / 2.0
    vertex = np.array(
        [
            [-s, -s, -s],
            [s, -s, -s],
            [s, s, -s],
            [-s, s, -s],
            [-s, -s, s],
            [s, -s, s],
            [s, s, s],
            [-s, s, s],
        ]
    ) + np.asarray(centre)
    faces = [
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (3, 7, 6),
        (3, 6, 2),
        (0, 4, 7),
        (0, 7, 3),
        (1, 2, 6),
        (1, 6, 5),
    ]
    return np.array([[vertex[a], vertex[b], vertex[c]] for a, b, c in faces])


def _prism_from_polygon(poly: list, z0: float = 0.0, z1: float = 1.0) -> np.ndarray:
    """A closed, consistently-wound, outward-oriented prism over a simple polygon.

    The polygon must be CCW in xy and star-shaped from its first vertex (the
    top/bottom caps are triangulated as a fan from vertex 0).
    """
    n = len(poly)

    def vertex_at(index: int, z: float) -> tuple:
        x, y = poly[index]
        return (float(x), float(y), float(z))

    triangles = []
    for i in range(1, n - 1):
        triangles.append((vertex_at(0, z1), vertex_at(i, z1), vertex_at(i + 1, z1)))
    for i in range(1, n - 1):
        triangles.append((vertex_at(0, z0), vertex_at(i + 1, z0), vertex_at(i, z0)))
    for i in range(n):
        j = (i + 1) % n
        a0, b0 = vertex_at(i, z0), vertex_at(j, z0)
        a1, b1 = vertex_at(i, z1), vertex_at(j, z1)
        triangles.append((a0, b0, b1))
        triangles.append((a0, b1, a1))
    return np.array(triangles, dtype=np.float64)


def _concave_l_shape_triangles() -> np.ndarray:
    """A closed, watertight, concave L-shaped prism (a long and a short arm)."""
    return _prism_from_polygon([(0, 0), (10, 0), (10, 1), (1, 1), (1, 10), (0, 10)])


def _two_tetrahedra_sharing_an_edge() -> np.ndarray:
    """Two standalone-watertight tetrahedra glued along one shared edge.

    Each tetrahedron alone is a valid closed manifold (every edge owned by
    exactly two triangles); combined, the shared edge is owned by four
    triangles instead of two — a non-manifold edge with no accompanying open
    edge or duplicate triangle, isolating that one failure mode.
    """

    def tetrahedron(p0, p1, p2, p3) -> np.ndarray:
        p0, p1, p2, p3 = (np.asarray(p, dtype=np.float64) for p in (p0, p1, p2, p3))
        return np.array([(p1, p2, p3), (p0, p3, p2), (p0, p1, p3), (p0, p2, p1)])

    shared_a, shared_b = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)
    tetra_1 = tetrahedron(shared_a, shared_b, (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    tetra_2 = tetrahedron(shared_a, shared_b, (0.0, -1.0, 0.0), (0.0, 0.0, -1.0))
    return np.concatenate([tetra_1, tetra_2], axis=0)


def test_valid_closed_cube_passes():
    report = audit_stl_mesh(_cube_triangles())
    assert report["disposition"] == "pass"
    assert report["component_count"] == 1
    assert report["n_open_edges"] == 0
    assert report["n_nonmanifold_edges"] == 0
    assert report["n_inconsistent_winding_edges"] == 0
    np.testing.assert_allclose(report["component_signed_volumes"], [1.0])


def test_open_edge_mesh_is_rejected():
    open_triangles = _cube_triangles()[:-1]
    with pytest.raises(StlAuditError, match="open edge"):
        audit_stl_mesh(open_triangles)


def test_nonmanifold_edge_mesh_is_rejected():
    with pytest.raises(StlAuditError, match="non-manifold edge"):
        audit_stl_mesh(_two_tetrahedra_sharing_an_edge())


def test_duplicate_triangle_is_rejected():
    cube = _cube_triangles()
    duplicated = np.concatenate([cube, cube[0:1]], axis=0)
    with pytest.raises(StlAuditError, match="duplicate triangle"):
        audit_stl_mesh(duplicated)


def test_degenerate_zero_area_triangle_is_rejected():
    cube = _cube_triangles()
    degenerate = np.concatenate(
        [cube, np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]])], axis=0
    )
    with pytest.raises(StlAuditError, match="degenerate triangle"):
        audit_stl_mesh(degenerate)


def test_non_finite_coordinate_is_rejected():
    cube = _cube_triangles()
    cube[0, 0, 0] = np.nan
    with pytest.raises(StlAuditError, match="non-finite"):
        audit_stl_mesh(cube)


def test_panel_count_over_budget_is_rejected_before_allocation():
    with pytest.raises(StlAuditError, match="max_panels"):
        audit_stl_mesh(_cube_triangles(), max_panels=4)


def test_disconnected_components_require_explicit_acknowledgement():
    two_cubes = np.concatenate(
        [_cube_triangles(centre=(0.0, 0.0, 0.0)), _cube_triangles(centre=(10.0, 0.0, 0.0))],
        axis=0,
    )
    with pytest.raises(StlAuditError, match="disconnected component"):
        audit_stl_mesh(two_cubes)

    report = audit_stl_mesh(two_cubes, expected_components=2)
    assert report["component_count"] == 2
    assert report["disposition"] == "pass"


def test_signed_volume_orientation_fixes_a_concave_body_the_centroid_heuristic_gets_wrong():
    correctly_oriented = _concave_l_shape_triangles()
    assert signed_volume(correctly_oriented) > 0.0

    # The centroid heuristic this replaces: flip a panel's normal whenever it
    # points toward the mean of all panel centres. On a long, thin concave
    # body the mean panel centre sits outside the material near several
    # panels, so it "corrects" panels that were already correctly oriented.
    centre = correctly_oriented.mean(axis=1)
    geometry_centre = centre.mean(axis=0)
    normal = _compute_unit_normals(correctly_oriented)
    flipped_by_centroid_heuristic = np.einsum("ij,ij->i", centre - geometry_centre, normal) < 0.0
    assert flipped_by_centroid_heuristic.any(), (
        "fixture no longer demonstrates the centroid heuristic's failure mode"
    )

    # The signed-volume method must leave an already-correct mesh unchanged.
    reoriented = orient_components_by_signed_volume(correctly_oriented)
    np.testing.assert_allclose(reoriented, correctly_oriented)

    # And it must correctly re-orient a globally flipped copy of the same body.
    flipped_copy = correctly_oriented.copy()
    flipped_copy[:, [1, 2], :] = flipped_copy[:, [2, 1], :]
    assert signed_volume(flipped_copy) < 0.0
    reoriented_from_flipped = orient_components_by_signed_volume(flipped_copy)
    np.testing.assert_allclose(reoriented_from_flipped, correctly_oriented)


def test_inconsistent_winding_is_rejected():
    cube = _cube_triangles()
    reversed_one_face = cube.copy()
    reversed_one_face[0, [1, 2], :] = cube[0, [2, 1], :]
    with pytest.raises(StlAuditError, match="inconsistent triangle winding"):
        audit_stl_mesh(reversed_one_face)


def test_overlapping_components_are_reported_not_rejected():
    overlapping = np.concatenate(
        [_cube_triangles(), _cube_triangles(centre=(0.3, 0.3, 0.0))], axis=0
    )
    report = audit_stl_mesh(overlapping, expected_components=2)

    # Interpenetration is reported for a human to judge, never silently
    # repaired and never fatal — the proximity test is a heuristic.
    assert report["disposition"] == "warn"
    assert report["candidate_self_intersections"]
    assert any("self-intersection" in warning for warning in report["warnings"])


def test_add_surface_rejects_invalid_geometry_before_allocating_gpu_state(tmp_path):
    stl_path = tmp_path / "open.stl"
    save_stl(str(stl_path), _cube_triangles()[:-1])

    solver = PanelSolver(max_n_panels=64, float_dtype="f32", linear_solver="SCIPY")
    with pytest.raises(StlAuditError, match="open edge"):
        solver.add_surface("body", str(stl_path))

    # The audit must precede allocation, so a rejected STL leaves no lattice
    # and no dense influence matrix behind.
    assert solver.lattice is None
    assert solver.aerodynamic_influence_coefficient is None


def test_add_surface_rejects_a_multi_component_stl(tmp_path):
    stl_path = tmp_path / "two_cubes.stl"
    two_cubes = np.concatenate(
        [_cube_triangles(centre=(0.0, 0.0, 0.0)), _cube_triangles(centre=(10.0, 0.0, 0.0))],
        axis=0,
    )
    save_stl(str(stl_path), two_cubes)

    # add_surface maps one file to one PanelBody with one uid/kinematics, so
    # separate shells must not be merged into it silently.
    solver = PanelSolver(max_n_panels=64, float_dtype="f32", linear_solver="SCIPY")
    with pytest.raises(StlAuditError, match="disconnected component"):
        solver.add_surface("body", str(stl_path))
    assert solver.lattice is None


def test_production_cube_flow_stl_orientation_is_unchanged_from_the_old_heuristic():
    vertex_position, _ = load_stl("tutorials/coupled_fvm_vpm/cube_flow/assets/cube.stl")

    centre = vertex_position.mean(axis=1)
    geometry_centre = centre.mean(axis=0)
    old_normal = _compute_unit_normals(vertex_position)
    flip = np.einsum("ij,ij->i", centre - geometry_centre, old_normal) < 0.0
    old_oriented = vertex_position.copy()
    old_oriented[flip, 1, :], old_oriented[flip, 2, :] = (
        vertex_position[flip, 2, :],
        vertex_position[flip, 1, :],
    )

    new_oriented = orient_components_by_signed_volume(vertex_position)

    np.testing.assert_allclose(
        _compute_unit_normals(new_oriented), _compute_unit_normals(old_oriented)
    )
