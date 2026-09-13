"""Physical geometry and flux invariants for the fully 3D inherited-cell study."""

import numpy as np
import pytest

from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.rectilinear import box_mesh_3d
from studies.coupler_accuracy.cube_boundary_oracle import restrict_mesh_3d


@pytest.mark.parametrize("spacing", [(0.25, 0.25, 0.25), (0.25, 0.5, 0.25)])
def test_inherited_cube_preserves_3d_cells_wall_and_closed_flux(spacing):
    axes = [np.arange(-2, 2 + h / 2, h) for h in spacing]
    full = box_mesh_3d(*axes, hole_box=(-0.5, 0.5) * 3, wall_patch_name="cube")
    full_geo = compute_mesh_geometry(full, gradient_scheme="gauss", compute_lsq=False)
    centres = full_geo["cell_centre"][: full["n_cells"]]
    selected = np.flatnonzero(np.max(np.abs(centres), axis=1) < 1.5)
    small, source_faces, signs = restrict_mesh_3d(full, selected)
    geo = compute_mesh_geometry(small, gradient_scheme="gauss", compute_lsq=False)

    np.testing.assert_allclose(geo["cell_centre"], centres[selected], rtol=0, atol=2e-14)
    np.testing.assert_allclose(
        geo["cell_volume"], full_geo["cell_volume"][selected], rtol=0, atol=2e-14
    )
    np.testing.assert_allclose(geo["cell_volume"].sum(), 3**3 - 1, rtol=0, atol=2e-13)
    np.testing.assert_allclose(
        geo["face_area_vector"],
        full_geo["face_area_vector"][source_faces] * signs[:, None],
        rtol=0,
        atol=2e-14,
    )
    assert {p["name"] for p in small["boundary"]} == {"cube", "numericalBoundary"}
    wall = next(p for p in small["boundary"] if p["name"] == "cube")
    wall_rows = slice(wall["start_face"], wall["start_face"] + wall["n_faces"])
    np.testing.assert_allclose(geo["face_area"][wall_rows].sum(), 6, rtol=0, atol=2e-14)

    patch = next(p for p in small["boundary"] if p["name"] == "numericalBoundary")
    rows = slice(patch["start_face"], patch["start_face"] + patch["n_faces"])
    normals = geo["face_area_vector"][rows] / geo["face_area"][rows, None]
    for axis in range(3):
        for direction in (-1, 1):
            assert np.any(normals[:, axis] * direction > 0.99)
    # A fully 3D divergence-free linear field: all velocity components vary.
    full_velocity = full_geo["face_centre"] * np.array([1, 1, -2])
    full_flux = np.einsum("ij,ij->i", full_velocity, full_geo["face_area_vector"])
    small_velocity = geo["face_centre"] * np.array([1, 1, -2])
    small_flux = np.einsum("ij,ij->i", small_velocity, geo["face_area_vector"])
    np.testing.assert_allclose(small_flux, full_flux[source_faces] * signs, rtol=0, atol=2e-14)
    np.testing.assert_allclose(small_flux[small["n_interior_faces"] :].sum(), 0, rtol=0, atol=2e-14)


def test_3d_surface_completion_reconstructs_uniform_field_inside_and_zero_outside():
    from studies.coupler_accuracy.cube_snapshot_induction import boundary_velocity

    mu, weight = np.polynomial.legendre.leggauss(24)
    phi = np.arange(64) * (2 * np.pi / 64)
    normals = np.column_stack(
        (
            (np.sqrt(1 - mu[:, None] ** 2) * np.cos(phi)).ravel(),
            (np.sqrt(1 - mu[:, None] ** 2) * np.sin(phi)).ravel(),
            np.broadcast_to(mu[:, None], (24, 64)).ravel(),
        )
    )
    radius = 3.0
    area = np.repeat(weight, 64) * (2 * np.pi / 64) * radius**2
    target = np.array([[0, 0, 0], [0.3, 0.2, 0.8], [-0.4, 0.6, 0.2], [12, 0, 0]])
    uniform = np.array([1, -2, 0.3])
    actual = boundary_velocity(
        target, radius * normals, area[:, None] * normals, np.tile(uniform, (len(normals), 1))
    )
    np.testing.assert_allclose(actual[:3], np.tile(uniform, (3, 1)), rtol=0, atol=2e-14)
    np.testing.assert_allclose(actual[3], 0, rtol=0, atol=2e-14)


def test_3d_volume_induction_preserves_all_strength_components_and_far_field():
    from studies.coupler_accuracy.cube_snapshot_induction import volume_velocity

    position = np.array([[0.2, -0.1, 0.4], [-0.2, 0.3, -0.4]])
    strength = np.array([[0.3, -0.2, 0.5], [-0.1, 0.7, 0.2]])
    targets = np.array([[2, 3, 4], [-3, 1, 2]], dtype=float)
    delta = targets[:, None] - position[None]
    expected = (
        np.cross(strength[None], delta)
        / (4 * np.pi * np.linalg.norm(delta, axis=-1)[..., None] ** 3)
    ).sum(axis=1)
    actual, potential = volume_velocity(targets, position, strength, np.array([0.01, 0.02]))
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=1e-16)
    np.testing.assert_array_equal(potential, 0)
