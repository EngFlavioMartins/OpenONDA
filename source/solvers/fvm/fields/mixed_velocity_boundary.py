"""Normal-value/tangential-gradient velocity boundary reconstruction."""

from __future__ import annotations

import numpy as np


def reconstruct_normal_velocity_tangential_gradient(
    owner_velocity: np.ndarray,
    normal: np.ndarray,
    normal_distance: np.ndarray,
    normal_velocity: np.ndarray,
    tangential_gradient: np.ndarray,
) -> np.ndarray:
    r"""Return face velocities satisfying the directional mixed condition.

    The boundary data prescribe ``U_f . n`` and
    ``(I - nn^T) (U_f - U_P) / d_n``.  Boundary ghost slots in the native FVM
    store face-centred values, so the reconstructed value can be consumed
    directly by convection, gradients, and face-flux evaluation.
    """
    owner = np.asarray(owner_velocity, dtype=np.float64)
    unit_normals = np.asarray(normal, dtype=np.float64)
    distance = np.asarray(normal_distance, dtype=np.float64).reshape(-1)
    prescribed_normal = np.asarray(normal_velocity, dtype=np.float64).reshape(-1)
    prescribed_tangent_gradient = np.asarray(tangential_gradient, dtype=np.float64)

    n_faces = owner.shape[0]
    expected_vector_shape = (n_faces, 3)
    if owner.shape != expected_vector_shape:
        raise ValueError(f"owner_velocity must have shape {expected_vector_shape}")
    if unit_normals.shape != expected_vector_shape:
        raise ValueError(f"normal must have shape {expected_vector_shape}")
    if prescribed_tangent_gradient.shape != expected_vector_shape:
        raise ValueError(f"tangential_gradient must have shape {expected_vector_shape}")
    if distance.shape != (n_faces,) or prescribed_normal.shape != (n_faces,):
        raise ValueError("normal_distance and normal_velocity must have one value per face")
    if not all(
        np.all(np.isfinite(values))
        for values in (
            owner,
            unit_normals,
            distance,
            prescribed_normal,
            prescribed_tangent_gradient,
        )
    ):
        raise ValueError("mixed velocity boundary reconstruction requires finite data")
    if np.any(distance <= 1.0e-14):
        raise ValueError("mixed velocity boundary requires positive owner-to-face distance")

    owner_normal = np.einsum("ij,ij->i", owner, unit_normals)
    return (
        owner
        + (prescribed_normal - owner_normal)[:, np.newaxis] * unit_normals
        + distance[:, np.newaxis] * prescribed_tangent_gradient
    )


def update_normal_velocity_tangential_gradient_boundary(
    velocity: np.ndarray,
    boundary: dict,
    mesh_data: dict,
    geo_data: dict,
) -> None:
    """Refresh one mixed patch's face-valued ghost slots in-place."""
    start = int(boundary["start_face"])
    n_faces = int(boundary["n_faces"])
    faces = np.arange(start, start + n_faces)
    n_cells = int(mesh_data["n_cells"])
    n_interior = int(mesh_data["n_interior_faces"])
    owners = np.asarray(mesh_data["owners"])[faces]
    ghosts = n_cells + (faces - n_interior)

    surface_vectors = np.asarray(geo_data["face_area_vector"], dtype=np.float64)[faces]
    areas = np.linalg.norm(surface_vectors, axis=1)
    if np.any(areas <= 1.0e-14):
        raise ValueError("mixed velocity boundary requires non-degenerate faces")
    normal = surface_vectors / areas[:, np.newaxis]
    owner_to_face = np.asarray(geo_data["cell_connection_vector"], dtype=np.float64)[faces]
    normal_distance = np.einsum("ij,ij->i", owner_to_face, normal)

    normal_velocity = boundary.get("normal_velocity_field")
    tangential_gradient = boundary.get("tangential_gradient_field")
    if normal_velocity is None or tangential_gradient is None:
        raise ValueError(
            f"Mixed velocity boundary {boundary.get('name')!r} has incomplete trace data"
        )
    velocity[ghosts] = reconstruct_normal_velocity_tangential_gradient(
        velocity[owners],
        normal,
        normal_distance,
        normal_velocity,
        tangential_gradient,
    )
