"""Normal-value/tangential-gradient velocity boundary reconstruction."""

from __future__ import annotations

import numpy as np

from .boundary_reconstruction import boundary_owner_gradient


def boundary_skew_velocity_increment(
    velocity: np.ndarray,
    mesh_data: dict,
    geo_data: dict,
    face_indices: np.ndarray,
    normal: np.ndarray,
    owner_to_face: np.ndarray,
) -> np.ndarray | None:
    """Return ``J_P r_t`` from real-cell stencils, or None on orthogonal faces.

    This local reconstruction consumes the field's existing processor halos;
    it performs no collectives, including when called by a patch setter. Only
    geometric stencil weights are cached. Physical face ghosts are excluded,
    so the correction does not depend on patch update order or a previous
    mixed reconstruction.
    """
    normal_distance = np.einsum("ij,ij->i", owner_to_face, normal)
    tangent_displacement = owner_to_face - normal_distance[:, np.newaxis] * normal
    # Ignore floating-point projection noise, retaining the previous
    # reconstruction and its arithmetic on orthogonal meshes.
    skew = np.linalg.norm(tangent_displacement, axis=1) > (
        1.0e-12 * np.linalg.norm(owner_to_face, axis=1)
    )
    if not np.any(skew):
        return None
    gradient = boundary_owner_gradient(
        velocity,
        mesh_data,
        geo_data,
        face_indices[skew],
        include_physical_ghosts=False,
        displacements=tangent_displacement[skew],
    )
    increment = np.zeros((len(face_indices), 3), dtype=np.float64)
    # Native gradients are indexed by derivative direction, then component.
    increment[skew] = np.einsum("fd,fdc->fc", tangent_displacement[skew], gradient)
    return increment


def reconstruct_normal_velocity_tangential_gradient(
    owner_velocity: np.ndarray,
    normal: np.ndarray,
    normal_distance: np.ndarray,
    normal_velocity: np.ndarray,
    tangential_gradient: np.ndarray,
    *,
    skew_velocity_increment: np.ndarray | None = None,
) -> np.ndarray:
    r"""Return face velocities satisfying the directional mixed condition.

    The boundary data prescribe ``U_f . n`` and the tangential part of
    ``dU/dn``. With ``r_t = C_f - C_P - d_n n``, the optional increment
    ``J_P r_t`` supplies the tangential owner-to-face displacement separately
    from the prescribed normal derivative. Boundary ghost slots in the native FVM
    store face-centred values, so the reconstructed value can be consumed
    directly by convection, gradients, and face-flux evaluation.

    The continuous coupling condition follows Billuart, Duponcheel,
    Winckelmans and Chatelain, "A weak coupling between a near-wall Eulerian
    solver and a Vortex Particle-Mesh method for the efficient simulation of
    2D external flows", Journal of Computational Physics 473 (2023), 111726,
    https://doi.org/10.1016/j.jcp.2022.111726, Section 3.1, Eqs. (11)-(12).
    On a planar boundary, with P = I - nn^T and omega = curl(U),
    P dU/dn = grad_t(U . n) - n x omega. The coupler supplies P J_VPM n
    from the induced-velocity Jacobian; equivalence to a separately sampled
    vorticity trace requires that trace to equal curl(U_VPM).

    The skew increment is projected tangentially, preserving the prescribed
    normal velocity and physical normal derivative. The solver's
    fixedFluxPressure condition separately enforces the prescribed normal
    flux; it is not an independent pressure trace prescribed by the paper.
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
    reconstructed = (
        owner
        + (prescribed_normal - owner_normal)[:, np.newaxis] * unit_normals
        + distance[:, np.newaxis] * prescribed_tangent_gradient
    )
    if skew_velocity_increment is not None:
        increment = np.asarray(skew_velocity_increment, dtype=np.float64)
        if increment.shape != expected_vector_shape or not np.all(np.isfinite(increment)):
            raise ValueError(
                f"skew_velocity_increment must be finite with shape {expected_vector_shape}"
            )
        normal_increment = np.einsum("ij,ij->i", increment, unit_normals)
        reconstructed += increment - normal_increment[:, np.newaxis] * unit_normals
    return reconstructed


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
        skew_velocity_increment=boundary_skew_velocity_increment(
            velocity, mesh_data, geo_data, faces, normal, owner_to_face
        ),
    )
