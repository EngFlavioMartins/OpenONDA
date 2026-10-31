"""Mixed velocity boundary: prescribed normal velocity and normal gradient of the tangential velocity.

On a coupling patch the condition is direction-aware. With an external face
velocity ``u_e`` (the particle velocity), each face blends two well-posed
closures by the inflow weight ``w`` of :func:`coupled_inflow_weight`:

* inflow faces (``u_e . n <= 0``, ``w = 1``) take the full external velocity,
  as the characteristics of an inflow boundary require;
* outflow faces (``u_e . n >= transition speed``, ``w = 0``) keep the mixed
  condition, ``U_b . n = u_e . n`` and ``P_t dU/dn = g_t``, which lets the
  FVM's own tangential velocity, and its vorticity, leave the domain;
* in between, ``w`` decreases smoothly with the prescribed normal velocity.

The weight depends only on the prescribed data, never on the FVM solution, so
the condition is linear in the solution and cannot chatter when the flow
reverses on a face. A tangential Neumann condition on an inflow face leaves the
tangential velocity uncontrolled: the incoming value is the owner's own value
plus an extrapolation, and on skewed boundary cells that loop is
anti-dissipative (the instability of the NACA 4412 coupled case).
"""

from __future__ import annotations

import numpy as np

from ..mesh.data import BoundaryPatchData, GeometryData, MeshData
from .boundary_reconstruction import boundary_owner_gradient


def coupled_inflow_weight(normal_velocity: np.ndarray, transition_speed: float) -> np.ndarray:
    """Return the Dirichlet weight ``w`` of each face from its prescribed normal velocity.

    ``w = 1`` for ``u_n <= 0`` (inflow and tangential flow); ``w`` decreases with
    a C1 smoothstep to ``0`` at ``u_n = transition_speed`` (m/s) and stays ``0``
    beyond it (outflow).
    """
    normal = np.asarray(normal_velocity, dtype=np.float64).reshape(-1)
    speed = float(transition_speed)
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError("transition_speed must be finite and positive")
    t = np.clip(normal / speed, 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


def boundary_skew_velocity_increment(
    velocity: np.ndarray,
    mesh_data: MeshData,
    geo_data: GeometryData,
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


def reconstruct_mixed_velocity_boundary(
    owner_velocity: np.ndarray,
    normal: np.ndarray,
    normal_distance: np.ndarray,
    normal_velocity: np.ndarray,
    tangential_velocity_normal_gradient: np.ndarray,
    *,
    skew_velocity_increment: np.ndarray | None = None,
    inflow_velocity: np.ndarray | None = None,
    inflow_weight: np.ndarray | None = None,
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
    boundary vorticity requires that vorticity to equal curl(U_VPM).

    The skew increment is projected tangentially, preserving the prescribed
    normal velocity and physical normal derivative.

    With ``inflow_velocity`` (``(N, 3)``, m/s) and ``inflow_weight`` (``(N,)``
    in ``[0, 1]``), the tangential face velocity is
    ``w P u_e + (1 - w) P U_mixed``; the normal component stays prescribed. The solver's
    fixed_flux_pressure condition separately enforces the prescribed normal
    flux; it is not an independent pressure boundary value prescribed by the paper.
    """
    owner = np.asarray(owner_velocity, dtype=np.float64)
    unit_normals = np.asarray(normal, dtype=np.float64)
    distance = np.asarray(normal_distance, dtype=np.float64).reshape(-1)
    prescribed_normal = np.asarray(normal_velocity, dtype=np.float64).reshape(-1)
    prescribed_normal_gradient = np.asarray(tangential_velocity_normal_gradient, dtype=np.float64)

    n_faces = owner.shape[0]
    expected_vector_shape = (n_faces, 3)
    if owner.shape != expected_vector_shape:
        raise ValueError(f"owner_velocity must have shape {expected_vector_shape}")
    if unit_normals.shape != expected_vector_shape:
        raise ValueError(f"normal must have shape {expected_vector_shape}")
    if prescribed_normal_gradient.shape != expected_vector_shape:
        raise ValueError(
            f"tangential_velocity_normal_gradient must have shape {expected_vector_shape}"
        )
    if distance.shape != (n_faces,) or prescribed_normal.shape != (n_faces,):
        raise ValueError("normal_distance and normal_velocity must have one value per face")
    if not all(
        np.all(np.isfinite(values))
        for values in (
            owner,
            unit_normals,
            distance,
            prescribed_normal,
            prescribed_normal_gradient,
        )
    ):
        raise ValueError("mixed velocity boundary reconstruction requires finite data")
    if np.any(distance <= 1.0e-14):
        raise ValueError("mixed velocity boundary requires positive owner-to-face distance")

    owner_normal = np.einsum("ij,ij->i", owner, unit_normals)
    reconstructed = (
        owner
        + (prescribed_normal - owner_normal)[:, np.newaxis] * unit_normals
        + distance[:, np.newaxis] * prescribed_normal_gradient
    )
    if skew_velocity_increment is not None:
        increment = np.asarray(skew_velocity_increment, dtype=np.float64)
        if increment.shape != expected_vector_shape or not np.all(np.isfinite(increment)):
            raise ValueError(
                f"skew_velocity_increment must be finite with shape {expected_vector_shape}"
            )
        normal_increment = np.einsum("ij,ij->i", increment, unit_normals)
        reconstructed += increment - normal_increment[:, np.newaxis] * unit_normals
    if (inflow_velocity is None) != (inflow_weight is None):
        raise ValueError("inflow_velocity and inflow_weight must be given together")
    if inflow_velocity is not None:
        external = np.asarray(inflow_velocity, dtype=np.float64)
        weight = np.asarray(inflow_weight, dtype=np.float64).reshape(-1)
        if external.shape != expected_vector_shape or weight.shape != (n_faces,):
            raise ValueError("inflow data must have one velocity and one weight per face")
        if not (np.all(np.isfinite(external)) and np.all((weight >= 0.0) & (weight <= 1.0))):
            raise ValueError("inflow data must be finite with weights in [0, 1]")
        external_tangential = (
            external - np.einsum("ij,ij->i", external, unit_normals)[:, np.newaxis] * unit_normals
        )
        mixed_tangential = (
            reconstructed
            - np.einsum("ij,ij->i", reconstructed, unit_normals)[:, np.newaxis] * unit_normals
        )
        # (1 - w) is exactly zero on inflow faces, so their value carries no
        # owner dependence, not even at round-off.
        reconstructed = (
            prescribed_normal[:, np.newaxis] * unit_normals
            + weight[:, np.newaxis] * external_tangential
            + (1.0 - weight)[:, np.newaxis] * mixed_tangential
        )
    return reconstructed


def update_mixed_velocity_boundary(
    velocity: np.ndarray,
    boundary: BoundaryPatchData,
    mesh_data: MeshData,
    geo_data: GeometryData,
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
    tangential_velocity_normal_gradient = boundary.get("tangential_velocity_normal_gradient_field")
    if normal_velocity is None or tangential_velocity_normal_gradient is None:
        raise ValueError(
            f"Mixed velocity boundary {boundary.get('name')!r} has incomplete boundary data"
        )
    inflow_velocity = boundary.get("inflow_velocity_field")
    inflow_weight = boundary.get("inflow_weight_field")
    velocity[ghosts] = reconstruct_mixed_velocity_boundary(
        velocity[owners],
        normal,
        normal_distance,
        normal_velocity,
        tangential_velocity_normal_gradient,
        skew_velocity_increment=mixed_skew_velocity_increment(
            velocity, mesh_data, geo_data, faces, normal, owner_to_face, inflow_weight
        ),
        inflow_velocity=inflow_velocity,
        inflow_weight=inflow_weight,
    )


def mixed_skew_velocity_increment(
    velocity: np.ndarray,
    mesh_data: MeshData,
    geo_data: GeometryData,
    face_indices: np.ndarray,
    normal: np.ndarray,
    owner_to_face: np.ndarray,
    inflow_weight: np.ndarray | None,
) -> np.ndarray | None:
    """Return ``J_P r_t``, zeroed on pure-Dirichlet faces (``w = 1``).

    The increment is evaluated on the whole patch so that its cached
    least-squares stencil does not depend on the current inflow faces.
    """
    increment = boundary_skew_velocity_increment(
        velocity, mesh_data, geo_data, face_indices, normal, owner_to_face
    )
    if increment is None or inflow_weight is None:
        return increment
    increment[np.asarray(inflow_weight, dtype=np.float64).reshape(-1) >= 1.0] = 0.0
    return increment
