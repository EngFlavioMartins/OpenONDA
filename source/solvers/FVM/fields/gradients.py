"""Gauss and inverse-distance least-squares gradients."""

import numpy as np

from ..schemes.boundaries import BOUNDARIES, BoundaryStrategy

_LSQ_QR_CONDITION_LIMIT = 1.0e8


def _is_empty_boundary(boundary, *, allow_source_type: bool = False) -> bool:
    type_u = boundary.get("bc_type_U")
    if type_u is not None:
        strategy = BOUNDARIES.strategy(type_u, "U", "gradient")
    elif boundary.get("bc_type") is None and allow_source_type:
        return boundary.get("type") == "empty"
    else:
        strategy = BOUNDARIES.strategy(boundary.get("bc_type"), "scalar", "gradient")
    return strategy is BoundaryStrategy.EMPTY


def _correct_boundary_gradient(grad_phi, phi, mesh_data, geo_data):
    """Replace the wall-normal part of a boundary gradient with the exact snGrad.

    OpenFOAM's ``fv::gaussGrad::correctBoundaryConditions`` — which *both* the
    Gauss and least-squares schemes call at the end of ``calcGrad`` — overwrites
    the normal component of the patch gradient with the boundary condition's own
    surface-normal derivative::

        gGrad_b += n * (snGrad_b - (n & gGrad_b))

    Extrapolating the owner-cell gradient instead leaves the normal component
    reconstructed from a one-sided stencil that never saw the boundary value.
    At a no-slip wall, where ``U`` goes to zero over half a cell, that is the
    dominant part of the tensor, so anything consuming a boundary-face gradient
    (the viscous stress, wall traction, Rhie-Chow) starts from the wrong number.

    ``snGrad`` is taken from the ghost value the boundary conditions already
    wrote, ``(phi_ghost - phi_owner) * deltaCoeffs``, with OpenFOAM's
    ``deltaCoeffs = 1 / (n . (Cf - CP))``.  Coupled (cyclic) and empty patches
    are skipped, exactly as the ``!coupled()`` guard does upstream.
    """
    n_elements = mesh_data["n_elements"]
    n_interior = mesh_data["n_interior_faces"]
    n_faces = mesh_data["n_faces"]
    if n_faces == n_interior:
        return grad_phi

    owners = mesh_data["owners"]
    face_sf = geo_data["face_sf"]
    face_cf_vector = geo_data.get("face_cf_vector")
    face_centroids = geo_data.get("face_centroids")
    element_centroids = geo_data.get("element_centroids")
    boundary_neighbours = np.asarray(
        mesh_data.get("boundary_neighbours", np.full(n_faces, -1, dtype=np.int32))
    )

    for boundary in mesh_data["boundary"]:
        if _is_empty_boundary(boundary, allow_source_type=True):
            continue
        start = boundary["startFace"]
        n_patch_faces = boundary["nFaces"]
        faces = np.arange(start, start + n_patch_faces)
        if np.any(boundary_neighbours[faces] >= 0):
            continue  # coupled patch: OpenFOAM leaves the gradient untouched

        ghosts = n_elements + (faces - n_interior)
        owner_cells = owners[faces]
        sf = face_sf[faces]
        area = np.linalg.norm(sf, axis=1)
        normals = sf / np.maximum(area, 1e-300)[:, np.newaxis]

        # deltaCoeffs = 1 / (n . (Cf - CP)); guard a degenerate normal distance.
        if face_cf_vector is not None:
            owner_to_face = face_cf_vector[faces]
        elif face_centroids is not None and element_centroids is not None:
            owner_to_face = face_centroids[faces] - element_centroids[owner_cells]
        else:
            raise KeyError(
                "Boundary-gradient correction requires face_cf_vector or both "
                "face_centroids and element_centroids"
            )
        normal_distance = np.sum(normals * owner_to_face, axis=1)
        delta_coeffs = 1.0 / np.where(np.abs(normal_distance) < 1e-300, 1e-300, normal_distance)

        # sn_grad[face, component]; grad_phi is (n_total, 3, n_components).
        sn_grad = (phi[ghosts] - phi[owner_cells]) * delta_coeffs[:, np.newaxis]
        patch_grad = grad_phi[ghosts]
        normal_part = np.einsum("fd,fdc->fc", normals, patch_grad)
        grad_phi[ghosts] += normals[:, :, np.newaxis] * (sn_grad - normal_part)[:, np.newaxis, :]

    return grad_phi


def compute_gauss_gradient(phi, mesh_data, geo_data):
    """Compute the gradient using the Gauss linear method (vectorised).

    Uses deterministic ``bincount`` reductions for face accumulation. The
    algorithm and interface are identical to
    :func:`compute_gradient_gauss_linear`.

    Args:
        phi:      Field values ``(n_total, n_components)``.
        mesh_data: Mesh dictionary.
        geo_data:  Geometry dictionary.

    Returns:
        Gradient field ``(n_total, 3, n_components)``.
    """

    # Determine field type
    if phi.ndim == 1:
        phi = phi.reshape(-1, 1)

    n_total = phi.shape[0]
    n_components = phi.shape[1]

    n_elements = mesh_data["n_elements"]
    n_interior_faces = mesh_data["n_interior_faces"]
    n_faces = mesh_data["n_faces"]
    n_boundary_faces = n_faces - n_interior_faces

    owners = mesh_data["owners"]
    neighbours = mesh_data["neighbours"]

    face_sf = geo_data["face_sf"]
    face_weights = geo_data["face_weights"]
    element_volumes = geo_data["element_volumes"]

    # Initialize gradient
    grad_phi = np.zeros((n_total, 3, n_components), dtype=np.float64)

    # --- INTERIOR FACES (Vectorized) ---
    owners_i = owners[:n_interior_faces]
    neighbours_i = neighbours[:n_interior_faces]
    weights_i = face_weights[:n_interior_faces]
    sf_i = face_sf[:n_interior_faces]

    for i_component in range(n_components):
        # Face interpolation
        phi_f = (
            weights_i * phi[neighbours_i, i_component]
            + (1 - weights_i) * phi[owners_i, i_component]
        )

        # Contribution to owners (vectorized accumulation)
        contribution = phi_f[:, np.newaxis] * sf_i  # (n_faces, 3)
        for dimension in range(3):
            grad_phi[:n_elements, dimension, i_component] += np.bincount(
                owners_i,
                weights=contribution[:, dimension],
                minlength=n_elements,
            )
            grad_phi[:n_elements, dimension, i_component] -= np.bincount(
                neighbours_i,
                weights=contribution[:, dimension],
                minlength=n_elements,
            )

    # --- BOUNDARY FACES (Vectorized) ---
    owners_b = owners[n_interior_faces:n_faces]
    sf_b = face_sf[n_interior_faces:n_faces]

    # Identify non-empty boundary face indices
    boundaries = mesh_data["boundary"]
    valid_b_face_indices = []

    for boundary in boundaries:
        if _is_empty_boundary(boundary):
            continue

        start = boundary["startFace"]
        nf = boundary["nFaces"]
        valid_b_face_indices.extend(range(start, start + nf))

    # Store indices of boundary elements that are part of non-empty patches
    # Store indices of owner cells corresponding to these non-empty boundary elements

    if valid_b_face_indices:
        valid_b_face_indices = np.array(valid_b_face_indices)
        # Relative indices for owners_b and sf_b
        rel_indices = valid_b_face_indices - n_interior_faces
        # Boundary element indices for these faces
        b_elem_indices = n_elements + rel_indices

        for i_component in range(n_components):
            phi_b = phi[b_elem_indices, i_component]
            boundary_neighbours = np.asarray(
                mesh_data.get("boundary_neighbours", np.full(n_faces, -1, dtype=np.int32))
            )[valid_b_face_indices]
            coupled = boundary_neighbours >= 0
            if np.any(coupled):
                face_weights_b = face_weights[valid_b_face_indices[coupled]]
                owner_indices = owners[valid_b_face_indices[coupled]]
                phi_b[coupled] = (
                    face_weights_b * phi[boundary_neighbours[coupled], i_component]
                    + (1.0 - face_weights_b) * phi[owner_indices, i_component]
                )
            contribution_b = phi_b[:, np.newaxis] * sf_b[rel_indices]
            for dimension in range(3):
                grad_phi[:n_elements, dimension, i_component] += np.bincount(
                    owners_b[rel_indices],
                    weights=contribution_b[:, dimension],
                    minlength=n_elements,
                )

    # --- Volume Averaging (Vectorized) ---
    for i_component in range(n_components):
        grad_phi[:n_elements, :, i_component] /= element_volumes[:, np.newaxis]

    # --- Boundary Gradients ---
    # Element gradients at boundaries equal to owner cell gradients
    # This provides a zero-gradient condition for the gradient field,
    # ensuring continuity for extrapolation/interpolation (e.g. in Rhie-Chow)
    # We apply this to ALL boundary faces, including 'empty' ones.
    i_boundary_elements = np.arange(n_elements, n_elements + n_boundary_faces)
    boundary_neighbours = np.asarray(
        mesh_data.get("boundary_neighbours", np.full(n_faces, -1, dtype=np.int32))
    )[n_interior_faces:n_faces]
    gradient_owners = np.where(
        boundary_neighbours >= 0, boundary_neighbours, owners[n_interior_faces:n_faces]
    )
    grad_phi[i_boundary_elements, :, :] = grad_phi[gradient_owners, :, :]
    _correct_boundary_gradient(grad_phi, phi, mesh_data, geo_data)

    return grad_phi


def compute_lsq_geometry(mesh_data, geo_data):
    """Precompute inverse-distance LSQ stencils and 3×3 inverses.

    Distances are in metres. ``lsq_nei_w2_dr`` therefore has units 1/m and
    ``lsq_M_inv`` has units m². Rank-deficient stencils use a pseudoinverse.
    """
    n_elements = mesh_data["n_elements"]
    n_interior = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"]
    neighbours = mesh_data["neighbours"]
    boundary = mesh_data["boundary"]
    elem_c = geo_data["element_centroids"]
    face_c = geo_data["face_centroids"]

    owners_i = np.asarray(owners[:n_interior], dtype=np.int32)
    neighbours_i = np.asarray(neighbours[:n_interior], dtype=np.int32)
    dr_i = elem_c[neighbours_i] - elem_c[owners_i]

    owner_parts = [owners_i, neighbours_i]
    neighbour_parts = [neighbours_i, owners_i]
    displacement_parts = [dr_i, -dr_i]
    boundary_neighbours = np.asarray(
        mesh_data.get("boundary_neighbours", np.full(mesh_data["n_faces"], -1)),
        dtype=np.int32,
    )

    for patch in boundary:
        if _is_empty_boundary(patch, allow_source_type=True):
            continue
        start = patch["startFace"]
        faces = np.arange(start, start + patch["nFaces"], dtype=np.int64)
        patch_owners = np.asarray(owners[faces], dtype=np.int32)
        paired = boundary_neighbours[faces]
        coupled = paired >= 0
        neighbour_indices = np.empty(len(faces), dtype=np.int32)
        displacements = np.empty((len(faces), 3), dtype=np.float64)
        neighbour_indices[coupled] = paired[coupled]
        displacements[coupled] = geo_data["face_cf_vector"][faces[coupled]]
        uncoupled = ~coupled
        neighbour_indices[uncoupled] = n_elements + faces[uncoupled] - n_interior
        displacements[uncoupled] = face_c[faces[uncoupled]] - elem_c[patch_owners[uncoupled]]
        owner_parts.append(patch_owners)
        neighbour_parts.append(neighbour_indices)
        displacement_parts.append(displacements)

    owner_cell = np.concatenate(owner_parts)
    nei_phi_idx = np.concatenate(neighbour_parts)
    dr = np.concatenate(displacement_parts)
    distance_squared = np.einsum("ij,ij->i", dr, dr)
    w2 = 1.0 / np.maximum(distance_squared, 1e-60)
    nei_w2_dr = w2[:, np.newaxis] * dr

    sum_w2dr = np.column_stack(
        [
            np.bincount(owner_cell, weights=nei_w2_dr[:, axis], minlength=n_elements)
            for axis in range(3)
        ]
    )
    moment = np.empty((n_elements, 3, 3), dtype=np.float64)
    for row in range(3):
        for column in range(row, 3):
            values = np.bincount(
                owner_cell,
                weights=w2 * dr[:, row] * dr[:, column],
                minlength=n_elements,
            )
            moment[:, row, column] = values
            moment[:, column, row] = values

    singular_values = np.linalg.svd(moment, compute_uv=False, hermitian=True)
    tolerance = 3.0 * np.finfo(np.float64).eps * singular_values[:, 0]
    lsq_rank = np.sum(singular_values > tolerance[:, np.newaxis], axis=1).astype(np.int8)
    with np.errstate(divide="ignore", invalid="ignore"):
        lsq_condition = singular_values[:, 0] / singular_values[:, -1]
    lsq_condition[singular_values[:, -1] == 0.0] = np.inf

    well_conditioned = (lsq_rank == 3) & (lsq_condition <= _LSQ_QR_CONDITION_LIMIT)
    M_inv = np.empty_like(moment)
    if np.any(well_conditioned):
        q, r = np.linalg.qr(moment[well_conditioned])
        M_inv[well_conditioned] = np.linalg.solve(r, np.swapaxes(q, 1, 2))
    if np.any(~well_conditioned):
        M_inv[~well_conditioned] = np.linalg.pinv(moment[~well_conditioned])
    lsq_solver_method = np.where(well_conditioned, "qr", "svd")

    return {
        "lsq_nei_phi_idx": nei_phi_idx,
        "lsq_owner_cell": owner_cell,
        "lsq_nei_w2_dr": nei_w2_dr,
        "lsq_sum_w2dr": sum_w2dr,
        "lsq_M_inv": M_inv,
        "lsq_condition": lsq_condition,
        "lsq_rank": lsq_rank,
        "lsq_solver_method": lsq_solver_method,
        "gradient_scheme": "lsq",
    }


def compute_lsq_gradient(phi, mesh_data, geo_data):
    """Compute the inverse-distance-weighted least-squares gradient.

    For each cell minimises  Σ w²(φ_n − φ_c − ∇φ·dr)².
    """
    if phi.ndim == 1:
        phi = phi.reshape(-1, 1)

    n_total = phi.shape[0]
    n_components = phi.shape[1]
    n_elements = mesh_data["n_elements"]
    n_owned = int(mesh_data.get("_n_owned", n_elements))
    n_interior = mesh_data["n_interior_faces"]
    n_faces = mesh_data["n_faces"]
    n_boundary = n_faces - n_interior

    nei_phi_idx = geo_data["lsq_nei_phi_idx"]
    owner_cell = geo_data["lsq_owner_cell"]
    nei_w2_dr = geo_data["lsq_nei_w2_dr"]
    sum_w2dr = geo_data["lsq_sum_w2dr"]
    M_inv = geo_data["lsq_M_inv"]

    grad = np.zeros((n_total, 3, n_components), dtype=np.float64)

    owned_stencil = owner_cell < n_owned
    owner_owned = owner_cell[owned_stencil]
    nei_owned = nei_phi_idx[owned_stencil]
    w2dr_owned = nei_w2_dr[owned_stencil]

    for ic in range(n_components):
        # RHS = Σ w²·φ_n·dr  −  φ_c · Σ w²·dr
        phi_nei = phi[nei_owned, ic]
        rhs = np.column_stack(
            [
                np.bincount(
                    owner_owned,
                    weights=phi_nei * w2dr_owned[:, dimension],
                    minlength=n_owned,
                )
                for dimension in range(3)
            ]
        )
        rhs[:, 0] -= phi[:n_owned, ic] * sum_w2dr[:n_owned, 0]
        rhs[:, 1] -= phi[:n_owned, ic] * sum_w2dr[:n_owned, 1]
        rhs[:, 2] -= phi[:n_owned, ic] * sum_w2dr[:n_owned, 2]

        # grad = M⁻¹ · rhs   (vectorised 3×3 matmul)
        g = M_inv[:n_owned] @ rhs[..., np.newaxis]
        grad[:n_owned, :, ic] = g.squeeze(-1)

    parallel = mesh_data.get("_parallel_context")
    if parallel is not None and parallel.is_partitioned:
        parallel.exchange_halo(grad[:n_elements])

    # Boundary ghost cells: copy owner gradient (same convention as Gauss)
    owners_b = mesh_data["owners"][n_interior:n_faces]
    i_boundary = np.arange(n_elements, n_elements + n_boundary)
    boundary_neighbours = np.asarray(
        mesh_data.get("boundary_neighbours", np.full(n_faces, -1, dtype=np.int32))
    )[n_interior:n_faces]
    gradient_owners = np.where(boundary_neighbours >= 0, boundary_neighbours, owners_b)
    grad[i_boundary, :, :] = grad[gradient_owners, :, :]
    _correct_boundary_gradient(grad, phi, mesh_data, geo_data)

    return grad


def _resolve_gradient_fn(geo_data):
    """Return the gradient function matching the configured scheme.

    Checks ``geo_data["gradient_scheme"]``:
    - ``"lsq"`` → :func:`compute_lsq_gradient`
    - anything else → :func:`compute_gauss_gradient`

    Args:
        geo_data: Geometry dictionary (must contain ``"gradient_scheme"``).

    Returns:
        A callable ``grad_fn(phi, mesh_data, geo_data) -> gradient``.
    """
    if geo_data.get("gradient_scheme") == "lsq":
        return compute_lsq_gradient
    return compute_gauss_gradient
