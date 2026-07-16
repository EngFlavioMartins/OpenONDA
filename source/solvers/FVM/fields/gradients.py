#!/usr/bin/env python3
"""
Gradient Computation for OpenONDA FVM Solver

Implements Gauss linear gradient computation using Green-Gauss theorem.

Converted from uFVM cfdComputeGradientGaussLinear0.m
"""

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


def _accumulate_interior_gradients(
    grad_phi, phi, owners, neighbours, face_sf, face_weights, n_interior_faces, n_components
):
    """Accumulate interior-face flux contributions into the cell gradient.

    For each interior face, interpolates *phi* to the face using
    geometric weights, then adds ``phi_f · S_f`` to the owner and
    subtracts it from the neighbour (the Gauss theorem contribution).

    Args:
        grad_phi:      Gradient accumulator array (mutated in place).
        phi:           Field values ``(n_total, n_components)``.
        owners:        Owner index array.
        neighbours:    Neighbour index array.
        face_sf:       Face area vectors ``(n_faces, 3)``.
        face_weights:  Interpolation weights ``(n_faces,)``.
        n_interior_faces: Number of interior faces.
        n_components:  Number of field components (1 for scalar, 3 for vector).
    """
    for i_component in range(n_components):
        phi_owner = phi[owners[:n_interior_faces], i_component]
        phi_neighbor = phi[neighbours[:n_interior_faces], i_component]
        phi_f = (
            face_weights[:n_interior_faces] * phi_neighbor
            + (1 - face_weights[:n_interior_faces]) * phi_owner
        )
        for i_face in range(n_interior_faces):
            sf = face_sf[i_face]
            grad_phi[owners[i_face], :, i_component] += phi_f[i_face] * sf
            grad_phi[neighbours[i_face], :, i_component] -= phi_f[i_face] * sf


def _accumulate_boundary_gradients(
    grad_phi,
    phi,
    owners_b,
    sf_b,
    boundaries,
    n_interior_faces,
    n_elements,
    n_components,
    boundary_neighbours,
    face_weights,
):
    """Accumulate boundary-face flux contributions into the cell gradient.

    Skips patches with type ``"empty"`` (2D front/back).  For each
    remaining boundary face, adds the boundary value times the area
    vector to the owner cell's gradient.

    Args:
        grad_phi:      Gradient accumulator array (mutated in place).
        phi:           Field values ``(n_total, n_components)``.
        owners_b:      Owner array for boundary faces ``(n_boundary_faces,)``.
        sf_b:          Face area vectors for boundary faces.
        boundaries:    List of boundary patch dictionaries.
        n_interior_faces: Number of interior faces.
        n_elements:    Number of interior elements.
        n_components:  Number of field components.
    """
    for boundary in boundaries:
        if _is_empty_boundary(boundary):
            continue
        start = boundary["startFace"]
        nf = boundary["nFaces"]
        rel_start = start - n_interior_faces
        rel_end = rel_start + nf
        for i_component in range(n_components):
            for k in range(rel_start, rel_end):
                face = n_interior_faces + k
                paired = boundary_neighbours[face]
                if paired >= 0:
                    weight = face_weights[face]
                    face_value = (
                        weight * phi[paired, i_component]
                        + (1.0 - weight) * phi[owners_b[k], i_component]
                    )
                else:
                    face_value = phi[n_elements + k, i_component]
                grad_phi[owners_b[k], :, i_component] += face_value * sf_b[k]


def compute_gradient_gauss_linear(phi, mesh_data, geo_data):
    """
    Compute gradient using Gauss linear method (Green-Gauss theorem).

    Algorithm:
    1. Interpolate field to faces using geometric weights
    2. Accumulate face contributions: grad += phi_f * Sf
    3. Divide by element volume

    Args:
        phi: Field values (n_elements + n_boundary_elements, [n_components])
            For scalar: (N,) or (N, 1)
            For vector: (N, 3)
        mesh_data: Dictionary with mesh connectivity
        geo_data: Dictionary with geometric data

    Returns:
        grad_phi: Gradient field (n_elements + n_boundary_elements, 3, n_components)
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

    # Initialize gradient array
    grad_phi = np.zeros((n_total, 3, n_components), dtype=np.float64)

    _accumulate_interior_gradients(
        grad_phi, phi, owners, neighbours, face_sf, face_weights, n_interior_faces, n_components
    )

    owners_b = owners[n_interior_faces:n_faces]
    sf_b = geo_data["face_sf"][n_interior_faces:n_faces]
    _accumulate_boundary_gradients(
        grad_phi,
        phi,
        owners_b,
        sf_b,
        mesh_data["boundary"],
        n_interior_faces,
        n_elements,
        n_components,
        np.asarray(mesh_data.get("boundary_neighbours", np.full(n_faces, -1))),
        face_weights,
    )

    # Volume-average the cell gradients (vectorized)
    grad_phi[:n_elements] /= element_volumes[:, np.newaxis, np.newaxis]
    parallel = mesh_data.get("_parallel_context")
    if parallel is not None and parallel.is_partitioned:
        parallel.exchange_halo(grad_phi[:n_elements])

    # Boundary element gradients equal owner cell gradients
    i_boundary_elements = np.arange(n_elements, n_elements + n_boundary_faces)
    boundary_neighbours = np.asarray(
        mesh_data.get("boundary_neighbours", np.full(n_faces, -1, dtype=np.int32))
    )[n_interior_faces:n_faces]
    gradient_owners = np.where(boundary_neighbours >= 0, boundary_neighbours, owners_b)
    grad_phi[i_boundary_elements, :, :] = grad_phi[gradient_owners, :, :]

    return grad_phi


def compute_gradient_gauss_linear_vectorized(phi, mesh_data, geo_data):
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

        # Collect indices for setting boundary gradients later
        owners_b[rel_indices]

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

    return grad_phi


# ═══════════════════════════════════════════════════════════════════════
# Least‑Squares Gradient (inverse‑distance weighted)
# ═══════════════════════════════════════════════════════════════════════


def compute_lsq_geometry(mesh_data, geo_data):
    """
    Pre‑compute LSQ gradient geometry data.

    For each cell, builds a stencil of neighbour cells and boundary faces,
    pre‑computes inverse‑distance weights and the 3×3 moment matrix inverse.

    Returns a dict of flat arrays for vectorised RHS assembly:

        lsq_nei_phi_idx   — index into phi array for each stencil point
        lsq_owner_cell    — owning cell for each stencil point
        lsq_nei_w2_dr     — w² · dr  (3‑vector per stencil point)
        lsq_sum_w2dr      — Σ w² · dr  per cell (3‑vector)
        lsq_M_inv         — M⁻¹ per cell (n, 3, 3)
        gradient_scheme   — "lsq" flag
    """
    n_elements = mesh_data["n_elements"]
    n_interior = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"]
    neighbours = mesh_data["neighbours"]
    boundary = mesh_data["boundary"]
    elem_c = geo_data["element_centroids"]
    face_c = geo_data["face_centroids"]

    stencil_nei = [[] for _ in range(n_elements)]
    stencil_dr = [[] for _ in range(n_elements)]

    # Interior faces: each face connects owner ↔ neighbour
    for face in range(n_interior):
        own = owners[face]
        nei = neighbours[face]
        dr = elem_c[nei] - elem_c[own]
        stencil_nei[own].append(nei)
        stencil_dr[own].append(dr)
        stencil_nei[nei].append(own)
        stencil_dr[nei].append(-dr)

    # Boundary faces (exclude empty patches)
    for patch in boundary:
        if _is_empty_boundary(patch, allow_source_type=True):
            continue
        start = patch["startFace"]
        nf = patch["nFaces"]
        for j in range(nf):
            face_idx = start + j
            own = owners[face_idx]
            boundary_neighbours = mesh_data.get("boundary_neighbours")
            if boundary_neighbours is not None and boundary_neighbours[face_idx] >= 0:
                stencil_nei[own].append(int(boundary_neighbours[face_idx]))
                stencil_dr[own].append(geo_data["face_cf_vector"][face_idx])
                continue
            bf_idx = face_idx - n_interior  # 0‑based boundary‑face number
            phi_idx = n_elements + bf_idx  # position in the phi array
            dr = face_c[face_idx] - elem_c[own]
            stencil_nei[own].append(-phi_idx - 1)  # negative ⇒ boundary
            stencil_dr[own].append(dr)

    # Build flat CSR‑like arrays
    offsets = np.zeros(n_elements + 1, dtype=np.int64)
    for c in range(n_elements):
        offsets[c + 1] = offsets[c] + len(stencil_nei[c])
    total = int(offsets[-1])

    nei_phi_idx = np.zeros(total, dtype=np.int32)
    owner_cell = np.zeros(total, dtype=np.int32)
    nei_w2_dr = np.zeros((total, 3), dtype=np.float64)
    sum_w2dr = np.zeros((n_elements, 3), dtype=np.float64)
    M_inv = np.zeros((n_elements, 3, 3), dtype=np.float64)
    lsq_condition = np.zeros(n_elements, dtype=np.float64)
    lsq_rank = np.zeros(n_elements, dtype=np.int8)
    lsq_solver_method = np.empty(n_elements, dtype="U3")

    for c in range(n_elements):
        s, e = int(offsets[c]), int(offsets[c + 1])
        M = np.zeros((3, 3))
        for j in range(s, e):
            k = j - s
            raw_idx = stencil_nei[c][k]
            dr = stencil_dr[c][k]
            w2 = 1.0 / max(dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2], 1e-60)

            if raw_idx >= 0:
                nei_phi_idx[j] = raw_idx
            else:
                nei_phi_idx[j] = -raw_idx - 1  # back to positive phi index
            owner_cell[j] = c
            nei_w2_dr[j] = w2 * dr
            sum_w2dr[c] += w2 * dr

            M[0, 0] += w2 * dr[0] * dr[0]
            M[0, 1] += w2 * dr[0] * dr[1]
            M[0, 2] += w2 * dr[0] * dr[2]
            M[1, 1] += w2 * dr[1] * dr[1]
            M[1, 2] += w2 * dr[1] * dr[2]
            M[2, 2] += w2 * dr[2] * dr[2]
        M[1, 0] = M[0, 1]
        M[2, 0] = M[0, 2]
        M[2, 1] = M[1, 2]

        lsq_condition[c] = np.linalg.cond(M)
        lsq_rank[c] = np.linalg.matrix_rank(M)
        if lsq_rank[c] == 3 and lsq_condition[c] <= _LSQ_QR_CONDITION_LIMIT:
            q, r = np.linalg.qr(M)
            M_inv[c] = np.linalg.solve(r, q.T)
            lsq_solver_method[c] = "qr"
        else:
            # SVD is deliberate for rank-deficient 2D stencils and severely
            # conditioned 3D cells; it returns the minimum-norm gradient.
            M_inv[c] = np.linalg.pinv(M)
            lsq_solver_method[c] = "svd"

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


def compute_gradient_lsq_vectorized(phi, mesh_data, geo_data):
    """
    Inverse‑distance‑weighted least‑squares gradient.

    For each cell minimises  Σ w²(φ_n − φ_c − ∇φ·dr)².

    Signature is identical to compute_gradient_gauss_linear_vectorized
    so the two are drop‑in replacements.
    """
    if phi.ndim == 1:
        phi = phi.reshape(-1, 1)

    n_total = phi.shape[0]
    n_components = phi.shape[1]
    n_elements = mesh_data["n_elements"]
    n_interior = mesh_data["n_interior_faces"]
    n_faces = mesh_data["n_faces"]
    n_boundary = n_faces - n_interior

    nei_phi_idx = geo_data["lsq_nei_phi_idx"]
    owner_cell = geo_data["lsq_owner_cell"]
    nei_w2_dr = geo_data["lsq_nei_w2_dr"]
    sum_w2dr = geo_data["lsq_sum_w2dr"]
    M_inv = geo_data["lsq_M_inv"]

    grad = np.zeros((n_total, 3, n_components), dtype=np.float64)

    for ic in range(n_components):
        # RHS = Σ w²·φ_n·dr  −  φ_c · Σ w²·dr
        phi_nei = phi[nei_phi_idx, ic]
        rhs = np.column_stack(
            [
                np.bincount(
                    owner_cell,
                    weights=phi_nei * nei_w2_dr[:, dimension],
                    minlength=n_elements,
                )
                for dimension in range(3)
            ]
        )
        rhs[:, 0] -= phi[:n_elements, ic] * sum_w2dr[:, 0]
        rhs[:, 1] -= phi[:n_elements, ic] * sum_w2dr[:, 1]
        rhs[:, 2] -= phi[:n_elements, ic] * sum_w2dr[:, 2]

        # grad = M⁻¹ · rhs   (vectorised 3×3 matmul)
        g = M_inv @ rhs[..., np.newaxis]  # (n, 3, 1)
        grad[:n_elements, :, ic] = g.squeeze(-1)

    # Boundary ghost cells: copy owner gradient (same convention as Gauss)
    owners_b = mesh_data["owners"][n_interior:n_faces]
    i_boundary = np.arange(n_elements, n_elements + n_boundary)
    boundary_neighbours = np.asarray(
        mesh_data.get("boundary_neighbours", np.full(n_faces, -1, dtype=np.int32))
    )[n_interior:n_faces]
    gradient_owners = np.where(boundary_neighbours >= 0, boundary_neighbours, owners_b)
    grad[i_boundary, :, :] = grad[gradient_owners, :, :]

    return grad


def _resolve_gradient_fn(geo_data):
    """Return the gradient function matching the configured scheme.

    Checks ``geo_data["gradient_scheme"]``:
    - ``"lsq"`` → :func:`compute_gradient_lsq_vectorized`
    - anything else → :func:`compute_gradient_gauss_linear_vectorized`

    Args:
        geo_data: Geometry dictionary (must contain ``"gradient_scheme"``).

    Returns:
        A callable ``grad_fn(phi, mesh_data, geo_data) -> gradient``.
    """
    if geo_data.get("gradient_scheme") == "lsq":
        return compute_gradient_lsq_vectorized
    return compute_gradient_gauss_linear_vectorized
