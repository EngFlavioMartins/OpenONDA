#!/usr/bin/env python3
"""
Diffusion Term Assembly for OpenONDA FVM Solver

Implements diffusion term discretization using:
- Orthogonal/non-orthogonal decomposition
- Geometric diffusion coefficients
- Minimum correction approach

Converted from uFVM cfdAssembleDiffusionTerm.m
"""

import numpy as np


def assemble_diffusion_term_interior(phi, grad_phi, gamma, mesh_data, geo_data):
    """
    Assemble diffusion term for interior faces.

    Discretization: ∇·(γ∇φ) ≈ Σ_faces γ_f (∇φ)_f · S_f

    Uses orthogonal/non-orthogonal decomposition:
    - Sf = Ef + Tf
    - Ef = |Sf| * e (orthogonal component)
    - Tf = Sf - Ef (non-orthogonal component)

    Args:
        phi: Field values (n_elements,)
        grad_phi: Field gradient (n_elements, 3)
        gamma: Diffusion coefficient (n_elements,)
        mesh_data: Mesh connectivity
        geo_data: Geometric data

    Returns:
        dict: Flux coefficients
            - flux_cf: Owner coefficient (n_interior_faces,)
            - flux_ff: Neighbor coefficient (n_interior_faces,)
            - flux_vf: Explicit correction (n_interior_faces,)
            - flux_tf: Total flux (n_interior_faces,)
    """

    n_interior_faces = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"][:n_interior_faces]
    neighbours = mesh_data["neighbours"][:n_interior_faces]

    # Get geometric data
    sf = geo_data["face_sf"][:n_interior_faces]  # Face area vectors
    cf_vector = geo_data["face_cf_vector"][:n_interior_faces]  # Owner to neighbor vector

    # Compute geometric quantities
    # e = CF / |CF| (unit vector from owner to neighbor)
    mag_cf = np.linalg.norm(cf_vector, axis=1)
    e = cf_vector / mag_cf[:, np.newaxis]

    # |Sf|
    mag_sf = np.linalg.norm(sf, axis=1)

    # Orthogonal component: Ef = |Sf| * e
    ef = mag_sf[:, np.newaxis] * e

    # Non-orthogonal component: Tf = Sf - Ef
    tf = sf - ef

    # Geometric diffusion: |Ef| / |CF|
    mag_ef = np.linalg.norm(ef, axis=1)
    geo_diff = mag_ef / mag_cf

    # Interpolate gamma to faces (linear interpolation using geometric weights)
    weights = geo_data["face_weights"][:n_interior_faces]
    gamma_f = weights * gamma[neighbours] + (1 - weights) * gamma[owners]

    # Handle scalar field gradients: squeeze if shape is (n, 3, 1)
    if grad_phi.ndim == 3 and grad_phi.shape[2] == 1:
        grad_phi = grad_phi.squeeze(-1)  # (n, 3, 1) -> (n, 3)

    # Interpolate element gradients to faces
    # Standard interpolation: grad_f = w * grad_neighbor + (1-w) * grad_owner
    grad_phi_f = (
        weights[:, np.newaxis] * grad_phi[neighbours]
        + (1 - weights[:, np.newaxis]) * grad_phi[owners]
    )

    # Linear flux coefficients
    # FluxCf: coefficient for owner cell
    # FluxFf: coefficient for neighbor cell
    flux_cf = gamma_f * geo_diff
    flux_ff = -gamma_f * geo_diff

    # Non-linear flux (explicit correction for non-orthogonality)
    # FluxVf = -gamma_f * (grad_phi_f · Tf)
    flux_vf = -gamma_f * np.sum(grad_phi_f * tf, axis=1)

    # Total flux: FluxTf = FluxCf * phi_owner + FluxFf * phi_neighbor + FluxVf
    flux_tf = flux_cf * phi[owners] + flux_ff * phi[neighbours] + flux_vf

    return {"flux_cf": flux_cf, "flux_ff": flux_ff, "flux_vf": flux_vf, "flux_tf": flux_tf}


def assemble_diffusion_term_boundary_fixed_value(phi, gamma, boundary_patch, mesh_data, geo_data):
    """
    Assemble diffusion term for boundary faces with fixed value BC.

    For Dirichlet BC: φ_boundary = φ_prescribed

    Args:
        phi: Field values including boundary elements
        gamma: Diffusion coefficient
        boundary_patch: Dict with patch info (start_face, n_faces, value)
        mesh_data: Mesh connectivity
        geo_data: Geometric data

    Returns:
        dict: Flux coefficients for this boundary patch
    """

    n_interior_faces = mesh_data["n_interior_faces"]
    n_elements = mesh_data["n_elements"]

    # Get boundary patch info
    start_face = boundary_patch["startFace"]
    n_faces = boundary_patch["nFaces"]
    end_face = start_face + n_faces

    # Boundary face indices
    b_face_indices = np.arange(start_face, end_face)

    # Owner cells for these boundary faces
    owners_b = mesh_data["owners"][b_face_indices]

    # Boundary element indices
    # In uFVM: boundary elements are stored after interior elements
    # Element index = n_elements + (face_index - n_interior_faces)
    b_elem_start = start_face - n_interior_faces
    b_elem_indices = np.arange(n_elements + b_elem_start, n_elements + b_elem_start + n_faces)

    # Get geometric data for boundary faces
    sf_b = geo_data["face_sf"][b_face_indices]
    wall_dist = geo_data["wall_dist"][b_face_indices]

    # Interpolate gamma to boundary
    gamma_b = gamma[owners_b]  # Use owner cell value

    # Geometric diffusion for boundary: |Sf| / wall_distance
    mag_sf_b = np.linalg.norm(sf_b, axis=1)
    geo_diff_b = mag_sf_b / wall_dist

    # Get boundary and owner values
    phi_b = phi[b_elem_indices]
    phi_c = phi[owners_b]

    # Linear flux coefficients
    # For Dirichlet BC: flux = gamma * (phi_b - phi_c) / dist
    # Matrix: A[owner, owner] += gamma * geo_diff
    # RHS: b[owner] -= gamma * geo_diff * phi_b (known boundary value)
    flux_cf = gamma_b * geo_diff_b
    flux_ff = np.zeros_like(flux_cf)  # No neighbor contribution (BC is fixed)

    # Explicit correction: contribution from known boundary value
    # This goes to RHS: b[owner] -= flux_vf
    # So flux_vf = -gamma * geo_diff * phi_b
    flux_vf = -gamma_b * geo_diff_b * phi_b

    # Total flux (for reference)
    flux_tf = flux_cf * phi_c + flux_vf

    return {
        "flux_cf": flux_cf,
        "flux_ff": flux_ff,
        "flux_vf": flux_vf,
        "flux_tf": flux_tf,
        "face_indices": b_face_indices,
    }


def assemble_diffusion_term(phi, grad_phi, gamma, mesh_data, geo_data, boundaries):
    """
    Assemble complete diffusion term for all faces.

    Args:
        phi: Field values (n_elements + n_boundary_elements,)
        grad_phi: Field gradient (n_elements, 3)
        gamma: Diffusion coefficient (n_elements,)
        mesh_data: Mesh connectivity
        geo_data: Geometric data
        boundaries: List of boundary patch dictionaries

    Returns:
        dict: Complete flux data for all faces
    """

    n_faces = mesh_data["n_faces"]

    # Initialize flux arrays
    flux_cf = np.zeros(n_faces)
    flux_ff = np.zeros(n_faces)
    flux_vf = np.zeros(n_faces)
    flux_tf = np.zeros(n_faces)

    # Assemble interior faces
    interior_fluxes = assemble_diffusion_term_interior(phi, grad_phi, gamma, mesh_data, geo_data)

    n_interior = mesh_data["n_interior_faces"]
    flux_cf[:n_interior] = interior_fluxes["flux_cf"]
    flux_ff[:n_interior] = interior_fluxes["flux_ff"]
    flux_vf[:n_interior] = interior_fluxes["flux_vf"]
    flux_tf[:n_interior] = interior_fluxes["flux_tf"]

    # Assemble boundary faces
    for boundary in boundaries:
        bc_type = boundary.get("bc_type", "zeroGradient")
        if bc_type == "empty" or boundary.get("type") == "empty":
            # Empty BC: no flux contribution
            continue

        elif bc_type in ["fixedValue", "Dirichlet", "noSlip"]:
            if bc_type == "noSlip":
                # Force zero value for noSlip regardless of field content
                # We need to ensure phi[b_elem] is 0 or passed correctly.
                # Actually, phi_b is read from phi[b_elem_indices].
                # For noSlip, it should be 0.
                pass

            b_fluxes = assemble_diffusion_term_boundary_fixed_value(
                phi, gamma, boundary, mesh_data, geo_data
            )

            indices = b_fluxes["face_indices"]
            flux_cf[indices] = b_fluxes["flux_cf"]
            flux_ff[indices] = b_fluxes["flux_ff"]
            flux_vf[indices] = b_fluxes["flux_vf"]
            flux_tf[indices] = b_fluxes["flux_tf"]

        elif bc_type == "zeroGradient":
            # Zero gradient: no flux contribution
            pass

        # Add more BC types as needed

    return {"flux_cf": flux_cf, "flux_ff": flux_ff, "flux_vf": flux_vf, "flux_tf": flux_tf}
