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

    # Over-relaxed orthogonal decomposition  Sf = Ef + Tf,  Ef ∥ e,
    # |Ef| = (Sf·Sf)/(Sf·e).  This keeps the implicit Laplacian aligned with the
    # owner→neighbour line (best conditioning / accuracy on non-orthogonal
    # meshes) and pushes the residual into the explicit Tf correction.  On an
    # orthogonal mesh Sf·e = |Sf| ⇒ Ef = Sf, Tf = 0, recovering the exact result.
    sf_dot_e = np.sum(sf * e, axis=1)
    sf_dot_e = np.where(np.abs(sf_dot_e) < 1e-30, 1e-30, sf_dot_e)
    mag_sf2 = np.sum(sf * sf, axis=1)
    ef_mag = mag_sf2 / sf_dot_e  # = |Ef|

    # Non-orthogonal component: Tf = Sf - Ef
    ef = ef_mag[:, np.newaxis] * e
    tf = sf - ef

    # Geometric diffusion: |Ef| / |CF|
    geo_diff = ef_mag / mag_cf

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


def assemble_diffusion_term(phi, grad_phi, gamma, mesh_data, geo_data, boundaries, face_flux=None):
    """
    Assemble complete diffusion term for all faces.

    Args:
        phi: Field values (n_elements + n_boundary_elements,)
        grad_phi: Field gradient (n_elements, 3)
        gamma: Diffusion coefficient (n_elements,)
        mesh_data: Mesh connectivity
        geo_data: Geometric data
        boundaries: List of boundary patch dictionaries.
        face_flux: Optional signed face flux.  Required to distinguish the
            inflow (Dirichlet) and outflow (zero-gradient) portions of an
            ``inletOutlet`` patch.

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
        if bc_type in ("empty", "slip", "symmetry") or boundary.get("type") == "empty":
            # Empty/slip/symmetry: zero diffusive flux through the plane (the
            # ghost value mirrors the tangential velocity, so the face-normal
            # gradient vanishes).
            continue

        elif bc_type == "cyclic":
            start = boundary["startFace"]
            indices = np.arange(start, start + boundary["nFaces"])
            owners_b = mesh_data["owners"][indices]
            neighbours_b = mesh_data["boundary_neighbours"][indices]
            if np.any(neighbours_b < 0):
                raise ValueError("Cyclic diffusion faces are missing paired owner cells")
            sf = geo_data["face_sf"][indices]
            cf_vector = geo_data["face_cf_vector"][indices]
            mag_cf = np.linalg.norm(cf_vector, axis=1)
            e = cf_vector / mag_cf[:, None]
            sf_dot_e = np.sum(sf * e, axis=1)
            sf_dot_e = np.where(np.abs(sf_dot_e) < 1e-30, 1e-30, sf_dot_e)
            ef_mag = np.sum(sf * sf, axis=1) / sf_dot_e
            tf = sf - ef_mag[:, None] * e
            weights = geo_data["face_weights"][indices]
            gamma_f = weights * gamma[neighbours_b] + (1.0 - weights) * gamma[owners_b]
            grad = grad_phi.squeeze(-1) if grad_phi.ndim == 3 else grad_phi
            grad_f = (
                weights[:, None] * grad[neighbours_b] + (1.0 - weights[:, None]) * grad[owners_b]
            )
            coefficient = gamma_f * ef_mag / mag_cf
            flux_cf[indices] = coefficient
            flux_ff[indices] = -coefficient
            flux_vf[indices] = -gamma_f * np.sum(grad_f * tf, axis=1)
            flux_tf[indices] = (
                coefficient * phi[owners_b] - coefficient * phi[neighbours_b] + flux_vf[indices]
            )

        elif bc_type in ["fixedValue", "Dirichlet", "noSlip", "directionMixed"]:
            b_fluxes = assemble_diffusion_term_boundary_fixed_value(
                phi, gamma, boundary, mesh_data, geo_data
            )

            if bc_type == "noSlip":
                # Enforce the mathematical value independently of ghost state.
                indices = b_fluxes["face_indices"]
                owners_b = mesh_data["owners"][indices]
                b_fluxes["flux_vf"][:] = 0.0
                b_fluxes["flux_tf"][:] = b_fluxes["flux_cf"] * phi[owners_b]

            indices = b_fluxes["face_indices"]
            flux_cf[indices] = b_fluxes["flux_cf"]
            flux_ff[indices] = b_fluxes["flux_ff"]
            flux_vf[indices] = b_fluxes["flux_vf"]
            flux_tf[indices] = b_fluxes["flux_tf"]

        elif bc_type in {"inletOutlet", "freestream"}:
            # Diffusion is Dirichlet only on reverse-flow/inflow faces; it is
            # zero-gradient on outflow.  Without a signed face flux, retain the
            # conservative zero-gradient behavior.
            if face_flux is None:
                continue
            b_fluxes = assemble_diffusion_term_boundary_fixed_value(
                phi, gamma, boundary, mesh_data, geo_data
            )
            indices = b_fluxes["face_indices"]
            inflow = np.asarray(face_flux)[indices] < 0.0
            flux_cf[indices] = np.where(inflow, b_fluxes["flux_cf"], 0.0)
            flux_ff[indices] = 0.0
            flux_vf[indices] = np.where(inflow, b_fluxes["flux_vf"], 0.0)
            flux_tf[indices] = np.where(inflow, b_fluxes["flux_tf"], 0.0)

        elif bc_type == "zeroGradient":
            # Zero gradient: no flux contribution
            pass

        else:
            raise ValueError(f"Unsupported diffusion boundary condition: {bc_type!r}")

    return {"flux_cf": flux_cf, "flux_ff": flux_ff, "flux_vf": flux_vf, "flux_tf": flux_tf}
