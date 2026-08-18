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

from ..schemes.boundaries import BOUNDARIES, BoundaryStrategy


def assemble_diffusion_term_interior(
    phi, grad_phi, gamma, mesh_data, geo_data, *, include_total_flux=True
):
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
    # Handle scalar field gradients: squeeze if shape is (n, 3, 1)
    if grad_phi.ndim == 3 and grad_phi.shape[2] == 1:
        grad_phi = grad_phi.squeeze(-1)  # (n, 3, 1) -> (n, 3)

    owners_all = mesh_data["owners"]
    neighbours_all = mesh_data["neighbours"]
    flux_cf = np.empty(n_interior_faces, dtype=np.float64)
    flux_ff = np.empty(n_interior_faces, dtype=np.float64)
    flux_vf = np.empty(n_interior_faces, dtype=np.float64)
    flux_tf = np.empty(n_interior_faces, dtype=np.float64) if include_total_flux else None

    # Keep the temporary edge/gradient tensors bounded independently of mesh
    # size.  On the cube reference partition the all-face implementation held
    # roughly 200 MiB here for each velocity component.
    chunk_size = 250_000
    for start in range(0, n_interior_faces, chunk_size):
        stop = min(start + chunk_size, n_interior_faces)
        face_slice = slice(start, stop)
        owners = owners_all[face_slice]
        neighbours = neighbours_all[face_slice]
        sf = geo_data["face_sf"][face_slice]
        cf_vector = geo_data["face_cf_vector"][face_slice]
        mag_cf = np.linalg.norm(cf_vector, axis=1)
        edge = cf_vector / mag_cf[:, np.newaxis]
        sf_dot_edge = np.sum(sf * edge, axis=1)
        sf_dot_edge = np.where(np.abs(sf_dot_edge) < 1e-30, 1e-30, sf_dot_edge)
        ef_mag = np.sum(sf * sf, axis=1) / sf_dot_edge
        nonorthogonal = sf - ef_mag[:, np.newaxis] * edge
        geo_diff = ef_mag / mag_cf

        weights = geo_data["face_weights"][face_slice]
        if np.isscalar(gamma):
            gamma_f = float(np.asarray(gamma).item())
        else:
            gamma_f = weights * gamma[neighbours] + (1.0 - weights) * gamma[owners]
        grad_face = (
            weights[:, np.newaxis] * grad_phi[neighbours]
            + (1.0 - weights[:, np.newaxis]) * grad_phi[owners]
        )
        coefficient = gamma_f * geo_diff
        correction = -gamma_f * np.sum(grad_face * nonorthogonal, axis=1)
        flux_cf[face_slice] = coefficient
        flux_ff[face_slice] = -coefficient
        flux_vf[face_slice] = correction
        if flux_tf is not None:
            flux_tf[face_slice] = (
                coefficient * phi[owners] - coefficient * phi[neighbours] + correction
            )

    result = {"flux_cf": flux_cf, "flux_ff": flux_ff, "flux_vf": flux_vf}
    if flux_tf is not None:
        result["flux_tf"] = flux_tf
    return result


def assemble_diffusion_term_boundary_fixed_value(
    phi, gamma, boundary_patch, mesh_data, geo_data, *, include_total_flux=True
):
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
    gamma_b = float(np.asarray(gamma).item()) if np.isscalar(gamma) else gamma[owners_b]

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
    result = {
        "flux_cf": flux_cf,
        "flux_ff": flux_ff,
        "flux_vf": flux_vf,
        "face_indices": b_face_indices,
    }
    if include_total_flux:
        result["flux_tf"] = flux_cf * phi_c + flux_vf
    return result


def assemble_diffusion_term(
    phi,
    grad_phi,
    gamma,
    mesh_data,
    geo_data,
    boundaries,
    face_flux=None,
    vector_field=None,
    component=None,
    *,
    include_total_flux=True,
):
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
        vector_field: Full vector velocity field, required by the directional
            mixed velocity condition.
        component: Component assembled from ``vector_field`` (0, 1, or 2).

    Returns:
        dict: Complete flux data for all faces
    """

    n_faces = mesh_data["n_faces"]

    # Initialize flux arrays
    flux_cf = np.zeros(n_faces)
    flux_ff = np.zeros(n_faces)
    flux_vf = np.zeros(n_faces)
    flux_tf = np.zeros(n_faces) if include_total_flux else None

    # Assemble interior faces
    interior_fluxes = assemble_diffusion_term_interior(
        phi,
        grad_phi,
        gamma,
        mesh_data,
        geo_data,
        include_total_flux=include_total_flux,
    )

    n_interior = mesh_data["n_interior_faces"]
    flux_cf[:n_interior] = interior_fluxes["flux_cf"]
    flux_ff[:n_interior] = interior_fluxes["flux_ff"]
    flux_vf[:n_interior] = interior_fluxes["flux_vf"]
    if flux_tf is not None:
        flux_tf[:n_interior] = interior_fluxes["flux_tf"]

    # Assemble boundary faces
    for boundary in boundaries:
        bc_type = boundary.get("bc_type")
        if boundary.get("bc_type_U") is not None:
            strategy = BOUNDARIES.strategy(boundary["bc_type_U"], "U", "diffusion")
        else:
            strategy = BOUNDARIES.strategy(bc_type, "scalar", "diffusion")
        if strategy in (
            BoundaryStrategy.EMPTY,
            BoundaryStrategy.SLIP,
            BoundaryStrategy.SYMMETRY,
        ):
            # Empty/slip/symmetry: zero diffusive flux through the plane (the
            # ghost value mirrors the tangential velocity, so the face-normal
            # gradient vanishes).
            continue

        elif strategy is BoundaryStrategy.CYCLIC:
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
            gamma_f = (
                float(np.asarray(gamma).item())
                if np.isscalar(gamma)
                else weights * gamma[neighbours_b] + (1.0 - weights) * gamma[owners_b]
            )
            grad = grad_phi.squeeze(-1) if grad_phi.ndim == 3 else grad_phi
            grad_f = (
                weights[:, None] * grad[neighbours_b] + (1.0 - weights[:, None]) * grad[owners_b]
            )
            coefficient = gamma_f * ef_mag / mag_cf
            flux_cf[indices] = coefficient
            flux_ff[indices] = -coefficient
            flux_vf[indices] = -gamma_f * np.sum(grad_f * tf, axis=1)
            if flux_tf is not None:
                flux_tf[indices] = (
                    coefficient * phi[owners_b] - coefficient * phi[neighbours_b] + flux_vf[indices]
                )

        elif strategy in (
            BoundaryStrategy.FIXED_VALUE,
            BoundaryStrategy.NO_SLIP,
        ):
            b_fluxes = assemble_diffusion_term_boundary_fixed_value(
                phi,
                gamma,
                boundary,
                mesh_data,
                geo_data,
                include_total_flux=include_total_flux,
            )

            if strategy is BoundaryStrategy.NO_SLIP:
                # Enforce the mathematical value independently of ghost state.
                indices = b_fluxes["face_indices"]
                owners_b = mesh_data["owners"][indices]
                b_fluxes["flux_vf"][:] = 0.0
                if include_total_flux:
                    b_fluxes["flux_tf"][:] = b_fluxes["flux_cf"] * phi[owners_b]

            indices = b_fluxes["face_indices"]
            flux_cf[indices] = b_fluxes["flux_cf"]
            flux_ff[indices] = b_fluxes["flux_ff"]
            flux_vf[indices] = b_fluxes["flux_vf"]
            if flux_tf is not None:
                flux_tf[indices] = b_fluxes["flux_tf"]

        elif strategy in (BoundaryStrategy.INLET_OUTLET, BoundaryStrategy.FREESTREAM):
            # Diffusion is Dirichlet only on reverse-flow/inflow faces; it is
            # zero-gradient on outflow.  Without a signed face flux, retain the
            # conservative zero-gradient behavior.
            if face_flux is None:
                continue
            b_fluxes = assemble_diffusion_term_boundary_fixed_value(
                phi,
                gamma,
                boundary,
                mesh_data,
                geo_data,
                include_total_flux=include_total_flux,
            )
            indices = b_fluxes["face_indices"]
            inflow = np.asarray(face_flux)[indices] < 0.0
            flux_cf[indices] = np.where(inflow, b_fluxes["flux_cf"], 0.0)
            flux_ff[indices] = 0.0
            flux_vf[indices] = np.where(inflow, b_fluxes["flux_vf"], 0.0)
            if flux_tf is not None:
                flux_tf[indices] = np.where(inflow, b_fluxes["flux_tf"], 0.0)

        elif strategy is BoundaryStrategy.NORMAL_VALUE_TANGENTIAL_GRADIENT:
            if vector_field is None or component not in (0, 1, 2):
                raise ValueError(
                    "normalValueTangentialGradient diffusion requires the full velocity "
                    "field and a component index"
                )
            start = int(boundary["startFace"])
            n_patch_faces = int(boundary["nFaces"])
            indices = np.arange(start, start + n_patch_faces)
            owners_b = mesh_data["owners"][indices]
            sf = np.asarray(geo_data["face_sf"], dtype=np.float64)[indices]
            area = np.linalg.norm(sf, axis=1)
            if np.any(area <= 1.0e-14):
                raise ValueError("Mixed diffusion boundary contains a degenerate face")
            normals = sf / area[:, np.newaxis]
            owner_to_face = np.asarray(geo_data["face_cf_vector"], dtype=np.float64)[indices]
            distance = np.einsum("ij,ij->i", owner_to_face, normals)
            if np.any(distance <= 1.0e-14):
                raise ValueError("Mixed diffusion boundary requires positive normal distance")

            gamma_b = (
                float(np.asarray(gamma).item())
                if np.isscalar(gamma)
                else np.asarray(gamma, dtype=np.float64)[owners_b]
            )
            diffusivity_area = gamma_b * area
            coefficient = diffusivity_area / distance
            n_i = normals[:, component]
            owner_velocity = np.asarray(vector_field, dtype=np.float64)[owners_b]
            owner_normal = np.einsum("ij,ij->i", owner_velocity, normals)
            cross_normal = owner_normal - n_i * owner_velocity[:, component]
            prescribed_normal = np.asarray(boundary["normal_velocity_field"], dtype=np.float64)
            prescribed_gradient = np.asarray(
                boundary["tangential_gradient_field"], dtype=np.float64
            )[:, component]

            # F_i = D n_i (n.U_P - u_n) - nu A g_t,i.  The owner-component
            # part is implicit; cross-components remain explicit and converge
            # through the existing PIMPLE outer iterations on non-axis-aligned
            # interfaces.
            flux_cf[indices] = coefficient * n_i * n_i
            flux_ff[indices] = 0.0
            flux_vf[indices] = (
                coefficient * n_i * (cross_normal - prescribed_normal)
                - diffusivity_area * prescribed_gradient
            )
            if flux_tf is not None:
                flux_tf[indices] = flux_cf[indices] * phi[owners_b] + flux_vf[indices]

        elif strategy is BoundaryStrategy.ZERO_GRADIENT:
            # Zero gradient: no flux contribution
            continue

        else:
            raise RuntimeError(f"Unhandled diffusion boundary strategy {strategy!r}")

    result = {"flux_cf": flux_cf, "flux_ff": flux_ff, "flux_vf": flux_vf}
    if flux_tf is not None:
        result["flux_tf"] = flux_tf
    return result
