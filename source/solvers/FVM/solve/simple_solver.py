#!/usr/bin/env python3
"""
SIMPLE Algorithm for OpenONDA FVM Solver

Semi-Implicit Method for Pressure-Linked Equations
Solves incompressible Navier-Stokes with pressure-velocity coupling.

Algorithm:
1. Solve momentum predictor: A_U * U* = H_U - ∇p
2. Solve pressure correction: ∇·(1/A_U ∇p') = ∇·U*
3. Correct velocity: U = U* - (1/A_U)∇p'
4. Update pressure: p = p + α_p * p'
5. Iterate until convergence

Converted from uFVM solve/ modules
"""

import numpy as np
from numba import njit

from ..assemble import matrix_assembly, momentum
from ..fields import gradients
from ..utils import cavity_utils


def _compute_rhie_chow_coefficients(volumes, A_U):
    """Compute the DU coefficients for Rhie-Chow interpolation.

    ``DU = V / A_U`` converts pressure-gradient cell values to velocity
    corrections: ``ΔU = −DU · ∇p'``.

    Args:
        volumes: Cell volumes ``(n_elements,)``.
        A_U:     Diagonal coefficients from the momentum equation
                 ``(n_elements, 3)`` (per component).

    Returns:
        DU array ``(n_elements, 3)``, with a small regulariser to avoid
        division by zero.
    """
    return volumes[:, np.newaxis] / (A_U + 1e-10)


def _compute_geometric_diffusion(DU_f, Sf, e, mag_CF):
    """
    Compute anisotropic geometric diffusion term for Rhie-Chow interpolation.

    Returns:
        geoDiff: Orthogonal diffusion conductance D_eff * (Sf·e) / |CF|
        nonOrth_flag: 0.0 placeholder (unused)
        k: Non-orthogonal vector Sf - (Sf·e)e
    """
    mag_Sf = np.linalg.norm(Sf)

    if mag_Sf < 1e-30:
        return 0.0, 0.0, np.zeros(3)

    n = Sf / mag_Sf

    # D_eff = n . D . n = n_x^2*D_x + n_y^2*D_y + n_z^2*D_z
    D_eff = (n[0] ** 2 * DU_f[0]) + (n[1] ** 2 * DU_f[1]) + (n[2] ** 2 * DU_f[2])

    # Orthogonal decomposition: Sf = Δ + k
    # Δ = (Sf·e) e  — orthogonal (implicit)
    # k = Sf - Δ    — explicit (non-orthogonal correction)
    sf_dot_e = np.dot(Sf, e)
    geoDiff = D_eff * sf_dot_e / (mag_CF + 1e-12)

    k = Sf - sf_dot_e * e

    return geoDiff, 0.0, k


def _process_interior_face_rhie_chow(
    i_face, owners, neighbours, face_weights, face_sf, face_cf_vector, DU, U_star, grad_p, rho, p
):
    """Process a single interior face for Rhie-Chow pressure-correction flux.

    Computes the interpolated velocity flux, geometric-diffusion (orthogonal)
    coefficients, and the explicit non-orthogonal and pressure-gradient
    correction terms for one interior face.

    Args:
        i_face:         Face index.
        owners:         Owner index array.
        neighbours:     Neighbour index array.
        face_weights:   Interpolation weights ``(n_faces,)``.
        face_sf:        Face area vectors ``(n_faces, 3)``.
        face_cf_vector: Centre-to-centre vectors ``(n_faces, 3)``.
        DU:             Rhie-Chow coefficients ``(n_elements, 3)``.
        U_star:         Predicted velocity ``(n_total, 3)``.
        grad_p:         Pressure gradient ``(n_total, 3)``.
        rho:            Density.
        p:              Pressure field ``(n_total,)``.

    Returns:
        Tuple ``(flux_cf, flux_ff, flux_vf)``: the owner coefficient,
        neighbour coefficient, and explicit RHS flux for this face.
    """
    own = owners[i_face]
    nei = neighbours[i_face]
    w = face_weights[i_face]

    Sf = face_sf[i_face]
    CF = face_cf_vector[i_face]
    e = CF / (np.linalg.norm(CF) + 1e-10)
    mag_CF = np.linalg.norm(CF)

    # Interpolate to face
    DU_f = w * DU[nei] + (1 - w) * DU[own]
    U_f = w * U_star[nei] + (1 - w) * U_star[own]
    grad_p_f = w * grad_p[nei] + (1 - w) * grad_p[own]

    # Term I: Interpolated Velocity Flux (contains implicit -grad_p_interpolated)
    flux_interpolated = rho * np.dot(U_f, Sf)

    # Term II: Geometric Diffusion (Coefficient Calculation)
    geoDiff, _, k = _compute_geometric_diffusion(DU_f, Sf, e, mag_CF)

    if geoDiff > 0:
        flux_cf = rho * geoDiff  # Owner Coeff (Positive)
        flux_ff = -rho * geoDiff  # Neighbor Coeff (Negative)
    else:
        flux_cf = 0.0
        flux_ff = 0.0

    # Term III: Rhie-Chow Smoothing
    # flux_vf = flux_interpolated + (Flux_GradP_Interp - Flux_GradP_Compact)

    # 1. Full Interpolated Gradient Flux
    DUSf = DU_f * Sf
    term_interp = rho * np.dot(grad_p_f, DUSf)

    # 2. Compact Gradient Flux (Orthogonal — uses orthogonal geoDiff)
    # Flux out = Gamma * (p_P - p_N)
    #          = Gamma * p_P - Gamma * p_N
    #          = flux_cf * p[own] + flux_ff * p[nei]
    term_compact = flux_cf * p[own] + flux_ff * p[nei]

    # 3. Non-Orthogonal Correction (explicit)
    # k = Sf - (Sf·e)e, handled explicitly in the RHS
    k_norm = np.linalg.norm(k)
    if k_norm > 1e-12:
        flux_nonortho = rho * np.dot(k, DU_f * grad_p_f)
    else:
        flux_nonortho = 0.0

    # Total Flux
    flux_vf = flux_interpolated + term_interp + term_compact + flux_nonortho

    return flux_cf, flux_ff, flux_vf


def _process_boundary_face_rhie_chow(
    i_face,
    boundary,
    owners,
    face_sf,
    face_cf_vector,
    U_star,
    DU,
    grad_p,
    rho,
    n_elements,
    n_interior,
    p,
):
    """Process a single boundary face for Rhie-Chow pressure-correction flux.

    Handles three pressure BC types:
    - ``zeroGradient`` / ``noSlip``: velocity flux only (no correction).
    - ``fixedValue``: full Rhie-Chow with geometric diffusion and non-orthogonal
      correction.
    - ``empty``: zero contribution (2-D face).

    Args:
        i_face:         Global face index.
        boundary:       Boundary patch dictionary.
        owners:         Owner index array.
        face_sf:        Face area vectors ``(n_faces, 3)``.
        face_cf_vector: Centre-to-centre vectors ``(n_faces, 3)``.
        U_star:         Predicted velocity ``(n_total, 3)``.
        DU:             Rhie-Chow coefficients ``(n_elements, 3)``.
        grad_p:         Pressure gradient ``(n_total, 3)``.
        rho:            Density.
        n_elements:     Number of interior elements.
        n_interior:     Number of interior faces.
        p:              Pressure field ``(n_total,)``.

    Returns:
        Tuple ``(flux_cf, flux_ff, flux_vf)``.
    """
    own = owners[i_face]
    b_elem_idx = n_elements + (i_face - n_interior)

    Sf = face_sf[i_face]
    U_b = U_star[b_elem_idx]
    bc_type_p = boundary.get("bc_type_p", "zeroGradient")

    # zeroGradient pressure (inlet, wall) or noSlip: No pressure correction flux
    if bc_type_p == "zeroGradient" or bc_type_p == "noSlip":
        return 0.0, 0.0, rho * np.dot(U_b, Sf)

    # fixedValue pressure (outlet): Include Rhie-Chow diffusion
    if bc_type_p == "fixedValue":
        CF = face_cf_vector[i_face]
        e = CF / (np.linalg.norm(CF) + 1e-10)
        mag_CF = np.linalg.norm(CF)

        DU_b = DU[own]
        grad_p_b = grad_p[own]

        # 1. Base Velocity Flux
        flux_vf = rho * np.dot(U_b, Sf)

        geoDiff, _, k = _compute_geometric_diffusion(DU_b, Sf, e, mag_CF)

        if geoDiff > 0:
            flux_cf = rho * geoDiff
            flux_ff = -rho * geoDiff

            # 2. Interpolated Gradient Flux
            term_interp = rho * DU_b[0] * grad_p_b[0] * Sf[0] \
                        + rho * DU_b[1] * grad_p_b[1] * Sf[1] \
                        + rho * DU_b[2] * grad_p_b[2] * Sf[2]

            # 3. Compact Pressure Drive
            val = boundary.get("value_p")
            if val is None:
                val = boundary.get("value")
            if val is None:
                val = 0.0

            term_compact = flux_cf * p[own] + flux_ff * val

            # 4. Non-Orthogonal Correction (explicit)
            k_norm = np.linalg.norm(k)
            if k_norm > 1e-12:
                flux_nonortho = rho * (k[0] * DU_b[0] * grad_p_b[0]
                                       + k[1] * DU_b[1] * grad_p_b[1]
                                       + k[2] * DU_b[2] * grad_p_b[2])
            else:
                flux_nonortho = 0.0

            # Total Flux
            flux_vf += term_interp + term_compact + flux_nonortho
        else:
            flux_cf = 0.0
            flux_ff = 0.0

        return flux_cf, flux_ff, flux_vf

    # Empty BC: For 2D simulations (extruded meshes), empty faces contribute
    # NOTHING to the pressure equation — no matrix entry, no source term, no flux.
    # This matches OpenFOAM's treatment and the behaviour of the diffusion/convection
    # assemblers in this solver.
    if boundary.get("bc_type_p") == "empty" or boundary.get("type") == "empty":
        return 0.0, 0.0, 0.0

    return 0.0, 0.0, 0.0


@njit(cache=True)
def _process_boundary_faces_jit(
    n_boundary_faces,
    n_interior,
    n_elements,
    rho,
    owners,
    face_sf,
    face_cf_vector,
    U_star,
    DU,
    grad_p,
    p,
    bc_type_codes,
    p_boundary_values,
    boundary_face_indices,
):
    """Numba-JITted boundary-face processing for Rhie-Chow assembly.

    Vectorised over all boundary faces using integer-coded BC types.
    Significantly faster than looping per patch in pure Python.

    Args:
        n_boundary_faces:     Total number of boundary faces.
        n_interior:           Number of interior faces.
        n_elements:           Number of interior elements.
        rho:                  Density.
        owners:               Owner index array.
        face_sf:              Face area vectors ``(n_faces, 3)``.
        face_cf_vector:       Centre-to-centre vectors ``(n_faces, 3)``.
        U_star:               Predicted velocity ``(n_total, 3)``.
        DU:                   Rhie-Chow coefficients ``(n_elements, 3)``.
        grad_p:               Pressure gradient ``(n_total, 3)``.
        p:                    Pressure field ``(n_total,)``.
        bc_type_codes:        Integer-coded BC: 0=zeroGradient, 1=fixedValue, 2=empty.
        p_boundary_values:    Fixed pressure values for fixedValue patches.
        boundary_face_indices: Global face index for each boundary face.

    Returns:
        Tuple ``(flux_cf_out, flux_ff_out, flux_vf_out)`` — coefficient
        arrays for all boundary faces.
    """
    n_total_boundary = len(boundary_face_indices)
    flux_cf_out = np.zeros(n_total_boundary, dtype=np.float64)
    flux_ff_out = np.zeros(n_total_boundary, dtype=np.float64)
    flux_vf_out = np.zeros(n_total_boundary, dtype=np.float64)

    for i in range(n_total_boundary):
        i_face = boundary_face_indices[i]
        own = owners[i_face]

        Sf0, Sf1, Sf2 = face_sf[i_face, 0], face_sf[i_face, 1], face_sf[i_face, 2]

        bc_code = bc_type_codes[i]

        # zeroGradient pressure (inlet, wall) or noSlip
        if bc_code == 0:
            b_elem_idx = n_elements + (i_face - n_interior)
            Ub0, Ub1, Ub2 = U_star[b_elem_idx, 0], U_star[b_elem_idx, 1], U_star[b_elem_idx, 2]
            flux_vf_out[i] = rho * (Ub0 * Sf0 + Ub1 * Sf1 + Ub2 * Sf2)
            continue

        # fixedValue pressure (outlet)
        if bc_code == 1:
            b_elem_idx = n_elements + (i_face - n_interior)
            Ub0, Ub1, Ub2 = U_star[b_elem_idx, 0], U_star[b_elem_idx, 1], U_star[b_elem_idx, 2]

            CF0, CF1, CF2 = face_cf_vector[i_face, 0], face_cf_vector[i_face, 1], face_cf_vector[i_face, 2]
            mag_CF = (CF0 * CF0 + CF1 * CF1 + CF2 * CF2) ** 0.5

            e0 = CF0 / (mag_CF + 1e-10)
            e1 = CF1 / (mag_CF + 1e-10)
            e2 = CF2 / (mag_CF + 1e-10)

            DU0, DU1, DU2 = DU[own, 0], DU[own, 1], DU[own, 2]
            gp0, gp1, gp2 = grad_p[own, 0], grad_p[own, 1], grad_p[own, 2]

            # Base velocity flux
            flux_vf = rho * (Ub0 * Sf0 + Ub1 * Sf1 + Ub2 * Sf2)

            # Geometric diffusion
            mag_Sf = (Sf0 * Sf0 + Sf1 * Sf1 + Sf2 * Sf2) ** 0.5
            if mag_Sf < 1e-30:
                continue
            n0, n1, n2 = Sf0 / mag_Sf, Sf1 / mag_Sf, Sf2 / mag_Sf
            D_eff = n0 * n0 * DU0 + n1 * n1 * DU1 + n2 * n2 * DU2
            sf_dot_e = Sf0 * e0 + Sf1 * e1 + Sf2 * e2
            geoDiff = D_eff * sf_dot_e / (mag_CF + 1e-12)

            if geoDiff > 0:
                cf = rho * geoDiff
                ff = -rho * geoDiff

                # Interpolated gradient flux
                term_interp = rho * (DU0 * gp0 * Sf0 + DU1 * gp1 * Sf1 + DU2 * gp2 * Sf2)

                # Compact pressure drive
                val = p_boundary_values[i]
                term_compact = cf * p[own] + ff * val

                # Non-orthogonal correction
                k0 = Sf0 - sf_dot_e * e0
                k1 = Sf1 - sf_dot_e * e1
                k2 = Sf2 - sf_dot_e * e2
                k_norm = (k0 * k0 + k1 * k1 + k2 * k2) ** 0.5
                if k_norm > 1e-12:
                    flux_nonortho = rho * (k0 * DU0 * gp0 + k1 * DU1 * gp1 + k2 * DU2 * gp2)
                else:
                    flux_nonortho = 0.0

                flux_vf = flux_vf + term_interp + term_compact + flux_nonortho
                flux_cf_out[i] = cf
                flux_ff_out[i] = ff

            flux_vf_out[i] = flux_vf
            continue

        # bc_code == 2 (empty): nothing to do — already zero-initialized
    return flux_cf_out, flux_ff_out, flux_vf_out


def _build_boundary_face_arrays(boundaries, n_interior, n_faces):
    """Pre-compute flat arrays for JIT boundary-face processing.

    Returns
    -------
    bc_type_codes : ndarray
        0=zeroGradient, 1=fixedValue, 2=empty
    p_boundary_values : ndarray
        Fixed pressure value (or 0.0 if not applicable)
    boundary_face_indices : ndarray
        Global face index for each boundary face
    """
    n_bnd = n_faces - n_interior
    bc_type_codes = np.empty(n_bnd, dtype=np.int32)
    p_boundary_values = np.empty(n_bnd, dtype=np.float64)
    boundary_face_indices = np.empty(n_bnd, dtype=np.int32)

    idx = 0
    for boundary in boundaries:
        start = boundary["startFace"]
        nf = boundary["nFaces"]
        bc_type_p = boundary.get("bc_type_p", "")
        bc_type = boundary.get("type", "")
        val = boundary.get("value_p")
        if val is None:
            val = boundary.get("value", 0.0)

        for j in range(nf):
            i_face = start + j
            boundary_face_indices[idx] = i_face
            if bc_type_p == "fixedValue":
                bc_type_codes[idx] = 1
                p_boundary_values[idx] = val if val is not None else 0.0
            elif bc_type_p == "empty" or bc_type == "empty":
                bc_type_codes[idx] = 2
                p_boundary_values[idx] = 0.0
            else:
                bc_type_codes[idx] = 0
                p_boundary_values[idx] = 0.0
            idx += 1

    return bc_type_codes, p_boundary_values, boundary_face_indices


def assemble_pressure_correction_equation_rhie_chow(
    U_star, A_U, p, rho, mesh_data, geo_data, boundaries, alpha_u=1.0
):
    """
    Assemble pressure correction equation using Modified Rhie-Chow interpolation.

    This implementation uses the "H-by-A" reconstruction method:
    1. Reconstruct velocity without pressure gradient at cell centers (HbyA).
    2. Interpolate HbyA to faces.
    3. Add compact pressure gradient drive at faces.

    This is more robust against checkerboarding than the standard correction method.

    Args:
        U_star: Predicted velocity field
        A_U: Momentum diagonal coefficients
        p: Current pressure field
        rho: Density
        mesh_data: Mesh connectivity
        geo_data: Geometric data
        boundaries: Boundary conditions
        alpha_u: Velocity under-relaxation factor

    Returns:
        tuple: (A_p, b_p, f_vf) where f_vf is the Rhie-Chow corrected flux (phi_star).
    """
    n_elements = mesh_data["n_elements"]
    n_interior = mesh_data["n_interior_faces"]
    n_faces = mesh_data["n_faces"]
    owners = mesh_data["owners"]
    neighbours = mesh_data["neighbours"]

    # 1. Compute DU and grad_p
    volumes = geo_data["element_volumes"]
    # Restore physical A_U from relaxed A_U for Rhie-Chow D-coefficients
    A_U_physical = A_U * alpha_u
    DU = _compute_rhie_chow_coefficients(volumes, A_U_physical)

    # Use direct gradient computation for full pressure field p
    _grad_fn = gradients._resolve_gradient_fn(geo_data)
    grad_p = _grad_fn(p, mesh_data, geo_data)
    if grad_p.ndim == 3:
        grad_p = grad_p.squeeze(-1)

    # Pre-allocate flux arrays
    flux_cf = np.zeros(n_faces)
    flux_ff = np.zeros(n_faces)
    flux_vf = np.zeros(n_faces)

    # 2. Interior Faces (Vectorized)
    # Gather all necessary data arrays
    w = geo_data["face_weights"][:n_interior]
    sf = geo_data["face_sf"][:n_interior]
    cf_vec = geo_data["face_cf_vector"][:n_interior]
    own = owners[:n_interior]
    nei = neighbours[:n_interior]

    # Pre-compute geometric properties
    mag_cf = np.linalg.norm(cf_vec, axis=1)

    # --- MODIFIED RHIE-CHOW INTERPOLATION ---
    # Standard Rhie-Chow: U_f = Avg(U) + Avg(D)*(Avg(GradP) - CompactGradP)
    # Modified Rhie-Chow: U_f = Avg(U + D*GradP) - Avg(D)*CompactGradP
    # This is much more robust on persistent checkerboarding.

    # 1. Reconstruct "H/A" velocity at cell centers (Velocity without pressure gradient)
    # U_center = H/A - grad_p * DU
    # So H/A = U_center + grad_p * DU
    # We use U_star as U_center (it includes -grad_p * DU approx)
    U_HbyA = U_star[:n_elements] + DU * grad_p[:n_elements]

    # 2. Interpolate DU and U_HbyA to faces
    # DU_f = w * DU[nei] + (1-w) * DU[own]
    du_f = w[:, np.newaxis] * DU[nei] + (1.0 - w[:, np.newaxis]) * DU[own]

    # U_HbyA_f = w * U_HbyA[nei] + (1-w) * U_HbyA[own]
    u_hbya_f = w[:, np.newaxis] * U_HbyA[nei] + (1.0 - w[:, np.newaxis]) * U_HbyA[own]

    # 3. Compute Face Normal, unit edge vector, and D_eff
    mag_sf = np.linalg.norm(sf, axis=1)
    valid_sf = mag_sf > 1e-30
    n_vec = np.zeros_like(sf)
    n_vec[valid_sf] = sf[valid_sf] / mag_sf[valid_sf][:, np.newaxis]

    # Unit owner→neighbour vector
    e_vec = cf_vec / (mag_cf[:, np.newaxis] + 1e-12)

    # Project diagonal D onto normal: D_eff = n . D . n
    d_eff = np.sum((n_vec**2) * du_f, axis=1)

    # Orthogonal projection: sf_dot_e = Sf · e
    sf_dot_e = np.sum(sf * e_vec, axis=1)

    # geoDiff uses the ORTHOGONAL component Sf·e (not mag_sf)
    geo_diff = d_eff * sf_dot_e / (mag_cf + 1e-12)

    # Non-orthogonal vector k = Sf - (Sf·e)e
    k_vec = sf - (sf_dot_e[:, np.newaxis] * e_vec)
    # Non-orthogonal flux: rho * k · (DU_f * grad_p_f)
    grad_p_interior = grad_p[:n_elements]
    grad_own = grad_p_interior[own]
    grad_nei = grad_p_interior[nei]
    grad_f = w[:, np.newaxis] * grad_nei + (1.0 - w[:, np.newaxis]) * grad_own
    flux_nonortho = rho * np.sum(k_vec * du_f * grad_f, axis=1)

    # 4. Construct Flux
    # Term I: HbyA Flux
    flux_hbya = rho * np.sum(u_hbya_f * sf, axis=1)

    # Term II: Compact Pressure Drive (orthogonal implicit)
    term_compact = rho * geo_diff * (p[own] - p[nei])

    # Coefficients for Pressure Equation (Laplacian) — orthogonal part only
    flux_cf[:n_interior] = rho * geo_diff
    flux_ff[:n_interior] = -rho * geo_diff

    # Total Flux = HbyA + Implicit Pressure + Non-Orthogonal Correction
    flux_vf[:n_interior] = flux_hbya + term_compact + flux_nonortho

    # 3. Boundary Faces (Numba JIT)
    n_boundary_faces = n_faces - n_interior
    if n_boundary_faces > 0:
        bc_codes, p_vals, bnd_face_idx = _build_boundary_face_arrays(
            boundaries, n_interior, n_faces
        )
        cf_b, ff_b, vf_b = _process_boundary_faces_jit(
            n_boundary_faces,
            n_interior,
            n_elements,
            rho,
            owners,
            geo_data["face_sf"],
            geo_data["face_cf_vector"],
            U_star,
            DU,
            grad_p,
            p,
            bc_codes,
            p_vals,
            bnd_face_idx,
        )
        for i in range(n_boundary_faces):
            i_face = bnd_face_idx[i]
            flux_cf[i_face] = cf_b[i]
            flux_ff[i_face] = ff_b[i]
            flux_vf[i_face] = vf_b[i]

    # 4. Assemble Matrix and RHS
    flux_data = {"flux_cf": flux_cf, "flux_ff": flux_ff, "flux_vf": flux_vf}

    A_p = matrix_assembly.assemble_matrix_from_fluxes_vectorized(flux_data, mesh_data)
    b_p = matrix_assembly.assemble_rhs_from_fluxes_vectorized(flux_data, mesh_data)

    # 5. Fix Pressure Reference
    # For incompressible flow, the pressure equation is singular (defined up
    # to a constant). Even when a few outlet faces have fixedValue p, the
    # constraint may be too weak to prevent the near-null-space mode from
    # growing (e.g. 10 outlet faces out of 2387 cells). This matches
    # OpenFOAM's setReference(pRefCell, pRefValue) which is always applied.
    if cavity_utils.needs_pressure_reference(boundaries, n_elements):
        A_p, b_p = cavity_utils.fix_pressure_reference(A_p, b_p)

    return A_p, b_p, flux_vf


def _extend_p_prime_bcs(p_prime, mesh_data, boundaries):
    """Extend the pressure-correction array with ghost-cell values.

    For ``fixedValue`` pressure boundaries, the ghost value is set to
    zero (``p' = 0`` at a fixed-pressure face).  For all other types,
    the ghost cell inherits the owner cell value (zero-gradient).

    Args:
        p_prime:    Pressure correction for interior cells ``(n_elements,)``.
        mesh_data:  Mesh dictionary.
        boundaries: List of boundary patch dictionaries.

    Returns:
        Extended ``p_prime`` array ``(n_elements + n_boundary_faces,)``.
    """
    n_elements = mesh_data["n_elements"]
    n_interior = mesh_data["n_interior_faces"]
    n_faces = mesh_data["n_faces"]
    owners = mesh_data["owners"]
    p_prime_ext = np.zeros(n_elements + (n_faces - n_interior))
    p_prime_ext[:n_elements] = p_prime
    for boundary in boundaries:
        start = boundary["startFace"]
        nf = boundary["nFaces"]
        idx = n_elements + (start - n_interior)
        own = owners[start : start + nf]
        bc_type_p = boundary.get("bc_type_p") or boundary.get("bc_type") or boundary.get("type")
        if bc_type_p == "fixedValue":
            p_prime_ext[idx : idx + nf] = 0.0
        else:
            p_prime_ext[idx : idx + nf] = p_prime[own]
    return p_prime_ext


def _correct_interior_fluxes(phi, p_prime, mesh_data, geo_data, DU, rho):
    """Correct interior face fluxes with the Rhie-Chow pressure correction.

    Applies the flux correction ``Δφ = ρ⋅g⋅(p'_P − p'_N)`` where *g* is
    the geometric diffusion coefficient based on the interpolated DU and
    face-normal projection.

    Args:
        phi:      Face flux array ``(n_faces,)`` (mutated in place).
        p_prime:  Pressure correction ``(n_elements,)``.
        mesh_data: Mesh dictionary.
        geo_data:  Geometry dictionary.
        DU:       Rhie-Chow coefficients ``(n_elements, 3)``.
        rho:      Density.
    """
    n_interior = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"]
    neighbours = mesh_data["neighbours"]
    w = geo_data["face_weights"][:n_interior]
    sf = geo_data["face_sf"][:n_interior]
    cf_vec = geo_data["face_cf_vector"][:n_interior]
    mag_cf = np.linalg.norm(cf_vec, axis=1)
    mag_sf = np.linalg.norm(sf, axis=1)
    n_vec = sf / (mag_sf[:, np.newaxis] + 1e-30)
    du_f = (
        w[:, np.newaxis] * DU[neighbours[:n_interior]]
        + (1.0 - w[:, np.newaxis]) * DU[owners[:n_interior]]
    )
    d_eff = np.sum((n_vec**2) * du_f, axis=1)
    geo_diff = d_eff * mag_sf / (mag_cf + 1e-12)
    phi[:n_interior] += (
        rho * geo_diff * (p_prime[owners[:n_interior]] - p_prime[neighbours[:n_interior]])
    )


def _correct_boundary_fluxes(
    phi, p_prime, boundaries, owners, geo_data, n_elements, n_interior, DU, rho
):
    """Correct boundary-face fluxes for ``fixedValue`` pressure boundaries.

    Only patches whose pressure BC type is ``fixedValue`` receive a
    correction, computed from the geometric diffusion and wall distance.

    Args:
        phi:        Face flux array ``(n_faces,)`` (mutated in place).
        p_prime:    Pressure correction ``(n_elements,)``.
        boundaries: List of boundary patch dictionaries.
        owners:     Owner index array.
        geo_data:   Geometry dictionary.
        n_elements: Number of interior elements.
        n_interior: Number of interior faces.
        DU:         Rhie-Chow coefficients ``(n_elements, 3)``.
        rho:        Density.
    """
    for boundary in boundaries:
        if boundary.get("bc_type_p") != "fixedValue":
            continue
        start = boundary["startFace"]
        nf = boundary["nFaces"]
        idx = np.arange(start, start + nf)
        own = owners[idx]
        wall_dist = geo_data["wall_dist"][idx]
        mag_sf_b = np.linalg.norm(geo_data["face_sf"][idx], axis=1)
        n_b = geo_data["face_sf"][idx] / (mag_sf_b[:, np.newaxis] + 1e-30)
        d_eff_b = np.sum((n_b**2) * DU[own], axis=1)
        geo_diff_b = d_eff_b * mag_sf_b / wall_dist
        phi[idx] += rho * geo_diff_b * p_prime[own]


def _apply_inlet_outlet_bc(U, phi, boundary, owners, n_elements, n_interior):
    """inletOutlet velocity BC: zeroGradient on outflow, fixed value on inflow.

    Per face: outgoing flux (φ ≥ 0) → extrapolate from the owner cell
    (zeroGradient); incoming flux (φ < 0) → impose the boundary ``value_U`` (the
    inletValue, default 0).  Physically bounds reverse flow at outlets, replacing
    the old patch-name heuristic.
    """
    start = boundary["startFace"]
    nf = boundary["nFaces"]
    idx = n_elements + (start - n_interior)
    own = owners[start : start + nf]
    inlet_val = np.asarray(boundary.get("value_U", [0.0, 0.0, 0.0]), dtype=float)
    outflow = phi[start : start + nf] >= 0.0
    U[idx : idx + nf] = np.where(outflow[:, np.newaxis], U[own], inlet_val)


def _apply_robin_bc(U, boundary, owners, geo_data, n_elements, n_interior):
    """directionMixed / Robin velocity BC (Billuart 2023, Eq. 11–14).

    Per face reconstruct the ghost value so the normal component is Dirichlet
    (``u·n̂ = u_target·n̂``) and the tangential component satisfies a vorticity-
    matched Neumann condition (``∂u_t/∂n = ω_target × n̂``):

        u_b = (u_target·n̂) n̂ + u_owner_t + d (ω_target × n̂),

    with ``u_owner_t`` the owner's tangential velocity and ``d`` the wall
    distance.  ``ω_target × n̂`` is already tangential.  Targets are stored per
    face by ``set_robin_velocity_boundary_condition``.
    """
    start = boundary["startFace"]
    nf = boundary["nFaces"]
    idx = n_elements + (start - n_interior)
    own = owners[start : start + nf]
    sf = geo_data["face_sf"][start : start + nf]
    n = sf / (np.linalg.norm(sf, axis=1, keepdims=True) + 1e-30)
    d = geo_data["wall_dist"][start : start + nf][:, np.newaxis]

    ut = boundary.get("u_target_field")
    om = boundary.get("omega_target_field")
    if ut is None:
        ut = np.zeros((nf, 3))
    if om is None:
        om = np.zeros((nf, 3))

    u_owner = U[own]
    u_owner_t = u_owner - np.sum(u_owner * n, axis=1, keepdims=True) * n
    u_target_n = np.sum(ut * n, axis=1, keepdims=True) * n
    U[idx : idx + nf] = u_target_n + u_owner_t + d * np.cross(om, n)


def _apply_zero_gradient_bc(U, phi, boundary, owners, n_elements, n_interior, boundaries):
    """Apply a zero-gradient velocity BC (extrapolate from the owner cell).

    Sets the ghost-cell velocity equal to the owner-cell value.

    Args:
        U:           Velocity array (mutated in place).
        phi:         Face flux array.
        boundary:    Boundary patch dictionary.
        owners:      Owner index array.
        n_elements:  Number of interior elements.
        n_interior:  Number of interior faces.
        boundaries:  List of all boundary patches.
    """
    start = boundary["startFace"]
    nf = boundary["nFaces"]
    idx = n_elements + (start - n_interior)
    own = owners[start : start + nf]
    U[idx : idx + nf] = U[own]


def _apply_fixed_value_bc(U, boundary, n_elements, n_interior):
    """Apply fixedValue or noSlip velocity BC.

    Honours a per-face ``value_U_field`` (n_faces_patch, 3) when present (e.g. a
    non-uniform coupler donor BC), otherwise the uniform ``value_U``.
    """
    start = boundary["startFace"]
    nf = boundary["nFaces"]
    idx = n_elements + (start - n_interior)
    bc_type_u = boundary.get("bc_type_U") or boundary.get("bc_type")
    if bc_type_u == "noSlip":
        U[idx : idx + nf] = [0.0, 0.0, 0.0]
    elif boundary.get("value_U_field") is not None:
        U[idx : idx + nf] = boundary["value_U_field"]
    elif "value_U" in boundary:
        U[idx : idx + nf] = np.array(boundary["value_U"])


def _apply_slip_bc(U, boundary, owners, geo_data, n_elements, n_interior):
    """Apply a slip / symmetry / empty velocity BC.

    Removes the component normal to the boundary face from the
    ghost-cell velocity, leaving only the tangential part.

    Args:
        U:          Velocity array (mutated in place).
        boundary:   Boundary patch dictionary.
        owners:     Owner index array.
        geo_data:   Geometry dictionary (needs ``face_sf``).
        n_elements: Number of interior elements.
        n_interior: Number of interior faces.
    """
    start = boundary["startFace"]
    nf = boundary["nFaces"]
    idx = n_elements + (start - n_interior)
    own = owners[start : start + nf]
    face_sf = geo_data["face_sf"][start : start + nf]
    for i in range(nf):
        U[idx + i] = _remove_normal_component(U[own[i]], face_sf[i])


def _update_velocity_bcs(U, phi, boundaries, owners, geo_data, n_elements, n_interior):
    """Update all velocity boundary conditions after a pressure-correction step.

    Dispatches to individual BC handlers based on each patch's
    ``bc_type_U``:

    - ``zeroGradient`` → :func:`_apply_zero_gradient_bc`
    - ``inletOutlet``  → :func:`_apply_inlet_outlet_bc`
    - ``directionMixed`` → :func:`_apply_robin_bc`
    - ``fixedValue`` / ``noSlip`` → :func:`_apply_fixed_value_bc`
    - ``empty`` / ``slip`` / ``symmetry`` → :func:`_apply_slip_bc`

    Args:
        U:           Velocity array (mutated in place).
        phi:         Face flux array.
        boundaries:  List of boundary patch dictionaries.
        owners:      Owner index array.
        geo_data:    Geometry dictionary.
        n_elements:  Number of interior elements.
        n_interior:  Number of interior faces.
    """
    for boundary in boundaries:
        bc_type_u = boundary.get("bc_type_U") or boundary.get("bc_type")
        if bc_type_u == "zeroGradient":
            _apply_zero_gradient_bc(U, phi, boundary, owners, n_elements, n_interior, boundaries)
        elif bc_type_u == "inletOutlet":
            _apply_inlet_outlet_bc(U, phi, boundary, owners, n_elements, n_interior)
        elif bc_type_u == "directionMixed":
            _apply_robin_bc(U, boundary, owners, geo_data, n_elements, n_interior)
        elif bc_type_u in ("fixedValue", "noSlip"):
            _apply_fixed_value_bc(U, boundary, n_elements, n_interior)
        elif bc_type_u in ("empty", "slip", "symmetry"):
            _apply_slip_bc(U, boundary, owners, geo_data, n_elements, n_interior)


def correct_velocity_and_flux(U, phi, p_prime, A_U, mesh_data, geo_data, boundaries, rho=1.0, alpha_u=1.0):
    """
    Apply pressure correction to velocity and persistent flux.

    U = U* - DU * grad(p')
    phi = phi* - rho * DU_f * (grad(p') . S)

    Uses UN-RELAXED diagonal for DU consistency: DU = V / (A_U * alpha_u).
    This matches the Rhie-Chow assembly which also uses the un-relaxed A_U.
    """
    n_elements = mesh_data["n_elements"]
    n_interior = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"]

    volumes = geo_data["element_volumes"]
    # Restore un-relaxed diagonal for DU consistency
    A_U_physical = A_U * alpha_u
    DU = _compute_rhie_chow_coefficients(volumes, A_U_physical)

    # 1. Correct Cell Velocity
    p_prime_ext = _extend_p_prime_bcs(p_prime, mesh_data, boundaries)
    _grad_fn = gradients._resolve_gradient_fn(geo_data)
    grad_p_prime = _grad_fn(
        p_prime_ext, mesh_data, geo_data
    )
    if grad_p_prime.ndim == 3:
        grad_p_prime = grad_p_prime.squeeze(-1)
    U[:n_elements] -= DU * grad_p_prime[:n_elements]

    # 2. Correct Face Fluxes
    _correct_interior_fluxes(phi, p_prime, mesh_data, geo_data, DU, rho)
    _correct_boundary_fluxes(
        phi, p_prime, boundaries, owners, geo_data, n_elements, n_interior, DU, rho
    )

    # 3. Update Velocity BCs
    _update_velocity_bcs(U, phi, boundaries, owners, geo_data, n_elements, n_interior)

    return U, phi


def _remove_normal_component(U_owner, face_vector):
    """Remove the normal component of velocity (slip / symmetry / empty BC).

    Projects out the velocity component along the face normal, leaving
    only the tangential component: ``U_t = U − (U·n̂) n̂``.

    Args:
        U_owner:     Velocity vector at the owner cell ``(3,)``.
        face_vector: Face area vector ``(3,)`` (direction defines normal).

    Returns:
        Tangential velocity vector ``(3,)``.
    """
    norm_Sf = np.linalg.norm(face_vector)
    if norm_Sf > 1e-10:
        n = face_vector / norm_Sf
        U_normal_mag = np.dot(U_owner, n)
        U_normal = U_normal_mag * n
        return U_owner - U_normal
    return U_owner


def _apply_scalar_bc(phi, indices, owners_b, bc_type, boundary, field_name):
    """Apply a boundary condition to a scalar field ghost-cell block.

    Zero-gradient BCs (including inlet, outlet, symmetry, empty, slip,
    noSlip) copy the owner value.  ``fixedValue`` sets the prescribed
    boundary value.

    Args:
        phi:        Scalar field array (mutated in place).
        indices:    Ghost-cell indices for this patch.
        owners_b:   Owner cell indices for the boundary faces.
        bc_type:    Boundary condition type string.
        boundary:   Boundary patch dictionary (may contain ``value_p`` etc.).
        field_name: Field name for value lookup (e.g. ``"p"``, ``"phi"``).
    """
    if bc_type in ("zeroGradient", "inlet", "outlet", "symmetry", "empty", "slip", "noSlip"):
        phi[indices] = phi[owners_b]
    elif bc_type == "fixedValue":
        val = boundary.get(f"value_{field_name}") or boundary.get("value")
        if val is not None:
            phi[indices] = val


def update_scalar_boundaries(phi, mesh_data, boundaries, field_name="p"):
    """
    Update boundary values (ghost cells) for a scalar field.

    Args:
        phi: Scalar field (n_elements + n_boundary)
        mesh_data: Mesh connectivity
        boundaries: Boundary conditions
        field_name: Name of field ('p', etc.) to check BC type
    """
    n_elements = mesh_data["n_elements"]
    n_interior = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"]

    for boundary in boundaries:
        start = boundary["startFace"]
        n_bfaces = boundary["nFaces"]

        # Boundary element indices
        b_elem_start = n_elements + (start - n_interior)
        b_elem_indices = np.arange(b_elem_start, b_elem_start + n_bfaces)
        owners_b = owners[start : start + n_bfaces]

        # Get BC type
        bc_type = (
            boundary.get(f"bc_type_{field_name}") or boundary.get("bc_type") or boundary.get("type")
        )
        _apply_scalar_bc(phi, b_elem_indices, owners_b, bc_type, boundary, field_name)


class SIMPLESolver:
    """
    SIMPLE algorithm solver for incompressible Navier-Stokes.
    """

    def __init__(self, mesh_data, geo_data, boundaries, params=None):
        """
        Initialize SIMPLE solver.

        Args:
            mesh_data: Mesh connectivity
            geo_data: Geometric data
            boundaries: Boundary conditions
            params: Dict with solver parameters:
                - alpha_u: Velocity under-relaxation (default: 0.7)
                - alpha_p: Pressure under-relaxation (default: 0.3)
                - max_iter: Maximum iterations (default: 100)
                - tolerance: Convergence tolerance (default: 1e-6)
        """
        self.mesh_data = mesh_data
        self.geo_data = geo_data
        self.boundaries = boundaries

        # Default parameters
        self.params = {
            "alpha_u": 0.7,
            "alpha_p": 0.3,
            "max_iter": 100,
            "tolerance": 1e-6,
            "convection_scheme": "deferred",
            "linear_solver": "spsolve",
        }

        if params:
            self.params.update(params)

        self.residuals = []

    def step(self, U, p, phi, U_old=None, dt=None, rho=1.0, nu=0.01, U_old_old=None,
             source_explicit=None, source_implicit=None):
        """
        Perform a single SIMPLE iteration.

        ``U_old_old`` is accepted for interface parity with the transient driver
        but is unused by steady SIMPLE.  ``source_explicit``/``source_implicit``
        are optional volumetric momentum sources (e.g. the coupling fringe
        S = λ(Utarget − U)) forwarded to the momentum predictor.
        """
        # 1. Solve momentum predictor
        U_star, A_U = momentum.solve_momentum_predictor(
            U,
            p,
            phi,
            rho,
            nu,
            self.mesh_data,
            self.geo_data,
            self.boundaries,
            convection_scheme=self.params["convection_scheme"],
            solver=self.params["linear_solver"],
            under_relaxation=self.params["alpha_u"],
            dt=dt,  # Use dt if provided (e.g. for PIMPLE)
            source_explicit=source_explicit,
            source_implicit=source_implicit,
        )

        # 2. Solve pressure correction
        A_p, b_p, phi_star = assemble_pressure_correction_equation_rhie_chow(
            U_star,
            A_U,
            p,
            rho,
            self.mesh_data,
            self.geo_data,
            self.boundaries,
            alpha_u=self.params["alpha_u"],
        )

        p_prime = matrix_assembly.solve_linear_system(
            A_p, b_p, method=self.params["linear_solver"], equation_type="pressure", tol=1e-6
        )

        # 3. Correct velocity and flux
        # Calculate residual before in-place modification of U_star
        self.last_res_u = np.linalg.norm(
            U_star[: self.mesh_data["n_elements"]] - U[: self.mesh_data["n_elements"]]
        ) / (np.linalg.norm(U[: self.mesh_data["n_elements"]]) + 1e-10)

        U, phi = correct_velocity_and_flux(
            U_star, phi_star, p_prime, A_U, self.mesh_data, self.geo_data, self.boundaries,
            rho=rho, alpha_u=self.params["alpha_u"],
        )

        # 4. Update pressure
        p[: self.mesh_data["n_elements"]] += self.params["alpha_p"] * p_prime

        # Update pressure boundaries
        update_scalar_boundaries(p, self.mesh_data, self.boundaries, field_name="p")

        # 5. Residuals
        self.last_res_p = np.linalg.norm(b_p) / (
            np.linalg.norm(p[: self.mesh_data["n_elements"]]) + 1e-10
        )

        return U, p, phi, {"p": self.last_res_p, "U": self.last_res_u}

    def solve(self, U_init, p_init, rho=1.0, nu=0.01):
        """Solve a steady incompressible flow using the SIMPLE algorithm.

        Iterates over ``max_iter`` SIMPLE steps, computing the momentum
        predictor, pressure correction, and velocity/pressure update at
        each iteration.  Convergence is declared when both the pressure
        and velocity residuals fall below ``tolerance``.

        Args:
            U_init: Initial velocity field ``(n_total, 3)``.
            p_init: Initial pressure field ``(n_total,)``.
            rho:    Fluid density (default 1.0).
            nu:     Kinematic viscosity (default 0.01).

        Returns:
            Tuple ``(U, p, phi, converged)`` where *converged* is a bool.
        """
        U = U_init.copy()
        p = p_init.copy()

        print("\\nSIMPLE Solver")
        print(f"  Max iterations: {self.params['max_iter']}")
        print(f"  Tolerance: {self.params['tolerance']}")
        print(f"  Under-relaxation: U={self.params['alpha_u']}, p={self.params['alpha_p']}")

        # Initialize Flux (phi) if not provided
        from ..assemble import convection

        phi = convection.compute_mass_flow_rate(U, self.mesh_data, self.geo_data)

        for iteration in range(int(self.params["max_iter"])):
            U, p, phi, residuals = self.step(U, p, phi, rho=rho, nu=nu)

            residual_p = self.last_res_p
            residual_u = self.last_res_u

            self.residuals.append({"iter": iteration, "R_p": residual_p, "R_u": residual_u})

            if iteration % 10 == 0 or residual_p < self.params["tolerance"]:
                print(f"  Iter {iteration:3d}: R_p={residual_p:.3e}, R_u={residual_u:.3e}")

            if residual_p < self.params["tolerance"] and residual_u < self.params["tolerance"]:
                print(f"  ✓ Converged in {iteration} iterations")
                return U, p, phi, True

        print(f"  ✗ Did not converge in {self.params['max_iter']} iterations")
        return U, p, phi, False
