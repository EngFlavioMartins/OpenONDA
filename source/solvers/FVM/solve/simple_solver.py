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

from numba import njit
import numpy as np

from ..assemble import matrix_assembly, momentum
from ..fields import diagnostics as field_diagnostics
from ..fields import gradients
from ..schemes.boundaries import BOUNDARIES, BoundaryStrategy
from ..utils import cavity_utils
from .contracts import OuterCorrectorDiagnostics


def _resolve_pressure_constraint(params) -> str:
    """Select the configured all-Neumann pressure treatment for this backend."""
    policy = str(params.get("pressure_nullspace_policy", "auto")).lower()
    backend = str(params.get("_linear_backend", "scipy")).lower()
    if policy == "auto":
        return "nullspace" if backend == "petsc" else "reference"
    if policy == "petsc":
        if backend != "petsc":
            raise ValueError("The PETSc pressure null space requires backend='petsc'")
        return "nullspace"
    if policy == "reference":
        return "reference"
    raise ValueError(f"Unknown pressure null-space policy {policy!r}")


def _pressure_requires_constraint(boundaries, U_star, mesh_data, geo_data) -> bool:
    """Return whether the assembled pressure operator has a constant null space."""
    n_elements = mesh_data["n_elements"]
    n_interior = mesh_data["n_interior_faces"]
    for boundary in boundaries:
        bc_type = boundary.get("bc_type_p")
        strategy = BOUNDARIES.strategy(bc_type, "p", "pressure")
        if strategy is BoundaryStrategy.FIXED_VALUE:
            local_requires_constraint = False
            break
        if strategy is BoundaryStrategy.FREESTREAM:
            start = boundary["startFace"]
            nf = boundary["nFaces"]
            ghosts = n_elements + np.arange(start - n_interior, start - n_interior + nf)
            flux = np.sum(U_star[ghosts] * geo_data["face_sf"][start : start + nf], axis=1)
            if np.any(flux >= 0.0):
                local_requires_constraint = False
                break
    else:
        local_requires_constraint = True
    parallel = mesh_data.get("_parallel_context")
    if parallel is not None and parallel.is_partitioned:
        return parallel.global_all(local_requires_constraint)
    return local_requires_constraint


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


def _compute_pressure_face_conductance(mesh_data, geo_data, DU):
    """Return the geometric Rhie--Chow conductance for every face.

    The pressure matrix and the post-solve flux correction must use exactly
    the same conductance.  Keeping this calculation in one function prevents
    the non-orthogonal inconsistency that previously used ``Sf·e`` during
    assembly but ``|Sf|`` during correction.

    ``DU`` is the cell-centred diagonal pressure-to-velocity coefficient.  It
    is linearly interpolated on interior faces and taken from the owner on
    boundary faces.  The returned value excludes density; callers apply the
    same ``rho`` scaling to matrix and correction terms.
    """
    n_faces = mesh_data["n_faces"]
    n_interior = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"]
    neighbours = mesh_data["neighbours"]

    sf = geo_data["face_sf"]
    cf_vec = geo_data["face_cf_vector"]
    mag_sf = np.linalg.norm(sf, axis=1)
    mag_cf = np.linalg.norm(cf_vec, axis=1)

    if np.any(mag_sf <= 1e-30) or np.any(mag_cf <= 1e-30):
        raise ValueError("Pressure conductance requires non-zero face area and cell distance")

    du_face = np.empty((n_faces, 3), dtype=np.float64)
    if n_interior:
        w = geo_data["face_weights"][:n_interior, np.newaxis]
        own_i = owners[:n_interior]
        nei_i = neighbours[:n_interior]
        du_face[:n_interior] = w * DU[nei_i] + (1.0 - w) * DU[own_i]
    if n_faces > n_interior:
        du_face[n_interior:] = DU[owners[n_interior:]]
    boundary_neighbours = np.asarray(
        mesh_data.get("boundary_neighbours", np.full(n_faces, -1, dtype=np.int32))
    )
    coupled = np.flatnonzero(boundary_neighbours >= 0)
    if coupled.size:
        w = geo_data["face_weights"][coupled, np.newaxis]
        du_face[coupled] = w * DU[boundary_neighbours[coupled]] + (1.0 - w) * DU[owners[coupled]]

    normal = sf / mag_sf[:, np.newaxis]
    edge = cf_vec / mag_cf[:, np.newaxis]
    d_eff = np.sum(normal * normal * du_face, axis=1)
    orthogonal_area = np.sum(sf * edge, axis=1)
    conductance = d_eff * orthogonal_area / mag_cf

    if np.any(conductance < -1e-14):
        raise ValueError(
            "Negative pressure-face conductance; check face orientation and mesh geometry"
        )
    return np.maximum(conductance, 0.0)


def _update_fixed_flux_pressure_boundaries(
    p, U_star, DU, mesh_data, geo_data, boundaries, grad_p=None
):
    """Update ``fixedFluxPressure`` face values from the normal momentum balance.

    For a prescribed boundary velocity ``U_b``, the pressure gradient must make
    the pressure-free predictor ``H/A`` deliver the same normal flux::

        U_b.n = (H/A).n - D_n dp/dn.

    The native solver stores boundary scalar values at face centres (rather
    than at reflected ghost-cell centres), so the resulting normal gradient is
    converted to a face-minus-owner pressure increment.  The increment is
    retained on the patch and re-applied after every pressure correction; this
    is the discrete analogue of OpenFOAM's ``fixedFluxPressure`` update.

    A diagnostic replay may set ``fixed_flux_pressure_external=True`` and
    provide ``fixed_flux_pressure_delta`` itself.  That mode is intentionally
    explicit: it is used only to measure the cropped-stencil floor against a
    recorded monolithic oracle.
    """
    n_elements = mesh_data["n_elements"]
    n_interior = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"]
    face_sf = geo_data["face_sf"]
    face_cf = geo_data["face_cf_vector"]
    _grad_fn = gradients._resolve_gradient_fn(geo_data)

    fixed_flux_patches = []
    for boundary in boundaries:
        strategy = BOUNDARIES.strategy(boundary.get("bc_type_p"), "p", "ghost")
        if strategy is BoundaryStrategy.FIXED_FLUX_PRESSURE:
            fixed_flux_patches.append(boundary)
    if not fixed_flux_patches:
        return grad_p

    if grad_p is None:
        grad_p = _grad_fn(p, mesh_data, geo_data)
        if grad_p.ndim == 3:
            grad_p = grad_p.squeeze(-1)

    # U_star was obtained with the *current* pressure gradient.  Reconstruct
    # H/A once from that same pair.  Iterating this formula without resolving
    # momentum would add the new gradient repeatedly and double the requested
    # pressure slope on every sweep.
    U_hbya = U_star[:n_elements] + DU * grad_p[:n_elements]
    changed = False
    for boundary in fixed_flux_patches:
        start = boundary["startFace"]
        nf = boundary["nFaces"]
        ghost = n_elements + (start - n_interior) + np.arange(nf)
        own = owners[start : start + nf]
        if boundary.get("fixed_flux_pressure_external", False):
            delta = boundary.get("fixed_flux_pressure_delta")
            if delta is not None:
                p[ghost] = p[own] + np.asarray(delta, dtype=float)
                changed = True
            continue

        sf = face_sf[start : start + nf]
        mag_sf = np.linalg.norm(sf, axis=1)
        normal = sf / mag_sf[:, np.newaxis]
        dr = face_cf[start : start + nf]
        normal_distance = np.einsum("ij,ij->i", dr, normal)
        D_normal = np.einsum("ij,ij->i", DU[own], normal * normal)
        target_normal = np.einsum("ij,ij->i", U_star[ghost], normal)
        hbya_normal = np.einsum("ij,ij->i", U_hbya[own], normal)
        dpdn = (hbya_normal - target_normal) / np.maximum(D_normal, 1.0e-30)
        delta = dpdn * normal_distance
        boundary["fixed_flux_pressure_delta"] = delta
        p[ghost] = p[own] + delta
        changed = True

    if changed:
        grad_p = _grad_fn(p, mesh_data, geo_data)
        if grad_p.ndim == 3:
            grad_p = grad_p.squeeze(-1)
    return grad_p


def _compute_geometric_diffusion(DU_f, Sf, e, mag_CF):
    """
    Compute anisotropic geometric diffusion term for Rhie-Chow interpolation.

    Returns:
        geoDiff: Orthogonal diffusion conductance D_eff * (Sf·e) / |CF|
        k: Non-orthogonal vector Sf - (Sf·e)e
    """
    mag_Sf = np.linalg.norm(Sf)

    if mag_Sf < 1e-30:
        return 0.0, np.zeros(3)

    n = Sf / mag_Sf

    # D_eff = n . D . n = n_x^2*D_x + n_y^2*D_y + n_z^2*D_z
    D_eff = (n[0] ** 2 * DU_f[0]) + (n[1] ** 2 * DU_f[1]) + (n[2] ** 2 * DU_f[2])

    # Orthogonal decomposition: Sf = Δ + k
    # Δ = (Sf·e) e  — orthogonal (implicit)
    # k = Sf - Δ    — explicit (non-orthogonal correction)
    sf_dot_e = np.dot(Sf, e)
    geoDiff = D_eff * sf_dot_e / (mag_CF + 1e-12)

    k = Sf - sf_dot_e * e

    return geoDiff, k


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
    geoDiff, k = _compute_geometric_diffusion(DU_f, Sf, e, mag_CF)

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
    flux_nonortho = rho * np.dot(k, DU_f * grad_p_f) if k_norm > 1e-12 else 0.0

    # Total Flux
    flux_vf = flux_interpolated + term_interp + term_compact + flux_nonortho

    return flux_cf, flux_ff, flux_vf


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
    face_conductance,
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

        b_elem_idx = n_elements + (i_face - n_interior)
        Ub0, Ub1, Ub2 = U_star[b_elem_idx, 0], U_star[b_elem_idx, 1], U_star[b_elem_idx, 2]
        velocity_flux = Ub0 * Sf0 + Ub1 * Sf1 + Ub2 * Sf2

        # zeroGradient pressure, including the inflow side of freestream
        if bc_code == 0 or (bc_code == 3 and velocity_flux < 0.0):
            flux_vf_out[i] = rho * (Ub0 * Sf0 + Ub1 * Sf1 + Ub2 * Sf2)
            continue

        # fixedValue pressure, including the outflow side of freestream
        if bc_code == 1 or bc_code == 3:
            CF0, CF1, CF2 = (
                face_cf_vector[i_face, 0],
                face_cf_vector[i_face, 1],
                face_cf_vector[i_face, 2],
            )
            mag_CF = (CF0 * CF0 + CF1 * CF1 + CF2 * CF2) ** 0.5

            e0 = CF0 / (mag_CF + 1e-10)
            e1 = CF1 / (mag_CF + 1e-10)
            e2 = CF2 / (mag_CF + 1e-10)

            DU0, DU1, DU2 = DU[own, 0], DU[own, 1], DU[own, 2]
            gp0, gp1, gp2 = grad_p[own, 0], grad_p[own, 1], grad_p[own, 2]

            # Base velocity flux
            flux_vf = rho * (Ub0 * Sf0 + Ub1 * Sf1 + Ub2 * Sf2)

            # Geometric diffusion.  This value is shared with pressure-matrix
            # assembly and post-solve flux correction.
            mag_Sf = (Sf0 * Sf0 + Sf1 * Sf1 + Sf2 * Sf2) ** 0.5
            if mag_Sf < 1e-30:
                continue
            sf_dot_e = Sf0 * e0 + Sf1 * e1 + Sf2 * e2
            geoDiff = face_conductance[i_face]

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
        0=zeroGradient, 1=fixedValue, 2=empty, 3=freestream
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
        bc_type_p = boundary.get("bc_type_p")
        strategy = BOUNDARIES.strategy(bc_type_p, "p", "pressure")
        val = boundary.get("value_p")
        if val is None:
            val = boundary.get("value", 0.0)

        for j in range(nf):
            i_face = start + j
            boundary_face_indices[idx] = i_face
            if strategy is BoundaryStrategy.FIXED_VALUE:
                bc_type_codes[idx] = 1
                p_boundary_values[idx] = val if val is not None else 0.0
            elif strategy is BoundaryStrategy.EMPTY:
                bc_type_codes[idx] = 2
                p_boundary_values[idx] = 0.0
            elif strategy is BoundaryStrategy.FREESTREAM:
                bc_type_codes[idx] = 3
                p_boundary_values[idx] = val if val is not None else 0.0
            elif strategy in (
                BoundaryStrategy.ZERO_GRADIENT,
                BoundaryStrategy.FIXED_FLUX_PRESSURE,
                BoundaryStrategy.CYCLIC,
            ):
                bc_type_codes[idx] = 0
                p_boundary_values[idx] = 0.0
            else:
                raise RuntimeError(f"Unhandled pressure boundary strategy {strategy!r}")
            idx += 1

    return bc_type_codes, p_boundary_values, boundary_face_indices


def assemble_pressure_correction_equation_rhie_chow(
    U_star,
    A_U,
    p,
    rho,
    mesh_data,
    geo_data,
    boundaries,
    alpha_u=1.0,
    pressure_constraint="reference",
    matrix_workspace=None,
    operator_backend="numpy",
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
    face_conductance = _compute_pressure_face_conductance(mesh_data, geo_data, DU)

    # Use direct gradient computation for full pressure field p
    _grad_fn = gradients._resolve_gradient_fn(geo_data)
    grad_p = _grad_fn(p, mesh_data, geo_data)
    if grad_p.ndim == 3:
        grad_p = grad_p.squeeze(-1)
    grad_p = _update_fixed_flux_pressure_boundaries(
        p, U_star, DU, mesh_data, geo_data, boundaries, grad_p=grad_p
    )

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

    # Unit owner→neighbour vector
    e_vec = cf_vec / (mag_cf[:, np.newaxis] + 1e-12)

    # Orthogonal projection: sf_dot_e = Sf · e
    sf_dot_e = np.sum(sf * e_vec, axis=1)

    # geoDiff uses the ORTHOGONAL component Sf·e (not mag_sf)
    geo_diff = face_conductance[:n_interior]

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
            face_conductance,
        )
        for i in range(n_boundary_faces):
            i_face = bnd_face_idx[i]
            flux_cf[i_face] = cf_b[i]
            flux_ff[i_face] = ff_b[i]
            flux_vf[i_face] = vf_b[i]

        boundary_neighbours = np.asarray(
            mesh_data.get("boundary_neighbours", np.full(n_faces, -1, dtype=np.int32))
        )
        cyclic_faces = np.flatnonzero(boundary_neighbours >= 0)
        if cyclic_faces.size:
            own_b = owners[cyclic_faces]
            nei_b = boundary_neighbours[cyclic_faces]
            weight_b = geo_data["face_weights"][cyclic_faces, np.newaxis]
            sf_b = geo_data["face_sf"][cyclic_faces]
            cf_b = geo_data["face_cf_vector"][cyclic_faces]
            mag_cf_b = np.linalg.norm(cf_b, axis=1)
            edge_b = cf_b / mag_cf_b[:, np.newaxis]
            du_b = weight_b * DU[nei_b] + (1.0 - weight_b) * DU[own_b]
            hbya_b = weight_b * U_HbyA[nei_b] + (1.0 - weight_b) * U_HbyA[own_b]
            grad_b = weight_b * grad_p[nei_b] + (1.0 - weight_b) * grad_p[own_b]
            orthogonal_area = np.sum(sf_b * edge_b, axis=1)
            nonorthogonal = sf_b - orthogonal_area[:, np.newaxis] * edge_b
            conductance_b = face_conductance[cyclic_faces]
            flux_cf[cyclic_faces] = rho * conductance_b
            flux_ff[cyclic_faces] = -rho * conductance_b
            flux_hbya = rho * np.sum(hbya_b * sf_b, axis=1)
            compact = rho * conductance_b * (p[own_b] - p[nei_b])
            nonorthogonal_flux = rho * np.sum(nonorthogonal * du_b * grad_b, axis=1)
            flux_vf[cyclic_faces] = flux_hbya + compact + nonorthogonal_flux

    # 4. Assemble Matrix and RHS
    flux_data = {"flux_cf": flux_cf, "flux_ff": flux_ff, "flux_vf": flux_vf}

    A_p = matrix_assembly.assemble_matrix_from_fluxes_vectorized(
        flux_data, mesh_data, workspace=matrix_workspace, backend=operator_backend
    )
    b_p = matrix_assembly.assemble_rhs_from_fluxes_vectorized(
        flux_data, mesh_data, backend=operator_backend
    )

    # 5. Fix Pressure Reference only for an all-Neumann pressure problem.
    if _pressure_requires_constraint(boundaries, U_star, mesh_data, geo_data):
        if pressure_constraint == "reference":
            A_p, b_p = cavity_utils.fix_pressure_reference(A_p, b_p)
        elif pressure_constraint == "nullspace":
            # A finite-volume all-Neumann RHS should already be compatible;
            # remove only accumulated roundoff before the backend projection.
            parallel = mesh_data.get("_parallel_context")
            if parallel is not None and parallel.is_partitioned:
                n_owned = parallel.n_owned
                global_sum = parallel.global_sum(float(np.sum(b_p[:n_owned])))
                b_p = b_p - global_sum / parallel.partition.global_n_cells
            else:
                b_p = b_p - np.mean(b_p)
        else:
            raise ValueError(
                "All-Neumann pressure requires pressure_constraint='reference' or 'nullspace'"
            )

    return A_p, b_p, flux_vf


def _extend_p_prime_bcs(p_prime, mesh_data, boundaries, face_flux=None):
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
        bc_type_p = boundary.get("bc_type_p")
        strategy = BOUNDARIES.strategy(bc_type_p, "p", "ghost")
        if strategy is BoundaryStrategy.FIXED_VALUE:
            p_prime_ext[idx : idx + nf] = 0.0
        elif strategy is BoundaryStrategy.CYCLIC:
            paired = mesh_data["boundary_neighbours"][start : start + nf]
            p_prime_ext[idx : idx + nf] = p_prime[paired]
        elif strategy is BoundaryStrategy.FREESTREAM:
            if face_flux is None:
                raise ValueError("Freestream pressure correction requires the predicted face flux")
            outflow = np.asarray(face_flux)[start : start + nf] >= 0.0
            p_prime_ext[idx : idx + nf] = np.where(outflow, 0.0, p_prime[own])
        elif strategy in (
            BoundaryStrategy.ZERO_GRADIENT,
            BoundaryStrategy.FIXED_FLUX_PRESSURE,
            BoundaryStrategy.EMPTY,
        ):
            p_prime_ext[idx : idx + nf] = p_prime[own]
        else:
            raise RuntimeError(f"Unhandled pressure ghost strategy {strategy!r}")
    return p_prime_ext


def _correct_interior_fluxes(phi, p_prime, mesh_data, face_conductance, rho):
    """Correct interior face fluxes with the Rhie-Chow pressure correction.

    Applies the flux correction ``Δφ = ρ⋅g⋅(p'_P − p'_N)`` where *g* is
    the geometric diffusion coefficient based on the interpolated DU and
    face-normal projection.

    Args:
        phi:      Face flux array ``(n_faces,)`` (mutated in place).
        p_prime:  Pressure correction ``(n_elements,)``.
        mesh_data: Mesh dictionary.
        face_conductance: Shared pressure-face conductance array.
        rho:      Density.
    """
    n_interior = mesh_data["n_interior_faces"]
    owners = mesh_data["owners"]
    neighbours = mesh_data["neighbours"]
    geo_diff = face_conductance[:n_interior]
    phi[:n_interior] += (
        rho * geo_diff * (p_prime[owners[:n_interior]] - p_prime[neighbours[:n_interior]])
    )


def _correct_boundary_fluxes(phi, p_prime, boundaries, owners, face_conductance, rho):
    """Correct boundary-face fluxes for ``fixedValue`` pressure boundaries.

    Only patches whose pressure BC type is ``fixedValue`` receive a
    correction, computed from the geometric diffusion and wall distance.

    Args:
        phi:        Face flux array ``(n_faces,)`` (mutated in place).
        p_prime:    Pressure correction ``(n_elements,)``.
        boundaries: List of boundary patch dictionaries.
        owners:     Owner index array.
        face_conductance: Shared pressure-face conductance array.
        rho:        Density.
    """
    for boundary in boundaries:
        start = boundary["startFace"]
        nf = boundary["nFaces"]
        idx = np.arange(start, start + nf)
        own = owners[idx]
        geo_diff_b = face_conductance[idx]
        bc_type = boundary.get("bc_type_p")
        strategy = BOUNDARIES.strategy(bc_type, "p", "flux")
        if strategy is BoundaryStrategy.FIXED_VALUE:
            phi[idx] += rho * geo_diff_b * p_prime[own]
        elif strategy is BoundaryStrategy.CYCLIC:
            paired = boundary.get("_paired_cells")
            if paired is None:
                # The mesh-level array is not part of this helper's historical
                # signature; cyclic setup stores the same view on each patch.
                raise ValueError(f"Cyclic patch {boundary.get('name')!r} lacks paired cells")
            phi[idx] += rho * geo_diff_b * (p_prime[own] - p_prime[paired])
        elif strategy is BoundaryStrategy.FREESTREAM:
            outflow = phi[idx] >= 0.0
            phi[idx] += np.where(outflow, rho * geo_diff_b * p_prime[own], 0.0)
        elif strategy not in (
            BoundaryStrategy.ZERO_GRADIENT,
            BoundaryStrategy.FIXED_FLUX_PRESSURE,
            BoundaryStrategy.EMPTY,
        ):
            raise RuntimeError(f"Unhandled pressure flux strategy {strategy!r}")


def _apply_inlet_outlet_bc(U, phi, boundary, owners, n_elements, n_interior):
    """inletOutlet velocity BC: zeroGradient on outflow, fixed value on inflow.

    Per face: outgoing flux (φ ≥ 0) → extrapolate from the owner cell
    (zeroGradient); incoming flux (φ < 0) → impose ``value_U_field`` when
    present, otherwise the uniform ``value_U`` (the inletValue, default 0).
    The per-face path is required by the FVM--VPM characteristic donor:
    pressure-correction refreshes must not replace its non-uniform donor trace
    with the uniform freestream.
    """
    start = boundary["startFace"]
    nf = boundary["nFaces"]
    idx = n_elements + (start - n_interior)
    own = owners[start : start + nf]
    if boundary.get("value_U_field") is not None:
        inlet_val = np.asarray(boundary["value_U_field"], dtype=float)
        if inlet_val.shape != (nf, 3):
            raise ValueError(
                f"Per-face inlet velocity for patch {boundary.get('name')!r} "
                f"must have shape ({nf}, 3), got {inlet_val.shape}"
            )
    else:
        inlet_val = np.asarray(boundary.get("value_U", [0.0, 0.0, 0.0]), dtype=float)
        if inlet_val.shape != (3,):
            raise ValueError(
                f"Uniform inlet velocity for patch {boundary.get('name')!r} "
                f"must have shape (3,), got {inlet_val.shape}"
            )
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


def _apply_cyclic_bc(U, boundary, mesh_data, n_elements, n_interior):
    """Copy paired owner values into the cyclic patch ghost layer."""
    start = boundary["startFace"]
    nf = boundary["nFaces"]
    idx = n_elements + (start - n_interior)
    paired = mesh_data["boundary_neighbours"][start : start + nf]
    if np.any(paired < 0):
        raise ValueError(f"Cyclic patch {boundary['name']!r} is not paired")
    U[idx : idx + nf] = U[paired]


def _apply_fixed_value_bc(U, boundary, n_elements, n_interior, strategy):
    """Apply fixedValue or noSlip velocity BC.

    Honours a per-face ``value_U_field`` (n_faces_patch, 3) when present (e.g. a
    non-uniform coupler donor BC), otherwise the uniform ``value_U``.
    """
    start = boundary["startFace"]
    nf = boundary["nFaces"]
    idx = n_elements + (start - n_interior)
    if strategy is BoundaryStrategy.NO_SLIP:
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


def _update_velocity_bcs(
    U, phi, boundaries, owners, geo_data, n_elements, n_interior, mesh_data=None
):
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
        strategy = BOUNDARIES.strategy(bc_type_u, "U", "ghost")
        if strategy is BoundaryStrategy.ZERO_GRADIENT:
            _apply_zero_gradient_bc(U, phi, boundary, owners, n_elements, n_interior, boundaries)
        elif strategy in (BoundaryStrategy.INLET_OUTLET, BoundaryStrategy.FREESTREAM):
            _apply_inlet_outlet_bc(U, phi, boundary, owners, n_elements, n_interior)
        elif strategy is BoundaryStrategy.DIRECTION_MIXED:
            _apply_robin_bc(U, boundary, owners, geo_data, n_elements, n_interior)
        elif strategy in (BoundaryStrategy.FIXED_VALUE, BoundaryStrategy.NO_SLIP):
            _apply_fixed_value_bc(U, boundary, n_elements, n_interior, strategy)
        elif strategy in (
            BoundaryStrategy.EMPTY,
            BoundaryStrategy.SLIP,
            BoundaryStrategy.SYMMETRY,
        ):
            _apply_slip_bc(U, boundary, owners, geo_data, n_elements, n_interior)
        elif strategy is BoundaryStrategy.CYCLIC:
            if mesh_data is None:
                raise ValueError("Cyclic boundary update requires mesh_data")
            _apply_cyclic_bc(U, boundary, mesh_data, n_elements, n_interior)
        else:
            raise RuntimeError(f"Unhandled velocity ghost strategy {strategy!r}")


def correct_velocity_and_flux(
    U, phi, p_prime, A_U, mesh_data, geo_data, boundaries, rho=1.0, alpha_u=1.0
):
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
    face_conductance = _compute_pressure_face_conductance(mesh_data, geo_data, DU)

    # 1. Correct Cell Velocity
    p_prime_ext = _extend_p_prime_bcs(p_prime, mesh_data, boundaries, face_flux=phi)
    _grad_fn = gradients._resolve_gradient_fn(geo_data)
    grad_p_prime = _grad_fn(p_prime_ext, mesh_data, geo_data)
    if grad_p_prime.ndim == 3:
        grad_p_prime = grad_p_prime.squeeze(-1)
    U[:n_elements] -= DU * grad_p_prime[:n_elements]

    # 2. Correct Face Fluxes
    _correct_interior_fluxes(phi, p_prime, mesh_data, face_conductance, rho)
    _correct_boundary_fluxes(phi, p_prime, boundaries, owners, face_conductance, rho)

    # 3. Update Velocity BCs
    _update_velocity_bcs(
        U, phi, boundaries, owners, geo_data, n_elements, n_interior, mesh_data=mesh_data
    )

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


def _apply_scalar_bc(
    phi,
    indices,
    owners_b,
    strategy,
    boundary,
    field_name,
    paired_owners=None,
    face_flux=None,
):
    """Apply a boundary condition to a scalar field ghost-cell block.

    Zero-gradient BCs (including inlet, outlet, symmetry, empty, slip,
    noSlip) copy the owner value.  ``fixedValue`` sets the prescribed
    boundary value.

    Args:
        phi:        Scalar field array (mutated in place).
        indices:    Ghost-cell indices for this patch.
        owners_b:   Owner cell indices for the boundary faces.
        strategy:   Validated boundary behavior.
        boundary:   Boundary patch dictionary (may contain ``value_p`` etc.).
        field_name: Field name for value lookup (e.g. ``"p"``, ``"phi"``).
    """
    if strategy in (BoundaryStrategy.ZERO_GRADIENT, BoundaryStrategy.EMPTY):
        phi[indices] = phi[owners_b]
    elif strategy is BoundaryStrategy.FIXED_FLUX_PRESSURE:
        delta = boundary.get("fixed_flux_pressure_delta")
        if delta is None:
            phi[indices] = phi[owners_b]
        else:
            phi[indices] = phi[owners_b] + np.asarray(delta, dtype=float)
    elif strategy is BoundaryStrategy.FIXED_VALUE:
        val = boundary.get(f"value_{field_name}")
        if val is None:
            val = boundary.get("value")
        if val is None:
            raise ValueError(
                f"Fixed-value {field_name} boundary {boundary.get('name')!r} has no value"
            )
        phi[indices] = val
    elif strategy is BoundaryStrategy.CYCLIC:
        if paired_owners is None or np.any(paired_owners < 0):
            raise ValueError(f"Cyclic patch {boundary.get('name')!r} is not paired")
        phi[indices] = phi[paired_owners]
    elif strategy is BoundaryStrategy.FREESTREAM:
        if face_flux is None:
            raise ValueError("Freestream scalar update requires face fluxes")
        val = boundary.get(f"value_{field_name}")
        if val is None:
            val = boundary.get("value")
        if val is None:
            raise ValueError(
                f"Freestream {field_name} boundary {boundary.get('name')!r} has no value"
            )
        phi[indices] = np.where(np.asarray(face_flux) >= 0.0, val, phi[owners_b])
    else:
        raise ValueError(
            f"Unsupported scalar boundary strategy {strategy!r} "
            f"for {field_name} on patch {boundary.get('name')!r}"
        )


def update_scalar_boundaries(phi, mesh_data, boundaries, field_name="p", face_flux=None):
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
        paired = mesh_data.get("boundary_neighbours")
        paired_owners = None if paired is None else paired[start : start + n_bfaces]
        patch_flux = None if face_flux is None else np.asarray(face_flux)[start : start + n_bfaces]

        # Get BC type
        bc_type = boundary.get(f"bc_type_{field_name}") or boundary.get("bc_type")
        registry_field = "p" if field_name == "p" else "scalar"
        strategy = BOUNDARIES.strategy(bc_type, registry_field, "ghost")
        _apply_scalar_bc(
            phi,
            b_elem_indices,
            owners_b,
            strategy,
            boundary,
            field_name,
            paired_owners=paired_owners,
            face_flux=patch_flux,
        )


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
        self.last_linear_results = ()
        self.last_outer_diagnostics = ()
        self._momentum_matrix_workspace = matrix_assembly.MatrixAssemblyWorkspace.create(mesh_data)
        self._pressure_matrix_workspace = matrix_assembly.MatrixAssemblyWorkspace.create(mesh_data)

    def step(
        self,
        U,
        p,
        phi,
        U_old=None,
        dt=None,
        rho=1.0,
        nu=0.01,
        U_old_old=None,
        source_explicit=None,
        source_implicit=None,
    ):
        """
        Perform a single SIMPLE iteration.

        ``U_old_old`` is accepted for interface parity with the transient driver
        but is unused by steady SIMPLE.  ``source_explicit``/``source_implicit``
        are optional volumetric momentum sources (e.g. the coupling fringe
        S = λ(Utarget − U)) forwarded to the momentum predictor.
        """
        # 1. Solve momentum predictor
        U_star, A_U, momentum_diagnostics = momentum.solve_momentum_predictor(
            U,
            p,
            phi,
            rho,
            nu,
            self.mesh_data,
            self.geo_data,
            self.boundaries,
            convection_scheme=self.params["convection_scheme"],
            solver=self.params.get("momentum_solver") or self.params["linear_solver"],
            under_relaxation=self.params["alpha_u"],
            dt=dt,  # Use dt if provided (e.g. for PIMPLE)
            source_explicit=source_explicit,
            source_implicit=source_implicit,
            linear_backend=self.params.get("_linear_backend", "scipy"),
            parallel_context=self.params.get("_parallel_context"),
            failure_policy=self.params.get("linear_failure_policy", "raise"),
            momentum_tol=self.params.get("momentum_tol", 1e-4),
            maxiter=self.params.get("momentum_maxiter", 1000),
            reuse_ilu=self.params.get("reuse_ilu", False),
            ilu_drop_tol=self.params.get("ilu_drop_tol", 1e-4),
            ilu_fill_factor=self.params.get("ilu_fill_factor", 10),
            ilu_reuse_tol=self.params.get("ilu_reuse_tol"),
            matrix_workspace=self._momentum_matrix_workspace,
            operator_backend=self.params.get("_operator_backend", "numpy"),
            return_diagnostics=True,
        )

        # 2. Solve pressure correction
        pressure_constraint = _resolve_pressure_constraint(self.params)
        has_pressure_nullspace = _pressure_requires_constraint(
            self.boundaries, U_star, self.mesh_data, self.geo_data
        )
        A_p, b_p, phi_star = assemble_pressure_correction_equation_rhie_chow(
            U_star,
            A_U,
            p,
            rho,
            self.mesh_data,
            self.geo_data,
            self.boundaries,
            alpha_u=self.params["alpha_u"],
            pressure_constraint=pressure_constraint,
            matrix_workspace=self._pressure_matrix_workspace,
            operator_backend=self.params.get("_operator_backend", "numpy"),
        )

        p_prime, pressure_result = matrix_assembly.solve_linear_system(
            A_p,
            b_p,
            method=self.params.get("pressure_solver") or self.params["linear_solver"],
            equation_type="pressure",
            tol=self.params.get("pressure_tol", 1e-8),
            maxiter=self.params.get("pressure_maxiter", 500),
            backend=self.params.get("_linear_backend", "scipy"),
            parallel_context=self.params.get("_parallel_context"),
            failure_policy=self.params.get("linear_failure_policy", "raise"),
            nullspace=(
                "constant"
                if has_pressure_nullspace and pressure_constraint == "nullspace"
                else None
            ),
            return_info=True,
        )

        # 3. Correct velocity and flux
        # Calculate residual before in-place modification of U_star
        velocity_increment = np.linalg.norm(
            U_star[: self.mesh_data["n_elements"]] - U[: self.mesh_data["n_elements"]]
        ) / (np.linalg.norm(U[: self.mesh_data["n_elements"]]) + 1e-10)

        U, phi = correct_velocity_and_flux(
            U_star,
            phi_star,
            p_prime,
            A_U,
            self.mesh_data,
            self.geo_data,
            self.boundaries,
            rho=rho,
            alpha_u=self.params["alpha_u"],
        )

        # 4. Update pressure
        p[: self.mesh_data["n_elements"]] += self.params["alpha_p"] * p_prime

        # Update pressure boundaries
        update_scalar_boundaries(p, self.mesh_data, self.boundaries, field_name="p", face_flux=phi)

        # 5. Residuals
        self.last_res_p = matrix_assembly.normalized_residual(A_p, p_prime, b_p)
        self.last_res_u = max(
            (values["final_residual"] for values in momentum_diagnostics.values()),
            default=0.0,
        )
        continuity = field_diagnostics.compute_continuity_error(phi, self.mesh_data, self.geo_data)
        volumes = self.geo_data["element_volumes"]
        continuity_max = float(np.max(np.abs(continuity) / (volumes + 1e-30)))
        self.last_linear_results = tuple(
            values["linear_result"] for values in momentum_diagnostics.values()
        ) + (pressure_result,)
        self.last_outer_diagnostics = (
            OuterCorrectorDiagnostics(
                index=0,
                momentum_residual=self.last_res_u,
                pressure_residual=self.last_res_p,
                continuity_max=continuity_max,
            ),
        )

        residuals = {"p": self.last_res_p, "U": self.last_res_u, "U_increment": velocity_increment}
        residuals.update(
            {
                f"U_{component}": values["final_residual"]
                for component, values in momentum_diagnostics.items()
            }
        )
        return U, p, phi, residuals

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
            residual_u = residuals["U_increment"]
            continuity = self.last_outer_diagnostics[-1].continuity_max

            self.residuals.append(
                {
                    "iter": iteration,
                    "R_p": residual_p,
                    "R_u": residual_u,
                    "continuity": continuity,
                }
            )

            if iteration % 10 == 0 or residual_p < self.params["tolerance"]:
                print(
                    f"  Iter {iteration:3d}: R_p={residual_p:.3e}, "
                    f"ΔU={residual_u:.3e}, continuity={continuity:.3e}"
                )

            if (
                residual_p < self.params["tolerance"]
                and residual_u < self.params["tolerance"]
                and continuity < self.params["tolerance"]
            ):
                print(f"  ✓ Converged in {iteration} iterations")
                return U, p, phi, True

        print(f"  ✗ Did not converge in {self.params['max_iter']} iterations")
        return U, p, phi, False
