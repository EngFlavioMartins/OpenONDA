#!/usr/bin/env python3
"""
Momentum Equation Assembly for OpenONDA FVM Solver

Assembles momentum equation for incompressible Navier-Stokes:
∂U/∂t + ∇·(UU) = -∇p/ρ + ∇·(nu∇U) + f

Components:
- Convection: ∇·(ρUU)
- Diffusion: ∇·(μ∇U)  where μ = ρnu
- Pressure gradient: -∇p
- Source terms: f (gravity, etc.)

Converted from uFVM NSAssembly/Momentum modules
"""

import numpy as np

from ..fields import gradients
from . import convection, diffusion, matrix_assembly


def _make_momentum_boundary(b: dict, i_comp: int) -> dict:
    """Create per-component momentum boundary dict from velocity boundary."""
    b_mom = b.copy()
    if "bc_type_U" in b:
        b_mom["bc_type"] = b["bc_type_U"]
    if "value_U" in b:
        val = b["value_U"]
        b_mom["value"] = val[i_comp] if (np.ndim(val) == 1 and len(val) == 3) else val
    return b_mom


def _add_transient_term(A, b_vec, rho, vol, dt, U_old_comp, U_curr_comp):
    """Apply Euler implicit transient term (only called when dt is not None)."""
    if U_old_comp is not None:
        coeff = rho * vol / dt
        b_vec += coeff * U_old_comp
    else:
        coeff = vol / dt
        b_vec += coeff * U_curr_comp
    A.setdiag(A.diagonal() + coeff)


def _apply_empty_bc_ustar(U_star, b_elem_indices, owners_b, face_sf):
    """Apply empty BC: zero out normal velocity component from owner cell."""
    for b_idx, own, sf in zip(b_elem_indices, owners_b, face_sf, strict=False):
        norm_sf = np.linalg.norm(sf)
        if norm_sf > 1e-10:
            n_hat = sf / norm_sf
            U_owner = U_star[own]
            U_star[b_idx] = U_owner - np.dot(U_owner, n_hat) * n_hat
        else:
            U_star[b_idx] = U_star[own]


def _apply_ustar_bc(U_star, U, boundary, mesh_data, geo_data, n_elements):
    """Apply post-solve boundary condition to the predicted velocity field."""
    start_face = boundary["startFace"]
    n_faces = boundary["nFaces"]
    b_elem_start = start_face - mesh_data["n_interior_faces"]
    b_elem_indices = np.arange(n_elements + b_elem_start, n_elements + b_elem_start + n_faces)
    bc_type = boundary.get("bc_type_U")

    if bc_type == "zeroGradient":
        owners_b = mesh_data["owners"][start_face : start_face + n_faces]
        U_star[b_elem_indices] = U_star[owners_b]
    elif bc_type in ("fixedValue", "noSlip"):
        if bc_type == "noSlip":
            U_star[b_elem_indices] = [0.0, 0.0, 0.0]
        elif "value_U" in boundary:
            U_star[b_elem_indices] = np.array(boundary["value_U"])
        else:
            U_star[b_elem_indices] = U[b_elem_indices]
    elif bc_type == "empty":
        owners_b = mesh_data["owners"][start_face : start_face + n_faces]
        face_sf = geo_data["face_sf"][start_face : start_face + n_faces]
        _apply_empty_bc_ustar(U_star, b_elem_indices, owners_b, face_sf)
    else:
        U_star[b_elem_indices] = U[b_elem_indices]


def assemble_momentum_equation(
    U,
    p,
    phi,
    rho,
    nu,
    mesh_data,
    geo_data,
    boundaries,
    convection_scheme="deferred",
    dt=None,
    U_old=None,
):
    """
    Assemble momentum equation for all three velocity components.

    Returns coefficient matrix and RHS for each component separately.

    Args:
        U: Velocity field (n_elements + n_boundary, 3)
        p: Pressure field (n_elements + n_boundary,)
        rho: Density (scalar or array)
        nu: Kinematic viscosity (scalar or array)
        mesh_data: Mesh connectivity
        geo_data: Geometric data
        boundaries: Boundary conditions
        convection_scheme: Convection discretization scheme
        dt: Time step size (optional)
        U_old: Previous time step velocity (optional, for transient term)

    Returns:
        dict: For each component (x, y, z):
            - A: Coefficient matrix
            - b: RHS vector
            - H: H operator (for pressure correction)
    """

    n_elements = mesh_data["n_elements"]

    # Ensure rho and nu are arrays
    if np.isscalar(rho):
        rho = np.full(n_elements, rho)
    if np.isscalar(nu):
        nu = np.full(n_elements, nu)

    # Dynamic viscosity: μ = ρnu
    mu = rho * nu

    # Resolve gradient scheme
    _grad_fn = gradients._resolve_gradient_fn(geo_data)

    # Compute velocity gradient
    grad_U = _grad_fn(U, mesh_data, geo_data)

    # Compute pressure gradient
    grad_p = _grad_fn(p, mesh_data, geo_data)
    if grad_p.ndim == 3:
        grad_p = grad_p.squeeze(-1)  # (n, 3, 1) -> (n, 3)

    # Compute mass flow rate for convection (DENSITY SCALED)
    # If phi is already rho-weighted, use as is.
    # In OpenONDA FVM, phi is usually U.Sf. Let's ensure rho scaling.
    mdot = phi * rho[0] if len(rho) > 0 else phi

    results = {}

    # Assemble for each component
    for i_comp, comp_name in enumerate(["x", "y", "z"]):
        # Get velocity component
        U_comp = U[:, i_comp]

        # Get gradient component
        grad_U_comp = (
            grad_U[:, :, i_comp] if grad_U.ndim == 3 else grad_U
        )  # (n_elements, 3) or full

        momentum_boundaries = [_make_momentum_boundary(b, i_comp) for b in boundaries]

        # 1. Diffusion term: ∇·(μ∇U)
        diff_flux = diffusion.assemble_diffusion_term(
            U_comp, grad_U_comp, mu, mesh_data, geo_data, momentum_boundaries
        )

        # 2. Convection term: ∇·(ρUU)
        conv_flux = convection.assemble_convection_term(
            U_comp, mdot, mesh_data, geo_data, boundaries, scheme=convection_scheme
        )

        # 3. Combine fluxes (diffusion + convection)
        combined_flux = {
            "flux_cf": diff_flux["flux_cf"] + conv_flux["flux_cf"],
            "flux_ff": diff_flux["flux_ff"] + conv_flux["flux_ff"],
            "flux_vf": diff_flux["flux_vf"] + conv_flux["flux_vf"],
            "flux_tf": diff_flux["flux_tf"] + conv_flux["flux_tf"],
        }

        # 4. Assemble matrix and RHS
        A = matrix_assembly.assemble_matrix_from_fluxes_vectorized(combined_flux, mesh_data)
        b = matrix_assembly.assemble_rhs_from_fluxes_vectorized(combined_flux, mesh_data)

        # 5. Add pressure gradient to RHS: -∇p * V
        vol = geo_data["element_volumes"]
        grad_p_comp = grad_p[:n_elements, i_comp]
        b -= grad_p_comp * vol

        # 6. Transient Term (Euler Implicit)
        if dt is not None:
            _add_transient_term(
                A,
                b,
                rho,
                vol,
                dt,
                U_old[:n_elements, i_comp] if U_old is not None else None,
                U[:n_elements, i_comp],
            )

        # 7. Compute H operator (diagonal of A, needed for pressure correction)
        H = A.diagonal().copy()

        results[comp_name] = {
            "A": A,
            "b": b,
            "H": H,
            "grad_p_comp": grad_p_comp,  # Store for debug
        }

    return results


def solve_momentum_predictor(
    U,
    p,
    phi,
    rho,
    nu,
    mesh_data,
    geo_data,
    boundaries,
    convection_scheme="deferred",
    solver="spsolve",
    under_relaxation=0.7,
    dt=None,
    U_old=None,
    **solver_kwargs,
):
    """
    Solve momentum equation to get predicted velocity (U*).
    """

    n_elements = mesh_data["n_elements"]
    n_boundary = mesh_data["n_faces"] - mesh_data["n_interior_faces"]

    # Assemble momentum equations
    mom_eqs = assemble_momentum_equation(
        U, p, phi, rho, nu, mesh_data, geo_data, boundaries, convection_scheme, dt=dt, U_old=U_old
    )

    # Solve for each component
    U_star = np.zeros((n_elements + n_boundary, 3))
    A_U = np.zeros((n_elements, 3))

    for i_comp, comp_name in enumerate(["x", "y", "z"]):
        A = mom_eqs[comp_name]["A"]
        b = mom_eqs[comp_name]["b"]

        # Apply under-relaxation to diagonal (Patankar method)
        # A_ii_new = A_ii / alpha
        # b_new = b + (1 - alpha) * A_ii_new * U_old
        # Note: we modify A in-place to avoid a full CSR copy.
        # This is safe because A is not used again after this point.

        diag_old = A.diagonal()
        diag_new = diag_old / under_relaxation
        A.setdiag(diag_new)
        A_relaxed = A

        # Add source term to RHS to ensure consistency at convergence
        # U[:n_elements, i_comp] is the old velocity value
        source_relax = (1.0 - under_relaxation) * diag_new * U[:n_elements, i_comp]
        b_relaxed = b + source_relax

        # Determine initial guess x0 for iterative solver: prefer U_old (previous time step) if available
        momentum_tol = solver_kwargs.get("momentum_tol", 1e-4)
        x0_vec = None
        if U_old is not None:
            try:
                x0_vec = U_old[:n_elements, i_comp]
            except Exception:
                x0_vec = None

        # Solve with optional initial guess and tuned tolerance
        solver_kwargs.pop("ilu_key", None)
        U_comp_star = matrix_assembly.solve_linear_system(
            A_relaxed,
            b_relaxed,
            method=solver,
            equation_type="momentum",
            tol=momentum_tol,
            x0=x0_vec,
            ilu_key=comp_name,
            **solver_kwargs,
        )

        # Store results
        U_star[:n_elements, i_comp] = U_comp_star
        # Return relaxed diagonal coefficients for pressure correction
        # A_U = A_ii / alpha (matching uFVM: DU = volumes/ac where ac is relaxed)
        A_U[:, i_comp] = diag_new

    for boundary in boundaries:
        _apply_ustar_bc(U_star, U, boundary, mesh_data, geo_data, n_elements)

    return U_star, A_U


def compute_H_operator(A_U):
    """
    Compute H operator for pressure correction.

    H = 1 / A_U (diagonal coefficients)

    Args:
        A_U: Diagonal coefficients from momentum equation (n_elements, 3)

    Returns:
        numpy.ndarray: H operator (n_elements, 3)
    """

    # Avoid division by zero
    H = 1.0 / (A_U + 1e-10)

    return H
