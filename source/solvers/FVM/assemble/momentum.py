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

from dataclasses import replace

import numpy as np

from ..fields import gradients
from . import convection, diffusion, matrix_assembly


def _make_momentum_boundary(b: dict, i_comp: int) -> dict:
    """Create per-component momentum boundary dict from velocity boundary.

    Args:
        b: Velocity boundary dictionary containing keys like ``bc_type_U``
            and ``value_U``.
        i_comp: Component index (0, 1, 2 for x, y, z).

    Returns:
        dict: Momentum boundary dictionary with ``bc_type`` and ``value``
            keys extracted for the specified component.
    """
    b_mom = b.copy()
    if "bc_type_U" in b:
        b_mom["bc_type"] = b["bc_type_U"]
    if "value_U" in b:
        val = b["value_U"]
        b_mom["value"] = val[i_comp] if (np.ndim(val) == 1 and len(val) == 3) else val
    return b_mom


def _add_transient_term(
    A, b_vec, rho, vol, dt, U_old_comp, U_curr_comp, U_old_old_comp=None, scheme="euler"
):
    """Apply the implicit transient ``∂(ρu)/∂t`` term (only called when dt is not None).

    Schemes (constant Δt):
      * ``"euler"`` / ``"backward_euler"`` — BDF1, first order:
        ``(ρV/Δt)(uⁿ⁺¹ − uⁿ)``.
      * ``"backward"`` — BDF2, second order:
        ``(ρV/Δt)(3/2 uⁿ⁺¹ − 2 uⁿ + 1/2 uⁿ⁻¹)``.  Falls back to BDF1 on the
        first step (when ``U_old_old_comp`` is None), which is the standard
        self-starting BDF2 procedure.

    BDF2 here assumes a constant time step. Configuration validation rejects
    adaptive time stepping with BDF2 before assembly.
    """
    coeff = rho * vol / dt
    if scheme == "backward" and U_old_comp is not None and U_old_old_comp is not None:
        A.setdiag(A.diagonal() + 1.5 * coeff)
        b_vec += coeff * (2.0 * U_old_comp - 0.5 * U_old_old_comp)
    else:
        # BDF1 (Euler implicit), also the BDF2 startup step.
        A.setdiag(A.diagonal() + coeff)
        b_vec += coeff * (U_old_comp if U_old_comp is not None else U_curr_comp)


def _apply_empty_bc_ustar(U_star, b_elem_indices, owners_b, face_sf):
    """Apply empty boundary condition: zero out normal velocity component.

    For ``empty`` boundaries (used in 2D extruded meshes for the
    non-solved dimension), the face-normal component of the predicted
    velocity is removed, leaving only the tangential components.

    Args:
        U_star: Predicted velocity field (n_elements + n_boundary, 3),
            modified in-place.
        b_elem_indices: Indices of boundary elements to update.
        owners_b: Owner element indices for each boundary face.
        face_sf: Face surface area vectors for each boundary face.
    """
    for b_idx, own, sf in zip(b_elem_indices, owners_b, face_sf, strict=False):
        norm_sf = np.linalg.norm(sf)
        if norm_sf > 1e-10:
            n_hat = sf / norm_sf
            U_owner = U_star[own]
            U_star[b_idx] = U_owner - np.dot(U_owner, n_hat) * n_hat
        else:
            U_star[b_idx] = U_star[own]


def _apply_ustar_bc(U_star, boundary, mesh_data, geo_data, n_elements):
    """Apply post-solve boundary condition to the predicted velocity field.

    Modifies ``U_star`` in-place at the boundary elements according to
    the boundary condition type specified in the boundary dictionary.

    Args:
        U_star: Predicted velocity field (n_elements + n_boundary, 3),
            modified in-place.
        boundary: Boundary dictionary with keys ``startFace``, ``nFaces``,
            ``bc_type_U``, and optionally ``value_U`` / ``value_U_field``.
        mesh_data: Mesh connectivity containing ``owners`` and
            ``n_interior_faces``.
        geo_data: Geometric data containing ``face_sf``.
        n_elements: Number of interior elements.
    """
    start_face = boundary["startFace"]
    n_faces = boundary["nFaces"]
    b_elem_start = start_face - mesh_data["n_interior_faces"]
    b_elem_indices = np.arange(n_elements + b_elem_start, n_elements + b_elem_start + n_faces)
    bc_type = boundary.get("bc_type_U")

    if bc_type in ("zeroGradient", "inletOutlet", "directionMixed"):
        owners_b = mesh_data["owners"][start_face : start_face + n_faces]
        U_star[b_elem_indices] = U_star[owners_b]
    elif bc_type == "freestream":
        # The boundary ghost was switched using the latest face flux before
        # momentum assembly; preserve that per-face inflow/outflow state.
        pass
    elif bc_type == "cyclic":
        paired = mesh_data["boundary_neighbours"][start_face : start_face + n_faces]
        U_star[b_elem_indices] = U_star[paired]
    elif bc_type in ("fixedValue", "noSlip"):
        if bc_type == "noSlip":
            U_star[b_elem_indices] = [0.0, 0.0, 0.0]
        elif boundary.get("value_U_field") is not None:
            U_star[b_elem_indices] = boundary["value_U_field"]
        elif "value_U" in boundary:
            U_star[b_elem_indices] = np.array(boundary["value_U"])
        else:
            raise ValueError(
                f"Fixed velocity boundary {boundary.get('name')!r} has no configured value"
            )
    elif bc_type in ("empty", "slip", "symmetry"):
        owners_b = mesh_data["owners"][start_face : start_face + n_faces]
        face_sf = geo_data["face_sf"][start_face : start_face + n_faces]
        _apply_empty_bc_ustar(U_star, b_elem_indices, owners_b, face_sf)
    else:
        raise ValueError(
            f"Unsupported momentum velocity boundary condition {bc_type!r} "
            f"on patch {boundary.get('name')!r}"
        )


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
    U_old_old=None,
    ddt_scheme="euler",
    source_explicit=None,
    source_implicit=None,
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
        source_explicit: Optional explicit volumetric source Su (n_elements, 3),
            per unit volume.  Added to the RHS as ``Su * V`` for each component
            (e.g. body force, MMS forcing, or the explicit part λ·Utarget of the
            coupling fringe source).
        source_implicit: Optional implicit volumetric source coefficient Sp
            (n_elements,), per unit volume.  Added to the diagonal as ``Sp * V``
            (e.g. the implicit part λ of the fringe source S = λ(Utarget − U)).
            Must be >= 0 to preserve diagonal dominance.

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

    if rho.shape != (n_elements,) or not np.all(np.isfinite(rho)) or np.any(rho <= 0.0):
        raise ValueError(f"rho must be finite and positive with shape ({n_elements},)")
    if nu.shape != (n_elements,) or not np.all(np.isfinite(nu)) or np.any(nu <= 0.0):
        raise ValueError(f"nu must be finite and positive with shape ({n_elements},)")

    # ``phi`` is volumetric flux U·Sf. Interpolate density to each face to
    # obtain mass flux without collapsing variable-density input to rho[0].
    owners = mesh_data["owners"]
    neighbours = mesh_data["neighbours"]
    n_interior = mesh_data["n_interior_faces"]
    face_rho = rho[owners].copy()
    weights = geo_data["face_weights"][:n_interior]
    face_rho[:n_interior] = (
        weights * rho[neighbours[:n_interior]] + (1.0 - weights) * rho[owners[:n_interior]]
    )
    boundary_neighbours = np.asarray(
        mesh_data.get("boundary_neighbours", np.full(mesh_data["n_faces"], -1, dtype=np.int32))
    )
    cyclic_faces = np.flatnonzero(boundary_neighbours >= 0)
    if cyclic_faces.size:
        weights_b = geo_data["face_weights"][cyclic_faces]
        face_rho[cyclic_faces] = (
            weights_b * rho[boundary_neighbours[cyclic_faces]]
            + (1.0 - weights_b) * rho[owners[cyclic_faces]]
        )
    mdot = np.asarray(phi) * face_rho

    results = {}
    spatial_matrix = None
    vol = geo_data["element_volumes"]

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
            U_comp,
            grad_U_comp,
            mu,
            mesh_data,
            geo_data,
            momentum_boundaries,
            face_flux=phi,
        )

        # 2. Convection term: ∇·(ρUU).  grad_U_comp feeds the gradient-based
        #    TVD limiter for high-resolution schemes (ignored by the others).
        conv_flux = convection.assemble_convection_term(
            U_comp,
            mdot,
            mesh_data,
            geo_data,
            boundaries,
            scheme=convection_scheme,
            grad_phi=grad_U_comp,
        )

        # 3. Combine fluxes (diffusion + convection)
        combined_flux = {
            "flux_cf": diff_flux["flux_cf"] + conv_flux["flux_cf"],
            "flux_ff": diff_flux["flux_ff"] + conv_flux["flux_ff"],
            "flux_vf": diff_flux["flux_vf"] + conv_flux["flux_vf"],
            "flux_tf": diff_flux["flux_tf"] + conv_flux["flux_tf"],
        }

        # 4. Assemble the common implicit matrix once.  Convection,
        # diffusion, transient, and implicit-source coefficients are identical
        # for x/y/z; only explicit corrections and boundary values differ.
        if spatial_matrix is None:
            spatial_matrix = matrix_assembly.assemble_matrix_from_fluxes_vectorized(
                combined_flux, mesh_data
            )
        A = spatial_matrix.copy()
        b = matrix_assembly.assemble_rhs_from_fluxes_vectorized(combined_flux, mesh_data)

        # 5. Add pressure gradient to RHS: -∇p * V
        grad_p_comp = grad_p[:n_elements, i_comp]
        b -= grad_p_comp * vol

        # 6. Transient Term (BDF1/BDF2, selected by ddt_scheme)
        if dt is not None:
            _add_transient_term(
                A,
                b,
                rho,
                vol,
                dt,
                U_old[:n_elements, i_comp] if U_old is not None else None,
                U[:n_elements, i_comp],
                U_old_old[:n_elements, i_comp] if U_old_old is not None else None,
                scheme=ddt_scheme,
            )

        # 6b. Generic volumetric source terms: S = Su + Sp·U (per unit volume).
        #     Su → RHS (+Su·V); Sp → diagonal (+Sp·V), keeping U implicit.
        #     Used by MMS forcing and the coupling fringe S = λ(Utarget − U).
        if source_explicit is not None:
            b += source_explicit[:n_elements, i_comp] * vol
        if source_implicit is not None:
            A.setdiag(A.diagonal() + source_implicit[:n_elements] * vol)

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
    U_old_old=None,
    ddt_scheme="euler",
    source_explicit=None,
    source_implicit=None,
    return_diagnostics=False,
    **solver_kwargs,
):
    """Solve momentum equation to get predicted velocity (U*).

    Assembles and solves the discretised momentum equation for each
    velocity component, applies under-relaxation, and enforces boundary
    conditions on the predicted field.

    Args:
        U: Current velocity field (n_elements + n_boundary, 3).
        p: Current pressure field (n_elements + n_boundary,).
        phi: Mass flow rate field.
        rho: Density (scalar or array).
        nu: Kinematic viscosity (scalar or array).
        mesh_data: Mesh connectivity data.
        geo_data: Geometric data.
        boundaries: List of boundary dictionaries.
        convection_scheme: Convection discretisation scheme
            (``'upwind'``, ``'central'``, ``'deferred'``, etc.).
            Defaults to ``'deferred'``.
        solver: Linear solver method. Defaults to ``'spsolve'``.
        under_relaxation: Under-relaxation factor for the velocity
            update (``0 < alpha <= 1``). Defaults to ``0.7``.
        dt: Time step size. If ``None``, steady-state mode.
        U_old: Velocity at previous time step
            (n_elements + n_boundary, 3).
        U_old_old: Velocity at two time steps ago (BDF2 only).
        ddt_scheme: Time discretisation scheme (``'euler'`` or
            ``'backward'``).
        source_explicit: Explicit volumetric source Su
            (n_elements, 3). Added to RHS as ``Su * V``.
        source_implicit: Implicit volumetric source coefficient Sp
            (n_elements,). Added to diagonal as ``Sp * V``.
            Must be >= 0.
        **solver_kwargs: Additional keyword arguments forwarded to the
            linear solver (e.g., ``momentum_tol``).

    Returns:
        tuple:
            - **U_star** (*numpy.ndarray*) -- Predicted velocity field
              (n_elements + n_boundary, 3).
            - **A_U** (*numpy.ndarray*) -- Relaxed diagonal coefficients
              (n_elements, 3), used in the pressure-correction step.
    """

    n_elements = mesh_data["n_elements"]
    n_boundary = mesh_data["n_faces"] - mesh_data["n_interior_faces"]

    # Assemble momentum equations
    mom_eqs = assemble_momentum_equation(
        U,
        p,
        phi,
        rho,
        nu,
        mesh_data,
        geo_data,
        boundaries,
        convection_scheme,
        dt=dt,
        U_old=U_old,
        U_old_old=U_old_old,
        ddt_scheme=ddt_scheme,
        source_explicit=source_explicit,
        source_implicit=source_implicit,
    )

    # Solve for each component
    U_star = np.zeros((n_elements + n_boundary, 3))
    A_U = np.zeros((n_elements, 3))
    solve_diagnostics = {}
    linear_backend = solver_kwargs.pop("linear_backend", "scipy")
    parallel_context = solver_kwargs.pop("parallel_context", None)

    if solver == "spsolve" and linear_backend == "scipy":
        # All three components share one matrix.  Solve a three-column RHS so
        # SuperLU performs one factorization instead of three.
        A_shared = mom_eqs["x"]["A"]
        diag_old = A_shared.diagonal()
        diag_new = diag_old / under_relaxation
        A_shared.setdiag(diag_new)

        rhs_columns = []
        for i_comp, comp_name in enumerate(["x", "y", "z"]):
            b = mom_eqs[comp_name]["b"]
            source_relax = (1.0 - under_relaxation) * diag_new * U[:n_elements, i_comp]
            rhs_columns.append(b + source_relax)
        B = np.column_stack(rhs_columns)
        X, shared_result = matrix_assembly.solve_linear_system(
            A_shared, B, method="spsolve", return_info=True
        )
        if X.ndim == 1:
            X = X[:, np.newaxis]

        for i_comp, comp_name in enumerate(["x", "y", "z"]):
            U_star[:n_elements, i_comp] = X[:, i_comp]
            A_U[:, i_comp] = diag_new
            b_relaxed = B[:, i_comp]
            x_initial = U_old[:n_elements, i_comp] if U_old is not None else np.zeros(n_elements)
            solve_diagnostics[comp_name] = {
                "initial_residual": matrix_assembly.normalized_residual(
                    A_shared, x_initial, b_relaxed
                ),
                "final_residual": matrix_assembly.normalized_residual(
                    A_shared, X[:, i_comp], b_relaxed
                ),
            }
            solve_diagnostics[comp_name]["linear_result"] = replace(
                shared_result,
                initial_residual=solve_diagnostics[comp_name]["initial_residual"],
                final_residual=solve_diagnostics[comp_name]["final_residual"],
            )

        for boundary in boundaries:
            _apply_ustar_bc(U_star, boundary, mesh_data, geo_data, n_elements)
        if return_diagnostics:
            return U_star, A_U, solve_diagnostics
        return U_star, A_U

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
            if np.asarray(U_old).shape[0] < n_elements:
                raise ValueError("U_old does not contain every interior cell")
            x0_vec = U_old[:n_elements, i_comp]

        # Solve with optional initial guess and tuned tolerance
        solver_kwargs.pop("ilu_key", None)
        x_initial = x0_vec if x0_vec is not None else np.zeros(n_elements)
        initial_residual = matrix_assembly.normalized_residual(A_relaxed, x_initial, b_relaxed)
        U_comp_star, linear_result = matrix_assembly.solve_linear_system(
            A_relaxed,
            b_relaxed,
            method=solver,
            equation_type="momentum",
            tol=momentum_tol,
            x0=x0_vec,
            ilu_key=comp_name,
            backend=linear_backend,
            parallel_context=parallel_context,
            return_info=True,
            **solver_kwargs,
        )
        final_residual = matrix_assembly.normalized_residual(A_relaxed, U_comp_star, b_relaxed)
        solve_diagnostics[comp_name] = {
            "initial_residual": initial_residual,
            "final_residual": final_residual,
            "linear_result": linear_result,
        }

        # Store results
        U_star[:n_elements, i_comp] = U_comp_star
        # Return relaxed diagonal coefficients for pressure correction
        # A_U = A_ii / alpha (matching uFVM: DU = volumes/ac where ac is relaxed)
        A_U[:, i_comp] = diag_new

    for boundary in boundaries:
        _apply_ustar_bc(U_star, boundary, mesh_data, geo_data, n_elements)

    if return_diagnostics:
        return U_star, A_U, solve_diagnostics
    return U_star, A_U
