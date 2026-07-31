"""Assemble ``∂U/∂t + ∇·(UU) = -∇(p/ρ) + ∇·(ν∇U) + f``."""

from dataclasses import replace

import numpy as np

from ..fields import gradients
from ..schemes.boundaries import BOUNDARIES, BoundaryStrategy
from ..solve.linear_interface import normalized_residual, solve_linear_system
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
    A, b_vec, vol, dt, U_old_comp, U_curr_comp, U_old_old_comp=None, scheme="euler"
):
    """Apply the implicit transient ``∂u/∂t`` term (only called when dt is not None).

    Schemes (constant Δt):
      * ``"euler"`` / ``"backward_euler"`` — BDF1, first order:
        ``(V/Δt)(uⁿ⁺¹ − uⁿ)``.
      * ``"backward"`` — BDF2, second order:
        ``(V/Δt)(3/2 uⁿ⁺¹ − 2 uⁿ + 1/2 uⁿ⁻¹)``.  Falls back to BDF1 on the
        first step (when ``U_old_old_comp`` is None), which is the standard
        self-starting BDF2 procedure.

    BDF2 here assumes a constant time step. Configuration validation rejects
    adaptive time stepping with BDF2 before assembly.
    """
    coeff = vol / dt
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
    # Empty/slip planes commonly account for O(n) faces in pseudo-2D cases.
    # Keep this path in ndarray operations: the scalar formulation below is
    # identical to the former per-face loop, including its degenerate-face
    # fallback.
    owner_velocity = U_star[owners_b]
    magnitudes = np.linalg.norm(face_sf, axis=1)
    valid = magnitudes > 1e-10
    projected = owner_velocity.copy()
    if np.any(valid):
        normals = face_sf[valid] / magnitudes[valid, np.newaxis]
        normal_velocity = np.sum(owner_velocity[valid] * normals, axis=1)
        projected[valid] -= normal_velocity[:, np.newaxis] * normals
    U_star[b_elem_indices] = projected


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
    strategy = BOUNDARIES.strategy(bc_type, "U", "ghost")

    if strategy in (
        BoundaryStrategy.ZERO_GRADIENT,
        BoundaryStrategy.INLET_OUTLET,
    ):
        owners_b = mesh_data["owners"][start_face : start_face + n_faces]
        U_star[b_elem_indices] = U_star[owners_b]
    elif strategy is BoundaryStrategy.FREESTREAM:
        # The boundary ghost was switched using the latest face flux before
        # momentum assembly; preserve that per-face inflow/outflow state.
        return
    elif strategy is BoundaryStrategy.CYCLIC:
        paired = mesh_data["boundary_neighbours"][start_face : start_face + n_faces]
        U_star[b_elem_indices] = U_star[paired]
    elif strategy in (BoundaryStrategy.FIXED_VALUE, BoundaryStrategy.NO_SLIP):
        if strategy is BoundaryStrategy.NO_SLIP:
            U_star[b_elem_indices] = [0.0, 0.0, 0.0]
        elif boundary.get("value_U_field") is not None:
            U_star[b_elem_indices] = boundary["value_U_field"]
        elif "value_U" in boundary:
            U_star[b_elem_indices] = np.array(boundary["value_U"])
        else:
            raise ValueError(
                f"Fixed velocity boundary {boundary.get('name')!r} has no configured value"
            )
    elif strategy in (
        BoundaryStrategy.EMPTY,
        BoundaryStrategy.SLIP,
        BoundaryStrategy.SYMMETRY,
    ):
        owners_b = mesh_data["owners"][start_face : start_face + n_faces]
        face_sf = geo_data["face_sf"][start_face : start_face + n_faces]
        _apply_empty_bc_ustar(U_star, b_elem_indices, owners_b, face_sf)


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
    matrix_workspace=None,
    operator_backend="numpy",
):
    """
    Assemble momentum equation for all three velocity components.

    Returns coefficient matrix and RHS for each component separately.

    Args:
        U: Velocity field (n_elements + n_boundary, 3)
        p: Kinematic pressure ``p/ρ`` [m²/s²], shape
            ``(n_elements + n_boundary,)``.
        rho: Positive constant reference density [kg/m³]. It cancels from
            this kinematic-pressure formulation and is validated only.
        nu: Kinematic viscosity (scalar or array)
        mesh_data: Mesh connectivity
        geo_data: Geometric data
        boundaries: Boundary conditions
        convection_scheme: Convection discretization scheme
        dt: Time step size (optional)
        U_old: Previous time step velocity (optional, for transient term)
        source_explicit: Optional acceleration source Su [m/s²], shape
            ``(n_elements, 3)``. Added to the RHS as ``Su * V`` for each
            component (e.g. body acceleration, MMS forcing, or the explicit
            part λ·Utarget of the coupling fringe source).
        source_implicit: Optional implicit volumetric source coefficient Sp
            [1/s], shape ``(n_elements,)``. Added to the diagonal as ``Sp * V``
            (e.g. the implicit part λ of the fringe source S = λ(Utarget − U)).
            Must be >= 0 to preserve diagonal dominance.

    Returns:
        dict: For each component (x, y, z):
            - A: Coefficient matrix
            - b: RHS vector
            - H: H operator (for pressure correction)
    """

    n_elements = mesh_data["n_elements"]

    density = np.asarray(rho, dtype=np.float64)
    if density.ndim != 0:
        raise ValueError("constant-density FVM requires rho to be a scalar")
    density_value = float(density.item())
    if not np.isfinite(density_value) or density_value <= 0.0:
        raise ValueError("rho must be finite and positive")

    viscosity = np.asarray(nu, dtype=np.float64)
    if viscosity.ndim == 0:
        nu = float(viscosity.item())
        if not np.isfinite(nu) or nu <= 0.0:
            raise ValueError("nu must be finite and positive")
    else:
        nu = viscosity
        if nu.shape != (n_elements,) or not np.all(np.isfinite(nu)) or np.any(nu <= 0.0):
            raise ValueError(f"nu must be finite and positive with shape ({n_elements},)")

    # Resolve gradient scheme
    _grad_fn = gradients._resolve_gradient_fn(geo_data)

    # Compute velocity gradient
    grad_U = _grad_fn(U, mesh_data, geo_data)

    # Compute pressure gradient
    grad_p = _grad_fn(p, mesh_data, geo_data)
    if grad_p.ndim == 3:
        grad_p = grad_p.squeeze(-1)  # (n, 3, 1) -> (n, 3)

    # The incompressible equation is divided by the constant reference
    # density. ``phi`` therefore remains volumetric flux and ``nu`` remains
    # kinematic viscosity throughout the operator.
    volumetric_flux = np.asarray(phi, dtype=np.float64)

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

        # 1. Diffusion term: ∇·(ν∇U)
        diff_flux = diffusion.assemble_diffusion_term(
            U_comp,
            grad_U_comp,
            nu,
            mesh_data,
            geo_data,
            momentum_boundaries,
            face_flux=phi,
        )

        # 2. Convection term: ∇·(UU). grad_U_comp feeds the gradient-based
        #    TVD limiter for high-resolution schemes (ignored by the others).
        conv_flux = convection.assemble_convection_term(
            U_comp,
            volumetric_flux,
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
                combined_flux,
                mesh_data,
                workspace=matrix_workspace,
                backend=operator_backend,
            )
        A = spatial_matrix.copy()
        b = matrix_assembly.assemble_rhs_from_fluxes_vectorized(
            combined_flux, mesh_data, backend=operator_backend
        )

        # 5. Add pressure gradient to RHS: -∇p * V
        grad_p_comp = grad_p[:n_elements, i_comp]
        b -= grad_p_comp * vol

        # 6. Transient Term (BDF1/BDF2, selected by ddt_scheme)
        if dt is not None:
            _add_transient_term(
                A,
                b,
                vol,
                dt,
                U_old[:n_elements, i_comp] if U_old is not None else None,
                U[:n_elements, i_comp],
                U_old_old[:n_elements, i_comp] if U_old_old is not None else None,
                scheme=ddt_scheme,
            )

        # 6b. Generic acceleration source terms: S = Su + Sp·U.
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
            "grad_p_comp": grad_p_comp,
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
    """Assemble and solve the three momentum predictors.

    Returns the predicted cell-and-boundary velocity and relaxed cell
    diagonals used by the pressure correction. ``source_explicit`` has shape
    ``(n_cells, 3)`` and ``source_implicit`` has shape ``(n_cells,)``.
    """

    n_elements = mesh_data["n_elements"]
    n_boundary = mesh_data["n_faces"] - mesh_data["n_interior_faces"]

    # Assemble momentum equations
    matrix_workspace = solver_kwargs.pop("matrix_workspace", None)
    operator_backend = solver_kwargs.pop("operator_backend", "numpy")
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
        matrix_workspace=matrix_workspace,
        operator_backend=operator_backend,
    )

    # Solve for each component
    U_star = np.zeros((n_elements + n_boundary, 3))
    A_U = np.zeros((n_elements, 3))
    solve_diagnostics = {}
    linear_backend = solver_kwargs.pop("linear_backend", "scipy")
    parallel_context = solver_kwargs.pop("parallel_context", None)
    partitioned_workspace = solver_kwargs.pop("partitioned_workspace", None)

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
        X, shared_result = solve_linear_system(A_shared, B, method="spsolve", return_info=True)
        if X.ndim == 1:
            X = X[:, np.newaxis]

        for i_comp, comp_name in enumerate(["x", "y", "z"]):
            U_star[:n_elements, i_comp] = X[:, i_comp]
            A_U[:, i_comp] = diag_new
            b_relaxed = B[:, i_comp]
            x_initial = U_old[:n_elements, i_comp] if U_old is not None else np.zeros(n_elements)
            solve_diagnostics[comp_name] = {
                "initial_residual": normalized_residual(A_shared, x_initial, b_relaxed),
                "final_residual": normalized_residual(A_shared, X[:, i_comp], b_relaxed),
            }
            solve_diagnostics[comp_name]["linear_result"] = replace(
                shared_result,
                initial_residual=solve_diagnostics[comp_name]["initial_residual"],
                final_residual=solve_diagnostics[comp_name]["final_residual"],
                setup_seconds=shared_result.setup_seconds if i_comp == 0 else 0.0,
                solve_seconds=shared_result.solve_seconds if i_comp == 0 else 0.0,
            )

        for boundary in boundaries:
            _apply_ustar_bc(U_star, boundary, mesh_data, geo_data, n_elements)
        if return_diagnostics:
            return U_star, A_U, solve_diagnostics
        return U_star, A_U

    requested_ilu_key = solver_kwargs.pop("ilu_key", None)
    shared_ilu_key = requested_ilu_key
    if shared_ilu_key is None and matrix_workspace is not None:
        # The three momentum components share this static matrix topology.  A
        # workspace namespace scopes the cache to one solver instance while
        # allowing x/y/z to reuse one ILU factorisation.
        shared_ilu_key = ("momentum", matrix_workspace.cache_namespace)

    for i_comp, comp_name in enumerate(["x", "y", "z"]):
        A = mom_eqs[comp_name]["A"]
        b = mom_eqs[comp_name]["b"]

        diag_new = A.diagonal() / under_relaxation
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
        x_initial = x0_vec if x0_vec is not None else np.zeros(n_elements)
        initial_residual = normalized_residual(A_relaxed, x_initial, b_relaxed)
        U_comp_star, linear_result = solve_linear_system(
            A_relaxed,
            b_relaxed,
            method=solver,
            equation_type="momentum",
            tol=momentum_tol,
            x0=x0_vec,
            ilu_key=shared_ilu_key,
            backend=linear_backend,
            parallel_context=parallel_context,
            partitioned_workspace=partitioned_workspace,
            return_info=True,
            **solver_kwargs,
        )
        if parallel_context is not None and parallel_context.is_partitioned:
            initial_residual = linear_result.initial_residual
            final_residual = linear_result.final_residual
        else:
            final_residual = normalized_residual(A_relaxed, U_comp_star, b_relaxed)
        solve_diagnostics[comp_name] = {
            "initial_residual": initial_residual,
            "final_residual": final_residual,
            "linear_result": linear_result,
        }

        # Store results
        U_star[:n_elements, i_comp] = U_comp_star
        A_U[:, i_comp] = diag_new

    if parallel_context is not None and parallel_context.is_partitioned:
        parallel_context.exchange_halo(A_U)

    for boundary in boundaries:
        _apply_ustar_bc(U_star, boundary, mesh_data, geo_data, n_elements)

    if return_diagnostics:
        return U_star, A_U, solve_diagnostics
    return U_star, A_U
