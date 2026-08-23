"""Assemble ``∂velocity/∂t + ∇·(UU) = -∇(kinematic_pressure/ρ) + ∇·(ν∇velocity) + ∇·(ν dev2(∇velocityᵀ)) + f``."""

from dataclasses import replace
from typing import Any

import numpy as np

from ..fields import gradients
from ..fields.mixed_velocity_boundary import (
    update_normal_velocity_tangential_gradient_boundary,
)
from ..schemes.boundaries import BOUNDARIES, BoundaryStrategy
from ..solve.linear_interface import normalized_residual, solve_linear_system
from . import convection, diffusion, matrix_assembly


def compute_dev2_stress_source(velocity_gradient, kinematic_viscosity, mesh_data, geo_data):
    r"""Return the explicit ``div(nuEff * dev2(T(grad(velocity))))`` acceleration.

    The viscous term is not a plain Laplacian. It is split as::

        -fvc::div(nuEff*dev2(T(fvc::grad(velocity)))) - fvm::laplacian(nuEff, velocity)

    so the momentum equation carries an explicit transpose-stress term
    alongside ``div(nuEff grad(velocity))``.  It is exactly the part of the deviatoric
    stress that survives a *variable* viscosity: expanding the divergence gives
    ``(grad nuEff) . (grad velocity)^T`` plus a dilatation piece that vanishes for a
    discretely divergence-free flux.  Dropping it is harmless for constant
    ``kinematic_viscosity``, but under LES ``nuEff`` falls from its peak to zero across the
    wall-adjacent cell, and there the term is the same order as the Laplacian
    itself.

    ``dev2(A) = A - (2/3) tr(A) I``, and with the native gradient layout
    ``velocity_gradient[c, i, j] = d(U_j)/d(x_i)``, the face flux of the transposed tensor
    is

    .. math::

        (S_f \cdot \nu \,\mathrm{dev2}(G^T))_j
            = \nu_f \left[ (G_f \cdot S_f)_j
                          - \tfrac{2}{3} S_{f,j} \,\mathrm{tr}(G_f) \right].

    Returns an acceleration ``(n_elements, 3)`` in m/s², i.e. already divided by
    the cell volume, ready to add to ``source_explicit``.  In a fully periodic
    constant-viscosity domain the transpose-stress divergence is identically
    zero in incompressible flow; the conservative face flux is the discrete
    divergence authority, so that special case returns zero.  Meshes with
    physical boundaries retain the term so momentum assembly and reported
    wall traction use the same discrete stress.
    """
    n_cells = mesh_data["n_cells"]
    kinematic_viscosity_array = np.asarray(kinematic_viscosity, dtype=np.float64)
    boundaries = mesh_data.get("boundary", ())
    fully_periodic = bool(boundaries) and all(
        boundary.get("velocity_type", boundary.get("type")) in ("cyclic", "empty")
        for boundary in boundaries
    )
    if kinematic_viscosity_array.ndim == 0 and fully_periodic:
        # Re-discretising grad(div(velocity)) from interpolated cell gradients injects
        # a spurious force even though the corrected periodic face flux is
        # divergence-free to solver tolerance.
        return np.zeros((n_cells, 3), dtype=np.float64)
    n_interior = mesh_data["n_interior_faces"]
    n_faces = mesh_data["n_faces"]
    owners = mesh_data["owners"]
    neighbours = mesh_data["neighbours"]
    face_area_vector = geo_data["face_area_vector"]
    weights = geo_data["face_interpolation_weight"]

    kinematic_viscosity_scalar = (
        float(kinematic_viscosity_array.item()) if kinematic_viscosity_array.ndim == 0 else None
    )
    kinematic_viscosity_cells = (
        None if kinematic_viscosity_scalar is not None else kinematic_viscosity_array
    )

    def _face_flux(grad_face, kinematic_viscosity_face, sf):
        """nu_f * [ G_f . Sf - (2/3) Sf tr(G_f) ] for one block of faces."""
        trace = np.einsum("fii->f", grad_face)
        transposed = np.einsum("fji,fi->fj", grad_face, sf)
        return kinematic_viscosity_face[:, np.newaxis] * (
            transposed - (2.0 / 3.0) * trace[:, np.newaxis] * sf
        )

    source = np.zeros((n_cells, 3), dtype=np.float64)

    # A full interpolated tensor costs 72 bytes per face before its flux and
    # interpolation temporaries.  Bound the working set independently of mesh
    # size; only the cell source persists.
    chunk_size = 200_000
    for start in range(0, n_interior, chunk_size):
        stop = min(start + chunk_size, n_interior)
        owners_i = owners[start:stop]
        neighbours_i = neighbours[start:stop]
        face_interpolation_weight = weights[start:stop]
        w = face_interpolation_weight[:, np.newaxis, np.newaxis]
        grad_face = w * velocity_gradient[neighbours_i] + (1.0 - w) * velocity_gradient[owners_i]
        if kinematic_viscosity_cells is None:
            kinematic_viscosity_face = np.full(
                stop - start, kinematic_viscosity_scalar, dtype=np.float64
            )
        else:
            kinematic_viscosity_face = (
                face_interpolation_weight * kinematic_viscosity_cells[neighbours_i]
                + (1.0 - face_interpolation_weight) * kinematic_viscosity_cells[owners_i]
            )
        flux = _face_flux(grad_face, kinematic_viscosity_face, face_area_vector[start:stop])
        np.add.at(source, owners_i, flux)
        np.add.at(source, neighbours_i, -flux)

    if n_faces > n_interior:
        # The patch value of the gradient already carries the exact wall-normal
        # derivative (see gradients._correct_boundary_gradient), which is what
        # fvc::interpolate reads on a boundary patch.
        for start in range(n_interior, n_faces, chunk_size):
            stop = min(start + chunk_size, n_faces)
            faces_b = np.arange(start, stop)
            owners_b = owners[faces_b]
            ghosts_b = n_cells + (faces_b - n_interior)
            kinematic_viscosity_face = (
                np.full(stop - start, kinematic_viscosity_scalar, dtype=np.float64)
                if kinematic_viscosity_cells is None
                else kinematic_viscosity_cells[owners_b]
            )
            flux_b = _face_flux(
                velocity_gradient[ghosts_b], kinematic_viscosity_face, face_area_vector[faces_b]
            )
            np.add.at(source, owners_b, flux_b)

    source /= geo_data["cell_volume"][:, np.newaxis]
    return source


def _make_momentum_boundary(b: dict, i_comp: int) -> dict:
    """Create per-component momentum boundary dict from velocity boundary.

    Args:
        b: Velocity boundary dictionary containing keys like ``bc_type_velocity``
            and ``value_velocity``.
        i_comp: Component index (0, 1, 2 for x, y, z).

    Returns:
        dict: Momentum boundary dictionary with ``boundary_condition_type`` and ``value``
            keys extracted for the specified component.
    """
    b_mom = b.copy()
    if "velocity_type" in b:
        b_mom["boundary_condition_type"] = b["velocity_type"]
    if "velocity_value" in b:
        val = b["velocity_value"]
        b_mom["value"] = val[i_comp] if (np.ndim(val) == 1 and len(val) == 3) else val
    return b_mom


def _add_transient_term(
    A,
    b_vec,
    vol,
    time_step_size,
    velocity_old_component,
    velocity_current_component,
    velocity_older_component=None,
    scheme="euler",
):
    """Apply the implicit transient ``∂velocity/∂t`` term (only called when
    ``time_step_size`` is not None).

    Schemes (constant Δt):
      * ``"euler"`` / ``"backward_euler"`` — BDF1, first order:
        ``(V/Δt)(uⁿ⁺¹ − uⁿ)``.
      * ``"backward"`` — BDF2, second order:
        ``(V/Δt)(3/2 uⁿ⁺¹ − 2 uⁿ + 1/2 uⁿ⁻¹)``.  Falls back to BDF1 on the
        first step (when ``velocity_older_component`` is None), which is the standard
        self-starting BDF2 procedure.

    BDF2 here assumes a constant time step. Configuration validation rejects
    adaptive time stepping with BDF2 before assembly.
    """
    coeff = vol / time_step_size
    if (
        scheme == "backward"
        and velocity_old_component is not None
        and velocity_older_component is not None
    ):
        A.setdiag(A.diagonal() + 1.5 * coeff)
        b_vec += coeff * (2.0 * velocity_old_component - 0.5 * velocity_older_component)
    else:
        # BDF1 (Euler implicit), also the BDF2 startup step.
        A.setdiag(A.diagonal() + coeff)
        b_vec += coeff * (
            velocity_old_component
            if velocity_old_component is not None
            else velocity_current_component
        )


def _apply_empty_bc_ustar(velocity_star, b_elem_indices, owners_b, face_area_vector):
    """Apply empty boundary condition: zero out normal velocity component.

    For ``empty`` boundaries (used in 2D extruded meshes for the
    non-solved dimension), the face-normal component of the predicted
    velocity is removed, leaving only the tangential components.

    Args:
        velocity_star: Predicted velocity field (n_elements + n_boundary, 3),
            modified in-place.
        b_elem_indices: Indices of boundary elements to update.
        owners_b: Owner element indices for each boundary face.
        face_area_vector: Face surface area vectors for each boundary face.
    """
    # Empty/slip planes commonly account for O(n) faces in pseudo-2D cases.
    # Keep this path in ndarray operations: the scalar formulation below is
    # identical to the former per-face loop, including its degenerate-face
    # fallback.
    owner_velocity = velocity_star[owners_b]
    magnitudes = np.linalg.norm(face_area_vector, axis=1)
    valid = magnitudes > 1e-10
    projected = owner_velocity.copy()
    if np.any(valid):
        normals = face_area_vector[valid] / magnitudes[valid, np.newaxis]
        normal_velocity = np.sum(owner_velocity[valid] * normals, axis=1)
        projected[valid] -= normal_velocity[:, np.newaxis] * normals
    velocity_star[b_elem_indices] = projected


def _apply_ustar_bc(velocity_star, boundary, mesh_data, geo_data, n_cells):
    """Apply post-solve boundary condition to the predicted velocity field.

    Modifies ``velocity_star`` in-place at the boundary elements according to
    the boundary condition type specified in the boundary dictionary.

    Args:
        velocity_star: Predicted velocity field (n_elements + n_boundary, 3),
            modified in-place.
        boundary: Boundary dictionary with keys ``startFace``, ``nFaces``,
            ``bc_type_velocity``, and optionally ``value_velocity`` / ``value_velocity_field``.
        mesh_data: Mesh connectivity containing ``owners`` and
            ``n_interior_faces``.
        geo_data: Geometric data containing ``face_area_vector``.
        n_elements: Number of interior elements.
    """
    start_face = boundary["start_face"]
    n_faces = boundary["n_faces"]
    b_elem_start = start_face - mesh_data["n_interior_faces"]
    b_elem_indices = np.arange(n_cells + b_elem_start, n_cells + b_elem_start + n_faces)
    boundary_condition_type = boundary.get("velocity_type")
    strategy = BOUNDARIES.strategy(boundary_condition_type, "velocity", "ghost")

    if strategy in (
        BoundaryStrategy.ZERO_GRADIENT,
        BoundaryStrategy.INLET_OUTLET,
    ):
        owners_b = mesh_data["owners"][start_face : start_face + n_faces]
        velocity_star[b_elem_indices] = velocity_star[owners_b]
    elif strategy is BoundaryStrategy.FREESTREAM:
        # The boundary ghost was switched using the latest face flux before
        # momentum assembly; preserve that per-face inflow/outflow state.
        return
    elif strategy is BoundaryStrategy.CYCLIC:
        paired = mesh_data["boundary_neighbour_cell"][start_face : start_face + n_faces]
        velocity_star[b_elem_indices] = velocity_star[paired]
    elif strategy in (BoundaryStrategy.FIXED_VALUE, BoundaryStrategy.NO_SLIP):
        if strategy is BoundaryStrategy.NO_SLIP:
            velocity_star[b_elem_indices] = [0.0, 0.0, 0.0]
        elif boundary.get("velocity_value_field") is not None:
            velocity_star[b_elem_indices] = boundary["velocity_value_field"]
        elif "velocity_value" in boundary:
            velocity_star[b_elem_indices] = np.array(boundary["velocity_value"])
        else:
            raise ValueError(
                f"Fixed velocity boundary {boundary.get('name')!r} has no configured value"
            )
    elif strategy is BoundaryStrategy.NORMAL_VALUE_TANGENTIAL_GRADIENT:
        update_normal_velocity_tangential_gradient_boundary(
            velocity_star, boundary, mesh_data, geo_data
        )
    elif strategy in (
        BoundaryStrategy.EMPTY,
        BoundaryStrategy.SLIP,
        BoundaryStrategy.SYMMETRY,
    ):
        owners_b = mesh_data["owners"][start_face : start_face + n_faces]
        face_area_vector = geo_data["face_area_vector"][start_face : start_face + n_faces]
        _apply_empty_bc_ustar(velocity_star, b_elem_indices, owners_b, face_area_vector)


def assemble_momentum_equation(
    velocity,
    kinematic_pressure,
    volumetric_face_flux,
    density,
    kinematic_viscosity,
    mesh_data,
    geo_data,
    boundaries,
    convection_scheme="deferred",
    time_step_size=None,
    velocity_old=None,
    velocity_older=None,
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
        velocity: Velocity field (n_elements + n_boundary, 3)
        kinematic_pressure: Kinematic pressure ``kinematic_pressure/ρ`` [m²/s²], shape
            ``(n_elements + n_boundary,)``.
        density: Positive constant reference density [kg/m³]. It cancels from
            this kinematic-pressure formulation and is validated only.
        kinematic_viscosity: Kinematic viscosity (scalar or array)
        mesh_data: Mesh connectivity
        geo_data: Geometric data
        boundaries: Boundary conditions
        convection_scheme: Convection discretization scheme
        time_step_size: Time step size (optional)
        velocity_old: Previous time step velocity (optional, for transient term)
        source_explicit: Optional acceleration source Su [m/s²], shape
            ``(n_elements, 3)``. Added to the RHS as ``Su * V`` for each
            component (e.g. body acceleration or MMS forcing).
        source_implicit: Optional implicit volumetric source coefficient Sp
            [1/s], shape ``(n_elements,)``. Added to the diagonal as ``Sp * V``
            Must be >= 0 to preserve diagonal dominance.

    Returns:
        dict: For each component (x, y, z):
            - A: Coefficient matrix
            - b: RHS vector
            - H: H operator (for pressure correction)
    """

    n_cells = mesh_data["n_cells"]

    density = np.asarray(density, dtype=np.float64)
    if density.ndim != 0:
        raise ValueError("constant-density FVM requires density to be a scalar")
    density_value = float(density.item())
    if not np.isfinite(density_value) or density_value <= 0.0:
        raise ValueError("density must be finite and positive")

    kinematic_viscosity_array = np.asarray(kinematic_viscosity, dtype=np.float64)
    if kinematic_viscosity_array.ndim == 0:
        kinematic_viscosity = float(kinematic_viscosity_array.item())
        if not np.isfinite(kinematic_viscosity) or kinematic_viscosity <= 0.0:
            raise ValueError("kinematic_viscosity must be finite and positive")
    else:
        kinematic_viscosity = kinematic_viscosity_array
        if (
            kinematic_viscosity.shape != (n_cells,)
            or not np.all(np.isfinite(kinematic_viscosity))
            or np.any(kinematic_viscosity <= 0.0)
        ):
            raise ValueError(
                f"kinematic_viscosity must be finite and positive with shape ({n_cells},)"
            )

    # Resolve gradient scheme
    _grad_fn = gradients._resolve_gradient_fn(geo_data)

    # Compute velocity gradient
    velocity_gradient = _grad_fn(velocity, mesh_data, geo_data)

    # Compute pressure gradient
    kinematic_pressure_gradient = _grad_fn(kinematic_pressure, mesh_data, geo_data)
    if kinematic_pressure_gradient.ndim == 3:
        kinematic_pressure_gradient = kinematic_pressure_gradient.squeeze(-1)  # (n, 3, 1) -> (n, 3)

    # The incompressible equation is divided by the constant reference
    # density. ``volumetric_face_flux`` therefore remains volumetric flux and
    # ``kinematic_viscosity`` remains
    # kinematic viscosity throughout the operator.
    volumetric_flux = np.asarray(volumetric_face_flux, dtype=np.float64)

    results = {}
    common_matrix = None
    common_diagonal = None
    vol = geo_data["cell_volume"]
    has_directional_mixed_bc = any(
        BOUNDARIES.strategy(boundary.get("velocity_type"), "velocity", "diffusion")
        is BoundaryStrategy.NORMAL_VALUE_TANGENTIAL_GRADIENT
        for boundary in boundaries
    )

    # Split the viscous stress into an implicit
    # laplacian(nuEff, velocity) and this explicit transpose-stress correction.  It is
    # assembled once for all three components because it couples them.
    dev2_source = compute_dev2_stress_source(
        velocity_gradient, kinematic_viscosity, mesh_data, geo_data
    )

    # Assemble for each component
    for i_comp, comp_name in enumerate(["x", "y", "z"]):
        # Get velocity component
        velocity_component = velocity[:, i_comp]

        # Get gradient component
        velocity_gradient_component = (
            velocity_gradient[:, :, i_comp] if velocity_gradient.ndim == 3 else velocity_gradient
        )  # (n_elements, 3) or full

        momentum_boundaries = [_make_momentum_boundary(b, i_comp) for b in boundaries]

        # 1. Diffusion term: ∇·(ν∇velocity)
        diff_flux = diffusion.assemble_diffusion_term(
            velocity_component,
            velocity_gradient_component,
            kinematic_viscosity,
            mesh_data,
            geo_data,
            momentum_boundaries,
            volumetric_face_flux=volumetric_face_flux,
            vector_field=velocity,
            component=i_comp,
            include_total_flux=False,
        )
        # The total face flux is a diagnostic output of the generic assembly
        # API.  Momentum matrix/RHS assembly only consumes cf/ff/vf, so do not
        # retain it while allocating the convection arrays.
        diff_flux.pop("flux_tf", None)

        # 2. Convection term: ∇·(UU). velocity_gradient_component feeds the gradient-based
        #    TVD limiter for high-resolution schemes (ignored by the others).
        conv_flux = convection.assemble_convection_term(
            velocity_component,
            volumetric_flux,
            mesh_data,
            geo_data,
            boundaries,
            scheme=convection_scheme,
            scalar_field_gradient=velocity_gradient_component,
            include_total_flux=False,
        )
        conv_flux.pop("flux_tf", None)

        # 3. Combine in the diffusion buffers.  Allocating four additional
        # face arrays here added about 45 MiB per reference-mesh rank.
        for key in ("flux_cf", "flux_ff", "flux_vf"):
            diff_flux[key] += conv_flux[key]
        combined_flux = diff_flux
        del conv_flux

        # 4. Assemble one shared matrix in the standard path.  The directional
        # mixed BC contributes D*n_i**2 at the boundary and therefore needs a
        # distinct scalar matrix for each Cartesian component.
        if common_matrix is None or has_directional_mixed_bc:
            assembled_matrix = matrix_assembly.assemble_matrix_from_fluxes_vectorized(
                combined_flux,
                mesh_data,
                workspace=matrix_workspace,
                backend=operator_backend,
            )
            # A reusable workspace owns one mutable CSR object.  Preserve each
            # component before the next workspace update overwrites its data.
            if has_directional_mixed_bc and matrix_workspace is not None:
                assembled_matrix = assembled_matrix.copy()
            if time_step_size is not None:
                use_bdf2 = (
                    ddt_scheme == "backward"
                    and velocity_old is not None
                    and velocity_older is not None
                )
                transient_diagonal = (1.5 if use_bdf2 else 1.0) * vol / time_step_size
                assembled_matrix.setdiag(assembled_matrix.diagonal() + transient_diagonal)
            if source_implicit is not None:
                assembled_matrix.setdiag(
                    assembled_matrix.diagonal() + source_implicit[:n_cells] * vol
                )
            if has_directional_mixed_bc:
                A = assembled_matrix
                equation_diagonal = A.diagonal()
            else:
                common_matrix = assembled_matrix
                common_diagonal = common_matrix.diagonal()
                A = common_matrix
                equation_diagonal = common_diagonal
        else:
            assert common_matrix is not None
            assert common_diagonal is not None
            A = common_matrix
            equation_diagonal = common_diagonal
        b = matrix_assembly.assemble_rhs_from_fluxes_vectorized(
            combined_flux, mesh_data, backend=operator_backend
        )

        # 5. Add pressure gradient to RHS: -∇kinematic_pressure * V
        grad_p_comp = kinematic_pressure_gradient[:n_cells, i_comp]
        b -= grad_p_comp * vol

        # 5b. Explicit transpose part of the deviatoric viscous stress.
        b += dev2_source[:, i_comp] * vol

        # 6. Transient RHS (the component-independent diagonal was added once
        #    to ``common_matrix`` above).
        if time_step_size is not None:
            coefficient = vol / time_step_size
            old_component = (
                velocity_old[:n_cells, i_comp]
                if velocity_old is not None
                else velocity[:n_cells, i_comp]
            )
            if ddt_scheme == "backward" and velocity_older is not None:
                b += coefficient * (2.0 * old_component - 0.5 * velocity_older[:n_cells, i_comp])
            else:
                b += coefficient * old_component

        # 6b. Generic acceleration source terms: S = Su + Sp·velocity.
        #     Su → RHS (+Su·V); Sp → diagonal (+Sp·V), keeping velocity implicit.
        #     Used by MMS forcing and other solver-owned source models.
        if source_explicit is not None:
            b += source_explicit[:n_cells, i_comp] * vol
        results[comp_name] = {
            "A": A,
            "b": b,
            "H": equation_diagonal,
        }
        # Without an explicit drop, the previous component's combined face
        # arrays survive while Python evaluates the next diffusion/convection
        # call on the right-hand side of its assignment.
        del combined_flux, diff_flux

    return results


def solve_momentum_predictor(
    velocity,
    kinematic_pressure,
    volumetric_face_flux,
    density,
    kinematic_viscosity,
    mesh_data,
    geo_data,
    boundaries,
    convection_scheme="deferred",
    solver="spsolve",
    under_relaxation=0.7,
    time_step_size=None,
    velocity_old=None,
    velocity_older=None,
    ddt_scheme="euler",
    source_explicit=None,
    source_implicit=None,
    return_diagnostics=False,
    **solver_kwargs,
) -> Any:
    """Assemble and solve the three momentum predictors.

    Returns the predicted cell-and-boundary velocity and relaxed cell
    diagonals used by the pressure correction. ``source_explicit`` has shape
    ``(n_cells, 3)`` and ``source_implicit`` has shape ``(n_cells,)``.
    """

    n_cells = mesh_data["n_cells"]
    n_boundary = mesh_data["n_faces"] - mesh_data["n_interior_faces"]

    # Assemble momentum equations
    matrix_workspace = solver_kwargs.pop("matrix_workspace", None)
    operator_backend = solver_kwargs.pop("operator_backend", "numpy")
    mom_eqs = assemble_momentum_equation(
        velocity,
        kinematic_pressure,
        volumetric_face_flux,
        density,
        kinematic_viscosity,
        mesh_data,
        geo_data,
        boundaries,
        convection_scheme,
        time_step_size=time_step_size,
        velocity_old=velocity_old,
        velocity_older=velocity_older,
        ddt_scheme=ddt_scheme,
        source_explicit=source_explicit,
        source_implicit=source_implicit,
        matrix_workspace=matrix_workspace,
        operator_backend=operator_backend,
    )

    # Solve for each component
    velocity_star = np.zeros((n_cells + n_boundary, 3))
    matrices_are_shared = all(mom_eqs[name]["A"] is mom_eqs["x"]["A"] for name in ("y", "z"))
    # Standard boundaries retain the compact shared diagonal.  The
    # directional mixed condition returns component diagonals for the
    # pressure/Rhie-Chow vector path that already supports them.
    momentum_diagonal = np.empty(n_cells if matrices_are_shared else (n_cells, 3), dtype=np.float64)
    solve_diagnostics = {}
    linear_backend = solver_kwargs.pop("linear_backend", "scipy")
    parallel_context = solver_kwargs.pop("parallel_context", None)
    partitioned_workspace = solver_kwargs.pop("partitioned_workspace", None)

    if solver == "spsolve" and linear_backend == "scipy" and matrices_are_shared:
        # All three components share one matrix.  Solve a three-column RHS so
        # SuperLU performs one factorization instead of three.
        A_shared = mom_eqs["x"]["A"]
        diag_old = A_shared.diagonal()
        diag_new = diag_old / under_relaxation
        A_shared.setdiag(diag_new)

        rhs_columns = []
        for i_comp, comp_name in enumerate(["x", "y", "z"]):
            b = mom_eqs[comp_name]["b"]
            source_relax = (1.0 - under_relaxation) * diag_new * velocity[:n_cells, i_comp]
            rhs_columns.append(b + source_relax)
        B = np.column_stack(rhs_columns)
        X, shared_result = solve_linear_system(A_shared, B, method="spsolve", return_info=True)
        if X.ndim == 1:
            X = X[:, np.newaxis]

        momentum_diagonal[:] = diag_new
        for i_comp, comp_name in enumerate(["x", "y", "z"]):
            velocity_star[:n_cells, i_comp] = X[:, i_comp]
            b_relaxed = B[:, i_comp]
            x_initial = (
                velocity_old[:n_cells, i_comp] if velocity_old is not None else np.zeros(n_cells)
            )
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
            _apply_ustar_bc(velocity_star, boundary, mesh_data, geo_data, n_cells)
        if return_diagnostics:
            return velocity_star, momentum_diagonal, solve_diagnostics
        return velocity_star, momentum_diagonal

    requested_ilu_key = solver_kwargs.pop("ilu_key", None)
    shared_ilu_key = requested_ilu_key
    if shared_ilu_key is None and matrix_workspace is not None:
        # The three momentum components share this static matrix topology.  A
        # workspace namespace scopes the cache to one solver instance while
        # allowing x/y/z to reuse one ILU factorisation.
        shared_ilu_key = ("momentum", matrix_workspace.cache_namespace)

    shared_relaxed_diagonal = None
    for i_comp, comp_name in enumerate(["x", "y", "z"]):
        A_relaxed = mom_eqs[comp_name]["A"]
        if matrices_are_shared:
            if shared_relaxed_diagonal is None:
                shared_relaxed_diagonal = A_relaxed.diagonal() / under_relaxation
                A_relaxed.setdiag(shared_relaxed_diagonal)
            diag_new = shared_relaxed_diagonal
            momentum_diagonal[:] = diag_new
            component_ilu_key = shared_ilu_key
        else:
            diag_new = A_relaxed.diagonal() / under_relaxation
            A_relaxed.setdiag(diag_new)
            momentum_diagonal[:, i_comp] = diag_new
            component_ilu_key = (shared_ilu_key, comp_name) if shared_ilu_key is not None else None
        b = mom_eqs[comp_name]["b"]

        # Add source term to RHS to ensure consistency at convergence
        # velocity[:n_elements, i_comp] is the old velocity value
        source_relax = (1.0 - under_relaxation) * diag_new * velocity[:n_cells, i_comp]
        b_relaxed = b + source_relax

        # Determine initial guess x0 for iterative solver: prefer velocity_old
        # (previous time step) if available
        momentum_tolerance = solver_kwargs.get("momentum_tolerance", 1e-4)
        momentum_relative_tolerance = solver_kwargs.get("momentum_relative_tolerance", 0.0)
        x0_vec = None
        if velocity_old is not None:
            if np.asarray(velocity_old).shape[0] < n_cells:
                raise ValueError("velocity_old does not contain every interior cell")
            x0_vec = velocity_old[:n_cells, i_comp]

        # Solve with optional initial guess and tuned tolerance.
        velocity_component_star, linear_result = solve_linear_system(
            A_relaxed,
            b_relaxed,
            method=solver,
            equation_type="momentum",
            tol=momentum_tolerance,
            rel_tol=momentum_relative_tolerance,
            x0=x0_vec,
            ilu_key=component_ilu_key,
            backend=linear_backend,
            parallel_context=parallel_context,
            partitioned_workspace=partitioned_workspace,
            return_info=True,
            **solver_kwargs,
        )
        solve_diagnostics[comp_name] = {
            "initial_residual": linear_result.initial_residual,
            "final_residual": linear_result.final_residual,
            "linear_result": linear_result,
        }

        # Store results
        velocity_star[:n_cells, i_comp] = velocity_component_star

    if parallel_context is not None and parallel_context.is_partitioned:
        parallel_context.exchange_halo(momentum_diagonal)

    for boundary in boundaries:
        _apply_ustar_bc(velocity_star, boundary, mesh_data, geo_data, n_cells)

    if return_diagnostics:
        return velocity_star, momentum_diagonal, solve_diagnostics
    return velocity_star, momentum_diagonal
