"""Assemble ``∂velocity/∂t + ∇·(UU) = -∇(kinematic_pressure/ρ) + ∇·(ν∇velocity) + ∇·(ν dev2(∇velocityᵀ)) + f``."""

from typing import TYPE_CHECKING, Any

import numpy as np

from openonda.core._numba import cacheable_njit as njit

from ..fields import gradients
from ..fields.boundary_values import update_velocity_boundaries
from ..schemes.boundaries import BoundaryStrategy, boundary_strategy
from ..solve.linear_interface import (
    solve_linear_system,
)
from . import convection, diffusion, matrix_assembly
from .time_integration import backward_coefficients

if TYPE_CHECKING:
    from openonda.core.log_style import StepLog

    from ..core.parallel import ParallelContext
    from ..solve.petsc_partitioned import PartitionedLinearWorkspace
    from .matrix_assembly import MatrixAssemblyWorkspace


@njit(cache=True)
def _dev2_interior_source(
    velocity_gradient: np.ndarray,
    viscosity: np.ndarray,
    owners: np.ndarray,
    neighbours: np.ndarray,
    weights: np.ndarray,
    areas: np.ndarray,
    n_interior: int,
    source: np.ndarray,
) -> None:
    """Add the interior-face fluxes of :func:`compute_dev2_stress_source` to ``source``.

    ``viscosity`` holds one value (constant viscosity) or one per cell.
    """
    gradient_face = np.empty((3, 3))
    for face in range(n_interior):
        owner, neighbour = owners[face], neighbours[face]
        weight = weights[face]
        for row in range(3):
            for column in range(3):
                gradient_face[row, column] = (
                    weight * velocity_gradient[neighbour, row, column]
                    + (1.0 - weight) * velocity_gradient[owner, row, column]
                )
        if len(viscosity) == 1:
            viscosity_face = viscosity[0]
        else:
            viscosity_face = weight * viscosity[neighbour] + (1.0 - weight) * viscosity[owner]
        trace = gradient_face[0, 0] + gradient_face[1, 1] + gradient_face[2, 2]
        for row in range(3):
            transposed = 0.0
            for column in range(3):
                transposed += gradient_face[row, column] * areas[face, column]
            flux = viscosity_face * (transposed - (2.0 / 3.0) * trace * areas[face, row])
            source[owner, row] += flux
            source[neighbour, row] -= flux


def compute_dev2_stress_source(
    velocity_gradient: np.ndarray,
    kinematic_viscosity: float | np.ndarray,
    mesh_data: dict[str, object],
    geo_data: dict[str, object],
) -> np.ndarray:
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

    Returns an acceleration ``(n_cells, 3)`` in m/s², i.e. already divided by
    the cell volume, added to the momentum right-hand side.  The term is
    assembled for every mesh and viscosity, so momentum assembly and reported
    wall traction use the same discrete stress.  Cyclic faces are interior
    faces between the owner and its paired cell.
    """
    n_cells = mesh_data["n_cells"]
    kinematic_viscosity_array = np.asarray(kinematic_viscosity, dtype=np.float64)
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

    def _face_flux(
        grad_face: np.ndarray, kinematic_viscosity_face: np.ndarray, sf: np.ndarray
    ) -> np.ndarray:
        """nu_f * [ G_f . Sf - (2/3) Sf tr(G_f) ] for one block of faces."""
        trace = np.einsum("fii->f", grad_face)
        transposed = np.einsum("fji,fi->fj", grad_face, sf)
        return kinematic_viscosity_face[:, np.newaxis] * (
            transposed - (2.0 / 3.0) * trace[:, np.newaxis] * sf
        )

    source = np.zeros((n_cells, 3), dtype=np.float64)

    _dev2_interior_source(
        velocity_gradient,
        np.atleast_1d(kinematic_viscosity_array),
        owners,
        neighbours,
        weights,
        face_area_vector,
        n_interior,
        source,
    )

    # The boundary faces are few; they use bounded NumPy blocks.
    chunk_size = 200_000
    if n_faces > n_interior:
        # The patch value of the gradient already carries the exact wall-normal
        # derivative (see gradients._correct_boundary_gradient), which is what
        # fvc::interpolate reads on a boundary patch.  A cyclic face
        # interpolates between its owner and the paired cell, as an interior
        # face does.
        boundary_neighbour_cell = np.asarray(
            mesh_data.get("boundary_neighbour_cell", np.full(n_faces, -1, dtype=np.int32))
        )
        for start in range(n_interior, n_faces, chunk_size):
            stop = min(start + chunk_size, n_faces)
            faces_b = np.arange(start, stop)
            owners_b = owners[faces_b]
            grad_face = velocity_gradient[n_cells + (faces_b - n_interior)]
            kinematic_viscosity_face = (
                np.full(stop - start, kinematic_viscosity_scalar, dtype=np.float64)
                if kinematic_viscosity_cells is None
                else kinematic_viscosity_cells[owners_b]
            )
            coupled = np.flatnonzero(boundary_neighbour_cell[faces_b] >= 0)
            if coupled.size:
                owners_c = owners_b[coupled]
                paired_c = boundary_neighbour_cell[faces_b[coupled]]
                weight_c = weights[faces_b[coupled]]
                w = weight_c[:, np.newaxis, np.newaxis]
                grad_face[coupled] = (
                    w * velocity_gradient[paired_c] + (1.0 - w) * velocity_gradient[owners_c]
                )
                if kinematic_viscosity_cells is not None:
                    kinematic_viscosity_face[coupled] = (
                        weight_c * kinematic_viscosity_cells[paired_c]
                        + (1.0 - weight_c) * kinematic_viscosity_cells[owners_c]
                    )
            flux_b = _face_flux(grad_face, kinematic_viscosity_face, face_area_vector[faces_b])
            for axis in range(3):
                source[:, axis] += np.bincount(owners_b, weights=flux_b[:, axis], minlength=n_cells)

    source /= geo_data["cell_volume"][:, np.newaxis]
    return source


def assemble_momentum_equation(
    velocity: np.ndarray,
    kinematic_pressure: np.ndarray,
    volumetric_face_flux: np.ndarray,
    density: float,
    kinematic_viscosity: float | np.ndarray,
    mesh_data: dict[str, object],
    geo_data: dict[str, object],
    boundaries: list[dict],
    convection_scheme: str,
    time_step_size: float | None = None,
    velocity_old: np.ndarray | None = None,
    velocity_older: np.ndarray | None = None,
    previous_time_step_size: float | None = None,
    ddt_scheme: str = "euler_implicit",
    matrix_workspace: "MatrixAssemblyWorkspace | None" = None,
    velocity_gradient: np.ndarray | None = None,
    kinematic_pressure_gradient: np.ndarray | None = None,
) -> dict[str, dict[str, object]]:
    """
    Assemble momentum equation for all three velocity components.

    Returns coefficient matrix and RHS for each component separately.

    Args:
        velocity: Velocity field (n_cells + n_boundary, 3)
        kinematic_pressure: Kinematic pressure ``kinematic_pressure/ρ`` [m²/s²], shape
            ``(n_cells + n_boundary,)``.
        density: Positive constant reference density [kg/m³]. It cancels from
            this kinematic-pressure formulation and is validated only.
        kinematic_viscosity: Kinematic viscosity (scalar or array)
        mesh_data: Mesh connectivity
        geo_data: Geometric data
        boundaries: Boundary conditions
        convection_scheme: Convection discretization scheme
        time_step_size: Time step size (optional)
        velocity_old: Previous time step velocity (optional, for transient term)
        previous_time_step_size: Preceding accepted step size for variable-step BDF2.
        velocity_gradient: Gradient of ``velocity`` that the caller already
            computed from the same array, ``(n_cells + n_boundary, 3, 3)``;
            ``None`` computes it here.
        kinematic_pressure_gradient: Gradient of ``kinematic_pressure`` that the
            caller already computed from the same array, ``(n_cells + n_boundary, 3)``;
            ``None`` computes it here. Neither gradient is modified.

    Returns:
        dict: For each component (x, y, z):
            - A: Coefficient matrix
            - b: RHS vector
            - H: diagonal of ``A``, used by the pressure correction
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

    if velocity_gradient is None:
        velocity_gradient = gradients.gradient(velocity, mesh_data, geo_data)
    if kinematic_pressure_gradient is None:
        kinematic_pressure_gradient = gradients.gradient(kinematic_pressure, mesh_data, geo_data)

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
        boundary_strategy(boundary.get("velocity_type"), "velocity")
        is BoundaryStrategy.MIXED_VELOCITY
        for boundary in boundaries
    )

    # Split the viscous stress into an implicit
    # laplacian(nuEff, velocity) and this explicit transpose-stress correction.  It is
    # assembled once for all three components because it couples them.
    dev2_source = compute_dev2_stress_source(
        velocity_gradient, kinematic_viscosity, mesh_data, geo_data
    )
    # BDF2 needs both earlier time levels and the earlier step size; the first
    # step and its matrix diagonal use BDF1.
    use_bdf2 = (
        time_step_size is not None
        and ddt_scheme == "backward"
        and velocity_old is not None
        and velocity_older is not None
        and previous_time_step_size is not None
    )

    # The implicit coefficients, the face geometry and the upwind selection do
    # not depend on the component, so the interior faces are assembled once for
    # all three. Only the per-component explicit parts differ, and the shared
    # matrix is built from the first component alone.
    interior_diffusion = diffusion.assemble_diffusion_term_interior_vector(
        velocity_gradient, kinematic_viscosity, mesh_data, geo_data
    )
    interior_convection = convection.assemble_convection_term_interior_vector(
        velocity, volumetric_flux, mesh_data, geo_data, velocity_gradient, convection_scheme
    )

    def _interior_slice(fluxes: dict[str, np.ndarray] | None, i_comp: int):
        if fluxes is None:
            return None
        return {
            "flux_cf": fluxes["flux_cf"],
            "flux_ff": fluxes["flux_ff"],
            "flux_vf": fluxes["flux_vf"][:, i_comp],
        }

    # Assemble for each component
    for i_comp, comp_name in enumerate(["x", "y", "z"]):
        # Only the first component, or every component with a directional
        # mixed boundary, builds a matrix and so needs cf and ff.
        needs_coefficients = common_matrix is None or has_directional_mixed_bc
        velocity_component = velocity[:, i_comp]

        velocity_gradient_component = (
            velocity_gradient[:, :, i_comp] if velocity_gradient.ndim == 3 else velocity_gradient
        )  # (n_cells, 3) or full

        # 1. Diffusion term: ∇·(ν∇velocity)
        diff_flux = diffusion.assemble_diffusion_term(
            velocity_component,
            velocity_gradient_component,
            kinematic_viscosity,
            mesh_data,
            geo_data,
            boundaries,
            volumetric_face_flux=volumetric_face_flux,
            vector_field=velocity,
            component=i_comp,
            include_total_flux=False,
            separate_slip_diagonal=True,
            interior_fluxes=_interior_slice(interior_diffusion, i_comp),
            coefficients=needs_coefficients,
        )
        # The total face flux is a diagnostic output of the generic assembly
        # API.  Momentum matrix/RHS assembly only consumes cf/ff/vf, so do not
        # retain it while allocating the convection arrays.
        diff_flux.pop("flux_tf", None)
        slip_diagonal = diff_flux.pop("slip_diagonal")

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
            component=i_comp,
            interior_fluxes=_interior_slice(interior_convection, i_comp),
            coefficients=needs_coefficients,
        )
        conv_flux.pop("flux_tf", None)

        # 3. Combine in the diffusion buffers.  Allocating four additional
        # face arrays here added about 45 MiB per reference-mesh rank.
        for key in conv_flux:
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
            )
            # A reusable workspace owns one mutable CSR object.  Preserve each
            # component before the next workspace update overwrites its data.
            if has_directional_mixed_bc and matrix_workspace is not None:
                assembled_matrix = assembled_matrix.copy()
            if time_step_size is not None:
                if use_bdf2:
                    assert previous_time_step_size is not None
                    transient_factor = backward_coefficients(
                        time_step_size, previous_time_step_size
                    )[0]
                else:
                    transient_factor = 1.0
                transient_diagonal = transient_factor * vol / time_step_size
                assembled_matrix.setdiag(assembled_matrix.diagonal() + transient_diagonal)
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
        if np.any(slip_diagonal):
            # The wall-normal slip coefficient belongs to this
            # component only; the others keep sharing the common matrix.
            if A is common_matrix:
                A = common_matrix.copy()
            A.setdiag(A.diagonal() + slip_diagonal)
            equation_diagonal = A.diagonal()
        b = matrix_assembly.assemble_rhs_from_fluxes_vectorized(combined_flux, mesh_data)

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
            if use_bdf2:
                _, coefficient_old, coefficient_older = backward_coefficients(
                    time_step_size, previous_time_step_size
                )
                b += coefficient * (
                    coefficient_old * old_component
                    - coefficient_older * velocity_older[:n_cells, i_comp]
                )
            else:
                b += coefficient * old_component

        results[comp_name] = {
            "A": A,
            "b": b,
            "H": equation_diagonal,
        }
        # Without an explicit drop, the previous component's combined face
        # arrays survive while Python evaluates the next diffusion/convection
        # call on the right-hand side of its assignment.
        del combined_flux, diff_flux

    del interior_diffusion, interior_convection
    return results


def solve_momentum_predictor(
    velocity: np.ndarray,
    kinematic_pressure: np.ndarray,
    volumetric_face_flux: np.ndarray,
    density: float,
    kinematic_viscosity: float | np.ndarray,
    mesh_data: dict[str, object],
    geo_data: dict[str, object],
    boundaries: list[dict],
    convection_scheme: str,
    solver: str = "spsolve",
    under_relaxation: float = 0.7,
    time_step_size: float | None = None,
    velocity_old: np.ndarray | None = None,
    velocity_older: np.ndarray | None = None,
    previous_time_step_size: float | None = None,
    ddt_scheme: str = "euler_implicit",
    return_diagnostics: bool = False,
    return_operator: bool = False,
    *,
    momentum_tolerance: float,
    momentum_relative_tolerance: float,
    maxiter: int,
    linear_backend: str = "scipy",
    parallel_context: "ParallelContext | None" = None,
    partitioned_workspace: "PartitionedLinearWorkspace | None" = None,
    matrix_workspace: "MatrixAssemblyWorkspace | None" = None,
    log_sink: "StepLog | None" = None,
    velocity_gradient: np.ndarray | None = None,
    kinematic_pressure_gradient: np.ndarray | None = None,
) -> Any:
    """Assemble and solve the three momentum predictors.

    ``velocity_gradient`` and ``kinematic_pressure_gradient`` are gradients the
    caller computed from ``velocity`` and ``kinematic_pressure`` as passed;
    ``None`` computes them in the assembly.

    Returns the predicted cell-and-boundary velocity and relaxed cell
    diagonals used by the pressure correction.
    With ``return_diagnostics`` the per-component linear-solve diagnostics
    follow; with ``return_operator`` the relaxed momentum matrices, keyed by
    ``"x"``, ``"y"`` and ``"z"``, come last. PISO correctors use them to
    update ``H(u)`` from the corrected velocity.
    """

    n_cells = mesh_data["n_cells"]
    n_boundary = mesh_data["n_faces"] - mesh_data["n_interior_faces"]

    # Assemble momentum equations
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
        previous_time_step_size=previous_time_step_size,
        ddt_scheme=ddt_scheme,
        matrix_workspace=matrix_workspace,
        velocity_gradient=velocity_gradient,
        kinematic_pressure_gradient=kinematic_pressure_gradient,
    )

    # Solve for each component
    velocity_star = np.zeros((n_cells + n_boundary, 3))
    # Boundary ghosts start from the current state; update_velocity_boundaries
    # then sets every one of them from the predicted interior and the prescribed values.
    velocity_star[n_cells:] = velocity[n_cells : n_cells + n_boundary]
    matrices_are_shared = all(mom_eqs[name]["A"] is mom_eqs["x"]["A"] for name in ("y", "z"))
    component_diagonal = not matrices_are_shared
    if parallel_context is not None and parallel_context.is_partitioned:
        # A rank without mixed boundary faces may still exchange with one
        # that has them. All halo messages and pressure operators need the
        # same trailing shape; local matrix sharing remains independent.
        component_diagonal = bool(parallel_context.global_max(int(component_diagonal)))
    # Standard boundaries retain the compact shared diagonal.  The
    # directional mixed condition returns component diagonals for the
    # pressure/Rhie-Chow vector path that already supports them.
    momentum_diagonal = np.empty((n_cells, 3) if component_diagonal else n_cells, dtype=np.float64)
    solve_diagnostics = {}

    # Relax each distinct matrix once: components may share one matrix.
    relaxed_diagonals: dict[int, np.ndarray] = {}
    for i_comp, comp_name in enumerate(["x", "y", "z"]):
        A_relaxed = mom_eqs[comp_name]["A"]
        diag_new = relaxed_diagonals.get(id(A_relaxed))
        if diag_new is None:
            diag_new = A_relaxed.diagonal() / under_relaxation
            A_relaxed.setdiag(diag_new)
            relaxed_diagonals[id(A_relaxed)] = diag_new
        if component_diagonal:
            momentum_diagonal[:, i_comp] = diag_new
        else:
            momentum_diagonal[:] = diag_new
        b = mom_eqs[comp_name]["b"]

        # Add source term to RHS to ensure consistency at convergence
        # velocity[:n_cells, i_comp] is the previous-iterate velocity
        source_relax = (1.0 - under_relaxation) * diag_new * velocity[:n_cells, i_comp]
        b_relaxed = b + source_relax

        # Determine initial guess x0 for iterative solver: prefer velocity_old
        # (previous time step) if available
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
            maxiter=maxiter,
            backend=linear_backend,
            parallel_context=parallel_context,
            partitioned_workspace=partitioned_workspace,
            matrix_workspace=matrix_workspace,
            log_sink=log_sink,
        )
        solve_diagnostics[comp_name] = {
            "initial_residual": linear_result.initial_residual,
            "final_residual": linear_result.final_residual,
            "linear_result": linear_result,
        }

        velocity_star[:n_cells, i_comp] = velocity_component_star

    if parallel_context is not None and parallel_context.is_partitioned:
        parallel_context.exchange_halo(momentum_diagonal)

    update_velocity_boundaries(velocity_star, volumetric_face_flux, boundaries, mesh_data, geo_data)

    return _predictor_result(
        velocity_star,
        momentum_diagonal,
        solve_diagnostics if return_diagnostics else None,
        {name: mom_eqs[name]["A"] for name in "xyz"} if return_operator else None,
    )


def _predictor_result(
    velocity_star: np.ndarray,
    momentum_diagonal: np.ndarray,
    diagnostics: dict | None,
    operator: dict | None,
) -> tuple:
    """Return the predictor outputs that the caller requested."""
    result = (velocity_star, momentum_diagonal)
    if diagnostics is not None:
        result += (diagnostics,)
    if operator is not None:
        result += (operator,)
    return result
