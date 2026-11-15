"""Transient incompressible PIMPLE solver."""

from typing import TYPE_CHECKING, Any

import numpy as np

from openonda.core.profiling import Profiler

from ..assemble import matrix_assembly, momentum
from ..assemble.time_integration import backward_coefficients
from ..diagnostics import fields as field_diagnostics
from ..diagnostics.step import OuterCorrectorDiagnostics
from ..fields import gradients
from ..fields.boundary_values import (
    prepare_freestream_boundaries,
    update_scalar_boundaries,
    update_velocity_boundaries,
)
from ..schemes.boundaries import BoundaryStrategy, boundary_strategy
from . import pressure_equation
from .linear_interface import solve_linear_system

if TYPE_CHECKING:
    from ..core.parallel import ParallelContext
    from .petsc_partitioned import PartitionedLinearWorkspace


def compute_ddt_flux_correction(
    velocity_old: np.ndarray,
    velocity_older: np.ndarray | None,
    volumetric_face_flux_old: np.ndarray,
    volumetric_face_flux_older: np.ndarray | None,
    time_step_size: float,
    mesh_data: dict[str, object],
    geo_data: dict[str, object],
    boundaries: list[dict],
    ddt_scheme: str,
    previous_time_step_size: float | None = None,
    mesh_face_flux: np.ndarray | None = None,
) -> np.ndarray | None:
    r"""Return the transient Rhie--Chow face-history correction.

    This is the limited difference between the committed face flux and the
    flux obtained by interpolating the committed cell velocity, divided by the
    time step.  Variable-step BDF2 uses the same two-level history combination
    as the cell derivative, and every non-coupled boundary contribution is zero.

    The face value of ``H/A`` interpolates the cell velocity, whose transient
    source carries ``V/dt * u_old``; scaled by the face pressure-velocity
    coefficient, this correction replaces that interpolated old velocity by
    the old face flux once, as ``fvc::ddtCorr`` does.  Fully periodic domains
    skip it: it carries the difference between the face flux and the
    interpolated cell velocity from step to step, and on a periodic
    Taylor-Green vortex that difference builds up to a velocity error several
    times larger, nearly independent of the time step.  On bounded domains it
    keeps a steady state independent of the time step.

    On a moving mesh the coupling coefficient is normalised by the old flux
    relative to the moving faces, ``phi_old - mesh_face_flux``, so the
    correction is independent of a uniform frame velocity.
    """
    if volumetric_face_flux_old is None or time_step_size is None or velocity_old is None:
        return None

    strategies = [boundary_strategy(b.get("velocity_type"), "velocity") for b in boundaries]
    if strategies and all(
        strategy in (BoundaryStrategy.CYCLIC, BoundaryStrategy.EMPTY) for strategy in strategies
    ):
        return None

    n_faces = mesh_data["n_faces"]
    n_interior = mesh_data["n_interior_faces"]

    if str(ddt_scheme).lower() == "backward" and (
        velocity_older is not None
        and volumetric_face_flux_older is not None
        and previous_time_step_size is not None
    ):
        _, coefft0, coefft00 = backward_coefficients(time_step_size, previous_time_step_size)
    else:
        coefft0, coefft00 = 1.0, 0.0

    volumetric_face_flux_old = np.asarray(volumetric_face_flux_old, dtype=np.float64)
    velocity_old = np.asarray(velocity_old, dtype=np.float64)
    older_flux = (
        np.zeros_like(volumetric_face_flux_old)
        if volumetric_face_flux_older is None
        else np.asarray(volumetric_face_flux_older, dtype=np.float64)
    )
    velocity_history = coefft0 * velocity_old
    if coefft00 != 0.0:
        velocity_history = velocity_history - coefft00 * np.asarray(
            velocity_older, dtype=np.float64
        )

    owners = mesh_data["owners"]
    neighbours = mesh_data["neighbours"]
    weights = geo_data["face_interpolation_weight"]
    face_area_vector = geo_data["face_area_vector"]
    coupling_flux = (
        volumetric_face_flux_old
        if mesh_face_flux is None
        else volumetric_face_flux_old - np.asarray(mesh_face_flux, dtype=np.float64)
    )
    correction = np.zeros(n_faces, dtype=np.float64)
    chunk_size = 250_000

    for start in range(0, n_interior, chunk_size):
        stop = min(start + chunk_size, n_interior)
        own = owners[start:stop]
        nei = neighbours[start:stop]
        w = weights[start:stop, np.newaxis]
        history_face = w * velocity_history[nei] + (1.0 - w) * velocity_history[own]
        old_face = w * velocity_old[nei] + (1.0 - w) * velocity_old[own]
        history_flux = coefft0 * volumetric_face_flux_old[start:stop]
        if coefft00 != 0.0:
            history_flux -= coefft00 * older_flux[start:stop]
        phi_corr = history_flux - np.einsum("ij,ij->i", history_face, face_area_vector[start:stop])
        reference = volumetric_face_flux_old[start:stop] - np.einsum(
            "ij,ij->i", old_face, face_area_vector[start:stop]
        )
        coupling = 1.0 - np.minimum(
            np.abs(reference) / (np.abs(coupling_flux[start:stop]) + np.finfo(np.float64).tiny),
            1.0,
        )
        correction[start:stop] = coupling * phi_corr / float(time_step_size)

    boundary_neighbour_cell = mesh_data.get("boundary_neighbour_cell")
    for boundary, strategy in zip(boundaries, strategies, strict=True):
        if strategy is not BoundaryStrategy.CYCLIC:
            continue
        start = boundary["start_face"]
        stop = start + boundary["n_faces"]
        faces = np.arange(start, stop)
        own = owners[faces]
        paired = boundary_neighbour_cell[faces]
        w = weights[faces, np.newaxis]
        history_face = w * velocity_history[paired] + (1.0 - w) * velocity_history[own]
        old_face = w * velocity_old[paired] + (1.0 - w) * velocity_old[own]
        history_flux = coefft0 * volumetric_face_flux_old[faces]
        if coefft00 != 0.0:
            history_flux -= coefft00 * older_flux[faces]
        phi_corr = history_flux - np.einsum("ij,ij->i", history_face, face_area_vector[faces])
        reference = volumetric_face_flux_old[faces] - np.einsum(
            "ij,ij->i", old_face, face_area_vector[faces]
        )
        coupling = 1.0 - np.minimum(
            np.abs(reference) / (np.abs(coupling_flux[faces]) + np.finfo(np.float64).tiny),
            1.0,
        )
        correction[faces] = coupling * phi_corr / float(time_step_size)

    return correction


class PIMPLESolver:
    """Transient PIMPLE algorithm for incompressible Navier–Stokes.

    Combines the PISO inner corrector loop (momentum predictor + multiple
    pressure correctors per time step) with outer correctors that re-solve
    the momentum equation with an updated pressure field.  This hybrid
    approach allows larger time steps than pure PISO while retaining good
    temporal accuracy.

    The time-advancement loop is driven by the
    :class:`~openonda.fvm.core.solver.FVMSolver`.

    References
    ----------
    - Issa, R. I. "Solution of the implicitly discretised fluid flow
      equations by operator-splitting." *J. Comput. Phys.*, 62(1):40–65, 1986.

    Examples
    --------
    >>> solver = PIMPLESolver(mesh_data, geo_data, boundaries, params)  # doctest: +SKIP
    >>> velocity, kinematic_pressure, volumetric_face_flux, residuals = solver.step(  # doctest: +SKIP
    ...     velocity, kinematic_pressure, volumetric_face_flux, velocity_old=velocity_old, time_step_size=0.01, density=1.225, kinematic_viscosity=1.5e-5
    ... )
    """

    def __init__(
        self,
        mesh_data: dict[str, object],
        geo_data: dict[str, object],
        boundaries: list[dict],
        params: dict[str, object],
    ) -> None:
        """Initialise PIMPLE with the resolved configuration parameters."""
        self.mesh_data = mesh_data
        self.geo_data = geo_data
        self.boundaries = boundaries
        self.params: dict[str, Any] = dict(params)
        # Momentum and pressure are sequential and share one static-topology
        # workspace. Its CSR values are overwritten between equations.
        self._momentum_matrix_workspace = matrix_assembly.MatrixAssemblyWorkspace.create(
            mesh_data,
            include_boundaries=mesh_data["n_faces"] > mesh_data["n_interior_faces"],
        )
        self._pressure_matrix_workspace = self._momentum_matrix_workspace
        self._pressure_boundary_layout = pressure_equation.build_pressure_boundary_layout(
            boundaries,
            mesh_data["n_interior_faces"],
            mesh_data["n_faces"],
        )
        self._profiler = self.params.get("_profiler") or Profiler(enabled=False)

        self.last_linear_results = ()
        self.last_outer_diagnostics = ()
        self._partitioned_linear_workspaces = {}

    def _partitioned_workspace(self, equation: str) -> "PartitionedLinearWorkspace | None":
        """Return the solver-owned PETSc workspace of ``"momentum"`` or ``"kinematic_pressure"``."""
        parallel = self.params.get("_parallel_context")
        if parallel is None or not parallel.is_partitioned:
            return None
        # Momentum and pressure have different operators. Separate workspaces
        # retain each operator's preconditioner across PIMPLE corrections.
        key = equation
        workspace = self._partitioned_linear_workspaces.get(key)
        if workspace is None:
            from .petsc_partitioned import PartitionedLinearWorkspace

            workspace = PartitionedLinearWorkspace(parallel, equation)
            self._partitioned_linear_workspaces[key] = workspace
        return workspace

    def rebuild_preconditioners(self) -> None:
        """Rebuild the reused serial and PETSc preconditioners at the next solves."""
        for workspace in self._partitioned_linear_workspaces.values():
            workspace.rebuild_preconditioner()
        self._pressure_matrix_workspace.amg_hierarchy = None

    def close(self) -> None:
        """Release collectively-owned persistent PETSc workspaces."""
        for workspace in self._partitioned_linear_workspaces.values():
            workspace.close()
        self._partitioned_linear_workspaces.clear()
        self.rebuild_preconditioners()
        self._momentum_matrix_workspace.direct_factorization = None

    def step(
        self,
        velocity: np.ndarray,
        kinematic_pressure: np.ndarray,
        volumetric_face_flux: np.ndarray,
        velocity_old: np.ndarray | None = None,
        time_step_size: float | None = None,
        density: float = 1.0,
        kinematic_viscosity: float | np.ndarray = 0.01,
        velocity_older: np.ndarray | None = None,
        volumetric_face_flux_old: np.ndarray | None = None,
        volumetric_face_flux_older: np.ndarray | None = None,
        previous_time_step_size: float | None = None,
        mesh_face_flux: np.ndarray | None = None,
        velocity_gradient: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
        """Perform one PIMPLE time step.

        Args:
            velocity: Cell and boundary-ghost velocity [m/s].
            kinematic_pressure: Kinematic pressure ``kinematic_pressure/ρ`` [m²/s²].
            volumetric_face_flux: Volumetric face flux ``velocity·Sf`` [m³/s].
            velocity_old: Velocity at the previous committed time [m/s].
            time_step_size: Positive time-step size [s].
            density: Positive constant reference density [kg/m³]. It cancels
                from the kinematic-pressure flow equations.
            kinematic_viscosity: Positive kinematic viscosity [m²/s].
            velocity_older: Velocity two committed time levels ago [m/s], required
                after BDF2 startup.
            previous_time_step_size: Preceding accepted step size [s], required
                with ``velocity_older`` for variable-step BDF2.
            mesh_face_flux: Face flux swept by a moving mesh, ``u_mesh . Sf``
                [m³/s], on the end-of-step geometry. Momentum is convected and
                inflow/outflow faces are selected with ``phi - mesh_face_flux``;
                continuity keeps the absolute flux ``phi``.
            velocity_gradient: Gradient of ``velocity`` as passed in, which the
                caller computed for the eddy viscosity. The first momentum
                assembly reuses it unless the boundary selection below changes
                the velocity ghosts.

        Returns:
            tuple: Updated ``(velocity, kinematic_pressure, volumetric_face_flux, residuals)``.
        """
        if velocity_old is None or time_step_size is None:
            raise ValueError("PIMPLESolver.step requires velocity_old and time_step_size")
        n_elem = self.mesh_data["n_cells"]
        ghost_velocity_before = None if velocity_gradient is None else velocity[n_elem:].copy()
        prepare_freestream_boundaries(
            velocity,
            kinematic_pressure,
            volumetric_face_flux,
            self.mesh_data,
            self.geo_data,
            self.boundaries,
            mesh_face_flux=mesh_face_flux,
        )
        if ghost_velocity_before is not None and not np.array_equal(
            ghost_velocity_before, velocity[n_elem:]
        ):
            velocity_gradient = None
        del ghost_velocity_before
        n_outer = int(self.params["n_outer_correctors"])
        n_corr = int(self.params["n_correctors"])
        velocity_relaxation = self.params["velocity_relaxation"]
        pressure_relaxation = self.params["pressure_relaxation"]
        momentum_method = self.params["linear_solver"]
        pressure_method = self.params.get("pressure_solver") or self.params["linear_solver"]
        pressure_constraint = pressure_equation._resolve_pressure_constraint(self.params)

        def _linear_tolerances(equation: str, *, final: bool) -> tuple[float, float]:
            absolute = float(self.params[f"{equation}_tolerance"])
            relative = 0.0 if final else float(self.params[f"{equation}_relative_tolerance"])
            return absolute, relative

        ts = self.params["time_scheme"]
        ddt_scheme = "backward" if ts == "backward" else "euler_implicit"
        if ddt_scheme == "backward" and velocity_older is None:
            ddt_scheme = "euler_implicit"  # self-starting first step

        # Frozen for the whole physical time step: only committed old-time
        # fields enter the face-history correction.
        ddt_flux_correction = compute_ddt_flux_correction(
            velocity_old,
            velocity_older,
            volumetric_face_flux_old,
            volumetric_face_flux_older,
            time_step_size,
            self.mesh_data,
            self.geo_data,
            self.boundaries,
            ddt_scheme,
            previous_time_step_size=previous_time_step_size,
            mesh_face_flux=mesh_face_flux,
        )

        update_scalar_boundaries(
            kinematic_pressure,
            self.mesh_data,
            self.boundaries,
            field_name="kinematic_pressure",
            volumetric_face_flux=volumetric_face_flux,
        )
        # Boundary types and geometry are fixed for the step.
        has_pressure_nullspace = pressure_equation._pressure_requires_constraint(
            self.boundaries, self.mesh_data, self.geo_data
        )
        initial_kinematic_pressure_residual = 0.0
        final_kinematic_pressure_residual = 0.0
        momentum_diagnostics = {}
        linear_results = []
        outer_diagnostics = []
        logger = self.params.get("_logger")
        parallel = self.params.get("_parallel_context")
        n_owned = parallel.n_owned if parallel is not None and parallel.is_partitioned else n_elem

        def _normalized_outer_state_update(
            previous_velocity: np.ndarray, previous_pressure: np.ndarray
        ) -> float:
            """Measure the nonlinear state change, independent of inner residuals."""
            if n_owned == 0:
                local_update = 0.0
            else:
                current_velocity = np.asarray(velocity[:n_owned], dtype=np.float64)
                velocity_scale = np.maximum(
                    np.maximum(
                        np.linalg.norm(current_velocity, axis=1),
                        np.linalg.norm(previous_velocity, axis=1),
                    ),
                    1.0e-12,
                )
                velocity_update = (
                    np.linalg.norm(current_velocity - previous_velocity, axis=1) / velocity_scale
                )
                current_pressure = np.asarray(kinematic_pressure[:n_owned], dtype=np.float64)
                pressure_scale = np.maximum(
                    np.maximum(np.abs(current_pressure), np.abs(previous_pressure)),
                    1.0e-12,
                )
                pressure_update = np.abs(current_pressure - previous_pressure) / pressure_scale
                local_update = float(
                    max(
                        np.max(velocity_update, initial=0.0),
                        np.max(pressure_update, initial=0.0),
                    )
                )
            if parallel is not None and parallel.is_partitioned:
                return float(parallel.global_max(local_update))
            return local_update

        # Apply the PIMPLE relaxation factors on every outer
        # corrector *except the last one*: ``fvMatrix::relax`` and
        # ``GeometricField::relax`` look the factor up under ``UFinal`` /
        # ``pFinal``. The relaxationFactors section does not define those
        # final-field entries, so the final
        # corrector runs unrelaxed.  That is what keeps the completed time step
        # time-consistent — relaxation only damps the *outer-iteration*
        # increment while the loop is still converging.  Relaxing the final
        # corrector too (as a plain SIMPLE sweep would) leaves a permanent
        # ``(1-α)/α · diag(A) · Δvelocity`` damping term in the committed solution,
        # which suppresses physically growing modes such as shear-layer rollup
        # and bluff-body vortex shedding.

        for outer in range(n_outer):
            outer_previous_velocity = np.asarray(velocity[:n_owned], dtype=np.float64).copy()
            outer_previous_pressure = np.asarray(
                kinematic_pressure[:n_owned], dtype=np.float64
            ).copy()
            final_iteration = outer == n_outer - 1
            outer_velocity_relaxation = 1.0 if final_iteration else velocity_relaxation
            outer_pressure_relaxation = 1.0 if final_iteration else pressure_relaxation
            momentum_tolerance, momentum_relative_tolerance = _linear_tolerances(
                "momentum", final=final_iteration
            )

            def _solve_predictor(
                predictor_pressure_gradient,
                reusable_velocity_gradient,
                volumetric_face_flux=(
                    volumetric_face_flux
                    if mesh_face_flux is None
                    else volumetric_face_flux - mesh_face_flux
                ),
                velocity_relaxation=outer_velocity_relaxation,
                momentum_tolerance=momentum_tolerance,
                momentum_relative_tolerance=momentum_relative_tolerance,
            ) -> Any:
                return momentum.solve_momentum_predictor(
                    velocity,
                    kinematic_pressure,
                    volumetric_face_flux,
                    density,
                    kinematic_viscosity,
                    self.mesh_data,
                    self.geo_data,
                    self.boundaries,
                    convection_scheme=self.params["convection_scheme"],
                    solver=momentum_method,
                    under_relaxation=velocity_relaxation,
                    time_step_size=time_step_size,
                    velocity_old=velocity_old,
                    velocity_older=velocity_older,
                    previous_time_step_size=previous_time_step_size,
                    ddt_scheme=ddt_scheme,
                    momentum_tolerance=momentum_tolerance,
                    momentum_relative_tolerance=momentum_relative_tolerance,
                    maxiter=self.params["momentum_max_iterations"],
                    linear_backend=self.params["_linear_backend"],
                    parallel_context=self.params.get("_parallel_context"),
                    # The default shared settings bounds full-mesh RAM.  The
                    # explicit separate settings retains equation-specific PETSc
                    # objects so pressure agglomeration can be cached.
                    partitioned_workspace=self._partitioned_workspace("momentum"),
                    log_sink=logger,
                    matrix_workspace=self._momentum_matrix_workspace,
                    return_diagnostics=True,
                    return_operator=True,
                    velocity_gradient=reusable_velocity_gradient,
                    kinematic_pressure_gradient=predictor_pressure_gradient,
                )

            with self._profiler.section("momentum predictor"):
                # The first pressure assembly of this outer iteration sees the
                # same pressure array, so it takes this gradient too.
                predictor_pressure_gradient = gradients.gradient(
                    kinematic_pressure, self.mesh_data, self.geo_data
                )
                velocity_star, momentum_diagonal, momentum_diagnostics, momentum_operator = (
                    _solve_predictor(predictor_pressure_gradient, velocity_gradient)
                )
                # Only the first assembly sees the velocity it was computed from;
                # release it before the pressure solve.
                velocity_gradient = None
                # Pressure assembly reuses the momentum CSR storage; later PISO
                # correctors need their own copy of the relaxed momentum matrix.
                momentum_operator = _copy_operator(momentum_operator) if n_corr > 1 else None
                linear_results.extend(
                    values["linear_result"] for values in momentum_diagnostics.values()
                )

            # The predictor is no longer needed as an immutable field: later
            # pressure correctors advance this same state in place. Reuse its
            # storage instead of retaining two full three-component velocity
            # arrays for the entire pressure loop.
            velocity_iter = velocity_star
            pressure_geometry = None
            # H/A is assembled once from the predictor and then advanced by
            # each corrector's neighbour-velocity change (Issa 1986). It is
            # never rebuilt from the pressure gradient: refreshed pressure
            # ghosts would silently change the predictor's momentum source.
            frozen_velocity_h_over_a = None
            corrected_velocity = velocity_star[:n_elem].copy()

            for _corr in range(n_corr):
                if _corr > 0:
                    # PISO: H(u) sees the previous correction's velocity
                    # change through the off-diagonal momentum coefficients.
                    # Without it every later corrector re-solves a pressure
                    # equation whose flux already satisfies continuity.
                    h_over_a_increment = _piso_h_over_a_increment(
                        momentum_operator,
                        velocity_iter[:n_elem] - corrected_velocity,
                        n_owned,
                        parallel,
                        outer_velocity_relaxation,
                    )
                    frozen_velocity_h_over_a = frozen_velocity_h_over_a + h_over_a_increment
                    velocity_iter[:n_elem] += h_over_a_increment
                    corrected_velocity = velocity_iter[:n_elem] - h_over_a_increment
                    del h_over_a_increment
                n_non_ortho = int(self.params["n_nonorthogonal_correctors"])
                pressure_before_nonorthogonal_sweeps = kinematic_pressure[:n_elem].copy()
                for non_ortho in range(n_non_ortho + 1):
                    # Coupling can replace a patch type between
                    # calls.  Keep cached indexing only while the patch
                    # layout is unchanged.
                    if (
                        self._pressure_boundary_layout.signature
                        != pressure_equation._pressure_boundary_signature(self.boundaries)
                    ):
                        self._pressure_boundary_layout = (
                            pressure_equation.build_pressure_boundary_layout(
                                self.boundaries,
                                self.mesh_data["n_interior_faces"],
                                self.mesh_data["n_faces"],
                            )
                        )
                        pressure_geometry = None
                    reuse_pressure_matrix = pressure_geometry is not None
                    with self._profiler.section("pressure assembly"):
                        (
                            pressure_matrix,
                            pressure_right_hand_side,
                            volumetric_face_flux_star,
                            pressure_workspace,
                        ) = pressure_equation.assemble_pressure_correction_equation_rhie_chow(
                            velocity_iter,
                            momentum_diagonal,
                            kinematic_pressure,
                            density,
                            self.mesh_data,
                            self.geo_data,
                            self.boundaries,
                            velocity_relaxation=outer_velocity_relaxation,
                            pressure_constraint=pressure_constraint,
                            matrix_workspace=self._pressure_matrix_workspace,
                            boundary_layout=self._pressure_boundary_layout,
                            ddt_flux_correction=ddt_flux_correction,
                            correction_workspace=pressure_geometry,
                            reuse_matrix=reuse_pressure_matrix,
                            frozen_velocity_h_over_a=frozen_velocity_h_over_a,
                            mesh_face_flux=mesh_face_flux,
                            kinematic_pressure_gradient=predictor_pressure_gradient,
                            pressure_requires_constraint=has_pressure_nullspace,
                        )
                        predictor_pressure_gradient = None
                        pressure_geometry = pressure_workspace
                        if frozen_velocity_h_over_a is None:
                            frozen_velocity_h_over_a = pressure_workspace.velocity_h_over_a

                    with self._profiler.section("pressure solve"):
                        final_pressure_solve = _corr == n_corr - 1 and non_ortho == n_non_ortho
                        pressure_tolerance, pressure_relative_tolerance = _linear_tolerances(
                            "pressure", final=final_pressure_solve
                        )
                        pressure_max_iterations = int(self.params["pressure_max_iterations"])
                        kinematic_pressure_correction, kinematic_pressure_result = (
                            solve_linear_system(
                                pressure_matrix,
                                pressure_right_hand_side,
                                method=pressure_method,
                                equation_type="kinematic_pressure",
                                tol=pressure_tolerance,
                                rel_tol=pressure_relative_tolerance,
                                maxiter=pressure_max_iterations,
                                backend=self.params["_linear_backend"],
                                parallel_context=self.params.get("_parallel_context"),
                                log_sink=logger,
                                nullspace=(
                                    "constant"
                                    if has_pressure_nullspace and pressure_constraint == "nullspace"
                                    else None
                                ),
                                partitioned_workspace=self._partitioned_workspace(
                                    "kinematic_pressure"
                                ),
                                matrix_values_unchanged=reuse_pressure_matrix,
                                matrix_workspace=self._pressure_matrix_workspace,
                            )
                        )
                        linear_results.append(kinematic_pressure_result)
                        initial_kinematic_pressure_residual = (
                            kinematic_pressure_result.initial_residual
                        )
                        final_kinematic_pressure_residual = kinematic_pressure_result.final_residual
                        parallel = self.params.get("_parallel_context")
                        del pressure_matrix, pressure_right_hand_side

                    if non_ortho < n_non_ortho:
                        # Iterate only the explicit non-orthogonal pressure
                        # contribution.  The momentum predictor (H/A), cell
                        # velocity, and persistent face flux remain frozen
                        # until the final sweep.  Treating this as another
                        # complete PISO correction over-corrects skewed meshes
                        # and made nNonOrthogonalCorrectors=1 identical to
                        # doubling nCorrectors.
                        kinematic_pressure[:n_elem] += kinematic_pressure_correction
                        pressure_flux_for_boundaries = volumetric_face_flux_star
                    else:
                        accumulated_kinematic_pressure_correction = (
                            kinematic_pressure[:n_elem]
                            - pressure_before_nonorthogonal_sweeps
                            + kinematic_pressure_correction
                        )
                        with self._profiler.section("velocity correction"):
                            velocity_iter, corrected_volumetric_face_flux = (
                                pressure_equation.correct_velocity_and_flux(
                                    velocity_iter,
                                    volumetric_face_flux_star,
                                    kinematic_pressure_correction,
                                    momentum_diagonal,
                                    self.mesh_data,
                                    self.geo_data,
                                    self.boundaries,
                                    density=density,
                                    velocity_relaxation=outer_velocity_relaxation,
                                    pressure_relaxation=outer_pressure_relaxation,
                                    workspace=pressure_workspace,
                                    velocity_kinematic_pressure_correction=(
                                        accumulated_kinematic_pressure_correction
                                    ),
                                )
                            )
                            volumetric_face_flux = corrected_volumetric_face_flux
                            pressure_flux_for_boundaries = volumetric_face_flux
                            kinematic_pressure[:n_elem] = (
                                pressure_before_nonorthogonal_sweeps
                                + outer_pressure_relaxation
                                * accumulated_kinematic_pressure_correction
                            )

                    update_scalar_boundaries(
                        kinematic_pressure,
                        self.mesh_data,
                        self.boundaries,
                        field_name="kinematic_pressure",
                        volumetric_face_flux=pressure_flux_for_boundaries,
                    )
                    if parallel is not None and parallel.is_partitioned:
                        parallel.exchange_halo(kinematic_pressure[:n_elem])
                    # Do not carry one corrector's face/cell work arrays into
                    # the next pressure assembly. Python otherwise keeps the
                    # previous assignment alive until the new call returns,
                    # nearly doubling the peak for nCorrectors/nonOrth loops.
                    del kinematic_pressure_correction
                    del volumetric_face_flux_star
                    if non_ortho == n_non_ortho:
                        del corrected_volumetric_face_flux
                        del accumulated_kinematic_pressure_correction

            update_velocity_boundaries(
                velocity_iter,
                volumetric_face_flux,
                self.boundaries,
                self.mesh_data,
                self.geo_data,
            )
            # Preserve updated ghost values for the next gradient assembly.
            velocity[:] = velocity_iter[:]
            parallel = self.params.get("_parallel_context")
            if parallel is not None and parallel.is_partitioned:
                parallel.exchange_halo(velocity[:n_elem])

            outer_velocity_residual = max(
                (values["final_residual"] for values in momentum_diagnostics.values()),
                default=0.0,
            )
            continuity = field_diagnostics.compute_continuity_error(
                volumetric_face_flux, self.mesh_data
            )
            volumes = self.geo_data["cell_volume"]
            parallel = self.params.get("_parallel_context")
            n_owned = (
                parallel.n_owned if parallel is not None and parallel.is_partitioned else n_elem
            )
            local_continuity = float(np.max(np.abs(continuity[:n_owned]) / volumes[:n_owned]))
            continuity_outer = (
                float(parallel.global_max(local_continuity))
                if parallel is not None and parallel.is_partitioned
                else local_continuity
            )
            state_update = _normalized_outer_state_update(
                outer_previous_velocity,
                outer_previous_pressure,
            )
            outer_diagnostics.append(
                OuterCorrectorDiagnostics(
                    index=outer,
                    velocity_residual=outer_velocity_residual,
                    kinematic_pressure_residual=final_kinematic_pressure_residual,
                    max_continuity_error=continuity_outer,
                    state_update=state_update,
                )
            )

            # The committed velocity/volumetric_face_flux fields now own the corrected state. Drop
            # predictor-sized arrays before the next outer momentum assembly.
            del velocity_star, momentum_diagonal, velocity_iter, pressure_geometry
            del momentum_operator, corrected_velocity

        final_velocity_component_residuals = {
            comp: values["final_residual"] for comp, values in momentum_diagnostics.items()
        }
        velocity_residual = max(final_velocity_component_residuals.values(), default=0.0)
        parallel = self.params.get("_parallel_context")
        n_owned = parallel.n_owned if parallel is not None and parallel.is_partitioned else n_elem
        increment_squared = float(np.sum((velocity[:n_owned] - velocity_old[:n_owned]) ** 2))
        velocity_squared = float(np.sum(velocity[:n_owned] ** 2))
        if parallel is not None and parallel.is_partitioned:
            increment_squared = parallel.global_sum(increment_squared)
            velocity_squared = parallel.global_sum(velocity_squared)
        velocity_increment = np.sqrt(increment_squared) / (np.sqrt(velocity_squared) + 1e-10)

        residuals = {
            "kinematic_pressure": final_kinematic_pressure_residual,
            "velocity": velocity_residual,
            "initial_kinematic_pressure": initial_kinematic_pressure_residual,
            "velocity_increment": velocity_increment,
            "nonlinear_state_update": (
                outer_diagnostics[-1].state_update if outer_diagnostics else 0.0
            ),
        }
        residuals.update(
            {
                f"velocity_{comp}": value
                for comp, value in final_velocity_component_residuals.items()
            }
        )
        self.last_linear_results = tuple(linear_results)
        self.last_outer_diagnostics = tuple(outer_diagnostics)

        return velocity, kinematic_pressure, volumetric_face_flux, residuals


def _copy_operator(operator: dict) -> dict:
    """Copy each distinct component matrix once, keeping shared matrices shared."""
    copies: dict[int, object] = {}
    return {name: copies.setdefault(id(matrix), matrix.copy()) for name, matrix in operator.items()}


def _piso_h_over_a_increment(
    operator: dict[str, object],
    velocity_change: np.ndarray,
    n_owned: int,
    parallel: "ParallelContext | None",
    velocity_relaxation: float,
) -> np.ndarray:
    """Change of H/A caused by one corrector's cell-velocity change.

    The relaxed momentum matrix is ``A = diag(a_P / alpha) + N``.  H/A and
    the Rhie--Chow coefficient ``V / a_P`` use the unrelaxed diagonal, so
    ``H(u)/a_P = (b - N u)/a_P`` changes by
    ``-N du / a_P = (du - (A du) / (a_P / alpha)) / alpha``.  Owned rows use
    the exchanged halo values of ``du``; halo rows are then refreshed from
    their owners.  Returns shape ``(n_cells, 3)``.
    """
    change = np.array(velocity_change, dtype=np.float64)
    if parallel is not None and parallel.is_partitioned:
        parallel.exchange_halo(change)
    increment = np.zeros_like(change)
    matrices = [operator[name] for name in "xyz"]
    shared = all(matrix is matrices[0] for matrix in matrices)
    groups = [(matrices[0], [0, 1, 2])] if shared else [(m, [c]) for c, m in enumerate(matrices)]
    for matrix, components in groups:
        rows = matrix[:n_owned]
        diagonal = matrix.diagonal()[:n_owned]
        applied = rows @ change[:, components]
        increment[:n_owned, components] = (
            change[:n_owned, components] - applied / diagonal[:, np.newaxis]
        ) / velocity_relaxation
    if parallel is not None and parallel.is_partitioned:
        parallel.exchange_halo(increment)
    return increment
