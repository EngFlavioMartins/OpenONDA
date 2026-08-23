#!/usr/bin/env python3
"""Transient incompressible PIMPLE solver."""

import os
from typing import Any

import numpy as np

from ..assemble import momentum
from ..fields import diagnostics as field_diagnostics
from ..io import logging
from . import simple_solver
from .contracts import OuterCorrectorDiagnostics
from .linear_interface import solve_linear_system


class PIMPLESolver(simple_solver.SIMPLESolver):
    """Transient PIMPLE algorithm for incompressible Navier–Stokes.

    Combines the PISO inner corrector loop (momentum predictor + multiple
    pressure correctors per time step) with outer correctors that re-solve
    the momentum equation with an updated pressure field.  This hybrid
    approach allows larger time steps than pure PISO while retaining good
    temporal accuracy.

    Inherits the SIMPLE boundary handling and pressure-correction operators;
    replaces the steady outer iteration with a time-advancement loop driven
    by the :class:`~source.solvers.fvm.core.solver.FVMSolver`.

    References
    ----------
    - Issa, R. I. "Solution of the implicitly discretised fluid flow
      equations by operator-splitting." *J. Comput. Phys.*, 62(1):40–65, 1986.
    - Issa (1986), solution of implicitly discretised fluid-flow equations

    Examples
    --------
    >>> solver = PIMPLESolver(mesh_data, geo_data, boundaries, params)  # doctest: +SKIP
    >>> velocity, kinematic_pressure, volumetric_face_flux, residuals = solver.step(  # doctest: +SKIP
    ...     velocity, kinematic_pressure, volumetric_face_flux, velocity_old=velocity_old, time_step_size=0.01, density=1.225, kinematic_viscosity=1.5e-5
    ... )
    """

    def __init__(self, mesh_data, geo_data, boundaries, params=None):
        """Initialise PIMPLE with transient defaults."""
        super().__init__(mesh_data, geo_data, boundaries, params)

        pimple_defaults = {
            "n_correctors": 2,
            "n_orthogonal_correctors": 0,
            "velocity_relaxation": 1.0,
            "pressure_relaxation": 1.0,
            "max_iterations": 20,
            "momentum_tolerance": 1e-4,
            "convection_scheme": "deferred",
            "ibm_second_solve": True,
        }

        for key, val in pimple_defaults.items():
            if key not in self.params:
                self.params[key] = val

        # Optional immersed-boundary forcing (set via FVMSolver.set_immersed_bodies).
        self.ibm = None
        self.last_linear_results = ()
        self.last_outer_diagnostics = ()
        self._partitioned_linear_workspaces = {}
        self._partitioned_workspace_policy = os.environ.get(
            "FVM_PETSC_WORKSPACE_POLICY", "shared"
        ).lower()
        if self._partitioned_workspace_policy not in {"shared", "separate"}:
            raise ValueError(
                "FVM_PETSC_WORKSPACE_POLICY must be 'shared' or 'separate', got "
                f"{self._partitioned_workspace_policy!r}"
            )

    def _partitioned_workspace(self, equation: str):
        """Return the solver-owned PETSc workspace for a partitioned equation."""
        parallel = self.params.get("_parallel_context")
        if parallel is None or not parallel.is_partitioned:
            return None
        key = equation if self._partitioned_workspace_policy == "separate" else "flow"
        workspace = self._partitioned_linear_workspaces.get(key)
        if workspace is None:
            from .petsc_partitioned import PartitionedLinearWorkspace

            workspace = PartitionedLinearWorkspace(parallel)
            self._partitioned_linear_workspaces[key] = workspace
        return workspace

    def close(self) -> None:
        """Release collectively-owned persistent PETSc workspaces."""
        for workspace in self._partitioned_linear_workspaces.values():
            workspace.close()
        self._partitioned_linear_workspaces.clear()
        super().close()

    def step(
        self,
        velocity,
        kinematic_pressure,
        volumetric_face_flux,
        velocity_old=None,
        time_step_size=None,
        density=1.0,
        kinematic_viscosity=0.01,
        velocity_older=None,
        source_explicit=None,
        source_implicit=None,
        volumetric_face_flux_old=None,
        volumetric_face_flux_older=None,
    ):
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
            source_explicit: Explicit acceleration source [m/s²].
            source_implicit: Non-negative implicit source coefficient [1/s].

        Returns:
            tuple: Updated ``(velocity, kinematic_pressure, volumetric_face_flux, residuals)``.
        """
        if velocity_old is None or time_step_size is None:
            raise ValueError("PIMPLESolver.step requires velocity_old and time_step_size")
        n_elem = self.mesh_data["n_cells"]
        n_outer = int(self.params.get("n_outer_correctors", 1))
        n_corr = int(self.params["n_correctors"])
        velocity_relaxation = self.params["velocity_relaxation"]
        pressure_relaxation = self.params["pressure_relaxation"]
        momentum_method = self.params.get("momentum_solver") or self.params["linear_solver"]
        pressure_method = self.params.get("pressure_solver") or self.params["linear_solver"]
        pressure_constraint = simple_solver._resolve_pressure_constraint(self.params)
        pressure_matrix_reusable = simple_solver._pressure_boundary_matrix_is_reusable(
            self.boundaries
        )

        def _linear_tolerances(equation: str, *, final: bool) -> tuple[float, float]:
            absolute = float(self.params.get(f"{equation}_tolerance", 1e-8))
            relative = self.params.get(f"{equation}_relative_tolerance", 0.0)
            if final:
                final_relative = self.params.get(f"{equation}_final_relative_tolerance", 0.0)
                if final_relative is not None:
                    relative = final_relative
            return absolute, float(relative)

        ts = str(self.params.get("time_scheme", "euler_implicit")).lower()
        ddt_scheme = "backward" if ts in ("backward", "bdf2") else "euler"
        if ddt_scheme == "backward" and velocity_older is None:
            ddt_scheme = "euler"  # self-starting first step

        # Frozen for the whole physical time step: only committed old-time
        # fields enter the face-history correction.
        ddt_flux_correction = None
        if bool(self.params.get("ddt_corr", True)):
            ddt_flux_correction = simple_solver.compute_ddt_flux_correction(
                velocity_old,
                velocity_older,
                volumetric_face_flux_old,
                volumetric_face_flux_older,
                time_step_size,
                self.mesh_data,
                self.geo_data,
                self.boundaries,
                ddt_scheme,
            )

        simple_solver.update_scalar_boundaries(
            kinematic_pressure,
            self.mesh_data,
            self.boundaries,
            field_name="kinematic_pressure",
            volumetric_face_flux=volumetric_face_flux,
        )
        initial_kinematic_pressure_residual = 0.0
        final_kinematic_pressure_residual = 0.0
        momentum_diagnostics = {}
        linear_results = []
        outer_diagnostics = []
        logger = self.params.get("_logger")

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
        final_iteration = False

        for outer in range(n_outer):
            final_iteration = final_iteration or outer == n_outer - 1
            alpha_u_outer = 1.0 if final_iteration else velocity_relaxation
            alpha_p_outer = 1.0 if final_iteration else pressure_relaxation
            momentum_tolerance, momentum_relative_tolerance = _linear_tolerances(
                "momentum", final=final_iteration
            )

            def _solve_predictor(
                src_explicit,
                volumetric_face_flux=volumetric_face_flux,
                velocity_relaxation=alpha_u_outer,
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
                    ddt_scheme=ddt_scheme,
                    source_explicit=src_explicit,
                    source_implicit=source_implicit,
                    reuse_ilu=self.params.get("reuse_ilu", False),
                    ilu_key=self.params.get("ilu_key", None),
                    ilu_drop_tolerance=self.params.get("ilu_drop_tolerance", 1e-4),
                    ilu_fill_factor=self.params.get("ilu_fill_factor", 10),
                    momentum_tolerance=momentum_tolerance,
                    momentum_relative_tolerance=momentum_relative_tolerance,
                    maxiter=self.params.get("momentum_max_iterations", 1000),
                    ilu_reuse_tolerance=self.params.get("ilu_reuse_tolerance", None),
                    linear_backend=self.params.get("_linear_backend", "scipy"),
                    parallel_context=self.params.get("_parallel_context"),
                    # The default shared policy bounds full-mesh RAM.  The
                    # explicit separate policy retains equation-specific PETSc
                    # objects so pressure agglomeration can be cached.
                    partitioned_workspace=self._partitioned_workspace("momentum"),
                    failure_policy=self.params.get("linear_failure_policy", "raise"),
                    log_sink=logger,
                    matrix_workspace=self._momentum_matrix_workspace,
                    operator_backend=self.params.get("_operator_backend", "numpy"),
                    return_diagnostics=True,
                )

            logging.Timer.start("Momentum Predictor")
            velocity_star, momentum_diagonal, momentum_diagnostics = _solve_predictor(
                source_explicit
            )
            linear_results.extend(
                values["linear_result"] for values in momentum_diagnostics.values()
            )
            logging.Timer.log(
                "Momentum Predictor",
                sink=logger,
            )

            ibm = getattr(self, "ibm", None)
            if ibm is not None:
                logging.Timer.start("IBM Forcing")
                n_loops = int(self.params.get("ibm_forcing_loops", 2))
                if bool(self.params["ibm_second_solve"]):
                    src_ibm = density * ibm.compute_force(velocity_star, time_step_size)
                    if source_explicit is not None:
                        src_ibm = src_ibm + source_explicit
                    velocity_star, momentum_diagonal, momentum_diagnostics = _solve_predictor(
                        src_ibm
                    )
                    linear_results.extend(
                        values["linear_result"] for values in momentum_diagnostics.values()
                    )
                    ibm.multidirect_correct(velocity_star, time_step_size, n_iterations=n_loops)
                else:
                    ibm.begin_step()
                    ibm.multidirect_correct(
                        velocity_star, time_step_size, n_iterations=max(n_loops, 2)
                    )
                logging.Timer.log(
                    "IBM Forcing",
                    sink=logger,
                )

            # The predictor is no longer needed as an immutable field: later
            # pressure correctors advance this same state in place. Reuse its
            # storage instead of retaining two full three-component velocity
            # arrays for the entire pressure loop.
            velocity_iter = velocity_star
            pressure_geometry = None

            for _corr in range(n_corr):
                n_non_ortho = int(self.params.get("n_orthogonal_correctors", 0))
                for non_ortho in range(n_non_ortho + 1):
                    # Coupling can intentionally replace a patch type between
                    # calls.  Keep cached indexing only while that structural
                    # contract is unchanged.
                    if (
                        self._pressure_boundary_layout.signature
                        != simple_solver._pressure_boundary_signature(self.boundaries)
                    ):
                        self._pressure_boundary_layout = (
                            simple_solver.build_pressure_boundary_layout(
                                self.boundaries,
                                self.mesh_data["n_interior_faces"],
                                self.mesh_data["n_faces"],
                            )
                        )
                        pressure_geometry = None
                    reuse_pressure_matrix = (
                        pressure_geometry is not None and pressure_matrix_reusable
                    )
                    logging.Timer.start("Pressure Assembly")
                    (
                        pressure_matrix,
                        pressure_right_hand_side,
                        volumetric_face_flux_star,
                        pressure_workspace,
                    ) = simple_solver.assemble_pressure_correction_equation_rhie_chow(
                        velocity_iter,
                        momentum_diagonal,
                        kinematic_pressure,
                        density,
                        self.mesh_data,
                        self.geo_data,
                        self.boundaries,
                        velocity_relaxation=alpha_u_outer,
                        pressure_constraint=pressure_constraint,
                        matrix_workspace=self._pressure_matrix_workspace,
                        operator_backend=self.params.get("_operator_backend", "numpy"),
                        boundary_layout=self._pressure_boundary_layout,
                        ddt_flux_correction=ddt_flux_correction,
                        correction_workspace=pressure_geometry,
                        reuse_matrix=reuse_pressure_matrix,
                        return_workspace=True,
                    )
                    if pressure_geometry is None:
                        pressure_geometry = pressure_workspace
                    logging.Timer.log(
                        "Pressure Assembly",
                        sink=logger,
                    )
                    has_pressure_nullspace = simple_solver._pressure_requires_constraint(
                        self.boundaries, velocity_iter, self.mesh_data, self.geo_data
                    )

                    logging.Timer.start("Pressure Solve")
                    final_pressure_solve = _corr == n_corr - 1 and non_ortho == n_non_ortho
                    pressure_tolerance, pressure_relative_tolerance = _linear_tolerances(
                        "pressure", final=final_pressure_solve
                    )
                    pressure_max_iterations = int(self.params.get("pressure_max_iterations", 500))
                    amg_tolerance = self.params.get("amg_tolerance")
                    amg_max_iterations = self.params.get("amg_max_iterations")
                    kinematic_pressure_correction, kinematic_pressure_result = solve_linear_system(
                        pressure_matrix,
                        pressure_right_hand_side,
                        method=pressure_method,
                        equation_type="kinematic_pressure",
                        tol=pressure_tolerance,
                        rel_tol=pressure_relative_tolerance,
                        maxiter=pressure_max_iterations,
                        amg_tolerance=None if amg_tolerance is None else float(amg_tolerance),
                        amg_max_iterations=(
                            pressure_max_iterations
                            if amg_max_iterations is None
                            else int(amg_max_iterations)
                        ),
                        amg_reuse_tolerance=float(self.params.get("amg_reuse_tolerance", 0.05)),
                        amg_key=(
                            "kinematic_pressure",
                            self._pressure_matrix_workspace.cache_namespace,
                        ),
                        backend=self.params.get("_linear_backend", "scipy"),
                        parallel_context=self.params.get("_parallel_context"),
                        failure_policy=self.params.get("linear_failure_policy", "raise"),
                        log_sink=logger,
                        nullspace=(
                            "constant"
                            if has_pressure_nullspace and pressure_constraint == "nullspace"
                            else None
                        ),
                        partitioned_workspace=self._partitioned_workspace("kinematic_pressure"),
                        matrix_values_unchanged=reuse_pressure_matrix,
                        return_info=True,
                    )
                    linear_results.append(kinematic_pressure_result)
                    initial_kinematic_pressure_residual = kinematic_pressure_result.initial_residual
                    final_kinematic_pressure_residual = kinematic_pressure_result.final_residual
                    parallel = self.params.get("_parallel_context")
                    del pressure_matrix, pressure_right_hand_side
                    logging.Timer.log(
                        "Pressure Solve",
                        sink=logger,
                    )

                    logging.Timer.start("Velocity Correction")
                    velocity_iter, corrected_volumetric_face_flux = (
                        simple_solver.correct_velocity_and_flux(
                            velocity_iter,
                            volumetric_face_flux_star,
                            kinematic_pressure_correction,
                            momentum_diagonal,
                            self.mesh_data,
                            self.geo_data,
                            self.boundaries,
                            density=density,
                            velocity_relaxation=alpha_u_outer,
                            pressure_relaxation=alpha_p_outer,
                            workspace=pressure_workspace,
                        )
                    )
                    if non_ortho == n_non_ortho:
                        volumetric_face_flux = corrected_volumetric_face_flux

                    logging.Timer.log(
                        "Velocity Correction",
                        sink=logger,
                    )

                    kinematic_pressure[:n_elem] += alpha_p_outer * kinematic_pressure_correction
                    simple_solver.update_scalar_boundaries(
                        kinematic_pressure,
                        self.mesh_data,
                        self.boundaries,
                        field_name="kinematic_pressure",
                        volumetric_face_flux=volumetric_face_flux,
                    )
                    if parallel is not None and parallel.is_partitioned:
                        parallel.exchange_halo(kinematic_pressure[:n_elem])
                    # Do not carry one corrector's face/cell work arrays into
                    # the next pressure assembly. Python otherwise keeps the
                    # previous assignment alive until the new call returns,
                    # nearly doubling the peak for nCorrectors/nonOrth loops.
                    del kinematic_pressure_correction
                    del volumetric_face_flux_star
                    del corrected_volumetric_face_flux

            simple_solver._update_velocity_bcs(
                velocity_iter,
                volumetric_face_flux,
                self.boundaries,
                self.mesh_data["owners"],
                self.geo_data,
                n_elem,
                self.mesh_data["n_interior_faces"],
                mesh_data=self.mesh_data,
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
                volumetric_face_flux, self.mesh_data, self.geo_data
            )
            volumes = self.geo_data["cell_volume"]
            parallel = self.params.get("_parallel_context")
            n_owned = (
                parallel.n_owned if parallel is not None and parallel.is_partitioned else n_elem
            )
            local_continuity = float(
                np.max(np.abs(continuity[:n_owned]) / (volumes[:n_owned] + 1e-30))
            )
            continuity_outer = (
                float(parallel.global_max(local_continuity))
                if parallel is not None and parallel.is_partitioned
                else local_continuity
            )
            outer_diagnostics.append(
                OuterCorrectorDiagnostics(
                    index=outer,
                    velocity_residual=outer_velocity_residual,
                    kinematic_pressure_residual=final_kinematic_pressure_residual,
                    max_continuity_error=continuity_outer,
                )
            )

            # The committed velocity/volumetric_face_flux fields now own the corrected state. Drop
            # predictor-sized arrays before the next outer momentum assembly.
            del velocity_star, momentum_diagonal, velocity_iter, pressure_geometry

            if final_iteration:
                break

            checks = []
            residual_tolerance = self.params.get("outer_residual_tolerance")
            if residual_tolerance is not None:
                checks.append(
                    max(outer_velocity_residual, final_kinematic_pressure_residual)
                    <= float(residual_tolerance)
                )
            continuity_tolerance = self.params.get("outer_continuity_tolerance")
            if continuity_tolerance is not None:
                checks.append(continuity_outer <= float(continuity_tolerance))
            minimum = int(self.params.get("min_outer_correctors", 1))
            if checks and all(checks) and outer + 1 >= minimum:
                # ``pimpleControl::loop`` does not stop on the iteration whose
                # residuals satisfied the criteria: it flags the next one final
                # and runs it unrelaxed before leaving the loop.
                final_iteration = True

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
        }
        residuals.update(
            {
                f"velocity_{comp}": value
                for comp, value in final_velocity_component_residuals.items()
            }
        )
        self.last_linear_results = tuple(linear_results)
        self.last_outer_diagnostics = tuple(outer_diagnostics)

        # Under the low-memory shared policy the workspace now contains the
        # final pressure GAMG hierarchy.  Destroy it before allocating
        # full-mesh diagnostics so those large lifetimes never overlap.  The
        # separate policy deliberately retains both equation workspaces.
        if self._partitioned_workspace_policy == "shared":
            flow_workspace = self._partitioned_linear_workspaces.get("flow")
            if flow_workspace is not None:
                flow_workspace.close()
        return velocity, kinematic_pressure, volumetric_face_flux, residuals
