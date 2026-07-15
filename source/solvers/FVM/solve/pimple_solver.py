#!/usr/bin/env python3
"""Transient incompressible PIMPLE solver."""

import numpy as np

from ..assemble import matrix_assembly, momentum
from ..fields import diagnostics as field_diagnostics
from ..io import logging
from . import simple_solver
from .contracts import OuterCorrectorDiagnostics


class PIMPLESolver(simple_solver.SIMPLESolver):
    """PIMPLE algorithm using the SIMPLE boundary and correction operators."""

    def __init__(self, mesh_data, geo_data, boundaries, params=None):
        """Initialise PIMPLE with transient defaults."""
        super().__init__(mesh_data, geo_data, boundaries, params)

        pimple_defaults = {
            "n_correctors": 2,
            "n_orthogonal_correctors": 0,
            "alpha_u": 1.0,
            "alpha_p": 1.0,
            "max_iter": 20,
            "momentum_tol": 1e-4,
            "convection_scheme": "deferred",
            "ibm_second_solve": True,
        }

        for key, val in pimple_defaults.items():
            if key not in self.params:
                self.params[key] = val

        # Optional immersed-boundary forcing (set via Solver.set_immersed_bodies).
        self.ibm = None
        self.last_linear_results = ()
        self.last_outer_diagnostics = ()

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
        """Perform one PIMPLE time step and return fields plus residuals."""
        assert U_old is not None and dt is not None, "PIMPLESolver.step requires U_old and dt"
        n_elem = self.mesh_data["n_elements"]
        n_outer = int(self.params.get("n_outer_correctors", 1))
        n_corr = int(self.params["n_correctors"])
        alpha_u = self.params["alpha_u"]
        alpha_p = self.params["alpha_p"]
        momentum_method = self.params.get("momentum_solver") or self.params["linear_solver"]
        pressure_method = self.params.get("pressure_solver") or self.params["linear_solver"]

        ts = str(self.params.get("time_scheme", "euler_implicit")).lower()
        ddt_scheme = "backward" if ts in ("backward", "bdf2") else "euler"
        if ddt_scheme == "backward" and U_old_old is None:
            ddt_scheme = "euler"  # self-starting first step

        simple_solver.update_scalar_boundaries(p, self.mesh_data, self.boundaries, field_name="p")
        pressure_initial_residual = 0.0
        pressure_final_residual = 0.0
        momentum_diagnostics = {}
        linear_results = []
        outer_diagnostics = []

        for outer in range(n_outer):

            def _solve_predictor(src_explicit, phi=phi):
                return momentum.solve_momentum_predictor(
                    U,
                    p,
                    phi,
                    rho,
                    nu,
                    self.mesh_data,
                    self.geo_data,
                    self.boundaries,
                    convection_scheme=self.params["convection_scheme"],
                    solver=momentum_method,
                    under_relaxation=alpha_u,
                    dt=dt,
                    U_old=U_old,
                    U_old_old=U_old_old,
                    ddt_scheme=ddt_scheme,
                    source_explicit=src_explicit,
                    source_implicit=source_implicit,
                    reuse_ilu=self.params.get("reuse_ilu", False),
                    ilu_key=self.params.get("ilu_key", None),
                    ilu_drop_tol=self.params.get("ilu_drop_tol", 1e-4),
                    ilu_fill_factor=self.params.get("ilu_fill_factor", 10),
                    momentum_tol=self.params.get("momentum_tol", 1e-4),
                    maxiter=self.params.get("momentum_maxiter", 1000),
                    ilu_reuse_tol=self.params.get("ilu_reuse_tol", None),
                    linear_backend=self.params.get("_linear_backend", "scipy"),
                    parallel_context=self.params.get("_parallel_context"),
                    failure_policy=self.params.get("linear_failure_policy", "raise"),
                    return_diagnostics=True,
                )

            logging.Timer.start("    Momentum Predictor")
            U_star, A_U, momentum_diagnostics = _solve_predictor(source_explicit)
            linear_results.extend(
                values["linear_result"] for values in momentum_diagnostics.values()
            )
            logging.Timer.log("    Momentum Predictor")

            ibm = getattr(self, "ibm", None)
            if ibm is not None:
                logging.Timer.start("    IBM Forcing")
                n_loops = int(self.params.get("ibm_forcing_loops", 2))
                if bool(self.params["ibm_second_solve"]):
                    src_ibm = rho * ibm.compute_force(U_star, dt)
                    if source_explicit is not None:
                        src_ibm = src_ibm + source_explicit
                    U_star, A_U, momentum_diagnostics = _solve_predictor(src_ibm)
                    linear_results.extend(
                        values["linear_result"] for values in momentum_diagnostics.values()
                    )
                    ibm.multidirect_correct(U_star, dt, n_iter=n_loops)
                else:
                    ibm.begin_step()
                    ibm.multidirect_correct(U_star, dt, n_iter=max(n_loops, 2))
                logging.Timer.log("    IBM Forcing")

            U_iter = U_star.copy()

            for _corr in range(n_corr):
                n_non_ortho = int(self.params.get("n_orthogonal_correctors", 0))
                for non_ortho in range(n_non_ortho + 1):
                    logging.Timer.start("    Pressure Correction")
                    A_p, b_p, phi_star = (
                        simple_solver.assemble_pressure_correction_equation_rhie_chow(
                            U_iter,
                            A_U,
                            p,
                            rho,
                            self.mesh_data,
                            self.geo_data,
                            self.boundaries,
                            alpha_u=alpha_u,
                        )
                    )
                    logging.Timer.log("    Pressure Assembly")

                    zero_guess = np.zeros_like(b_p)
                    pressure_initial_residual = matrix_assembly.normalized_residual(
                        A_p, zero_guess, b_p
                    )
                    logging.Timer.start("    Pressure Solve")
                    pressure_tol = float(self.params.get("pressure_tol", 1e-8))
                    pressure_maxiter = int(self.params.get("pressure_maxiter", 500))
                    amg_tol = self.params.get("amg_tol")
                    amg_maxiter = self.params.get("amg_maxiter")
                    p_prime, pressure_result = matrix_assembly.solve_linear_system(
                        A_p,
                        b_p,
                        method=pressure_method,
                        equation_type="pressure",
                        tol=pressure_tol,
                        maxiter=pressure_maxiter,
                        amg_tol=pressure_tol if amg_tol is None else float(amg_tol),
                        amg_maxiter=(pressure_maxiter if amg_maxiter is None else int(amg_maxiter)),
                        amg_reuse_tol=float(self.params.get("amg_reuse_tol", 0.05)),
                        backend=self.params.get("_linear_backend", "scipy"),
                        parallel_context=self.params.get("_parallel_context"),
                        failure_policy=self.params.get("linear_failure_policy", "raise"),
                        return_info=True,
                    )
                    linear_results.append(pressure_result)
                    pressure_final_residual = matrix_assembly.normalized_residual(A_p, p_prime, b_p)
                    logging.Timer.log("    Pressure Solve")

                    logging.Timer.start("    Velocity Correction")
                    U_iter, corrected_phi = simple_solver.correct_velocity_and_flux(
                        U_iter,
                        phi_star.copy(),
                        p_prime,
                        A_U,
                        self.mesh_data,
                        self.geo_data,
                        self.boundaries,
                        rho=rho,
                        alpha_u=alpha_u,
                    )
                    if non_ortho == n_non_ortho:
                        phi = corrected_phi

                    logging.Timer.log("    Velocity Correction")

                    p[:n_elem] += alpha_p * p_prime
                    simple_solver.update_scalar_boundaries(
                        p, self.mesh_data, self.boundaries, field_name="p"
                    )

            simple_solver._update_velocity_bcs(
                U_iter,
                phi,
                self.boundaries,
                self.mesh_data["owners"],
                self.geo_data,
                n_elem,
                self.mesh_data["n_interior_faces"],
            )
            # Preserve updated ghost values for the next gradient assembly.
            U[:] = U_iter[:]

            momentum_outer = max(
                (values["final_residual"] for values in momentum_diagnostics.values()),
                default=0.0,
            )
            continuity = field_diagnostics.compute_continuity_error(
                phi, self.mesh_data, self.geo_data
            )
            volumes = self.geo_data["element_volumes"]
            continuity_outer = float(np.max(np.abs(continuity) / (volumes + 1e-30)))
            outer_diagnostics.append(
                OuterCorrectorDiagnostics(
                    index=outer,
                    momentum_residual=momentum_outer,
                    pressure_residual=pressure_final_residual,
                    continuity_max=continuity_outer,
                )
            )

            checks = []
            residual_tolerance = self.params.get("outer_residual_tolerance")
            if residual_tolerance is not None:
                checks.append(
                    max(momentum_outer, pressure_final_residual) <= float(residual_tolerance)
                )
            continuity_tolerance = self.params.get("outer_continuity_tolerance")
            if continuity_tolerance is not None:
                checks.append(continuity_outer <= float(continuity_tolerance))
            minimum = int(self.params.get("min_outer_correctors", 1))
            if checks and all(checks) and outer + 1 >= minimum:
                break

        momentum_final = {
            comp: values["final_residual"] for comp, values in momentum_diagnostics.items()
        }
        residual_u = max(momentum_final.values(), default=0.0)
        increment_u = np.linalg.norm(U[:n_elem] - U_old[:n_elem]) / (
            np.linalg.norm(U[:n_elem]) + 1e-10
        )

        residuals = {
            "p": pressure_final_residual,
            "U": residual_u,
            "p_initial": pressure_initial_residual,
            "U_increment": increment_u,
        }
        residuals.update({f"U_{comp}": value for comp, value in momentum_final.items()})
        self.last_linear_results = tuple(linear_results)
        self.last_outer_diagnostics = tuple(outer_diagnostics)
        return U, p, phi, residuals
