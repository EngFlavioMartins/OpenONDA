#!/usr/bin/env python3
"""
PIMPLE Algorithm for OpenONDA FVM Solver

PIMPLE = SIMPLE + PISO (Transient SIMPLE)

Algorithm:
1. Outer Loop (Time Step): t -> t+dt
   Store U_old

   2. Sub-iterations (PIMPLE Loop):
      Solve Momentum Predictor (using U_old for transient term)

      3. Pressure Corrector Loop (PISO-like or SIMPLE-like inner loop):
         Solve Pressure Equation
         Correct Velocity
         Update Pressure (with under-relaxation)

   Loop until convergence or fixed number of correctors.

3. Move to next time step
"""

import numpy as np

from ..assemble import matrix_assembly, momentum
from ..fields import field_io
from ..io import logging
from . import simple_solver


class PIMPLESolver(simple_solver.SIMPLESolver):
    """
    PIMPLE algorithm solver for transient incompressible flow.
    Inherits from SIMPLESolver to reuse boundary updates and correction logic.
    """

    def __init__(self, mesh_data, geo_data, boundaries, params=None):
        super().__init__(mesh_data, geo_data, boundaries, params)

        # PIMPLE-specific defaults — only applied if user did NOT explicitly
        # set them via SolverParams.  The parent __init__ already stored user
        # params in self.params, so we only inject defaults for missing keys.
        pimple_defaults = {
            "n_correctors": 2,
            "n_orthogonal_correctors": 0,
            "alpha_u": 1.0,  # Transient: no under-relaxation
            "alpha_p": 1.0,  # Transient: no under-relaxation
            "max_iter": 20,
            "momentum_tol": 1e-4,
            "convection_scheme": "deferred",
        }

        for key, val in pimple_defaults.items():
            if key not in self.params:
                self.params[key] = val

    def step(self, U, p, phi, U_old=None, dt=None, rho=1.0, nu=0.01, U_old_old=None,
             source_explicit=None, source_implicit=None):
        """
        Perform a single PIMPLE time step.

        PIMPLE = Outer (SIMPLE-like) loop + Inner (PISO-like) corrector loop.
          Outer loop: momentum predictor → inner correctors → under-relaxation
          Inner loop: pressure correction → velocity/flux correction → pressure update

        Args:
            U: Current velocity field
            p: Current pressure field
            U_old: Velocity field from previous time step
            dt: Time step size
            rho: Density
            nu: Viscosity

        Returns:
            tuple: (U_new, p_new, phi_new, residuals)
        """
        assert U_old is not None and dt is not None, "PIMPLESolver.step requires U_old and dt"
        n_elem = self.mesh_data["n_elements"]
        n_outer = int(self.params.get("n_outer_correctors", 1))
        n_corr = int(self.params["n_correctors"])
        alpha_u = self.params["alpha_u"]
        alpha_p = self.params["alpha_p"]

        # Resolve the time-discretisation scheme: BDF2 ("backward") needs the
        # second history level u^{n-1}; otherwise fall back to BDF1 ("euler").
        ts = str(self.params.get("time_scheme", "euler_implicit")).lower()
        ddt_scheme = "backward" if ts in ("backward", "bdf2") else "euler"
        if ddt_scheme == "backward" and U_old_old is None:
            ddt_scheme = "euler"  # self-starting first step

        # 0. Update Pressure Boundaries (Ghost Cells)
        simple_solver.update_scalar_boundaries(p, self.mesh_data, self.boundaries, field_name="p")

        # 1. PIMPLE Outer Loop
        for outer in range(n_outer):
            # Save pre-outer state for the between-outer under-relaxation below
            # (must be set on every iteration, incl. outer==0, since the blend
            # runs for all but the last outer corrector).
            U_before = U[:n_elem].copy()
            p_before = p[:n_elem].copy()

            # ---- 1a. Momentum Predictor (once per outer iteration) ----
            logging.Timer.start("    Momentum Predictor")
            U_star, A_U = momentum.solve_momentum_predictor(
                U, p, phi, rho, nu,
                self.mesh_data, self.geo_data, self.boundaries,
                convection_scheme=self.params["convection_scheme"],
                solver=self.params["linear_solver"],
                under_relaxation=alpha_u,
                dt=dt, U_old=U_old, U_old_old=U_old_old, ddt_scheme=ddt_scheme,
                source_explicit=source_explicit, source_implicit=source_implicit,
                reuse_ilu=self.params.get("reuse_ilu", False),
                ilu_key=self.params.get("ilu_key", None),
                ilu_drop_tol=self.params.get("ilu_drop_tol", 1e-4),
                ilu_fill_factor=self.params.get("ilu_fill_factor", 10),
                momentum_tol=self.params.get("momentum_tol", 1e-4),
                ilu_reuse_tol=self.params.get("ilu_reuse_tol", None),
            )
            logging.Timer.log("    Momentum Predictor")

            U_iter = U_star.copy()

            # ---- 1b. PISO-like Inner Corrector Loop ----
            for _corr in range(n_corr):
                logging.Timer.start("    Pressure Correction")
                A_p, b_p, phi_star = simple_solver.assemble_pressure_correction_equation_rhie_chow(
                    U_iter, A_U, p, rho,
                    self.mesh_data, self.geo_data, self.boundaries,
                    alpha_u=alpha_u,
                )
                logging.Timer.log("    Pressure Assembly")

                logging.Timer.start("    Pressure Solve")
                p_prime = matrix_assembly.solve_linear_system(
                    A_p, b_p,
                    method=self.params["linear_solver"],
                    equation_type="pressure",
                    amg_tol=self.params.get("amg_tol", 4e-4),
                    amg_maxiter=self.params.get("amg_maxiter", 100),
                )
                logging.Timer.log("    Pressure Solve")

                # Note: fix_pressure_reference() is applied inside the assembly,
                # pinning p'[ref_cell]=0 via matrix modification (OpenFOAM setReference).

                logging.Timer.start("    Velocity Correction")
                U_iter, phi = simple_solver.correct_velocity_and_flux(
                    U_iter, phi_star, p_prime, A_U,
                    self.mesh_data, self.geo_data, self.boundaries,
                    rho=rho, alpha_u=alpha_u,
                )

                # Velocity clipping to prevent divergence in transient steps
                # Optional hard velocity cap.  Disabled by default: clipping
                # silently masks divergence, which the continuity diagnostic and
                # the finite-value guard now surface instead.  Set
                # solver.max_velocity_clip to re-enable as a last resort.
                cap = self.params.get("max_velocity_clip", None)
                if cap is not None:
                    np.clip(U_iter[:n_elem], -float(cap), float(cap), out=U_iter[:n_elem])

                logging.Timer.log("    Velocity Correction")

                # Update pressure
                p[:n_elem] += alpha_p * p_prime
                simple_solver.update_scalar_boundaries(p, self.mesh_data, self.boundaries, field_name="p")

            # ---- 1c. Non-Orthogonal Corrector Loop ----
            n_ortho = int(self.params.get("n_orthogonal_correctors", 0))
            for _ in range(n_ortho):
                # Re-evaluate the pressure equation with updated p for
                # explicit non-orthogonal correction.  The implicit matrix
                # (orthogonal part) stays the same; the RHS changes.
                _, b_p_new, _ = simple_solver.assemble_pressure_correction_equation_rhie_chow(
                    U_iter, A_U, p, rho,
                    self.mesh_data, self.geo_data, self.boundaries,
                    alpha_u=alpha_u,
                )
                # Solve with the same matrix A_p but updated RHS
                p_prime = matrix_assembly.solve_linear_system(
                    A_p, b_p_new,
                    method=self.params["linear_solver"],
                    equation_type="pressure",
                    amg_tol=self.params.get("amg_tol", 4e-4),
                    amg_maxiter=self.params.get("amg_maxiter", 100),
                )
                U_iter, phi = simple_solver.correct_velocity_and_flux(
                    U_iter, phi_star, p_prime, A_U,
                    self.mesh_data, self.geo_data, self.boundaries,
                    rho=rho, alpha_u=alpha_u,
                )
                # Optional hard velocity cap.  Disabled by default: clipping
                # silently masks divergence, which the continuity diagnostic and
                # the finite-value guard now surface instead.  Set
                # solver.max_velocity_clip to re-enable as a last resort.
                cap = self.params.get("max_velocity_clip", None)
                if cap is not None:
                    np.clip(U_iter[:n_elem], -float(cap), float(cap), out=U_iter[:n_elem])
                p[:n_elem] += alpha_p * p_prime
                simple_solver.update_scalar_boundaries(p, self.mesh_data, self.boundaries, field_name="p")

            # ---- 1d. Update U boundary conditions and copy back ----
            simple_solver._update_velocity_bcs(
                U_iter, phi, self.boundaries,
                self.mesh_data["owners"], self.geo_data,
                n_elem, self.mesh_data["n_interior_faces"],
            )
            U[:n_elem] = U_iter[:n_elem]

            # ---- 1e. Under-relaxation between outer iterations ----
            if outer < n_outer - 1:
                U[:n_elem] = U_before + alpha_u * (U[:n_elem] - U_before)
                p[:n_elem] = p_before + alpha_p * (p[:n_elem] - p_before)
                simple_solver.update_scalar_boundaries(p, self.mesh_data, self.boundaries, field_name="p")

        # 2. Residual computation (last corrector)
        residual_p = np.linalg.norm(b_p) / (np.linalg.norm(p[:n_elem]) + 1e-10)
        residual_u = np.linalg.norm(U[:n_elem] - U_old[:n_elem]) / (np.linalg.norm(U[:n_elem]) + 1e-10)

        return U, p, phi, {"p": residual_p, "U": residual_u}

    def solve(
        self,
        U_init,
        p_init,
        rho=1.0,
        nu=0.01,
        dt=None,
        n_steps=None,
        write_interval=None,
        case_dir=None,
    ):
        """
        Solve transient flow using PIMPLE (backward compatible loop).
        """
        assert dt is not None, "PIMPLESolver.solve requires dt"
        assert n_steps is not None, "PIMPLESolver.solve requires n_steps"
        U = U_init.copy()
        p = p_init.copy()
        U_old = U.copy()

        import os

        current_time = 0.0

        print("\nPIMPLE Solver Started")
        print(f"  Time step: {dt}")
        print(f"  Total steps: {n_steps}")
        print(f"  Correctors: {self.params['n_correctors']}")

        # Initialize phi
        from ..assemble import convection

        phi = convection.compute_mass_flow_rate(U, self.mesh_data, self.geo_data)

        for step in range(1, n_steps + 1):
            current_time += dt
            print(f"\nTime: {current_time:.5f} (Step {step}/{n_steps})")

            U_old[:] = U[:]
            U, p, phi, residuals = self.step(U, p, phi, U_old, dt, rho, nu)

            # Write Output
            if write_interval and step % write_interval == 0 and case_dir:
                time_dir = f"{current_time:.5g}"
                output_dir = os.path.join(case_dir, time_dir)
                os.makedirs(output_dir, exist_ok=True)

                print(f"  Writing results to {output_dir}")

                U_data = {"name": "U", "type": "volVectorField", "phi": U}
                p_data = {"name": "p", "type": "volScalarField", "phi": p}

                field_io.write_foam_field(os.path.join(output_dir, "U"), self.mesh_data, U_data)
                field_io.write_foam_field(os.path.join(output_dir, "p"), self.mesh_data, p_data)

        return U, p
