#!/usr/bin/env python3
"""
Complete Equation Solver for OpenONDA FVM

Integrates all discretization terms:
- Diffusion
- Convection
- Transient
- Source terms

Solves: ∂(ρφ)/∂t + ∇·(ρUφ) = ∇·(γ∇φ) + S
"""

import numpy as np

from ..assemble import convection, diffusion, matrix_assembly, time_integration
from ..fields import gradients
from .simple_solver import update_scalar_boundaries


class ScalarEquationSolver:
    """
    Solver for scalar transport equations.

    Handles complete assembly and solution of:
    ∂(ρφ)/∂t + ∇·(ρUφ) = ∇·(γ∇φ) + S
    """

    def __init__(self, mesh_data, geo_data, boundaries):
        """
        Initialize solver.

        Args:
            mesh_data: Mesh connectivity
            geo_data: Geometric data
            boundaries: Boundary conditions
        """
        self.mesh_data = mesh_data
        self.geo_data = geo_data
        self.boundaries = boundaries
        self.n_elements = mesh_data["n_elements"]
        self._grad_fn = gradients._resolve_gradient_fn(geo_data)

    def _mass_flow_rate(self, velocity, density):
        """Return face mass flux using linearly interpolated density."""
        density = np.asarray(density, dtype=np.float64)
        if density.ndim == 0:
            density = np.full(self.n_elements, float(density))
        if density.ndim != 1 or len(density) < self.n_elements:
            raise ValueError(
                f"density must be scalar or contain at least {self.n_elements} cell values"
            )
        if not np.all(np.isfinite(density[: self.n_elements])) or np.any(
            density[: self.n_elements] <= 0.0
        ):
            raise ValueError("density values must be finite and positive")

        volumetric_flux = convection.compute_mass_flow_rate(velocity, self.mesh_data, self.geo_data)
        owners = self.mesh_data["owners"]
        neighbours = self.mesh_data["neighbours"]
        n_interior = self.mesh_data["n_interior_faces"]
        face_density = density[owners].copy()
        weights = self.geo_data["face_weights"][:n_interior]
        face_density[:n_interior] = (
            weights * density[neighbours[:n_interior]]
            + (1.0 - weights) * density[owners[:n_interior]]
        )
        return volumetric_flux * face_density

    def solve_steady_diffusion(self, phi_initial, gamma, solver="spsolve", **kwargs):
        """
        Solve steady diffusion equation: ∇·(γ∇φ) = 0

        Args:
            phi_initial: Initial field (n_elements + n_boundary,)
            gamma: Diffusion coefficient
            solver: Linear solver method
            **kwargs: Solver options

        Returns:
            numpy.ndarray: Solution (n_elements + n_boundary,)
        """

        # Ensure gamma is array
        if np.isscalar(gamma):
            gamma = np.full(self.n_elements, gamma)

        # Compute gradient
        grad_phi = self._grad_fn(phi_initial, self.mesh_data, self.geo_data)

        # Assemble diffusion term
        flux_data = diffusion.assemble_diffusion_term(
            phi_initial, grad_phi, gamma, self.mesh_data, self.geo_data, self.boundaries
        )

        # Assemble matrix and RHS
        A = matrix_assembly.assemble_matrix_from_fluxes_vectorized(flux_data, self.mesh_data)
        b = matrix_assembly.assemble_rhs_from_fluxes_vectorized(flux_data, self.mesh_data)

        # Solve for interior cells only
        phi_interior = matrix_assembly.solve_linear_system(
            A, b, method=solver, equation_type="scalar", tol=1e-6, **kwargs
        )

        # Combine with boundary values
        n_boundary = len(phi_initial) - self.n_elements
        phi_solution = np.zeros(self.n_elements + n_boundary)
        phi_solution[: self.n_elements] = phi_interior
        phi_solution[self.n_elements :] = phi_initial[self.n_elements :]  # Preserve BCs

        return phi_solution

    def solve_steady_advection_diffusion(
        self,
        phi_initial,
        velocity,
        gamma,
        density=1.0,
        convection_scheme="deferred",
        solver="spsolve",
        **kwargs,
    ):
        """
        Solve steady advection-diffusion: ∇·(ρUφ) = ∇·(γ∇φ)

        Args:
            phi_initial: Initial field
            velocity: Velocity field
            gamma: Diffusion coefficient
            density: Density (scalar or array)
            convection_scheme: 'upwind', 'central', or 'deferred'
            solver: Linear solver
            **kwargs: Solver options

        Returns:
            numpy.ndarray: Solution
        """

        # Compute gradient
        grad_phi = self._grad_fn(phi_initial, self.mesh_data, self.geo_data)

        # Compute mass flow rate
        mdot = self._mass_flow_rate(velocity, density)

        # Assemble diffusion
        diff_flux = diffusion.assemble_diffusion_term(
            phi_initial, grad_phi, gamma, self.mesh_data, self.geo_data, self.boundaries
        )

        # Assemble convection
        conv_flux = convection.assemble_convection_term(
            phi_initial,
            mdot,
            self.mesh_data,
            self.geo_data,
            self.boundaries,
            scheme=convection_scheme,
        )

        # Combine fluxes
        combined_flux = {
            "flux_cf": diff_flux["flux_cf"] + conv_flux["flux_cf"],
            "flux_ff": diff_flux["flux_ff"] + conv_flux["flux_ff"],
            "flux_vf": diff_flux["flux_vf"] + conv_flux["flux_vf"],
            "flux_tf": diff_flux["flux_tf"] + conv_flux["flux_tf"],
        }

        # Assemble and solve
        A = matrix_assembly.assemble_matrix_from_fluxes_vectorized(combined_flux, self.mesh_data)
        b = matrix_assembly.assemble_rhs_from_fluxes_vectorized(combined_flux, self.mesh_data)

        phi_interior = matrix_assembly.solve_linear_system(
            A, b, method=solver, equation_type="scalar", tol=1e-6, **kwargs
        )
        phi_solution = np.asarray(phi_initial, dtype=np.float64).copy()
        phi_solution[: self.n_elements] = phi_interior
        update_scalar_boundaries(phi_solution, self.mesh_data, self.boundaries, field_name="phi")
        return phi_solution

    def solve_transient_diffusion(
        self,
        phi_initial,
        gamma,
        density,
        dt,
        n_steps,
        time_scheme="euler_implicit",
        solver="spsolve",
        **kwargs,
    ):
        """
        Solve transient diffusion: ∂(ρφ)/∂t = ∇·(γ∇φ)

        Args:
            phi_initial: Initial field (n_elements + n_boundary,)
            gamma: Diffusion coefficient
            density: Density
            dt: Time step
            n_steps: Number of time steps
            time_scheme: 'euler_implicit' or 'euler_explicit'
            solver: Linear solver
            **kwargs: Solver options

        Returns:
            list: Solution at each time step (each with n_elements + n_boundary,)
        """

        # Ensure arrays
        if np.isscalar(density):
            density = np.full(self.n_elements, density)
        if np.isscalar(gamma):
            gamma = np.full(self.n_elements, gamma)

        # Initialize time integrator
        integrator = time_integration.TimeIntegrator(dt, scheme=time_scheme)

        # Storage
        solutions = [phi_initial.copy()]  # Store full field including boundaries
        phi = phi_initial.copy()

        for _step in range(n_steps):
            # Store old field
            integrator.store_old_fields(phi=phi[: self.n_elements])

            # Compute gradient
            grad_phi = self._grad_fn(phi, self.mesh_data, self.geo_data)

            # Assemble diffusion
            diff_flux = diffusion.assemble_diffusion_term(
                phi, grad_phi, gamma, self.mesh_data, self.geo_data, self.boundaries
            )

            # Assemble transient term
            trans_contrib = integrator.get_transient_contribution(
                "phi", phi, density, self.mesh_data, self.geo_data
            )

            # Build matrix: A = A_diff + A_trans
            A_diff = matrix_assembly.assemble_matrix_from_fluxes_vectorized(
                diff_flux, self.mesh_data
            )

            # Add transient diagonal contribution (in-place on CSR)
            A_diff.setdiag(A_diff.diagonal() + trans_contrib["ac"])
            A = A_diff

            # Build RHS: b = b_diff + b_trans
            b_diff = matrix_assembly.assemble_rhs_from_fluxes_vectorized(diff_flux, self.mesh_data)
            b = b_diff + trans_contrib["bc"]

            # Solve for interior cells
            phi_new_interior = matrix_assembly.solve_linear_system(
                A, b, method=solver, equation_type="scalar", tol=1e-6, **kwargs
            )
            phi[: self.n_elements] = phi_new_interior
            update_scalar_boundaries(phi, self.mesh_data, self.boundaries, field_name="phi")

            # Advance time
            integrator.advance_time()

            # Store full solution including boundaries
            solutions.append(phi.copy())

        return solutions

    def solve_transient_advection_diffusion(
        self,
        phi_initial,
        velocity,
        gamma,
        density,
        dt,
        n_steps,
        convection_scheme="deferred",
        time_scheme="euler_implicit",
        solver="spsolve",
        **kwargs,
    ):
        """Solve transient advection-diffusion over multiple time steps.

        Integrates the full transport equation:

        .. math:: \\frac{\\partial(\\rho\\phi)}{\\partial t}
                  + \\nabla\\cdot(\\rho U\\phi)
                  = \\nabla\\cdot(\\gamma\\nabla\\phi)

        Args:
            phi_initial: Initial scalar field
                (n_elements + n_boundary,).
            velocity: Velocity field (n_elements + n_boundary, 3).
            gamma: Diffusion coefficient (scalar or
                n_elements array).
            density: Density (scalar or n_elements array).
            dt: Time step size.
            n_steps: Number of time steps to advance.
            convection_scheme: Convection discretisation scheme
                (``'upwind'``, ``'central'``, ``'deferred'``).
                Defaults to ``'deferred'``.
            time_scheme: Time integration scheme
                (``'euler_implicit'`` or ``'euler_explicit'``).
                Defaults to ``'euler_implicit'``.
            solver: Linear solver method. Defaults to ``'spsolve'``.
            **kwargs: Additional solver keyword arguments.

        Returns:
            list: Solution at each time step (including initial), each
                entry is a numpy array of shape
                (n_elements + n_boundary,).
        """

        # Ensure arrays
        if np.isscalar(density):
            density = np.full(self.n_elements, density)
        if np.isscalar(gamma):
            gamma = np.full(self.n_elements, gamma)

        # Initialize time integrator
        integrator = time_integration.TimeIntegrator(dt, scheme=time_scheme)

        # Storage
        solutions = [phi_initial.copy()]
        phi = phi_initial.copy()

        # Mass flow rate (assuming steady flow for now)
        mdot = self._mass_flow_rate(velocity, density)

        for _step in range(n_steps):
            # Store old field
            integrator.store_old_fields(phi=phi[: self.n_elements])

            # Compute gradient
            grad_phi = self._grad_fn(phi, self.mesh_data, self.geo_data)

            # Assemble diffusion
            diff_flux = diffusion.assemble_diffusion_term(
                phi, grad_phi, gamma, self.mesh_data, self.geo_data, self.boundaries
            )

            # Assemble convection
            conv_flux = convection.assemble_convection_term(
                phi, mdot, self.mesh_data, self.geo_data, self.boundaries, scheme=convection_scheme
            )

            # Assemble transient term
            trans_contrib = integrator.get_transient_contribution(
                "phi", phi, density, self.mesh_data, self.geo_data
            )

            # Combine fluxes
            combined_flux = {
                "flux_cf": diff_flux["flux_cf"] + conv_flux["flux_cf"],
                "flux_ff": diff_flux["flux_ff"] + conv_flux["flux_ff"],
                "flux_vf": diff_flux["flux_vf"] + conv_flux["flux_vf"],
                "flux_tf": diff_flux["flux_tf"] + conv_flux["flux_tf"],
            }

            # Build Matrix and RHS
            A_combined = matrix_assembly.assemble_matrix_from_fluxes_vectorized(
                combined_flux, self.mesh_data
            )
            b_combined = matrix_assembly.assemble_rhs_from_fluxes_vectorized(
                combined_flux, self.mesh_data
            )

            # Add transient contributions (in-place on CSR)
            A_combined.setdiag(A_combined.diagonal() + trans_contrib["ac"])
            A = A_combined

            b = b_combined + trans_contrib["bc"]

            # Solve
            phi_new_interior = matrix_assembly.solve_linear_system(
                A, b, method=solver, equation_type="scalar", tol=1e-6, **kwargs
            )
            phi[: self.n_elements] = phi_new_interior
            update_scalar_boundaries(phi, self.mesh_data, self.boundaries, field_name="phi")

            # Advance
            integrator.advance_time()
            solutions.append(phi.copy())

        return solutions


def solve_scalar_equation(equation_config, mesh_data, geo_data, boundaries):
    """
    High-level interface to solve scalar transport equation.

    Args:
        equation_config: Dict with equation parameters:
            - 'type': 'steady' or 'transient'
            - 'terms': list of terms ['diffusion', 'convection', 'transient']
            - 'phi_initial': initial field
            - 'gamma': diffusion coefficient
            - 'velocity': velocity field (if convection)
            - 'density': density
            - 'dt': time step (if transient)
            - 'n_steps': number of steps (if transient)
            - 'solver': linear solver method
        mesh_data: Mesh connectivity
        geo_data: Geometric data
        boundaries: Boundary conditions

    Returns:
        Solution (array or list of arrays for transient)
    """

    solver = ScalarEquationSolver(mesh_data, geo_data, boundaries)

    eq_type = equation_config.get("type", "steady")
    terms = equation_config.get("terms", ["diffusion"])
    linear_options = equation_config.get("linear_options", {})
    if not isinstance(linear_options, dict):
        raise TypeError("linear_options must be a dictionary")

    if eq_type == "steady":
        if "convection" in terms:
            return solver.solve_steady_advection_diffusion(
                equation_config["phi_initial"],
                equation_config["velocity"],
                equation_config["gamma"],
                density=equation_config.get("density", 1.0),
                convection_scheme=equation_config.get("convection_scheme", "deferred"),
                solver=equation_config.get("solver", "spsolve"),
                **linear_options,
            )
        else:
            return solver.solve_steady_diffusion(
                equation_config["phi_initial"],
                equation_config["gamma"],
                solver=equation_config.get("solver", "spsolve"),
                **linear_options,
            )

    elif eq_type == "transient":
        if "convection" in terms:
            return solver.solve_transient_advection_diffusion(
                equation_config["phi_initial"],
                equation_config["velocity"],
                equation_config["gamma"],
                equation_config["density"],
                equation_config["dt"],
                equation_config["n_steps"],
                convection_scheme=equation_config.get("convection_scheme", "deferred"),
                time_scheme=equation_config.get("time_scheme", "euler_implicit"),
                solver=equation_config.get("solver", "spsolve"),
                **linear_options,
            )
        else:
            return solver.solve_transient_diffusion(
                equation_config["phi_initial"],
                equation_config["gamma"],
                equation_config["density"],
                equation_config["dt"],
                equation_config["n_steps"],
                time_scheme=equation_config.get("time_scheme", "euler_implicit"),
                solver=equation_config.get("solver", "spsolve"),
                **linear_options,
            )

    else:
        raise ValueError(f"Unknown equation type: {eq_type}")
