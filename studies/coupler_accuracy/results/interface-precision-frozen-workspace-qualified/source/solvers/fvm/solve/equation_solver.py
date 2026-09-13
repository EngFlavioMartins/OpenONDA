"""Solve finite-volume scalar transport equations."""

import numpy as np

from ..assemble import convection, diffusion, matrix_assembly, time_integration
from ..fields import gradients
from .linear_interface import solve_linear_system
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
        self.n_cells = mesh_data["n_cells"]
        self._grad_fn = gradients._resolve_gradient_fn(geo_data)

    def _advance_transient_step(
        self,
        scalar_field_old,
        spatial_matrix,
        spatial_rhs,
        density,
        time_step_size,
        time_scheme,
        solver,
        linear_options,
    ):
        """Advance one scalar step from the assembled steady spatial balance."""
        if time_scheme == "euler_explicit":
            return time_integration.advance_euler_explicit(
                scalar_field_old,
                spatial_matrix,
                spatial_rhs,
                time_step_size,
                density,
                self.geo_data["cell_volume"],
            )
        if time_scheme != "euler_implicit":
            raise ValueError(f"Unknown scalar time scheme: {time_scheme}")

        transient = time_integration.assemble_transient_term_euler_implicit(
            scalar_field_old, time_step_size, density, self.geo_data
        )
        spatial_matrix.setdiag(spatial_matrix.diagonal() + transient["ac"])
        rhs = spatial_rhs + transient["bc"]
        return solve_linear_system(
            spatial_matrix,
            rhs,
            method=solver,
            equation_type="scalar",
            tol=1e-6,
            **linear_options,
        )

    def _compute_mass_flux(self, velocity, density):
        """Return face mass flux using linearly interpolated density."""
        density = np.asarray(density, dtype=np.float64)
        if density.ndim == 0:
            density = np.full(self.n_cells, float(density))
        if density.ndim != 1 or len(density) < self.n_cells:
            raise ValueError(
                f"density must be scalar or contain at least {self.n_cells} cell values"
            )
        if not np.all(np.isfinite(density[: self.n_cells])) or np.any(
            density[: self.n_cells] <= 0.0
        ):
            raise ValueError("density values must be finite and positive")

        volumetric_face_flux = convection.compute_volumetric_face_flux(
            velocity, self.mesh_data, self.geo_data
        )
        owners = self.mesh_data["owners"]
        neighbours = self.mesh_data["neighbours"]
        n_interior = self.mesh_data["n_interior_faces"]
        face_density = density[owners].copy()
        weights = self.geo_data["face_interpolation_weight"][:n_interior]
        face_density[:n_interior] = (
            weights * density[neighbours[:n_interior]]
            + (1.0 - weights) * density[owners[:n_interior]]
        )
        return volumetric_face_flux * face_density

    def solve_steady_diffusion(self, scalar_field_initial, diffusivity, solver="spsolve", **kwargs):
        """
        Solve steady diffusion equation: ∇·(γ∇scalar_field) = 0

        Args:
            scalar_field_initial: Initial field (n_elements + n_boundary,)
            diffusivity: Diffusion coefficient
            solver: Linear solver method
            **kwargs: Solver options

        Returns:
            numpy.ndarray: Solution (n_elements + n_boundary,)
        """

        # Ensure diffusivity is array
        if np.isscalar(diffusivity):
            diffusivity = np.full(self.n_cells, diffusivity)

        # Compute gradient
        scalar_field_gradient = self._grad_fn(scalar_field_initial, self.mesh_data, self.geo_data)

        # Assemble diffusion term
        flux_data = diffusion.assemble_diffusion_term(
            scalar_field_initial,
            scalar_field_gradient,
            diffusivity,
            self.mesh_data,
            self.geo_data,
            self.boundaries,
        )

        # Assemble matrix and RHS
        A = matrix_assembly.assemble_matrix_from_fluxes_vectorized(flux_data, self.mesh_data)
        b = matrix_assembly.assemble_rhs_from_fluxes_vectorized(flux_data, self.mesh_data)

        # Solve for interior cells only
        scalar_field_interior = solve_linear_system(
            A, b, method=solver, equation_type="scalar", tol=1e-6, **kwargs
        )

        # Combine with boundary values
        n_boundary = len(scalar_field_initial) - self.n_cells
        scalar_field_solution = np.zeros(self.n_cells + n_boundary)
        scalar_field_solution[: self.n_cells] = scalar_field_interior
        scalar_field_solution[self.n_cells :] = scalar_field_initial[self.n_cells :]  # Preserve BCs

        return scalar_field_solution

    def solve_steady_advection_diffusion(
        self,
        scalar_field_initial,
        velocity,
        diffusivity,
        density=1.0,
        convection_scheme="deferred",
        solver="spsolve",
        **kwargs,
    ):
        """
        Solve steady advection-diffusion: ∇·(density·velocity·scalar_field) =
        ∇·(γ∇scalar_field)

        Args:
            scalar_field_initial: Initial field
            velocity: Velocity field
            diffusivity: Diffusion coefficient
            density: Density (scalar or array)
            convection_scheme: 'upwind', 'central', or 'deferred'
            solver: Linear solver
            **kwargs: Solver options

        Returns:
            numpy.ndarray: Solution
        """

        # Compute gradient
        scalar_field_gradient = self._grad_fn(scalar_field_initial, self.mesh_data, self.geo_data)

        # Compute mass flow rate
        mass_flux = self._compute_mass_flux(velocity, density)

        # Assemble diffusion
        diff_flux = diffusion.assemble_diffusion_term(
            scalar_field_initial,
            scalar_field_gradient,
            diffusivity,
            self.mesh_data,
            self.geo_data,
            self.boundaries,
        )

        # Assemble convection
        conv_flux = convection.assemble_convection_term(
            scalar_field_initial,
            mass_flux,
            self.mesh_data,
            self.geo_data,
            self.boundaries,
            scheme=convection_scheme,
            scalar_field_gradient=scalar_field_gradient,
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

        scalar_field_interior = solve_linear_system(
            A, b, method=solver, equation_type="scalar", tol=1e-6, **kwargs
        )
        scalar_field_solution = np.asarray(scalar_field_initial, dtype=np.float64).copy()
        scalar_field_solution[: self.n_cells] = scalar_field_interior
        update_scalar_boundaries(
            scalar_field_solution,
            self.mesh_data,
            self.boundaries,
            field_name="scalar_field",
        )
        return scalar_field_solution

    def solve_transient_diffusion(
        self,
        scalar_field_initial,
        diffusivity,
        density,
        time_step_size,
        n_steps,
        time_scheme="euler_implicit",
        solver="spsolve",
        **kwargs,
    ):
        """
        Solve transient diffusion: ∂(density·scalar_field)/∂t =
        ∇·(γ∇scalar_field)

        Args:
            scalar_field_initial: Initial field (n_elements + n_boundary,)
            diffusivity: Diffusion coefficient
            density: Density
            time_step_size: Time step
            n_steps: Number of time steps
            time_scheme: 'euler_implicit' or 'euler_explicit'
            solver: Linear solver
            **kwargs: Solver options

        Returns:
            list: Solution at each time step (each with n_elements + n_boundary,)
        """

        # Ensure arrays
        if np.isscalar(density):
            density = np.full(self.n_cells, density, dtype=np.float64)
        if np.isscalar(diffusivity):
            diffusivity = np.full(self.n_cells, diffusivity, dtype=np.float64)

        # Storage
        solutions = [scalar_field_initial.copy()]  # Store full field including boundaries
        scalar_field = scalar_field_initial.copy()

        for _step in range(n_steps):
            scalar_field_old = scalar_field[: self.n_cells].copy()

            # Compute gradient
            scalar_field_gradient = self._grad_fn(scalar_field, self.mesh_data, self.geo_data)

            # Assemble diffusion
            diff_flux = diffusion.assemble_diffusion_term(
                scalar_field,
                scalar_field_gradient,
                diffusivity,
                self.mesh_data,
                self.geo_data,
                self.boundaries,
            )

            # Assemble the steady spatial balance A·scalar_field = b.  The selected time
            # scheme determines whether it is solved implicitly or used as the
            # residual in a forward-Euler update.
            A_diff = matrix_assembly.assemble_matrix_from_fluxes_vectorized(
                diff_flux, self.mesh_data
            )
            b_diff = matrix_assembly.assemble_rhs_from_fluxes_vectorized(diff_flux, self.mesh_data)
            scalar_field_new_interior = self._advance_transient_step(
                scalar_field_old,
                A_diff,
                b_diff,
                density,
                time_step_size,
                time_scheme,
                solver,
                kwargs,
            )
            scalar_field[: self.n_cells] = scalar_field_new_interior
            update_scalar_boundaries(
                scalar_field,
                self.mesh_data,
                self.boundaries,
                field_name="scalar_field",
            )

            # Store full solution including boundaries
            solutions.append(scalar_field.copy())

        return solutions

    def solve_transient_advection_diffusion(
        self,
        scalar_field_initial,
        velocity,
        diffusivity,
        density,
        time_step_size,
        n_steps,
        convection_scheme="deferred",
        time_scheme="euler_implicit",
        solver="spsolve",
        **kwargs,
    ):
        """Solve transient advection-diffusion over multiple time steps.

        Integrates the full transport equation:

        .. math:: \\frac{\\partial(\\mathrm{density}\\,\\mathrm{scalar_field})}{\\partial t}
                  + \\nabla\\cdot(\\mathrm{density}\\,\\mathrm{velocity}\\,\\mathrm{scalar_field})
                  = \\nabla\\cdot(\\mathrm{diffusivity}\\nabla\\mathrm{scalar_field})

        Args:
            scalar_field_initial: Initial scalar field
                (n_elements + n_boundary,).
            velocity: Velocity field (n_elements + n_boundary, 3).
            diffusivity: Diffusion coefficient (scalar or
                n_elements array).
            density: Density (scalar or n_elements array).
            time_step_size: Time step size.
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
            density = np.full(self.n_cells, density, dtype=np.float64)
        if np.isscalar(diffusivity):
            diffusivity = np.full(self.n_cells, diffusivity, dtype=np.float64)

        # Storage
        solutions = [scalar_field_initial.copy()]
        scalar_field = scalar_field_initial.copy()

        # Mass flow rate (assuming steady flow for now)
        mass_flux = self._compute_mass_flux(velocity, density)

        for _step in range(n_steps):
            scalar_field_old = scalar_field[: self.n_cells].copy()

            # Compute gradient
            scalar_field_gradient = self._grad_fn(scalar_field, self.mesh_data, self.geo_data)

            # Assemble diffusion
            diff_flux = diffusion.assemble_diffusion_term(
                scalar_field,
                scalar_field_gradient,
                diffusivity,
                self.mesh_data,
                self.geo_data,
                self.boundaries,
            )

            # Assemble convection
            conv_flux = convection.assemble_convection_term(
                scalar_field,
                mass_flux,
                self.mesh_data,
                self.geo_data,
                self.boundaries,
                scheme=convection_scheme,
                scalar_field_gradient=scalar_field_gradient,
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

            scalar_field_new_interior = self._advance_transient_step(
                scalar_field_old,
                A_combined,
                b_combined,
                density,
                time_step_size,
                time_scheme,
                solver,
                kwargs,
            )
            scalar_field[: self.n_cells] = scalar_field_new_interior
            update_scalar_boundaries(
                scalar_field,
                self.mesh_data,
                self.boundaries,
                field_name="scalar_field",
            )
            solutions.append(scalar_field.copy())

        return solutions


def solve_scalar_equation(equation_config, mesh_data, geo_data, boundaries):
    """
    High-level interface to solve scalar transport equation.

    Args:
        equation_config: Dict with equation parameters:
            - 'type': 'steady' or 'transient'
            - 'terms': list of terms ['diffusion', 'convection', 'transient']
            - 'scalar_field_initial': initial field
            - 'diffusivity': diffusion coefficient
            - 'velocity': velocity field (if convection)
            - 'density': density
            - 'time_step_size': time step (if transient)
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
                equation_config["scalar_field_initial"],
                equation_config["velocity"],
                equation_config["diffusivity"],
                density=equation_config.get("density", 1.0),
                convection_scheme=equation_config.get("convection_scheme", "deferred"),
                solver=equation_config.get("solver", "spsolve"),
                **linear_options,
            )
        else:
            return solver.solve_steady_diffusion(
                equation_config["scalar_field_initial"],
                equation_config["diffusivity"],
                solver=equation_config.get("solver", "spsolve"),
                **linear_options,
            )

    elif eq_type == "transient":
        if "convection" in terms:
            return solver.solve_transient_advection_diffusion(
                equation_config["scalar_field_initial"],
                equation_config["velocity"],
                equation_config["diffusivity"],
                equation_config["density"],
                equation_config["time_step_size"],
                equation_config["n_steps"],
                convection_scheme=equation_config.get("convection_scheme", "deferred"),
                time_scheme=equation_config.get("time_scheme", "euler_implicit"),
                solver=equation_config.get("solver", "spsolve"),
                **linear_options,
            )
        else:
            return solver.solve_transient_diffusion(
                equation_config["scalar_field_initial"],
                equation_config["diffusivity"],
                equation_config["density"],
                equation_config["time_step_size"],
                equation_config["n_steps"],
                time_scheme=equation_config.get("time_scheme", "euler_implicit"),
                solver=equation_config.get("solver", "spsolve"),
                **linear_options,
            )

    else:
        raise ValueError(f"Unknown equation type: {eq_type}")
