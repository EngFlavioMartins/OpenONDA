#!/usr/bin/env python3
"""
Time Integration for OpenONDA FVM Solver

Implements transient term discretization:
- Euler implicit (first-order, unconditionally stable)
- Euler explicit (first-order, conditionally stable)

Converted from uFVM cfdAssembleTransientTermEuler.m
"""

import numpy as np


def assemble_transient_term_euler_implicit(phi, phi_old, dt, rho, mesh_data, geo_data):
    """
    Assemble transient term using Euler implicit scheme.

    ∂(ρφ)/∂t ≈ (ρφ^{n+1} - ρφ^n) / Δt

    Implicit: φ^{n+1} appears in coefficient matrix (unconditionally stable)

    Args:
        phi: Current field values (n_elements + n_boundary,)
        phi_old: Previous time step values (n_elements,)
        dt: Time step size
        rho: Density (n_elements,)
        mesh_data: Mesh connectivity
        geo_data: Geometric data

    Returns:
        dict: Transient term contribution
            - ac: Diagonal coefficients (n_elements,)
            - bc: RHS contribution (n_elements,)
    """

    mesh_data["n_elements"]
    volumes = geo_data["element_volumes"]

    # Coefficient: ρV/Δt
    ac = rho * volumes / dt

    # RHS: (ρV/Δt) * φ_old
    bc = ac * phi_old

    return {"ac": ac, "bc": bc}


def assemble_transient_term_euler_explicit(phi_old, dt, rho, mesh_data, geo_data):
    """
    Assemble transient term using Euler explicit scheme.

    ∂(ρφ)/∂t ≈ (ρφ^{n+1} - ρφ^n) / Δt

    Explicit: φ^{n+1} does NOT appear in matrix (conditionally stable)
    All contribution goes to RHS.

    Args:
        phi_old: Previous time step values
        dt: Time step
        rho: Density
        mesh_data: Mesh connectivity
        geo_data: Geometric data

    Returns:
        dict: Transient term contribution
    """

    n_elements = mesh_data["n_elements"]
    volumes = geo_data["element_volumes"]

    # No diagonal contribution for explicit
    ac = np.zeros(n_elements)

    # RHS: ρV*φ_old/Δt
    bc = rho * volumes * phi_old / dt

    return {"ac": ac, "bc": bc}


def compute_time_step_cfl(velocity, mesh_data, geo_data, cfl_max=0.5):
    """
    Compute time step based on CFL condition.

    CFL = |U| * Δt / Δx ≤ CFL_max

    Δt = CFL_max * Δx / |U|_max

    Args:
        velocity: Velocity field (n_elements + n_boundary, 3)
        mesh_data: Mesh connectivity
        geo_data: Geometric data
        cfl_max: Maximum CFL number

    Returns:
        float: Time step size
    """

    n_elements = mesh_data["n_elements"]

    # Compute characteristic length: cube root of volume
    volumes = geo_data["element_volumes"]
    char_length = np.cbrt(volumes)

    # Compute velocity magnitude
    vel_mag = np.linalg.norm(velocity[:n_elements], axis=1)

    # Avoid division by zero
    vel_mag = np.maximum(vel_mag, 1e-10)

    # Local time step for each element
    dt_local = cfl_max * char_length / vel_mag

    # Global time step: minimum of all local time steps
    dt = np.min(dt_local)

    return dt


def compute_time_step_diffusion(gamma, rho, mesh_data, geo_data, fo_max=0.5):
    """
    Compute time step based on diffusion (Fourier number) constraint.

    Fo = γ * Δt / (ρ * Δx²) ≤ Fo_max

    Δt = Fo_max * ρ * Δx² / γ

    Args:
        gamma: Diffusion coefficient (n_elements,)
        rho: Density (n_elements,)
        mesh_data: Mesh connectivity
        geo_data: Geometric data
        fo_max: Maximum Fourier number

    Returns:
        float: Time step size
    """

    volumes = geo_data["element_volumes"]
    char_length = np.cbrt(volumes)

    # Avoid division by zero
    gamma = np.maximum(gamma, 1e-10)

    # Local time step
    dt_local = fo_max * rho * char_length**2 / gamma

    # Global time step
    dt = np.min(dt_local)

    return dt


def compute_adaptive_time_step(velocity, gamma, rho, mesh_data, geo_data, cfl_max=0.5, fo_max=0.5):
    """
    Compute adaptive time step considering both convection and diffusion.

    Args:
        velocity: Velocity field
        gamma: Diffusion coefficient
        rho: Density
        mesh_data: Mesh connectivity
        geo_data: Geometric data
        cfl_max: Maximum CFL number
        fo_max: Maximum Fourier number

    Returns:
        float: Time step size
    """

    # CFL constraint
    dt_cfl = compute_time_step_cfl(velocity, mesh_data, geo_data, cfl_max)

    # Diffusion constraint
    dt_diff = compute_time_step_diffusion(gamma, rho, mesh_data, geo_data, fo_max)

    # Take minimum
    dt = min(dt_cfl, dt_diff)

    return dt


class TimeIntegrator:
    """
    Time integration manager for transient simulations.
    """

    def __init__(self, dt, scheme="euler_implicit"):
        """
        Initialize time integrator.

        Args:
            dt: Time step size (can be 'auto' for adaptive)
            scheme: 'euler_implicit' or 'euler_explicit'
        """
        self.dt = dt
        self.scheme = scheme
        self.time = 0.0
        self.iteration = 0
        self.fields_old = {}

    def store_old_fields(self, **fields):
        """Store field values from previous time step.

        Saves deep copies of the provided fields so they can be used
        in transient term assembly at the next time step.

        Args:
            **fields: Named field arrays to store. Each value must be a
                numpy array that supports ``.copy()``.
        """
        self.fields_old = {name: field.copy() for name, field in fields.items()}

    def advance_time(self, dt=None):
        """Advance to the next time step.

        Increments the internal time and iteration counters.

        Args:
            dt: Time step increment. If ``None``, uses the default
                time step stored at initialisation.

        Returns:
            float: Updated simulation time after advancing.
        """
        if dt is None:
            dt = self.dt
        self.time += dt
        self.iteration += 1
        return self.time

    def get_transient_contribution(self, field_name, phi, rho, mesh_data, geo_data):
        """
        Get transient term contribution for a field.

        Args:
            field_name: Name of field
            phi: Current field values
            rho: Density
            mesh_data: Mesh connectivity
            geo_data: Geometric data

        Returns:
            dict: Transient contribution (ac, bc)
        """

        if field_name not in self.fields_old:
            # First time step: no transient term
            n_elements = mesh_data["n_elements"]
            return {"ac": np.zeros(n_elements), "bc": np.zeros(n_elements)}

        phi_old = self.fields_old[field_name]

        if self.scheme == "euler_implicit":
            return assemble_transient_term_euler_implicit(
                phi, phi_old, self.dt, rho, mesh_data, geo_data
            )
        elif self.scheme == "euler_explicit":
            return assemble_transient_term_euler_explicit(
                phi_old, self.dt, rho, mesh_data, geo_data
            )
        else:
            raise ValueError(f"Unknown scheme: {self.scheme}")
