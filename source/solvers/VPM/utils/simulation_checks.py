"""
Pre-run sanity checks: particle statistics, per-scheme time-step estimates, and
time-step validation.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import numpy as np

from ..config.constants import EPSILON

def _compute_particle_statistics(system):
    """Extract particle statistics for time step validation."""
    positions = system.particles.position_cpu()
    velocities = system.particles.velocity_cpu()
    radii = system.particles.radius_cpu()
    viscosities = system.particles.viscosity_cpu()

    N = len(positions)

    # Particle spacing
    h_array = 2.0 * radii
    h_min, h_max, h_mean = float(np.min(h_array)), float(np.max(h_array)), float(np.mean(h_array))

    # Velocity
    u_mag = np.linalg.norm(velocities, axis=1)
    u_max = float(np.max(u_mag)) if N > 0 else 0.0
    u_mean = float(np.mean(u_mag)) if N > 0 else 0.0

    # Viscosity
    nu_molecular = float(np.max(viscosities)) if N > 0 else 1e-5

    # Turbulent viscosity
    nu_turbulent_max = 0.0
    if hasattr(system, "LES") and system.LES is not None:
        try:
            nu_t_array = system.particles.viscosity_turbulent_cpu()
            nu_turbulent_max = float(np.max(nu_t_array)) if len(nu_t_array) > 0 else 0.0
        except Exception:
            pass

    nu_eff_max = nu_molecular + nu_turbulent_max

    # Strain rate
    try:
        gradU = system.particles.velocity_gradient_cpu()
        grad_u_mag = np.linalg.norm(gradU.reshape(N, 9), axis=1)
        grad_u_max = float(np.max(grad_u_mag)) if N > 0 else 1e-5
    except Exception:
        grad_u_max = 1e-5

    return {
        "h_min": h_min,
        "h_max": h_max,
        "h_mean": h_mean,
        "u_max": u_max,
        "u_mean": u_mean,
        "nu_molecular": nu_molecular,
        "nu_turbulent_max": nu_turbulent_max,
        "nu_eff_max": nu_eff_max,
        "grad_u_max": grad_u_max,
    }

def _compute_scheme_timestep(scheme_name, stats, safety_factor, system_dt, use_mean_spacing=False):
    """Compute time step limits for a specific viscous scheme."""
    CFL_advection = 0.5
    C_diff = 0.125 if scheme_name == "CS" else 1.0  # RWM uses larger coefficient
    C_stretching = 1.0

    h = stats["h_mean"] if use_mean_spacing else stats["h_min"]

    # Advection limit
    dt_adv = CFL_advection * h / stats["u_max"] if stats["u_max"] > EPSILON else 1.0

    # Diffusion limit (None scheme doesn't have this)
    if scheme_name == "NONE":
        dt_diff = float("inf")
    else:
        dt_diff = C_diff * h**2 / stats["nu_eff_max"] if stats["nu_eff_max"] > EPSILON else 1.0

    # Stretching limit
    dt_stretch = C_stretching / stats["grad_u_max"] if stats["grad_u_max"] > EPSILON else 1.0

    # Find minimum
    components = [dt_adv, dt_diff, dt_stretch] if scheme_name != "NONE" else [dt_adv, dt_stretch]
    dt_limit = safety_factor * min(components)

    comp_names = (
        ["advection", "diffusion", "stretching"]
        if scheme_name != "NONE"
        else ["advection", "stretching"]
    )
    limiting_idx = np.argmin(components)

    result = {
        "dt_limit": dt_limit,
        "dt_recommended": dt_limit,
        "dt_adv_component": dt_adv,
        "dt_stretch_component": dt_stretch,
        "limiting_component": comp_names[limiting_idx],
        "status": "stable" if system_dt <= dt_limit else "WARNING: too large",
    }

    if scheme_name != "NONE":
        result["dt_diff_component"] = dt_diff

    return result

#TODO: is this method even in use anywhere? Haven't we moved this to the viscous schemes initialization? If so, delete this. Check if its used. If its still used, make sure its plugged and adopted by all viscous methods.
def _validate_time_step_sizing(system, safety_factor=0.8, verbose=True):
    """
    Check and recommend time-step sizing constraints for all viscous schemes.

    This method computes the maximum stable time step for each viscous scheme based on:
    - Advection CFL constraint: Δt_adv ≤ CFL * h / |u_max|
    - Viscous diffusion constraint: Δt_visc ≤ C_visc * h² / nu
    - RWM stochastic stability: Δt_rwm ≤ h² / (C_rwm * nu)
    - Stretching term constraint: Δt_stretch ≤ min(Δt / |∇u_max|)

    Must be called AFTER add_vortex_particles() to have particle data available.

    Args:
            safety_factor: Reduction factor applied to computed limits (0 < factor ≤ 1).
                        Default 0.8 is conservative; use 0.9-1.0 for less conservative.
            verbose: If True, print formatted summary to console. If False, return dict only.

    Returns:
            dict: Timestep recommendations with keys:
                - 'h_min': Minimum particle spacing [m]
                - 'h_max': Maximum particle spacing [m]
                - 'h_mean': Mean particle spacing [m]
                - 'u_max': Maximum particle velocity magnitude [m/s]
                - 'nu_molecular': Molecular kinematic viscosity [m²/s]
                - 'nu_turbulent_max': Maximum turbulent viscosity (if computed) [m²/s]
                - 'schemes': dict with per-scheme recommendations:
                        'CS': {'dt_limit': float, 'dt_recommended': float, 'status': str}
                        'RWM': {'dt_limit': float, 'dt_recommended': float, 'status': str}
                        'NONE': {'dt_limit': float, 'dt_recommended': float, 'status': str}
                - 'current_dt': Current time step from config
                - 'current_scheme': Currently selected scheme
                - 'issues': List of warnings/issues if any

    Raises:
            RuntimeError: If no particles have been added yet
    """

    # Check preconditions
    if system.particles.number_of_particles == 0:
        raise RuntimeError("No particles in system. Call add_vortex_particles() first.")

    if not 0 < safety_factor <= 1.0:
        raise ValueError(f"safety_factor must be in (0, 1], got {safety_factor}")

    # Compute statistics
    stats = _compute_particle_statistics(system)
    h_min = stats["h_min"]
    h_max = stats["h_max"]
    h_mean = stats["h_mean"]
    u_max = stats["u_max"]
    u_mean = stats["u_mean"]
    nu_molecular = stats["nu_molecular"]
    nu_turbulent_max = stats["nu_turbulent_max"]
    nu_eff_max = stats["nu_eff_max"]
    grad_u_max = stats["grad_u_max"]
    N = system.particles.number_of_particles

    # Initialise issue list
    issues = []
    schemes_info = {
        "CS": _compute_scheme_timestep(
            "CS", stats, safety_factor, system.time_step_size, use_mean_spacing=False
        ),
        "RWM": _compute_scheme_timestep(
            "RWM", stats, safety_factor, system.time_step_size, use_mean_spacing=True
        ),
        "NONE": _compute_scheme_timestep(
            "NONE", stats, safety_factor, system.time_step_size, use_mean_spacing=True
        ),
    }
    # ---- Check for issues ----

    if system.time_step_size > schemes_info[system.viscous_scheme]["dt_limit"]:
        issues.append(
            f"Current Δt = {system.time_step_size:.3e}s exceeds {system.viscous_scheme} limit "
            f"({schemes_info[system.viscous_scheme]['dt_limit']:.3e}s). Risk of instability!"
        )

    if nu_eff_max < EPSILON:
        issues.append("Effective viscosity is effectively zero. Check viscosity initialization.")

    if u_max < EPSILON:
        issues.append("Maximum velocity is very small. Flow appears to be static.")

    if h_min / h_max > 2.0:
        issues.append(
            f"Large variation in particle spacing (h_min/h_max = {h_min / h_max:.2f}). "
            "Consider more uniform distribution."
        )

    # ---- Assemble results ----

    results = {
        "h_min": h_min,
        "h_max": h_max,
        "h_mean": h_mean,
        "h_ratio": h_min / h_max if h_max > 0 else 1.0,
        "u_max": u_max,
        "u_mean": u_mean,
        "nu_molecular": nu_molecular,
        "nu_turbulent_max": nu_turbulent_max,
        "nu_eff_max": nu_eff_max,
        "grad_u_max": grad_u_max,
        "schemes": schemes_info,
        "current_dt": system.time_step_size,
        "current_scheme": system.viscous_scheme,
        "issues": issues,
        "num_particles": N,
        "reynolds_number": (u_max * h_mean / nu_molecular) if nu_molecular > 0 else float("inf"),
    }

    # ---- Print summary if verbose ----

    if verbose:
        system._print_timestep_validation_summary(results)

    return results
