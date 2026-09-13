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
    position = system.particles.position_cpu()
    velocity = system.particles.velocity_cpu()
    core_radius = system.particles.core_radius_cpu()
    kinematic_viscosity = system.particles.kinematic_viscosity_cpu()

    n_particles_total = len(position)

    # Particle spacing
    h_array = 2.0 * core_radius
    min_particle_spacing = float(np.min(h_array))
    max_particle_spacing = float(np.max(h_array))
    mean_particle_spacing = float(np.mean(h_array))

    # Velocity
    velocity_magnitude = np.linalg.norm(velocity, axis=1)
    max_velocity_magnitude = float(np.max(velocity_magnitude)) if n_particles_total > 0 else 0.0
    mean_velocity_magnitude = float(np.mean(velocity_magnitude)) if n_particles_total > 0 else 0.0

    # Viscosity
    max_kinematic_viscosity = float(np.max(kinematic_viscosity)) if n_particles_total > 0 else 1e-5

    # Turbulent viscosity
    max_eddy_viscosity = 0.0
    if getattr(system, "turbulence_model", None) is not None:
        try:
            eddy_viscosity = system.particles.eddy_viscosity_cpu()
            max_eddy_viscosity = float(np.max(eddy_viscosity)) if len(eddy_viscosity) > 0 else 0.0
        except Exception:
            pass

    max_effective_viscosity = max_kinematic_viscosity + max_eddy_viscosity

    # Strain rate
    try:
        velocity_gradient = system.particles.velocity_gradient_cpu()
        velocity_gradient_magnitude = np.linalg.norm(
            velocity_gradient.reshape(n_particles_total, 9), axis=1
        )
        max_velocity_gradient_magnitude = (
            float(np.max(velocity_gradient_magnitude)) if n_particles_total > 0 else 1e-5
        )
    except Exception:
        max_velocity_gradient_magnitude = 1e-5

    return {
        "min_particle_spacing": min_particle_spacing,
        "max_particle_spacing": max_particle_spacing,
        "mean_particle_spacing": mean_particle_spacing,
        "max_velocity_magnitude": max_velocity_magnitude,
        "mean_velocity_magnitude": mean_velocity_magnitude,
        "max_kinematic_viscosity": max_kinematic_viscosity,
        "max_eddy_viscosity": max_eddy_viscosity,
        "max_effective_viscosity": max_effective_viscosity,
        "max_velocity_gradient_magnitude": max_velocity_gradient_magnitude,
    }


def _compute_scheme_time_step(
    scheme_name, stats, safety_factor, system_time_step_size, use_mean_spacing=False
):
    """Compute time step limits for a specific viscous scheme."""
    CFL_advection = 0.5
    C_diff = 1.0  # RWM accuracy coefficient
    C_stretching = 1.0

    particle_spacing = (
        stats["mean_particle_spacing"] if use_mean_spacing else stats["min_particle_spacing"]
    )

    # Advection limit
    advection_time_step_size_limit = (
        CFL_advection * particle_spacing / stats["max_velocity_magnitude"]
        if stats["max_velocity_magnitude"] > EPSILON
        else 1.0
    )

    # Diffusion limit (NONE and CS don't have an explicit diffusion step: CS
    # spreads cores analytically, so there is no parabolic-CFL stability bound).
    if scheme_name in ("NONE", "CS"):
        diffusion_time_step_size_limit = float("inf")
    else:
        diffusion_time_step_size_limit = (
            C_diff * particle_spacing**2 / stats["max_effective_viscosity"]
            if stats["max_effective_viscosity"] > EPSILON
            else 1.0
        )

    # Stretching limit
    stretching_time_step_size_limit = (
        C_stretching / stats["max_velocity_gradient_magnitude"]
        if stats["max_velocity_gradient_magnitude"] > EPSILON
        else 1.0
    )

    # Find minimum
    components = (
        [
            advection_time_step_size_limit,
            diffusion_time_step_size_limit,
            stretching_time_step_size_limit,
        ]
        if scheme_name != "NONE"
        else [advection_time_step_size_limit, stretching_time_step_size_limit]
    )
    time_step_size_limit = safety_factor * min(components)

    comp_names = (
        ["advection", "diffusion", "stretching"]
        if scheme_name != "NONE"
        else ["advection", "stretching"]
    )
    limiting_idx = np.argmin(components)

    result = {
        "time_step_size_limit": time_step_size_limit,
        "recommended_time_step_size": time_step_size_limit,
        "advection_time_step_size_limit": advection_time_step_size_limit,
        "stretching_time_step_size_limit": stretching_time_step_size_limit,
        "limiting_component": comp_names[limiting_idx],
        "status": "stable"
        if system_time_step_size <= time_step_size_limit
        else "WARNING: too large",
    }

    if scheme_name not in ("NONE", "CS"):
        result["diffusion_time_step_size_limit"] = diffusion_time_step_size_limit

    return result


# NOTE: This function appears to be unused (no callers found in the codebase).
# It may have been superseded by the viscous scheme initialization.
# Consider removing if no longer needed after verification.
def _validate_time_step_sizing(system, safety_factor=0.8, verbose=True):
    """
    Check and recommend time-step sizing constraints for all viscous schemes.

    This method computes the maximum stable time step for each viscous scheme based on:
    - Advection CFL constraint: Δt_adv ≤ CFL * particle_spacing / max_velocity_magnitude
    - Viscous diffusion constraint: Δt_visc ≤ C_visc * particle_spacing² / kinematic_viscosity
    - RWM stochastic stability: Δt_rwm ≤ particle_spacing² / (C_rwm * kinematic_viscosity)
    - Stretching term constraint: Δt_stretch ≤ 1 / max_velocity_gradient_magnitude

    Must be called AFTER add_vortex_particles() to have particle data available.

    Args:
            safety_factor: Reduction factor applied to computed limits (0 < factor ≤ 1).
                        Default 0.8 is conservative; use 0.9-1.0 for less conservative.
            verbose: If True, print formatted summary to console. If False, return dict only.

    Returns:
            dict: Timestep recommendations with keys:
                - 'min_particle_spacing': Minimum particle spacing [m]
                - 'max_particle_spacing': Maximum particle spacing [m]
                - 'mean_particle_spacing': Mean particle spacing [m]
                - 'max_velocity_magnitude': Maximum particle velocity magnitude [m/s]
                - 'max_kinematic_viscosity': Molecular kinematic viscosity [m²/s]
                - 'max_eddy_viscosity': Maximum turbulent viscosity (if computed) [m²/s]
                - 'schemes': dict with per-scheme recommendations:
                        'CS': {'time_step_size_limit': float, 'recommended_time_step_size': float, 'status': str}
                        'RWM': {'time_step_size_limit': float, 'recommended_time_step_size': float, 'status': str}
                        'NONE': {'time_step_size_limit': float, 'recommended_time_step_size': float, 'status': str}
                - 'time_step_size': Current time step from config
                - 'viscous_scheme': Currently selected scheme
                - 'issues': List of warnings/issues if any

    Raises:
            RuntimeError: If no particles have been added yet
    """

    # Check preconditions
    if system.particles.n_particles_total == 0:
        raise RuntimeError("No particles in system. Call add_vortex_particles() first.")

    if not 0 < safety_factor <= 1.0:
        raise ValueError(f"safety_factor must be in (0, 1], got {safety_factor}")

    # Compute statistics
    stats = _compute_particle_statistics(system)
    min_particle_spacing = stats["min_particle_spacing"]
    max_particle_spacing = stats["max_particle_spacing"]
    mean_particle_spacing = stats["mean_particle_spacing"]
    max_velocity_magnitude = stats["max_velocity_magnitude"]
    mean_velocity_magnitude = stats["mean_velocity_magnitude"]
    max_kinematic_viscosity = stats["max_kinematic_viscosity"]
    max_eddy_viscosity = stats["max_eddy_viscosity"]
    max_effective_viscosity = stats["max_effective_viscosity"]
    max_velocity_gradient_magnitude = stats["max_velocity_gradient_magnitude"]
    n_particles_total = system.particles.n_particles_total

    # Initialise issue list
    issues = []
    schemes_info = {
        "CS": _compute_scheme_time_step(
            "CS", stats, safety_factor, system.time_step_size, use_mean_spacing=False
        ),
        "RWM": _compute_scheme_time_step(
            "RWM", stats, safety_factor, system.time_step_size, use_mean_spacing=True
        ),
        "NONE": _compute_scheme_time_step(
            "NONE", stats, safety_factor, system.time_step_size, use_mean_spacing=True
        ),
    }
    # ---- Check for issues ----

    if system.time_step_size > schemes_info[system.viscous_scheme]["time_step_size_limit"]:
        issues.append(
            f"Current Δt = {system.time_step_size:.3e}s exceeds {system.viscous_scheme} limit "
            f"({schemes_info[system.viscous_scheme]['time_step_size_limit']:.3e}s). Risk of instability!"
        )

    if max_effective_viscosity < EPSILON:
        issues.append("Effective viscosity is effectively zero. Check viscosity initialization.")

    if max_velocity_magnitude < EPSILON:
        issues.append("Maximum velocity is very small. Flow appears to be static.")

    particle_spacing_ratio = (
        min_particle_spacing / max_particle_spacing if max_particle_spacing > 0 else 1.0
    )
    if particle_spacing_ratio < 0.5:
        issues.append(
            f"Large variation in particle spacing (min/max = {particle_spacing_ratio:.2f}). "
            "Consider more uniform distribution."
        )

    # ---- Assemble results ----

    results = {
        "min_particle_spacing": min_particle_spacing,
        "max_particle_spacing": max_particle_spacing,
        "mean_particle_spacing": mean_particle_spacing,
        "particle_spacing_ratio": particle_spacing_ratio,
        "max_velocity_magnitude": max_velocity_magnitude,
        "mean_velocity_magnitude": mean_velocity_magnitude,
        "max_kinematic_viscosity": max_kinematic_viscosity,
        "max_eddy_viscosity": max_eddy_viscosity,
        "max_effective_viscosity": max_effective_viscosity,
        "max_velocity_gradient_magnitude": max_velocity_gradient_magnitude,
        "schemes": schemes_info,
        "time_step_size": system.time_step_size,
        "viscous_scheme": system.viscous_scheme,
        "issues": issues,
        "n_particles_total": n_particles_total,
        "reynolds_number": (
            max_velocity_magnitude * mean_particle_spacing / max_kinematic_viscosity
            if max_kinematic_viscosity > 0
            else float("inf")
        ),
    }

    # ---- Print summary if verbose ----

    if verbose:
        system._print_time_step_validation_summary(results)

    return results
