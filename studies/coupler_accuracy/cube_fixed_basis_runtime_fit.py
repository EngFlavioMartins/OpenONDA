"""Study-only fixed-Gaussian-basis fit with a protected physical renewal belt.

The supplied operator and moment-row classes must come from the same solver
source loaded by the caller. No source modules are imported at module load,
so historical cube checkpoints can be restored under their original schema.
"""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg


def fit_outer_strength(
    position: np.ndarray,
    strength: np.ndarray,
    core_radius: np.ndarray,
    particle_volume: np.ndarray,
    *,
    operator_type: Callable,
    moment_nullspace_type: Callable,
    gaussian_moment_rows: Callable,
    renewal_bounds: np.ndarray,
    spacing: float = 0.06,
    regularization: float = 0.1,
) -> tuple[np.ndarray, dict]:
    """Fit a fixed Gaussian basis while preserving the FVM renewal region.

    Parameters
    ----------
    position : ndarray, shape (N, 3)
        Particle centres in m.
    strength : ndarray, shape (N, 3)
        Particle vortex strengths in m³/s; the input is read only.
    core_radius, particle_volume : ndarray, shape (N,)
        Positive Gaussian core radii in m and particle volumes in m³.
    operator_type, moment_nullspace_type, gaussian_moment_rows : callable
        Gaussian-grid and nine-moment operators from the same VPM source tree.
    renewal_bounds : ndarray, shape (6,)
        Ordered lower/upper bounds in m for each Cartesian axis. Strengths of
        particles within this closed box remain exactly unchanged.
    spacing : float, optional
        Diagnostic padded-grid spacing in m; default 0.06.
    regularization : float, optional
        Positive dimensionless Tikhonov weight; default 0.1.

    Returns
    -------
    ndarray, shape (N, 3)
        Float64 corrected strengths in m³/s, with the same particle order.
    dict
        Fit residual, population, correction, and wall-time measurements.

    Notes
    -----
    The fit solves a regularized normal equation for the Helmholtz residual
    of the reconstructed vorticity. Corrections are restricted to the outer
    particles' null space of net strength and both impulses. This research
    operator does not enforce the production solver's quadratic-invariant
    gates and does not mutate a solver or change particle positions/cores.
    """
    position = np.asarray(position, dtype=np.float64)
    strength = np.asarray(strength, dtype=np.float64)
    core_radius = np.asarray(core_radius, dtype=np.float64)
    particle_volume = np.asarray(particle_volume, dtype=np.float64)
    renewal_bounds = np.asarray(renewal_bounds, dtype=np.float64)
    if position.shape != strength.shape or position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("Particle position and strength must both have shape (N, 3)")
    if core_radius.shape != (len(position),) or particle_volume.shape != (len(position),):
        raise ValueError("Core radius and particle volume must match the particle count")
    if renewal_bounds.shape != (6,) or np.any(renewal_bounds[::2] >= renewal_bounds[1::2]):
        raise ValueError("Physical renewal bounds must contain six ordered values")
    if not np.all(np.isfinite(strength)) or not np.all(np.isfinite(position)):
        raise ValueError("Particle fit input is non-finite")
    if not np.all(np.isfinite(core_radius)) or not np.all(np.isfinite(particle_volume)):
        raise ValueError("Particle cores and volumes must be finite")
    if not np.all(np.isfinite(renewal_bounds)):
        raise ValueError("Physical renewal bounds must be finite")
    if not np.all(core_radius > 0) or not np.all(particle_volume > 0):
        raise ValueError("Particle cores and volumes must be positive")
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Diagnostic grid spacing must be positive and finite")
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("Regularization must be positive and finite")
    frozen = np.all((position >= renewal_bounds[::2]) & (position <= renewal_bounds[1::2]), axis=1)
    mutable = ~frozen
    if not np.any(mutable):
        raise ValueError("No VPM-owned outer particles are available for projection")
    started = perf_counter()
    operator = operator_type(
        position, core_radius, np.linalg.norm(strength, axis=1), spacing=spacing
    )
    residual, divergence_before, _ = operator.relaxation_residual(strength)
    sqrt_volume = np.sqrt(particle_volume)
    nullspace = moment_nullspace_type(
        gaussian_moment_rows(position[mutable], core_radius[mutable]),
        particle_volume[mutable],
    )

    def restrict(vector: np.ndarray) -> np.ndarray:
        """Keep only mutable moment-free transformed corrections, shape (N, 3)."""
        values = np.zeros_like(vector)
        values[mutable] = nullspace.project_transformed(vector[mutable])
        return values

    def matrix_vector(vector: np.ndarray) -> np.ndarray:
        """Apply the regularized normal equation to a flat transformed vector."""
        values = restrict(vector.reshape(-1, 3))
        applied = operator.apply(sqrt_volume[:, None] * values)
        transformed = sqrt_volume[:, None] * applied + regularization * values
        return restrict(transformed).ravel()

    iterations = 0

    def count_iteration(_vector: np.ndarray) -> None:
        """Count SciPy CG iterations without storing trial vectors."""
        nonlocal iterations
        iterations += 1

    system = LinearOperator(
        (3 * len(position), 3 * len(position)), matvec=matrix_vector, dtype=np.float64
    )
    rhs = restrict(sqrt_volume[:, None] * residual)
    solution, info = cg(
        system, rhs.ravel(), rtol=1e-5, atol=0.0, maxiter=30, callback=count_iteration
    )
    if info != 0:
        raise RuntimeError(f"Fixed-basis runtime projection did not converge: CG={info}")
    correction = sqrt_volume[:, None] * restrict(solution.reshape(-1, 3))
    candidate = strength + correction
    if not np.array_equal(candidate[frozen], strength[frozen]):
        raise AssertionError("Projection modified a renewable-belt source")
    residual_after, divergence_after, _ = operator.relaxation_residual(candidate)
    metadata = {
        "particles": len(position),
        "frozen_particles": int(frozen.sum()),
        "mutable_particles": int(mutable.sum()),
        "grid_shape": operator.shape,
        "cg_iterations": iterations,
        "fit_wall_seconds": perf_counter() - started,
        "correction_strength_relative": float(np.linalg.norm(correction))
        / max(float(np.linalg.norm(strength)), np.finfo(float).tiny),
        "initial_residual_norm": float(np.linalg.norm(residual)),
        "final_residual_ratio": float(np.linalg.norm(residual_after))
        / max(float(np.linalg.norm(residual)), np.finfo(float).tiny),
        "grid_divergence_before": divergence_before,
        "grid_divergence_after": divergence_after,
        "net_correction_inside_belt_m3_s": correction[frozen].sum(axis=0).tolist(),
        "net_correction_outside_belt_m3_s": correction[mutable].sum(axis=0).tolist(),
    }
    return candidate, metadata


__all__ = ["fit_outer_strength"]
