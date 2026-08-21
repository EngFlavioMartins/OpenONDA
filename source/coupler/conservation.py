"""Circulation and impulse conservation for the FVM-to-VPM transfer."""

from __future__ import annotations

import numpy as np


def invariants(positions: np.ndarray, vortex_strength: np.ndarray) -> dict[str, np.ndarray]:
    """Return total vortex strength, linear impulse, and raw angular impulse."""
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    vortex_strength = np.asarray(vortex_strength, dtype=np.float64).reshape(-1, 3)
    if len(positions) != len(vortex_strength):
        raise ValueError("positions and vortex_strength must contain the same number of particles")
    if len(positions) == 0:
        zero = np.zeros(3)
        return {
            "total_vortex_strength": zero.copy(),
            "linear_impulse": zero.copy(),
            "angular_impulse": zero.copy(),
        }
    position_cross_strength = np.cross(positions, vortex_strength)
    return {
        "total_vortex_strength": np.sum(vortex_strength, axis=0),
        "linear_impulse": 0.5 * np.sum(position_cross_strength, axis=0),
        "angular_impulse": (1.0 / 3.0)
        * np.sum(np.cross(positions, position_cross_strength), axis=0),
    }


def recover_invariants(
    positions: np.ndarray,
    vortex_strength: np.ndarray,
    target: dict[str, np.ndarray],
    *,
    volumes: np.ndarray,
) -> np.ndarray:
    """Recover total vortex strength and linear impulse with minimum volume weighting."""
    positions = np.asarray(positions, dtype=float)
    vortex_strength = np.asarray(vortex_strength, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    count = len(positions)
    if positions.shape != (count, 3) or vortex_strength.shape != (count, 3):
        raise ValueError("positions and vortex_strength must have shape (N, 3)")
    if volumes.shape != (count,) or np.any(volumes <= 0.0):
        raise ValueError("volumes must be positive with shape (N,)")
    if not all(np.isfinite(values).all() for values in (positions, vortex_strength, volumes)):
        raise ValueError("particle data must be finite")

    target_total_strength = np.asarray(target["total_vortex_strength"], dtype=float)
    target_impulse = np.asarray(target["linear_impulse"], dtype=float)
    if target_total_strength.shape != (3,) or target_impulse.shape != (3,):
        raise ValueError("target invariants must have shape (3,)")
    residual_scale = np.linalg.norm(np.concatenate((target_total_strength, target_impulse)))
    if count == 0:
        if residual_scale > 1.0e-14:
            raise ValueError("cannot recover non-zero invariants without particles")
        return vortex_strength
    if count < 2:
        raise ValueError("at least two particles are required for invariant recovery")

    reference = np.average(positions, weights=volumes, axis=0)
    relative = positions - reference
    current_total_strength = vortex_strength.sum(axis=0)
    current_impulse = 0.5 * np.cross(relative, vortex_strength).sum(axis=0)
    target_impulse_relative = target_impulse - 0.5 * np.cross(reference, target_total_strength)
    residual = np.concatenate(
        (
            target_total_strength - current_total_strength,
            target_impulse_relative - current_impulse,
        )
    )
    if np.linalg.norm(residual) <= 1.0e-14:
        return vortex_strength

    matrix = np.zeros((6, 6))
    for column, probe in enumerate(np.eye(6)):
        delta = volumes[:, None] * (probe[:3] + 0.5 * np.cross(relative, probe[3:]))
        matrix[:3, column] = delta.sum(axis=0)
        matrix[3:, column] = 0.5 * np.cross(relative, delta).sum(axis=0)

    condition = np.linalg.cond(matrix)
    if not np.isfinite(condition) or condition > 1.0e12:
        raise np.linalg.LinAlgError(
            f"invariant recovery matrix is ill-conditioned ({condition:.3e})"
        )
    multipliers = np.linalg.solve(matrix, residual)
    correction = volumes[:, None] * (multipliers[:3] + 0.5 * np.cross(relative, multipliers[3:]))
    return vortex_strength + correction


__all__ = ["invariants", "recover_invariants"]
