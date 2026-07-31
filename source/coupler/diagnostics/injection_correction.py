"""Volume-weighted circulation and impulse recovery after particle pruning."""

from contextlib import contextmanager
import ctypes
import ctypes.util

import numpy as np

_FE_ALL_EXCEPT = 0x01 | 0x04 | 0x08
_libm_path = ctypes.util.find_library("m")
try:
    _libm = ctypes.CDLL(_libm_path) if _libm_path else None
except OSError:
    _libm = None


@contextmanager
def _suspend_fpe_traps():
    if _libm is None or not hasattr(_libm, "fedisableexcept"):
        yield
        return
    previous = _libm.fedisableexcept(_FE_ALL_EXCEPT)
    try:
        yield
    finally:
        if previous & _FE_ALL_EXCEPT:
            _libm.feenableexcept(previous & _FE_ALL_EXCEPT)


def recover_invariants(
    positions: np.ndarray,
    circulation: np.ndarray,
    target: dict[str, np.ndarray],
    *,
    volumes: np.ndarray,
) -> np.ndarray:
    """Recover total circulation and linear impulse with minimum volume weight."""
    positions = np.asarray(positions, dtype=float)
    circulation = np.asarray(circulation, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    count = len(positions)
    if positions.shape != (count, 3) or circulation.shape != (count, 3):
        raise ValueError("positions and circulation must have shape (N, 3)")
    if volumes.shape != (count,) or np.any(volumes <= 0.0):
        raise ValueError("volumes must be positive with shape (N,)")
    if not all(np.isfinite(values).all() for values in (positions, circulation, volumes)):
        raise ValueError("particle data must be finite")

    target_circulation = np.asarray(target["circulation"], dtype=float)
    target_impulse = np.asarray(target["linear_impulse"], dtype=float)
    if target_circulation.shape != (3,) or target_impulse.shape != (3,):
        raise ValueError("target invariants must have shape (3,)")
    residual_scale = np.linalg.norm(np.concatenate((target_circulation, target_impulse)))
    if count == 0:
        if residual_scale > 1.0e-14:
            raise ValueError("cannot recover non-zero invariants without particles")
        return circulation
    if count < 2:
        raise ValueError("at least two particles are required for invariant recovery")

    reference = np.average(positions, weights=volumes, axis=0)
    relative = positions - reference
    current_circulation = circulation.sum(axis=0)
    current_impulse = 0.5 * np.cross(relative, circulation).sum(axis=0)
    target_impulse_relative = target_impulse - 0.5 * np.cross(reference, target_circulation)
    residual = np.concatenate(
        (
            target_circulation - current_circulation,
            target_impulse_relative - current_impulse,
        )
    )
    if np.linalg.norm(residual) <= 1.0e-14:
        return circulation

    matrix = np.zeros((6, 6))
    for column, probe in enumerate(np.eye(6)):
        delta = volumes[:, None] * (probe[:3] + 0.5 * np.cross(relative, probe[3:]))
        matrix[:3, column] = delta.sum(axis=0)
        matrix[3:, column] = 0.5 * np.cross(relative, delta).sum(axis=0)

    with _suspend_fpe_traps():
        condition = np.linalg.cond(matrix)
        if not np.isfinite(condition) or condition > 1.0e12:
            raise np.linalg.LinAlgError(
                f"invariant recovery matrix is ill-conditioned ({condition:.3e})"
            )
        multipliers = np.linalg.solve(matrix, residual)

    correction = volumes[:, None] * (multipliers[:3] + 0.5 * np.cross(relative, multipliers[3:]))
    return circulation + correction
