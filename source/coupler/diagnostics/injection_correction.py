"""Constrained circulation correction for FVM-to-particle injection."""

from contextlib import contextmanager
import ctypes
import ctypes.util
from dataclasses import dataclass
import logging
from typing import Literal

import numpy as np

logger = logging.getLogger("coupler")

# OpenFOAM can enable glibc hardware traps that conflict with LAPACK's IEEE
# capability probe. Preserve and restore that mask around the two LAPACK calls.
_FE_INVALID = 0x01
_FE_DIVBYZERO = 0x04
_FE_OVERFLOW = 0x08
_FE_ALL_EXCEPT = _FE_INVALID | _FE_DIVBYZERO | _FE_OVERFLOW

_libm = None
_libm_path = ctypes.util.find_library("m")
if _libm_path:
    try:
        _libm = ctypes.CDLL(_libm_path)
    except OSError:
        _libm = None


@contextmanager
def _suspend_fpe_traps():
    """Suspend glibc FPE traps when that API is available."""
    if _libm is None or not hasattr(_libm, "fedisableexcept"):
        yield
        return
    old_traps = _libm.fedisableexcept(_FE_ALL_EXCEPT)
    try:
        yield
    finally:
        if old_traps & _FE_ALL_EXCEPT:
            _libm.feenableexcept(old_traps & _FE_ALL_EXCEPT)


@dataclass
class CorrectionStats:
    """Statistics from invariant correction."""

    correction_norm: float  # RMS correction magnitude
    max_correction: float  # Maximum single-particle correction
    relative_correction: float  # Correction / original circulation
    residual_error: float  # Post-correction conservation error
    condition_number: float  # Condition number of system matrix
    num_particles: int  # Number of particles corrected


def recover_invariants(
    pos: np.ndarray,
    circ: np.ndarray,
    target_invariants: dict[str, np.ndarray],
    reference_point: np.ndarray | None = None,
    return_stats: bool = False,
    conserve_circulation: bool = True,
    conserve_linear_impulse: bool = True,
    conserve_angular_impulse: bool = True,
    volumes: np.ndarray | None = None,
    weighting: Literal["volume", "circulation"] = "volume",
    correction_tolerance: float = 1e-14,
    max_condition_number: float = 1e12,
) -> np.ndarray | tuple[np.ndarray, CorrectionStats]:
    """Redistribute circulation to recover selected integral invariants.

    ``target_invariants`` contains three-component ``circulation``,
    ``linear_impulse`` and ``angular_impulse`` arrays. Volume weighting requires
    one positive volume per particle. Ill-conditioned correction systems are
    rejected using ``max_condition_number``.
    """
    n_particles = len(pos)
    if pos.shape != (n_particles, 3) or circ.shape != (n_particles, 3):
        raise ValueError(f"pos and circ must be (N, 3), got {pos.shape}, {circ.shape}")
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(circ)):
        raise ValueError("pos and circ must contain only finite values")
    if not np.isfinite(correction_tolerance) or correction_tolerance < 0.0:
        raise ValueError("correction_tolerance must be finite and non-negative")
    if not np.isfinite(max_condition_number) or max_condition_number < 1.0:
        raise ValueError("max_condition_number must be finite and at least one")

    required_keys = ["circulation", "linear_impulse", "angular_impulse"]
    for key in required_keys:
        if key not in target_invariants:
            raise ValueError(f"target_invariants missing key: {key}")
        if target_invariants[key].shape != (3,):
            raise ValueError(f"{key} must be shape (3,), got {target_invariants[key].shape}")
        if not np.all(np.isfinite(target_invariants[key])):
            raise ValueError(f"{key} must contain only finite values")

    if reference_point is not None:
        reference_point = np.asarray(reference_point, dtype=float)
        if reference_point.shape != (3,) or not np.all(np.isfinite(reference_point)):
            raise ValueError("reference_point must be a finite vector with shape (3,)")

    if n_particles == 0:
        active = []
        if conserve_circulation:
            active.append(target_invariants["circulation"])
        if conserve_linear_impulse:
            active.append(target_invariants["linear_impulse"])
        if conserve_angular_impulse:
            active.append(target_invariants["angular_impulse"])
        residual = np.linalg.norm(np.concatenate(active)) if active else 0.0
        if residual > correction_tolerance:
            raise ValueError("Cannot recover non-zero invariants without particles")
        stats = CorrectionStats(0, 0, 0, residual, 1, 0)
        return (circ, stats) if return_stats else circ

    circ_corrected, stats = _recover_invariants_numpy(
        pos,
        circ,
        target_invariants,
        reference_point,
        conserve_circulation,
        conserve_linear_impulse,
        conserve_angular_impulse,
        volumes=volumes,
        weighting=weighting,
        correction_tolerance=correction_tolerance,
        max_condition_number=max_condition_number,
    )

    if return_stats:
        return circ_corrected, stats
    return circ_corrected


def _recover_invariants_numpy(
    pos: np.ndarray,
    circ: np.ndarray,
    target_invariants: dict[str, np.ndarray],
    reference_point: np.ndarray | None = None,
    conserve_circulation: bool = True,
    conserve_linear_impulse: bool = True,
    conserve_angular_impulse: bool = True,
    volumes: np.ndarray | None = None,
    weighting: Literal["volume", "circulation"] = "volume",
    correction_tolerance: float = 1e-14,
    max_condition_number: float = 1e12,
) -> tuple[np.ndarray, CorrectionStats]:
    """
    NumPy implementation of invariant recovery.

    Uses the 9-probe method to build the system matrix efficiently.
    Cost: O(N) for matrix build + O(1) for 9×9 solve.
    """
    N = len(pos)

    active_indices = []
    if conserve_circulation:
        active_indices.extend([0, 1, 2])
    if conserve_linear_impulse:
        active_indices.extend([3, 4, 5])
    if conserve_angular_impulse:
        active_indices.extend([6, 7, 8])

    n_constraints = len(active_indices)

    if n_constraints == 0:
        logger.debug("No invariants selected for conservation.")
        return circ, CorrectionStats(0, 0, 0, 0, 1.0, N)

    if n_constraints > 3 * N:
        raise ValueError(
            f"Cannot conserve {n_constraints} scalar constraints with only {3 * N} "
            "particle-circulation degrees of freedom"
        )

    if reference_point is None:
        w_mag = np.linalg.norm(circ, axis=1)
        if np.sum(w_mag) > 1e-12:
            reference_point = np.average(pos, weights=w_mag, axis=0)
        else:
            reference_point = np.mean(pos, axis=0)

    pos_rel = pos - reference_point

    current_Gamma = np.sum(circ, axis=0)
    x_cross_G = np.cross(pos_rel, circ)
    current_I = 0.5 * np.sum(x_cross_G, axis=0)
    current_A = (1.0 / 3.0) * np.sum(np.cross(pos_rel, x_cross_G), axis=0)

    R = reference_point
    G_tgt = target_invariants["circulation"]
    I_tgt_0 = target_invariants["linear_impulse"]
    A_tgt_0 = target_invariants["angular_impulse"]

    I_tgt_R = I_tgt_0 - 0.5 * np.cross(R, G_tgt)

    A_tgt_R = (
        A_tgt_0
        - np.cross(R, I_tgt_0)
        - np.cross(R, I_tgt_R)
        + (1.0 / 3.0) * np.cross(R, np.cross(R, G_tgt))
    )

    R_Gamma = G_tgt - current_Gamma
    R_I = I_tgt_R - current_I
    R_A = A_tgt_R - current_A

    R_total_full = np.concatenate([R_Gamma, R_I, R_A])
    R_active = R_total_full[active_indices]

    residual_norm = np.linalg.norm(R_active)
    if residual_norm <= correction_tolerance:
        return circ, CorrectionStats(0, 0, 0, 0, 1.0, N)

    if weighting == "volume":
        if volumes is None or np.asarray(volumes).shape != (N,):
            raise ValueError(f"volume weighting requires volumes with shape ({N},)")
        weights = np.asarray(volumes, dtype=float)
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("particle volumes must be finite and positive")
    elif weighting == "circulation":
        weights = np.linalg.norm(circ, axis=1)
    else:
        raise ValueError(f"Unknown invariant-correction weighting {weighting!r}")
    total_weight = np.sum(weights)

    if total_weight < 1e-14:
        raise ValueError("Invariant correction has zero total particle weight")

    M = np.zeros((9, 9))
    basis_vectors = np.eye(9)

    for i in range(9):
        L_vec = basis_vectors[i]
        L_Gamma = L_vec[0:3]
        L_I = L_vec[3:6]
        L_A = L_vec[6:9]

        term1 = L_Gamma
        term2 = 0.5 * np.cross(pos_rel, L_I)
        term3 = (1.0 / 3.0) * np.cross(pos_rel, np.cross(pos_rel, L_A))

        dGamma_probe = (term1 + term2 + term3) * weights[:, None]

        res_Gamma = np.sum(dGamma_probe, axis=0)
        x_cross_dG = np.cross(pos_rel, dGamma_probe)
        res_I = 0.5 * np.sum(x_cross_dG, axis=0)
        res_A = (1.0 / 3.0) * np.sum(np.cross(pos_rel, x_cross_dG), axis=0)

        M[:, i] = np.concatenate([res_Gamma, res_I, res_A])

    M_active = M[np.ix_(active_indices, active_indices)]

    if (not np.all(np.isfinite(M_active))) or (not np.all(np.isfinite(R_active))):
        raise FloatingPointError("Invariant correction matrix or residual contains non-finite data")

    with _suspend_fpe_traps():
        cond = np.linalg.cond(M_active)
        if not np.isfinite(cond) or cond > max_condition_number:
            raise np.linalg.LinAlgError(
                f"Invariant correction condition number {cond:.3e} exceeds "
                f"{max_condition_number:.3e}"
            )

        lambdas_active, _, rank, _ = np.linalg.lstsq(M_active, R_active, rcond=None)
        if rank < n_constraints:
            raise np.linalg.LinAlgError(
                f"Invariant correction matrix rank {rank} is below {n_constraints}"
            )

    lambdas = np.zeros(9)
    lambdas[active_indices] = lambdas_active

    L_Gamma = lambdas[0:3]
    L_I = lambdas[3:6]
    L_A = lambdas[6:9]

    term1 = L_Gamma
    term2 = 0.5 * np.cross(pos_rel, L_I)
    term3 = (1.0 / 3.0) * np.cross(pos_rel, np.cross(pos_rel, L_A))

    correction = (term1 + term2 + term3) * weights[:, None]
    circ_corrected = circ + correction

    correction_norm = np.sqrt(np.mean(np.sum(correction**2, axis=1)))
    max_correction = np.max(np.linalg.norm(correction, axis=1))
    original_norm = np.sqrt(np.mean(np.sum(circ**2, axis=1)))
    relative_correction = correction_norm / (original_norm + 1e-16)

    new_Gamma = np.sum(circ_corrected, axis=0)
    new_x_cross_G = np.cross(pos_rel, circ_corrected)
    new_I = 0.5 * np.sum(new_x_cross_G, axis=0)
    new_A = (1.0 / 3.0) * np.sum(np.cross(pos_rel, new_x_cross_G), axis=0)

    residual_after_full = np.concatenate([G_tgt - new_Gamma, I_tgt_R - new_I, A_tgt_R - new_A])
    residual_after = np.linalg.norm(residual_after_full[active_indices])

    stats = CorrectionStats(
        correction_norm=correction_norm,
        max_correction=max_correction,
        relative_correction=relative_correction,
        residual_error=residual_after,
        condition_number=cond,
        num_particles=N,
    )

    return circ_corrected, stats


def validate_conservation(
    pos: np.ndarray,
    circ: np.ndarray,
    target_invariants: dict[str, np.ndarray],
    tolerance: float = 1e-10,
) -> dict[str, float]:
    """
    Validate that particle field matches target invariants.

    Args:
        pos: (N, 3) Particle positions
        circ: (N, 3) Particle circulations
        target_invariants: Target values
        tolerance: Relative error threshold

    Returns:
        Dict with keys:
            - 'circulation_error': Relative error in Γ
            - 'linear_impulse_error': Relative error in I
            - 'angular_impulse_error': Relative error in A
            - 'passed': True if all errors < tolerance
    """
    # Compute current invariants
    current_Gamma = np.sum(circ, axis=0)
    x_cross_G = np.cross(pos, circ)
    current_I = 0.5 * np.sum(x_cross_G, axis=0)
    current_A = (1.0 / 3.0) * np.sum(np.cross(pos, x_cross_G), axis=0)

    # Compute relative errors
    def rel_error(current, target):
        diff = np.linalg.norm(current - target)
        norm = np.linalg.norm(target)
        return diff / (norm + 1e-16)

    errors = {
        "circulation_error": rel_error(current_Gamma, target_invariants["circulation"]),
        "linear_impulse_error": rel_error(current_I, target_invariants["linear_impulse"]),
        "angular_impulse_error": rel_error(current_A, target_invariants["angular_impulse"]),
    }

    errors["passed"] = all(err < tolerance for err in errors.values())

    return errors


def compute_correction_quality(
    pos_original: np.ndarray,
    circ_original: np.ndarray,
    pos_kept: np.ndarray,
    circ_kept: np.ndarray,
    circ_corrected: np.ndarray,
) -> dict[str, float]:
    """
    Assess quality of correction by comparing filtered+corrected field
    to original full field.

    Args:
        pos_original: Original particle positions (before filtering)
        circ_original: Original circulations (before filtering)
        pos_kept: Kept particle positions (after filtering)
        circ_kept: Kept circulations (after filtering, before correction)
        circ_corrected: Corrected circulations (after recover_invariants)

    Returns:
        Dict with quality metrics:
            - 'spatial_fidelity': Measure of how well correction preserves
              spatial distribution (ideally close to 1.0)
            - 'magnitude_distortion': RMS change in circulation magnitude
            - 'direction_distortion': RMS change in circulation direction
    """
    # Compute metrics
    circ_kept_mag = np.linalg.norm(circ_kept, axis=1)
    circ_corrected_mag = np.linalg.norm(circ_corrected, axis=1)

    # Magnitude distortion
    mag_change = np.abs(circ_corrected_mag - circ_kept_mag)
    magnitude_distortion = np.sqrt(np.mean(mag_change**2))

    # Direction distortion (cosine similarity)
    mask = (circ_kept_mag > 1e-12) & (circ_corrected_mag > 1e-12)
    if np.any(mask):
        cos_sim = np.sum(circ_kept[mask] * circ_corrected[mask], axis=1) / (
            circ_kept_mag[mask] * circ_corrected_mag[mask]
        )
        direction_distortion = np.sqrt(np.mean((1.0 - cos_sim) ** 2))
    else:
        direction_distortion = 0.0

    # Spatial fidelity: Compare centroid shifts
    original_centroid = np.average(
        pos_original, weights=np.linalg.norm(circ_original, axis=1), axis=0
    )
    corrected_centroid = np.average(pos_kept, weights=circ_corrected_mag, axis=0)

    centroid_shift = np.linalg.norm(corrected_centroid - original_centroid)
    domain_size = np.max(pos_original, axis=0) - np.min(pos_original, axis=0)
    spatial_fidelity = 1.0 - centroid_shift / (np.linalg.norm(domain_size) + 1e-12)

    return {
        "spatial_fidelity": spatial_fidelity,
        "magnitude_distortion": magnitude_distortion,
        "direction_distortion": direction_distortion,
        "centroid_shift": centroid_shift,
    }
