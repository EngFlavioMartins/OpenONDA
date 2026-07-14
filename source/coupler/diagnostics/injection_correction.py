"""
Moment-Conserving Correction for Particle Injection.
=====================================================

This module implements a constrained least-squares algorithm to adjust particle
circulations after threshold filtering, ensuring exact conservation of global
invariants (circulation, linear impulse, angular impulse).

Mathematical Framework
----------------------
After filtering weak particles, we seek a small correction δΓ to each kept
particle such that the global invariants match the original FVM field.

We minimize the perturbation weighted by particle strength:
    min Σ |δΓ_i|² / |Γ_i|

Subject to conservation constraints:
    Σ (Γ_i + δΓ_i) = Γ_target              (circulation)
    Σ r_i × (Γ_i + δΓ_i) = I_target        (linear impulse)
    Σ r_i × (r_i × (Γ_i + δΓ_i)) = A_target (angular impulse)

Solution via Lagrange Multipliers
----------------------------------
The correction has the analytic form:
    δΓ_i = w_i [λ_Γ + 0.5(r_i × λ_I) + (1/3)(r_i × (r_i × λ_A))]

where w_i = |Γ_i| is the particle weight and λ = [λ_Γ, λ_I, λ_A] are
Lagrange multipliers found by solving a 9×9 linear system.

References
----------
- Cottet & Koumoutsakos (2000), "Vortex Methods", Chapter 8
- Winckelmans (1993), PhD thesis on vortex particle methods
- Barba et al. (2005), "Advances in viscous vortex methods"

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: February 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import contextlib
from contextlib import contextmanager
import ctypes
import ctypes.util
from dataclasses import dataclass
import logging
import signal

import numpy as np

logger = logging.getLogger("coupler")

# =========================================================
# FPE guard: temporarily disable hardware FPE traps installed by OpenFOAM
# =========================================================
# OpenFOAM calls feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW)
# which makes the CPU raise SIGFPE on any such exception.  LAPACK's ieeeck
# routine intentionally divides by zero to test IEEE-754 compliance, so it
# triggers a fatal signal when OpenFOAM's handler is active.
#
# The fix: temporarily call fedisableexcept() to clear the hardware FPE
# traps around LAPACK calls (np.linalg.cond, np.linalg.lstsq, etc.), then
# restore them.  This is done via ctypes to call the glibc functions
# directly, which is the same mechanism OpenFOAM itself uses.
# =========================================================

# FE_* constants from <fenv.h> on Linux/glibc
_FE_INVALID = 0x01
_FE_DIVBYZERO = 0x04
_FE_OVERFLOW = 0x08
_FE_ALL_EXCEPT = _FE_INVALID | _FE_DIVBYZERO | _FE_OVERFLOW

# Try to load glibc for fenv control
_libm = None
try:
    _libm_path = ctypes.util.find_library("m")
    if _libm_path:
        _libm = ctypes.CDLL(_libm_path)
except Exception:
    pass


@contextmanager
def _suspend_fpe_traps():
    """Context manager to disable hardware FPE traps around LAPACK calls.

    On Linux with glibc, uses ``fedisableexcept`` / ``feenableexcept`` to
    temporarily clear the FE_DIVBYZERO, FE_INVALID and FE_OVERFLOW traps
    that OpenFOAM enables.  On exit the original traps are restored.

    If glibc is not available (e.g. macOS, musl), falls back to setting
    the Python-level SIGFPE handler to SIG_DFL for the duration of the
    block.
    """
    if _libm is not None and hasattr(_libm, "fedisableexcept"):
        # fedisableexcept returns the old set of enabled traps
        old_traps = _libm.fedisableexcept(_FE_ALL_EXCEPT)
        try:
            yield
        finally:
            # Restore exactly the traps that were active before
            if old_traps & _FE_ALL_EXCEPT:
                _libm.feenableexcept(old_traps & _FE_ALL_EXCEPT)
    else:
        old_handler = signal.getsignal(signal.SIGFPE)
        signal.signal(signal.SIGFPE, signal.SIG_DFL)
        try:
            yield
        finally:
            if callable(old_handler) or old_handler in (signal.SIG_DFL, signal.SIG_IGN):
                with contextlib.suppress(OSError, ValueError):
                    signal.signal(signal.SIGFPE, old_handler)


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
    use_taichi: bool = False,
    return_stats: bool = False,
    tolerance_ratio: float = 0.05,
    conserve_circulation: bool = True,
    conserve_linear_impulse: bool = True,
    conserve_angular_impulse: bool = True,
    volumes: np.ndarray | None = None,
) -> np.ndarray:
    """
    Adjust particle circulations to exactly match target invariants.

    This is the main API. After threshold filtering removes weak particles,
    this function redistributes the lost circulation/impulse onto the
    remaining particles in a physically consistent way.

    Args:
        pos: (N, 3) Particle positions [m]
        circ: (N, 3) Particle circulations (filtered set) [m²/s]
        target_invariants: Dict with keys:
            - 'circulation': (3,) Total circulation vector [m²/s]
            - 'linear_impulse': (3,) Total linear impulse [m³/s]
            - 'angular_impulse': (3,) Total angular impulse [m⁴/s]
            These should be the values from the ORIGINAL full FVM field.
        reference_point: (3,) Optional reference point for the moment correction.
            If None, the centroid of the particles is used. Shifting the reference
            to the centroid improves numerical conditioning and prevents
            non-physical "moving body" effects in low-circulation regions.
        use_taichi: Reserved for a future accelerated implementation.
        return_stats: If True, return (corrected_circ, stats) tuple
        tolerance_ratio: If net circulation target is less than this fraction
             of the integrated vorticity magnitude, the correction is bypassed
             to prevent signal destruction in non-isolated systems.
        volumes: (N,) optional particle volumes. When provided, corrections are
            weighted by volume (uniform redistribution per unit volume) instead
            of by |Γ| (which piles corrections onto the strongest particles).
            Volume weighting avoids the "moving body" artifact (coupler_design.md §WS3).

    Returns:
        circ_corrected: (N, 3) Adjusted circulations
        stats: CorrectionStats (if return_stats=True)

    Example:
        >>> # After threshold filtering
        >>> fvm_invariants = compute_invariants(pos_all, circ_all, radii_all)
        >>> circ_corrected = recover_invariants(
        ...     pos_kept, circ_kept, fvm_invariants
        ... )
        >>> # Now circ_corrected conserves FVM invariants exactly
    """
    N = len(pos)
    if N == 0:
        if return_stats:
            return circ, CorrectionStats(0, 0, 0, 0, 0, 0)
        return circ

    # Validate inputs
    if pos.shape != (N, 3) or circ.shape != (N, 3):
        raise ValueError(f"pos and circ must be (N, 3), got {pos.shape}, {circ.shape}")

    required_keys = ["circulation", "linear_impulse", "angular_impulse"]
    for key in required_keys:
        if key not in target_invariants:
            raise ValueError(f"target_invariants missing key: {key}")
        if target_invariants[key].shape != (3,):
            raise ValueError(f"{key} must be shape (3,), got {target_invariants[key].shape}")

    if use_taichi:
        raise NotImplementedError("Taichi invariant correction is not implemented")
    circ_corrected, stats = _recover_invariants_numpy(
        pos,
        circ,
        target_invariants,
        reference_point,
        tolerance_ratio,
        conserve_circulation,
        conserve_linear_impulse,
        conserve_angular_impulse,
        volumes=volumes,
    )

    if return_stats:
        return circ_corrected, stats
    return circ_corrected


def _recover_invariants_numpy(
    pos: np.ndarray,
    circ: np.ndarray,
    target_invariants: dict[str, np.ndarray],
    reference_point: np.ndarray | None = None,
    tolerance_ratio: float = 0.05,
    conserve_circulation: bool = True,
    conserve_linear_impulse: bool = True,
    conserve_angular_impulse: bool = True,
    volumes: np.ndarray | None = None,
) -> tuple[np.ndarray, CorrectionStats]:
    """
    NumPy implementation of invariant recovery.

    Uses the 9-probe method to build the system matrix efficiently.
    Cost: O(N) for matrix build + O(1) for 9×9 solve.
    """
    N = len(pos)

    # Determine active constraints
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

    # Guard: cannot correct with too few particles
    # Need at least N particles for N constraints
    if n_constraints > N:
        logger.debug(
            f"Cannot correct with only {N} particles (need ≥{n_constraints} for well-posed system)"
        )
        return circ, CorrectionStats(0, 0, 0, 0, 1.0, 0)

    # 1. Coordinate shifting for numerical stability
    if reference_point is None:
        # Default to vorticity centroid (weighted by circulation magnitude)
        w_mag = np.linalg.norm(circ, axis=1)
        if np.sum(w_mag) > 1e-12:
            reference_point = np.average(pos, weights=w_mag, axis=0)
        else:
            reference_point = np.mean(pos, axis=0)

    pos_rel = pos - reference_point

    # 2. Invariant bypass check
    # Only skip correction if the target is truly zero (absolute threshold).
    # NOTE: The old check compared target_gamma_mag to sum_abs_circ (L1 norm), which
    # always bypasses correction for bipolar vorticity fields — the vector sum is
    # near-zero even when individual magnitudes are large. This caused the correction
    # to be silently skipped on every step.
    target_gamma_mag = np.linalg.norm(target_invariants["circulation"])
    if target_gamma_mag < 1e-14:
        logger.debug("Bypassing moment correction: target Γ is effectively zero")
        return circ, CorrectionStats(0, 0, 0, 0, 1.0, N)

    # 3. Compute current invariants of the filtered set (relative to reference_point)
    # We compute these from scratch in the local frame to ensure consistency.
    current_Gamma = np.sum(circ, axis=0)
    x_cross_G = np.cross(pos_rel, circ)
    current_I = 0.5 * np.sum(x_cross_G, axis=0)
    current_A = (1.0 / 3.0) * np.sum(np.cross(pos_rel, x_cross_G), axis=0)

    # 4. Shift target invariants to the reference_point frame
    # Formula for Linear Impulse shift: I(R) = I(0) - 0.5 * R x Γ
    # Formula for Angular Impulse shift: A(R) = A(0) - R x I(0) - R x I(R) + (1/3) R x (R x Γ)
    R = reference_point
    G_tgt = target_invariants["circulation"]
    I_tgt_0 = target_invariants["linear_impulse"]
    A_tgt_0 = target_invariants["angular_impulse"]

    I_tgt_R = I_tgt_0 - 0.5 * np.cross(R, G_tgt)

    # Derivation for A(R) shift in 3D is complex; we use the direct definition shift:
    # A(R) = 1/3 Σ (r-R) x ((r-R) x Γ)
    # But since we don't have the original field i.e. the individual Γ_i, we must use
    # the shift formula for the total moments.
    # For a vorticity distribution, the shifts are:
    # L_1(R) = L_1(0) - R x L_0
    # L_2(R) = L_2(0) - R x L_1(0) - R x L_1(R) + R x (R x L_0) ... but with 1/2, 1/3 factors.
    # Correct relation for A = 1/3 Σ r x (r x ω):
    # A(R) = A(0) - np.cross(R, I_tgt_0) - np.cross(R, I_tgt_R) + (1.0/3.0) * np.cross(R, np.cross(R, G_tgt))
    A_tgt_R = (
        A_tgt_0
        - np.cross(R, I_tgt_0)
        - np.cross(R, I_tgt_R)
        + (1.0 / 3.0) * np.cross(R, np.cross(R, G_tgt))
    )

    # 5. Calculate residuals (deficits)
    R_Gamma = G_tgt - current_Gamma
    R_I = I_tgt_R - current_I
    R_A = A_tgt_R - current_A

    # Check if correction is needed
    # Select active residuals
    R_total_full = np.concatenate([R_Gamma, R_I, R_A])
    R_active = R_total_full[active_indices]

    # Check if correction is needed
    residual_norm = np.linalg.norm(R_active)
    if residual_norm < 1e-14:
        logger.debug("No correction needed (residual < 1e-14)")
        return circ, CorrectionStats(0, 0, 0, 0, 1.0, N)

    # 3. Compute particle weights (avoid division by zero)
    # WS3: volume weighting redistributes corrections uniformly per unit volume,
    # preventing the ∝|Γ| weighting from piling corrections onto strong particles
    # and causing "moving body" distortion (coupler_design.md §WS3).
    if volumes is not None and len(volumes) == N:
        weights = np.asarray(volumes, dtype=float)
        weights = np.maximum(weights, 1e-30)
    else:
        weights = np.linalg.norm(circ, axis=1)
    total_weight = np.sum(weights)

    if total_weight < 1e-14:
        logger.warning("All particles have near-zero circulation - cannot correct")
        return circ, CorrectionStats(0, 0, 0, residual_norm, np.inf, N)

    # 6. Build system matrix using 9-probe method (relative coordinates)
    M = np.zeros((9, 9))
    basis_vectors = np.eye(9)

    for i in range(9):
        L_vec = basis_vectors[i]
        L_Gamma = L_vec[0:3]
        L_I = L_vec[3:6]
        L_A = L_vec[6:9]

        # Compute induced correction for this basis vector
        term1 = L_Gamma
        term2 = 0.5 * np.cross(pos_rel, L_I)
        term3 = (1.0 / 3.0) * np.cross(pos_rel, np.cross(pos_rel, L_A))

        dGamma_probe = (term1 + term2 + term3) * weights[:, None]

        # Project back to invariants
        res_Gamma = np.sum(dGamma_probe, axis=0)
        x_cross_dG = np.cross(pos_rel, dGamma_probe)
        res_I = 0.5 * np.sum(x_cross_dG, axis=0)
        res_A = (1.0 / 3.0) * np.sum(np.cross(pos_rel, x_cross_dG), axis=0)

        M[:, i] = np.concatenate([res_Gamma, res_I, res_A])

    # Extract submatrix for active constraints
    M_active = M[np.ix_(active_indices, active_indices)]

    # 5. Guard against NaN/Inf before any LAPACK routines
    if (not np.all(np.isfinite(M_active))) or (not np.all(np.isfinite(R_active))):
        logger.warning("Correction skipped: non-finite entries in system matrix or residuals")
        return circ, CorrectionStats(0, 0, 0, residual_norm, np.inf, N)

    # 6. Solve the 9×9 system with FPE traps suspended.
    #    OpenFOAM enables hardware FPE traps (feenableexcept) which conflict
    #    with LAPACK's internal IEEE-754 compliance check (ieeeck divides by
    #    zero on purpose).  We disable the traps for the duration of the
    #    np.linalg calls and restore them afterward.
    with _suspend_fpe_traps():
        # 6a. Check condition number
        try:
            cond = np.linalg.cond(M_active)
            if cond > 1e12:
                logger.warning(
                    f"System matrix poorly conditioned (cond={cond:.2e}), skipping correction"
                )
                return circ, CorrectionStats(0, 0, 0, residual_norm, cond, N)
        except Exception:
            cond = np.inf
            logger.warning("Correction skipped: failed to compute condition number")
            return circ, CorrectionStats(0, 0, 0, residual_norm, cond, N)

        # 6b. Solve for Lagrange multipliers via least-squares
        try:
            lambdas_active, _, rank, _ = np.linalg.lstsq(M_active, R_active, rcond=None)
            if rank < n_constraints:
                logger.warning(f"System matrix rank deficient (rank={rank}/{n_constraints})")
        except np.linalg.LinAlgError as e:
            logger.error(f"Failed to solve correction system: {e}")
            return circ, CorrectionStats(0, 0, 0, residual_norm, np.inf, N)

    # Map active lambdas back to full 9-component vector
    lambdas = np.zeros(9)
    lambdas[active_indices] = lambdas_active

    L_Gamma = lambdas[0:3]
    L_I = lambdas[3:6]
    L_A = lambdas[6:9]

    # 7. Apply correction
    term1 = L_Gamma
    term2 = 0.5 * np.cross(pos_rel, L_I)
    term3 = (1.0 / 3.0) * np.cross(pos_rel, np.cross(pos_rel, L_A))

    correction = (term1 + term2 + term3) * weights[:, None]
    circ_corrected = circ + correction

    # 8. Compute statistics
    correction_norm = np.sqrt(np.mean(np.sum(correction**2, axis=1)))
    max_correction = np.max(np.linalg.norm(correction, axis=1))
    original_norm = np.linalg.norm(current_Gamma)
    relative_correction = correction_norm / (original_norm + 1e-16)

    # Verify correction worked
    new_Gamma = np.sum(circ_corrected, axis=0)
    new_x_cross_G = np.cross(pos_rel, circ_corrected)
    new_I = 0.5 * np.sum(new_x_cross_G, axis=0)
    new_A = (1.0 / 3.0) * np.sum(np.cross(pos_rel, new_x_cross_G), axis=0)

    residual_after = np.linalg.norm(
        np.concatenate([I_tgt_R - new_I, A_tgt_R - new_A, G_tgt - new_Gamma])
    )

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
