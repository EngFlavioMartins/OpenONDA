"""Unqualified represented-residual experiment; never selected by production.

It preserves a consistent Gaussian field algebraically, but unregularized
iteration amplified particle strengths in the frozen 3D cube experiment.
Keep this implementation available to reproduce that failed candidate.
"""

from __future__ import annotations

import numpy as np

from source.coupler.stable_renewal import (
    DEFAULT_AMPLIFICATION_CAP,
    RepresentedStateBlend,
    _vectors,
    gaussian_represented_vortex_strength,
)


def blend_represented_state(
    vpm_vortex_strength: np.ndarray,
    fvm_target_vortex_strength: np.ndarray,
    fvm_authority: np.ndarray,
    shape: tuple[int, int, int],
    particle_spacing: float,
    *,
    core_radius: float,
    amplification_cap: float = DEFAULT_AMPLIFICATION_CAP,
    output_weight: np.ndarray | None = None,
    compute_final_representation: bool = True,
) -> RepresentedStateBlend:
    """Correct the represented-field residual without resetting particle coefficients.

    ``fvm_target_vortex_strength`` contains physical vorticity samples times
    cell volume, whereas ``vpm_vortex_strength`` contains Gaussian coefficients.
    An existing field that already represents the target must be a fixed point.
    The optional second correction approximately deconvolves only the update.
    """
    cap = float(amplification_cap)
    if not np.isfinite(cap) or cap < 1.0:
        raise ValueError("amplification_cap must be finite and at least one")
    vpm_strength = _vectors("vpm_vortex_strength", vpm_vortex_strength)
    fvm_strength = _vectors("fvm_target_vortex_strength", fvm_target_vortex_strength)
    authority = np.asarray(fvm_authority, dtype=np.float64).reshape(-1)
    if vpm_strength.shape != fvm_strength.shape or authority.shape != (len(vpm_strength),):
        raise ValueError("blend inputs must share one lattice shape")
    weight: np.ndarray | None = None
    if output_weight is not None:
        weight = np.asarray(output_weight, dtype=np.float64).reshape(-1)
        if weight.shape != (len(vpm_strength),):
            raise ValueError("output_weight must share the transfer lattice shape")

    represented_vpm = gaussian_represented_vortex_strength(
        vpm_strength,
        shape,
        particle_spacing,
        core_radius=core_radius,
    )
    physical_target = represented_vpm + authority[:, None] * (fvm_strength - represented_vpm)

    blended_strength = vpm_strength + authority[:, None] * (fvm_strength - represented_vpm)
    represented_blend = gaussian_represented_vortex_strength(
        blended_strength,
        shape,
        particle_spacing,
        core_radius=core_radius,
    )
    residual = physical_target - represented_blend
    denominator = float(np.linalg.norm(physical_target)) + 1.0e-30
    residual_before = float(np.linalg.norm(residual)) / denominator

    correction_gain = min(cap - 1.0, 1.0)
    corrected_strength = blended_strength + correction_gain * residual
    if weight is not None:
        corrected_strength = corrected_strength * weight[:, None]
    target_maximum = float(np.linalg.norm(physical_target, axis=1).max(initial=0.0)) + 1.0e-30
    maximum_amplification = (
        float(np.linalg.norm(corrected_strength, axis=1).max(initial=0.0)) / target_maximum
    )

    represented_corrected: np.ndarray | None = None
    residual_after: float | None = None
    if compute_final_representation:
        represented_corrected = gaussian_represented_vortex_strength(
            corrected_strength,
            shape,
            particle_spacing,
            core_radius=core_radius,
        )
        residual_after = (
            float(np.linalg.norm(physical_target - represented_corrected)) / denominator
        )
    return RepresentedStateBlend(
        vortex_strength=corrected_strength,
        physical_target=physical_target,
        represented_vortex_strength=represented_corrected,
        residual_before_correction=residual_before,
        residual_after_correction=residual_after,
        maximum_amplification=maximum_amplification,
    )
