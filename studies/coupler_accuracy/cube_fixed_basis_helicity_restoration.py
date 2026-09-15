"""Test whether an outer-only Helmholtz fit can preserve energy and helicity.

This is a frozen, three-dimensional operator study. It does not change a VPM
solver or its acceptance gates. The two moment-free restoration directions
solve the exact quadratic Gaussian Fourier energy and helicity equations after
the fixed-basis fit; the represented solenoidal enstrophy is audited separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cube_wake_particle_probe import direct_gaussian, rms
import h5py
from numba import set_num_threads
import numpy as np
from scipy import fft
from scipy.optimize import root
from threadpoolctl import threadpool_limits

from source.solvers.vpm.numerics.fourier_integrals import gaussian_fourier_integrals
from source.solvers.vpm.stabilization.divergence_relaxation import (
    GaussianParticleGridOperator,
    _MomentNullspace,
    _wave_numbers,
    gaussian_invariant_rows,
)
from source.solvers.vpm.stabilization.filament_refinement import gaussian_particle_moments


def _spectral_velocity(operator: GaussianParticleGridOperator, strength: np.ndarray) -> np.ndarray:
    """Return padded periodic Biot--Savart velocity in m/s from Gaussian strengths."""
    vorticity = operator.smooth(operator.scatter(strength))
    transformed = [fft.rfftn(vorticity[..., axis], workers=4) for axis in range(3)]
    wave = _wave_numbers(operator.shape, operator.spacing)
    squared = sum(component**2 for component in wave)
    inverse = np.zeros_like(squared)
    inverse[squared > 0] = 1.0 / squared[squared > 0]
    velocity_hat = [
        1j
        * (
            wave[(axis + 1) % 3] * transformed[(axis + 2) % 3]
            - wave[(axis + 2) % 3] * transformed[(axis + 1) % 3]
        )
        * inverse
        for axis in range(3)
    ]
    return np.stack(
        [fft.irfftn(component, s=operator.shape, workers=4).real for component in velocity_hat],
        axis=-1,
    )


def _solenoidal_enstrophy(operator: GaussianParticleGridOperator, strength: np.ndarray) -> float:
    """Return the padded-grid enstrophy of curl-compatible vorticity in m³/s²."""
    represented = operator.smooth(operator.scatter(strength))
    solenoidal, _, _ = operator.helmholtz_project(represented)
    return float(np.sum(solenoidal**2, dtype=np.float64) * operator.spacing**3)


def run(args: argparse.Namespace) -> None:
    """Restore two physical quadratic measures without changing frozen strengths."""
    if args.output.exists():
        raise FileExistsError(args.output)
    with h5py.File(args.checkpoint) as saved:
        particles = saved["particles"]
        position, strength, radius, volume = (
            np.asarray(particles[name], dtype=np.float64)
            for name in ("position", "vortex_strength", "core_radius", "particle_volume")
        )
    with np.load(args.candidate) as stored:
        fitted = stored["vortex_strength"].astype(np.float64)
    if fitted.shape != strength.shape:
        raise ValueError("The fitted strength cloud does not match the saved checkpoint")
    screen = json.loads((args.candidate.parent / "screen.json").read_text())
    if (
        screen["candidate_strength_sha256"]
        != hashlib.sha256(args.candidate.read_bytes()).hexdigest()
    ):
        raise ValueError("The fitted strength cloud does not match its provenance report")
    if screen["checkpoint_sha256"] != hashlib.sha256(args.checkpoint.read_bytes()).hexdigest():
        raise ValueError("The fitted strength cloud was obtained from another checkpoint")
    bounds = np.asarray(screen["renewal_bounds_m"], dtype=np.float64)
    frozen = np.all((position >= bounds[::2]) & (position <= bounds[1::2]), axis=1)
    mutable = ~frozen
    if not np.array_equal(fitted[frozen], strength[frozen]):
        raise ValueError("The fitted cloud modified the FVM renewal belt")
    operator = GaussianParticleGridOperator(
        position, radius, np.linalg.norm(strength, axis=1), spacing=0.06
    )
    sqrt_volume = np.sqrt(volume[mutable])
    nullspace = _MomentNullspace(
        gaussian_invariant_rows(position[mutable], radius[mutable]), volume[mutable]
    )
    baseline_norm = float(np.linalg.norm(strength))

    def direction(transformed: np.ndarray) -> np.ndarray:
        """Extend a mutable, nine-moment-free physical direction to the whole cloud."""
        physical = np.zeros_like(strength)
        physical[mutable] = nullspace.to_correction(transformed[mutable] / sqrt_volume[:, None])
        physical *= baseline_norm / max(float(np.linalg.norm(physical)), np.finfo(float).tiny)
        return physical

    energy_direction = direction(strength)
    # For H = integral u dot w, dH/dw = 2u on a periodic solenoidal grid.
    # Gaussian reconstruction and M4 scattering are transposed by smooth/gather.
    helicity_gradient = operator.gather(operator.smooth(_spectral_velocity(operator, fitted)))
    helicity_direction = direction(helicity_gradient)
    helicity_direction -= (
        np.vdot(helicity_direction, energy_direction) / np.vdot(energy_direction, energy_direction)
    ) * energy_direction
    helicity_norm = float(np.linalg.norm(helicity_direction))
    if helicity_norm < 1e-10 * baseline_norm:
        raise RuntimeError("The outer particle cloud has no independent helicity direction")
    helicity_direction *= baseline_norm / helicity_norm

    def measures(values: np.ndarray) -> np.ndarray:
        """Return exact Gaussian Fourier energy and helicity for a strength trial."""
        integral = gaussian_fourier_integrals(position, values, radius, volume, spacing=0.06)
        return np.array([integral.total_kinetic_energy, integral.total_helicity])

    baseline = measures(strength)
    values = np.stack(
        [
            measures(fitted),
            measures(fitted + energy_direction),
            measures(fitted - energy_direction),
            measures(fitted + helicity_direction),
            measures(fitted - helicity_direction),
            measures(fitted + energy_direction + helicity_direction),
        ],
        axis=1,
    )
    linear = np.stack(
        (0.5 * (values[:, 1] - values[:, 2]), 0.5 * (values[:, 3] - values[:, 4])), axis=1
    )
    diagonal = np.stack(
        (
            0.5 * (values[:, 1] + values[:, 2]) - values[:, 0],
            0.5 * (values[:, 3] + values[:, 4]) - values[:, 0],
        ),
        axis=1,
    )
    cross = 0.5 * (
        values[:, 5] - values[:, 0] - linear[:, 0] - linear[:, 1] - diagonal[:, 0] - diagonal[:, 1]
    )
    scales = np.maximum(np.abs(baseline), [1.0, 0.1])

    def error(coefficients: np.ndarray) -> np.ndarray:
        """Return normalized exact quadratic errors for energy and helicity."""
        first, second = coefficients
        predicted = (
            values[:, 0]
            + linear @ coefficients
            + diagonal[:, 0] * first**2
            + 2 * cross * first * second
            + diagonal[:, 1] * second**2
        )
        return (predicted - baseline) / scales

    initial = np.linalg.lstsq(
        linear / scales[:, None], (baseline - values[:, 0]) / scales, rcond=1e-12
    )[0]
    restored = root(error, initial, method="hybr")
    if not restored.success or not np.all(np.isfinite(restored.x)):
        raise RuntimeError(
            f"Energy/helicity restoration has no converged local root: {restored.message}"
        )
    correction = restored.x[0] * energy_direction + restored.x[1] * helicity_direction
    final = fitted + correction
    if not np.array_equal(final[frozen], strength[frozen]):
        raise AssertionError("Quadratic restoration modified a renewable-belt source")
    final_measures = measures(final)
    original_residual, _, _ = operator.relaxation_residual(strength)
    fitted_residual, _, _ = operator.relaxation_residual(fitted)
    final_residual, _, _ = operator.relaxation_residual(final)
    original_moments = gaussian_particle_moments(position, strength, radius)
    final_moments = gaussian_particle_moments(position, final, radius)
    with np.load(args.reference_cache) as cached:
        points = cached["points"].astype(np.float64)
        reference = cached["reference_0"].astype(np.float64)
    before_velocity, _ = direct_gaussian(points, position, strength, radius)
    fitted_velocity, _ = direct_gaussian(points, position, fitted, radius)
    final_velocity, _ = direct_gaussian(points, position, final, radius)
    seam = (points[:, 0] >= 1.25) & (points[:, 0] <= 1.62)
    outer = points[:, 0] > 1.62
    freestream = np.array([1.0, 0.0, 0.0])
    report = {
        "scope": __doc__,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        "reference_cache_sha256": hashlib.sha256(args.reference_cache.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "restoration_coefficients": restored.x.tolist(),
        "energy_helicity_before": baseline.tolist(),
        "energy_helicity_fitted": values[:, 0].tolist(),
        "energy_helicity_final": final_measures.tolist(),
        "quadratic_relative_errors": (np.abs(final_measures - baseline) / scales).tolist(),
        "solenoidal_enstrophy_before": _solenoidal_enstrophy(operator, strength),
        "solenoidal_enstrophy_final": _solenoidal_enstrophy(operator, final),
        "fitted_residual_ratio": float(
            np.linalg.norm(fitted_residual) / np.linalg.norm(original_residual)
        ),
        "final_residual_ratio": float(
            np.linalg.norm(final_residual) / np.linalg.norm(original_residual)
        ),
        "correction_relative": float(np.linalg.norm(final - strength) / baseline_norm),
        "moment_errors_net_linear_angular": [
            float(np.linalg.norm(final_moments[i] - original_moments[i])) for i in (0, 2, 3)
        ],
        "regions": {
            name: {
                "points": int(mask.sum()),
                "reference_error_before_particle_plus_freestream_rms_m_s": rms(
                    (before_velocity + freestream - reference)[mask]
                ),
                "reference_error_fitted_particle_plus_freestream_rms_m_s": rms(
                    (fitted_velocity + freestream - reference)[mask]
                ),
                "reference_error_restored_particle_plus_freestream_rms_m_s": rms(
                    (final_velocity + freestream - reference)[mask]
                ),
            }
            for name, mask in (("renewal_seam", seam), ("outer_wake", outer))
        },
        "limit": "Velocity comparisons omit the separately refreshed body panel; this test does not satisfy or alter the production total-variation and correction-norm gates.",
    }
    args.output.mkdir(parents=True)
    np.savez_compressed(
        args.output / "restored_strength.npz", vortex_strength=final.astype(np.float32)
    )
    (args.output / "restoration.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    """Run a nonmutating, frozen quadratic-restoration falsification screen."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_num_threads(4)
    with threadpool_limits(limits=4):
        run(args)


if __name__ == "__main__":
    main()
