"""Screen a fixed-particle-basis Helmholtz correction on a saved 3D cube state.

This is a study-only normal-equation fit. It keeps all particle centres and
Gaussian cores fixed and does not apply the solver's invariant restoration,
projection acceptance gates, panel refresh, or coupled advancement. The
periodic padded grid is a diagnostic approximation to the free-space target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from cube_wake_particle_probe import direct_gaussian, rms
from cube_wake_vorticity_consistency import gaussian_vorticity_and_divergence
import h5py
from numba import set_num_threads
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from threadpoolctl import threadpool_limits

from source.solvers.vpm.stabilization.divergence_relaxation import (
    GaussianParticleGridOperator,
    _MomentNullspace,
    gaussian_invariant_rows,
)
from source.solvers.vpm.stabilization.filament_refinement import gaussian_particle_moments


def _curl(jacobian: np.ndarray) -> np.ndarray:
    """Return the shape-(N, 3) curl in s^-1 from J[i,j] = partial_j u_i."""
    return np.column_stack(
        (
            jacobian[:, 2, 1] - jacobian[:, 1, 2],
            jacobian[:, 0, 2] - jacobian[:, 2, 0],
            jacobian[:, 1, 0] - jacobian[:, 0, 1],
        )
    )


def _probe_points() -> np.ndarray:
    """Return the existing 392 three-dimensional cube-wake probe positions in m."""
    x = np.array([0.9, 1.14, 1.26, 1.38, 1.5, 1.62, 1.86, 2.34])
    yz = np.linspace(-0.54, 0.54, 7)
    return np.stack(np.meshgrid(x, yz, yz, indexing="ij"), axis=-1).reshape(-1, 3)


def _regional_metrics(
    points: np.ndarray,
    accepted_velocity: np.ndarray,
    reference_velocity: np.ndarray,
    before_velocity: np.ndarray,
    after_velocity: np.ndarray,
    before_jacobian: np.ndarray,
    after_jacobian: np.ndarray,
    before_vorticity: np.ndarray,
    after_vorticity: np.ndarray,
    before_divergence: np.ndarray,
    after_divergence: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Measure the unchanged 3D probe masks; all velocities are in m/s."""
    masks = {
        "authority_ramp": points[:, 0] < 1.25,
        "renewal_seam": (points[:, 0] >= 1.25) & (points[:, 0] <= 1.62),
        "outer_wake": points[:, 0] > 1.62,
    }
    before_curl = _curl(before_jacobian)
    after_curl = _curl(after_jacobian)
    baseline_particle_plus_freestream = accepted_velocity + np.array([1.0, 0.0, 0.0])
    estimated_particle_plus_freestream = (
        baseline_particle_plus_freestream + after_velocity - before_velocity
    )
    return {
        name: {
            "points": int(np.count_nonzero(mask)),
            "reference_error_before_particle_plus_freestream_rms_m_s": rms(
                (baseline_particle_plus_freestream - reference_velocity)[mask]
            ),
            "reference_error_after_particle_plus_freestream_rms_m_s": rms(
                (estimated_particle_plus_freestream - reference_velocity)[mask]
            ),
            "self_velocity_change_rms_m_s": rms((after_velocity - before_velocity)[mask]),
            "self_velocity_change_max_m_s": float(
                np.linalg.norm(after_velocity - before_velocity, axis=1)[mask].max()
            ),
            "velocity_curl_change_rms_s_inv": rms((after_curl - before_curl)[mask]),
            "vorticity_curl_mismatch_before_relative": rms((before_vorticity - before_curl)[mask])
            / max(rms(before_vorticity[mask]), np.finfo(float).tiny),
            "vorticity_curl_mismatch_after_relative": rms((after_vorticity - after_curl)[mask])
            / max(rms(after_vorticity[mask]), np.finfo(float).tiny),
            "represented_divergence_before_rms_s2_inv": float(
                np.sqrt(np.mean(before_divergence[mask] ** 2))
            ),
            "represented_divergence_after_rms_s2_inv": float(
                np.sqrt(np.mean(after_divergence[mask] ** 2))
            ),
        }
        for name, mask in masks.items()
    }


def screen(
    checkpoint: Path,
    accepted_fields: Path,
    reference_cache: Path,
    output: Path,
    *,
    regularization: float,
    preserve_moments: bool,
    freeze_renewal_belt: bool,
    expected_time: float,
) -> None:
    """Fit a fixed-basis correction and record its velocity/representation costs.

    The fitted equation is ``(V**0.5 K V**0.5 + lambda I) z =
    V**0.5 G(PB Gamma - B Gamma)``, with ``delta Gamma = V**0.5 z``.
    ``B`` is M4 scatter followed by Gaussian reconstruction, ``G`` its
    transpose gather, and ``P`` the padded periodic Helmholtz projector.
    ``Gamma`` has units m³/s, and the unchanged spacing is 0.06 m.
    """
    if output.exists():
        raise FileExistsError(output)
    if not np.isfinite(regularization) or regularization <= 0.0:
        raise ValueError("regularization must be positive and finite")
    with h5py.File(checkpoint) as state:
        saved_time = float(state["solver"].attrs["time"])
        particles = state["particles"]
        position, strength, core_radius, volume = (
            np.asarray(particles[name], dtype=np.float64)
            for name in ("position", "vortex_strength", "core_radius", "particle_volume")
        )
    if not np.isclose(saved_time, expected_time, rtol=0.0, atol=1e-8):
        raise ValueError(f"Expected a saved t={expected_time} cube state, got {saved_time}")
    if not np.allclose(core_radius, 0.066, rtol=0.0, atol=1e-7):
        raise ValueError("This screen requires the unchanged common Gaussian core of 0.066 m")
    if not np.allclose(volume, 0.06**3, rtol=0.0, atol=1e-9):
        raise ValueError("This screen requires the unchanged particle volume h³")
    if np.any(np.all(np.abs(position) < 0.5, axis=1)):
        raise ValueError("The saved accepted state contains particle centres inside the cube")

    points = _probe_points()
    with np.load(accepted_fields) as accepted:
        np.testing.assert_array_equal(accepted["points"], points)
        accepted_velocity = accepted["accepted_velocity"].astype(np.float64)
    with np.load(reference_cache) as reference:
        np.testing.assert_array_equal(reference["points"], points)
        reference_velocity = reference["reference_0"].astype(np.float64)

    started = perf_counter()
    operator = GaussianParticleGridOperator(
        position, core_radius, np.linalg.norm(strength, axis=1), spacing=0.06
    )
    residual, grid_divergence_before, _ = operator.relaxation_residual(strength)
    initial_residual = float(np.linalg.norm(residual))
    sqrt_volume = np.sqrt(volume)
    # The historical cube uses a [-1.25, 1.25]^3 transfer box and a 0.135 m
    # physical renewal buffer. Sources in this belt are absolute-replaced at
    # the next transfer, so an outer correction must leave them untouched.
    renewal_bounds = np.array([-1.385, 1.385] * 3)
    frozen = (
        np.all(
            (position >= renewal_bounds[::2]) & (position <= renewal_bounds[1::2]),
            axis=1,
        )
        if freeze_renewal_belt
        else np.zeros(len(position), dtype=bool)
    )
    mutable = ~frozen
    nullspace = (
        _MomentNullspace(
            gaussian_invariant_rows(position[mutable], core_radius[mutable]), volume[mutable]
        )
        if preserve_moments
        else None
    )
    iterations = 0

    def restrict(vector: np.ndarray) -> np.ndarray:
        result = np.zeros_like(vector)
        if nullspace is None:
            result[mutable] = vector[mutable]
        else:
            result[mutable] = nullspace.project_transformed(vector[mutable])
        return result

    def matrix_vector(vector: np.ndarray) -> np.ndarray:
        values = restrict(vector.reshape(-1, 3))
        applied = operator.apply(sqrt_volume[:, None] * values)
        transformed = sqrt_volume[:, None] * applied + regularization * values
        return restrict(transformed).ravel()

    def count_iteration(_vector: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1

    linear_operator = LinearOperator(
        (3 * len(position), 3 * len(position)), matvec=matrix_vector, dtype=np.float64
    )
    right_hand_side = restrict(sqrt_volume[:, None] * residual)
    solution, info = cg(
        linear_operator,
        right_hand_side.ravel(),
        rtol=1e-5,
        atol=0.0,
        maxiter=30,
        callback=count_iteration,
    )
    if info != 0:
        raise RuntimeError(f"Fixed-basis projection normal equation did not converge: {info}")
    correction = sqrt_volume[:, None] * restrict(solution.reshape(-1, 3))
    candidate = strength + correction
    residual_after, grid_divergence_after, _ = operator.relaxation_residual(candidate)
    elapsed_projection = perf_counter() - started

    before_velocity, before_jacobian = direct_gaussian(points, position, strength, core_radius)
    after_velocity, after_jacobian = direct_gaussian(points, position, candidate, core_radius)
    before_vorticity, before_divergence = gaussian_vorticity_and_divergence(
        points, position, strength, core_radius
    )
    after_vorticity, after_divergence = gaussian_vorticity_and_divergence(
        points, position, candidate, core_radius
    )
    before_moments = gaussian_particle_moments(position, strength, core_radius)
    after_moments = gaussian_particle_moments(position, candidate, core_radius)
    report = {
        "scope": __doc__,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "accepted_fields": str(accepted_fields.resolve()),
        "accepted_fields_sha256": hashlib.sha256(accepted_fields.read_bytes()).hexdigest(),
        "reference_cache": str(reference_cache.resolve()),
        "reference_cache_sha256": hashlib.sha256(reference_cache.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "physical_time_s": saved_time,
        "particle_count_before_and_after": len(position),
        "particle_centres_inside_cube_before_and_after": 0,
        "grid_shape": operator.shape,
        "grid_spacing_m": operator.spacing,
        "core_radius_m": 0.066,
        "regularization": regularization,
        "raw_particle_moments_constrained": preserve_moments,
        "freeze_renewal_belt": freeze_renewal_belt,
        "renewal_bounds_m": renewal_bounds.tolist() if freeze_renewal_belt else None,
        "frozen_particle_count": int(np.count_nonzero(frozen)),
        "mutable_particle_count": int(np.count_nonzero(mutable)),
        "correction_net_inside_renewal_belt_m3_s": correction[frozen].sum(axis=0).tolist(),
        "correction_net_outside_renewal_belt_m3_s": correction[mutable].sum(axis=0).tolist(),
        "cg_iterations": iterations,
        "projection_wall_seconds": elapsed_projection,
        "fixed_basis_residual_before_norm": initial_residual,
        "fixed_basis_residual_after_ratio": float(np.linalg.norm(residual_after))
        / max(initial_residual, np.finfo(float).tiny),
        "grid_divergence_before_relative": grid_divergence_before,
        "grid_divergence_after_relative": grid_divergence_after,
        "particle_strength_correction_relative": float(np.linalg.norm(correction))
        / max(float(np.linalg.norm(strength)), np.finfo(float).tiny),
        "net_strength_change_m3_s": (after_moments[0] - before_moments[0]).tolist(),
        "linear_impulse_change_m4_s": (after_moments[2] - before_moments[2]).tolist(),
        "angular_impulse_change_m5_s": (after_moments[3] - before_moments[3]).tolist(),
        "regions": _regional_metrics(
            points,
            accepted_velocity,
            reference_velocity,
            before_velocity,
            after_velocity,
            before_jacobian,
            after_jacobian,
            before_vorticity,
            after_vorticity,
            before_divergence,
            after_divergence,
        ),
        "reference_error_scope": "Particle-induced velocity plus the unit streamwise freestream. Body/panel response is omitted from both before and after; this is not complete VPM velocity or a coupled result.",
        "self_velocity_change_rms_m_s": rms(after_velocity - before_velocity),
        "accepted_self_velocity_rms_m_s": rms(accepted_velocity),
    }
    output.mkdir(parents=True, exist_ok=False)
    candidate_path = output / "candidate_strength.npz"
    np.savez_compressed(candidate_path, vortex_strength=candidate.astype(np.float32))
    report["candidate_strength_sha256"] = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    report["candidate_storage_dtype"] = "float32, matching the saved VPM particle state"
    quantized_moments = gaussian_particle_moments(
        position, candidate.astype(np.float32).astype(np.float64), core_radius
    )
    report["quantized_net_strength_change_m3_s"] = (
        quantized_moments[0] - before_moments[0]
    ).tolist()
    report["quantized_linear_impulse_change_m4_s"] = (
        quantized_moments[2] - before_moments[2]
    ).tolist()
    report["quantized_angular_impulse_change_m5_s"] = (
        quantized_moments[3] - before_moments[3]
    ).tolist()
    (output / "screen.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    """Run the frozen study without mutating any native or tutorial output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--accepted-fields", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regularization", type=float, default=0.1)
    parser.add_argument("--expected-time", type=float, default=6.0)
    parser.add_argument("--preserve-moments", action="store_true")
    parser.add_argument("--freeze-renewal-belt", action="store_true")
    args = parser.parse_args()
    set_num_threads(4)
    with threadpool_limits(limits=4):
        screen(
            args.checkpoint,
            args.accepted_fields,
            args.reference_cache,
            args.output,
            regularization=args.regularization,
            preserve_moments=args.preserve_moments,
            freeze_renewal_belt=args.freeze_renewal_belt,
            expected_time=args.expected_time,
        )


if __name__ == "__main__":
    main()
