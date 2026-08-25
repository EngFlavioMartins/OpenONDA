"""Run the decisive Gaussian representability oracle on a captured cube handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from source.coupler.renewal_projection import (  # noqa: E402
    evaluate_sparse_gaussian_vorticity,
    solve_sparse_renewal_projection,
)


def _relative_rms(actual: np.ndarray, expected: np.ndarray, floor: float) -> float:
    residual = float(np.sqrt(np.mean((actual - expected) ** 2)))
    scale = max(float(np.sqrt(np.mean(expected**2))), floor, np.finfo(float).tiny)
    return residual / scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("oracle", type=Path)
    parser.add_argument("--tail-cutoff", type=float, default=1.0e-12)
    parser.add_argument("--tolerance", type=float, default=1.0e-12)
    parser.add_argument("--maximum-iterations", type=int, default=20_000)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    with np.load(arguments.oracle) as data:
        fit_position = np.asarray(data["fit_position"], dtype=np.float64)
        fit_target = np.asarray(data["fit_target"], dtype=np.float64)
        verification_position = np.asarray(data["verification_position"], dtype=np.float64)
        verification_target = np.asarray(data["verification_target"], dtype=np.float64)
        particle_position = np.asarray(data["solve_position"], dtype=np.float64)
        core_radius = np.asarray(data["solve_radius"], dtype=np.float64)
        prior = np.asarray(data["solve_prior"], dtype=np.float64)
        activity_floor = float(data["activity_floor"])
        production_fit_error = float(data["fit_error"])
        production_verification_error = float(data["verification_error"])

    collocation_position = np.vstack((fit_position, verification_position))
    target = np.vstack((fit_target, verification_target))
    result = solve_sparse_renewal_projection(
        collocation_position=collocation_position,
        target_vorticity=target,
        particle_position=particle_position,
        core_radius=core_radius,
        prior_vortex_strength=prior,
        relative_tail_cutoff=arguments.tail_cutoff,
        relative_tolerance=arguments.tolerance,
        maximum_iterations=arguments.maximum_iterations,
    )
    fit_field = evaluate_sparse_gaussian_vorticity(
        fit_position,
        particle_position,
        result.vortex_strength,
        core_radius,
        relative_tail_cutoff=arguments.tail_cutoff,
    )
    verification_field = evaluate_sparse_gaussian_vorticity(
        verification_position,
        particle_position,
        result.vortex_strength,
        core_radius,
        relative_tail_cutoff=arguments.tail_cutoff,
    )
    report = {
        "oracle": str(arguments.oracle.resolve()),
        "n_particles": int(len(particle_position)),
        "n_fit_points": int(len(fit_position)),
        "n_verification_points": int(len(verification_position)),
        "production_fit_error": production_fit_error,
        "production_verification_error": production_verification_error,
        "best_combined_fit_error": _relative_rms(fit_field, fit_target, activity_floor),
        "best_combined_verification_error": _relative_rms(
            verification_field,
            verification_target,
            activity_floor,
        ),
        "solver_converged": bool(result.converged),
        "solver_iterations": int(result.iteration_count),
        "condition_number_estimate": float(result.condition_number),
        "operator_nonzeros": int(result.operator_nonzeros),
        "maximum_strength": float(result.maximum_strength),
        "rms_strength": float(result.rms_strength),
        "maximum_to_rms_strength": float(result.maximum_to_rms_strength),
        "passes_vorticity_gate": bool(
            max(
                _relative_rms(fit_field, fit_target, activity_floor),
                _relative_rms(verification_field, verification_target, activity_floor),
            )
            < 5.0e-3
        ),
    }
    output = json.dumps(report, indent=2)
    print(output)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
