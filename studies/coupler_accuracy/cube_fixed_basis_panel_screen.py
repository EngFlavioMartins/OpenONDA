"""Measure a fixed-basis cube projection through the historical VPM body path.

This frozen-state screen changes only the vortex strengths of one saved
cloud in a private solver instance. It does not advance either solver or alter
the tutorial outputs. It uses the original checkpoint-compatible source tree.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

from cube_wake_particle_probe import load_case, rms
from numba import set_num_threads
import numpy as np
from threadpoolctl import threadpool_limits


def _regions(points: np.ndarray, velocity: np.ndarray, reference: np.ndarray) -> dict:
    """Return complete-velocity RMS errors on the fixed three-dimensional masks."""
    masks = {
        "authority_ramp": points[:, 0] < 1.25,
        "renewal_seam": (points[:, 0] >= 1.25) & (points[:, 0] <= 1.62),
        "outer_wake": points[:, 0] > 1.62,
    }
    return {
        name: {
            "points": int(mask.sum()),
            "complete_velocity_error_rms_m_s": rms((velocity - reference)[mask]),
            "transverse_velocity_error_rms_m_s": rms((velocity - reference)[mask, 1:]),
        }
        for name, mask in masks.items()
    }


def run(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    with (args.candidate.parent / "screen.json").open() as handle:
        fit_report = json.load(handle)
    if (
        hashlib.sha256(args.candidate.read_bytes()).hexdigest()
        != fit_report["candidate_strength_sha256"]
    ):
        raise ValueError("Candidate strength checksum does not match its fitting report")
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != fit_report["checkpoint_sha256"]:
        raise ValueError("Candidate was fitted to a different particle checkpoint")
    if (
        hashlib.sha256(args.reference_cache.read_bytes()).hexdigest()
        != fit_report["reference_cache_sha256"]
    ):
        raise ValueError("Candidate was fitted against a different reference cache")

    # This must precede imports from source: the checkpoint schema is historical.
    case = load_case(args.source_tree)
    from source.solvers.vpm import VPMSolver

    with np.load(args.candidate) as stored:
        candidate = stored["vortex_strength"].astype(np.float32)
    with np.load(args.reference_cache) as stored:
        points = stored["points"].astype(np.float64)
        reference = stored["reference_0"].astype(np.float64)
    if not np.all(np.isfinite(candidate)):
        raise ValueError("Candidate has non-finite vortex strengths")
    if not np.all(np.isfinite(reference)):
        raise ValueError("Fine reference has non-finite probe values")

    policy = replace(
        case.VPM_CASE,
        directory=args.output / "runtime",
        samplers=case.vpm.Samplers(),
        numerics=replace(
            case.VPM_CASE.numerics,
            compute_device=args.device,
            max_n_particles=300000,
            max_evaluation_points=300000,
        ),
    )
    solver = VPMSolver(policy)
    started = perf_counter()
    try:
        solver.load_backup(args.checkpoint)
        if len(solver.particles) != len(candidate):
            raise ValueError("Candidate particle count does not match restored solver")
        if abs(solver.time - args.expected_time) > 1e-8:
            raise ValueError(f"Expected the saved t={args.expected_time} cube state")
        solver.physics.configure_body_box(np.array([-0.5, 0.5] * 3))
        solver.physics.configure_grid_lattice_anchor(np.array([-0.03] * 3), 0.06)
        panel_scope = None if solver.panel_solver is None else solver.panel_solver.coupling_scope
        solver.refresh_boundary_element_solution()
        baseline_particle_plus_freestream = solver.compute_velocity_at_points(
            points, include_body=False
        ).astype(np.float64)
        baseline_complete = solver.compute_velocity_at_points(points).astype(np.float64)
        baseline_time = perf_counter() - started
        solver.set_particles_properties(vortex_strength=candidate)
        solver.notify_external_particle_mutation()
        candidate_particle_plus_freestream = solver.compute_velocity_at_points(
            points, include_body=False
        ).astype(np.float64)
        solver.refresh_boundary_element_solution()
        candidate_complete = solver.compute_velocity_at_points(points).astype(np.float64)
        candidate_time = perf_counter() - started - baseline_time
        report = {
            "scope": __doc__,
            "source_tree": str(args.source_tree.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": fit_report["checkpoint_sha256"],
            "candidate_strength_sha256": fit_report["candidate_strength_sha256"],
            "reference_cache_sha256": fit_report["reference_cache_sha256"],
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "time_s": float(solver.time),
            "panel_coupling_scope": panel_scope,
            "study_compute_device": solver.compute_device,
            "baseline_probe_wall_s": baseline_time,
            "candidate_probe_wall_s": candidate_time,
            "body_correction_before_rms_m_s": rms(
                baseline_complete - baseline_particle_plus_freestream
            ),
            "body_correction_after_rms_m_s": rms(
                candidate_complete - candidate_particle_plus_freestream
            ),
            "body_correction_change_rms_m_s": rms(
                (candidate_complete - candidate_particle_plus_freestream)
                - (baseline_complete - baseline_particle_plus_freestream)
            ),
            "complete_velocity_change_rms_m_s": rms(candidate_complete - baseline_complete),
            "complete_velocity_change_max_m_s": float(
                np.max(np.linalg.norm(candidate_complete - baseline_complete, axis=1))
            ),
            "before": _regions(points, baseline_complete, reference),
            "after": _regions(points, candidate_complete, reference),
            "interpretation_limit": (
                f"Frozen t={args.expected_time:g} state only; no particle evolution, "
                "transfer, FVM boundary feedback, Cd, or stability claim."
            ),
        }
        np.savez_compressed(
            args.output / "fields.npz",
            points=points,
            reference=reference,
            baseline_complete=baseline_complete,
            candidate_complete=candidate_complete,
            baseline_particle_plus_freestream=baseline_particle_plus_freestream,
            candidate_particle_plus_freestream=candidate_particle_plus_freestream,
        )
        (args.output / "screen.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        solver.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("AUTO", "CPU"), default="AUTO")
    parser.add_argument("--expected-time", type=float, default=6.0)
    args = parser.parse_args()
    set_num_threads(4)
    with threadpool_limits(limits=4):
        run(args)


if __name__ == "__main__":
    main()
