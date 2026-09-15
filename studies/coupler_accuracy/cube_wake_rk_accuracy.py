"""Check actual temporary RK states and one-step time refinement in 3D.

Only the inviscid operator is applied; body strengths and Gaussian radii are
held fixed. Both half steps start from the same accepted backup as the full
step. No diffusion/remeshing cadence changes enter this comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from cube_wake_operator_audit import snapshot
from cube_wake_particle_probe import direct_gaussian, load_case, rms
from numba import set_num_threads
import numpy as np
from threadpoolctl import threadpool_limits


class StageObserver:
    """Delegate the native RHS unchanged and copy each evaluated temporary state."""

    def __init__(self, solver):
        self.solver = solver
        self.records = []

    def __getattr__(self, name):
        return getattr(self.solver.stage_rhs, name)

    def evaluate(self, state, time, rates):
        self.solver.stage_rhs.evaluate(state, time, rates)
        download = self.solver.physics._download_vector_field
        self.records.append(
            {
                "stage_index": state.stage_index,
                "time": time,
                "position": download(state.position, state.count).copy(),
                "strength": download(state.vortex_strength, state.count).copy(),
                "velocity": download(rates.velocity, state.count).copy(),
                "rate": download(rates.vortex_strength_rate, state.count).copy(),
            }
        )


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    case = load_case(args.source_tree)
    from source.solvers.vpm import VPMSolver

    policy = replace(
        case.VPM_CASE,
        directory=args.output / "runtime",
        samplers=case.vpm.Samplers(),
        numerics=replace(
            case.VPM_CASE.numerics, max_n_particles=300000, max_evaluation_points=300000
        ),
    )
    report = {"description": __doc__, "times": []}
    solver = VPMSolver(policy)
    try:
        for time in args.times:
            path = args.solution / f"vpm_{round(100 * time):06d}.h5"
            solver.load_backup(path)
            solver.refresh_boundary_element_solution()
            original = snapshot(solver)
            p = original["position"]
            candidates = np.flatnonzero((p[:, 0] >= 1.25) & (p[:, 0] <= 1.65))
            selected = np.random.default_rng(927).choice(
                candidates, min(192, len(candidates)), replace=False
            )
            observer = StageObserver(solver)
            solver.integrator.advance(
                position=solver.particles.position,
                vortex_strength=solver.particles.vortex_strength,
                core_radius=solver.particles.core_radius,
                count=len(p),
                time=time,
                time_step_size=0.01,
                right_hand_side=observer,
            )
            solver.particles.touch_state()
            full = snapshot(solver)
            row = {"time": time, "temporary_stages": []}
            assert len(observer.records) == 2
            for record in observer.records:
                stage_p, stage_g = (
                    record["position"].astype(float),
                    record["strength"].astype(float),
                )
                direct_u, direct_j = direct_gaussian(
                    stage_p[selected], stage_p, stage_g, original["core_radius"]
                )
                body_u = solver.panel_solver.compute_induced_velocity(stage_p[selected])
                body_j = solver.panel_solver.compute_induced_velocity_gradient(stage_p[selected])
                rate = np.einsum("nji,nj->ni", direct_j + body_j, stage_g[selected])
                row["temporary_stages"].append(
                    {
                        "stage": record["stage_index"],
                        "time": record["time"],
                        "velocity_error_rms": rms(
                            record["velocity"][selected] - direct_u - body_u - [1, 0, 0]
                        ),
                        "stretching_relative_error": rms(record["rate"][selected] - rate)
                        / rms(rate),
                        "maximum_position_change_from_accepted": float(
                            np.max(np.linalg.norm(stage_p - p, axis=1))
                        ),
                    }
                )
            solver.load_backup(path)
            solver.refresh_boundary_element_solution()
            for offset in (0.0, 0.005):
                solver.time = time + offset
                solver.stepper._advance_particles(0.005)
            half = snapshot(solver)
            with np.load(args.operator_audit / f"fields_t{time:g}.npz") as data:
                points = data["points"].copy()
            full_u, _ = direct_gaussian(
                points, full["position"], full["vortex_strength"], full["core_radius"]
            )
            half_u, _ = direct_gaussian(
                points, half["position"], half["vortex_strength"], half["core_radius"]
            )
            mask = (points[:, 0] >= 1.25) & (points[:, 0] <= 1.62)
            row["full_vs_two_half_velocity_rms"] = rms(full_u - half_u)
            row["seam_full_vs_two_half_velocity_rms"] = rms((full_u - half_u)[mask])
            row["seam_full_vs_two_half_transverse_rms"] = rms((full_u - half_u)[mask, 1:])
            np.savez_compressed(
                args.output / f"fields_t{time:g}.npz",
                points=points,
                full_velocity=full_u,
                half_velocity=half_u,
            )
            report["times"].append(row)
            (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    finally:
        solver.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--operator-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--times", type=float, nargs="+", default=[6, 8])
    args = parser.parse_args()
    set_num_threads(2)
    with threadpool_limits(limits=2):
        run(args)


if __name__ == "__main__":
    main()
