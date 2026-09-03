#!/usr/bin/env python3
"""Diagnose the maintained rotor's accepted-step CFL exceedance."""

from __future__ import annotations

import csv
import json
import shutil

import numpy as np
from run_actual_rotor import RESULTS_DIR, rotor_setup
from scipy.spatial import cKDTree
import taichi as ti

from source.solvers.vpm.physics.stage_rhs import StageRates, StageState

DIAGNOSTIC_STEPS = {50, 53, 54}
OUTPUT = RESULTS_DIR / "actual_rotor_cfl_diagnosis.csv"


class RecordingRHS:
    def __init__(self, solver, wrapped) -> None:
        self.solver = solver
        self.wrapped = wrapped
        self.stage_number = 0
        self.accepted_step = None
        self.records: list[dict[str, object]] = []

    def evaluate(self, stage_state: StageState, stage_time: float, stage_rates: StageRates) -> None:
        self.wrapped.evaluate(stage_state, stage_time, stage_rates)
        current_step = self.solver.step + 1
        if self.accepted_step != current_step:
            self.accepted_step = current_step
            self.stage_number = 0
        self.stage_number += 1
        if current_step not in DIAGNOSTIC_STEPS:
            return
        count = int(stage_state.count)
        position = stage_state.position.to_numpy()[:count].astype(np.float64)
        strength = stage_state.vortex_strength.to_numpy()[:count].astype(np.float64)
        core_radius = stage_state.core_radius.to_numpy()[:count].astype(np.float64)
        complete_velocity = stage_rates.velocity.to_numpy()[:count].astype(np.float64)
        self_velocity = self.solver.induction.workspace.velocity.to_numpy()[:count].astype(
            np.float64
        )
        gradient = self.solver.induction.workspace.gradient.to_numpy()[:count].astype(np.float64)
        background = np.broadcast_to(
            self.solver.particles.velocity_background_cpu().astype(np.float64), (count, 3)
        ).copy()
        vlm_velocity = np.zeros((count, 3), dtype=np.float64)
        vlm_position = ti.Vector.field(3, dtype=self.solver.compute_dtype, shape=count)
        vlm_output = ti.Vector.field(3, dtype=self.solver.compute_dtype, shape=count)
        vlm_position.from_numpy(position.astype(self.solver.np_dtype))
        vlm_output.fill(0.0)
        self.solver.vlm_solver.add_stage_velocity(vlm_position, vlm_output, count, stage_time)
        vlm_velocity[:] = vlm_output.to_numpy()[:count]
        body_velocity = complete_velocity - self_velocity - background - vlm_velocity
        strain = 0.5 * (gradient + np.swapaxes(gradient, 1, 2))
        particle_cfl = self.solver.time_step_size * np.abs(strain).sum(axis=(1, 2))
        nearest = cKDTree(position).query(position, k=2)[0][:, 1]
        top = np.argsort(particle_cfl)[-20:][::-1]
        for index in top:
            self.records.append(
                {
                    "accepted_step": current_step,
                    "stage": self.stage_number,
                    "stage_time": float(stage_time),
                    "particle_index": int(index),
                    "group_id": int(self.solver.particles.group_id_cpu()[index]),
                    "zone_id": int(self.solver.particles.zone_id_cpu()[index]),
                    "source_surface": "unknown",
                    "particle_age_accepted_steps": "unknown",
                    "position": position[index].tolist(),
                    "vortex_strength": strength[index].tolist(),
                    "core_radius": float(core_radius[index]),
                    "nearest_neighbour_distance": float(nearest[index]),
                    "characteristic_length": "not used by health.py",
                    "complete_velocity": complete_velocity[index].tolist(),
                    "freestream_velocity_contribution": background[index].tolist(),
                    "vlm_velocity_contribution": vlm_velocity[index].tolist(),
                    "self_induced_fmm_velocity_contribution": self_velocity[index].tolist(),
                    "body_motion_contribution": body_velocity[index].tolist(),
                    "velocity_decomposition_residual": float(
                        np.linalg.norm(
                            complete_velocity[index]
                            - background[index]
                            - vlm_velocity[index]
                            - self_velocity[index]
                            - body_velocity[index]
                        )
                    ),
                    "calculated_lagrangian_cfl": float(particle_cfl[index]),
                }
            )


def main() -> int:
    case_dir = RESULTS_DIR / "actual_rotor_cfl_diagnosis"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    solver = rotor_setup.vpm.VPMSolver(
        rotor_setup.build_rotor_case(
            steps=54,
            max_n_particles=rotor_setup.MAX_N_PARTICLES,
            directory=case_dir,
            time_step_size=0.006,
        )
    )
    recorder = RecordingRHS(solver, solver.stage_rhs)
    solver.stage_rhs = recorder
    failure = None
    try:
        solver._build_initial_conditions()
        for _ in range(54):
            try:
                solver.advance()
            except Exception as error:
                failure = f"{type(error).__name__}: {error}"
                break
    finally:
        solver.close()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(recorder.records[0]))
        writer.writeheader()
        writer.writerows(recorder.records)
    metadata = {
        "health_formula": "cfl = time_step_size * max_i(sum_j(abs(S[i,j]))), S=0.5*(J+J.T)",
        "failure": failure,
        "records": len(recorder.records),
        "positive_finite_denominators": True,
        "diagnostic_note": "health.py uses no characteristic-length denominator; particle age and source surface are not fields in the maintained solver",
    }
    (OUTPUT.with_suffix(".json")).write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    return 0 if failure else 1


if __name__ == "__main__":
    raise SystemExit(main())
