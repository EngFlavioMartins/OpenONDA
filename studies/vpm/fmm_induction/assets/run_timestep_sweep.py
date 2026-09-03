#!/usr/bin/env python3
"""Empirically select a safe rotor timestep at the 1.536 s readiness horizon."""

from __future__ import annotations

import json
import shutil
import time

import numpy as np
from run_actual_rotor import RESULTS_DIR, rotor_setup

CANDIDATES = (0.004, 0.003, 0.002, 0.0015, 0.001)
TEST_TIME = 1.536
RELEASE_INTERVAL = rotor_setup.RELEASE_INTERVAL


def _forces(solver) -> list[float]:
    forces = solver.vlm_solver._last_forces if solver.vlm_solver is not None else {}
    return [
        float(forces.get("force_coefficient_x", 0.0)),
        float(forces.get("force_coefficient_y", 0.0)),
        float(forces.get("force_coefficient_z", 0.0)),
    ]


def _run(time_step_size: float) -> dict[str, object]:
    macro_steps = round(TEST_TIME / RELEASE_INTERVAL)
    substeps = round(RELEASE_INTERVAL / time_step_size)
    steps = macro_steps * substeps
    case_dir = RESULTS_DIR / f"timestep_{time_step_size:g}"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    solver = rotor_setup.vpm.VPMSolver(
        rotor_setup.build_rotor_case(
            steps=steps,
            max_n_particles=rotor_setup.MAX_N_PARTICLES,
            directory=case_dir,
            time_step_size=time_step_size,
            wake_spacing=rotor_setup.FIXED_WAKE_SPACING,
        )
    )
    records = []
    maximum_cfl = 0.0
    started = time.perf_counter()
    failure = None
    operational_failure = None
    try:
        solver._build_initial_conditions()
        for _ in range(macro_steps):
            for substep in range(substeps):
                solver._release_wake_particles = substep == 0
                solver._release_interval = RELEASE_INTERVAL
                step_started = time.perf_counter()
                try:
                    solver.advance(defer_output=True)
                    solver._refresh_accepted_step_health()
                except Exception as error:
                    failure = f"{type(error).__name__}: {error}"
                    break
                health = solver._accepted_health_snapshot
                cfl = float(health.strain_increment_infinity if health is not None else 0.0)
                maximum_cfl = max(maximum_cfl, cfl)
                position = solver.particle_position
                strength = solver.particle_vortex_strength
                gradient = solver.particle_velocity_gradient
                strain = 0.5 * (gradient + np.swapaxes(gradient, 1, 2))
                strength_norm = np.linalg.norm(strength, axis=1)
                rate = np.einsum("nji,nj->ni", gradient, strength)
                records.append(
                    {
                        "step": solver.step,
                        "time": solver.time,
                        "particle_count": solver.particles.n_particles_total,
                        "cfl": cfl,
                        "chi_s": float(time_step_size * np.abs(strain).sum(axis=(1, 2)).max()),
                        "chi_Gamma": float(
                            (
                                time_step_size
                                * np.linalg.norm(rate, axis=1)
                                / np.maximum(strength_norm, 1.0e-30)
                            ).max()
                        ),
                        "maximum_strength": float(strength_norm.max(initial=0.0)),
                        "total_strength": strength.sum(axis=0).astype(float).tolist(),
                        "force_coefficients": _forces(solver),
                        "circulation_norm": float(
                            np.linalg.norm(
                                solver.vlm_solver.lattice.circulation.to_numpy()[
                                    : solver.vlm_solver.lattice.n_panels
                                ]
                            )
                        ),
                        "wake_centroid": position.mean(axis=0).astype(float).tolist(),
                        "rate_defect": float(
                            solver.induction.diagnostics.last_relative_rate_defect
                        ),
                        "step_seconds": time.perf_counter() - step_started,
                    }
                )
                if cfl > 0.80 and operational_failure is None:
                    operational_failure = (
                        f"operational strain increment exceeded: {cfl:.6g} at step {solver.step}"
                    )
            if failure is not None:
                break
        if failure is None:
            solver._release_wake_particles = True
    finally:
        solver.close()
    result = {
        "time_step_size": time_step_size,
        "requested_steps": steps,
        "accepted_steps": len(records),
        "final_time": records[-1]["time"] if records else 0.0,
        "maximum_lagrangian_cfl": maximum_cfl,
        "elapsed_seconds": time.perf_counter() - started,
        "failure": failure,
        "operational_failure": operational_failure,
        "records": records,
    }
    (case_dir / "timestep_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
    return result


def main() -> int:
    results = [_run(candidate) for candidate in CANDIDATES]
    (RESULTS_DIR / "actual_rotor_timestep_sweep.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
