"""Bounded two-wing temporal self-convergence with fixed geometry and core rule.

This is a small coupled implementation check, not rotor qualification. Both
surfaces rotate and share the free wake. The span-controlled overlap core stays
fixed while dt changes. Native health limits and primary state are untouched.
"""

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tests/vpm"))

from test_vlm_production_contract import _case

import openonda.vpm as vpm


def main():
    """Compare final circulation, force and all components at fixed flow probes."""
    output = Path(__file__).with_name("coupled_time_refinement")
    output.mkdir(exist_ok=False)
    targets = np.array(
        [[x, y, z] for x in (0.25, 1.4, 2.8, 4.0) for y in (-0.7, 0.25) for z in (0.2, 0.7)]
    )
    rows = []
    for policy in ("lagged", "responsive"):
        for dt in (0.02, 0.01, 0.005, 0.0025):
            directory = output / f"{policy}_{dt}"
            case = _case(directory, responsive=(policy == "responsive"))
            case = replace(
                case,
                backup=vpm.Backup(interval_steps=0),
                numerics=replace(
                    case.numerics, time_step_size=dt, max_n_particles=2048, integrator=vpm.SSPRK3()
                ),
            )
            solver = vpm.VPMSolver(case)
            try:
                for _ in range(round(0.2 / dt)):
                    solver.advance()
                lattice = solver.vlm_solver.lattice
                rows.append(
                    {
                        "policy": policy,
                        "dt": dt,
                        "step": solver.step,
                        "time": solver.time,
                        "particles": len(solver.particles),
                        "circulation": lattice.circulation.to_numpy()[: lattice.n_panels].tolist(),
                        "force": lattice.panel_force.to_numpy()[: lattice.n_panels]
                        .sum(axis=0)
                        .tolist(),
                        "probe_velocity": solver.compute_velocity_at_points(targets).tolist(),
                        "core_range": [
                            float(np.min(solver.particles.core_radius_cpu())),
                            float(np.max(solver.particles.core_radius_cpu())),
                        ],
                    }
                )
                (output / "runs.json").write_text(json.dumps(rows, indent=2) + "\n")
            finally:
                solver.close()
    summary = {"scope": __doc__, "targets": targets.tolist(), "policies": {}}
    for policy in ("lagged", "responsive"):
        levels = [row for row in rows if row["policy"] == policy]
        metrics = {}
        for field in ("circulation", "force", "probe_velocity"):
            values = [np.asarray(row[field]) for row in levels]
            changes = [
                float(np.linalg.norm(a - b)) for a, b in zip(values, values[1:], strict=False)
            ]
            metrics[field] = {
                "successive_absolute_l2_changes": changes,
                "observed_orders": np.log2(np.array(changes[:-1]) / changes[1:]).tolist(),
                "last_relative_l2_change": changes[-1] / float(np.linalg.norm(values[-1])),
            }
        summary["policies"][policy] = metrics
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
