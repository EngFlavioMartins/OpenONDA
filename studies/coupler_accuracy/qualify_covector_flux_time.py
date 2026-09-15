"""Qualify source-only time integration against a high-accuracy ODE solution.

Positions, cores and occupied-cell taper remain fixed. Both free-space psi and
the self-induced Jacobian evolve with strength, plus a solenoidal affine
background. DOP853 independently integrates the same nonlinear spatial ODE;
this assesses the midpoint integrator, not continuum spatial consistency or
the order of a full transport/remeshing/renewal composition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cube_covector_flux_step import SourceCells, midpoint_increment
from cube_wake_particle_probe import direct_gaussian, rms
from numba import set_num_threads
import numpy as np
from scipy.integrate import solve_ivp
from threadpoolctl import threadpool_limits


def run(output: Path) -> None:
    """Check temporal convergence and actual integrated circulation/impulse."""
    output.mkdir(parents=True, exist_ok=False)
    axis = (np.arange(8) - 3.5) * 0.06
    position = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    radius = np.full(len(position), 0.066)
    cells = SourceCells.from_particles(position, radius)
    envelope = np.exp(-np.sum(position**2, axis=1) / 0.12**2)
    initial = 0.06**3 * envelope[:, None] * (3 * position / 0.12 + [0, 0, 0.8])
    background = np.array([[0.3, -0.7, 0], [0.7, -0.3, 0], [0, 0, 0]])
    stages = []

    def rate(strength):
        _, jacobian = direct_gaussian(position, position, strength, radius)
        source, budget = cells.rate(strength, jacobian + background)
        stages.append(budget)
        return source

    def ode(_, value):
        return rate(value.reshape(initial.shape)).ravel()

    references = []
    for relative in (1e-10, 1e-12):
        solution = solve_ivp(
            ode, (0, 0.4), initial.ravel(), method="DOP853", rtol=relative, atol=relative * 1e-5
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        references.append(solution.y[:, -1].reshape(initial.shape))
    rows = []
    for duration in (0.1, 0.05, 0.025):
        state = initial.copy()
        integrated_impulse = np.zeros(3)
        stage_pairs = []
        for _ in range(round(0.4 / duration)):
            stages.clear()
            increment = midpoint_increment(state, duration, rate)
            integrated_impulse += duration * np.asarray(
                stages[-1]["source_impulse_rate_per_density"]
            )
            stage_pairs.append([stage["strength_sha256"] for stage in stages])
            state += increment
        rows.append(
            {
                "duration": duration,
                "strength_error_rms": rms(state - references[-1]),
                "net_strength_change": (state - initial).sum(axis=0).tolist(),
                "impulse_balance_error": (
                    0.5 * np.cross(position, state - initial).sum(axis=0) - integrated_impulse
                ).tolist(),
                "midpoint_recomputed": all(first != second for first, second in stage_pairs),
            }
        )
    for index in (1, 2):
        rows[index]["convergence_order"] = float(
            np.log2(rows[index - 1]["strength_error_rms"] / rows[index]["strength_error_rms"])
        )
    reference_error = rms(references[0] - references[1])
    checks = {
        "second_order_source": rows[-1]["convergence_order"] > 1.8,
        "reference_resolved": reference_error < 0.01 * rows[-1]["strength_error_rms"],
        "net_conserved": max(np.linalg.norm(row["net_strength_change"]) for row in rows) < 1e-13,
        "impulse_accounted": max(np.linalg.norm(row["impulse_balance_error"]) for row in rows)
        < 1e-13,
        "midpoint_state_updated": all(row["midpoint_recomputed"] for row in rows),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    report = {
        "scope": __doc__,
        "particles": len(position),
        "reference_difference_rms": reference_error,
        "rows": rows,
        "checks": checks,
        "passed": all(checks.values()),
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(__file__).with_name("cube_covector_flux_step.py"),
                Path(__file__).with_name("cube_covector_flux_control.py"),
            )
        },
    }
    (output / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise AssertionError("Compact source failed temporal qualification")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_num_threads(2)
    with threadpool_limits(limits=2):
        run(args.output)
