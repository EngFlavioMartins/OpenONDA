"""Exercise the authored rotor for ten startup steps; not a physical validation."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import openonda.vpm as vpm
from tests._tutorial_helpers import load_tutorial_module


def main():
    """Run real three-blade geometry and native outputs in an isolated directory."""
    setup = load_tutorial_module("vpm/rotor_flow")
    directory = Path(__file__).with_name("rotor_smoke")
    if directory.exists():
        raise FileExistsError(f"Preserve existing pilot: {directory}")
    case = setup.build_case(steps=10)
    case = replace(
        case,
        directory=directory,
        numerics=replace(
            case.numerics, compute_device="CPU", max_n_particles=2048, max_evaluation_points=20000
        ),
        run=replace(case.run, runtime_compute_device="CPU", wall_time_limit_seconds=180),
    )
    started = time.perf_counter()
    solver = vpm.VPMSolver(case)
    try:
        solver.run()
        result = {
            "status": "STARTUP_SMOKE_ONLY" if solver.step == 10 else "INCOMPLETE_WALL_TIME_LIMIT",
            "requested_steps": 10,
            "step": solver.step,
            "time": solver.time,
            "particles": len(solver.particles),
            "wall_seconds": time.perf_counter() - started,
        }
        Path(__file__).with_name("rotor_smoke_summary.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    finally:
        solver.close()


if __name__ == "__main__":
    main()
