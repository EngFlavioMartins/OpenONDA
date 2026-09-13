"""Run an isolated native rotor case with explicit resolution and resource records.

Completion of a short pilot is NOT aerodynamic qualification. Use the native
validator and refinement comparison for production-length evidence.
"""

import argparse
import cProfile
from dataclasses import replace
import json
from pathlib import Path
import pstats
import re
import sys
import time

import h5py

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import openonda.vpm as vpm
from tests._tutorial_helpers import load_tutorial_module


def main():
    """Preserve every attempt and let native run health govern completion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--dt", type=float, default=.006)
    parser.add_argument("--device", choices=("CPU", "METAL"), default="CPU")
    parser.add_argument("--wall-seconds", type=float, default=600)
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        raise ValueError("tag must be a simple name")
    directory = Path(__file__).with_name(args.tag)
    if directory.exists():
        raise FileExistsError(f"Preserve existing evidence: {directory}")
    setup = load_tutorial_module("vpm/rotor_flow")
    case = setup.build_case(time_step_size=args.dt, steps=args.steps)
    capacity = args.capacity or max(2048, args.steps * 135 + 256)
    declared_device = args.device
    if args.resume:
        with h5py.File(args.resume) as archive:
            saved = json.loads(archive["solver"].attrs["numerical_configuration"])
        capacity, declared_device = saved["max_n_particles"], saved["compute_device"]
        if args.capacity is not None and args.capacity != capacity:
            raise ValueError("restart must preserve checkpoint capacity; omit --capacity")
        recorded = json.loads((args.resume.parent / "vpm_metadata.json").read_text())
        groups = [surface["group_id"] for surface in recorded["configuration"]["numerics"]["vlm"]["surfaces"]]
        case = replace(case, numerics=replace(case.numerics, vlm=replace(
            case.numerics.vlm, surfaces=tuple(replace(surface, group_id=group) for surface, group in zip(case.numerics.vlm.surfaces, groups, strict=True))
        )))
        # Preserve the recorded execution settings as well as the physics;
        # a restart is not permission to bypass numerical-identity validation.
        tree = saved["induction"]
        case = replace(case, numerics=replace(case.numerics, induction=vpm.TreecodeInduction(
            theta=tree["theta"], multipole_order=tree["multipole_order"],
            stretching_scheme=tree["stretching_scheme"],
            sort_particle_targets=tree["sort_particle_targets"],
            traversal_block_dim=tree["traversal_block_dim"],
        )))
    case = replace(
        case, directory=directory,
        numerics=replace(case.numerics, compute_device=declared_device,
                         max_n_particles=capacity, max_evaluation_points=20000),
        run=replace(case.run, runtime_compute_device=args.device,
                    wall_time_limit_seconds=args.wall_seconds),
    )
    directory.mkdir(parents=True)
    result = {"request": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "status": "RUNNING", "scope": "native rotor run; qualification requires independent postprocessing"}
    started = time.perf_counter()
    profiler = cProfile.Profile()
    solver = None
    try:
        solver = vpm.VPMSolver(case)
        if args.resume:
            solver.load_backup(args.resume, time_step_size=args.dt)
        initial_step = solver.step
        profiler.runcall(solver.run)
        result.update(status="COMPLETED_REQUEST" if solver.step == initial_step + args.steps else "INCOMPLETE",
                      step=solver.step, time=solver.time, particles=len(solver.particles))
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        if solver is not None:
            result.update(step=solver.step, time=solver.time, particles=len(solver.particles))
        raise
    finally:
        result["wall_seconds"] = time.perf_counter() - started
        (directory / "run_summary.json").write_text(json.dumps(result, indent=2) + "\n")
        with (directory / "profile.txt").open("w") as output:
            profiler.create_stats()
            if profiler.stats:
                pstats.Stats(profiler, stream=output).sort_stats("cumtime").print_stats(50)
        if solver is not None:
            solver.close()


if __name__ == "__main__":
    main()
