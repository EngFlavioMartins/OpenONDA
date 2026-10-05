#!/usr/bin/env python3
"""Run one isolated reference or coupled phase-benchmark stage (no cleaning)."""

import argparse
from functools import partial
import json
from pathlib import Path
import time

from . import phase_benchmark as case


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("reference", "coupled"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    # Keep the portable planar backend default and allow explicit device selection.
    parser.add_argument("--device", choices=("CPU", "CUDA", "VULKAN", "METAL"), default="CPU")
    args = parser.parse_args(argv)
    if args.kind == "reference" and args.pilot:
        parser.error("The reference runs the full declared horizon")
    return args


def _publish_result(path, result, started):
    """Write once after either shared run_stages has closed its solver."""
    from openonda.runtime import detected_world_size
    from source.coupler.parallel import collective_phase

    comm = None
    if detected_world_size() > 1:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    with collective_phase(comm, "publish phase stage result"):
        if comm is None or comm.Get_rank() == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            result.update(wall_seconds=time.monotonic() - started)
            path.write_text(json.dumps(result, indent=2) + "\n")


def main():
    args = parse_args()
    out = args.root.resolve() / args.kind
    label = "pilot" if args.pilot else "continuation" if args.resume else "run"
    result_path = out / f"{label}-result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    if not args.resume and any((out / n).exists() for n in ("samples", "solution")):
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")
    started = time.monotonic()
    try:
        if args.kind == "reference":
            reference = case.load_case_module(case.CASE / "reference_flow")
            with reference.create_solver(
                "phase_h004", case.H, output_root=out, end_time=case.END
            ) as solver:
                reference.run_solver(solver, start_from="latest" if args.resume else "initial")
                assert solver.run_status == "complete" and abs(solver.time - case.END) < 1e-8
                result = {"status": "completed", "time": solver.time, "step": solver.step}
        else:
            flow, particles, settings, mesh = case.coupled_case(end=case.END, device=args.device)
            initial = partial(
                case.module.cylinder_initial_velocity,
                freestream_velocity=case.module.STARTUP_FREESTREAM_VELOCITY,
                **case.module.INITIAL_PERTURBATION,
            )
            with case.module.coupling.create_coupler(
                flow, particles, settings, mesh=mesh, case_dir=out
            ) as solver:
                last = solver.run(
                    start_from="latest" if args.resume else "initial",
                    max_coupling_steps=20 if args.pilot else None,
                    backup_at_stop=True,
                    initial_velocity=initial,
                )
            expected = case.steps(case.END, case.EXCHANGE)
            if args.pilot:
                assert 0 < last <= expected
                if not args.resume:
                    assert last == min(20, expected)
            else:
                assert last == expected
            result = {
                "status": "pilot-completed" if args.pilot else "completed",
                "time": last * case.EXCHANGE,
                "step": last,
            }
    except BaseException as error:
        result = {"status": "failed", "error": repr(error)}
        raise
    finally:
        _publish_result(result_path, result, started)


if __name__ == "__main__":
    main()
