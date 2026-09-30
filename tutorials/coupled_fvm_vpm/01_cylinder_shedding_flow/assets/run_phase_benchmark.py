#!/usr/bin/env python3
"""Run one isolated reference or coupled phase-benchmark stage (no cleaning)."""
import argparse
import json
from pathlib import Path
import time

import phase_benchmark as case


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("reference", "coupled"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    # The case uses FMM beneath SlipSlabInduction. CUDA is not a qualified
    # backend for that evaluator; keep the default aligned with coupled_case.
    parser.add_argument("--device", choices=("CPU", "VULKAN", "METAL"), default="CPU")
    args = parser.parse_args(argv)
    if args.kind == "reference" and args.pilot:
        parser.error("The reference runs the full declared horizon")
    return args


def main():
    args = parse_args()
    out = args.root.resolve() / args.kind
    label = "pilot" if args.pilot else "continuation" if args.resume else "run"
    result_path = out / f"{label}-result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    if not args.resume and any((out / n).exists() for n in ("samples", "solution")):
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")
    from openonda.cylinder_campaign import initialize_cylinder_perturbation
    started = time.monotonic()
    if args.kind == "reference":
        setup, mesh = case.reference_case()
        with case.fvm.create_fvm_solver(setup, case_dir=out, mesh=mesh) as solver:
            if not args.resume:
                initialize_cylinder_perturbation(solver, case.SPAN)
            try:
                solver.run(start_from="latest" if args.resume else None)
                assert solver.run_status == "complete" and abs(solver.time-case.END) < 1e-8
                result = dict(status="completed", time=solver.time, step=solver.step)
            except BaseException as error:
                result = dict(status="failed", error=repr(error))
                raise
            finally:
                if solver.parallel.is_root:
                    result.update(wall_seconds=time.monotonic()-started)
                    result_path.write_text(json.dumps(result, indent=2)+"\n")
    else:
        import openonda.coupler as coupling
        from source.coupler.parallel import collective_phase
        setup, particles, coupled, mesh = case.coupled_case(device=args.device)
        with coupling.create_coupler(setup, particles, coupled, mesh=mesh, case_dir=out) as solver:
            solver.initialize()
            if not args.resume:
                initialize_cylinder_perturbation(solver.fvm_solver, case.SPAN)
            try:
                # solve() performs initial FVM->VPM synchronization AFTER seeding.
                if args.resume:
                    last = solver.run(start_from="latest")
                else:
                    last = solver.solve(max_coupling_steps=20 if args.pilot else None,
                                        backup_at_start=True, backup_at_stop=True)
                expected = 20 if args.pilot else case.steps(case.END, case.EXCHANGE)
                assert last == expected
                result = dict(status="pilot-completed" if args.pilot else "completed",
                              time=last*case.EXCHANGE, step=last)
            except BaseException as error:
                result = dict(status="failed", error=repr(error))
                raise
            finally:
                with collective_phase(solver._comm, "publish phase stage result"):
                    if solver._is_master:
                        result.update(wall_seconds=time.monotonic()-started)
                        result_path.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
