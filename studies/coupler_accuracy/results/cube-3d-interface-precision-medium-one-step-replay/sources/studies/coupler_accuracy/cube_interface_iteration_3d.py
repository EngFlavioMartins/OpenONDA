#!/usr/bin/env python3
"""Test fixed-predictor interface iteration in the real matched 3D cube pair."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import taichi as ti

import openonda.fvm as fvm
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.cube_coupled_trial import run as run_trial
from studies.coupler_accuracy.experimental_interface_iteration import fixed_predictor_iteration


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    paths = [Path(__file__).resolve()]
    paths += [ROOT / name for name in (
        "studies/coupler_accuracy/experimental_interface_iteration.py", "studies/coupler_accuracy/cube_coupled_trial.py",
        "source/coupler/solver.py", "source/coupler/boundary.py", "source/coupler/vorticity_transfer.py", "source/coupler/stable_renewal.py",
        "source/solvers/fvm/io/backup.py", "source/solvers/fvm/core/solver.py", "source/solvers/vpm/core/solver.py")]
    sources = [hash_file(path) for path in paths]
    for path in paths:
        archive = args.output / "sources" / path.relative_to(ROOT)
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(path.read_bytes())
    force = fvm.ForceSampler(patch_names=["cube"], reference_area=1, reference_length=1)
    report = {"schema": "openonda-interface-iteration-3d/1", "status": "running", "spatial_dimensions": 3,
              "maximum_sweeps": args.iterations, "relaxation": args.relaxation,
              "normal_tolerance": args.normal_tolerance, "gradient_tolerance": args.gradient_tolerance,
              "execution_environment": {name: os.environ.get(name) for name in ("TI_CPU_MAX_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
              "sources": sources, "sweeps": [], "limitations": [
                  "Only interface consistency is changed. The FVM start and advected VPM predictor are fixed within every interval; particle renewal never uses a prior sweep as its starting cloud.",
                  "The first endpoint map in each interval is replayed from the saved states and required to match numerical fields and clocks bitwise.",
                  "Zero sweeps selects the original unmodified partitioned algorithm. One sweep tests the wrapper and replay path without a numerical correction.",
                  "The fixed tolerances measure interface residuals; they are not acceptance tolerances for force or velocity agreement with the reference.",
                  "A completed bounded run can contain unconverged intervals; convergence flags are explicit and no production mode is changed.",
                  "Serial buffered-M4 mixed/flux-pressure study without a consistency band; no developed-wake validation is implied."]}
    report_path = args.output / "interface-iteration-3d.json"
    report_path.write_text(json.dumps(report, indent=2)+"\n")

    def observe(coupler, row, applied, post):
        row["runtime_cpu_threads"] = int(ti.lang.impl.current_cfg().cpu_max_num_threads)
        row["hybrid_drag_coefficient"] = float(force.sample(coupler.fvm_solver)["cube"]["coeffs"]["drag_coefficient"])
        path = args.output / f"boundary-step-{row['coupling_step']:06d}-sweep-{row['sweep']:02d}.npz"
        np.savez_compressed(path, applied_velocity=applied[0], applied_normal_velocity=applied[1], applied_tangential_gradient=applied[2],
                            post_velocity=post[0], post_normal_velocity=post[1], post_tangential_gradient=post[2])
        row["boundary_fields"] = hash_file(path)
        report["sweeps"].append(row)
        report_path.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps({"event": "interface_iteration", **{k:v for k,v in row.items() if k != "map_replay"}}), flush=True)

    trial_args = SimpleNamespace(oracle=args.oracle, output=args.output / "trial", particle_spacing=args.particle_spacing,
                                 steps=args.steps, substeps=5, initial_cutoff=.02, audit_pressure=False, mixed_convection="native",
                                 pressure_history="accepted", transfer_cutoff=.05, transfer_method="buffered_m4_renewal",
                                 transfer_amplification=1.8, frozen_renewals=0, experimental_residual_blend=False, boundary_mode="vorticity_mixed")
    context = (fixed_predictor_iteration(args.output / "interval-starts", iterations=args.iterations, relaxation=args.relaxation,
                                        normal_tolerance=args.normal_tolerance, gradient_tolerance=args.gradient_tolerance, callback=observe)
               if args.iterations else nullcontext())
    try:
        with context:
            run_trial(trial_args)
        report["status"] = "complete"
        if ti.lang.impl.get_runtime().prog is not None:
            report["runtime_cpu_threads_after_run"] = int(ti.lang.impl.current_cfg().cpu_max_num_threads)
        last = {row["coupling_step"]: row for row in report["sweeps"]}
        report["converged_intervals"] = sum(row["converged"] for row in last.values())
        report["observed_intervals"] = len(last)
        report["comparison"] = json.loads((args.output / "trial/cube-coupled-trial.json").read_text())["comparison"]
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter()-started
        report_path.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--particle-spacing", type=float, default=.0625)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--relaxation", type=float, default=1.)
    parser.add_argument("--normal-tolerance", type=float, default=1e-6)
    parser.add_argument("--gradient-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    if args.steps < 1 or args.iterations < 0 or args.particle_spacing <= 0:
        parser.error("Positive steps and spacing, and a nonnegative sweep count, are required")
    args.oracle, args.output = args.oracle.resolve(), args.output.resolve()
    run(args)
