#!/usr/bin/env python3
"""Separate complete RK2/GBD substeps from the FVM exchange and renewal cadence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from source.coupler.config.types import CouplerSetup
from studies.coupler_accuracy import cube_profile_checkpoint_trial_3d as checkpoints
from studies.coupler_accuracy import experimental_split_subcycling_3d as experiment
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    frozen = json.loads((ROOT / "frozen-workspace.json").read_text())
    assert frozen["status"] == "complete" and frozen["snapshot_root"] == str(ROOT)
    for row in frozen["records"]:
        assert hash_file(ROOT / row["path"])["sha256"] == row["sha256"]
    paths = [Path(__file__).resolve(), Path(experiment.__file__).resolve(), Path(checkpoints.__file__).resolve()]
    paths += [ROOT / name for name in (
        "source/coupler/config/types.py", "source/coupler/solver.py", "source/solvers/vpm/core/solver.py",
        "source/solvers/vpm/core/evolution.py", "source/solvers/vpm/numerics/runge_kutta.py",
        "source/solvers/vpm/numerics/rk_tableaux.py", "source/solvers/vpm/physics/stage_rhs.py",
        "source/solvers/vpm/stabilization/manager.py", "source/solvers/vpm/physics/diffusion/grid.py")]
    sources = [hash_file(path) for path in paths]
    archive = {path: path.read_bytes() for path in paths}
    correction_limit = CouplerSetup().transfer_discretization_error_limit
    assert correction_limit == .08
    report_path = args.output / "split-subcycling-3d.json"
    record = {
        "schema": "openonda-split-subcycling-3d/1", "status": "running", "spatial_dimensions": 3,
        "vpm_substeps": args.vpm_substeps, "substep_dt": .05 / args.vpm_substeps,
        "outer_dt": .05, "fvm_dt": .01, "requested_exchanges": args.steps,
        "gbd_correction_limit": correction_limit, "frozen_original_files_verified": len(frozen["records"]),
        "sources": sources, "outer_advances": [], "limitations": [
            "Each inner update is the native RK2 advection/stretching followed by GBD diffusion/remapping. Their physical stage times advance while the canonical outer clock commits once.",
            "FVM dt, exchange/accepted renewal and outer stabilization retain their original schedules. Every inner GBD recovery receives the same unchanged production gate.",
            "This changes diffusion/remapping frequency and operator splitting together; it does not isolate their individual errors.",
            "The vpm_boundary_condition panel scope still excludes the body velocity/gradient from particle stages while including them in target queries.",
            "This is a fully 3D serial laminar cube experiment with fixed-predictor interface replay, not a production option or developed-wake qualification.",
        ],
    }
    started = time.perf_counter()
    try:
        with experiment.split_subcycling(args.vpm_substeps, correction_limit, record, report_path):
            checkpoints.run(args)
        assert len(record["outer_advances"]) == args.steps
        assert [row["accepted_step"] for row in record["outer_advances"]] == list(range(1, args.steps + 1))
        assert [hash_file(path) for path in paths] == sources
        for row in frozen["records"]:
            assert hash_file(ROOT / row["path"])["sha256"] == row["sha256"]
        record["status"] = "complete"
    except Exception as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["elapsed_seconds"] = time.perf_counter() - started
        if args.output.exists():
            for path, data in archive.items():
                target = args.output / "split-sources" / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            record["child_reports"] = [hash_file(args.output / name) for name in (
                "accepted-fvm-checkpoints-3d.json", "profile-observation-3d.json", "interface-cadence-3d.json",
                "interface-iteration-3d.json", "panel-target-precision-3d.json", "trial/cube-coupled-trial.json") if (args.output / name).exists()]
            report_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: record[key] for key in ("status", "vpm_substeps", "requested_exchanges", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vpm-substeps", type=int, choices=(1, 2, 5), required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--profile-every-fvm-steps", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--relaxation", type=float, default=1.)
    parser.add_argument("--normal-tolerance", type=float, default=1e-6)
    parser.add_argument("--gradient-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    if args.steps < 1 or args.iterations < 0 or args.profile_every_fvm_steps < 5 or args.profile_every_fvm_steps % 5:
        parser.error("Require positive steps and a positive profile interval divisible by five")
    args.oracle, args.output = args.oracle.resolve(), args.output.resolve()
    args.substeps, args.particle_spacing = 5, .0625
    run(args)
