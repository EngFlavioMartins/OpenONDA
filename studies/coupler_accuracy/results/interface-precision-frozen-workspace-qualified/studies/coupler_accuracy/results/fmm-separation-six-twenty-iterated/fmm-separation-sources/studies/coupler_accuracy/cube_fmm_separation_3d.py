#!/usr/bin/env python3
"""Isolate FMM stage accuracy in the original fully 3D coupled cube trajectory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from studies.coupler_accuracy import cube_profile_checkpoint_trial_3d as checkpoints
from studies.coupler_accuracy import experimental_fmm_separation_3d as experiment
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    frozen = json.loads((ROOT / "frozen-workspace.json").read_text())
    assert frozen["status"] == "complete" and frozen["snapshot_root"] == str(ROOT)
    for row in frozen["records"]:
        assert hash_file(ROOT / row["path"])["sha256"] == row["sha256"]
    qualifications = []
    for path, factor in ((args.stage_qualification, args.separation), (args.native_stage_control, 3.)):
        qualification = json.loads(path.read_text())
        assert qualification["status"] == "complete" and qualification["schema"] == "openonda-particle-stage-separation-probe-3d/1"
        assert qualification["spatial_dimensions"] == 3 and qualification["geometric_separation_factor"] == factor
        assert qualification["replayed_stage_bitwise_equal"] and qualification["primary_fields_and_clocks_unchanged"]
        if factor == 3.:
            assert qualification["native_control_bitwise_equal"]
        for row in qualification["sources"]:
            assert hash_file(ROOT / row["path"]) == row
        qualifications.append(hash_file(path))
    paths = [Path(__file__).resolve(), Path(experiment.__file__).resolve(), Path(checkpoints.__file__).resolve()]
    paths += [ROOT / name for name in (
        "source/solvers/vpm/physics/induction/fmm/device.py", "source/solvers/vpm/kernels/base.py", "source/coupler/solver.py", "source/solvers/vpm/core/solver.py",
        "source/solvers/vpm/core/evolution.py", "source/solvers/vpm/numerics/runge_kutta.py",
        "source/solvers/vpm/numerics/rk_tableaux.py", "source/solvers/vpm/physics/stage_rhs.py",
        "source/solvers/vpm/stabilization/manager.py", "source/solvers/vpm/physics/diffusion/grid.py")]
    sources = [hash_file(path) for path in paths]
    archive = {path: path.read_bytes() for path in paths}
    report_path = args.output / "fmm-separation-trial-3d.json"
    record = {
        "schema": "openonda-fmm-separation-trial-3d/1", "status": "running", "spatial_dimensions": 3,
        "geometric_separation_factor": args.separation, "native_factor": 3.,
        "stage_qualifications": qualifications,
        "execution_environment": {key: os.environ.get(key) for key in ("TI_CPU_MAX_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TI_OFFLINE_CACHE_FILE_PATH")},
        "outer_dt": .05, "fvm_dt": .01, "requested_exchanges": args.steps,
        "frozen_original_files_verified": len(frozen["records"]),
        "sources": sources, "outer_advances": [], "limitations": [
            "Only the internal FMM geometric separation factor changes, before compilation in a fresh runtime. The fixed expansion order, kernel-tail criteria and f32 particle precision are unchanged.",
            "FVM mesh/numerics, physical RK/GBD step, exchange/renewal and outer stabilization retain their original settings. Point queries retain the backend's direct regularized fallback.",
            "The selected vpm_boundary_condition scope still excludes the body velocity/gradient from RK stages. This experiment isolates induction accuracy from the separate body-transport change.",
            "The native transposed stretching scheme and all production conservation/recovery limits remain unchanged.",
            "This is a fully 3D serial laminar cube experiment with fixed-predictor interface replay, not a production option or developed-wake qualification.",
        ],
    }
    started = time.perf_counter()
    try:
        with experiment.fmm_separation(args.separation, record, report_path):
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
                target = args.output / "fmm-separation-sources" / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            record["child_reports"] = [hash_file(args.output / name) for name in (
                "accepted-fvm-checkpoints-3d.json", "profile-observation-3d.json", "interface-cadence-3d.json",
                "interface-iteration-3d.json", "panel-target-precision-3d.json", "trial/cube-coupled-trial.json") if (args.output / name).exists()]
            report_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: record[key] for key in ("status", "geometric_separation_factor", "requested_exchanges", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--separation", type=float, choices=(3., 4.5, 6.), required=True)
    parser.add_argument("--stage-qualification", type=Path, required=True)
    parser.add_argument("--native-stage-control", type=Path, required=True)
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
    args.stage_qualification, args.native_stage_control = args.stage_qualification.resolve(), args.native_stage_control.resolve()
    args.substeps, args.particle_spacing = 5, .0625
    run(args)
