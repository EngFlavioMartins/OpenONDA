#!/usr/bin/env python3
"""Test the accepted body field in fully 3D particle advection and stretching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from studies.coupler_accuracy import cube_profile_checkpoint_trial_3d as checkpoints
from studies.coupler_accuracy import experimental_body_transport_3d as experiment
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    frozen = json.loads((ROOT / "frozen-workspace.json").read_text())
    assert frozen["status"] == "complete" and frozen["snapshot_root"] == str(ROOT)
    for row in frozen["records"]:
        assert hash_file(ROOT / row["path"])["sha256"] == row["sha256"]
    qualification = json.loads(args.gradient_qualification.read_text())
    assert qualification["status"] == "complete" and qualification["schema"] == "openonda-analytical-panel-gradient-qualification-3d/1"
    assert qualification["spatial_dimensions"] == 3 and qualification["independent_targets"] == 256
    for row in qualification["sources"]:
        assert hash_file(ROOT / row["path"]) == row
    paths = [Path(__file__).resolve(), Path(experiment.__file__).resolve(), Path(checkpoints.__file__).resolve()]
    paths += [ROOT / name for name in (
        "studies/coupler_accuracy/analytical_panel_gradient_3d.py", "studies/coupler_accuracy/native_volume_induction_3d.py", "source/coupler/solver.py", "source/solvers/vpm/core/solver.py",
        "source/solvers/vpm/core/evolution.py", "source/solvers/vpm/numerics/runge_kutta.py",
        "source/solvers/vpm/numerics/rk_tableaux.py", "source/solvers/vpm/physics/stage_rhs.py",
        "source/solvers/vpm/stabilization/manager.py", "source/solvers/vpm/physics/diffusion/grid.py")]
    sources = [hash_file(path) for path in paths]
    archive = {path: path.read_bytes() for path in paths}
    report_path = args.output / "body-transport-3d.json"
    record = {
        "schema": "openonda-body-transport-3d/1", "status": "running", "spatial_dimensions": 3,
        "body_transport_enabled": args.enable_body_transport,
        "gradient_qualification": hash_file(args.gradient_qualification),
        "outer_dt": .05, "fvm_dt": .01, "requested_exchanges": args.steps,
        "frozen_original_files_verified": len(frozen["records"]),
        "sources": sources, "outer_advances": [], "limitations": [
            "Only RK stage callbacks change: the native source-panel velocity and a qualified analytical f64 source Jacobian are added at actual temporary stage positions.",
            "Stored particle/panel precision, FVM dt, RK/GBD step, physical exchange/renewal and outer stabilization retain their existing settings.",
            "Panel strengths are fixed within each VPM interval, as in the baseline. Panel boundary solves still occur in the existing FVM-coupling path; panel advancement and shedding remain disabled.",
            "The analytical source gradient uses J[i,j]=du_i/dx_j; the native StageRHS retains its transposed stretching convention and all production recovery gates.",
            "This is a fully 3D serial laminar cube experiment with fixed-predictor interface replay, not a production option or developed-wake qualification.",
        ],
    }
    started = time.perf_counter()
    try:
        with experiment.body_transport(args.enable_body_transport, record, report_path):
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
                target = args.output / "body-transport-sources" / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            record["child_reports"] = [hash_file(args.output / name) for name in (
                "accepted-fvm-checkpoints-3d.json", "profile-observation-3d.json", "interface-cadence-3d.json",
                "interface-iteration-3d.json", "panel-target-precision-3d.json", "trial/cube-coupled-trial.json") if (args.output / name).exists()]
            report_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: record[key] for key in ("status", "body_transport_enabled", "requested_exchanges", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enable-body-transport", action="store_true")
    parser.add_argument("--gradient-qualification", type=Path, required=True)
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
    args.gradient_qualification = args.gradient_qualification.resolve()
    args.substeps, args.particle_spacing = 5, .0625
    run(args)
