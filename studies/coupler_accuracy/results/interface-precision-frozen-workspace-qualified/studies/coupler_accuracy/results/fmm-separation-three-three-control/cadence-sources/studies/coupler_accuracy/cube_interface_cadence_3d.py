#!/usr/bin/env python3
"""Vary the exchange interval with the qualified 3D interface/precision runners.

The FVM time step, initial fields, meshes and reference problem are fixed.
Changing the number of FVM substeps also changes the VPM RK2 step and renewal
frequency. This is a combined hybrid time-resolution experiment; it cannot
attribute a change to one of those operations in isolation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time
from types import SimpleNamespace

from studies.coupler_accuracy import cube_interface_iteration_3d as iteration
from studies.coupler_accuracy import cube_interface_precision_3d as precision
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file


@contextmanager
def exchange_interval(substeps, record):
    """Override only the trial's existing substep input, restoring it on exit."""
    original = iteration.run_trial

    def invoke(args):
        if args.substeps != 5 or record["trial_invocations"]:
            raise ValueError("Expected one invocation of the qualified five-substep runner")
        trial = SimpleNamespace(**{**vars(args), "substeps": substeps})
        record["trial_invocations"] += 1
        return original(trial)

    iteration.run_trial = invoke
    try:
        yield
    finally:
        iteration.run_trial = original


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    oracle = json.loads((args.oracle / "cube-boundary-oracle.json").read_text())
    assert oracle["spatial_dimensions"] == 3 and oracle["valid_for_comparison"]
    assert oracle["dt"] == .01 and args.particle_spacing == .0625
    paths = [Path(__file__).resolve(), Path(iteration.__file__).resolve(), Path(precision.__file__).resolve()]
    sources = [hash_file(path) for path in paths]
    frozen = ROOT / "frozen-workspace.json"
    record = {
        "schema": "openonda-interface-cadence-3d/1", "status": "running", "spatial_dimensions": 3,
        "fvm_substeps": args.substeps, "fvm_dt": oracle["dt"],
        "exchange_dt": oracle["dt"] * args.substeps, "coupling_steps": args.steps,
        "accepted_fvm_steps": args.substeps * args.steps,
        "elapsed_flow_time": oracle["dt"] * args.substeps * args.steps,
        "trial_invocations": 0, "sources": sources,
        "frozen_workspace_manifest": hash_file(frozen) if frozen.exists() else None,
        "limitations": [
            "FVM dt and spatial resolution stay fixed. VPM step, exchange interval and renewal frequency change together.",
            "The qualified fixed-predictor replay and f64 auxiliary panel-query scopes are otherwise unchanged.",
            "The reference evolves independently and only measures the hybrid at accepted endpoints.",
            "VPM velocity metrics sample matched FVM cell centres; they are not exterior-wake profile measurements.",
            "This is a fully 3D serial laminar cube study, not a production option or developed-wake validation.",
        ],
    }
    started = time.perf_counter()
    try:
        with exchange_interval(args.substeps, record):
            precision.run(args)
        assert record["trial_invocations"] == 1
        child = json.loads((args.output / "trial/cube-coupled-trial.json").read_text())
        assert child["status"] == "complete"
        assert child["fvm_dt"] == record["fvm_dt"] and child["vpm_dt"] == record["exchange_dt"]
        assert child["requested_coupling_steps"] == args.steps
        assert [hash_file(path) for path in paths] == sources
        record["status"] = "complete"
    except Exception as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["elapsed_seconds"] = time.perf_counter() - started
        if args.output.exists():
            for path in paths:
                target = args.output / "cadence-sources" / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
            for name in ("interface-iteration-3d.json", "panel-target-precision-3d.json", "trial/cube-coupled-trial.json"):
                path = args.output / name
                if path.exists():
                    record.setdefault("artifacts", []).append(hash_file(path))
            (args.output / "interface-cadence-3d.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: record[key] for key in ("status", "fvm_substeps", "exchange_dt", "accepted_fvm_steps", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--substeps", type=int, choices=(1, 2, 5), required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--particle-spacing", type=float, default=.0625)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--relaxation", type=float, default=1.)
    parser.add_argument("--normal-tolerance", type=float, default=1e-6)
    parser.add_argument("--gradient-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    if args.steps < 1 or args.iterations < 0 or args.particle_spacing != .0625:
        parser.error("Positive steps, a nonnegative sweep count and matched medium spacing are required")
    args.oracle, args.output = args.oracle.resolve(), args.output.resolve()
    run(args)
