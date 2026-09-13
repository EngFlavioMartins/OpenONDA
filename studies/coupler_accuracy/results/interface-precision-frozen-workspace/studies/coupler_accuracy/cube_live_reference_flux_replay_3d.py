#!/usr/bin/env python3
"""Recover the live comparison reference flux with the same stepping horizon."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

import openonda.fvm as fvm
from source.solvers.fvm.io.backup import save_backup
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file, setup_for
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    trial_path = args.trial / "cube-coupled-trial.json"
    field_path = args.trial / "latest-comparison-fields.npz"
    trial, expected = json.loads(trial_path.read_text()), read_arrays(field_path)
    assert trial["status"] == "complete" and trial["spatial_dimensions"] == 3
    oracle_path = args.oracle / "cube-boundary-oracle.json"
    oracle = json.loads(oracle_path.read_text())
    mesh_path, initial_path, map_path = (args.oracle / name for name in ("full-native-mesh.npz", "initial-cell-fields.npz", "cell-and-face-map.npz"))
    mesh, initial, mapping = load_native_mesh(mesh_path), read_arrays(initial_path), read_arrays(map_path)
    dt, steps = trial["fvm_dt"], round(float(expected["elapsed_time"])/trial["fvm_dt"])
    assert steps == trial["requested_coupling_steps"]*round(trial["vpm_dt"]/dt)
    config = {"turbulence": oracle["numerics"]["sgs"] != "none", "outer_correctors": oracle["numerics"]["outer_correctors"]}
    paths = [Path(__file__).resolve(), trial_path, field_path, oracle_path, mesh_path, initial_path, map_path]
    paths += [ROOT / name for name in (
        "studies/coupler_accuracy/cube_coupled_trial.py", "studies/coupler_accuracy/cube_boundary_oracle.py",
        "studies/coupler_accuracy/cube_native_moment_reconstruction_3d.py", "source/solvers/fvm/core/solver.py",
        "source/solvers/fvm/io/backup.py", "source/solvers/fvm/io/mesh_storage.py", "source/solvers/fvm/mesh/geometry.py",
        "source/solvers/fvm/fields/gradients.py", "source/solvers/fvm/fields/diagnostics.py", "source/solvers/fvm/assemble/momentum.py",
        "source/solvers/fvm/assemble/convection.py", "source/solvers/fvm/assemble/diffusion.py", "source/solvers/fvm/assemble/time_integration.py",
        "source/solvers/fvm/solve/pimple_solver.py", "source/solvers/fvm/solve/simple_solver.py")]
    sources = [hash_file(path) for path in paths]
    original = {row["path"]: row for row in trial["sources"]}
    for row in sources:
        if row["path"] in original:
            assert row == original[row["path"]]
    for path in paths:
        if path.suffix == ".py":
            archive = args.output / "sources" / path.relative_to(ROOT)
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_bytes(path.read_bytes())
    (args.output / "sources-at-start.json").write_text(json.dumps(sources, indent=2)+"\n")
    # This is the existing live runner's configured horizon, not a new endpoint
    # equal to the observation time. Clipping the final step changes its rounding.
    setup = setup_for(mesh, "full", dt, round(20/dt), **config)
    with fvm.create_fvm_solver(setup, case_dir=args.output / "full", mesh=copy.deepcopy(mesh)) as solver:
        solver.set_initial_state(initial["velocity"], initial["pressure"])
        for step in range(steps):
            solver.advance()
            if (step+1) % 10 == 0 or step+1 == steps:
                print(json.dumps({"step": step+1, "total": steps, "elapsed_seconds": time.perf_counter()-started}), flush=True)
        actual = solver.get_velocity_field()[mapping["cell_ids"]]
        maximum = float(np.max(np.abs(actual-expected["full_velocity"])))
        np.testing.assert_array_equal(actual, expected["full_velocity"])
        force = fvm.ForceSampler(patch_names=["cube"], reference_area=1, reference_length=1).sample(solver)["cube"]["coeffs"]
        np.testing.assert_array_equal(float(force["drag_coefficient"]), trial["comparison"][-1]["full_drag_coefficient"])
        backup = args.output / "final-fvm.npz"
        save_backup(solver, backup)
        endpoint = {"name": "final", "physical_time": trial["comparison"][-1]["physical_time"],
                    "solver_time": float(solver.time), "solver_step": int(solver.step),
                    "maximum_shared_velocity_replay_difference": maximum,
                    "full_drag_coefficient": float(force["drag_coefficient"]), "backup": hash_file(backup)}
    result = {"schema": "openonda-live-reference-flux-replay-3d/1", "status": "complete", "spatial_dimensions": 3,
              "full_fvm_cells": mesh["n_cells"], "shared_fvm_cells": len(mapping["cell_ids"]), "configured_end_time": setup.time.end_time,
              "dt": dt, "source_sgs": oracle["numerics"]["sgs"], "records": [endpoint], "sources": sources,
              "elapsed_seconds": time.perf_counter()-started,
              "limitations": ["The reference velocity comparison covers every shared native cell; exterior velocities and face fluxes were not saved by the original live comparison.",
                              "The replay uses the live reference's full configuration and reproduces its shared velocity and drag bitwise. It does not change any hybrid state."]}
    (args.output / "reference-flux-replay-3d.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.oracle, args.trial, args.output = (path.resolve() for path in (args.oracle, args.trial, args.output))
    run(args)
