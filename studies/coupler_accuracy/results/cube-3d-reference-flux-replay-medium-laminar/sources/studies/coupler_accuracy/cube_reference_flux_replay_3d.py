#!/usr/bin/env python3
"""Recover pressure-corrected reference fluxes while replaying saved 3D fields."""

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
    report_path = args.oracle / "cube-boundary-oracle.json"
    mesh_path = args.oracle / "full-native-mesh.npz"
    initial_path = args.oracle / "initial-cell-fields.npz"
    final_path = args.oracle / "final-full-cell-fields.npz"
    report = json.loads(report_path.read_text())
    assert report["valid_for_comparison"] and report["spatial_dimensions"] == 3
    initial, final = read_arrays(initial_path), read_arrays(final_path)
    dt, steps = report["dt"], report["steps"]
    warmup_steps = round(float(initial["physical_time"])/dt)
    np.testing.assert_allclose(warmup_steps*dt, initial["physical_time"], rtol=0, atol=1e-12)
    assert warmup_steps > 0 and report["numerics"]["time_scheme"] == "euler_implicit"
    mesh = load_native_mesh(mesh_path)
    config = {"turbulence": report["numerics"]["sgs"] != "none", "outer_correctors": report["numerics"]["outer_correctors"]}
    paths = [Path(__file__).resolve(), report_path, mesh_path, initial_path, final_path]
    paths += [ROOT / name for name in (
        "studies/coupler_accuracy/cube_boundary_oracle.py", "studies/coupler_accuracy/cube_native_moment_reconstruction_3d.py",
        "source/solvers/fvm/core/solver.py", "source/solvers/fvm/io/backup.py", "source/solvers/fvm/io/mesh_storage.py",
        "source/solvers/fvm/mesh/geometry.py", "source/solvers/fvm/fields/gradients.py", "source/solvers/fvm/fields/diagnostics.py",
        "source/solvers/fvm/assemble/momentum.py", "source/solvers/fvm/assemble/convection.py", "source/solvers/fvm/assemble/diffusion.py",
        "source/solvers/fvm/assemble/time_integration.py", "source/solvers/fvm/solve/pimple_solver.py", "source/solvers/fvm/solve/simple_solver.py")]
    sources = [hash_file(path) for path in paths]
    for path in paths:
        if path.suffix == ".py":
            archive = args.output / "sources" / path.relative_to(ROOT)
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_bytes(path.read_bytes())
    (args.output / "sources-at-start.json").write_text(json.dumps(sources, indent=2)+"\n")
    records = []

    def capture(solver, name, target, physical_time):
        u = solver.get_velocity_field()
        p = solver.kinematic_pressure[:mesh["n_cells"]]
        difference = {"velocity": float(np.max(np.abs(u-target["velocity"]))),
                      "pressure": float(np.max(np.abs(p-target["pressure"])))}
        np.testing.assert_allclose(u, target["velocity"], rtol=0, atol=2e-12)
        np.testing.assert_allclose(p, target["pressure"], rtol=0, atol=2e-12)
        np.testing.assert_allclose(solver.get_cell_centre_coordinates(), initial["centres"], rtol=0, atol=1e-13)
        backup = args.output / (name+"-fvm.npz")
        save_backup(solver, backup)
        row = {"name": name, "physical_time": float(physical_time), "solver_time": float(solver.time),
               "solver_step": int(solver.step), "maximum_field_replay_difference": difference, "backup": hash_file(backup)}
        records.append(row)
        print(json.dumps(row), flush=True)

    def advance(solver, count, phase):
        for step in range(count):
            solver.advance()
            if (step+1) % 10 == 0 or step+1 == count:
                print(json.dumps({"phase": phase, "step": step+1, "total": count,
                                  "elapsed_seconds": time.perf_counter()-started}), flush=True)

    with fvm.create_fvm_solver(setup_for(mesh, "flux-warmup", dt, warmup_steps, **config),
                               case_dir=args.output / "warmup", mesh=copy.deepcopy(mesh)) as solver:
        advance(solver, warmup_steps, "warmup")
        capture(solver, "warmup", initial, initial["physical_time"])
    with fvm.create_fvm_solver(setup_for(mesh, "flux-reference", dt, steps, **config),
                               case_dir=args.output / "reference", mesh=copy.deepcopy(mesh)) as solver:
        solver.set_initial_state(initial["velocity"], initial["pressure"])
        capture(solver, "reset-initial", initial, initial["physical_time"])
        advance(solver, steps, "reference")
        capture(solver, "final", final, float(initial["physical_time"])+solver.time)
    result = {"schema": "openonda-reference-flux-replay-3d/1", "status": "complete", "spatial_dimensions": 3,
              "full_fvm_cells": mesh["n_cells"], "particle_spacing": report["requested_wall_spacing"],
              "source_sgs": report["numerics"]["sgs"], "dt": dt, "warmup_steps": warmup_steps, "reference_steps": steps,
              "records": records, "sources": sources, "elapsed_seconds": time.perf_counter()-started,
              "limitations": [
                  "The accepted warmup and reset-initial snapshots have the same cell velocity and pressure, but different flux-history contracts.",
                  "The reset deliberately reproduces the existing oracle's set_initial_state call; it is not asserted to preserve the warmup's pressure-corrected face flux.",
                  "This recovers reference flux observations. It does not change or requalify the advancing hybrid transfer."]}
    (args.output / "reference-flux-replay-3d.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.oracle, args.output = args.oracle.resolve(), args.output.resolve()
    run(args)
