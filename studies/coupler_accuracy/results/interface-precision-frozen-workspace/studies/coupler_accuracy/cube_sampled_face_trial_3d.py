#!/usr/bin/env python3
"""Advance the matched medium 3D pair with sampled native boundary observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.cube_coupled_trial import run as run_trial
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays
from studies.coupler_accuracy.experimental_sampled_face_boundary import sampled_face_boundary
from studies.coupler_accuracy.native_face_velocity_sampling_3d import InteriorFaceVelocitySampler


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    paths = [Path(__file__)]
    paths += [ROOT / name for name in ("studies/coupler_accuracy/experimental_sampled_face_boundary.py",
                                      "studies/coupler_accuracy/native_face_velocity_sampling_3d.py", "studies/coupler_accuracy/native_face_trace_3d.py",
                                      "studies/coupler_accuracy/cube_coupled_trial.py", "source/solvers/fvm/fields/gradients.py",
                                      "source/solvers/fvm/assemble/diffusion.py", "source/solvers/fvm/mesh/geometry.py",
                                      "tests/coupler/test_experimental_sampled_face_boundary.py")]
    paths += [args.oracle / name for name in ("full-native-mesh.npz", "small-native-mesh.npz", "cell-and-face-map.npz")]
    sources = [hash_file(path) for path in paths]
    for path in paths:
        if path.suffix == ".py":
            target = args.output / "sources" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    mesh = load_native_mesh(args.oracle / "full-native-mesh.npz")
    for patch in mesh["boundary"]:
        patch["velocity_type"] = "fixedValue"
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    small = load_native_mesh(args.oracle / "small-native-mesh.npz")
    mapping = read_arrays(args.oracle / "cell-and-face-map.npz")
    patch = next(p for p in small["boundary"] if p["name"] == "numericalBoundary")
    rows = slice(patch["start_face"], patch["start_face"]+patch["n_faces"])
    sampler = InteriorFaceVelocitySampler(mesh, geometry, mapping["face_ids"][rows], mapping["signs"][rows])
    np.savez_compressed(args.output / "boundary-sampling-geometry.npz", sample_cells=sampler.sample_cells, gradient_cells=sampler.gradient_cells,
                        full_face_ids=sampler.faces, signs=sampler.trace.signs, sample_position=geometry["cell_centre"][sampler.sample_cells],
                        face_position=geometry["face_centre"][sampler.faces], face_normal=sampler.trace.normal, face_area=sampler.trace.area)
    report = {"schema": "openonda-live-sampled-face-boundary-3d/1", "status": "running", "spatial_dimensions": 3, "mode": args.mode,
              "sources": sources, "sample_cells": len(sampler.sample_cells), "gradient_cells": len(sampler.gradient_cells),
              "boundary_faces": len(sampler.faces), "requested_coupling_steps": args.steps, "boundary_calls": [],
              "limitations": ["The native sampling stencil uses full-reference mesh geometry only. Evolving reference field values never enter a boundary query.",
                              "Only the small inherited FVM mesh is advanced by the hybrid; external stencil locations are VPM evaluation points.",
                              "This changes boundary observations with the existing particle transfer, convection and pressure policies. It does not make cropped momentum fluxes identical to internal full-FVM fluxes.",
                              "The native face velocity is an interpolation of VPM samples, not an advancing pressure-corrected Rhie--Chow flux.",
                              "Short laminar matched-medium experiment, not developed tutorial force/profile or LES validation."]}
    report_path = args.output / "sampled-boundary-trial.json"
    report_path.write_text(json.dumps(report, indent=2)+"\n")

    def record(solver, sampled, observed, velocity):
        record = {"call": len(report["boundary_calls"]), "step": int(solver.step), "elapsed_time": float(solver.time),
                  "sample_velocity_rms": float(np.sqrt(np.mean(np.sum(sampled**2, axis=1)))),
                  "face_velocity_rms": float(np.sqrt(np.mean(np.sum(velocity**2, axis=1)))),
                  "tangential_gradient_rms": float(np.sqrt(np.average(np.sum(observed["tangential_gradient"]**2, axis=1), weights=sampler.trace.area)))}
        report["boundary_calls"].append(record)
        name = "initial-boundary-observation.npz" if record["call"] == 0 else "latest-boundary-observation.npz"
        np.savez_compressed(args.output / name, sampled_velocity=sampled, returned_velocity=velocity,
                            step=record["step"], elapsed_time=record["elapsed_time"], **observed)
        report_path.write_text(json.dumps(report, indent=2)+"\n")

    trial_args = SimpleNamespace(oracle=args.oracle, output=args.output / "trial", particle_spacing=args.particle_spacing,
                                 steps=args.steps, substeps=5, initial_cutoff=.02, audit_pressure=False, mixed_convection="native",
                                 pressure_history="accepted", transfer_cutoff=.05, transfer_method="buffered_m4_renewal",
                                 transfer_amplification=1.8, frozen_renewals=0, experimental_residual_blend=False, boundary_mode="vorticity_mixed")
    try:
        with sampled_face_boundary(sampler, mode=args.mode, callback=record):
            run_trial(trial_args)
        assert report["boundary_calls"]
        report["status"] = "complete"
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
    parser.add_argument("--mode", choices=("native_gradient", "native_both"), required=True)
    parser.add_argument("--particle-spacing", type=float, default=.0625)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.oracle, args.output = args.oracle.resolve(), args.output.resolve()
    run(args)
