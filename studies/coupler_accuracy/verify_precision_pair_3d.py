#!/usr/bin/env python3
"""Verify a frozen-source native/promoted interface-derivative comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from source.solvers.fvm.io.backup import decode_state
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays
from studies.coupler_accuracy.verify_sampled_face_trials_3d import independent_cube_force


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    snapshot_path = ROOT / "frozen-workspace.json"
    snapshot = json.loads(snapshot_path.read_text())
    assert snapshot["status"] == "complete" and snapshot["snapshot_root"] == str(ROOT)
    for row in snapshot["records"]:
        assert hash_file(ROOT / row["path"])["sha256"] == row["sha256"]
    verification = json.loads(args.promoted_verification.read_text())
    assert verification["status"] == "complete"
    for row in verification["sources"]:
        assert hash_file(ROOT / row["path"]) == row
    native, promoted = (json.loads((directory / "interface-iteration-3d.json").read_text()) for directory in (args.native, args.promoted))
    nc, pc = (json.loads((directory / "trial/cube-coupled-trial.json").read_text()) for directory in (args.native, args.promoted))
    assert not (args.native / "panel-target-precision-3d.json").exists()
    precision = json.loads((args.promoted / "panel-target-precision-3d.json").read_text())
    assert precision["status"] == "complete" and precision["mode"] == "f64_auxiliary_panel_queries"
    checked = next(row for row in verification["experiments"] if row["directory"] == str(args.promoted.relative_to(ROOT)))
    assert checked["auxiliary_panel_precision_mode"] == "f64_auxiliary_panel_queries"
    assert all(report["status"] == "complete" and report["spatial_dimensions"] == 3 for report in (native, promoted, nc, pc))
    assert native["sources"] == promoted["sources"] and nc["sources"] == pc["sources"]
    for report in (native, nc):
        for row in report["sources"]:
            assert hash_file(ROOT / row["path"]) == row
    for key in ("maximum_sweeps", "relaxation", "normal_tolerance", "gradient_tolerance", "execution_environment"):
        assert native[key] == promoted[key]
    assert nc["requested_coupling_steps"] == pc["requested_coupling_steps"] == 3
    assert native["comparison"] == nc["comparison"] and promoted["comparison"] == pc["comparison"]
    assert native["comparison"][0] == promoted["comparison"][0]
    for key in ("coupling_step", "physical_time", "full_drag_coefficient"):
        assert [r[key] for r in native["comparison"]] == [r[key] for r in promoted["comparison"]]
    mesh_path = args.native / "trial/hybrid/solution/mesh.npz"
    mesh = load_native_mesh(mesh_path)
    geo = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    patch = next(row for row in mesh["boundary"] if row["name"] == "numericalBoundary")
    faces = slice(patch["start_face"], patch["start_face"]+patch["n_faces"])
    area, volume = np.linalg.norm(geo["face_area_vector"][faces], axis=1), geo["cell_volume"]
    errors = []
    last = {}
    for row in native["sweeps"]:
        assert row["runtime_cpu_threads"] == 1 and row["accepted_fvm_step"] == 5*row["coupling_step"]
        assert row["accepted_transfer_step"] == row["coupling_step"]+1 and row["map_evaluations"] == row["sweep"]+1
        if row["sweep"] == 1:
            assert row["map_replay"]["bitwise_equal"] and row["map_replay"]["numerical_arrays_and_clocks"] == 24
        assert hash_file(ROOT / row["boundary_fields"]["path"]) == row["boundary_fields"]
        f = read_arrays(ROOT / row["boundary_fields"]["path"])
        for mode, metric in (("normal_velocity", "normal_residual_rms"), ("tangential_gradient", "gradient_residual_rms")):
            delta = f["post_"+mode]-f["applied_"+mode]
            square = delta*delta if delta.ndim == 1 else np.sum(delta*delta, axis=1)
            difference = abs(float(np.sqrt(np.dot(area, square)/area.sum()))-row[metric])
            assert difference < 2e-14
            errors.append(difference)
        assert row["converged"] == (row["normal_residual_rms"] <= native["normal_tolerance"] and row["gradient_residual_rms"] <= native["gradient_tolerance"])
        last[row["coupling_step"]] = row
    assert set(last) == {1, 2, 3} and sum(row["converged"] for row in last.values()) == native["converged_intervals"]
    manifest_path = args.native / "trial/hybrid/solution/backups/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["fvm_step"] == 15 and manifest["coupling_step"] == manifest["vpm_step"] == 3
    for key, name in manifest["artifacts"].items():
        assert hash_file(manifest_path.parent / name)["sha256"] == manifest["artifact_sha256"][key]
    state = decode_state(read_arrays(manifest_path.parent / manifest["artifacts"]["fvm"]))
    fields = read_arrays(args.native / "trial/latest-comparison-fields.npz")
    pf = read_arrays(args.promoted / "trial/latest-comparison-fields.npz")
    np.testing.assert_array_equal(fields["full_velocity"], pf["full_velocity"])
    np.testing.assert_array_equal(state["velocity"][:mesh["n_cells"]], fields["hybrid_velocity"])
    force = independent_cube_force(state, mesh, geo)
    difference = abs(force["drag_coefficient"]-native["comparison"][-1]["hybrid_drag_coefficient"])
    assert difference < 2e-13
    errors.append(difference)
    for ids, key in ((np.arange(mesh["n_cells"]), "fvm_velocity_rms_over_Uinf"),
                     (np.flatnonzero(np.max(np.abs(geo["cell_centre"]), axis=1) < 1), "fvm_near_body_velocity_rms_over_Uinf")):
        delta = fields["hybrid_velocity"][ids]-fields["full_velocity"][ids]
        actual = float(np.sqrt(np.average(np.sum(delta*delta, axis=1), weights=volume[ids])))
        difference = abs(actual-native["comparison"][-1][key])
        assert difference < 2e-13
        errors.append(difference)
    sources = [hash_file(p) for p in (snapshot_path, args.promoted_verification, Path(__file__).resolve(), mesh_path, manifest_path)]
    for directory in (args.native, args.promoted):
        sources += [hash_file(directory / name) for name in ("interface-iteration-3d.json", "trial/cube-coupled-trial.json", "trial/latest-comparison-fields.npz")]
    result = {"schema": "openonda-interface-precision-pair-3d/1", "status": "complete", "spatial_dimensions": 3,
              "frozen_files_verified": len(snapshot["records"]), "additional_independent_metrics": len(errors), "maximum_metric_difference": max(errors),
              "native_independent_force": force,
              "cases": [{"directory": str(directory.relative_to(ROOT)), "converged_intervals": report["converged_intervals"],
                         "final_sweeps": [{k:v[k] for k in ("coupling_step", "sweep", "normal_residual_rms", "gradient_residual_rms", "converged")}
                                          for v in {r["coupling_step"]:r for r in report["sweeps"]}.values()],
                         "comparison": report["comparison"]} for directory, report in ((args.native, native), (args.promoted, promoted))],
              "sources": sources,
              "limitations": ["Three short matched 3D intervals on one immutable source snapshot; no developed-wake or complete 20-interval precision conclusion.",
                              "The promoted zero/one-sweep checkpoint identity is checked separately. Each native and promoted interval replays its own endpoint map; different derivative operators are not required to share a first-map fingerprint."]}
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("frozen_files_verified", "additional_independent_metrics", "maximum_metric_difference", "cases")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--promoted", type=Path, required=True)
    parser.add_argument("--promoted-verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.native, args.promoted, args.promoted_verification, args.output = (p.resolve() for p in (args.native, args.promoted, args.promoted_verification, args.output))
    run(args)
