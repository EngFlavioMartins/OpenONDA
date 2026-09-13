#!/usr/bin/env python3
"""Verify interface sweeps, a one-sweep control, stored fields and wall forces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from source.solvers.fvm.io.backup import decode_state
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays
from studies.coupler_accuracy.verify_sampled_face_trials_3d import independent_cube_force


def rms(value, weight):
    square = value*value
    if square.ndim == 2:
        square = np.sum(square, axis=1)
    return float(np.sqrt(np.sum(weight*square)/np.sum(weight)))


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    directories = [args.control, args.replay, *args.candidate]
    loaded, sources, differences, checks = [], [], [], []
    for directory in directories:
        path = directory / "interface-iteration-3d.json"
        report = json.loads(path.read_text())
        child_path = directory / "trial/cube-coupled-trial.json"
        child = json.loads(child_path.read_text())
        assert report["status"] == child["status"] == "complete" and report["spatial_dimensions"] == child["spatial_dimensions"] == 3
        assert report["execution_environment"]["TI_CPU_MAX_NUM_THREADS"] == "1"
        assert child["identical_native_shared_cells"] and child["particle_spacing"] == child["requested_wall_spacing"] == .0625
        assert child["small_cells"] == 16936 and child["fvm_numerics"]["sgs"] == "none"
        for record, archive in ((report, directory), (child, directory / "trial")):
            for row in record["sources"]:
                assert hash_file(ROOT / row["path"]) == row
                copy = archive / "sources" / row["path"]
                if copy.exists():
                    assert hash_file(copy)["sha256"] == row["sha256"]
                sources.append(row)
        mesh_path = directory / "trial/hybrid/solution/mesh.npz"
        mesh = load_native_mesh(mesh_path)
        geo = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
        cut = next(patch for patch in mesh["boundary"] if patch["name"] == "numericalBoundary")
        face_ids = np.arange(cut["start_face"], cut["start_face"]+cut["n_faces"])
        area = np.linalg.norm(geo["face_area_vector"][face_ids], axis=1)
        normal = geo["face_area_vector"][face_ids]/area[:, None]
        first = {}
        for row in report["sweeps"]:
            assert row["runtime_cpu_threads"] == 1
            assert row["accepted_fvm_step"] == 5*row["coupling_step"] and row["accepted_transfer_step"] == row["coupling_step"]+1
            if row["sweep"] == 1:
                assert row["map_replay"]["bitwise_equal"] and row["map_replay"]["numerical_arrays_and_clocks"] == 24
                assert len(row["map_replay"]["sha256"]) == 24
                first[row["coupling_step"]] = row["map_replay"]["sha256"]
            fields_path = ROOT / row["boundary_fields"]["path"]
            assert hash_file(fields_path) == row["boundary_fields"]
            fields = read_arrays(fields_path)
            for mode, key in (("normal_velocity", "normal_residual_rms"), ("tangential_gradient", "gradient_residual_rms")):
                actual = rms(fields["post_"+mode]-fields["applied_"+mode], area)
                difference = abs(actual-row[key])
                assert difference < 2e-14
                differences.append(difference)
            expected = row["normal_residual_rms"] <= report["normal_tolerance"] and row["gradient_residual_rms"] <= report["gradient_tolerance"]
            assert row["converged"] == expected and row["map_evaluations"] == row["sweep"]+1
            for phase in ("applied", "post"):
                np.testing.assert_allclose(np.sum(fields[phase+"_velocity"]*normal, axis=1), fields[phase+"_normal_velocity"], rtol=0, atol=2e-13)
            sources.append(row["boundary_fields"])
        manifest_path = directory / "trial/hybrid/solution/backups/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["coupling_step"] == manifest["vpm_step"] == child["requested_coupling_steps"]
        assert manifest["fvm_step"] == 5*child["requested_coupling_steps"]
        artifacts = {key: manifest_path.parent / name for key, name in manifest["artifacts"].items()}
        for key, artifact in artifacts.items():
            assert hash_file(artifact)["sha256"] == manifest["artifact_sha256"][key]
            sources.append(hash_file(artifact))
        state = decode_state(read_arrays(artifacts["fvm"]))
        field_path = directory / "trial/latest-comparison-fields.npz"
        fields = read_arrays(field_path)
        np.testing.assert_array_equal(state["velocity"][:mesh["n_cells"]], fields["hybrid_velocity"])
        assert int(state["step"]) == 5*child["requested_coupling_steps"]
        np.testing.assert_allclose(state["time"], manifest["time"], rtol=0, atol=1e-12)
        assert report["comparison"] == child["comparison"]
        last = child["comparison"][-1]
        force = independent_cube_force(state, mesh, geo)
        difference = abs(force["drag_coefficient"]-last["hybrid_drag_coefficient"])
        assert difference < 2e-13
        differences.append(difference)
        near = np.max(np.abs(geo["cell_centre"]), axis=1) < 1
        error = fields["hybrid_velocity"]-fields["full_velocity"]
        query = fields["vpm_query_ids"]
        for actual, expected in ((rms(error, geo["cell_volume"]), last["fvm_velocity_rms_over_Uinf"]),
                                 (rms(error[near], geo["cell_volume"][near]), last["fvm_near_body_velocity_rms_over_Uinf"]),
                                 (rms(fields["vpm_velocity"]-fields["full_velocity"][query], geo["cell_volume"][query]), last["vpm_sampled_velocity_rms_over_Uinf"])):
            difference = abs(actual-expected)
            assert difference < 2e-13
            differences.append(difference)
        sources += [hash_file(p) for p in (path, child_path, mesh_path, manifest_path, field_path)]
        loaded.append((report, child, artifacts, state, fields, first))
        checks.append({"directory": str(directory.relative_to(ROOT)), "maximum_sweeps": report["maximum_sweeps"],
                       "independent_force": force, "final_comparison": last, "converged_intervals": report["converged_intervals"],
                       "observed_intervals": report["observed_intervals"]})
    control, replay = loaded[:2]
    assert control[0]["maximum_sweeps"] == 0 and replay[0]["maximum_sweeps"] == 1
    assert control[0]["comparison"] == replay[0]["comparison"]
    counts = {}
    for artifact in ("fvm", "vpm_boundary_condition"):
        a, b = (decode_state(read_arrays(item[2][artifact])) for item in (control, replay))
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        counts[artifact] = len(a)
    with h5py.File(control[2]["vpm"], "r") as a, h5py.File(replay[2]["vpm"], "r") as b:
        names = []
        a.visititems(lambda name, item: names.append(name) if isinstance(item, h5py.Dataset) and item.dtype.kind in "biufc" else None)
        for name in names:
            np.testing.assert_array_equal(a[name][...], b[name][...])
        counts["vpm_numeric_datasets"] = len(names)
    for candidate in loaded[2:]:
        assert [row["full_drag_coefficient"] for row in candidate[0]["comparison"]] == [row["full_drag_coefficient"] for row in control[0]["comparison"]]
        np.testing.assert_array_equal(candidate[4]["full_velocity"], control[4]["full_velocity"])
        assert candidate[5][1] == replay[5][1]
    own = [Path(__file__).resolve(), ROOT / "studies/coupler_accuracy/verify_sampled_face_trials_3d.py"]
    archive = args.output.parent / (args.output.stem+"-sources")
    for path in own:
        target = archive / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        sources.append(hash_file(path))
    result = {"schema": "openonda-interface-iteration-verification-3d/1", "status": "complete", "spatial_dimensions": 3,
              "source_and_artifact_records": len(sources), "independent_metrics": len(differences),
              "maximum_metric_difference": max(differences), "one_sweep_control_bitwise_checkpoint_counts": counts,
              "experiments": checks, "sources": sources,
              "limitations": ["The control identity covers the tested interval count and complete numeric checkpoint datasets, not every possible flow or parallel backend.",
                              "Interface convergence and force/profile agreement with the full FVM remain separate acceptance questions."]}
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: result[key] for key in ("source_and_artifact_records", "independent_metrics", "maximum_metric_difference", "one_sweep_control_bitwise_checkpoint_counts", "experiments")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.control, args.replay, args.output = (p.resolve() for p in (args.control, args.replay, args.output))
    args.candidate = [p.resolve() for p in args.candidate]
    run(args)
