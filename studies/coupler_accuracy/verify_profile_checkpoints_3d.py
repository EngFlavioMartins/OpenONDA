#!/usr/bin/env python3
"""Recompute full and hybrid wall forces from accepted 3D profile checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from source.solvers.fvm.io.backup import decode_state, mesh_hash
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays
from studies.coupler_accuracy.verify_sampled_face_trials_3d import independent_cube_force


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = []
    report_path = args.run / "accepted-fvm-checkpoints-3d.json"
    report = json.loads(report_path.read_text())
    assert report["status"] == "complete" and report["spatial_dimensions"] == 3
    profile_path = args.run / "profile-observation-3d.json"
    assert hash_file(profile_path) == report["profile_report"]
    profiles = json.loads(profile_path.read_text())
    assert profiles["status"] == "complete" and profiles["spatial_dimensions"] == 3
    child_path = args.run / "trial/cube-coupled-trial.json"
    child = json.loads(child_path.read_text())
    assert child["status"] == "complete" and child["spatial_dimensions"] == 3
    assert child["fvm_dt"] == .01 and child["particle_spacing"] == child["requested_wall_spacing"] == .0625
    substeps = round(child["vpm_dt"] / child["fvm_dt"])
    assert child["vpm_dt"] == substeps * child["fvm_dt"]
    end_tick = substeps * child["requested_coupling_steps"]
    ticks = sorted({*range(0, end_tick + 1, profiles["profile_every_fvm_steps"]), end_tick})
    assert [row["fvm_step"] for row in report["frames"]] == ticks
    assert [row["fvm_step"] for row in profiles["frames"]] == ticks
    for item, archive_directory in ((report, "checkpoint-sources"), (profiles, "profile-sources")):
        for row in item["sources"]:
            assert hash_file(ROOT / row["path"]) == row
            assert hash_file(args.run / archive_directory / row["path"])["sha256"] == row["sha256"]
            sources.append(row)
    qualifier = None
    if args.observer_qualification:
        qualifier = json.loads(args.observer_qualification.read_text())
        assert qualifier["status"] == "complete" and qualifier["comparison_histories_bitwise_equal"]
        assert qualifier["checkpoint_bitwise_equal_counts"] == {"fvm": 17, "vpm_boundary_condition": 11, "vpm_numeric_datasets": 11}
        assert any(row["path"] == str(profile_path.relative_to(ROOT)) and row == hash_file(profile_path) for row in qualifier["sources"])
        for row in qualifier["sources"]:
            assert hash_file(ROOT / row["path"]) == row
            sources.append(row)
        sources.append(hash_file(args.observer_qualification))
    meshes, geometry = {}, {}
    for name, row in report["meshes"].items():
        assert name in ("full", "hybrid") and hash_file(ROOT / row["path"]) == row
        meshes[name] = load_native_mesh(ROOT / row["path"])
        geometry[name] = compute_mesh_geometry(meshes[name], gradient_scheme="gauss", compute_lsq=False)
        sources.append(row)
    assert meshes["full"]["n_cells"] == 53752 and meshes["hybrid"]["n_cells"] == 16936
    differences, rows = [], []
    for checkpoint_frame, profile_frame in zip(report["frames"], profiles["frames"], strict=True):
        for key in ("fvm_step", "coupling_step", "physical_time"):
            assert checkpoint_frame[key] == profile_frame[key]
        assert checkpoint_frame["state_bitwise_unchanged"] and profile_frame["observer_state_bitwise_unchanged"]
        assert checkpoint_frame["state_fingerprints"] == profile_frame["state_fingerprints"]
        comparison = child["comparison"][checkpoint_frame["coupling_step"]]
        assert comparison["physical_time"] == checkpoint_frame["physical_time"]
        field_path = ROOT / profile_frame["fields"]["path"]
        assert hash_file(field_path) == profile_frame["fields"]
        fields = read_arrays(field_path)
        force_record = {}
        for name, checkpoint in checkpoint_frame["checkpoints"].items():
            path = ROOT / checkpoint["path"]
            assert hash_file(path) == checkpoint
            raw = read_arrays(path)
            metadata = json.loads(raw["metadata"].item())
            assert metadata["mesh_hash"] == mesh_hash(meshes[name])
            state = decode_state(raw)
            assert int(state["step"]) == checkpoint_frame["fvm_step"]
            np.testing.assert_allclose(float(state["time"]), comparison["elapsed_flow_time"], rtol=0, atol=1e-12)
            field_name = "full_cell_velocity" if name == "full" else "small_cell_velocity"
            np.testing.assert_array_equal(state["velocity"][:meshes[name]["n_cells"]], fields[field_name])
            force = independent_cube_force(state, meshes[name], geometry[name])
            key = name + "_drag_coefficient"
            assert checkpoint_frame[key] == comparison[key]
            difference = abs(force["drag_coefficient"] - comparison[key])
            assert difference < 2e-13
            differences.append(difference)
            force_record[name] = force
            sources.append(checkpoint)
        sources.append(profile_frame["fields"])
        rows.append({"fvm_step": checkpoint_frame["fvm_step"], "physical_time": checkpoint_frame["physical_time"], "forces": force_record})
    own = [Path(__file__).resolve(), ROOT / "studies/coupler_accuracy/verify_sampled_face_trials_3d.py"]
    for path in own:
        target = args.output.parent / (args.output.stem + "-sources") / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        sources.append(hash_file(path))
    sources.extend(hash_file(path) for path in (report_path, profile_path, child_path))
    result = {
        "schema": "openonda-profile-checkpoint-verification-3d/1", "status": "complete", "spatial_dimensions": 3,
        "coupling_intervals": child["requested_coupling_steps"], "profile_frames": len(rows),
        "independently_recomputed_forces": len(differences), "maximum_force_difference": max(differences),
        "observer_qualification_intervals": qualifier["advancing_comparison_intervals"] if qualifier else None,
        "frames": rows, "sources": sources,
        "limitations": [
            "Both wall forces are reconstructed independently at every saved profile time. Intermediate coupling times use the original observer's records.",
            "An advancing observer control is claimed only when supplied for this exact run; canonical snapshots alone do not establish a separate unobserved trajectory.",
            "These checks establish observation consistency, not agreement between the hybrid and full flow or a statistically developed wake.",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("profile_frames", "independently_recomputed_forces", "maximum_force_difference", "observer_qualification_intervals")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--observer-qualification", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.run, args.output = args.run.resolve(), args.output.resolve()
    if args.observer_qualification:
        args.observer_qualification = args.observer_qualification.resolve()
    run(args)
