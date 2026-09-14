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
    qualification_control = args.replay_control or args.control
    directories = [qualification_control, args.replay]
    if args.replay_control:
        assert args.replay_control != args.control
        directories.append(args.control)
    directories += args.candidate
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
        precision_mode = "native"
        precision_path = directory / "panel-target-precision-3d.json"
        if precision_path.exists():
            precision = json.loads(precision_path.read_text())
            assert precision["status"] == "complete" and precision["spatial_dimensions"] == 3
            assert precision["mode"] == "f64_auxiliary_panel_queries" and precision["auxiliary_calls"] > 0
            assert precision["stored_panel_precision"] == precision["particle_precision"] == "f32"
            assert hash_file(path) == precision["interface_report"]
            for row in precision["sources"]:
                assert hash_file(ROOT / row["path"]) == row
                archive = directory / "precision-sources" / row["path"]
                if archive.exists():
                    assert hash_file(archive)["sha256"] == row["sha256"]
                sources.append(row)
            query_path = ROOT / precision["first_query"]["path"]
            assert hash_file(query_path) == precision["first_query"]
            query = read_arrays(query_path)
            assert query["original_velocity"].dtype == np.float32 and query["promoted_velocity"].dtype == np.float64
            actual = float(np.sqrt(np.mean(np.sum((query["original_velocity"]-query["promoted_velocity"])**2, axis=1))))
            assert actual == precision["first_query_velocity_difference_rms"]
            sources += [hash_file(precision_path), precision["first_query"]]
            precision_mode = precision["mode"]
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
        first, interval_rows = {}, {}
        for row in report["sweeps"]:
            interval_rows.setdefault(row["coupling_step"], []).append(row)
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
        steps = child["requested_coupling_steps"]
        assert len(child["comparison"]) == steps+1
        assert [row["coupling_step"] for row in child["comparison"]] == list(range(steps+1))
        if report["maximum_sweeps"]:
            assert set(first) == set(interval_rows) == set(range(1, steps+1))
            for rows in interval_rows.values():
                assert [row["sweep"] for row in rows] == list(range(1, len(rows)+1))
                assert len(rows) <= report["maximum_sweeps"]
                assert not any(row["converged"] for row in rows[:-1])
                assert rows[-1]["converged"] or len(rows) == report["maximum_sweeps"]
            assert report["observed_intervals"] == steps
        else:
            assert not interval_rows and report["observed_intervals"] == 0
        assert report["converged_intervals"] == sum(rows[-1]["converged"] for rows in interval_rows.values())
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
        loaded.append((report, child, artifacts, state, fields, first, precision_mode))
        accepted = child["comparison"][1:]
        relative_drag = np.array([row["drag_coefficient_difference"]/row["full_drag_coefficient"] for row in accepted])
        trajectory_metrics = {"coupling_endpoints": len(accepted), "initial_state_excluded": True,
                              "drag_relative_difference_rms_percent": float(100*np.sqrt(np.mean(relative_drag**2))),
                              "drag_relative_difference_maximum_percent": float(100*np.max(np.abs(relative_drag))),
                              **{key: float(np.sqrt(np.mean([row[key]**2 for row in accepted]))) for key in (
                                  "fvm_velocity_rms_over_Uinf", "fvm_near_body_velocity_rms_over_Uinf", "vpm_sampled_velocity_rms_over_Uinf")}}
        checks.append({"directory": str(directory.relative_to(ROOT)), "maximum_sweeps": report["maximum_sweeps"],
                       "auxiliary_panel_precision_mode": precision_mode,
                       "independent_force": force, "final_comparison": last, "converged_intervals": report["converged_intervals"],
                       "observed_intervals": report["observed_intervals"], "trajectory_metrics": trajectory_metrics})
    qualification, replay = loaded[:2]
    control = loaded[2] if args.replay_control else qualification
    candidates = loaded[3:] if args.replay_control else loaded[2:]
    assert qualification[0]["maximum_sweeps"] == control[0]["maximum_sweeps"] == 0 and replay[0]["maximum_sweeps"] == 1
    assert qualification[6] == replay[6] == control[6]
    assert qualification[1]["requested_coupling_steps"] == replay[1]["requested_coupling_steps"]
    assert qualification[0]["comparison"] == replay[0]["comparison"]
    assert control[1]["requested_coupling_steps"] >= qualification[1]["requested_coupling_steps"]
    assert control[0]["comparison"][:len(qualification[0]["comparison"])] == qualification[0]["comparison"]
    counts = {}
    for artifact in ("fvm", "vpm_boundary_condition"):
        a, b = (decode_state(read_arrays(item[2][artifact])) for item in (qualification, replay))
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        counts[artifact] = len(a)
    with h5py.File(qualification[2]["vpm"], "r") as a, h5py.File(replay[2]["vpm"], "r") as b:
        names = []
        a.visititems(lambda name, item: names.append(name) if isinstance(item, h5py.Dataset) and item.dtype.kind in "biufc" else None)
        for name in names:
            np.testing.assert_array_equal(a[name][...], b[name][...])
        counts["vpm_numeric_datasets"] = len(names)
    for candidate in candidates:
        assert candidate[6] == control[6]
        assert candidate[1]["requested_coupling_steps"] == control[1]["requested_coupling_steps"]
        for key in ("coupling_step", "elapsed_flow_time", "physical_time", "full_drag_coefficient"):
            assert [row[key] for row in candidate[0]["comparison"]] == [row[key] for row in control[0]["comparison"]]
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
              "one_sweep_control_intervals": qualification[1]["requested_coupling_steps"],
              "comparison_intervals": control[1]["requested_coupling_steps"],
              "comparison_control_directory": str(args.control.relative_to(ROOT)),
              "experiments": checks, "sources": sources,
              "limitations": ["The control identity covers one_sweep_control_intervals and complete numeric checkpoint datasets. Longer candidate runs replay the first endpoint map bitwise in every interval; they are not a separate longer zero-versus-one-sweep trajectory comparison.",
                              "Trajectory statistics are equally weighted over recorded coupling endpoints, excluding the shared initial state. Final forces and velocity metrics are independently recomputed from fields; earlier endpoints use the trial observer's records.",
                              "Interface convergence and force/profile agreement with the full FVM remain separate acceptance questions."]}
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: result[key] for key in ("source_and_artifact_records", "independent_metrics", "maximum_metric_difference", "one_sweep_control_bitwise_checkpoint_counts", "experiments")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--replay-control", type=Path, help="Optional shorter zero-sweep control paired with --replay; --control remains the candidate trajectory control")
    parser.add_argument("--candidate", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.control, args.replay, args.output = (p.resolve() for p in (args.control, args.replay, args.output))
    args.candidate = [p.resolve() for p in args.candidate]
    if args.replay_control:
        args.replay_control = args.replay_control.resolve()
    run(args)
