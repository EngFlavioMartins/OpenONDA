#!/usr/bin/env python3
"""Check the one-substep wrapper against an unmodified advancing cube control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from source.solvers.fvm.io.backup import decode_state
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = []

    def checked(row):
        assert hash_file(ROOT / row["path"]) == row
        sources.append(row)
        return ROOT / row["path"]

    def report(path):
        sources.append(hash_file(path))
        value = json.loads(path.read_text())
        assert value["status"] == "complete"
        for row in value.get("sources", []):
            checked(row)
        return value

    wrapped = report(args.wrapped / "inviscid-subcycling-3d.json")
    assert wrapped["spatial_dimensions"] == 3 and wrapped["inviscid_substeps"] == 1
    assert wrapped["requested_exchanges"] == 3 and wrapped["outer_dt"] == .05
    for row in wrapped["sources"]:
        archive = args.wrapped / "subcycling-sources" / row["path"]
        assert hash_file(archive)["sha256"] == row["sha256"]
        sources.append(hash_file(archive))
    for row in wrapped["child_reports"]:
        checked(row)
    assert len(wrapped["outer_advances"]) == 3
    for step, row in enumerate(wrapped["outer_advances"], 1):
        assert row["start_step"] == step - 1 and row["accepted_step"] == step
        assert row["schedule_checks_passed"] and row["rk_preserved_outer_clock_and_population"]
        assert len(row["rk_calls"]) == 1 and row["rk_calls"][0]["dt"] == .05
        assert [item["index"] for item in row["stages"]] == [0, 1]
        np.testing.assert_allclose([item["time"] for item in row["stages"]], [.05 * (step - 1), .05 * step], rtol=0, atol=3e-17)
        assert [item["dt"] for item in row["diffusion_calls"]] == [.05]
        assert row["stabilization_phases"] == ["pre_evolution", "pre_strength", "post_evolution", "post_step"]

    datasets, histories, profiles, checkpoints = [], [], [], []
    for directory in (args.control, args.wrapped):
        iteration = report(directory / "interface-iteration-3d.json")
        trial = report(directory / "trial/cube-coupled-trial.json")
        assert iteration["maximum_sweeps"] == 0 and trial["requested_coupling_steps"] == 3
        assert trial["fvm_dt"] == .01 and trial["vpm_dt"] == .05
        assert iteration["comparison"] == trial["comparison"]
        histories.append(trial["comparison"])
        manifest_path = directory / "trial/hybrid/solution/backups/manifest.json"
        sources.append(hash_file(manifest_path))
        manifest = json.loads(manifest_path.read_text())
        assert manifest["fvm_step"] == 15 and manifest["vpm_step"] == manifest["coupling_step"] == 3
        artifacts = {}
        for key, name in manifest["artifacts"].items():
            path = manifest_path.parent / name
            assert hash_file(path)["sha256"] == manifest["artifact_sha256"][key]
            sources.append(hash_file(path))
            artifacts[key] = path
        datasets.append(artifacts)
        profiles.append(report(directory / "profile-observation-3d.json"))
        checkpoints.append(report(directory / "accepted-fvm-checkpoints-3d.json"))
    assert histories[0] == histories[1]
    counts = {}
    for key in ("fvm", "vpm_boundary_condition"):
        left, right = (decode_state(read_arrays(row[key])) for row in datasets)
        assert left.keys() == right.keys()
        for name in left:
            np.testing.assert_array_equal(left[name], right[name])
        counts[key] = len(left)
    with h5py.File(datasets[0]["vpm"], "r") as left, h5py.File(datasets[1]["vpm"], "r") as right:
        names = [[], []]
        for handle, items in zip((left, right), names, strict=True):
            handle.visititems(lambda name, value, items=items: items.append(name) if isinstance(value, h5py.Dataset) and value.dtype.kind in "biufc" else None)
        assert names[0] == names[1]
        for name in names[0]:
            np.testing.assert_array_equal(left[name][...], right[name][...])
        counts["vpm_numeric_datasets"] = len(names[0])
    profile_arrays, checkpoint_entries = 0, 0
    for record in profiles + checkpoints:
        assert [row["fvm_step"] for row in record["frames"]] == [0, 5, 10, 15]
    for left, right in zip(profiles[0]["frames"], profiles[1]["frames"], strict=True):
        a, b = (read_arrays(checked(row["fields"])) for row in (left, right))
        assert a.keys() == b.keys()
        for name in a:
            np.testing.assert_array_equal(a[name], b[name])
        profile_arrays += len(a)
    for left, right in zip(checkpoints[0]["frames"], checkpoints[1]["frames"], strict=True):
        for solver in ("full", "hybrid"):
            a, b = (decode_state(read_arrays(checked(row["checkpoints"][solver]))) for row in (left, right))
            assert a.keys() == b.keys()
            for name in a:
                np.testing.assert_array_equal(a[name], b[name])
            checkpoint_entries += len(a)
    for directory in (args.control, args.wrapped):
        sources.append(hash_file(directory / "trial/latest-comparison-fields.npz"))
    a, b = (read_arrays(directory / "trial/latest-comparison-fields.npz") for directory in (args.control, args.wrapped))
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    sources += [hash_file(Path(__file__).resolve())]
    result = {
        "schema": "openonda-inviscid-subcycling-wrapper-verification-3d/1", "status": "complete",
        "spatial_dimensions": 3, "advancing_intervals": 3,
        "control_directory": str(args.control.relative_to(ROOT)), "wrapped_directory": str(args.wrapped.relative_to(ROOT)),
        "histories_bitwise_equal": True, "final_comparison_arrays_bitwise_equal": len(a),
        "final_checkpoint_entries_bitwise_equal": counts,
        "profile_arrays_bitwise_equal": profile_arrays, "accepted_checkpoint_entries_bitwise_equal": checkpoint_entries,
        "sources": list({row["path"]: row for row in sources}.values()),
        "limitations": ["This verifies the one-substep observation/wrapper path over three intervals. Smaller RK steps intentionally change the numerical solution and require separate comparison.",
                        "No force or velocity accuracy improvement is claimed by bitwise preservation of the control."],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("sources", "limitations")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--wrapped", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.control, args.wrapped, args.output = (value.resolve() for value in (args.control, args.wrapped, args.output))
    run(args)
