#!/usr/bin/env python3
"""Qualify profile observation against an unchanged advancing 3D control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from source.solvers.fvm.io.backup import decode_state
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays


def norms(error):
    return {"vector_rms_over_Uinf": float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
            "streamwise_rms_over_Uinf": float(np.sqrt(np.mean(error[:, 0]**2))),
            "vector_maximum_over_Uinf": float(np.max(np.linalg.norm(error, axis=1)))}


def run(args):
    if args.output.exists() or args.output.with_suffix(".png").exists():
        raise FileExistsError(args.output)
    qualification = json.loads(args.baseline_verification.read_text())
    assert qualification["status"] == "complete" and qualification["comparison_intervals"] == 3
    baseline = ROOT / qualification["comparison_control_directory"]
    observed = args.observed
    sources = [hash_file(args.baseline_verification)]
    for row in qualification["sources"]:
        assert hash_file(ROOT / row["path"]) == row
        sources.append(row)
    reports, checkpoints, fields = [], [], []
    for directory in (baseline, observed):
        report_path = directory / "interface-iteration-3d.json"
        report = json.loads(report_path.read_text())
        child_path = directory / "trial/cube-coupled-trial.json"
        child = json.loads(child_path.read_text())
        assert report["status"] == child["status"] == "complete" and report["spatial_dimensions"] == 3
        assert report["maximum_sweeps"] == 0 and child["requested_coupling_steps"] == 3
        assert child["vpm_dt"] == .05 and child["fvm_dt"] == .01
        assert report["comparison"] == child["comparison"]
        for record in (report, child):
            for row in record["sources"]:
                assert hash_file(ROOT / row["path"]) == row
                sources.append(row)
        manifest_path = directory / "trial/hybrid/solution/backups/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["fvm_step"] == 15 and manifest["vpm_step"] == manifest["coupling_step"] == 3
        artifacts = {key: manifest_path.parent / value for key, value in manifest["artifacts"].items()}
        for key, path in artifacts.items():
            assert hash_file(path)["sha256"] == manifest["artifact_sha256"][key]
            sources.append(hash_file(path))
        path = directory / "trial/latest-comparison-fields.npz"
        fields.append(read_arrays(path))
        reports.append(report)
        checkpoints.append(artifacts)
        sources.extend(hash_file(path) for path in (report_path, child_path, manifest_path, path))
    assert reports[0]["comparison"] == reports[1]["comparison"]
    assert reports[0]["sources"] == reports[1]["sources"]
    assert fields[0].keys() == fields[1].keys()
    for key in fields[0]:
        np.testing.assert_array_equal(fields[0][key], fields[1][key])
    counts = {}
    for name in ("fvm", "vpm_boundary_condition"):
        a, b = (decode_state(read_arrays(paths[name])) for paths in checkpoints)
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        counts[name] = len(a)
    with h5py.File(checkpoints[0]["vpm"], "r") as a, h5py.File(checkpoints[1]["vpm"], "r") as b:
        names_a, names_b = [], []
        for handle, names in ((a, names_a), (b, names_b)):
            handle.visititems(lambda name, item, names=names: names.append(name) if isinstance(item, h5py.Dataset) and item.dtype.kind in "biufc" else None)
        assert names_a == names_b
        for name in names_a:
            np.testing.assert_array_equal(a[name][...], b[name][...])
        counts["vpm_numeric_datasets"] = len(names_a)
    observation_path = observed / "profile-observation-3d.json"
    observation = json.loads(observation_path.read_text())
    assert observation["status"] == "complete" and observation["spatial_dimensions"] == 3
    assert [row["fvm_step"] for row in observation["frames"]] == [0, 5, 10, 15]
    assert observation["profile_every_fvm_steps"] == 5
    for row in observation["sources"]:
        assert hash_file(ROOT / row["path"]) == row
        archive = observed / "profile-sources" / row["path"]
        assert hash_file(archive)["sha256"] == row["sha256"]
        sources.append(row)
    for row in (observation["cadence_report"], observation["geometry"]):
        assert hash_file(ROOT / row["path"]) == row
        sources.append(row)
    geometry = read_arrays(ROOT / observation["geometry"]["path"])
    position, fluid, small = (geometry[key] for key in ("position", "fluid_mask", "small_mask"))
    assert position.shape == (418, 3) and not np.any(position[:, 2])
    np.testing.assert_array_equal(fluid, np.max(np.abs(position), axis=1) > .5)
    np.testing.assert_array_equal(small, fluid & (np.max(np.abs(position), axis=1) <= 1.5))
    ids = geometry["shared_cell_ids"]
    coordinate_error = float(np.max(np.abs(geometry["full_cell_centres"][ids] - geometry["small_cell_centres"])))
    assert coordinate_error == observation["shared_cell_coordinate_maximum_difference"] < 1e-13
    frame_metrics = []
    for row in observation["frames"]:
        assert row["observer_state_bitwise_unchanged"] and len(row["state_fingerprints"]) >= 25
        path = ROOT / row["fields"]["path"]
        assert hash_file(path) == row["fields"]
        source = read_arrays(path)
        for name, field, prefix in (("full_profile", source["full_cell_velocity"], "full"),
                                    ("small_profile", source["small_cell_velocity"], "small"),
                                    ("reference_on_small_stencil", source["full_cell_velocity"][ids], "small")):
            weights, values = geometry[prefix + "_weights"], field[geometry[prefix + "_indices"]]
            reconstructed = np.einsum("qk,qkj->qj", weights, values)
            # Independent summation can reorder the 12 weighted terms. Bound
            # the difference by their absolute sum and float64 roundoff.
            absolute_sum = np.einsum("qk,qkj->qj", np.abs(weights), np.abs(values))
            bound = 2 * weights.shape[1] * np.finfo(float).eps * absolute_sum
            assert np.all(np.abs(reconstructed - source[name]) <= np.maximum(bound, np.finfo(float).tiny))
        if row["fvm_step"] == 0:
            np.testing.assert_array_equal(source["small_profile"], source["reference_on_small_stencil"])
        point = position[fluid]
        region_metrics = {}
        for line_y, name in ((0., "centreline"), (.75, "offaxis_y075")):
            line = point[:, 1] == line_y
            for region, mask in (("whole_line", line), ("exterior", line & (np.abs(point[:, 0]) > 1.5))):
                region_metrics[name + "_" + region + "_vpm"] = norms(source["vpm_profile"][mask] - source["full_profile"][mask])
            line_small = position[small, 1] == line_y
            region_metrics[name + "_small_fvm"] = norms((source["small_profile"] - source["reference_on_small_stencil"])[line_small])
            in_full = small[fluid] & line
            region_metrics[name + "_reference_stencil_difference"] = norms(source["full_profile"][in_full] - source["reference_on_small_stencil"][line_small])
        frame_metrics.append({"physical_time": row["physical_time"], "fvm_step": row["fvm_step"], "metrics": region_metrics})
        sources.append(row["fields"])
    figure, axes = plt.subplot_mosaic([["centre", "offaxis"], ["force", "force"]], figsize=(11, 7), layout="constrained")
    final_time = observation["frames"][-1]["physical_time"]
    for y, name, title in ((0., "centre", "Centreline"), (.75, "offaxis", "Off-axis: y/D = 0.75")):
        ax = axes[name]
        line = position[:, 1] == y
        for key, mask, color, label, style in (("full_profile", fluid, "0.25", "Full FVM", "-."),
                                               ("small_profile", small, "#156d91", "Hybrid FVM", "-"),
                                               ("vpm_profile", fluid, "#ad572f", "VPM", "-")):
            velocity = np.full(len(position), np.nan)
            velocity[mask] = source[key][:, 0]
            ax.plot(position[line, 0], velocity[line], linestyle=style, color=color, label=label)
        ax.axvspan(-1.5, 1.5, color="0.9", alpha=.4)
        if y == 0:
            ax.axvspan(-.5, .5, color="0.7")
        ax.set(title=title, xlabel="x/D", ylabel="uₓ/U∞", xlim=(-3, 10))
        ax.grid(alpha=.15)
        ax.legend(fontsize=9)
    history = reports[1]["comparison"]
    for key, label, color in (("full_drag_coefficient", "Full FVM", "0.25"), ("hybrid_drag_coefficient", "Hybrid FVM", "#156d91")):
        axes["force"].plot([row["physical_time"] for row in history], [row[key] for row in history], "o-", label=label, color=color)
    axes["force"].set(xlabel="Physical time", ylabel="Drag coefficient", title="Advancing observer qualification; original coupling")
    axes["force"].grid(alpha=.15)
    axes["force"].legend(fontsize=9)
    figure.suptitle(f"Fully 3D matched medium cube: profiles at t = {final_time:g}")
    figure.savefig(args.output.with_suffix(".png"), dpi=170)
    plt.close(figure)
    own = Path(__file__).resolve()
    archive = args.output.parent / (args.output.stem + "-sources") / own.relative_to(ROOT)
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(own.read_bytes())
    sources += [hash_file(own), hash_file(observation_path)]
    result = {
        "schema": "openonda-profile-observer-verification-3d/1", "status": "complete", "spatial_dimensions": 3,
        "advancing_comparison_intervals": 3, "comparison_histories_bitwise_equal": True,
        "checkpoint_bitwise_equal_counts": counts, "comparison_arrays_bitwise_equal": len(fields[0]),
        "profile_frames": frame_metrics, "sources": sources, "figure": hash_file(args.output.with_suffix(".png")),
        "limitations": [
            "Neutrality is qualified over three advancing intervals, including initial and endpoint samples; it is not a complete long-run qualification.",
            "The two one-dimensional profiles observe the real 3D solution. Exterior means line points with |x| > 1.5, not a volume norm.",
            "FVM profile reconstruction is independently recomputed from saved cells and weights. VPM profile error norms use saved velocity queries.",
            "This original-coupling short transient does not demonstrate developed-wake or machine-precision agreement.",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("checkpoint_bitwise_equal_counts", "comparison_arrays_bitwise_equal", "profile_frames")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-verification", type=Path, required=True)
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.baseline_verification, args.observed, args.output = (path.resolve() for path in (args.baseline_verification, args.observed, args.output))
    run(args)
