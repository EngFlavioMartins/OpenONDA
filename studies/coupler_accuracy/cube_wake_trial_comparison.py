"""Compare completed native 3D trial states against the committed cube baseline.

Only exactly coincident physical times are accepted. Both coupled cases must
use the same native mesh. The fine reference is reconstructed at those cell
centres, with volume weights, using the tutorial's affine k=12 interpolation.
No simulation files are changed and no missing states are time-interpolated.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

from cube_wake_drift_audit import (
    comparison_geometry,
    frame,
    ordered_fields,
    reattachment_x,
    regions,
    vector_metrics,
)
from cube_wake_particle_probe import rms
from cube_wake_symmetry import project_rotations, rotation_probe
import numpy as np
import pandas as pd
import pyvista as pv
from threadpoolctl import threadpool_limits


def exact_rows(path, time):
    data = pd.read_csv(path)
    result = data[np.isclose(data["time"], time, rtol=0, atol=1e-8)]
    if result.empty:
        raise ValueError(f"No exact sample at t={time:g} in {path}")
    return result


def read_diagnostics(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        # An advancing run may still be writing its last JSON line.
        return [json.loads(line) for line in stream if line.endswith("\n") and line.strip()]


def drag_history(samples, start, end, output):
    """Use every exactly matched force sample, with trapezoidal time weights."""
    tables = {}
    for name, directory in samples.items():
        data = pd.read_csv(directory / "forces_history.csv")
        data = data[(data["time"] >= start - 1e-8) & (data["time"] <= end + 1e-8)].copy()
        if not (data["patch"] == "cube").all():
            raise ValueError("Force history contains another patch")
        data["time_key"] = data["time"].round(8)
        if data["time_key"].duplicated().any():
            raise ValueError("Force history contains duplicate times")
        tables[name] = data.set_index("time_key")
    common = sorted(set.intersection(*(set(t.index) for t in tables.values())))
    if not common:
        raise ValueError("No common force sample times")
    time = tables["reference"].loc[common, "time"].to_numpy()
    values = {}
    for name, table in tables.items():
        selected = table.loc[common]
        np.testing.assert_allclose(selected["time"], time, atol=1e-8, rtol=0)
        values[name] = selected["drag_coefficient"].to_numpy()
        if not np.all(np.isfinite(values[name])):
            raise ValueError("Nonfinite drag coefficient")
    if abs(time[0] - start) > 1e-8 or abs(time[-1] - end) > 1e-8:
        raise ValueError("The matched force interval is incomplete")
    weights = np.ones(len(time))
    if len(time) > 1:
        intervals = np.diff(time)
        weights[0], weights[-1] = intervals[0] / 2, intervals[-1] / 2
        weights[1:-1] = (intervals[:-1] + intervals[1:]) / 2
    reference_rms = float(np.sqrt(np.average(values["reference"] ** 2, weights=weights)))
    report = {
        "method": "All common physical sample times; trapezoidal time weights; relative RMS divides by reference Cd RMS",
        "time_start": float(time[0]),
        "time_end": float(time[-1]),
        "matched_samples": len(time),
        "available_samples": {name: len(table) for name, table in tables.items()},
        "maximum_sample_gap": float(np.diff(time).max()) if len(time) > 1 else None,
        "errors": {},
    }
    for name in ("baseline", "candidate"):
        difference = values[name] - values["reference"]
        error = float(np.sqrt(np.average(difference**2, weights=weights)))
        report["errors"][name] = {
            "rms": error,
            "relative_rms": error / reference_rms,
            "mean_bias": float(np.average(difference, weights=weights)),
            "maximum_absolute": float(np.abs(difference).max()),
        }
    pd.DataFrame({"time": time, **values}).to_csv(output / "drag_history.csv", index=False)
    return report


def compare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source_tree.resolve()))
    from source.solvers.fvm.sampling.fields import _PointProbe

    directories = {
        "baseline": args.baseline / "solution/coupled",
        "candidate": args.trial / "solution",
        "reference": args.baseline / "solution/reference_fine",
    }
    samples = {
        "baseline": args.baseline / "samples/coupled",
        "candidate": args.trial / "samples",
        "reference": args.baseline / "samples/reference_fine",
    }
    diagnostics = {
        "baseline": read_diagnostics(args.baseline / "telemetry/coupler.jsonl.gz"),
        "candidate": read_diagnostics(args.trial / "solution/coupler_diagnostics.jsonl"),
    }
    mesh_hash = {
        name: hashlib.sha256((path / "mesh.npz").read_bytes()).hexdigest()
        for name, path in directories.items()
    }
    if mesh_hash["baseline"] != mesh_hash["candidate"]:
        raise ValueError("The two coupled trials must have the identical native mesh")
    centres, volumes = {}, {}
    for name, path in directories.items():
        if name == "candidate":
            centres[name], volumes[name] = centres["baseline"], volumes["baseline"]
            continue
        geometry = comparison_geometry(path / "mesh.npz", args.output.parent / "geometry-cache")
        centres[name], volumes[name] = geometry["cell_centre"], geometry["cell_volume"]
    volume_probe = _PointProbe(centres["baseline"], k=12, reconstruction="affine")
    masks = regions(centres["baseline"])
    points, rotations, permutations = rotation_probe(0.1)
    symmetry_probes = {
        name: _PointProbe(points, k=12, reconstruction="affine") for name in directories
    }
    report = {
        "description": __doc__,
        "trial": str(args.trial.resolve()),
        "trial_provenance": json.loads((args.trial / "trial.json").read_text()),
        "baseline_manifest_sha256": hashlib.sha256(
            (args.baseline / "manifest.json").read_bytes()
        ).hexdigest(),
        "mesh_sha256": mesh_hash,
        "units": "U_infinity=D=rho=1; velocity differences are fractions of U_infinity",
        "wake_boundary_region": "All native fluid cells with x/D >= 1.25, up to the FVM boundary at 1.5",
        "sampled_plane_comparison": "All three velocity components at the actual saved z=0 VPM wake probes (spacing 0.12); point RMS, not a 3D volume integral",
        "symmetry_probe_spacing": 0.1,
        "symmetry_projection": "One interpolation per point followed by exact C4 orbit permutations",
        "reattachment": "First downstream negative-to-positive crossing; null means not resolved inside available profile coverage, not zero recirculation.",
        "times": [],
    }
    report["drag_history"] = drag_history(samples, min(args.times), max(args.times), args.output)
    for time in args.times:
        fields = {
            name: ordered_fields(
                frame(path, "fine" if name == "reference" else "coupled_replacement_flow", time),
                len(centres[name]),
            )
            for name, path in directories.items()
        }
        reference = volume_probe._interpolate(fields["reference"]["velocity"], centres["reference"])
        row = {"time": time, "native_3d": {}, "symmetry": {}, "profiles": {}, "drag": {}}
        row["boundary_trace"] = {}
        for name, records in diagnostics.items():
            matching = [r for r in records if abs(r["time"] - time) < 1e-8]
            if len(matching) != 1:
                raise ValueError(f"Expected one {name} boundary diagnostic at t={time:g}")
            row["boundary_trace"][name] = matching[0]["fvm_boundary_trace"]
        for name in ("baseline", "candidate"):
            row["native_3d"][name] = {
                region: vector_metrics(
                    fields[name]["velocity"][mask], reference[mask], volumes[name][mask]
                )
                for region, mask in masks.items()
            }
        projected, sampled = {}, {}
        for name in directories:
            sampled[name] = symmetry_probes[name]._interpolate(
                fields[name]["velocity"], centres[name]
            )
            projected[name] = project_rotations(sampled[name], rotations, permutations)
        for name in ("baseline", "candidate"):
            error = sampled[name] - sampled["reference"]
            symmetric = projected[name] - projected["reference"]
            row["symmetry"][name] = {
                "total_error_rms": rms(error),
                "symmetric_error_rms": rms(symmetric),
                "asymmetric_error_rms": rms(error - symmetric),
                "own_asymmetry_rms": rms(sampled[name] - projected[name]),
                "orthogonality_residual": float(
                    np.mean(np.sum(symmetric * (error - symmetric), axis=1))
                ),
            }
        for line_name in ("centreline", "offaxis_y075"):
            old_line = exact_rows(samples["baseline"] / f"vpm_{line_name}.csv", time)
            new_line = exact_rows(samples["candidate"] / f"vpm_{line_name}.csv", time)
            coordinates = ["position_" + a for a in "xyz"]
            velocity_names = ["velocity_" + a for a in "xyz"]
            line_points = old_line[coordinates].to_numpy()
            np.testing.assert_allclose(new_line[coordinates], line_points, atol=1e-9, rtol=0)
            fluid = ~np.all(np.abs(line_points) <= 0.5 + 1e-10, axis=1)
            inside = fluid & np.all(np.abs(line_points) <= 1.5, axis=1)
            values = {
                "baseline_vpm": old_line[velocity_names].to_numpy(),
                "candidate_vpm": new_line[velocity_names].to_numpy(),
            }
            for name in directories:
                valid = fluid if name == "reference" else inside
                values[name] = np.full_like(line_points, np.nan)
                probe = _PointProbe(line_points[valid], k=12, reconstruction="affine")
                values[name][valid] = probe._interpolate(fields[name]["velocity"], centres[name])
            rows = dict(zip(coordinates, line_points.T, strict=True))
            for name, value in values.items():
                value[~fluid] = np.nan
                for component, column in zip("xyz", value.T, strict=True):
                    rows[f"{name}_velocity_{component}"] = column
            pd.DataFrame(rows).to_csv(args.output / f"{line_name}_t{time:g}.csv", index=False)
            if line_name == "centreline":
                row["profiles"]["reattachment_x"] = {
                    name: reattachment_x(line_points, value) for name, value in values.items()
                }
        planes = {
            name: pv.read(frame(samples[name], "vpm_wake_slice_z0", time))
            for name in ("baseline", "candidate")
        }
        plane_points = np.asarray(planes["baseline"].points, dtype=float)
        np.testing.assert_array_equal(planes["candidate"].points, plane_points)
        fluid = ~np.all(np.abs(plane_points) <= 0.5 + 1e-8, axis=1)
        plane_reference = np.full_like(plane_points, np.nan)
        plane_reference[fluid] = _PointProbe(
            plane_points[fluid], k=12, reconstruction="affine"
        )._interpolate(fields["reference"]["velocity"], centres["reference"])
        plane_masks = {
            "wake": fluid & (plane_points[:, 0] > 0.5),
            "near_xmax": fluid & (plane_points[:, 0] >= 1.25) & (plane_points[:, 0] <= 1.75),
            "outer_wake": fluid & (plane_points[:, 0] > 1.5),
        }
        plane_rows = dict(zip(["position_" + a for a in "xyz"], plane_points.T, strict=True))
        plane_values = {"reference": plane_reference}
        row["sampled_plane"] = {}
        for name, plane in planes.items():
            velocity = np.asarray(plane.point_data["velocity"], dtype=float).copy()
            if not np.all(np.isfinite(velocity[fluid])):
                raise ValueError("Nonfinite velocity at a fluid-plane probe")
            velocity[~fluid] = np.nan
            plane_values[name] = velocity
            row["sampled_plane"][name] = {
                region: {
                    "points": int(mask.sum()),
                    **vector_metrics(velocity[mask], plane_reference[mask], np.ones(mask.sum())),
                }
                for region, mask in plane_masks.items()
            }
        for name, value in plane_values.items():
            for component, column in zip("xyz", value.T, strict=True):
                plane_rows[f"{name}_velocity_{component}"] = column
        pd.DataFrame(plane_rows).to_csv(args.output / f"wake_plane_z0_t{time:g}.csv", index=False)
        for name, directory in samples.items():
            force = exact_rows(directory / "forces_history.csv", time)
            if len(force) != 1 or force.iloc[0]["patch"] != "cube":
                raise ValueError(f"Expected exactly one cube wall-force record at t={time:g}")
            row["drag"][name] = float(force.iloc[0]["drag_coefficient"])
        report["times"].append(row)
        (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "time": time,
                    "velocity_rms": {k: v["whole"]["rms"] for k, v in row["native_3d"].items()},
                    "drag": row["drag"],
                    "symmetry": row["symmetry"],
                    "reattachment": row["profiles"]["reattachment_x"],
                }
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--times", type=float, nargs="+", required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        compare(args)


if __name__ == "__main__":
    main()
