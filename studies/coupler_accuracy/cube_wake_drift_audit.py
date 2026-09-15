"""Audit the preserved September cube run without changing its output files.

Run after capture_cube_wake_baseline.py and the baseline commit. Native 3D
fields are compared at the coupled mesh's cell centres using the same affine
reconstruction as the tutorial. This measures discrepancies, not causation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = ROOT / "studies/coupler_accuracy/results/cube-wake-drift-2026-09-15/baseline"


def comparison_geometry(mesh_file, cache_directory):
    """Cache only cell centres/volumes, keyed by native mesh and geometry code.

    Rebuilding all face geometry for every completed checkpoint needlessly
    competes with the advancing MPI run for memory. Raw solutions are untouched.
    """
    from source.solvers.fvm.io.mesh_storage import load_native_mesh
    from source.solvers.fvm.mesh.geometry import compute_mesh_geometry

    digest = hashlib.sha256(mesh_file.read_bytes())
    module_directory = Path(sys.modules[compute_mesh_geometry.__module__].__file__).parent
    for source in sorted(module_directory.rglob("*.py")):
        digest.update(str(source.relative_to(module_directory)).encode())
        digest.update(source.read_bytes())
    cache_directory.mkdir(parents=True, exist_ok=True)
    cached = cache_directory / f"{digest.hexdigest()}.npz"
    if cached.exists():
        with np.load(cached, allow_pickle=False) as data:
            return {name: data[name] for name in ("cell_centre", "cell_volume")}
    geometry = compute_mesh_geometry(load_native_mesh(mesh_file), compute_lsq=False)
    result = {name: geometry[name].copy() for name in ("cell_centre", "cell_volume")}
    del geometry
    with tempfile.NamedTemporaryFile(dir=cache_directory, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **result)
    try:
        os.replace(temporary, cached)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def frame(directory, name, time):
    matches = [
        directory / row.attrib["file"]
        for row in ET.parse(directory / f"{name}.pvd").iter("DataSet")
        if abs(float(row.attrib["timestep"]) - time) < 1e-8
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one saved state at {time} in {directory}")
    return matches[0]


def ordered_fields(path, count):
    import pyvista as pv

    grid = pv.read(path)
    ids = np.asarray(grid.cell_data["global_cell_id"], dtype=int)
    if not np.array_equal(np.sort(ids), np.arange(count)):
        raise ValueError(f"Native cells must occur exactly once in {path}")
    result = {}
    for key in ("velocity", "vorticity", "eddy_viscosity", "kinematic_pressure"):
        values = np.asarray(grid.cell_data[key], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Nonfinite {key} in {path}")
        result[key] = np.empty_like(values)
        result[key][ids] = values
    return result


def regions(points):
    x = points[:, 0]
    return {
        "whole": np.ones(len(x), dtype=bool),
        "upstream": x < -0.6,
        "body_flanks": (x > -0.5) & (x < 0.5),
        "wake_inner": (x > 0.6) & (x < 0.89),
        "wake_blend": (x >= 0.89) & (x < 1.25),
        "wake_boundary": x >= 1.25,
    }


def vector_metrics(actual, expected, weights):
    difference = actual - expected
    return {
        "rms": float(np.sqrt(np.average(np.sum(difference**2, axis=-1), weights=weights))),
        "component_rms": np.sqrt(np.average(difference**2, axis=0, weights=weights)).tolist(),
        "component_bias": np.average(difference, axis=0, weights=weights).tolist(),
        "maximum": float(np.linalg.norm(difference, axis=-1).max()),
    }


def reattachment_x(points, velocity):
    x, u = points[:, 0], velocity[:, 0]
    order = np.argsort(x)
    x, u = x[order], u[order]
    crossings = np.flatnonzero((x[:-1] > 0.5) & (u[:-1] < 0) & (u[1:] >= 0))
    if not len(crossings):
        return None
    j = crossings[0]
    return float(x[j] - u[j] * (x[j + 1] - x[j]) / (u[j + 1] - u[j]))


def audit(args):
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source_tree.resolve()))
    from source.solvers.fvm.io.mesh_storage import load_native_mesh
    from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
    from source.solvers.fvm.sampling.fields import _PointProbe

    b = args.baseline
    metadata = {p.stem: json.loads(p.read_text()) for p in (b / "metadata").glob("*.json")}
    telemetry = {}
    for name in ("coupler", "coupled_fvm", "reference_fine"):
        with gzip.open(b / "telemetry" / f"{name}.jsonl.gz", "rt") as stream:
            telemetry[name] = [json.loads(line) for line in stream if line.strip()]
    geometry = {}
    for name in ("coupled", "reference_fine"):
        mesh = load_native_mesh(b / "solution" / name / "mesh.npz")
        g = compute_mesh_geometry(mesh, compute_lsq=False)
        geometry[name] = {key: g[key] for key in ("cell_centre", "cell_volume")}
    centres = geometry["coupled"]["cell_centre"]
    volumes = geometry["coupled"]["cell_volume"]
    reference_centres = geometry["reference_fine"]["cell_centre"]
    distance, nearest = cKDTree(reference_centres).query(centres)
    probe = _PointProbe(centres, k=12, reconstruction="affine")
    masks = regions(centres)
    # Interior cell centroids cannot sample the solid or an outer-box extension.
    if np.any(np.all(np.abs(centres) <= 0.5, axis=1)):
        raise ValueError("Coupled native centres include the solid cube")
    vpm_line = pd.read_csv(b / "samples/coupled/vpm_centreline.csv")
    output = {
        "baseline_manifest_sha256": hashlib.sha256((b / "manifest.json").read_bytes()).hexdigest(),
        "source_tree": str(args.source_tree.resolve()),
        "source_revision": (args.source_tree / "SOURCE_REVISION").read_text().strip(),
        "velocity_scale": "U_infinity=1 m/s; differences reported as fractions, not percent",
        "native_comparison": "volume-weighted 3D; affine k=12 reference at coupled cell centroids",
        "reattachment": "first negative-to-positive centreline crossing downstream of x=0.5; linear between native sample points",
        "mesh": {
            "coupled_cells": len(centres),
            "reference_cells": len(reference_centres),
            "coincident_cell_centres": int(np.count_nonzero(distance < 1e-9)),
            "nearest_reference_centre_distance_quantiles": np.quantile(
                distance, [0, 0.5, 0.95, 1]
            ).tolist(),
            "coupled_size_quantiles": np.quantile(np.cbrt(volumes), [0, 0.5, 0.95, 1]).tolist(),
            "reference_size_at_nearest_quantiles": np.quantile(
                np.cbrt(geometry["reference_fine"]["cell_volume"][nearest]), [0, 0.5, 0.95, 1]
            ).tolist(),
        },
        "identical_fvm_settings": {
            key: metadata["coupled_fvm"]["configuration"][key]
            == metadata["reference_fine"]["configuration"][key]
            for key in ("schemes", "pimple", "transport", "turbulence")
        },
        "times": [],
    }
    for time in args.times:
        fields = {}
        for name, stem in (("coupled", "coupled_replacement_flow"), ("reference_fine", "fine")):
            fields[name] = ordered_fields(
                frame(b / "solution" / name, stem, time), len(geometry[name]["cell_centre"])
            )
        reference = fields["reference_fine"]
        sampled = {
            key: probe._interpolate(value, reference_centres) for key, value in reference.items()
        }
        row = {"time": time, "native_3d": {}}
        coupled = fields["coupled"]
        for name, mask in masks.items():
            record = {"volume": float(volumes[mask].sum()), "cells": int(mask.sum())}
            for key in ("velocity", "vorticity"):
                record[key] = vector_metrics(coupled[key][mask], sampled[key][mask], volumes[mask])
            for solver, values in (("coupled", coupled), ("reference", sampled)):
                record[solver] = {
                    "mean_velocity": np.average(
                        values["velocity"][mask], axis=0, weights=volumes[mask]
                    ).tolist(),
                    "enstrophy": float(
                        0.5 * np.sum(volumes[mask] * np.sum(values["vorticity"][mask] ** 2, axis=1))
                    ),
                    "mean_eddy_viscosity": float(
                        np.average(values["eddy_viscosity"][mask], weights=volumes[mask])
                    ),
                    "reverse_flow_volume": float(
                        volumes[mask & (values["velocity"][:, 0] < 0)].sum()
                    ),
                }
            row["native_3d"][name] = record
        with np.load(b / "samples/coupled/comparison" / f"fields_t{time:.9f}.npz") as cached:
            line = vpm_line[np.isclose(vpm_line["time"], time, atol=1e-8, rtol=0)]
            points = cached["centreline_points"]
            if not np.allclose(line[["position_" + a for a in "xyz"]], points, rtol=0, atol=1e-6):
                raise ValueError("VPM and reference profile coordinates differ")
            row["reattachment_x"] = {
                "reference": reattachment_x(points, cached["centreline_reference"]),
                "coupled_fvm": reattachment_x(points, cached["centreline_fvm"]),
                "vpm": reattachment_x(points, line[["velocity_" + a for a in "xyz"]].to_numpy()),
            }
        with h5py.File(b / "solution/coupled" / f"vpm_{round(time * 100):06d}.h5") as h5:
            p = h5["particles"]
            position, strength = np.array(p["position"]), np.array(p["vortex_strength"])
            row["particles"] = {
                "count": len(position),
                "bounds": [position.min(axis=0).tolist(), position.max(axis=0).tolist()],
                "strength_l1": float(np.linalg.norm(strength, axis=1).sum()),
            }
        row["telemetry"] = {}
        for name, records in telemetry.items():
            item = min(records, key=lambda r: abs(r["time"] - time))
            if abs(item["time"] - time) > 1e-8:
                raise ValueError(f"No telemetry at matched time {time} for {name}")
            keys = (
                (
                    "fvm_boundary_trace",
                    "transfer",
                    "timing_seconds",
                    "interface_iteration",
                    "gbd_moment_recovery",
                )
                if name == "coupler"
                else ("time_step_size", "max_courant_number", "residuals")
            )
            row["telemetry"][name] = {key: item.get(key) for key in keys}
        output["times"].append(row)
        (args.output / "audit.json").write_text(json.dumps(output, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "time": time,
                    "velocity_3d_rms": row["native_3d"]["whole"]["velocity"]["rms"],
                    "reattachment_x": row["reattachment_x"],
                }
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--times", type=float, nargs="+", default=[1, 3, 5, 6, 7, 8, 10, 12, 15])
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        audit(args)


if __name__ == "__main__":
    main()
