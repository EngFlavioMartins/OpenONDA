"""Measure local mesh agreement and autonomous cylinder force histories."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry

from .analyze_force_cycles import FIELDS, complete_cycles, window_statistics
from .run_coupled_checkpoint_control import digest, read_force_history

CASE = Path(__file__).resolve().parents[3] / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"


def mesh_statistics(paths):
    meshes, result = [], {}
    for label, path in paths.items():
        mesh = load_native_mesh(path)
        geometry = compute_mesh_geometry(mesh, compute_lsq=False)
        centres = geometry["cell_centre"]
        selected = np.linalg.norm(centres[:, :2], axis=1) < 1.0
        wall = next(patch for patch in mesh["boundary"] if patch["name"] == "cylinder")
        faces = np.arange(wall["start_face"], wall["start_face"] + wall["n_faces"])
        meshes.append(
            (
                centres[selected],
                geometry["cell_volume"][selected],
                geometry["face_centre"][faces],
                geometry["face_area_vector"][faces],
            )
        )
        result[label] = {
            "path": str(path),
            "sha256": digest(path),
            "cylinder_face_count": len(faces),
            "cells_within_radius_1m": int(selected.sum()),
            "local_in_plane_cell_sizes_m": np.unique(mesh["cell_sizes"][selected]).tolist(),
            "cylinder_surface_area_m2": float(geometry["face_area"][faces].sum()),
            "cell_volume_m3_min_median_max": [
                float(operation(geometry["cell_volume"][selected]))
                for operation in (np.min, np.median, np.max)
            ],
        }
    a, b = meshes
    distance, index = cKDTree(b[0]).query(a[0])
    wall_distance, wall_index = cKDTree(b[2]).query(a[2])
    result["agreement"] = {
        "maximum_near_body_cell_centre_difference_m": float(distance.max()),
        "maximum_relative_near_body_cell_volume_difference": float(
            np.max(np.abs(a[1] - b[1][index]) / b[1][index])
        ),
        "maximum_wall_face_centre_difference_m": float(wall_distance.max()),
        "maximum_wall_area_vector_difference_m2": float(
            np.linalg.norm(a[3] - b[3][wall_index], axis=1).max()
        ),
        "same_local_cell_count": len(a[0]) == len(b[0]),
        "same_wall_face_count": len(a[2]) == len(b[2]),
    }
    return result


def compare(directory, histories, start, end):
    directory.mkdir(parents=True, exist_ok=True)
    records, forces = {}, {}
    for label, path in histories.items():
        values = read_force_history(path)
        cycles = {field: complete_cycles(values, field, 2.0) for field in FIELDS}
        records[label] = {
            "source": str(path),
            "sha256": digest(path),
            "statistics": window_statistics(values, cycles, start, end, 2.0),
        }
        forces[label] = values
    reference = records["reference"]["statistics"]
    for label, record in records.items():
        if label == "reference":
            continue
        record["ratios_to_reference"] = {}
        for field in FIELDS:
            data = record["statistics"][field]
            record["ratios_to_reference"][field] = {
                name: data[name] / reference[field][name]
                if data[name] is not None and reference[field][name]
                else None
                for name in ("median_peak_to_peak_drift_corrected", "median_period")
            }
            if field == "drag_coefficient":
                record["ratios_to_reference"][field]["mean"] = (
                    data["mean"] / reference[field]["mean"]
                )
            record["ratios_to_reference"][field]["mean_difference"] = (
                data["mean"] - reference[field]["mean"]
            )
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, layout="constrained")
    for label, values in forces.items():
        selected = (values["time"] >= start) & (values["time"] <= end)
        for axis, field in zip(axes, FIELDS, strict=True):
            axis.plot(values["time"][selected], values[field][selected], label=label, lw=1.2)
    axes[0].set_ylabel("Drag coefficient")
    axes[1].set_ylabel("Lift coefficient")
    axes[1].set_xlabel("Time (s)")
    axes[0].legend(frameon=False)
    figure.savefig(directory / "force_comparison.png", dpi=180)
    plt.close(figure)
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "interval_s": [start, end],
        "time_alignment": "Original clocks; no time or force adjustment",
        "histories": records,
    }
    (directory / "force_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=[], metavar="LABEL=CASE_PATH")
    parser.add_argument("--start", type=float, default=55.0)
    parser.add_argument("--end", type=float, default=95.0)
    parser.add_argument("--mesh-only", action="store_true")
    options = parser.parse_args()
    options.directory.mkdir(parents=True, exist_ok=True)
    mesh = mesh_statistics(
        {
            "coupled": CASE / "solution/fvm/mesh.npz",
            "reference": CASE / "reference_flow/solution/fvm/mesh.npz",
        }
    )
    (options.directory / "near_body_mesh.json").write_text(json.dumps(mesh, indent=2) + "\n")
    if not options.mesh_only:
        histories = {
            "reference": CASE / "reference_flow/samples/forces_history.csv",
            "baseline": CASE / "samples/forces_history.csv",
        }
        for item in options.candidate:
            label, path = item.split("=", 1)
            histories[label] = Path(path) / "samples/forces_history.csv"
        report = compare(options.directory, histories, options.start, options.end)
        print(
            json.dumps(
                {
                    label: record.get("ratios_to_reference")
                    for label, record in report["histories"].items()
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
