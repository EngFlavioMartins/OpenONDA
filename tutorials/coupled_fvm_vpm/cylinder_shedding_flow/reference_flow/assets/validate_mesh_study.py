#!/usr/bin/env python3
"""Qualify and plot every mesh in the cylinder grid-independence study."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/openonda-matplotlib-cache")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
import numpy as np  # noqa: E402

from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.validation import validate_geometry, validate_topology

CASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_DIR))

import setup as reference  # noqa: E402

GRIDS = (
    ("very_coarse", 1.0 / 12.0, 17),
    ("coarse", 1.0 / 24.0, 35),
    ("medium", 1.0 / 36.0, 53),
    ("fine", 1.0 / 54.0, 80),
)

LIMITS = {
    "max_non_orthogonality_deg": 45.0,
    "max_skewness": 0.495,
    "max_lsq_condition": 5.5,
    "max_wall_cell_in_plane_edge_ratio": 20.0,
    "out_of_bounds_interpolation_weights": 0,
    "rank_deficient_lsq_cells": 0,
    "svd_lsq_cells": 0,
    "transition_to_lattice_ratio_max": 1.0,
}


def _patch(mesh: dict, name: str) -> dict:
    return next(patch for patch in mesh["boundary"] if patch["name"] == name)


def _face_edge_ratio(points: np.ndarray, face) -> float:
    polygon = points[np.asarray(face, dtype=np.int64), :2]
    edge_lengths = np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1)
    positive = edge_lengths[edge_lengths > 1.0e-14]
    return float(np.max(positive) / np.min(positive))


def _planar_metrics_and_segments(mesh: dict) -> tuple[float, list[np.ndarray]]:
    points = np.asarray(mesh["vertex_position"])
    owners = np.asarray(mesh["owners"])
    layer_index = np.asarray(mesh["boundary_layer_index"])
    zmin = _patch(mesh, "zmin")
    start = int(zmin["start_face"])
    stop = start + int(zmin["n_faces"])
    wall_ratios = []
    segments = []
    for face_index in range(start, stop):
        face = mesh["faces"][face_index]
        polygon = points[np.asarray(face, dtype=np.int64), :2]
        centre = np.mean(polygon, axis=0)
        if np.max(np.abs(centre)) <= 1.48:
            segments.extend(
                np.stack((polygon, np.roll(polygon, -1, axis=0)), axis=1)
            )
        if layer_index[owners[face_index]] == 0:
            wall_ratios.append(_face_edge_ratio(points, face))
    return float(np.max(wall_ratios)), segments


def _quality_record(
    name: str, dx: float, expected_transition_layers: int, mesh: dict
) -> tuple[dict, list[np.ndarray]]:
    validate_topology(mesh)
    geometry = compute_mesh_geometry(mesh, gradient_scheme="lsq", compute_lsq=True)
    quality = validate_geometry(mesh, geometry)
    generation = mesh["mesh_generation"]
    boundary_layer = generation["boundary_layer"]
    wall = _patch(mesh, "cylinder")
    wall_faces = np.arange(
        int(wall["start_face"]),
        int(wall["start_face"]) + int(wall["n_faces"]),
    )
    wall_owners = np.asarray(mesh["owners"])[wall_faces]
    wall_centre_distance = np.linalg.norm(
        np.asarray(geometry["cell_centre"])[wall_owners]
        - np.asarray(geometry["face_centre"])[wall_faces],
        axis=1,
    )
    wall_edge_ratio, segments = _planar_metrics_and_segments(mesh)

    record = {
        "case": name,
        "dx": dx,
        "cells": int(mesh["n_cells"]),
        "faces": int(mesh["n_faces"]),
        "wall_faces": int(wall["n_faces"]),
        "wall_layers": int(boundary_layer["wall_layers"]),
        "transition_layers": int(boundary_layer["transition_layers"]),
        "first_cell_height": float(boundary_layer["first_cell_height"]),
        "wall_centre_distance_min": float(np.min(wall_centre_distance)),
        "wall_centre_distance_max": float(np.max(wall_centre_distance)),
        "transition_to_lattice_ratio_max": float(
            boundary_layer["transition_to_lattice_ratio_max"]
        ),
        "max_wall_cell_in_plane_edge_ratio": wall_edge_ratio,
        "spanwise_cells": int(generation["spanwise_cells"]),
        "spanwise_cell_size": float(generation["spanwise_cell_size"]),
        **quality,
    }

    exact_checks = {
        "wall_layers": record["wall_layers"] == 10,
        "transition_layers": record["transition_layers"] == expected_transition_layers,
        "first_cell_height": bool(
            np.isclose(
                record["first_cell_height"], dx / 16.0, rtol=0.0, atol=1.0e-14
            )
        ),
        "wall_centre_distance": bool(
            np.allclose(
                wall_centre_distance,
                0.5 * record["first_cell_height"],
                rtol=3.0e-3,
                atol=1.0e-12,
            )
        ),
    }
    limit_checks = {
        metric: bool(
            record[metric] <= limit + 1.0e-12
            if limit > 0.0
            else record[metric] == limit
        )
        for metric, limit in LIMITS.items()
    }
    record["checks"] = {**exact_checks, **limit_checks}
    record["passed"] = bool(all(record["checks"].values()))
    return record, segments


def main() -> None:
    figure, axes = plt.subplots(2, 4, figsize=(12.0, 6.2), constrained_layout=True)
    records = []
    for column, (name, dx, transition_layers) in enumerate(GRIDS):
        print(f"Qualifying {name} (dx={dx:.8f})", flush=True)
        mesh = reference.grid_mesh(dx).build()
        record, segments = _quality_record(name, dx, transition_layers, mesh)
        records.append(record)
        for row, extent in enumerate((1.45, 0.62)):
            axis = axes[row, column]
            axis.add_collection(
                LineCollection(segments, colors="#35546f", linewidths=0.22, rasterized=True)
            )
            cylinder = plt.Circle((0.0, 0.0), 0.5, color="#d9e2e8", ec="#172b3a", lw=0.8)
            axis.add_patch(cylinder)
            axis.set_xlim(-extent, extent)
            axis.set_ylim(-extent, extent)
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(
                    f"{name.replace('_', ' ')}\n"
                    f"$\\Delta x/D={dx:.4f}$, {record['cells']:,} cells",
                    fontsize=9,
                )
            if column == 0:
                axis.set_ylabel("interface" if row == 0 else "wall detail", fontsize=9)

    report = {
        "status": "passed" if all(record["passed"] for record in records) else "failed",
        "limits": LIMITS,
        "aspect_ratio_note": (
            "The 3-D max_aspect_ratio is dominated by the fixed 0.5D spanwise "
            "extrusion. max_wall_cell_in_plane_edge_ratio measures the resolved "
            "two-dimensional cylinder plane used by this reference study."
        ),
        "cases": records,
    }
    solution = CASE_DIR / "solution"
    figures = CASE_DIR / "figures"
    solution.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    (solution / "mesh_quality_study.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    for suffix in ("png", "pdf"):
        figure.savefig(figures / f"mesh_quality_study.{suffix}", dpi=240)
    plt.close(figure)

    if report["status"] != "passed":
        failed = {
            record["case"]: [name for name, passed in record["checks"].items() if not passed]
            for record in records
            if not record["passed"]
        }
        raise SystemExit(f"Mesh study failed qualification: {failed}")
    print("All four meshes passed qualification", flush=True)


if __name__ == "__main__":
    main()
