#!/usr/bin/env python3
"""Reconstruct a slab-centred spanwise line from an MPI FVM checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import pickle

import numpy as np

from source.solvers.fvm.io.checkpoint import decode_state
from source.solvers.fvm.sampling.fields import LineSampler


def _global_velocity(checkpoint: Path) -> tuple[np.ndarray, int, float]:
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    global_ids = []
    local_velocity = []
    step = 0
    time = 0.0
    for name in manifest["files"]:
        with np.load(checkpoint / name, allow_pickle=False) as encoded:
            state = decode_state({key: encoded[key] for key in encoded.files})
        ids = np.asarray(state["global_cell_id"], dtype=int)
        global_ids.append(ids)
        local_velocity.append(np.asarray(state["velocity"][: ids.size], dtype=float))
        step = int(state["step"])
        time = float(state["time"])
    ids = np.concatenate(global_ids)
    values = np.concatenate(local_velocity)
    counts = np.bincount(ids, minlength=int(manifest["n_global_cells"]))
    if np.any(counts == 0):
        raise ValueError("Checkpoint does not cover every global cell")
    velocity = np.column_stack(
        [
            np.bincount(ids, weights=values[:, axis], minlength=counts.size) / counts
            for axis in range(3)
        ]
    )
    return velocity, step, time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path, help="reference case containing solution/")
    parser.add_argument("output", type=Path, help="destination spanwise_line.csv")
    args = parser.parse_args()
    case = args.case.resolve()
    metadata = json.loads(
        (case / "solution" / "benchmark_metadata.json").read_text(encoding="utf-8")
    )
    span = float(metadata["physics"]["cylinder_length"])
    spacing = float(metadata["mesh"]["spanwise_cell_size"])
    with (case / "solution" / "reference_mesh.pkl").open("rb") as stream:
        payload = pickle.load(stream)
    mesh = payload["mesh"] if "mesh" in payload else payload
    points = np.asarray(mesh["vertex_position"], dtype=float)
    cells = np.asarray(mesh["cell_vertex_indices"], dtype=int)
    centres = np.mean(points[cells], axis=1)
    velocity, step, time = _global_velocity(case / "solution" / "checkpoint")

    start_z = -0.5 * span + 0.5 * spacing
    end_z = 0.5 * span - 0.5 * spacing
    sampler = LineSampler(
        start=[1.5, 0.0, start_z],
        end=[1.5, 0.0, end_z],
        spacing=spacing,
        k=12,
        reconstruction="affine",
    )
    sampled = np.column_stack(
        [sampler._interpolate(velocity[:, axis], centres) for axis in range(3)]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "time",
                "step",
                "position_x",
                "position_y",
                "position_z",
                "velocity_x",
                "velocity_y",
                "velocity_z",
            )
        )
        writer.writerows(
            (time, step, *point, *value) for point, value in zip(sampler.points, sampled)
        )
    print(
        json.dumps(
            {
                "time": time,
                "step": step,
                "points": int(sampled.shape[0]),
                "velocity_range": np.ptp(sampled, axis=0).tolist(),
                "maximum_absolute_velocity_z": float(np.max(np.abs(sampled[:, 2]))),
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
