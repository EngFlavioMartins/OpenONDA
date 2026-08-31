#!/usr/bin/env python3
"""Create a two-dimensional restart by averaging an extruded 3-D checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import tempfile
from uuid import uuid4

import numpy as np

from source.solvers.fvm.io.checkpoint import decode_state, encode_state


CELL_FIELDS = ("velocity", "velocity_old", "velocity_older", "kinematic_pressure")
FACE_FIELDS = (
    "volumetric_face_flux",
    "volumetric_face_flux_old",
    "volumetric_face_flux_older",
)


def _average_by_group(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Replace every row by the arithmetic mean of its integer group."""
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int64)
    count = np.bincount(groups).astype(np.float64)
    if np.any(count == 0.0):
        raise ValueError("Spanwise grouping contains an empty group")
    if values.ndim == 1:
        sums = np.bincount(groups, weights=values)
        return sums[groups] / count[groups]
    result = np.empty_like(values)
    for component in range(values.shape[1]):
        sums = np.bincount(groups, weights=values[:, component])
        result[:, component] = sums[groups] / count[groups]
    return result


def _cell_groups(mesh: dict) -> np.ndarray:
    points = np.asarray(mesh["vertex_position"], dtype=np.float64)
    vertices = np.asarray(mesh["cell_vertex_indices"], dtype=np.int64)
    centres_xy = points[vertices, :2].mean(axis=1)
    _, groups, counts = np.unique(
        np.round(centres_xy, decimals=12), axis=0, return_inverse=True, return_counts=True
    )
    if np.unique(counts).size != 1 or counts[0] < 2:
        raise ValueError(f"Mesh is not a uniformly extruded cell stack; counts={np.unique(counts)}")
    return groups


def _face_groups(mesh: dict) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(mesh["vertex_position"], dtype=np.float64)
    faces = np.asarray(mesh["faces"], dtype=np.int64)
    vertices = points[faces]
    z_range = np.ptp(vertices[:, :, 2], axis=1)
    horizontal = z_range < 1.0e-12

    # A vertical face is uniquely identified by its sorted x-y vertex set. This
    # remains exact at refinement transitions where a centre/direction key can
    # alias neighbouring partial faces.
    vertical = ~horizontal
    xy = np.round(vertices[vertical, :, :2], decimals=12)
    order = np.lexsort((xy[:, :, 1], xy[:, :, 0]), axis=1)
    sorted_xy = np.take_along_axis(xy, order[:, :, None], axis=1)
    keys = sorted_xy.reshape(len(sorted_xy), -1)
    _, local_groups, counts = np.unique(
        keys,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    if np.unique(counts).size != 1 or counts[0] < 2:
        raise ValueError(f"Mesh is not a uniformly extruded face stack; counts={np.unique(counts)}")
    groups = np.full(len(faces), -1, dtype=np.int64)
    groups[vertical] = local_groups
    return groups, horizontal


def _global_field(states: list[dict], ids_name: str, field_name: str, size: int) -> np.ndarray:
    sample = states[0][field_name]
    shape = (size,) + sample.shape[1:]
    total = np.zeros(shape, dtype=np.float64)
    count = np.zeros(size, dtype=np.int64)
    for state in states:
        ids = np.asarray(state[ids_name], dtype=np.int64)
        values = np.asarray(state[field_name], dtype=np.float64)[: len(ids)]
        np.add.at(total, ids, values)
        np.add.at(count, ids, 1)
    if np.any(count == 0):
        raise ValueError(f"Checkpoint does not cover every global entry for {field_name}")
    divisor = count.reshape((size,) + (1,) * (total.ndim - 1))
    return total / divisor


def _atomic_npz(path: Path, arrays: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def spanwise_average_checkpoint(
    mesh_cache: Path, source: Path, destination: Path, *, reset_clock: bool = False
) -> Path:
    """Average cells and conservative face-flux histories into a new checkpoint."""
    payload = pickle.loads(mesh_cache.read_bytes())
    mesh = payload["mesh"] if "mesh" in payload else payload
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    states = []
    for name in manifest["files"]:
        with np.load(source / name, allow_pickle=False) as archive:
            states.append(
                decode_state({key: np.array(archive[key], copy=True) for key in archive.files})
            )

    cell_groups = _cell_groups(mesh)
    face_groups, horizontal_faces = _face_groups(mesh)
    n_cells = int(mesh["n_cells"])
    n_faces = int(mesh["n_faces"])

    averaged_cells = {}
    for name in CELL_FIELDS:
        values = _global_field(states, "global_cell_id", name, n_cells)
        values = _average_by_group(values, cell_groups)
        if name.startswith("velocity"):
            values[:, 2] = 0.0
        averaged_cells[name] = values

    averaged_faces = {}
    vertical = ~horizontal_faces
    for name in FACE_FIELDS:
        values = _global_field(states, "global_face_id", name, n_faces)
        values[vertical] = _average_by_group(values[vertical], face_groups[vertical])
        values[horizontal_faces] = 0.0
        averaged_faces[name] = values

    for state in states:
        cell_ids = np.asarray(state["global_cell_id"], dtype=np.int64)
        face_ids = np.asarray(state["global_face_id"], dtype=np.int64)
        for name, values in averaged_cells.items():
            state[name][: len(cell_ids)] = values[cell_ids]
            if name.startswith("velocity"):
                state[name][len(cell_ids) :, 2] = 0.0
        for name, values in averaged_faces.items():
            state[name][:] = values[face_ids]
        state["max_courant_number"] = np.asarray(0.0)
        if reset_clock:
            state["time"] = np.asarray(0.0)
            state["step"] = np.asarray(0, dtype=np.int64)
            # The three averaged history levels remain valid. Two committed
            # levels are enough to make that explicit without retaining an
            # unrelated production step count in the derived case.
            state["n_committed_time_steps"] = np.asarray(2, dtype=np.int64)
            state["time_since_last_write"] = np.asarray(0.0)
            state["n_consecutive_accepted_steps"] = np.zeros_like(
                state["n_consecutive_accepted_steps"]
            )

    destination.mkdir(parents=True, exist_ok=False)
    generation = uuid4().hex
    files = [f"rank-{rank:05d}-{generation}.npz" for rank in range(len(states))]
    for name, state in zip(files, states, strict=True):
        _atomic_npz(destination / name, encode_state(state))
    output_manifest = dict(manifest)
    output_manifest["generation"] = generation
    output_manifest["files"] = files
    temporary_manifest = destination / f".manifest-{generation}.tmp"
    temporary_manifest.write_text(json.dumps(output_manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, destination / "manifest.json")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh_cache", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--reset-clock",
        action="store_true",
        help="start the derived case at step zero while retaining valid field histories",
    )
    args = parser.parse_args()
    output = spanwise_average_checkpoint(
        args.mesh_cache.resolve(),
        args.source.resolve(),
        args.destination.resolve(),
        reset_clock=args.reset_clock,
    )
    print(output)


if __name__ == "__main__":
    main()
