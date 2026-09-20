"""Audit span-probe interpolation using a saved native mesh and FVM checkpoint.

Read-only with respect to simulation files. A checkpoint directory may rotate;
use --expected-checkpoint-sha256 to require the exact original evidence state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from source.solvers.fvm.io.backup import decode_state
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.sampling.fields import _PointProbe


def checkpoint_digest(path):
    """Match the coupled backup's directory-artifact SHA-256 convention."""
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        with child.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def audit(run, checkpoint, sample_time, expected_hash=None):
    """Compare saved, reconstructed and raw-cell span diagnostics at one time.

    run and checkpoint are pathlib.Path directories; sample_time is in s.
    expected_hash optionally requires the coupled checkpoint-directory SHA-256.
    The native mesh, every rank field and nine saved probe points must match
    that time. Inconsistent, incomplete or nonextruded input raises ValueError.

    Return a JSON-serializable evidence dictionary containing source hashes,
    raw velocities (m/s), probe ranges (m/s) and exact-planar reconstruction
    controls. The planar control exists only in temporary host arrays; no
    simulation file or physical state is changed. Memory scales with global
    cell count, and nearest-neighbour interpolation uses native cell centres.
    """
    actual_hash = checkpoint_digest(checkpoint)
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(f"Checkpoint hash mismatch: expected {expected_hash}, found {actual_hash}")
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    mesh_path = run / "solution/fvm/mesh.npz"
    mesh = load_native_mesh(mesh_path)
    centres = compute_mesh_geometry(mesh, compute_lsq=False)["cell_centre"][: mesh["n_cells"]]
    velocity = np.full_like(centres, np.nan)
    duplicate_max = 0.0
    for name in manifest["files"]:
        with np.load(checkpoint / name, allow_pickle=False) as archive:
            data = decode_state(dict(archive))
        if not np.isclose(float(data["time"]), sample_time, rtol=0, atol=1e-8):
            raise ValueError(f"Rank checkpoint {name} is not at requested time {sample_time}")
        ids = data["global_cell_id"]
        values = data["velocity"][: len(ids)]
        duplicate = np.isfinite(velocity[ids, 0])
        if duplicate.any():
            duplicate_max = max(
                duplicate_max,
                float(np.max(np.linalg.norm(velocity[ids[duplicate]] - values[duplicate], axis=1))),
            )
        velocity[ids] = values
    if not np.isfinite(velocity).all() or duplicate_max > 1e-12:
        raise ValueError("Checkpoint does not supply a consistent finite global cell field")
    xy, stack = np.unique(np.round(centres[:, :2], 9), axis=0, return_inverse=True)
    counts = np.bincount(stack)
    if np.any(counts != counts[0]) or counts[0] < 2:
        raise ValueError("Audit requires an extruded mesh with uniform span-layer counts")
    means = np.stack(
        [np.bincount(stack, weights=velocity[:, i]) / counts for i in range(3)], axis=1
    )
    planar = means[stack]
    table = pd.read_csv(run / "samples/span_probe.csv")
    frame = table[np.isclose(table.time, sample_time, rtol=0, atol=1e-8)].sort_values("position_z")
    if len(frame) != 9:
        raise ValueError("Expected exactly nine saved span samples at the requested time")
    points = frame[["position_x", "position_y", "position_z"]].to_numpy()
    saved = frame[["velocity_x", "velocity_y", "velocity_z"]].to_numpy()
    affine = _PointProbe(points, k=12, reconstruction="affine")
    reproduced = affine._interpolate(velocity, centres)
    planar_sample = affine._interpolate(planar, centres)
    nearest = _PointProbe(points, k=1, reconstruction="idw")
    indices, _ = nearest._interpolation_stencil(centres)
    selected_ids = indices[:, 0]
    selected_stacks = np.unique(stack[selected_ids])
    if len(selected_stacks) != 1 or len(np.unique(centres[selected_ids, 2])) != counts[0]:
        raise ValueError("Nearest probe does not observe every layer of a single XY stack")
    selected = stack == selected_stacks[0]
    nearest_sample = nearest._interpolate(velocity, centres)
    return {
        "case": str(run.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_artifact_sha256": actual_hash,
        "native_mesh": str(mesh_path.resolve()),
        "time": sample_time,
        "n_cells": len(centres),
        "n_xy_stacks": len(xy),
        "span_layers": int(counts[0]),
        "duplicate_ghost_max_difference": duplicate_max,
        "full_stack_max_vector_deviation": float(np.linalg.norm(velocity - planar, axis=1).max()),
        "full_max_w": float(abs(velocity[:, 2]).max()),
        "saved_probe_range": np.ptp(saved, axis=0).tolist(),
        "offline_affine_reproduction_max_error": float(abs(reproduced - saved).max()),
        "affine_probe_range_on_exactly_planar_field": np.ptp(planar_sample, axis=0).tolist(),
        "nearest_probe_range": np.ptp(nearest_sample, axis=0).tolist(),
        "nearest_probe_selected_centres": centres[selected_ids].tolist(),
        "raw_selected_stack_centres": centres[selected].tolist(),
        "raw_selected_stack_velocity": velocity[selected].tolist(),
        "raw_selected_stack_range": np.ptp(velocity[selected], axis=0).tolist(),
        "method": "Native geometry plus decoded checkpoint; exact-planar control replaces each XY stack by its span average only offline. Original simulation fields and samples are not changed.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="FVM rank checkpoint directory, absolute or relative to run",
    )
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument(
        "--output", type=Path, help="Write a new JSON; an existing file is never overwritten"
    )
    args = parser.parse_args()
    if not np.isfinite(args.time):
        parser.error("--time must be finite")
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else args.run / args.checkpoint
    result = audit(args.run, checkpoint, args.time, args.expected_checkpoint_sha256)
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
