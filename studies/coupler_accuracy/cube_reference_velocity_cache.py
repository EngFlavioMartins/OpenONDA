"""Cache native 3D fine-reference velocity traces for a bounded curl-transfer trial."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cube_lattice_phase_study import reference_gradient
from cube_wake_drift_audit import frame, ordered_fields
from cube_wake_particle_probe import load_case
from numba import set_num_threads
import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits


def run(args):
    load_case(args.source_tree)
    from source.coupler.interpolation import FVMVelocityInterpolator
    from source.solvers.fvm.io.mesh_storage import load_native_mesh
    from source.solvers.fvm.mesh.geometry import compute_mesh_geometry

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.lattice_inputs) as data:
        shape = data["shape"].astype(int) + 2
        spacing = float(data["spacing"])
        origin = data["position"].min(axis=0) - spacing
    axes = [origin[axis] + np.arange(shape[axis]) * spacing for axis in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    mesh = load_native_mesh(args.reference / "mesh.npz")
    geometry = compute_mesh_geometry(mesh, compute_lsq=False)
    trace = FVMVelocityInterpolator(
        geometry["cell_centre"], cKDTree(geometry["cell_centre"]), neighbour_count=4
    )
    values = {}
    for physical_time in (6, 7):
        state = ordered_fields(frame(args.reference, "fine", physical_time), mesh["n_cells"])
        gradient = reference_gradient(state["velocity"], mesh, geometry)
        values[f"velocity_t{physical_time}"] = trace.sample(points, state["velocity"], gradient)
        print(f"Cached {len(points)} 3D native-reference traces at t={physical_time}.", flush=True)
    np.savez_compressed(
        args.output, position=points, origin=origin, shape=shape, spacing=spacing, **values
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "reference": str(args.reference.resolve()),
                "times": [6, 7],
                "reconstruction": "Native Gauss gradients and four-donor Taylor traces, including cutoff ties; no body mask here.",
                "points": len(points),
                "spacing": spacing,
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--lattice-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_num_threads(2)
    with threadpool_limits(limits=2):
        run(args)
