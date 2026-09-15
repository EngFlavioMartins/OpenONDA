"""Measure the fully meshed reference's late 3D symmetry breaking.

The probe grid closes exactly under z reflection. This distinguishes a shifted
physical symmetry-breaking transient from a blanket assumption that every
out-of-plane velocity is intrinsically unphysical.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from cube_wake_drift_audit import comparison_geometry, frame, ordered_fields
from cube_wake_particle_probe import rms
import numpy as np
from threadpoolctl import threadpool_limits


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source_tree.resolve()))
    from source.solvers.fvm.sampling.fields import _PointProbe

    geometry = comparison_geometry(
        args.reference / "mesh.npz", args.output.parent / "geometry-cache"
    )
    x, yz = np.array([0.9, 1.14, 1.26, 1.38, 1.5, 1.62, 1.86, 2.34]), np.linspace(-0.54, 0.54, 7)
    points = np.stack(np.meshgrid(x, yz, yz, indexing="ij"), axis=-1).reshape(-1, 3)
    probe = _PointProbe(points, k=12, reconstruction="affine")
    masks = {"seam": (points[:, 0] >= 1.25) & (points[:, 0] <= 1.62), "wake": points[:, 0] > 1.62}
    report = {"description": __doc__, "times": []}
    for time in args.times:
        fields = ordered_fields(frame(args.reference, "fine", time), len(geometry["cell_centre"]))
        u = probe._interpolate(fields["velocity"], geometry["cell_centre"])
        grid = u.reshape(8, 7, 7, 3)
        asym = ((grid - grid[:, :, ::-1, :] * [1, 1, -1]) * 0.5).reshape(-1, 3)
        row = {
            "time": time,
            "regions": {
                name: {
                    "reflection_asymmetry_rms": rms(asym[m]),
                    "maximum_speed": float(np.max(np.linalg.norm(u[m], axis=1))),
                }
                for name, m in masks.items()
            },
        }
        report["times"].append(row)
        np.savez_compressed(
            args.output / f"fields_t{time:g}.npz", points=points, velocity=u, asymmetry=asym
        )
        print(json.dumps(row), flush=True)
        (args.output / "history.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--times", type=float, nargs="+", default=[3, 6, 7, 8, 10, 15, 20, 25, 30])
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args)


if __name__ == "__main__":
    main()
