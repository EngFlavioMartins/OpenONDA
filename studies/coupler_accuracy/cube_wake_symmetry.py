"""Measure fourfold symmetry loss in the saved three-dimensional cube fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from cube_wake_drift_audit import comparison_geometry, frame, ordered_fields
from cube_wake_particle_probe import rms
import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits


def rotation_probe(spacing):
    """Return a closed Cartesian query grid and exact rotation permutations.

    Sample each point once, then permute the observations. Separate sampling
    of roundoff-separated rotated copies can choose different interpolation
    donors and spoil orthogonality of the numerical C4 projection.
    """
    if spacing <= 0:
        raise ValueError("Probe spacing must be positive")
    half_count = round(1.4 / spacing)
    if not np.isclose(half_count * spacing, 1.4, rtol=0, atol=1e-12):
        raise ValueError("Probe spacing must divide the half-width of 1.4")
    axis = np.arange(-half_count, half_count + 1)
    indices = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    indices = indices[np.any(np.abs(indices * spacing) > 0.5 + 1e-10, axis=1)]
    rotations = [
        np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) for c, s in ((1, 0), (0, 1), (-1, 0), (0, -1))
    ]
    tree = cKDTree(indices)
    permutations = []
    for rotation in rotations:
        distance, rows = tree.query(indices @ rotation.T)
        if np.any(distance) or len(np.unique(rows)) != len(indices):
            raise ValueError("The probe set must close exactly under rotations")
        permutations.append(rows)
    return indices * spacing, rotations, permutations


def project_rotations(values, rotations, permutations):
    return np.mean(
        [values[rows] @ rotation for rotation, rows in zip(rotations, permutations, strict=True)],
        axis=0,
    )


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source_tree.resolve()))
    from source.solvers.fvm.sampling.fields import _PointProbe

    points, rotations, permutations = rotation_probe(args.spacing)
    centres, probes = {}, {}
    for name in ("coupled", "reference_fine"):
        centres[name] = comparison_geometry(
            args.baseline / "solution" / name / "mesh.npz", args.output.parent / "geometry-cache"
        )["cell_centre"]
        probes[name] = _PointProbe(points, k=12, reconstruction="affine")
    report = {
        "description": __doc__,
        "spacing": args.spacing,
        "bounds": [-1.4, 1.4] * 3,
        "fluid_query_count": len(points),
        "projection": "One interpolation per point followed by exact C4 orbit permutations",
        "times": [],
    }
    for time in args.times:
        values, symmetry = {}, {}
        row = {"time": time}
        for name, stem in (("coupled", "coupled_replacement_flow"), ("reference_fine", "fine")):
            velocity = ordered_fields(
                frame(args.baseline / "solution" / name, stem, time), len(centres[name])
            )["velocity"]
            values[name] = probes[name]._interpolate(velocity, centres[name])
            symmetry[name] = project_rotations(values[name], rotations, permutations)
            row[name] = {
                "asymmetric_velocity_rms": rms(values[name] - symmetry[name]),
                "sampled_velocity_rms": rms(values[name]),
            }
        difference = values["coupled"] - values["reference_fine"]
        symmetric_difference = symmetry["coupled"] - symmetry["reference_fine"]
        asymmetric_difference = difference - symmetric_difference
        row.update(
            {
                "total_error_rms": rms(difference),
                "symmetric_error_rms": rms(symmetric_difference),
                "asymmetric_error_rms": rms(asymmetric_difference),
                "asymmetric_squared_error_fraction": rms(asymmetric_difference) ** 2
                / rms(difference) ** 2,
                "orthogonality_residual": float(
                    np.mean(np.sum(symmetric_difference * asymmetric_difference, axis=1))
                ),
            }
        )
        report["times"].append(row)
        (args.output / "symmetry.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--times", type=float, nargs="+", default=[1, 3, 5, 6, 8, 10, 12, 15])
    parser.add_argument("--spacing", type=float, default=0.1)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args)


if __name__ == "__main__":
    main()
