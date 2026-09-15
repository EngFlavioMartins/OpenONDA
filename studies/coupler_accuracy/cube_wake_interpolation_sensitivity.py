"""Check whether the measured cube improvement depends on reference sampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from cube_wake_drift_audit import (
    comparison_geometry,
    frame,
    ordered_fields,
    regions,
    vector_metrics,
)
from threadpoolctl import threadpool_limits


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source_tree.resolve()))
    from source.solvers.fvm.sampling.fields import _PointProbe

    directories = {
        "baseline": args.baseline / "solution/coupled",
        "candidate": args.trial / "solution",
        "reference": args.baseline / "solution/reference_fine",
    }
    geometry = {
        name: comparison_geometry(path / "mesh.npz", args.output.parent / "geometry-cache")
        for name, path in directories.items()
        if name != "candidate"
    }
    geometry["candidate"] = geometry["baseline"]
    if (directories["baseline"] / "mesh.npz").read_bytes() != (
        directories["candidate"] / "mesh.npz"
    ).read_bytes():
        raise ValueError("Coupled meshes differ")
    fields = {
        name: ordered_fields(
            frame(path, "fine" if name == "reference" else "coupled_replacement_flow", args.time),
            len(geometry[name]["cell_centre"]),
        )["velocity"]
        for name, path in directories.items()
    }
    points = geometry["baseline"]["cell_centre"]
    volumes = geometry["baseline"]["cell_volume"]
    masks = regions(points)
    report = {
        "description": __doc__,
        "time": args.time,
        "method": "Native 3D cell-volume-weighted vector RMS / U_infinity; affine reference reconstruction with varied donor counts",
        "source_tree": str(args.source_tree.resolve()),
        "baseline": str(args.baseline.resolve()),
        "trial": str(args.trial.resolve()),
        "neighbour_counts": [],
    }
    for count in args.neighbours:
        sampled = _PointProbe(points, k=count, reconstruction="affine")._interpolate(
            fields["reference"], geometry["reference"]["cell_centre"]
        )
        row = {"neighbours": count, "errors": {}}
        for name in ("baseline", "candidate"):
            row["errors"][name] = {
                region: vector_metrics(fields[name][mask], sampled[mask], volumes[mask])
                for region, mask in masks.items()
            }
        report["neighbour_counts"].append(row)
        (args.output / "sensitivity.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "neighbours": count,
                    "whole": {name: data["whole"]["rms"] for name, data in row["errors"].items()},
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
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--neighbours", type=int, nargs="+", default=[8, 12, 20])
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args)


if __name__ == "__main__":
    main()
