#!/usr/bin/env python3
"""Compare full/hybrid normal-moment corrections on identical native faces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from source.solvers.fvm.io.mesh_storage import load_native_mesh
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    paths, parents, arrays = [args.mapping, Path(__file__).resolve()], {}, {}
    for name, directory in (("full", args.full), ("hybrid", args.hybrid)):
        report_path = directory / "normal-moment-lift-3d.json"
        report = json.loads(report_path.read_text())
        assert report["status"] == "complete" and report["spatial_dimensions"] == 3
        assert report["snapshot"]["name"] == name
        for row in report["sources"]:
            assert hash_file(ROOT / row["path"]) == row
            if row["path"].endswith(".py"):
                assert hash_file(directory / "sources" / row["path"])["sha256"] == row["sha256"]
        for row in report["records"]:
            path = ROOT / row["fields"]["path"]
            assert hash_file(path) == row["fields"]
            arrays[name, row["name"]] = read_arrays(path)
            paths.append(path)
        paths.append(report_path)
        parents[name] = report
    for key in ("fvm_step", "physical_time"):
        assert parents["full"]["snapshot"][key] == parents["hybrid"]["snapshot"][key]
    assert [r["name"] for r in parents["full"]["records"]] == [r["name"] for r in parents["hybrid"]["records"]]
    mesh_path = ROOT / next(r["path"] for r in parents["hybrid"]["sources"] if r["path"].endswith("trial/hybrid/solution/mesh.npz"))
    mesh = load_native_mesh(mesh_path)
    mapping = read_arrays(args.mapping)
    ids = mapping["face_ids"]
    assert len(ids) == mesh["n_faces"] and len(mapping["cell_ids"]) == mesh["n_cells"]
    paths.append(mesh_path)
    records = []
    for pair in parents["hybrid"]["records"]:
        name = pair["name"]
        full, small = arrays["full", name], arrays["hybrid", name]
        np.testing.assert_allclose(full["face_centroid"][ids], small["face_centroid"], rtol=0, atol=1e-13)
        np.testing.assert_allclose(full["face_area"][ids], small["face_area"], rtol=1e-13, atol=1e-15)
        free = np.zeros(mesh["n_faces"], dtype=bool)
        free[small["free_faces"]] = True
        inside = np.arange(mesh["n_faces"]) < mesh["n_interior_faces"]
        radius = np.max(np.abs(small["face_centroid"]), axis=1)
        regions = {"all_shared_free_faces": free, "shared_interior_faces": free & inside,
                   "near_body_free_faces": free & (radius < .8), "coupling_faces": free & ~inside}
        result = {"name": name, "regions": {}}
        for region, mask in regions.items():
            area = small["face_area"][mask]
            result["regions"][region] = {"faces": int(mask.sum()), "area": float(area.sum())}
            for state_name, state, selected in (("full", full, ids[mask]), ("hybrid", small, np.flatnonzero(mask))):
                energy = state["normal_change_energy_by_face"][selected]
                maximum = state["maximum_normal_change_by_face"][selected]
                peak = selected[int(np.argmax(maximum))]
                result["regions"][region][state_name] = {
                    "normal_change_rms": float(np.sqrt(energy.sum() / area.sum())),
                    "maximum_normal_change": float(maximum.max()),
                    "peak_face_centroid": state["face_centroid"][peak].tolist(),
                }
        records.append(result)
    sources = [hash_file(path) for path in paths]
    own = Path(__file__).resolve()
    target = args.output.parent / (args.output.stem + "-sources") / own.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(own.read_bytes())
    report = {"schema": "openonda-normal-moment-lift-comparison-3d/1", "status": "complete", "spatial_dimensions": 3,
              "physical_time": parents["hybrid"]["snapshot"]["physical_time"], "shared_cells": mesh["n_cells"],
              "shared_faces": mesh["n_faces"], "sources": sources, "records": records,
              "limitations": [
                  "The two fits are posed on different whole domains, but the reported norms use identical mapped native faces and areas.",
                  "These are fitted normal-trace changes needed by the moment constraints, not hybrid/reference flow errors or production flow velocities.",
                  "The full-domain fit also frees within-face variation on its outer physical boundary; it is a diagnostic control, not a change to the reference simulation.",
              ]}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(next(row for row in records if row["name"] == "stored_cell_velocity-3-modes"), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.full, args.hybrid, args.mapping, args.output = args.full.resolve(), args.hybrid.resolve(), args.mapping.resolve(), args.output.resolve()
    run(args)
