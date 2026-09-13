#!/usr/bin/env python3
"""Measure frozen 3D reconstructions against accepted conservative cut-face flux."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from source.solvers.fvm.io.backup import decode_state
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, field_rms, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays
from studies.coupler_accuracy.shared_trace_response_3d import response


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    replay_path = args.reference / "reference-flux-replay-3d.json"
    parent_path = args.induction / "cube-shared-trace-induction-3d.json"
    physical_path = args.induction / "shared-trace-physical-fields.npz"
    replay, parent = (json.loads(path.read_text()) for path in (replay_path, parent_path))
    paths = [Path(__file__).resolve(), replay_path, parent_path, physical_path]
    for report in (replay, parent):
        assert report["status"] == "complete" and report["spatial_dimensions"] == 3
        for row in report["sources"]:
            assert hash_file(ROOT / row["path"]) == row
    source_path = ROOT / next(row["path"] for row in parent["sources"] if row["path"].endswith("/shared-trace-source-fields.npz"))
    mesh_path = ROOT / next(row["path"] for row in parent["sources"] if row["path"].endswith("/full-native-mesh.npz"))
    source, physical = read_arrays(source_path), read_arrays(physical_path)
    mesh = load_native_mesh(mesh_path)
    geo = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    paths += [source_path, mesh_path]
    faces, signs, area = physical["full_face_ids"], physical["signs"], physical["area"]
    assert np.all(faces < mesh["n_interior_faces"])
    np.testing.assert_array_equal(geo["cell_centre"], source["full_centres"])
    sf = geo["face_area_vector"][faces]*signs[:, None]
    np.testing.assert_allclose(np.linalg.norm(sf, axis=1), area, rtol=0, atol=1e-14)
    np.testing.assert_allclose(sf/area[:, None], physical["normal"], rtol=0, atol=1e-14)
    np.testing.assert_allclose(geo["face_centre"][faces], physical["position"][physical["target__boundary"]], rtol=0, atol=1e-13)
    targets = {"linear_cell_velocity": physical["reference_normal_velocity"]}
    w = geo["face_interpolation_weight"][faces, None]
    face_u = (1-w)*source["full_velocity"][mesh["owners"][faces]]+w*source["full_velocity"][mesh["neighbours"][faces]]
    np.testing.assert_allclose(np.sum(sf*face_u, axis=1)/area, targets["linear_cell_velocity"], rtol=0, atol=2e-14)
    snapshots = []
    for name in ("warmup", "reset-initial"):
        row = next(item for item in replay["records"] if item["name"] == name)
        path = ROOT / row["backup"]["path"]
        assert hash_file(path) == row["backup"]
        np.testing.assert_allclose(row["physical_time"], parent["physical_time"], rtol=0, atol=1e-14)
        state = decode_state(read_arrays(path))
        np.testing.assert_array_equal(state["velocity"][:mesh["n_cells"]], source["full_velocity"])
        un = state["volumetric_face_flux"][faces]*signs/area
        if name == "warmup":
            targets["accepted_conservative_flux"] = un
            np.testing.assert_allclose(area @ un, 0, rtol=0, atol=2e-13)
        else:
            np.testing.assert_allclose(un, targets["linear_cell_velocity"], rtol=0, atol=2e-14)
        snapshots.append({"name": name, "physical_time": row["physical_time"], "solver_step": int(state["step"]),
                          "cut_flux": float(area @ un), "maximum_cell_velocity_difference": 0.,
                          "difference_from_linear_reference_rms": field_rms((un-targets["linear_cell_velocity"])[:, None], area)})
        paths.append(path)
    experiments = [("shared_trace", physical, parent)]
    if args.continuous:
        report_path = args.continuous / "cube-continuous-curl-induction-3d.json"
        field_path = args.continuous / "continuous-curl-physical-fields.npz"
        report, data = json.loads(report_path.read_text()), read_arrays(field_path)
        assert report["status"] == "complete" and report["spatial_dimensions"] == 3
        for row in report["sources"]:
            assert hash_file(ROOT / row["path"]) == row
        np.testing.assert_array_equal(data["position"], physical["position"])
        experiments.append(("continuous_curl", data, report))
        paths += [report_path, field_path]
    paths += [ROOT / name for name in (
        "source/solvers/fvm/io/backup.py", "source/solvers/fvm/io/mesh_storage.py", "source/solvers/fvm/mesh/geometry.py",
        "studies/coupler_accuracy/cube_boundary_oracle.py", "studies/coupler_accuracy/cube_native_moment_reconstruction_3d.py",
        "studies/coupler_accuracy/shared_trace_response_3d.py")]
    paths = list(dict.fromkeys(paths))
    sources = [hash_file(path) for path in paths]
    for path in paths:
        if path.suffix == ".py":
            archive = args.output / "sources" / path.relative_to(ROOT)
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_bytes(path.read_bytes())
    (args.output / "sources-at-start.json").write_text(json.dumps(sources, indent=2)+"\n")
    records, changes = [], []
    saved = {"area": area, "normal": physical["normal"], "full_face_ids": faces, "signs": signs,
             "face_group": physical["face_group"], **targets}
    for experiment, data, report in experiments:
        for index, name in enumerate(data["source_names"].tolist()):
            row = {"experiment": experiment, "name": name, "normal_velocity_errors": {}}
            for mode in ("point", "native"):
                un = data[name+"__"+mode+"_normal_velocity"]
                np.testing.assert_allclose(area @ un, 0, rtol=0, atol=2e-12)
                errors = {kind: field_rms((un-value)[:, None], area) for kind, value in targets.items()}
                np.testing.assert_allclose(errors["linear_cell_velocity"], report["records"][index][mode+"_normal_velocity_error_rms"], rtol=0, atol=1e-14)
                row["normal_velocity_errors"][mode] = errors
                saved[experiment+"__"+name+"__"+mode+"_normal_velocity"] = un
            records.append(row)
        pairs = [("point", 2, 4), ("cell_average", 5, 7)] if experiment == "shared_trace" else [
            ("point_affine", 2, 0), ("point_curl", 2, 1), ("cell_average_affine", 5, 2), ("cell_average_curl", 5, 3)]
        for kind, base, updated in pairs:
            base_name, updated_name = physical["source_names"][base], data["source_names"][updated]
            old = physical[base_name+"__native_normal_velocity"]
            new = data[updated_name+"__native_normal_velocity"]
            normal_response = {key: response((old-value)[:, None], (new-old)[:, None], area) for key, value in targets.items()}
            gt0, gt1 = physical[base_name+"__tangential_gradient"], data[updated_name+"__tangential_gradient"]
            derivative_response = response(gt0-physical["reference_tangential_gradient"], gt1-gt0, area)
            upper = [normal_response["accepted_conservative_flux"]["positive_scale_improvement_upper_bound"],
                     derivative_response["positive_scale_improvement_upper_bound"]]
            changes.append({"experiment": experiment, "input": kind, "baseline": str(base_name), "updated": str(updated_name),
                            "native_normal_response": normal_response, "native_derivative_response": derivative_response,
                            "joint_accepted_flux_and_derivative_improvement_upper_bound": None if any(v is None for v in upper) else min(upper)})
    difference = targets["accepted_conservative_flux"]-targets["linear_cell_velocity"]
    fields_path = args.output / "accepted-flux-boundary-fields.npz"
    np.savez_compressed(fields_path, **saved)
    result = {"schema": "openonda-accepted-flux-boundary-3d/1", "status": "complete", "spatial_dimensions": 3,
              "physical_time": parent["physical_time"], "boundary_faces": len(faces), "small_fvm_cells": parent["small_fvm_cells"],
              "snapshots": snapshots, "reference_difference_rms": field_rms(difference[:, None], area),
              "reference_difference_maximum": float(np.max(np.abs(difference))),
              "reference_difference_by_side": {name: field_rms(difference[physical["face_group"] == i, None], area[physical["face_group"] == i])
                                               for i, name in enumerate(("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"))},
              "records": records, "responses": changes, "fields": hash_file(fields_path), "sources": sources,
              "limitations": [
                  "Both reference observations use identical FVM cell velocities and native geometry, but the accepted warmup flux and reset-initial linear flux are distinct states.",
                  "The earlier linear-cell-velocity errors are replayed and retained. The conservative-flux target is an additional observation, not a replacement solver result.",
                  "Only the normal-velocity reference changes. The native tangential-derivative target is unchanged.",
                  "The response calculation diagnoses frozen source directions using the reference; it is not a production damping rule or an advancing force prediction."]}
    (args.output / "accepted-flux-boundary-3d.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: result[key] for key in ("snapshots", "reference_difference_rms", "reference_difference_by_side", "records")}, indent=2))
    print(json.dumps({"responses": [{key: row[key] for key in ("input", "joint_accepted_flux_and_derivative_improvement_upper_bound")}
                                    for row in changes]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--induction", type=Path, required=True)
    parser.add_argument("--continuous", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.reference, args.induction, args.output = (path.resolve() for path in (args.reference, args.induction, args.output))
    if args.continuous:
        args.continuous = args.continuous.resolve()
    run(args)
