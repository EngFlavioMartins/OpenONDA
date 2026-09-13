#!/usr/bin/env python3
"""Separate a continuous velocity correction from its Biot--Savart projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from source.solvers.fvm.io.mesh_storage import load_native_mesh
from studies.coupler_accuracy.continuous_velocity_curl_3d import ContinuousVelocityCurl
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, field_rms, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays
from studies.coupler_accuracy.native_volume_induction_3d import NativeVolumeSources


def parent_values(subdivision, velocity, points, parents):
    """Locate in known native cells using independent 4-by-4 barycentrics."""
    chosen, weights = [], []
    for point, parent in zip(points, parents, strict=True):
        candidates = np.flatnonzero(subdivision.parent == parent)
        vertex = subdivision.position[subdivision.tetrahedra[candidates]]
        matrix = np.concatenate((vertex, np.ones((*vertex.shape[:2], 1))), axis=2).transpose(0, 2, 1)
        right = np.broadcast_to(np.r_[point, 1.], (len(candidates), 4))
        barycentric = np.linalg.solve(matrix, right[..., None])[..., 0]
        selected = int(np.argmax(barycentric.min(axis=1)))
        if np.min(barycentric[selected]) < -1e-9:
            raise ValueError("A target does not belong to its mapped native cell")
        chosen.append(candidates[selected])
        weights.append(barycentric[selected])
    chosen, weights = np.asarray(chosen), np.asarray(weights)
    nodal = velocity[:, subdivision.tetrahedra[chosen]]
    return np.einsum("ni,snij->snj", weights, nodal), chosen, weights


def independent_tetrahedron_divergence(subdivision, velocity):
    """Integrate normal velocity over four oriented faces of every tetrahedron."""
    vertices = subdivision.position[subdivision.tetrahedra]
    result = np.zeros((len(velocity), len(vertices)))
    for ids in ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)):
        face = vertices[:, ids]
        vector_area = np.cross(face[:, 1]-face[:, 0], face[:, 2]-face[:, 0])/2
        mean_velocity = velocity[:, subdivision.tetrahedra[:, ids]].mean(axis=2)
        result += np.einsum("ti,sti->st", vector_area, mean_velocity)
    return result/subdivision.volume[None]


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source_report_path = args.source / "cube-continuous-curl-sources-3d.json"
    source_path = args.source / "continuous-curl-source-fields.npz"
    baseline_report_path = args.baseline / "cube-shared-trace-induction-3d.json"
    baseline_path = args.baseline / "shared-trace-physical-fields.npz"
    source_report, baseline_report = [json.loads(path.read_text()) for path in (source_report_path, baseline_report_path)]
    for report in (source_report, baseline_report):
        assert report["status"] == "complete" and report["spatial_dimensions"] == 3
        for row in report["sources"]:
            assert hash_file(ROOT / row["path"]) == row
    parent = ROOT / source_report["parent_source"]
    donor_path = parent / "shared-trace-source-fields.npz"
    mesh_path = ROOT / next(row["path"] for row in source_report["sources"] if row["path"].endswith("/small-native-mesh.npz"))
    paths = [Path(__file__).resolve(), source_report_path, source_path, baseline_report_path, baseline_path, donor_path, mesh_path]
    paths += [ROOT / name for name in (
        "studies/coupler_accuracy/continuous_velocity_curl_3d.py", "studies/coupler_accuracy/native_volume_induction_3d.py",
        "studies/coupler_accuracy/native_linear_volume_induction_3d.py", "studies/coupler_accuracy/cube_boundary_oracle.py",
        "studies/coupler_accuracy/cube_native_moment_reconstruction_3d.py", "source/solvers/fvm/io/mesh_storage.py")]
    completed, completed_report = None, None
    if args.completed_induction:
        completed_path = args.completed_induction / "continuous-curl-physical-fields.npz"
        completed_report_path = args.completed_induction / "cube-continuous-curl-induction-3d.json"
        checkpoint_path = args.completed_induction / "continuous-curl-induction-checkpoint.npz"
        completed = read_arrays(completed_path)
        completed_report = json.loads(completed_report_path.read_text())
        assert completed_report["status"] == "complete"
        paths += [completed_path, completed_report_path, checkpoint_path]
    sources = [hash_file(path) for path in paths]
    for path in paths:
        if path.suffix == ".py":
            target = args.output / "sources" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    (args.output / "sources-at-start.json").write_text(json.dumps(sources, indent=2)+"\n")
    source, base, donor = [read_arrays(path) for path in (source_path, baseline_path, donor_path)]
    mesh = load_native_mesh(mesh_path)
    c = ContinuousVelocityCurl.from_mesh(mesh, source["cell_centroid"])
    np.testing.assert_array_equal(c.position, source["node_position"])
    velocity, omega = source["nodal_velocity"], source["tetrahedron_vorticity"]
    theta = np.trace(c.gradient(velocity), axis1=-2, axis2=-1)
    independent_theta = independent_tetrahedron_divergence(c, velocity)
    np.testing.assert_allclose(theta, independent_theta, rtol=0, atol=2e-12)
    np.testing.assert_allclose(theta @ c.volume, 0, rtol=0, atol=2e-14)
    ids = base["near_body__cell_ids"]
    full_to_small = np.full(len(donor["full_centres"]), -1)
    full_to_small[donor["inside_cell_ids"]] = np.arange(mesh["n_cells"])
    near_parents = full_to_small[ids]
    assert np.all(near_parents >= 0)
    near = base["position"][base["target__near_body"]]
    np.testing.assert_array_equal(near, donor["full_centres"][ids])
    rng = np.random.default_rng(20260913)
    selected = np.concatenate([np.sort(rng.choice(np.flatnonzero(base["face_group"] == i), 4, replace=False)) for i in range(6)])
    exterior = base["position"][base["target__boundary"]][selected]
    points = np.vstack((near, exterior))
    field, tetrahedron, barycentric = parent_values(c, velocity, near, near_parents)
    preimage = np.zeros((2, len(points), 3))
    preimage[:, :len(near)] = field
    active = np.any(omega != 0, axis=(0, 2)) | np.any(theta != 0, axis=0)
    support = c.position[c.tetrahedra[active]]
    bounds = np.stack((support.min(axis=(0, 1)), support.max(axis=(0, 1))))
    distance = np.linalg.norm(np.maximum(bounds[0]-exterior, 0)+np.maximum(exterior-bounds[1], 0), axis=1)
    assert distance.min() > .2
    combined_omega = np.concatenate((omega, np.zeros_like(omega)))
    combined_theta = np.concatenate((np.zeros_like(theta), theta))
    coefficient = c.source.coefficients(combined_omega, combined_theta)
    used = np.any(coefficient != 0, axis=(1, 2))
    native = c.source
    compact = NativeVolumeSources(*(getattr(native, key)[used] for key in
        ("triangles", "normals", "tangents", "outward", "lengths", "owners", "neighbours", "face_ids")), native.n_cells)

    def progress(done, total):
        if done % 64 == 0 or done == total:
            print(json.dumps({"stage": "hodge_induction", "done": done, "total": total, "elapsed_seconds": time.perf_counter()-started}), flush=True)

    induced = compact.evaluate(points, coefficient[used], progress=progress).transpose(1, 0, 2)
    solenoidal, dilatation = induced[:2], induced[2:]
    # With a zero boundary trace, v_h = B(curl v_h) + integral div(v_h) r/(4 pi r^3).
    identity_error = solenoidal+dilatation-preimage
    np.testing.assert_allclose(solenoidal+dilatation, preimage, rtol=0, atol=1e-11)
    near_weight = donor["full_volume"][ids]
    records = []
    saved = {}
    for state, kind in enumerate(("point", "cell_average")):
        baseline_name = str(base["source_names"][source["baseline_state_indices"][state]])
        error = base[baseline_name+"__velocity"][base["target__near_body"]]-donor["full_velocity"][ids]
        projection_change = -dilatation[state, :len(near)]
        raw = solenoidal[state, :len(near)]
        expected = preimage[state, :len(near)]
        row = {"input": kind, "baseline": baseline_name,
               "velocity_divergence_rms": field_rms(theta[state, :, None], c.volume),
               "velocity_divergence_maximum": float(np.max(np.abs(theta[state]))),
               "independent_divergence_maximum_difference": float(np.max(np.abs(theta[state]-independent_theta[state]))),
               "helmholtz_identity_maximum_difference": float(np.max(np.abs(identity_error[state]))),
               "near_preimage_velocity_rms": field_rms(expected, near_weight),
               "near_projection_change_rms": field_rms(projection_change, near_weight),
               "near_projected_source_velocity_rms": field_rms(raw, near_weight),
               "near_baseline_error_rms": field_rms(error, near_weight),
               "near_error_with_unprojected_change_rms": field_rms(error+expected, near_weight),
               "near_error_with_projected_source_change_rms": field_rms(error+raw, near_weight),
               "exterior_preimage_maximum": float(np.max(np.abs(preimage[state, len(near):]))),
               "exterior_projected_source_velocity_rms": field_rms(solenoidal[state, len(near):], None)}
        stages = {"preimage": expected, "hodge_projection": projection_change}
        if completed is not None:
            current_name = completed_report["records"][2*state+1]["name"]
            body = completed[current_name+"__body_velocity_change"][base["target__near_body"]]
            actual = completed[current_name+"__velocity"][base["target__near_body"]]
            reconstructed = base[baseline_name+"__velocity"][base["target__near_body"]]+raw+body
            np.testing.assert_allclose(reconstructed, actual, rtol=0, atol=2e-13)
            row["completed_physical_replay_maximum_difference"] = float(np.max(np.abs(reconstructed-actual)))
            row["near_body_potential_change_rms"] = field_rms(body, near_weight)
            row["near_completed_error_rms"] = field_rms(error+raw+body, near_weight)
            np.testing.assert_allclose(row["near_completed_error_rms"], completed_report["records"][2*state+1]["cell_velocity_rms_over_Uinf"]["near_body"], rtol=0, atol=1e-14)
            stages["body_potential"] = body
            saved[kind+"__body_potential"] = body
        row["baseline_error_inner_products"] = {name: float(np.average(np.sum(error*value, axis=1), weights=near_weight))
                                                 for name, value in stages.items()}
        records.append(row)
        saved[kind+"__baseline_error"] = error
        print(json.dumps(row, indent=2), flush=True)
    np.savez_compressed(args.output / "continuous-curl-hodge-fields.npz", position=points, near_cell_ids=ids,
                        near_parent_ids=near_parents, near_tetrahedron_ids=tetrahedron, near_barycentric=barycentric,
                        near_weights=near_weight, exterior_face_rows=selected, preimage_velocity=preimage,
                        solenoidal_velocity=solenoidal, dilatation_velocity=dilatation,
                        velocity_divergence=theta, independent_velocity_divergence=independent_theta,
                        helmholtz_identity_error=identity_error, **saved)
    result = {"schema": "openonda-continuous-curl-hodge-3d/1", "status": "complete", "spatial_dimensions": 3,
              "small_fvm_cells": mesh["n_cells"], "source_sgs": source_report["source_sgs"], "physical_time": source_report["physical_time"],
              "near_targets": len(near), "exterior_targets": len(exterior), "nonzero_induction_faces": len(compact.triangles),
              "records": records, "sources": sources, "elapsed_seconds": time.perf_counter()-started,
              "limitations": [
                  "The decomposed field is the reconstruction correction, not the whole FVM or VPM solution.",
                  "Its nonzero divergence describes the reconstructed velocity change, not the FVM solver's pressure-corrected face-flux divergence.",
                  "Adding the unprojected velocity change is a diagnostic counterfactual; it is not an admissible incompressible transfer proposal.",
                  "Stage errors and error inner products are local observations. They are not globally orthogonal error budgets or force predictions.",
                  "The medium pure-source identity can be checked before the full medium body response completes; no missing body response is inferred."]}
    (args.output / "continuous-curl-hodge-3d.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--completed-induction", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.source, args.baseline, args.output = args.source.resolve(), args.baseline.resolve(), args.output.resolve()
    if args.completed_induction:
        args.completed_induction = args.completed_induction.resolve()
    run(args)
