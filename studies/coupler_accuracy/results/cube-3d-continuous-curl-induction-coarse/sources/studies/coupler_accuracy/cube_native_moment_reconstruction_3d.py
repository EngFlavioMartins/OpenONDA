#!/usr/bin/env python3
"""Test moment acquisition from cell circulation or native velocity data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.manufactured_cube_field_3d import NoSlipCubeField
from studies.coupler_accuracy.native_linear_volume_induction_3d import LinearNativeVolumeSources
from studies.coupler_accuracy.native_moment_reconstruction_3d import (
    CellLeastSquares,
    reconstruct_linear_face_velocity,
    weak_curl_moments,
)
from studies.coupler_accuracy.native_volume_induction_3d import NativeVolumeSources


def read_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def rms(field, weight=None):
    return float(np.sqrt(np.average(np.sum(np.asarray(field)**2, axis=-1), weights=weight)))


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    parent = json.loads((args.manufactured / "cube-manufactured-induction-3d.json").read_text())
    exact = json.loads((args.exact_moments / "cube-manufactured-linear-induction-3d.json").read_text())
    assert parent["status"] == exact["status"] == "complete"
    assert parent["spatial_dimensions"] == exact["spatial_dimensions"] == 3
    mesh_path = ROOT / next(p["path"] for p in parent["sources"] if p["path"].endswith("full-native-mesh.npz"))
    paths = [Path(__file__), mesh_path]
    paths += [args.manufactured / name for name in ("cube-manufactured-induction-3d.json", "manufactured-source-fields.npz", "manufactured-induction-fields.npz")]
    paths += [args.exact_moments / name for name in ("cube-manufactured-linear-induction-3d.json", "linear-source-fields.npz", "linear-induction-fields.npz")]
    paths += [ROOT / p for p in ("studies/coupler_accuracy/native_moment_reconstruction_3d.py", "studies/coupler_accuracy/native_linear_volume_induction_3d.py",
                                 "studies/coupler_accuracy/native_volume_induction_3d.py", "studies/coupler_accuracy/manufactured_cube_field_3d.py",
                                 "source/solvers/fvm/mesh/geometry.py", "tests/coupler/test_native_moment_reconstruction_3d.py")]
    sources = [hash_file(path) for path in paths]
    for path in paths:
        if path.suffix == ".py":
            destination = args.output / "sources" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
    (args.output / "sources-at-start.json").write_text(json.dumps(sources, indent=2)+"\n")
    source = read_arrays(args.manufactured / "manufactured-source-fields.npz")
    target = read_arrays(args.manufactured / "manufactured-induction-fields.npz")
    oracle = read_arrays(args.exact_moments / "linear-source-fields.npz")
    mesh = load_native_mesh(mesh_path)
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    native = NativeVolumeSources.from_mesh(mesh)
    linear = LinearNativeVolumeSources.from_native(native, geometry["cell_centre"])
    for key, value in (("volume", linear.volume), ("centroid", linear.centroid), ("covariance", linear.covariance)):
        np.testing.assert_array_equal(oracle[key], value)
    fields = [NoSlipCubeField(field["width"]) for field in parent["manufactured_fields"]]
    fit_poly = CellLeastSquares.from_mesh(mesh, linear.centroid)
    fit_fvm = CellLeastSquares.from_mesh(mesh, geometry["cell_centre"])
    names, gamma, moment, records = [], [], [], []
    boundary = np.stack([field.velocity(geometry["face_centre"]) for field in fields])
    cube = next(p for p in mesh["boundary"] if p["name"] == "cube")
    wall = slice(cube["start_face"], cube["start_face"]+cube["n_faces"])
    np.testing.assert_allclose(boundary[:, wall], 0, rtol=0, atol=1e-12)
    boundary[:, wall] = 0
    point_u = source["point_velocity"]
    point_grad = fit_fvm.gradient(point_u)
    # Point input is extrapolated to the true centroid for the corresponding
    # linear estimate of the cell integral. This correction vanishes when
    # stored and true centroids coincide and preserves affine velocity exactly.
    estimated_average = point_u+np.einsum("ci,scij->scj", linear.centroid-geometry["cell_centre"], point_grad)
    estimated_integral = estimated_average*linear.volume[None, :, None]
    native_faces = boundary.copy()
    n = mesh["n_interior_faces"]
    own, nei, w = mesh["owners"][:n], mesh["neighbours"], geometry["face_interpolation_weight"][:n]
    native_faces[:, :n] = (1-w[None, :, None])*point_u[:, own]+w[None, :, None]*point_u[:, nei]
    native_gamma, native_moment = weak_curl_moments(native, linear.centroid, estimated_integral,
                                                   geometry["face_centre"], native_faces)
    native_reference = np.stack([source["gamma"][parent["source_names"].index(f"width_{field.width}__native_curl_of_point_velocity")]
                                 for field in fields])
    native_circulation_replay = float(np.max(np.abs(native_gamma-native_reference)/linear.volume[None, :, None]))
    assert native_circulation_replay < 1e-10
    reconstructed_face_u, reconstructed_face_gradient = reconstruct_linear_face_velocity(mesh, geometry, point_u, point_grad, boundary)
    linear_gamma, linear_moment = weak_curl_moments(native, linear.centroid, estimated_integral, geometry["face_centre"],
                                                   reconstructed_face_u, reconstructed_face_gradient)
    average_u = source["velocity_integral"]/linear.volume[None, :, None]
    average_grad = fit_poly.gradient(average_u)
    average_face_u, average_face_gradient = reconstruct_linear_face_velocity(mesh, geometry, average_u, average_grad, boundary,
                                                                            cell_positions=linear.centroid)
    average_gamma, average_moment = weak_curl_moments(native, linear.centroid, source["velocity_integral"], geometry["face_centre"],
                                                     average_face_u, average_face_gradient)
    exact_circulation = source["vorticity_integral"]
    exact_circulation_gradient = fit_poly.gradient(exact_circulation/linear.volume[None, :, None])
    native_circulation_gradient = fit_poly.gradient(native_reference/linear.volume[None, :, None])
    candidates = [("lsq_exact_circulation", exact_circulation, linear.covariance[None] @ exact_circulation_gradient),
                  ("lsq_native_curl", native_reference, linear.covariance[None] @ native_circulation_gradient),
                  ("weak_native_faces", native_reference, native_moment),
                  ("weak_linear_faces_point_input", linear_gamma, linear_moment),
                  ("weak_linear_faces_average_input", average_gamma, average_moment)]
    radius = np.max(np.abs(geometry["cell_centre"]), axis=1)
    regions = {"all": np.ones(len(radius), dtype=bool), "near_body": radius < .8,
               "outer_layer": (radius > 1.1) & (radius < 1.5)}
    for index, field in enumerate(fields):
        exact_gradient = np.linalg.solve(linear.covariance, oracle["first_moment"][index])
        for name, strengths, moments in candidates:
            names.append(f"width_{field.width}__{name}")
            gamma.append(strengths[index])
            moment.append(moments[index])
            gradient = np.linalg.solve(linear.covariance, moments[index])
            delta = gradient-exact_gradient
            moment_error2 = np.einsum("nij,nik,nkj->n", delta, linear.covariance, delta)
            moment_norm2 = np.einsum("nij,nik,nkj->n", exact_gradient, linear.covariance, exact_gradient)
            record = {"name": names[-1], "method": name, "width": field.width, "source_error": {},
                      "total_circulation": strengths[index].sum(axis=0).tolist(),
                      "global_first_vorticity_moment": (moments[index].sum(axis=0)+np.einsum("ni,nj->ij", linear.centroid, strengths[index])).tolist()}
            for region, rows in regions.items():
                volume = linear.volume[rows]
                gamma_error = rms((strengths[index]-exact_circulation[index])[rows]/volume[:, None], volume)
                gamma_norm = rms(exact_circulation[index, rows]/volume[:, None], volume)
                record["source_error"][region] = {"circulation_density_rms": gamma_error, "circulation_relative_error": gamma_error/gamma_norm,
                                                   "moment_variation_density_rms": float(np.sqrt(moment_error2[rows].sum()/volume.sum())),
                                                   "moment_variation_relative_error": float(np.sqrt(moment_error2[rows].sum()/moment_norm2[rows].sum()))}
            records.append(record)
            print(json.dumps({"stage": "sources", **record}), flush=True)
    gamma, moment = np.stack(gamma), np.stack(moment)
    np.savez_compressed(args.output / "reconstructed-moment-source-fields.npz", source_names=np.asarray(names),
                        circulation=gamma, first_moment=moment, estimated_cell_velocity_integral=estimated_integral,
                        true_cell_velocity_integral=source["velocity_integral"], centres=geometry["cell_centre"],
                        volume=linear.volume, centroid=linear.centroid, covariance=linear.covariance)
    coefficients, recovered_gradient = linear.coefficients(gamma, moment)
    recovery_error = float(np.max(np.abs(linear.covariance[None] @ recovered_gradient-moment)/linear.volume[None, :, None, None]**(4/3)))
    assert recovery_error < 1e-10

    def progress(done, total):
        if done % 128 == 0 or done == total:
            print(json.dumps({"stage": "induction", "done": done, "total": total,
                              "elapsed_seconds": time.perf_counter()-started}), flush=True)

    velocity = linear.evaluate(target["position"], coefficients, progress=progress).transpose(1, 0, 2)
    np.savez_compressed(args.output / "reconstructed-moment-induction-fields.npz", position=target["position"],
                        velocity=velocity, exact_velocity=target["exact_velocity"])
    for state, record in enumerate(records):
        field_index = state//len(candidates)
        record["velocity_error"] = {}
        for group, limits in parent["target_slices"].items():
            rows = slice(*limits)
            error = velocity[state, rows]-target["exact_velocity"][field_index, rows]
            record["velocity_error"][group] = {"rms_over_reference_speed": rms(error),
                                               "maximum_over_reference_speed": float(np.max(np.linalg.norm(error, axis=1)))}
        print(json.dumps({"stage": "result", **record}), flush=True)
    report = {"schema": "openonda-cube-native-moment-reconstruction-3d/1", "status": "complete", "spatial_dimensions": 3,
              "parent_manufactured_directory": str(args.manufactured.relative_to(ROOT)), "exact_moment_directory": str(args.exact_moments.relative_to(ROOT)),
              "source_cells": native.n_cells, "source_triangles": len(native.triangles), "target_slices": parent["target_slices"],
              "source_names": names, "records": records, "sources": sources, "elapsed_seconds": time.perf_counter()-started,
              "native_circulation_replay_maximum_difference_over_volume": native_circulation_replay,
              "moment_recovery_maximum_difference_over_V_to_four_thirds": recovery_error,
              "lsq_polyhedron_maximum_condition": float(fit_poly.condition.max()), "lsq_fvm_maximum_condition": float(fit_fvm.condition.max()),
              "minimum_neighbour_count": int(fit_poly.neighbour_count.min()),
              "limitations": ["Kinematic 3D source reconstruction; not an advancing solver or a physical force prediction.",
                              "Only lsq_exact_circulation uses exact circulation as input. No candidate uses exact first moments or target velocities.",
                              "Point and true cell-average velocity inputs are separate measurements of the same analytic field.",
                              "Weak traces use the actual shared face fans and constant prescribed velocity over each boundary face. General variable VPM boundary traces need separate treatment.",
                              "The native-face control keeps the original native circulation. Weak linear-face methods also change circulation through a shared Stokes sum.",
                              "No source threshold, core-radius tuning, strength fit, f32 rounding or time advance is applied.",
                              "Moment relative error is the L2 error in the corresponding affine density variation, using the actual cell covariance."]}
    (args.output / "cube-native-moment-reconstruction-3d.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manufactured", type=Path, required=True)
    parser.add_argument("--exact-moments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.manufactured, args.exact_moments, args.output = args.manufactured.resolve(), args.exact_moments.resolve(), args.output.resolve()
    run(args)
