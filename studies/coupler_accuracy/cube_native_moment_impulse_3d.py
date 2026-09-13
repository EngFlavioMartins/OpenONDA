#!/usr/bin/env python3
"""Track the raw impulse added by native-face moments in two saved FVM states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays
from studies.coupler_accuracy.native_moment_reconstruction_3d import weak_curl_moments
from studies.coupler_accuracy.native_volume_induction_3d import NativeVolumeSources


def antisymmetric_impulse(moment):
    return .5*np.array([moment[1, 2]-moment[2, 1], moment[2, 0]-moment[0, 2], moment[0, 1]-moment[1, 0]])


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__).resolve()]
    paths += [args.oracle / name for name in ("small-native-mesh.npz", "final-full-cell-fields.npz")]
    paths += [args.source / name for name in ("cube-shared-trace-sources-3d.json", "shared-trace-source-fields.npz")]
    paths += [ROOT / name for name in ("studies/coupler_accuracy/native_moment_reconstruction_3d.py",
                                      "studies/coupler_accuracy/native_volume_induction_3d.py",
                                      "studies/coupler_accuracy/cube_boundary_oracle.py",
                                      "studies/coupler_accuracy/cube_native_moment_reconstruction_3d.py",
                                      "source/solvers/fvm/mesh/geometry.py")]
    hashes = [hash_file(path) for path in paths]
    for path in paths:
        if path.suffix == ".py":
            target = args.output / "sources" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    report = json.loads((args.source / "cube-shared-trace-sources-3d.json").read_text())
    assert report["status"] == "complete" and report["spatial_dimensions"] == 3
    source = read_arrays(args.source / "shared-trace-source-fields.npz")
    final = read_arrays(args.oracle / "final-full-cell-fields.npz")
    np.testing.assert_array_equal(final["centres"], source["full_centres"])
    np.testing.assert_array_equal(final["cell_volumes"], source["full_volume"])
    mesh = load_native_mesh(args.oracle / "small-native-mesh.npz")
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    native = NativeVolumeSources.from_mesh(mesh)
    weight, centre, volume = source["volume_weight"], source["centroid"], source["volume"]
    ids = source["inside_cell_ids"]
    ni = mesh["n_interior_faces"]
    owner, neighbour = mesh["owners"][:ni], mesh["neighbours"]
    interpolation = geometry["face_interpolation_weight"][:ni]
    triangle_centre = native.triangles.mean(axis=1)
    vector_area = np.cross(native.triangles[:, 1]-native.triangles[:, 0], native.triangles[:, 2]-native.triangles[:, 0])/2
    displacement = weight[native.owners, None]*(triangle_centre-centre[native.owners])
    interior = native.neighbours >= 0
    displacement[interior] -= weight[native.neighbours[interior], None]*(triangle_centre[interior]-centre[native.neighbours[interior]])
    rows, saved = [], {}
    states = (("initial", report["physical_time"], source["cell_velocity"], source["native_circulation"]),
              ("final", float(final["physical_time"]), final["velocity"][ids], final["vorticity"][ids]*final["cell_volumes"][ids, None]))
    for label, time, velocity, gamma in states:
        face_velocity = np.zeros((1, mesh["n_faces"], 3))
        face_velocity[0, :ni] = velocity[owner]*(1-interpolation[:, None])+velocity[neighbour]*interpolation[:, None]
        integral = volume[None, :, None]*velocity[None]
        reconstructed_gamma, moment = weak_curl_moments(native, centre, integral, geometry["face_centre"], face_velocity)
        replay = float(np.max(np.abs(reconstructed_gamma[0, weight > 0]-gamma[weight > 0])/volume[weight > 0, None]))
        assert replay < 1e-10
        weighted_moment = weight[:, None, None]*moment[0]
        if label == "initial":
            np.testing.assert_allclose(weighted_moment, source["first_moment"][5], rtol=0, atol=1e-15)
        # Independent global boundary sum, without per-cell moment assembly.
        face_curl = np.cross(vector_area, face_velocity[0, native.face_ids])
        boundary_moment = np.sum(displacement[:, :, None]*face_curl[:, None], axis=0)
        integral_moment = np.cross(np.eye(3), np.sum(weight[:, None]*integral[0], axis=0))
        independent = boundary_moment-integral_moment
        summed = np.sum(weighted_moment, axis=0)
        difference = float(np.max(np.abs(summed-independent)))
        np.testing.assert_allclose(summed, independent, rtol=0, atol=2e-12)
        impulse = antisymmetric_impulse(summed)
        row = {"state": label, "physical_time": time, "moment_representation_impulse_change": impulse.tolist(),
               "independent_global_moment_maximum_difference": difference, "native_gamma_replay_maximum_difference_over_volume": replay}
        rows.append(row)
        saved[label+"__weighted_moment"] = weighted_moment
        saved[label+"__velocity"] = velocity
        saved[label+"__independent_global_moment"] = independent
    interval = rows[1]["physical_time"]-rows[0]["physical_time"]
    assert interval > 0
    rate = (np.asarray(rows[1]["moment_representation_impulse_change"])-rows[0]["moment_representation_impulse_change"])/interval
    np.savez_compressed(args.output / "native-moment-impulse-fields.npz", volume=volume, centroid=centre, volume_weight=weight, **saved)
    output = {"schema": "openonda-native-moment-impulse-3d/1", "status": "complete", "spatial_dimensions": 3,
              "source_sgs": report["source_sgs"], "small_fvm_cells": mesh["n_cells"], "small_fvm_bounds": report["small_fvm_bounds"],
              "states": rows, "interval": interval, "mean_rate_of_representation_impulse_difference": rate.tolist(), "sources": hashes,
              "limitations": ["Two frozen full-FVM states, not a coupled trajectory or particle-emission history.",
                              "The measurement is only the raw impulse added by weighted native-face moments relative to the same zero-moment cell sources.",
                              "Neither exterior vorticity cutoff nor particle core affects this difference, because the circulation and all Gaussian contributions cancel between representations.",
                              "The mean impulse-difference rate is not a drag prediction. Finite-domain flux terms, physical impulse evolution and the distinction between raw source vorticity and induced velocity curl remain."]}
    (args.output / "native-moment-impulse-3d.json").write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps({key: value for key, value in output.items() if key != "sources"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.oracle, args.source, args.output = args.oracle.resolve(), args.source.resolve(), args.output.resolve()
    run(args)
