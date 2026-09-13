#!/usr/bin/env python3
"""Test prescribed wall velocity in quadratic reconstruction on the same 3D fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays, rms
from studies.coupler_accuracy.manufactured_cube_field_3d import NoSlipCubeField
from studies.coupler_accuracy.native_boundary_quadratic_3d import BoundaryQuadraticCellLeastSquares
from studies.coupler_accuracy.native_linear_volume_induction_3d import LinearNativeVolumeSources
from studies.coupler_accuracy.native_quadratic_moments_3d import (
    reconstruct_quadratic_faces,
    weak_quadratic_curl_moments,
)
from studies.coupler_accuracy.native_volume_induction_3d import NativeVolumeSources


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    parent = json.loads((args.manufactured / "cube-manufactured-induction-3d.json").read_text())
    exact = json.loads((args.exact_moments / "cube-manufactured-linear-induction-3d.json").read_text())
    assert parent["status"] == exact["status"] == "complete"
    assert parent["spatial_dimensions"] == exact["spatial_dimensions"] == 3
    mesh_path = ROOT / next(p["path"] for p in parent["sources"] if p["path"].endswith("full-native-mesh.npz"))
    paths = [Path(__file__), mesh_path]
    paths += [args.previous_quadratic / name for name in ("cube-quadratic-moment-reconstruction-3d.json", "quadratic-moment-source-fields.npz")]
    paths += [ROOT / name for name in ("studies/coupler_accuracy/native_boundary_quadratic_3d.py", "studies/coupler_accuracy/native_velocity_curl_integrals_3d.py", "tests/coupler/test_native_boundary_quadratic_3d.py")]
    paths += [args.manufactured / name for name in ("cube-manufactured-induction-3d.json", "manufactured-source-fields.npz", "manufactured-induction-fields.npz")]
    paths += [args.exact_moments / name for name in ("cube-manufactured-linear-induction-3d.json", "linear-source-fields.npz", "linear-induction-fields.npz")]
    paths += [ROOT / name for name in ("studies/coupler_accuracy/native_quadratic_moments_3d.py", "studies/coupler_accuracy/cube_native_moment_reconstruction_3d.py",
                                      "studies/coupler_accuracy/native_linear_volume_induction_3d.py", "studies/coupler_accuracy/native_volume_induction_3d.py",
                                      "studies/coupler_accuracy/manufactured_cube_field_3d.py", "source/solvers/fvm/mesh/geometry.py",
                                      "tests/coupler/test_native_quadratic_moments_3d.py")]
    sources = [hash_file(path) for path in paths]
    for path in paths:
        if path.suffix == ".py":
            destination = args.output / "sources" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
    (args.output / "sources-at-start.json").write_text(json.dumps(sources, indent=2)+"\n")
    source, target = [read_arrays(args.manufactured / name) for name in ("manufactured-source-fields.npz", "manufactured-induction-fields.npz")]
    oracle = read_arrays(args.exact_moments / "linear-source-fields.npz")
    previous = read_arrays(args.previous_quadratic / "quadratic-moment-source-fields.npz")
    previous_report = json.loads((args.previous_quadratic / "cube-quadratic-moment-reconstruction-3d.json").read_text())
    assert previous_report["status"] == "complete"
    mesh = load_native_mesh(mesh_path)
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    native = NativeVolumeSources.from_mesh(mesh)
    linear = LinearNativeVolumeSources.from_native(native, geometry["cell_centre"])
    for key, value in (("volume", linear.volume), ("centroid", linear.centroid), ("covariance", linear.covariance)):
        np.testing.assert_array_equal(oracle[key], value)
    fields = [NoSlipCubeField(row["width"]) for row in parent["manufactured_fields"]]
    boundary = np.stack([field.velocity(geometry["face_centre"]) for field in fields])
    cube = next(patch for patch in mesh["boundary"] if patch["name"] == "cube")
    wall = slice(cube["start_face"], cube["start_face"]+cube["n_faces"])
    np.testing.assert_allclose(boundary[:, wall], 0, rtol=0, atol=1e-12)
    boundary[:, wall] = 0
    candidates, diagnostics, saved = [], {}, {}
    for kind in ("point", "average"):
        is_average = kind == "average"
        centres = linear.centroid if is_average else geometry["cell_centre"]
        fit = BoundaryQuadraticCellLeastSquares.from_mesh(mesh, centres, linear.volume,
                                                  boundary_faces=np.arange(wall.start, wall.stop),
                                                  average_covariance=linear.covariance/linear.volume[:, None, None] if is_average else None)
        value = source["velocity_integral"]/linear.volume[None, :, None] if is_average else source["point_velocity"]
        prescribed = np.zeros((len(fields), len(fit.boundary_position), 3))
        gradient, hessian = fit.derivatives(value, prescribed)
        integral = fit.cell_integrals(value, gradient, hessian, linear.volume, linear.centroid, linear.covariance)
        if is_average:
            np.testing.assert_allclose(integral, source["velocity_integral"], rtol=0, atol=2e-16)
        face_u, face_g, face_h = reconstruct_quadratic_faces(mesh, geometry, fit, value, gradient, hessian, boundary)
        gamma, moment = weak_quadratic_curl_moments(native, linear.centroid, integral, geometry["face_centre"], face_u, face_g, face_h)
        changed = fit.boundary_observation_count > 0
        source_changed = changed.copy()
        interior = mesh["n_interior_faces"]
        own, nei = mesh["owners"][:interior], mesh["neighbours"]
        np.logical_or.at(source_changed, own, changed[nei])
        np.logical_or.at(source_changed, nei, changed[own])
        indices = [previous_report["source_names"].index(f"width_{field.width}__weak_quadratic_faces_{kind}_input") for field in fields]
        np.testing.assert_array_equal(gamma[:, ~source_changed], previous["circulation"][indices][:, ~source_changed])
        np.testing.assert_array_equal(moment[:, ~source_changed], previous["first_moment"][indices][:, ~source_changed])
        saved[kind+"__source_change_mask"] = source_changed
        saved[kind+"__boundary_observation_count"] = fit.boundary_observation_count
        saved[kind+"__boundary_position"] = fit.boundary_position
        saved[kind+"__boundary_face_ids"] = fit.boundary_face_ids
        saved[kind+"__boundary_area_weight"] = fit.boundary_area_weight
        candidates.append(("weak_boundary_quadratic_faces_"+kind+"_input", gamma, moment))
        saved[kind+"__cell_velocity_integral"] = integral
        saved.update({kind+"__"+key: getattr(fit, key) for key in ("condition", "neighbour_count", "rings")})
        diagnostics[kind] = {"maximum_condition": float(fit.condition.max()), "minimum_neighbour_count": int(fit.neighbour_count.min()),
                              "maximum_neighbour_count": int(fit.neighbour_count.max()), "maximum_rings": int(fit.rings.max()),
                              "cells_requiring_third_ring": int(np.sum(fit.rings > 2)),
                              "cells_with_wall_observations": int(changed.sum()), "potentially_changed_source_cells": int(source_changed.sum()),
                              "boundary_faces": cube["n_faces"], "boundary_points": len(fit.boundary_position),
                              "minimum_positive_boundary_observations": int(fit.boundary_observation_count[changed].min()),
                              "maximum_boundary_observations": int(fit.boundary_observation_count.max())}
        print(json.dumps({"stage": "reconstruction", "kind": kind, **diagnostics[kind], "elapsed_seconds": time.perf_counter()-started}), flush=True)
        del fit, gradient, hessian, face_u, face_g, face_h
    native_gamma = np.stack([source["gamma"][parent["source_names"].index(f"width_{field.width}__native_curl_of_point_velocity")]
                             for field in fields])
    candidates.append(("native_circulation_boundary_quadratic_point_moments", native_gamma, candidates[0][2]))
    radius = np.max(np.abs(geometry["cell_centre"]), axis=1)
    regions = {"all": np.ones(len(radius), dtype=bool), "near_body": radius < .8, "outer_layer": (radius > 1.1) & (radius < 1.5)}
    names, circulations, moments, records = [], [], [], []
    for index, field in enumerate(fields):
        truth_gamma = source["vorticity_integral"][index]
        truth_gradient = np.linalg.solve(linear.covariance, oracle["first_moment"][index])
        for name, gamma, moment in candidates:
            names.append(f"width_{field.width}__{name}")
            circulations.append(gamma[index])
            moments.append(moment[index])
            difference = np.linalg.solve(linear.covariance, moment[index])-truth_gradient
            squared = np.einsum("nij,nik,nkj->n", difference, linear.covariance, difference)
            norm2 = np.einsum("nij,nik,nkj->n", truth_gradient, linear.covariance, truth_gradient)
            record = {"name": names[-1], "method": name, "width": field.width, "source_error": {},
                      "total_circulation": gamma[index].sum(axis=0).tolist(),
                      "global_first_vorticity_moment": (moment[index].sum(axis=0)+np.einsum("ni,nj->ij", linear.centroid, gamma[index])).tolist()}
            for region, selected in regions.items():
                volume = linear.volume[selected]
                error = rms((gamma[index]-truth_gamma)[selected]/volume[:, None], volume)
                norm = rms(truth_gamma[selected]/volume[:, None], volume)
                record["source_error"][region] = {"circulation_density_rms": error, "circulation_relative_error": error/norm,
                                                   "moment_variation_density_rms": float(np.sqrt(squared[selected].sum()/volume.sum())),
                                                   "moment_variation_relative_error": float(np.sqrt(squared[selected].sum()/norm2[selected].sum()))}
            records.append(record)
            print(json.dumps({"stage": "sources", **record}), flush=True)
    circulation, moment = np.stack(circulations), np.stack(moments)
    np.savez_compressed(args.output / "boundary-quadratic-source-fields.npz", source_names=names, circulation=circulation, first_moment=moment,
                        centres=geometry["cell_centre"], volume=linear.volume, centroid=linear.centroid, covariance=linear.covariance, **saved)
    coefficients, _ = linear.coefficients(circulation, moment)

    def progress(done, total):
        if done % 128 == 0 or done == total:
            print(json.dumps({"stage": "induction", "done": done, "total": total, "elapsed_seconds": time.perf_counter()-started}), flush=True)

    velocity = linear.evaluate(target["position"], coefficients, progress=progress).transpose(1, 0, 2)
    np.savez_compressed(args.output / "boundary-quadratic-induction-fields.npz", position=target["position"], velocity=velocity, exact_velocity=target["exact_velocity"])
    for state, record in enumerate(records):
        record["velocity_error"] = {}
        for group, limits in parent["target_slices"].items():
            selected = slice(*limits)
            difference = velocity[state, selected]-target["exact_velocity"][state//len(candidates), selected]
            record["velocity_error"][group] = {"rms_over_reference_speed": rms(difference), "maximum_over_reference_speed": float(np.linalg.norm(difference, axis=1).max())}
        print(json.dumps({"stage": "result", **record}), flush=True)
    report = {"schema": "openonda-cube-boundary-quadratic-moments-3d/1", "status": "complete", "spatial_dimensions": 3,
              "source_cells": native.n_cells, "source_triangles": len(native.triangles), "target_slices": parent["target_slices"],
              "source_names": names, "records": records, "stencil_diagnostics": diagnostics, "sources": sources,
              "elapsed_seconds": time.perf_counter()-started,
              "limitations": ["Kinematic fully 3D reconstruction; no physical flow time advance or forces.",
                              "Prescribed cube velocity is added to the cell fit through a distance-scaled face-area residual; no wall-penalty parameter is tuned.",
                              "Only sources touching a changed cell polynomial may differ; all other Gamma and M reproduce the preceding fit exactly.",
                              "Quadratic cell velocity provides circulation and first moments to the existing affine-vorticity induction kernel.",
                              "Point and exact-average velocity inputs are separate; neither receives exact derivatives, moments or target velocities.",
                              "The native-circulation control retains original Gamma while using the point-input quadratic first moments.",
                              "Only full-rank 3D stencils are accepted. Two rings expand to three only when required by rank/conditioning.",
                              "Boundary traces are constant zero on the cube and exponentially small on the outer box; variable VPM boundary data are not tested.",
                              "All original full-mesh cells and common targets are retained; no pruning or tuned core size."]}
    (args.output / "cube-boundary-quadratic-moments-3d.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manufactured", type=Path, required=True)
    parser.add_argument("--exact-moments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-quadratic", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.manufactured, arguments.exact_moments, arguments.output = arguments.manufactured.resolve(), arguments.exact_moments.resolve(), arguments.output.resolve()
    arguments.previous_quadratic = arguments.previous_quadratic.resolve()
    run(arguments)
