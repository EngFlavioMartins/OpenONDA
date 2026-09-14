#!/usr/bin/env python3
"""Measure velocity-mean compatibility of conservative flux traces in 3D.

This is an offline observation of accepted reference and hybrid states. It
does not change either solver or assert that a volume reconstruction exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from source.solvers.fvm.io.backup import decode_state, mesh_hash
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays
from studies.coupler_accuracy.native_flux_moments_3d import (
    cell_flux_moments,
    enforce_face_flux,
    linear_face_flux_moments,
)
from studies.coupler_accuracy.native_moment_reconstruction_3d import (
    CellLeastSquares,
    reconstruct_linear_face_velocity,
)
from studies.coupler_accuracy.native_volume_induction_3d import NativeVolumeSources


def weighted_rms(field, volume):
    return float(np.sqrt(np.sum(volume * np.sum(field**2, axis=-1)) / volume.sum()))


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest_path = ROOT / "frozen-workspace.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "complete"
    for row in manifest["records"]:
        assert hash_file(ROOT / row["path"])["sha256"] == row["sha256"]
    report_path = args.run / "accepted-fvm-checkpoints-3d.json"
    profile_path = args.run / "profile-observation-3d.json"
    parent = json.loads(report_path.read_text())
    profile = json.loads(profile_path.read_text())
    assert parent["status"] == profile["status"] == "complete"
    assert parent["spatial_dimensions"] == profile["spatial_dimensions"] == 3
    assert hash_file(profile_path) == parent["profile_report"]
    frames = [frame for frame in parent["frames"] if frame["fvm_step"] in args.steps]
    assert sorted(frame["fvm_step"] for frame in frames) == sorted(set(args.steps))
    profile_geometry_path = ROOT / profile["geometry"]["path"]
    assert hash_file(profile_geometry_path) == profile["geometry"]
    profile_geometry = read_arrays(profile_geometry_path)
    paths = [Path(__file__).resolve(), manifest_path, report_path, profile_path, profile_geometry_path]
    paths += [ROOT / name for name in (
        "studies/coupler_accuracy/native_flux_moments_3d.py",
        "studies/coupler_accuracy/native_moment_reconstruction_3d.py",
        "studies/coupler_accuracy/native_volume_induction_3d.py",
        "studies/coupler_accuracy/cube_native_moment_reconstruction_3d.py",
        "studies/coupler_accuracy/cube_boundary_oracle.py",
        "source/solvers/fvm/io/backup.py", "source/solvers/fvm/io/mesh_storage.py",
        "source/solvers/fvm/mesh/geometry.py", "tests/coupler/test_native_flux_moments_3d.py")]
    paths += [ROOT / row["path"] for row in parent["meshes"].values()]
    paths += [ROOT / row["path"] for frame in frames for row in frame["checkpoints"].values()]
    sources = [hash_file(path) for path in paths]
    for path in paths:
        if path.suffix == ".py":
            target = args.output / "sources" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    (args.output / "sources-at-start.json").write_text(json.dumps(sources, indent=2) + "\n")
    records, geometry_records = [], []
    shared_centre = profile_geometry["small_cell_centres"]
    radius = np.max(np.abs(shared_centre), axis=1)
    regions = {"all_shared": np.ones(len(radius), dtype=bool), "near_body_0.8": radius < .8,
               "near_body_1.0": radius < 1, "overlap": (radius >= .75) & (radius < 1.25),
               "outer_shared": radius >= 1.25}
    for name in ("full", "hybrid"):
        mesh_path = ROOT / parent["meshes"][name]["path"]
        assert hash_file(mesh_path) == parent["meshes"][name]
        mesh = load_native_mesh(mesh_path)
        geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
        native = NativeVolumeSources.from_mesh(mesh)
        volume, centre = native.cell_geometry(geometry["cell_centre"])
        ids = profile_geometry["shared_cell_ids"] if name == "full" else np.arange(mesh["n_cells"])
        np.testing.assert_allclose(geometry["cell_centre"][ids], shared_centre, rtol=0, atol=1e-13)
        origin, sf = geometry["face_centre"], geometry["face_area_vector"]
        n, ni, nf = mesh["n_cells"], mesh["n_interior_faces"], mesh["n_faces"]
        fit = CellLeastSquares.from_mesh(mesh, geometry["cell_centre"])
        summary_patch, triangle_patch, area_closure = [], [], []
        for axis in np.eye(3):
            flux, moment = linear_face_flux_moments(native, origin, np.broadcast_to(axis, origin.shape))
            np.testing.assert_allclose(flux, sf @ axis, rtol=0, atol=3e-14)
            net, first = cell_flux_moments(mesh, origin, flux, moment, centre)
            _, summary = cell_flux_moments(mesh, origin, flux, np.zeros_like(moment), centre)
            triangle_patch.append(float(np.max(np.abs(first / volume[:, None] - axis))))
            summary_patch.append(float(np.max(np.abs(summary / geometry["cell_volume"][:, None] - axis))))
            area_closure.append(float(np.max(np.abs(net / volume))))
        geometry_records.append({"name": name, "cells": n, "shared_cells": len(ids), "triangles": len(native.triangles),
                                 "summary_constant_patch_maximum_component_errors": summary_patch,
                                 "triangle_constant_patch_maximum_component_errors": triangle_patch,
                                 "area_closure_over_volume_maximum_components": area_closure,
                                 "maximum_relative_volume_difference": float(np.max(np.abs(volume / geometry["cell_volume"] - 1))),
                                 "maximum_centroid_displacement": float(np.max(np.linalg.norm(centre - geometry["cell_centre"], axis=1)))})
        own, nei = mesh["owners"][:ni], mesh["neighbours"]
        weight = geometry["face_interpolation_weight"][:ni, None]
        in_shared = np.zeros(n, dtype=bool)
        in_shared[ids] = True
        shared_faces = in_shared[own] & in_shared[nei]
        for frame in frames:
            checkpoint = frame["checkpoints"][name]
            assert hash_file(ROOT / checkpoint["path"]) == checkpoint
            raw = read_arrays(ROOT / checkpoint["path"])
            assert json.loads(raw["metadata"].item())["mesh_hash"] == mesh_hash(mesh)
            state = decode_state(raw)
            assert int(state["step"]) == frame["fvm_step"]
            u, phi = state["velocity"], state["volumetric_face_flux"]
            constant = np.empty((nf, 3))
            constant[:ni] = (1 - weight) * u[own] + weight * u[nei]
            constant[ni:] = u[n:]
            gradient = fit.gradient(u[:n])[0]
            affine, affine_gradient = reconstruct_linear_face_velocity(mesh, geometry, u[:n], gradient, constant)
            names = ["face_summary_constant_normal", "native_constant_velocity_trace", "native_affine_velocity_trace"]
            means, moments, shifts, raw_fluxes = [], [], [], []
            for variant in names:
                if variant == names[0]:
                    flux, moment, shift = phi, np.zeros((nf, 3)), np.zeros(nf)
                    raw_flux = phi
                else:
                    values, derivatives = ((constant, None) if variant == names[1] else (affine[0], affine_gradient[0]))
                    raw_flux, raw_moment = linear_face_flux_moments(native, origin, values, derivatives)
                    flux, moment, shift = enforce_face_flux(native, origin, raw_flux, raw_moment, phi)
                net, first = cell_flux_moments(mesh, origin, flux, moment, centre)
                means.append(first / volume[:, None])
                moments.append(moment)
                shifts.append(shift)
                raw_fluxes.append(raw_flux)
            means = np.stack(means)
            row = {"name": name, "physical_time": frame["physical_time"], "fvm_step": frame["fvm_step"],
                   "reset_initial_state": frame["fvm_step"] == 0, "variants": []}
            for variant, mean, shift in zip(names, means, shifts, strict=True):
                measures = {}
                for region, mask in regions.items():
                    selected = ids[mask]
                    error = mean[selected] - u[selected]
                    transported_error = mean[selected] - u[selected] * (geometry["cell_volume"][selected] / volume[selected])[:, None]
                    measures[region] = {"cells": len(selected), "mean_velocity_difference_rms_over_Uinf": weighted_rms(error, volume[selected]),
                                         "mean_velocity_difference_maximum_over_Uinf": float(np.linalg.norm(error, axis=1).max()),
                                         "transported_momentum_difference_rms_as_velocity": weighted_rms(transported_error, volume[selected]),
                                         "stored_flux_divergence_rms": weighted_rms((net[selected] / volume[selected])[:, None], volume[selected])}
                row["variants"].append({"name": variant, "regions": measures,
                                         "shared_interior_normal_adjustment_rms": weighted_rms(shift[:ni][shared_faces, None], geometry["face_area"][:ni][shared_faces])})
            output = args.output / f"{name}-step-{frame['fvm_step']:06d}-flux-moments.npz"
            np.savez_compressed(output, shared_cell_ids=ids, cell_centroid=centre, polyhedron_volume=volume,
                                fvm_volume=geometry["cell_volume"], fvm_cell_centre=geometry["cell_centre"], stored_velocity=u[:n],
                                face_origin=origin, stored_flux=phi, net_flux=net, variant_names=np.asarray(names),
                                implied_velocity=means, face_first_moment=np.stack(moments), raw_face_flux=np.stack(raw_fluxes),
                                normal_shift=np.stack(shifts), constant_face_velocity=constant, affine_face_velocity=affine[0],
                                affine_face_gradient=affine_gradient[0])
            row["fields"] = hash_file(output)
            records.append(row)
            print(json.dumps({"name": name, "step": frame["fvm_step"], "physical_time": frame["physical_time"],
                              "near_0.8_rms": {v["name"]: v["regions"]["near_body_0.8"]["mean_velocity_difference_rms_over_Uinf"] for v in row["variants"]}}), flush=True)
    for row in sources:
        assert hash_file(ROOT / row["path"]) == row
    for row in manifest["records"]:
        assert hash_file(ROOT / row["path"])["sha256"] == row["sha256"]
    result = {"schema": "openonda-flux-moment-compatibility-3d/1", "status": "complete", "spatial_dimensions": 3,
              "frozen_original_files_verified": len(manifest["records"]), "sources": sources,
              "geometry": geometry_records, "records": records, "elapsed_seconds": time.perf_counter() - started,
              "limitations": [
                  "This observes normal-trace moments. It does not construct an H(div) volume field, particles, or a time-advancing coupling.",
                  "Only a vanishing first divergence moment makes the boundary first flux moment equal the cell velocity integral; zero net cell flux alone is insufficient.",
                  "Native affine traces use local neighbouring cell velocities and constant stored boundary-face velocities. Variable traces over coupling boundary faces are not inferred.",
                  "The summary control deliberately omits within-face moments and fails constant reproduction on warped faces. It is a diagnostic approximation, not the FVM momentum algorithm.",
                  "Both stored cell-velocity and stored-volume momentum comparisons are shown; true fan geometry differs slightly from the FVM summaries.",
                  "Step zero, if requested, is the existing reset initial state and is not an accepted conservative step.",
                  "The reference and hybrid comparisons use the same mapped native cells; no reference values are supplied to the hybrid reconstruction.",
                  "Reducing this necessary moment discrepancy does not establish reference force or profile agreement.",
              ]}
    (args.output / "flux-moment-compatibility-3d.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[15])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.run, args.output = args.run.resolve(), args.output.resolve()
    run(args)
