#!/usr/bin/env python3
"""Qualify derivative traces and move native-volume support inside the FVM rim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from source.coupler.boundary import evaluate_vpm_velocity
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, field_rms, hash_file
from studies.coupler_accuracy.cube_panel_resolution_3d import particle_velocities
from studies.coupler_accuracy.joint_reconstruction_3d import CubePanelResponse
from studies.coupler_accuracy.native_volume_induction_3d import NativeVolumeSources


def smooth_volume_weight(radius, inner, outer):
    s = np.clip((radius-inner)/(outer-inner), 0., 1.)
    return 1-s*s*(3-2*s)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = json.loads((args.study / "cube-native-volume-induction-3d.json").read_text())
    prior = json.loads((args.partition / "cube-partitioned-volume-induction-3d.json").read_text())
    assert all(x["status"] == "complete" and x["spatial_dimensions"] == 3 for x in (report, prior))
    paths = {ROOT / record["path"] for previous in (report, prior) for record in previous["sources"]}
    paths.update((Path(__file__), args.partition / "partitioned-induction-fields.npz",
                  args.partition / "cube-partitioned-volume-induction-3d.json"))
    sources = [hash_file(path) for path in sorted(paths)]
    for path in paths:
        if path.suffix == ".py":
            destination = args.output / "sources" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
    (args.output / "sources-at-start.json").write_text(json.dumps(sources, indent=2)+"\n")
    with np.load(args.study / "volume-induction-comparison-fields.npz", allow_pickle=False) as data:
        fields = {k: data[k].copy() for k in data.files}
    with np.load(args.study / "native-induction-fields.npz", allow_pickle=False) as data:
        omega = data["cell_vorticity"].copy()
    with np.load(args.boundary / "boundary-fields.npz", allow_pickle=False) as data:
        boundary = {k: data[k].copy() for k in data.files}
    mesh = load_native_mesh(args.oracle / "small-native-mesh.npz")
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    native = NativeVolumeSources.from_mesh(mesh)
    centres, volumes = fields["native_cell_centres"], fields["native_cell_volume"]
    distance, inside_ids = cKDTree(centres).query(geometry["cell_centre"])
    np.testing.assert_allclose(distance, 0, rtol=0, atol=1e-13)
    outside = np.ones(len(centres), dtype=bool)
    outside[inside_ids] = False
    outside &= np.linalg.norm(omega, axis=1) >= 0.02
    assert outside.sum() == 68
    poly_volume, _ = native.cell_geometry(geometry["cell_centre"])
    radius = np.max(np.abs(geometry["cell_centre"]), axis=1)
    names = ["gaussian_control", "sharp_at_fvm_boundary", "sharp_inner_1.25",
             "taper_0.75_to_1.25", "taper_1.0_to_1.25"]
    weights = np.stack((np.zeros_like(radius), np.ones_like(radius), (radius < 1.25).astype(float),
                        smooth_volume_weight(radius, 0.75, 1.25), smooth_volume_weight(radius, 1.0, 1.25)))
    assert np.all((weights >= 0) & (weights <= 1))
    gamma = omega[inside_ids]*volumes[inside_ids, None]
    volume_density = weights[:, :, None]*gamma[None]/poly_volume[None, :, None]
    gaussian_gamma = (1-weights[:, :, None])*gamma[None]
    np.testing.assert_allclose(volume_density*poly_volume[None, :, None]+gaussian_gamma,
                               np.broadcast_to(gamma, gaussian_gamma.shape), rtol=0, atol=3e-17)
    position = np.vstack((centres[inside_ids], centres[outside].astype(np.float32).astype(float)))
    outer_gamma = (omega[outside]*volumes[outside, None]).astype(np.float32).astype(float)
    strengths = np.concatenate((gaussian_gamma, np.broadcast_to(outer_gamma, (len(names), len(outer_gamma), 3))), axis=1)
    selections = {k: slice(*v) for k, v in report["target_slices"].items()}
    points = [fields["position"]]
    count = len(points[0])
    epsilon = report["derivative_step"]
    for name, multiplier in (("plus2", 2), ("minus2", -2), ("plus_half", 0.5), ("minus_half", -0.5)):
        p = boundary["position"]+multiplier*epsilon*boundary["normal"]
        selections[name] = slice(count, count+len(p))
        count += len(p)
        points.append(p)
    targets = np.vstack(points)

    def progress(done, total):
        if done % 512 == 0 or done == total:
            print(json.dumps({"stage": "overlap_volumes", "done": done, "total": total,
                              "elapsed_seconds": time.perf_counter()-started}), flush=True)

    induced = native.evaluate(targets, native.coefficients(volume_density), progress=progress).transpose(1, 0, 2)
    induced += particle_velocities(targets, position, np.full(len(position), 0.125), strengths)
    print(json.dumps({"stage": "induction_complete", "elapsed_seconds": time.perf_counter()-started}), flush=True)
    body = CubePanelResponse()
    distance, cut_face_ids = cKDTree(geometry["face_centre"]).query(boundary["position"])
    np.testing.assert_allclose(distance, 0, rtol=0, atol=1e-13)
    owners = np.asarray(mesh["owners"])[cut_face_ids]
    n = boundary["native_unit_normal"]
    planar = np.array([np.max(np.abs((mesh["vertex_position"][mesh["faces"][face]]-boundary["position"][row]) @ n[row])) < 1e-12
                       for row, face in enumerate(cut_face_ids)])
    assert planar.sum() > 0
    legacy_length = np.linalg.norm(boundary["normal"], axis=1)

    def tangent(derivative):
        derivative = derivative/legacy_length[:, None]
        return derivative-np.sum(derivative*n, axis=1)[:, None]*n

    bounds = np.column_stack((mesh["vertex_position"].min(axis=0), mesh["vertex_position"].max(axis=0))).ravel()
    adapter = SimpleNamespace(particles=SimpleNamespace(n_particles_total=len(position)))
    records, output = [], {}
    for state, name in enumerate(names):
        body.panel.solve(np.array([1., 0., 0.]), induced[state, selections["collocation"]], time=report["physical_time"])
        actual = induced[state]+[1, 0, 0]+body.panel.compute_induced_velocity(targets)
        panel_flux = float(body.area @ body.panel.lattice.source_strength.to_numpy()[:body.count])
        assert abs(panel_flux) < 1e-10
        if name == "sharp_at_fvm_boundary":
            with np.load(args.partition / "partitioned-induction-fields.npz", allow_pickle=False) as data:
                np.testing.assert_allclose(actual[:len(fields["position"])], data["near_volume_outer_gaussian_seed__panel108__velocity"], rtol=0, atol=1e-12)
        centre = actual[selections["boundary"]]
        plus, minus = actual[selections["plus"]], actual[selections["minus"]]
        half_plus, half_minus = actual[selections["plus_half"]], actual[selections["minus_half"]]
        gt = {"centred": tangent((plus-minus)/(2*epsilon)),
              "centred_half": tangent((half_plus-half_minus)/epsilon),
              "exterior": tangent((-3*centre+4*plus-actual[selections["plus2"]])/(2*epsilon)),
              "interior": tangent((3*centre-4*minus+actual[selections["minus2"]])/(2*epsilon)),
              "exterior_half": tangent((-3*centre+4*half_plus-plus)/epsilon),
              "interior_half": tangent((3*centre-4*half_minus+minus)/epsilon)}
        cut, flux = evaluate_vpm_velocity(adapter, boundary["position"], boundary["normal"], boundary["area"],
                                          freestream_velocity=np.array([1., 0., 0.]), fvm_box=bounds,
                                          particle_spacing=0.125, evaluated_velocity=centre)
        un = np.sum(cut*boundary["normal"], axis=1)
        errors = {kind: field_rms(value-boundary["fvm_native_flux_tangential_normal_gradient"], boundary["native_vector_area"])
                  for kind, value in gt.items()}
        predicted_jump = np.cross(n, volume_density[state, owners])
        actual_jump = gt["exterior_half"]-gt["interior_half"]
        jump_error = float(np.max(np.abs(actual_jump[planar]-predicted_jump[planar])))
        assert jump_error < 1e-6
        active = weights[state] > 0
        gap = None
        if active.any():
            rows = active[native.owners] | ((native.neighbours >= 0) & active[np.maximum(native.neighbours, 0)])
            low, high = native.triangles[rows].min(axis=(0, 1)), native.triangles[rows].max(axis=(0, 1))
            delta = np.maximum(np.maximum(low-boundary["position"], boundary["position"]-high), 0)
            gap = float(np.linalg.norm(delta, axis=1).min())
        wall = actual[selections["wall"]]
        wall_un = np.sum(wall*fields["wall_normal"], axis=1)
        record = {"name": name, "positive_volume_cells": int(active.sum()),
                  "minimum_coupling_target_distance_to_volume_support_box": gap,
                  "cell_velocity_rms_over_Uinf": {},
                  "boundary_normal_velocity_rms_error_over_Uinf": field_rms((un-boundary["fvm_normal_velocity"])[:, None], boundary["area"]),
                  "boundary_native_tangential_gradient_rms_error": errors,
                  "all_face_step_halving_maximum_gradient_difference": {side: float(np.max(np.abs(gt[side]-gt[side+"_half"])))
                                                                         for side in ("centred", "exterior", "interior")},
                  "all_face_one_sided_tangential_gradient_jump_rms": field_rms(actual_jump, boundary["native_vector_area"]),
                  "planar_face_jump_maximum_difference_from_n_cross_omega": jump_error,
                  "wall_normal_velocity_rms_over_Uinf": float(np.sqrt(np.mean(wall_un**2))),
                  "wall_tangential_velocity_rms_over_Uinf": float(np.sqrt(np.mean(np.sum((wall-wall_un[:, None]*fields["wall_normal"])**2, axis=1)))),
                  "boundary_flux": flux, "body_source_flux": panel_flux}
        for group in report["cell_target_counts"]:
            ids = fields[group+"__cell_ids"]
            record["cell_velocity_rms_over_Uinf"][group] = field_rms(actual[selections[group]]-fields["native_cell_velocity"][ids], volumes[ids])
        records.append(record)
        output[name+"__velocity"] = actual
        output[name+"__boundary_normal_velocity"] = un
        output.update({name+"__gradient_"+kind: value for kind, value in gt.items()})
        print(json.dumps(record), flush=True)
    np.savez_compressed(args.output / "volume-overlap-fields.npz", position=targets, inside_cell_ids=inside_ids,
                        volume_weights=weights, volume_density=volume_density, gaussian_position=position,
                        gaussian_strength=strengths, cut_face_ids=cut_face_ids, planar_cut_faces=planar, **output)
    result = {"schema": "openonda-cube-volume-overlap-3d/1", "status": "complete", "spatial_dimensions": 3,
              "physical_time": report["physical_time"], "near_fvm_cells": len(inside_ids), "exterior_particles": 68,
              "state_names": names, "targets": len(targets), "target_slices": {k: [v.start, v.stop] for k, v in selections.items()},
              "planar_coupling_faces_for_jump_identity": int(planar.sum()), "coupling_faces": len(owners),
              "derivative_steps": [epsilon, epsilon/2], "records": records, "sources": sources,
              "elapsed_seconds": time.perf_counter()-started,
              "limitations": ["Frozen induction only; no coupled evolution, force validation, emission rule or particle transport test.",
                              "Each near-cell omega*V_FVM is split by a nonnegative geometric weight between native volume and Gaussian sources.",
                              "The same 68 exterior seed particles remain fixed in every state. Near-cell Gaussian complements use sigma=0.125 and unrounded FVM values.",
                              "One-sided derivatives use second-order formulas and are checked by halving the step on all 864 coupling faces.",
                              "The sharp-volume jump identity is checked only on geometrically planar coupling faces; warped face-centre traces need their own geometric interpretation.",
                              "Moving or tapering volume support inward removes its sharp boundary from the coupling faces; it does not prove smooth transport across the interior overlap."]}
    (args.output / "cube-volume-overlap-3d.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    results = ROOT / "studies/coupler_accuracy/results"
    parser.add_argument("--study", type=Path, default=results / "cube-3d-native-volume-induction")
    parser.add_argument("--partition", type=Path, default=results / "cube-3d-partitioned-volume-induction")
    parser.add_argument("--oracle", type=Path, default=results / "cube-3d-oracle")
    parser.add_argument("--boundary", type=Path, default=results / "cube-3d-integrated-velocity-curl-reconstruction-boundary")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
