#!/usr/bin/env python3
"""Restrict native volume induction to the small FVM domain of the frozen cube."""

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
from studies.coupler_accuracy.cube_snapshot_induction import volume_velocity
from studies.coupler_accuracy.joint_reconstruction_3d import CubePanelResponse
from studies.coupler_accuracy.native_volume_induction_3d import NativeVolumeSources


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    paths = [Path(__file__), args.study / "cube-native-volume-induction-3d.json",
             args.study / "native-induction-fields.npz", args.study / "volume-induction-comparison-fields.npz",
             args.oracle / "small-native-mesh.npz", args.boundary / "boundary-fields.npz"]
    paths += [ROOT / p for p in (
        "studies/coupler_accuracy/native_volume_induction_3d.py",
        "studies/coupler_accuracy/cube_snapshot_induction.py", "studies/coupler_accuracy/joint_reconstruction_3d.py",
        "studies/coupler_accuracy/cube_boundary_oracle.py", "source/coupler/boundary.py",
        "source/solvers/fvm/mesh/geometry.py", "source/solvers/vpm/boundary_elements/panels/kernels/source_velocity.py",
        "source/solvers/vpm/boundary_elements/panels/kernels/induced_velocity.py",
        "source/solvers/vpm/boundary_elements/panels/solver/linear_solvers.py",
        "source/solvers/vpm/boundary_elements/panels/solver/panel_solver.py",
        "tutorials/coupled_fvm_vpm/02_cube_flow/assets/cube.stl")]
    sources = [hash_file(p) for p in paths]
    for path in paths:
        if path.suffix == ".py":
            destination = args.output / "sources" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
    (args.output / "sources-at-start.json").write_text(json.dumps(sources, indent=2)+"\n")
    report = json.loads((args.study / "cube-native-volume-induction-3d.json").read_text())
    if report["status"] != "complete" or report["spatial_dimensions"] != 3:
        raise ValueError("A completed fully 3D whole-volume comparison is required")
    with np.load(args.study / "volume-induction-comparison-fields.npz", allow_pickle=False) as data:
        fields = {k: data[k].copy() for k in data.files}
    with np.load(args.study / "native-induction-fields.npz", allow_pickle=False) as data:
        omega = data["cell_vorticity"].copy()
        full_native = data["velocity"][:, 1].copy()
        full_poly_volume = data["polyhedron_volume"].copy()
    with np.load(args.boundary / "boundary-fields.npz", allow_pickle=False) as data:
        boundary = {k: data[k].copy() for k in data.files}
    mesh = load_native_mesh(args.oracle / "small-native-mesh.npz")
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    centres, volumes = fields["native_cell_centres"], fields["native_cell_volume"]
    distance, inside_ids = cKDTree(centres).query(geometry["cell_centre"])
    np.testing.assert_allclose(distance, 0, rtol=0, atol=1e-13)
    assert len(np.unique(inside_ids)) == mesh["n_cells"]
    np.testing.assert_allclose(geometry["cell_volume"], volumes[inside_ids], rtol=0, atol=1e-14)
    outside = np.ones(len(centres), dtype=bool)
    outside[inside_ids] = False
    native = NativeVolumeSources.from_mesh(mesh)
    poly_volume, _ = native.cell_geometry(geometry["cell_centre"])
    np.testing.assert_allclose(poly_volume, full_poly_volume[inside_ids], rtol=0, atol=1e-14)
    targets = fields["position"]
    selections = {k: slice(*v) for k, v in report["target_slices"].items()}
    coefficient = native.coefficients(omega[inside_ids]*(volumes[inside_ids]/poly_volume)[:, None])

    def progress(done, total):
        if done % 512 == 0 or done == total:
            print(json.dumps({"stage": "near_volume", "done": done, "total": total,
                              "elapsed_seconds": time.perf_counter()-started}), flush=True)

    near_volume = native.evaluate(targets, coefficient, progress=progress)[:, 0]
    exterior_velocities, counts = {}, {}
    for name, prune in (("all", False), ("seed", True)):
        keep = outside & (np.linalg.norm(omega, axis=1) >= 0.02) if prune else outside
        position, strength = centres[keep], (omega*volumes[:, None])[keep]
        if prune:
            position = position.astype(np.float32).astype(float)
            strength = strength.astype(np.float32).astype(float)
        induced = np.zeros_like(targets)
        for start in range(0, len(targets), 64):
            induced[start:start+64] = volume_velocity(targets[start:start+64], position, strength, 0.125)[0]
        exterior_velocities[name] = induced
        counts[name] = len(position)
    full_gaussian = fields["gaussian_sigma_0.125__freestream__velocity"]-[1, 0, 0]
    candidates = {
        "near_volume_outer_gaussian_all": near_volume+exterior_velocities["all"],
        "near_volume_outer_gaussian_seed": near_volume+exterior_velocities["seed"],
        "near_gaussian_outer_volume_control": full_gaussian-exterior_velocities["all"]+full_native-near_volume,
    }
    np.testing.assert_allclose(candidates["near_volume_outer_gaussian_all"]
                               + candidates["near_gaussian_outer_volume_control"],
                               full_native+full_gaussian, rtol=0, atol=5e-16)
    body = CubePanelResponse()
    bounds = np.column_stack((mesh["vertex_position"].min(axis=0), mesh["vertex_position"].max(axis=0))).ravel()
    adapter = SimpleNamespace(particles=SimpleNamespace(n_particles_total=counts["all"]))
    epsilon = report["derivative_step"]
    legacy_length = np.linalg.norm(boundary["normal"], axis=1)
    records, output_fields = [], {}
    for name, induced in candidates.items():
        body.panel.solve(np.array([1., 0., 0.]), induced[selections["collocation"]], time=report["physical_time"])
        panel_velocity = body.panel.compute_induced_velocity(targets)
        source_flux = float(body.area @ body.panel.lattice.source_strength.to_numpy()[:body.count])
        assert abs(source_flux) < 1e-10
        for variant, actual in (("freestream", induced+[1, 0, 0]),
                                ("panel108", induced+[1, 0, 0]+panel_velocity)):
            cut, mass = evaluate_vpm_velocity(adapter, boundary["position"], boundary["normal"], boundary["area"],
                                              freestream_velocity=np.array([1., 0., 0.]), fvm_box=bounds,
                                              particle_spacing=0.125, evaluated_velocity=actual[selections["boundary"]])
            un = np.sum(cut*boundary["normal"], axis=1)
            derivative = (actual[selections["plus"]]-actual[selections["minus"]])/(2*epsilon)
            check = (actual[selections["check_plus"]]-actual[selections["check_minus"]])/(4*epsilon)
            normal_derivative = derivative/legacy_length[:, None]
            tangent = normal_derivative-np.sum(normal_derivative*boundary["native_unit_normal"], axis=1)[:, None]*boundary["native_unit_normal"]
            wall = actual[selections["wall"]]
            wall_un = np.sum(wall*fields["wall_normal"], axis=1)
            record = {"representation": name, "completion": variant,
                      "cell_velocity_rms_over_Uinf": {},
                      "wall_normal_velocity_rms_over_Uinf": float(np.sqrt(np.mean(wall_un**2))),
                      "wall_tangential_velocity_rms_over_Uinf": float(np.sqrt(np.mean(np.sum(
                          (wall-wall_un[:, None]*fields["wall_normal"])**2, axis=1)))),
                      "boundary_normal_velocity_rms_error_over_Uinf": field_rms((un-boundary["fvm_normal_velocity"])[:, None], boundary["area"]),
                      "boundary_native_tangential_gradient_rms_error": field_rms(tangent-boundary["fvm_native_flux_tangential_normal_gradient"], boundary["native_vector_area"]),
                      "derivative_step_check_maximum_difference": float(np.max(np.abs(derivative[:32]-check))),
                      "boundary_flux": mass, "panel_source_flux": source_flux if variant == "panel108" else None}
            for group in report["cell_target_counts"]:
                ids = fields[group+"__cell_ids"]
                record["cell_velocity_rms_over_Uinf"][group] = field_rms(
                    actual[selections[group]]-fields["native_cell_velocity"][ids], volumes[ids])
            key = name+"__"+variant
            output_fields[key+"__velocity"] = actual
            output_fields[key+"__boundary_normal_velocity"] = un
            output_fields[key+"__boundary_native_tangential_gradient"] = tangent
            records.append(record)
            print(json.dumps(record), flush=True)
    np.savez_compressed(args.output / "partitioned-induction-fields.npz", position=targets, inside_cell_ids=inside_ids,
                        near_volume_induction=near_volume, outer_gaussian_all=exterior_velocities["all"],
                        outer_gaussian_seed=exterior_velocities["seed"], **output_fields)
    result = {"schema": "openonda-cube-partitioned-volume-induction-3d/1", "status": "complete", "spatial_dimensions": 3,
              "physical_time": report["physical_time"], "near_volume_cells": len(inside_ids),
              "near_volume_triangles": len(native.triangles), "exterior_particles": counts, "body_panels": body.count,
              "small_fvm_bounds": bounds.tolist(), "target_slices": report["target_slices"],
              "records": records, "sources": sources, "elapsed_seconds": time.perf_counter()-started,
              "limitations": ["A frozen induction partition only; no particle injection, advection, FVM update or force calculation.",
                              "Near-volume cells preserve each original omega*V_FVM. All other cell fields are represented by Gaussian particles of radius 0.125.",
                              "The seed variant keeps only exterior sources with |omega| >= 0.02 and rounds their positions/strengths to the original f32 seed representation.",
                              "The reversed partition is a diagnostic control and uses unavailable full-domain exterior volume data.",
                              "The exterior particle states are initially seeded from the full FVM oracle, as in the preceding frozen experiments."]}
    (args.output / "cube-partitioned-volume-induction-3d.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    results = ROOT / "studies/coupler_accuracy/results"
    parser.add_argument("--study", type=Path, default=results / "cube-3d-native-volume-induction")
    parser.add_argument("--oracle", type=Path, default=results / "cube-3d-oracle")
    parser.add_argument("--boundary", type=Path, default=results / "cube-3d-integrated-velocity-curl-reconstruction-boundary")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
