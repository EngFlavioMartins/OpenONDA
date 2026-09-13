#!/usr/bin/env python3
"""Compare solenoidal tetrahedral curl and its identical-moment affine density."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import taichi as ti

import openonda.vpm as vpm
from source.coupler.boundary import evaluate_vpm_velocity
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from studies.coupler_accuracy.continuous_velocity_curl_3d import ContinuousVelocityCurl
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, field_rms, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays
from studies.coupler_accuracy.cube_shared_trace_induction_3d import unique_targets
from studies.coupler_accuracy.native_face_velocity_sampling_3d import InteriorFaceVelocitySampler
from studies.coupler_accuracy.native_linear_volume_induction_3d import (
    LinearNativeVolumeSources,
    curl_affine,
)
from studies.coupler_accuracy.native_volume_induction_3d import NativeVolumeSources


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source_report_path = args.source / "cube-continuous-curl-sources-3d.json"
    source_path = args.source / "continuous-curl-source-fields.npz"
    baseline_report_path = args.baseline / "cube-shared-trace-induction-3d.json"
    baseline_path = args.baseline / "shared-trace-physical-fields.npz"
    source_report = json.loads(source_report_path.read_text())
    baseline_report = json.loads(baseline_report_path.read_text())
    for report in (source_report, baseline_report):
        assert report["status"] == "complete" and report["spatial_dimensions"] == 3
        for row in report["sources"]:
            assert hash_file(ROOT / row["path"]) == row
    parent = ROOT / source_report["parent_source"]
    parent_report = json.loads((parent / "cube-shared-trace-sources-3d.json").read_text())
    assert any(row["path"] == str((parent / "shared-trace-source-fields.npz").relative_to(ROOT)) for row in baseline_report["sources"])
    small_path = ROOT / next(row["path"] for row in baseline_report["sources"] if row["path"].endswith("/small-native-mesh.npz"))
    full_path = ROOT / next(row["path"] for row in baseline_report["sources"] if row["path"].endswith("/full-native-mesh.npz"))
    surface = ROOT / next(row["path"] for row in baseline_report["sources"] if row["path"].endswith("/refined-cube.stl"))
    paths = [Path(__file__).resolve(), source_report_path, source_path, baseline_report_path, baseline_path,
             parent / "cube-shared-trace-sources-3d.json", parent / "shared-trace-source-fields.npz", small_path, full_path, surface]
    paths += [ROOT / name for name in (
        "studies/coupler_accuracy/continuous_velocity_curl_3d.py", "studies/coupler_accuracy/cube_shared_trace_induction_3d.py",
        "studies/coupler_accuracy/native_face_velocity_sampling_3d.py", "studies/coupler_accuracy/native_face_trace_3d.py",
        "studies/coupler_accuracy/native_linear_volume_induction_3d.py", "studies/coupler_accuracy/native_volume_induction_3d.py",
        "studies/coupler_accuracy/cube_boundary_oracle.py", "studies/coupler_accuracy/cube_native_moment_reconstruction_3d.py",
        "source/coupler/boundary.py", "source/solvers/fvm/io/mesh_storage.py", "source/solvers/fvm/mesh/geometry.py",
        "source/solvers/fvm/fields/gradients.py", "source/solvers/fvm/assemble/diffusion.py",
        "source/solvers/vpm/boundary_elements/panels/solver/panel_solver.py", "source/solvers/vpm/boundary_elements/panels/solver/linear_solvers.py",
        "source/solvers/vpm/boundary_elements/panels/kernels/source_velocity.py", "source/solvers/vpm/boundary_elements/panels/kernels/induced_velocity.py")]
    sources = [hash_file(path) for path in paths]
    for path in paths:
        if path.suffix == ".py":
            archive = args.output / "sources" / path.relative_to(ROOT)
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_bytes(path.read_bytes())
    (args.output / "sources-at-start.json").write_text(json.dumps(sources, indent=2)+"\n")
    source, base, donor = read_arrays(source_path), read_arrays(baseline_path), read_arrays(parent / "shared-trace-source-fields.npz")
    small = load_native_mesh(small_path)
    native = NativeVolumeSources.from_mesh(small)
    linear = LinearNativeVolumeSources.from_native(native, donor["native_centres"])
    c = ContinuousVelocityCurl.from_mesh(small, linear.centroid)
    for key, value in (("node_position", c.position), ("tetrahedron_nodes", c.tetrahedra), ("tetrahedron_parent", c.parent),
                       ("tetrahedron_volume", c.volume), ("tetrahedron_centroid", c.centroid)):
        np.testing.assert_array_equal(source[key], value)
    omega = c.curl(source["nodal_velocity"])
    np.testing.assert_array_equal(omega, source["tetrahedron_vorticity"])
    gamma, moment = c.curl_moments(omega)
    np.testing.assert_array_equal(gamma, source["cell_circulation"])
    np.testing.assert_array_equal(moment, source["first_moment"])
    compact, coefficient = c.active_source(omega)
    np.testing.assert_array_equal(compact.face_ids, source["compact_source_face_ids"])
    affine_coefficient, _ = linear.coefficients(gamma, moment)
    mesh = load_native_mesh(full_path)
    for patch in mesh["boundary"]:
        patch["velocity_type"] = "fixedValue"
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    sampler = InteriorFaceVelocitySampler(mesh, geometry, base["full_face_ids"], base["signs"])
    for name in ("sample_cells", "gradient_cells", "gradient_faces"):
        np.testing.assert_array_equal(getattr(sampler, name), base[name])
    normal, area = base["normal"], base["area"]
    side_names = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
    rng = np.random.default_rng(20260913)
    selected = np.concatenate([np.sort(rng.choice(np.flatnonzero(base["face_group"] == i), 4, replace=False)) for i in range(6)])
    curl_position = base["position"][base["target__boundary"]][selected]
    step = parent_report["particle_spacing"]*1e-3
    offsets = np.eye(3)[:, None]*np.array([1., -1., .5, -.5])[None, :, None]*step
    derivative_targets = curl_position[:, None, None]+offsets[None]
    active = np.any(omega != 0, axis=(0, 2))
    support = c.position[c.tetrahedra[active]]
    bounds = np.stack((support.min(axis=(0, 1)), support.max(axis=(0, 1))))
    distance = np.linalg.norm(np.maximum(bounds[0]-derivative_targets, 0)+np.maximum(derivative_targets-bounds[1], 0), axis=-1)
    assert distance.min() > 10*step
    targets, indices = unique_targets({"physical": base["position"], "curl": derivative_targets.reshape(-1, 3)})
    print(json.dumps({"stage": "geometry", "targets": len(targets), "physical_targets": len(base["position"]),
                      "continuous_induction_faces": len(compact.triangles), "affine_induction_faces": len(native.triangles)}), flush=True)

    def progress(stage):
        def update(done, total):
            if done % 512 == 0 or done == total:
                print(json.dumps({"stage": stage, "done": done, "total": total, "elapsed_seconds": time.perf_counter()-started}), flush=True)
        return update

    continuous = compact.evaluate(targets, coefficient, progress=progress("continuous_curl_induction")).transpose(1, 0, 2)
    np.savez_compressed(args.output / "continuous-induction-checkpoint.npz", position=targets, continuous_induction=continuous,
                        source_archive_sha256=hash_file(source_path)["sha256"])
    affine = linear.evaluate(targets, affine_coefficient, progress=progress("affine_moment_induction")).transpose(1, 0, 2)
    names = ["point_affine_moments", "point_continuous_curl", "cell_average_affine_moments", "cell_average_continuous_curl"]
    correction = np.stack((affine[0], continuous[0], affine[1], continuous[1]))
    np.savez_compressed(args.output / "continuous-curl-induction-checkpoint.npz", position=targets, correction_induction=correction,
                        physical_target_indices=indices["physical"], curl_target_indices=indices["curl"],
                        source_names=names, source_archive_sha256=hash_file(source_path)["sha256"])
    curl_values = correction[:, indices["curl"]].reshape(4, len(curl_position), 3, 4, 3)
    jacobian = (curl_values[:, :, :, 0]-curl_values[:, :, :, 1])/(2*step)
    jacobian_half = (curl_values[:, :, :, 2]-curl_values[:, :, :, 3])/step
    curl, curl_half = curl_affine(jacobian), curl_affine(jacobian_half)
    np.testing.assert_allclose(curl_half[[1, 3]], 0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(jacobian_half, jacobian, rtol=0, atol=1e-6)
    count = baseline_report["body_panels"]
    if ti.lang.impl.get_runtime().prog is None:
        ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1)
    panel = vpm.PanelSolver(max_n_panels=max(128, 1 << (count-1).bit_length()), float_dtype="f64", linear_solver="SCIPY",
                            boundary_condition_type="NEUMANN", density=1, freestream_velocity=np.array([1., 0., 0.]),
                            coupling_scope="vpm_boundary_condition", far_field_min_panels=count+1)
    panel.add_surface("cube", str(surface), reference_area=1)
    panel.initialize()
    lattice = panel.lattice
    assert lattice.n_panels == count
    for field, key in ((lattice.panel_centre, "panel_centre"), (lattice.normal, "panel_normal"), (lattice.area, "panel_area")):
        np.testing.assert_array_equal(field.to_numpy()[:count], base[key])
    matrix = panel.aerodynamic_influence_coefficient.to_numpy()[:count, :count]
    adapter = SimpleNamespace(particles=SimpleNamespace(n_particles_total=len(donor["gaussian_position"])))
    saved, records = {}, []
    for state, name in enumerate(names):
        family = state//2
        baseline_state = int(source["baseline_state_indices"][family])
        baseline_name = str(base["source_names"][baseline_state])
        baseline_observed = sampler.evaluate(base[baseline_name+"__velocity"][base["target__sample"]])
        np.testing.assert_array_equal(baseline_observed["tangential_gradient"], base[baseline_name+"__tangential_gradient"])
        delta = correction[state, indices["physical"]]
        panel.solve(np.zeros(3), delta[base["target__collocation"]], time=source_report["physical_time"])
        body = panel.compute_induced_velocity(base["position"])
        assert panel._last_far_field_fraction == 0
        strength = lattice.source_strength.to_numpy()[:count].copy()
        assert abs(base["panel_area"] @ strength) < 1e-10
        actual = base[baseline_name+"__velocity"]+delta+body
        observed = sampler.evaluate(actual[base["target__sample"]])
        normals, flux = {}, {}
        for mode, velocity in (("point", actual[base["target__boundary"]]), ("native", observed["face_velocity"])):
            corrected, flux[mode] = evaluate_vpm_velocity(adapter, base["position"][base["target__boundary"]], normal, area,
                freestream_velocity=np.array([1., 0., 0.]), fvm_box=np.asarray(parent_report["small_fvm_bounds"]),
                particle_spacing=parent_report["particle_spacing"], evaluated_velocity=velocity)
            normals[mode] = np.sum(corrected*normal, axis=1)
        gt_error = observed["tangential_gradient"]-base["reference_tangential_gradient"]
        wall_u = actual[base["target__wall"]]
        wall_un = np.sum(wall_u*base["wall_normal"], axis=1)
        collocation_residual = matrix @ strength+np.sum(delta[base["target__collocation"]]*base["panel_normal"], axis=1)
        row = {"name": name, "baseline": baseline_name, "baseline_state": baseline_state,
               "cell_velocity_rms_over_Uinf": {group: field_rms(actual[base["target__"+group]]-donor["full_velocity"][base[group+"__cell_ids"]],
                                                                donor["full_volume"][base[group+"__cell_ids"]])
                                               for group in ("near_body", "held_outer", "wake")},
               "native_gradient_error_rms": field_rms(gt_error, area),
               "point_normal_velocity_error_rms": field_rms((normals["point"]-base["reference_normal_velocity"])[:, None], area),
               "native_normal_velocity_error_rms": field_rms((normals["native"]-base["reference_normal_velocity"])[:, None], area),
               "wall_normal_velocity_rms_over_Uinf": field_rms(wall_un[:, None], None),
               "wall_tangential_velocity_rms_over_Uinf": field_rms(wall_u-wall_un[:, None]*base["wall_normal"], None),
               "body_source_flux_change": float(base["panel_area"] @ strength), "boundary_flux": flux,
               "correction_neumann_collocation_residual_rms": field_rms(collocation_residual[:, None], base["panel_area"]),
               "exterior_curl_rms": field_rms(curl_half[state], None),
               "exterior_curl_step_halving_difference_rms": field_rms(curl_half[state]-curl[state], None),
               "face_groups": {}}
        for number, side_name in enumerate(side_names):
            mask = base["face_group"] == number
            row["face_groups"][side_name] = {"native_gradient_error_rms": field_rms(gt_error[mask], area[mask]),
                "native_normal_velocity_error_rms": field_rms((normals["native"]-base["reference_normal_velocity"])[mask, None], area[mask])}
        records.append(row)
        saved.update({name+"__velocity": actual, name+"__body_velocity_change": body, name+"__body_strength_change": strength,
                      name+"__point_normal_velocity": normals["point"], name+"__native_normal_velocity": normals["native"],
                      name+"__collocation_residual_change": collocation_residual})
        saved.update({name+"__"+key: value for key, value in observed.items()})
        print(json.dumps(row), flush=True)
    np.savez_compressed(args.output / "continuous-curl-physical-fields.npz", position=base["position"], source_names=names,
                        curl_position=curl_position, curl_selected_face_rows=selected, curl_side=base["face_group"][selected],
                        curl_step=step, curl_values=curl_values, curl=curl, curl_half=curl_half,
                        curl_jacobian=jacobian, curl_jacobian_half=jacobian_half, support_bounds=bounds, **saved)
    result = {"schema": "openonda-cube-continuous-curl-induction-3d/1", "status": "complete", "spatial_dimensions": 3,
              "source_names": names, "source_sgs": source_report["source_sgs"], "physical_time": source_report["physical_time"],
              "small_fvm_cells": small["n_cells"], "body_panels": count, "boundary_faces": len(area),
              "physical_targets": len(base["position"]), "induction_targets": len(targets),
              "minimum_curl_target_distance_from_source_bounding_box": float(distance.min()),
              "records": records, "sources": sources, "elapsed_seconds": time.perf_counter()-started,
              "limitations": [
                  "Each pair has identical raw cell circulation and first moment; only the source representation changes.",
                  "The reconstructed continuous velocity is not incompressible; its exact curl is solenoidal. Biot--Savart induces the solenoidal velocity projection.",
                  "The body response is recomputed for each correction. The earlier baseline and all observation positions remain identical.",
                  "Exterior curl norms use 24 points and two centred finite-difference steps; they are not whole-boundary error norms.",
                  "The reconstructed common-node trace differs from the earlier face-wise quadratic update. Comparisons against that older update are not representation-only comparisons.",
                  "Frozen source/body comparisons do not establish advancing particle emission, force histories or velocity-profile agreement."]}
    (args.output / "cube-continuous-curl-induction-3d.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.source, args.baseline, args.output = args.source.resolve(), args.baseline.resolve(), args.output.resolve()
    run(args)
