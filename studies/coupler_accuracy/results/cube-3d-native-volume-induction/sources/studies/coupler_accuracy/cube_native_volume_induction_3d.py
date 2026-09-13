#!/usr/bin/env python3
"""Frozen fully 3D cube: native polyhedral volumes versus omega*V Gaussians."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

import openonda.fvm as fvm
from source.coupler.boundary import evaluate_vpm_velocity
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from studies.coupler_accuracy.cube_boundary_oracle import CASE, ROOT, field_rms, hash_file, setup_for
from studies.coupler_accuracy.cube_panel_resolution_3d import fixed_wall_samples
from studies.coupler_accuracy.cube_snapshot_induction import volume_velocity
from studies.coupler_accuracy.joint_reconstruction_3d import CubePanelResponse
from studies.coupler_accuracy.native_cell_integrals_3d import NativeCellIntegration
from studies.coupler_accuracy.native_velocity_curl_integrals_3d import triangle_rule
from studies.coupler_accuracy.native_volume_induction_3d import (
    NativeVolumeSources,
    piecewise_constant_boundary_completion,
    triangle_source_integrals,
)


def qualify_native_cells(mesh, geometry, ids):
    integration = NativeCellIntegration.from_mesh(mesh, geometry, ids)
    errors, refinements = [], []
    for row, cell in enumerate(ids):
        triangles = integration.tetrahedra[row][:, 1:]
        target = integration.centre[row]
        relative = triangles-target
        vector_area = np.cross(relative[:, 1]-relative[:, 0], relative[:, 2]-relative[:, 0])/2
        normal = vector_area/np.linalg.norm(vector_area, axis=1)[:, None]
        p = triangle_source_integrals(target[None], triangles)[0][0]
        analytical = np.sum(p[:, None]*normal, axis=0)
        determinant = np.einsum("ti,ti->t", relative[:, 0], np.cross(relative[:, 1], relative[:, 2]))
        answers = []
        for order in (16, 20):
            barycentric, weight = triangle_rule(order)
            q = np.einsum("qa,tad->tqd", barycentric, relative)
            answers.append(-np.einsum("t,q,tqd->d", determinant/2, weight,
                                      q/np.linalg.norm(q, axis=2)[:, :, None]**3))
        scale = np.cbrt(geometry["cell_volume"][cell])
        errors.append(float(np.max(np.abs(analytical-answers[-1]))/scale))
        refinements.append(float(np.max(np.abs(answers[0]-answers[1]))/scale))
    assert max(errors) < 1e-9
    assert max(refinements) < 1e-9
    return {"cell_ids": np.asarray(ids).tolist(), "volume_angular_orders": [16, 20],
            "maximum_analytical_vs_radial_volume_difference_over_cell_length": max(errors),
            "maximum_volume_refinement_difference_over_cell_length": max(refinements),
            "per_cell_analytical_difference_over_cell_length": errors}


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source_paths = [Path(__file__), CASE / "assets/cube.stl"]
    source_paths += [ROOT / p for p in (
        "studies/coupler_accuracy/native_volume_induction_3d.py",
        "studies/coupler_accuracy/native_cell_integrals_3d.py",
        "studies/coupler_accuracy/native_velocity_curl_integrals_3d.py",
        "studies/coupler_accuracy/cube_boundary_oracle.py",
        "studies/coupler_accuracy/cube_panel_resolution_3d.py",
        "studies/coupler_accuracy/cube_snapshot_induction.py",
        "studies/coupler_accuracy/joint_reconstruction_3d.py",
        "source/coupler/boundary.py", "source/coupler/renewal_projection.py",
        "source/solvers/fvm/core/solver.py", "source/solvers/fvm/fields/gradients.py",
        "source/solvers/fvm/fields/diagnostics.py", "source/solvers/fvm/coupling/coupler_interface.py",
        "source/solvers/fvm/mesh/geometry.py",
        "source/solvers/vpm/boundary_elements/panels/kernels/source_potential.py",
        "source/solvers/vpm/boundary_elements/panels/kernels/source_velocity.py",
        "source/solvers/vpm/boundary_elements/panels/kernels/induced_velocity.py",
        "source/solvers/vpm/boundary_elements/panels/solver/panel_solver.py",
        "source/solvers/vpm/boundary_elements/panels/solver/linear_solvers.py",
        "tests/coupler/test_native_volume_induction_3d.py", "tests/vpm/test_source_panel_potential.py")]
    source_paths += [args.oracle / p for p in ("cube-boundary-oracle.json", "full-native-mesh.npz",
                                               "small-native-mesh.npz", "initial-cell-fields.npz")]
    source_paths += [args.study / p for p in ("held-fields.npz", "native-reconstruction-3d.json")]
    source_paths += [args.boundary / p for p in ("boundary-fields.npz", "native-curl-3d.json")]
    sources = [hash_file(p) for p in source_paths]
    for path in source_paths:
        if path.suffix == ".py":
            destination = args.output / "sources" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
    (args.output / "sources-at-start.json").write_text(json.dumps(sources, indent=2)+"\n")
    metadata = json.loads((args.oracle / "cube-boundary-oracle.json").read_text())
    if metadata.get("spatial_dimensions") != 3 or not metadata.get("valid_for_comparison"):
        raise ValueError("A qualified fully 3D native oracle is required")
    with np.load(args.oracle / "initial-cell-fields.npz", allow_pickle=False) as data:
        centres, velocity, pressure = (data[k].copy() for k in ("centres", "velocity", "pressure"))
        physical_time = float(data["physical_time"])
    fit_metadata = json.loads((args.study / "native-reconstruction-3d.json").read_text())
    assert physical_time == fit_metadata["physical_time"]
    with np.load(args.study / "held-fields.npz", allow_pickle=False) as data:
        held = {k: data[k].copy() for k in data.files}
    with np.load(args.boundary / "boundary-fields.npz", allow_pickle=False) as data:
        boundary = {k: data[k].copy() for k in data.files}
    mesh = load_native_mesh(args.oracle / "full-native-mesh.npz")
    small_mesh = load_native_mesh(args.oracle / "small-native-mesh.npz")
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    np.testing.assert_allclose(centres, geometry["cell_centre"], rtol=0, atol=1e-13)
    volumes = geometry["cell_volume"]
    face_velocity = np.zeros((mesh["n_faces"], 3))
    with fvm.create_fvm_solver(setup_for(mesh, "native-volume-donor", 0.01, 1,
                                        turbulence=metadata["numerics"]["sgs"] != "none"),
                               mesh=copy.deepcopy(mesh), case_dir=args.output / "donor") as solver:
        solver.set_initial_state(velocity, pressure)
        omega = solver.get_vorticity_field().copy()
        divergence = np.trace(solver.get_velocity_gradient_field(), axis1=1, axis2=2).copy()
        for patch in mesh["boundary"]:
            first, count = patch["start_face"], patch["n_faces"]
            face_velocity[first:first+count] = solver.get_boundary_face_velocity(patch["name"])
    distance, held_ids = cKDTree(centres).query(held["position"])
    np.testing.assert_allclose(distance, 0, rtol=0, atol=1e-13)
    np.testing.assert_array_equal(velocity[held_ids], held["fvm_velocity"])
    np.testing.assert_allclose(omega[held_ids], held["fvm_vorticity"], rtol=0, atol=2e-14)
    native = NativeVolumeSources.from_mesh(mesh)
    poly_volume, poly_centroid = native.cell_geometry(centres)
    volume_difference = (poly_volume-volumes)/volumes
    ids = np.unique(np.r_[np.argmax(np.abs(volume_difference)), held_ids[::64],
                          np.random.default_rng(1285).choice(len(centres), 8, replace=False)])
    qualification = qualify_native_cells(mesh, geometry, ids)
    (args.output / "kernel-qualification.json").write_text(json.dumps(qualification, indent=2)+"\n")
    body = CubePanelResponse()
    points, selections, selected_cells, count = [], {}, {}, 0

    def add(name, values):
        nonlocal count
        selections[name] = slice(count, count+len(values))
        count += len(values)
        points.append(values)

    generator = np.random.default_rng(20260913)
    near_ids = np.flatnonzero(np.max(np.abs(centres), axis=1) < 0.8)
    wake_ids = np.flatnonzero((centres[:, 0] > 1.5) & (centres[:, 0] < 5)
                              & (np.max(np.abs(centres[:, 1:]), axis=1) < 1.5))
    for name, ids in (("near_body", near_ids), ("held_outer", held_ids), ("wake", wake_ids)):
        if name != "held_outer":
            ids = np.sort(generator.choice(ids, min(args.samples_per_region, len(ids)), replace=False))
        selected_cells[name] = ids
        add(name, centres[ids])
    wall, wall_normal = fixed_wall_samples(1e-6)
    add("wall", wall)
    add("boundary", boundary["position"])
    epsilon = fit_metadata["particle_spacing"]*1e-4
    add("plus", boundary["position"]+epsilon*boundary["normal"])
    add("minus", boundary["position"]-epsilon*boundary["normal"])
    add("check_plus", boundary["position"][:32]+2*epsilon*boundary["normal"][:32])
    add("check_minus", boundary["position"][:32]-2*epsilon*boundary["normal"][:32])
    add("collocation", body.centres)
    targets = np.vstack(points)
    ratio = volumes/poly_volume
    input_omega = np.stack((omega, omega*ratio[:, None], np.zeros_like(omega), np.zeros_like(omega)))
    input_divergence = np.stack((np.zeros_like(divergence), np.zeros_like(divergence), divergence, divergence*ratio))
    coefficients = native.coefficients(input_omega, input_divergence)

    def progress(done, total):
        if done % 128 == 0 or done == total:
            print(json.dumps({"stage": "native_volume", "targets_done": done, "targets_total": total,
                              "elapsed_seconds": time.perf_counter()-started}), flush=True)

    print(json.dumps({"stage": "native_start", "source_cells": len(centres), "source_triangles": len(native.triangles),
                      "targets": len(targets), "qualification": qualification}), flush=True)
    native_velocity = native.evaluate(targets, coefficients, progress=progress)
    np.savez_compressed(args.output / "native-induction-fields.npz", position=targets, velocity=native_velocity,
                        cell_vorticity=omega, cell_divergence=divergence, polyhedron_volume=poly_volume,
                        polyhedron_centroid=poly_centroid)
    outer = np.zeros(mesh["n_faces"], dtype=bool)
    for patch in mesh["boundary"]:
        if patch["name"] != "cube":
            outer[patch["start_face"]:patch["start_face"]+patch["n_faces"]] = True
    rows = outer[native.face_ids]
    completion = piecewise_constant_boundary_completion(
        targets, native.triangles[rows], face_velocity[native.face_ids[rows]]-[1, 0, 0])
    representations = [("native_constant_omega", native_velocity[:, 0], native_velocity[:, 2]),
                       ("native_preserved_circulation", native_velocity[:, 1], native_velocity[:, 3])]
    for name, radius, prune in (("gaussian_sigma_0.0625", 0.0625, False),
                                ("gaussian_sigma_0.125", 0.125, False),
                                ("gaussian_sigma_0.25", 0.25, False),
                                ("gaussian_seed_threshold_0.02", 0.125, True)):
        source_position = centres
        strength = omega*volumes[:, None]
        divergence_strength = divergence*volumes
        if prune:
            keep = np.linalg.norm(omega, axis=1) >= 0.02
            source_position = centres[keep].astype(np.float32).astype(float)
            strength = strength[keep].astype(np.float32).astype(float)
            divergence_strength = divergence_strength[keep]
        induced, potential = np.zeros_like(targets), np.zeros_like(targets)
        for start in range(0, len(targets), 64):
            induced[start:start+64], potential[start:start+64] = volume_velocity(
                targets[start:start+64], source_position, strength, radius, divergence_strength)
        representations.append((name, induced, potential))
        print(json.dumps({"stage": name, "sources": len(source_position),
                          "elapsed_seconds": time.perf_counter()-started}), flush=True)
    bounds = np.column_stack((small_mesh["vertex_position"].min(axis=0), small_mesh["vertex_position"].max(axis=0))).ravel()
    adapter = SimpleNamespace(particles=SimpleNamespace(n_particles_total=len(centres)))
    legacy_length = np.linalg.norm(boundary["normal"], axis=1)
    records, output_fields = [], {}
    for name, induced, potential in representations:
        body.panel.solve(np.array([1., 0., 0.]), induced[selections["collocation"]], time=physical_time)
        panel_velocity = body.panel.compute_induced_velocity(targets)
        flux = float(body.area @ body.panel.lattice.source_strength.to_numpy()[:body.count])
        assert abs(flux) < 1e-10
        variants = {"freestream": induced+[1, 0, 0], "outer_completion": induced+[1, 0, 0]+completion,
                    "outer_and_native_divergence": induced+[1, 0, 0]+completion+potential,
                    "panel108": induced+[1, 0, 0]+panel_velocity}
        for variant, actual in variants.items():
            cut, mass = evaluate_vpm_velocity(adapter, boundary["position"], boundary["normal"], boundary["area"],
                                              freestream_velocity=np.array([1., 0., 0.]), fvm_box=bounds,
                                              particle_spacing=0.125, evaluated_velocity=actual[selections["boundary"]])
            un = np.sum(cut*boundary["normal"], axis=1)
            derivative = (actual[selections["plus"]]-actual[selections["minus"]])/(2*epsilon)
            check = (actual[selections["check_plus"]]-actual[selections["check_minus"]])/(4*epsilon)
            normal_derivative = derivative/legacy_length[:, None]
            tangent = normal_derivative-np.sum(normal_derivative*boundary["native_unit_normal"], axis=1)[:, None]*boundary["native_unit_normal"]
            wall_velocity = actual[selections["wall"]]
            wall_un = np.sum(wall_velocity*wall_normal, axis=1)
            record = {"representation": name, "completion": variant,
                      "cell_velocity_rms_over_Uinf": {group: field_rms(actual[selections[group]]-velocity[ids], volumes[ids])
                                                     for group, ids in selected_cells.items()},
                      "wall_normal_velocity_rms_over_Uinf": float(np.sqrt(np.mean(wall_un**2))),
                      "wall_tangential_velocity_rms_over_Uinf": float(np.sqrt(np.mean(np.sum(
                          (wall_velocity-wall_un[:, None]*wall_normal)**2, axis=1)))),
                      "boundary_normal_velocity_rms_error_over_Uinf": field_rms((un-boundary["fvm_normal_velocity"])[:, None], boundary["area"]),
                      "boundary_native_tangential_gradient_rms_error": field_rms(tangent-boundary["fvm_native_flux_tangential_normal_gradient"], boundary["native_vector_area"]),
                      "derivative_step_check_maximum_difference": float(np.max(np.abs(derivative[:32]-check))),
                      "boundary_flux": mass, "panel_source_flux": flux if variant == "panel108" else None}
            key = name+"__"+variant
            output_fields[key+"__velocity"] = actual
            output_fields[key+"__boundary_normal_velocity"] = un
            output_fields[key+"__boundary_native_tangential_gradient"] = tangent
            if name == "gaussian_seed_threshold_0.02" and variant == "panel108":
                np.testing.assert_allclose(actual[selections["held_outer"]], held["donor_volume_vorticity__velocity"], rtol=0, atol=1e-12)
                np.testing.assert_allclose(un, boundary["donor_volume_vorticity__normal_velocity"], rtol=0, atol=1e-12)
            records.append(record)
            print(json.dumps(record), flush=True)
    np.savez_compressed(args.output / "volume-induction-comparison-fields.npz", position=targets,
                        native_cell_centres=centres, native_cell_volume=volumes, native_cell_velocity=velocity,
                        outer_completion=completion, wall_normal=wall_normal,
                        **{name+"__cell_ids": ids for name, ids in selected_cells.items()}, **output_fields)
    report = {"schema": "openonda-cube-native-volume-induction-3d/1", "status": "complete", "spatial_dimensions": 3,
              "physical_time": physical_time, "source_cells": len(centres), "native_triangles": len(native.triangles),
              "targets": len(targets), "target_slices": {k: [v.start, v.stop] for k, v in selections.items()},
              "cell_target_counts": {k: len(v) for k, v in selected_cells.items()}, "body_panels": body.count,
              "native_divergence_rms": field_rms(divergence[:, None], volumes),
              "geometry": {"maximum_relative_fan_vs_fvm_volume_difference": float(np.max(np.abs(volume_difference))),
                           "rms_relative_fan_vs_fvm_volume_difference": float(np.sqrt(np.mean(volume_difference**2))),
                           "maximum_fan_vs_fvm_centroid_distance": float(np.max(np.linalg.norm(poly_centroid-centres, axis=1)))},
              "kernel_qualification": qualification, "derivative_step": epsilon,
              "records": records, "elapsed_seconds": time.perf_counter()-started, "sources": sources,
              "limitations": ["Frozen initial full FVM field at the same time as the preceding fits; no time integration or force comparison.",
                              "All full native cells are included, except the explicitly named old thresholded Gaussian seed control.",
                              "Native constant vorticity is reconstructed from the FVM discrete curl, not known continuum vorticity.",
                              "Native preserved circulation uses omega*V_FVM/V_polyhedron so its cell integral equals the original omega*V_FVM.",
                              "Boundary completion uses piecewise constant native boundary-face velocities; no slip makes the body surface term zero.",
                              "Divergence completion is a diagnostic of FVM kinematics, not an incompressible VPM proposal.",
                              "Velocity derivatives of piecewise constant volume sources can have jumps; centred derivatives and step checks are reported, not assumed smooth.",
                              "Body source panels impose normal velocity only; panel and full-boundary completions are separate experiments."]}
    (args.output / "cube-native-volume-induction-3d.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"status": "complete", "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    results = ROOT / "studies/coupler_accuracy/results"
    parser.add_argument("--oracle", type=Path, default=results / "cube-3d-oracle")
    parser.add_argument("--study", type=Path, default=results / "cube-3d-integrated-velocity-curl-reconstruction")
    parser.add_argument("--boundary", type=Path, default=results / "cube-3d-integrated-velocity-curl-reconstruction-boundary")
    parser.add_argument("--samples-per-region", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
