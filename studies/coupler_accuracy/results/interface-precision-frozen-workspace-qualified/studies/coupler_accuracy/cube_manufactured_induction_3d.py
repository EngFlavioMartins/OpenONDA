#!/usr/bin/env python3
"""Separate analytic source integration, face quadrature and native FVM curl."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import qmc

import openonda.fvm as fvm
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, field_rms, hash_file, setup_for
from studies.coupler_accuracy.cube_panel_resolution_3d import (
    fixed_wall_samples,
    particle_velocities,
)
from studies.coupler_accuracy.manufactured_cube_field_3d import (
    NoSlipCubeField,
    accumulate_faces,
    native_manufactured_integrals,
)
from studies.coupler_accuracy.native_cell_integrals_3d import NativeCellIntegration
from studies.coupler_accuracy.native_volume_induction_3d import NativeVolumeSources


def fixed_fluid_samples():
    generator = qmc.Sobol(3, scramble=True, seed=8413)
    near = (generator.random_base2(11)-0.5)*1.6
    near = near[np.max(np.abs(near), axis=1) > 0.5][:256]
    generator = qmc.Sobol(3, scramble=True, seed=6172)
    outer = (generator.random_base2(11)-0.5)*3
    outer = outer[np.max(np.abs(outer), axis=1) > 1.1][:256]
    assert len(near) == len(outer) == 256
    return near, outer


def volume_integral_checks(mesh, geometry, fields, integral):
    centres = geometry["cell_centre"]
    radius = np.max(np.abs(centres), axis=1)
    relevant = np.flatnonzero((radius > 0.5) & (radius < 0.9))
    strongest = np.max(np.linalg.norm(integral["vorticity_integral"], axis=2), axis=0)/geometry["cell_volume"]
    ids = np.unique(np.r_[np.argmax(strongest), np.random.default_rng(7473).choice(relevant, 8, replace=False)])
    integration = NativeCellIntegration.from_mesh(mesh, geometry, ids)
    differences, refinements = [], []
    for row, cell in enumerate(ids):
        answers = []
        for order in (12, 16):
            point, weight = integration.rule(row, order)
            answers.append(np.array([[weight @ field.velocity(point), weight @ field.vorticity(point)] for field in fields]))
        surface = np.stack((integral["velocity_integral"][:, cell], integral["vorticity_integral"][:, cell]), axis=1)
        differences.append(float(np.max(np.abs(surface-answers[-1]))/geometry["cell_volume"][cell]))
        refinements.append(float(np.max(np.abs(answers[-1]-answers[0]))/geometry["cell_volume"][cell]))
    assert max(differences) < 1e-8 and max(refinements) < 1e-8
    return {"cell_ids": ids.tolist(), "volume_orders": [12, 16],
            "maximum_surface_vs_volume_difference_over_fvm_volume": max(differences),
            "maximum_volume_refinement_difference_over_fvm_volume": max(refinements)}


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source_paths = [Path(__file__), args.oracle / "full-native-mesh.npz", args.oracle / "cube-boundary-oracle.json"]
    source_paths += [ROOT / p for p in (
        "studies/coupler_accuracy/manufactured_cube_field_3d.py", "studies/coupler_accuracy/native_volume_induction_3d.py",
        "studies/coupler_accuracy/native_cell_integrals_3d.py", "studies/coupler_accuracy/native_velocity_curl_integrals_3d.py",
        "studies/coupler_accuracy/cube_boundary_oracle.py", "studies/coupler_accuracy/cube_panel_resolution_3d.py",
        "source/coupler/renewal_projection.py", "source/solvers/fvm/fields/gradients.py",
        "source/solvers/fvm/fields/diagnostics.py", "source/solvers/fvm/mesh/geometry.py",
        "source/solvers/fvm/core/solver.py", "source/solvers/fvm/coupling/coupler_interface.py",
        "tests/coupler/test_manufactured_cube_field_3d.py")]
    sources = [hash_file(p) for p in source_paths]
    for path in source_paths:
        if path.suffix == ".py":
            destination = args.output / "sources" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
    (args.output / "sources-at-start.json").write_text(json.dumps(sources, indent=2)+"\n")
    metadata = json.loads((args.oracle / "cube-boundary-oracle.json").read_text())
    if metadata.get("spatial_dimensions") != 3 or not metadata.get("valid_for_comparison"):
        raise ValueError("A qualified fully 3D cube mesh is required")
    fields = [NoSlipCubeField(width) for width in args.widths]
    mesh = load_native_mesh(args.oracle / "full-native-mesh.npz")
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    centres, volumes = geometry["cell_centre"], geometry["cell_volume"]
    native = NativeVolumeSources.from_mesh(mesh)
    poly_volume, poly_centroid = native.cell_geometry(centres)
    history, previous, accepted = [], None, None

    def surface_progress(done, total, order):
        print(json.dumps({"stage": "surface_integrals", "order": order, "triangles_done": done,
                          "triangles_total": total, "elapsed_seconds": time.perf_counter()-started}), flush=True)

    for order in (6, 8, 10, 12, 14):
        current = native_manufactured_integrals(native, mesh, fields, order=order, progress=surface_progress)
        difference = None
        if previous is not None:
            difference = max(float(np.max(np.abs(current[key]-previous[key])/volumes[None, :, None]))
                             for key in ("velocity_integral", "vorticity_integral"))
        history.append({"surface_order": order, "maximum_successive_cell_integral_difference_over_fvm_volume": difference})
        print(json.dumps({"stage": "surface_refinement", **history[-1]}), flush=True)
        if difference is not None and difference < 1e-9:
            accepted = current
            break
        previous = current
    if accepted is None:
        raise RuntimeError("Manufactured cell integrals did not converge")
    qualification = volume_integral_checks(mesh, geometry, fields, accepted)
    print(json.dumps({"stage": "volume_qualified", **qualification}), flush=True)
    names, strengths, source_records = [], [], []
    point_velocity = np.stack([field.velocity(centres) for field in fields])
    point_omega = np.stack([field.vorticity(centres) for field in fields])
    n_interior = mesh["n_interior_faces"]
    wall = next(p for p in mesh["boundary"] if p["name"] == "cube")
    wall_faces = np.arange(wall["start_face"], wall["start_face"]+wall["n_faces"])
    cell_radius = np.max(np.abs(centres), axis=1)
    groups = {"all": np.ones(len(centres), dtype=bool), "near_body": cell_radius < 0.8,
              "outer_layer": (cell_radius > 1.1) & (cell_radius < 1.5)}
    native_divergence = []
    for index, field in enumerate(fields):
        face_u = field.velocity(geometry["face_centre"])
        np.testing.assert_allclose(face_u[wall_faces], 0, rtol=0, atol=1e-12)
        face_u[wall_faces] = 0
        true_gamma = accepted["vorticity_integral"][index]
        gamma_midpoint = accumulate_faces(mesh, np.cross(geometry["face_area_vector"], face_u))
        face_average = accepted["face_velocity_integral"][index]/geometry["face_area"][:, None]
        gamma_face_average = accumulate_faces(mesh, np.cross(geometry["face_area_vector"], face_average))
        cases = [("exact_cell_circulation", true_gamma),
                 ("point_vorticity_times_fvm_volume", point_omega[index]*volumes[:, None]),
                 ("exact_face_centre_velocity", gamma_midpoint),
                 ("exact_face_average_velocity", gamma_face_average)]
        config = setup_for(mesh, "manufactured-curl", 0.01, 1, turbulence=False)
        with fvm.create_fvm_solver(config, mesh=copy.deepcopy(mesh), case_dir=args.output / f"curl-width-{field.width}") as solver:
            for patch in mesh["boundary"]:
                first, count = patch["start_face"], patch["n_faces"]
                solver.set_dirichlet_velocity_boundary_condition_vec(face_u[first:first+count], patch["name"])
            for name, velocity in (("native_curl_of_point_velocity", point_velocity[index]),
                                   ("native_curl_of_cell_average_velocity", accepted["velocity_integral"][index]/poly_volume[:, None])):
                solver.set_initial_state(velocity, np.zeros(len(centres)))
                omega = solver.get_vorticity_field().copy()
                gradient = solver.get_velocity_gradient_field().copy()
                interpolated = face_u.copy()
                w = geometry["face_interpolation_weight"][:n_interior, None]
                interpolated[:n_interior] = ((1-w)*velocity[mesh["owners"][:n_interior]]+w*velocity[mesh["neighbours"]])
                replay_gamma = accumulate_faces(mesh, np.cross(geometry["face_area_vector"], interpolated))
                replay_error = float(np.max(np.abs(replay_gamma/volumes[:, None]-omega)))
                assert replay_error < 1e-10
                cases.append((name, omega*volumes[:, None]))
                native_divergence.append({"width": field.width, "source": name, "curl_replay_maximum_difference": replay_error,
                                          "divergence_rms": field_rms(np.trace(gradient, axis1=1, axis2=2)[:, None], volumes)})
        for name, gamma in cases:
            key = f"width_{field.width}__{name}"
            names.append(key)
            strengths.append(gamma)
            record = {"name": key, "width": field.width, "source": name, "circulation_error": {},
                      "total_gamma": gamma.sum(axis=0).tolist(), "l1_gamma": float(np.linalg.norm(gamma, axis=1).sum())}
            for group, selected in groups.items():
                difference = (gamma-true_gamma)[selected]/poly_volume[selected, None]
                reference = true_gamma[selected]/poly_volume[selected, None]
                rms = field_rms(difference, poly_volume[selected])
                denominator = field_rms(reference, poly_volume[selected])
                record["circulation_error"][group] = {"density_rms_error": rms, "relative_to_exact_cell_circulation": rms/denominator}
            source_records.append(record)
    strengths = np.stack(strengths)
    near, outer = fixed_fluid_samples()
    wall_targets, wall_normal = fixed_wall_samples(0.)
    target = np.vstack((near, outer, wall_targets))
    selections = {"near_body": slice(0, len(near)), "outer_layer": slice(len(near), len(near)+len(outer)),
                  "wall": slice(len(near)+len(outer), len(target))}
    exact_velocity = np.stack([field.velocity(target) for field in fields])
    np.testing.assert_allclose(exact_velocity[:, selections["wall"]], 0, rtol=0, atol=1e-12)
    bounding = np.array([-5, 10, -5, 5, -5, 5])
    minimum_outer_distance = np.min(np.minimum(target-bounding[::2], bounding[1::2]-target))
    extent = bounding[1::2]-bounding[::2]
    outer_area = 2*(extent[0]*extent[1]+extent[1]*extent[2]+extent[2]*extent[0])
    completion_bounds = [2*field.outer_box_velocity_bound(bounding)*outer_area/(4*np.pi*minimum_outer_distance**2) for field in fields]
    assert max(completion_bounds) < 1e-25
    np.savez_compressed(args.output / "manufactured-source-fields.npz", centres=centres, fvm_volume=volumes,
                        polyhedron_volume=poly_volume, polyhedron_centroid=poly_centroid, source_names=np.asarray(names),
                        gamma=strengths, point_velocity=point_velocity, point_vorticity=point_omega,
                        **accepted)

    def induction_progress(done, total):
        if done % 128 == 0 or done == total:
            print(json.dumps({"stage": "native_induction", "done": done, "total": total,
                              "elapsed_seconds": time.perf_counter()-started}), flush=True)

    native_velocity = native.evaluate(target, native.coefficients(strengths/poly_volume[None, :, None]),
                                      progress=induction_progress).transpose(1, 0, 2)
    np.savez_compressed(args.output / "native-velocity-checkpoint.npz", position=target, velocity=native_velocity)
    representations = [("native_volume", native_velocity)]
    for kernel, sigma in (("gaussian_nominal_spacing", np.full(len(centres), args.particle_spacing)),
                           ("gaussian_cell_volume_scale", np.cbrt(volumes))):
        values = particle_velocities(target, centres, sigma, strengths)
        representations.append((kernel, values))
        print(json.dumps({"stage": kernel, "elapsed_seconds": time.perf_counter()-started}), flush=True)
    records, output_fields = [], {}
    for kernel, velocity in representations:
        output_fields[kernel+"__velocity"] = velocity
        for state, name in enumerate(names):
            field_index = state//6
            record = {"name": name, "kernel": kernel, "width": fields[field_index].width,
                      "source": source_records[state]["source"], "velocity_errors": {}}
            for group, rows in selections.items():
                error = velocity[state, rows]-exact_velocity[field_index, rows]
                rms = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))
                expected_rms = float(np.sqrt(np.mean(np.sum(exact_velocity[field_index, rows]**2, axis=1))))
                record["velocity_errors"][group] = {"rms_over_reference_speed": rms, "exact_velocity_rms": expected_rms,
                                                   "maximum_over_reference_speed": float(np.max(np.linalg.norm(error, axis=1)))}
            records.append(record)
            print(json.dumps(record), flush=True)
    np.savez_compressed(args.output / "manufactured-induction-fields.npz", position=target,
                        exact_velocity=exact_velocity, wall_normal=wall_normal, **output_fields)
    report = {"schema": "openonda-cube-manufactured-induction-3d/1", "status": "complete", "spatial_dimensions": 3,
              "source_cells": len(centres), "source_triangles": len(native.triangles), "particle_spacing": args.particle_spacing,
              "manufactured_fields": [{"width": f.width, "amplitude": f.amplitude, "direction": f.direction.tolist(),
                                        "unit_speed_reference_position": f.reference_position.tolist()} for f in fields],
              "outer_boundary_completion_velocity_bounds": completion_bounds,
              "quadrature_history": history, "volume_qualification": qualification,
              "native_curl_replays": native_divergence,
              "target_slices": {key: [value.start, value.stop] for key, value in selections.items()},
              "source_names": names, "source_records": source_records, "records": records, "sources": sources,
              "elapsed_seconds": time.perf_counter()-started,
              "limitations": ["Kinematic manufactured verification, not the physical Re=1000 cube or an advancing hybrid run.",
                              "The same off-grid samples and wall points are used across mesh resolutions. Exact velocity is available at every target.",
                              "The analytic vector potential gives zero velocity on the cube and vanishes at infinity. No body-panel correction or freestream is added.",
                              "The omitted finite-box Helmholtz boundary contribution is bounded by the reported product-factor estimate.",
                              "Exact cell circulation comes from actual face-fan Stokes integrals, independently checked by volume integration.",
                              "Native FVM curl is measured both from point-centre and true cell-average velocity with exact Dirichlet wall/outer data.",
                              "All native cells are included without particle pruning, fitting, f32 rounding or source thresholds.",
                              "Gaussian nominal spacing and local cube-root-volume radii are distinct source representations; neither is assumed identical to polyhedral volumes."]}
    (args.output / "cube-manufactured-induction-3d.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--particle-spacing", type=float, required=True)
    parser.add_argument("--widths", nargs="+", type=float, default=[0.35, 0.15])
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
