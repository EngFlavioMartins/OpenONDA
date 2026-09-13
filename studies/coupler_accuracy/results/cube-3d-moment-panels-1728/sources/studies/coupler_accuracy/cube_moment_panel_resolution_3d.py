#!/usr/bin/env python3
"""Refine only the body response of the frozen reconstructed-moment overlap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree
import taichi as ti

import openonda.vpm as vpm
from source.coupler.boundary import evaluate_vpm_velocity
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.vpm.boundary_elements.panels.geometry.stl_io import load_stl, save_stl
from studies.coupler_accuracy.cube_boundary_oracle import CASE, ROOT, field_rms, hash_file
from studies.coupler_accuracy.cube_panel_resolution_3d import (
    particle_velocities,
    subdivide_triangles,
)
from studies.coupler_accuracy.induction_derivatives_3d import normal_derivative_estimates
from studies.coupler_accuracy.native_linear_volume_induction_3d import LinearNativeVolumeSources
from studies.coupler_accuracy.native_volume_induction_3d import NativeVolumeSources


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def run(args):
    if args.subdivisions not in (0, 1, 2, 3):
        raise ValueError("Use zero to three body triangle subdivisions")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    parent = json.loads((args.moments / "cube-reconstructed-moment-overlap-3d.json").read_text())
    old = json.loads((args.overlap / "cube-volume-overlap-3d.json").read_text())
    assert parent["status"] == old["status"] == "complete"
    assert parent["spatial_dimensions"] == old["spatial_dimensions"] == 3
    paths = [Path(__file__), CASE / "assets/cube.stl", args.oracle / "small-native-mesh.npz",
             args.boundary / "boundary-fields.npz", args.full / "volume-induction-comparison-fields.npz"]
    paths += [args.moments / key for key in ("cube-reconstructed-moment-overlap-3d.json", "moment-overlap-induction-checkpoint.npz", "reconstructed-moment-overlap-fields.npz")]
    paths += [args.overlap / key for key in ("cube-volume-overlap-3d.json", "induction-checkpoint.npz")]
    paths += [ROOT / key for key in ("studies/coupler_accuracy/native_linear_volume_induction_3d.py",
                                     "studies/coupler_accuracy/native_volume_induction_3d.py", "studies/coupler_accuracy/induction_derivatives_3d.py",
                                     "studies/coupler_accuracy/cube_panel_resolution_3d.py", "source/coupler/boundary.py", "source/coupler/renewal_projection.py",
                                     "source/solvers/fvm/mesh/geometry.py", "source/solvers/vpm/boundary_elements/panels/geometry/stl_io.py",
                                     "source/solvers/vpm/boundary_elements/panels/solver/panel_solver.py", "source/solvers/vpm/boundary_elements/panels/solver/linear_solvers.py",
                                     "source/solvers/vpm/boundary_elements/panels/kernels/source_velocity.py", "source/solvers/vpm/boundary_elements/panels/kernels/induced_velocity.py")]
    sources = [hash_file(path) for path in paths]
    for path in paths:
        if path.suffix == ".py":
            destination = args.output / "sources" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
    (args.output / "sources-at-start.json").write_text(json.dumps(sources, indent=2)+"\n")
    moments = arrays(args.moments / "moment-overlap-induction-checkpoint.npz")
    previous = arrays(args.moments / "reconstructed-moment-overlap-fields.npz")
    overlap = arrays(args.overlap / "induction-checkpoint.npz")
    reference = arrays(args.full / "volume-induction-comparison-fields.npz")
    boundary = arrays(args.boundary / "boundary-fields.npz")
    baseline = old["state_names"].index("taper_0.75_to_1.25")
    target = moments["position"]
    np.testing.assert_array_equal(target, overlap["position"])
    np.testing.assert_array_equal(target, previous["position"])
    incident_target = overlap["induced_velocity"][baseline][None]+moments["moment_induction"]
    names = parent["source_names"]
    mesh = load_native_mesh(args.oracle / "small-native-mesh.npz")
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    native = NativeVolumeSources.from_mesh(mesh)
    linear = LinearNativeVolumeSources.from_native(native, geometry["cell_centre"])
    for key, value in (("polyhedron_volume", linear.volume), ("polyhedron_centroid", linear.centroid), ("covariance", linear.covariance)):
        np.testing.assert_array_equal(moments[key], value)
    gamma = moments["volume_weights"][:, None]*moments["native_circulation"]
    # Retain the original summation order: P0 + Gaussian, then the zero-Gamma
    # first-moment correction. Combining P0 and P1 coefficients changes face
    # cancellation roundoff near the body, despite exact mathematical linearity.
    density = overlap["volume_density"][baseline]
    np.testing.assert_array_equal(density, gamma/linear.volume[:, None])
    constant_coefficient = native.coefficients(density)
    coefficient, _ = linear.coefficients(np.zeros((len(names), *gamma.shape)), moments["weighted_first_moment"])
    gaussian_position = overlap["gaussian_position"]
    gaussian_strength = overlap["gaussian_strength"][baseline:baseline+1]
    surface = CASE / "assets/cube.stl"
    if args.subdivisions:
        vertices, _ = load_stl(str(surface))
        for _ in range(args.subdivisions):
            vertices = subdivide_triangles(vertices)
        surface = args.output / "refined-cube.stl"
        save_stl(str(surface), vertices)
    count = 108*4**args.subdivisions
    if ti.lang.impl.get_runtime().prog is None:
        ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1)
    panel = vpm.PanelSolver(max_n_panels=max(128, 1 << (count-1).bit_length()), float_dtype="f64",
                            linear_solver="SCIPY", boundary_condition_type="NEUMANN", density=1,
                            freestream_velocity=np.array([1., 0., 0.]), coupling_scope="vpm_boundary_condition",
                            far_field_min_panels=count+1)
    panel.add_surface("cube", str(surface), reference_area=1)
    panel.initialize()
    lattice = panel.lattice
    assert lattice.n_panels == count
    panel_centre = lattice.panel_centre.to_numpy()[:count]
    panel_normal = lattice.normal.to_numpy()[:count]
    panel_area = lattice.area.to_numpy()[:count]
    vertices = lattice.vertex_position.to_numpy()[:count]
    np.testing.assert_allclose(panel_area.sum(), 6, rtol=0, atol=2e-12)
    np.testing.assert_allclose(np.sum(panel_area*np.sum(panel_centre*panel_normal, axis=1))/3, 1, rtol=0, atol=2e-12)
    np.testing.assert_array_equal(vertices.min(axis=(0, 1)), [-.5]*3)
    np.testing.assert_array_equal(vertices.max(axis=(0, 1)), [.5]*3)
    print(json.dumps({"stage": "body_initialized", "panels": count, "elapsed_seconds": time.perf_counter()-started}), flush=True)
    constant_collocation = native.evaluate(panel_centre, constant_coefficient).transpose(1, 0, 2)
    constant_collocation += particle_velocities(panel_centre, gaussian_position, np.full(len(gaussian_position), .125), gaussian_strength)
    incident_collocation = constant_collocation+linear.evaluate(panel_centre, coefficient).transpose(1, 0, 2)
    selections = {key: slice(*limits) for key, limits in parent["target_slices"].items()}
    incidence_check = None
    if args.subdivisions == 0:
        np.testing.assert_array_equal(panel_centre, target[selections["collocation"]])
        incidence_check = float(np.max(np.abs(incident_collocation-incident_target[:, selections["collocation"]])))
        assert incidence_check < 1e-12
    minimum_wall_distance = float(cKDTree(panel_centre).query(target[selections["wall"]])[0].min())
    assert minimum_wall_distance > 1e-5
    a = panel.aerodynamic_influence_coefficient.to_numpy()[:count, :count]
    adapter = SimpleNamespace(particles=SimpleNamespace(n_particles_total=len(gaussian_position)))
    epsilon = parent["derivative_steps"][0]
    normal, legacy_length = boundary["native_unit_normal"], np.linalg.norm(boundary["normal"], axis=1)
    records, output, strengths, baseline_difference = [], {}, [], None

    def tangent(value):
        value = value/legacy_length[:, None]
        return value-np.sum(value*normal, axis=1)[:, None]*normal

    for state, name in enumerate(names):
        panel.solve(np.array([1., 0., 0.]), incident_collocation[state], time=parent["physical_time"])
        strength = lattice.source_strength.to_numpy()[:count]
        strengths.append(strength)
        flux = float(panel_area @ strength)
        assert abs(flux) < 1e-10
        response = panel.compute_induced_velocity(target)
        assert panel._last_far_field_fraction == 0
        velocity = incident_target[state]+[1, 0, 0]+response
        if args.subdivisions == 0:
            difference = float(np.max(np.abs(velocity-previous[name+"__velocity"])))
            baseline_difference = max(baseline_difference or 0, difference)
            print(json.dumps({"stage": "baseline_comparison", "name": name, "velocity_maximum_difference": difference,
                              "collocation_incidence_maximum_difference": float(np.max(np.abs(incident_collocation[state]-incident_target[state, selections["collocation"]])))}), flush=True)
            if difference >= 1e-12:
                np.savez_compressed(args.output / "baseline-discrepancy.npz", name=name, position=target,
                                    actual=velocity, previous=previous[name+"__velocity"], strength=strength,
                                    incidence=incident_collocation[state], previous_incidence=incident_target[state, selections["collocation"]],
                                    panel_centre=panel_centre, panel_normal=panel_normal, matrix=a)
            assert difference < 1e-12
        samples = {key: velocity[selections[key]] for key in ("boundary", "plus", "minus", "plus2", "minus2", "plus_half", "minus_half")}
        derivative = {key: tangent(value) for key, value in normal_derivative_estimates(samples, epsilon).items()}
        changes = {key: float(np.max(np.abs(derivative[key]-derivative[key+"_half"]))) for key in ("centred", "exterior", "interior")}
        assert max(changes.values()) < 1e-6
        cut, mass = evaluate_vpm_velocity(adapter, boundary["position"], boundary["normal"], boundary["area"],
                                          freestream_velocity=np.array([1., 0., 0.]), fvm_box=np.asarray(parent["small_fvm_bounds"]),
                                          particle_spacing=.125, evaluated_velocity=samples["boundary"])
        un = np.sum(cut*boundary["normal"], axis=1)
        wall = velocity[selections["wall"]]
        wall_un = np.sum(wall*reference["wall_normal"], axis=1)
        residual = a @ strength+np.sum((incident_collocation[state]+[1, 0, 0])*panel_normal, axis=1)
        record = {"name": name, "panels": count, "cell_velocity_rms_over_Uinf": {},
                  "boundary_normal_velocity_rms_error_over_Uinf": field_rms((un-boundary["fvm_normal_velocity"])[:, None], boundary["area"]),
                  "boundary_native_tangential_gradient_rms_error": {key: field_rms(value-boundary["fvm_native_flux_tangential_normal_gradient"], boundary["native_vector_area"])
                                                                    for key, value in derivative.items()},
                  "derivative_step_halving_maximum_difference": changes,
                  "one_sided_derivative_difference_rms": field_rms(derivative["exterior_half"]-derivative["interior_half"], boundary["native_vector_area"]),
                  "wall_normal_velocity_rms_over_Uinf": float(np.sqrt(np.mean(wall_un**2))),
                  "wall_tangential_velocity_rms_over_Uinf": float(np.sqrt(np.mean(np.sum((wall-wall_un[:, None]*reference["wall_normal"])**2, axis=1)))),
                  "discrete_neumann_collocation_residual_rms": field_rms(residual[:, None], panel_area),
                  "body_source_flux": flux, "boundary_flux": mass}
        for group in ("near_body", "held_outer", "wake"):
            ids = reference[group+"__cell_ids"]
            record["cell_velocity_rms_over_Uinf"][group] = field_rms(velocity[selections[group]]-reference["native_cell_velocity"][ids], reference["native_cell_volume"][ids])
        records.append(record)
        output[name+"__velocity"] = velocity
        output[name+"__body_velocity"] = response
        output[name+"__boundary_normal_velocity"] = un
        output[name+"__collocation_residual"] = residual
        output.update({name+"__gradient_"+key: value for key, value in derivative.items()})
        print(json.dumps(record), flush=True)
    np.savez_compressed(args.output / "moment-panel-resolution-fields.npz", position=target, incident_velocity=incident_target,
                        incident_collocation_velocity=incident_collocation, panel_centre=panel_centre, panel_normal=panel_normal,
                        panel_area=panel_area, panel_strength=np.stack(strengths), **output)
    report = {"schema": "openonda-cube-moment-panel-resolution-3d/1", "status": "complete", "spatial_dimensions": 3,
              "subdivisions": args.subdivisions, "panels": count, "physical_time": parent["physical_time"],
              "panel_velocity_evaluation": "exact triangular source kernel; far_field_min_panels=panel_count+1",
              "small_fvm_cells": parent["small_fvm_cells"], "exterior_particles": parent["exterior_particles"],
              "small_fvm_bounds": parent["small_fvm_bounds"], "target_slices": parent["target_slices"], "derivative_steps": parent["derivative_steps"],
              "source_names": names, "minimum_wall_sample_distance_to_collocation": minimum_wall_distance,
              "baseline_collocation_incidence_maximum_difference": incidence_check, "baseline_velocity_maximum_difference": baseline_difference,
              "records": records, "sources": sources, "surface": hash_file(surface), "elapsed_seconds": time.perf_counter()-started,
              "limitations": ["Frozen physical cube source field. Only the body panel discretization changes; no time advance, force calculation or emission scheme.",
                              "All native circulation, reconstructed first moments, overlap weights, Gaussian complements and 68 exterior particles are fixed.",
                              "Common wall samples differ from panel collocation points. The discrete Neumann residual is not the independent wall penetration measurement.",
                              "The source-panel boundary condition imposes normal velocity; tangential wall velocity is measured but not imposed.",
                              "The same 2,840-cell small FVM box and actual native derivative targets remain unchanged."]}
    (args.output / "cube-moment-panel-resolution-3d.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "studies/coupler_accuracy/results"
    parser.add_argument("--moments", type=Path, default=base / "cube-3d-reconstructed-moment-overlap")
    parser.add_argument("--overlap", type=Path, default=base / "cube-3d-volume-overlap")
    parser.add_argument("--full", type=Path, default=base / "cube-3d-native-volume-induction")
    parser.add_argument("--oracle", type=Path, default=base / "cube-3d-oracle")
    parser.add_argument("--boundary", type=Path, default=base / "cube-3d-integrated-velocity-curl-reconstruction-boundary")
    parser.add_argument("--subdivisions", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
