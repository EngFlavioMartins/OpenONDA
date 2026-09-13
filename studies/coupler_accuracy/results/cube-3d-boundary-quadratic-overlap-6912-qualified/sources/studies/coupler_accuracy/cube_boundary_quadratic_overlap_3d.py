#!/usr/bin/env python3
"""Compare wall-aware quadratic reconstruction in the frozen physical cube."""

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
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, field_rms, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays
from studies.coupler_accuracy.induction_derivatives_3d import normal_derivative_estimates
from studies.coupler_accuracy.native_boundary_quadratic_3d import BoundaryQuadraticCellLeastSquares
from studies.coupler_accuracy.native_linear_volume_induction_3d import LinearNativeVolumeSources
from studies.coupler_accuracy.native_quadratic_moments_3d import (
    QuadraticCellLeastSquares,
    reconstruct_quadratic_faces,
    weak_quadratic_curl_moments,
)
from studies.coupler_accuracy.native_volume_induction_3d import NativeVolumeSources


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    parent = json.loads((args.panels / "cube-moment-panel-resolution-3d.json").read_text())
    assert parent["status"] == "complete" and parent["spatial_dimensions"] == 3
    surface = ROOT / parent["surface"]["path"]
    paths = [Path(__file__), surface, args.oracle / "small-native-mesh.npz", args.full / "volume-induction-comparison-fields.npz",
             args.boundary / "boundary-fields.npz", args.moments / "moment-overlap-induction-checkpoint.npz"]
    paths += [args.panels / name for name in ("cube-moment-panel-resolution-3d.json", "moment-panel-resolution-fields.npz")]
    paths += [ROOT / name for name in ("studies/coupler_accuracy/native_quadratic_moments_3d.py", "studies/coupler_accuracy/native_boundary_quadratic_3d.py",
                                      "studies/coupler_accuracy/native_linear_volume_induction_3d.py", "studies/coupler_accuracy/native_volume_induction_3d.py",
                                      "studies/coupler_accuracy/native_velocity_curl_integrals_3d.py", "studies/coupler_accuracy/cube_native_moment_reconstruction_3d.py",
                                      "studies/coupler_accuracy/induction_derivatives_3d.py", "source/coupler/boundary.py", "source/solvers/fvm/mesh/geometry.py",
                                      "source/solvers/vpm/boundary_elements/panels/solver/panel_solver.py", "source/solvers/vpm/boundary_elements/panels/solver/linear_solvers.py",
                                      "source/solvers/vpm/boundary_elements/panels/kernels/source_velocity.py", "source/solvers/vpm/boundary_elements/panels/kernels/induced_velocity.py",
                                      "tests/coupler/test_native_boundary_quadratic_3d.py")]
    sources = [hash_file(path) for path in paths]
    for path in paths:
        if path.suffix == ".py":
            destination = args.output / "sources" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
    (args.output / "sources-at-start.json").write_text(json.dumps(sources, indent=2)+"\n")
    base = read_arrays(args.panels / "moment-panel-resolution-fields.npz")
    old = read_arrays(args.moments / "moment-overlap-induction-checkpoint.npz")
    full = read_arrays(args.full / "volume-induction-comparison-fields.npz")
    boundary = read_arrays(args.boundary / "boundary-fields.npz")
    mesh = load_native_mesh(args.oracle / "small-native-mesh.npz")
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    native = NativeVolumeSources.from_mesh(mesh)
    linear = LinearNativeVolumeSources.from_native(native, geometry["cell_centre"])
    for key, value in (("polyhedron_volume", linear.volume), ("polyhedron_centroid", linear.centroid), ("covariance", linear.covariance)):
        np.testing.assert_array_equal(old[key], value)
    np.testing.assert_array_equal(base["position"], old["position"])
    velocity = full["native_cell_velocity"][old["inside_cell_ids"]]
    gamma, weight = old["native_circulation"], old["volume_weights"]
    active = weight > 0
    cube = next(patch for patch in mesh["boundary"] if patch["name"] == "cube")
    wall_faces = np.arange(cube["start_face"], cube["start_face"]+cube["n_faces"])
    for patch in mesh["boundary"]:
        if patch["name"] != "cube":
            rows = slice(patch["start_face"], patch["start_face"]+patch["n_faces"])
            assert not np.any(active[mesh["owners"][rows]])
    # Outer placeholders do not touch active source faces; no exterior FVM
    # cells or velocity samples enter either cell reconstruction stencil.
    names = ["constant_volume_control", "linear_face_moment_control"]
    delta_gamma = [np.zeros_like(gamma), np.zeros_like(gamma)]
    moments = [np.zeros_like(old["weighted_first_moment"][0]), old["weighted_first_moment"][3]]
    saved, diagnostics = {}, {}
    for kind in ("cell_only", "wall_observations"):
        if kind == "cell_only":
            fit = QuadraticCellLeastSquares.from_mesh(mesh, geometry["cell_centre"], linear.volume)
            g, h = fit.derivatives(velocity)
        else:
            fit = BoundaryQuadraticCellLeastSquares.from_mesh(mesh, geometry["cell_centre"], linear.volume, boundary_faces=wall_faces)
            g, h = fit.derivatives(velocity, np.zeros((len(fit.boundary_position), 3)))
        integral = fit.cell_integrals(velocity, g, h, linear.volume, linear.centroid, linear.covariance)
        faces = reconstruct_quadratic_faces(mesh, geometry, fit, velocity, g, h, np.zeros((mesh["n_faces"], 3)))
        reconstructed_gamma, moment = weak_quadratic_curl_moments(native, linear.centroid, integral, geometry["face_centre"], *faces)
        names.append("native_gamma_quadratic_"+kind+"_moments")
        delta_gamma.append(np.zeros_like(gamma))
        moments.append(weight[:, None, None]*moment[0])
        saved[kind+"__cell_velocity_integral"] = integral[0]
        saved[kind+"__unweighted_circulation"] = reconstructed_gamma[0]
        saved[kind+"__unweighted_moment"] = moment[0]
        diagnostics[kind] = {"maximum_condition": float(fit.condition.max()), "maximum_rings": int(fit.rings.max())}
        if kind == "wall_observations":
            names.append("quadratic_wall_gamma_and_moments")
            delta_gamma.append(weight[:, None]*(reconstructed_gamma[0]-gamma))
            moments.append(moments[-1])
            diagnostics[kind]["cells_with_wall_observations"] = int(np.sum(fit.boundary_observation_count > 0))
        print(json.dumps({"stage": "reconstruction", "kind": kind, **diagnostics[kind]}), flush=True)
    delta_gamma, moments = np.stack(delta_gamma), np.stack(moments)
    targets = np.vstack((base["position"], base["panel_centre"]))
    coefficient, _ = linear.coefficients(delta_gamma, moments)

    def progress(done, total):
        if done % 512 == 0 or done == total:
            print(json.dumps({"stage": "correction_induction", "done": done, "total": total, "elapsed_seconds": time.perf_counter()-started}), flush=True)

    correction = linear.evaluate(targets, coefficient, progress=progress).transpose(1, 0, 2)
    n_target, count = len(base["position"]), parent["panels"]
    np.testing.assert_array_equal(correction[0], 0)
    np.savez_compressed(args.output / "physical-boundary-quadratic-checkpoint.npz", position=base["position"], panel_centre=base["panel_centre"],
                        correction_induction=correction[:, :n_target], collocation_correction_induction=correction[:, n_target:],
                        delta_circulation=delta_gamma, first_moment=moments, native_circulation=gamma, volume_weights=weight,
                        source_names=names, volume=linear.volume, centroid=linear.centroid, covariance=linear.covariance, **saved)
    if ti.lang.impl.get_runtime().prog is None:
        ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1)
    panel = vpm.PanelSolver(max_n_panels=max(128, 1 << (count-1).bit_length()), float_dtype="f64", linear_solver="SCIPY",
                            boundary_condition_type="NEUMANN", density=1, freestream_velocity=np.array([1., 0., 0.]),
                            coupling_scope="vpm_boundary_condition", far_field_min_panels=count+1)
    panel.add_surface("cube", str(surface), reference_area=1)
    panel.initialize()
    lattice = panel.lattice
    assert lattice.n_panels == count
    np.testing.assert_array_equal(lattice.panel_centre.to_numpy()[:count], base["panel_centre"])
    np.testing.assert_array_equal(lattice.area.to_numpy()[:count], base["panel_area"])
    np.testing.assert_array_equal(lattice.normal.to_numpy()[:count], base["panel_normal"])
    records, output = [], {}
    selections = {key: slice(*limits) for key, limits in parent["target_slices"].items()}
    adapter = SimpleNamespace(particles=SimpleNamespace(n_particles_total=2908))
    epsilon = parent["derivative_steps"][0]
    for state, name in enumerate(names):
        # The constrained Neumann solve is linear. Solve the changed incident
        # field and add its body response to the qualified constant-volume state.
        panel.solve(np.zeros(3), correction[state, n_target:], time=parent["physical_time"])
        body_change = panel.compute_induced_velocity(base["position"])
        assert panel._last_far_field_fraction == 0
        actual = base["constant_volume_control__velocity"]+correction[state, :n_target]+body_change
        baseline_difference = None
        if state < 2:
            control = "constant_volume_control" if state == 0 else "linear_face_moments"
            baseline_difference = float(np.max(np.abs(actual-base[control+"__velocity"])))
            assert baseline_difference < 1e-12
        strengths = base["panel_strength"][0]+lattice.source_strength.to_numpy()[:count]
        flux = float(base["panel_area"] @ strengths)
        assert abs(flux) < 1e-10
        samples = {key: actual[selections[key]] for key in ("boundary", "plus", "minus", "plus2", "minus2", "plus_half", "minus_half")}
        derivatives = normal_derivative_estimates(samples, epsilon)
        normal = boundary["native_unit_normal"]
        for key, value in derivatives.items():
            value = value/np.linalg.norm(boundary["normal"], axis=1)[:, None]
            derivatives[key] = value-np.sum(value*normal, axis=1)[:, None]*normal
        changes = {key: float(np.max(np.abs(derivatives[key]-derivatives[key+"_half"]))) for key in ("centred", "exterior", "interior")}
        assert max(changes.values()) < 1e-6
        cut, mass = evaluate_vpm_velocity(adapter, boundary["position"], boundary["normal"], boundary["area"], freestream_velocity=np.array([1., 0., 0.]),
                                          fvm_box=np.asarray(parent["small_fvm_bounds"]), particle_spacing=.125, evaluated_velocity=samples["boundary"])
        un = np.sum(cut*boundary["normal"], axis=1)
        wall = actual[selections["wall"]]
        wall_un = np.sum(wall*full["wall_normal"], axis=1)
        row = {"name": name, "cell_velocity_rms_over_Uinf": {}, "baseline_maximum_velocity_difference": baseline_difference,
               "boundary_normal_velocity_rms_error_over_Uinf": field_rms((un-boundary["fvm_normal_velocity"])[:, None], boundary["area"]),
               "boundary_native_tangential_gradient_rms_error": {key: field_rms(value-boundary["fvm_native_flux_tangential_normal_gradient"], boundary["native_vector_area"])
                                                                 for key, value in derivatives.items()},
               "derivative_step_halving_maximum_difference": changes, "body_source_flux": flux, "boundary_flux": mass,
               "wall_normal_velocity_rms_over_Uinf": field_rms(wall_un[:, None], None),
               "wall_tangential_velocity_rms_over_Uinf": field_rms(wall-wall_un[:, None]*full["wall_normal"], None),
               "total_circulation_change": delta_gamma[state].sum(axis=0).tolist(),
               "circulation_change_L1": float(np.linalg.norm(delta_gamma[state], axis=1).sum())}
        for group in ("near_body", "held_outer", "wake"):
            ids = full[group+"__cell_ids"]
            row["cell_velocity_rms_over_Uinf"][group] = field_rms(actual[selections[group]]-full["native_cell_velocity"][ids], full["native_cell_volume"][ids])
        records.append(row)
        output.update({name+"__velocity": actual, name+"__body_velocity_change": body_change, name+"__panel_strength": strengths, name+"__boundary_normal_velocity": un})
        output.update({name+"__gradient_"+key: value for key, value in derivatives.items()})
        print(json.dumps(row), flush=True)
    np.savez_compressed(args.output / "physical-boundary-quadratic-fields.npz", position=base["position"], **output)
    report = {"schema": "openonda-cube-boundary-quadratic-overlap-3d/1", "status": "complete", "spatial_dimensions": 3,
              "small_fvm_cells": native.n_cells, "small_fvm_bounds": parent["small_fvm_bounds"], "body_panels": count,
              "physical_time": parent["physical_time"], "source_names": names, "target_slices": parent["target_slices"],
              "derivative_steps": parent["derivative_steps"], "stencil_diagnostics": diagnostics, "records": records,
              "sources": sources, "elapsed_seconds": time.perf_counter()-started,
              "limitations": ["Frozen physical cube induction; no time advance, force validation or particle emission.",
                              "Every source retains the same Gaussian complements, 68 exterior particles and native-volume taper inside the unchanged small FVM box.",
                              "Only the final state changes weighted native circulation, using its shared quadratic wall-fit Stokes sum. Earlier states retain all original Gamma.",
                              "The physical cell values are treated as point inputs to this candidate; this does not assert that FVM values are exact physical point samples.",
                              "The cell stencils use only the small FVM mesh and prescribed cube velocity; outer placeholder traces cannot touch active source faces.",
                              "Both previous physical controls are reproduced through the actual constrained linear Neumann response, with exact source-panel velocity evaluation."]}
    (args.output / "cube-boundary-quadratic-overlap-3d.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "studies/coupler_accuracy/results"
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, default=base / "cube-3d-oracle")
    parser.add_argument("--full", type=Path, default=base / "cube-3d-native-volume-induction")
    parser.add_argument("--moments", type=Path, default=base / "cube-3d-reconstructed-moment-overlap")
    parser.add_argument("--boundary", type=Path, default=base / "cube-3d-integrated-velocity-curl-reconstruction-boundary")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
