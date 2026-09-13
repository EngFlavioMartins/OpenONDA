#!/usr/bin/env python3
"""Build compact shared-trace source updates on the real small 3D FVM mesh."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

import openonda.fvm as fvm
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file, setup_for
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays
from studies.coupler_accuracy.cube_volume_overlap_3d import smooth_volume_weight
from studies.coupler_accuracy.native_boundary_quadratic_3d import BoundaryQuadraticCellLeastSquares
from studies.coupler_accuracy.native_linear_volume_induction_3d import LinearNativeVolumeSources
from studies.coupler_accuracy.native_moment_reconstruction_3d import (
    CellLeastSquares,
    reconstruct_linear_face_velocity,
    weak_curl_moments,
)
from studies.coupler_accuracy.native_quadratic_moments_3d import (
    reconstruct_quadratic_faces,
    weak_quadratic_curl_moments,
)
from studies.coupler_accuracy.native_shared_trace_update_3d import (
    global_first_moment,
    impulse,
    shared_trace_update,
)
from studies.coupler_accuracy.native_volume_induction_3d import NativeVolumeSources


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    oracle_report = json.loads((args.oracle / "cube-boundary-oracle.json").read_text())
    assert oracle_report["valid_for_comparison"] and oracle_report["spatial_dimensions"] == 3
    paths = [Path(__file__)]
    paths += [args.oracle / name for name in ("cube-boundary-oracle.json", "full-native-mesh.npz", "small-native-mesh.npz",
                                            "initial-cell-fields.npz", "cell-and-face-map.npz")]
    paths += [ROOT / name for name in ("studies/coupler_accuracy/native_shared_trace_update_3d.py", "studies/coupler_accuracy/native_boundary_quadratic_3d.py",
                                      "studies/coupler_accuracy/native_quadratic_moments_3d.py", "studies/coupler_accuracy/native_moment_reconstruction_3d.py",
                                      "studies/coupler_accuracy/native_linear_volume_induction_3d.py", "studies/coupler_accuracy/native_volume_induction_3d.py",
                                      "studies/coupler_accuracy/native_velocity_curl_integrals_3d.py", "studies/coupler_accuracy/cube_volume_overlap_3d.py",
                                      "studies/coupler_accuracy/cube_boundary_oracle.py", "studies/coupler_accuracy/cube_native_moment_reconstruction_3d.py",
                                      "source/solvers/fvm/core/solver.py", "source/solvers/fvm/coupling/coupler_interface.py", "source/solvers/fvm/mesh/geometry.py",
                                      "source/solvers/fvm/fields/gradients.py", "source/solvers/fvm/fields/diagnostics.py", "source/solvers/fvm/assemble/time_integration.py",
                                      "tests/coupler/test_native_shared_trace_update_3d.py")]
    if args.coarse_replay:
        paths += [args.coarse_replay / "physical-boundary-quadratic-checkpoint.npz"]
    sources = [hash_file(path) for path in paths]
    for path in paths:
        if path.suffix == ".py":
            target = args.output / "sources" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    (args.output / "sources-at-start.json").write_text(json.dumps(sources, indent=2)+"\n")
    full_mesh = load_native_mesh(args.oracle / "full-native-mesh.npz")
    mesh = load_native_mesh(args.oracle / "small-native-mesh.npz")
    geo = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    seed = read_arrays(args.oracle / "initial-cell-fields.npz")
    mapping = read_arrays(args.oracle / "cell-and-face-map.npz")
    ids = mapping["cell_ids"]
    full_setup = setup_for(full_mesh, "source-donor", .01, 1, turbulence=oracle_report["numerics"]["sgs"] != "none")
    with fvm.create_fvm_solver(full_setup, mesh=copy.deepcopy(full_mesh), case_dir=args.output / "donor") as solver:
        solver.set_initial_state(seed["velocity"], seed["pressure"])
        full_omega = solver.get_vorticity_field().copy()
        full_gradient = solver.get_velocity_gradient_field().copy()
        full_volume = solver.get_cell_volume().copy()
        full_centres = solver.get_cell_centre_coordinates().copy()
    np.testing.assert_allclose(geo["cell_centre"], full_centres[ids], rtol=0, atol=1e-13)
    np.testing.assert_allclose(geo["cell_volume"], full_volume[ids], rtol=0, atol=1e-14)
    velocity = seed["velocity"][ids]
    gamma = full_omega[ids]*full_volume[ids, None]
    native = NativeVolumeSources.from_mesh(mesh)
    linear = LinearNativeVolumeSources.from_native(native, geo["cell_centre"])
    weight = smooth_volume_weight(np.max(np.abs(geo["cell_centre"]), axis=1), .75, 1.25)
    active = weight > 0
    wall = next(patch for patch in mesh["boundary"] if patch["name"] == "cube")
    wall_faces = np.arange(wall["start_face"], wall["start_face"]+wall["n_faces"])
    for patch in mesh["boundary"]:
        if patch["name"] != "cube":
            rows = slice(patch["start_face"], patch["start_face"]+patch["n_faces"])
            assert not np.any(active[mesh["owners"][rows]])
    count = mesh["n_interior_faces"]
    owner, neighbour = mesh["owners"][:count], mesh["neighbours"]
    interpolation = geo["face_interpolation_weight"][:count]
    native_faces = [np.zeros((1, mesh["n_faces"])+(3,)*(i+1)) for i in range(3)]
    native_faces[0][0, :count] = (1-interpolation[:, None])*velocity[owner]+interpolation[:, None]*velocity[neighbour]
    point_g = CellLeastSquares.from_mesh(mesh, geo["cell_centre"]).gradient(velocity)
    point_integral = (velocity[None]+np.einsum("ni,snij->snj", linear.centroid-geo["cell_centre"], point_g))*linear.volume[None, :, None]
    linear_faces = reconstruct_linear_face_velocity(mesh, geo, velocity[None], point_g, np.zeros((mesh["n_faces"], 3)))
    _, linear_m = weak_curl_moments(native, linear.centroid, point_integral, geo["face_centre"], *linear_faces)
    names = ["constant_volume_control", "linear_face_point_moment_control"]
    delta_gamma = [np.zeros_like(gamma), np.zeros_like(gamma)]
    moments = [np.zeros((native.n_cells, 3, 3)), weight[:, None, None]*linear_m[0]]
    saved, fits, budgets = {}, {}, []
    for kind in ("point", "cell_average"):
        if kind == "point":
            centres, covariance = geo["cell_centre"], None
            base_integral = point_integral
        else:
            centres, covariance = linear.centroid, linear.covariance/linear.volume[:, None, None]
            base_integral = linear.volume[None, :, None]*velocity[None]
        fit = BoundaryQuadraticCellLeastSquares.from_mesh(mesh, centres, linear.volume, average_covariance=covariance, boundary_faces=wall_faces)
        gradient, hessian = fit.derivatives(velocity, np.zeros((len(fit.boundary_position), 3)))
        integrated_polynomial = fit.cell_integrals(velocity, gradient, hessian, linear.volume, linear.centroid, linear.covariance)
        integral = integrated_polynomial
        if kind == "cell_average":
            # Preserve the supplied cell mean U, so the FVM's transported V_FVM U
            # remains unchanged. The source integral uses its actual polyhedron
            # volume V_poly U; those two absolute measures are tracked separately.
            np.testing.assert_array_equal(integrated_polynomial, base_integral)
            integral = base_integral
        faces = reconstruct_quadratic_faces(mesh, geo, fit, velocity, gradient, hessian, np.zeros((mesh["n_faces"], 3)))
        base_gamma, base_moment = weak_quadratic_curl_moments(native, linear.centroid, base_integral, geo["face_centre"], *native_faces)
        new_gamma, new_moment = weak_quadratic_curl_moments(native, linear.centroid, integral, geo["face_centre"], *faces)
        replay = float(np.max(np.abs(base_gamma[0, active]-gamma[active])/linear.volume[active, None]))
        assert replay < 1e-10
        update = shared_trace_update(mesh, native, linear.centroid, weight, geo["face_centre"], base_integral, integral, native_faces, faces)
        global_h = global_first_moment(linear.centroid, update.circulation, update.first_moment)
        expected_h = -np.cross(np.eye(3)[None], update.cell_velocity_integral_change.sum(axis=1)[:, None])
        np.testing.assert_allclose(update.circulation.sum(axis=1), 0, rtol=0, atol=1e-12)
        np.testing.assert_allclose(global_h, expected_h, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(update.circulation[:, ~active], 0)
        np.testing.assert_array_equal(update.first_moment[:, ~active], 0)
        base_weighted_moment = weight[:, None, None]*base_moment[0]
        names += [kind+"_native_face_moments", kind+"_cell_weighted_quadratic", kind+"_shared_trace_quadratic"]
        cell_delta = weight[:, None]*(new_gamma[0]-gamma)
        delta_gamma += [np.zeros_like(gamma), cell_delta, update.circulation[0]]
        moments += [base_weighted_moment, weight[:, None, None]*new_moment[0], base_weighted_moment+update.first_moment[0]]
        for label, dg, dm in (("cell_weighted", cell_delta, weight[:, None, None]*new_moment[0]-base_weighted_moment),
                              ("shared_trace", update.circulation[0], update.first_moment[0])):
            h = global_first_moment(linear.centroid, dg, dm)
            row = {"input": kind, "update": label, "circulation_change": dg.sum(axis=0).tolist(),
                   "circulation_change_L1": float(np.linalg.norm(dg, axis=1).sum()),
                   "first_spatial_moment_change": h.tolist(), "impulse_change": impulse(h).tolist(),
                   "weighted_cell_integral_change": update.cell_velocity_integral_change.sum(axis=1)[0].tolist(),
                   "first_moment_budget_maximum_difference": float(np.max(np.abs(h-expected_h[0])))}
            budgets.append(row)
        saved.update({kind+"__base_cell_integral": base_integral[0], kind+"__new_cell_integral": integral[0],
                      kind+"__base_circulation": base_gamma[0], kind+"__base_moment": base_moment[0],
                      kind+"__new_circulation": new_gamma[0], kind+"__new_moment": new_moment[0],
                      kind+"__shared_delta_circulation": update.circulation[0], kind+"__shared_delta_moment": update.first_moment[0],
                      kind+"__weighted_cell_integral_change": update.cell_velocity_integral_change[0]})
        fits[kind] = {"maximum_condition": float(fit.condition.max()), "maximum_rings": int(fit.rings.max()),
                      "cells_with_wall_observations": int(np.sum(fit.boundary_observation_count > 0)),
                      "native_circulation_replay_maximum_difference_over_volume": replay,
                      "integrated_cell_mean_maximum_difference_from_input": float(np.max(np.abs(integral[0]/linear.volume[:, None]-velocity)))}
        print(json.dumps({"stage": "reconstructed", "input": kind, **fits[kind], "budgets": budgets[-2:]}), flush=True)
    delta_gamma, moments = np.stack(delta_gamma), np.stack(moments)
    outside = np.ones(full_mesh["n_cells"], dtype=bool)
    outside[ids] = False
    outside &= np.linalg.norm(full_omega, axis=1) >= .02
    gaussian_position = np.vstack((geo["cell_centre"], full_centres[outside].astype(np.float32).astype(float)))
    gaussian_gamma = np.vstack(((1-weight[:, None])*gamma, (full_omega[outside]*full_volume[outside, None]).astype(np.float32).astype(float)))
    replay_check = None
    if args.coarse_replay:
        previous = read_arrays(args.coarse_replay / "physical-boundary-quadratic-checkpoint.npz")
        np.testing.assert_array_equal(gamma, previous["native_circulation"])
        np.testing.assert_array_equal(weight, previous["volume_weights"])
        replay_check = {}
        for new_state, old_state in ((0, 0), (1, 1), (3, 4)):
            for key, actual, expected in (("circulation", delta_gamma[new_state], previous["delta_circulation"][old_state]),
                                          ("moment", moments[new_state], previous["first_moment"][old_state])):
                value = float(np.max(np.abs(actual-expected)))
                assert value < 1e-14
                replay_check[names[new_state]+"__"+key] = value
    source_h = global_first_moment(linear.centroid, weight[None, :, None]*gamma[None]+delta_gamma, moments)
    source_h += np.einsum("ni,nj->ij", gaussian_position, gaussian_gamma)[None]
    source_gamma = np.sum(weight[None, :, None]*gamma[None]+delta_gamma, axis=1)+gaussian_gamma.sum(axis=0)[None]
    np.savez_compressed(args.output / "shared-trace-source-fields.npz", source_names=names, delta_circulation=delta_gamma,
                        first_moment=moments, native_circulation=gamma, volume_weight=weight, face_weight=update.face_weight,
                        gaussian_position=gaussian_position, gaussian_circulation=gaussian_gamma, inside_cell_ids=ids,
                        volume=linear.volume, centroid=linear.centroid, covariance=linear.covariance,
                        native_centres=geo["cell_centre"], native_volume=geo["cell_volume"], cell_velocity=velocity,
                        full_centres=full_centres, full_volume=full_volume, full_velocity=seed["velocity"],
                        full_velocity_gradient=full_gradient, full_vorticity=full_omega,
                        total_source_circulation=source_gamma, total_source_first_moment=source_h, **saved)
    relative_volume_difference = (geo["cell_volume"]-linear.volume)/linear.volume
    report = {"schema": "openonda-cube-shared-trace-sources-3d/1", "status": "complete", "spatial_dimensions": 3,
              "physical_time": float(seed["physical_time"]), "source_sgs": oracle_report["numerics"]["sgs"],
              "full_fvm_cells": full_mesh["n_cells"], "small_fvm_cells": mesh["n_cells"], "source_names": names,
              "positive_volume_cells": int(active.sum()), "exterior_particles": int(outside.sum()),
              "particle_spacing": oracle_report["requested_wall_spacing"], "small_fvm_bounds": oracle_report["small_bounds"],
              "maximum_relative_native_to_polyhedron_volume_difference": float(np.max(np.abs(relative_volume_difference))),
              "native_minus_polyhedron_cell_momentum_sum": np.sum((geo["cell_volume"]-linear.volume)[:, None]*velocity, axis=0).tolist(),
              "stencil_diagnostics": fits, "update_budgets": budgets, "coarse_source_replay": replay_check, "sources": sources,
              "elapsed_seconds": time.perf_counter()-started,
              "limitations": ["Source construction only; induction, body response and particle emission are separate qualifications.",
                              "The two reconstructions interpret the same FVM velocity values as points at native centres or means over actual fan polyhedra. Neither is an exact physical measurement claim.",
                              "Preserving mean U preserves the FVM's V_FVM U state while reconstructed source integrals use V_poly U. The geometric difference is measured explicitly; velocity is not rescaled by a volume ratio.",
                              "Shared corrections preserve total raw source circulation and have the stated first-moment/impulse budget relative to their corresponding native-face-moment baseline.",
                              "The cell-average shared update preserves that baseline's impulse. It does not claim identical impulse to the zero-moment constant-volume control.",
                              "Overlap face weights are constant per face and equal the minimum of adjacent cell weights. This is a compact discrete trace rule, not an exact continuous taper product.",
                              "Near-cell fits use only the small mesh and prescribed zero cube velocity. The unchanged initial Gaussian seed/complement originates from the full reference snapshot.",
                              "Affine vorticity sources need not be divergence free. Raw source moments are distinct from cell moments of the curl of their projected induced velocity."]}
    (args.output / "cube-shared-trace-sources-3d.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--coarse-replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.oracle, args.output = args.oracle.resolve(), args.output.resolve()
    if args.coarse_replay:
        args.coarse_replay = args.coarse_replay.resolve()
    run(args)
