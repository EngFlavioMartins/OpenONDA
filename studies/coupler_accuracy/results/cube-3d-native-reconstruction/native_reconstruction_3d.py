#!/usr/bin/env python3
"""Frozen 3D reconstruction with native curl, wall samples and conserved moments.

Fits only donor cells and the known stationary wall condition. The actual outer
coupling faces are reserved for a separate boundary audit. Every candidate keeps
the original outside particles, core widths and source positions fixed.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack
from scipy.sparse.linalg import LinearOperator, lsmr
from scipy.spatial import cKDTree

import openonda.fvm as fvm
from source.coupler.renewal_projection import (
    geometric_renewal_mask,
    sparse_gaussian_vorticity_basis,
)
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from studies.coupler_accuracy.cube_boundary_oracle import (
    CASE,
    ROOT,
    field_rms,
    hash_file,
    setup_for,
)
from studies.coupler_accuracy.joint_reconstruction_3d import (
    CubePanelResponse,
    constrained_least_squares,
    gaussian_velocity_curl_operator,
)
from studies.coupler_accuracy.moment_budget import MomentBudgetProjector, particle_moments
from studies.coupler_accuracy.native_curl_3d import native_gauss_curl_stencil


def selection_operator(ids, total_points):
    ids = np.asarray(ids, dtype=int)
    columns = (3 * ids[:, None] + np.arange(3)).ravel()
    return coo_matrix((np.ones(len(columns)), (np.arange(len(columns)), columns)),
                      shape=(len(columns), 3 * total_points)).tocsr()


def fit_native(velocity_map, background, observation, target, prior, position,
               *, conserve_moments, max_iterations):
    penalty = 0.05 / np.linalg.norm(prior)
    n = prior.size

    def apply(x):
        return np.r_[observation @ (velocity_map @ x), penalty * x]

    def transpose(y):
        return velocity_map.T @ (observation.T @ y[:-n]) + penalty * y[-n:]

    operator = LinearOperator((observation.shape[0] + n, n), matvec=apply, rmatvec=transpose, dtype=float)
    rhs = np.r_[target - observation @ (velocity_map @ prior.ravel() + background), np.zeros(n)]
    unconstrained = lsmr(operator, rhs, atol=1e-9, btol=1e-9, maxiter=1000)
    initial = prior + unconstrained[0].reshape(-1, 3)
    budget = 2 * np.linalg.norm(prior, axis=1).sum()
    projector = MomentBudgetProjector(position, prior, budget) if conserve_moments else None
    candidate, diagnostics = constrained_least_squares(
        operator, rhs, prior, budget, initial, max_iterations=max_iterations, projector=projector)
    diagnostics.update({"lsmr_stop": int(unconstrained[1]), "lsmr_iterations": int(unconstrained[2]),
                        "prior_weight": 0.05, "l1_budget_ratio": 2.0,
                        "conserve_moments": conserve_moments,
                        "maximum_moment_projection_iterations": projector.max_iterations_used if projector else 0})
    return candidate, diagnostics


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source_paths = [Path(__file__), args.failure, args.oracle / "initial-cell-fields.npz",
                    args.oracle / "full-native-mesh.npz", CASE / "assets/cube.stl"]
    source_paths += [ROOT / p for p in (
        "studies/coupler_accuracy/cube_boundary_oracle.py",
        "studies/coupler_accuracy/joint_reconstruction_3d.py",
        "studies/coupler_accuracy/moment_budget.py",
        "studies/coupler_accuracy/native_curl_3d.py",
        "source/coupler/renewal_projection.py",
        "source/solvers/fvm/fields/gradients.py", "source/solvers/fvm/fields/diagnostics.py",
        "source/solvers/fvm/coupling/coupler_interface.py",
        "source/solvers/vpm/boundary_elements/panels/solver/panel_solver.py",
        "source/solvers/vpm/boundary_elements/panels/solver/linear_solvers.py",
        "source/solvers/vpm/boundary_elements/panels/kernels/induced_velocity.py",
        "source/solvers/vpm/boundary_elements/panels/kernels/source_velocity.py")]
    sources = [hash_file(p) for p in source_paths]
    (args.output / "sources-at-start.json").write_text(json.dumps(sources, indent=2) + "\n")
    for path in source_paths:
        if path.parent == Path(__file__).parent:
            (args.output / path.name).write_bytes(path.read_bytes())
    with np.load(args.failure, allow_pickle=False) as data:
        failure = {k: data[k].copy() for k in data.files}
    with np.load(args.oracle / "initial-cell-fields.npz", allow_pickle=False) as data:
        centres, velocity, pressure = (data[k].copy() for k in ("centres", "velocity", "pressure"))
        physical_time = float(data["physical_time"])
    tree = cKDTree(centres)
    fit, held = failure["fit_position"], failure["verification_position"]
    fit_distance, fit_ids = tree.query(fit)
    held_distance, held_ids = tree.query(held)
    np.testing.assert_allclose(np.r_[fit_distance, held_distance], 0, rtol=0, atol=1e-13)
    assert not set(fit_ids) & set(held_ids)
    ids = np.r_[fit_ids, held_ids]
    mesh = load_native_mesh(args.oracle / "full-native-mesh.npz")
    with fvm.create_fvm_solver(setup_for(mesh, "donor", 0.01, 1), mesh=copy.deepcopy(mesh),
                               case_dir=args.output / "donor") as solver:
        solver.set_initial_state(velocity, pressure)
        omega, volume = solver.get_vorticity_field().copy(), solver.get_cell_volume().copy()
        stencil, dependencies = native_gauss_curl_stencil(solver.mesh_data, solver.geo_data, ids)
        replay = (stencil @ solver.velocity[dependencies].ravel()).reshape(-1, 3)
        np.testing.assert_allclose(replay, omega[ids], rtol=0, atol=5e-13)
        boundary_ids = dependencies[dependencies >= mesh["n_cells"]]
        wall = next(p for p in solver.mesh_data["boundary"] if p["name"] == "cube")
        wall_ghosts = mesh["n_cells"] + np.arange(wall["start_face"], wall["start_face"] + wall["n_faces"]) - mesh["n_interior_faces"]
        assert np.all(np.isin(boundary_ids, wall_ghosts))
        np.testing.assert_array_equal(solver.velocity[boundary_ids], 0)
    _, small_ids = tree.query(failure["fvm_position"])
    np.testing.assert_allclose(omega[small_ids], failure["fvm_vorticity"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(velocity[small_ids], failure["fvm_velocity"], rtol=0, atol=1e-13)
    h = float(failure["particle_spacing"])
    retained = np.linalg.norm(omega, axis=1) >= 0.02
    seed_position = centres[retained].astype(np.float32).astype(float)
    seed_strength = (omega * volume[:, None])[retained].astype(np.float32).astype(float)
    preserved = ~geometric_renewal_mask(seed_position, failure["renewal_bounds"], particle_spacing=h)
    outer_position, outer_strength = seed_position[preserved], seed_strength[preserved]
    position, radius, prior = (failure[k] for k in ("solve_position", "solve_radius", "solve_prior"))
    nonzero = np.linalg.norm(prior, axis=1) > 0
    distance, seed_ids = cKDTree(seed_position).query(position[nonzero])
    np.testing.assert_allclose(distance, 0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(prior[nonzero], seed_strength[seed_ids], rtol=0, atol=1e-12)
    panel = CubePanelResponse()
    fluid = dependencies < mesh["n_cells"]
    dependency_cells = dependencies[fluid]
    wall_points = panel.centres + 1e-6 * panel.normal
    evaluation_points = np.vstack((centres[dependency_cells], wall_points))
    u, background = panel.velocity_operator(evaluation_points, position, radius)
    outer, _ = panel.velocity_operator(evaluation_points, outer_position, h)
    background += outer @ outer_strength.ravel()
    del outer
    native = hstack((stencil[:, np.repeat(fluid, 3)], csr_matrix((stencil.shape[0], 3 * panel.count))), format="csr")
    fit_rows, held_rows = np.searchsorted(dependency_cells, fit_ids), np.searchsorted(dependency_cells, held_ids)
    np.testing.assert_array_equal(dependency_cells[fit_rows], fit_ids)
    np.testing.assert_array_equal(dependency_cells[held_rows], held_ids)
    fit_select = selection_operator(fit_rows, len(evaluation_points))
    wall_select = selection_operator(np.arange(len(dependency_cells), len(evaluation_points)), len(evaluation_points))
    weights_omega = np.repeat(np.sqrt(volume[fit_ids] / volume[fit_ids].sum()) / field_rms(omega[fit_ids], volume[fit_ids]), 3)
    weights_velocity = 5 * np.repeat(np.sqrt(volume[fit_ids] / volume[fit_ids].sum()), 3)
    weights_wall = 5 * np.repeat(np.sqrt(panel.area / panel.area.sum()), 3)
    observations = [native[:3 * len(fit)].multiply(weights_omega[:, None]),
                    fit_select.multiply(weights_velocity[:, None]), wall_select.multiply(weights_wall[:, None])]
    targets = [weights_omega * omega[fit_ids].ravel(), weights_velocity * velocity[fit_ids].ravel(), np.zeros(3 * panel.count)]
    gheld = sparse_gaussian_vorticity_basis(held, position, radius)
    raw_outer = sparse_gaussian_vorticity_basis(held, outer_position, h) @ outer_strength
    cheld = gaussian_velocity_curl_operator(held, position, radius)
    curl_outer = (gaussian_velocity_curl_operator(held, outer_position, h) @ outer_strength.ravel()).reshape(-1, 3)
    barycentric = np.array([[2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]])
    held_wall_points = np.einsum("qv,pvd->pqd", barycentric, panel.vertices).reshape(-1, 3) + 1e-6 * np.repeat(panel.normal, 3, axis=0)
    wall_u, wall_background = panel.velocity_operator(held_wall_points, position, radius)
    outer_wall, _ = panel.velocity_operator(held_wall_points, outer_position, h)
    wall_background += outer_wall @ outer_strength.ravel()
    records, fields = [], {}
    prior_l1 = np.linalg.norm(prior, axis=1).sum()
    prior_moments = particle_moments(position, prior)
    print(json.dumps({"event": "native_operators_ready", "fluid_dependencies": len(dependency_cells),
                      "wall_fit_points": panel.count, "wall_held_points": len(held_wall_points),
                      "fit_cells": len(fit), "held_cells": len(held), "sources": len(position)}), flush=True)

    def measure(name, strength, diagnostics):
        values = (u @ strength.ravel() + background).reshape(-1, 3)
        discrete_curl = (native @ values.ravel()).reshape(-1, 3)
        raw = gheld @ strength + raw_outer
        physical = (cheld @ strength.ravel()).reshape(-1, 3) + curl_outer
        wall_values = (wall_u @ strength.ravel() + wall_background).reshape(-1, 3)
        moments = particle_moments(position, strength)
        record = {"name": name, **diagnostics,
                  "fit_velocity_rms_over_Uinf": field_rms(values[fit_rows] - velocity[fit_ids], volume[fit_ids]),
                  "held_velocity_rms_over_Uinf": field_rms(values[held_rows] - velocity[held_ids], volume[held_ids]),
                  "fit_native_curl_relative_error": field_rms(discrete_curl[:len(fit)] - omega[fit_ids], volume[fit_ids]) / field_rms(omega[fit_ids], volume[fit_ids]),
                  "held_native_curl_relative_error": field_rms(discrete_curl[len(fit):] - omega[held_ids], volume[held_ids]) / field_rms(omega[held_ids], volume[held_ids]),
                  "held_raw_vorticity_relative_error": field_rms(raw - omega[held_ids], volume[held_ids]) / field_rms(omega[held_ids], volume[held_ids]),
                  "held_velocity_curl_relative_error": field_rms(physical - omega[held_ids], volume[held_ids]) / field_rms(omega[held_ids], volume[held_ids]),
                  "fit_wall_velocity_rms_over_Uinf": field_rms(values[len(dependency_cells):], panel.area),
                  "held_wall_velocity_rms_over_Uinf": field_rms(wall_values, np.repeat(panel.area / 3, 3)),
                  "strength_l1_over_prior": float(np.linalg.norm(strength, axis=1).sum() / prior_l1),
                  "net_strength_change_norm": float(np.linalg.norm(moments[0] - prior_moments[0])),
                  "impulse_change_norm": float(np.linalg.norm(moments[1] - prior_moments[1])),
                  "elapsed_seconds": time.perf_counter() - started}
        records.append(record)
        fields.update({name + "__strength": strength, name + "__velocity": values[held_rows],
                       name + "__raw_vorticity": raw, name + "__velocity_curl": physical,
                       name + "__native_curl": discrete_curl[len(fit):], name + "__held_wall_velocity": wall_values})
        print(json.dumps(record), flush=True)
        (args.output / "history.json").write_text(json.dumps(records, indent=2) + "\n")

    measure("donor_volume_vorticity", prior, {})
    measure("original_unregularized_fit", failure["solve_strength"], {})
    cases = [("native_curl_only", 1, False), ("native_curl_velocity", 2, False),
             ("native_curl_velocity_wall", 3, False), ("native_curl_velocity_wall_moments", 3, True)]
    for name, count, conserve in cases:
        candidate, diagnostics = fit_native(u, background, vstack(observations[:count], format="csr"),
                                            np.concatenate(targets[:count]), prior, position,
                                            conserve_moments=conserve, max_iterations=args.max_iterations)
        measure(name, candidate, diagnostics)
    np.savez_compressed(args.output / "held-fields.npz", position=held, volume=volume[held_ids],
                        fvm_velocity=velocity[held_ids], fvm_vorticity=omega[held_ids],
                        renewable_position=position, radius=radius, held_wall_position=held_wall_points, **fields)
    report = {"schema": "openonda-frozen-native-reconstruction-3d/1", "status": "complete",
              "spatial_dimensions": 3, "physical_time": physical_time, "particle_spacing": h,
              "renewal_bounds": failure["renewal_bounds"].tolist(), "results": records, "sources": sources,
              "limitations": ["Frozen coarse 3D cube; no production checkpoint or acceptance gate is changed.",
                              "Actual coupling faces are excluded from fitting and evaluated in a separate audit.",
                              "The moment target is this frozen donor; evolving regional exchange is not qualified.",
                              "Native-curl fit does not by itself establish continuous-field vorticity or no-slip accuracy.",
                              "Reused development verification cells; fresh 3D acceptance data remain required."]}
    (args.output / "native-reconstruction-3d.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure", type=Path, default=ROOT / "studies/coupler_accuracy/results/cube-3d-frozen-projected-guard/hybrid/renewal_projection_failure_oracle.npz")
    parser.add_argument("--oracle", type=Path, default=ROOT / "studies/coupler_accuracy/results/cube-3d-oracle")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int, default=6000)
    run(parser.parse_args())
