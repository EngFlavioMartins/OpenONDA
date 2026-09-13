#!/usr/bin/env python3
"""Refine only the cube's source panels while holding particle states fixed."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc
import taichi as ti

import openonda.fvm as fvm
import openonda.vpm as vpm
from source.coupler.boundary import evaluate_vpm_velocity
from source.coupler.renewal_projection import gaussian_velocity_operator, geometric_renewal_mask
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.vpm.boundary_elements.panels.geometry.stl_io import load_stl, save_stl
from studies.coupler_accuracy.cube_boundary_oracle import CASE, ROOT, field_rms, hash_file, setup_for


def subdivide_triangles(vertices):
    a, b, c = np.asarray(vertices, dtype=float).transpose(1, 0, 2)
    ab, bc, ca = (a+b)/2, (b+c)/2, (c+a)/2
    children = [np.stack(x, axis=1) for x in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))]
    return np.stack(children, axis=1).reshape(-1, 3, 3)


def fixed_wall_samples(epsilon):
    uv = qmc.Sobol(2, scramble=True, seed=1837).random_base2(8)-0.5
    points, normals = [], []
    for axis in range(3):
        transverse = [i for i in range(3) if i != axis]
        for sign in (-1, 1):
            x, n = np.zeros((len(uv), 3)), np.zeros((len(uv), 3))
            x[:, transverse] = uv
            x[:, axis] = sign*(0.5+epsilon)
            n[:, axis] = sign
            points.append(x)
            normals.append(n)
    return np.concatenate(points), np.concatenate(normals)


def particle_velocities(points, position, radius, strengths):
    result = np.empty((len(strengths), len(points), 3))
    coefficients = strengths.reshape(len(strengths), -1).T
    for start in range(0, len(points), 128):
        target = points[start:start+128]
        values = gaussian_velocity_operator(target, position, radius) @ coefficients
        result[:, start:start+len(target)] = values.reshape(len(target), 3, len(strengths)).transpose(2, 0, 1)
    return result


def run(args):
    if args.subdivisions not in (0, 1, 2, 3):
        raise ValueError("Use zero to three triangle subdivisions")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    paths = [Path(__file__), CASE / "assets/cube.stl", args.study / "native-reconstruction-3d.json",
             args.study / "held-fields.npz", args.boundary / "boundary-fields.npz",
             args.boundary / "native-curl-3d.json", args.failure,
             *[args.oracle / f for f in ("full-native-mesh.npz", "small-native-mesh.npz", "initial-cell-fields.npz")]]
    paths += [ROOT / p for p in (
        "source/coupler/renewal_projection.py", "source/coupler/boundary.py",
        "source/solvers/fvm/core/solver.py", "source/solvers/fvm/fields/gradients.py",
        "source/solvers/fvm/coupling/coupler_interface.py", "source/solvers/fvm/mesh/geometry.py",
        "source/solvers/vpm/boundary_elements/panels/geometry/stl_io.py",
        "source/solvers/vpm/boundary_elements/panels/solver/panel_solver.py",
        "source/solvers/vpm/boundary_elements/panels/solver/linear_solvers.py",
        "source/solvers/vpm/boundary_elements/panels/kernels/source_velocity.py",
        "source/solvers/vpm/boundary_elements/panels/kernels/induced_velocity.py")]
    sources = [hash_file(p) for p in paths]
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (args.output / "sources-at-start.json").write_text(json.dumps(sources, indent=2) + "\n")
    metadata = json.loads((args.study / "native-reconstruction-3d.json").read_text())
    with np.load(args.study / "held-fields.npz", allow_pickle=False) as data:
        fields = {key: data[key].copy() for key in data.files}
    with np.load(args.boundary / "boundary-fields.npz", allow_pickle=False) as data:
        boundary = {key: data[key].copy() for key in data.files}
    with np.load(args.oracle / "initial-cell-fields.npz", allow_pickle=False) as data:
        centres, velocity, pressure = (data[key].copy() for key in ("centres", "velocity", "pressure"))
    with np.load(args.failure, allow_pickle=False) as data:
        small_position = data["fvm_position"].copy()
    mesh = load_native_mesh(args.oracle / "full-native-mesh.npz")
    small_mesh = load_native_mesh(args.oracle / "small-native-mesh.npz")
    with fvm.create_fvm_solver(setup_for(mesh, "panel-donor", 0.01, 1), mesh=copy.deepcopy(mesh),
                               case_dir=args.output / "donor") as solver:
        solver.set_initial_state(velocity, pressure)
        omega, volume = solver.get_vorticity_field().copy(), solver.get_cell_volume().copy()
    distance, small_ids = cKDTree(centres).query(small_position)
    np.testing.assert_allclose(distance, 0, rtol=0, atol=1e-13)
    distance, held_rows = cKDTree(small_position).query(fields["position"])
    np.testing.assert_allclose(distance, 0, rtol=0, atol=1e-13)
    h = metadata["particle_spacing"]
    retained = np.linalg.norm(omega, axis=1) >= 0.02
    seed_position = centres[retained].astype(np.float32).astype(float)
    seed_strength = (omega*volume[:, None])[retained].astype(np.float32).astype(float)
    preserved = ~geometric_renewal_mask(seed_position, metadata["renewal_bounds"], particle_spacing=h)
    position = np.vstack((fields["renewable_position"], seed_position[preserved]))
    radius = np.r_[fields["radius"], np.full(preserved.sum(), h)]
    names = [row["name"] for row in metadata["results"] if row["name"] != "original_unregularized_fit"]
    strengths = np.stack([np.vstack((fields[name+"__strength"], seed_strength[preserved])) for name in names])
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
                            freestream_velocity=np.array([1., 0., 0.]), coupling_scope="vpm_boundary_condition")
    panel.add_surface("cube", str(surface), reference_area=1)
    panel.initialize()
    lattice = panel.lattice
    assert lattice.n_panels == count
    vertices = lattice.vertex_position.to_numpy()[:count]
    normal = lattice.normal.to_numpy()[:count]
    panel_centres = lattice.panel_centre.to_numpy()[:count]
    area = lattice.area.to_numpy()[:count]
    np.testing.assert_allclose(area.sum(), 6, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.sum(area*np.sum(panel_centres*normal, axis=1))/3, 1, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(vertices.min(axis=(0, 1)), [-0.5]*3)
    np.testing.assert_array_equal(vertices.max(axis=(0, 1)), [0.5]*3)
    assert np.all(np.sum(panel_centres*normal, axis=1) > 0)
    wall_epsilon = 1e-6
    wall_points, wall_normal = fixed_wall_samples(wall_epsilon)
    minimum_wall_sample_distance = float(cKDTree(panel_centres).query(wall_points)[0].min())
    assert minimum_wall_sample_distance > 1e-5
    epsilon = 1e-4*h
    positions, selections, total = [], {}, 0

    def add(name, values):
        nonlocal total
        selections[name] = slice(total, total+len(values))
        total += len(values)
        positions.append(values)

    add("small", small_position)
    add("wall", wall_points)
    add("boundary", boundary["position"])
    add("plus", boundary["position"]+epsilon*boundary["normal"])
    add("minus", boundary["position"]-epsilon*boundary["normal"])
    add("check_plus", boundary["position"][:16]+2*epsilon*boundary["normal"][:16])
    add("check_minus", boundary["position"][:16]-2*epsilon*boundary["normal"][:16])
    add("collocation", panel_centres+wall_epsilon*normal)
    targets = np.vstack(positions)
    particle = particle_velocities(targets, position, radius, strengths)
    incident = particle_velocities(panel_centres, position, radius, strengths)
    bounds = np.column_stack((small_mesh["vertex_position"].min(axis=0), small_mesh["vertex_position"].max(axis=0))).ravel()
    adapter = SimpleNamespace(particles=SimpleNamespace(n_particles_total=len(position)))
    near = np.max(np.abs(small_position), axis=1) < 0.8
    legacy_length = np.linalg.norm(boundary["normal"], axis=1)
    records, output_fields = [], {}
    for index, name in enumerate(names):
        panel.solve(np.array([1., 0., 0.]), incident[index], time=metadata["physical_time"])
        values = particle[index]+panel.compute_induced_velocity(targets)+[1, 0, 0]
        wall = values[selections["wall"]]
        wall_un = np.sum(wall*wall_normal, axis=1)
        collocation_un = np.sum(values[selections["collocation"]]*normal, axis=1)
        cut, flux = evaluate_vpm_velocity(adapter, boundary["position"], boundary["normal"], boundary["area"],
                                          freestream_velocity=np.array([1., 0., 0.]), fvm_box=bounds,
                                          particle_spacing=h, evaluated_velocity=values[selections["boundary"]])
        cut_un = np.sum(cut*boundary["normal"], axis=1)
        derivative = (values[selections["plus"]]-values[selections["minus"]])/(2*epsilon)
        check = (values[selections["check_plus"]]-values[selections["check_minus"]])/(4*epsilon)
        difference = float(np.max(np.abs(derivative[:16]-check)))
        assert difference < 1e-7
        tangent = derivative-np.sum(derivative*boundary["normal"], axis=1)[:, None]*boundary["normal"]
        native_derivative = derivative/legacy_length[:, None]
        native_tangent = native_derivative-np.sum(native_derivative*boundary["native_unit_normal"], axis=1)[:, None]*boundary["native_unit_normal"]
        small = values[selections["small"]]
        if args.subdivisions == 0:
            np.testing.assert_allclose(small[held_rows], fields[name+"__velocity"], rtol=0, atol=1e-12)
        sigma = lattice.source_strength.to_numpy()[:count]
        record = {"name": name, "panel_count": count,
                  "whole_small_cell_velocity_rms_over_Uinf": field_rms(small-velocity[small_ids], volume[small_ids]),
                  "near_body_cell_velocity_rms_over_Uinf": field_rms(small[near]-velocity[small_ids[near]], volume[small_ids[near]]),
                  "held_cell_velocity_rms_over_Uinf": field_rms(small[held_rows]-fields["fvm_velocity"], fields["volume"]),
                  "collocation_normal_velocity_rms_over_Uinf": field_rms(collocation_un[:, None], area),
                  "independent_wall_normal_velocity_rms_over_Uinf": float(np.sqrt(np.mean(wall_un**2))),
                  "independent_wall_tangential_velocity_rms_over_Uinf": float(np.sqrt(np.mean(np.sum((wall-wall_un[:, None]*wall_normal)**2, axis=1)))),
                  "boundary_normal_velocity_rms_error_over_Uinf": field_rms((cut_un-boundary["fvm_normal_velocity"])[:, None], boundary["area"]),
                  "boundary_tangential_normal_gradient_rms_error": field_rms(tangent-boundary["fvm_interpolated_tangential_normal_gradient"], boundary["area"]),
                  "boundary_native_flux_tangential_normal_gradient_rms_error": field_rms(native_tangent-boundary["fvm_native_flux_tangential_normal_gradient"], boundary["native_vector_area"]),
                  "boundary_flux": flux, "derivative_step_check_maximum_difference": difference,
                  "body_source_flux": float(area@sigma), "elapsed_seconds": time.perf_counter()-started}
        assert abs(record["body_source_flux"]) < 1e-10
        records.append(record)
        output_fields.update({name+"__small_velocity": small, name+"__wall_velocity": wall,
                              name+"__boundary_normal_velocity": cut_un,
                              name+"__boundary_native_tangential_gradient": native_tangent})
        print(json.dumps(record), flush=True)
        (args.output / "history.json").write_text(json.dumps(records, indent=2) + "\n")
    np.savez_compressed(args.output / "panel-resolution-fields.npz", small_position=small_position,
                        wall_position=wall_points, wall_normal=wall_normal, boundary_position=boundary["position"],
                        panel_position=panel_centres, panel_area=area, particle_position=position,
                        particle_radius=radius, particle_strengths=strengths, **output_fields)
    report = {"schema": "openonda-frozen-cube-panel-resolution-3d/1", "status": "complete", "spatial_dimensions": 3,
              "subdivisions": args.subdivisions, "panels": count, "particle_count": len(position),
              "wall_samples": len(wall_points), "minimum_wall_sample_to_collocation_distance": minimum_wall_sample_distance,
              "wall_epsilon": wall_epsilon, "particle_spacing": h, "physical_time": metadata["physical_time"],
              "near_body_definition": "Native fluid cells with max(abs(x),abs(y),abs(z)) < 0.8", "near_body_cells": int(near.sum()),
              "surface": hash_file(surface), "sources": sources, "results": records,
              "limitations": ["Particle positions, radii and strengths are frozen; only the body source-panel surface is subdivided.",
                              "Common scrambled Sobol samples on all six cube faces are distinct from collocation points at every tested resolution.",
                              "Tangential wall velocity is measured, but the source-panel solve imposes only normal velocity.",
                              "No new particle fit or evolving hybrid calculation is part of this isolation."]}
    (args.output / "cube-panel-resolution-3d.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    results = ROOT / "studies/coupler_accuracy/results"
    parser.add_argument("--study", type=Path, default=results / "cube-3d-integrated-velocity-curl-reconstruction")
    parser.add_argument("--boundary", type=Path, default=results / "cube-3d-integrated-velocity-curl-reconstruction-boundary")
    parser.add_argument("--oracle", type=Path, default=results / "cube-3d-oracle")
    parser.add_argument("--failure", type=Path, default=results / "cube-3d-frozen-projected-guard/hybrid/renewal_projection_failure_oracle.npz")
    parser.add_argument("--subdivisions", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
