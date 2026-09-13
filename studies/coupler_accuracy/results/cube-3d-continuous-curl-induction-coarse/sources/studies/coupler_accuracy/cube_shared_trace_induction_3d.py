#!/usr/bin/env python3
"""Compare compact shared-trace sources through induction and a fixed 3D body."""

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
from studies.coupler_accuracy.cube_panel_resolution_3d import (
    fixed_wall_samples,
    particle_velocities,
)
from studies.coupler_accuracy.native_face_velocity_sampling_3d import InteriorFaceVelocitySampler
from studies.coupler_accuracy.native_linear_volume_induction_3d import LinearNativeVolumeSources
from studies.coupler_accuracy.native_volume_induction_3d import NativeVolumeSources


def select_regions(centres, prior=None):
    if prior is not None:
        return {name: prior[name+"__cell_ids"] for name in ("near_body", "held_outer", "wake")}
    radius = np.max(np.abs(centres), axis=1)
    masks = {"near_body": radius < .8,
             "held_outer": (radius > 1.5) & (radius < 2.5),
             "wake": (centres[:, 0] > 1.5) & (centres[:, 0] < 5)
                     & (np.max(np.abs(centres[:, 1:]), axis=1) < 1.5)}
    rng = np.random.default_rng(20260913)
    return {name: np.sort(rng.choice(np.flatnonzero(mask), min(256, int(mask.sum())), replace=False))
            for name, mask in masks.items()}


def unique_targets(groups):
    position, inverse = np.unique(np.vstack(list(groups.values())), axis=0, return_inverse=True)
    indices, start = {}, 0
    for name, values in groups.items():
        indices[name] = inverse[start:start+len(values)]
        np.testing.assert_array_equal(position[indices[name]], values)
        start += len(values)
    return position, indices


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result_root = ROOT / "studies/coupler_accuracy/results"
    source_report = json.loads((args.source / "cube-shared-trace-sources-3d.json").read_text())
    panel_report = json.loads((args.panels / "cube-moment-panel-resolution-3d.json").read_text())
    assert source_report["status"] == panel_report["status"] == "complete"
    assert source_report["spatial_dimensions"] == panel_report["spatial_dimensions"] == 3
    surface = ROOT / panel_report["surface"]["path"]
    assert hash_file(surface) == panel_report["surface"]
    paths = [Path(__file__).resolve(), surface, args.panels / "cube-moment-panel-resolution-3d.json",
             args.panels / "moment-panel-resolution-fields.npz"]
    paths += [args.source / name for name in ("cube-shared-trace-sources-3d.json", "shared-trace-source-fields.npz")]
    paths += [args.oracle / name for name in ("full-native-mesh.npz", "small-native-mesh.npz", "cell-and-face-map.npz")]
    paths += [ROOT / name for name in (
        "studies/coupler_accuracy/native_face_velocity_sampling_3d.py", "studies/coupler_accuracy/native_face_trace_3d.py",
        "studies/coupler_accuracy/native_linear_volume_induction_3d.py", "studies/coupler_accuracy/native_volume_induction_3d.py",
        "studies/coupler_accuracy/cube_native_moment_reconstruction_3d.py", "studies/coupler_accuracy/cube_panel_resolution_3d.py",
        "studies/coupler_accuracy/cube_boundary_oracle.py", "source/coupler/boundary.py", "source/coupler/renewal_projection.py",
        "source/solvers/fvm/fields/gradients.py", "source/solvers/fvm/assemble/diffusion.py", "source/solvers/fvm/mesh/geometry.py",
        "source/solvers/fvm/solve/simple_solver.py", "source/solvers/vpm/boundary_elements/panels/geometry/stl_io.py",
        "source/solvers/vpm/boundary_elements/panels/solver/panel_solver.py", "source/solvers/vpm/boundary_elements/panels/solver/linear_solvers.py",
        "source/solvers/vpm/boundary_elements/panels/kernels/source_velocity.py", "source/solvers/vpm/boundary_elements/panels/kernels/induced_velocity.py")]
    prior_cells, prior, prior_report, prior_sampled = None, None, None, None
    if args.coarse_replay:
        prior_cell_path = result_root / "cube-3d-native-volume-induction/volume-induction-comparison-fields.npz"
        prior_sampled_path = result_root / "cube-3d-sampled-native-face/sampled-native-face-fields.npz"
        prior = read_arrays(args.coarse_replay / "physical-boundary-quadratic-fields.npz")
        prior_report = json.loads((args.coarse_replay / "cube-boundary-quadratic-overlap-3d.json").read_text())
        prior_cells, prior_sampled = read_arrays(prior_cell_path), read_arrays(prior_sampled_path)
        assert prior_report["status"] == "complete" and prior_report["body_panels"] == panel_report["panels"]
        paths += [prior_cell_path, prior_sampled_path, args.coarse_replay / "physical-boundary-quadratic-fields.npz",
                  args.coarse_replay / "cube-boundary-quadratic-overlap-3d.json"]
    if args.induction_checkpoint:
        paths.append(args.induction_checkpoint)
    sources = [hash_file(path) for path in paths]
    for path in paths:
        if path.suffix == ".py":
            target = args.output / "sources" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    (args.output / "sources-at-start.json").write_text(json.dumps(sources, indent=2)+"\n")
    source = read_arrays(args.source / "shared-trace-source-fields.npz")
    source_digest = hash_file(args.source / "shared-trace-source-fields.npz")["sha256"]
    names = source["source_names"].tolist()
    assert names == source_report["source_names"] and len(names) == 8
    mesh = load_native_mesh(args.oracle / "full-native-mesh.npz")
    for patch in mesh["boundary"]:
        patch["velocity_type"] = "fixedValue"
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    small = load_native_mesh(args.oracle / "small-native-mesh.npz")
    small_geometry = compute_mesh_geometry(small, gradient_scheme="gauss", compute_lsq=False)
    mapping = read_arrays(args.oracle / "cell-and-face-map.npz")
    cut = next(patch for patch in small["boundary"] if patch["name"] == "numericalBoundary")
    cut_rows = slice(cut["start_face"], cut["start_face"]+cut["n_faces"])
    faces, signs = mapping["face_ids"][cut_rows], mapping["signs"][cut_rows]
    sampler = InteriorFaceVelocitySampler(mesh, geometry, faces, signs)
    np.testing.assert_array_equal(geometry["cell_centre"], source["full_centres"])
    np.testing.assert_array_equal(mapping["cell_ids"], source["inside_cell_ids"])
    np.testing.assert_allclose(geometry["face_centre"][faces], small_geometry["face_centre"][cut_rows], rtol=0, atol=1e-13)
    native = NativeVolumeSources.from_mesh(small)
    linear = LinearNativeVolumeSources.from_native(native, small_geometry["cell_centre"])
    for key, value in (("volume", linear.volume), ("centroid", linear.centroid), ("covariance", linear.covariance)):
        np.testing.assert_array_equal(source[key], value)
    reference = sampler.evaluate(source["full_velocity"][sampler.sample_cells])
    reference_full = sampler.trace.evaluate(source["full_velocity"], source["full_velocity_gradient"],
                                            np.zeros(mesh["n_cells"]), np.zeros((mesh["n_cells"], 3)))
    reference_replay = float(np.max(np.abs(reference["tangential_gradient"]-reference_full["native_flux_tangential_gradient"])))
    assert reference_replay < 1e-12
    normal, area = sampler.trace.normal, sampler.trace.area
    reference_un = np.sum(reference["face_velocity"]*normal, axis=1)
    panels = read_arrays(args.panels / "moment-panel-resolution-fields.npz")
    count = panel_report["panels"]
    cell_groups = select_regions(source["full_centres"], prior_cells)
    wall, wall_normal = fixed_wall_samples(1e-6)
    groups = {name: source["full_centres"][ids] for name, ids in cell_groups.items()}
    groups.update(wall=wall, boundary=geometry["face_centre"][faces], sample=source["full_centres"][sampler.sample_cells],
                  collocation=panels["panel_centre"])
    targets, index = unique_targets(groups)
    print(json.dumps({"stage": "geometry", "targets": len(targets), "faces": len(faces), "sample_cells": len(sampler.sample_cells),
                      "triangles": len(native.triangles), "panels": count, "reference_gradient_replay": reference_replay}), flush=True)

    def progress(stage):
        def report(done, total):
            if done % 512 == 0 or done == total:
                print(json.dumps({"stage": stage, "done": done, "total": total,
                                  "elapsed_seconds": time.perf_counter()-started}), flush=True)
        return report

    if args.induction_checkpoint:
        cached = read_arrays(args.induction_checkpoint)
        np.testing.assert_array_equal(cached["position"], targets)
        np.testing.assert_array_equal(cached["source_names"], source["source_names"])
        assert str(cached["source_archive_sha256"]) == source_digest
        baseline, correction = cached["baseline_induction"], cached["correction_induction"]
    else:
        density = source["volume_weight"][:, None]*source["native_circulation"]/linear.volume[:, None]
        baseline = native.evaluate(targets, native.coefficients(density), progress=progress("constant_induction")).transpose(1, 0, 2)
        baseline += particle_velocities(targets, source["gaussian_position"],
                                        np.full(len(source["gaussian_position"]), source_report["particle_spacing"]),
                                        source["gaussian_circulation"][None])
        np.savez_compressed(args.output / "constant-induction-checkpoint.npz", position=targets, baseline_induction=baseline,
                            source_archive_sha256=source_digest)
        coefficient, _ = linear.coefficients(source["delta_circulation"], source["first_moment"])
        correction = linear.evaluate(targets, coefficient, progress=progress("moment_induction")).transpose(1, 0, 2)
    np.testing.assert_array_equal(correction[0], 0)
    np.savez_compressed(args.output / "shared-trace-induction-checkpoint.npz", position=targets, baseline_induction=baseline,
                        correction_induction=correction, source_names=names, source_archive_sha256=source_digest)
    if ti.lang.impl.get_runtime().prog is None:
        ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1)
    panel = vpm.PanelSolver(max_n_panels=max(128, 1 << (count-1).bit_length()), float_dtype="f64", linear_solver="SCIPY",
                            boundary_condition_type="NEUMANN", density=1, freestream_velocity=np.array([1., 0., 0.]),
                            coupling_scope="vpm_boundary_condition", far_field_min_panels=count+1)
    panel.add_surface("cube", str(surface), reference_area=1)
    panel.initialize()
    lattice = panel.lattice
    assert lattice.n_panels == count
    for field, expected in ((lattice.panel_centre, "panel_centre"), (lattice.normal, "panel_normal"), (lattice.area, "panel_area")):
        np.testing.assert_array_equal(field.to_numpy()[:count], panels[expected])
    a = panel.aerodynamic_influence_coefficient.to_numpy()[:count, :count]
    adapter = SimpleNamespace(particles=SimpleNamespace(n_particles_total=len(source["gaussian_position"])))
    side = 2*np.argmax(np.abs(normal), axis=1)+(np.sum(normal, axis=1) > 0).astype(int)
    side_names = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
    saved, records = {}, []
    for state, name in enumerate(names):
        if state == 0:
            panel.solve(np.array([1., 0., 0.]), baseline[0, index["collocation"]], time=source_report["physical_time"])
            base_body = panel.compute_induced_velocity(targets)
            base_strength = lattice.source_strength.to_numpy()[:count].copy()
            base_velocity = baseline[0]+[1., 0., 0.]+base_body
            actual, body, strength = base_velocity, base_body, base_strength
        else:
            panel.solve(np.zeros(3), correction[state, index["collocation"]], time=source_report["physical_time"])
            body_change = panel.compute_induced_velocity(targets)
            actual = base_velocity+correction[state]+body_change
            body, strength = base_body+body_change, base_strength+lattice.source_strength.to_numpy()[:count]
        assert panel._last_far_field_fraction == 0
        flux = float(panels["panel_area"] @ strength)
        assert abs(flux) < 1e-10
        observed = sampler.evaluate(actual[index["sample"]])
        normal_values, fluxes = {}, {}
        for mode, velocity in (("point", actual[index["boundary"]]), ("native", observed["face_velocity"])):
            corrected, mass = evaluate_vpm_velocity(adapter, groups["boundary"], normal, area,
                                                    freestream_velocity=np.array([1., 0., 0.]),
                                                    fvm_box=np.asarray(source_report["small_fvm_bounds"]),
                                                    particle_spacing=source_report["particle_spacing"], evaluated_velocity=velocity)
            normal_values[mode] = np.sum(corrected*normal, axis=1)
            fluxes[mode] = mass
        gt_error = observed["tangential_gradient"]-reference["tangential_gradient"]
        wall_velocity = actual[index["wall"]]
        wall_un = np.sum(wall_velocity*wall_normal, axis=1)
        incident = baseline[0, index["collocation"]]+correction[state, index["collocation"]]
        residual = a @ strength+np.sum((incident+[1., 0., 0.])*panels["panel_normal"], axis=1)
        row = {"name": name, "cell_velocity_rms_over_Uinf": {}, "body_source_flux": flux, "boundary_flux": fluxes,
               "native_gradient_error_rms": field_rms(gt_error, area),
               "point_normal_velocity_error_rms": field_rms((normal_values["point"]-reference_un)[:, None], area),
               "native_normal_velocity_error_rms": field_rms((normal_values["native"]-reference_un)[:, None], area),
               "wall_normal_velocity_rms_over_Uinf": field_rms(wall_un[:, None], None),
               "wall_tangential_velocity_rms_over_Uinf": field_rms(wall_velocity-wall_un[:, None]*wall_normal, None),
               "discrete_neumann_collocation_residual_rms": field_rms(residual[:, None], panels["panel_area"]),
               "face_groups": {}}
        for group, ids in cell_groups.items():
            row["cell_velocity_rms_over_Uinf"][group] = field_rms(actual[index[group]]-source["full_velocity"][ids], source["full_volume"][ids])
        sample_index = np.full(mesh["n_cells"], -1)
        sample_index[sampler.sample_cells] = np.arange(len(sampler.sample_cells))
        row["gradient_input_cell_velocity_error_rms"] = field_rms(actual[index["sample"]][sample_index[sampler.gradient_cells]]
                                                                  -source["full_velocity"][sampler.gradient_cells],
                                                                  source["full_volume"][sampler.gradient_cells])
        for number, side_name in enumerate(side_names):
            mask = side == number
            row["face_groups"][side_name] = {"faces": int(mask.sum()), "native_gradient_error_rms": field_rms(gt_error[mask], area[mask]),
                "point_normal_velocity_error_rms": field_rms((normal_values["point"]-reference_un)[mask, None], area[mask]),
                "native_normal_velocity_error_rms": field_rms((normal_values["native"]-reference_un)[mask, None], area[mask])}
        if args.coarse_replay and state in (0, 1, 3):
            old_state = {0: 0, 1: 1, 3: 4}[state]
            old_name = prior_report["source_names"][old_state]
            replay = {}
            for group in ("near_body", "held_outer", "wake", "wall", "boundary"):
                old_rows = slice(*prior_report["target_slices"][group])
                np.testing.assert_allclose(groups[group], prior["position"][old_rows], rtol=0, atol=1e-13)
                replay[group] = float(np.max(np.abs(actual[index[group]]-prior[old_name+"__velocity"][old_rows])))
            prefix = f"panel{count}__{old_name}__"
            replay["native_gradient"] = float(np.max(np.abs(observed["tangential_gradient"]-prior_sampled[prefix+"tangential_gradient"])))
            assert max(replay.values()) < 1e-12
            row["coarse_replay_maximum_differences"] = replay
        records.append(row)
        saved.update({name+"__velocity": actual, name+"__body_velocity": body, name+"__panel_strength": strength,
                      name+"__point_normal_velocity": normal_values["point"], name+"__native_normal_velocity": normal_values["native"],
                      name+"__collocation_residual": residual})
        saved.update({name+"__"+key: value for key, value in observed.items()})
        print(json.dumps({key: value for key, value in row.items() if key not in ("face_groups", "boundary_flux")}), flush=True)
    np.savez_compressed(args.output / "shared-trace-physical-fields.npz", position=targets, normal=normal, area=area, face_group=side,
                        source_names=names, wall_normal=wall_normal, sample_cells=sampler.sample_cells, gradient_cells=sampler.gradient_cells,
                        gradient_faces=sampler.gradient_faces, full_face_ids=faces, signs=signs,
                        panel_centre=panels["panel_centre"], panel_normal=panels["panel_normal"], panel_area=panels["panel_area"],
                        reference_tangential_gradient=reference["tangential_gradient"], reference_normal_velocity=reference_un,
                        reference_face_velocity=reference["face_velocity"], reference_cell_gradient=reference["cell_gradient"],
                        **{"target__"+key: value for key, value in index.items()},
                        **{key+"__cell_ids": value for key, value in cell_groups.items()}, **saved)
    report = {"schema": "openonda-cube-shared-trace-induction-3d/1", "status": "complete", "spatial_dimensions": 3,
              "physical_time": source_report["physical_time"], "source_sgs": source_report["source_sgs"],
              "small_fvm_bounds": source_report["small_fvm_bounds"], "small_fvm_cells": small["n_cells"], "full_fvm_cells": mesh["n_cells"],
              "particle_spacing": source_report["particle_spacing"], "body_panels": count, "boundary_faces": len(faces),
              "sample_cells": len(sampler.sample_cells), "gradient_cells": len(sampler.gradient_cells), "unique_targets": len(targets),
              "region_cell_counts": {key: len(value) for key, value in cell_groups.items()}, "source_names": names,
              "reference_gradient_replay_maximum_difference": reference_replay, "records": records, "sources": sources,
              "elapsed_seconds": time.perf_counter()-started,
              "limitations": ["Frozen physical field only; no advancing solution, forces or particle-emission change.",
                              "The coarse LES and medium laminar snapshots are different physical states; their comparison is not mesh convergence.",
                              "All eight states share the Gaussian complement, exterior particles and fixed body mesh. Source corrections use only the small FVM velocity and prescribed no slip.",
                              "Every cut face receives a native derivative from induced velocities on a complete geometric stencil. Exterior reference values enter only error measurements.",
                              "Normal velocity uses true unit normals and vector-area magnitudes with the public mass correction, matching the current live convention. Earlier frozen normal-velocity metrics used a different area convention.",
                              "Body evaluation uses the exact f64 triangular source kernel without far-field grouping. Source response is the constant baseline plus the linear moment/circulation correction.",
                              "Source circulation and impulse budgets refer to raw affine vorticity, not local moments of the curl of its projected induced velocity.",
                              "The cell-average shared update conserves its native-face-moment baseline, not the zero-moment constant-volume control."]}
    (args.output / "cube-shared-trace-induction-3d.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--panels", type=Path, default=ROOT / "studies/coupler_accuracy/results/cube-3d-moment-panels-1728")
    parser.add_argument("--coarse-replay", type=Path)
    parser.add_argument("--induction-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("source", "oracle", "panels", "coarse_replay", "induction_checkpoint", "output"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    run(args)
