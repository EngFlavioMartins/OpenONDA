#!/usr/bin/env python3
"""Compare continuous and native-face observations of the same frozen velocity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import taichi as ti

from source.coupler.boundary import evaluate_vpm_velocity
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.vpm.boundary_elements.panels.geometry.stl_io import load_stl
from source.solvers.vpm.boundary_elements.panels.kernels.induced_velocity import (
    compute_source_induced_velocity_kernel,
)
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, field_rms, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays
from studies.coupler_accuracy.cube_panel_resolution_3d import particle_velocities
from studies.coupler_accuracy.native_face_trace_3d import tangential
from studies.coupler_accuracy.native_face_velocity_sampling_3d import InteriorFaceVelocitySampler
from studies.coupler_accuracy.native_linear_volume_induction_3d import LinearNativeVolumeSources
from studies.coupler_accuracy.native_volume_induction_3d import NativeVolumeSources


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    experiments = []
    paths = [Path(__file__), args.overlap / "induction-checkpoint.npz", args.full / "volume-induction-comparison-fields.npz",
             args.boundary / "boundary-fields.npz"]
    paths += [args.oracle / name for name in ("full-native-mesh.npz", "small-native-mesh.npz", "cell-and-face-map.npz")]
    for directory in args.physical:
        report = json.loads((directory / "cube-boundary-quadratic-overlap-3d.json").read_text())
        assert report["status"] == "complete" and report["spatial_dimensions"] == 3
        parent_path = ROOT / next(row["path"] for row in report["sources"] if row["path"].endswith("/cube-moment-panel-resolution-3d.json"))
        parent = json.loads(parent_path.read_text())
        assert parent["status"] == "complete" and parent["panels"] == report["body_panels"]
        surface = ROOT / parent["surface"]["path"]
        experiments.append((directory, report, parent_path.parent, surface))
        paths += [directory / name for name in ("cube-boundary-quadratic-overlap-3d.json", "physical-boundary-quadratic-checkpoint.npz",
                                                "physical-boundary-quadratic-fields.npz")]
        paths += [parent_path, parent_path.parent / "moment-panel-resolution-fields.npz", surface]
    paths += [ROOT / name for name in (
        "studies/coupler_accuracy/native_face_velocity_sampling_3d.py", "studies/coupler_accuracy/native_face_trace_3d.py",
        "studies/coupler_accuracy/native_linear_volume_induction_3d.py", "studies/coupler_accuracy/native_volume_induction_3d.py",
        "studies/coupler_accuracy/cube_native_moment_reconstruction_3d.py", "studies/coupler_accuracy/cube_panel_resolution_3d.py",
        "studies/coupler_accuracy/cube_boundary_oracle.py", "source/coupler/boundary.py", "source/coupler/renewal_projection.py",
        "source/solvers/fvm/fields/gradients.py", "source/solvers/fvm/assemble/diffusion.py", "source/solvers/fvm/mesh/geometry.py",
        "source/solvers/fvm/solve/simple_solver.py", "source/solvers/vpm/boundary_elements/panels/geometry/stl_io.py",
        "source/solvers/vpm/boundary_elements/panels/kernels/source_velocity.py", "source/solvers/vpm/boundary_elements/panels/kernels/induced_velocity.py",
        "tests/coupler/test_native_face_velocity_sampling_3d.py")]
    paths = sorted(set(paths))
    sources = [hash_file(path) for path in paths]
    for path in paths:
        if path.suffix == ".py":
            destination = args.output / "sources" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
    (args.output / "sources-at-start.json").write_text(json.dumps(sources, indent=2)+"\n")

    mesh = load_native_mesh(args.oracle / "full-native-mesh.npz")
    # The native gradient kernel requires resolved patch metadata. These zero
    # remote patch placeholders cannot touch any selected gradient cell; the
    # sampler rejects that dependency before the field is evaluated.
    for patch in mesh["boundary"]:
        patch["velocity_type"] = "fixedValue"
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    small = load_native_mesh(args.oracle / "small-native-mesh.npz")
    small_geometry = compute_mesh_geometry(small, gradient_scheme="gauss", compute_lsq=False)
    cut = next(patch for patch in small["boundary"] if patch["name"] == "numericalBoundary")
    cut_rows = slice(cut["start_face"], cut["start_face"]+cut["n_faces"])
    mapping = read_arrays(args.oracle / "cell-and-face-map.npz")
    faces, signs = mapping["face_ids"][cut_rows], mapping["signs"][cut_rows]
    sampler = InteriorFaceVelocitySampler(mesh, geometry, faces, signs)
    boundary = read_arrays(args.boundary / "boundary-fields.npz")
    truth = read_arrays(args.full / "volume-induction-comparison-fields.npz")
    source = read_arrays(experiments[0][0] / "physical-boundary-quadratic-checkpoint.npz")
    overlap = read_arrays(args.overlap / "induction-checkpoint.npz")
    np.testing.assert_array_equal(geometry["cell_centre"], truth["native_cell_centres"])
    np.testing.assert_allclose(geometry["face_centre"][faces], boundary["position"], rtol=0, atol=1e-13)
    np.testing.assert_allclose(sampler.trace.normal, boundary["native_unit_normal"], rtol=0, atol=1e-14)
    np.testing.assert_allclose(sampler.trace.area, boundary["native_vector_area"], rtol=0, atol=1e-14)
    np.testing.assert_array_equal(source["volume_weights"], overlap["volume_weights"][3])
    native = NativeVolumeSources.from_mesh(small)
    linear = LinearNativeVolumeSources.from_native(native, small_geometry["cell_centre"])
    for key, value in (("volume", linear.volume), ("centroid", linear.centroid), ("covariance", linear.covariance)):
        np.testing.assert_array_equal(source[key], value)
    np.testing.assert_array_equal(overlap["volume_density"][3], source["volume_weights"][:, None]*source["native_circulation"]/linear.volume[:, None])
    names = source["source_names"].tolist()
    assert all(report["source_names"] == names for _, report, _, _ in experiments)
    reference = sampler.evaluate(truth["native_cell_velocity"][sampler.sample_cells])
    reference_gt = boundary["fvm_native_flux_tangential_normal_gradient"]
    reference_replay = float(np.max(np.abs(reference["tangential_gradient"]-reference_gt)))
    assert reference_replay < 1e-12
    ref_un_from_value = np.sum(reference["face_velocity"]*boundary["normal"], axis=1)
    reference_flux_replay = float(np.max(np.abs(ref_un_from_value-boundary["fvm_normal_velocity"])))
    assert reference_flux_replay < 1e-12
    n_sample = len(sampler.sample_cells)
    targets = np.vstack((geometry["cell_centre"][sampler.sample_cells], boundary["position"]))
    print(json.dumps({"stage": "stencil", "sample_cells": n_sample, "gradient_cells": len(sampler.gradient_cells),
                      "faces": len(faces), "reference_gradient_replay_maximum_difference": reference_replay}), flush=True)

    def progress(done, total):
        if done % 512 == 0 or done == total:
            print(json.dumps({"stage": "source_induction", "done": done, "total": total,
                              "elapsed_seconds": time.perf_counter()-started}), flush=True)

    # Preserve the qualified summation order: constant volumes plus Gaussians,
    # followed by the affine correction. All three body counts share this field.
    baseline = native.evaluate(targets, native.coefficients(overlap["volume_density"][3]), progress=progress).transpose(1, 0, 2)
    baseline += particle_velocities(targets, overlap["gaussian_position"], np.full(len(overlap["gaussian_position"]), .125), overlap["gaussian_strength"][3:4])
    coefficients, _ = linear.coefficients(source["delta_circulation"], source["first_moment"])
    correction = linear.evaluate(targets, coefficients, progress=progress).transpose(1, 0, 2)
    incident = baseline+correction
    np.savez_compressed(args.output / "sampled-face-source-induction.npz", position=targets, sample_cells=sampler.sample_cells,
                        gradient_cells=sampler.gradient_cells, gradient_faces=sampler.gradient_faces,
                        full_face_ids=faces, signs=signs, baseline_induction=baseline, correction_induction=correction,
                        incident_velocity=incident, source_names=names)
    if ti.lang.impl.get_runtime().prog is None:
        ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1)
    records, saved, replay_errors = [], {}, []
    normal, area = sampler.trace.normal, sampler.trace.area
    side = 2*np.argmax(np.abs(normal), axis=1)+(np.sum(normal, axis=1) > 0).astype(int)
    side_names = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
    adapter = SimpleNamespace(particles=SimpleNamespace(n_particles_total=len(overlap["gaussian_position"])))
    for directory, report, panel_directory, surface in experiments:
        prior = read_arrays(directory / "physical-boundary-quadratic-fields.npz")
        checkpoint = read_arrays(directory / "physical-boundary-quadratic-checkpoint.npz")
        panels = read_arrays(panel_directory / "moment-panel-resolution-fields.npz")
        for key in ("delta_circulation", "first_moment", "native_circulation", "volume_weights", "volume", "centroid", "covariance", "source_names"):
            np.testing.assert_array_equal(checkpoint[key], source[key])
        vertices, _ = load_stl(str(surface))
        count = report["body_panels"]
        assert len(vertices) == count
        cross = np.cross(vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0])
        np.testing.assert_allclose(cross/np.linalg.norm(cross, axis=1)[:, None], panels["panel_normal"], rtol=0, atol=1e-15)
        np.testing.assert_allclose(vertices.mean(axis=1), panels["panel_centre"], rtol=0, atol=2e-16)
        np.testing.assert_allclose(np.linalg.norm(cross, axis=1)/2, panels["panel_area"], rtol=0, atol=2e-16)
        cut_slice = slice(*report["target_slices"]["boundary"])
        bounds = np.asarray(report["small_fvm_bounds"])
        for state, name in enumerate(names):
            prefix = f"panel{count}__{name}"
            body = np.empty((len(targets), 3))
            strengths = np.ascontiguousarray(prior[name+"__panel_strength"])
            compute_source_induced_velocity_kernel(np.ascontiguousarray(vertices), np.ascontiguousarray(panels["panel_normal"]),
                                                   strengths, np.ascontiguousarray(targets), body)
            actual = incident[state]+[1., 0., 0.]+body
            replay = float(np.max(np.abs(actual[n_sample:]-prior[name+"__velocity"][cut_slice])))
            replay_errors.append(replay)
            assert replay < 1e-12
            native_trace = sampler.evaluate(actual[:n_sample])
            continuous_gt = prior[name+"__gradient_centred_half"]
            native_gt = native_trace["tangential_gradient"]
            e_continuous, e_native = continuous_gt-reference_gt, native_gt-reference_gt
            change = native_gt-continuous_gt
            error_dot_change = float(np.average(np.sum(e_continuous*change, axis=1), weights=area))
            rms_continuous, rms_native = field_rms(e_continuous, area), field_rms(e_native, area)
            rms_change = field_rms(change, area)
            np.testing.assert_allclose(rms_native**2, rms_continuous**2+2*error_dot_change+rms_change**2, rtol=0, atol=1e-17)
            normal_values, mass = {}, {}
            for observation, velocity in (("point", actual[n_sample:]), ("native", native_trace["face_velocity"])):
                corrected, flux = evaluate_vpm_velocity(adapter, boundary["position"], boundary["normal"], boundary["area"],
                                                        freestream_velocity=np.array([1., 0., 0.]), fvm_box=bounds,
                                                        particle_spacing=.125, evaluated_velocity=velocity)
                normal_values[observation] = np.sum(corrected*boundary["normal"], axis=1)
                mass[observation] = flux
            np.testing.assert_allclose(normal_values["point"], prior[name+"__boundary_normal_velocity"], rtol=0, atol=1e-12)
            owner, neighbour = sampler.trace.mesh["owners"], sampler.trace.mesh["neighbours"]
            cell_index = np.full(mesh["n_cells"], -1)
            cell_index[sampler.sample_cells] = np.arange(n_sample)
            # Independent two-point part; the remainder measures the native
            # nonorthogonal gradient correction, not continuous derivative data.
            distance = np.sum(geometry["cell_connection_vector"][faces]*normal, axis=1)
            two_point = (actual[cell_index[neighbour]]-actual[cell_index[owner]])/distance[:, None]
            row = {"panels": count, "name": name, "boundary_velocity_replay_maximum_difference": replay,
                   "continuous_gradient_error_rms": rms_continuous, "native_gradient_error_rms": rms_native,
                   "gradient_observation_change_rms": rms_change, "error_dot_observation_change": error_dot_change,
                   "value_vs_flux_gradient_rms": field_rms(native_trace["value_tangential_gradient"]-native_gt, area),
                   "nonorthogonal_gradient_correction_rms": field_rms(tangential(native_trace["normal_gradient"]-two_point, normal), area),
                   "point_normal_velocity_error_rms": field_rms((normal_values["point"]-boundary["fvm_normal_velocity"])[:, None], boundary["area"]),
                   "native_normal_velocity_error_rms": field_rms((normal_values["native"]-boundary["fvm_normal_velocity"])[:, None], boundary["area"]),
                   "normal_velocity_observation_change_rms": field_rms((normal_values["native"]-normal_values["point"])[:, None], boundary["area"]),
                   "face_velocity_observation_change_rms": field_rms(native_trace["face_velocity"]-actual[n_sample:], area),
                   "gradient_input_cell_velocity_error_rms": field_rms(actual[cell_index[sampler.gradient_cells]]-truth["native_cell_velocity"][sampler.gradient_cells],
                                                                      geometry["cell_volume"][sampler.gradient_cells]),
                   "body_source_flux": float(panels["panel_area"] @ strengths), "boundary_flux": mass, "face_groups": {}}
            assert abs(row["body_source_flux"]) < 1e-10
            for number, side_name in enumerate(side_names):
                mask = side == number
                row["face_groups"][side_name] = {"faces": int(mask.sum()),
                    "continuous_gradient_error_rms": field_rms(e_continuous[mask], area[mask]),
                    "native_gradient_error_rms": field_rms(e_native[mask], area[mask]),
                    "gradient_observation_change_rms": field_rms(change[mask], area[mask]),
                    "point_normal_velocity_error_rms": field_rms((normal_values["point"]-boundary["fvm_normal_velocity"])[mask, None], boundary["area"][mask]),
                    "native_normal_velocity_error_rms": field_rms((normal_values["native"]-boundary["fvm_normal_velocity"])[mask, None], boundary["area"][mask])}
            records.append(row)
            saved.update({prefix+"__velocity": actual, prefix+"__body_velocity": body,
                          prefix+"__continuous_tangential_gradient": continuous_gt,
                          prefix+"__point_normal_velocity": normal_values["point"], prefix+"__native_normal_velocity": normal_values["native"]})
            saved.update({prefix+"__"+key: value for key, value in native_trace.items()})
            print(json.dumps({key: value for key, value in row.items() if key not in ("face_groups", "boundary_flux")}), flush=True)
    np.savez_compressed(args.output / "sampled-native-face-fields.npz", position=targets, normal=normal, area=area, face_group=side,
                        reference_tangential_gradient=reference_gt, reference_normal_velocity=boundary["fvm_normal_velocity"],
                        reference_face_velocity=reference["face_velocity"], reference_cell_gradient=reference["cell_gradient"], **saved)
    result = {"schema": "openonda-cube-sampled-native-face-3d/1", "status": "complete", "spatial_dimensions": 3,
              "physical_time": experiments[0][1]["physical_time"], "small_fvm_bounds": experiments[0][1]["small_fvm_bounds"],
              "full_fvm_cells": mesh["n_cells"], "small_fvm_cells": small["n_cells"], "boundary_faces": len(faces),
              "sample_cells": n_sample, "gradient_cells": len(sampler.gradient_cells), "source_names": names,
              "body_counts": [report["body_panels"] for _, report, _, _ in experiments],
              "reference_gradient_replay_maximum_difference": reference_replay,
              "reference_face_velocity_flux_replay_maximum_difference": reference_flux_replay,
              "boundary_velocity_replay_maximum_difference": max(replay_errors), "records": records, "sources": sources,
              "elapsed_seconds": time.perf_counter()-started,
              "limitations": ["Frozen coarse physical cube; no advancing boundary condition, force validation or emission change.",
                              "Native observations use VPM velocities sampled at full-mesh cell centres on both sides of the cut. Exterior reference velocities do not enter the source field.",
                              "The sampled gradient uses the actual FVM Gauss kernel and the flux uses its actual diffusion kernel. Remote zero patch placeholders cannot influence selected gradients.",
                              "Continuous derivatives are the qualified centred half-step observations of the identical saved source/body field, with every boundary point velocity replayed.",
                              "The native normal-velocity observation uses face interpolation followed by the same global mass correction as the point observation. It does not reproduce a pressure-corrected advancing Rhie--Chow flux.",
                              "Saved source-panel strengths are evaluated with the exact production triangular kernel. No body re-solve or fitted correction is used."]}
    (args.output / "cube-sampled-native-face-3d.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "studies/coupler_accuracy/results"
    parser.add_argument("--physical", nargs="+", type=Path, default=[base / f"cube-3d-boundary-quadratic-overlap-{count}-qualified" for count in (108, 1728, 6912)])
    parser.add_argument("--oracle", type=Path, default=base / "cube-3d-oracle")
    parser.add_argument("--full", type=Path, default=base / "cube-3d-native-volume-induction")
    parser.add_argument("--boundary", type=Path, default=base / "cube-3d-integrated-velocity-curl-reconstruction-boundary")
    parser.add_argument("--overlap", type=Path, default=base / "cube-3d-volume-overlap")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
