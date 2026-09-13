#!/usr/bin/env python3
"""Add reconstructed first moments inside the qualified small-domain overlap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from source.coupler.boundary import evaluate_vpm_velocity
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, field_rms, hash_file
from studies.coupler_accuracy.joint_reconstruction_3d import CubePanelResponse
from studies.coupler_accuracy.native_linear_volume_induction_3d import LinearNativeVolumeSources
from studies.coupler_accuracy.native_moment_reconstruction_3d import (
    CellLeastSquares,
    reconstruct_linear_face_velocity,
    weak_curl_moments,
)
from studies.coupler_accuracy.native_volume_induction_3d import NativeVolumeSources


def read_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    overlap = json.loads((args.overlap / "cube-volume-overlap-3d.json").read_text())
    full = json.loads((args.full / "cube-native-volume-induction-3d.json").read_text())
    assert overlap["status"] == full["status"] == "complete"
    assert overlap["spatial_dimensions"] == full["spatial_dimensions"] == 3
    paths = [Path(__file__), args.oracle / "small-native-mesh.npz", args.boundary / "boundary-fields.npz"]
    paths += [args.overlap / name for name in ("cube-volume-overlap-3d.json", "induction-checkpoint.npz", "volume-overlap-fields.npz")]
    paths += [args.full / name for name in ("cube-native-volume-induction-3d.json", "native-induction-fields.npz", "volume-induction-comparison-fields.npz")]
    paths += [ROOT / p for p in ("studies/coupler_accuracy/native_moment_reconstruction_3d.py", "studies/coupler_accuracy/native_linear_volume_induction_3d.py",
                                 "studies/coupler_accuracy/native_volume_induction_3d.py", "studies/coupler_accuracy/joint_reconstruction_3d.py",
                                 "source/coupler/boundary.py", "source/solvers/fvm/mesh/geometry.py", "tests/coupler/test_native_moment_reconstruction_3d.py",
                                 "source/solvers/vpm/boundary_elements/panels/kernels/source_velocity.py",
                                 "source/solvers/vpm/boundary_elements/panels/kernels/induced_velocity.py",
                                 "source/solvers/vpm/boundary_elements/panels/solver/linear_solvers.py",
                                 "source/solvers/vpm/boundary_elements/panels/solver/panel_solver.py",
                                 "tutorials/coupled_fvm_vpm/02_cube_flow/assets/cube.stl")]
    sources = [hash_file(path) for path in paths]
    for path in paths:
        if path.suffix == ".py":
            destination = args.output / "sources" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
    (args.output / "sources-at-start.json").write_text(json.dumps(sources, indent=2)+"\n")
    previous = read_arrays(args.overlap / "volume-overlap-fields.npz")
    checkpoint = read_arrays(args.overlap / "induction-checkpoint.npz")
    fields = read_arrays(args.full / "volume-induction-comparison-fields.npz")
    omega = read_arrays(args.full / "native-induction-fields.npz")["cell_vorticity"]
    boundary = read_arrays(args.boundary / "boundary-fields.npz")
    mesh = load_native_mesh(args.oracle / "small-native-mesh.npz")
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    native = NativeVolumeSources.from_mesh(mesh)
    linear = LinearNativeVolumeSources.from_native(native, geometry["cell_centre"])
    ids = checkpoint["inside_cell_ids"]
    np.testing.assert_allclose(geometry["cell_centre"], fields["native_cell_centres"][ids], rtol=0, atol=1e-13)
    np.testing.assert_allclose(geometry["cell_volume"], fields["native_cell_volume"][ids], rtol=0, atol=1e-14)
    assert len(ids) == 2840
    baseline_index = overlap["state_names"].index("taper_0.75_to_1.25")
    weight = checkpoint["volume_weights"][baseline_index]
    active = weight > 0
    assert active.sum() == 2112 and checkpoint["gaussian_position"].shape[0] == 2908
    gamma = omega[ids]*fields["native_cell_volume"][ids, None]
    np.testing.assert_allclose(checkpoint["volume_density"][baseline_index]*linear.volume[:, None], weight[:, None]*gamma,
                               rtol=0, atol=1e-16)
    cell_u = fields["native_cell_velocity"][ids]
    fit_fvm = CellLeastSquares.from_mesh(mesh, geometry["cell_centre"])
    fit_poly = CellLeastSquares.from_mesh(mesh, linear.centroid)
    gradient_u = fit_fvm.gradient(cell_u)[0]
    integral_u = (cell_u+np.einsum("ci,cij->cj", linear.centroid-geometry["cell_centre"], gradient_u))*linear.volume[:, None]
    n = mesh["n_interior_faces"]
    own, nei = mesh["owners"][:n], mesh["neighbours"]
    interpolation = geometry["face_interpolation_weight"][:n]
    face_u = np.zeros((mesh["n_faces"], 3))
    face_u[:n] = (1-interpolation[:, None])*cell_u[own]+interpolation[:, None]*cell_u[nei]
    # Outer placeholders cannot influence active source cells. Cell gradient
    # reconstruction uses only interior neighbours, not boundary placeholders.
    for patch in mesh["boundary"]:
        if patch["name"] != "cube":
            rows = slice(patch["start_face"], patch["start_face"]+patch["n_faces"])
            assert not np.any(active[mesh["owners"][rows]])
    weak_gamma, native_moment = weak_curl_moments(native, linear.centroid, integral_u, geometry["face_centre"], face_u)
    replay = float(np.max(np.abs(weak_gamma[0, active]-gamma[active])/linear.volume[active, None]))
    assert replay < 1e-10
    face_linear, face_gradient = reconstruct_linear_face_velocity(mesh, geometry, cell_u, gradient_u, np.zeros_like(face_u))
    _, face_moment = weak_curl_moments(native, linear.centroid, integral_u[None], geometry["face_centre"], face_linear, face_gradient)
    lsq_moment = linear.covariance @ fit_poly.gradient(gamma/linear.volume[:, None])[0]
    names = ["constant_volume_control", "circulation_lsq_moments", "native_face_moments", "linear_face_moments"]
    moments = weight[None, :, None, None]*np.stack((np.zeros_like(lsq_moment), lsq_moment, native_moment[0], face_moment[0]))
    targets = checkpoint["position"]
    np.testing.assert_array_equal(targets, previous["position"])
    selections = {name: slice(*limits) for name, limits in overlap["target_slices"].items()}
    coefficient, _ = linear.coefficients(np.zeros((len(names), native.n_cells, 3)), moments)

    def progress(done, total):
        if done % 512 == 0 or done == total:
            print(json.dumps({"stage": "moment_induction", "done": done, "total": total,
                              "elapsed_seconds": time.perf_counter()-started}), flush=True)

    correction = linear.evaluate(targets, coefficient, progress=progress).transpose(1, 0, 2)
    np.testing.assert_array_equal(correction[0], 0)
    induced = checkpoint["induced_velocity"][baseline_index][None]+correction
    np.savez_compressed(args.output / "moment-overlap-induction-checkpoint.npz", position=targets, moment_induction=correction,
                        weighted_first_moment=moments, source_names=np.asarray(names), inside_cell_ids=ids, volume_weights=weight,
                        native_circulation=gamma, polyhedron_volume=linear.volume, polyhedron_centroid=linear.centroid,
                        covariance=linear.covariance, cell_velocity_integral=integral_u)
    body = CubePanelResponse()
    epsilon = overlap["derivative_steps"][0]
    normal = boundary["native_unit_normal"]
    legacy_length = np.linalg.norm(boundary["normal"], axis=1)
    bounds = np.column_stack((mesh["vertex_position"].min(axis=0), mesh["vertex_position"].max(axis=0))).ravel()
    adapter = SimpleNamespace(particles=SimpleNamespace(n_particles_total=len(checkpoint["gaussian_position"])))
    volume_rows = active[native.owners] | ((native.neighbours >= 0) & active[np.maximum(native.neighbours, 0)])
    low, high = native.triangles[volume_rows].min(axis=(0, 1)), native.triangles[volume_rows].max(axis=(0, 1))
    distance = np.maximum(np.maximum(low-boundary["position"], boundary["position"]-high), 0)
    gap = float(np.linalg.norm(distance, axis=1).min())
    assert gap > .249

    def tangent(gradient):
        gradient = gradient/legacy_length[:, None]
        return gradient-np.sum(gradient*normal, axis=1)[:, None]*normal

    records, output = [], {}
    for state, name in enumerate(names):
        body.panel.solve(np.array([1., 0., 0.]), induced[state, selections["collocation"]], time=overlap["physical_time"])
        actual = induced[state]+[1, 0, 0]+body.panel.compute_induced_velocity(targets)
        source_flux = float(body.area @ body.panel.lattice.source_strength.to_numpy()[:body.count])
        assert abs(source_flux) < 1e-10
        if state == 0:
            np.testing.assert_allclose(actual, previous["taper_0.75_to_1.25__velocity"], rtol=0, atol=1e-12)
        centre, plus, minus = [actual[selections[key]] for key in ("boundary", "plus", "minus")]
        ph, mh = [actual[selections[key]] for key in ("plus_half", "minus_half")]
        derivatives = {"centred": tangent((plus-minus)/(2*epsilon)), "centred_half": tangent((ph-mh)/epsilon),
                       "exterior": tangent((-3*centre+4*plus-actual[selections["plus2"]])/(2*epsilon)),
                       "interior": tangent((3*centre-4*minus+actual[selections["minus2"]])/(2*epsilon)),
                       "exterior_half": tangent((-3*centre+4*ph-plus)/epsilon),
                       "interior_half": tangent((3*centre-4*mh-minus)/epsilon)}
        cut, mass = evaluate_vpm_velocity(adapter, boundary["position"], boundary["normal"], boundary["area"],
                                          freestream_velocity=np.array([1., 0., 0.]), fvm_box=bounds,
                                          particle_spacing=.125, evaluated_velocity=centre)
        un = np.sum(cut*boundary["normal"], axis=1)
        wall = actual[selections["wall"]]
        wall_un = np.sum(wall*fields["wall_normal"], axis=1)
        record = {"name": name, "cell_velocity_rms_over_Uinf": {},
                  "boundary_normal_velocity_rms_error_over_Uinf": field_rms((un-boundary["fvm_normal_velocity"])[:, None], boundary["area"]),
                  "boundary_native_tangential_gradient_rms_error": {key: field_rms(value-boundary["fvm_native_flux_tangential_normal_gradient"], boundary["native_vector_area"])
                                                                    for key, value in derivatives.items()},
                  "derivative_step_halving_maximum_difference": {key: float(np.max(np.abs(derivatives[key]-derivatives[key+"_half"])))
                                                                  for key in ("centred", "exterior", "interior")},
                  "one_sided_derivative_difference_rms": field_rms(derivatives["exterior_half"]-derivatives["interior_half"], boundary["native_vector_area"]),
                  "wall_normal_velocity_rms_over_Uinf": float(np.sqrt(np.mean(wall_un**2))),
                  "wall_tangential_velocity_rms_over_Uinf": float(np.sqrt(np.mean(np.sum((wall-wall_un[:, None]*fields["wall_normal"])**2, axis=1)))),
                  "boundary_flux": mass, "body_source_flux": source_flux}
        for group in full["cell_target_counts"]:
            selected = fields[group+"__cell_ids"]
            record["cell_velocity_rms_over_Uinf"][group] = field_rms(actual[selections[group]]-fields["native_cell_velocity"][selected],
                                                                    fields["native_cell_volume"][selected])
        records.append(record)
        output[name+"__velocity"] = actual
        output[name+"__boundary_normal_velocity"] = un
        output.update({name+"__gradient_"+key: value for key, value in derivatives.items()})
        print(json.dumps(record), flush=True)
    np.savez_compressed(args.output / "reconstructed-moment-overlap-fields.npz", position=targets, **output)
    report = {"schema": "openonda-cube-reconstructed-moment-overlap-3d/1", "status": "complete", "spatial_dimensions": 3,
              "physical_time": overlap["physical_time"], "small_fvm_cells": native.n_cells, "exterior_particles": 68,
              "positive_volume_cells": int(active.sum()), "body_panels": body.count, "small_fvm_bounds": bounds.tolist(),
              "minimum_coupling_target_distance_to_volume_support_box": gap, "native_circulation_replay_maximum_difference_over_volume": replay,
              "lsq_fvm_maximum_condition": float(fit_fvm.condition.max()), "lsq_polyhedron_maximum_condition": float(fit_poly.condition.max()),
              "source_names": names, "target_slices": overlap["target_slices"], "derivative_steps": overlap["derivative_steps"],
              "records": records, "sources": sources, "elapsed_seconds": time.perf_counter()-started,
              "limitations": ["Frozen physical cube induction, not a live coupler, time advance, force validation or emission scheme.",
                              "Every original near-cell circulation, Gaussian complement and exterior particle is unchanged; only zero-circulation affine first-moment corrections are added.",
                              "Moments are reconstructed from the small FVM mesh and its velocity/circulation values only. No exterior cells enter these stencils.",
                              "Moment support uses the previously qualified 0.75-to-1.25 taper, with more than 0.249 D separation from every coupling target.",
                              "Outer boundary placeholders cannot enter active source-cell traces; that topology condition is checked. Cube boundary velocity is zero.",
                              "The physical FVM snapshot is the same full-oracle initial state at t=0.5 used in preceding overlap tests.",
                              "Linear-face moments are used while retaining original native Gamma. The changed Stokes Gamma from that face reconstruction is not adopted."]}
    (args.output / "cube-reconstructed-moment-overlap-3d.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "studies/coupler_accuracy/results"
    parser.add_argument("--overlap", type=Path, default=base / "cube-3d-volume-overlap")
    parser.add_argument("--full", type=Path, default=base / "cube-3d-native-volume-induction")
    parser.add_argument("--oracle", type=Path, default=base / "cube-3d-oracle")
    parser.add_argument("--boundary", type=Path, default=base / "cube-3d-integrated-velocity-curl-reconstruction-boundary")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
