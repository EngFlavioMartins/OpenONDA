#!/usr/bin/env python3
"""Build a continuous compact velocity correction from the qualified 3D seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np

from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from studies.coupler_accuracy.continuous_velocity_curl_3d import (
    ContinuousVelocityCurl,
    compact_polynomial_trace,
)
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.cube_native_moment_reconstruction_3d import read_arrays
from studies.coupler_accuracy.native_boundary_quadratic_3d import BoundaryQuadraticCellLeastSquares
from studies.coupler_accuracy.native_linear_volume_induction_3d import LinearNativeVolumeSources
from studies.coupler_accuracy.native_quadratic_moments_3d import (
    reconstruct_quadratic_faces,
    weak_quadratic_curl_moments,
)
from studies.coupler_accuracy.native_shared_trace_update_3d import global_first_moment, impulse
from studies.coupler_accuracy.native_volume_induction_3d import NativeVolumeSources


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report_path = args.source / "cube-shared-trace-sources-3d.json"
    source_path = args.source / "shared-trace-source-fields.npz"
    report = json.loads(report_path.read_text())
    assert report["status"] == "complete" and report["spatial_dimensions"] == 3
    for row in report["sources"]:
        assert hash_file(ROOT / row["path"]) == row
    mesh_path = ROOT / next(row["path"] for row in report["sources"] if row["path"].endswith("/small-native-mesh.npz"))
    test_path = ROOT / "studies/coupler_accuracy/results/3d-continuous-velocity-curl-regression.xml"
    suites = ET.parse(test_path).getroot().findall(".//testsuite")
    assert sum(int(suite.get("tests", 0)) for suite in suites) == 5
    assert all(int(suite.get(key, 0)) == 0 for suite in suites for key in ("failures", "errors", "skipped"))
    paths = [Path(__file__).resolve(), report_path, source_path, mesh_path, test_path]
    paths += [ROOT / name for name in (
        "studies/coupler_accuracy/continuous_velocity_curl_3d.py", "studies/coupler_accuracy/native_boundary_quadratic_3d.py",
        "studies/coupler_accuracy/native_quadratic_moments_3d.py", "studies/coupler_accuracy/native_shared_trace_update_3d.py",
        "studies/coupler_accuracy/native_linear_volume_induction_3d.py", "studies/coupler_accuracy/native_volume_induction_3d.py",
        "studies/coupler_accuracy/cube_boundary_oracle.py", "studies/coupler_accuracy/cube_native_moment_reconstruction_3d.py",
        "source/solvers/fvm/io/mesh_storage.py", "source/solvers/fvm/mesh/geometry.py",
        "tests/coupler/test_continuous_velocity_curl_3d.py")]
    sources = [hash_file(path) for path in paths]
    for path in paths:
        if path.suffix == ".py":
            target = args.output / "sources" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    (args.output / "sources-at-start.json").write_text(json.dumps(sources, indent=2)+"\n")
    source = read_arrays(source_path)
    mesh = load_native_mesh(mesh_path)
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    native = NativeVolumeSources.from_mesh(mesh)
    linear = LinearNativeVolumeSources.from_native(native, geometry["cell_centre"])
    for key in ("volume", "centroid", "covariance"):
        np.testing.assert_array_equal(source[key], getattr(linear, key))
    c = ContinuousVelocityCurl.from_mesh(mesh, linear.centroid)
    np.testing.assert_allclose(c.cell_volume, linear.volume, rtol=2e-15, atol=0)
    count = mesh["n_interior_faces"]
    weight = geometry["face_interpolation_weight"][:count]
    values = source["cell_velocity"]
    native_face = np.zeros((1, mesh["n_faces"], 3))
    native_face[0, :count] = ((1-weight[:, None])*values[mesh["owners"][:count]]
                              +weight[:, None]*values[mesh["neighbours"]])
    wall = next(patch for patch in mesh["boundary"] if patch["name"] == "cube")
    wall_faces = np.arange(wall["start_face"], wall["start_face"]+wall["n_faces"])
    polynomials, integrals, fits = [], [], []
    for kind in ("point", "cell_average"):
        centres = geometry["cell_centre"] if kind == "point" else linear.centroid
        covariance = None if kind == "point" else linear.covariance/linear.volume[:, None, None]
        fit = BoundaryQuadraticCellLeastSquares.from_mesh(mesh, centres, linear.volume, average_covariance=covariance,
                                                         boundary_faces=wall_faces)
        gradient, hessian = fit.derivatives(values, np.zeros((len(fit.boundary_position), 3)))
        q = fit.cell_integrals(values, gradient, hessian, linear.volume, linear.centroid, linear.covariance)
        np.testing.assert_array_equal(q[0], source[kind+"__new_cell_integral"])
        face = reconstruct_quadratic_faces(mesh, geometry, fit, values, gradient, hessian, np.zeros((mesh["n_faces"], 3)))
        gamma, moment = weak_quadratic_curl_moments(native, linear.centroid, q, geometry["face_centre"], *face)
        np.testing.assert_array_equal(gamma[0], source[kind+"__new_circulation"])
        np.testing.assert_array_equal(moment[0], source[kind+"__new_moment"])
        polynomials.append((face[0]-native_face, face[1], face[2]))
        integrals.append(source[kind+"__weighted_cell_integral_change"])
        fits.append({"input": kind, "maximum_condition": float(fit.condition.max()), "polynomial_replay": "bitwise"})
    polynomial = tuple(np.concatenate([family[i] for family in polynomials]) for i in range(3))
    trace, face_weight, vertex_weight = compact_polynomial_trace(mesh, geometry, c, source["volume_weight"], polynomial)
    integral = np.stack(integrals)
    velocity = c.complete_cell_integrals(trace, integral)
    actual_integral = c.velocity_integrals(velocity)
    np.testing.assert_allclose(actual_integral, integral, rtol=0, atol=1e-16)
    np.testing.assert_array_equal(velocity[:, c.boundary_nodes], 0)
    omega = c.curl(velocity)
    np.testing.assert_array_equal(omega[:, source["volume_weight"][c.parent] == 0], 0)
    gamma, moment = c.curl_moments(omega)
    normal_jump = c.normal_curl_jumps(omega)
    np.testing.assert_allclose(normal_jump, 0, rtol=0, atol=1e-10)
    h = global_first_moment(linear.centroid, gamma, moment)
    expected_h = -np.cross(np.eye(3)[None], integral.sum(axis=1)[:, None])
    np.testing.assert_allclose(gamma.sum(axis=1), 0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(h, expected_h, rtol=0, atol=1e-12)
    compact, _ = c.active_source(omega)
    names = ["point_continuous_curl", "cell_average_continuous_curl"]
    records = []
    for state, name in enumerate(names):
        records.append({"name": name, "maximum_cell_integral_difference": float(np.max(np.abs(actual_integral[state]-integral[state]))),
                        "maximum_normal_curl_jump": float(np.max(np.abs(normal_jump[state]))),
                        "total_circulation": gamma[state].sum(axis=0).tolist(),
                        "total_first_moment": h[state].tolist(), "expected_first_moment": expected_h[state].tolist(),
                        "impulse": impulse(h[state]).tolist(), "expected_impulse": integral[state].sum(axis=0).tolist(),
                        "maximum_first_moment_budget_difference": float(np.max(np.abs(h[state]-expected_h[state])))})
    np.savez_compressed(args.output / "continuous-curl-source-fields.npz", source_names=names,
                        baseline_state_indices=[2, 5], previous_shared_state_indices=[4, 7],
                        node_position=c.position, tetrahedron_nodes=c.tetrahedra, tetrahedron_parent=c.parent,
                        tetrahedron_volume=c.volume, tetrahedron_centroid=c.centroid, cell_centroid=linear.centroid,
                        nodal_velocity=velocity, tetrahedron_vorticity=omega, cell_circulation=gamma, first_moment=moment,
                        cell_velocity_integral_change=integral, actual_cell_velocity_integral_change=actual_integral,
                        face_weight=face_weight, vertex_weight=vertex_weight, volume_weight=source["volume_weight"],
                        normal_curl_jump=normal_jump, compact_source_face_ids=compact.face_ids)
    result = {"schema": "openonda-cube-continuous-curl-sources-3d/1", "status": "complete", "spatial_dimensions": 3,
              "source_names": names, "parent_source": str(args.source.relative_to(ROOT)), "physical_time": report["physical_time"],
              "source_sgs": report["source_sgs"], "particle_spacing": report["particle_spacing"],
              "small_fvm_cells": mesh["n_cells"], "tetrahedra": len(c.tetrahedra), "unique_tetrahedral_faces": len(c.face_nodes),
              "nonzero_induction_faces": len(compact.triangles), "stencil_fits": fits, "records": records,
              "sources": sources, "elapsed_seconds": time.perf_counter()-started,
              "limitations": [
                  "The continuous trace averages the preceding shared polynomial differences at common nodes; it changes their face representation.",
                  "Each cell velocity integral is imposed through its internal node, including a zero change for the mean-input variant.",
                  "Continuous velocity is not necessarily divergence free. Its piecewise constant curl is divergence free in the distributional sense.",
                  "The physical comparison must pair this exact tetrahedral curl with an affine density having identical cell circulation and first moments.",
                  "Only exact-zero induction coefficients are removed. No vorticity magnitude threshold or core smoothing is introduced.",
                  "These are frozen corrections relative to the same native-face-moment baselines, not particle emission or advancing force validation."]}
    (args.output / "cube-continuous-curl-sources-3d.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.source, args.output = args.source.resolve(), args.output.resolve()
    run(args)
