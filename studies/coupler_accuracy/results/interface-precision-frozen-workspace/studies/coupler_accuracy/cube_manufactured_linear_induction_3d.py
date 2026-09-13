#!/usr/bin/env python3
"""Measure the information gained by retaining exact 3D cell first moments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.manufactured_cell_moments_3d import native_first_vorticity_moment
from studies.coupler_accuracy.manufactured_cube_field_3d import NoSlipCubeField
from studies.coupler_accuracy.native_cell_integrals_3d import NativeCellIntegration
from studies.coupler_accuracy.native_linear_volume_induction_3d import LinearNativeVolumeSources
from studies.coupler_accuracy.native_velocity_curl_integrals_3d import triangle_rule
from studies.coupler_accuracy.native_volume_induction_3d import (
    NativeVolumeSources,
    triangle_geometry,
)


def read_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def qualification(mesh, geometry, linear, fields, gamma, moment, cell_ids):
    integration = NativeCellIntegration.from_mesh(mesh, geometry, cell_ids)
    moment_errors, moment_refinements, induction_errors, induction_refinements = [], [], [], []
    geometry_errors = []
    for row, cell in enumerate(cell_ids):
        answers = []
        for order in (12, 16):
            point, weight = integration.rule(row, order)
            answers.append(np.array([np.einsum("q,qi,qj->ij", weight, point-linear.centroid[cell], field.vorticity(point)) for field in fields]))
        scale = linear.volume[cell]**(4/3)
        moment_errors.append(float(np.max(np.abs(moment[:, cell]-answers[-1]))/scale))
        moment_refinements.append(float(np.max(np.abs(answers[-1]-answers[0]))/scale))
        triangles = integration.tetrahedra[row][:, 1:]
        source = NativeVolumeSources(*triangle_geometry(triangles), np.zeros(len(triangles), dtype=int),
                                     np.full(len(triangles), -1), np.arange(len(triangles)), 1)
        one = LinearNativeVolumeSources.from_native(source, linear.centroid[cell:cell+1])
        geometry_errors.append(float(np.max(np.abs(one.covariance[0]-linear.covariance[cell]))))
        coefficients, gradient = one.coefficients(gamma[:, cell:cell+1], moment[:, cell:cell+1])
        targets = np.array([one.centroid[0], [1.7, -.83, .91], [-1.3, 1.41, -.61]])
        actual = one.evaluate(targets, coefficients).transpose(1, 0, 2)
        answers = []
        for order in (16, 20):
            value = np.zeros_like(actual)
            q, weight = integration.rule(row, order)
            density = gamma[:, cell, None]/linear.volume[cell]+np.einsum("qi,sij->sqj", q-one.centroid[0], gradient[:, 0])
            delta = targets[1:, None]-q
            value[:, 1:] = np.einsum("q,spqd->spd", weight, np.cross(density[:, None], delta[None])
                                     /(4*np.pi*np.linalg.norm(delta, axis=2)[None, :, :, None]**3))
            # Independent target-apex radial integration, exact in the radial
            # coordinate for the affine density, regular angular quadrature.
            barycentric, weight = triangle_rule(order)
            relative = triangles-targets[0]
            q = np.einsum("qa,tad->tqd", barycentric, relative)
            determinant = np.einsum("ti,ti->t", relative[:, 0], np.cross(relative[:, 1], relative[:, 2]))
            density_at_target = gamma[:, cell]/one.volume[0]
            numerator = np.cross(density_at_target[:, None, None], q[None])
            numerator += .5*np.cross(np.einsum("tqi,sij->stqj", q, gradient[:, 0]), q[None])
            value[:, 0] = -np.einsum("t,q,stqd->sd", determinant/2, weight,
                                     numerator/np.linalg.norm(q, axis=2)[None, :, :, None]**3)/(4*np.pi)
            answers.append(value)
        induction_errors.append(float(np.max(np.abs(actual-answers[-1]))))
        induction_refinements.append(float(np.max(np.abs(answers[-1]-answers[0]))))
    assert max(moment_errors) < 1e-8 and max(moment_refinements) < 1e-8
    assert max(induction_errors) < 1e-9 and max(induction_refinements) < 1e-9
    return {"cell_ids": np.asarray(cell_ids).tolist(), "moment_volume_orders": [12, 16], "induction_orders": [16, 20],
            "maximum_surface_vs_volume_moment_difference_over_V_to_four_thirds": max(moment_errors),
            "maximum_volume_moment_refinement_difference_over_V_to_four_thirds": max(moment_refinements),
            "maximum_induction_vs_volume_quadrature_difference_over_Uref": max(induction_errors),
            "maximum_induction_quadrature_refinement_difference_over_Uref": max(induction_refinements),
            "maximum_isolated_cell_covariance_difference": max(geometry_errors)}


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    parent = json.loads((args.manufactured / "cube-manufactured-induction-3d.json").read_text())
    assert parent["status"] == "complete" and parent["spatial_dimensions"] == 3
    mesh_path = ROOT / next(p["path"] for p in parent["sources"] if p["path"].endswith("full-native-mesh.npz"))
    paths = [Path(__file__), mesh_path]
    paths += [args.manufactured / name for name in ("cube-manufactured-induction-3d.json", "manufactured-source-fields.npz", "manufactured-induction-fields.npz")]
    paths += [ROOT / p for p in ("studies/coupler_accuracy/manufactured_cell_moments_3d.py",
                                 "studies/coupler_accuracy/manufactured_cube_field_3d.py", "studies/coupler_accuracy/native_linear_volume_induction_3d.py",
                                 "studies/coupler_accuracy/native_volume_induction_3d.py", "studies/coupler_accuracy/native_cell_integrals_3d.py",
                                 "studies/coupler_accuracy/native_velocity_curl_integrals_3d.py", "tests/coupler/test_native_linear_volume_induction_3d.py")]
    sources = [hash_file(path) for path in paths]
    for path in paths:
        if path.suffix == ".py":
            destination = args.output / "sources" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
    (args.output / "sources-at-start.json").write_text(json.dumps(sources, indent=2)+"\n")
    source = read_arrays(args.manufactured / "manufactured-source-fields.npz")
    target = read_arrays(args.manufactured / "manufactured-induction-fields.npz")
    mesh = load_native_mesh(mesh_path)
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    native = NativeVolumeSources.from_mesh(mesh)
    linear = LinearNativeVolumeSources.from_native(native, geometry["cell_centre"])
    np.testing.assert_array_equal(linear.volume, source["polyhedron_volume"])
    np.testing.assert_array_equal(linear.centroid, source["polyhedron_centroid"])
    fields = [NoSlipCubeField(f["width"]) for f in parent["manufactured_fields"]]
    gamma = source["vorticity_integral"]
    history, previous, accepted = [], None, None

    def progress(done, total, order):
        print(json.dumps({"stage": "first_moment", "order": order, "done": done, "total": total,
                          "elapsed_seconds": time.perf_counter()-started}), flush=True)

    for order in (6, 8, 10, 12):
        current = native_first_vorticity_moment(native, linear.centroid, fields, source["velocity_integral"], order=order, progress=progress)
        difference = None if previous is None else float(np.max(np.abs(current-previous)/linear.volume[None, :, None, None]**(4/3)))
        history.append({"order": order, "maximum_moment_change_over_V_to_four_thirds": difference})
        print(json.dumps({"stage": "moment_refinement", **history[-1]}), flush=True)
        if difference is not None and difference < 1e-9:
            accepted = current
            break
        previous = current
    if accepted is None:
        raise RuntimeError("Manufactured first moments did not converge")
    checks = qualification(mesh, geometry, linear, fields, gamma, accepted, parent["volume_qualification"]["cell_ids"])
    print(json.dumps({"stage": "qualified", **checks}), flush=True)
    coefficients, gradient = linear.coefficients(gamma, accepted)
    recovered_moment = linear.covariance[None] @ gradient
    recovery_difference = float(np.max(np.abs(recovered_moment-accepted)/linear.volume[None, :, None, None]**(4/3)))
    assert recovery_difference < 1e-10
    np.savez_compressed(args.output / "linear-source-fields.npz", circulation=gamma, first_moment=accepted,
                        gradient=gradient, volume=linear.volume, centroid=linear.centroid, covariance=linear.covariance)

    def induction_progress(done, total):
        if done % 128 == 0 or done == total:
            print(json.dumps({"stage": "linear_induction", "done": done, "total": total,
                              "elapsed_seconds": time.perf_counter()-started}), flush=True)

    velocity = linear.evaluate(target["position"], coefficients, progress=induction_progress).transpose(1, 0, 2)
    np.savez_compressed(args.output / "linear-induction-fields.npz", position=target["position"], velocity=velocity,
                        exact_velocity=target["exact_velocity"])
    records = []
    for index, field in enumerate(fields):
        record = {"width": field.width, "velocity_errors": {}}
        for group, limits in parent["target_slices"].items():
            rows = slice(*limits)
            difference = velocity[index, rows]-target["exact_velocity"][index, rows]
            record["velocity_errors"][group] = {"rms_over_reference_speed": float(np.sqrt(np.mean(np.sum(difference**2, axis=1)))),
                                                "maximum_over_reference_speed": float(np.max(np.linalg.norm(difference, axis=1)))}
        records.append(record)
        print(json.dumps(record), flush=True)
    report = {"schema": "openonda-cube-manufactured-linear-induction-3d/1", "status": "complete", "spatial_dimensions": 3,
              "source_cells": native.n_cells, "source_triangles": len(native.triangles), "target_slices": parent["target_slices"],
              "parent_manufactured_directory": str(args.manufactured.relative_to(ROOT)), "quadrature_history": history,
              "quadrature_qualification": checks, "first_moment_recovery_difference_over_V_to_four_thirds": recovery_difference,
              "cell_covariance_maximum_condition_number": float(np.max(np.linalg.cond(linear.covariance))),
              "raw_affine_vorticity_divergence_volume_rms": [float(np.sqrt(np.average(np.trace(g, axis1=1, axis2=2)**2, weights=linear.volume))) for g in gradient],
              "records": records, "sources": sources, "elapsed_seconds": time.perf_counter()-started,
              "limitations": ["Kinematic exact-moment manufactured experiment; no FVM time advance or live transfer implementation.",
                              "Same full 3D mesh and all targets as the parent manufactured study; no source pruning, panels, strength fit or core radius.",
                              "All cell circulations and first central moments are preserved. Raw affine vorticity is not constrained to be solenoidal.",
                              "Exact manufactured moments are an information-content control, not data already available from the physical FVM solution.",
                              "Direct face integration is unaccelerated. No performance comparison with FMM or particle evolution is claimed."]}
    (args.output / "cube-manufactured-linear-induction-3d.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manufactured", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.manufactured, args.output = args.manufactured.resolve(), args.output.resolve()
    run(args)
